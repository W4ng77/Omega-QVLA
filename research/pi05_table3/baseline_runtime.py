"""Existing Marlin/QServe packed kernels; no full floating weight inference.

Loader constructs the architecture and loads only NONquantized weights plus
packed targets. It never requires the original checkpoint's target weights.
"""

from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from research.pi05_table3 import kernels


@triton.jit
def _prepare8(X, SM, Q, SC, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    x = tl.div_rn(x, tl.load(SM + k, k < K, 1)).to(tl.float16).to(tl.float32)
    scale = (tl.maximum(tl.max(tl.abs(x)), 1e-5) / 127).to(tl.float16).to(tl.float32)
    q = tl.minimum(tl.maximum(libdevice.nearbyint(tl.div_rn(x, scale)), -128), 127).to(tl.int8)
    tl.store(Q + row * K + k, q, k < K)
    tl.store(SC + row, scale)


class BaselineLinear(nn.Module):
    def __init__(self, record, name):
        super().__init__()
        self.name = name
        self.backend = record["backend"]
        self.in_features, self.out_features = record["original_K"], record["original_N"]
        self.k, self.n = record["K"], record["N"]
        self.external_dtype = getattr(torch, record["external_dtype"].removeprefix("torch."))
        for key in ["B", "s", "s2", "z2", "z", "bias"]:
            self.register_buffer(key, record.get(key) if record.get(key) is not None else torch.empty(0))
        self.register_buffer("_dtype_only", torch.empty(0, dtype=self.external_dtype))
        self.calls = 0
        self.shapes = set()
        if self.backend == "marlin":
            self._kernel = kernels.marlin()
            self.register_buffer("workspace", torch.zeros(self.n // 128 * 16, dtype=torch.int32), persistent=False)
        elif self.backend == "awq":
            self._kernel = kernels.awq()
            self.group = record["group"]
            self.compute_dtype = getattr(torch, record.get("compute_dtype", "float16"))
        else:
            assert self.backend == "qserve"
            self._kernel = kernels.qserve()
            smooth = torch.ones(self.k, dtype=torch.float32)
            smooth[: self.in_features] = record["smooth"]
            self.register_buffer("smooth", smooth)

    @property
    def weight(self):
        return self._dtype_only

    def forward(self, x):
        self.calls += 1
        self.shapes.add(tuple(x.shape))
        shape = x.shape
        xx = x.reshape(-1, self.in_features).contiguous()
        if self.k != self.in_features:
            xx = F.pad(xx, (0, self.k - self.in_features))
        m = len(xx)
        if self.backend == "marlin":
            y = torch.empty((m, self.n), device=x.device, dtype=torch.float16)
            # Pinned upstream's parallel-M G128 path fails the frozen-weight
            # oracle on SM89 (e.g. M128+). Keep its existing verified kernel
            # with independent M64 problems, never a dense floating fallback.
            self._kernel.mul(xx.half(), self.B, y, self.s, self.workspace, max_par=1)
        elif self.backend == "awq":
            aa = xx.to(self.compute_dtype)
            if m < 8:
                y = self._kernel.gemv_forward_cuda_new(aa, self.B, self.s, self.z, m, self.n, self.k, self.group)
            else:
                y = self._kernel.gemm_forward_cuda_new(aa, self.B, self.s, self.z)
        else:
            y = torch.empty((m, self.n), device=x.device, dtype=torch.float16)
            q = torch.empty((m, self.k), device=x.device, dtype=torch.int8)
            asc = torch.empty(m, device=x.device, dtype=torch.float16)
            _prepare8[(m,)](
                xx,
                self.smooth,
                q,
                asc,
                self.k,
                triton.next_power_of_2(self.k),
                num_warps=4 if self.k < 4096 else 8,
            )
            self._kernel.gemm_forward_cuda(q, self.B, self.z2, self.s2, self.s, asc, y)
        y = y[:, : self.out_features].to(self.external_dtype)
        if self.bias.numel():
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)


def load_policy(directory, cfg, steps):
    from safetensors.torch import load_model
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.training import checkpoints
    from openpi.policies import policy as policy_module
    from openpi import transforms
    from transformers.modeling_utils import no_init_weights

    directory = Path(directory)
    manifest = json.loads((directory / "deployment.json").read_text())
    packed = torch.load(directory / "packed.pt", map_location="cpu", mmap=True, weights_only=False)
    with no_init_weights():
        model = PI0Pytorch(config=cfg.model)
    # HF no_init_weights skips init_weights' tie step. Restore the original
    # embedding/lm_head alias BEFORE safetensors alias-aware partial loading.
    model.paligemma_with_expert.paligemma.tie_weights()
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    missing, unexpected = load_model(model, str(directory / "model_nonquant.safetensors"), strict=False)
    expected_missing = set(manifest["excluded_state_keys"])
    assert set(missing) == expected_missing, {
        "extra_missing": sorted(set(missing) - expected_missing),
        "unexpectedly_present": sorted(expected_missing - set(missing)),
    }
    assert not unexpected, unexpected
    modules = dict(model.named_modules())
    for name, record in packed["layers"].items():
        assert isinstance(modules[name], nn.Linear)
        parent, attr = name.rsplit(".", 1)
        setattr(modules[parent], attr, BaselineLinear(record, name))
    del modules
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data.asset_id is not None
    stats = checkpoints.load_norm_stats(directory / "assets", data.asset_id)
    return policy_module.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": steps},
        metadata=cfg.policy_metadata,
        is_pytorch=True,
        pytorch_device="cuda",
    )
