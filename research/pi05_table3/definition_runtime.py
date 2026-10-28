"""Same frozen signed15 core grid: FP32 oracle versus explicit packed kernels.

LLM per-token Sa permits nativeS4/INT32 +FP32 row/column epilogue. Head Sa[k]
remains inside K: decode packed A4/W4 tiles for FP16 MMA/FP32 accumulation.
No output rotation, step weight banks, requantization or silent fallback.
"""

import json
from pathlib import Path
import torch
from torch import nn
import triton
import triton.language as tl
from research.pi05_table3.paper_math import block_transform, token_quantize, signed15, pack_int4, equation_oracle


@triton.jit
def _channel_a4w4(
    A,
    SA,
    W,
    SW,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        pa = tl.load(A + m[:, None] * (K // 2) + k[None, :] // 2, (m[:, None] < M) & (k[None, :] < K), 0).to(
            tl.int32
        )
        qa = (pa >> ((k[None, :] % 2) * 4)) & 15
        qa = tl.where(qa >= 8, qa - 16, qa)
        sa = tl.load(SA + k, k < K, 0).to(tl.float32)
        av = (qa.to(tl.float32) * sa[None, :]).to(tl.float16)
        pw = tl.load(W + n[:, None] * (K // 2) + k[None, :] // 2, (n[:, None] < N) & (k[None, :] < K), 0).to(
            tl.int32
        )
        qw = (pw >> ((k[None, :] % 2) * 4)) & 15
        qw = tl.where(qw >= 8, qw - 16, qw).to(tl.float16)
        acc = tl.dot(av, tl.trans(qw), acc)
    # Sw is per output, so it may be outside K. Sa[k] above may not.
    acc *= tl.load(SW + n, n < N, 0).to(tl.float32)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def unpack_codes(p):
    q = torch.stack((p & 15, (p >> 4) & 15), -1).flatten(-2).to(torch.int8)
    return torch.where(q >= 8, q - 16, q)


class DefinitionLinear(nn.Module):
    def __init__(self, record, name, implementation):
        super().__init__()
        self.name = name
        self.in_features = record["K"]
        self.out_features = record["N"]
        self.act_mode = record["act_mode"]
        self.calls = 0
        self.shapes = set()
        self.steps_seen = set()
        for key in ["B", "s", "rot_in", "perm", "a_scale", "bias"]:
            self.register_buffer(key, record.get(key) if record.get(key) is not None else torch.empty(0))
        self.register_buffer("_dtype_only", torch.empty(0, dtype=torch.bfloat16))
        assert (
            self.s.shape == (self.out_features, 1)
            and self.s.dtype == torch.float32
            and self.rot_in.dtype == torch.float32
        )
        self.set_implementation(implementation)
        if implementation == "reference":
            self.register_buffer("reference_codes", unpack_codes(self.B))

    @property
    def weight(self):
        return self._dtype_only

    def set_implementation(self, implementation):
        assert implementation in ["reference", "packed"]
        self.implementation = implementation
        if implementation == "packed" and self.act_mode == "token":
            from research.pi05_table3.native_llm import extension

            self._native = extension()

    def prepare(self, x):
        xr = block_transform(x.float(), self.rot_in, self.perm)
        if self.act_mode == "token":
            q, s = token_quantize(xr)
        else:
            from gr00t.quantization.dit_step_context import get_current_dit_step, get_total_dit_steps

            step = get_current_dit_step()
            assert step is not None and 0 <= step < 8 and get_total_dit_steps() == 8, (self.name, step)
            self.steps_seen.add(step)
            s = self.a_scale[step]
            q = signed15(xr, s)
        return q.reshape(-1, self.in_features), s.reshape(-1, 1) if self.act_mode == "token" else s

    def forward(self, x):
        self.calls += 1
        self.shapes.add(tuple(x.shape))
        q, sa = self.prepare(x)
        with torch.autocast("cuda", enabled=False):
            if self.implementation == "reference":
                y = equation_oracle(q, self.reference_codes, sa, self.s, self.bias)
            else:
                aa = pack_int4(q)
                m = len(q)
                if self.act_mode == "token":
                    y = self._native.matmul(aa, self.B).float() * sa * self.s.t()
                else:
                    y = torch.empty((m, self.out_features), device=x.device, dtype=torch.bfloat16)
                    _channel_a4w4[(triton.cdiv(m, 32), triton.cdiv(self.out_features, 64))](
                        aa,
                        sa,
                        self.B,
                        self.s,
                        y,
                        m,
                        self.out_features,
                        self.in_features,
                        32,
                        64,
                        64,
                        num_warps=4,
                        num_stages=3,
                    )
                if self.bias.numel():
                    y = y.float() + self.bias.float()
        return y.bfloat16().reshape(*x.shape[:-1], self.out_features)


def load_policy(directory, cfg, implementation):
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies import policy as policy_module
    from openpi.training import checkpoints
    from openpi import transforms
    from transformers.modeling_utils import no_init_weights
    from safetensors.torch import load_model
    from gr00t.quantization.int4_storage import unpack_int4
    from gr00t.quantization.rtn_layers import RTNLinear
    from research.pi05_table3.vision_packed import PackedVision

    directory = Path(directory)
    manifest = json.loads((directory / "deployment.json").read_text())
    pack = torch.load(directory / "packed_core.pt", map_location="cpu", mmap=True, weights_only=False)
    vision = torch.load(directory / "vision_int4.pt", map_location="cpu", mmap=True, weights_only=False)
    with no_init_weights():
        model = PI0Pytorch(config=cfg.model)
    model.paligemma_with_expert.paligemma.tie_weights()
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    missing, unexpected = load_model(model, str(directory / "model_nonquant.safetensors"), strict=False)
    assert set(missing) == set(manifest["excluded_state_keys"]) and not unexpected
    modules = dict(model.named_modules())
    for name, record in pack["layers"].items():
        parent, attr = name.rsplit(".", 1)
        setattr(modules[parent], attr, DefinitionLinear(record, name, implementation))
    for key, record in vision.items():
        name = key[:-7]
        parent, attr = name.rsplit(".", 1)
        source = modules[name]
        if implementation == "reference":
            source.weight.data.copy_(unpack_int4(record).to(source.weight.dtype))
            replacement = RTNLinear(
                source, name=name, group_size=record["group_size"], wbits=4, abits=4, prequantized=True
            )
        else:
            replacement = PackedVision(source, record, name)
        setattr(modules[parent], attr, replacement)
    del modules
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
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
        sample_kwargs={"num_steps": 8},
        metadata=cfg.policy_metadata,
        is_pytorch=True,
        pytorch_device="cuda",
    )
