"""One fixed deployment versus its exact-grid equation oracle; no backend search.

Deployment uses real nibble-packed W4 with the EXISTING tile_fp16 kernel. A4
is numerical quantization with floating transient operands, NOT nativeS4 MMA.
No full floating weight cache is allocated by the packed implementation.
"""
import os
import torch
from torch import nn
import triton
from gr00t.quantization.perstep_backends import unpack_int4
from gr00t.quantization.dit_step_context import get_current_dit_step
from research.holoq_bench.packed_runtime import _tile_fp16
from research.rebuttal.method import block_transform, token_quantize, signed15, equation_oracle


class PaperLinear(nn.Module):
    def __init__(self, record, name, implementation):
        super().__init__()
        assert implementation in ("reference", "packed")
        self.implementation = implementation
        self.backend = "paper_equation_fp32_reference" if implementation == "reference" else "paper_packed_w4_fp16_mma"
        self.name = name
        self.in_features, self.out_features = record["K"], record["N"]
        self.act_mode = record["act_mode"]
        self.calls = 0
        self.input_shapes = set()
        self.steps_seen = set()
        for key in ("B", "s", "rot_in", "perm", "a_scale", "bias"):
            value = record.get(key)
            self.register_buffer(key, value if value is not None else torch.empty(0))
        assert self.s.shape == (self.out_features, 1) and self.s.dtype == torch.float32
        assert self.rot_in.dtype == torch.float32
        if implementation == "reference":
            self.register_buffer("reference_codes", unpack_int4(self.B, self.in_features).float())

    def prepare(self, x):
        # Match the FP16 base model's Linear input conversion; preserve FP32
        # rotation/quantizer arithmetic afterward, before packed-MMA rounding.
        xr = block_transform(x.half().float(), self.rot_in, self.perm)
        if self.act_mode == "token":
            q, scale = token_quantize(xr)
        else:
            step = get_current_dit_step()
            if step is None or not 0 <= step < 8:
                raise RuntimeError(f"Missing/out-of-range8-step scale for {self.name}")
            self.steps_seen.add(step)
            scale = self.a_scale[step]
            q = signed15(xr, scale)
        return q.reshape(-1, self.in_features), scale.reshape(-1, 1) if self.act_mode == "token" else scale

    def forward(self, x):
        self.calls += 1
        self.input_shapes.add(tuple(x.shape))
        q, scale = self.prepare(x)
        shape = x.shape
        with torch.autocast(x.device.type, enabled=False):
            if self.implementation == "reference":
                y = equation_oracle(q, self.reference_codes, scale, self.s, self.bias)
            else:
                aa = (q.float() * scale).half().contiguous()
                m, k, n = len(aa), self.in_features, self.out_features
                y = torch.empty((m, n), dtype=torch.float16, device=x.device)
                _tile_fp16[(triton.cdiv(m, 32), triton.cdiv(n, 64))](aa, self.B, self.s, y, m, n, k, k, 32, 64, 64, num_warps=4, num_stages=3)
                # Sw is applied during in-register weight decode. This is the
                # same mathematical weight; FP16 operand/output rounding is
                # explicitly measured against the FP32 equation oracle.
                y = y.float()
                if self.bias.numel():
                    y = y + self.bias.float()
        return y.half().reshape(*shape[:-1], self.out_features)


def enable(model, path):
    implementation = os.environ.get("HOLOQ_REBUTTAL_IMPLEMENTATION")
    if implementation not in ("reference", "packed"):
        raise ValueError("Explicit HOLOQ_REBUTTAL_IMPLEMENTATION=reference|packed is required; no fallback")
    pack = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    assert pack["meta"]["runtime_plugin"] == "submission_contract_v2"
    names = dict(model.named_modules())
    for name, record in pack["layers"].items():
        source = names[name]
        assert isinstance(source, nn.Linear) and source.in_features == record["K"] and source.out_features == record["N"]
        parent, attr = name.rsplit(".", 1)
        setattr(names[parent], attr, PaperLinear(record, name, implementation))
    assert len(pack["layers"]) == pack["meta"]["expected_layers"]
    print("EXPLICIT_SUBMISSION_CONTRACT_V2", implementation, len(pack["layers"]), "NO_OUTPUT_ROTATION_NO_WEIGHT_GROUPING_NO_FALLBACK", flush=True)
    return pack["meta"]
