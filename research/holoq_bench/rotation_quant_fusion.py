"""Explicit research prototype: input block rotation + static A4 in one launch.

Preserve the intermediate FP16 rotation rounding and FP32 per-column scale.
Not selected by any deployment loader. Requires its own complete validation.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from research.holoq_bench.fused_rotation_strided import StridedPackedLinear
from gr00t.quantization.dit_step_context import get_current_dit_step


@triton.jit
def _rotate_quant(X, R, P, A, Y, M: tl.constexpr, K: tl.constexpr,
                  S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                  QMIN: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    kk = tl.arange(0, 64)
    cols = block * 64 + kk
    input_cols = tl.load(P + cols)
    x = tl.load(X + rows[:, None] * K + input_cols[None, :], rows[:, None] < M, 0)
    r = tl.load(R + block * S0 + kk[:, None] * S1 + kk[None, :] * S2)
    # Same dot shape/operand order and explicit rounding as strided v2.
    rotated = tl.dot(x, r).to(tl.float16).to(tl.float32)
    a = tl.load(A + cols).to(tl.float32)
    q = libdevice.nearbyint(tl.div_rn(rotated, a[None, :]))
    q = tl.minimum(tl.maximum(q, QMIN), 7)
    y = (q * a[None, :]).to(tl.float16)
    tl.store(Y + rows[:, None] * K + cols[None, :], y, rows[:, None] < M)


def rotate_quant(x, blocks, perm, scale, qmin=-7):
    if x.dtype != torch.float16 or blocks.dtype != torch.float16 or scale.dtype != torch.float32:
        raise ValueError("Explicit FP16 rotation and FP32 scale contract required")
    shape = x.shape
    k = shape[-1]
    assert k % 64 == 0 and blocks.shape == (k // 64, 64, 64)
    assert perm.shape == (k,) and perm.is_contiguous() and scale.shape == (k,) and scale.is_contiguous()
    xx = x.reshape(-1, k).contiguous()
    out = torch.empty_like(xx)
    _rotate_quant[(triton.cdiv(len(xx), 16), k // 64)](xx, blocks, perm, scale, out, len(xx), k, *blocks.stride(), qmin, 16, num_warps=4)
    return out.reshape(shape)


class RotationQuantLinear(StridedPackedLinear):
    def forward(self, x):
        self.calls += 1
        self.input_shapes.add(tuple(x.shape))
        step = get_current_dit_step()
        if step is None or not 0 <= step < 8:
            raise RuntimeError("Missing paper denoising step")
        aa = rotate_quant(x.half(), self._fused_in, self.perm, self.a_scale[step], self.act_qmin)
        aa = aa.reshape(-1, self.in_features)
        y = torch.empty((len(aa), self.out_features), device=x.device, dtype=torch.float16)
        self._marlin.mul(aa, self.B, y, self.s, self.workspace)
        y = y.reshape(*x.shape[:-1], self.out_features)
        return y + self._fused_bias if self._fused_bias.numel() else y


def install(policy):
    count = 0
    for name, module in policy.model.named_modules():
        if not name.startswith("action_head.") or not isinstance(module, StridedPackedLinear):
            continue
        if module.backend != "marlin" or module.act_bits != 4 or module.act_mode != "step":
            raise ValueError("Prototype requires explicitly selected paper original-grid DiT")
        if module._fused_out.numel() or module.smooth.numel():
            raise ValueError("Input-only unsmoothed paper variant required")
        module.__class__ = RotationQuantLinear
        count += 1
    if count != 96:
        raise RuntimeError(("Unexpected prototype coverage", count))
    graphs = policy.model.action_head.get_action
    if not hasattr(graphs, "cache") or not hasattr(graphs, "original"):
        raise ValueError("Explicit head Graph adapter required for this paired probe")
    # Captured graphs retain the old kernels. Force recapture during warmup;
    # otherwise a prototype could report an entirely unchanged execution.
    torch.cuda.synchronize()
    graphs.cache.clear()
    print("EXPLICIT_RESEARCH_ROTATION_STATICQ_FUSION", count, flush=True)
    return {"name": "research_input_rotation_static_A4_fusion_96_DiT", "logits_shapes": set()}
