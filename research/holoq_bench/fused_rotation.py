"""Explicit block64 rotation backend, preserving all exported coefficients.

Fuse input permutation, block product and contiguous output layout. Existing
BMM packs never enter this plugin. The integer/Marlin operator is unchanged.
"""

import torch
import triton
import triton.language as tl
from research.holoq_bench.packed_runtime import PackedLinear
from research.holoq_bench.cutlass_s4 import CutlassLinear


@triton.jit
def _rotate(X, R, P, Y, M: tl.constexpr, K: tl.constexpr, PERM: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    kk = tl.arange(0, 64)
    cols = block * 64 + kk
    if PERM:
        input_cols = tl.load(P + cols)
    else:
        input_cols = cols
    x = tl.load(X + rows[:, None] * K + input_cols[None, :], rows[:, None] < M, 0)
    r = tl.load(R + block * 4096 + kk[:, None] * 64 + kk[None, :])
    y = tl.dot(x, r).to(tl.float16)
    tl.store(Y + rows[:, None] * K + cols[None, :], y, rows[:, None] < M)


def rotate(x, blocks, perm=None):
    if x.dtype != torch.float16 or blocks.dtype != torch.float16:
        raise ValueError("Fused rotation explicitly requires FP16 operands")
    shape = x.shape
    k = shape[-1]
    assert k % 64 == 0 and blocks.shape == (k // 64, 64, 64)
    xx = x.reshape(-1, k).contiguous()
    out = torch.empty_like(xx)
    _rotate[(triton.cdiv(len(xx), 16), k // 64)](
        xx, blocks, perm, out, len(xx), k, perm is not None and perm.numel() > 0, 16, num_warps=4
    )
    return out.reshape(shape)


class _FusedMixin:
    def forward(self, x):
        x = x.half()
        if self._fused_in.numel():
            x = rotate(x, self._fused_in, self.perm)
        y = super().forward(x)
        if self._fused_out.numel():
            y = rotate(y, self._fused_out)
        if self._fused_bias.numel():
            y = y + self._fused_bias
        return y


class FusedPackedLinear(_FusedMixin, PackedLinear):
    pass


class FusedCutlassLinear(_FusedMixin, CutlassLinear):
    pass


def install(model):
    count = 0
    for module in model.modules():
        if not isinstance(module, PackedLinear) or not module.rot_in.numel():
            continue
        module.__class__ = FusedCutlassLinear if isinstance(module, CutlassLinear) else FusedPackedLinear
        for src, dest in [("rot_in", "_fused_in"), ("rot_out", "_fused_out"), ("bias", "_fused_bias")]:
            module.register_buffer(dest, getattr(module, src))
            setattr(module, src, torch.empty(0))
        count += 1
    if count not in (116, 180):
        raise RuntimeError(("Fused rotation layer count", count))
    print("EXPLICIT_FUSED_BLOCK64", count, flush=True)
