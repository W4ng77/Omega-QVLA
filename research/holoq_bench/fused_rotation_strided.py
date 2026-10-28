"""Version 2 rotation: respect serialized matrix strides, including R.T.

The legacy plugin is deliberately retained to reproduce the failed experiment.
This backend is selected only by the explicit triton_block64_strided_v2 tag.
"""

import torch
import triton
import triton.language as tl
from research.holoq_bench.packed_runtime import PackedLinear
from research.holoq_bench.cutlass_s4 import CutlassLinear


@triton.jit
def _rotate(X, R, P, Y, M: tl.constexpr, K: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
            PERM: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    kk = tl.arange(0, 64)
    cols = block * 64 + kk
    input_cols = tl.load(P + cols) if PERM else cols
    x = tl.load(X + rows[:, None] * K + input_cols[None, :], rows[:, None] < M, 0)
    r = tl.load(R + block * S0 + kk[:, None] * S1 + kk[None, :] * S2)
    y = tl.dot(x, r).to(tl.float16)
    tl.store(Y + rows[:, None] * K + cols[None, :], y, rows[:, None] < M)


def rotate(x, blocks, perm=None):
    if x.dtype != torch.float16 or blocks.dtype != torch.float16:
        raise ValueError("Strided rotation requires FP16 operands")
    shape = x.shape
    k = shape[-1]
    assert k % 64 == 0 and blocks.shape == (k // 64, 64, 64)
    if perm is not None and perm.numel():
        assert perm.is_contiguous() and perm.numel() == k
    xx = x.reshape(-1, k).contiguous()
    out = torch.empty_like(xx)
    _rotate[(triton.cdiv(len(xx), 16), k // 64)](
        xx, blocks, perm, out, len(xx), k, *blocks.stride(),
        perm is not None and perm.numel() > 0, 16, num_warps=4,
    )
    return out.reshape(shape)


class _StridedMixin:
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


class StridedPackedLinear(_StridedMixin, PackedLinear):
    pass


class StridedCutlassLinear(_StridedMixin, CutlassLinear):
    pass


def install(model):
    count = 0
    for module in model.modules():
        if not isinstance(module, PackedLinear) or not module.rot_in.numel():
            continue
        module.__class__ = StridedCutlassLinear if isinstance(module, CutlassLinear) else StridedPackedLinear
        for src, dest in [("rot_in", "_fused_in"), ("rot_out", "_fused_out"), ("bias", "_fused_bias")]:
            module.register_buffer(dest, getattr(module, src))
            setattr(module, src, torch.empty(0, device=getattr(module, dest).device))
        count += 1
    if count not in (116, 180):
        raise RuntimeError(("Strided rotation layer count", count))
    print("EXPLICIT_STRIDED_BLOCK64_V2", count, flush=True)
