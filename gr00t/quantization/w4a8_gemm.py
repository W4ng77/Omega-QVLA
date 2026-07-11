"""Real W4A8 GEMM: int4-packed weights, int8 activations, INT8 tensor cores.

Weights are stored packed (two signed 4-bit values per uint8, +8 offset):
    pack[n, k2] = (w[n, 2*k2] + 8) | ((w[n, 2*k2+1] + 8) << 4)
with a per-row fp32 scale s_w[n]. Activations are quantized per token:
    a_i8[m, k] = round(x[m, k] / s_a[m]),  s_a[m] = amax_k |x[m, k]| / 127.

The kernel splits the K reduction into even/odd columns so the packed weight
tile can be used directly with two int8 tl.dot calls (int32 accumulate):
    acc = dot(a_even, w_lo^T) + dot(a_odd, w_hi^T)
Epilogue: out[m, n] = acc * s_a[m] * s_w[n] (+ bias), stored in x.dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a8_gemm_kernel(
    a_ptr, pack_ptr, sa_ptr, sw_ptr, bias_ptr, out_ptr,
    M, N, K2,                      # K2 = K // 2
    stride_am, stride_pn, stride_om,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK2: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    m_mask = rm < M
    n_mask = rn < N

    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k2 in range(0, K2, BK2):
        rk2 = k2 + tl.arange(0, BK2)
        k_mask = rk2 < K2
        # a_even[m, k2] = a[m, 2*k2], a_odd[m, k2] = a[m, 2*k2+1]
        a_even = tl.load(
            a_ptr + rm[:, None] * stride_am + (rk2 * 2)[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0,
        )
        a_odd = tl.load(
            a_ptr + rm[:, None] * stride_am + (rk2 * 2 + 1)[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0,
        )
        p = tl.load(
            pack_ptr + rn[:, None] * stride_pn + rk2[None, :],
            mask=n_mask[:, None] & k_mask[None, :], other=136,  # 136 = (0+8)|((0+8)<<4)
        )
        w_lo = (p & 0xF).to(tl.int8) - 8
        w_hi = (p >> 4).to(tl.int8) - 8
        acc += tl.dot(a_even, tl.trans(w_lo), out_dtype=tl.int32)
        acc += tl.dot(a_odd, tl.trans(w_hi), out_dtype=tl.int32)

    sa = tl.load(sa_ptr + rm, mask=m_mask, other=0.0)
    sw = tl.load(sw_ptr + rn, mask=n_mask, other=0.0)
    y = acc.to(tl.float32) * sa[:, None] * sw[None, :]
    if HAS_BIAS:
        b = tl.load(bias_ptr + rn, mask=n_mask, other=0.0)
        y += b[None, :]
    tl.store(
        out_ptr + rm[:, None] * stride_om + rn[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@torch.no_grad()
def pack_int4_rows(w: torch.Tensor):
    """Per-row symmetric int4 quantization + nibble packing.
    Returns (pack[N, K//2] uint8, s_w[N] fp32)."""
    assert w.dim() == 2 and w.shape[1] % 2 == 0
    wf = w.detach().float()
    s = (wf.abs().amax(dim=1) / 7.0).clamp_min(1e-12)
    q = torch.clamp(torch.round(wf / s[:, None]), -8, 7).to(torch.int16) + 8  # [0..15]
    pack = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).contiguous()
    return pack, s.to(torch.float32).contiguous()


def w4a8_linear(x2d: torch.Tensor, pack: torch.Tensor, s_w: torch.Tensor,
                bias: torch.Tensor | None, a_qmax: float = 127.0) -> torch.Tensor:
    """x2d [M, K] float -> [M, N] in x2d.dtype. Per-token dynamic A8."""
    M, K = x2d.shape
    N, K2 = pack.shape
    assert K == K2 * 2
    s_a = (x2d.detach().float().abs().amax(dim=1) / a_qmax).clamp_min(1e-12)
    a_i8 = torch.clamp(torch.round(x2d.float() / s_a[:, None]), -a_qmax - 1, a_qmax).to(torch.int8).contiguous()
    out = torch.empty(M, N, dtype=x2d.dtype, device=x2d.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _w4a8_gemm_kernel[grid](
        a_i8, pack, s_a.to(torch.float32), s_w,
        bias if bias is not None else s_w,  # dummy ptr when no bias
        out,
        M, N, K2,
        a_i8.stride(0), pack.stride(0), out.stride(0),
        HAS_BIAS=bias is not None,
        BM=64, BN=64, BK2=64,
        num_warps=4,
    )
    return out
