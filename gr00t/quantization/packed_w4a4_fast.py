"""Arithmetic W4 decoding and fused A4 quantization/packing, BF16 MMA.

The original calibrated scales, codes and stored-dtype rounding are retained.
Sparse codebook exceptions are handled separately from normal symmetric groups.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _quant_pack_a4(X, S, P, K: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    k = (2 * i) % K
    s0 = tl.load(S + k, i < TOTAL, other=1).to(tl.float32)
    s1 = tl.load(S + k + 1, i < TOTAL, other=1).to(tl.float32)
    x0 = tl.load(X + 2 * i, i < TOTAL, other=0)
    x1 = tl.load(X + 2 * i + 1, i < TOTAL, other=0)
    # PyTorch division returns the input dtype. Preserve that rounding before
    # round-to-integer; otherwise BF16 half-integer thresholds change A4 codes.
    d0 = tl.div_rn(x0.to(tl.float32), s0).to(X.dtype.element_ty).to(tl.float32)
    d1 = tl.div_rn(x1.to(tl.float32), s1).to(X.dtype.element_ty).to(tl.float32)
    q0 = tl.minimum(tl.maximum(libdevice.rint(d0), -8), 7).to(tl.int32) + 8
    q1 = tl.minimum(tl.maximum(libdevice.rint(d1), -8), 7).to(tl.int32) + 8
    tl.store(P + i, (q0 | (q1 << 4)).to(tl.uint8), i < TOTAL)


@triton.jit
def _gemm_a4w4_arithmetic(
    A, W, SW, FMAP, FCBOOK, SA, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
    A_FP16: tl.constexpr, A_BF16: tl.constexpr,
    W_STORED_FP16: tl.constexpr, W_STORED_BF16: tl.constexpr,
    HAS_FALLBACK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        pa = tl.load(A + m[:, None] * (K // 2) + k[None, :] // 2,
                     (m[:, None] < M) & (k[None, :] < K), other=136)
        qa = ((pa.to(tl.int32) >> ((k[None, :] % 2) * 4)) & 15) - 8
        sa = tl.load(SA + k, k < K, other=0).to(tl.float32)
        a = qa.to(tl.float32) * sa[None, :]
        if A_FP16:
            a = a.to(tl.float16).to(tl.float32)
        elif A_BF16:
            a = a.to(tl.bfloat16).to(tl.float32)

        # Load row-major W as [Ntile,Ktile] for contiguous packed K accesses.
        mask = (n[:, None] < N) & (k[None, :] < K)
        pw = tl.load(W + n[:, None] * (K // 2) + k[None, :] // 2, mask, other=136)
        code = (pw.to(tl.int32) >> ((k[None, :] % 2) * 4)) & 15
        # A reduction tile stays inside one weight group when BK divides G.
        # Load scale/map once per output row rather than once per weight.
        group = n * (K // G) + (start * BK) // G
        sw = tl.load(SW + group, n < N, other=0).to(tl.float32)
        w = (code - 8).to(tl.float32) * sw[:, None]
        if W_STORED_FP16:
            w = w.to(tl.float16).to(tl.float32)
        elif W_STORED_BF16:
            w = w.to(tl.bfloat16).to(tl.float32)
        if HAS_FALLBACK:
            fallback_id = tl.load(FMAP + group, n < N, other=-1)
            special = tl.load(FCBOOK + fallback_id[:, None] * 16 + code,
                              mask & (fallback_id[:, None] >= 0), other=0).to(tl.float32)
            w = tl.where(fallback_id[:, None] >= 0, special, w)
        acc += tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
    tl.store(Y + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def launch_gemm(packed_a, packed_w, scales, fallback_map, fallback_codebook, sa,
                m, group_size, stored_kind, *, config=None):
    """Shared launch entry for the operator and shape-specific measurements."""
    n, k2 = packed_w.shape
    if config is None:
        config = (16, 64, 128, 4, 2) if m <= 16 else (32, 64, 64, 4, 2)
    bm, bn, bk, warps, stages = config
    if group_size % bk:
        raise ValueError("GEMM BK must divide the weight group size")
    out = torch.empty((m, n), device=packed_a.device, dtype=torch.bfloat16)
    _gemm_a4w4_arithmetic[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        packed_a, packed_w, scales, fallback_map, fallback_codebook, sa, out,
        m, n, k2 * 2, group_size, sa.dtype == torch.float16, sa.dtype == torch.bfloat16,
        stored_kind == 0, stored_kind == 1, fallback_codebook.numel() > 0,
        bm, bn, bk, num_warps=warps, num_stages=stages,
    )
    return out


@torch.library.custom_op("gr00t::packed_w4a4_fast_bf16_gemm", mutates_args=())
def packed_w4a4_fast_linear(
    x: torch.Tensor, sa: torch.Tensor, packed_w: torch.Tensor, scales: torch.Tensor,
    fallback_map: torch.Tensor, fallback_codebook: torch.Tensor,
    group_size: int, stored_kind: int,
) -> torch.Tensor:
    m, k = x.shape
    if not x.is_cuda or not x.is_contiguous() or sa.shape != (k,) or k % 2 or k % group_size:
        raise ValueError("Expected CUDA contiguous [M,K], even grouped K, and K activation scales")
    if x.dtype != sa.dtype:
        raise ValueError("Input and scale must have matching dtype for reference division rounding")
    packed_a = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    _quant_pack_a4[(triton.cdiv(m * k // 2, 256),)](x, sa, packed_a, k, m * k // 2, 256)
    return launch_gemm(packed_a, packed_w, scales, fallback_map, fallback_codebook,
                       sa, m, group_size, stored_kind)


@packed_w4a4_fast_linear.register_fake
def _fake(x, sa, packed_w, scales, fallback_map, fallback_codebook, group_size, stored_kind):
    return x.new_empty((x.shape[0], packed_w.shape[0]), dtype=torch.bfloat16)


def symmetric_metadata(record, model_dtype):
    if record["encoding"] != "int4_symmetric_v1":
        raise ValueError("Arithmetic path requires a symmetric W4 pack")
    scales = record["scales"].float().contiguous()
    fallback_map = torch.full((scales.numel(),), -1, dtype=torch.int32)
    ids = record["fallback_indices"].long()
    fallback_map[ids] = torch.arange(ids.numel(), dtype=torch.int32)
    fallback_codebook = record["fallback_codebooks"].to(model_dtype).contiguous()
    source_dtype = record["fallback_codebooks"].dtype
    kind = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[source_dtype]
    return scales, fallback_map, fallback_codebook, kind


@triton.jit
def _decode_w4(W, SW, FMAP, FTABLE, OUT, TOTAL: tl.constexpr, G: tl.constexpr,
               KIND: tl.constexpr, HAS_FALLBACK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    p = tl.load(W + i // 2, i < TOTAL, other=136).to(tl.int32)
    code = (p >> ((i % 2) * 4)) & 15
    group = i // G
    scale = tl.load(SW + group, i < TOTAL, other=0).to(tl.float32)
    w = (code - 8).to(tl.float32) * scale
    if KIND == 0:
        w = w.to(tl.float16).to(tl.float32)
    elif KIND == 1:
        w = w.to(tl.bfloat16).to(tl.float32)
    if HAS_FALLBACK:
        f = tl.load(FMAP + group, i < TOTAL, other=-1)
        special = tl.load(FTABLE + f * 16 + code, (i < TOTAL) & (f >= 0), other=0).to(tl.float32)
        w = tl.where(f >= 0, special, w)
    tl.store(OUT + i, w, i < TOTAL)


@triton.jit
def _decode_a4(A, SA, OUT, K: tl.constexpr, TOTAL: tl.constexpr,
               A_FP16: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    p = tl.load(A + i // 2, i < TOTAL, other=136).to(tl.int32)
    q = ((p >> ((i % 2) * 4)) & 15) - 8
    scale = tl.load(SA + i % K, i < TOTAL, other=0).to(tl.float32)
    a = q.to(tl.float32) * scale
    if A_FP16:
        a = a.to(tl.float16).to(tl.float32)
    tl.store(OUT + i, a, i < TOTAL)


@torch.library.custom_op("gr00t::packed_w4a4_decode_cublas", mutates_args=())
def packed_w4a4_decode_cublas(
    x: torch.Tensor, sa: torch.Tensor, packed_w: torch.Tensor, scales: torch.Tensor,
    fallback_map: torch.Tensor, fallback_codebook: torch.Tensor,
    group_size: int, stored_kind: int,
) -> torch.Tensor:
    """A4/W4 stored codes, transient BF16 decoding, then the reference GEMM.

    This alternative trades transient workspace for cuBLAS throughput and its
    reference reduction order. It never changes A4 into unquantized A16.
    """
    m, k = x.shape
    n = packed_w.shape[0]
    if not x.is_cuda or not x.is_contiguous() or x.dtype != sa.dtype or sa.shape != (k,):
        raise ValueError("Expected CUDA contiguous [M,K] with matching K scale dtype")
    packed_a = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    a = torch.empty((m, k), device=x.device, dtype=torch.bfloat16)
    w = torch.empty((n, k), device=x.device, dtype=torch.bfloat16)
    _quant_pack_a4[(triton.cdiv(m * k // 2, 256),)](x, sa, packed_a, k, m * k // 2, 256)
    _decode_a4[(triton.cdiv(m * k, 1024),)](packed_a, sa, a, k, m * k, x.dtype == torch.float16, 1024)
    _decode_w4[(triton.cdiv(n * k, 1024),)](packed_w, scales, fallback_map, fallback_codebook, w,
                                        n * k, group_size, stored_kind, fallback_codebook.numel() > 0, 1024)
    return torch.nn.functional.linear(a, w)


@packed_w4a4_decode_cublas.register_fake
def _fake_cublas(x, sa, packed_w, scales, fallback_map, fallback_codebook, group_size, stored_kind):
    return x.new_empty((x.shape[0], packed_w.shape[0]), dtype=torch.bfloat16)
