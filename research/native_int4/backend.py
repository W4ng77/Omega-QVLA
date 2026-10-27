"""Research-only native S4 MMA backend for the existing integer per-step formats.

Preserves exported INT4 codes and FP32 group scales. A is quantized once and
packed into PTX fragments. B is reordered once before timing, without requant.
No changes are made to the separately running LIBERO checkout.
"""

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load

    path = Path(__file__).resolve().parent
    return load(
        name="omega_native_s4_sm89",
        sources=[str(path / "native.cpp"), str(path / "native.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        verbose=True,
    )


@triton.jit
def _pack_a(
    X,
    A,
    OUT,
    M: tl.constexpr,
    K: tl.constexpr,
    KT: tl.constexpr,
    G: tl.constexpr,
    GROUPED: tl.constexpr,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
    QMIN: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    lane = i % 32
    reg = (i // 32) % 4
    kb = (i // 128) % KT
    mb = i // (128 * KT)
    row = mb * 16 + lane // 4 + (reg % 2) * 8
    k0 = kb * 64 + (lane % 4) * 8 + (reg // 2) * 32
    j = tl.arange(0, 8)
    kk = k0[:, None] + j[None, :]
    mask = (i[:, None] < TOTAL) & (row[:, None] < M) & (kk < K)
    x = tl.load(X + row[:, None] * K + kk, mask, 0).to(tl.float32)
    if GROUPED:
        idx = kk // G
    else:
        idx = kk
    a = tl.load(A + idx, mask, 1)
    q = tl.minimum(tl.maximum(libdevice.nearbyint(tl.div_rn(x, a)), QMIN), 7.0).to(tl.int32)
    word = tl.sum((q & 15) << (j[None, :] * 4), axis=1)
    tl.store(OUT + i, word, i < TOTAL)


@triton.jit
def _pack_b(
    P,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    PK: tl.constexpr,
    KT: tl.constexpr,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    lane = i % 32
    reg = (i // 32) % 2
    kb = (i // 64) % KT
    nb = i // (64 * KT)
    n = nb * 8 + lane // 4
    byte0 = kb * 32 + (lane % 4) * 4 + reg * 16
    j = tl.arange(0, 4)
    pos = byte0[:, None] + j[None, :]
    p = tl.load(P + n[:, None] * PK + pos, (i[:, None] < TOTAL) & (n[:, None] < N) & (pos < PK), 0).to(tl.int32)
    word = tl.sum(p << (j[None, :] * 8), axis=1)
    tl.store(OUT + i, word, i < TOTAL)


@triton.jit
def _reduce(P, Y, TOTAL: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for split in range(SK):
        acc += tl.load(P + split * TOTAL + i, i < TOTAL, 0)
    tl.store(Y + i, acc, i < TOTAL)


def pack_weight(packed, n, k):
    if packed.dtype != torch.uint8 or packed.shape != (n, triton.cdiv(k, 2)):
        raise ValueError("Expected [N,ceil(K/2)] signed-nibble source weights")
    kt = triton.cdiv(k, 64)
    out = torch.empty((triton.cdiv(n, 8), kt, 2, 32), device=packed.device, dtype=torch.int32)
    _pack_b[(triton.cdiv(out.numel(), 128),)](packed, out, n, k, packed.shape[1], kt, out.numel(), 128)
    return out


def pack_activation(x, act, group, grouped, qmin=-8):
    if not x.is_contiguous() or act.dtype != torch.float32:
        raise ValueError("Expected contiguous inputs and FP32 scales")
    m, k = x.shape
    kt = triton.cdiv(k, 64)
    out = torch.empty((triton.cdiv(m, 16), kt, 4, 32), device=x.device, dtype=torch.int32)
    _pack_a[(triton.cdiv(out.numel(), 128),)](x, act, out, m, k, kt, group, grouped, out.numel(), 128, qmin)
    return out


def from_packed(a, b, scales_t, act, m, n, k, group, mode, warps_n=4, split_k=1):
    partial = extension().group_mma(a, b, scales_t, act, m, n, k, group, mode, warps_n, split_k)
    if split_k == 1:
        return partial[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    _reduce[(triton.cdiv(m * n, 256),)](partial, out, m * n, split_k, 256)
    return out


def linear(x, b, scales_t, act, n, group, mode, warps_n=4, split_k=1, qmin=-8):
    m, k = x.shape
    a = pack_activation(x, act, group, mode == 2, qmin)
    return from_packed(a, b, scales_t, act, m, n, k, group, mode, warps_n, split_k)


def cached_linear(x, packed, scales, act, group, mode, cache, qmin=-8):
    """Full-runtime adapter: prepack selected banks once, preserve leading dims."""
    if torch.cuda.get_device_capability(x.device) != (8, 9):
        raise RuntimeError("This native S4 integration is validated only on SM89")
    if group not in (64, 128, 256):
        raise ValueError("Native S4 supports groups 64/128/256")
    key = (packed.device, packed.data_ptr(), scales.data_ptr())
    if key not in cache:
        n, k = packed.shape[0], x.shape[-1]
        cache[key] = (pack_weight(packed, n, k), scales.t().contiguous())
    b, st = cache[key]
    shape = x.shape
    xx = x.reshape(-1, shape[-1]).contiguous()
    m, k = xx.shape
    n = packed.shape[0]
    if k >= 4096:
        wn, sk = (4, 8) if m <= 32 else ((4, 4) if m <= 64 else (8, 2))
    elif n >= 4096:
        wn, sk = (8, 2) if m <= 32 else ((8, 1) if m <= 64 else (2, 1))
    else:
        wn, sk = (4, 4) if m <= 64 else (8, 1)
    sk = min(sk, scales.shape[1])
    y = linear(xx, b, st, act, n, group, 2 if mode == "group_int" else 1, wn, sk, qmin)
    return y.reshape(*shape[:-1], n)
