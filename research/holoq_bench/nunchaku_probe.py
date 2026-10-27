"""Check the paper's actual Nunchaku backend without changing active policies.

Keep integer codes unchanged and repeat existing scales over its native G64.
This probe does not install a model adapter or select an inference backend.
"""

from functools import lru_cache
from pathlib import Path
import torch
import triton
import triton.language as tl
from research.holoq_bench.kernels import HALF_FLAGS

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load

    source = ROOT / "third_party/nunchaku"
    return load(
        name="holoq_nunchaku_int4",
        sources=[str(ROOT / "research/native_int4" / n) for n in ("nunchaku_bindings.cpp", "nunchaku_gemm.cu")],
        extra_include_paths=[str(source / "src"), str(source / "third_party/spdlog/include")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-DENABLE_BF16", "--expt-relaxed-constexpr", "--expt-extended-lambda"] + HALF_FLAGS,
        verbose=True,
    )


@triton.jit
def _codes(Q, P, ROWS: tl.constexpr, K: tl.constexpr, WEIGHT: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    reg = i % 4
    lane = i // 4 % 32
    if WEIGHT:
        tile = i // 128 % 8
        kg = i // 1024 % (K // 64)
        block = i // (1024 * (K // 64))
        ar = tl.where(reg == 1, 2, tl.where(reg == 2, 1, reg))
        row = block * 128 + tile * 16 + lane // 4 + (ar % 2) * 8
    else:
        tile = i // 128 % 2
        warp = i // 256 % 8
        kg = i // 2048 % (K // 64)
        block = i // (2048 * (K // 64))
        ar = reg
        row = block * 256 + warp * 32 + tile * 16 + lane // 4 + (ar % 2) * 8
    col = kg * 64 + (lane % 4) * 8 + (ar // 2) * 32
    value = tl.full((B,), 0, tl.int32)
    for nibble in range(8):
        q = tl.load(Q + row * K + col + nibble, (i < TOTAL) & (row < ROWS), 0).to(tl.int32)
        value = value | ((q & 15) << (4 * nibble))
    tl.store(P + i, value, i < TOTAL)


@triton.jit
def _scales(S, OUT, ROWS: tl.constexpr, PAD: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr,
            WEIGHT: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    if WEIGHT:
        el = i % 4
        lane = i // 4 % 32
        kg = i // 128 % (K // 64)
        block = i // (128 * (K // 64))
        row = block * 128 + lane // 4 * 16 + lane % 4 * 2 + el % 2 + el // 2 * 8
    else:
        el = i % 2
        lane = i // 2 % 16
        warp = i // 32 % 8
        kg = i // 256 % (K // 64)
        block = i // (256 * (K // 64))
        row = block * 256 + warp * 32 + lane // 8 * 16 + lane % 8 + el * 8
    value = tl.load(S + row * (K // GROUP) + kg * 64 // GROUP, (i < TOTAL) & (row < ROWS), 1)
    tl.store(OUT + i, value, i < TOTAL)


def pack(q, scales, group, weight=False):
    rows, k = q.shape
    pad = triton.cdiv(rows, 128 if weight else 256) * (128 if weight else 256)
    assert not weight or rows == pad
    packed = torch.empty((pad, k // 8), device=q.device, dtype=torch.int32)
    packed_scales = torch.empty((pad * k // 64,), device=q.device, dtype=torch.float16)
    _codes[(triton.cdiv(packed.numel(), 256),)](q, packed, rows, k, weight, packed.numel(), 256)
    _scales[(triton.cdiv(packed_scales.numel(), 256),)](
        scales, packed_scales, rows, pad, k, group, weight, packed_scales.numel(), 256
    )
    return packed, packed_scales


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = extension()
    torch.manual_seed(42)
    for m, n, k, group in [(1, 128, 128, 128), (16, 1024, 1024, 128), (49, 512, 2048, 2048), (559, 128, 8192, 128)]:
        a = torch.randint(-7, 8, (m, k), device="cuda", dtype=torch.int8)
        w = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
        asc = torch.rand((m, 1), device="cuda", dtype=torch.float16) * 0.02 + 0.01
        wsc = torch.rand((n, k // group), device="cuda", dtype=torch.float16) * 0.02 + 0.01
        ap, asp = pack(a, asc, k)
        wp, wsp = pack(w, wsc, group, True)
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        ext.gemm(ap, wp, asp, wsp, out)
        ref = (a.float() * asc.float()) @ (w.float() * wsc.float().repeat_interleave(group, 1)).T
        torch.cuda.synchronize()
        nmse = ((out.float() - ref).square().mean() / ref.square().mean()).item()
        print("NUNCHAKU_PROBE", m, n, k, group, "nmse", nmse, "finite", out.isfinite().all().item(), flush=True)
        assert torch.isfinite(out).all() and nmse < 1e-4
    print("NUNCHAKU_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
