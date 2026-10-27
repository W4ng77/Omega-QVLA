"""Pinned QuaRot CUTLASS S4 GEMM with exact token/output scale contract."""

from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from research.holoq_bench.packed_runtime import PackedLinear, apply_blocks

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load

    source = ROOT / "third_party/QuaRot"
    dest = ROOT / ".kernel_sources/quarot"
    dest.mkdir(parents=True, exist_ok=True)
    original = (source / "quarot/kernels/gemm.cu").read_text()
    # Keep the official CUTLASS template and shape configuration unchanged.
    text = "#include <ATen/cuda/CUDAContext.h>\n" + original.replace(
        "gemmOp(arguments)", "gemmOp(arguments, nullptr, at::cuda::getCurrentCUDAStream())"
    )
    path = dest / "gemm.cu"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return load(
        name="holoq_quarot_s4_cuda",
        sources=[str(Path(__file__).with_name("cutlass_bindings.cpp")), str(path)],
        extra_include_paths=[str(source / "quarot/kernels/include"), str(source / "third-party/cutlass/include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=True,
    )


@triton.jit
def _pack(X, SM, A, S, K: tl.constexpr, SMOOTH: tl.constexpr, QMIN: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    lo = 2 * j
    even = tl.load(X + row * K + lo, lo < K, 0).to(tl.float32)
    odd = tl.load(X + row * K + lo + 1, lo + 1 < K, 0).to(tl.float32)
    if SMOOTH:
        even = tl.div_rn(even, tl.load(SM + lo, lo < K, 1).to(tl.float32))
        odd = tl.div_rn(odd, tl.load(SM + lo + 1, lo + 1 < K, 1).to(tl.float32))
    even = even.to(tl.float16).to(tl.float32)
    odd = odd.to(tl.float16).to(tl.float32)
    scale = (
        (tl.maximum(tl.maximum(tl.max(tl.abs(even)), tl.max(tl.abs(odd))), 1e-5) / 7).to(tl.float16).to(tl.float32)
    )
    qe = tl.minimum(tl.maximum(libdevice.nearbyint(tl.div_rn(even, scale)), QMIN), 7).to(tl.int32)
    qo = tl.minimum(tl.maximum(libdevice.nearbyint(tl.div_rn(odd, scale)), QMIN), 7).to(tl.int32)
    tl.store(A + row * (K // 2) + j, (qe & 15) | ((qo & 15) << 4), j < K // 2)
    tl.store(S + row, scale)


@triton.jit
def _rescale(C, AS, WS, Y, N: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(C + i, i < TOTAL, 0).to(tl.float32)
    value = value * tl.load(WS + i % N, i < TOTAL, 0).to(tl.float32)
    value = value * tl.load(AS + i // N, i < TOTAL, 0).to(tl.float32)
    tl.store(Y + i, value, i < TOTAL)


class CutlassLinear(PackedLinear):
    def __init__(self, record, name):
        nn.Module.__init__(self)
        assert record["act_mode"] == "token" and record["act_bits"] == 4
        assert record["s"].shape == (record["N"], 1)
        self.name = name
        self.backend = "cutlass_s4"
        self.in_features, self.out_features = record["K"], record["N"]
        self.act_qmin = record["act_qmin"]
        self.calls = 0
        self.input_shapes = set()
        for key in ("B", "s", "smooth", "rot_in", "rot_out", "perm", "bias"):
            value = record.get(key)
            self.register_buffer(key, value if value is not None else torch.empty(0))
        self._extension = extension()

    def forward(self, x):
        self.calls += 1
        self.input_shapes.add(tuple(x.shape))
        x = x.half()
        if self.rot_in.numel():
            x = apply_blocks(x, self.rot_in, self.perm)
        shape = x.shape
        xx = x.reshape(-1, self.in_features).contiguous()
        m, k = xx.shape
        aa = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
        asc = torch.empty(m, dtype=torch.float16, device=x.device)
        _pack[(m,)](
            xx,
            self.smooth,
            aa,
            asc,
            k,
            bool(self.smooth.numel()),
            self.act_qmin,
            triton.next_power_of_2(k // 2),
            num_warps=4 if k < 4096 else 8,
        )
        integers = self._extension.matmul(aa, self.B)
        y = torch.empty((m, self.out_features), dtype=torch.float16, device=x.device)
        _rescale[(triton.cdiv(y.numel(), 256),)](integers, asc, self.s, y, self.out_features, y.numel(), 256)
        y = y.reshape(*shape[:-1], self.out_features)
        if self.rot_out.numel():
            y = apply_blocks(y, self.rot_out)
        if self.bias.numel():
            y = y + self.bias
        return y
