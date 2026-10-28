"""Pinned official packed kernels; JIT builds without replacing the Torch env."""

from functools import lru_cache
import importlib.util
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HALF_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
]


@lru_cache(None)
def marlin():
    from torch.utils.cpp_extension import load

    src = ROOT / "third_party/marlin/marlin"
    ext = load(
        name="holoq_marlin_cuda",
        sources=[str(src / "marlin_cuda.cpp"), str(src / "marlin_cuda_kernel.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    sys.modules["marlin_cuda"] = ext
    spec = importlib.util.spec_from_file_location("holoq_official_marlin", src / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_stream_copy(source, destination):
    """Copy official CUDA sources and patch only legacy default-stream launches."""
    for original in source.rglob("*"):
        if not original.is_file():
            continue
        path = destination / original.relative_to(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = original.read_text()
        if path.suffix != ".cu":
            if not path.exists() or path.read_text() != text:
                path.write_text(text)
            continue
        if "#include <ATen/cuda/CUDAContext.h>" not in text:
            text = "#include <ATen/cuda/CUDAContext.h>\n" + text

        def patch(match):
            args = match.group(1).split(",")
            if len(args) == 3:
                return "<<<" + match.group(1) + ", at::cuda::getCurrentCUDAStream()>>>"
            if len(args) == 2:
                return "<<<" + match.group(1) + ", 0, at::cuda::getCurrentCUDAStream()>>>"
            return match.group(0)

        text = re.sub(r"<<<([^<>\n]+)>>>", patch, text)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)


@lru_cache(None)
def awq():
    from torch.utils.cpp_extension import load

    source = ROOT / "third_party/llm-awq/awq/kernels/csrc/quantization_new"
    dest = ROOT / ".kernel_sources/awq"
    current_stream_copy(source, dest)
    cpp = dest / "bindings.cpp"
    binding = """#include <torch/extension.h>
#include "gemm/gemm_cuda.h"
#include "gemv/gemv_cuda.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("gemm_forward_cuda_new", &gemm_forward_cuda_new);
 m.def("gemv_forward_cuda_new", &gemv_forward_cuda_new);
}
"""
    if not cpp.exists() or cpp.read_text() != binding:
        cpp.write_text(binding)
    return load(
        name="holoq_awq_cuda",
        sources=[str(cpp), str(dest / "gemm/gemm_cuda.cu"), str(dest / "gemv/gemv_cuda.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"] + HALF_FLAGS,
        verbose=True,
    )


@lru_cache(None)
def qserve():
    from torch.utils.cpp_extension import load

    source = ROOT / "third_party/omniserve/kernels/csrc/qgemm/w4a8_per_group"
    dest = ROOT / ".kernel_sources/qserve"
    current_stream_copy(source, dest)
    return load(
        name="holoq_qserve_cuda",
        sources=[str(dest / "pybind.cpp"), str(dest / "gemm_cuda.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"] + HALF_FLAGS,
        verbose=True,
    )


def pack_marlin(q, scales):
    """Signed codes [N,K], scales [N,G]; preserve codes in the official layout."""
    from types import SimpleNamespace

    n, k = q.shape
    group = k // scales.shape[1]
    if group not in (k, 128):
        raise ValueError("Official Marlin requires per-channel or G128 scales")
    mod = marlin()
    layer = mod.Layer(k, n, -1 if group == k else group)
    s = scales.half().cpu()
    w = (q.cpu().float() * s.float().repeat_interleave(group, 1)).half()
    recovered = (w.float() / s.float().repeat_interleave(group, 1)).round().int()
    if not torch.equal(recovered, q.cpu().int()):
        raise ValueError("FP16 parameter conversion would change weight codes")
    layer.pack(SimpleNamespace(weight=w), s)
    return {"backend": "marlin", "B": layer.B.cpu(), "s": layer.s.cpu(), "group": group}


def pack_awq(q, scales, zeros, dtype=torch.float16):
    """Official AWQ uint4 interleave; vectorize only the code preparation."""
    awq()
    spec = importlib.util.spec_from_file_location(
        "holoq_awq_packer", ROOT / "third_party/llm-awq/awq/quantize/qmodule.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["awq_inference_engine"] = awq()
    spec.loader.exec_module(module)
    n, k = q.shape
    group = k // scales.shape[1]
    if group != 128:
        raise ValueError("Official AWQ selected kernel requires G128")
    padded = ((scales.shape[1] + 7) // 8) * 8
    s = torch.zeros((padded, n), dtype=dtype)
    z = torch.zeros_like(s)
    s[: scales.shape[1]] = scales.to(dtype).cpu().t()
    z[: scales.shape[1]] = -(scales.to(dtype).cpu() * zeros.cpu()).to(dtype).t()
    return {"backend": "awq", "B": module.pack_intweight(q.cpu().int(), 4, 64), "s": s, "z": z, "group": group}


def pack_qserve(q, scales):
    """Per-output symmetric W4 represented exactly by QServe's integer G128 path."""
    n, k = q.shape
    if scales.shape != (n, 1):
        raise ValueError("QServe adapter requires a single W scale per output")
    qq = q.cpu().int() + 8
    if qq.min() < 0 or qq.max() > 15:
        raise ValueError("Expected signed INT4 codes")
    w = qq.reshape(n // 32, 2, 2, 8, k // 32, 2, 4, 4).permute(0, 4, 3, 6, 1, 5, 2, 7).contiguous()
    w = w.permute(0, 1, 2, 3, 5, 6, 7, 4).contiguous().to(torch.int8)
    b = ((w[..., 1] << 4) + w[..., 0]).reshape(n, k // 2)
    return {
        "backend": "qserve",
        "B": b,
        "s": scales.half().cpu().flatten(),
        "s2": torch.ones((k // 128, n), dtype=torch.int8),
        "z2": torch.full((k // 128, n), -8, dtype=torch.int8),
        "group": 128,
    }


if __name__ == "__main__":
    for name, fn in (("marlin", marlin), ("awq", awq), ("qserve", qserve)):
        print("BUILD", name, flush=True)
        fn()
        print("READY", name, flush=True)
