from functools import lru_cache
from pathlib import Path

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
        name="holoq_pi05_definition_s4",
        sources=[str(Path(__file__).with_name("cutlass_bindings.cpp")), str(path)],
        extra_include_paths=[str(source / "quarot/kernels/include"), str(source / "third-party/cutlass/include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=True,
    )
