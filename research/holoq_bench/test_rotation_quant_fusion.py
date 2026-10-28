"""Compare fused preparation directly with both original GPU kernels."""
import pytest
import torch
import triton
from research.holoq_bench.fused_rotation_strided import rotate
from research.holoq_bench.packed_runtime import _prepare
from research.holoq_bench.rotation_quant_fusion import rotate_quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required")


@pytest.mark.parametrize("m,k", [(1, 1536), (49, 1536), (49, 6144)])
@pytest.mark.parametrize("transposed", [False, True])
def test_same_fp16_rotation_rounding_static_quantization_current_stream_and_graph(m, k, transposed):
    torch.manual_seed(61)
    x = torch.randn((1, m, k), device="cuda", dtype=torch.float16) * 3
    r = (torch.randn((k // 64, 64, 64), device="cuda") / 8).half()
    if transposed:
        r = r.transpose(-1, -2)
    p = torch.randperm(k, device="cuda")
    scale = torch.logspace(-4, 0, k, device="cuda")
    rotated = rotate(x, r, p).reshape(m, k)
    expected = torch.empty_like(rotated)
    empty = torch.empty(0, device="cuda", dtype=torch.float16)
    _prepare[(m,)](rotated, empty, scale, expected, empty, k, 2, 4, False, False, -7, triton.next_power_of_2(k), num_warps=4 if k < 4096 else 8)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        actual = rotate_quant(x, r, p, scale)
    torch.cuda.current_stream().wait_stream(side)
    torch.testing.assert_close(actual.reshape(m, k), expected, rtol=0, atol=0)
    for _ in range(3):
        rotate_quant(x, r, p, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = rotate_quant(x, r, p, scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
