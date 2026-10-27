"""Independent integer oracle and grid/stream/graph validation for CUTLASS S4."""

import pytest
import torch
from gr00t.quantization.perstep_backends import pack_int4
from research.holoq_bench.cutlass_s4 import CutlassLinear, extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA GPU required")


@pytest.mark.parametrize("m", [1, 49, 558])
def test_exact_integer_arithmetic_stream_and_graph(m):
    torch.manual_seed(32)
    n, k = 256, 512
    a = torch.randint(-7, 8, (m, k), dtype=torch.int32, device="cuda")
    b = torch.randint(-8, 8, (n, k), dtype=torch.int32, device="cuda")
    aa = pack_int4(a)
    bb = pack_int4(b)
    ext = extension()
    expected = a.cpu().long() @ b.cpu().long().t()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = ext.matmul(aa, bb)
    torch.cuda.current_stream().wait_stream(side)
    assert torch.equal(out.cpu(), expected.int())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        for _ in range(3):
            ext.matmul(aa, bb)
    side.synchronize()
    with torch.cuda.graph(graph, stream=side):
        y = ext.matmul(aa, bb)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y.cpu(), expected.int())


@pytest.mark.parametrize("m", [49, 558])
@pytest.mark.parametrize("smooth", [False, True])
def test_canonical_token_quantization_and_scale_contract(m, smooth):
    torch.manual_seed(33)
    n, k = 256, 512
    q = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    sw = (torch.rand(n, 1) * 0.03 + 0.01).half()
    sm = (torch.rand(k) * 2 + 0.3).half() if smooth else torch.empty(0)
    layer = CutlassLinear(
        dict(K=k, N=n, B=pack_int4(q), s=sw, smooth=sm, act_bits=4, act_mode="token", act_qmin=-7), "s4_contract"
    ).cuda()
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    xx = x.float()
    if smooth:
        xx = (xx / sm.cuda().float()).half().float()
    sa = (xx.abs().amax(1, keepdim=True).clamp_min(1e-5) / 7).half().float()
    qa = (xx / sa).round().clamp(-7, 7).int()
    exact = (qa.cpu().long() @ q.cpu().long().t()).cuda().float() * sw.cuda().float().t() * sa
    actual = layer(x)
    torch.testing.assert_close(actual, exact.half(), rtol=0, atol=0)
