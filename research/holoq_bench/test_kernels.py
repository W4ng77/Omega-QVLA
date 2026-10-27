"""Independent arithmetic, stream and graph checks before policy evaluation."""

import pytest
import torch

from gr00t.quantization.perstep_backends import pack_int4
from research.holoq_bench.kernels import pack_awq, pack_marlin, pack_qserve
from research.holoq_bench.packed_runtime import PackedLinear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual GPU required")


@pytest.mark.parametrize("backend", ["marlin", "awq", "qserve", "tile_fp16"])
@pytest.mark.parametrize("m", [1, 32, 49, 128, 384])
def test_packed_arithmetic_stream_and_graph(backend, m):
    torch.manual_seed(13)
    n, k, group = 256, 512, 128
    q = torch.randint(-7, 8, (n, k), dtype=torch.int8)
    s = (torch.rand(n, 4) * 0.04 + 0.005).half()
    if backend == "qserve":
        s = s[:, :1]
        packed = pack_qserve(q, s)
    elif backend == "awq":
        zero = torch.randint(4, 12, (n, 4))
        qu = torch.randint(0, 16, (n, k), dtype=torch.int32)
        q = qu - zero.repeat_interleave(group, 1)
        packed = pack_awq(qu, s, zero)
    elif backend == "marlin":
        packed = pack_marlin(q, s)
    else:
        packed = {"backend": backend, "B": pack_int4(q), "s": s.float(), "group": group}
    bits = 8 if backend == "qserve" else 16
    record = dict(
        packed,
        K=k,
        N=n,
        act_bits=bits,
        act_mode="token" if bits < 16 else "none",
        act_qmin=-127 if bits == 8 else -32768,
    )
    layer = PackedLinear(record, "test").cuda()
    x = torch.randn(1, m, k, device="cuda", dtype=torch.float16)
    xx = x.float().reshape(m, k)
    if bits == 8:
        a = (xx.abs().amax(1, keepdim=True).clamp_min(1e-5) / 127).half().float()
        xx = (xx / a).round().clamp(-127, 127) * a
    w = q.float() * s.float().repeat_interleave(k // s.shape[1], 1)
    oracle = (xx @ w.cuda().t()).half().reshape(1, m, n)
    result = layer(x)
    nmse = ((result.float() - oracle.float()).square().sum() / oracle.float().square().sum()).item()
    assert torch.isfinite(result).all() and nmse < 1e-4, (backend, m, nmse)
    assert result.dtype == x.dtype and result.shape == (1, m, n)
    assert not hasattr(layer, "weight"), "Dense weight must not persist"
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        streamed = layer(x)
    torch.cuda.current_stream().wait_stream(side)
    torch.testing.assert_close(streamed, result, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        for _ in range(3):
            layer(x)
    side.synchronize()
    with torch.cuda.graph(graph, stream=side):
        graphed = layer(x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graphed, result, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 32, 49, 384])
def test_marlin_per_output_scale_contract(m):
    torch.manual_seed(39)
    n, k = 256, 512
    q = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    s = (torch.rand(n, 1) * 0.02 + 0.01).half()
    layer = PackedLinear(
        dict(
            pack_marlin(q, s),
            K=k,
            N=n,
            act_bits=4,
            act_mode="channel",
            act_qmin=-7,
            a_scale=torch.rand(k) * 0.01 + 0.01,
        ),
        "per_output",
    ).cuda()
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    aa = ((x.float() / layer.a_scale).round().clamp(-7, 7) * layer.a_scale).half()
    ww = (q.float() * s.float()).half().cuda()
    expected = (aa.float() @ ww.float().t()).half()
    actual = layer(x)
    nmse = (actual.float() - expected.float()).square().sum() / expected.float().square().sum()
    assert nmse < 1e-4


def test_native_minus_seven_activation_grid():
    from research.native_int4.test_native import decode_a
    from research.native_int4.backend import pack_activation
    from gr00t.quantization.perstep_backends import PerStepBackend, make_backend_record

    torch.manual_seed(99)
    k, n = 267, 35
    record = make_backend_record(torch.randn(n, k) * 0.1, torch.rand(8, k) * 0.04 + 0.01, "step_weight", 128)
    record["act_qmin"] = -7
    layer = PerStepBackend(record).cuda()
    x = torch.randn(19, k, device="cuda")
    for step in (0, 7):
        _, _, a = layer.selected(step)
        pa = pack_activation(x, a, 128, False, qmin=-7)
        expected = (x / a).round().clamp(-7, 7).int().cpu()
        assert torch.equal(decode_a(pa, 19, k), expected)
        actual = layer(x, step, engine="native_s4")
        oracle = layer(x, step, engine="torch_reference")
        torch.testing.assert_close(actual, oracle, rtol=1e-4, atol=1e-4)
