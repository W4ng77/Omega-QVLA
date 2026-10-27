"""Signed codes, unchanged scale contracts, all step banks and graph streams."""

import pytest
import torch
import triton
from research.holoq_bench.nunchaku_probe import extension, pack
from research.holoq_bench.nunchaku_runtime import NunchakuLinear, _pack_input
from gr00t.quantization.dit_step_context import set_dit_quant_step


@pytest.mark.parametrize("m,k,step,smooth", [(1, 128, False, False), (16, 1024, True, False), (49, 2048, False, True), (559, 8192, False, False)])
def test_activation_codes_and_scales(m, k, step, smooth):
    torch.manual_seed(100)
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    sm = torch.rand((k,), device="cuda", dtype=torch.float16) + 0.5 if smooth else torch.empty(0, device="cuda")
    table = torch.rand((k,), device="cuda", dtype=torch.float32) + 0.05 if step else torch.empty(0, device="cuda")
    xx = (x.float() / sm.float()).half() if smooth else x
    scale = table if step else (xx.abs().float().amax(1, keepdim=True).clamp_min(1e-5) / 7).half()
    codes = (xx.float() / scale.float()).round().clamp(-7, 7).to(torch.int8)
    reference, rs = pack(codes, torch.ones((m, 1), device="cuda", dtype=torch.float16) if step else scale, k)
    actual, asc = torch.empty_like(reference), torch.empty_like(rs)
    _pack_input[(actual.shape[0],)](x, sm, table, actual, asc, m, k, step, smooth, -7, triton.next_power_of_2(k // 8), num_warps=4 if k < 4096 else 8)
    assert torch.equal(actual, reference)
    assert torch.equal(asc, rs)


@pytest.mark.parametrize("step", [False, True])
def test_module_step_dispatch_and_signed_gemm(step):
    torch.manual_seed(123)
    m, n, k = 16, 128, 1024
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    weights = torch.randint(-8, 8, (9 if step else 1, n, k), device="cuda", dtype=torch.int8)
    ws = torch.rand((9 if step else 1, n, k // 128 if step else 1), device="cuda", dtype=torch.float16) * 0.03 + 0.005
    bp, sp = zip(*[pack(q, s, 128 if step else k, True) for q, s in zip(weights, ws)])
    table = torch.rand((8, k), device="cuda", dtype=torch.float32) * 0.4 + 0.05 if step else torch.empty(0, device="cuda")
    record = dict(backend="nunchaku_s4", act_bits=4, act_qmin=-7, act_mode="step" if step else "token", N=n, K=k,
                  B=torch.stack(bp) if step else bp[0], s=torch.stack(sp) if step else sp[0], a_scale=table)
    layer = NunchakuLinear(record, "test", False)
    for index in range(8) if step else [0]:
        with set_dit_quant_step(index if step else None):
            actual = layer(x)
        scale = table[index] if step else (x.float().abs().amax(1, keepdim=True) / 7).half()
        q = (x.float() / scale.float()).round().clamp(-7, 7)
        w = weights[index].float() * ws[index].float().repeat_interleave(128 if step else k, 1)
        reference = q @ w.T if step else (q * scale.float()) @ w.T
        nmse = (actual.float() - reference).square().mean() / reference.square().mean()
        assert actual.isfinite().all() and nmse < 1e-4
    if step:
        with pytest.raises(RuntimeError, match="Missing explicit step"):
            layer(x)


def test_current_stream_and_cuda_graph():
    ext = extension()
    a = torch.randint(-7, 8, (16, 128), device="cuda", dtype=torch.int8)
    w = torch.randint(-8, 8, (128, 128), device="cuda", dtype=torch.int8)
    ap, asc = pack(a, torch.ones((16, 1), device="cuda", dtype=torch.float16), 128)
    wp, wsc = pack(w, torch.ones((128, 1), device="cuda", dtype=torch.float16), 128, True)
    out = torch.empty((16, 128), device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ext.gemm(ap, wp, asc, wsc, out)
    torch.cuda.current_stream().wait_stream(stream)
    reference = (a.float() @ w.float().T).half()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ext.gemm(ap, wp, asc, wsc, out)
    graph.replay()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
