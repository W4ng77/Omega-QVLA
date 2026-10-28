import pytest
import torch
from research.holoq_bench.fused_rotation_strided import rotate, install
from research.holoq_bench.packed_runtime import apply_blocks, PackedLinear
from research.holoq_bench.kernels import pack_marlin

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA GPU required")


@pytest.mark.parametrize("m,k", [(1, 128), (49, 1536), (558, 2048), (559, 8192)])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_coefficients_layout_and_stream(m, k, permuted, transposed):
    torch.manual_seed(49)
    x = torch.randn(1, m, k, device="cuda", dtype=torch.float16)
    r = (torch.randn(k // 64, 64, 64, device="cuda") / 8).half()
    if transposed:
        r = r.transpose(-1, -2)
        assert not r.is_contiguous()
    p = torch.randperm(k, device="cuda") if permuted else None
    expected = apply_blocks(x, r, p)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = rotate(x, r, p)
    torch.cuda.current_stream().wait_stream(side)
    nmse = (out.float() - expected.float()).square().sum() / expected.float().square().sum()
    assert nmse < 1e-6 and out.is_contiguous()
    for _ in range(3):
        rotate(x, r, p)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        y = rotate(x, r, p)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, out, rtol=0, atol=0)


def test_plugin_keeps_weight_and_scale_buffers_without_dense_cache():
    torch.manual_seed(50)
    n, k = 256, 512
    q = torch.randint(-7, 8, (n, k), dtype=torch.int8)
    s = (torch.rand(n, 1) * 0.02 + 0.01).half()
    records = dict(
        pack_marlin(q, s),
        K=k,
        N=n,
        act_bits=4,
        act_mode="channel",
        act_qmin=-7,
        a_scale=torch.rand(k) * 0.03 + 0.01,
        rot_in=(torch.randn(k // 64, 64, 64) / 8).half(),
        rot_out=(torch.randn(n // 64, 64, 64) / 8).half().transpose(-1, -2),
        perm=torch.randperm(k),
        bias=torch.randn(n).half(),
    )
    # install validates full model coverage; wrap the same audited record under
    # 116 names to exercise buffer migration without a separate model download.
    model = torch.nn.ModuleList([PackedLinear(records, str(i)) for i in range(116)])
    before = {id(m): m.B.clone() for m in model}
    install(model)
    model.cuda()
    layer = model[0]
    x = torch.randn(1, 49, k, device="cuda", dtype=torch.float16)
    actual = layer(x)
    reference = PackedLinear(records, "reference").cuda()(x)
    nmse = (actual.float() - reference.float()).square().sum() / reference.float().square().sum()
    assert nmse < 1e-4 and torch.isfinite(actual).all()
    assert not hasattr(layer, "weight")
    assert all(torch.equal(m.B.cpu(), before[id(m)]) for m in model)
