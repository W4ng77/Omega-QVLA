import pytest
import torch

from gr00t.quantization.perstep_backends import PerStepBackend, make_backend_record, unpack_int4
from research.native_int4.backend import from_packed, pack_activation, pack_weight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA GPU required")


def decode_a(words, m, k):
    words = words.cpu().long()
    result = torch.zeros((words.shape[0] * 16, words.shape[1] * 64), dtype=torch.int32)
    for mb in range(words.shape[0]):
        for kb in range(words.shape[1]):
            for reg in range(4):
                for lane in range(32):
                    row = mb * 16 + lane // 4 + (reg % 2) * 8
                    lo = kb * 64 + (lane % 4) * 8 + (reg // 2) * 32
                    q = (words[mb, kb, reg, lane] >> (torch.arange(8) * 4)) & 15
                    result[row, lo : lo + 8] = torch.where(q >= 8, q - 16, q)
    return result[:m, :k]


@pytest.mark.parametrize("mode", ["step_weight", "group_int"])
@pytest.mark.parametrize("group", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_matches_existing_integer_backend(mode, group, dtype):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(43)
    n, k, m = 35, 267, 19
    weight = torch.randn(n, k) * 0.1
    table = torch.rand(3, k) * 0.2 + 0.01
    b = PerStepBackend(make_backend_record(weight, table, mode, group)).cuda()
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    for step in (0, 2, None):
        p, scales, act = b.selected(step)
        pa = pack_activation(x, act, group, mode == "group_int")
        pb = pack_weight(p, n, k)
        scale_vec = act.repeat_interleave(group)[:k] if mode == "group_int" else act
        expected_q = (x.float() / scale_vec).round().clamp(-8, 7).int().cpu()
        assert torch.equal(decode_a(pa, m, k), expected_q)
        oracle = b(x, step, engine="torch_reference")
        for wn, sk in ((1, 1), (4, 1), (8, 2)):
            y = from_packed(
                pa, pb, scales.t().contiguous(), act, m, n, k, group, 2 if mode == "group_int" else 1, wn, sk
            )
            err = ((y - oracle).square().sum() / oracle.square().sum().clamp_min(1e-30)).item()
            assert err < 1e-6, (mode, group, dtype, step, wn, sk, err)


def test_mma_packing_and_integer_products_are_exact():
    from gr00t.quantization.perstep_backends import pack_int4

    torch.manual_seed(47)
    m, n, k = 33, 41, 191
    aq = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int32)
    wq = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    act = torch.ones(k, device="cuda")
    scales = torch.ones((3, n), device="cuda")
    pa = pack_activation(aq.float(), act, 64, False)
    pb = pack_weight(pack_int4(wq), n, k)
    y = from_packed(pa, pb, scales, act, m, n, k, 64, 1, 4, 1)
    expected = (aq.cpu().long() @ wq.cpu().long().t()).float().cuda()
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    torch.testing.assert_close(unpack_int4(pack_int4(wq), k), wq)


@pytest.mark.parametrize("mode", ["step_weight", "group_int"])
def test_runtime_engine_dispatch_preserves_batch_shape_and_caches_banks(mode):
    torch.manual_seed(71)
    b = PerStepBackend(make_backend_record(torch.randn(35, 267), torch.rand(3, 267) + 0.1, mode, 128)).cuda()
    x = torch.randn(2, 19, 267, device="cuda", dtype=torch.bfloat16)
    for step in (0, 2, None, 0):
        expected = b(x, step, engine="triton")
        actual = b(x, step, engine="native_s4")
        assert actual.shape == (2, 19, 35)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert b._native_calls == 4
    assert len(b._native_cache) == (3 if mode == "step_weight" else 1)
