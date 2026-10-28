"""Numerical and dispatch checks for the experimental W4A4 storage backend."""

import pytest
import torch

from gr00t.quantization.int4_storage import pack_codebook, pack_symmetric, unpack_int4
from gr00t.quantization.packed_w4a4 import packed_w4a4_linear, weight_codebooks


@pytest.mark.parametrize("encoding", ["symmetric", "codebook"])
@pytest.mark.parametrize("stored_dtype", [torch.float16, torch.bfloat16])
def test_codebook_reconstructs_original_values(encoding, stored_dtype):
    generator = torch.Generator().manual_seed(17)
    q = torch.randint(-8, 8, (8, 256), generator=generator)
    w = (q * torch.rand((8, 2, 1), generator=generator).repeat(1, 1, 128).reshape(8, 256)).to(stored_dtype)
    record = pack_symmetric(w) if encoding == "symmetric" else pack_codebook(w, 128)
    packed, table = weight_codebooks(record, torch.bfloat16)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(-1, 128)
    assert torch.equal(table.gather(1, codes.long()).reshape_as(w), w.bfloat16())
    assert torch.equal(unpack_int4(record), w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [1, 50, 129, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_all_m_and_per_channel_scales_match_reference(m, dtype):
    torch.manual_seed(19)
    n, k, g = 96, 256, 128
    w = (torch.randint(-8, 8, (n, k)) * 0.021).bfloat16()
    record = pack_symmetric(w, group_size=g)
    packed, table = weight_codebooks(record, w.dtype)
    packed, table = packed.cuda(), table.cuda()
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    scale = torch.logspace(-2, 0, k, device="cuda").to(dtype)
    # Test multiple genuinely different timestep/channel scales.
    for factor in (1.0, 2.7):
        s = scale * factor
        xq = torch.round(x / s).clamp(-8, 7) * s
        actual = packed_w4a4_linear(xq.contiguous(), s, packed, table, g)
        expected = torch.nn.functional.linear(xq.bfloat16(), w.cuda())
        diff = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
        assert diff < 0.004, (m, dtype, diff.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_custom_op_compiles_without_graph_break():
    w = (torch.randint(-8, 8, (64, 128)) * 0.03).bfloat16()
    packed, table = weight_codebooks(pack_symmetric(w), w.dtype)
    packed, table = packed.cuda(), table.cuda()
    s = torch.full((128,), 0.1, device="cuda", dtype=torch.bfloat16)
    x = torch.randn((50, 128), device="cuda", dtype=torch.bfloat16)
    xq = torch.round(x / s).clamp(-8, 7) * s
    compiled = torch.compile(packed_w4a4_linear, fullgraph=True)
    torch.testing.assert_close(compiled(xq, s, packed, table, 128),
                               packed_w4a4_linear(xq, s, packed, table, 128))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("version", [1, 2, 3])
def test_gptq_layer_keeps_rotations_bias_and_timestep_dispatch(tmp_path, version):
    from gr00t.quantization.dit_step_context import set_dit_quant_step
    from gr00t.quantization.gptq_layers import GptqConfig, GptqLinear
    from gr00t.quantization.packed_w4a4 import enable_packed_w4a4

    torch.manual_seed(23)
    w = (torch.randint(-8, 8, (128, 128)) * 0.02).bfloat16()
    blocks = torch.linalg.qr(torch.randn(2, 64, 64)).Q
    rec = {"format": "dit_svdquant_v1", "weight_res_q_int4": pack_symmetric(w),
           "lowrank_A": torch.empty(128, 0), "lowrank_B": torch.empty(128, 0),
           "act_scale_table": torch.stack((torch.full((128,), 0.03), torch.full((128,), 0.2))),
           "duquant_rotation_blocks": blocks, "duquant_rotation_perm": torch.randperm(128),
           "duquant_rotation_out_blocks": blocks, "a_bits": 4, "weight_bits": 4}
    records = {"0": rec}
    path = tmp_path / "pack.pt"
    torch.save(records, path)
    model = torch.nn.Sequential(GptqLinear(torch.nn.Linear(128, 128, dtype=torch.bfloat16),
                                          "0", GptqConfig(enabled=True, path=str(path), act_bits=4))).cuda()
    x = torch.randn((50, 128), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        reference = []
        for step in range(2):
            with set_dit_quant_step(step, total=2):
                reference.append(model(x))
        enable_packed_w4a4(model, records, version=version)
        assert model[0]._weight_q.numel() == 0
        assert model[0]._w_promoted_cache is None
        assert not torch.equal(reference[0], reference[1])
        for step in range(2):
            with set_dit_quant_step(step, total=2):
                actual = model(x)
            relative_error = (actual.float() - reference[step].float()).norm() / reference[step].float().norm()
            assert relative_error < 0.006
            if version == 3:
                torch.testing.assert_close(actual, reference[step], atol=0, rtol=0)
        with pytest.raises(RuntimeError, match="missing/out-of-range"):
            model(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_direct_pack_keeps_quantization_codes_at_rounding_boundaries(dtype):
    import triton
    from gr00t.quantization.packed_w4a4_fast import _quant_pack_a4

    s = torch.logspace(-3, 1, 128, device="cuda").to(dtype)
    thresholds = torch.arange(-9, 9, device="cuda", dtype=torch.float32) + 0.5
    x = (thresholds[:, None] * s.float()[None, :]).to(dtype)
    x = torch.cat((x, torch.nextafter(x, torch.full_like(x, float("inf"))),
                   torch.nextafter(x, torch.full_like(x, -float("inf")))), dim=0).contiguous()
    output = torch.empty((x.shape[0], 64), device="cuda", dtype=torch.uint8)
    total = output.numel()
    _quant_pack_a4[(triton.cdiv(total, 256),)](x, s, output, 128, total, 256)
    decoded = torch.stack((output & 15, output >> 4), -1).reshape_as(x).to(torch.int32) - 8
    expected = torch.round(x / s).clamp(-8, 7).to(torch.int32)
    assert torch.equal(decoded, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [10, 129, 968])
def test_arithmetic_path_with_sparse_fallback_and_fp16_pack(m):
    from gr00t.quantization.packed_w4a4_fast import packed_w4a4_fast_linear, symmetric_metadata

    torch.manual_seed(29)
    w = (torch.randint(-8, 8, (96, 256)) * 0.0317).half()
    w[0, :128] = torch.tensor([0., 0.1, 0.4, 0.9], dtype=torch.float16).repeat(32)
    record = pack_symmetric(w)
    assert record["fallback_indices"].numel()
    scales, fmap, table, kind = symmetric_metadata(record, torch.bfloat16)
    packed = record["packed"].reshape(96, 128).cuda()
    x = torch.randn((m, 256), device="cuda", dtype=torch.bfloat16)
    sa = torch.logspace(-2, 0, 256, device="cuda").bfloat16()
    actual = packed_w4a4_fast_linear(x, sa, packed, scales.cuda(), fmap.cuda(), table.cuda(), 128, kind)
    xq = torch.round(x / sa).clamp(-8, 7) * sa
    expected = torch.nn.functional.linear(xq, w.bfloat16().cuda())
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative < 0.004
