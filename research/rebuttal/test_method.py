import pytest
import torch
from research.rebuttal.method import direct_input_rotation, serpentine_descending_input_energy_v1, equation_oracle, rtn_per_output, gptq_per_output_signed15


def test_input_space_rotation_preserves_unquantized_operator_and_unrotated_bias():
    torch.manual_seed(2)
    x, w, b = torch.randn(5, 128), torch.randn(192, 128), torch.randn(192)
    wr, r, p, signs = direct_input_rotation(w, 51)
    from research.rebuttal.method import block_transform

    actual = block_transform(x, r, p) @ wr.t() + b
    expected = x @ w.t() + b
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-5)
    assert r.dtype == torch.float32 and set(signs.unique().tolist()) <= {-1, 1}


def test_cross_block_zigzag_is_a_named_interpretation_and_not_legacy_chunking():
    w = torch.arange(1, 257).float()[None, :].repeat(3, 1)
    p = serpentine_descending_input_energy_v1(w)
    assert torch.equal(p.sort().values, torch.arange(256))
    assert set(p.reshape(4, 64)[:, 0].tolist()) == {252, 253, 254, 255}


def test_column_scales_cannot_be_replaced_by_one_output_epilogue_scale():
    qx = torch.tensor([[1, 2], [2, 1]], dtype=torch.int8)
    qw = torch.tensor([[1, -1]], dtype=torch.int8)
    a, sw = torch.tensor([1., 10.]), torch.tensor([[2.]])
    correct = equation_oracle(qx, qw, a, sw)
    torch.testing.assert_close(correct, torch.tensor([[-38.], [-16.]]), rtol=0, atol=0)
    incorrect = (qx.float() @ qw.float().t()) * a.mean() * sw.t()
    assert not torch.equal(correct, incorrect)


def test_weight_scale_is_one_fp32_coefficient_per_output_and_never_uses_minus8():
    w = torch.tensor([[-8., -3., 1., 7.], [0., 0., 0., 0.]])
    q, s = rtn_per_output(w)
    assert q.min() == -7 and q.max() <= 7 and s.shape == (2, 1)
    assert s.dtype == torch.float32 and torch.equal(q[1], torch.zeros(4, dtype=torch.int8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required")
def test_gptq_processing128_does_not_introduce_weight_group128_or_half_scale():
    torch.manual_seed(7)
    w = torch.randn(256, 256, device="cuda") * .01357
    x = torch.randn(384, 256, device="cuda")
    q, s, config = gptq_per_output_signed15(w, x.t() @ x / len(x), len(x))
    assert s.shape == (256, 1) and s.dtype == torch.float32
    assert bool((s != s.half().float()).any())
    assert q.min() >= -7 and q.max() <= 7 and config["weight_group_size"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required")
def test_existing_packed_backend_matches_same_frozen_grid_for_all8_column_scale_rows_and_strides():
    from research.rebuttal.runtime import PaperLinear
    from gr00t.quantization.perstep_backends import pack_int4
    from gr00t.quantization.dit_step_context import set_dit_quant_step

    torch.manual_seed(55)
    w = torch.randn(192, 128, device="cuda") * .02
    _, r, p, _ = direct_input_rotation(w, 99)
    q, s = rtn_per_output(w)
    # Distinct column and step factors test placement INSIDE the K reduction.
    a = torch.logspace(-1, .2, 128, device="cuda")[None] * torch.linspace(.8, 1.5, 8, device="cuda")[:, None]
    record = {"K": 128, "N": 192, "B": pack_int4(q), "s": s, "rot_in": r, "perm": p,
              "a_scale": a, "bias": torch.randn(192, device="cuda").half() * .01, "act_mode": "step"}
    ref, packed = PaperLinear(record, "test", "reference"), PaperLinear(record, "test", "packed")
    x = torch.randn(2, 17, 256, device="cuda").half()[..., ::2]
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="Missing/out-of-range"):
        packed(x)
    for step in range(8):
        with set_dit_quant_step(step, total=8):
            qr, sr = ref.prepare(x)
            qp, sp = packed.prepare(x)
            assert torch.equal(qr, qp) and torch.equal(sr, sp) and torch.equal(sr, a[step])
            expected, actual = ref(x).float(), packed(x).float()
            assert torch.isfinite(actual).all()
            assert float((actual - expected).square().sum() / expected.square().sum()) < 1e-5
    assert not hasattr(packed, "reference_codes")
    assert packed.steps_seen == set(range(8))
