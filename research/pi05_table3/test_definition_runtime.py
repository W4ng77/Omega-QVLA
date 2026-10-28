"""Numerical guards for core layout, scale location, and step dispatch."""

import pytest
import torch
from gr00t.quantization.dit_step_context import set_dit_quant_step
from research.pi05_table3.definition_runtime import DefinitionLinear
from research.pi05_table3.paper_math import pack_int4, direct_input_rotation, block_transform


@pytest.mark.parametrize("mode", ["token", "step"])
@pytest.mark.parametrize("m", [7, 161])
def test_frozen_codes_scales_noncontiguous_and_steps(mode, m):
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    k, n = 128, 192
    q = torch.randint(-7, 8, (n, k), device="cuda", dtype=torch.int8)
    rotation = torch.eye(64, device="cuda").repeat(k // 64, 1, 1)
    tables = torch.rand(8, k, device="cuda") * 0.1 + 0.01
    record = {
        "K": k,
        "N": n,
        "B": pack_int4(q),
        "s": torch.rand(n, 1, device="cuda") * 0.01 + 0.001,
        "rot_in": rotation,
        "perm": torch.arange(k - 1, -1, -1, device="cuda"),
        "a_scale": tables,
        "bias": torch.zeros(n, device="cuda"),
        "act_mode": mode,
    }
    reference = DefinitionLinear(record, "test", "reference").cuda()
    packed = DefinitionLinear(record, "test", "packed").cuda()
    x = torch.randn(m, k * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert not x.is_contiguous()
    with torch.inference_mode():
        for step in range(8):
            with set_dit_quant_step(step, 8):
                expected, actual = reference(x), packed(x)
            nmse = float((expected.float() - actual.float()).square().sum() / expected.float().square().sum())
            assert nmse < 1e-4, nmse
        if mode == "step":
            assert packed.steps_seen == set(range(8))
            with pytest.raises(AssertionError), set_dit_quant_step(0, 10):
                packed(x)
    assert packed.weight.numel() == 0


def test_input_only_transform_equivalence():
    torch.manual_seed(23)
    w = torch.randn(192, 128, device="cuda")
    x = torch.randn(7, 256, device="cuda")[:, ::2]
    wr, r, p, signs = direct_input_rotation(w, 42)
    torch.backends.cuda.matmul.allow_tf32 = False
    expected = x @ w.t()
    actual = block_transform(x, r, p) @ wr.t()
    assert float((expected - actual).square().sum() / expected.square().sum()) < 1e-9
    assert set(signs.unique().tolist()) == {-1, 1}
