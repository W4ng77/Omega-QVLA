"""No source weight may change during the historical pack deployment export."""
import torch
from research.holoq_bench.released_grid import recover_exact_grid


def test_fp32_scale_fp16_weight_values_are_recovered_exactly_without_requantization():
    torch.manual_seed(72)
    q = torch.randint(-8, 8, (32, 4, 128), dtype=torch.int8)
    scale = torch.rand(32, 4) * .117 + .000013
    w = (q.float() * scale[..., None]).half().flatten(1)
    q2, s2 = recover_exact_grid(w)
    actual = (q2.reshape(32, 4, 128).float() * s2[..., None]).half().flatten(1)
    assert torch.equal(actual, w) and s2.dtype == torch.float32


def test_zero_group_and_missing_smallest_code_remain_lossless():
    q = torch.tensor([0, 2, 4, 6, -2, -4, -6, -8], dtype=torch.float32).repeat(16)
    w = (q[None, :] * .01703417).half().repeat(3, 1)
    w[0].zero_()
    q2, s2 = recover_exact_grid(w)
    assert torch.equal((q2.float() * s2).half(), w)


def test_characterize_legacy_first_forward_freeze_instead_of_32_batch_calibration():
    from types import SimpleNamespace
    from gr00t.quantization.gptq_layers import GptqLinear
    from gr00t.quantization.duquant_preprocess import PercentileCalibrator

    legacy = SimpleNamespace(cfg=SimpleNamespace(act_bits=4, act_percentile=99.9),
                             calibrator=PercentileCalibrator(99.9, 32), _act_scale_initialized=False,
                             _act_scale=None, _dump_act_stats=lambda value: None)
    x = torch.arange(1, 129).reshape(32, 4).half()
    first = GptqLinear._get_act_scale(legacy, x).clone()
    for _ in range(31):
        current = GptqLinear._get_act_scale(legacy, x * 10)
        assert torch.equal(current, first)
    assert legacy.calibrator._seen == 1 and legacy._act_scale_initialized
