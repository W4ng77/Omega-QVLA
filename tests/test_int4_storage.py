import pytest
import torch

from gr00t.quantization.gptq_layers import GptqLinear
from gr00t.quantization.int4_storage import pack_codebook, pack_rtn, pack_symmetric, unpack_int4
from gr00t.quantization.rtn_layers import RTNLinear, _quant_dequant_weight_groupwise


def test_symmetric_int4_round_trip() -> None:
    codes = torch.randint(-8, 8, (8, 256), generator=torch.Generator().manual_seed(7))
    scales = torch.tensor([0.013, 0.071, 0.003, 0.18, 0.09, 0.001, 0.22, 0.04])
    weights = (codes * scales[:, None]).to(torch.float16)
    packed = pack_symmetric(weights)

    assert packed["packed"].dtype == torch.uint8
    assert torch.equal(unpack_int4(packed), weights)


def test_codebook_int4_round_trip() -> None:
    weights = torch.arange(16, dtype=torch.bfloat16).repeat(4, 4)
    packed = pack_codebook(weights, 64)

    assert packed["packed"].dtype == torch.uint8
    assert torch.equal(unpack_int4(packed), weights)


def test_rtn_int4_round_trip() -> None:
    weights = torch.randn((32, 128), generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
    packed = pack_rtn(weights, 64)
    expected = _quant_dequant_weight_groupwise(weights, 64, 4).to(weights.dtype)

    assert packed["packed"].dtype == torch.uint8
    assert torch.equal(unpack_int4(packed), expected)


def test_symmetric_falls_back_to_codebook() -> None:
    values = torch.tensor([0.0, 0.1, 0.4, 0.9], dtype=torch.float16)
    weights = values[torch.arange(128) % 4].reshape(1, 128)
    packed = pack_symmetric(weights)

    assert packed["fallback_indices"].numel() == 1
    assert torch.equal(unpack_int4(packed), weights)


def test_codebook_rejects_more_than_16_values() -> None:
    weights = torch.arange(32, dtype=torch.float16).repeat(1, 2)

    with pytest.raises(ValueError, match="more than 16"):
        pack_codebook(weights, 64)


def test_gptq_loader_accepts_packed_weight() -> None:
    weight = (torch.randint(-8, 8, (8, 128)) * 0.015).to(torch.float16)
    record = {"weight_res_q_int4": pack_symmetric(weight)}

    loaded = GptqLinear._build_weight_from_record(record, "test", torch.empty_like(weight))

    assert torch.equal(loaded, weight)


def test_rtn_prequantized_weight_is_not_modified() -> None:
    layer = torch.nn.Linear(64, 8, bias=False, dtype=torch.bfloat16)
    quantized = _quant_dequant_weight_groupwise(layer.weight, 64, 4).to(layer.weight.dtype)
    layer.weight.data.copy_(quantized)

    wrapped = RTNLinear(layer, name="test", group_size=64, wbits=4, abits=4, prequantized=True)

    assert torch.equal(wrapped.weight, quantized)
