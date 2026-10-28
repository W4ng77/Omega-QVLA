import pytest
import torch
from torch import nn
from gr00t.quantization.int4_storage import unpack_int4
from gr00t.quantization.rtn_layers import RTNLinear
from research.pi05_table3.vision_packed import PackedVision


@pytest.mark.parametrize("group", [16, 64])
def test_real_record_with_original_grid_and_strided_input(group):
    records = torch.load('/home/xinyu/models/omega-qvla-pi05-414-w4a4-int4/vision_int4.pt', map_location='cpu', mmap=True, weights_only=False)
    name, record = next((n, r) for n, r in records.items() if r['group_size'] == group)
    n, k = record['shape']
    torch.manual_seed(group)
    base = nn.Linear(k, n, bias=True, device='cuda', dtype=torch.bfloat16)
    base.weight.data.copy_(unpack_int4(record).cuda())
    reference = RTNLinear(base, name=name, group_size=group, wbits=4, abits=4, prequantized=True)
    packed = PackedVision(reference, record, name)
    x = torch.randn(2, 17, 2*k, device='cuda', dtype=torch.bfloat16)[..., ::2]
    with torch.inference_mode():
        expected, actual = reference(x).float(), packed(x).float()
    assert torch.isfinite(actual).all()
    nmse = float((actual-expected).square().sum()/expected.square().sum())
    # BF16 output/reduction roundoff is larger than FP16's. This local1e-4
    # engineering bound matches the predeclared policy layer guard; the
    # independent complete-action gate remains1e-2 and is not relaxed.
    assert nmse < 1e-4, nmse
    assert packed.weight.numel() == 0 and packed.weight.dtype == torch.bfloat16
