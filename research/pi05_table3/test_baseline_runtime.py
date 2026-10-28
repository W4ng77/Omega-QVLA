from pathlib import Path
import pytest
import torch
from research.pi05_table3.baseline_runtime import BaselineLinear


@pytest.mark.parametrize("part", ["vision", "language"])
@pytest.mark.parametrize("m", [7, 161])
def test_qserve_real_packed_control_matches_integer_oracle_with_padding(part, m):
    root = Path(__file__).resolve().parents[2] / "results/pi05_table3_414_20261028"
    manifest = __import__("json").loads((root / "baseline_capture/manifest.json").read_text())
    item = next(
        x
        for x in manifest["layers"]
        if ("vision_tower" in x["name"] if part == "vision" else "language_model" in x["name"])
    )
    cache = root / "baseline_build/smoothquant_qserve414/layers" / f"{item['index']:03d}.pt"
    entry = torch.load(cache, map_location="cpu", mmap=True, weights_only=False)
    layer = BaselineLinear(entry["record"], entry["name"]).cuda()
    torch.manual_seed(m)
    x = torch.randn(m, item["K"], device="cuda", dtype=torch.bfloat16)
    q = entry["canonical_codes"].cuda().float()
    sw = entry["canonical_scales"].cuda().float().t()
    smooth = entry["record"]["smooth"].cuda()
    xr = (x.float() / smooth).half().float()
    sa = (xr.abs().amax(1, keepdim=True).clamp_min(1e-5) / 127).half().float()
    qa = (xr / sa).round().clamp(-128, 127)
    with torch.inference_mode():
        expected = ((qa @ q.t()) * sa * sw).half().to(layer.external_dtype)
        if layer.bias.numel():
            expected = expected + layer.bias
        actual = layer(x)
    assert torch.isfinite(actual).all()
    nmse = float((actual.float() - expected.float()).square().sum() / expected.float().square().sum())
    assert nmse < 1e-4, nmse
    assert layer.weight.numel() == 0


@pytest.mark.parametrize('m',[7,161])
def test_marlin_packed_control_matches_frozen_weight_oracle(m):
    root=Path(__file__).resolve().parents[2]/'results/pi05_table3_414_20261028'
    manifest=__import__('json').loads((root/'baseline_capture/manifest.json').read_text())
    item=next(x for x in manifest['layers'] if 'language_model' in x['name'])
    entry=torch.load(root/'baseline_build/gptq_marlin252/layers'/f"{item['index']:03d}.pt",map_location='cpu',mmap=True,weights_only=False)
    layer=BaselineLinear(entry['record'],entry['name']).cuda()
    torch.manual_seed(m)
    torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(m,item['K'],device='cuda',dtype=torch.bfloat16)
    q=entry['canonical_codes'].cuda().float()
    sw=entry['canonical_scales'].cuda().float().repeat_interleave(128,1)
    with torch.inference_mode():
        expected=torch.nn.functional.linear(x.half(),(q*sw).half()).to(layer.external_dtype)
        actual=layer(x)
    nmse=float((actual.float()-expected.float()).square().sum()/expected.float().square().sum())
    assert nmse<1e-4,nmse
    assert layer.weight.numel()==0 and torch.count_nonzero(layer.workspace)==0
