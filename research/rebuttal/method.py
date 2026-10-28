"""Explicit interpretation of the user-verified submitted HoloQ equations.

Unspecified choices are named, frozen and reported, never selected by scores.
No output rotation, groupwise original weight quantizer or low-rank branch.
"""
import sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def serpentine_descending_input_energy_v1(weight, block=64):
    """One disclosed cross-block zigzag mapping; submission is not unique."""
    k = weight.shape[1]
    assert k % block == 0
    energy = weight.float().square().sum(0)
    order = energy.argsort(descending=True, stable=True).reshape(block, k // block)
    order[1::2] = order[1::2].flip(1)
    return order.t().contiguous().reshape(-1)


def block_transform(x, rotation, perm=None):
    shape = x.shape
    assert rotation.dtype == torch.float32
    flat = x.float().reshape(-1, shape[-1])
    if perm is not None:
        flat = flat.index_select(1, perm)
    nb, b, _ = rotation.shape
    assert shape[-1] == nb * b and b == 64
    with torch.autocast(x.device.type, enabled=False):
        out = torch.bmm(flat.reshape(-1, nb, b).transpose(0, 1).contiguous(), rotation)
    return out.transpose(0, 1).contiguous().reshape(shape)


@torch.no_grad()
def direct_input_rotation(weight, d_seed):
    n, k = weight.shape
    permutation = serpentine_descending_input_energy_v1(weight)
    wp = weight.float()[:, permutation]
    # In paper orientation each block is[64,Cout]; U is the LEFT basis.
    blocks = wp.reshape(n, k // 64, 64).permute(1, 2, 0).contiguous()
    u, _, _ = torch.linalg.svd(blocks, full_matrices=False)
    pivot = u.abs().argmax(-2, keepdim=True)
    u = u * u.gather(-2, pivot).sign()  #disclosed SVD sign convention
    generator = torch.Generator(device="cpu").manual_seed(d_seed)
    signs = (torch.randint(0, 2, (k // 64, 64), generator=generator) * 2 - 1).to(torch.int8)
    h = torch.ones((1, 1), device=weight.device, dtype=torch.float32)
    while len(h) < 64:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    with torch.autocast(weight.device.type, enabled=False):
        rotation = (u * signs.to(weight.device).float()[:, None, :]) @ (h / 8)
    wr = block_transform(weight.float(), rotation, permutation)
    return wr, rotation, permutation, signs


def positive_scale(peak):
    # Eq6 has an undefined0/0 for an all-zero vector. Scale1/codes0 preserves
    # that vector exactly; count and disclose these guards in calibration.
    return torch.where(peak > 0, peak / 7, torch.ones_like(peak))


def signed15(x, scale):
    return (x.float() / scale.float()).round().clamp(-7, 7).to(torch.int8)


def token_quantize(x):
    scale = positive_scale(x.float().abs().amax(-1, keepdim=True))
    return signed15(x, scale), scale


def rtn_per_output(weight):
    scale = positive_scale(weight.float().abs().amax(1, keepdim=True))
    return signed15(weight, scale), scale


def equation_oracle(qx, qw, activation_scale, weight_scale, bias=None):
    """Correct Eq9: Sa[k] inside K, Sw[n] outside; FP32 accuracy reference."""
    assert weight_scale.shape == (qw.shape[0], 1)
    with torch.autocast(qx.device.type, enabled=False):
        y = torch.nn.functional.linear(qx.float() * activation_scale.float(), qw.float())
        y = y * weight_scale.float().t()
        return y + bias.float() if bias is not None and bias.numel() else y


@torch.no_grad()
def gptq_per_output_signed15(weight, hessian, token_count):
    """Pinned official GPTQ; processing128, damp.01, NO weight grouping.

    Full-H propagation is an explicit reconstruction choice: legacy block-local
    MSE scaling conflicts with the submission's single per-output scale.
    """
    sys.path.insert(0, str(ROOT / "third_party/gptq"))
    from research.holoq_bench.quantizers import load_file

    module = load_file("rebuttal_official_gptq", ROOT / "third_party/gptq/gptq.py")
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=torch.float32)
    layer.weight.copy_(weight.float())
    solver = module.GPTQ(layer)
    solver.H = hessian.float().clone()
    solver.nsamples = token_count
    solver.quantizer = module.Quantizer()
    solver.quantizer.configure(4, perchannel=True, sym=True, mse=False)

    def parameters(values, weight=False):
        assert weight
        solver.quantizer.scale = positive_scale(values.float().abs().amax(1, keepdim=True))
        solver.quantizer.zero = torch.full_like(solver.quantizer.scale, 7)
        solver.quantizer.maxq = torch.tensor(14, device=values.device)

    solver.quantizer.find_params = parameters
    dead = int((torch.diag(hessian) == 0).sum())
    solver.fasterquant(blocksize=128, percdamp=.01, groupsize=-1, actorder=False)
    scale = solver.quantizer.scale.detach().clone()
    codes = signed15(layer.weight, scale)
    assert torch.equal(codes.float() * scale, layer.weight)
    solver.free()
    return codes, scale, {"processing_block": 128, "weight_group_size": None, "weight_scale_granularity": "one FP32 scalar per output over all input channels", "damping": .01, "activation_order": False, "dead_input_channels": dead}
