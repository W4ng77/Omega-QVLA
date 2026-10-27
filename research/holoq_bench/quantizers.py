"""Auditable GR00T linear adapters of the pinned paper quantizers."""

import ast
import copy
import gc
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_function(path, name, namespace):
    """Load an unchanged upstream function without its unrelated model imports."""
    parsed = ast.parse(path.read_text())
    node = next(x for x in ast.walk(parsed) if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace
    )
    return namespace[name]


@torch.no_grad()
def gptq(weight, inputs, group=128, damping=0.01):
    sys.path.insert(0, str(ROOT / "third_party/gptq"))
    module = load_file("holoq_official_gptq", ROOT / "third_party/gptq/gptq.py")
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="cuda", dtype=torch.float16)
    linear.weight.copy_(weight)
    solver = module.GPTQ(linear)
    solver.quantizer = module.Quantizer()
    solver.quantizer.configure(4, perchannel=True, sym=True, mse=False)
    parameters = []
    original_find = solver.quantizer.find_params

    def find(*args, **kwargs):
        original_find(*args, **kwargs)
        # FP16 scale storage is the actual packed Marlin deployment contract.
        # Quantize using those exact parameters, avoiding a later code change.
        solver.quantizer.scale = solver.quantizer.scale.half().float()
        parameters.append((solver.quantizer.scale.detach().clone(), solver.quantizer.zero.detach().clone()))

    solver.quantizer.find_params = find
    solver.add_batch(inputs.unsqueeze(0), None)
    solver.fasterquant(blocksize=128, percdamp=damping, groupsize=group, actorder=False)
    count = 1 if group == -1 else weight.shape[1] // group
    params = parameters[-count:]
    scales = torch.cat([p[0] for p in params], dim=1)
    zeros = torch.cat([p[1] for p in params], dim=1)
    g = weight.shape[1] // count
    q = (
        (linear.weight.float() / scales.repeat_interleave(g, 1) + zeros.repeat_interleave(g, 1))
        .round()
        .clamp(0, 15)
    )
    q = q.to(torch.int8) - 8
    reconstructed = (q.float() * scales.repeat_interleave(g, 1)).half()
    if not torch.equal(reconstructed, linear.weight):
        raise RuntimeError("GPTQ export failed exact FP16 grid reconstruction")
    solver.free()
    return q, scales, {"weight_grid_exact": True, "group": g, "damping": damping, "block": 128}


@torch.no_grad()
def awq(weight, inputs, name):
    qpath = ROOT / "third_party/llm-awq/awq/quantize"
    ns = {"torch": torch, "gc": gc}
    pseudo = official_function(qpath / "quantizer.py", "pseudo_quantize_tensor", ns)
    config = {"q_group_size": 128, "zero_point": True}
    ns["w_quantize_func"] = lambda w: pseudo(w, n_bit=4, **config).detach()
    ns["get_act_scale"] = lambda x: x.abs().view(-1, x.shape[-1]).mean(0)
    search = official_function(qpath / "auto_scale.py", "_search_module_scale", ns)
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.float16, device="cuda")
    layer.weight.copy_(weight)
    # This is a linear-inspection adapter, not an unmodified Qwen/DiT block
    # entrypoint. The upstream search, clipping and affine quantizer are intact.
    smooth = search(layer, [layer], inputs)
    transformed = weight * smooth[None, :]
    scaled_inputs = inputs / smooth
    skip_clip = any(part in name for part in ("q_proj", "k_proj", "to_q", "to_k"))
    if not skip_clip:
        clip = official_function(
            qpath / "auto_clip.py", "auto_clip_layer", dict(ns, pseudo_quantize_tensor=pseudo)
        )
        bounds = clip(
            transformed, scaled_inputs, n_bit=4, q_config=config, n_grid=20, max_shrink=0.5, n_sample_token=512
        )
        transformed = transformed.reshape(*bounds.shape[:2], -1).clamp(-bounds, bounds).reshape(weight.shape)
    quant, scales, zeros = pseudo(transformed, n_bit=4, get_scale_zp=True, **config)
    q = (
        (quant.float() / scales.float().repeat_interleave(128, 1) + zeros.float().repeat_interleave(128, 1))
        .round()
        .clamp(0, 15)
    )
    return (
        q.to(torch.int32),
        scales,
        zeros,
        smooth,
        {
            "scale_candidates": 20,
            "clip_shrink": 0.5,
            "search_tokens": len(inputs),
            "clip_skipped_for_attention_qk": skip_clip,
            "adapter": "upstream linear-inspection scale loss, not whole-transformer-block loss",
        },
    )


def omniquant(weight, inputs):
    official = load_file("holoq_official_omni_quantizer", ROOT / "third_party/OmniQuant/quantize/quantizer.py")
    qzr = official.UniformAffineQuantizer(
        n_bits=4,
        symmetric=False,
        per_channel_axes=[0],
        dynamic_method="per_channel",
        group_size=128,
        shape=weight.shape,
        lwc=True,
    ).cuda()
    log_scale = nn.Parameter(torch.zeros(weight.shape[1], device="cuda"))
    optimizer = torch.optim.Adam(list(qzr.parameters()) + [log_scale], lr=0.005)
    x = inputs.detach().float()
    w = weight.detach().float()
    with torch.no_grad():
        target = F.linear(x, w)
    best_loss = float("inf")
    best = None
    for iteration in range(200):
        optimizer.zero_grad(set_to_none=True)
        scale = log_scale.exp()
        quantized = qzr(w * scale)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = F.linear(x / scale, quantized)
        loss = (prediction.float() - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite OmniQuant reconstruction loss")
        if loss.item() < best_loss:
            best_loss = loss.item()
            best = (copy.deepcopy(qzr.state_dict()), log_scale.detach().clone())
        loss.backward()
        optimizer.step()
    qzr.load_state_dict(best[0])
    with torch.no_grad():
        smooth = best[1].exp()
        quant = qzr(w * smooth)
        scales = qzr.scale.reshape(weight.shape[0], -1).detach()
        zeros = qzr.round_zero_point.reshape(weight.shape[0], -1).detach()
        q = (quant / scales.repeat_interleave(128, 1) + zeros.repeat_interleave(128, 1)).round().clamp(0, 15)
    return (
        q.to(torch.int32),
        scales.half(),
        zeros,
        smooth.half(),
        {
            "iterations": 200,
            "lr": 0.005,
            "loss": best_loss,
            "adapter": "upstream LWC plus input scaling, linear reconstruction",
        },
    )


@torch.no_grad()
def rtn(weight, group=-1, scale_half=True, mse=False):
    n, k = weight.shape
    group = k if group == -1 else group
    w = weight.float().reshape(n, -1, group)
    scale = (w.abs().amax(-1) / 7).clamp_min(1e-8)
    if mse:
        best = torch.full_like(scale, float("inf"))
        selected = scale.clone()
        for alpha in (0.5, 0.75, 1.0, 1.25, 1.5):
            ss = scale / alpha
            if scale_half:
                ss = ss.half().float()
            q = (w / ss[..., None]).round().clamp(-8, 7)
            error = ((q * ss[..., None] - w) ** 2).mean(-1)
            selected = torch.where(error < best, ss, selected)
            best = torch.minimum(error, best)
        scale = selected
    if scale_half:
        scale = scale.half().float()
    q = (w / scale[..., None]).round().clamp(-8, 7).reshape(n, k).to(torch.int8)
    return q, scale


@torch.no_grad()
def rotations(weight, composite, permute=True):
    """Block SVD bases via the equivalent small weight Gram eigensystems."""
    n, k = weight.shape
    w = weight.float()
    if permute:
        from gr00t.quantization.duquant_preprocess import zigzag_permutation

        perm = torch.from_numpy(zigzag_permutation(w.square().mean(0).cpu().numpy())).to(w.device)
    else:
        perm = torch.arange(k, device=w.device)
    wp = w[:, perm]
    blocks = wp.reshape(n, k // 64, 64).permute(1, 2, 0).contiguous()
    gram = blocks @ blocks.transpose(1, 2)
    _, basis = torch.linalg.eigh(gram)
    basis = basis.flip(-1)
    largest = basis.abs().argmax(-2, keepdim=True)
    basis *= basis.gather(-2, largest).sign()
    if composite:
        had = torch.ones((1, 1), device=w.device)
        while had.shape[0] < 64:
            had = torch.cat((torch.cat((had, had), 1), torch.cat((had, -had), 1)), 0)
        basis = basis @ (had / 8)
    rin = basis.half()
    wt = torch.bmm(wp.reshape(n, k // 64, 64).transpose(0, 1), rin.float()).transpose(0, 1).reshape(n, k)
    rows = w.reshape(n // 64, 64, k)
    _, rout = torch.linalg.eigh(rows @ rows.transpose(1, 2))
    rout = rout.flip(-1)
    largest = rout.abs().argmax(-2, keepdim=True)
    rout *= rout.gather(-2, largest).sign()
    rout = rout.half()
    wt = torch.bmm(rout.float(), wt.reshape(n // 64, 64, k)).reshape(n, k)
    return wt, rin, rout, perm
