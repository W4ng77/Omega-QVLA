#!/usr/bin/env python
"""QVLA (arXiv 2602.03782) port for GR00T N1.5: channel-wise bit allocation.

Method (faithful to the released QVLA pipeline, openvla/qvla/*):
  1. Per-layer input Gram H from LIBERO calib forwards (one pass, all layers).
  2. Hessian proxy: d = diag(chol(inv(H))); per-row loss_b = sum((W - Q_b(W))^2 / d^2).
  3. Global greedy demotion over bits {2,4,8,16} to a target average bit-width
     (0-bit pruning disabled: the released greedy over-prunes and kills the policy).
  4. Bake per-row symmetric fake-quantized weights into a checkpoint copy.

Scope follows the QVLA paper: LLM backbone + vision tower Linears; action head
(DiT), projectors, embeddings and lm_head stay FP.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    ensure_libero_runtime,
    get_named_module,
    load_libero_samples,
    load_policy,
    normalized_input_no_inference,
    seed_everything,
)
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402

QVLA_INCLUDE = r".*backbone\.eagle_model\.(language_model|vision_model)\..*"
QVLA_EXCLUDE = r"(?:^|\.)(lm_head|embed|embeddings|norm|ln|layernorm|mlp1|rotary)(?:\.|$)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--dataset-path", default=os.environ.get("LIBERO_ROOT", "") + "/datasets/lerobot_libero_10")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task")
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-regex", default=QVLA_INCLUDE)
    p.add_argument("--exclude-regex", default=QVLA_EXCLUDE)
    p.add_argument("--scope", default="")
    p.add_argument("--bits", default="2,4,8,16")
    p.add_argument("--target-avg-bits", type=float, default=4.0)
    p.add_argument("--token-cap", type=int, default=1024)
    p.add_argument("--percdamp", type=float, default=0.01)
    return p.parse_args()


@torch.no_grad()
def quant_rows_sym(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Vectorized per-row symmetric fake quant (QVLA _quantize_row_sym)."""
    if bits >= 16:
        return W
    qmax = (1 << (bits - 1)) - 1
    scale = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
    return torch.clamp(torch.round(W / scale), -(qmax + 1), qmax) * scale


def diag_hinv(H: torch.Tensor, percdamp: float) -> torch.Tensor:
    n = H.shape[0]
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = percdamp * torch.diag(H).mean()
    idx = torch.arange(n, device=H.device)
    H[idx, idx] += damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    return torch.diag(U)


def greedy_allocate(proxies: dict, bit_list, target_avg: float):
    bit_desc = sorted(bit_list, reverse=True)
    highest = bit_desc[0]

    def next_lower(b):
        for bb in bit_desc:
            if bb < b:
                return bb
        return -1

    layer_bits, heap, step = {}, [], 0
    total_bits, n_ch = 0, 0
    for layer, pmap in proxies.items():
        C = next(iter(pmap.values())).numel()
        layer_bits[layer] = [highest] * C
        total_bits += highest * C
        n_ch += C
    for layer, pmap in proxies.items():
        nb = next_lower(highest)
        if nb < 0 or nb not in pmap:
            continue
        costs = pmap[nb]
        for ci in range(costs.numel()):
            heapq.heappush(heap, (float(costs[ci]) / (highest - nb), step, layer, ci, nb))
            step += 1
    avg = total_bits / max(1, n_ch)
    while heap and avg > target_avg:
        _, _, layer, ci, nb = heapq.heappop(heap)
        cur = layer_bits[layer][ci]
        if nb >= cur:
            continue
        total_bits -= cur - nb
        layer_bits[layer][ci] = nb
        avg = total_bits / n_ch
        nnb = next_lower(nb)
        if nnb >= 0 and nnb in proxies[layer]:
            cost = float(proxies[layer][nnb][ci])
            heapq.heappush(heap, (cost / (nb - nnb), step, layer, ci, nnb))
            step += 1
    hist = {}
    for arr in layer_bits.values():
        for b in arr:
            hist[b] = hist.get(b, 0) + 1
    return layer_bits, {"final_avg_bits": avg, "total_channels": n_ch, "bit_hist": hist}


def main():
    args = parse_args()
    ensure_libero_runtime()
    bits = sorted({int(x) for x in args.bits.split(",")})
    low_bits = [b for b in bits if b < 16]

    rng = np.random.default_rng(args.seed)
    print(f"[qvla] loading FP policy: {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()

    targets = select_targets(
        policy.model,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        scope_prefix=args.scope or None,
        whitelist=None,
        blacklist=None,
    )
    names = [n for n, _ in targets]
    print(f"[qvla] target layers: {len(names)}")
    assert names, "no target layers matched"

    print(f"[qvla] collecting {args.num_samples} LIBERO calib observations")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    normed = [
        {"seed": s["seed"], "normalized": normalized_input_no_inference(policy, s["obs"])}
        for s in samples
    ]
    print(f"[qvla] prepared {len(normed)} calib samples")

    # --- one-pass Gram collection over all target layers ---
    grams: dict[str, torch.Tensor] = {}
    tokens: dict[str, int] = {}
    handles = []

    def make_hook(lname):
        def hook(_m, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            if args.token_cap > 0 and x.shape[0] > args.token_cap:
                idx = torch.from_numpy(
                    rng.choice(x.shape[0], size=args.token_cap, replace=False)
                ).to(x.device)
                x = x.index_select(0, idx)
            with torch.autocast(device_type="cuda", enabled=False):
                gg = x.t() @ x
            g = grams.get(lname)
            if g is None:
                grams[lname] = gg.float()
            else:
                g += gg.float()
            tokens[lname] = tokens.get(lname, 0) + int(x.shape[0])
        return hook

    for lname in names:
        m = get_named_module(policy.model, lname)
        handles.append(m.register_forward_pre_hook(make_hook(lname)))

    autocast = (
        torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE)
        if args.device.startswith("cuda")
        else torch.no_grad()
    )
    with torch.no_grad(), autocast:
        for s in normed:
            seed_everything(s["seed"])
            policy.model.get_action(s["normalized"])
    for h in handles:
        h.remove()
    print(f"[qvla] grams collected for {len(grams)} layers")

    # --- proxy + allocation ---
    proxies: dict[str, dict[int, torch.Tensor]] = {}
    for lname in names:
        if lname not in grams:
            continue
        m = get_named_module(policy.model, lname)
        W = m.weight.data.float()
        H = grams[lname].float() / max(1, tokens[lname])
        d = diag_hinv(H.to(device=W.device, dtype=torch.float32), args.percdamp)
        d2 = (d.float() ** 2).clamp_min(1e-12)
        pmap = {}
        for b in low_bits:
            q = quant_rows_sym(W, b)
            pmap[b] = (((W - q) ** 2) / d2.unsqueeze(0)).sum(dim=1).cpu()
        proxies[lname] = pmap
        del grams[lname]
    print(f"[qvla] proxies done for {len(proxies)} layers")

    layer_bits, stats = greedy_allocate(proxies, bits, args.target_avg_bits)
    print(f"[qvla] allocation: {stats}")

    # --- bake fake-quantized weights ---
    baked = 0
    for lname, arr in layer_bits.items():
        m = get_named_module(policy.model, lname)
        W = m.weight.data
        bt = torch.tensor(arr, device=W.device)
        for b in low_bits:
            mask = bt == b
            if mask.any():
                W[mask] = quant_rows_sym(W[mask].float(), b).to(W.dtype)
        baked += 1
    print(f"[qvla] baked {baked} layers")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy.model.save_pretrained(str(out))
    src = Path(args.checkpoint)
    if (src / "experiment_cfg").is_dir():
        shutil.copytree(src / "experiment_cfg", out / "experiment_cfg", dirs_exist_ok=True)
    json.dump(
        {"stats": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in stats.items()},
         "assign": layer_bits},
        open(out / "qvla_gates.json", "w"),
    )
    print(f"[qvla] saved quantized checkpoint -> {out}")


if __name__ == "__main__":
    main()
