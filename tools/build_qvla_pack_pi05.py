#!/usr/bin/env python
"""QVLA (arXiv 2602.03782) port for pi0.5 (openpi PyTorch): channel-wise bits.

Same recipe as build_qvla_pack.py (GR00T): per-layer Gram -> Hessian proxy ->
global greedy allocation over {2,4,8,16} to avg 4.0 -> bake per-row symmetric
fake-quant into a checkpoint copy. Scope: PaliGemma language model + vision
tower Linears; the Gemma action expert (action head), projector, embeddings
and small heads stay FP, matching the QVLA paper's exclusions.

Run inside the openpi venv.
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

QVLA_INCLUDE = r".*paligemma_with_expert\.paligemma\.model\.(language_model|vision_tower)\..*"
QVLA_EXCLUDE = r"(?:^|\.)(lm_head|embed|embed_tokens|embeddings|norm|ln|layernorm|multi_modal_projector|rotary)(?:\.|$)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="pi05 PyTorch checkpoint dir")
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--obs-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--include-regex", default=QVLA_INCLUDE)
    p.add_argument("--exclude-regex", default=QVLA_EXCLUDE)
    p.add_argument("--bits", default="2,4,8,16")
    p.add_argument("--target-avg-bits", type=float, default=4.0)
    p.add_argument("--token-cap", type=int, default=1024)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def quant_rows_sym(W: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return W
    qmax = (1 << (bits - 1)) - 1
    scale = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
    return torch.clamp(torch.round(W / scale), -(qmax + 1), qmax) * scale


def diag_hinv(H: torch.Tensor, percdamp: float) -> torch.Tensor:
    n = H.shape[0]
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    idx = torch.arange(n, device=H.device)
    H[idx, idx] += percdamp * torch.diag(H).mean()
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    return torch.diag(U)


def greedy_allocate(proxies, bit_list, target_avg):
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
            heapq.heappush(heap, (float(proxies[layer][nnb][ci]) / (nb - nnb), step, layer, ci, nnb))
            step += 1
    hist = {}
    for arr in layer_bits.values():
        for b in arr:
            hist[b] = hist.get(b, 0) + 1
    return layer_bits, {"final_avg_bits": avg, "total_channels": n_ch, "bit_hist": hist}


def main():
    args = parse_args()
    bits = sorted({int(x) for x in args.bits.split(",")})
    low_bits = [b for b in bits if b < 16]

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    print(f"[qvla-pi05] loading policy from {args.checkpoint}")
    train_cfg = _config.get_config(args.data_config)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint, pytorch_device=args.device)
    assert getattr(policy, "_is_pytorch_model", False), "need a PyTorch pi05 checkpoint"
    model = policy._model
    model.eval()

    inc = re.compile(args.include_regex)
    exc = re.compile(args.exclude_regex)
    names = [
        n for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and inc.search(n) and not exc.search(n)
    ]
    print(f"[qvla-pi05] target layers: {len(names)}")
    assert names, "no target layers matched; check include regex against model names"

    data = torch.load(args.obs_path, weights_only=False)
    samples = data["samples"] if isinstance(data, dict) else data
    samples = samples[: args.num_samples]
    print(f"[qvla-pi05] {len(samples)} calib obs")

    name_to_mod = dict(model.named_modules())
    grams: dict[str, torch.Tensor] = {}
    tokens: dict[str, int] = {}
    import numpy as np
    rng = np.random.default_rng(args.seed)
    handles = []

    def make_hook(lname):
        def hook(_m, inputs):
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x = x.detach().reshape(-1, x.shape[-1]).float()
            if args.token_cap > 0 and x.shape[0] > args.token_cap:
                idx = torch.from_numpy(rng.choice(x.shape[0], size=args.token_cap, replace=False)).to(x.device)
                x = x.index_select(0, idx)
            with torch.autocast(device_type="cuda", enabled=False):
                gg = (x.t() @ x).float()
            if lname in grams:
                grams[lname] += gg
            else:
                grams[lname] = gg
            tokens[lname] = tokens.get(lname, 0) + int(x.shape[0])
        return hook

    for lname in names:
        handles.append(name_to_mod[lname].register_forward_pre_hook(make_hook(lname)))
    with torch.no_grad():
        for i, obs in enumerate(samples, 1):
            policy.infer(obs)
            if i % 10 == 0:
                print(f"[qvla-pi05] calib {i}/{len(samples)}")
    for h in handles:
        h.remove()
    print(f"[qvla-pi05] grams for {len(grams)} layers")

    proxies = {}
    for lname in names:
        if lname not in grams:
            continue
        W = name_to_mod[lname].weight.data.float()
        H = grams[lname] / max(1, tokens[lname])
        d = diag_hinv(H.to(W.device), args.percdamp)
        d2 = (d.float() ** 2).clamp_min(1e-12)
        pmap = {}
        for b in low_bits:
            q = quant_rows_sym(W, b)
            pmap[b] = (((W - q) ** 2) / d2.unsqueeze(0)).sum(dim=1).cpu()
        proxies[lname] = pmap
        del grams[lname]

    layer_bits, stats = greedy_allocate(proxies, bits, args.target_avg_bits)
    print(f"[qvla-pi05] allocation: {stats}")

    for lname, arr in layer_bits.items():
        m = name_to_mod[lname]
        W = m.weight.data
        bt = torch.tensor(arr, device=W.device)
        for b in low_bits:
            mask = bt == b
            if mask.any():
                W[mask] = quant_rows_sym(W[mask].float(), b).to(W.dtype)
    print(f"[qvla-pi05] baked {len(layer_bits)} layers")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.checkpoint)
    for item in src.iterdir():
        if item.name == "model.safetensors":
            continue
        if item.is_dir():
            shutil.copytree(item, out / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, out / item.name)
    from safetensors.torch import save_file

    sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    json.dump(
        {"stats": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in stats.items()},
         "assign": layer_bits},
        open(out / "qvla_gates.json", "w"),
    )
    print(f"[qvla-pi05] saved -> {out}")


if __name__ == "__main__":
    main()
