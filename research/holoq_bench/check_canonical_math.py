"""Accuracy-only dense oracle of frozen HoloQ grids; never a speed baseline.

Check real layer inputs, every denoising-step scale, rotations and full policy
actions. Dense floating weights exist only in this diagnostic process. The
deployment implementation and all measured benchmark rows remain packed.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from torch import nn
from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.packed_runtime import PackedLinear, apply_blocks
from gr00t.quantization.dit_step_context import get_current_dit_step


class DenseCanonicalOracle(nn.Module):
    def __init__(self, source, weight):
        super().__init__()
        self.source = source
        self.register_buffer("oracle_weight", weight)

    def forward(self, x):
        src = self.source
        x = x.half()
        rin = getattr(src, "_fused_in", src.rot_in)
        rout = getattr(src, "_fused_out", src.rot_out)
        bias = getattr(src, "_fused_bias", src.bias)
        if rin.numel():
            x = apply_blocks(x, rin, src.perm)
        if src.backend == "native_s4":
            step = get_current_dit_step()
            assert step is not None and 0 <= step < 8
            # Independently unpack the actual converted bank and compute its
            # integer-grid formula. Never compare conversion to original W.
            p, s, a = src.integer.selected(step)
            y = src.integer.reference(x, p, s, a).half()
            if rout.numel():
                y = apply_blocks(y, rout)
            return y + bias if bias.numel() else y
        if getattr(src, "legacy_activation_arithmetic", False):
            if src.act_mode == "step":
                step = get_current_dit_step()
                assert step is not None and 0 <= step < 8
                scale = src.a_scale[step].half()
            else:
                scale = src.a_scale.half()
            xq = ((x / scale).round().clamp(-8, 7) * scale).half()
        elif getattr(src, "act_mode", "token") == "token":
            scale = (x.float().abs().amax(-1, keepdim=True).clamp_min(1e-5) / 7).half().float()
        else:
            step = get_current_dit_step()
            assert step is not None and 0 <= step < 8
            scale = src.a_scale[step]
        if not getattr(src, "legacy_activation_arithmetic", False):
            xq = ((x.float() / scale).round().clamp(src.act_qmin, 7) * scale).half()
        with torch.autocast("cuda", enabled=False):
            y = torch.nn.functional.linear(xq.float(), self.oracle_weight.float()).half()
        if rout.numel():
            y = apply_blocks(y, rout)
        if bias.numel():
            y = y + bias
        return y


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    deployment = args.deployment or args.root / "deployments/holoq"
    os.environ.update(HOLOQ_DEPLOY=str(deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16")
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=str(deployment.resolve()), modality_config=config.modality_config(),
                         modality_transform=config.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    modules = {name: m for name, m in policy.model.named_modules() if isinstance(m, PackedLinear)}
    oracles = {}
    cache = args.cache or args.root / "packs/layer_cache/holoq"
    for file in sorted(cache.glob("*.pt")):
        entry = torch.load(file, map_location="cpu", weights_only=False)
        canonical = entry["canonical"]
        weight = (canonical["q"].float() * canonical["scales"].float().repeat_interleave(canonical["group"], 1)).half().cuda()
        oracles[entry["name"]] = DenseCanonicalOracle(modules[entry["name"]], weight)
    assert set(oracles) == set(modules) and len(modules) == 180
    errors = defaultdict(list)

    def hook(name):
        def check(module, inputs, output):
            reference = oracles[name](inputs[0])
            actual, reference = output.float(), reference.float()
            num = (actual - reference).square().sum().item()
            den = reference.square().sum().item()
            errors[name].append({"step": get_current_dit_step(), "nmse": num / max(den, 1e-30),
                                 "max_abs": (actual - reference).abs().max().item(),
                                 "reference_peak": reference.abs().max().item(),
                                 "finite": bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())})
        return check

    handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
    observations = torch.load(args.root / "heldout_init10_19/observations.pt", map_location="cpu", weights_only=False)
    selected = [observations[i * 10] for i in range(10)]
    packed = []
    with torch.inference_mode():
        for sample in selected:
            seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
            packed.append(policy.get_action(sample["obs"]))
    for handle in handles:
        handle.remove()
    names = dict(policy.model.named_modules())
    for name, oracle in oracles.items():
        parent, attr = name.rsplit(".", 1)
        setattr(names[parent], attr, oracle)
    action_errors = []
    with torch.inference_mode():
        for sample, actual in zip(selected, packed):
            seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
            reference = policy.get_action(sample["obs"])
            num = sum(np.square(actual[k].astype(float) - reference[k].astype(float)).sum() for k in actual)
            den = sum(np.square(reference[k].astype(float)).sum() for k in actual)
            action_errors.append({"task_id": sample["task_id"], "init_index": sample["init_index"],
                                  "nmse": float(num / max(den, 1e-30)),
                                  "max_abs": max(float(np.max(np.abs(actual[k] - reference[k]))) for k in actual)})
    rows = {name: {"max_nmse": max(v["nmse"] for v in values), "calls": len(values), "comparisons": values} for name, values in errors.items()}
    output = {"purpose": "accuracy-only oracle; no speed/storage/memory baseline", "deployment": str(deployment),
              "command": sys.argv, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_pack_sha256": json.loads((deployment / "deployment.json").read_text())["source_pack_sha256"],
              "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "denoising_steps": 8, "layers": rows, "full_policy_action_errors": action_errors,
              "max_layer_nmse": max(r["max_nmse"] for r in rows.values()),
              "finite": all(v["finite"] for values in errors.values() for v in values)}
    output["review_required"] = not output["finite"] or output["max_layer_nmse"] > 1e-3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("CANONICAL_MATH_ORACLE", output["max_layer_nmse"], "finite", output["finite"], "action_nmse", [v["nmse"] for v in action_errors], flush=True)


if __name__ == "__main__":
    main()
