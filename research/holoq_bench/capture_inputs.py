"""Shared teacher activations from uniformly sampled frames of ten trajectories."""

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import torch

from research.holoq_bench.collect_calibration import seed

LLM = r"backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
DIT = r"action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))$"


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "layers").mkdir()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed(42)
    os.environ["HOLOQ_COMPUTE_DTYPE"] = "float16"
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.quantization.dit_step_context import get_current_dit_step

    cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )
    linears = {
        name: module for name, module in policy.model.named_modules() if isinstance(module, torch.nn.Linear)
    }
    targets = {name: module for name, module in linears.items() if re.search(LLM, name) or re.search(DIT, name)}
    assert len(targets) == 180, len(targets)
    weights = {
        name: {
            "weight": m.weight.detach().half().cpu(),
            "bias": m.bias.detach().half().cpu() if m.bias is not None else None,
        }
        for name, m in targets.items()
    }
    torch.save(weights, args.output / "weights.pt")
    selected = torch.load(args.calibration / "observations.pt", weights_only=False, map_location="cpu")
    assert len(selected) == 100 and len({s["task_id"] for s in selected}) == 10
    manifest = {
        "calibration": str(args.calibration.resolve()),
        "checkpoint": args.checkpoint,
        "observations": len(selected),
        "max_rows_per_observation_per_layer_step": 52,
        "selected_layers": len(targets),
        "all_linear_layers": [],
        "layers": [],
    }
    for name, m in linears.items():
        part = (
            "language"
            if "language_model" in name
            else ("action_head" if name.startswith("action_head") else "vision_or_other")
        )
        manifest["all_linear_layers"].append(
            {"name": name, "K": m.in_features, "N": m.out_features, "part": part, "selected": name in targets}
        )
    buffers = {name: {} for name in targets}
    dtypes = {name: set() for name in targets}
    generators = {name: np.random.default_rng(42 + i) for i, name in enumerate(targets)}

    def hook(name):
        def collect(module, args):
            step = get_current_dit_step()
            bank = int(step) if step is not None else -1
            x = args[0].detach().reshape(-1, module.in_features)
            dtypes[name].add(str(x.dtype))
            n = min(52, x.shape[0])
            idx = generators[name].choice(x.shape[0], n, replace=False)
            chunk = x[torch.as_tensor(idx, device=x.device)].half().cpu()
            if not torch.isfinite(chunk).all():
                raise RuntimeError(f"Nonfinite calibration activations: {name}/{bank}")
            buffers[name].setdefault(bank, []).append(chunk)

        return collect

    handles = [m.register_forward_pre_hook(hook(name)) for name, m in targets.items()]
    try:
        for i, sample in enumerate(selected):
            seed(sample["seed"])
            policy.get_action(sample["obs"])
            if i % 10 == 9:
                print("Captured observations", i + 1, "/", len(selected), flush=True)
    finally:
        for h in handles:
            h.remove()
    for i, name in enumerate(targets):
        data = {bank: torch.cat(chunks) for bank, chunks in buffers[name].items()}
        expected = {-1} if "language_model" in name else set(range(8))
        assert set(data) == expected, (name, set(data))
        filename = f"layers/{i:03d}.pt"
        torch.save(data, args.output / filename)
        manifest["layers"].append(
            {
                "name": name,
                "file": filename,
                "input_dtypes": sorted(dtypes[name]),
                "rows": {str(bank): len(x) for bank, x in data.items()},
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Shared teacher capture complete:", len(targets), "layers", flush=True)


if __name__ == "__main__":
    main()
