"""Refit official QuantVLA ATM/OHB on the same traced calibration trajectories."""

import argparse
import ast
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import torch

from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.quantizers import ROOT


def upstream_components():
    source = ROOT / "third_party/QuantVLA/tools/calibrate_atm_dit.py"
    names = {"StdCollector", "RMSCollector", "compute_alpha_json", "compute_beta_values"}
    tree = ast.parse(source.read_text())
    nodes = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    namespace = {"torch": torch, "defaultdict": defaultdict, "Dict": Dict, "List": List, "math": math}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace
    )
    return namespace


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--packs", nargs="+", required=True, type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["HOLOQ_COMPUTE_DTYPE"] = "float16"
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.atm import register_atm_capture, register_ohb_capture, clear_atm_capture

    components = upstream_components()
    samples = torch.load(args.observations, map_location="cpu", weights_only=False)

    def collect(pack):
        if pack is None:
            os.environ.pop("HOLOQ_PACK", None)
        else:
            os.environ["HOLOQ_PACK"] = str(pack)
        seed(42)
        cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
        policy = Gr00tPolicy(
            model_path=args.checkpoint,
            modality_config=cfg.modality_config(),
            modality_transform=cfg.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=8,
        )
        std, rms = components["StdCollector"](), components["RMSCollector"]()
        register_atm_capture(policy.model, std)
        register_ohb_capture(policy.model, rms)
        for i, sample in enumerate(samples):
            seed(sample["seed"])
            policy.get_action(sample["obs"])
            if i % 20 == 19:
                print("ATM calibration", pack, i + 1, flush=True)
        clear_atm_capture(policy.model)
        result = {"std": {k: v.cpu() for k, v in std.finalize().items()}, "rms": rms.finalize()}
        del policy, std, rms
        gc.collect()
        torch.cuda.empty_cache()
        return result

    teacher_file = args.output / "atm_teacher.pt"
    if teacher_file.exists():
        teacher = torch.load(teacher_file, weights_only=False)
    else:
        teacher = collect(None)
        torch.save(teacher, teacher_file)
    for path in args.packs:
        output = path.with_name(path.stem + "_atm.pt")
        if output.exists():
            print("REUSE ATM", output, flush=True)
            continue
        student = collect(path)
        alpha = components["compute_alpha_json"](teacher["std"], student["std"])
        beta = components["compute_beta_values"](teacher["rms"], student["rms"], log_clamp=0.30, neutral=0.03)
        for name, entry in alpha.items():
            entry["beta"] = beta[name]
        if len(alpha) != 16:
            raise ValueError(f"Expected 16 calibrated attention modules, got {len(alpha)}")
        packed = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        packed["atm"] = alpha
        packed["meta"].update(atm_ohb="recalibrated_shared_100_frames", atm_observations=args.observations)
        torch.save(packed, output)
        (args.output / (path.stem + "_atm.json")).write_text(json.dumps(alpha, indent=2) + "\n")
        torch.save(student, args.output / (path.stem + "_atm_student.pt"))
        print("EXPORTED", output, flush=True)


if __name__ == "__main__":
    main()
