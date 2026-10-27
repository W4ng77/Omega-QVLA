"""Full-policy timing on identical saved rollout observations and fixed seeds."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from research.native_int4.summarize_libero import stats


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--pack", required=True)
    p.add_argument("--engine", choices=["native_s4", "triton"], required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--compare-engines", action="store_true")
    args = p.parse_args()
    os.environ.update(
        GR00T_GPTQ="1",
        GR00T_GPTQ_PATH=args.pack,
        GR00T_GPTQ_SCOPE="",
        GR00T_GPTQ_ABITS="4",
        GR00T_GPTQ_MISSING="error",
        GR00T_PERSTEP_ENGINE=args.engine,
        GR00T_CALIB_SUITE="libero_object",
    )
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from research.native_int4.run_libero import CHECKPOINT

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=CHECKPOINT,
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )
    policy.warmup(num_steps=32)
    report = {"pack": args.pack, "engine": args.engine, "warmup": 32, "rows": []}
    outputs = {}
    torch.cuda.reset_peak_memory_stats()
    paths = sorted(args.observations.glob("observation_*.pt"))
    if len(paths) != 10:
        raise ValueError(f"Expected one observation per task, got {len(paths)}")
    for path in paths:
        obs = torch.load(path, map_location="cpu", weights_only=False)
        values = {engine: [] for engine in (["triton", "native_s4"] if args.compare_engines else [args.engine])}
        action_errors = []
        for rep in range(args.repeats + 1):
            engines = list(values)
            if rep % 2:
                engines.reverse()
            results = {}
            for engine in engines:
                os.environ["GR00T_PERSTEP_ENGINE"] = engine
                torch.manual_seed(1000 + rep)
                np.random.seed(1000 + rep)
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = policy.get_action(obs)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                results[engine] = result
                if rep:
                    values[engine].append(ms)
            if not rep:
                outputs[path.name] = results
            if args.compare_engines:
                old = np.concatenate([np.asarray(v).reshape(-1) for v in results["triton"].values()])
                new = np.concatenate([np.asarray(results["native_s4"][k]).reshape(-1) for k in results["triton"]])
                action_errors.append(float(np.square(new - old).sum() / max(float(np.square(old).sum()), 1e-30)))
        report["rows"].append(
            {
                "observation": path.name,
                "task": str(obs.get("annotation.human.action.task_description")),
                "values_ms": values,
                "engines": {engine: stats(v) for engine, v in values.items()},
                "action_nmse_max": max(action_errors, default=0),
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(path.name, report["rows"][-1]["engines"], flush=True)
    report["overall"] = {
        engine: stats([v for row in report["rows"] for v in row["values_ms"][engine]]) for engine in values
    }
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    native = [m for m in policy.model.modules() if getattr(m, "_native_calls", 0)]
    report["active_native_layers"] = len(native)
    report["native_calls"] = sum(m._native_calls for m in native)
    if args.engine == "native_s4" and len(native) != 96:
        raise RuntimeError("Full policy did not use all 96 native layers")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(outputs, args.output.with_suffix(".actions.pt"))
    print(report["overall"], flush=True)


if __name__ == "__main__":
    main()
