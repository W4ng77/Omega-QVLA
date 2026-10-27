"""Collect ten actual full-precision Object trajectories, one initial state/task."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--init-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--observations-per-trajectory", type=int, default=10)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith(("GR00T_GPTQ", "GR00T_DUQUANT", "GR00T_RTN", "GR00T_ATM", "GR00T_OHB", "HOLOQ_PACK")):
            os.environ.pop(key)
    os.environ["HOLOQ_COMPUTE_DTYPE"] = "float16"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed(args.seed)
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy, COMPUTE_DTYPE
    from tools.analyze_layerwise_quant_drift import ensure_libero_runtime, _convert_libero_observation

    ensure_libero_runtime()
    from libero.libero import benchmark
    from examples.Libero.eval.utils import get_libero_env, get_libero_dummy_action
    from examples.Libero.eval.run_libero_eval import ExternalPolicy, get_max_steps

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )
    policy.warmup(num_steps=4)
    converter = ExternalPolicy.__new__(ExternalPolicy)
    converter.policy_output_format = "gr00t_dict"
    converter.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    manifest = {
        "checkpoint": args.checkpoint,
        "compute_dtype": str(COMPUTE_DTYPE),
        "seed": args.seed,
        "suite": "libero_object",
        "init_index": args.init_index,
        "denoising_steps": 8,
        "action_horizon": 16,
        "replan": "first action each simulator step",
        "raw_resolution": 256,
        "trajectory_count": 10,
        "frame_sampling": "ten evenly spaced observed frames per trajectory",
        "started": time.time(),
        "trajectories": [],
    }
    selected = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        env, description = get_libero_env(task, resolution=256)
        observations = []
        try:
            env.reset()
            states = suite.get_task_init_states(task_id)
            obs = env.set_init_state(states[args.init_index])
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            done = False
            for step in range(get_max_steps("libero_object", "groot")):
                converted = _convert_libero_observation(obs, task.language)
                observations.append(converted)
                seed(args.seed + task_id * 10000 + args.init_index * 1000 + step)
                result = policy.get_action(converted)
                action = converter._convert_to_libero_action(result, 0)
                if not np.isfinite(action).all():
                    raise RuntimeError("Non-finite full-precision calibration action")
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
            path = args.output / f"trajectory_{task_id:02d}.pt"
            torch.save(observations, path)
            idx = np.linspace(
                0, len(observations) - 1, min(len(observations), args.observations_per_trajectory), dtype=int
            )
            for step in idx:
                selected.append(
                    {
                        "task_id": task_id,
                        "init_index": args.init_index,
                        "trajectory_step": int(step),
                        "seed": args.seed + task_id * 10000 + int(step),
                        "obs": observations[int(step)],
                    }
                )
            manifest["trajectories"].append(
                {
                    "task_id": task_id,
                    "task": description,
                    "steps": len(observations),
                    "success": bool(done),
                    "selected_steps": idx.tolist(),
                    "initial_state_sha256": hashlib.sha256(
                        np.asarray(states[args.init_index]).tobytes()
                    ).hexdigest(),
                }
            )
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(manifest["trajectories"][-1], flush=True)
        finally:
            env.close()
    torch.save(selected, args.output / "observations.pt")
    manifest.update(finished=time.time(), observation_count=len(selected))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Calibration complete:", len(selected), "observations from ten actual trajectories", flush=True)


if __name__ == "__main__":
    main()
