"""Prepare fixed replay observations and independent evaluation state hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from tools.analyze_layerwise_quant_drift import ensure_libero_runtime, _convert_libero_observation

    ensure_libero_runtime()
    from libero.libero import benchmark
    from examples.Libero.eval.utils import get_libero_env, get_libero_dummy_action

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    samples = []
    hashes = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        env, _ = get_libero_env(task, resolution=256)
        try:
            states = suite.get_task_init_states(task_id)
            for init in range(10, 20):
                env.reset()
                obs = env.set_init_state(states[init])
                for _ in range(10):
                    obs, _, _, _ = env.step(get_libero_dummy_action())
                samples.append(
                    {
                        "task_id": task_id,
                        "init_index": init,
                        "obs": _convert_libero_observation(obs, task.language),
                    }
                )
                hashes.append(
                    {
                        "task_id": task_id,
                        "init_index": init,
                        "sha256": hashlib.sha256(np.asarray(states[init]).tobytes()).hexdigest(),
                    }
                )
        finally:
            env.close()
    torch.save(samples, args.output / "observations.pt")
    (args.output / "manifest.json").write_text(
        json.dumps({"suite": "libero_object", "states": hashes}, indent=2) + "\n"
    )
    print("Held-out initial observations:", len(samples), flush=True)


if __name__ == "__main__":
    main()
