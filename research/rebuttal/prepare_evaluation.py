"""Predeclared LIBERO states/observations; no policy, no scored episodes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.deployment import sha256
from research.rebuttal.build import json_write, tensor_write


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--suite", choices=("object", "long"), default="object")
    p.add_argument("--init-start", type=int, default=20)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert a.init_start >= 20, "Earlier windows already inspected; require a separate disclosed diagnostic script"
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from tools.analyze_layerwise_quant_drift import ensure_libero_runtime, _convert_libero_observation

    ensure_libero_runtime()
    from libero.libero import benchmark
    from examples.Libero.eval.utils import get_libero_env, get_libero_dummy_action

    assert not a.output.exists(), "Preserve the existing state/observation freeze; use a new directory for a new protocol"
    a.output.mkdir(parents=True)
    calibration = json.loads((a.calibration / "manifest.json").read_text())
    calibration_hashes = {t["initial_state_sha256"] for t in calibration["trajectories"]}
    suite_name = "libero_object" if a.suite == "object" else "libero_10"
    suite = benchmark.get_benchmark_dict()[suite_name]()
    rows, samples = [], []
    for task_id in range(10):
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        env, description = get_libero_env(task, resolution=256)
        try:
            for init in range(a.init_start, a.init_start + a.trials):
                digest = hashlib.sha256(np.asarray(states[init]).tobytes()).hexdigest()
                assert digest not in calibration_hashes, (task_id, init, "calibration/evaluation state overlap")
                noise_seed = 42 + task_id * 10000 + init * 1000
                seed(noise_seed)
                env.seed(0)
                env.reset()
                obs = env.set_init_state(states[init])
                for _ in range(10):
                    obs, _, _, _ = env.step(get_libero_dummy_action())
                samples.append({"task_id": task_id, "init_index": init, "seed": noise_seed, "obs": _convert_libero_observation(obs, task.language)})
                rows.append({"task_id": task_id, "init_index": init, "task": description, "initial_state_sha256": digest})
            print("FROZEN_EVALUATION_STATES", task_id, len(rows), flush=True)
        finally:
            env.close()
    tensor_write(a.output / "observations.pt", samples)
    manifest = {"suite": suite_name, "init_indices": list(range(a.init_start, a.init_start + a.trials)),
                "episodes": rows, "episode_count": len(rows), "raw_resolution": 256,
                "settling_steps": 10, "env_seed_per_episode": 0,
                "policy_noise_seed": "42 + task_id*10000 + init_index*1000 + control_step",
                "protocol_note": "No policy evaluated; freeze state hashes and settled observations before scoring. Repeat env.seed(0) per episode in all new callers.",
                "calibration_state_hashes": sorted(calibration_hashes), "calibration_disjoint_verified": True,
                "observations_sha256": sha256(a.output / "observations.pt"),
                "source_sha256": sha256(Path(__file__)), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    json_write(a.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
