"""Compare a common dead-logits optimization on complete packed policies.

Existing rollout scores are never relabeled. Require bitwise hidden features
and report all full-action differences before making any accuracy reuse claim.
Original frozen evaluation runners/packs remain unchanged.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.common_logits import install
from research.holoq_bench.run_policy import stats, memory


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    lock = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "result.json").exists(), "Retain existing result; use a new directory"
    os.environ.update(HOLOQ_DEPLOY=str(args.deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from research.holoq_bench.graphs import install as install_graphs

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=str(args.deployment.resolve()), modality_config=config.modality_config(),
                         modality_transform=config.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    graphs = install_graphs(policy)
    observations = torch.load(args.observations, map_location="cpu", weights_only=False)
    assert len(observations) == 100
    for i in range(32):
        seed(2000 + i)
        policy.get_action(observations[(i % 10) * 10]["obs"])

    def invoke(sample):
        seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
        return policy.get_action(sample["obs"])

    def features_and_actions():
        original = policy.model.action_head.get_action
        hashes = []

        def capture(backbone, action):
            value = backbone["backbone_features"].detach().cpu().numpy()
            hashes.append(hashlib.sha256(value.tobytes()).hexdigest())
            return original(backbone, action)

        policy.model.action_head.get_action = capture
        try:
            actions = [invoke(sample) for sample in observations]
        finally:
            policy.model.action_head.get_action = original
        assert len(hashes) == 100
        return hashes, actions

    def timed():
        values = []
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(3):
            for sample in observations:
                seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
                torch.cuda.synchronize()
                start = time.perf_counter()
                action = policy.get_action(sample["obs"])
                torch.cuda.synchronize()
                values.append({"repeat": repeat, "task_id": sample["task_id"], "init_index": sample["init_index"],
                               "ms": (time.perf_counter() - start) * 1000})
                assert all(np.isfinite(v).all() for v in action.values())
        return stats([row["ms"] for row in values]), memory(), values

    before_features, before_actions = features_and_actions()
    # Some packed floating-MMA backends have pre-existing numerical variance.
    # Measure self-repeat before attributing any difference to this optimization.
    self_features, self_actions = features_and_actions()
    self_errors = []
    for a, b in zip(before_actions, self_actions):
        num = sum(np.square(a[k].astype(float) - b[k].astype(float)).sum() for k in a)
        den = sum(np.square(a[k].astype(float)).sum() for k in a)
        self_errors.append(float(num / max(den, 1e-30)))
    before, before_memory, before_calls = timed()
    state = install(policy)
    for i in range(32):
        seed(2000 + i)
        policy.get_action(observations[(i % 10) * 10]["obs"])
    after_features, after_actions = features_and_actions()
    after, after_memory, after_calls = timed()
    differences = []
    exact = 0
    for sample, a, b in zip(observations, before_actions, after_actions):
        equal = all(np.array_equal(a[key], b[key]) for key in a)
        exact += equal
        num = sum(np.square(a[k].astype(float) - b[k].astype(float)).sum() for k in a)
        den = sum(np.square(a[k].astype(float)).sum() for k in a)
        differences.append({"task_id": sample["task_id"], "init_index": sample["init_index"],
                            "exact": equal, "nmse": float(num / max(den, 1e-30))})
    output = {"command": sys.argv, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_pack_sha256": json.loads((args.deployment / "deployment.json").read_text())["source_pack_sha256"],
              "optimization": state["name"], "source_sha256": hashlib.sha256(Path(__file__).with_name("common_logits.py").read_bytes()).hexdigest(),
              "before": before, "after": after, "speedup": before["mean_ms"] / after["mean_ms"],
              "before_memory": before_memory, "after_memory": after_memory,
              "graph_count": len(graphs.cache), "all_100_backbone_features_exact": before_features == after_features,
              "baseline_self_feature_exact_count": sum(a == b for a, b in zip(before_features, self_features)),
              "baseline_self_action_nmse": self_errors,
              "full_actions_exact_count": int(exact), "full_action_comparisons": differences,
              "logits_shapes_after": sorted(state["logits_shapes"]),
              "scope": "Complete get_action; same8 steps/horizon16/seed/input. No profiler during timing. Fixed packs and main evaluation unchanged.",
              "accuracy_note": "No LIBERO task score is assigned to this timing/semantic pilot. If actions differ, independent rollout is mandatory before any accuracy claim.",
              "memory_note": "Same process: reserved memory retains pre-optimization allocator cache; peak allocated reflects each replay phase."}
    (args.output / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    for label, rows in [("before", before_calls), ("after", after_calls)]:
        (args.output / (label + "_calls.jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print("COMMON_LOGITS_PILOT", args.deployment.name, "before", before["mean_ms"], "after", after["mean_ms"],
          "features_exact", output["all_100_backbone_features_exact"], "actions_exact", exact, flush=True)
    output["difference_guard_nmse"] = max(.02, 10 * max(self_errors))
    output["needs_independent_accuracy_evaluation"] = exact != 100 or before_features != after_features
    (args.output / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    if max(row["nmse"] for row in differences) >= output["difference_guard_nmse"]:
        raise SystemExit("Large policy drift beyond self-repeat variance; investigate")


if __name__ == "__main__":
    main()
