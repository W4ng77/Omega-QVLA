"""After task evaluations, offer the same public optimization to all methods."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from research.holoq_bench.main_selection import COHORTS, read_selection


def main():
    root = Path("results/holoq_object")
    while True:
        ready = [json.loads((root / cohort / "queue.json").read_text()).get("finished")
                 if (root / cohort / "queue.json").exists() else False for cohort in COHORTS]
        if all(ready):
            break
        time.sleep(15)
    cases, _, _ = read_selection(root)
    out = root / "common_logits_confirmed"
    out.mkdir(parents=True, exist_ok=True)
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    env = os.environ.copy()
    env["HOLOQ_PILOT_LOCK_FD"] = str(guard.fileno())
    status = {"started": time.time(), "cases": cases, "commands": [], "results": {}}
    rows = []

    def save():
        temp = out / "queue.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(out / "queue.json")

    save()
    for case in cases:
        directory = out / case
        directory.mkdir(parents=True, exist_ok=True)
        result = directory / "result.json"
        if not result.exists():
            cmd = [sys.executable, "-u", "-m", "research.holoq_bench.common_logits_worker",
                   "--deployment", str(root / "deployments" / case), "--observations",
                   str(root / "heldout_init10_19/observations.pt"), "--output", str(directory)]
            status["commands"].append({"cmd": cmd, "started": time.time(), "borrowed_lock": True})
            save()
            print("ALL_COMMON_START", case, flush=True)
            with (directory / "run.log").open("a") as log:
                ret = subprocess.run(cmd, env=env, pass_fds=(guard.fileno(),), stdout=log, stderr=subprocess.STDOUT)
            status["results"][case] = {"returncode": ret.returncode, "finished": time.time()}
        if result.exists():
            data = json.loads(result.read_text())
            max_error = max(row["nmse"] for row in data["full_action_comparisons"])
            validation = "failed" if max_error >= data.get("difference_guard_nmse", .02) else (
                "bitwise" if not data.get("needs_independent_accuracy_evaluation", True) else "within_guard_requires_rollout")
            rows.append({"case": case, "before_mean_ms": data["before"]["mean_ms"],
                         "after_mean_ms": data["after"]["mean_ms"], "after_p50_ms": data["after"]["p50_ms"],
                         "after_p95_ms": data["after"]["p95_ms"], "speedup": data["speedup"],
                         "features_exact": data["all_100_backbone_features_exact"],
                         "actions_exact_count": data["full_actions_exact_count"],
                         "validation_status": validation,
                         "needs_independent_accuracy_evaluation": data.get("needs_independent_accuracy_evaluation", True),
                         "allocated_before": data["before_memory"]["peak_allocated_bytes"],
                         "allocated_after": data["after_memory"]["peak_allocated_bytes"],
                         "source_pack_sha256": data["source_pack_sha256"]})
        save()
    if rows:
        with (out / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"rows": rows, "scope": "Contiguous common optimization replay/semantic checks; no task score assigned"}, indent=2) + "\n")
    status["finished"] = time.time()
    save()
    print("ALL_COMMON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
