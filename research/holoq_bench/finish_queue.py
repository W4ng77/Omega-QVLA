"""Run native-S4 deployment additions after the primary queue, then merge audits."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    a = p.parse_args()
    primary = a.root / "evaluation_v3"
    extra = a.root / "evaluation_best_s4"
    while True:
        status = json.loads((primary / "queue.json").read_text())
        if status.get("finished"):
            break
        time.sleep(15)
    print("PRIMARY_FINISHED; submitting explicit S4 additions", flush=True)
    env = os.environ.copy()
    env.update(GPU="0", CUDA_HOME="/usr/local/cuda-12.4", RUN_OUTPUT=str(extra))
    command = [
        "bash",
        "scripts/run_holoq_object_baselines.sh",
        "--cases",
        "smoothquant_a4_cutlass",
        "holoq_llm_s4",
        "holoq_step_weight_llm_s4",
    ]
    subprocess.run(command, env=env, check=True)
    merged = a.root / "evaluation_all"
    merged.mkdir(parents=True, exist_ok=True)
    records = []
    for root in (primary, extra):
        data = json.loads((root / "queue.json").read_text())
        records.append(data)
        for case in data["valid_cases"]:
            shutil.copytree(root / case, merged / case, dirs_exist_ok=True)
    common = records[0]["common_mode"]
    combined = {
        "common_mode": common,
        "input_queues": [str(primary), str(extra)],
        "queue_modes": [d["common_mode"] for d in records],
        "cases": {case: values for data in records for case, values in data["cases"].items()},
        "valid_cases": [case for data in records for case in data["valid_cases"]],
        "commands": [cmd for data in records for cmd in data["commands"]],
        "finished": time.time(),
    }
    (merged / "queue.json").write_text(json.dumps(combined, indent=2) + "\n")
    with (merged / "gpu_monitor.jsonl").open("w") as file:
        for root in (primary, extra):
            file.write((root / "gpu_monitor.jsonl").read_text())
    shutil.copy2(primary / "environment.txt", merged / "environment.txt")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.holoq_bench.summarize",
            "--runs",
            str(merged),
            "--deployments",
            str(a.root / "deployments"),
            "--output",
            str(merged / "report"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.holoq_bench.provenance",
            "--root",
            str(a.root),
            "--output",
            "experiment_results/holoq_object_20261027/protocol",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.holoq_bench.archive",
            "--runs",
            str(merged),
            "--output",
            "experiment_results/holoq_object_20261027/evaluation",
        ],
        check=True,
    )
    print("ALL_COMPLETE", merged, flush=True)


if __name__ == "__main__":
    main()
