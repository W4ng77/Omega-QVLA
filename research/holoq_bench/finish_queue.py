"""Finish explicitly selected best backends after the common accuracy cohort."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    a = p.parse_args()
    primary = a.root / "evaluation_v3"
    extra = a.root / "evaluation_best_s4"
    while True:
        q = json.loads((primary / "queue.json").read_text())
        pilot_path = extra / "queue.json"
        pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else {}
        if pilot.get("pilot_finished") and pilot.get("pilot_extension_complete"):
            break
        time.sleep(15)
    selection = json.loads((extra / "backend_selection.json").read_text())
    order = ["holoq", "holoq_step_weight", "quantvla_a8", "quantvla_a4", "duquant_a8", "duquant_a4", "smoothquant_a4"]
    winners = [selection["families"][family]["selected"] for family in order]
    new = [d for d in winners if Path(d["root"]).resolve() == extra.resolve()]
    unique = {d["case"]: d for d in new}
    pilot.update(
        valid_cases=list(unique),
        selected_modes={c: d["mode"] for c, d in unique.items()},
        phase="selected_accuracy",
    )

    def save():
        tmp = extra / "queue.tmp"
        tmp.write_text(json.dumps(pilot, indent=2) + "\n")
        tmp.replace(extra / "queue.json")

    save()
    # The primary sampler covered all timing pilots. This cohort may interleave
    # whole stages with primary accuracy, still owning exclusive policy locks.
    (extra / "gpu_monitor.jsonl").write_text((primary / "gpu_monitor.jsonl").read_text())
    stop = threading.Event()

    def monitor():
        with (extra / "gpu_monitor.jsonl").open("a", buffering=1) as file:
            while not stop.is_set():
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                file.write(json.dumps({"time": time.time(), "gpus": query.stdout.strip().splitlines()}) + "\n")
                stop.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def job(case, mode, stage, trials):
        out = extra / case / mode / stage
        out.mkdir(parents=True, exist_ok=True)
        marker = out / "completed.json"
        if marker.exists():
            return json.loads(marker.read_text())["returncode"] == 0
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "research.holoq_bench.run_policy",
            "--deployment",
            str(a.root / "deployments" / case),
            "--observations",
            str(a.root / "heldout_init10_19/observations.pt"),
            "--output",
            str(out),
            "--mode",
            mode,
            "--stage",
            "rollout",
            "--trials",
            str(trials),
        ]
        pilot["commands"].append({"cmd": cmd, "started": time.time(), "cwd": str(Path.cwd())})
        save()
        print("SELECTED_ACCURACY_START", case, mode, stage, flush=True)
        with (out / "run.log").open("a") as file:
            ret = subprocess.run(cmd, stdout=file, stderr=subprocess.STDOUT)
        record = {"returncode": ret.returncode, "finished": time.time()}
        marker.write_text(json.dumps(record, indent=2) + "\n")
        pilot["cases"].setdefault(case, {})[mode + "_" + stage] = record
        save()
        print("SELECTED_ACCURACY_FINISH", case, mode, stage, ret.returncode, flush=True)
        return ret.returncode == 0

    try:
        for case, data in unique.items():
            if job(case, data["mode"], "rollout_smoke", 1):
                job(case, data["mode"], "rollout", 10)
        pilot.update(finished=time.time(), phase="complete")
        save()
    finally:
        stop.set()
        thread.join(timeout=10)
    print("SELECTED_ACCURACY_COMPLETE_WAIT_PRIMARY_FOR_MERGE", flush=True)
    while not json.loads((primary / "queue.json").read_text()).get("finished"):
        time.sleep(15)
    # Allow the audit/reuse process to finish merging the peaks from its two
    # explicitly documented measurement phases before snapshotting the results.
    time.sleep(6)
    merged = a.root / "evaluation_all"
    merged.mkdir(parents=True, exist_ok=True)
    records = [json.loads((root / "queue.json").read_text()) for root in (primary, extra)]
    for root, data in zip((primary, extra), records):
        for case in data["valid_cases"]:
            shutil.copytree(root / case, merged / case, dirs_exist_ok=True)
    combined = {
        "common_mode": records[0]["common_mode"],
        "input_queues": [str(primary), str(extra)],
        "queue_modes": [d["common_mode"] for d in records],
        "cases": {case: values for data in records for case, values in data["cases"].items()},
        "valid_cases": [case for data in records for case in data["valid_cases"]],
        "commands": [cmd for data in records for cmd in data["commands"]],
        "backend_selection": selection,
        "finished": time.time(),
    }
    (merged / "queue.json").write_text(json.dumps(combined, indent=2) + "\n")
    samples = {}
    for root in (primary, extra):
        for line in (root / "gpu_monitor.jsonl").read_text().splitlines():
            row = json.loads(line)
            samples[row["time"]] = row
    (merged / "gpu_monitor.jsonl").write_text("".join(json.dumps(samples[k]) + "\n" for k in sorted(samples)))
    shutil.copy2(primary / "environment.txt", merged / "environment.txt")
    shutil.copy2(extra / "backend_selection.json", merged / "backend_selection.json")
    # Keep nonselected timing candidates separately; no task score is invented.
    candidates = merged / "timing_candidates"
    for case in records[0].get("accuracy_focus", {}).get("preserved_prior_valid_cases", []):
        if case not in records[0]["valid_cases"]:
            shutil.copytree(primary / case, candidates / ("common_" + case), dirs_exist_ok=True)
    if (primary / "queue_before_backend_focus.json").exists():
        shutil.copy2(primary / "queue_before_backend_focus.json", merged / "queue_before_backend_focus.json")
    for case in pilot.get("candidate_cases", []):
        shutil.copytree(extra / case, candidates / case, dirs_exist_ok=True)
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
            "research.holoq_bench.audit_results",
            "--runs",
            str(merged),
            "--output",
            str(merged / "report/audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.holoq_bench.final_report",
            "--runs",
            str(merged),
            "--output",
            "experiment_results/holoq_object_20261027/RESULTS.md",
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
