"""Merge only functionally eligible v2 winners with the unchanged baseline cohort."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    args = parser.parse_args()
    primary = args.root / "evaluation_v3"
    extra = args.root / "evaluation_rotation_v2"
    while True:
        records = [json.loads((r / "queue.json").read_text()) if (r / "queue.json").exists() else {}
                   for r in (primary, extra)]
        if all(r.get("finished") for r in records):
            break
        time.sleep(15)
    # Let the smoke reuse worker merge the inference peaks before snapshotting.
    time.sleep(6)
    selection = json.loads((extra / "backend_selection.json").read_text())
    merged = args.root / "evaluation_verified_v2"
    merged.mkdir(parents=True, exist_ok=True)
    valid = list(records[0]["valid_cases"])
    for family in selection["families"].values():
        case = family["selected"]["case"]
        assert "fusedrot" not in case, "Invalid legacy plugin cannot be selected"
        if case not in valid:
            valid.append(case)
    for case in valid:
        source = extra if (extra / case).exists() else primary
        shutil.copytree(source / case, merged / case, dirs_exist_ok=True)
    combined = {
        "common_mode": "head_graph", "valid_cases": valid,
        "cases": {c: values for r in records for c, values in r["cases"].items() if c in valid},
        "commands": [cmd for r in records for cmd in r["commands"]],
        "input_queues": [str(primary), str(extra)], "backend_selection": selection,
        "functional_regression_excluded": json.loads((args.root / "fused_rotation_regression.json").read_text()),
        "finished": time.time(),
    }
    (merged / "queue.json").write_text(json.dumps(combined, indent=2) + "\n")
    (merged / "backend_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    samples = {}
    for source in (primary, args.root / "evaluation_best_s4", extra):
        for line in (source / "gpu_monitor.jsonl").read_text().splitlines():
            row = json.loads(line)
            samples[row["time"]] = row
    (merged / "gpu_monitor.jsonl").write_text("".join(json.dumps(samples[k]) + "\n" for k in sorted(samples)))
    shutil.copy2(primary / "environment.txt", merged / "environment.txt")
    for case in records[1]["candidate_cases"]:
        if case not in valid:
            shutil.copytree(extra / case, merged / "timing_candidates" / case, dirs_exist_ok=True)
    # Nonfused losing controls are useful for bottleneck interpretation. Their
    # timing-only status and lack of 100 scores stay explicit.
    for source in (primary, args.root / "evaluation_best_s4"):
        for case in ["holoq_step_weight", "holoq_llm_s4", "holoq_step_weight_llm_s4",
                     "smoothquant_a4_cutlass", "smoothquant_a4_nunchaku", "duquant_a8_g128",
                     "duquant_a4_g128", "quantvla_selective_a8_atm_g128", "quantvla_selective_a4_atm_g128"]:
            if case not in valid and (source / case).exists():
                shutil.copytree(source / case, merged / "timing_candidates" / case, dirs_exist_ok=True)
    shutil.copytree(extra / "functional_checks", merged / "functional_checks", dirs_exist_ok=True)
    shutil.copytree(extra / "strided_tests", merged / "strided_tests", dirs_exist_ok=True)
    publish = Path("experiment_results/holoq_object_20261027")
    commands = [
        ["summarize", "--runs", str(merged), "--deployments", str(args.root / "deployments"), "--output", str(merged / "report")],
        ["audit_results", "--runs", str(merged), "--output", str(merged / "report/audit.json")],
        ["final_report", "--runs", str(merged), "--output", str(publish / "RESULTS.md")],
        ["provenance", "--root", str(args.root), "--output", str(publish / "protocol")],
        ["archive", "--runs", str(merged), "--output", str(publish / "evaluation")],
    ]
    for module, *options in commands:
        subprocess.run([sys.executable, "-m", "research.holoq_bench." + module, *options], check=True)
    print("VERIFIED_V2_ALL_COMPLETE", merged, flush=True)


if __name__ == "__main__":
    main()
