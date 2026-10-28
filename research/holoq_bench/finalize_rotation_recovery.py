"""Merge only functionally eligible v2 winners with the unchanged baseline cohort."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from research.holoq_bench.main_selection import COHORTS, read_selection


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    args = parser.parse_args()
    primary = args.root / "evaluation_v3"
    extra = args.root / "evaluation_rotation_v2"
    paper = args.root / "evaluation_paper_holoq"
    cohorts = [args.root / name for name in COHORTS]
    while True:
        records = [json.loads((r / "queue.json").read_text()) if (r / "queue.json").exists() else {}
                   for r in cohorts]
        common_path = args.root / "common_logits_confirmed/queue.json"
        common = json.loads(common_path.read_text()) if common_path.exists() else {}
        if all(r.get("finished") for r in records) and common.get("finished"):
            break
        time.sleep(15)
    # Let the smoke reuse worker merge the inference peaks before snapshotting.
    time.sleep(6)
    subprocess.run([sys.executable, "-u", "-m", "research.holoq_bench.confirm_replay",
                    "--root", str(args.root)], check=True)
    confirmation_root = args.root / "evaluation_final_replay_v2"
    confirmation = json.loads((confirmation_root / "confirmation.json").read_text())
    valid, _, selection = read_selection(args.root)
    merged = args.root / "evaluation_verified_v2"
    merged.mkdir(parents=True, exist_ok=True)
    for case in valid:
        assert "fusedrot" not in case, "Invalid legacy plugin cannot be selected"
        source = next(source for source in (paper, extra, primary) if (source / case).exists())
        shutil.copytree(source / case, merged / case, dirs_exist_ok=True)
        for mode in confirmation["modes"][case]:
            destination = merged / case / mode / "replay"
            if destination.exists():
                shutil.copytree(destination, destination.with_name("replay_prior"), dirs_exist_ok=True)
                shutil.rmtree(destination)
            shutil.copytree(confirmation_root / case / mode / "replay", destination)
    combined = {
        "common_mode": "head_graph", "valid_cases": valid,
        "cases": {c: values for r in records for c, values in r["cases"].items() if c in valid},
        "commands": [cmd for r in records for cmd in r["commands"]],
        "input_queues": [str(source) for source in cohorts], "backend_selection": selection,
        "two_sided_holoq_controls_excluded": "Extra output rotation, legacy permutation and grid differ from paper Eqs1–8; raw evaluations retained separately",
        "functional_regression_excluded": json.loads((args.root / "fused_rotation_regression.json").read_text()),
        "final_replay_confirmation": confirmation,
        "finished": time.time(),
    }
    (merged / "queue.json").write_text(json.dumps(combined, indent=2) + "\n")
    (merged / "backend_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    samples = {}
    for source in (primary, args.root / "evaluation_best_s4", extra, paper, confirmation_root):
        for line in (source / "gpu_monitor.jsonl").read_text().splitlines():
            row = json.loads(line)
            samples[row["time"]] = row
    (merged / "gpu_monitor.jsonl").write_text("".join(json.dumps(samples[k]) + "\n" for k in sorted(samples)))
    shutil.copy2(primary / "environment.txt", merged / "environment.txt")
    for source, record in zip(cohorts[1:], records[1:]):
        for case in record["candidate_cases"]:
            if case not in valid:
                folder = "two_sided_controls" if case.startswith("holoq") and not case.startswith("holoq_paper") else "timing_candidates"
                shutil.copytree(source / case, merged / folder / case, dirs_exist_ok=True)
    shutil.copytree(primary / "holoq", merged / "two_sided_controls/holoq", dirs_exist_ok=True)
    # Nonfused losing controls are useful for bottleneck interpretation. Their
    # timing-only status and lack of 100 scores stay explicit.
    for source in (primary, args.root / "evaluation_best_s4"):
        for case in ["holoq_step_weight", "holoq_llm_s4", "holoq_step_weight_llm_s4",
                     "smoothquant_a4_cutlass", "smoothquant_a4_nunchaku", "duquant_a8_g128",
                     "duquant_a4_g128", "quantvla_selective_a8_atm_g128", "quantvla_selective_a4_atm_g128"]:
            if case not in valid and (source / case).exists():
                shutil.copytree(source / case, merged / "timing_candidates" / case, dirs_exist_ok=True)
    shutil.copytree(extra / "functional_checks", merged / "functional_checks", dirs_exist_ok=True)
    shutil.copytree(paper / "functional_checks", merged / "paper_functional_checks", dirs_exist_ok=True)
    shutil.copytree(paper / "canonical_math", merged / "paper_canonical_math", dirs_exist_ok=True)
    shutil.copytree(paper / "canonical_math_candidates", merged / "paper_canonical_math_candidates", dirs_exist_ok=True)
    shutil.copy2(args.root / "paper_cfg_build/configuration.json", merged / "paper_configuration.json")
    shutil.copytree(extra / "strided_tests", merged / "strided_tests", dirs_exist_ok=True)
    shutil.copytree(args.root / "common_logits_confirmed", merged / "common_logits", dirs_exist_ok=True)
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
