"""Verified full-policy backend pilots and explicit latency-only selection.

Reserve the same host-wide lock across the batch. Children borrow its open-file
identity, preserving shared/exclusive mode transitions without deadlocking.
No quantized codes, scales, calibration or inference computation are changed.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CANDIDATES = [
    "holoq_llm_s4",
    "holoq_step_weight_llm_s4",
    "smoothquant_a4_cutlass",
    "holoq_llm_s4_fusedrot",
    "holoq_step_weight_llm_s4_fusedrot",
    "duquant_a8_g128_fusedrot",
    "duquant_a4_g128_fusedrot",
    "quantvla_selective_a8_atm_g128_fusedrot",
    "quantvla_selective_a4_atm_g128_fusedrot",
]
FAMILIES = {
    "holoq": ["holoq", "holoq_llm_s4", "holoq_llm_s4_fusedrot"],
    "holoq_step_weight": ["holoq_step_weight", "holoq_step_weight_llm_s4", "holoq_step_weight_llm_s4_fusedrot"],
    "smoothquant_a4": ["smoothquant_a4", "smoothquant_a4_cutlass"],
    "duquant_a8": ["duquant_a8_g128", "duquant_a8_g128_fusedrot"],
    "duquant_a4": ["duquant_a4_g128", "duquant_a4_g128_fusedrot"],
    "quantvla_a8": ["quantvla_selective_a8_atm_g128", "quantvla_selective_a8_atm_g128_fusedrot"],
    "quantvla_a4": ["quantvla_selective_a4_atm_g128", "quantvla_selective_a4_atm_g128_fusedrot"],
}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    a = p.parse_args()
    out = a.root / "evaluation_best_s4"
    out.mkdir(parents=True, exist_ok=True)
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    print("WAITING_FOR_EXCLUSIVE_PILOT_BATCH", flush=True)
    fcntl.flock(guard, fcntl.LOCK_EX)
    print("EXCLUSIVE_PILOT_BATCH_ACQUIRED", flush=True)
    env = os.environ.copy()
    env["HOLOQ_PILOT_LOCK_FD"] = str(guard.fileno())
    status = {
        "started": time.time(),
        "common_mode": "head_graph",
        "cases": {},
        "commands": [],
        "candidate_cases": CANDIDATES,
        "valid_cases": [],
        "phase": "timing_pilot",
    }

    def save():
        temp = out / "queue.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(out / "queue.json")

    # Prepare immutable metadata-only exports/deployment files while no policy is
    # being timed. All original artifacts stay in their original directories.
    subprocess.run(
        [sys.executable, "-m", "research.holoq_bench.export_fused_rotation", "--root", str(a.root)], check=True
    )
    packs = [str(a.root / "packs_v5" / (case + ".pt")) for case in CANDIDATES if case.endswith("fusedrot")]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.holoq_bench.deployment",
            "--packs",
            *packs,
            "--state",
            str(a.root / "teacher_capture/fp16_model_state.pt"),
            "--output",
            str(a.root / "deployments"),
        ],
        check=True,
    )

    def job(case, mode, stage):
        directory = out / case / mode / stage
        directory.mkdir(parents=True, exist_ok=True)
        done = directory / "completed.json"
        if done.exists():
            return json.loads(done.read_text())["returncode"] == 0
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "research.holoq_bench.pilot_worker",
            "--deployment",
            str(a.root / "deployments" / case),
            "--observations",
            str(a.root / "heldout_init10_19/observations.pt"),
            "--output",
            str(directory),
            "--mode",
            mode,
            "--stage",
            stage,
        ]
        status["commands"].append(
            {"cmd": cmd, "started": time.time(), "cwd": str(Path.cwd()), "borrowed_timing_lock": True}
        )
        save()
        print("PILOT_START", case, mode, stage, flush=True)
        with (directory / "run.log").open("a") as file:
            result = subprocess.run(
                cmd, env=env, pass_fds=(guard.fileno(),), stdout=file, stderr=subprocess.STDOUT
            )
        record = {"returncode": result.returncode, "finished": time.time()}
        done.write_text(json.dumps(record, indent=2) + "\n")
        status["cases"].setdefault(case, {})[mode + "_" + stage] = record
        save()
        print("PILOT_FINISH", case, mode, stage, result.returncode, flush=True)
        return result.returncode == 0

    for case in CANDIDATES:
        eager = job(case, "eager", "smoke")
        graph = eager and job(case, "head_graph", "smoke")
        if not eager:
            continue
        status["valid_cases"].append(case)
        save()
        job(case, "eager", "replay")
        if graph:
            job(case, "head_graph", "replay")
    selected = {}
    for family, cases in FAMILIES.items():
        choices = []
        for case in cases:
            roots = [a.root / "evaluation_v3", out]
            for root in roots:
                for mode in ("eager", "head_graph"):
                    path = root / case / mode / "replay/summary.json"
                    if path.exists():
                        d = json.loads(path.read_text())
                        choices.append(
                            {
                                "case": case,
                                "mode": mode,
                                "root": str(root),
                                "mean_ms": d["latency"]["mean_ms"],
                                "source_pack_sha256": d["source_pack_sha256"],
                                "replay_summary": str(path),
                                "code_commit": d["code_commit"],
                            }
                        )
        if choices:
            winner = min(choices, key=lambda d: d["mean_ms"])
            selected[family] = {"selected": winner, "candidates": choices}
    selection = {
        "rule": "minimum full-policy fixed-input replay mean among verified execution variants; no task score is read",
        "frozen_at": time.time(),
        "families": selected,
    }
    (out / "backend_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    status.update(pilot_finished=time.time(), phase="awaiting_primary_accuracy")
    save()
    # The primary monitor runs through this reserved batch. Save its coverage;
    # the final controller appends monitoring after the primary queue finishes.
    (out / "gpu_monitor.jsonl").write_text((a.root / "evaluation_v3/gpu_monitor.jsonl").read_text())
    fcntl.flock(guard, fcntl.LOCK_UN)
    print(
        "PILOT_SELECTION_FROZEN", json.dumps({k: v["selected"]["case"] for k, v in selected.items()}), flush=True
    )


if __name__ == "__main__":
    main()
