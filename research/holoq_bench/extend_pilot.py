"""Check fused rotations with the existing Marlin LLM as explicit candidates.

The native-S4 LLM pilot can be slower; do not assume an integer kernel wins.
Preserve the first selection and add this missing combination before accuracy.
"""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("results/holoq_object")
OUT = ROOT / "evaluation_best_s4"
ADDITIONS = ["holoq_fusedrot", "holoq_step_weight_fusedrot"]


def main():
    while not (OUT / "backend_selection.json").exists():
        time.sleep(5)
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    status = json.loads((OUT / "queue.json").read_text())
    shutil.copy2(OUT / "backend_selection.json", OUT / "backend_selection_initial9.json")
    status["pilot_finished_initial"] = status.pop("pilot_finished")
    status.update(phase="timing_pilot_extension")

    def save():
        p = OUT / "queue.tmp"
        p.write_text(json.dumps(status, indent=2) + "\n")
        p.replace(OUT / "queue.json")

    save()
    env = os.environ.copy()
    env["HOLOQ_PILOT_LOCK_FD"] = str(guard.fileno())
    subprocess.run(
        [sys.executable, "-m", "research.holoq_bench.export_fused_rotation", "--root", str(ROOT)], check=True
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.holoq_bench.deployment",
            "--packs",
            *[str(ROOT / "packs_v5" / (c + ".pt")) for c in ADDITIONS],
            "--state",
            str(ROOT / "teacher_capture/fp16_model_state.pt"),
            "--output",
            str(ROOT / "deployments"),
        ],
        check=True,
    )

    def job(case, mode, stage):
        directory = OUT / case / mode / stage
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "completed.json"
        if marker.exists():
            return json.loads(marker.read_text())["returncode"] == 0
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "research.holoq_bench.pilot_worker",
            "--deployment",
            str(ROOT / "deployments" / case),
            "--observations",
            str(ROOT / "heldout_init10_19/observations.pt"),
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
        print("EXTENSION_START", case, mode, stage, flush=True)
        with (directory / "run.log").open("a") as log:
            ret = subprocess.run(cmd, env=env, pass_fds=(guard.fileno(),), stdout=log, stderr=subprocess.STDOUT)
        record = {"returncode": ret.returncode, "finished": time.time()}
        marker.write_text(json.dumps(record, indent=2) + "\n")
        status["cases"].setdefault(case, {})[mode + "_" + stage] = record
        save()
        print("EXTENSION_FINISH", case, mode, stage, ret.returncode, flush=True)
        return ret.returncode == 0

    for case in ADDITIONS:
        status["candidate_cases"].append(case)
        eager = job(case, "eager", "smoke")
        graph = eager and job(case, "head_graph", "smoke")
        if not eager:
            continue
        status["valid_cases"].append(case)
        save()
        job(case, "eager", "replay")
        if graph:
            job(case, "head_graph", "replay")
    prior = json.loads((OUT / "backend_selection_initial9.json").read_text())
    for family, case in zip(["holoq", "holoq_step_weight"], ADDITIONS):
        choices = prior["families"][family]["candidates"]
        for mode in ["eager", "head_graph"]:
            p = OUT / case / mode / "replay/summary.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            choices.append(
                {
                    "case": case,
                    "mode": mode,
                    "root": str(OUT),
                    "mean_ms": d["latency"]["mean_ms"],
                    "source_pack_sha256": d["source_pack_sha256"],
                    "replay_summary": str(p),
                    "code_commit": d["code_commit"],
                }
            )
        prior["families"][family] = {"selected": min(choices, key=lambda v: v["mean_ms"]), "candidates": choices}
    prior["frozen_at"] = time.time()
    prior["extension_reason"] = (
        "Native-S4 LLM did not beat Marlin in first full-policy replay; test fused rotation with both backends before accuracy."
    )
    (OUT / "backend_selection.json").write_text(json.dumps(prior, indent=2) + "\n")
    status.update(pilot_finished=time.time(), phase="awaiting_primary_accuracy", pilot_extension_complete=True)
    save()
    (OUT / "gpu_monitor.jsonl").write_text((ROOT / "evaluation_v3/gpu_monitor.jsonl").read_text())
    fcntl.flock(guard, fcntl.LOCK_UN)
    print(
        "EXTENDED_SELECTION_FROZEN",
        json.dumps({k: v["selected"]["case"] for k, v in prior["families"].items()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
