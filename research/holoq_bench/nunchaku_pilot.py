"""Check the paper's mature backend and revise timing-only selection explicitly.

Own one exclusive timing lock across compile, arithmetic checks, export and
complete-policy pilots. A failed candidate does not block the baseline cohort.
"""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from research.holoq_bench.export_nunchaku import CANDIDATES


def main():
    root = Path("results/holoq_object")
    out = root / "evaluation_best_s4"
    status = json.loads((out / "queue.json").read_text())
    if "nunchaku_pilot_finished" in status:
        status.setdefault("nunchaku_prior_attempts", []).append(
            {key: status.pop(key) for key in ["nunchaku_pilot_finished", "nunchaku_pilot_failure", "nunchaku_pilot_success"] if key in status}
        )
    status.update(nunchaku_pilot_requested=time.time(), pilot_extension_complete=False)

    def save():
        p = out / "queue.nunchaku.tmp"
        p.write_text(json.dumps(status, indent=2) + "\n")
        p.replace(out / "queue.json")

    save()
    print("NUNCHAKU_WAIT_EXCLUSIVE_STAGE_BOUNDARY", flush=True)
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    status["phase"] = "timing_pilot_nunchaku"
    env = os.environ.copy()
    env["HOLOQ_PILOT_LOCK_FD"] = str(guard.fileno())
    save()

    def command(cmd, log):
        print("NUNCHAKU_COMMAND", cmd, flush=True)
        status["commands"].append({"cmd": cmd, "cwd": str(Path.cwd()), "started": time.time(), "borrowed_timing_lock": True})
        save()
        with log.open("a") as file:
            ret = subprocess.run(cmd, env=env, pass_fds=(guard.fileno(),), stdout=file, stderr=subprocess.STDOUT)
        return ret.returncode

    try:
        preparatory = [
            ([sys.executable, "-u", "-m", "research.holoq_bench.nunchaku_probe"], root / "nunchaku_probe.log"),
            ([sys.executable, "-m", "pytest", "research/holoq_bench/test_nunchaku.py", "-q"], root / "nunchaku_tests.log"),
            ([sys.executable, "-u", "-m", "research.holoq_bench.export_nunchaku"], root / "nunchaku_export.log"),
            ([sys.executable, "-u", "-m", "research.holoq_bench.deployment", "--packs", *[str(root / "packs_v6" / (c + ".pt")) for c in CANDIDATES], "--state", str(root / "teacher_capture/fp16_model_state.pt"), "--output", str(root / "deployments")], root / "nunchaku_deployments.log"),
        ]
        for cmd, log in preparatory:
            code = command(cmd, log)
            if code:
                raise RuntimeError({"command": cmd, "returncode": code, "log": str(log)})
        for case in CANDIDATES:
            (out / case).mkdir(exist_ok=True)
            if case not in status["candidate_cases"]:
                status["candidate_cases"].append(case)
        save()

        def job(case, mode, stage):
            directory = out / case / mode / stage
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / "completed.json"
            if marker.exists():
                return json.loads(marker.read_text())["returncode"] == 0
            cmd = [sys.executable, "-u", "-m", "research.holoq_bench.pilot_worker", "--deployment", str(root / "deployments" / case), "--observations", str(root / "heldout_init10_19/observations.pt"), "--output", str(directory), "--mode", mode, "--stage", stage]
            code = command(cmd, directory / "run.log")
            record = {"returncode": code, "finished": time.time()}
            marker.write_text(json.dumps(record, indent=2) + "\n")
            status["cases"].setdefault(case, {})[mode + "_" + stage] = record
            save()
            print("NUNCHAKU_STAGE_FINISHED", case, mode, stage, code, flush=True)
            return code == 0

        for case in CANDIDATES:
            eager = job(case, "eager", "smoke")
            graph = eager and job(case, "head_graph", "smoke")
            if eager:
                if case not in status["valid_cases"]:
                    status["valid_cases"].append(case)
                save()
                job(case, "eager", "replay")
                if graph:
                    job(case, "head_graph", "replay")
        selection = json.loads((out / "backend_selection.json").read_text())
        backup = out / "backend_selection_before_nunchaku.json"
        if not backup.exists():
            shutil.copy2(out / "backend_selection.json", backup)
        for family, case in zip(["holoq", "holoq_step_weight", "smoothquant_a4"], CANDIDATES):
            choices = selection["families"][family]["candidates"]
            for mode in ["eager", "head_graph"]:
                path = out / case / mode / "replay/summary.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                assert data.get("finished") and data["latency"]["count"] == 300
                choices.append({"case": case, "mode": mode, "root": str(out), "mean_ms": data["latency"]["mean_ms"], "source_pack_sha256": data["source_pack_sha256"], "replay_summary": str(path), "code_commit": data["code_commit"]})
            selection["families"][family] = {"selected": min(choices, key=lambda d: d["mean_ms"]), "candidates": choices}
        selection.update(frozen_at=time.time(), nunchaku_revision_reason="Appendix Table8 uses Nunchaku. Explicitly check compatible token/G128-scale paths; original DiT column scales cannot be replaced. Selection still reads no task scores.")
        tmp = out / "backend_selection.nunchaku.tmp"
        tmp.write_text(json.dumps(selection, indent=2) + "\n")
        tmp.replace(out / "backend_selection.json")
        print("NUNCHAKU_SELECTION_FROZEN", {k: v["selected"]["case"] for k, v in selection["families"].items()}, flush=True)
        status["nunchaku_pilot_success"] = True
    except Exception as exc:
        status["nunchaku_pilot_failure"] = repr(exc)
        print("NUNCHAKU_PILOT_FAILED_CONTINUE_BASELINES", repr(exc), flush=True)
    finally:
        status.update(nunchaku_pilot_finished=time.time(), pilot_extension_complete=True, phase="awaiting_primary_accuracy")
        save()
        fcntl.flock(guard, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
