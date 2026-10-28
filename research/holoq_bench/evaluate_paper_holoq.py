"""Independent paper-input-only HoloQ validation, latency selection and Object100.

Never borrow task outcomes from earlier two-sided reconstruction controls.
"""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def main():
    root = Path("results/holoq_object")
    out = root / "evaluation_paper_holoq"
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "queue.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        "started": time.time(), "common_mode": "head_graph", "commands": [],
        "cases": {}, "valid_cases": [], "candidate_cases": [], "selected_modes": {},
        "phase": "waiting_for_paper_input_only_build"}

    def save():
        temp = out / "queue.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)

    save()
    while not (root / "paper_cfg_reencoding/finished.json").exists():
        time.sleep(10)
    assert json.loads((root / "paper_cfg_reencoding/finished.json").read_text())["built"]
    shutil.copy2(root / "evaluation_v3/environment.txt", out / "environment.txt")
    stop = threading.Event()

    def monitor():
        with (out / "gpu_monitor.jsonl").open("a", buffering=1) as file:
            while not stop.is_set():
                result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                file.write(json.dumps({"time": time.time(), "gpus": result.stdout.strip().splitlines()}) + "\n")
                stop.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def command(cmd, directory, owns_lock=False):
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "completed.json"
        if marker.exists():
            assert json.loads(marker.read_text())["returncode"] == 0, directory
            return
        status["commands"].append({"cmd": cmd, "started": time.time(), "cwd": str(Path.cwd())})
        save()
        print("PAPER_EVAL_START", directory, flush=True)
        guard = None
        if owns_lock:
            guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
            fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            with (directory / "run.log").open("a") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        finally:
            if guard:
                guard.close()
        marker.write_text(json.dumps({"returncode": result.returncode, "finished": time.time()}) + "\n")
        assert result.returncode == 0, directory
        print("PAPER_EVAL_FINISH", directory, flush=True)

    def policy(case, mode, stage, trials=10):
        directory = out / case / mode / stage
        command([sys.executable, "-u", "-m", "research.holoq_bench.run_policy", "--deployment", str(root / "deployments" / case), "--observations", str(root / "heldout_init10_19/observations.pt"), "--output", str(directory), "--mode", mode, "--stage", "rollout" if stage == "rollout_smoke" else stage, "--trials", str(trials)], directory)
        status["cases"].setdefault(case, {})[mode + "_" + stage] = json.loads((directory / "completed.json").read_text())
        save()

    try:
        diagnostic = out / "canonical_math"
        command([sys.executable, "-u", "-m", "research.holoq_bench.check_canonical_math", "--deployment", str(root / "deployments/holoq_paper_g128"), "--cache", str(root / "paper_cfg_build/layer_cache"), "--output", str(diagnostic / "result.json")], diagnostic, owns_lock=True)
        assert not json.loads((diagnostic / "result.json").read_text())["review_required"]
        selection = {"families": {}, "frozen_by": "minimum full-policy replay latency after same-grid correctness; never task outcomes", "paper_configuration": str(root / "paper_cfg_build/configuration.json")}
        for family, bases in {
            "holoq": ["holoq_paper_g128", "holoq_paper_g128_llm_s4"],
            "holoq_step_weight": ["holoq_paper_g128_step_weight", "holoq_paper_g128_step_weight_llm_s4"],
        }.items():
            choices = []
            for case in bases:
                check = out / "functional_checks" / case
                oracle = check / "result.json"
                command([sys.executable, "-u", "-m", "research.holoq_bench.check_rotation_policy", "--deployment", str(root / "deployments" / case), "--observations", str(root / "heldout_init10_19/observations.pt"), "--output", str(oracle)], check)
                validation = json.loads(oracle.read_text())
                assert validation["passed"] and validation["comparisons"] == 8520
                shutil.copy2(oracle, root / "deployments" / case / "functional_validation.json")
                if case != "holoq_paper_g128":
                    math = out / "canonical_math_candidates" / case
                    command([sys.executable, "-u", "-m", "research.holoq_bench.check_canonical_math", "--deployment", str(root / "deployments" / case), "--cache", str(root / "paper_cfg_build/layer_cache"), "--output", str(math / "result.json")], math, owns_lock=True)
                    assert not json.loads((math / "result.json").read_text())["review_required"]
                if case not in status["candidate_cases"]:
                    status["candidate_cases"].append(case)
                if case not in status["valid_cases"]:
                    status["valid_cases"].append(case)
                save()
                for mode in ("eager", "head_graph"):
                    policy(case, mode, "smoke")
                    smoke = out / case / mode / "smoke"
                    assert "EXPLICIT_STRIDED_BLOCK64_V2" in (smoke / "run.log").read_text()
                    assert json.loads((smoke / "metadata.json").read_text())["source_files_sha256"]["research/holoq_bench/fused_rotation_strided.py"] == validation["plugin_source_sha256"]
                    policy(case, mode, "replay")
                    path = out / case / mode / "replay/summary.json"
                    data = json.loads(path.read_text())
                    choices.append({"case": case, "mode": mode, "root": str(out), "mean_ms": data["latency"]["mean_ms"], "replay_summary": str(path), "source_pack_sha256": data["source_pack_sha256"], "code_commit": data["code_commit"]})
            winner = min(choices, key=lambda row: row["mean_ms"])
            selection["families"][family] = {"selected": winner, "candidates": choices, "frozen_at": time.time()}
            (out / "backend_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
            print("PAPER_LATENCY_WINNER", family, winner, flush=True)
            case = winner["case"]
            status["selected_modes"][case] = winner["mode"]
            save()
            for mode in dict.fromkeys(["head_graph", winner["mode"]]):
                policy(case, mode, "rollout_smoke", 1)
                # Give the audited smoke-reuse worker one complete poll window.
                time.sleep(6)
                policy(case, mode, "rollout", 10)
            status["completed_families"] = list(selection["families"])
            save()
        status.update(phase="complete", finished=time.time(), backend_selection=selection)
        save()
    finally:
        stop.set()
        thread.join(timeout=10)
    print("PAPER_HOLOQ_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
