"""Accuracy-first completion of released original-grid and separate conversion."""
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
    out = root / "evaluation_released_grid"
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "queue.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        "started": time.time(), "common_mode": "head_graph", "commands": [], "cases": {},
        "valid_cases": [], "phase": "accuracy_first_release_grid", "common_optimizations": ["qwen3_last_token_logits"]}

    def save():
        temp = out / "queue.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)

    save()
    while not (root / "released_grid_build/finished.json").exists() or not (root / "released_grid_validation/canonical_math.json").exists():
        time.sleep(10)
    assert json.loads((root / "released_grid_build/export_integrity_audit.json").read_text())["passed"]
    assert not json.loads((root / "released_grid_validation/canonical_math.json").read_text())["review_required"]
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

    def command(cmd, directory, own_lock=False):
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "completed.json"
        if marker.exists():
            assert json.loads(marker.read_text())["returncode"] == 0, directory
            return
        status["commands"].append({"cmd": cmd, "started": time.time(), "cwd": str(Path.cwd())})
        save()
        print("RELEASE_EVAL_START", directory, flush=True)
        guard = None
        if own_lock:
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
        print("RELEASE_EVAL_FINISH", directory, flush=True)

    try:
        for case in ("holoq_release_grid", "holoq_release_step_weight"):
            if case.endswith("step_weight"):
                diagnostic = out / "conversion_math"
                command([sys.executable, "-u", "-m", "research.holoq_bench.check_canonical_math", "--deployment", str(root / "deployments" / case), "--cache", str(root / "released_grid_build/layer_cache"), "--output", str(diagnostic / "result.json")], diagnostic, own_lock=True)
                assert not json.loads((diagnostic / "result.json").read_text())["review_required"]
            status["valid_cases"].append(case) if case not in status["valid_cases"] else None
            save()
            # Accuracy first; no latency-based backend search on this cohort.
            for stage, trials in (("smoke", 10), ("rollout_smoke", 1), ("rollout", 10), ("replay", 10)):
                directory = out / case / "head_graph" / stage
                command([sys.executable, "-u", "-m", "research.holoq_bench.run_policy", "--deployment", str(root / "deployments" / case), "--observations", str(root / "heldout_init10_19/observations.pt"), "--output", str(directory), "--mode", "head_graph", "--stage", "rollout" if stage == "rollout_smoke" else stage, "--trials", str(trials), "--common-logits"], directory)
                status["cases"].setdefault(case, {})["head_graph_" + stage] = json.loads((directory / "completed.json").read_text())
                save()
                if stage == "rollout_smoke":
                    time.sleep(6)
        status.update(finished=time.time(), phase="complete")
        save()
    finally:
        stop.set()
        thread.join(timeout=10)
    print("RELEASE_GRID_ACCURACY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
