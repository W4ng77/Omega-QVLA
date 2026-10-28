"""Contiguous final replay block for all selected deployments and baselines.

Repeat only shared-input performance, keeping prior timings immutable. This
controls for wall-clock drift across the many-hour task-evaluation queue.
The already frozen backend choices are not changed using these measurements.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from research.holoq_bench.main_selection import read_selection


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    args = parser.parse_args()
    cases, modes, _ = read_selection(args.root)
    out = args.root / "evaluation_final_replay_v2"
    out.mkdir(parents=True, exist_ok=True)
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    status = {"started": time.time(), "cases": cases, "modes": modes,
              "purpose": "contemporaneous final replay block; frozen methods unchanged", "commands": []}
    env = os.environ.copy()
    env["HOLOQ_PILOT_LOCK_FD"] = str(guard.fileno())
    stop = threading.Event()

    def save():
        (out / "confirmation.json").write_text(json.dumps(status, indent=2) + "\n")

    def monitor():
        with (out / "gpu_monitor.jsonl").open("a", buffering=1) as file:
            while not stop.is_set():
                result = subprocess.run(["nvidia-smi",
                    "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
                file.write(json.dumps({"time": time.time(), "gpus": result.stdout.strip().splitlines()}) + "\n")
                stop.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for case in cases:
            for mode in modes[case]:
                directory = out / case / mode / "replay"
                directory.mkdir(parents=True, exist_ok=True)
                marker = directory / "completed.json"
                if marker.exists():
                    assert json.loads(marker.read_text())["returncode"] == 0
                    continue
                cmd = [sys.executable, "-u", "-m", "research.holoq_bench.pilot_worker",
                       "--deployment", str(args.root / "deployments" / case), "--observations",
                       str(args.root / "heldout_init10_19/observations.pt"), "--output", str(directory),
                       "--mode", mode, "--stage", "replay"]
                status["commands"].append({"cmd": cmd, "started": time.time(), "cwd": str(Path.cwd()),
                                           "borrowed_timing_lock": True})
                save()
                print("FINAL_REPLAY_START", case, mode, flush=True)
                with (directory / "run.log").open("a") as log:
                    result = subprocess.run(cmd, env=env, pass_fds=(guard.fileno(),), stdout=log, stderr=subprocess.STDOUT)
                marker.write_text(json.dumps({"returncode": result.returncode, "finished": time.time()}) + "\n")
                assert result.returncode == 0, (case, mode)
                print("FINAL_REPLAY_FINISH", case, mode, flush=True)
        status["finished"] = time.time()
        save()
    finally:
        stop.set()
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
