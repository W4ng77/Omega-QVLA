"""Keep all required selected 100-episode cases, skip redundant losing backends.

Finish the already-running original HoloQ control. Acquire the exclusive lock
at its stage boundary before restarting only the queue controller; no measured
episode is interrupted and no prior result or command is overwritten.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


KEEP = [
    "holoq",
    "awq_qwen_selective",
    "smoothquant_a8",
    "gptq_selective",
    "omniquant_selective",
    "smoothquant_a4",
    "awq_qwen_full",
    "gptq_full",
]


def arguments(pid):
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return []


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    args = parser.parse_args()
    runs = args.root / "evaluation_v3"
    parent_args = arguments(args.parent_pid)
    assert b"research.holoq_bench.submit" in parent_args, parent_args
    assert str(runs).encode() in parent_args, parent_args
    selection = json.loads((args.root / "evaluation_best_s4/backend_selection.json").read_text())
    assert all(v["selected"]["case"] != "holoq" for k, v in selection["families"].items() if k == "holoq")
    print("WAIT_FOR_CURRENT_ORIGINAL_100_STAGE_BOUNDARY", flush=True)
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    summary = json.loads((runs / "holoq/head_graph/rollout/summary.json").read_text())
    assert summary.get("finished") and summary["episodes"] == 100
    marker = runs / "holoq/head_graph/rollout/completed.json"
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists() and json.loads(marker.read_text())["returncode"] == 0
    # A subsequent worker may already exist but cannot load GPU weights while
    # this exclusive lock is held. Freeze the controller before enumerating it.
    os.kill(args.parent_pid, signal.SIGSTOP)
    children = []
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            parent = next(line.split()[1] for line in (directory / "status").read_text().splitlines() if line.startswith("PPid:"))
            if int(parent) == args.parent_pid:
                pid = int(directory.name)
                cmd = arguments(pid)
                if b"research.holoq_bench.run_policy" not in cmd:
                    continue  # e.g. the controller's read-only nvidia-smi sampler
                out = cmd[cmd.index(b"--output") + 1]
                assert out.startswith(str(runs).encode() + b"/")
                children.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    for pid in children:
        os.kill(pid, signal.SIGTERM)
    os.kill(args.parent_pid, signal.SIGTERM)
    os.kill(args.parent_pid, signal.SIGCONT)
    time.sleep(2)
    old = json.loads((runs / "queue.json").read_text())
    (runs / "queue_before_backend_focus.json").write_text(json.dumps(old, indent=2) + "\n")
    old["accuracy_focus"] = {
        "time": time.time(),
        "reason": "Backend selection frozen without scores; independently evaluate every selected backend, omit losing duplicate backends from additional100 runs.",
        "already_running_control_finished": "holoq/head_graph/rollout",
        "common_accuracy_cases": KEEP,
        "selected_accuracy_cases": [v["selected"]["case"] for v in selection["families"].values()],
        "preserved_prior_valid_cases": old["valid_cases"],
        "controller_pid": args.parent_pid,
        "interrupted_waiting_workers": children,
    }
    temp = runs / "queue.focus.tmp"
    temp.write_text(json.dumps(old, indent=2) + "\n")
    temp.replace(runs / "queue.json")
    cmd = [sys.executable, "-u", "-m", "research.holoq_bench.submit", "--root", str(args.root), "--output", str(runs), "--cases", *KEEP]
    print("FOCUSED_QUEUE", cmd, flush=True)
    fcntl.flock(guard, fcntl.LOCK_UN)
    with (args.root / "evaluation_focused.log").open("a") as log:
        ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print("FOCUSED_QUEUE_FINISHED", ret.returncode, flush=True)
    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    main()
