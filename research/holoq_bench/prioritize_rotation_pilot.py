"""Temporarily reorder pending jobs without interrupting an active GPU worker.

Suspend only the original submit controller while its current rollout finishes.
Resume once the repaired HoloQ pilot has frozen a valid choice, or on timeout,
controller exit, signal, or any failure. No episode/timer process is suspended.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--primary-pid", type=int, required=True)
    parser.add_argument("--recovery-pid", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    root = Path("results/holoq_object")
    proc = Path("/proc") / str(args.primary_pid)
    command = (proc / "cmdline").read_bytes().split(b"\0")
    assert b"research.holoq_bench.submit" in command
    identity = (proc / "stat").read_text().split()[21]
    recovery = Path("/proc") / str(args.recovery_pid)
    assert b"research.holoq_bench.rotation_recovery" in (recovery / "cmdline").read_bytes().split(b"\0")
    note = {"primary_controller_pid": args.primary_pid, "primary_starttime": identity,
            "recovery_controller_pid": args.recovery_pid, "started": time.time(),
            "reason": "Current rollout continues; prioritize corrected HoloQ pilot before remaining pending baseline jobs"}
    record = root / "pilot_priority.json"

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        os.kill(args.primary_pid, signal.SIGSTOP)
        record.write_text(json.dumps(note, indent=2) + "\n")
        print("PENDING_PRIMARY_JOBS_HELD_CURRENT_ROLLOUT_CONTINUES", flush=True)
        while time.time() - note["started"] < args.timeout_seconds:
            selection = root / "evaluation_rotation_v2/backend_selection.json"
            if selection.exists() and "holoq" in json.loads(selection.read_text()).get("families", {}):
                note["resume_reason"] = "Corrected HoloQ full-policy timing pilot completed and valid choice frozen"
                break
            if not recovery.exists():
                note["resume_reason"] = "Recovery controller exited"
                break
            time.sleep(5)
        else:
            note["resume_reason"] = "Bounded priority window expired"
    finally:
        if proc.exists() and (proc / "stat").read_text().split()[21] == identity:
            os.kill(args.primary_pid, signal.SIGCONT)
        note["resumed"] = time.time()
        record.write_text(json.dumps(note, indent=2) + "\n")
        print("PRIMARY_PENDING_QUEUE_RESUMED", note.get("resume_reason", "finally"), flush=True)


if __name__ == "__main__":
    main()
