"""Hold pending queue jobs while profiling; leave the active policy untouched."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--finished", type=Path, required=True)
    a = p.parse_args()
    proc = Path("/proc") / str(a.pid)
    assert b"research.holoq_bench.rotation_recovery" in (proc / "cmdline").read_bytes().split(b"\0")
    identity = (proc / "stat").read_text().split()[21]
    a.finished.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        os.kill(a.pid, signal.SIGSTOP)
        print("PENDING_QUEUE_HELD_ACTIVE_CHILD_UNCHANGED", a.pid, flush=True)
        while not a.finished.exists() and time.time() - started < 3600:
            time.sleep(5)
    finally:
        if proc.exists() and (proc / "stat").read_text().split()[21] == identity:
            os.kill(a.pid, signal.SIGCONT)
        a.finished.with_name("queue_hold.json").write_text(json.dumps({
            "pid": a.pid, "started": started, "resumed": time.time(),
            "completed_marker": a.finished.exists(), "active_child_interrupted": False}, indent=2) + "\n")
        print("PENDING_QUEUE_RESUMED", flush=True)


if __name__ == "__main__":
    main()
