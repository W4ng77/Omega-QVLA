"""Borrow a scheduler's GPU experiment lock; policy computations are unchanged."""

import builtins
import fcntl
import os
from types import SimpleNamespace
from research.holoq_bench import run_policy


def borrow_lock(path, mode="r", *args, **kwargs):
    if str(path) == "/tmp/holoq-vla-policy-timing.lock":
        return os.fdopen(os.dup(int(os.environ["HOLOQ_PILOT_LOCK_FD"])), mode)
    return builtins.open(path, mode, *args, **kwargs)


if __name__ == "__main__":
    run_policy.open = borrow_lock
    # Keep the batch reservation exclusive even for correctness-only children.
    # Replacing this module reference leaves stdlib/Torch file locks intact.
    run_policy.fcntl = SimpleNamespace(
        LOCK_SH=fcntl.LOCK_SH,
        LOCK_EX=fcntl.LOCK_EX,
        flock=lambda fd, mode: fcntl.flock(fd, fcntl.LOCK_EX),
    )
    run_policy.main()
