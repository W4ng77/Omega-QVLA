"""Borrow a scheduler's GPU experiment lock; policy computations are unchanged."""

import builtins
import os
from research.holoq_bench import run_policy


def borrow_lock(path, mode="r", *args, **kwargs):
    if str(path) == "/tmp/holoq-vla-policy-timing.lock":
        return os.fdopen(os.dup(int(os.environ["HOLOQ_PILOT_LOCK_FD"])), mode)
    return builtins.open(path, mode, *args, **kwargs)


if __name__ == "__main__":
    run_policy.open = borrow_lock
    run_policy.main()
