"""Borrow an exclusive common-pilot batch lock without changing inference."""
import builtins
import os
from research.holoq_bench import bench_common_logits


def borrow(path, mode="r", *args, **kwargs):
    if str(path) == "/tmp/holoq-vla-policy-timing.lock":
        return os.fdopen(os.dup(int(os.environ["HOLOQ_PILOT_LOCK_FD"])), mode)
    return builtins.open(path, mode, *args, **kwargs)


if __name__ == "__main__":
    bench_common_logits.open = borrow
    bench_common_logits.main()
