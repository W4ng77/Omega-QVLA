"""Opt-in full get_action wall timing after warmup, excluding RPC/serialization."""

import json
import os
from pathlib import Path
import time

import torch


def install_timing(policy, port):
    root = Path(os.environ["GR00T_LATENCY_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / f"server_{port}.jsonl").open("w", buffering=1)
    original = policy.get_action
    seen = set()
    call = 0
    torch.cuda.reset_peak_memory_stats()

    def timed(obs):
        nonlocal call
        task = str(obs.get("annotation.human.action.task_description", "unknown"))
        if task not in seen:
            seen.add(task)
            torch.save(obs, root / f"observation_{port}_{len(seen):02d}.pt")
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(obs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        call += 1
        stream.write(
            json.dumps(
                {
                    "call": call,
                    "task": task,
                    "policy_ms": elapsed_ms,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            + "\n"
        )
        return result

    policy.get_action = timed
