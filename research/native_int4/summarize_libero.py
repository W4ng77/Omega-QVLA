"""Verify complete trials and aggregate measured full-policy latency."""

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values):
    v = np.asarray(values, dtype=float)
    return {
        "calls": len(v),
        "mean_ms": float(v.mean()),
        "p50_ms": float(np.percentile(v, 50)),
        "p95_ms": float(np.percentile(v, 95)),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    root = args.run
    merged = json.loads((root / "merged_summary.json").read_text())
    episodes = [
        json.loads(line)
        for f in (root / "summaries").glob("*_episodes.jsonl")
        for line in f.read_text().splitlines()
        if line
    ]
    samples = [
        json.loads(line)
        for f in (root / "policy_latency").glob("*.jsonl")
        for line in f.read_text().splitlines()
        if line
    ]
    assert merged["total_episodes"] == len(episodes) == 100
    assert len(merged["task_summaries"]) == 10
    assert all(row["episodes"] == 10 for row in merged["task_summaries"])
    assert all(row["error"] is None for row in episodes), "Evaluation exceptions found"
    assert sum(ep["policy_calls"] for ep in episodes) == len(samples)
    for f in (root / "logs").glob("server_shard_*.log"):
        assert "[native_s4] active_layers=96" in f.read_text()
    report = {
        "successes": merged["total_successes"],
        "episodes": 100,
        "success_rate": merged["total_success_rate"],
        "server_policy": stats([s["policy_ms"] for s in samples]),
        "client_policy_rpc": stats([v for ep in episodes for v in ep["policy_rpc_ms"]]),
        "peak_allocated_gib": max(s["peak_allocated_bytes"] for s in samples) / 2**30,
        "peak_reserved_gib": max(s["peak_reserved_bytes"] for s in samples) / 2**30,
        "rollout_total_s": sum(ep["rollout_s"] for ep in episodes),
        "tasks": [],
    }
    for task in merged["task_summaries"]:
        ss = [s["policy_ms"] for s in samples if task["task_description"] in s["task"]]
        if not ss:
            raise ValueError(f"No timing samples for {task['task_id']}")
        report["tasks"].append(dict(task, server_policy=stats(ss)))
    (root / "timing_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Native S4 LIBERO Object: step_weight + block rotations",
        "",
        f"Successes: {report['successes']}/100",
        "",
        "| Task | Successes | Policy mean ms | Policy p50 ms | Policy p95 ms | Rollout mean s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in report["tasks"]:
        s = task["server_policy"]
        lines.append(
            f"| {task['task_description']} | {task['successes']}/10 | {s['mean_ms']:.2f} | "
            f"{s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {task['rollout_mean_s']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Timing includes full get_action, all 8 denoising steps, rotations, preprocessing and postprocessing.",
            "Server timing excludes IPC; client RPC timing and memory are in timing_summary.json.",
            "The converted step_weight format and block-rotation reduction order differ from the original dense-pack method.",
            "No claim of exact original-method equivalence is made.",
        ]
    )
    (root / "timing_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
