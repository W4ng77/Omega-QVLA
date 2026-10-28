"""Cumulative GPU work and interval union of the full414 Graph trace."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from research.pi05_table3.measure import write


def interval_union(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if end is None or left > end:
            total += right - left
            end = right
        elif right > end:
            total += right - end
            end = right
    return total


def main():
    root = Path(__file__).resolve().parents[2] / "results/pi05_table3_414_20261028/full414"
    trace = json.loads((root / "policy_trace.json").read_text())
    kernels = [x for x in trace["traceEvents"] if x.get("cat") == "kernel"]
    groups = defaultdict(lambda: Counter())
    per_name = defaultdict(lambda: Counter())
    for e in kernels:
        name = e["name"]
        grid = e.get("args", {}).get("grid", [0])[0]
        if "_gemm_a4w4_arithmetic" in name:
            kind = "language_GEMM" if grid > 1 else "expert_GEMM_all10steps"
        elif "_vision_gemm" in name:
            kind = "vision_GEMM"
        elif "_quant_pack_a4" in name:
            kind = "language_A4_pack" if grid >= 1000 else "expert_A4_pack_all10steps"
        else:
            kind = "other_GPU_work"
        for target in [groups[kind], per_name[name]]:
            target["count"] += 1
            target["gpu_ms"] += e["dur"] / 1000
    summary = {
        "calls": 5,
        "cumulative_groups": {
            k: {"count_per_call": v["count"] / 5, "GPU_ms_per_call": v["gpu_ms"] / 5}
            for k, v in sorted(groups.items(), key=lambda x: -x[1]["gpu_ms"])
        },
        "top_kernels": {k: dict(v) for k, v in sorted(per_name.items(), key=lambda x: -x[1]["gpu_ms"])[:40]},
        "all_kernel_work_ms_per_call": sum(e["dur"] for e in kernels) / 5000,
        "kernel_busy_union_ms_per_call": interval_union([(e["ts"], e["ts"] + e["dur"]) for e in kernels]) / 5000,
        "streams": sorted({e.get("args", {}).get("stream") for e in kernels}),
        "phase_mapping_note": "Observed shapes: languageM968=>GEMMgrid.x31;expertM10=>grid.x1. A4 pack grid distinguishes these disjoint shape sets. Vision has a dedicated kernel. Other work is NOT automatically called rotation.",
        "interpretation": "Profiler cumulative work, not formal latency. Never add overlapped kernels or profiler CPU time to the unprofiled complete-policy latency.",
    }
    write(root / "cumulative_graph_profile.json", summary)


if __name__ == "__main__":
    main()
