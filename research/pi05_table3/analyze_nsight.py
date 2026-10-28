"""Actual Graph work/overlap and host wait; never add host wait to GPU time."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3


def union(intervals):
    total = 0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def main():
    data = Path(__file__).resolve().parents[2] / "results/pi05_table3_414_20261028"
    db = sqlite3.connect(data / "nsight_full414.sqlite")
    names = dict(db.execute("select id,value from StringIds"))
    kernels = list(db.execute("select start,end,shortName,streamId,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL"))
    copies = list(db.execute("select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"))
    ranges = list(db.execute("select start,end,text,textId from NVTX_EVENTS where end is not null"))
    named = defaultdict(list)
    for a, b, text, tid in ranges:
        named[text or names.get(tid, "")].append((a, b))
    complete = next(v for k, v in named.items() if "complete" in k)
    assert len(complete) == 5, {k: len(v) for k, v in named.items()}
    calls = len(complete)
    times = Counter()
    counts = Counter()
    for a, b, n, stream, node in kernels:
        times[names[n]] += b - a
        counts[names[n]] += 1
    target = {k: counts[k] for k in ["_vision_gemm", "_gemm_a4w4_arithmetic", "_quant_pack_a4"]}
    assert target == {"_vision_gemm": 2430, "_gemm_a4w4_arithmetic": 6930, "_quant_pack_a4": 6930}, target
    api = defaultdict(lambda: Counter())
    for a, b, n in db.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[names[n]]["calls"] += 1
        api[names[n]]["host_ms"] += (b - a) / 1e6
    busy = union([(a, b) for a, b, *_ in kernels] + [(a, b) for a, b, *_ in copies])
    result = {
        "profiled_calls": calls,
        "target_counts": target,
        "complete_policy_NVTX_mean_ms": sum(b - a for a, b in complete) / calls / 1e6,
        "GPU_kernel_work_ms_per_policy": sum(times.values()) / calls / 1e6,
        "GPU_busy_union_ms_per_policy": busy / calls / 1e6,
        "GPU_copy_work_ms_per_policy": sum(b - a for a, b, *_ in copies) / calls / 1e6,
        "GPU_streams": dict(Counter(k[3] for k in kernels)),
        "graph_kernel_nodes": sum(k[4] is not None and k[4] != 0 for k in kernels),
        "kernel_ms_per_policy": {n: t / calls / 1e6 for n, t in times.most_common()},
        "host_api": dict(api),
        "NVTX_ranges": {
            n: {"calls": len(v), "mean_host_ms": sum(b - a for a, b in v) / len(v) / 1e6} for n, v in named.items()
        },
        "note": "Host cudaMemcpyAsync duration includes blocking D2H completion while GPU runs; it is not device copy work. Do not sum that duration with GPU kernel work. NVTX/profiling latency is diagnostic, formal latency is in report.json.",
    }
    (data / "nsight_verified_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["kernel_ms_per_policy", "host_api", "NVTX_ranges"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
