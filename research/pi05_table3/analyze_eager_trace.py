"""Attribute raw GPU activities through launch correlation and CPU scopes.

Rejects duplicated/propagated FunctionEvent device totals; validates actual
kernel counts and keeps overlap-aware interval union separate from work sums.
"""

from bisect import bisect_right
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def union(intervals):
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
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--definition8', action='store_true')
    args=parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "results/pi05_table3_414_20261028" / ("definition8_profile" if args.definition8 else "scope_controls/both")
    events = json.loads((root / "eager_policy_trace.json").read_text())["traceEvents"]
    kernels = [x for x in events if x.get("cat") == "kernel"]
    activities = [x for x in events if x.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]]
    counts = {
        name: sum(name in x["name"] for x in kernels)
        for name in ["_vision_gemm", "_gemm_a4w4_arithmetic", "_quant_pack_a4"]
    }
    expected = {"_vision_gemm": 2430, "_gemm_a4w4_arithmetic": 6930, "_quant_pack_a4": 6930}
    if args.definition8:
        counts={name:sum(name in x['name'] for x in kernels) for name in ['_vision_gemm','_channel_a4w4','integer_subbyteILi4ELb1E']}
        expected={'_vision_gemm':2430,'_channel_a4w4':5040,'integer_subbyteILi4ELb1E':630}
    assert counts == expected, {
        "observed": counts,
        "expected": expected,
        "reason": "Do not use a trace with extra/missing target executions",
    }
    scopes = defaultdict(list)
    ops = defaultdict(list)
    launches = {}
    complete = []
    for e in events:
        cat = e.get("cat")
        name = e.get("name", "")
        tid = e.get("tid")
        if cat == "user_annotation" and name.startswith(("layer/", "stage/", "complete_policy/")):
            scopes[tid].append(e)
            if name.startswith("complete_policy/"):
                complete.append(e)
        elif cat == "cpu_op":
            ops[tid].append(e)
        if cat in ["cuda_runtime", "cuda_driver"] and "correlation" in e.get("args", {}):
            cid = e["args"]["correlation"]
            if cid not in launches or e.get("dur", 1e99) < launches[cid].get("dur", 1e99):
                launches[cid] = e
    for store in [scopes, ops]:
        for tid, values in store.items():
            values.sort(key=lambda x: x["ts"])
    scope_starts = {t: [x["ts"] for x in v] for t, v in scopes.items()}
    op_starts = {t: [x["ts"] for x in v] for t, v in ops.items()}
    # Prefix maximum end provides a quick proof that no earlier interval can
    # contain a launch. Native extension launches often have no aten owner;
    # scanning all previous closed ops for each such launch is quadratic.
    def prefix_ends(store):
        result={}
        for tid,values in store.items():
            bound=float('-inf')
            ends=[]
            for event in values:
                bound=max(bound,event['ts']+event.get('dur',0))
                ends.append(bound)
            result[tid]=ends
        return result
    scope_ends=prefix_ends(scopes)
    op_ends=prefix_ends(ops)

    def containing(store, starts, ends, tid, ts):
        values = store.get(tid, [])
        index = bisect_right(starts.get(tid, []), ts) - 1
        while index >= 0:
            if ends[tid][index]<ts:
                break
            e = values[index]
            if e["ts"] <= ts <= e["ts"] + e.get("dur", 0):
                return e["name"]
            index -= 1
        return ""

    groups = defaultdict(lambda: Counter())
    missing = Counter()
    for e in activities:
        launch = launches.get(e.get("args", {}).get("correlation"))
        label = owner = ""
        if launch:
            tid, ts = launch.get("tid"), launch["ts"]
            label = containing(scopes, scope_starts, scope_ends, tid, ts)
            owner = containing(ops, op_starts, op_ends, tid, ts)
        else:
            missing[e.get("cat")] += 1
        phase = label.split("/")[1] if label.startswith(("layer/", "stage/")) else "other"
        name = e["name"]
        if any(t in name for t in ['_vision_gemm','_gemm_a4w4_arithmetic','_channel_a4w4','integer_subbyteILi4ELb1E']):
            category = "GEMM_with_W_decode_scales"
        elif "_quant_pack_a4" in name:
            category = "A4_quantize_pack"
        elif owner == "aten::bmm" and label.startswith(("layer/language/", "layer/expert/")):
            category = "rotation_input_or_output"
        elif owner == "aten::index_select" and label.startswith(("layer/language/", "layer/expert/")):
            category = "input_permutation"
        elif e.get("cat") in ["gpu_memcpy", "gpu_memset"] or any(
            t in owner for t in ["copy", "clone", "contiguous"]
        ):
            category = "data_copy_dtype_contiguous"
        elif label.startswith("layer/vision/"):
            category = "vision_A4_quantize_prepare"
        elif args.definition8 and label.startswith(('layer/language/','layer/expert/')):
            category = "A4_quantize_pack_scale_epilogue_or_bias"
        else:
            category = "other_compute"
        record = groups[phase + "/" + category]
        record["calls"] += 1
        record["gpu_us"] += e.get("dur", 0)
    result = {
        "source": "raw Chrome GPU activities + launch correlation IDs + enclosing CPU scopes",
        "profiled_calls": len(complete),
        "validated_target_counts": counts,
        "missing_launch_correlations": dict(missing),
        "groups": {
            k: {"calls_per_policy": v["calls"] / 5, "gpu_ms_per_policy": v["gpu_us"] / 5000}
            for k, v in sorted(groups.items(), key=lambda x: -x[1]["gpu_us"])
        },
        "GPU_work_ms_per_policy": sum(x.get("dur", 0) for x in activities) / 5000,
        "GPU_busy_union_ms_per_policy": union([(x["ts"], x["ts"] + x.get("dur", 0)) for x in activities]) / 5000,
        "profiled_CPU_policy_ms_mean": sum(e["dur"] for e in complete) / len(complete) / 1000,
        "host_sync_call_ms_per_policy": sum(
            e.get("dur", 0)
            for e in events
            if e.get("cat") == "cuda_runtime" and "Synchronize" in e.get("name", "")
        )
        / 5000,
        "note": "Profiler wall includes instrumentation overhead. Work sums may overlap. Host sync includes waiting for GPU work and must not be added to GPU time. CPU/GPU totals from the rejected FunctionEvent association are not used.",
    }
    assert len(complete) == 5
    assert (
        abs(sum(v["gpu_ms_per_policy"] for v in result["groups"].values()) - result["GPU_work_ms_per_policy"])
        < 1e-5
    )
    (root / "verified_eager_attribution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
