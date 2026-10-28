"""Reconcile whole-policy CUDA activity, CPU scopes and idle intervals.

Never add synthetic gpu_user_annotation spans to real CUDA activities. Sum of
kernel durations is work, not wall time; unions/overlap are reported explicitly.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def length(intervals):
    return sum(b - a for a, b in merge(intervals))


def intersection(a, b):
    aa, bb = merge(a), merge(b)
    i = j = 0
    total = 0
    while i < len(aa) and j < len(bb):
        total += max(0, min(aa[i][1], bb[j][1]) - max(aa[i][0], bb[j][0]))
        if aa[i][1] < bb[j][1]:
            i += 1
        else:
            j += 1
    return total


def sqlite_events(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    events = []
    for row in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL AND start IS NOT NULL"):
        name = row["text"] or strings.get(row["textId"], "")
        if name.startswith("holoq/"):
            events.append(dict(name=name, cat="user_annotation", ts=row["start"] / 1000,
                               dur=(row["end"] - row["start"]) / 1000, tid=row["globalTid"], args={}))
    for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        events.append(dict(name=strings[row["nameId"]], cat="cuda_runtime", ts=row["start"] / 1000,
                           dur=(row["end"] - row["start"]) / 1000, tid=row["globalTid"],
                           args={"correlation": row["correlationId"]}))
    for table, cat in [("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                       ("CUPTI_ACTIVITY_KIND_MEMCPY", "gpu_memcpy"),
                       ("CUPTI_ACTIVITY_KIND_MEMSET", "gpu_memset")]:
        if table not in tables:
            continue
        for row in db.execute("SELECT * FROM " + table):
            name = strings[row["demangledName"]] if cat == "kernel" else cat
            args = {"correlation": row["correlationId"], "stream": row["streamId"], "device": row["deviceId"]}
            if cat == "kernel":
                args["grid"] = [row["gridX"], row["gridY"], row["gridZ"]]
                args["block"] = [row["blockX"], row["blockY"], row["blockZ"]]
                args["graph_node"] = row["graphNodeId"]
            if cat == "gpu_memcpy":
                args.update(bytes=row["bytes"], copy_kind=row["copyKind"])
            events.append(dict(name=name, cat=cat, ts=row["start"] / 1000,
                               dur=(row["end"] - row["start"]) / 1000, tid=row["streamId"], args=args))
    for table in tables:
        if "GRAPH" not in table or table == "CUDA_GRAPH_EVENTS":
            continue
        columns = {row[1] for row in db.execute("PRAGMA table_info(" + table + ")")}
        if not {"start", "end", "deviceId", "streamId", "correlationId"} <= columns:
            continue
        for row in db.execute("SELECT * FROM " + table):
            events.append(dict(name="CUDA_GRAPH_BODY", cat="gpu_graph", ts=row["start"] / 1000,
                               dur=(row["end"] - row["start"]) / 1000, tid=row["streamId"],
                               args={"correlation": row["correlationId"], "stream": row["streamId"]}))
    return events


def operation(event):
    name = event["name"].lower()
    if event["cat"] in ("gpu_memcpy", "gpu_memset"):
        return event["cat"]
    if event["cat"] == "gpu_graph":
        return "graph_body_inclusive"
    if name.startswith("_rotate"):
        return "rotation"
    if name.startswith(("_prepare", "_pack_input", "_pack")):
        return "quantization_pack"
    if name.startswith("_rescale"):
        return "gemm_scale_epilogue"
    if "marlin" in name or "gemm" in name or "gemv" in name or name.startswith("_zn7cutlass6kernel"):
        return "gemm"
    if "direct_copy" in name or "copy_device_to_device" in name:
        return "copy_cast_kernel"
    return "other_compute"


def write_csv(path, rows):
    if not rows:
        # A corrected re-analysis must not leave a stale, previously derived
        # table (notably false idle gaps from incomplete coarse Graph traces).
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(events, output):
    calls = sorted([e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("holoq/get_action/")], key=lambda e: e["ts"])
    assert len(calls) == 5, ("Expected five complete get_action ranges", len(calls))
    n = len(calls)
    scopes = [e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("holoq/stage/")]
    steps = [e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("holoq/step/")]
    runtime = {e["args"]["correlation"]: e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_graph")]
    phase_op = defaultdict(lambda: [0.0, 0])
    kernels = defaultdict(lambda: [0.0, 0])
    step_stats = defaultdict(lambda: [0.0, 0])
    missing = Counter()
    for event in gpu:
        source = runtime.get(event["args"].get("correlation"))
        parents = [s for s in scopes if source and s["tid"] == source["tid"] and s["ts"] <= source["ts"] < s["ts"] + s["dur"]]
        phase = min(parents, key=lambda s: s["dur"])["name"].split("/")[-1] if parents else "other_policy"
        op = operation(event)
        if not parents:
            missing[event["name"]] += event["dur"]
        phase_op[phase, op][0] += event["dur"]
        phase_op[phase, op][1] += 1
        key = (phase, op, event["name"], json.dumps(event["args"].get("grid", [])))
        kernels[key][0] += event["dur"]
        kernels[key][1] += 1
        for step in steps:
            if source and step["tid"] == source["tid"] and step["ts"] <= source["ts"] < step["ts"] + step["dur"]:
                key = (step["name"], op)
                step_stats[key][0] += event["dur"]
                step_stats[key][1] += 1
    rows = [{"phase": phase, "operation": op, "gpu_total_ms": v[0] / 1000,
             "gpu_ms_per_call": v[0] / (1000 * n), "calls_total": v[1], "calls_per_policy": v[1] / n}
            for (phase, op), v in phase_op.items()]
    rows.sort(key=lambda row: row["gpu_total_ms"], reverse=True)
    kernel_rows = [{"phase": phase, "operation": op, "kernel": name, "grid": grid,
                    "gpu_total_ms": v[0] / 1000, "gpu_ms_per_call": v[0] / (1000 * n),
                    "count_total": v[1], "count_per_call": v[1] / n, "mean_kernel_us": v[0] / v[1]}
                   for (phase, op, name, grid), v in kernels.items()]
    kernel_rows.sort(key=lambda row: row["gpu_total_ms"], reverse=True)
    sync = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and "Synchronize" in e["name"]]
    host_api = [e for e in events if any(c["ts"] <= e["ts"] < c["ts"] + c["dur"] for c in calls) and
                e.get("cat") in ("cuda_runtime", "cuda_driver") and
                any(term in e["name"] for term in ("Synchronize", "Memcpy", "GraphLaunch"))]
    api_rows = []
    all_busy = merge([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    for name in sorted({e["name"] for e in host_api}):
        selected = [e for e in host_api if e["name"] == name]
        intervals = [(e["ts"], e["ts"] + e["dur"]) for e in selected]
        overlap = intersection(intervals, all_busy)
        api_rows.append({"api": name, "count_per_call": len(selected) / n,
                         "cpu_api_elapsed_ms_per_call": length(intervals) / (1000 * n),
                         "overlap_gpu_busy_ms_per_call": overlap / (1000 * n),
                         "during_observed_gpu_idle_ms_per_call": (length(intervals) - overlap) / (1000 * n),
                         "note": "host API elapsed, includes implicit synchronization for pageable DtoH; not additive to GPU compute"})
    per_call = []
    gaps = []
    for call in calls:
        a, b = call["ts"], call["ts"] + call["dur"]
        intervals = [(max(a, e["ts"]), min(b, e["ts"] + e["dur"])) for e in gpu if e["ts"] < b and e["ts"] + e["dur"] > a]
        busy = merge(intervals)
        sync_intervals = [(max(a, e["ts"]), min(b, e["ts"] + e["dur"])) for e in sync if e["ts"] < b and e["ts"] + e["dur"] > a]
        sync_gpu_overlap = intersection(sync_intervals, busy)
        per_call.append({"call": call["name"], "profiled_wall_ms": (b - a) / 1000,
                         "gpu_activity_sum_ms": sum(y - x for x, y in intervals) / 1000,
                         "gpu_busy_union_ms": length(busy) / 1000,
                         "gpu_overlap_excess_ms": (sum(y - x for x, y in intervals) - length(busy)) / 1000,
                         "gpu_idle_within_policy_ms": ((b - a) - length(busy)) / 1000,
                         "cpu_sync_blocked_ms": length(sync_intervals) / 1000,
                         "cpu_sync_overlapping_gpu_busy_ms": sync_gpu_overlap / 1000,
                         "cpu_sync_during_gpu_idle_ms": (length(sync_intervals) - sync_gpu_overlap) / 1000,
                         "sync_calls": sum(a <= e["ts"] < b for e in sync)})
        cursor = a
        for start, end in busy + [[b, b]]:
            if start - cursor >= 10:
                overlapping = [s["name"] for s in scopes if s["ts"] < start and s["ts"] + s["dur"] > cursor]
                gaps.append({"call": call["name"], "start_us": cursor, "end_us": start,
                             "idle_ms": (start - cursor) / 1000, "overlapping_cpu_phases": ";".join(overlapping)})
            cursor = max(cursor, end)
    cpu_ops = defaultdict(lambda: [0.0, 0])
    for e in events:
        if e.get("cat") == "cpu_op" and any(x in e["name"] for x in ("aten::to", "aten::_to_copy", "aten::copy_", "aten::contiguous", "aten::clone", "aten::item", "aten::_local_scalar_dense")):
            cpu_ops[e["name"]][0] += e["dur"]
            cpu_ops[e["name"]][1] += 1
    prep = [{"operator": k, "cpu_inclusive_ms_per_call": v[0] / (1000 * n), "count_per_call": v[1] / n,
             "note": "nested inclusive CPU times; not additive; may include GPU wait"} for k, v in cpu_ops.items()]
    summary = {"profiled_calls": n, "phase_operation": rows,
               "phase_gpu_work_ms_per_call": {p: sum(r["gpu_ms_per_call"] for r in rows if r["phase"] == p) for p, _ in phase_op},
               "means": {k: sum(row[k] for row in per_call) / n for k in per_call[0] if k != "call"},
               "per_call": per_call, "sync_api_count": dict(Counter(e["name"] for e in sync)),
               "host_api_overlap": api_rows,
               "data_preparation_cpu": prep, "unattributed_gpu_ms_per_call": sum(missing.values()) / (1000 * n),
               "unattributed_kernels": dict(missing),
               "step_operation": [{"step": step, "operation": op, "gpu_ms_per_call": v[0] / (1000 * n), "count_per_call": v[1] / n} for (step, op), v in step_stats.items()],
               "notes": ["Only actual CUDA activity is summed; gpu_user_annotation excluded.",
                         "CPU synchronization blocking overlaps GPU compute; never add it to GPU busy.",
                         "GPU idle contains CPU preprocessing/launch gaps and profiler overhead, not necessarily synchronization.",
                         "Kernel sums are work; GPU unions and excess overlap reconcile wall time.",
                         "Copy/cast is a cross-cutting phase/operation cell. CPU contiguous/to/copy ranges are nested and not additive."]}
    graph_launches = [e for e in events if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e["name"]]
    head_work = sum(row["gpu_ms_per_call"] for row in rows if row["phase"] == "action_head" and
                    row["operation"] in ("gemm", "graph_body_inclusive"))
    summary["gpu_coverage_complete"] = not graph_launches or head_work > 1
    if not summary["gpu_coverage_complete"]:
        summary["notes"].append("INCOMPLETE GRAPH GPU COVERAGE: graph launch APIs present but graph body activities absent. Observed GPU work is a lower bound. Idle cannot be inferred; use the complete node-level capture.")
        summary["means"]["gpu_idle_within_policy_ms"] = None
        summary["means"]["cpu_sync_during_gpu_idle_ms"] = None
        for row in per_call:
            row["gpu_idle_within_policy_ms"] = None
            row["cpu_sync_during_gpu_idle_ms"] = None
        for row in api_rows:
            row["during_observed_gpu_idle_ms_per_call"] = None
            row["note"] += "; incomplete graph coverage: GPU overlap is a lower bound"
        gaps = []
    output.mkdir(parents=True, exist_ok=True)
    (output / "breakdown.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output / "phase_operation.csv", rows)
    write_csv(output / "kernels_by_total_time.csv", kernel_rows)
    write_csv(output / "per_call_overlap.csv", per_call)
    write_csv(output / "data_preparation_cpu.csv", prep)
    write_csv(output / "host_api_overlap.csv", api_rows)
    write_csv(output / "largest_gpu_idle_gaps.csv", sorted(gaps, key=lambda r: r["idle_ms"], reverse=True)[:100])
    print("PROFILE_RECONCILIATION", json.dumps(summary["means"]), flush=True)
    print("PHASE_GPU_WORK", summary["phase_gpu_work_ms_per_call"], flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--torch-trace", type=Path)
    group.add_argument("--nsys-sqlite", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    events = json.loads(args.torch_trace.read_text())["traceEvents"] if args.torch_trace else sqlite_events(args.nsys_sqlite)
    analyze(events, args.output)


if __name__ == "__main__":
    main()
