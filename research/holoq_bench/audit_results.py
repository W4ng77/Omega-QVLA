"""Verify completed statistics/protocol/state isolation from saved records."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def packed_trace(base):
    """Check real packed launches, allowing multiple Marlin passes per GEMM.

    Unquantized vision/auxiliary layers legitimately launch floating kernels;
    their presence is not evidence that a replaced projection fell back.
    """
    meta = json.loads((base / "smoke/metadata.json").read_text())
    trace_path = base / "smoke/cuda_kernels.json"
    trace = json.loads(trace_path.read_text())
    checks = {}
    patterns = {
        "marlin": lambda name: name.startswith("void Marlin<"),
        "awq": lambda name: "gemm_w4a16_" in name or "gemv_forward_cuda" in name,
        "qserve": lambda name: name.startswith("void dense_kernel"),
        "native_s4": lambda name: name.startswith("void native_group_kernel<"),
        "cutlass_s4": lambda name: "integer_subbyteILi4ELb1EE" in name,
        "nunchaku_s4": lambda name: "gemm_w4a4_kernel" in name and "nunchaku" in name,
    }
    for backend, layers in meta["backend_layers"].items():
        assert backend in patterns, ("Unrecognized backend", backend)
        launches = sum(count for name, count in trace.items() if patterns[backend](name))
        # This cohort only has homogeneous all-target / LLM / DiT partitions.
        expected = {84: 84, 96: 768, 116: 340, 180: 852}[layers]
        assert launches >= expected, (backend, launches, expected)
        checks[backend] = {"layers": layers, "packed_cuda_launches": launches, "minimum": expected}
    return {"backends": checks, "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    calibration = json.loads((a.root / "calibration_fp16_init0_v2/manifest.json").read_text())
    heldout = json.loads((a.root / "heldout_init10_19/manifest.json").read_text())
    calib_hashes = {r["initial_state_sha256"] for r in calibration["trajectories"]}
    state_hashes = {(r["task_id"], r["init_index"]): r["sha256"] for r in heldout["states"]}
    assert not calib_hashes & set(state_hashes.values()), "Calibration/evaluation state overlap"
    output = {
        "calibration_evaluation_state_disjoint": True,
        "cases": {},
        "packed_kernel_traces": {},
        "errors": [],
    }
    queue = json.loads((a.runs / "queue.json").read_text())
    gpu = [json.loads(line) for line in (a.runs / "gpu_monitor.jsonl").read_text().splitlines()]
    for case in queue["valid_cases"]:
        for mode in ["eager", "head_graph"]:
            base = a.runs / case / mode
            if (base / "smoke/cuda_kernels.json").exists():
                try:
                    output["packed_kernel_traces"][case + "/" + mode] = packed_trace(base)
                except Exception as exc:
                    output["errors"].append({"case": case + "/" + mode + "/packed_trace", "error": repr(exc)})
            for stage in ["replay", "rollout"]:
                path = base / stage / "summary.json"
                if not path.exists():
                    continue
                d = json.loads(path.read_text())
                if not d.get("finished"):
                    continue
                key = case + "/" + mode + "/" + stage
                try:
                    assert (
                        d["denoising_steps"] == 8
                        and d["action_horizon"] == 16
                        and d["batch"] == 1
                        and d["warmup"] == 32
                    )
                    assert d["raw_image_shape_each_camera"] == [256, 256, 3] and d[
                        "transform_image_shape_each_camera"
                    ] == [224, 224, 3]
                    assert d["quantized_layers"] in (116, 180)
                    assert len(d["active_layer_calls_after_smoke"]) == d["quantized_layers"]
                    assert all(n > 0 for n in d["active_layer_calls_after_smoke"].values())
                    source = a.root / "deployments" / case / "deployment.json"
                    assert d["source_pack_sha256"] == json.loads(source.read_text())["source_pack_sha256"]
                    file = base / stage / ("calls.jsonl" if stage == "replay" else "rollout_calls.jsonl")
                    calls = [json.loads(line) for line in file.read_text().splitlines()]
                    if stage == "replay":
                        assert len(calls) == 300 and {r["repeat"] for r in calls} == {0, 1, 2}
                        for repeat in range(3):
                            assert {
                                (r["task_id"], r["init_index"]) for r in calls if r["repeat"] == repeat
                            } == set(state_hashes)
                    else:
                        episodes = [
                            json.loads(line) for line in (base / stage / "episodes.jsonl").read_text().splitlines()
                        ]
                        assert len(episodes) == 100
                        assert {(e["task_id"], e["init_index"]) for e in episodes} == set(state_hashes)
                        assert all(
                            e["initial_state_sha256"] == state_hashes[e["task_id"], e["init_index"]]
                            for e in episodes
                        )
                        assert d["successes"] == sum(e["success"] for e in episodes)
                        assert len(calls) == sum(e["control_steps"] for e in episodes)
                        for episode in episodes:
                            frames = [
                                r["control_step"]
                                for r in calls
                                if r["task_id"] == episode["task_id"] and r["init_index"] == episode["init_index"]
                            ]
                            assert sorted(frames) == list(range(episode["control_steps"]))
                        calls = [r for r in calls if not r["cold_capture"]]
                    values = [r["ms"] for r in calls]
                    expected = [
                        float(np.mean(values)),
                        float(np.percentile(values, 50)),
                        float(np.percentile(values, 95)),
                    ]
                    actual = [d["latency"][k] for k in ["mean_ms", "p50_ms", "p95_ms"]]
                    assert np.allclose(expected, actual, rtol=0, atol=1e-9)
                    windows = [(d["started"], d["finished"])]
                    reused_path = path.with_name("reused_smoke.json")
                    if stage == "rollout" and reused_path.exists():
                        reused = json.loads(reused_path.read_text())
                        assert reused["source_pack_sha256"] == d["source_pack_sha256"]
                        assert reused["episodes"] == 10
                        source_episodes = Path(reused["source"]) / "episodes.jsonl"
                        assert (
                            hashlib.sha256(source_episodes.read_bytes()).hexdigest() == reused["episodes_sha256"]
                        )
                        windows.append((reused["source_started"], reused["source_finished"]))
                    monitor = [r for r in gpu if any(start <= r["time"] <= end for start, end in windows)]
                    assert monitor, "No GPU monitor coverage"
                    gpu1 = [
                        line.split(",")
                        for sample in monitor
                        for line in sample["gpus"]
                        if line.split(",")[0].strip() == "1"
                    ]
                    max_util = max(float(r[2]) for r in gpu1)
                    max_mem = max(float(r[3]) for r in gpu1)
                    assert max_util == 0 and max_mem < 100, (max_util, max_mem)
                    output["cases"][key] = {
                        "verified": True,
                        "calls": len(values),
                        "gpu1_max_util": max_util,
                        "gpu1_max_memory_MB": max_mem,
                        "source_hash_manifest_sha256": hashlib.sha256(
                            json.dumps(d["source_files_sha256"], sort_keys=True).encode()
                        ).hexdigest(),
                    }
                except Exception as exc:
                    output["errors"].append({"case": key, "error": repr(exc)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + "\n")
    print("AUDIT", len(output["cases"]), "verified;", len(output["errors"]), "errors", flush=True)
    if output["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
