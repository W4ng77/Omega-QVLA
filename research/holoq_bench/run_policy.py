"""Audited complete-policy replay/rollout with independent LIBERO states.

Use one process/GPU, never parallelize timed runs. All timings synchronize
before/after the entire get_action, including preprocessing/transfers/output.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from research.holoq_bench.collect_calibration import seed


def write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def stats(values):
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def memory():
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["eager", "head_graph"], required=True)
    p.add_argument("--stage", choices=["smoke", "replay", "rollout"], required=True)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--warmup", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.deployment = args.deployment.resolve()
    os.environ.update(HOLOQ_DEPLOY=str(args.deployment), HOLOQ_COMPUTE_DTYPE="float16")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed(42)
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy
    from research.holoq_bench.packed_runtime import PackedLinear

    cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=str(args.deployment),
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )
    pack = torch.load(args.deployment / "packed.pt", map_location="cpu", mmap=True, weights_only=False)
    layers = {n: m for n, m in policy.model.named_modules() if isinstance(m, PackedLinear)}
    assert set(layers) == set(pack["layers"])
    source_root = Path(__file__).resolve().parents[2]
    source_files = [
        *source_root.glob("research/holoq_bench/*.py"),
        *source_root.glob("research/native_int4/*"),
        source_root / "gr00t/model/policy.py",
        source_root / "gr00t/model/action_head/flow_matching_action_head.py",
        source_root / "gr00t/quantization/perstep_backends.py",
    ]
    source_hashes = {
        str(f.relative_to(source_root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source_files
        if f.is_file() and f.suffix in (".py", ".cpp", ".cu")
    }
    report = {
        "source_files_sha256": source_hashes,
        "method": pack["meta"]["method"],
        "mode": args.mode,
        "stage": args.stage,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": subprocess.list2cmdline(__import__("sys").argv),
        "source_pack_sha256": json.loads((args.deployment / "deployment.json").read_text())["source_pack_sha256"],
        "backend_layers": dict(Counter(m.backend for m in layers.values())),
        "checkpoint": pack["meta"]["checkpoint"],
        "quantized_layers": len(layers),
        "load_memory": memory(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "denoising_steps": 8,
        "action_horizon": 16,
        "batch": 1,
        "raw_image_shape_each_camera": [256, 256, 3],
        "transform_image_shape_each_camera": [224, 224, 3],
        "warmup": args.warmup,
        "timing_boundary": "synchronize; perf_counter; complete Gr00tPolicy.get_action; synchronize; perf_counter",
        "includes": [
            "CPU image/text/state preprocessing",
            "H2D",
            "vision/language backbone",
            "all 8 denoising steps",
            "rotation",
            "activation quantize/pack",
            "packed GEMM",
            "reductions",
            "D2H",
            "action unnormalization",
        ],
        "excludes": [
            "simulator",
            "raw LIBERO observation conversion",
            "RPC",
            "per-call RNG reset",
            "cold graph capture",
        ],
        "noise_seed": "42 + task_id*10000 + initial_state_index*1000 + control_step",
        "init_indices": list(range(10, 10 + args.trials)),
        "started": time.time(),
    }
    samples = torch.load(args.observations, map_location="cpu", weights_only=False)
    selected = [samples[i * 10] for i in range(10)]
    refs = []
    for i, sample in enumerate(selected):
        seed(1000 + i)
        refs.append(policy.get_action(sample["obs"]))
    eager_self_checks = []
    for i, sample in enumerate(selected):
        seed(1000 + i)
        out = policy.get_action(sample["obs"])
        num = sum(np.square(out[k].astype(float) - refs[i][k].astype(float)).sum() for k in out)
        den = sum(np.square(refs[i][k].astype(float)).sum() for k in out)
        eager_self_checks.append(
            {
                "sample": i,
                "nmse": float(num / max(den, 1e-30)),
                "max_abs": max(float(np.max(np.abs(out[k] - refs[i][k]))) for k in out),
            }
        )
    report["eager_self_checks"] = eager_self_checks
    graphs = None
    if args.mode == "head_graph":
        from research.holoq_bench.graphs import install

        graphs = install(policy)
    checks = []
    for i in list(range(10)) + list(reversed(range(10))):
        seed(1000 + i)
        out = policy.get_action(selected[i]["obs"])
        assert all(np.isfinite(v).all() for v in out.values()), "Nonfinite smoke action"
        num = sum(np.square(out[k].astype(float) - refs[i][k].astype(float)).sum() for k in out)
        den = sum(np.square(refs[i][k].astype(float)).sum() for k in out)
        checks.append(
            {
                "sample": i,
                "nmse": float(num / max(den, 1e-30)),
                "max_abs": max(float(np.max(np.abs(out[k] - refs[i][k]))) for k in out),
            }
        )
    report["execution_drift_checks"] = checks
    # Do not infer task equivalence from this tolerance. Each deployed execution
    # mode has its OWN rollout score, even if differences are only FP16 roundoff.
    drift_limit = max(0.02, 10 * max(c["nmse"] for c in eager_self_checks))
    report["drift_guard_nmse"] = drift_limit
    report["drift_guard_note"] = (
        "coarse bug guard relative to eager self variance, not numerical equivalence or a task-accuracy guarantee"
    )
    assert max(c["nmse"] for c in checks) < drift_limit, (
        "Large eager/deployed action drift, investigate before rollout"
    )
    assert all(m.calls > 0 for m in layers.values())
    report["active_layer_calls_after_smoke"] = {n: m.calls for n, m in layers.items()}
    report["per_layer_input_shapes"] = {n: sorted(m.input_shapes) for n, m in layers.items()}
    for i in range(args.warmup):
        seed(2000 + i)
        policy.get_action(selected[i % 10]["obs"])
    torch.cuda.synchronize()
    report["graph_count"] = len(graphs.cache) if graphs else 0
    write(args.output / "metadata.json", report)
    if args.stage == "smoke":
        # Record actual CUDA kernel names once outside all measurement phases.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            policy.get_action(selected[0]["obs"])
            torch.cuda.synchronize()
        events = Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
        write(args.output / "cuda_kernels.json", dict(events))
        print("SMOKE_OK", report["method"], args.mode, "drift", max(c["nmse"] for c in checks), flush=True)
        return

    def timed(obs, value):
        seed(value)
        torch.cuda.synchronize()
        start = time.perf_counter()
        action = policy.get_action(obs)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        if not all(np.isfinite(v).all() for v in action.values()):
            raise RuntimeError("Nonfinite policy action")
        return action, latency

    if args.stage == "replay":
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        with (args.output / "calls.jsonl").open("w") as file:
            for repeat in range(args.repeats):
                for sample in samples:
                    value = 42 + sample["task_id"] * 10000 + sample["init_index"] * 1000
                    _, latency = timed(sample["obs"], value)
                    latencies.append(latency)
                    file.write(
                        json.dumps(
                            {
                                "repeat": repeat,
                                "task_id": sample["task_id"],
                                "init_index": sample["init_index"],
                                "ms": latency,
                            }
                        )
                        + "\n"
                    )
        report.update(latency=stats(latencies), inference_memory=memory(), finished=time.time())
        write(args.output / "summary.json", report)
        print("REPLAY", report["method"], args.mode, report["latency"], flush=True)
        return
    from tools.analyze_layerwise_quant_drift import ensure_libero_runtime, _convert_libero_observation

    ensure_libero_runtime()
    from libero.libero import benchmark
    from examples.Libero.eval.utils import get_libero_env, get_libero_dummy_action
    from examples.Libero.eval.run_libero_eval import ExternalPolicy, get_max_steps

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    converter = ExternalPolicy.__new__(ExternalPolicy)
    converter.policy_output_format = "gr00t_dict"
    converter.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    episodes_path = args.output / "episodes.jsonl"
    previous = (
        [json.loads(line) for line in episodes_path.read_text().splitlines()] if episodes_path.exists() else []
    )
    complete = {(e["task_id"], e["init_index"]) for e in previous}
    episodes = list(previous)
    peak_allocated = peak_reserved = 0
    for task_id in range(10):
        task = suite.get_task(task_id)
        env, description = get_libero_env(task, resolution=256)
        try:
            states = suite.get_task_init_states(task_id)
            for init in range(10, 10 + args.trials):
                if (task_id, init) in complete:
                    continue
                seed(42 + task_id * 10000 + init * 1000)
                env.reset()
                obs = env.set_init_state(states[init])
                for _ in range(10):
                    obs, _, _, _ = env.step(get_libero_dummy_action())
                torch.cuda.reset_peak_memory_stats()
                latencies = []
                done = False
                with (args.output / "rollout_calls.jsonl").open("a", buffering=1) as call_file:
                    for step in range(get_max_steps("libero_object", "groot")):
                        converted = _convert_libero_observation(obs, task.language)
                        before = len(graphs.cache) if graphs else 0
                        result, latency = timed(converted, 42 + task_id * 10000 + init * 1000 + step)
                        after = len(graphs.cache) if graphs else 0
                        call_file.write(
                            json.dumps(
                                {
                                    "task_id": task_id,
                                    "init_index": init,
                                    "control_step": step,
                                    "ms": latency,
                                    "cold_capture": after != before,
                                }
                            )
                            + "\n"
                        )
                        if before == after:
                            latencies.append(latency)
                        action = converter._convert_to_libero_action(result, 0)
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            break
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                episode = {
                    "task_id": task_id,
                    "task": description,
                    "init_index": init,
                    "success": bool(done),
                    "control_steps": step + 1,
                    "policy_latency": stats(latencies),
                    "initial_state_sha256": hashlib.sha256(np.asarray(states[init]).tobytes()).hexdigest(),
                }
                episodes.append(episode)
                with episodes_path.open("a") as file:
                    file.write(json.dumps(episode) + "\n")
                scores = [sum(e["success"] for e in episodes if e["task_id"] == t) for t in range(10)]
                report.update(
                    episodes=len(episodes),
                    successes=sum(scores),
                    per_task_successes=scores,
                    inference_memory={
                        "peak_allocated_bytes": peak_allocated,
                        "peak_reserved_bytes": peak_reserved,
                    },
                    updated=time.time(),
                )
                write(args.output / "summary.json", report)
                print(
                    "EPISODE",
                    report["method"],
                    args.mode,
                    task_id,
                    init,
                    bool(done),
                    "TOTAL",
                    sum(scores),
                    "/",
                    len(episodes),
                    flush=True,
                )
        finally:
            env.close()
    calls = [json.loads(line) for line in (args.output / "rollout_calls.jsonl").read_text().splitlines()]
    report.update(
        latency=stats([c["ms"] for c in calls if not c["cold_capture"]]),
        cold_capture_calls=sum(c["cold_capture"] for c in calls),
        finished=time.time(),
    )
    write(args.output / "summary.json", report)
    print(
        "ROLLOUT_COMPLETE",
        report["method"],
        args.mode,
        report["successes"],
        report["episodes"],
        report["latency"],
        flush=True,
    )


if __name__ == "__main__":
    main()
