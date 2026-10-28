"""Audited complete-policy replay/rollout with independent LIBERO states.

Use one process/GPU, never parallelize timed runs. All timings synchronize
before/after the entire get_action, including preprocessing/transfers/output.
"""

import argparse
import fcntl
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
    p.add_argument("--common-logits", action="store_true", help="Same unused-logits optimization offered to every method")
    p.add_argument("--init-start", type=int, default=10)
    p.add_argument("--suite", choices=("libero_object", "libero_10"), default="libero_object")
    p.add_argument("--episode-env-seed", type=int)
    p.add_argument("--evaluation-manifest", type=Path)
    p.add_argument("--validation", type=Path, help="Required correctness-gate summary for submission_contract_v2 rollout")
    args = p.parse_args()
    # This experiment has multiple independent controllers but one GPU0.
    # Serialize smoke/compilation as well: two large packed models must not
    # race for memory when controllers reach their smoke stage together.
    timing_guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(timing_guard, fcntl.LOCK_EX)
    args.output.mkdir(parents=True, exist_ok=True)
    args.deployment = args.deployment.resolve()
    os.environ.update(HOLOQ_DEPLOY=str(args.deployment), HOLOQ_COMPUTE_DTYPE="float16")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(42)
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy
    from research.holoq_bench.packed_runtime import PackedLinear
    from research.rebuttal.runtime import PaperLinear

    cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=str(args.deployment),
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )
    if args.common_logits:
        from research.holoq_bench.common_logits import install

        install(policy)
    pack = torch.load(args.deployment / "packed.pt", map_location="cpu", mmap=True, weights_only=False)
    layers = {n: m for n, m in policy.model.named_modules() if isinstance(m, (PackedLinear, PaperLinear))}
    assert set(layers) == set(pack["layers"])
    source_root = Path(__file__).resolve().parents[2]
    source_files = [
        *source_root.glob("research/holoq_bench/*.py"),
        *source_root.glob("research/rebuttal/*.py"),
        *source_root.glob("research/native_int4/*"),
        source_root / "research/holoq_bench/cutlass_bindings.cpp",
        source_root / "gr00t/model/policy.py",
        source_root / "gr00t/model/action_head/flow_matching_action_head.py",
        source_root / "gr00t/quantization/perstep_backends.py",
        source_root / "gr00t/quantization/dit_step_context.py",
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
        "init_indices": list(range(args.init_start, args.init_start + args.trials)),
        "suite": args.suite,
        "episode_env_seed": args.episode_env_seed,
        "started": time.time(),
    }
    if args.common_logits:
        report["common_optimizations"] = ["qwen3_last_token_logits"]
    from research.holoq_bench.deployment import sha256
    from research.holoq_bench.rollout_resume import validate_resume_metadata

    report["observation_file_sha256"] = sha256(args.observations)
    if args.evaluation_manifest:
        evaluation = json.loads(args.evaluation_manifest.read_text())
        assert evaluation["observations_sha256"] == report["observation_file_sha256"]
        assert evaluation["suite"] == args.suite and evaluation["env_seed_per_episode"] == args.episode_env_seed
        assert set(report["init_indices"]) <= set(evaluation["init_indices"])
        args.expected_state_hashes = {(r["task_id"], r["init_index"]): r["initial_state_sha256"] for r in evaluation["episodes"]}
        report["evaluation_manifest_sha256"] = sha256(args.evaluation_manifest)
    if pack["meta"].get("runtime_plugin") == "submission_contract_v2":
        report["runtime_implementation"] = os.environ["HOLOQ_REBUTTAL_IMPLEMENTATION"]
        report["method"] += "_" + report["runtime_implementation"]
        reference_mode = report["runtime_implementation"] == "reference"
        report["deployment_arithmetic"] = "FP32 corrected equation oracle; accuracy reference, not optimized deployment" if reference_mode else pack["meta"]["deployment_arithmetic"]
        report["role"] = "accuracy reference only; do not use its timing to claim superiority over an optimized quantized baseline" if reference_mode else "actual packed deployment"
        report["includes"] = ["A4 rounding/clamping and floating reconstruction (no nibble-packed activation buffer)" if item == "activation quantize/pack" else ("FP32 equation oracle" if reference_mode else "packed W4 decode + FP16 MMA/FP32 accumulation") if item == "packed GEMM" else item for item in report["includes"]]
        if args.stage == "rollout":
            assert args.validation is not None and args.evaluation_manifest is not None
            gate = json.loads(args.validation.read_text())
            assert gate["passed"] and gate["pack_sha256"] == report["source_pack_sha256"]
            assert gate["observation_sha256"] == report["observation_file_sha256"]
            report["validation_sha256"] = sha256(args.validation)
    previous_memory, previous_windows = validate_resume_metadata(args.output, report)
    report["resume_measurement_windows"] = previous_windows
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
    report["stable_memory_after_warmup"] = memory()
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
    from research.holoq_bench.rollout_driver import run

    run(args, report, graphs, timed, previous_memory, stats, memory, write)


if __name__ == "__main__":
    main()
