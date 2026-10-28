#!/usr/bin/env python
"""Paired eager baseline / Omega W4A4 / packed-W4A4 inference benchmark.

All modes use the same checkpoint, observation, initial noise and step count.
Optional CUDA Graph runs compare all modes under matched replay execution.
Scope is 252 LM/expert linears; vision stays floating point.
The experimental packed backend uses BF16 MMA, not INT4 MMA. It preserves
per-step channel scales and performs no M-dependent dense fallback.
"""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit_inference(policy, obs, noise, num_steps):
    from gr00t.quantization.dit_step_context import get_current_dit_step
    from gr00t.quantization.gptq_layers import GptqLinear

    counts, handles = {}, []
    token_counts = Counter()
    for name, layer in policy._model.named_modules():
        if not isinstance(layer, GptqLinear):
            continue
        bits = layer._record_a_bits if layer._record_a_bits is not None else layer.cfg.act_bits
        if bits != 4 or layer.weight_bits != 4 or not layer._has_act_scale_table:
            raise RuntimeError(f"{name}: expected a calibrated W4A4 layer")
        rows = layer._act_scale_table.shape[0]
        if rows not in (1, num_steps):
            raise RuntimeError(f"{name}: {rows} scale buckets vs {num_steps} inference steps")
        layer._strict_perstep = True
        counts[name] = Counter()

        def hook(mod, args, name=name, rows=rows):
            step = get_current_dit_step()
            if rows > 1 and (step is None or not 0 <= step < rows):
                raise RuntimeError(f"{name}: invalid timestep {step}")
            if rows == 1 and step is not None:
                raise RuntimeError(f"{name}: prefix layer unexpectedly called during denoising")
            counts[name]["prefix" if rows == 1 else str(step)] += 1
            x = args[0]
            token_counts[str(x.numel() // x.shape[-1])] += 1

        handles.append(layer.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            output = policy.infer(obs, noise=noise)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    expected = {"prefix"} | {str(i) for i in range(num_steps)}
    observed = set().union(*(set(v) for v in counts.values()))
    if counts and (len(counts) != 252 or observed != expected):
        raise RuntimeError(f"Incomplete W4A4 coverage: {len(counts)} layers, buckets={observed}")
    for name, buckets in counts.items():
        want = {"prefix": 1} if "prefix" in buckets else {str(i): 1 for i in range(num_steps)}
        if dict(buckets) != want:
            raise RuntimeError(f"{name}: unexpected dispatch counts {buckets}")
    return output, {"layers": len(counts), "token_counts": dict(token_counts),
                    "layer_step_counts": {k: dict(v) for k, v in counts.items()},
                    "perstep_dispatch_verified": bool(counts)}


def measure(policy, obs, noise, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            policy.infer(obs, noise=noise)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            policy.infer(obs, noise=noise)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)),
            "std_ms": float(np.std(samples)), "samples_ms": samples,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def profile_inference(policy, obs, noise, trace_path):
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        policy.infer(obs, noise=noise)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    kernels = {}
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        name = event["name"]
        item = kernels.setdefault(name, {"calls": 0, "gpu_ms": 0.0})
        item["calls"] += 1
        item["gpu_ms"] += event.get("dur", 0) / 1000
    operators = [{"name": event.key, "calls": event.count,
                  "self_cpu_ms": event.self_cpu_time_total / 1000,
                  "self_gpu_ms": event.self_device_time_total / 1000,
                  "inclusive_gpu_ms": event.device_time_total / 1000}
                 for event in prof.key_averages()]
    return {"trace": str(trace_path), "gpu_kernel_count": sum(v["calls"] for v in kernels.values()),
            "gpu_kernel_ms": sum(v["gpu_ms"] for v in kernels.values()),
            "kernels": dict(sorted(kernels.items(), key=lambda item: -item[1]["gpu_ms"])),
            "operators": sorted(operators, key=lambda item: -item["self_gpu_ms"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/home/xinyu/models/pi05_libero_pytorch")
    parser.add_argument("--pack", default="/home/xinyu/models/omega-qvla-pi05-414-w4a4-int4/pi05_long/quantized.pt")
    parser.add_argument("--obs-path", default=str(ROOT / "deploy_extras/pi05_calib_obs.pt"))
    parser.add_argument("--obs-index", type=int, default=30)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--backend-version", type=int, choices=(1, 2, 3), default=3,
                        help="1: original LUT GEMM; 2: experimental arithmetic GEMM; 3: decode + cuBLAS")
    parser.add_argument("--cuda-graph", action="store_true", help="Also compare matched CUDA Graph sampling")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("Need positive warmup and repeat counts")
    for key in list(os.environ):
        if key.startswith(("GR00T_", "QVLA_ACT", "OPENPI_GPTQ")):
            os.environ.pop(key)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no CPU timing substitute")
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    from openpi.policies import policy_config
    from openpi.training import config
    from gr00t.quantization.gptq_layers import GptqConfig, wrap_gptq
    from gr00t.quantization.packed_w4a4 import enable_packed_w4a4
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from tools.bench_pi05_speed import INCLUDE_BOTH
    import re

    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    data = torch.load(args.obs_path, map_location="cpu", weights_only=False)
    observations = data["samples"] if isinstance(data, dict) else data
    obs = observations[args.obs_index]
    noise = np.random.default_rng(args.seed).standard_normal(
        (cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
    result = {"arguments": {**vars(args), "output": str(args.output)},
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "model_config": {"name": "pi05_libero", "action_horizon": cfg.model.action_horizon,
                               "action_dim": cfg.model.action_dim, "max_token_len": cfg.model.max_token_len},
              "execution": ("eager plus matched CUDA Graph replay; no torch.compile" if args.cuda_graph
                            else "eager for all modes; no torch.compile"),
              "scope": "252 LM/expert linears; vision floating point",
              "note": "Packed W4A4 uses BF16 MMA, not native INT4 MMA. No LIBERO success-rate evaluation.",
              "modes": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def build():
        policy = policy_config.create_trained_policy(cfg, args.checkpoint,
                                                     pytorch_device="cuda",
                                                     sample_kwargs={"num_steps": args.steps})
        policy._model.eval()
        policy._model.requires_grad_(False)
        install_pi05_step_context(policy._model, fixed_loop=args.cuda_graph)
        policy._sample_actions = policy._model.sample_actions
        return policy

    def graph_measure(mode, eager_output):
        from gr00t.quantization.pi05_cuda_graph import Pi05CudaGraphSampler

        runner = Pi05CudaGraphSampler(policy._model.sample_actions)
        policy._sample_actions = runner
        try:
            with torch.inference_mode():
                output = policy.infer(obs, noise=noise)
            error = np.asarray(output["actions"]) - np.asarray(eager_output["actions"])
            max_error = float(np.max(np.abs(error)))
            if not np.isfinite(error).all() or max_error > 1e-6:
                raise RuntimeError(f"CUDA Graph changed outputs: max error {max_error}")
            key = mode + "_cuda_graph"
            result["modes"][key] = {**measure(policy, obs, noise, args.warmup, args.repeats),
                                     "same_mode_eager_action_max_abs": max_error,
                                     "execution": "CUDA Graph replay; same quantization and per-step scales"}
            save()
            print("[w4a4]", key, result["modes"][key]["mean_ms"], flush=True)
        finally:
            policy._sample_actions = policy._model.sample_actions
            del runner
            gc.collect()
            torch.cuda.empty_cache()

    save()
    policy = build()
    baseline_output, baseline_audit = audit_inference(policy, obs, noise, args.steps)
    result["modes"]["baseline_bf16"] = {**measure(policy, obs, noise, args.warmup, args.repeats),
                                          "audit": baseline_audit}
    if args.profile:
        result["modes"]["baseline_bf16"]["profile"] = profile_inference(
            policy, obs, noise, args.output.parent / "baseline_trace.json")
    save()
    print("[w4a4] baseline", result["modes"]["baseline_bf16"]["mean_ms"], flush=True)
    if args.cuda_graph:
        graph_measure("baseline_bf16", baseline_output)
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    policy = build()
    names = [name for name, mod in policy._model.named_modules()
             if isinstance(mod, torch.nn.Linear) and re.fullmatch(INCLUDE_BOTH, name)]
    if len(names) != 252:
        raise RuntimeError(f"Expected 252 target layers, found {len(names)}")
    wrap_gptq(policy._model, names, GptqConfig(enabled=True, path=args.pack, act_bits=4,
                                            weight_bits=4, missing="error"))
    policy._model.to("cuda")
    reference_output, reference_audit = audit_inference(policy, obs, noise, args.steps)
    result["modes"]["omega_w4a4_reference"] = {
        **measure(policy, obs, noise, args.warmup, args.repeats), "audit": reference_audit}
    if args.profile:
        result["modes"]["omega_w4a4_reference"]["profile"] = profile_inference(
            policy, obs, noise, args.output.parent / "reference_trace.json")
    save()
    print("[w4a4] reference", result["modes"]["omega_w4a4_reference"]["mean_ms"], flush=True)
    if args.cuda_graph:
        graph_measure("omega_w4a4_reference", reference_output)

    records = torch.load(args.pack, map_location="cpu", mmap=True, weights_only=False)
    backend = enable_packed_w4a4(policy._model, records, version=args.backend_version)
    gc.collect()
    torch.cuda.empty_cache()
    packed_output, packed_audit = audit_inference(policy, obs, noise, args.steps)
    result["modes"]["omega_w4a4_packed"] = {
        **measure(policy, obs, noise, args.warmup, args.repeats), "audit": packed_audit,
        "backend": backend}
    if args.profile:
        result["modes"]["omega_w4a4_packed"]["profile"] = profile_inference(
            policy, obs, noise, args.output.parent / "packed_trace.json")
    if args.cuda_graph:
        graph_measure("omega_w4a4_packed", packed_output)
    for label, output in (("reference_vs_baseline", reference_output),
                          ("packed_vs_baseline", packed_output)):
        diff = np.asarray(output["actions"]) - np.asarray(baseline_output["actions"])
        result[label] = {"action_rmse": float(np.sqrt(np.mean(diff ** 2))),
                         "action_max_abs": float(np.max(np.abs(diff)))}
    diff = np.asarray(packed_output["actions"]) - np.asarray(reference_output["actions"])
    ref = np.asarray(reference_output["actions"])
    result["packed_vs_reference"] = {"action_rmse": float(np.sqrt(np.mean(diff ** 2))),
                                      "action_max_abs": float(np.max(np.abs(diff))),
                                      "action_relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-8)),
                                      "finite": bool(np.isfinite(diff).all())}
    for mode, values in result["modes"].items():
        baseline_mode = "baseline_bf16_cuda_graph" if mode.endswith("_cuda_graph") else "baseline_bf16"
        values["latency_baseline_mode"] = baseline_mode
        values["latency_ratio_to_baseline"] = values["mean_ms"] / result["modes"][baseline_mode]["mean_ms"]
    save()
    print("[w4a4] summary", json.dumps({k: v["mean_ms"] for k, v in result["modes"].items()}), flush=True)
    print("[w4a4] packed_vs_reference", result["packed_vs_reference"], flush=True)
    print("[w4a4] saved", args.output, flush=True)


if __name__ == "__main__":
    main()
