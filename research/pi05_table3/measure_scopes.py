"""Backbone/head/both deployment isolation and cumulative eager profiling.

No task-success experiment. Same frozen historical grid in every subset.
"""

import argparse
from collections import Counter, defaultdict
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
import torch
from research.pi05_table3.measure import sha, write, memory, sample_gpu


def part(name):
    return "vision" if "vision_tower" in name else "expert" if "gemma_expert" in name else "language"


def annotate(obj, attr, label):
    original = getattr(obj, attr)

    def forward(*args, **kwargs):
        with torch.profiler.record_function(label(*args, **kwargs) if callable(label) else label):
            return original(*args, **kwargs)

    setattr(obj, attr, forward)


def profile_policy(policy, sample, noise, output, raw_only=False):
    from gr00t.quantization.gptq_layers import GptqLinear
    from research.pi05_table3.vision_packed import PackedVision
    from research.pi05_table3.definition_runtime import DefinitionLinear

    model = policy._model
    policy._sample_actions = model.sample_actions
    annotate(policy, "_input_transform", "stage/preprocess")
    annotate(policy, "_output_transform", "stage/postprocess")
    for name, module in model.named_modules():
        if isinstance(module, (GptqLinear, PackedVision, DefinitionLinear)):
            annotate(module, "forward", "layer/" + part(name) + "/" + name)
    annotate(model.paligemma_with_expert.paligemma.model.vision_tower, "forward", "stage/vision")

    def joint(*args, **kwargs):
        inputs = kwargs.get("inputs_embeds")
        return "stage/language" if inputs is not None and inputs[0] is not None else "stage/expert"

    annotate(model.paligemma_with_expert, "forward", joint)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        for i in range(5):
            with torch.profiler.record_function(f"complete_policy/{i}"):
                policy.infer(sample, noise=noise)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output / "eager_policy_trace.json"))
    if raw_only:
        # Derive device totals from raw activities/correlation IDs. The
        # propagated FunctionEvent associations can multiply-count kernels.
        return
    (output / "top_self_cuda.txt").write_text(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40) + "\n"
    )
    rows = []
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        ancestors = []
        current = event
        while current is not None:
            ancestors.append(current.name)
            current = current.cpu_parent
        phase = next((x.split("/")[1] for x in ancestors if x.startswith(("layer/", "stage/"))), "other")
        for kernel in event.kernels:
            name = kernel.name
            if "_gemm_a4w4_arithmetic" in name or "_vision_gemm" in name:
                category = "GEMM_with_W_decode_and_scales"
            elif "_quant_pack_a4" in name:
                category = "A4_quantize_pack"
            elif event.name == "aten::bmm" and any(
                x.startswith("layer/") and "/vision/" not in x for x in ancestors
            ):
                category = "input_or_output_rotation"
            elif phase == "vision" and any(x.startswith("layer/vision/") for x in ancestors):
                category = "vision_A4_prepare_or_data"
            elif any(token in event.name for token in ["copy", "contiguous", "clone", "to_copy"]):
                category = "copy_dtype_contiguous"
            else:
                category = "other"
            rows.append(
                {
                    "phase": phase,
                    "category": category,
                    "kernel": name,
                    "gpu_us": kernel.duration,
                    "cpu_owner": event.name,
                }
            )
    totals = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    for row in rows:
        entry = totals[row["phase"] + "/" + row["category"]]
        entry["calls"] += 1
        entry["gpu_ms"] += row["gpu_us"] / 1000
    gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    direct = sum(e.device_time_total for e in gpu)
    assigned = sum(x["gpu_us"] for x in rows)
    summary = {
        "profiled_calls": 5,
        "groups": {
            k: {"calls_per_policy": v["calls"] / 5, "gpu_ms_per_policy": v["gpu_ms"] / 5}
            for k, v in sorted(totals.items(), key=lambda x: -x[1]["gpu_ms"])
        },
        "assigned_GPU_us": assigned,
        "direct_GPU_us": direct,
        "assigned_matches_direct": abs(assigned - direct) < max(1e-3, direct * 1e-5),
        "note": "Cumulative GPU work, not wall latency. Phase attribution uses eager CPU ancestry; CUDA Graph is separately timed. Do not add overlapping GPU work to CPU wall time.",
        "cpu_sync_wait_us": sum(
            e.self_cpu_time_total
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CPU
            and any(t in e.name.lower() for t in ["synchronize", "streamwait"])
        ),
    }
    write(output / "profile_kernel_attribution.json", rows)
    write(output / "profile_summary.json", summary)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--scope", choices=["both", "backbone", "head"], required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--method", choices=["historical", "definition8"], default="historical")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    data = root / "results/pi05_table3_414_20261028"
    out = data / ("scope_controls" if a.method == "historical" else "definition8_scope_controls") / a.scope
    deployment = (
        data / ("holoq_deployments" if a.method == "historical" else "definition8_scope_deployments") / a.scope
    )
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    out.mkdir(parents=True, exist_ok=False)
    for k in list(os.environ):
        if k.startswith(("GR00T_", "OPENPI_GPTQ", "QVLA_ACT")):
            os.environ.pop(k)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from gr00t.quantization.pi05_cuda_graph import Pi05CudaGraphSampler
    from gr00t.quantization.gptq_layers import GptqLinear
    from research.pi05_table3.vision_packed import PackedVision

    if a.method == "historical":
        from research.pi05_table3.historical_loader import export_scope, load_policy

        steps = 10
        linear_type = GptqLinear
    else:
        from research.pi05_table3.definition_scopes import export_scope
        from research.pi05_table3.definition_runtime import load_policy as load_definition, DefinitionLinear

        def load_policy(directory, cfg):
            return load_definition(directory, cfg, "packed")

        steps = 8
        linear_type = DefinitionLinear

    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    export_scope(deployment, a.scope)
    policy = load_policy(deployment, cfg)
    model = policy._model
    model.eval().requires_grad_(False)
    install_pi05_step_context(model, fixed_loop=True)
    policy._sample_actions = model.sample_actions
    path = Path("/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt")
    samples = torch.load(path, map_location="cpu", weights_only=False)["samples"][10:20]
    noise = (
        np.random.default_rng(42)
        .standard_normal((cfg.model.action_horizon, cfg.model.action_dim))
        .astype(np.float32)
    )
    modules = {n: m for n, m in model.named_modules() if isinstance(m, (linear_type, PackedVision))}
    expected = {"both": 414, "backbone": 288, "head": 126}[a.scope]
    assert len(modules) == expected
    report = {
        "scope": a.scope,
        "quantized_layers": expected,
        "actual_parts": dict(Counter(part(n) for n in modules)),
        "deployment_manifest_sha256": sha(deployment / "deployment.json"),
        "observations_sha256": sha(path),
        "sample_indices": list(range(10, 20)),
        "steps": steps,
        "method": a.method,
        "action_horizon": cfg.model.action_horizon,
        "batch": 1,
        "warmup": 32,
        "modes": {},
        "started": time.time(),
        "role": "Deployment-isolation only. No task-success claim. Same frozen quantized grid retained in selected subsets.",
        "source_sha256": {str(f.relative_to(root)): sha(f) for f in (root / "research/pi05_table3").glob("*.py")},
    }
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, out / "gpu_samples.jsonl"), daemon=True)
    sampler.start()
    eager = []
    try:
        with torch.inference_mode():
            for mode in ["eager", "sample_graph"]:
                if mode == "sample_graph":
                    policy._sample_actions = Pi05CudaGraphSampler(model.sample_actions)
                for i in range(32):
                    policy.infer(samples[i % 10], noise=noise)
                torch.cuda.synchronize()
                if mode == "eager":
                    eager = [policy.infer(s, noise=noise)["actions"].copy() for s in samples]
                else:
                    errors = [
                        float(np.max(np.abs(policy.infer(s, noise=noise)["actions"] - eager[i])))
                        for i, s in enumerate(samples)
                    ]
                    if a.method == "historical":
                        assert max(errors) <= 1e-6, errors
                    else:
                        nmse = [
                            float(
                                np.square(policy.infer(s, noise=noise)["actions"].astype(float) - eager[i]).sum()
                                / max(np.square(eager[i].astype(float)).sum(), 1e-30)
                            )
                            for i, s in enumerate(samples)
                        ]
                        assert max(nmse) <= 0.01, nmse
                        report["graph_eager_nmse"] = nmse
                    report["graph_eager_max_abs"] = errors
                stable = memory()
                torch.cuda.reset_peak_memory_stats()
                times = []
                with (out / (mode + "_calls.jsonl")).open("w") as stream:
                    for rep in range(10):
                        for index, obs in enumerate(samples):
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            output = policy.infer(obs, noise=noise)
                            torch.cuda.synchronize()
                            ms = (time.perf_counter() - start) * 1000
                            assert np.isfinite(output["actions"]).all()
                            times.append(ms)
                            stream.write(json.dumps({"repeat": rep, "sample": 10 + index, "ms": ms}) + "\n")
                report["modes"][mode] = {
                    "mean_ms": float(np.mean(times)),
                    "p50_ms": float(np.percentile(times, 50)),
                    "p95_ms": float(np.percentile(times, 95)),
                    "count": len(times),
                    "policy_calls_per_s": 1000 / float(np.mean(times)),
                    "stable_memory": stable,
                    "inference_memory": memory(),
                }
                write(out / "report.json", report)
                print("SCOPE_TIMING", a.scope, mode, report["modes"][mode], flush=True)
            # Separate diagnostic, after all formal timings. Resetting peaks
            # here cannot alter the already-recorded full-policy memory peaks.
            policy._sample_actions = model.sample_actions
            original = model.denoise_step
            head = {}
            count = [0]

            def denoise(*args, **kwargs):
                if count[0] == 0:
                    head["entry_allocated_bytes"] = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                count[0] += 1
                return original(*args, **kwargs)

            model.denoise_step = denoise
            policy.infer(samples[0], noise=noise)
            torch.cuda.synchronize()
            model.denoise_step = original
            head.update(
                through_policy_completion=memory(),
                steps=count[0],
                note="From first denoise entry through final policy completion, over resident model and prefix state. Increment is not total process memory.",
            )
            head["incremental_peak_bytes"] = (
                head["through_policy_completion"]["peak_allocated_bytes"] - head["entry_allocated_bytes"]
            )
            report["head_stage_memory"] = head
            if a.profile:
                profile_policy(policy, samples[0], noise, out)
            report.update(status="complete", finished=time.time())
            write(out / "report.json", report)
    finally:
        stop.set()
        sampler.join(timeout=25)


if __name__ == "__main__":
    main()
