"""Validate frozen reference/packed actions before complete-policy timing.

No task-success claim. Ten initial snapshots are not ten trajectories.
All engineering gates precede formal timing and are saved even on failure.
"""

import dataclasses
import fcntl
import gc
import json
import os
from pathlib import Path
import threading
import time
from collections import Counter
import numpy as np
import torch
from research.pi05_table3.measure import sha, write, memory, sample_gpu


def difference(value, reference):
    a, b = np.asarray(value, dtype=float), np.asarray(reference, dtype=float)
    return {
        "nmse": float(np.square(a - b).sum() / max(np.square(b).sum(), 1e-30)),
        "max_abs": float(np.abs(a - b).max()),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    data = root / "results/pi05_table3_414_20261028"
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    out = data / "definition8_measurement"
    out.mkdir(exist_ok=False)
    for key in list(os.environ):
        if key.startswith(("GR00T_", "OPENPI_GPTQ", "QVLA_ACT")):
            os.environ.pop(key)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from gr00t.quantization.pi05_cuda_graph import Pi05CudaGraphSampler
    from research.pi05_table3.definition_runtime import load_policy, DefinitionLinear, unpack_codes
    from research.pi05_table3.paper_math import equation_oracle
    from research.pi05_table3.vision_packed import PackedVision

    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    deployment = data / "definition8_deployment_fp64"
    source = Path("/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt")
    samples = torch.load(source, map_location="cpu", weights_only=False)["samples"][10:20]
    noise = (
        np.random.default_rng(42)
        .standard_normal((cfg.model.action_horizon, cfg.model.action_dim))
        .astype(np.float32)
    )
    report = {
        "method": "definition8_core_with_released_vision_rtn",
        "steps": 8,
        "scope": {"vision": 162, "language": 126, "expert": 126},
        "batch": 1,
        "quantized_layers": 414,
        "action_horizon": cfg.model.action_horizon,
        "warmup": 32,
        "sample_indices": list(range(10, 20)),
        "noise_seed": 42,
        "observations_sha256": sha(source),
        "deployment_manifest_sha256": sha(deployment / "deployment.json"),
        "local_nmse_limit": 1e-4,
        "action_nmse_limit": 0.01,
        "calibration_caveat": "10 initial observations, not10 trajectories; no success-rate claim",
        "source_sha256": {str(f.relative_to(root)): sha(f) for f in (root / "research/pi05_table3").glob("*.py")},
        "modes": {},
        "started": time.time(),
        "status": "validating",
    }
    write(out / "report.json", report)

    def ready(implementation):
        policy = load_policy(deployment, cfg, implementation)
        policy._model.eval().requires_grad_(False)
        # PI0 constructor sets high matmul precision. Disable TF32 explicitly
        # after construction so the FP32 oracle really uses FP32 products.
        torch.backends.cuda.matmul.allow_tf32 = False
        install_pi05_step_context(policy._model, fixed_loop=True)
        policy._sample_actions = policy._model.sample_actions
        return policy

    def capture(policy):
        original = policy._model.denoise_step
        velocities = []

        def denoise(*args, **kwargs):
            value = original(*args, **kwargs)
            velocities.append(value.float().cpu().numpy().copy())
            return value

        policy._model.denoise_step = denoise
        actions, steps = [], []
        try:
            for sample in samples:
                velocities.clear()
                actions.append(policy.infer(sample, noise=noise)["actions"].copy())
                assert len(velocities) == 8, len(velocities)
                steps.append(np.stack(velocities))
        finally:
            policy._model.denoise_step = original
        return np.stack(actions), np.stack(steps)

    with torch.inference_mode():
        reference = ready("reference")
        ref_actions, ref_steps = capture(reference)
        np.savez(out / "reference_outputs.npz", actions=ref_actions, velocities=ref_steps, noise=noise)
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        policy = ready("packed")
        core = {n: m for n, m in policy._model.named_modules() if isinstance(m, DefinitionLinear)}
        vision = {n: m for n, m in policy._model.named_modules() if isinstance(m, PackedVision)}
        assert len(core) == 252 and len(vision) == 162
        local = []

        def oracle_hook(module, args, output):
            q, sa = module.prepare(args[0])
            expected = (
                equation_oracle(q, unpack_codes(module.B), sa, module.s, module.bias)
                .bfloat16()
                .reshape(output.shape)
            )
            error = difference(output.float().cpu().numpy(), expected.float().cpu().numpy())
            local.append({"layer": module.name, "shape": list(args[0].shape), **error})

        handles = [m.register_forward_hook(oracle_hook) for m in core.values()]
        try:
            policy.infer(samples[0], noise=noise)
        finally:
            for handle in handles:
                handle.remove()
        write(out / "local_oracle_checks.json", local)
        assert len(local) == 126 + 126 * 8, len(local)
        report["local_checks"] = {"count": len(local), "max_nmse": max(x["nmse"] for x in local)}
        write(out / "report.json", report)
        assert report["local_checks"]["max_nmse"] <= report["local_nmse_limit"], report["local_checks"]
        actions, steps = capture(policy)
        np.savez(out / "packed_outputs.npz", actions=actions, velocities=steps)
        report["reference_packed_action"] = [difference(a, b) for a, b in zip(actions, ref_actions)]
        report["reference_packed_per_step"] = [
            [difference(a, b) for a, b in zip(row, ref)] for row, ref in zip(steps, ref_steps)
        ]
        report["max_action_nmse"] = max(x["nmse"] for x in report["reference_packed_action"])
        write(out / "report.json", report)
        assert np.isfinite(actions).all() and report["max_action_nmse"] <= report["action_nmse_limit"], report[
            "max_action_nmse"
        ]
        report["coverage"] = {
            n: {"act_mode": m.act_mode, "steps_seen": sorted(m.steps_seen), "shapes": sorted(m.shapes)}
            for n, m in core.items()
        }
        assert all(m.steps_seen == set(range(8)) for m in core.values() if m.act_mode == "step")
        report["validation_passed"] = True
        write(out / "report.json", report)
        print("DEFINITION8_VALIDATED", report["max_action_nmse"], flush=True)
        stop = threading.Event()
        sampler = threading.Thread(target=sample_gpu, args=(stop, out / "gpu_samples.jsonl"), daemon=True)
        sampler.start()
        try:
            eager = None
            for mode in ["eager", "sample_graph"]:
                if mode == "sample_graph":
                    policy._sample_actions = Pi05CudaGraphSampler(policy._model.sample_actions)
                for i in range(32):
                    policy.infer(samples[i % 10], noise=noise)
                torch.cuda.synchronize()
                current = [policy.infer(s, noise=noise)["actions"].copy() for s in samples]
                if eager is None:
                    eager = current
                else:
                    report["graph_eager"] = [difference(a, b) for a, b in zip(current, eager)]
                    assert max(x["nmse"] for x in report["graph_eager"]) <= 0.01
                stable = memory()
                torch.cuda.reset_peak_memory_stats()
                times = []
                with (out / (mode + "_calls.jsonl")).open("w") as file:
                    for rep in range(10):
                        for index, sample in enumerate(samples):
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            result = policy.infer(sample, noise=noise)
                            torch.cuda.synchronize()
                            ms = (time.perf_counter() - start) * 1000
                            assert np.isfinite(result["actions"]).all()
                            times.append(ms)
                            file.write(json.dumps({"repeat": rep, "sample": 10 + index, "ms": ms}) + "\n")
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
                print("DEFINITION8_TIMING", mode, report["modes"][mode], flush=True)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                policy.infer(samples[0], noise=noise)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(out / "graph_trace.json"))
            report["actual_cuda_kernel_names"] = dict(
                Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
            )
            report.update(
                status="complete",
                finished=time.time(),
                boundary="Complete synchronized policy.infer including preprocessing,H2D,all414 selected linears,8denoise steps,online transforms,packing,GEMM,D2H,output transforms; excludes RPC/simulator/cold capture",
            )
            write(out / "report.json", report)
        finally:
            stop.set()
            sampler.join(timeout=25)


if __name__ == "__main__":
    main()
