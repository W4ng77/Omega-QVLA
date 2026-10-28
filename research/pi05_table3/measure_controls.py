"""Matched full-policy controls only, no LIBERO success-rate inference."""

import argparse
from contextlib import nullcontext
from collections import Counter
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
import torch
from research.pi05_table3.measure import sha, write, memory, sample_gpu


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--kind",
        choices=["bf16", "fp16", "gptq_marlin252", "gptq_awq252", "gptq_awq_bf16252", "smoothquant_qserve414"],
        required=True,
    )
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--steps", type=int, default=10, choices=[8, 10])
    p.add_argument("--output-group", default="controls")
    p.add_argument("--ieee-matmul", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    data = root / "results/pi05_table3_414_20261028"
    out = data / a.output_group / a.kind
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    out.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith(("GR00T_", "OPENPI_GPTQ", "QVLA_ACT")):
            os.environ.pop(key)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    from openpi.training import config
    from openpi.policies import policy_config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from gr00t.quantization.pi05_cuda_graph import Pi05CudaGraphSampler
    from research.pi05_table3.baseline_runtime import BaselineLinear, load_policy

    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    if a.kind in ["bf16", "fp16"]:
        policy = policy_config.create_trained_policy(
            cfg,
            "/home/xinyu/models/pi05_libero_pytorch",
            pytorch_device="cuda",
            sample_kwargs={"num_steps": a.steps},
        )
        if a.kind == "fp16":
            for value in policy._model.parameters():
                if value.dtype == torch.bfloat16:
                    value.data = value.data.half()
        role = (
            "High precision source BF16"
            if a.kind == "bf16"
            else "FP16 autocast; sensitive sourceFP32 parameters retained"
        )
    else:
        policy = load_policy(data / "baseline_deployments" / a.kind, cfg, a.steps)
        role = "Real packed quantized deployment; disclosed reconstructed calibration, no success-rate claim"
    policy._model.eval().requires_grad_(False)
    if a.ieee_matmul:
        # The model constructor calls set_float32_matmul_precision('high').
        # Set this AFTER loading, matching the definition8 oracle/deployment.
        torch.backends.cuda.matmul.allow_tf32 = False
    install_pi05_step_context(policy._model, fixed_loop=True)
    policy._sample_actions = policy._model.sample_actions
    path = Path("/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt")
    samples = torch.load(path, map_location="cpu", weights_only=False)["samples"][10:20]
    noise = (
        np.random.default_rng(42)
        .standard_normal((cfg.model.action_horizon, cfg.model.action_dim))
        .astype(np.float32)
    )
    modules = {n: m for n, m in policy._model.named_modules() if isinstance(m, BaselineLinear)}
    expected = 0 if a.kind in ["bf16", "fp16"] else (252 if a.kind.endswith("252") else 414)
    assert len(modules) == expected
    report = {
        "method": a.kind,
        "role": role,
        "marlin_max_par": 1 if a.kind == "gptq_marlin252" else None,
        "quantized_layers": len(modules),
        "backend_counts": dict(Counter(m.backend for m in modules.values())),
        "checkpoint": "/home/xinyu/models/pi05_libero_pytorch",
        "observations_sha256": sha(path),
        "sample_indices": list(range(10, 20)),
        "steps": a.steps,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "calibration_steps": 10 if modules else None,
        "action_horizon": cfg.model.action_horizon,
        "batch": 1,
        "noise_seed": 42,
        "warmup": 32,
        "modes": {},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {str(f.relative_to(root)): sha(f) for f in (root / "research/pi05_table3").glob("*.py")},
        "started": time.time(),
    }
    if modules:
        report["deployment_manifest_sha256"] = sha(data / "baseline_deployments" / a.kind / "deployment.json")
    if a.kind.startswith("gptq_awq"):
        oracle = data / (
            "gptq_awq_bf16_full_policy_oracle.json"
            if a.kind == "gptq_awq_bf16252"
            else "gptq_awq_full_policy_oracle.json"
        )
        checked = json.loads(oracle.read_text())
        assert checked["passed"] and checked["deployment_manifest_sha256"] == report["deployment_manifest_sha256"]
        assert checked["runtime_sha256"] == sha(root / "research/pi05_table3/baseline_runtime.py")
        report["frozen_grid_oracle_sha256"] = sha(oracle)
    write(out / "report.json", report)
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, out / "gpu_samples.jsonl"), daemon=True)
    sampler.start()
    eager = []
    try:
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16) if a.kind == "fp16" else nullcontext(),
        ):
            for mode in ["eager", "sample_graph"]:
                if mode == "sample_graph":
                    policy._sample_actions = Pi05CudaGraphSampler(policy._model.sample_actions)
                for i in range(32):
                    value = policy.infer(samples[i % 10], noise=noise)
                    assert np.isfinite(value["actions"]).all()
                torch.cuda.synchronize()
                if mode == "eager":
                    eager = [policy.infer(s, noise=noise)["actions"].copy() for s in samples]
                else:
                    outputs = [policy.infer(s, noise=noise)["actions"].copy() for s in samples]
                    errors = [float(np.max(np.abs(v - eager[i]))) for i, v in enumerate(outputs)]
                    nmse = [
                        float(
                            np.square(v.astype(float) - eager[i]).sum()
                            / max(np.square(eager[i].astype(float)).sum(), 1e-30)
                        )
                        for i, v in enumerate(outputs)
                    ]
                    if a.kind == "gptq_marlin252":
                        diagnostic = data / "marlin_maxpar1_full_policy_oracle.json"
                        check = json.loads(diagnostic.read_text())
                        assert check["passed"] and check["max_par"] == 1
                        assert check["runtime_sha256"] == sha(root / "research/pi05_table3/baseline_runtime.py")
                        assert check["deployment_manifest_sha256"] == report["deployment_manifest_sha256"]
                        # Use the same predeclared complete-action numerical
                        # guard as HoloQ, after independent per-call1e-4 checks.
                        assert max(nmse) <= 0.01, nmse
                        nonzero = sum(int(torch.count_nonzero(m.workspace)) for m in modules.values())
                        assert nonzero == 0, nonzero
                        report["graph_guard"] = {
                            "diagnostic_sha256": sha(diagnostic),
                            "limit_nmse": 0.01,
                            "nonzero_workspace_elements": nonzero,
                            "note": "Numerical execution guard, not task-success equivalence",
                        }
                    else:
                        assert max(errors) <= 1e-6, errors
                    report["graph_eager_max_abs"] = errors
                    report["graph_eager_nmse"] = nmse
                stable = memory()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                with (out / (mode + "_calls.jsonl")).open("w") as file:
                    for rep in range(a.repeats):
                        for index, obs in enumerate(samples):
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            result = policy.infer(obs, noise=noise)
                            torch.cuda.synchronize()
                            ms = (time.perf_counter() - started) * 1000
                            assert np.isfinite(result["actions"]).all()
                            timings.append(ms)
                            file.write(json.dumps({"repeat": rep, "sample": 10 + index, "ms": ms}) + "\n")
                report["modes"][mode] = {
                    "mean_ms": float(np.mean(timings)),
                    "p50_ms": float(np.percentile(timings, 50)),
                    "p95_ms": float(np.percentile(timings, 95)),
                    "count": len(timings),
                    "policy_calls_per_s": 1000 / float(np.mean(timings)),
                    "stable_memory": stable,
                    "inference_memory": memory(),
                }
                report["boundary"] = (
                    f"Complete synchronized policy.infer, includingCPU preprocessing,H2D,vision,language,all{a.steps} expert steps,online transforms,quantization,GEMM,D2H,output transforms. NoRPC/simulator/cold capture."
                )
                write(out / "report.json", report)
                print("MATCHED_CONTROL", a.kind, mode, report["modes"][mode], flush=True)
            if modules:
                report["active_layers"] = {
                    n: {"backend": m.backend, "calls": m.calls, "shapes": sorted(m.shapes)}
                    for n, m in modules.items()
                }
                assert all(m.calls > 0 for m in modules.values())
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    policy.infer(samples[0], noise=noise)
                    torch.cuda.synchronize()
                report["actual_cuda_kernel_names"] = dict(
                    Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
                )
            report.update(status="complete", finished=time.time())
            write(out / "report.json", report)
    finally:
        stop.set()
        sampler.join(timeout=25)


if __name__ == "__main__":
    main()
