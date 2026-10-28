"""Check repeat variance and Graph arithmetic before choosing baseline modes."""

import dataclasses
import fcntl
import os
from pathlib import Path
import numpy as np
import torch
from research.pi05_table3.measure import write, sha


def diff(a, b):
    delta = a.astype(float) - b.astype(float)
    return {
        "nmse": float(np.square(delta).sum() / max(np.square(b.astype(float)).sum(), 1e-30)),
        "max_abs": float(np.abs(delta).max()),
    }


def main():
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    for k in list(os.environ):
        if k.startswith(("GR00T_", "OPENPI_GPTQ", "QVLA_ACT")):
            os.environ.pop(k)
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from gr00t.quantization.pi05_cuda_graph import Pi05CudaGraphSampler
    from research.pi05_table3.baseline_runtime import load_policy, BaselineLinear

    root = Path(__file__).resolve().parents[2]
    data = root / "results/pi05_table3_414_20261028"
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = load_policy(data / "baseline_deployments/gptq_marlin252", cfg, 10)
    policy._model.eval().requires_grad_(False)
    install_pi05_step_context(policy._model, fixed_loop=True)
    policy._sample_actions = policy._model.sample_actions
    samples = torch.load(
        "/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt", map_location="cpu", weights_only=False
    )["samples"][10:20]
    noise = (
        np.random.default_rng(42)
        .standard_normal((cfg.model.action_horizon, cfg.model.action_dim))
        .astype(np.float32)
    )
    result = {
        "source_sha256": sha(Path(__file__)),
        "purpose": "Numerical Graph/repeat diagnostic, not timing or success evidence",
        "limit_nmse": 0.01,
        "eager_repeat": [],
        "graph_eager": [],
    }
    with torch.inference_mode():
        first = [policy.infer(s, noise=noise)["actions"].copy() for s in samples]
        for i, s in enumerate(samples):
            result["eager_repeat"].append(diff(policy.infer(s, noise=noise)["actions"], first[i]))
        policy._sample_actions = Pi05CudaGraphSampler(policy._model.sample_actions)
        for i, s in enumerate(samples):
            result["graph_eager"].append(diff(policy.infer(s, noise=noise)["actions"], first[i]))
        torch.cuda.synchronize()
        result["nonzero_workspace_elements"] = sum(
            int(torch.count_nonzero(m.workspace))
            for m in policy._model.modules()
            if isinstance(m, BaselineLinear) and m.backend == "marlin"
        )
        result["max_eager_repeat_nmse"] = max(x["nmse"] for x in result["eager_repeat"])
        result["max_graph_eager_nmse"] = max(x["nmse"] for x in result["graph_eager"])
        result["passed"] = result["max_graph_eager_nmse"] <= 0.01 and result["nonzero_workspace_elements"] == 0
        write(data / "marlin_graph_diagnostic.json", result)
        print("MARLIN_GRAPH_DIAGNOSTIC", result, flush=True)


if __name__ == "__main__":
    main()
