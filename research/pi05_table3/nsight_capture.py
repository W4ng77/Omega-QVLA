"""Five warmed complete policy calls in the actual sampler-Graph mode."""

import dataclasses
import argparse
import fcntl
import os
from pathlib import Path
import numpy as np
import torch


def wrap(obj, attr, label):
    original = getattr(obj, attr)

    def call(*args, **kwargs):
        torch.cuda.nvtx.range_push(label)
        try:
            return original(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    setattr(obj, attr, call)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--definition8", action="store_true")
    args = parser.parse_args()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    for key in list(os.environ):
        if key.startswith(("GR00T_", "OPENPI_GPTQ", "QVLA_ACT")):
            os.environ.pop(key)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from gr00t.quantization.pi05_cuda_graph import Pi05CudaGraphSampler

    if args.definition8:
        from research.pi05_table3.definition_runtime import load_policy as load_definition

        def load_policy(directory, cfg):
            return load_definition(directory, cfg, "packed")
    else:
        from research.pi05_table3.historical_loader import load_policy

    root = Path(__file__).resolve().parents[2]
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    directory = "definition8_deployment_fp64" if args.definition8 else "holoq_deployments/both"
    policy = load_policy(root / "results/pi05_table3_414_20261028" / directory, cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    policy._model.eval().requires_grad_(False)
    install_pi05_step_context(policy._model, fixed_loop=True)
    policy._sample_actions = Pi05CudaGraphSampler(policy._model.sample_actions)
    sample = torch.load(
        "/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt", map_location="cpu", weights_only=False
    )["samples"][10]
    noise = (
        np.random.default_rng(42)
        .standard_normal((cfg.model.action_horizon, cfg.model.action_dim))
        .astype(np.float32)
    )
    with torch.inference_mode():
        for _ in range(32):
            policy.infer(sample, noise=noise)
        torch.cuda.synchronize()
        wrap(policy, "_input_transform", "policy/preprocess")
        wrap(policy, "_sample_actions", "policy/full_sample_graph")
        wrap(policy, "_output_transform", "policy/postprocess")
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(5):
            torch.cuda.nvtx.range_push("policy/complete_infer")
            policy.infer(sample, noise=noise)
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
