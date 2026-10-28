"""Every real QServe target call vs independent INT8/INT4 integer oracle."""

import dataclasses
import fcntl
import json
from pathlib import Path
import numpy as np
import torch
from research.pi05_table3.measure import write, sha


def main():
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from research.pi05_table3.baseline_runtime import load_policy, BaselineLinear

    root = Path(__file__).resolve().parents[2]
    data = root / "results/pi05_table3_414_20261028"
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    deployment = data / "baseline_deployments/smoothquant_qserve414"
    policy = load_policy(deployment, cfg, 8)
    model = policy._model.eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    install_pi05_step_context(model, fixed_loop=True)
    policy._sample_actions = model.sample_actions
    manifest = json.loads((data / "baseline_capture/manifest.json").read_text())
    cache = {}
    for item in manifest["layers"]:
        e = torch.load(
            data / "baseline_build/smoothquant_qserve414/layers" / f"{item['index']:03d}.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        cache[item["name"]] = (e["canonical_codes"].cuda(), e["canonical_scales"].cuda())
    rows = []

    def hook(module, args, actual):
        q, s = cache[module.name]
        x = (args[0].float() / module.smooth[: module.in_features]).half().float()
        sa = (x.abs().amax(-1, keepdim=True).clamp_min(1e-5) / 127).half().float()
        qa = (x / sa).round().clamp(-128, 127).reshape(-1, module.in_features)
        expected = (
            ((qa @ q.float().t()) * sa.reshape(-1, 1) * s.float().t())
            .half()
            .to(module.external_dtype)
            .reshape(actual.shape)
        )
        if module.bias.numel():
            expected = expected + module.bias
        diff = actual.float() - expected.float()
        rows.append(
            {
                "layer": module.name,
                "shape": list(args[0].shape),
                "nmse": float(diff.square().sum() / expected.float().square().sum().clamp_min(1e-30)),
                "max_abs": float(diff.abs().max()),
            }
        )

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, BaselineLinear)]
    sample = torch.load(
        "/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt", map_location="cpu", weights_only=False
    )["samples"][10]
    noise = (
        np.random.default_rng(42)
        .standard_normal((cfg.model.action_horizon, cfg.model.action_dim))
        .astype(np.float32)
    )
    with torch.inference_mode():
        policy.infer(sample, noise=noise)
        torch.cuda.synchronize()
    for h in handles:
        h.remove()
    result = {
        "source_sha256": sha(Path(__file__)),
        "runtime_sha256": sha(root / "research/pi05_table3/baseline_runtime.py"),
        "deployment_manifest_sha256": sha(deployment / "deployment.json"),
        "backend": "official_qserve_INT8_INT32",
        "steps": 8,
        "count": len(rows),
        "max_nmse": max(x["nmse"] for x in rows),
        "limit_nmse": 1e-4,
        "rows": rows,
        "note": "Independent arithmetic correctness only; no task-success inference. Dense floating oracle never timed as a quantized baseline.",
    }
    result["passed"] = len(rows) == 486 + 126 + 126 * 8 and result["max_nmse"] < 1e-4
    write(data / "qserve_full_policy_oracle.json", result)
    assert result["passed"], (result["count"], result["max_nmse"])
    print("QSERVE_FULL_ORACLE", result["count"], result["max_nmse"], flush=True)


if __name__ == "__main__":
    main()
