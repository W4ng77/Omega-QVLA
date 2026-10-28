"""Teacher inputs for deployment baseline construction, no scored episodes.

Uses the available10 initial Object observations, not10 full trajectories.
This is disclosed deployment reconstruction, not paper calibration replication.
"""

import dataclasses
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.pi05_table3.measure import sha, write


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "results/pi05_table3_414_20261028/baseline_capture"
    out.mkdir(parents=True, exist_ok=True)
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    for key in list(os.environ):
        if key.startswith(("GR00T_", "OPENPI_GPTQ", "QVLA_ACT")):
            os.environ.pop(key)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    from openpi.policies import policy_config
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from gr00t.quantization.dit_step_context import get_current_dit_step

    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    checkpoint = Path("/home/xinyu/models/pi05_libero_pytorch")
    policy = policy_config.create_trained_policy(
        cfg, checkpoint, pytorch_device="cuda", sample_kwargs={"num_steps": 10}
    )
    policy._model.eval().requires_grad_(False)
    install_pi05_step_context(policy._model, fixed_loop=True)
    policy._sample_actions = policy._model.sample_actions
    source = Path("/home/xinyu/models/omega-qvla-pi05-414-w4a4-int4")
    m = json.loads((source / "manifest.json").read_text())
    v = torch.load(source / "vision_int4.pt", map_location="cpu", mmap=True, weights_only=False)
    expected = {n[:-7] for n in set(m["lm_shapes"]) | set(v)}
    modules = {n: x for n, x in policy._model.named_modules() if n in expected}
    assert len(modules) == 414 and all(isinstance(x, torch.nn.Linear) for x in modules.values())
    data_path = Path("/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt")
    observations = torch.load(data_path, map_location="cpu", weights_only=False)["samples"][10:20]
    noise = (
        np.random.default_rng(42)
        .standard_normal((cfg.model.action_horizon, cfg.model.action_dim))
        .astype(np.float32)
    )
    current = {}

    def hook(index):
        def record(module, inputs):
            bank = get_current_dit_step()
            bank = -1 if bank is None else bank
            current.setdefault(index, {}).setdefault(bank, []).append(
                inputs[0].detach().reshape(-1, module.in_features).cpu()
            )

        return record

    handles = [mod.register_forward_pre_hook(hook(i)) for i, mod in enumerate(modules.values())]
    manifest = {
        "checkpoint": str(checkpoint),
        "source_observations": str(data_path),
        "source_observation_sha256": sha(data_path),
        "calibration": "10 available initial Object observations, one per task, ALL actual input tokens. Not10 trajectories. No success-driven data selection.",
        "steps": 10,
        "seed": 42,
        "noise": "fixed explicit NumPy standard_normal action noise",
        "teacher_dtype": "OpenPI default BF16 with original sensitiveFP32 parameters",
        "source_sha256": sha(Path(__file__)),
        "layers": [
            {
                "name": n,
                "index": i,
                "K": mod.in_features,
                "N": mod.out_features,
                "weight_dtype": str(mod.weight.dtype),
            }
            for i, (n, mod) in enumerate(modules.items())
        ],
    }
    write(out / "configuration.json", manifest)
    try:
        with torch.inference_mode():
            for frame, obs in enumerate(observations):
                current.clear()
                folder = out / f"{frame:02d}"
                folder.mkdir(exist_ok=True)
                assert not (folder / "complete.json").exists(), (
                    "Do not mix existing captures; use a frozen completed manifest or a new output"
                )
                output = policy.infer(obs, noise=noise)
                assert np.isfinite(output["actions"]).all() and len(current) == 414
                entries = []
                for i, item in enumerate(manifest["layers"]):
                    values = {b: torch.cat(parts) for b, parts in current[i].items()}
                    path = folder / f"{i:03d}.pt"
                    torch.save(values, path)
                    entries.append(
                        {"index": i, "rows": {str(k): len(x) for k, x in values.items()}, "sha256": sha(path)}
                    )
                write(
                    folder / "complete.json", {"frame": frame, "observation_index": 10 + frame, "layers": entries}
                )
                print("BASELINE_TEACHER_CAPTURE", frame + 1, "/10", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    write(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
