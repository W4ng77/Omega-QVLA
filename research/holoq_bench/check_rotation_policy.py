"""Accuracy-only real-pack regression check; no diagnostic times are reported.

Start from the nonfused packed deployment. Compare ten independent full-policy
actions before/after installing v2 and compare every real projection input with
the original packed operator, including all eight per-step activation scales.
"""

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.packed_runtime import PackedLinear, apply_blocks
from research.holoq_bench.cutlass_s4 import CutlassLinear
from research.holoq_bench.fused_rotation_strided import install
from research.holoq_bench.fused_rotation import rotate as legacy_rotate
from gr00t.quantization.dit_step_context import get_current_dit_step


def nmse(actual, reference):
    a, r = actual.float(), reference.float()
    return float((a - r).square().sum() / r.square().sum().clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(HOLOQ_DEPLOY=str(args.deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16")
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=str(args.deployment.resolve()), modality_config=config.modality_config(),
                         modality_transform=config.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    modules = {n: m for n, m in policy.model.named_modules() if isinstance(m, PackedLinear)}
    assert len(modules) in (116, 180)
    assert all(m.rot_in.numel() for m in modules.values())
    observations = torch.load(args.observations, map_location="cpu", weights_only=False)
    selected = [observations[i * 10] for i in range(10)]
    reference_actions = []
    with torch.inference_mode():
        for sample in selected:
            seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
            reference_actions.append(policy.get_action(sample["obs"]))
    strides = {name: {k: list(getattr(m, k).stride()) for k in ("rot_in", "rot_out")}
               for name, m in modules.items()}
    original_types = {name: CutlassLinear if isinstance(m, CutlassLinear) else PackedLinear
                      for name, m in modules.items()}
    install(policy.model)
    errors = defaultdict(list)
    legacy_proof = []

    def hook(name):
        def check(module, inputs, output):
            # Direct forward bypasses the forward-hook dispatcher. Share the
            # exact original packed buffers; never build a dense float weight.
            fused_type = module.__class__
            try:
                module.__class__ = original_types[name]
                for key in ("rot_in", "rot_out", "bias"):
                    setattr(module, key, getattr(module, {"rot_in": "_fused_in", "rot_out": "_fused_out", "bias": "_fused_bias"}[key]))
                reference = original_types[name].forward(module, inputs[0])
                if not legacy_proof and module.rot_out.numel() and not module.rot_out.is_contiguous():
                    x = reference.half()
                    expected = apply_blocks(x, module.rot_out)
                    legacy_proof.append({"layer": name, "stride": list(module.rot_out.stride()),
                                         "nmse": nmse(legacy_rotate(x, module.rot_out), expected)})
            finally:
                module.__class__ = fused_type
                for key in ("rot_in", "rot_out", "bias"):
                    setattr(module, key, torch.empty(0, device=inputs[0].device))
            errors[name].append({"step": get_current_dit_step(), "nmse": nmse(output, reference),
                                 "finite": bool(torch.isfinite(output).all() and torch.isfinite(reference).all())})
        return check

    handles = [m.register_forward_hook(hook(n)) for n, m in modules.items()]
    action_errors = []
    with torch.inference_mode():
        for sample, reference in zip(selected, reference_actions):
            seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
            actual = policy.get_action(sample["obs"])
            num = sum(np.square(actual[k].astype(float) - reference[k].astype(float)).sum() for k in actual)
            den = sum(np.square(reference[k].astype(float)).sum() for k in actual)
            action_errors.append({"task_id": sample["task_id"], "init_index": sample["init_index"],
                                  "nmse": float(num / max(den, 1e-30)),
                                  "finite": all(np.isfinite(actual[k]).all().item() for k in actual)})
    for handle in handles:
        handle.remove()
    max_layer = max(row["nmse"] for rows in errors.values() for row in rows)
    finite = all(row["finite"] for rows in errors.values() for row in rows) and all(row["finite"] for row in action_errors)
    output = {"purpose": "accuracy-only same-packed-operator oracle; no speed baseline",
              "deployment": str(args.deployment), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "plugin_source_sha256": hashlib.sha256(Path(__file__).with_name("fused_rotation_strided.py").read_bytes()).hexdigest(),
              "source_pack_sha256": json.loads((args.deployment / "deployment.json").read_text())["source_pack_sha256"],
              "matrix_strides": strides, "legacy_bug_proof": legacy_proof,
              "layers": dict(errors), "comparisons": sum(map(len, errors.values())),
              "max_layer_nmse": max_layer, "action_errors": action_errors, "finite": finite,
              "passed": finite and max_layer < 1e-4 and max(row["nmse"] for row in action_errors) < .02}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("ROTATION_REAL_POLICY_CHECK", output["passed"], "max_layer_nmse", max_layer,
          "actions", [r["nmse"] for r in action_errors], "legacy", legacy_proof, flush=True)
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
