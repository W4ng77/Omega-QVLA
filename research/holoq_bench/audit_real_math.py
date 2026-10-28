"""Independent accuracy-only audits of actual rotations, native banks and graphs.

Floating diagnostics below verify algebra only; they are never speed baselines.
"""

import argparse
from collections import defaultdict, Counter
import fcntl
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.packed_runtime import PackedLinear, apply_blocks
from gr00t.quantization.perstep_backends import PerStepBackend
from gr00t.quantization.dit_step_context import get_current_dit_step


def error(a, b):
    return float((a.float() - b.float()).square().sum() / b.float().square().sum().clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.root
    manifest = json.loads((root / "teacher_capture/manifest.json").read_text())
    entries = {row["name"]: row for row in manifest["layers"]}
    weights = torch.load(root / "teacher_capture/weights.pt", map_location="cpu", mmap=True, weights_only=False)
    base = torch.load(root / "deployments/holoq/packed.pt", map_location="cpu", mmap=True, weights_only=False)
    converted = torch.load(root / "deployments/holoq_step_weight/packed.pt", map_location="cpu", mmap=True, weights_only=False)
    names = ["backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj",
             "backbone.eagle_model.language_model.model.layers.0.mlp.down_proj",
             "action_head.model.transformer_blocks.0.attn1.to_k",
             "action_head.model.transformer_blocks.0.ff.net.0.proj",
             "action_head.model.transformer_blocks.0.ff.net.2"]
    rotations = []
    native = []
    with torch.inference_mode():
        for name in names:
            rr = base["layers"][name]
            data = torch.load(root / "teacher_capture" / entries[name]["file"], map_location="cpu", weights_only=False)
            x = next(iter(data.values()))[:49].half().cuda()
            w = weights[name]["weight"].float().cuda()
            rin, rout, perm = (rr[k].cuda() for k in ("rot_in", "rot_out", "perm"))
            n, k = w.shape
            wp = w[:, perm]
            wr = torch.bmm(wp.reshape(n, k // 64, 64).transpose(0, 1), rin.float()).transpose(0, 1).reshape(n, k)
            wr = torch.bmm(rout.float(), wr.reshape(n // 64, 64, k)).reshape(n, k).half()
            xr = apply_blocks(x, rin, perm)
            predicted = apply_blocks(torch.nn.functional.linear(xr.float(), wr.float()).half(), rout)
            reference = torch.nn.functional.linear(x.float(), w)
            rotations.append({"name": name, "shape": [n, k], "unquantized_rotation_roundtrip_nmse": error(predicted, reference),
                              "output_rotation_stride": list(rout.stride())})
            assert rotations[-1]["unquantized_rotation_roundtrip_nmse"] < 2e-5
            if name.startswith("action_head"):
                integer = PerStepBackend(converted["layers"][name]["integer"]).cuda()
                assert integer.act_qmin == -7 and integer.act_scale_table.dtype == torch.float32
                pointers = []
                for step in range(8):
                    p, scales, act = integer.selected(step)
                    pointers.append(p.data_ptr())
                    xt = apply_blocks(data[step][:49].half().cuda(), rin, perm).contiguous()
                    actual = integer(xt, step, engine="native_s4")
                    expected = integer.reference(xt, p, scales, act)
                    nmse = error(actual, expected)
                    native.append({"name": name, "step": step, "nmse": nmse,
                                   "shape": list(xt.shape), "scale_dtype": str(scales.dtype)})
                    assert nmse < 1e-8 and torch.isfinite(actual).all()
                assert len(set(pointers)) == 8
                del integer
            del x, w, rin, rout, perm, wr, xr, predicted, reference
        del weights, base, converted
        gc.collect()
        torch.cuda.empty_cache()

    deployment = root / "deployments/holoq_llm_s4_rotv2"
    os.environ.update(HOLOQ_DEPLOY=str(deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16")
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from research.holoq_bench.graphs import install

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=str(deployment.resolve()), modality_config=config.modality_config(),
                         modality_transform=config.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    observations = torch.load(root / "heldout_init10_19/observations.pt", map_location="cpu", weights_only=False)
    samples = [observations[i * 10] for i in range(10)]
    constants = []
    original = policy.model.action_head.get_action

    def record_types(backbone, action):
        constants.append([{key: type(value).__name__ for key, value in dict(tree).items()
                           if not isinstance(value, torch.Tensor)} for tree in (backbone, action)])
        return original(backbone, action)

    policy.model.action_head.get_action = record_types
    eager = []
    for sample in samples:
        seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
        eager.append(policy.get_action(sample["obs"]))
    policy.model.action_head.get_action = original
    assert all(not fields for call in constants for fields in call), constants
    seen = defaultdict(Counter)
    addresses = defaultdict(dict)

    def inspect(name):
        def hook(module, inputs):
            step = get_current_dit_step()
            assert step is not None and 0 <= step < 8
            seen[name][step] += 1
            addresses[name][step] = module.a_scale[step].data_ptr()
        return hook

    handles = [m.register_forward_pre_hook(inspect(name)) for name, m in policy.model.named_modules()
               if isinstance(m, PackedLinear) and getattr(m, "act_mode", None) == "step"]
    assert len(handles) == 96
    graphs = install(policy)
    action_errors = []
    for sample, reference in zip(samples, eager):
        seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
        actual = policy.get_action(sample["obs"])
        num = sum(np.square(actual[k].astype(float) - reference[k].astype(float)).sum() for k in actual)
        den = sum(np.square(reference[k].astype(float)).sum() for k in actual)
        action_errors.append(float(num / max(den, 1e-30)))
    for handle in handles:
        handle.remove()
    assert all(set(count) == set(range(8)) for count in seen.values())
    assert all(len(set(values.values())) == 8 for values in addresses.values())
    assert max(action_errors) < .02
    output = {"purpose": "accuracy only, no floating speed benchmark", "rotations": rotations,
              "native_real_bank_comparisons": native, "real_policy_non_tensor_inputs": constants,
              "current_protocol_uses_tensor_only_graph_inputs": True,
              "per_step_capture_counts": {k: dict(v) for k, v in seen.items()},
              "all_96_DiT_layers_capture_distinct_8_scale_rows": True,
              "graph_count": len(graphs.cache), "graph_vs_eager_action_nmse": action_errors,
              "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("REAL_MATH_AUDIT_PASS", "rotations", len(rotations), "native_step_checks", len(native),
          "max_native_nmse", max(row["nmse"] for row in native), "action_nmse", action_errors, flush=True)


if __name__ == "__main__":
    main()
