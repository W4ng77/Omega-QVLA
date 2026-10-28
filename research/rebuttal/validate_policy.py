"""Accuracy-only same-frozen-grid layer/step/action gates BEFORE scored rollout.

Sequential reference/packed execution of the very same loaded model; no second
checkpoint and no quantizer rebuild. This run is never a timing comparison.
"""
import argparse
import fcntl
import os
from pathlib import Path

import numpy as np
import torch

from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.deployment import sha256
from research.rebuttal.build import json_write, tensor_write


def difference(actual, reference):
    x, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    assert x.shape == r.shape
    assert np.isfinite(x).all() and np.isfinite(r).all()
    return {"nmse": float(np.square(x - r).sum() / max(np.square(r).sum(), 1e-30)),
            "max_abs": float(np.abs(x - r).max()), "rms": float(np.sqrt(np.square(x - r).mean()))}


def flat(value):
    return np.concatenate([np.asarray(value[k]).reshape(-1) for k in sorted(value)])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    a.output.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith(("GR00T_GPTQ", "GR00T_DUQUANT", "GR00T_RTN", "GR00T_ATM", "GR00T_OHB", "GR00T_DIT_CORRECTION", "HOLOQ_PACK", "HOLOQ_DEPLOY")):
            os.environ.pop(key)
    os.environ.update(HOLOQ_DEPLOY=str(a.deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16", HOLOQ_REBUTTAL_IMPLEMENTATION="reference")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from gr00t.quantization.dit_step_context import get_current_dit_step
    from gr00t.quantization.perstep_backends import unpack_int4
    from research.rebuttal.method import equation_oracle
    from research.rebuttal.runtime import PaperLinear
    from research.holoq_bench.common_logits import install as logits
    from research.holoq_bench.graphs import install as install_graphs

    c = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=str(a.deployment.resolve()), modality_config=c.modality_config(), modality_transform=c.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    logits(policy)
    modules = {name: m for name, m in policy.model.named_modules() if isinstance(m, PaperLinear)}
    assert len(modules) == 180
    samples = torch.load(a.observations, map_location="cpu", weights_only=False)[::10]
    assert len(samples) == 10 and len({s["task_id"] for s in samples}) == 10
    head = policy.model.action_head
    target_noise_shape = (1, head.config.action_horizon, head.config.action_dim)
    original_randn = torch.randn
    draws, step_outputs, final_euler = [], [], []

    def randn(*args, **kwargs):
        result = original_randn(*args, **kwargs)
        if tuple(result.shape) == target_noise_shape and not torch.cuda.is_current_stream_capturing():
            draws.append(result.detach().cpu().clone())
        return result

    def step_hook(module, inputs, output):
        step_outputs.append({"step": get_current_dit_step(), "output": output.detach().cpu().clone()})

    def head_hook(module, inputs, output):
        final_euler.append(output["action_pred"].detach().cpu().clone())

    # get_action is a direct method, so an nn.Module forward hook on head will
    # not run. Wrap that method explicitly to record final Euler state.
    original_head = head.get_action

    def recorded_head(*args, **kwargs):
        result = original_head(*args, **kwargs)
        head_hook(head, args, result)
        return result

    head.get_action = recorded_head
    step_handle = head.model.register_forward_hook(step_hook)
    torch.randn = randn
    reference = []
    try:
        with torch.inference_mode():
            for sample in samples:
                draws.clear()
                step_outputs.clear()
                final_euler.clear()
                seed(sample["seed"])
                action = policy.get_action(sample["obs"])
                assert len(draws) == 1 and len(step_outputs) == 8 and len(final_euler) == 1
                assert [s["step"] for s in step_outputs] == list(range(8))
                reference.append({"task_id": sample["task_id"], "action": action, "steps": list(step_outputs), "noise": draws[0], "final_euler": final_euler[0]})
                print("REFERENCE_FIXED_NOISE", sample["task_id"], flush=True)
    finally:
        torch.randn = original_randn
    tensor_write(a.output / "reference_fixed_inputs.pt", reference)
    for m in modules.values():
        m.implementation = "packed"
        m.backend = "paper_packed_w4_fp16_mma"
        del m.reference_codes
    torch.cuda.empty_cache()
    local_rows = []
    saved_local = {}

    def local_before(name):
        def hook(m, inputs):
            q, sa = m.prepare(inputs[0])
            expected = equation_oracle(q, unpack_int4(m.B, m.in_features), sa, m.s, m.bias).half()
            saved_local[name] = (expected, get_current_dit_step())
        return hook

    def local_after(name):
        def hook(m, inputs, output):
            expected, step = saved_local.pop(name)
            actual = output.reshape(-1, m.out_features).float()
            expected = expected.float()
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), (name, step)
            local_rows.append({"name": name, "step": step, "input_shape": list(inputs[0].shape),
                               "nmse": float((actual - expected).square().sum() / expected.square().sum().clamp_min(1e-30)),
                               "max_abs": float((actual - expected).abs().max())})
        return hook

    handles = [h for name, m in modules.items() for h in (m.register_forward_pre_hook(local_before(name)), m.register_forward_hook(local_after(name)))]
    actions, packed_cases = [], []
    torch.randn = randn
    try:
        with torch.inference_mode():
            for i, sample in enumerate(samples):
                draws.clear()
                step_outputs.clear()
                final_euler.clear()
                seed(sample["seed"])
                action = policy.get_action(sample["obs"])
                ref = reference[i]
                assert len(draws) == 1 and torch.equal(draws[0], ref["noise"]), "Eager reference/packed noise mismatch"
                assert [s["step"] for s in step_outputs] == list(range(8))
                actions.append(action)
                packed_cases.append({"task_id": sample["task_id"], "noise_bitwise_equal": True,
                                     "action": difference(flat(action), flat(ref["action"])),
                                     "final_euler": difference(final_euler[0], ref["final_euler"]),
                                     "steps": [{"step": j, **difference(step_outputs[j]["output"], ref["steps"][j]["output"])} for j in range(8)]})
                print("PACKED_FIXED_NOISE", packed_cases[-1], flush=True)
    finally:
        torch.randn = original_randn
        for h in handles:
            h.remove()
        step_handle.remove()
        head.get_action = original_head
    assert len(local_rows) == 8520, len(local_rows)
    assert not saved_local
    graph_cases = []
    graphs = install_graphs(policy)
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            seed(sample["seed"])
            output = policy.get_action(sample["obs"])
            graph_cases.append({"task_id": sample["task_id"], "action": difference(flat(output), flat(actions[i]))})
    limits = {"max_local_nmse": 1e-4, "max_final_action_nmse": .01, "max_graph_eager_action_nmse": 1e-5}
    # Engineering numerical gates, NOT a statistical task-success criterion.
    measured = {"max_local_nmse": max(x["nmse"] for x in local_rows), "max_final_action_nmse": max(x["action"]["nmse"] for x in packed_cases), "max_graph_eager_action_nmse": max(x["action"]["nmse"] for x in graph_cases)}
    result = {"purpose": "Correctness-only; no timing claim, no success-rate claim", "pack_sha256": sha256(a.deployment / "packed.pt"),
              "observation_sha256": sha256(a.observations), "source_sha256": sha256(Path(__file__)),
              "limits_fixed_before_run": limits, "measured": measured, "passed": all(measured[k] <= v for k, v in limits.items()),
              "local_checks": len(local_rows), "cases": packed_cases, "graph_cases": graph_cases, "graphs": len(graphs.cache)}
    json_write(a.output / "layer_checks.json", local_rows)
    json_write(a.output / "summary.json", result)
    print("FROZEN_GRID_POLICY_GATE", result, flush=True)
    assert result["passed"], "Gate failed; preserve evidence and diagnose before scored rollout"


if __name__ == "__main__":
    main()
