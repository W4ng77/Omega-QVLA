#!/usr/bin/env python
"""Compare FP vs QVLA-baked vs uniform-W4 GR00T actions on calib obs."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    ensure_libero_runtime,
    get_named_module,
    load_libero_samples,
    load_policy,
    normalized_input_no_inference,
    seed_everything,
)
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402
from tools.build_qvla_pack import QVLA_INCLUDE, QVLA_EXCLUDE, quant_rows_sym  # noqa: E402
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402


def get_actions(policy, normed):
    outs = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        for s in normed:
            seed_everything(s["seed"])
            a = policy.model.get_action(s["normalized"])
            t = a["action_pred"] if isinstance(a, dict) and "action_pred" in a else a
            if isinstance(t, dict):
                t = next(iter(t.values()))
            outs.append(t.detach().float().cpu())
    return torch.stack([o.flatten() for o in outs])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp-checkpoint", required=True)
    p.add_argument("--qvla-checkpoint", required=True)
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--dataset-path", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task")
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ensure_libero_runtime()
    data_config = load_data_config(args.data_config)

    # FP baseline
    args.checkpoint = args.fp_checkpoint
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    normed = [
        {"seed": s["seed"], "normalized": normalized_input_no_inference(policy, s["obs"])}
        for s in samples
    ]
    a_fp = get_actions(policy, normed)
    print("FP action[0][:8]:", np.round(a_fp[0][:8].numpy(), 4))

    # uniform W4 on same scope (bake in-place on the FP model)
    targets = select_targets(policy.model, include_regex=QVLA_INCLUDE, exclude_regex=QVLA_EXCLUDE,
                             scope_prefix=None, whitelist=None, blacklist=None)
    for lname, _ in targets:
        m = get_named_module(policy.model, lname)
        m.weight.data = quant_rows_sym(m.weight.data.float(), 4).to(m.weight.dtype)
    a_u4 = get_actions(policy, normed)
    print("uniform-W4 mean|d|:", (a_u4 - a_fp).abs().mean().item())
    del policy
    torch.cuda.empty_cache()

    # QVLA-baked
    args.checkpoint = args.qvla_checkpoint
    policy_q = load_policy(args, data_config, quantized_layers=None)
    policy_q.model.eval()
    a_q = get_actions(policy_q, normed)
    print("qvla-W4   mean|d|:", (a_q - a_fp).abs().mean().item())
    print("qvla action[0][:8]:", np.round(a_q[0][:8].numpy(), 4))

    # QVLA-baked + A4 hooks
    import os
    os.environ["QVLA_ACT_BITS"] = "4"
    from gr00t.quantization.qvla_act import enable_qvla_act_if_configured
    enable_qvla_act_if_configured(policy_q.model)
    a_qa4 = get_actions(policy_q, normed)
    print("qvla-W4A4 mean|d|:", (a_qa4 - a_fp).abs().mean().item())


if __name__ == "__main__":
    main()
