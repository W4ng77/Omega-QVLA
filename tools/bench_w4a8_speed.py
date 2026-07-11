#!/usr/bin/env python
"""Benchmark GR00T N1.5 policy latency: FP16 vs Omega-QVLA W4A8 (GptqLinear).

Measures end-to-end get_action latency on real LIBERO calib obs, plus a
per-op profiler breakdown of the W4A8 path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    ensure_libero_runtime,
    load_libero_samples,
    load_policy,
    normalized_input_no_inference,
    seed_everything,
)
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402


LLM_RE = r"backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
DIT_MLP_RE = r"action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)"
EXCLUDE_RE = r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)"


def build(args, mode: str):
    # hermetic: clear all quant env, then set per mode
    for k in list(os.environ):
        if k.startswith(("GR00T_GPTQ", "GR00T_DUQUANT", "GR00T_RTN", "GR00T_ATM",
                         "GR00T_OHB", "GR00T_HYBRID", "QVLA_ACT")):
            os.environ.pop(k)
    ckpt = args.checkpoint

    if mode.startswith("w4a8"):  # Omega-QVLA GPTQ pack runtime
        os.environ["GR00T_GPTQ"] = "1"
        os.environ["GR00T_GPTQ_PATH"] = args.pack
        os.environ["GR00T_GPTQ_CALIB_STEPS"] = "6"
        os.environ["GR00T_GPTQ_ABITS"] = str(args.abits)
        os.environ["GR00T_GPTQ_INCLUDE"] = args.include
        os.environ["GR00T_GPTQ_MISSING"] = "fallback"
        if "fused" in mode:
            os.environ["GR00T_GPTQ_FUSED"] = "1"
        if "qserve" in mode or "fold" in mode:
            os.environ["GR00T_GPTQ_QSERVE"] = "1"
        if "fp8" in mode or "fold" in mode:
            os.environ["GR00T_GPTQ_FP8"] = "1"
        if "fold" in mode:
            os.environ["GR00T_GPTQ_FOLD_ROT"] = "1"
        if "a16" in mode:
            os.environ["GR00T_GPTQ_A16_LARGE_M"] = "1"
    elif mode.startswith("qvla"):  # QVLA: baked fake-quant W + act hooks
        ckpt = args.qvla_checkpoint
        os.environ["QVLA_ACT_BITS"] = str(args.abits)
    elif mode.startswith("quantvla"):  # runtime DuQuant + ATM/OHB (preset)
        os.environ["GR00T_HYBRID_QUANT"] = "1"
        os.environ["GR00T_DUQUANT_INCLUDE"] = f".*({LLM_RE}|{DIT_MLP_RE}).*"
        os.environ["GR00T_DUQUANT_EXCLUDE"] = EXCLUDE_RE
        os.environ["GR00T_DUQUANT_BLOCK"] = "64"
        os.environ["GR00T_DUQUANT_BLOCK_OUT"] = "64"
        os.environ["GR00T_DUQUANT_PERMUTE"] = "0"
        os.environ["GR00T_DUQUANT_ROW_ROT"] = "restore"
        os.environ["GR00T_DUQUANT_ACT_PCT"] = "99.9"
        os.environ["GR00T_DUQUANT_CALIB_STEPS"] = "2"
        os.environ["GR00T_DUQUANT_LS"] = "0.15"
        os.environ["GR00T_DUQUANT_PERM_SCORE"] = "weight"
        os.environ["GR00T_DUQUANT_ROT_MODE"] = "svd_hadamard"
        os.environ["GR00T_DUQUANT_ABITS"] = str(args.abits)
        os.environ["GR00T_ATM_ENABLE"] = "1"
        os.environ["GR00T_ATM_SCOPE"] = "dit"
        os.environ["GR00T_ATM_ALPHA_PATH"] = str(REPO_ROOT / "atm_alpha_beta_spatial.json")
        os.environ["GR00T_OHB_ENABLE"] = "1"
        os.environ["GR00T_OHB_FALLBACK"] = "1.0"
        os.environ["GR00T_OHB_SCOPE"] = "dit"

    if "graph" in mode and "fast" not in mode and "fold" not in mode:
        os.environ["GR00T_DIT_CUDA_GRAPH"] = "1"
        os.environ["GR00T_BACKBONE_CUDA_GRAPH"] = "1"
    else:
        # for fast/fold modes the graph is enabled AFTER freeze (see main loop)
        os.environ.pop("GR00T_DIT_CUDA_GRAPH", None)
        os.environ.pop("GR00T_BACKBONE_CUDA_GRAPH", None)

    data_config = load_data_config(args.data_config)
    if mode.startswith("quantvla"):
        # load_policy() strips GR00T_DUQUANT_/ATM_/OHB_ env during load
        # (isolated_quant_env) — build the policy directly instead.
        from gr00t.model.policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        policy = Gr00tPolicy(
            model_path=ckpt,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            denoising_steps=args.denoising_steps,
            device=args.device,
        )
    else:
        saved = args.checkpoint
        args.checkpoint = ckpt
        policy = load_policy(args, data_config, quantized_layers=None)
        args.checkpoint = saved
    policy.model.eval()
    return policy, data_config


def timed(policy, normed, n_warm=3, n_iter=15):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        for s in normed[:n_warm]:
            seed_everything(s["seed"])
            policy.model.get_action(s["normalized"])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        k = 0
        for i in range(n_iter):
            s = normed[i % len(normed)]
            seed_everything(s["seed"])
            policy.model.get_action(s["normalized"])
            k += 1
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / k
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/work/sicheng/ckpts/gr00t-n1.5-libero-spatial-posttrain")
    p.add_argument("--pack", default="/work/sicheng/models/omega-qvla-gr00t-w4a4/gr00t_spatial/quantized.pt")
    p.add_argument("--abits", type=int, default=8)
    p.add_argument("--include", default=(
        r".*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"
        r"|.*action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*"
    ))
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--dataset-path", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task")
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="all")
    p.add_argument(
        "--qvla-checkpoint",
        default="/work/sicheng/models/qvla-groot-w4/spatial",
        help="QVLA baked fake-quant checkpoint (for qvla_* modes)",
    )
    p.add_argument("--fused", action="store_true")
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()

    ensure_libero_runtime()
    args.checkpoint = args.checkpoint  # load_policy reads args.checkpoint

    results = {}
    actions = {}
    modes = (
        ["fp16", "fp16_graph", "w4a8", "w4a8_graph",
         "qvla_w4a8", "qvla_w4a8_graph", "quantvla_w4a8", "quantvla_w4a8_graph"]
        if args.mode == "all"
        else args.mode.split(",")
    )
    normed_cache = None
    for mode in modes:
        policy, data_config = build(args, mode)
        if normed_cache is None:
            _, _, samples = load_libero_samples(args, data_config)
            samples = samples[: args.num_samples]
            normed_cache = [
                {"seed": s["seed"], "normalized": normalized_input_no_inference(policy, s["obs"])}
                for s in samples
            ]
        if ("fast" in mode or "fold" in mode) and mode.startswith("w4a8"):
            # calibrate on the slow path, freeze, then (optionally) arm the
            # CUDA graph so it captures the frozen fast path
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                for s in normed_cache or []:
                    seed_everything(s["seed"])
                    policy.model.get_action(s["normalized"])
            if normed_cache is not None:
                from gr00t.quantization.gptq_layers import freeze_gptq_fast_paths

                freeze_gptq_fast_paths(policy.model)
                if "graph" in mode:
                    os.environ["GR00T_DIT_CUDA_GRAPH"] = "1"
                    os.environ["GR00T_BACKBONE_CUDA_GRAPH"] = "1"
        # record actions for numerics comparison
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            outs = []
            for s in normed_cache:
                seed_everything(s["seed"])
                a = policy.model.get_action(s["normalized"])
                t = None
                if torch.is_tensor(a):
                    t = a
                elif hasattr(a, "values"):
                    for v in a.values():
                        if torch.is_tensor(v):
                            t = v
                            break
                assert t is not None, f"no tensor in get_action output: {type(a)}"
                outs.append(t.detach().float().cpu().flatten())
            actions[mode] = torch.stack(outs)
        dt = timed(policy, normed_cache)
        results[mode] = dt
        print(f"[bench] {mode}: {dt*1000:.1f} ms / get_action")

        if args.profile and mode.startswith("w4a8"):
            from torch.profiler import profile, ProfilerActivity
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    s = normed_cache[0]
                    seed_everything(s["seed"])
                    policy.model.get_action(s["normalized"])
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
        del policy
        torch.cuda.empty_cache()

    if len(results) > 1 and "fp16" in results:
        base = results["fp16"]
        for m, dt in results.items():
            if m != "fp16":
                print(f"[bench] {m} vs fp16: {dt/base:.2f}x slower")
    if "w4a8" in actions and "w4a8_fused" in actions:
        d = (actions["w4a8"] - actions["w4a8_fused"]).abs()
        print(f"[bench] fused vs eager action diff: max={d.max():.5f} mean={d.mean():.6f}")
    if "fp16" in actions and "w4a8" in actions:
        d = (actions["fp16"] - actions["w4a8"]).abs()
        print(f"[bench] w4a8 vs fp16 action diff: max={d.max():.5f} mean={d.mean():.6f}")


if __name__ == "__main__":
    main()
