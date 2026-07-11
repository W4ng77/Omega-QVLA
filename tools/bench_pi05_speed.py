#!/usr/bin/env python
"""Benchmark pi0.5 (openpi PyTorch) policy latency: FP16 vs Omega-QVLA packs.

Runs policy.infer on recorded LIBERO obs (long-suite prompts by default).
Modes: fp16 | w4a8 | w4a4 (GPTQ pack runtime), each optionally +compile.
Run inside the openpi venv with PYTHONPATH=<Omega-QVLA repo>.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

INCLUDE_BOTH = (
    r".*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)"
    r"\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"
)
# QuantVLA paper layout (arXiv 2602.20309): LM all linears + expert MLP only
INCLUDE_QUANTVLA_PAPER = (
    r".*paligemma_with_expert\.("
    r"paligemma\.model\.language_model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|gemma_expert\.model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)"
    r").*"
)


def clear_env():
    for k in list(os.environ):
        if k.startswith(("GR00T_GPTQ", "GR00T_DUQUANT", "GR00T_RTN", "QVLA_ACT")):
            os.environ.pop(k)


def strip_pack(src: str, drop_prefixes: tuple, tag: str) -> str:
    """Create a copy of the pack with record keys dropped (speed ablation only —
    weights stay in the rotated basis, so outputs are NOT numerically valid)."""
    out = Path(os.environ.get("TMPDIR", "/tmp")) / f"pi05_pack_{tag}.pt"
    if out.exists():
        return str(out)
    pack = torch.load(src, weights_only=False, map_location="cpu")
    n = 0
    for rec in pack.values():
        if not isinstance(rec, dict):
            continue
        for k in list(rec.keys()):
            if k.startswith(drop_prefixes):
                rec.pop(k)
                n += 1
    torch.save(pack, out)
    print(f"[bench-pi05] stripped {n} entries ({tag}) -> {out}")
    return str(out)


def build(args, mode: str):
    clear_env()
    abits = 4 if "a4" in mode else 8
    if mode.endswith("_real") or "_real_" in mode:
        # unified real-quant deployment: baked method weights + QServe small-M
        # + A16 large-M. Method identity lives in the checkpoint weights only.
        os.environ["GR00T_GPTQ"] = "1"
        os.environ["GR00T_GPTQ_PATH"] = args.empty_pack
        os.environ["GR00T_GPTQ_INCLUDE"] = (
            INCLUDE_QUANTVLA_PAPER if "paper" in mode else INCLUDE_BOTH
        )
        os.environ["GR00T_GPTQ_ABITS"] = "8"
        os.environ["GR00T_GPTQ_CALIB_STEPS"] = "4"
        os.environ["GR00T_GPTQ_MISSING"] = "fallback"
        os.environ["GR00T_GPTQ_QSERVE"] = "1"
        os.environ["GR00T_GPTQ_A16_LARGE_M"] = "1"
    elif mode.startswith("qvla"):
        # QVLA (SJTU) method: fake-quant weights baked offline into the
        # checkpoint + uniform per-token A8 hooks. No fast kernels.
        os.environ["QVLA_ACT_BITS"] = "8"
        os.environ["QVLA_ACT_INCLUDE"] = INCLUDE_BOTH
    elif mode.startswith("quantvla"):
        # QuantVLA preset: runtime DuQuant (block rotation + RTN). No packs.
        os.environ["GR00T_DUQUANT_INCLUDE"] = INCLUDE_BOTH
        os.environ["GR00T_DUQUANT_BLOCK"] = "64"
        os.environ["GR00T_DUQUANT_BLOCK_OUT"] = "64"
        os.environ["GR00T_DUQUANT_PERMUTE"] = "0"
        os.environ["GR00T_DUQUANT_ROT_MODE"] = "svd_hadamard"
        os.environ["GR00T_DUQUANT_ABITS"] = "8"
        os.environ["GR00T_DUQUANT_CALIB_STEPS"] = "4"
    elif mode.startswith("rtn"):
        os.environ["GR00T_RTN"] = "1"
        os.environ["GR00T_RTN_INCLUDE"] = INCLUDE_BOTH
        os.environ["GR00T_RTN_WBITS"] = "4"
        os.environ["GR00T_RTN_ABITS"] = str(abits)
        os.environ["GR00T_RTN_GROUP_SIZE"] = "64"
    elif mode.startswith(("w4a8", "w4a4", "pack")):
        pack = args.pack
        if "norot" in mode and "nostep" in mode or "bare" in mode:
            pack = strip_pack(pack, ("duquant_rotation", "act_scale_table"), "bare")
        elif "norot" in mode:
            pack = strip_pack(pack, ("duquant_rotation",), "norot")
        elif "nostep" in mode:
            pack = strip_pack(pack, ("act_scale_table",), "nostep")
        os.environ["GR00T_GPTQ"] = "1"
        os.environ["GR00T_GPTQ_PATH"] = pack
        os.environ["GR00T_GPTQ_INCLUDE"] = INCLUDE_BOTH
        os.environ["GR00T_GPTQ_ABITS"] = str(abits)
        os.environ["GR00T_GPTQ_CALIB_STEPS"] = "4"
        os.environ["GR00T_GPTQ_MISSING"] = "fallback"
        os.environ["GR00T_GPTQ_WBITS_DEFAULT"] = "4"
        if "qserve" in mode:
            os.environ["GR00T_GPTQ_QSERVE"] = "1"
        if "fp8" in mode:
            os.environ["GR00T_GPTQ_FP8"] = "1"
        if "fold" in mode:
            os.environ["GR00T_GPTQ_FOLD_ROT"] = "1"
            os.environ["GR00T_GPTQ_QSERVE"] = "1"
            if "a16" in mode:
                os.environ["GR00T_GPTQ_A16_LARGE_M"] = "1"
            else:
                os.environ["GR00T_GPTQ_FP8"] = "1"

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    if mode.startswith("qvla_real"):
        ckpt = args.qvla_checkpoint
    elif mode.startswith("quantvla_real"):
        ckpt = (
            args.quantvla_paper_checkpoint
            if "paper" in mode
            else args.quantvla_checkpoint
        )
    elif mode.startswith("qvla"):
        ckpt = args.qvla_checkpoint
    else:
        ckpt = args.checkpoint
    train_cfg = _config.get_config(args.data_config)
    policy = _policy_config.create_trained_policy(
        train_cfg, ckpt, pytorch_device=args.device
    )
    assert getattr(policy, "_is_pytorch_model", False)

    if mode.startswith(("w4a8", "w4a4", "pack", "rtn", "fast", "qvla", "quantvla")) or "_real" in mode:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from openpi_inference_service import _apply_quantization_if_requested

        _apply_quantization_if_requested(policy)
    policy._model.eval()
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/work/sicheng/models/pi05_libero_pytorch")
    p.add_argument("--pack", default="/work/sicheng/models/omega-qvla-pi05-w4a4/pi05_long/quantized.pt")
    p.add_argument("--obs-path", default="/work/sicheng/vla-quant/out/pi05_calib_obs.pt")
    p.add_argument("--qvla-checkpoint", default="/work/sicheng/models/qvla-pi05-w4")
    p.add_argument("--quantvla-checkpoint", default="/work/sicheng/models/quantvla-pi05-w4")
    p.add_argument("--quantvla-paper-checkpoint", default="/work/sicheng/models/quantvla-pi05-w4-paper")
    p.add_argument("--empty-pack", default="/work/sicheng/vla-quant/out/empty_pack.pt")
    p.add_argument("--obs-slice", default="30:40", help="python slice into obs list (long suite = 30:40)")
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", default="fp16,w4a8")
    p.add_argument("--n-iter", type=int, default=12)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--phase", action="store_true", help="split prefill vs denoise-loop time")
    args = p.parse_args()

    data = torch.load(args.obs_path, weights_only=False)
    samples = data["samples"] if isinstance(data, dict) else data
    lo, hi = (int(x) for x in args.obs_slice.split(":"))
    samples = samples[lo:hi]
    print(f"[bench-pi05] {len(samples)} obs (slice {args.obs_slice})")

    results = {}
    for mode in args.mode.split(","):
        policy = build(args, mode)
        s0 = samples[0]  # single prompt: matches per-episode reality, avoids
        # per-shape dynamo recompiles dominating the measurement
        with torch.no_grad():
            if "fast" in mode or "fold" in mode or "_real" in mode:
                # calibrate eagerly (no pointless compile of the slow path),
                # freeze, then compile only the frozen fast path
                torch.compiler.set_stance("force_eager")
                for s in samples[:4]:
                    policy.infer(s)
                from gr00t.quantization.gptq_layers import freeze_gptq_fast_paths

                freeze_gptq_fast_paths(policy._model)
                torch.compiler.set_stance("default")
            for _ in range(4):
                policy.infer(s0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            k = 0
            for _ in range(args.n_iter):
                policy.infer(s0)
                k += 1
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / k
        results[mode] = dt
        print(f"[bench-pi05] {mode}: {dt*1000:.1f} ms / infer", flush=True)

        if args.phase:
            # phase split via step-count extrapolation: T(n) = prefill + n * per_step
            with torch.no_grad():
                policy._sample_kwargs["num_steps"] = 2
                for _ in range(3):
                    policy.infer(s0)  # recompile warmup at 2 steps
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(8):
                    policy.infer(s0)
                torch.cuda.synchronize()
                t2 = (time.perf_counter() - t0) / 8 * 1000
                policy._sample_kwargs.pop("num_steps", None)
            per_step = (dt * 1000 - t2) / 8.0
            denoise10 = per_step * 10
            print(
                f"[bench-pi05][{mode}] denoise(10 steps): {denoise10:.1f} ms "
                f"({per_step:.2f}/step) | prefill+rest: {dt*1000 - denoise10:.1f} ms",
                flush=True,
            )

        if args.profile:
            from torch.profiler import profile, ProfilerActivity

            with torch.no_grad():
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    policy.infer(samples[0])
            ka = prof.key_averages()
            print(f"===== profile top ops ({mode}) =====", flush=True)
            print(ka.table(sort_by="cuda_time_total", row_limit=14), flush=True)
            cats = {"matmul/linear": 0.0, "copy/cast": 0.0, "quant(round/clamp/div/mul)": 0.0, "other": 0.0}
            total = 0.0
            for ev in ka:
                t = ev.self_device_time_total
                total += t
                n = ev.key
                if any(s in n for s in ("mm", "linear", "bmm", "matmul", "gemm", "addmm")):
                    cats["matmul/linear"] += t
                elif any(s in n for s in ("copy", "_to_copy", "to(")):
                    cats["copy/cast"] += t
                elif any(s in n for s in ("round", "clamp", "div", "mul", "quantile", "amax", "abs")):
                    cats["quant(round/clamp/div/mul)"] += t
                else:
                    cats["other"] += t
            for k, v in cats.items():
                print(f"[bench-pi05][{mode}] {k}: {v/1000:.1f} ms ({100*v/max(total,1):.0f}%)", flush=True)
        del policy
        torch.cuda.empty_cache()

    if "fp16" in results:
        for m, dt in results.items():
            if m != "fp16":
                print(f"[bench-pi05] {m} vs fp16: {dt/results['fp16']:.2f}x")


if __name__ == "__main__":
    main()
