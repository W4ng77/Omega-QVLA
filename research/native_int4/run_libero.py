"""Reproducible full-policy native S4 LIBERO runner; explicit output per run."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/xinyu/miniconda3/envs/groot_test/bin/python"
CHECKPOINT = "/home/xinyu/.cache/huggingface/hub/models--youliangtan--gr00t-n1.5-libero-object-posttrain/snapshots/f422266b016fd917127d3a0348062a5cd750df43"


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{ROOT}:/home/xinyu/LIBERO",
        PYTHONNOUSERSITE="1",
        CONDA_ROOT="/home/xinyu/miniconda3",
        QUANTVLA_ROOT=str(ROOT),
        QUANTVLA_CONDA_ENV="groot_test",
        LIBERO_CONDA_ENV="libero_test",
        LIBERO_ROOT="/home/xinyu/LIBERO",
        CUDA_HOME="/usr/local/cuda-12.4",
        TORCH_CUDA_ARCH_LIST="8.9",
        MAX_JOBS="2",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        GR00T_CPU_THREADS="8",
        GR00T_INTEROP_THREADS="1",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        NO_ALBUMENTATIONS_UPDATE="1",
        TF_CPP_MIN_LOG_LEVEL="2",
        HF_HUB_DISABLE_XET="1",
        TOKENIZERS_PARALLELISM="false",
        GR00T_PERSTEP_ENGINE="native_s4",
        GR00T_EXPECT_NATIVE_LAYERS="96",
        GR00T_GPTQ_SCOPE="",
        GR00T_CALIB_SUITE="libero_object",
        TORCH_COMPILE_DISABLE="1",
        TORCHDYNAMO_DISABLE="1",
        TORCH_CUDA_GRAPH_DISABLE="1",
        TORCHINDUCTOR_DISABLE_CUDAGRAPHS="1",
    )
    env["PATH"] = "/home/xinyu/.venv-vllm/bin:/usr/local/cuda-12.4/bin:" + env.get("PATH", "")
    pack = str(args.pack.resolve())
    manifest = {
        "engine": "native_s4",
        "format": "step_weight",
        "pack": pack,
        "checkpoint": CHECKPOINT,
        "suite": "libero_object",
        "trials_per_subtask": 10,
        "init_offset": 10,
        "denoising_steps": 8,
        "gpus": args.gpus,
        "smoke_only": args.smoke,
        "start_unix": time.time(),
        "timing": "server full get_action wall time with CUDA synchronization; client includes processing and local RPC",
    }
    if args.smoke:
        env.update(
            CUDA_VISIBLE_DEVICES=args.gpus,
            GR00T_GPTQ="1",
            GR00T_GPTQ_PATH=pack,
            GR00T_GPTQ_ABITS="4",
            GR00T_GPTQ_MISSING="error",
        )
        cmd = [
            PYTHON,
            "-u",
            "scripts/inference_service.py",
            "--server",
            "--warmup-only",
            "--warmup-steps",
            "2",
            "--model-path",
            CHECKPOINT,
            "--data-config",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            "--embodiment-tag",
            "new_embodiment",
            "--denoising-steps",
            "8",
        ]
    else:
        env.update(
            SUITE="object",
            CHECKPOINT=CHECKPOINT,
            GPU_LIST=args.gpus,
            PORT_BASE="5900",
            LLM_QUANT="gptq",
            DIT_QUANT="gptq",
            DIT_ATTN="1",
            WBITS="4",
            ABITS="4",
            LLM_GPTQ_PACK=pack,
            DIT_GPTQ_PACK=pack,
            GR00T_GPTQ_PATH_OVERRIDE=pack,
            GR00T_EVAL_INIT_OFFSET="10",
            NUM_TRIALS_PER_TASK="10",
            OUTPUT_ROOT=str(out),
            GR00T_LATENCY_DIR=str(out / "policy_latency"),
            GR00T_SAVE_VIDEOS="0",
            GR00T_SHARD_STAGGER_MODE="wait_eval",
            GR00T_SHARD_WAIT_EVAL_TIMEOUT_S="240",
        )
        cmd = ["bash", "scripts/run_groot_benchmark.sh"]
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    code = subprocess.call(cmd, cwd=ROOT, env=env)
    manifest.update(exit_code=code, end_unix=time.time())
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
