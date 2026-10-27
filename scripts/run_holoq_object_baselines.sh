#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${LIBERO_ROOT:-/home/xinyu/LIBERO}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
export PATH="${NINJA_BIN:-/home/xinyu/.venv-vllm/bin}:$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PYTHON="${GR00T_PYTHON:-/home/xinyu/miniconda3/envs/groot_test/bin/python}"
ROOT="${EXPERIMENT_ROOT:-results/holoq_object}"
OUTPUT="${RUN_OUTPUT:-$ROOT/evaluation_v1}"
mkdir -p "$ROOT"
"$PYTHON" -m research.holoq_bench.bootstrap
# Quantization/calibration construction commands are in docs; this entrypoint
# deliberately requires completed immutable deployment artifacts, never chooses
# another pack/backend or starts fake quantization if one is absent.
"$PYTHON" -u -m research.holoq_bench.submit --root "$ROOT" --output "$OUTPUT" --trials 10 "$@"
"$PYTHON" -m research.holoq_bench.summarize --runs "$OUTPUT" --deployments "$ROOT/deployments" --output "$OUTPUT/report"
