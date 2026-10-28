#!/usr/bin/env bash
set -euo pipefail
cd /home/xinyu/holoq-vla-baselines
export PYTHONPATH="$PWD:/home/xinyu/LIBERO"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/xinyu/.venv-vllm/bin:/usr/local/cuda-12.4/bin:/usr/bin:/bin
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
exec /home/xinyu/miniconda3/envs/groot_test/bin/python -u -m research.holoq_bench.evaluate_paper_holoq "$@"
