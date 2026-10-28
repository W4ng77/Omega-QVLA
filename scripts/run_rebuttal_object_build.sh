#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
export PYTHONPATH="$task_root:/home/xinyu/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
export PATH="/home/xinyu/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1
task_data="${REBUTTAL_DATA:-$task_root/results/rebuttal_object_20261028}"
task_python="${REBUTTAL_PYTHON:-/home/xinyu/miniconda3/envs/groot_test/bin/python}"
exec "$task_python" -u -m research.rebuttal.build \
  --source-capture "${SOURCE_CAPTURE:-$task_data/source_existing_capture}" \
  --calibration "${CALIBRATION:-$task_data/source_teacher_trajectories}" \
  --output "$task_data/submission_contract_v2_build" \
  --deployment "$task_data/deployments/submission_contract_v2" \
  "$@"
