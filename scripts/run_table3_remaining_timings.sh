#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
export PYTHONPATH="$task_root:/home/xinyu/openpi/src:/home/xinyu/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
export PATH="/home/xinyu/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
task_python=/home/xinyu/openpi/.venv/bin/python
"$task_python" -u -m research.pi05_table3.measure_controls --kind gptq_marlin252 \
  > results/pi05_table3_414_20261028/control_gptq_marlin252.log 2>&1
for task_scope in backbone head; do
  "$task_python" -u -m research.pi05_table3.measure_scopes --scope "$task_scope" \
    > "results/pi05_table3_414_20261028/scope_${task_scope}.log" 2>&1
done
