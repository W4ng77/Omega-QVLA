#!/usr/bin/env bash
# This follows an audit: OpenPI resets the global matmul flag at construction.
set -uo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
export PYTHONPATH="$task_root:/home/xinyu/openpi/src:/home/xinyu/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
export PATH="/home/xinyu/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
task_python=/home/xinyu/openpi/.venv/bin/python
task_data=results/pi05_table3_414_20261028
for task_kind in fp16 bf16 gptq_marlin252 smoothquant_qserve414; do
  "$task_python" -u -m research.pi05_table3.measure_controls --steps 8 --ieee-matmul --output-group controls8_ieee --kind "$task_kind" > "$task_data/control8_ieee_${task_kind}.log" 2>&1
done
for task_scope in backbone head; do
  "$task_python" -u -m research.pi05_table3.measure_scopes --method definition8 --scope "$task_scope" > "$task_data/definition8_ieee_scope_${task_scope}.log" 2>&1
done
