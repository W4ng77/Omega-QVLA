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
task_data=results/pi05_table3_414_20261028
flock /tmp/holoq-vla-policy-timing.lock "$task_python" -m pytest -q research/pi05_table3/test_baseline_runtime.py > "$task_data/quantized_baseline_kernel_checks_final.log" 2>&1
"$task_python" -u -m research.pi05_table3.validate_marlin_maxpar1 > "$task_data/marlin_maxpar1_full_policy_oracle.log" 2>&1
for task_group in controls8_ieee controls; do
  if [ -d "$task_data/$task_group/gptq_marlin252" ]; then
    mv "$task_data/$task_group/gptq_marlin252" "$task_data/$task_group/gptq_marlin252_invalid_parallelM"
  fi
done
"$task_python" -u -m research.pi05_table3.measure_controls --steps 8 --ieee-matmul --output-group controls8_ieee --kind gptq_marlin252 > "$task_data/control8_ieee_marlin_maxpar1.log" 2>&1
"$task_python" -u -m research.pi05_table3.measure_controls --kind gptq_marlin252 > "$task_data/control10_marlin_maxpar1.log" 2>&1
