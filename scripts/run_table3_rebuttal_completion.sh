#!/usr/bin/env bash
# Serial, fixed protocol; preserve failures and continue independent evidence.
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
flock /tmp/holoq-vla-policy-timing.lock "$task_python" -m pytest -q research/pi05_table3/test_definition_runtime.py > "$task_data/definition_kernel_checks.log" 2>&1
task_test_status=$?
if [ "$task_test_status" -eq 0 ]; then
  "$task_python" -u -m research.pi05_table3.measure_definition8 > "$task_data/measure_definition8.log" 2>&1
  task_definition_status=$?
  if [ "$task_definition_status" -eq 0 ]; then
    for task_scope in backbone head; do
      "$task_python" -u -m research.pi05_table3.measure_scopes --method definition8 --scope "$task_scope" > "$task_data/definition8_scope_${task_scope}.log" 2>&1
    done
  fi
fi
for task_kind in fp16 bf16 gptq_marlin252 smoothquant_qserve414; do
  "$task_python" -u -m research.pi05_table3.measure_controls --steps 8 --output-group controls8 --kind "$task_kind" > "$task_data/control8_${task_kind}.log" 2>&1
done
if [ -d "$task_data/controls/gptq_marlin252" ]; then
  mv "$task_data/controls/gptq_marlin252" "$task_data/controls/gptq_marlin252_failed_bitwise_guard"
fi
"$task_python" -u -m research.pi05_table3.measure_controls --kind gptq_marlin252 > "$task_data/control_gptq_marlin252_numerical_guard.log" 2>&1
