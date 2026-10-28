#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
export PYTHONPATH="$task_root:/home/xinyu/openpi/src:/home/xinyu/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
task_output="$task_root/results/pi05_table3_414_20261028/nsight_full414"
exec /usr/local/cuda-12.4/bin/nsys profile --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output="$task_output" \
  /home/xinyu/openpi/.venv/bin/python -u -m research.pi05_table3.nsight_capture
