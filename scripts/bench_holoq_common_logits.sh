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
COMMON_PYTHON=/home/xinyu/miniconda3/envs/groot_test/bin/python
COMMON_OUTPUT=${COMMON_OUTPUT:-results/holoq_object/common_logits_pilot}
if [ "$#" -eq 0 ]; then
    set -- holoq_llm_s4_rotv2 smoothquant_a8 awq_qwen_selective gptq_full
fi
mkdir -p "$COMMON_OUTPUT"
for COMMON_CASE in "$@"; do
    if [ -f "$COMMON_OUTPUT/$COMMON_CASE/result.json" ]; then
        continue
    fi
    mkdir -p "$COMMON_OUTPUT/$COMMON_CASE"
    if "$COMMON_PYTHON" -u -m research.holoq_bench.bench_common_logits \
        --deployment "results/holoq_object/deployments/$COMMON_CASE" \
        --observations results/holoq_object/heldout_init10_19/observations.pt \
        --output "$COMMON_OUTPUT/$COMMON_CASE" \
        > "$COMMON_OUTPUT/$COMMON_CASE/run.log" 2>&1; then
        printf '%s\n' "COMMON_PILOT_COMPLETED $COMMON_CASE"
    else
        printf '%s\n' "COMMON_PILOT_FAILED $COMMON_CASE (see run.log)"
    fi
done
