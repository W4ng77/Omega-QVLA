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
PROF_PYTHON=/home/xinyu/miniconda3/envs/groot_test/bin/python
PROF_OUTPUT=${1:?Pass a new output directory; existing runs are retained}
PROF_DEPLOY=${PROF_DEPLOY:-results/holoq_object/deployments/holoq_llm_s4_rotv2}
PROF_OBS=results/holoq_object/heldout_init10_19/observations.pt
test ! -e "$PROF_OUTPUT"
mkdir -p "$PROF_OUTPUT"
for PROF_MODE in head_graph eager; do
    "$PROF_PYTHON" -u -m research.holoq_bench.profile_policy \
        --deployment "$PROF_DEPLOY" --observations "$PROF_OBS" \
        --output "$PROF_OUTPUT/torch_$PROF_MODE" --mode "$PROF_MODE" --engine torch \
        > "$PROF_OUTPUT/torch_$PROF_MODE.log" 2>&1
    "$PROF_PYTHON" -m research.holoq_bench.analyze_policy_trace \
        --torch-trace "$PROF_OUTPUT/torch_$PROF_MODE/policy_trace.json" \
        --output "$PROF_OUTPUT/torch_$PROF_MODE/analysis"
done
for PROF_GRANULARITY in node graph; do
    nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --cuda-graph-trace="$PROF_GRANULARITY" --capture-range=cudaProfilerApi \
        --capture-range-end=stop -o "$PROF_OUTPUT/nsys_$PROF_GRANULARITY" \
        "$PROF_PYTHON" -u -m research.holoq_bench.profile_policy \
        --deployment "$PROF_DEPLOY" --observations "$PROF_OBS" \
        --output "$PROF_OUTPUT/nsys_${PROF_GRANULARITY}_worker" --mode head_graph --engine nsys \
        > "$PROF_OUTPUT/nsys_$PROF_GRANULARITY.log" 2>&1
    nsys export --type sqlite --output "$PROF_OUTPUT/nsys_$PROF_GRANULARITY.sqlite" \
        "$PROF_OUTPUT/nsys_$PROF_GRANULARITY.nsys-rep" \
        > "$PROF_OUTPUT/nsys_${PROF_GRANULARITY}_export.log" 2>&1
    "$PROF_PYTHON" -m research.holoq_bench.analyze_policy_trace \
        --nsys-sqlite "$PROF_OUTPUT/nsys_$PROF_GRANULARITY.sqlite" \
        --output "$PROF_OUTPUT/nsys_${PROF_GRANULARITY}_analysis"
done
