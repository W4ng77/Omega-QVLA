# HoloQ-VLA / quantized baseline results

Common execution: **head_graph**. Missing scores are pending, not zero.
All latency rows measure complete get_action; replay and rollout are distinct. Storage is decimal GB.

| Method | Variant/backend | W/A | Scope | Success | Replay mean / p50 / p95 ms | Rollout mean / p50 / p95 ms | Peak alloc / reserve GB | Deployment GB | Speedup vs fastest quant baseline |
|---|---|---|---|---|---|---|---|---|---|
| HoloQ-VLA | original-definition / packed Marlin + block rotations | 4/4 | 180 (84 LLM + 96 DiT) | 72/100 | 85.36 / 84.14 / 94.39 | 87.67 / 84.32 / 108.03 | 4.25 / 4.57 | 3.98 | 0.69 |
| GPTQ | Marlin / exclude DiT attention | 4/16 | 116 (84 LLM + 32 DiT MLP) | 93/100 | 65.26 / 61.96 / 76.97 | 64.46 / 60.44 / 75.07 | 4.37 / 4.72 | 4.11 | — |
| GPTQ | Marlin / full targets | 4/16 | 180 (84 LLM + 96 DiT) | pending | 67.78 / 65.45 / 79.16 | pending / pending / pending | 4.14 / 4.45 | 3.87 | — |
| AWQ | Qwen grouped search/folding + DiT linear port / packed AWQ | 4/16 | 116 (84 LLM + 32 DiT MLP) | 93/100 | 60.54 / 59.37 / 72.48 | 63.55 / 58.91 / 77.26 | 4.39 / 4.74 | 4.13 | — |
| AWQ | Qwen grouped search/folding + DiT linear port / packed AWQ | 4/16 | 180 (84 LLM + 96 DiT) | pending | 60.11 / 56.23 / 73.20 | pending / pending / pending | 4.16 / 4.47 | 3.89 | — |
| OmniQuant | linear LWC/scale port / packed AWQ | 4/16 | 116 (84 LLM + 32 DiT MLP) | 88/100 | 60.59 / 58.04 / 75.88 | 54.87 / 54.10 / 58.07 | 4.39 / 4.74 | 4.13 | — |
| SmoothQuant | QServe packed INT8 MMA | 4/8 | 180 (84 LLM + 96 DiT) | 88/100 | 58.91 / 57.03 / 67.67 | 57.38 / 55.26 / 74.01 | 4.14 / 4.45 | 3.87 | — |
| SmoothQuant | QServe INT8 container, numerical A4 | 4/4 | 180 (84 LLM + 96 DiT) | pending | 61.09 / 56.83 / 77.00 | pending / pending / pending | 4.14 / 4.45 | 3.87 | — |

## Complete-policy speedups against every measured quantized baseline

| Quantized baseline | holoq |
|---|---|
| gptq_selective | 0.765x |
| gptq_full | 0.794x |
| awq_qwen_selective | 0.709x |
| awq_qwen_full | 0.704x |
| omniquant_selective | 0.710x |
| smoothquant_a8 | 0.690x |
| smoothquant_a4 | 0.716x |

Speedup = baseline full-policy replay mean / our full-policy replay mean; below 1 means slower.
See speedups.json for every baseline, main.csv for all execution modes and exact bytes.

**Fidelity limits:** AWQ uses a Qwen3 grouped adapter and a DiT linear adapter; OmniQuant is a linear LWC/scaling port, not the full upstream block/LET implementation. DuQuant uses the paper GR00T SVD recipe. Numerical A4/A8 feeding Marlin uses FP16 MMA. These distinctions constrain any claim about an optimized original method.

100-episode score differences are descriptive; do not infer statistical improvement.
