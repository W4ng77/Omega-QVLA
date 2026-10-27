# HoloQ-VLA / quantized baseline results

Common execution: **head_graph**. Missing scores are pending, not zero.
All latency rows measure complete get_action; replay and rollout are distinct. Storage is decimal GB.

| Method | Variant/backend | W/A | Scope | Success | Replay mean / p50 / p95 ms | Rollout mean / p50 / p95 ms | Peak alloc / reserve GB | Deployment GB |
|---|---|---|---|---|---|---|---|---|
| HoloQ-VLA | original-definition / packed Marlin + block rotations | 4/4 | 180 (84 LLM + 96 DiT) | pending | 85.36 / 84.14 / 94.39 | pending / pending / pending | 4.25 / 4.57 | 3.98 |
| HoloQ-VLA conversion | step_weight / native S4 DiT + Marlin LLM | 4/4 | 180 (84 LLM + 96 DiT) | pending | 81.86 / 77.09 / 103.74 | pending / pending / pending | 8.40 / 8.73 | 5.98 |
| GPTQ | Marlin / exclude DiT attention | 4/16 | 116 (84 LLM + 32 DiT MLP) | pending | 65.26 / 61.96 / 76.97 | pending / pending / pending | 4.37 / 4.72 | 4.11 |
| GPTQ | Marlin / full targets | 4/16 | 180 (84 LLM + 96 DiT) | pending | 67.78 / 65.45 / 79.16 | pending / pending / pending | 4.14 / 4.45 | 3.87 |
| AWQ | Qwen grouped search/folding + DiT linear port / packed AWQ | 4/16 | 116 (84 LLM + 32 DiT MLP) | pending | 60.54 / 59.37 / 72.48 | pending / pending / pending | 4.39 / 4.74 | 4.13 |
| AWQ | Qwen grouped search/folding + DiT linear port / packed AWQ | 4/16 | 180 (84 LLM + 96 DiT) | pending | 60.11 / 56.23 / 73.20 | pending / pending / pending | 4.16 / 4.47 | 3.89 |
| OmniQuant | linear LWC/scale port / packed AWQ | 4/16 | 116 (84 LLM + 32 DiT MLP) | pending | 60.59 / 58.04 / 75.88 | pending / pending / pending | 4.39 / 4.74 | 4.13 |
| SmoothQuant | QServe packed INT8 MMA | 4/8 | 180 (84 LLM + 96 DiT) | pending | 58.91 / 57.03 / 67.67 | pending / pending / pending | 4.14 / 4.45 | 3.87 |
| SmoothQuant | QServe INT8 container, numerical A4 | 4/4 | 180 (84 LLM + 96 DiT) | pending | 61.09 / 56.83 / 77.00 | pending / pending / pending | 4.14 / 4.45 | 3.87 |
| DuQuant | paper GR00T SVD port / Marlin FP16 MMA | 4/8 | 180 (84 LLM + 96 DiT) | pending | 85.41 / 84.16 / 98.91 | pending / pending / pending | 4.26 / 4.58 | 3.99 |
| DuQuant | paper GR00T SVD port / Marlin FP16 MMA | 4/4 | 180 (84 LLM + 96 DiT) | pending | 89.98 / 84.67 / 109.58 | pending / pending / pending | 4.26 / 4.58 | 3.99 |
| QuantVLA | released GR00T port + ATM/OHB / Marlin | 4/8 | 116 (84 LLM + 32 DiT MLP) | pending | 71.97 / 70.45 / 86.12 | pending / pending / pending | 4.47 / 4.85 | 4.20 |
| QuantVLA | released GR00T port + ATM/OHB / Marlin | 4/4 | 116 (84 LLM + 32 DiT MLP) | pending | 74.07 / 69.24 / 95.45 | pending / pending / pending | 4.47 / 4.85 | 4.20 |

Speedup = baseline full-policy replay mean / our full-policy replay mean; below 1 means slower.
See speedups.json for every baseline, main.csv for all execution modes and exact bytes.

**Fidelity limits:** AWQ and OmniQuant use explicit GR00T linear adapters, not unmodified whole-block official adapters. DuQuant uses the paper GR00T SVD recipe. Numerical A4/A8 feeding Marlin uses FP16 MMA. These distinctions constrain any claim about an optimized original method.

100-episode score differences are descriptive; do not infer statistical improvement.
