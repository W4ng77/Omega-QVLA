# Legacy fused rotation regression

{
  "status": "withdrawn_functional_regression",
  "case": "holoq_fusedrot",
  "successes": 0,
  "episodes": 100,
  "cause": "Legacy Triton rotation assumes contiguous R; serialized output R has stride (4096,1,64). Reads transpose incorrectly. Prior tests used only contiguous R.",
  "withdrawn_timing_claims_ms": [
    80.133,
    74.186
  ],
  "affected": "All triton_block64 fused variants; Nunchaku hybrids using same rotation also unvalidated. Nonfused baseline records unaffected.",
  "action": "Selected accuracy controller stopped before next GPU stage; primary baseline controller continues. Repair explicit v2 source/pack/output; no reuse of legacy fused episodes.",
  "time": 1793145600.5638144
}

The 80.13 ms and 74.19 ms figures are withdrawn as valid HoloQ performance comparisons. The old source and immutable packs remain reproducible. Version 2 is separately named and must pass real-input layer and full-policy comparisons before timing or task evaluation.

## Repair validation

Explicit strided v2: 17 GPU layout/stream/Graph tests passed. The original-grid real-policy check compared 8520 projection calls, all eight denoising steps, plus ten complete policy outputs: zero measured difference from the nonfused packed path. The old kernel gives NMSE 1.958473 on a real transposed output rotation. These correctness observations are restricted to the tested inputs; new replay and independent100 episode results are still pending.
