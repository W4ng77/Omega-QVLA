# Pi0.5 Table3 scope: 414 quantized projections

The completed reviewer packet is [MAIN_REPORT.rst](../../docs/pi05_table3_rebuttal_20261028/MAIN_REPORT.rst), with the [English reply](../../docs/pi05_table3_rebuttal_20261028/READY_TO_PASTE.txt) and [primary eight-step table](../../docs/pi05_table3_rebuttal_20261028/DEPLOYMENT_MAIN_TABLE.csv). That table uses a definition-aligned language/expert core plus released vision RTN, validated GPTQ with the existing BF16/FP32 AWQ execution backend, and a disclosed W4A8 QServe control. HoloQ does not have a latency advantage. No new success rate is claimed.

The notes below describe the **separate historical ten-step artifact**, not the primary eight-step reconstruction. Its frozen grids and floating MMA behavior remain documented rather than silently being relabeled as the new configuration. Rejected Marlin and FP16-accumulation AWQ results are retained in the audit archive and excluded from the primary comparison.

This work package replaces the mismatched GR00T180-projection timing as evidence for the Pi0.5 Table3 scope. That GR00T timing remains a valid separate measurement but must not be attached to Pi0.5 storage or success results.

The source artifact is `/home/xinyu/models/omega-qvla-pi05-414-w4a4-int4`, Object pack SHA256 `77391d01f0424a24ffe3ff09dae484136cceb48972eacc95cff89fce3598341f`. It comprises 162 vision RTN linears,126 language linears and126 expert linears. Every target must match a module and execute in the audited complete inference. Vision executions can occur more than once per camera; expert projections must select every stored denoising-step row. A shape or dispatch mismatch aborts.

This is the **historical frozen pack**, not a new claim of submission-method reproduction. It has10 expert scale rows, static per-channel language scales, damping.05 and output rotations. Those differ from the user-verified submitted description. The timing matches all10 stored rows and is explicitly labeled10-step; it cannot be called the submitted8-step setting. No weights or activation tables are rebuilt to match scores.

All414 targets use real nibble-packed W4 and A4 buffers, with on-chip floating decode, BF16 MMA and FP32 accumulation. The252 LM/expert projections use the already-existing `_gemm_a4w4_arithmetic` implementation. The162 vision projections preserve their original asymmetric RTN G64/G16 grid using `_vision_gemm`, preserving stored-weight dtype and activation rounding. There is no nativeINT4xINT4 claim, no step-weight conversion, no M-dependent or dense-weight GEMM fallback and no full floating weight cache in the timed packed configuration. The SigLIP `weight.dtype` compatibility property is a zero-element tensor, not a decoded matrix. A dense reference runs only for correctness and is never an optimized speed baseline.

Table3's displayed5.41GB matches the original414 target weight tensors:5,408,612,352bytes. The original full model has7,233,515,040 tensor bytes, including1,824,902,688 other bytes. These scopes must be reported separately. The available packed artifact contains1,715,578,848 target tensor bytes when scales, zero points, permutations and input/output rotations are included. This is an observed storage ledger, **not** a reproduction of Table3's1.39GB/74.2% claim. The code must not discard necessary metadata to force those values. Runtime memory additionally includes nonquantized parameters, activations, cached transform casts, workspaces and CUDA Graph pools.

Run on GPU0 only:

```bash
cd /home/xinyu/holoq-vla-pi05-table3
bash scripts/run_table3_414_timing.sh \
  > results/pi05_table3_414_20261028/full414.log 2>&1
```

The script owns `/tmp/holoq-vla-policy-timing.lock`, uses the OpenPI Python environment and disables compile explicitly. It validates ten fixed Object observations with the same supplied noise, then measures complete `policy.infer` under common eager and complete-sampler CUDA Graph execution. Warmup32;10 observations×10 repeats per mode. Timing includes preprocessing,H2D,all vision/language/expert calls,all10 denoising steps,rotations,quantization/packing,GEMMs,scales,D2H and output transforms. RPC,simulator,loading and cold capture are excluded. The same-mode Graph/eager outputs must agree; reference/packed action NMSE must be below the fixed0.01 engineering guard. This does not establish retained task success. No LIBERO scored episodes are run.

Raw commands, source/pack/checkpoint hashes,414-module/step call ledgers, mean/p50/p95,throughput,stable/peak allocated/reserved memory and five complete-call profiler traces are saved. A sampler records both GPUs and compute processes during timing. Kernel duration sums are diagnostic work, not an alternative end-to-end wall time. All failures remain archived.

Existing relevant modified/untracked source files were copied from `/home/xinyu/omegaqvla` into this independent worktree. Their original hashes are in `legacy_source_snapshot.json`; the original workspace and its changes remain untouched.
