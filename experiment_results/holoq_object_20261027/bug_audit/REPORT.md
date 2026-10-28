# Bug audit before further optimization

Core math audit passed. Five actual weight/input rotation round trips have maximum NMSE 2.8769162e-07; 24 native INT4 checks across three real DiT shapes and all eight steps have maximum NMSE 7.12212339e-14. All96 DiT layers capture eight distinct activation-scale rows. Ten full policy Graph/eager comparisons have NMSE between 0.0001451 and 0.0003501; this is not bitwise equality or a task-success guarantee.

## Bugs fixed

- Graph cache ignored Python scalar values: changed constants could silently retain the first captured value. Typed constants are now part of the key. Current GR00T inputs were inspected over ten tasks and are tensor-only; their original tensor signatures remain exactly unchanged.
- Interrupted rollout journals could retain incomplete-episode timing fragments and count them again after restart. Recovery now archives original bytes, retains only completed episodes and rejects duplicate/missing control records.
- Resume could mix different packs/inputs/code, and lose prior peak memory. Resume now checks pack/input SHA, mathematical source fingerprints and protocol, archives old metadata, carries peaks and records earlier measurement windows.
- Calibration seed metadata omitted initial_state_index*1000 for nonzero calibration states. Current experiment uses state0, so prior calibration records are unaffected.
- Trace analysis now excludes synthetic GPU stage annotations, separates kernel sums from activity unions, and counts implicit blocking in cudaMemcpyAsync. Coarse Nsight Graph capture omitted its GPU body and is flagged incomplete; no missing Graph interval is labeled idle.

Five behavioral regression tests passed. No weights or quantization/scaling definitions were retuned.

## Existing-result audit

9 complete100 journals were checked for exactly one timing row per executed control step and remained byte-for-byte unchanged. Primary result audit verifies24 completed rows with zero errors. Current completed scores:

| Case | Successes |
|---|---:|
| awq_qwen_full | 92/100 |
| gptq_full | 98/100 |
| smoothquant_a8 | 88/100 |
| gptq_selective | 93/100 |
| omniquant_selective | 88/100 |
| smoothquant_a4 | 53/100 |
| awq_qwen_selective | 93/100 |
| holoq | 72/100 |
| holoq_llm_s4_rotv2 | 70/100 |

## Measurement evidence correction

Holding the pending queue with SIGSTOP also suspended that controller's GPU monitoring thread while its already-running eager replay finished. One conversion pilot therefore lacks monitor coverage. Its raw data is preserved in replay_unmonitored_prior and excluded; the same300-call replay has been repeated successfully under an independent sampler; the replacement passes the GPU coverage audit. Monitoring evidence is never fabricated retroactively. Final audit and selection must use the replacement.

The earlier legacy fused-rotation0/100 regression remains withdrawn. The present audit does not turn70/100 into a stronger accuracy claim. Full VLA baseline and conversion task evaluations remain pending.

Final audit after replacement: primary24 completed rows +rotation8 completed rows =32 verified rows, zero errors. Nine completed100-episode journals remain unchanged. Current GR00T tensor-only Graph signatures are unchanged by the scalar-constant fix. Backend/scaling math is frozen before continuing the pending conversion/VLA measurements.
