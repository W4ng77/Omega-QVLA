# Rebuttal evidence status

This is an interim audit, not an accuracy or speed claim. HoloQ-VLA and Omega-QVLA denote the same method. The authoritative submission source is the user's page-verified description in [submission_contract.md](submission_contract.md). Exact reviewer texts were unavailable; the question topics below were supplied by the user.

## Confirmed evidence and limits

- The old 70/72 reconstructions added output rotations and omitted the randomized Hadamard signs. They remain reconstruction controls, not submission reproductions. The detailed alignment matrix separates historical, reconstructed and adopted choices.
- Historical pack SHA256 `0fa5f367e7ed6e7be25efce25fce967078cb5dc86b39ae5dd53b9dac4ec107ba` stores dense rounded weights without their original integer-code/weight-scale metadata. For every one of its 180 layers, sampled output rows contain more than 16 distinct values across the full input dimension. This rules out a single uniform 4-bit grid per output for this artifact. Source code recomputes scales within 128-column processing blocks. Sampling does not establish every weight's exact original scale.
- Graph initial noise previously inherited incoming feature FP16 dtype, whereas eager noise used FP32 after conditioning. Commit `d96d2db` fixes that mismatch. In the actual historical packed policy, all 10 tested observations then had bitwise-equal initial noise and identical final actions. This does not validate every new backend or every observation. Earlier Graph scores keep their old protocol label.
- Six CUDA/CPU mathematical tests passed: input-only unquantized equivalence; disclosed zigzag mapping; scale-inside-K counterexample; signed15 per-output RTN; actual official GPTQ processing128 with FP32 per-output scales; and the existing packed backend on non-contiguous inputs over all eight distinct column-scale rows. These are local correctness evidence, not full-policy task accuracy.
- The adopted reconstruction uses a single frozen packed codes/scales/rotation artifact for the FP32 equation oracle and packed execution. Its actual deployed arithmetic is packed W4 decoded in registers to FP16 operands, A4 values reconstructed into transient FP16 operands, and FP16 tensor-core MMA with FP32 accumulation. It is **not** a native INT4×INT4 pipeline. There is no full floating-weight cache in packed mode. Full-policy validation remains pending.
- The selected calibration source comprises 10 saved teacher trajectories, **9 successful and 1 unsuccessful**, with 10 uniformly selected frames each. All 10 are retained. The new capture retains all actual input tokens. These are disclosed reconstruction choices, not recovered historical-paper calibration identity.

## Reviewer evidence checklist

|Reviewer topic supplied by user|Evidence needed|Available now|Still missing|
|---|---|---|---|
|JvpR: real/simulated W4A4; column scales; W4A8/W4A16 deployments|Correct expression, runtime operator graph, traced arithmetic, same-version deployment metrics|Corrected equation; explicit fixed packed/runtime source; local kernel tests|Full-policy frozen-grid validation and matched success/latency/memory; representative quantized-baseline deployment table|
|BY2H: graph and transform overhead; backbone/head isolation|Cumulative full-get_action profiler; scope ledger; backbone-only/head-only/both; head storage breakdown|Explicit scope targets 84 language +96 DiT projections; no output rotations in new implementation|New frozen scope/byte census, isolation experiments, cumulative profiling after correctness gate|
|gXcZ: representative hardware latency, throughput, peak VRAM|Exclusive GPU, full policy timing boundaries, mean/p50/p95, stable and peak memory|Actual RTX4080SUPER correctness checks; exclusive shared-GPU lock|Formal unprofiled measurements for validated frozen policy; no new H100 evidence|
|aAUc: storage versus timed scope; eight versus ten steps|Exact selected/excluded modules and assets; effective-step trace|New construction explicitly requires 8 calibration and inference steps; scope is not all operators|Frozen deployment byte ledger, actual full-call step trace and matched timing; Pi0.5 ten-step results cannot substitute|

## Same-protocol primary results

No new 100-episode final result exists yet. Dashes mean unmeasured, not zero.

|Configuration|Frozen identity|Object /100|Full policy mean/p50/p95|Memory|Deployment storage|Status|
|---|---|---|---|---|---|---|
|Historical implementation|Published pack SHA above; runtime and Graph version must be recorded|—|—|—|—|Retain history; run under shared new protocol|
|Submission-contract numerical reference|Pending frozen build|—|Accuracy reference, not optimized speed baseline|—|—|Input-only UDH;15-level;per-output W scale;all-token disclosed calibration|
|Same-grid actual packed deployment|Must share reference codes/scales/R|—|—|—|—|Existing FP16-MMA packed W4; no native-S4 claim|
|step_weight / group_int|Separate identities|—|—|—|—|Independent variants; cannot replace proof of original-method implementation|

Long per-step versus pooled-static ablation follows the Object correctness/accuracy freeze. Object alone cannot establish the submitted per-step benefit. No seed, episode subset or sign pattern will be selected to match the submitted 92% score.
