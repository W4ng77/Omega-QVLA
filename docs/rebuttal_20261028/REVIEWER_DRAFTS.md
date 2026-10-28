# Interim English response material

Drafts below use only verified evidence. They are not ready to submit as completed experimental replies. Exact review text is unavailable; reviewer topics are the user-provided summaries. Numerical slots remain unfilled until matched experiments finish.

## JvpR

Thank you for pointing out the treatment of channel-wise activation scales. For the DiT head, the activation scale depends on the input channel and therefore must remain inside the reduction:

`Y[m,n] = Sw[n] * sum_k(Qx[m,k] * Sa[layer,t,k] * Qw[n,k]) + bias[n]`.

We will replace the scalar-factor shorthand in Eq. 9 with this expression and disclose the actual execution graph. Our current audited deployment stores the weights in packed 4-bit form, reconstructs the quantized activations with their channel-wise scales, and decodes weight tiles on chip for FP16 MMA with FP32 accumulation. Thus numerical W4A4 quantization and packed weight storage should not be confused with native INT4×INT4 execution. We do not claim that W4A4 necessarily runs faster than optimized W4A8 or W4A16 kernels. Matched end-to-end comparisons: **[PENDING; no speed claim]**. Matched task success: **[PENDING]**.

Evidence: `research/rebuttal/method.py:equation_oracle`, `research/rebuttal/runtime.py`, `research/rebuttal/test_method.py`; frozen full-policy trace and result files still required.

## BY2H

The quantized scope is the language-backbone attention/MLP projections and the complete selected DiT attention/MLP linear projections, rather than every operation of the policy. The GR00T Object target list contains 84 language and 96 DiT projections. Vision and excluded auxiliary modules remain in floating point. The adopted input transform is permutation followed by block64 UDH; it does not introduce an output rotation or a low-rank branch. We will report the actual selected and excluded module list, cumulative transform/quantization/GEMM costs across the full eight-step policy, and separate backbone-only, head-only and combined evidence. These measurements are **[PENDING]**. Action-head storage and runtime-memory breakdown: **[PENDING]**.

Evidence: `docs/rebuttal_20261028/METHOD_ALIGNMENT.md`, `research/rebuttal/build.py`; new scope ledger, isolation results and profiler trace still required.

## gXcZ

We agree that static model size alone is insufficient evidence of deployment efficiency. We are measuring complete get_action calls on an RTX 4080 SUPER, including all eight denoising steps and the method's additional transforms, and will report mean/p50/p95 latency, throughput, steady-state memory, and peak allocated/reserved memory. Fixed-observation replay and real rollout measurements will be distinguished, with timing boundaries and common execution optimizations stated explicitly. Results: **[PENDING]**. These new measurements must not be described as H100 measurements.

Evidence: exclusive-GPU experiment configuration and completed correctness audit; formal same-version timing/memory artifacts still required.

## aAUc

We will explicitly align storage accounting with the timed operator scope, listing packed weights, scales, rotations, nonquantized parameters and required metadata. The new primary GR00T protocol uses eight Euler steps for both scale calibration and inference. A previous ten-step Pi0.5 timing configuration cannot validate an eight-step calibrated GR00T model or be merged into the same table. The matched storage/scope and effective-step trace are **[PENDING]**. We also correct the GR00T FP16 average in the narrative to 86.5, consistent with the four entries in Table 2; the separate 87.0 narrative value was inconsistent.

Evidence: user-verified submission contract; `research/rebuttal/build.py` eight-step checks. Frozen byte ledger and full-call trace still required.

## Reproduction qualification

The older 70/72% reconstruction controls differ from the submitted input-only randomized transform and must not be labeled paper reproductions. Historical released packs also exhibit implementation differences, which we are reporting separately. The submitted text does not uniquely specify the random-sign seed, zigzag pseudocode, robust estimator or full calibration/evaluation provenance. The reconstruction choices and unresolved historical details will be disclosed rather than silently attributed to the submission. A same-protocol historical/reference/packed comparison is **[PENDING]**; no score recovery or retained task success is claimed yet.
