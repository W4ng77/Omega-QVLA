Pi0.5 / Table3 deployment evidence
==================================

This is an implementation/performance packet, not a new task-success experiment. HoloQ-VLA and Omega-QVLA are the same method.
The submitted-method source is the user-verified eight-page description. The separately available arXiv appendix is not used to fill missing submission details.

Scope: 162 vision + 126 language + 126 expert = 414 selected projections. Full module names and shapes are in the captured manifests and per-deployment reports.
The released-grid historical variant and the definition-aligned8-step core with released visionRTN remain separate configurations. See scope_and_claims.txt and method_alignment.csv.

Complete-policy measurement
---------------------------

RTX4080SUPER16GB, batch1, horizon10, 224x224 supplied cameras, identical fixed observations/noise,32warmup and100calls per mode. Eager and full-sampler CUDA Graph are available to every measured method; compile is disabled. Includes preprocessing,H2D,vision/language/all Euler steps,transforms,quantization/packing,GEMM,D2H and output transforms. Excludes RPC,simulator,loading and cold graph capture.
Allocated/reserved statistics refer to warmed inference, reset immediately before formal calls. Reserved memory includes allocator history; it is not a minimum required VRAM claim. Model static files and shared tokenizer bytes are separate.

.. csv-table:: Matched whole-policy timings and memory
   :file: main_results.csv
   :header-rows: 1

The speed ratios are computed only within matching step counts and execution modes. A ratio baseline_ms/HoloQ_ms below1 means HoloQ is slower; see speed_ratios.csv. No inherent W4A4 speed advantage or superiority to unmeasured VLA-specific implementations is claimed.

Correctness and execution
-------------------------

Definition8 local same-input oracle checks: 1134, maximum NMSE 9.93927202e-07. Ten-observation final action maximum NMSE 0.00491390433, max absolute discrepancy 0.103797986. These are backend arithmetic comparisons on the SAME frozen grid, not quantization-to-FP16 errors or success rates.
Graph/eager actions agree in the recorded fixed-input checks. The engineering action bound0.01 was fixed before timing; passing it does not prove equal task success.

Language: packed signed15 W4/A4, existing CUTLASS nativeS4 products/INT32 accumulation andFP32 scale epilogue. Expert: packed signed15 W4/A4, Sa[t,k] insideK, on-chip FP16 operands/FP32 accumulation. Vision: released asymmetricRTN packedW4/A4, BF16MMA/FP32 accumulation. Outside selected targets, floating attention,normalization,embeddings,auxiliary projections and image patch computation remain. No dense selected-weight cache or floating F.linear fallback is used by the timed packed targets.

Correct expert equation: Y[m,n] = Sw[n] * sum_k(QX[m,k] * Sa[layer,t,k] * QW[n,k]) + bias[n]. One W4 matrix per layer; no step_weight banks or weight requantization.

Storage, attribution and limitations
------------------------------------

Table3 displayed5.41GB matches5,408,612,352bytes of selected414 weights. The full source checkpoint has7,233,515,040tensor bytes. Required metadata must be included; see complete_storage_audit.json and definition_storage_audit.json in the evidence archive. The current deployment does not establish the submitted1.39GB/74.2% number when all necessary metadata is included. Static target saving, complete model storage and runtime memory are different quantities.

Profiling uses raw GPU activity durations linked by CUDA correlation IDs to enclosing CPU scopes. Rejected FunctionEvent association sums are preserved but not cited. GPU overlap is handled by interval union. Nsight confirms the actual historical CUDA Graph deployment; CUDA API memcpy host time contains GPU completion waits and must not be added to GPU kernel work. See profile_breakdown.csv and nsight_verified_summary.json.

Calibration is10 initial Object snapshots/all tokens, not the submission10 trajectories. The definition8 builder retains signed15,full-K per-outputSw,input-onlyUDH64,8-step per-channelSa,DiTRTN and GPTQprocessing128/damp.01. Zigzag mapping,Dseed andq99.9 are explicitly disclosed choices where the body is incomplete. FirstFP32 factorization failure and separateFP64Gram/factorization retry are preserved. Baseline static weights were calibrated with10-step teacher inputs;8-step deployment timing does not make this an equal-calibration accuracy comparison.

No new LIBERO success rates or real-rollout timings are established here. Do not attach submitted success rates to these newly measured deployment rows. No historicalH100 native kernel trace was recovered. Optimized QuantVLA/QVLA/ActQuant superiority is not established. Those claims must be omitted or narrowed, not filled from unrelated local ports.

Reviewer replies and reproducibility
------------------------------------

READY_TO_PASTE.txt gives the short combined response. REVIEWER_REPLIES.txt gives topic-specific replies with evidence references. reviewer_evidence.csv lists answer status and remaining unsupported claims. Reproduction commands, package/source versions, pack hashes and raw records accompany this packet.

Incomplete cases at collection time
-----------------------------------

definition8_scope_controls_ieee/backbone/report.json
definition8_scope_controls_ieee/head/report.json
controls8_ieee/gptq_marlin252/report.json
controls8_ieee/smoothquant_qserve414/report.json

