HoloQ-VLA: Pi0.5 Table 3 deployment evidence
=========================================

The measured 414-layer deployment reduces storage and warmed inference memory relative to the high-precision control, but is slower than both validated packed quantized controls. HoloQ-VLA and Omega-QVLA are the same method.

This packet answers the supplied kernel, scale, timing, storage, memory and coverage questions. It does not establish a new LIBERO success rate, real-rollout latency or historical H100 execution path. Those claims are explicitly left unsupported rather than filled with paper scores.

Primary deployment comparison
-----------------------------

+------------------+--------------+--------------------------+---------+-------------------+----------+
| Method           | W/A; targets | Mean/p50/p95 ms          | Calls/s | Peak alloc/res GB | Files GB |
+==================+==============+==========================+=========+===================+==========+
| FP16 control     | 16/16; 0     | 106.83 / 106.38 / 110.94 | 9.361   | 7.514 / 8.338     | 7.234    |
+------------------+--------------+--------------------------+---------+-------------------+----------+
| GPTQ / BF16-AWQ  | 4/16; 252    | 109.09 / 108.84 / 111.00 | 9.167   | 4.147 / 4.446     | 3.866    |
+------------------+--------------+--------------------------+---------+-------------------+----------+
| SQ-W4 / QServe   | 4/8; 414     | 86.99 / 88.27 / 89.99    | 11.495  | 3.502 / 3.790     | 3.229    |
+------------------+--------------+--------------------------+---------+-------------------+----------+
| HoloQ core + RTN | 4/4; 414     | 309.58 / 308.99 / 312.87 | 3.230   | 3.712 / 4.245     | 3.441    |
+------------------+--------------+--------------------------+---------+-------------------+----------+

The complete precision/scope labels and speed ratios are in ``DEPLOYMENT_MAIN_TABLE.csv``.

All rows use the same Pi0.5 checkpoint, fixed Object observations/noise, batch 1, action horizon 10, 224x224 model inputs, 32 warmup calls and 100 unprofiled complete calls per mode. The two supplied cameras and the actual masked third camera slot are processed by every method. All methods receive Eager and full-sampler CUDA Graph settings; compile is disabled. Both GPUs/processes are logged, with no competing load on GPU 1.

Timing wraps the complete synchronized ``policy.infer``: CPU preprocessing, H2D, vision, language, all denoising steps, permutations/rotations, quantization/packing, GEMM, scales, D2H and output transforms. RPC, simulator, loading and cold graph capture are excluded. The internal asynchronous ``infer_ms`` field is not used as complete latency. Raw Eager measurements have substantial host variability; mean/p50/p95 and every sample are retained. CPU affinity/frequency was not fixed.

The primary means are 309.58 ms for HoloQ, 109.09 ms for GPTQ/BF16-AWQ, 86.99 ms for the SmoothQuant-W4/QServe control, and 106.83 ms for FP16. HoloQ is 2.84 times slower than this GPTQ deployment and 3.56 times slower than this QServe deployment. This is a deployment comparison with different activation precisions and explicitly different coverage, not a claim of identical quantization accuracy. No speed superiority to unmeasured optimized QuantVLA, QVLA or ActQuant is established.

Method definition and execution
-------------------------------

The submitted-definition source is the author's page-verified description of the eight-page submission, not the separately available arXiv appendix. ``method_alignment.csv`` covers definitions, unspecified choices, historical artifacts and adopted behavior. The historical 70/72 GR00T reconstructions are not paper reproduction and are not Pi0.5 Table 3 evidence.

The core reconstruction has input-only block64 rotation, direct left SVD, U D H64/8 ordering, a disclosed cross-block serpentine interpretation, signed[-7,7] codes, full-K per-output weight scales, dynamic per-token language activation scales, and eight per-layer/step/input-channel expert scale rows. Language uses GPTQ processing block128/damping0.01 without weight grouping; the expert uses RTN. The first FP32 Hessian factorization failure and separately named FP64 Gram/factorization retry are preserved. D seed, SVD sign convention and q99.9 robust estimator are disclosed choices where the submission is incomplete.

The vision targets retain the released asymmetric RTN G64/G16 recipe. The supplied body does not uniquely determine that vision recipe; this packet does not claim a unique full-paper calibration reconstruction. Calibration uses ten initial Object observations/all actual tokens, not ten trajectories. Baseline static quantizers use ten-step teacher captures from the same observations; this is disclosed and is not equal-calibration success-rate evidence.

There are 162 vision, 126 language and 126 expert quantized targets. Embeddings, AdaRMS conditioning, action/time auxiliary projections, patch projection and nonlinear operators remain floating outside these targets. Full target names and all remaining checkpoint tensors are listed. Each eight-step policy executes 486 quantized vision GEMMs, 126 language GEMMs and 1008 expert GEMMs, for 1620 target executions; distinct layers and repeated calls are different counts.

``kernel_execution.csv`` and ``operator_graph.csv`` map stages to functions, formats and arithmetic:

* Language: physical signed-A4/W4 packing, existing CUTLASS native S4 products, INT32 accumulation, FP32 per-token/per-output scale epilogue.
* Expert: physical signed-A4/W4 packing, per-input-channel scale inside K, on-chip FP16 MMA operands and FP32 accumulation. One weight matrix per layer; no step-weight conversion or full floating-weight cache.
* Vision: physical unsigned-A4/W4 packing, on-chip RTN decode, BF16 MMA and FP32 accumulation.

In paper row-vector orientation, the corrected expert expression is::

   Y[m,n] = Sw[n] * sum_k(QX[m,k] * Sa[layer,t,k] * QW[k,n]) + bias[n]

Sa[layer,t,k] cannot be moved outside the K reduction. The scalar shorthand in Eq.9 must be corrected. ``step_weight`` and ``group_int`` are separate algorithm-changing variants and are not used as proof of the original method.

Correctness evidence
--------------------

All 252 core unquantized transforms pass equivalence checks: maximum NMSE 1.13e-10. Five core layout/stride/step-dispatch tests pass. Actual packed core calls pass 1134 same-input frozen-grid oracle checks, maximum NMSE 9.94e-7. Across ten fixed observations, final action maximum NMSE is 0.004914 and maximum absolute difference is 0.1038; per-step velocities are retained. These numerical engineering checks do not guarantee equal task success. Graph/eager actions agree exactly in the recorded checks.

The original local NMSE limit1e-4 and complete-action limit0.01 were not relaxed to admit failing backends. Tested Marlin default-parallel and max_par1 paths failed; the latter fails 54 actual MLP calls. The same-grid AWQ FP16 path uses FP16 accumulation and fails one strict check. They are excluded from the validated table. A separately named GPTQ build fixes BF16 scales before reconstruction and uses the existing AWQ BF16/FP32 path: all 1134 actual calls pass, maximum NMSE3.88e-5. It remains GPTQ quantization, not AWQ quantization/search. QServe passes an independent integer oracle for all 1620 actual eight-step target calls, maximum NMSE1.92e-10. Dense floating references are correctness oracles only and are never substituted for packed baseline timing.

The main HoloQ driver resets matmul TF32 after model construction. An audit found that the constructor overrides pre-load settings in some first control/subset scripts. Only the affected eight-step cases were repeated under ``controls8_ieee`` / ``definition8_scope_controls_ieee``; flags are recorded after load. Matmul TF32 is false and cuDNN TF32 remains true in all primary cases, the latter concerning floating image patch computation outside quantized targets.

Where the complete time goes
---------------------------

+------------------+--------------+-------+----------+---------+--------------------+------------+-------+
| Stage            | GPU total ms | GEMM  | Rotation | Permute | A-quant/pack/scale | Data/dtype | Other |
+==================+==============+=======+==========+=========+====================+============+=======+
| vision           | 94.73        | 75.67 | 0.00     | 0.00    | 13.47              | 3.28       | 2.32  |
+------------------+--------------+-------+----------+---------+--------------------+------------+-------+
| language         | 94.25        | 8.09  | 6.69     | 6.74    | 36.17              | 27.70      | 8.87  |
+------------------+--------------+-------+----------+---------+--------------------+------------+-------+
| expert (8 steps) | 115.36       | 72.84 | 2.55     | 1.84    | 7.38               | 13.67      | 17.08 |
+------------------+--------------+-------+----------+---------+--------------------+------------+-------+
| other            | 0.93         | 0.00  | 0.00     | 0.00    | 0.00               | 0.10       | 0.83  |
+------------------+--------------+-------+----------+---------+--------------------+------------+-------+

See ``profile_breakdown.csv`` for counts and source records. Quant/pack/scale groups include epilogue or bias work where the trace does not separate it.

The eight-step trace has about305.3 ms of cumulative GPU work. Vision GEMM costs75.7 ms, all expert GEMMs72.8 ms, and language integer GEMM8.1 ms. Across phases, quantize/pack/scale-or-bias processing costs57.0 ms, permutations plus rotations17.8 ms, copy/dtype/contiguous44.8 ms, and other work29.1 ms. Each activity is assigned once using raw CUDA correlation IDs and enclosing CPU scopes. Propagated FunctionEvent totals that duplicate work are rejected. Timestamp interval union is reported separately; sums must not be added across overlaps or added to CPU wait.

Nsight verifies the real eight-step Graph mode: five Graph launches and exactly630 language S4,5040 expert and2430 vision target kernels across five calls. Its diagnostic mean325.8 ms differs from unprofiled latency and is not used as formal timing. About25.1 ms/call of host Graph-launch API duration is visible under instrumentation; Graph-node execution itself has about1.47 ms of internal gaps. CPU memcpy API duration includes waiting for GPU completion; device copy work is only about0.042 ms/call. The timeline supports GPU-compute and Graph-submission costs, not a conclusion that thousands of Python synchronizations dominate Graph execution.

Memory and storage
------------------

All sizes in the main table are decimal GB. The source has5,408,612,352bytes in the414 selected weights and7,233,515,040bytes in complete checkpoint tensors. HoloQ's selected packed tensors plus scales, rotations and permutations total1,615,668,192bytes:70.13% below the selected original weights. Complete deployed tensor payload is3,440,570,880bytes; all model files, normalization assets and required loader manifest total3,441,209,843bytes. The common4,264,023-byte tokenizer is separately listed. Original export inventory fields omit the manifest written afterwards; corrected complete-file accounting includes it without modifying frozen manifests.

The warmed Graph peak allocated/reserved figures are3.712/4.245GB for HoloQ versus7.514/8.338GB for FP16. Peak allocated is about50.6% lower than FP16, but is higher than QServe's3.502GB. Reserved memory reflects allocator history, not minimum required VRAM. These statistics exclude loading/cold capture and do not convert a static-table percentage into runtime savings. The available metadata-inclusive artifacts do not reproduce1.39GB/74.2%; that claim must be qualified/corrected.

With identical frozen core grids, backbone-only/head-only/both full Graph policy means are232.10/192.33/309.58ms. These isolate performance and memory, not success. The expert's126 target weights fall from622.85MB to216.65MB including necessary metadata. Active action-head static tensors including remaining floating auxiliaries fall from860.20MB to453.99MB. Resident registered tensors differ because sensitive auxiliaries are promoted toFP32; the full-deployment ledger shows about691.34MB of active head model tensors plus the unchanged526.65MB unused expert vocabulary projection. That unused projection is retained equally by all methods and never executed. The separately measured head-stage incremental allocated peak is1.98MB above the already-resident model and prefix; it is not total action-head memory.

Deliverables and limits
----------------------

* ``READY_TO_PASTE.txt``: short combined English response.
* ``REVIEWER_REPLIES.txt``: JvpR, BY2H, gXcZ, aAUc and cd6x drafts with evidence filenames.
* ``reviewer_evidence.csv``: eight-topic answer/correction/limitation mapping.
* ``baseline_checklist.csv``, ``method_alignment.csv``, ``kernel_execution.csv`` and ``operator_graph.csv``: definitions and actual execution scope.
* ``main_results.csv``: all28 Eager/Graph rows, including separately named historical ten-step cases. The primary four-row table uses only eight-step validated deployments.
* ``AUDIT_DECISIONS.txt``: failures, superseded results and reuse decisions. No failed record is removed.
* ``REPRODUCE.txt`` and the independent branch scripts: commands, environment and required inputs.
* ``experiment_results/pi05_table3_414_20261028``: raw reports, per-call JSONL, provenance, pack/checkpoint identifiers, compressed logs/traces, original observation source and Nsight reports. Large checkpoint/pack files stay local with exact SHA256 identifiers and rebuild commands.

No new task-success scores, real-rollout timings or original H100 kernel traces are claimed. This packet is sufficient to answer the supplied execution/resource questions by providing measurements and correcting unsupported claims; it is not evidence that the newly compiled deployment reproduces the paper's success rates.
