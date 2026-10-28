# Subsequent paper-definition and real-input overflow audit

The earlier REPORT.md describes execution correctness of the **two-sided reconstruction controls**, not conformance to paper Eqs1–8. Their raw72, optimized70, converted66/100 must not be reported as paper-input-only HoloQ results.

Further paper review found an extra output-side rotation (also in LLM), legacy global high/low rather than cross-block snake distribution, deterministic H without randomized D, and16-level rather than literal signed15 LLM W. These records and packs remain intact. A separate paper-input-only rebuild uses the same shared teacher corpus, U D H, cross-block snake, block64, signed[-7,7] original W/A and8 per-step rows; no output rotation. DuQuant/QuantVLA output restore remains as Table5/released GR00T defaults require.

The first paper pack was rejected by a real-input canonical math gate: per-output Marlin had an unscaled FP16 reduction infinity at LLM layer2.mlp.down_proj while its independent mathematical output was finite. It has no accepted task score. Distinct G128 packs repeat the EXACT same output coefficient every128 weights, preserving all codes and mathematical scales. The independent8,520-call diagnostic now reports all finite and maximum layer NMSE3.85443255e-7. Full actions still differ (NMSE0.00052–0.00897), requiring fresh100-task-episode evaluations; no success is inferred from numerical diagnostics.

Three paper-construction tests passed. Six additional rotation/static-A4 fusion tests passed, comparing the exact FP16-rounded output of the original two GPU kernels across actual DiT shapes, matrix strides, current-stream execution and CUDA Graph replay. This fusion remains an explicit isolated research prototype with no assigned task score and no automatic runtime selection.

Single-GPU smoke jobs are now exclusive too: simultaneous controllers can no longer load two large models into GPU0 during smoke. Task-journal, resume, typed Graph constant and profiler-coverage fixes from the original audit remain in force.

New command logs and immutable variants are under `results/holoq_object/paper_cfg_build`, `paper_cfg_reencoding`, `evaluation_paper_holoq`, `evaluation_paper_holoq_row_scale_overflow_v1`, and `rotation_quant_prototype`. The final experiment report must wait for fresh independent task cohorts and re-audit actual GPU traces; no final advantage conclusion is established by this addendum.
