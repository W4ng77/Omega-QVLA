# AWQ selective: completed100-episode milestone

This is one completed baseline, not the final full comparison. All100 pairs use the shared new protocol; no historical score is reused.

Qwen3 grouped AWQ +DiT linear port, real packed AWQ W4A16;116 targets (84LLM +32DiTMLP),64DiTattention projections keptFP16.

Success: 93/100. Per-task: [10, 8, 9, 10, 9, 10, 7, 10, 10, 10].

Rollout full-policy latency: {'count': 15383, 'mean_ms': 63.55247400653339, 'p50_ms': 58.907754020765424, 'p95_ms': 77.25518122315407}. Fixed replay is separately saved. The neighboring audit verifies18 completed timing/rollout rows, including both completed100-episode cases, with zero errors.

Calibration/source/port limitations are in the experiment protocol.100-episode differences are descriptive, not a claim of statistical improvement.
