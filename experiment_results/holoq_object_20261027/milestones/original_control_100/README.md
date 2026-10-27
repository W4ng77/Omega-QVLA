# First completed100: un-fused controlled HoloQ rebuild

This is a progress milestone, **not the completed optimized-method comparison**.
The selected optimized backends and required baseline100 rows are still running.

The canonical paper/repository-recipe rebuild `holoq`, head-only CUDA Graph,
scores **72/100** on the new independent Object states10–19. Per-task successes:
`[8, 8, 8, 5, 5, 8, 7, 9, 8, 6]`.

Real rollout includes19,398 complete get_action calls: mean87.668914ms,
p50 84.323846ms, p95 108.027234ms. Fixed-input replay is a distinct measurement
(mean85.363309ms; use its source JSON for exact values). This row retains the
original BMM block-rotation path, not the selected fused-rotation backend.

This pack was newly calibrated from the shared ten FP16 teacher trajectories;
it is not the historical HF pack and is not a reproduction of the paper92%
under identical states/protocol. Historical88/100 and90/100 remain excluded.
No conclusion about the optimized implementation's retained success follows
from this control alone. Dense canonical-math checks are queued for accuracy
diagnosis only and do not create a floating speed baseline.

The JSON, per-episode records and compressed raw calls/logs are actual results.
Ten new, hash-verified smoke episodes plus90 new executions form the100 unique
task/state pairs; `reused_smoke.json` records that audit. No historical episode
is reused. `audit_first100.json` verifies17 completed primary replay/rollout
rows, including this100, with zero errors and GPU1 idle. Full final deployment
memory accounting also merges the explicitly documented smoke peak window.
