# Frozen Object work package

Branch: `experiment/holoq-paper-rebuttal-4080`. Audits and evidence-qualified drafts are public at `experiment_results/rebuttal_20261028/` and this directory. Large source packs, raw observations and full-token activation shards remain local. SHA256 identifiers are recorded in `configuration.json`; they are not silently replaced by regenerated data.

Current host worktree: `/home/xinyu/holoq-vla-rebuttal`.
Python: `/home/xinyu/miniconda3/envs/groot_test/bin/python`.
Data: `results/rebuttal_object_20261028`.

The data directory contains read-only symlinks:

- `source_teacher_trajectories` → previous `results/holoq_object/calibration_fp16_init0_v2`: 10 actual trajectories, 9 successes and 1 failure, 10 uniform frames each. All are retained. This is a disclosed reconstructed calibration protocol, not recovered proof of historical submission calibration.
- `source_existing_capture` → previous `results/holoq_object/teacher_capture`: source weights and original checkpoint path only. The old 52-token activations are **not** reused by the new capture. Every target source weight is verified against the newly loaded checkpoint.
- `source_previous_results` → previous `results/holoq_object`: historical-pack provenance, exact factored rotation representation and frozen historical first-call LLM activation scales.

Run these stages sequentially; do not start independent timed jobs in parallel:

```bash
cd /home/xinyu/holoq-vla-rebuttal
bash scripts/run_rebuttal_object_build.sh \
  > results/rebuttal_object_20261028/build_submission_contract_v2.log 2>&1
bash scripts/run_rebuttal_object_queue.sh \
  > results/rebuttal_object_20261028/object_primary_queue.log 2>&1
```

The builder owns `/tmp/holoq-vla-policy-timing.lock`. It directly constructs/fixes input-only block64 UDH, captures all tokens from the 100 fixed teacher calls, constructs the signed15/per-output frozen grid and exports one packed artifact. Checkpoints, sources, configuration and per-frame shards carry identities. It refuses configuration changes in the same output directory. Incomplete frames are rerun, and only complete per-frame journals are reused. No seed/sign choice uses evaluation success.

The Object queue may be launched while the build is running: it waits for that lock and refuses to continue if export failed. It freezes evaluation states20..29 and verifies their hashes are disjoint from the recorded calibration states. It does not score episodes during observation collection. New comparisons consistently use env.seed(0) per episode, 10 settling steps, 256px raw cameras transformed to224px, batch1, eight Euler steps, horizon16 and first-action execution each simulator step. The reconstruction protocol is disclosed; identity with unspecified historical-paper details is not claimed.

Before any scored rollout, `validate_policy.py` runs the same loaded model with the same packed codes/scales/rotations in the FP32 equation oracle and actual packed backend. It checks 8,520 local linear executions, eight DiT step outputs per observation, actual initial-noise equality, final Euler states, final actions and Graph/eager behavior. Predetermined engineering limits are local NMSE1e-4, final-action NMSE0.01 and Graph/eager action NMSE1e-5. These are bug guards, not evidence of task-success equivalence. Failure stops the queue for diagnosis and retains evidence.

After the gate, the queue runs historical implementation, reference and actual packed deployment on100 episodes each. Reference/packed additionally run smoke checks. Interrupted rollouts retain only completed episodes; incomplete call fragments are archived and rerun from the frozen state/seed. Pack, runtime implementation, execution math sources, states, Graph protocol and validation identity must match before resuming. The queue then measures unprofiled packed replay,100 observations×3 repeats. CPU preprocessing, H2D, vision/language, all eight steps, transforms, A quantization/reconstruction, GEMM, D2H and action unnormalization are included; simulator, raw-observation conversion, RNG resetting and cold Graph capture are excluded. Real rollout timing remains separate.

The packed runtime uses packed W4 storage and numerical A4 quantization with transient floating activations and on-chip floating weight decode. It does not use native INT4×INT4 MMA and does not nibble-pack the activation buffer. The FP32 oracle is an accuracy reference, never an optimized speed baseline. Layer tests are not substituted for complete-policy results.

`primary_queue/status.json` records commands, source revision, PIDs and exit codes. A five-second sampler records both GPUs and compute processes. Timings must be checked against these samples for interference; ownership of a local lock alone does not prove system-wide exclusivity. The queue archives raw JSON and compressed logs under `experiment_results/rebuttal_20261028/primary` when it exits. A completed primary package is explicitly **not** a completed rebuttal: baseline comparability, cumulative profiling, memory/storage splits and Long per-step ablations still require their own evidence review. No score recovery or acceleration is assumed.
