# GR00T per-step W4A4 performance research

Measured on 2026-10-27, RTX 4080 SUPER (SM89), PyTorch 2.5.1+cu124,
Triton 3.1.0, CUDA 12.4. Base revision:
`a6b227397cc6eccd17a7170d8958c5ed289e25ce`.
The research checkout is `/home/xinyu/omegaqvla-perstep-hp`; the independently
running original LIBERO checkout is `/home/xinyu/omegaqvla-perstep-4080`.
The initial operator study did not modify runtime files. Subsequent user-requested
LIBERO integration adds an opt-in `GR00T_PERSTEP_ENGINE=native_s4` in this checkout
only, plus full-policy/rollout timing. The original evaluation checkout remains
unchanged.

## Findings

There are two independent opportunities: execute the existing integer formats
with actual S4 tensor instructions, and stop computing known block rotations as
dense matrices. Neither operator benchmark establishes policy latency or LIBERO
success rate. The native integer route preserves the **converted** methods;
it does not remove their differences from the original Omega quantization grid.

### Native S4 instead of unpacked INT8

The existing Triton kernel quantizes the same input in every output-N tile,
unpacks weights to INT8, and converts group accumulators to FP32 inside a long
K-group loop. Its fixed output tile is 16x32. Larger tiles alone did not solve
the layout/register costs in the diagnostic experiments.

The new research implementation does the following:

1. `backend.py:pack_activation` quantizes once with the same FP32 division and
   round-to-nearest rule and packs signed nibbles directly into MMA fragments.
2. `backend.py:pack_weight` reorders existing packed codes offline. It does not
   requantize them. Weight scales are transposed to contiguous `[groups,N]`.
3. `native.cu:native_group_kernel` uses
   `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`. Integer group sums receive
   the existing FP32 scales; `group_int` also applies its activation group scale.
4. `backend.py:from_packed` optionally reduces split-K partials. Split-K and
   number of N warps are selected by shape. Runtime quantization, packing and
   split reduction are all included in the reported total timings.

SASS confirms `IMMA.16864.S4.S4`; compilation reports 39 registers and no spills.
The fragment mapping comes from the [NVIDIA PTX 8.4 specification](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html).
It is an SM89-tested research kernel, not a portable production backend.

Nineteen GPU tests passed (three input dtypes, groups 64/128/256, odd tails,
step selection including fallback, multiple launch configurations, and an exact
integer arithmetic case). Additional real-input validation covered both integer
formats, all 96 DiT layers, eight steps, M=32/128 and split-K=1/4: **6,144
comparisons passed**, maximum NMSE against the existing Triton kernel
`2.5507311310324875e-14`. All **3,072 split-K=1 comparisons matched every
output value exactly**. This compares the same converted weights/scales, not
converted weights against the original source pack.

### Measured residual GEMM performance

Three real held-out layers, step 0, M=32/49/128, both integer formats: 18 cases.
Both paths use CUDA Graphs. The reference caches its BF16 weight before timing;
the original exporter benchmark's per-call FP16-to-BF16 weight conversion is
therefore removed. The native total includes activation preparation/reduction;
offline weight packing is excluded. Rotation, bias and low-rank work are excluded
from both residual-only paths.

Representative `step_weight` measurements, microseconds:

| M,N,K | Cached BF16 reference | Existing integer kernel | Native S4 total |
|---|---:|---:|---:|
| 32,1536,1536 | 16.90 | 36.62 | 5.32 |
| 32,1536,6144 | 33.17 | 142.92 | 8.53 |
| 128,6144,1536 | 49.17 | 191.84 | 17.64 |
| 128,1536,6144 | 38.01 | 289.83 | 20.28 |

Across all cases native is **1.748–3.889x** faster than cached BF16 with warm
repeated operands, and **6.88–17.81x** faster than the existing integer kernel.
After a 128 MB L2 flush, native versus BF16 is **1.00–2.63x**. The smaller cold
gains matter: a policy traverses many weights, so warm microbenchmarks alone
would overstate the expected application improvement. These are single-run
benchmarks, not confidence intervals, and the first GPU was running evaluation.

The reference is a performance comparator, not an accuracy-equivalent oracle:
the converted formats already differ from the original source quantization.
Native-versus-converted-oracle maximum NMSE in the tuning sweep was
`1.3751e-13`. Converted-oracle-versus-source-reference NMSE in these 18 cases
ranged from `0.000785` to `0.03323`. Full-export accuracy evidence is in the
original benchmark manifest and must be assessed separately.

### Dense rotations are a separate major cost

The downloaded GR00T Object source contains **276 dense rotations**:
84 LLM input rotations, 96 DiT input rotations and 96 DiT output rotations.
Every matrix was proved to be a row-permuted block diagonal matrix with block
size 64. Off-block coefficients are exactly zero; each block owns exactly 64
unique rows. All original coefficient values are retained in the factorization.
All output rotations have identity permutation.

| Representation | Rotation storage |
|---|---:|
| Source FP16 dense matrices | 4,739,563,520 bytes |
| Factored FP16 blocks plus input permutations | 89,587,712 bytes |
| Current constructor's FP32 dense buffers | 9,479,127,040 bytes |
| Corresponding FP32 blocks plus permutations | 175,570,944 bytes |

This saves **9.304 GB (8.665 GiB)** in rotation buffers under the current FP32
constructor storage convention. It is not a measured whole-policy peak-memory
delta: a later module dtype conversion, retained tensors and allocator state can
change that delta. The factored complete source pack is 2,236,316,250 bytes,
versus the original 6,886,223,590 bytes. Weight tensors and per-step tables are
unchanged. Existing `GptqLinear` already accepts these block/perm fields.

Real-input rotation measurements cover four first-block DiT input rotations,
M=32/49/128, with and without BF16 autocast (24 cases). For FF-down K=6144,
M=32, the existing dense BF16 rotation takes **119.43 us**, versus **9.13 us**
for gather plus existing block BMM; cold timings are **163.84 vs 21.50 us**.
Smaller K cases can be slower with block BMM, so shape dispatch or a fused
gather/rotation kernel remains useful.

**Coefficient preservation is not bitwise forward equivalence.** Changing
reduction order changes some BF16 outputs. The measured FF-down rotation has
NMSE up to `7.83e-6`, and the maximum downstream A4 code mismatch fraction over
the eight source scale rows is about **0.126%** in these cases. This is a
rotation-input comparison, not an all-layer/all-step policy equivalence test.
Full action and LIBERO checks are required before adopting it by default.
The existing baseline remains untouched.

Do not fold a general input rotation across activation quantization: in general,
`Q(xR)` cannot be replaced by `Q(x)` followed by a weight-only transformation.
Instead retain online `x[:,perm] @ blocks`, then smooth scaling, then the
original per-step A4 operation. A fused kernel can combine gather, block
rotation, smooth scaling, A4 and fragment packing; low-rank consumers, if
present, still need the unquantized smoothed activation.

## Which method does this accelerate?

For original per-time, per-channel scales, the residual has the form

`y_n = sum_k qA_t,k * a_t,k * W_n,k`.

An INT4 MMA sums products of integer codes; a different real factor for every
K channel cannot in general be applied once after that sum. This is the
constraint behind the existing converted algorithms:

* `step_weight` retains original channel A4 thresholds but absorbs their scales
  into weights and requantizes one W4 bank per step, plus fallback. Native S4
  changes neither its codes nor scales; **T+1 bank storage remains**.
* `group_int` shares a scale within each K group, enabling integer accumulation
  with one scale per group, but **changes the original channel A4 grid**.
* `fused_float` retains channel A4 thresholds while dequantizing within floating
  MMA. This prototype does not implement a faster `fused_float` path.

If exact original Omega channel thresholds and original weight grid are required,
prefer optimizing block rotation, quantization-once, and a floating-input W4
kernel that loads `qA*a_t,k` into its MMA operand. Export original W4 codes and
scales at build time rather than reconstructing them by requantizing the saved
dense FP16 residual. The downloaded source stores `weight_res_q` but no original
integer-code/group-scale metadata. A fast INT4-code kernel for the converted
formats is not proof of an exact original-method kernel.

The official [Marlin](https://github.com/IST-DASLab/marlin) kernel was also built
and tested at `1f25790bdd49fba53106164a24666dade68d7c90`. It is FP16xINT4:
A4 numerical values are represented as FP16 operands and scales are FP16.
In this adaptation it was slower than native S4 in all 18 cases and introduced
oracle NMSE up to `0.001293`. The benchmark verified recovered weight codes,
but did not isolate scale rounding from internal reduction errors. It is not a
drop-in accuracy-equivalent solution under the present FP32-scale contract.

## Integration order

1. Keep the original source pack and baseline results. Add explicit opt-in
   factored rotation packs and verify all-layer pre-A4 values, A4 codes, action
   outputs, memory and matched LIBERO episodes. Numerical thresholds must be
   chosen using those measurements rather than assuming coefficient equality
   guarantees forward equality.
2. Add a native S4 engine only to the two integer converted formats. Prepack
   every step/fallback weight bank and transpose scales once before warmup.
   Preserve step selection, group scaling, dtype and device behavior. Reject
   unsupported group sizes (group 32 is unsupported by this prototype) and use
   an explicit existing-engine fallback for other hardware/shapes.
3. Freeze shape launch choices from a separate tuning set; avoid searching
   during inference. Use split-K when small M/N leaves too little parallelism;
   reduce it when extra partial-output traffic dominates. Current timings are
   tuned on the same representative cases and need independent verification.
4. Profile the full policy before spending effort on asynchronous weight loads,
   multi-M-tile weight reuse, double buffering or persistent scheduling. Use
   fusion to eliminate intermediate rotation/quantization buffers and launches
   only after checking register pressure and numerical behavior.
5. Capture CUDA Graphs/compile only after stabilizing buffers and step selection.
   Compare equally cached/compiled/graphed baselines, report p50/p95 policy
   latency, real peak allocated/reserved memory, pack size and task success.
   Include rotations, model components and host overhead. Operator speedups do
   not multiply into an end-to-end speedup.

## Reproduction and artifacts

Source: `ucmp137538/Omega-QVLA-GR00T-N1.5-LIBERO-W4A4`, revision
`01252878485fa9295b53ab833eb13d564f6911ad`, `gr00t_object/quantized.pt`.
SHA256: `0fa5f367e7ed6e7be25efce25fce967078cb5dc86b39ae5dd53b9dac4ec107ba`.
Held-out captures and converted packs come from the original checkout's
`results/perstep_full_object_20261027*` outputs, with capture offset 10.

Run from this checkout using the existing GR00T environment. Local defaults in
the scripts point to the source/capture/export paths above. Marlin is an optional
independent source checkout in `research/marlin`, built in place, not installed
over the evaluation environment.

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PATH="/home/xinyu/.venv-vllm/bin:$PATH"
PY=/home/xinyu/miniconda3/envs/groot_test/bin/python
"$PY" -m pytest -q research/native_int4/test_native.py
"$PY" research/native_int4/bench.py --output results/native_int4_20261027/recheck.json
"$PY" research/native_int4/validate_all.py --output results/native_int4_20261027/recheck_validation.json
"$PY" research/native_int4/bench_rotations.py --output results/native_int4_20261027/recheck_rotations.json
```

The extension needs Ninja on PATH (this host used `/home/xinyu/.venv-vllm/bin`).
`audit_rotations.py --output audit.json --save-pack factored.pt` performs the
CPU proof/export and refuses to overwrite an existing output pack. Run it
before `bench_rotations.py` if the factored pack is absent.

Artifacts in `results/native_int4_20261027/`:

* `benchmark.json`: 18 cases with all 16 launch candidates, oracle errors,
  independent timing recheck, cold timing and optional Marlin comparison.
* `validation_all.json` / `.log`: all 6,144 real-input comparisons.
* `tests.log`, `build.log`, `native.sass`: tests, compiler output and machine code.
* `rotation_audit.json`: all 276 structural proofs and byte accounting.
* `object_block_rotations.pt`: source weights/tables with factored rotations.
* `rotation_benchmark.json`: all 24 rotation cases and A4 mismatch fractions.

## Full-policy integration and requested LIBERO run

`PerStepBackend.forward(engine="native_s4")` now calls
`research/native_int4/backend.py:cached_linear`. The adapter preserves leading
batch dimensions, lazily packs each selected weight bank once, retains FP32
scales and selects a fixed shape-based launch. It explicitly requires SM89 and
groups 64/128/256. It never silently falls back to BF16. Existing `triton` and
`torch_reference` routes remain available. The model's rotation, smoothing,
low-rank, output rotation, bias and step-context code remain on the existing
full-policy path.

Runtime-dispatch tests plus existing backend tests: **42 passed**. A full-policy
smoke run with two warmup calls confirmed **96 active native layers and 1,536
native calls**, with peak PyTorch allocated memory **9,274,808,320 bytes**.

The unmodified dense-rotation `step_weight` pack exceeded the 16 GB card during
model placement, before evaluation. The full run therefore uses
`step_weight_blockrot.pt`: the existing converted integer banks and scales are
unchanged, and the 276 rotations use the coefficient-preserving block
representation above. Pack size: **3,538,568,974 bytes**. Both the converted
weight-grid differences and rotation reduction-order differences apply.

The requested protocol is LIBERO Object, ten tasks, ten trials per task,
initial-state indices 10 through 19, eight denoising steps, using GPU 0 for
tasks 0/2/4/6/8 and GPU 1 for tasks 1/3/5/7/9. Results are in
`results/native_int4_20261027/libero_object_step_weight_native_block/`.

```bash
"$PY" research/native_int4/run_libero.py \
  --pack results/native_int4_20261027/step_weight_blockrot.pt \
  --output results/native_int4_20261027/libero_object_step_weight_native_block \
  --gpus 0,1
"$PY" research/native_int4/summarize_libero.py \
  results/native_int4_20261027/libero_object_step_weight_native_block
```

The runner refuses to reuse an existing output directory. Server JSONL records
measure synchronized full `policy.get_action` wall time after 32 warmups, including
image preprocessing, LLM, all denoising steps, rotations and action postprocessing;
they exclude IPC serialization. Client episode JSONL records include preprocessing
and local RPC, rollout elapsed time, initial-state index, success and exceptions.
`summarize_libero.py` requires exactly 100 episodes, ten per task, no evaluation
exceptions, matching timing counts, and logs proving all 96 native layers were
active. It produces `timing_summary.json` and `timing_summary.md`.

`replay_policy.py` measures the full policy on one identical saved observation
per task with fixed seeds, enabling matched-rotation comparisons of the native
engine, old Triton engine and original residual path. This latency replay is
separate from task-success evaluation.

## Completed full-policy results

The requested run finished successfully: **90/100 successes**, exactly ten
episodes for each of ten Object tasks, with no recorded inference/evaluation
exceptions. All 96 target DiT layers used native S4; the full model retained
eight denoising steps. Wall time including startup: **45.09 minutes on two
GPUs**. Earlier original dense-pack W4A4 baseline: **88/100** on the same
initial-state range. This is a two-episode observed difference, not evidence of
a statistically established accuracy improvement or original-method equivalence.

| Task | Original dense baseline | Native step_weight + block rotations |
|---|---:|---:|
| alphabet soup | 10/10 | 10/10 |
| cream cheese | 7/10 | 7/10 |
| salad dressing | 10/10 | 10/10 |
| bbq sauce | 7/10 | 7/10 |
| ketchup | 10/10 | 9/10 |
| tomato sauce | 10/10 | 10/10 |
| butter | 8/10 | 9/10 |
| milk | 10/10 | 10/10 |
| chocolate pudding | 8/10 | 10/10 |
| orange juice | 8/10 | 8/10 |
| Total | 88/100 | 90/100 |

The 15,859 real rollout `get_action` calls averaged **301.06 ms**, p50
**266.84 ms**, p95 **480.51 ms**. Client-side preprocessing plus RPC averaged
**304.40 ms**. Peak PyTorch allocated/reserved memory was **8.638/8.766 GiB**.
Summed rollout time across both shards was 5,028.78 seconds; that sum is not
parallel wall time. Latency varied substantially over the run; per-task values
are in `timing_summary.md`. Do not compare these concurrent-rollout averages
directly with the single-GPU replay values below.

After evaluation, ten saved rollout observations (one per task) were replayed
on GPU 0, each with 32 startup warmups and ten measured repeats. Native and old
integer engines were alternated within one loaded model, with the same noise
seeds and converted weights. Original-source controls used separate processes
on the same observations and seeds. All reported times cover full
`get_action`, not just GEMM.

| Full-policy replay configuration | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| Original W4A4, dense rotations, BF16 GEMM | 257.10 | 257.10 | 257.32 |
| Original W4A4, block rotations, BF16 GEMM | **133.75** | **133.25** | **136.49** |
| Converted step_weight, block rotations, old Triton integer engine | 201.04 | 200.92 | 201.80 |
| Converted step_weight, block rotations, native S4 engine | 155.83 | 155.30 | 159.02 |

**The native engine is not the fastest full-policy configuration.** It is
1.29x faster than the old integer engine (22.49% lower latency), but **16.51%
slower than the block-rotation BF16 original-weight path**. Versus the original
dense-rotation pipeline it is 1.65x faster; block rotation alone yields 1.92x
in this replay. Thus the earlier operator gains do not justify making the
native engine the default fastest backend. The old integer and block-only BF16
configurations do not have new 100-episode task scores from this experiment.

Original dense/block control peak allocations were **14.129/5.473 GiB**. The
native step-weight format uses multiple weight banks and its present adapter
keeps original and reordered packed banks. Rotation savings must not be
attributed entirely to the integer kernel.

The fixed-seed old/native full-action comparison had maximum NMSE
**0.0002460421** across the tested observations/seeds. Split-K changes FP32
group-sum ordering, and iterative quantized denoising can amplify tiny operator
differences. The action outputs are therefore not bitwise equivalent, despite
the very small operator errors above. Full-policy validation matters.

Completed artifacts:

* `libero_object_step_weight_native_block/merged_summary.json`: success counts.
* `libero_object_step_weight_native_block/timing_summary.json` / `.md`: verified
  complete trials, real rollout timings and memory.
* `libero_object_step_weight_native_block/timing_controls.json`: all three
  successful replay runs, including paired old/native engines.
* `libero_object_step_weight_native_block/replay_*.json` / `.log` /
  `.actions.pt`: timing samples and saved fixed-seed actions.
* `libero_object_step_weight_native_block/source_provenance.json`: runtime file
  hashes and base revision; `run_manifest.json`: protocol and exit status.
