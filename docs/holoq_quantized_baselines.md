# HoloQ-VLA: GR00T Object quantized-baseline experiment

HoloQ-VLA is the renamed Omega-QVLA, not a separate method. This experiment
uses arXiv **2605.28803v3 (11 August 2026)**:
[paper](https://arxiv.org/html/2605.28803v3),
[PDF](https://arxiv.org/pdf/2605.28803v3).

## Paper-grounded baseline checklist

The main accuracy comparison is **Table 2, PDF page 5**. Configurations are
specified in **Table 5, PDF page 11**, and the calibration/evaluation protocol
is in the appendix **Description of Baselines and Benchmarks, PDF page 10**.
Table 3, PDF page 6, measures **Pi-0.5 static storage**, not GR00T latency.
The paper ran H100; these experiments explicitly target RTX 4080 SUPER SM89.

| Method | Priority paper configuration | Object result in Table 2, for identification only | Configuration source | Algorithm / backend source |
|---|---|---:|---|---|
| GPTQ | W4A16, exclude DiT attention | 81% | block 128, damping .01, Hessian reconstruction; Table 5 | [official GPTQ](https://github.com/IST-DASLab/gptq), [Marlin](https://github.com/IST-DASLab/marlin) |
| AWQ | W4A16, exclude DiT attention | 95% | 20 scale candidates, shrink .5, 512 clipping/search tokens; Table 5 | [official AWQ](https://github.com/mit-han-lab/llm-awq), official packed AWQ GEMM/GEMV |
| OmniQuant | W4A16, exclude DiT attention | 68% | Adam .005, 200 reconstruction iterations; Table 5 | [official OmniQuant](https://github.com/OpenGVLab/OmniQuant), packed AWQ-compatible affine W4 kernels |
| SmoothQuant | W4A8, full target set | 94% | smoothing alpha .5, clipping threshold 1000; Table 5 | [official SmoothQuant](https://github.com/mit-han-lab/smoothquant), [QServe](https://github.com/mit-han-lab/omniserve) where the row-scale contract matches |
| DuQuant | W4A8, full target set | 86% | paper's SVD block 64, weight permutation/output restore, percentile 99.9; Table 5 | HoloQ/QuantVLA's GR00T port, checked against [upstream DuQuant](https://github.com/Hsu1023/DuQuant); packed Marlin when activation scales are channel-wise |
| QuantVLA | W4A8, exclude DiT attention | 92% | released defaults, ATM/OHB, preserve attention exclusions; Table 2 and released code | [official QuantVLA](https://github.com/AIoT-MLSys-Lab/QuantVLA), packed backend adapter |
| HoloQ-VLA original definition | W4A4, full target set | 92% | SVD-Hadamard block 64, zigzag permutation, LLM GPTQ .01, DiT RTN, eight channel-scale rows | this repository; preserve its frozen quantized grid/scales during optimization |
| HoloQ step-weight conversion | W4A4, full target set | not a paper method | absorb each activation-scale row into W, **requantize** W, T+1 banks | existing native S4 research engine, independently named |

AWQ and SmoothQuant are included because they are particularly strong on Object;
the experiment must not select only the weaker GPTQ/full-attention ablations.
The W4A4 SmoothQuant, DuQuant and selective QuantVLA rows are also relevant
precision-matched controls. Full-target GPTQ/AWQ variants distinguish coverage
effects from the paper's stronger attention-excluded configurations. A method
name alone is not a kernel name or a claim of original implementation fidelity.

QVLA and ActQuant appear in Tables 1/3, but not the GR00T main-table benchmark.
Their official releases were checked: [QVLA](https://github.com/AutoLab-SAI-SJTU/QVLA)
targets OpenVLA/OpenVLA-OFT/UniVLA, and
[ActQuant](https://github.com/arashakb/ActQuant) supplies OpenVLA-OFT and Pi-0.5
GGUF/GGML backends. Neither supplies a GR00T adapter/checkpoint. They must not
be replaced by an unrelated GPTQ model and relabeled as those methods. Their
GR00T support limitation is recorded separately; QuantVLA is the mandatory
VLA-specific baseline that does have a released GR00T implementation.

## Coverage and common protocol

The audited checkpoint contains **391 nn.Linear modules**. The repository's
full quantization target regex matches **180**: 84 Eagle language projections
and 96 DiT attention/MLP projections. Attention-excluded variants match **116**:
the same 84 language projections and 32 DiT MLP projections. The full set still
leaves 166 vision/other, 44 action-head auxiliary/norm projections and one
language output linear unquantized, as well as embeddings and non-linear
parameters. Exact names/shapes are in `teacher_capture/manifest.json`; “full”
does not mean all 391 policy linears.

Common checkpoint: local snapshot `f422266b016fd917127d3a0348062a5cd750df43`
of `youliangtan/gr00t-n1.5-libero-object-posttrain`. Batch 1, eight Euler steps,
action horizon 16, raw two-camera 256x256 observations, original data transforms,
first predicted action executed each simulator step, max 280 control steps,
10 settling steps. Model/activation FP16 is explicit and common; source GR00T
normally autocasts BF16. All model weights must be converted before enabling
FP16 autocast, including Eagle image/text embeddings.

Calibration: ten **actual full-precision trajectories**, one per Object task,
initial-state index 0, seed 42 with a task/frame-derived deterministic noise
schedule. Ten evenly spaced observed frames per trajectory are selected (100
total), with at most 52 input tokens per selected frame/layer/denoising step.
All methods share this teacher corpus. AWQ's explicit 512-token search subset
is drawn deterministically from that common corpus. Trajectory state hashes,
selected frame indices and raw observations are retained.

Evaluation: ten episodes per task, initial-state indices **10–19**, separate
from calibration state 0. Task/frame seeds, preprocessing, quantization range,
warmup 32 and synchronized timing must be common and recorded. The timing
boundary is complete `get_action`: preprocessing, transfers, the language
backbone, all denoising steps, rotations, activation preparation, packed GEMMs,
reductions and action postprocessing. RPC-inclusive latency is separate.
Measure inference-stage allocated/reserved peaks after warmup, plus load peak
separately. Deployment storage includes all retained nonquantized weights and
all packed weights, scales, zero points, rotations, permutations, step banks,
ATM/OHB and necessary model metadata.

Use GPU 0 only and leave the other GPU idle during timing. Use one CPU intra-op
thread and one inter-op thread consistently. Common eager execution is the
first comparison; CUDA Graph/compile eligibility and verified output behavior
must be checked for every method before any best-deployment table is produced.
Unsupported optimized paths must be labeled explicitly, without silent fallback.

## Existing-record audit

The prior 88/100 original and 90/100 step-weight runs remain intact in their
original worktrees. They have no explicit per-episode inference seed protocol,
the converted run timed two GPUs concurrently, and the original source pack's
metadata reports GPTQ damping **.05**, versus **.01** in Table 5. The repository
builder's default `one_per_task` samples one initial observation per task, not
ten complete teacher trajectories. The source pack metadata does not identify
calibration initial-state indices. The prior results are therefore historical
measurements, **not reusable primary rows** under this new protocol.

The original method and step-weight conversion must have separate names,
storage accounting and success results. Reordering packed codes alone is
different from requantizing scale-absorbed weights. Operator oracle checks and
full policy validation are both required; neither can replace task evaluation.

## Backend audit requirements

Packed weights must persist as packed codes plus metadata. Do not retain a
dense dequantized cache or represent baseline latency by full-weight
dequantization followed by `F.linear`. Marlin/AWQ use FP16 MMA with W4 storage;
this is a real packed kernel, not native S4 arithmetic. A4/A8 numerical
activation quantization feeding FP16 operands must be labeled separately from
native INT4/INT8 MMA. QServe is applicable only when its activation/weight scale
contract matches; incompatible channel scales cannot be silently changed to
per-token scales for a faster label.

Official AWQ/QServe source launches on the default stream. The local build
adapter must use PyTorch's current CUDA stream, with a documented source patch,
before graph capture or non-default-stream correctness testing. This is an
execution-port change, not a quantizer change.

Official algorithm/kernel commits, source hashes, commands, packs, numerical
checks, run logs and final JSON/CSV will be recorded alongside this document.
No speed/accuracy victory is claimed until the full primary matrix completes.

## Implementation and reproduction status (2026-10-27)

The independent branch is `experiment/holoq-quant-baselines-4080`. Pinned source
revisions are in `research/holoq_bench/sources.json`; fetch with
`python -m research.holoq_bench.bootstrap`. Runtime remains Torch 2.5.1+cu124,
Triton 3.1.0, CUDA 12.4, SM89; the official kernel sources are JIT-compiled
without replacing this working environment. `kernel_build_v2.log` and
`kernel_tests.log` retain the 20 initial arithmetic/current-stream/graph checks.
Five additional GPU checks cover the actual per-output Marlin scale contract
and native S4 activation clamp [-7,7].

All packs are newly calibrated from the same true-FP16 teacher trajectories.
The original-definition HoloQ row preserves the rebuilt canonical weight codes,
FP16 weight scales, FP32 activation tables and per-step behavior in its optimized
execution path. It is **not** a bitwise reproduction of the older HF pack, whose
calibration/protocol cannot be fully verified. Activation scale absorption and
weight requantization occur only in the explicitly separate step-weight row.
LLM per-token and DiT per-step/per-column scales are never interchanged for a
faster kernel. Eight DiT scale rows cover the eight denoising steps; the native
conversion stores eight step banks plus one fallback mean-scale bank.

The GPTQ block size is 128; deployment weight group size is separately 128 for
the GPTQ baseline, whereas original-definition HoloQ uses per-output weight
scales. Damping is .01. Marlin scale metadata is FP16, frozen before GPTQ selects
weight codes. Native step conversion uses group 128. AWQ and OmniQuant have
explicit affine W4 group-128 grids. AWQ uses the official 20-candidate linear
inspection search and clipping; OmniQuant uses official LWC plus learned input
scaling with **linear** reconstruction. These are working GR00T ports and real
packed inference, but neither is an unmodified whole-transformer-block official
adapter. Do not label their scores as exact paper-score reproduction. This
limitation remains relevant even if the result favors HoloQ.

DuQuant follows Table 5's GR00T SVD recipe; upstream DuQuant's LLM dual-transform
entrypoint is a different implementation. QuantVLA follows its released GR00T
rotation/default exclusion path plus official ATM/OHB collectors and formulas,
recalibrated on the same 100 frames. Initial exploratory DuQuant/QuantVLA packs
used another MSE mini-grid; they are preserved but **not used** for primary runs.
`packs_v2` uses the released five-candidate grid
`maxabs/7 / [0.5, .75, 1, 1.25, 1.5]`, with deployed FP16 scales frozen before
choosing codes. Rotations are computed through the equivalent small block Gram
eigensystem; sign/order and finite precision can differ from a direct SVD.

`deployment.py` creates actual loadable model directories containing packed
weights, the exact remaining unquantized state, config/experiment metadata and
a byte/hash manifest. The loader constructs the architecture from config,
strictly checks missing state keys against exactly the replaced weight/bias
keys, and never reads original floating model shards. Storage includes required
packaged tokenizer/processor assets; excludes the common software/CUDA runtime
and compiled kernel libraries. Deployment manifest bytes are added to the final
reported total. Offline teacher inputs, calibration trajectories and diagnostic
unpacked codes are not deployment files.

`run_policy.py` measures complete `Gr00tPolicy.get_action` with synchronized
host-wall timing. It includes both-camera preprocessing/tokenization, CPU/GPU
transfers, vision/language inference, all eight denoising steps, rotations,
activation quantization/packing, packed GEMMs and reductions, output transfer
and action unnormalization. It excludes simulator work, conversion of raw LIBERO
observations, RPC (in-process evaluation) and deterministic per-call RNG reset.
Warmup is 32, fixed replay is 100 held-out first observations repeated three
times. Rollout reports every get_action separately and uses only the first
predicted action per simulator step. Any new graph capture is marked cold and
reported separately from steady-state latency. Memory peaks reset after warmup;
rollout memory resets per episode. Load-stage peaks are recorded separately.

`submit.py` serializes all GPU work and records both GPUs' utilization, memory,
power and clocks every five seconds. It preflights all methods, then offers the
same head-only CUDA Graph optimization to all. If every valid method passes,
Graph is the common setting; otherwise eager is common and supported Graph
variants receive their own independent rollout scores. No method uses compile
in this comparison. Eager replay timings are also retained for all methods.
Graph output drift and eager self-repeat drift are saved, not called bitwise
equivalent. Every execution mode used for a success claim is actually evaluated;
no score is copied between modes.

Calibration and held-out replay artifacts are under `results/holoq_object/`.
The initial calibration attempt and initial strict dtype/graph-check failures
are retained. Primary commands and run-specific source commits are stored in
queue/metadata JSON. A final report must distinguish paper-prescribed recipes,
the explicit port/storage dtype choices above, and these common deployment
settings; none imply a same-W/A comparison between W4A16 and W4A4.
