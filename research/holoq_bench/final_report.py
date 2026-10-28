"""Render the completed, audited experiment without filling missing scores."""

import argparse
import json
from pathlib import Path


def table(rows, fastest):
    lines = [
        "| Method / case | Backend / execution | W/A | Actual scope | Success | Replay mean / p50 / p95 ms | Rollout mean / p50 / p95 ms | Peak allocated / reserved GB | Deployment GB | Speedup vs fastest quant baseline |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        assert row["episodes"] == 100, (row["case"], row["episodes"])
        replay = " / ".join(f"{row[k]:.2f}" for k in ("policy_mean_ms", "policy_p50_ms", "policy_p95_ms"))
        rollout = " / ".join(f"{row[k]:.2f}" for k in ("rollout_mean_ms", "rollout_p50_ms", "rollout_p95_ms"))
        memory = " / ".join(f"{row[k] / 1e9:.3f}" for k in ("peak_allocated_bytes", "peak_reserved_bytes"))
        speedup = f"{fastest['policy_mean_ms'] / row['policy_mean_ms']:.3f}x" if row["case"].startswith("holoq") else "—"
        lines.append(
            f"| {row['method']} / `{row['case']}` | {row['variant']} / {row['execution']} | {row['W/A']} | "
            f"{row['scope']} | {row['successes']}/100 | {replay} | {rollout} | {memory} | "
            f"{row['deployment_bytes'] / 1e9:.3f} | {speedup} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = args.runs / "report"
    audit = json.loads((report / "audit.json").read_text())
    assert not audit["errors"], audit["errors"]
    data = json.loads((report / "main.json").read_text())
    selected = json.loads((report / "best.json").read_text())
    common = [row for row in data["rows"] if row["common"]]
    best = selected["rows"]
    baseline_best = [row for row in best if not row["case"].startswith("holoq")]
    fastest = min(baseline_best, key=lambda row: row["policy_mean_ms"])
    highest_score = max(row["successes"] for row in baseline_best)
    strongest = min((row for row in baseline_best if row["successes"] == highest_score),
                    key=lambda row: row["policy_mean_ms"])
    ours = [row for row in best if row["case"].startswith("holoq")]
    assert len(ours) == 2, [row["case"] for row in ours]
    lines = [
        "# HoloQ-VLA / quantized baselines on RTX 4080 SUPER",
        "",
        "HoloQ-VLA and Omega-QVLA name the same method. These are actual complete-policy measurements, not isolated GEMM estimates.",
        "",
        f"The fastest measured quantized deployment is **{fastest['case']}**, with full-policy replay mean "
        f"**{fastest['policy_mean_ms']:.2f} ms** and **{fastest['successes']}/100** Object successes.",
        f"The highest observed baseline success count is **{highest_score}/100**. Among tied deployments, "
        f"**{strongest['case']}** has the lowest replay mean, **{strongest['policy_mean_ms']:.2f} ms**. "
        "This descriptive ranking does not establish a statistically superior method.",
    ]
    for row in ours:
        ratio = fastest["policy_mean_ms"] / row["policy_mean_ms"]
        result = "faster" if ratio > 1 else "slower"
        lines += [
            f"`{row['case']}`: **{row['policy_mean_ms']:.2f} ms**, **{row['successes']}/100**, "
            f"**{ratio:.3f}x** relative to that baseline ({result})."
        ]
        lines += [f"Against the highest-observed-success baseline `{strongest['case']}`, this deployment has "
                  f"**{strongest['policy_mean_ms'] / row['policy_mean_ms']:.3f}x** replay speed and "
                  f"**{row['successes']}/100 versus {highest_score}/100** successes."]
    lines += [
        "",
        "These 100-episode success differences are descriptive. They do not establish statistical improvement, and kernel speed alone does not establish retained task success.",
        "",
        "## Common execution and each method's best tested deployment",
        "",
        "All methods received eager and the same head-only CUDA Graph path. The primary cohort uses the common supported Graph setting. No method uses torch.compile. The separately frozen backend selection minimizes full-policy replay mean without reading task scores; every selected changed execution receives its own 100 episodes. This is the best among the explicitly tested backends, not a claim of globally optimal implementations.",
        "",
        "The first table includes common configurations and the independently scored changed backends. `best.csv` identifies the exact selected subset; timing-only losing candidates have no borrowed success score.",
        "",
        *table(common, fastest),
        "",
        "### Selected deployments",
        "",
        *table(best, fastest),
        "",
        "### Speedup against each quantized baseline (selected deployments)",
        "",
        "| Quantized baseline | " + " | ".join(row["case"] for row in ours) + " |",
        "|---|" + "---|" * len(ours),
    ]
    for base in baseline_best:
        lines.append(
            "| " + base["case"] + " | "
            + " | ".join(f"{base['policy_mean_ms'] / row['policy_mean_ms']:.3f}x" for row in ours) + " |"
        )
    lines += [
        "",
        "Speedup = baseline full-policy mean / HoloQ full-policy mean; below 1 means HoloQ is slower. `evaluation/report/speedups.csv` also gives rollout ratios for every measured HoloQ/backend against every quantized baseline.",
        "",
        "## Successes per Object task",
        "",
        "Task IDs map to the exact evaluated language descriptions in `evaluation/report/tasks.json`; `per_task.csv` also includes names and the ten-trial denominator.",
        "",
        "| Case | " + " | ".join(f"Task {i}" for i in range(10)) + " | Total |",
        "|---|" + "---|" * 11,
    ]
    for row in best:
        values = json.loads(row["per_task_successes"])
        assert len(values) == 10 and sum(values) == row["successes"]
        lines.append("| " + row["case"] + " | " + " | ".join(f"{v}/10" for v in values) + f" | {sum(values)}/100 |")
    lines += [
        "",
        "## What is and is not reproduced",
        "",
        "The paper's main LIBERO table is Table 2 (PDF p5); configurations are Table 5 (p11), and the shared ten-trajectory calibration protocol is in the baseline appendix (p10). Table3 is Pi0.5 static storage; appendix Table8 (p13) measures complete Pi0.5 sample_actions on RTX4080 SUPER using Nunchaku INT4, ten denoising steps and batches1/8/16/24. It is not a GR00T quantized-baseline latency comparison. Main accuracy experiments used H100; this report uses RTX 4080 SUPER. The exact paper/source checklist and port differences are in [the experiment protocol](../../docs/holoq_quantized_baselines.md) and pinned `research/holoq_bench/sources.json`.",
        "",
        "This is a controlled deployment comparison using paper recipes, not an assertion of exact upstream reproduction for every baseline: GPTQ uses the official solver and real Marlin; AWQ uses official search/clipping plus a Qwen3 grouped adapter and DiT linear adapter; OmniQuant is only the LWC/scale linear reconstruction port, not full block/LET OmniQuant. SmoothQuant uses alpha .5 with real QServe W4A8; A4 uses a numerical signed-4 grid in an INT8 container. DuQuant follows the paper's GR00T SVD recipe, which differs from upstream dual-transform LLM code. QuantVLA includes its GR00T attention exclusions, ATM and OHB. QuantVLA/DuQuant global corpus-percentile collection differs numerically from the released batchwise/lazy collector. GR00T adapters are unavailable in the checked QVLA and ActQuant releases; they are recorded N/A rather than simulated under another method's name.",
        "",
        "Full target coverage means 180 = 84 LLM + 96 DiT linears; selective means 116 = 84 LLM + 32 DiT MLP, excluding 64 attention projections. All 391 nn.Linear names and shapes are audited. Both leave 166 vision/other, 44 action auxiliary/norm and one language output linear, plus embeddings/custom matrix banks, in FP16. W4A16 and W4A4 are different precision configurations, and selective versus full are different coverage settings. Their deployment speeds are comparable, but they are not identical precision/scope experiments.",
        "",
        "The original-grid HoloQ candidate preserves canonical packed codes, weight/activation scales and the eight per-step rows from the newly shared calibration; it is a controlled paper/repository-recipe rebuild, not bitwise the historical HF pack. Its activation contract follows paper Eqs6–8 (per-token LLM, per-step/channel DiT, signed[-7,7]); the LLM weight solver retains official GPTQ's16-level grid, and the Hadamard/sign convention follows the released deterministic helper. These construction differences constrain paper-reproduction claims. A column activation scale lies inside the GEMM reduction, so ordinary token/output scalar INT4 epilogues cannot express it. The Marlin path keeps this scale contract and uses real packed W4 storage with FP16 MMA. Native LLM S4 is used only where token/output scaling factors correctly; fused rotation changes FP16 arithmetic order and is independently task-evaluated.",
        "",
        "The separately named step_weight conversion absorbs the activation scale into each step's weight and requantizes W with group 128. It stores eight step banks plus a fallback bank, additional scales and runtime integer reorder caches. It is not strictly equivalent to the original method. Actual deployment and peak memory above include these costs; no generic memory-saving claim follows from the W4 label.",
        "",
        "## Measurement and provenance",
        "",
        "Additional mature-backend checks used the official Nunchaku INT4 core. Its standalone arithmetic tests and packed-launch traces passed. However, original/conversion Nunchaku hybrids reused the invalid legacy rotation plugin, so their84.23/79.55ms policy figures are withdrawn as functional comparisons. Rotation-free SmoothQuant-A4 Nunchaku measured70.11ms and lost to QServe; no success score is borrowed. The converted Nunchaku allocation6.27GB is historical diagnostic data, not a validated-method result. Initial padding-metadata failures and their correction remain archived.",
        "",
        "The initial fused-rotation backend hardcoded contiguous R addresses while actual output R tensors are transposed views. It completed0/100 and is excluded from all valid main comparisons. Its80.13/74.19ms claims are withdrawn. The stride-correct v2 has an explicit new tag, new packs and independent timing/task records. Its functional gate compares every real packed projection with the nonfused operator on the same inputs and ten full-policy outputs before selection. The old code, failure logs and100 episodes remain in the regression archive. Small layer/action drift remains insufficient to establish task equivalence, so each selected v2 path is independently scored.",
        "",
        "An accuracy-only frozen-grid diagnostic compares8,520 real layer calls across ten held-out first observations and all eight denoising steps. Maximum layer NMSE is6.01e-7, while complete action NMSE is0.0012–0.0074. This is not bitwise policy equivalence or a success guarantee, and does not isolate the whole success gap. Dense oracle weights exist only in that diagnostic process; no floating speed baseline is created. The adjacent validation/extra artifacts contain the exact source hashes, comparisons and candidate timings.",
        "",
        "Checkpoint revision f422266b016fd917127d3a0348062a5cd750df43; batch1; raw two-camera 256x256 ->224 preprocessing; eight Euler denoising steps; action horizon16, executing index0 each simulator step; ten settling steps; max280 control steps. Calibration uses ten successful FP16 teacher trajectories at initial state0, 100 sampled frames. Evaluation uses independently hashed states10–19 for each task. Per-call seed is 42 + task_id*10000 + initial_state_index*1000 + control_step. No historical 88/100 or 90/100 result is reused. Verified new smoke state10 episodes are reused exactly once within their own 100-row cohort, with identical frozen operators, pack, state hash, seed and execution mode; remaining90 are actually executed.",
        "",
        "Every timing uses GPU0 exclusively with GPU1 idle, synchronized host-wall time around the full Gr00tPolicy.get_action API, 32 warmups and the same input/noise schedule. Final replay is one contiguous exclusive block across all primary and selected deployments, after task evaluation, to control wall-clock drift. Prior replay data stays archived as replay_prior. Replay is the 100 held-out first observations repeated three times; rollout measures every real policy call, with cold captures separately excluded/count-recorded. Included: CPU crop/resize/tokenization/normalization, host/device transfers, vision/language, all8 denoising steps, rotation, activation packing, GEMM/reductions, output transfer and model action unnormalization. Excluded: raw LIBERO observation conversion, deterministic RNG reset, environment simulation/RPC, and external action-index extraction/gripper binarization. Thus the complete policy API includes its internal preprocessing/postprocessing, but not all environment-driver work.",
        "",
        "Table memory is Torch inference-stage peak allocated/reserved after warmup and graph creation; rollout per-episode peaks and model-load peaks are separately saved. It excludes EGL/driver allocations not tracked by Torch. Storage is actual deployment bytes, including retained unquantized weights, packed codes, scales, permutations/rotations, all step banks, ATM/OHB, tokenizer/processor/config and manifest. Common software, CUDA/JIT binaries and offline calibration data are excluded. Standalone loading requires no original floating checkpoint shards and has no full dequantized weight cache or F.linear quantization simulation.",
        "",
        f"Audit: **{len(audit['cases'])} completed timing/rollout rows verified, zero errors**, with calibration/evaluation state disjointness, exact 100 episode pairs, all task/control-step records, raw timing recomputation and GPU1 inactivity checked. The same policy applies to the reused smoke timing windows. Raw JSON/CSV, compressed calls/logs, commands, source/pack hashes and archived environment are in the adjacent evaluation/protocol artifacts. Source and setup details are in the experiment protocol; the code and paper baselines are not renamed as kernels.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("FINAL_REPORT", args.output, flush=True)


if __name__ == "__main__":
    main()
