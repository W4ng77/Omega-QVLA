"""Evidence-linked drafts; unsupported accuracy/H100 claims remain explicit."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/pi05_table3_rebuttal_20261028"
DATA = ROOT / "results/pi05_table3_414_20261028"


def polish(text):
    terms = [
        "FP16",
        "BF16",
        "FP32",
        "INT4",
        "INT32",
        "H100",
        "W4A4",
        "W4A8",
        "W4A16",
        "S4",
        "D2H",
        "H2D",
        "G128",
        "Pi0.5",
        "cd6x",
    ]
    out = []
    for line in text.splitlines():
        if line.startswith("Evidence:"):
            out.append(line)
            continue
        placeholders = {}
        for i, term in enumerate(terms):
            key = "TOKEN" + chr(65 + i) + "TOKEN"
            if term in line:
                placeholders[key] = term
                line = line.replace(term, key)
        line = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", line)
        line = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", line)
        line = re.sub(r",(?=\S)", ", ", line)
        line = re.sub(r"%(?=[a-zA-Z])", "% ", line)
        for key, term in placeholders.items():
            line = line.replace(key, term)
        for a, b in {
            "visionGEMM": "vision GEMM",
            "expert-stepGEMM": "expert-step GEMM",
            "languageS4GEMM": "language S4 GEMM",
            "nativeS4": "native S4",
            "nativeINT4": "native INT4",
            "FP16MMA": "FP16 MMA",
            "BF16MMA": "BF16 MMA",
            "FP32accumulation": "FP32 accumulation",
            "FP32parameters": "FP32 parameters",
            "output-sideSVD": "output-side SVD",
            "insideK": "inside K",
            "toFP32": "to FP32",
            "totalhead": "total head",
            "universalINT4": "universal INT4",
            "sharedFP32": "shared FP32",
        }.items():
            line = line.replace(a, b)
        out.append(line)
    return "\n".join(out) + "\n"


def main():
    rows = list(csv.DictReader((DOC / "main_results.csv").open()))

    def select(name):
        return next(r for r in rows if r["method"] == name and r["mode"] == "sample_graph" and r["steps"] == "8")

    h = select("HoloQ definition8 core + released vision RTN")
    g = select("GPTQ / official AWQ BF16-FP32 backend")
    q = select("SmoothQuant-W4 extension / QServe")
    f = select("FP16 control")
    compact = []
    for r in [f, g, q, h]:
        compact.append(
            {
                "Method": r["method"],
                "W/A": r["W_A_bits"],
                "Actual quantized scope": r["actual_scope"],
                "Success rate": "not measured",
                "Policy mean/p50/p95 ms": " / ".join(
                    f"{float(r[k]):.2f}" for k in ["mean_ms", "p50_ms", "p95_ms"]
                ),
                "Policy calls/s": f"{float(r['policy_calls_per_s']):.3f}",
                "Peak allocated/reserved GB": f"{int(r['peak_allocated_bytes']) / 1e9:.3f} / {int(r['peak_reserved_bytes']) / 1e9:.3f}",
                "Complete model files GB": f"{int(r['model_artifact_bytes']) / 1e9:.3f}",
                "GPTQ ms / this ms": f"{float(g['mean_ms']) / float(r['mean_ms']):.3f}",
                "QServe ms / this ms": f"{float(q['mean_ms']) / float(r['mean_ms']):.3f}",
            }
        )
    with (DOC / "DEPLOYMENT_MAIN_TABLE.csv").open("w") as file:
        w = csv.DictWriter(file, fieldnames=list(compact[0]))
        w.writeheader()
        w.writerows(compact)
    prof = json.loads((DATA / "definition8_profile/verified_eager_attribution.json").read_text())

    def ms(r):
        return float(r["mean_ms"])

    prefix = "experiment_results/pi05_table3_414_20261028/"
    ready = f"""We distinguish 4-bit quantization and storage from hardware arithmetic. Our additional Pi0.5 run covers all 414 Table 3 target projections: 162 vision, 126 language and 126 expert layers. The language/expert core follows the submitted input-only rotation and signed[-7,7] definition; vision retains the released RTN recipe. We store packed W4 and pack A4 GEMM inputs. Language uses native INT4xINT4/INT32; the expert decodes tiles for FP16 MMA/FP32 accumulation, and vision uses BF16 MMA/FP32 accumulation. There is no full floating-weight cache or F.linear fallback for these timed targets. We do not claim an entirely native-INT4 policy.

The expert scale must remain inside the reduction. In the paper's row-vector convention, Eq.9 should read:
Y[m,n] = sW[n] * sum_k(QX[m,k] * sX[layer,t,k] * QW[k,n]) + bias[n].
There is one weight matrix per layer and eight activation-scale rows, without per-step weight requantization.

On one RTX 4080 SUPER, batch 1, horizon 10 and eight Euler steps, with 32 warmup calls and 100 unprofiled calls per configuration, common CUDA Graph execution gives HoloQ mean/p50/p95 {ms(h):.2f}/{float(h["p50_ms"]):.2f}/{float(h["p95_ms"]):.2f} ms and {float(h["policy_calls_per_s"]):.2f} policy calls/s. Complete FP16, packed GPTQ W4A16 and SmoothQuant-W4/QServe W4A8 controls take {ms(f):.2f}, {ms(g):.2f} and {ms(q):.2f} ms. Timing includes preprocessing, transfers, vision/language, all steps, transformations, quantization/packing, GEMM and postprocessing. Precision and layer coverage differ and are disclosed. HoloQ is slower; we will remove the claim that uniform W4A4 necessarily outperforms optimized mixed-bitwidth execution. QServe is an execution control here, not a QuantVLA reproduction.

The 414 target tensors require 1.616 GB including scales, rotations and permutations, versus 5.409 GB originally. Complete model files require {int(h["model_artifact_bytes"]) / 1e9:.3f} GB. Warmed inference peak allocated/reserved memory is {int(h["peak_allocated_bytes"]) / 1e9:.3f}/{int(h["peak_reserved_bytes"]) / 1e9:.3f} GB, versus {int(f["peak_allocated_bytes"]) / 1e9:.3f}/{int(f["peak_reserved_bytes"]) / 1e9:.3f} GB for FP16. The 74.2% static-table claim is not runtime-VRAM saving; available metadata-inclusive artifacts do not reproduce 1.39 GB. We provide separate storage, overhead and backbone/head isolation tables.

These are fixed-input deployment measurements using ten initial observations, not the paper's ten-trajectory calibration or scored rollouts. We do not attach submitted success rates to these kernels or claim exact historical H100/accuracy reproduction.
"""
    (DOC / "READY_TO_PASTE.txt").write_text(ready)
    individual = f"""Evidence-backed reviewer drafts. These respond to the supplied kernel/time/memory topics; unseen review text is not reconstructed.

JvpR — execution,channel scales,and quantized baselines

Thank you for identifying the insufficient scale factorization in Eq.9. For the expert, sX[layer,t,k] belongs inside sum_k; it cannot generally be a scalar epilogue. The corrected expression is Y[m,n]=sW[n]sum_k(QX[m,k]sX[layer,t,k]QW[k,n])+bias[n]. Our packed deployment preserves one weight grid per layer and all8 channel-scale rows. It uses nativeS4/INT32 for per-token-scaled language layers and explicitly packed-decode FP16MMA/FP32 accumulation for per-channel-scaled expert layers. It is not an all-nativeINT4 policy, and step_weight is not used as original-method proof.

We measured real packed controls rather than a dense dequantize-then-F.linear timing proxy. The validated GPTQ control uses the existing official AWQ BF16/FP32 kernel, with BF16 scales fixed before GPTQ reconstruction; the quantization method remains GPTQ. We rejected the tested Marlin configurations after their model-wide oracle failures. The QServe control is a disclosed SmoothQuant-W4 extension, not QuantVLA. Under common8-step Graph execution GPTQ and QServe take {ms(g):.2f} and {ms(q):.2f}ms, while HoloQ takes {ms(h):.2f}ms. We will withdraw the general argument that mixed bitwidth negates acceleration. We do not establish superiority to unmeasured optimized QuantVLA/QVLA/ActQuant deployments.

Evidence: kernel_execution.csv; DEPLOYMENT_MAIN_TABLE.csv; method_alignment.csv; {prefix}definition8_measurement/report.json; {prefix}gptq_awq_bf16_full_policy_oracle.json; {prefix}qserve_full_policy_oracle.json; {prefix}marlin_maxpar1_full_policy_oracle.json.

BY2H — full graph,transformation overhead,head isolation

The original-method core graph is input permutation -> block64 UDH rotation -> A4 quantize/pack -> packed GEMM -> output scale/bias, with no added output-sideSVD restore or low-rank branch. The expert's per-input-channel scale is decoded insideK; weights are not re-quantized per step. Offline coefficient construction,online floating transforms,physical packing,and MMA precision are listed separately. Embeddings,normalization,attention softmax and auxiliary conditioning projections remain floating outside the414 selected projections.

Five complete-call PyTorch traces attribute {prof["GPU_work_ms_per_policy"]:.1f}ms of cumulative GPU work: visionGEMM75.7ms,all8expert-stepGEMM72.8ms,languageS4GEMM8.1ms. Across phases,quantize/pack/scale processing is57.0ms,permutation+rotation17.8ms,copy/dtype/contiguous44.8ms,and remaining work29.1ms. These are diagnostic GPU-work totals, not a substitute for the unprofiled{ms(h):.2f}ms complete-call mean. Nsight verifies all414 targets in actual Graph execution and exposes Graph-launch delay; host D2H wait overlaps GPU work and is not added to device totals.

With identical frozen definition8 tensors, backbone-only,head-only and both Graph policy means are232.10,192.33 and309.58ms respectively. These are performance/memory isolation measurements, not success-rate ablations. The126 expert target weights occupy622.9MB before quantization and216.6MB including required packed data/metadata. Active-head static tensors including floating auxiliaries are860.2MB before and454.0MB after; registered resident tensors differ because the loader promotes sensitive auxiliaries toFP32. The full-deployment tensor ledger separately shows the unchanged unused526.6MB expert vocabulary projection. The measured extra allocation from first denoise entry through completion is1.98MB above the already-resident model/prefix, not totalhead memory.

Evidence: operator_graph.csv; profile_breakdown.csv; {prefix}definition8_profile/verified_eager_attribution.json; {prefix}nsight_definition8_verified_summary.json; {prefix}definition8_memory_inspection.json; {prefix}action_head_memory_breakdown.json; scope rows in main_results.csv.

gXcZ — representative hardware,time,throughput,peak memory

We provide complete synchronized policy.infer measurements on RTX4080SUPER16GB: batch1,horizon10,8Euler steps,identical observations/noise,32warmup and100calls per mode. All methods receive both common Eager and full-sampler CUDA Graph execution; compile is disabled. Complete timing includes CPU preprocessing,H2D,vision/language,all denoising,online transforms,quantization/packing,GEMM,D2H and output transforms. RPC,simulator,loading and cold capture are excluded. GPU1 remained idle; GPU/process logs accompany each run. Raw Eager samples show substantial host variability, so p50/p95 and per-call logs are retained rather than attributing all wall time to kernels.

HoloQ Graph mean/p50/p95 is {ms(h):.2f}/{float(h["p50_ms"]):.2f}/{float(h["p95_ms"]):.2f}ms,throughput{float(h["policy_calls_per_s"]):.3f}policy calls/s. Warmed inference peak allocated/reserved memory is {int(h["peak_allocated_bytes"]) / 1e9:.3f}/{int(h["peak_reserved_bytes"]) / 1e9:.3f}GB. Reserved reflects allocator history, not minimum required VRAM. Relative to the FP16 control, peak allocated is about50.6%lower; relative to QServe it is higher, not lower. There is no latency advantage over the measured packed quantized controls. No new H100 result,scored success rate or real-rollout latency is inferred from these fixed-input measurements.

Evidence: DEPLOYMENT_MAIN_TABLE.csv; main_results.csv; raw *_calls.jsonl and gpu_samples.jsonl; source/version/archive manifests.

aAUc — static scope and8-versus10 steps

We agree that414-target storage cannot support a252-target timing claim without explicit qualification. The additional HoloQ run matches all162vision+126language+126expert projections and executes486vision,126language and1008expert GEMMs per8-step policy (vision is processed for3image slots,including the masked slot). Full module-name/shape and CUDA-launch ledgers are provided. Auxiliary projections/embeddings are listed separately, rather than describing every policy operation as INT4.

The definition8 core is calibrated and executed at8steps. The available historical pack stores10 expert scale rows and differs in language scales,weight grids and output rotation; it is retained as a separately named10-step configuration. We do not silently interpolate its scales or compare its latency against8-step baselines. Baseline static quantizers were calibrated from10-step teacher inputs using the same10initial observations; this budget difference is disclosed and prevents equal-calibration accuracy claims. Calibration provenance remains incomplete for the original paper pack.

The5.41GB denominator matches414 selected original weight tensors, not the7.23GB complete checkpoint. Necessary metadata gives1.616GB for the reconstructed target tensors and about3.44GB for full model files, with shared tokenizer bytes separate. We cannot reproduce1.39GB/74.2%from the available metadata-inclusive artifact and will qualify/correct that storage claim; it is not runtime-VRAM reduction.

Evidence: method_alignment.csv; kernel_execution.csv; {prefix}definition_storage_audit.json; {prefix}complete_storage_audit.json; manifests for8-step and historical10-step artifacts.

cd6x — execution coverage and original experiment qualification

The new physical packed deployment and the historical Python quantization reference are explicitly distinct. Available historical-loader metadata describes unpacking weights and floating computation; it does not identify the exact original H100 run. The new timed path keeps all414 selected weights packed and packs A4 GEMM inputs, while documenting floating transforms and decoded floating MMA in vision/expert layers. Only the language path is nativeS4. Thus we will not equate the4-bit numerical method,compressed files,and universalINT4 hardware execution, or attach historical success rates to a newly compiled runtime.

Evidence: source manifest, kernel_execution.csv, actual CUDA kernel names, resident tensor ledger, and independent frozen-grid oracle checks.
"""
    (DOC / "REVIEWER_REPLIES.txt").write_text(polish(individual))
    notes = """Main table protocol and limits

All four primary rows are8-step full policy replay with common sampler CUDA Graph, not nakedGEMM. No new scored episodes: success is deliberately blank. Ratios above1 mean faster than the corresponding baseline; HoloQ ratios below1 mean slower. GPTQ uses252 targets,visionBF16,group128,BF16scales fixed before reconstruction and officialAWQ BF16/FP32MMA. SmoothQuant-W4/QServe uses414targets,A8 and FP16scales; it is not QuantVLA. HoloQ uses414targets with signed15core plus released visionRTN. None are claimed to be identical quantization-accuracy settings.

Target and full model files include all necessary metadata,remaining parameters,normalization assets and loader manifests; the shared4,264,023-byte tokenizer is separate. Runtime allocated/reserved peaks are after warmup and reset immediately before100formal calls,excluding loading/cold capture. Reserved contains allocation history. FP16 runtime loads the same16-bit source checkpoint and retains sensitiveFP32parameters; storage is the actual reused source files,not a fabricated export.

Calibration here is10initial Object observations/all actual tokens,not10trajectories. HoloQcore8-step calibration and baseline10-step static-weight calibration are separately disclosed. No repeated evaluation-seed tuning,episode selection or success-number borrowing occurred in this packet. The original historicalH100 path and exact paper calibration are not recovered. HistoricalGR00T180 and70/72 results are not Pi0.5Table3 evidence. The originalmethod and step_weight/group_int are not merged.

The main numerical guards are unchanged: same-input localNMSE1e-4 and full-actionNMSE0.01; they are engineering checks,not task-success equivalence. HoloQ localmax9.94e-7 over1134actualcalls and finalactionmax0.004914 over10observations; GPTQ/BF16AWQ localmax3.88e-5 over1134calls; QServe has its separate integer-oracle report. Final action maxabsolute error is0.104,so do not present this as exact action identity. Graph/eager checks are exact for the reported configurations. Failed Marlin/default,max_par1,and FP16AWQ oracle results remain archived and do not enter the validated main table. No claim against the strongest unmeasured VLA-specific deployment is supported.
"""
    (DOC / "TABLE_NOTES.txt").write_text(notes)
    print("WROTE_REPLIES", len(ready.split()), "combined words")


if __name__ == "__main__":
    main()
