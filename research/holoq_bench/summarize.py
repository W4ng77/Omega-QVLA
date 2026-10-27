"""Generate the main comparison from raw completed run artifacts only."""

import argparse
import csv
import json
from pathlib import Path

CONFIGS = {
    "holoq_llm_nunchaku_fusedrot": (
        "HoloQ-VLA", "original grid / Nunchaku S4 LLM + Marlin DiT, fused rotation", "4/4", "180 (84 LLM + 96 DiT)"
    ),
    "holoq_step_weight_nunchaku_f16sw_fusedrot": (
        "HoloQ-VLA conversion", "step_weight / Nunchaku S4 all targets, FP16 bank-scale cast + fused rotation", "4/4", "180 (84 LLM + 96 DiT)"
    ),
    "smoothquant_a4_nunchaku": (
        "SmoothQuant", "original grid / Nunchaku S4 all targets", "4/4", "180 (84 LLM + 96 DiT)"
    ),
    "holoq_fusedrot": (
        "HoloQ-VLA",
        "original grid / Marlin + fused block rotation",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "holoq_step_weight_fusedrot": (
        "HoloQ-VLA conversion",
        "step_weight / native S4 DiT, Marlin LLM + fused rotation",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "holoq_llm_s4_fusedrot": (
        "HoloQ-VLA",
        "original grid / LLM CUTLASS S4, fused block rotation",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "holoq_step_weight_llm_s4_fusedrot": (
        "HoloQ-VLA conversion",
        "step_weight / all-target S4, fused block rotation",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "duquant_a8_g128_fusedrot": (
        "DuQuant",
        "paper GR00T port / Marlin G128 + fused rotation",
        "4/8",
        "180 (84 LLM + 96 DiT)",
    ),
    "duquant_a4_g128_fusedrot": (
        "DuQuant",
        "paper GR00T port / Marlin G128 + fused rotation",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "quantvla_selective_a8_atm_g128_fusedrot": (
        "QuantVLA",
        "GR00T port + ATM/OHB / Marlin + fused rotation",
        "4/8",
        "116 (84 LLM + 32 DiT MLP)",
    ),
    "quantvla_selective_a4_atm_g128_fusedrot": (
        "QuantVLA",
        "GR00T port + ATM/OHB / Marlin + fused rotation",
        "4/4",
        "116 (84 LLM + 32 DiT MLP)",
    ),
    "holoq_llm_s4": ("HoloQ-VLA", "original grid / LLM CUTLASS S4 + DiT Marlin", "4/4", "180 (84 LLM + 96 DiT)"),
    "holoq_step_weight_llm_s4": (
        "HoloQ-VLA conversion",
        "step_weight / LLM CUTLASS S4 + DiT native S4",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "smoothquant_a4_cutlass": (
        "SmoothQuant",
        "original grid / CUTLASS native S4 all targets",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "holoq": (
        "HoloQ-VLA",
        "original-definition / packed Marlin + block rotations",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "holoq_step_weight": (
        "HoloQ-VLA conversion",
        "step_weight / native S4 DiT + Marlin LLM",
        "4/4",
        "180 (84 LLM + 96 DiT)",
    ),
    "gptq_selective": ("GPTQ", "Marlin / exclude DiT attention", "4/16", "116 (84 LLM + 32 DiT MLP)"),
    "gptq_full": ("GPTQ", "Marlin / full targets", "4/16", "180 (84 LLM + 96 DiT)"),
    "awq_qwen_selective": (
        "AWQ",
        "Qwen grouped search/folding + DiT linear port / packed AWQ",
        "4/16",
        "116 (84 LLM + 32 DiT MLP)",
    ),
    "awq_qwen_full": (
        "AWQ",
        "Qwen grouped search/folding + DiT linear port / packed AWQ",
        "4/16",
        "180 (84 LLM + 96 DiT)",
    ),
    "omniquant_selective": (
        "OmniQuant",
        "linear LWC/scale port / packed AWQ",
        "4/16",
        "116 (84 LLM + 32 DiT MLP)",
    ),
    "smoothquant_a8": ("SmoothQuant", "QServe packed INT8 MMA", "4/8", "180 (84 LLM + 96 DiT)"),
    "smoothquant_a4": ("SmoothQuant", "QServe INT8 container, numerical A4", "4/4", "180 (84 LLM + 96 DiT)"),
    "duquant_a8_g128": ("DuQuant", "paper GR00T SVD port / Marlin FP16 MMA", "4/8", "180 (84 LLM + 96 DiT)"),
    "duquant_a4_g128": ("DuQuant", "paper GR00T SVD port / Marlin FP16 MMA", "4/4", "180 (84 LLM + 96 DiT)"),
    "quantvla_selective_a8_atm_g128": (
        "QuantVLA",
        "released GR00T port + ATM/OHB / Marlin",
        "4/8",
        "116 (84 LLM + 32 DiT MLP)",
    ),
    "quantvla_selective_a4_atm_g128": (
        "QuantVLA",
        "released GR00T port + ATM/OHB / Marlin",
        "4/4",
        "116 (84 LLM + 32 DiT MLP)",
    ),
}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--deployments", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    queue = json.loads((a.runs / "queue.json").read_text()) if (a.runs / "queue.json").exists() else {}
    common = queue.get("common_mode", "pending")
    selection = queue.get("backend_selection", {})
    selected = {
        (v["selected"]["case"], v["selected"]["mode"])
        for v in selection.get("families", {}).values()
    }
    selected_families = set(selection.get("families", {}))
    rows = []
    for case, (method, variant, bits, scope) in CONFIGS.items():
        if queue.get("valid_cases") and case not in queue["valid_cases"]:
            continue
        for mode in ("eager", "head_graph"):
            base = a.runs / case / mode
            replay_path = base / "replay/summary.json"
            rollout_path = base / "rollout/summary.json"
            if not replay_path.exists() and not rollout_path.exists():
                continue
            replay = json.loads(replay_path.read_text()) if replay_path.exists() else {}
            rollout = json.loads(rollout_path.read_text()) if rollout_path.exists() else {}
            rollout_complete = bool(rollout.get("finished") and rollout.get("episodes") == 100)
            completion_path = base / "rollout/completed.json"
            completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
            rollout_status = "complete" if rollout_complete else "failed" if completion.get("returncode", 0) else "pending"
            deployment = json.loads((a.deployments / case / "deployment.json").read_text())
            storage = (
                deployment["deployment_bytes_excluding_manifest"]
                + (a.deployments / case / "deployment.json").stat().st_size
            )
            lat = replay.get("latency", {})
            rol = rollout.get("latency", {}) if rollout_complete else {}
            mem = replay.get("inference_memory", {})
            row = {
                "case": case,
                "method": method,
                "variant": variant,
                "execution": mode,
                "common": mode == common,
                "selected_best": (case, mode) in selected
                or (
                    case not in selected_families
                    and not any(case == d[0] for d in selected)
                    and not case.startswith(("holoq", "duquant", "quantvla", "smoothquant_a4"))
                    and mode == common
                ),
                "W/A": bits,
                "scope": scope,
                "successes": rollout.get("successes", "") if rollout_complete else "",
                "episodes": 100 if rollout_complete else "",
                "rollout_status": rollout_status,
                "completed_episode_count": rollout.get("episodes", 0),
                "policy_mean_ms": lat.get("mean_ms", ""),
                "policy_p50_ms": lat.get("p50_ms", ""),
                "policy_p95_ms": lat.get("p95_ms", ""),
                "rollout_mean_ms": rol.get("mean_ms", ""),
                "rollout_p50_ms": rol.get("p50_ms", ""),
                "rollout_p95_ms": rol.get("p95_ms", ""),
                "peak_allocated_bytes": mem.get("peak_allocated_bytes", ""),
                "peak_reserved_bytes": mem.get("peak_reserved_bytes", ""),
                "rollout_peak_allocated_bytes": rollout.get("inference_memory", {}).get(
                    "peak_allocated_bytes", ""
                ),
                "rollout_peak_reserved_bytes": rollout.get("inference_memory", {}).get(
                    "peak_reserved_bytes", ""
                ),
                "rollout_cold_capture_calls": rollout.get("cold_capture_calls", ""),
                "deployment_bytes": storage,
                "per_task_successes": json.dumps(rollout.get("per_task_successes", []) if rollout_complete else []),
                "code_commit": replay.get("code_commit", rollout.get("code_commit", "")),
                "replay_code_commit": replay.get("code_commit", ""),
                "rollout_code_commit": rollout.get("code_commit", ""),
                "source_pack_sha256": deployment["source_pack_sha256"],
                "actual_backend_layers": json.dumps(replay.get("backend_layers", rollout.get("backend_layers", {}))),
            }
            rows.append(row)
    quantized_common = [
        r for r in rows if r["common"] and not r["case"].startswith("holoq") and r["policy_mean_ms"]
    ]
    fastest = min(quantized_common, key=lambda r: r["policy_mean_ms"]) if quantized_common else None
    for row in rows:
        row["fastest_quantized_baseline"] = fastest["case"] if fastest else ""
        row["speedup_vs_fastest_quantized_baseline"] = (
            fastest["policy_mean_ms"] / row["policy_mean_ms"]
            if fastest and row["common"] and row["policy_mean_ms"] and row["case"].startswith("holoq")
            else ""
        )
    if rows:
        with (a.output / "main.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (a.output / "main.json").write_text(json.dumps({"common_mode": common, "rows": rows}, indent=2) + "\n")
    text = [
        "# HoloQ-VLA / quantized baseline results",
        "",
        f"Common execution: **{common}**. Missing scores are pending, not zero.",
        "All latency rows measure complete get_action; replay and rollout are distinct. Storage is decimal GB.",
        "",
        "| Method | Variant/backend | W/A | Scope | Success | Replay mean / p50 / p95 ms | Rollout mean / p50 / p95 ms | Peak alloc / reserve GB | Deployment GB | Speedup vs fastest quant baseline |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "pending"

    for r in rows:
        if not r["common"]:
            continue
        success = f"{r['successes']}/{r['episodes']}" if r["episodes"] else "pending"
        text.append(
            "| "
            + " | ".join(
                [
                    r["method"],
                    r["variant"],
                    r["W/A"],
                    r["scope"],
                    success,
                    " / ".join(fmt(r[k]) for k in ["policy_mean_ms", "policy_p50_ms", "policy_p95_ms"]),
                    " / ".join(fmt(r[k]) for k in ["rollout_mean_ms", "rollout_p50_ms", "rollout_p95_ms"]),
                    " / ".join(
                        fmt(r[k] / 1e9) if r[k] else "pending"
                        for k in ["peak_allocated_bytes", "peak_reserved_bytes"]
                    ),
                    fmt(r["deployment_bytes"] / 1e9),
                    fmt(r["speedup_vs_fastest_quantized_baseline"])
                    if r["speedup_vs_fastest_quantized_baseline"]
                    else "—",
                ]
            )
            + " |"
        )
    quant = {
        r["case"]: r for r in rows if r["common"] and not r["case"].startswith("holoq") and r["policy_mean_ms"]
    }
    speeds = {}
    speed_rows = []
    for r in rows:
        if r["common"] and r["case"].startswith("holoq") and r["policy_mean_ms"]:
            speeds[r["case"]] = {n: b["policy_mean_ms"] / r["policy_mean_ms"] for n, b in quant.items()}
            for name, value in speeds[r["case"]].items():
                baseline = quant[name]
                speed_rows.append(
                    {
                        "ours": r["case"],
                        "ours_execution": r["execution"],
                        "baseline": name,
                        "baseline_execution": baseline["execution"],
                        "replay_speedup": value,
                        "rollout_speedup": baseline["rollout_mean_ms"] / r["rollout_mean_ms"]
                        if baseline["rollout_mean_ms"] and r["rollout_mean_ms"]
                        else "",
                    }
                )
    (a.output / "speedups.json").write_text(json.dumps(speeds, indent=2) + "\n")
    if speed_rows:
        with (a.output / "speedups.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(speed_rows[0]))
            writer.writeheader()
            writer.writerows(speed_rows)
        ours = list(speeds)
        text += [
            "",
            "## Complete-policy speedups against every measured quantized baseline",
            "",
            "| Quantized baseline | " + " | ".join(ours) + " |",
            "|---|" + "---|" * len(ours),
        ]
        for name in quant:
            text.append("| " + name + " | " + " | ".join(f"{speeds[c][name]:.3f}x" for c in ours) + " |")
    task_rows = []
    for r in rows:
        values = json.loads(r["per_task_successes"])
        if not values:
            continue
        for task, successes in (values.items() if isinstance(values, dict) else enumerate(values)):
            task_rows.append(
                {
                    "case": r["case"],
                    "execution": r["execution"],
                    "task": task,
                    "successes": successes,
                }
            )
    if task_rows:
        with (a.output / "per_task.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(task_rows[0]))
            writer.writeheader()
            writer.writerows(task_rows)
    best_rows = [r for r in rows if r["selected_best"]]
    if best_rows:
        with (a.output / "best.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(best_rows[0]))
            writer.writeheader()
            writer.writerows(best_rows)
        (a.output / "best.json").write_text(
            json.dumps({"selection": selection, "rows": best_rows}, indent=2) + "\n"
        )
    text += [
        "",
        "Speedup = baseline full-policy replay mean / our full-policy replay mean; below 1 means slower.",
        "See speedups.json for every baseline, main.csv for all execution modes and exact bytes.",
        "",
        "**Fidelity limits:** AWQ uses a Qwen3 grouped adapter and a DiT linear adapter; OmniQuant is a linear LWC/scaling port, not the full upstream block/LET implementation. DuQuant uses the paper GR00T SVD recipe. Numerical A4/A8 feeding Marlin uses FP16 MMA. These distinctions constrain any claim about an optimized original method.",
        "",
        "100-episode score differences are descriptive; do not infer statistical improvement.",
    ]
    (a.output / "main.md").write_text("\n".join(text) + "\n")
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
