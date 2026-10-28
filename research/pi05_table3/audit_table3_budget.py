"""Compare the author-supplied Table3 with shape-based storage budgets.

Budget agreement is NOT proof of the historical quantizer or layer selection.
The G128 cost below is a hypothetical storage allowance, not a change to the
submitted HoloQ per-output weight-scale definition. No kernels are modified.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = json.loads(args.manifest.read_text())["layers"]
    assert len(layers) == 414
    total = sum(2 * x["K"] * x["N"] for x in layers)
    selections = {
        "HoloQ414_target_set": layers,
        "GPTQ252_core_candidate": [x for x in layers if "vision_tower" not in x["name"]],
        "QuantVLA108_MLP_only_candidate": [
            x for x in layers if "vision_tower" not in x["name"] and ".mlp." in x["name"]
        ],
        "QVLA_ActQuant288_vision_language_candidate": [x for x in layers if "gemma_expert" not in x["name"]],
    }
    rows = []
    for name, selected in selections.items():
        raw = sum(2 * x["K"] * x["N"] for x in selected)
        for metadata in ["none", "uniform_G128_FP16_allowance", "per_output_FP16", "per_output_FP32"]:
            overhead = 0
            if metadata == "uniform_G128_FP16_allowance":
                overhead = sum(x["K"] * x["N"] / 128 * 2 for x in selected)
            elif metadata.startswith("per_output"):
                overhead = sum(x["N"] for x in selected) * (2 if metadata.endswith("FP16") else 4)
            footprint = total - raw + raw / 4 + overhead
            rows.append(
                {
                    "candidate": name,
                    "Q_layers": len(selected),
                    "metadata_budget": metadata,
                    "original414_universe_bytes": total,
                    "selected_original_bytes": raw,
                    "remaining_FP16_weight_bytes": total - raw,
                    "W4_code_bytes": raw // 4,
                    "metadata_allowance_bytes": overhead,
                    "footprint_bytes": footprint,
                    "footprint_GB": footprint / 1e9,
                    "saving_percent": 100 * (1 - footprint / total),
                    "exclusions": "rotation,activation-scale tables,permutations,zero points,loader/runtime metadata; fractional uniform groups are an allowance, not exact packed tail layout",
                    "interpretation": "Candidate arithmetic only; not proof of historical method scope or scale granularity",
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "TABLE3_BUDGET_CHECK.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "source": "Table3 supplied by the author in this conversation",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "actual_original414_weight_bytes": total,
        "submitted_table": [
            {"method": "FP16 baseline", "Q_layers": 0, "footprint_GB": 5.41, "saving_percent": None},
            {"method": "GPTQ W4A16", "Q_layers": 252, "footprint_GB": 2.00, "saving_percent": 62.9},
            {"method": "QuantVLA W4A8", "Q_layers": 108, "footprint_GB": 2.38, "saving_percent": 55.9},
            {"method": "QVLA", "Q_layers": 288, "footprint_GB": 1.82, "saving_percent": 66.3},
            {"method": "ActQuant 4bpw", "Q_layers": 288, "footprint_GB": 1.81, "saving_percent": 66.5},
            {"method": "HoloQ-VLA", "Q_layers": 414, "footprint_GB": 1.39, "saving_percent": 74.2},
        ],
        "inference": "HoloQ414,core252,andMLP-only108 budgets with uniformFP16/G128 scale allowance round to the corresponding three reported footprint/saving rows. This is an arithmetic fingerprint, NOT recovered experiment provenance.",
        "unresolved": "OriginalTable3 generator, actual method-specific target lists, metadata inclusion and ActQuant1.81 are not recovered. Layer counts alone do not establish these mappings.",
        "quantizer_unchanged": "HoloQ core remains per-outputSw,processing128,damping.01; no G128 weight-quantization change was made to fit this table.",
    }
    (args.output / "TABLE3_BUDGET_CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
    print("TABLE3_BUDGET_AUDIT", total, len(rows))


if __name__ == "__main__":
    main()
