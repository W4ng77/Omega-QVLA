"""Snapshot compact source/calibration/protocol manifests for publication."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "teacher_capture/manifest.json": "coverage_and_capture.json",
        "calibration_fp16_init0_v2/manifest.json": "calibration_manifest.json",
        "heldout_init10_19/manifest.json": "heldout_manifest.json",
        "teacher_capture/fp16_model_state.json": "original_model_tensor_manifest.json",
    }
    for source, target in paths.items():
        shutil.copy2(a.root / source, a.output / target)
    checkpoint = Path(json.loads((a.root / "teacher_capture/manifest.json").read_text())["checkpoint"])
    original = Path("/home/xinyu/models/omega-qvla-gr00t-w4a4/gr00t_object/quantized.pt")
    files = [f for f in checkpoint.rglob("*") if f.is_file()]
    info = {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint),
        "checkpoint_files": {
            str(f.relative_to(checkpoint)): {"bytes": f.stat().st_size, "sha256": digest(f)} for f in files
        },
        "historical_pack": {
            "path": str(original),
            "bytes": original.stat().st_size,
            "sha256": digest(original),
            "used_for_primary_runs": False,
        },
        "paper": {
            "version": "arXiv:2605.28803v3",
            "url": "https://arxiv.org/pdf/2605.28803v3",
            "main_table": "Table 2 p5",
            "efficiency_table": "Table 3 p6 (Pi-0.5 static footprint, not latency)",
            "config_table": "Table 5 p11",
            "calibration": "Baseline appendix p10",
        },
        "unavailable_GR00T_baselines": {
            "QVLA": {
                "source": "https://github.com/AutoLab-SAI-SJTU/QVLA",
                "reason": "Official architectures OpenVLA/OpenVLA-OFT/UniVLA; no GR00T diffusion adapter/checkpoint. Not relabeled GPTQ.",
            },
            "ActQuant": {
                "source": "https://github.com/arashakb/ActQuant",
                "reason": "Official GGUF/GGML artifacts/backends support OpenVLA-OFT/Pi-0.5, not GR00T. Not a fake-quant speed row.",
            },
        },
    }
    (a.output / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    manifests = {}
    for directory in (a.root / "deployments").iterdir():
        if (directory / "deployment.json").exists():
            d = json.loads((directory / "deployment.json").read_text())
            manifests[directory.name] = d
    (a.output / "deployment_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")


if __name__ == "__main__":
    main()
