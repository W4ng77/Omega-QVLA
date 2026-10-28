"""Publishable evidence packet; preserve originals and avoid large model files."""

import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/pi05_table3_414_20261028"
OUT = ROOT / "experiment_results/pi05_table3_414_20261028"


def digest(file):
    h = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    excluded = []
    for file in sorted(DATA.rglob("*")):
        if not file.is_file():
            continue
        rel = file.relative_to(DATA)
        trace = file.name.endswith("trace.json") or file.name in [
            "policy_trace.json",
            "eager_policy_trace.json",
            "graph_trace.json",
        ]
        include = file.suffix in [".json", ".jsonl", ".csv", ".txt", ".log", ".npz", ".nsys-rep"]
        if not include:
            continue
        if file.name == "profile_kernel_attribution.json":
            excluded.append(
                {
                    "path": str(rel),
                    "bytes": file.stat().st_size,
                    "sha256": digest(file),
                    "reason": "Rejected duplicated FunctionEvent association output; original retained locally; raw trace archived instead",
                }
            )
            continue
        target = OUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hash = digest(file)
        if trace or file.suffix == ".log":
            target = target.with_name(target.name + ".gz")
            with file.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, 2**20)
        else:
            shutil.copy2(file, target)
        inventory.append(
            {
                "path": str(target.relative_to(OUT)),
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "original_local_path": str(file),
                "original_bytes": file.stat().st_size,
                "original_sha256": original_hash,
            }
        )
    observations = Path("/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt")
    target = OUT / "source_observations.pt.gz"
    with observations.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    inventory.append(
        {
            "path": target.name,
            "bytes": target.stat().st_size,
            "sha256": digest(target),
            "original_local_path": str(observations),
            "original_bytes": observations.stat().st_size,
            "original_sha256": digest(observations),
        }
    )
    versions = {}
    for name in ["torch", "triton", "transformers", "safetensors", "numpy", "jax", "flax", "sentencepiece"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed in archive process environment; see runtime reports"
    revisions = {}
    for name in ["gptq", "marlin", "omniserve", "QuaRot", "llm-awq", "QuantVLA", "QVLA", "ActQuant"]:
        path = ROOT / "third_party" / name
        revisions[name] = {
            "commit": subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip(),
            "url": subprocess.check_output(
                ["git", "-C", str(path), "remote", "get-url", "origin"], text=True
            ).strip(),
            "dirty": subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True),
        }
    revisions["openpi"] = {
        "commit": subprocess.check_output(
            ["git", "-C", "/home/xinyu/openpi", "rev-parse", "HEAD"], text=True
        ).strip()
    }
    source_files = {
        str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "research/pi05_table3").glob("*") if p.is_file()
    }
    replacements = {
        str(p.relative_to("/home/xinyu/openpi/src/openpi/models_pytorch/transformers_replace")): digest(p)
        for p in Path("/home/xinyu/openpi/src/openpi/models_pytorch/transformers_replace").rglob("*.py")
    }
    result = {
        "source_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "python_executable": "/home/xinyu/openpi/.venv/bin/python",
        "packages": versions,
        "source_revisions": revisions,
        "final_source_sha256": source_files,
        "openpi_transformers_replacement_source_sha256": replacements,
        "environment_allowlist": {
            k: os.environ.get(k)
            for k in [
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "TORCH_CUDA_ARCH_LIST",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ]
        },
        "files": inventory,
        "excluded_derived_files": excluded,
        "large_model_policy": "No checkpoint, quantized pack, canonical build cache, teacher activation shard or exportedSQLite is committed. Exact byte sizes/SHA256 and local locations are in deployment manifests; code rebuilds them. Raw nsys-rep can regenerate SQLite.",
        "scope": "All414 targets for HoloQ; no task-success or real-rollout claim. Rejected runs and first stricter-oracle failures retained separately.",
    }
    (OUT / "archive_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ARCHIVE", len(inventory), "files", sum(x["bytes"] for x in inventory), "bytes", flush=True)


if __name__ == "__main__":
    main()
