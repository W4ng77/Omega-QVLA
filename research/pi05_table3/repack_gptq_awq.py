"""Same frozen GPTQ codes/scales in the already-available official AWQ format.

No AWQ quantization/search or new method is run. This isolates the inference
backend after Marlin's model-wide oracle failure; no dense inference fallback.
"""

import fcntl
import json
from pathlib import Path
import shutil
import torch
from research.pi05_table3 import kernels
from research.pi05_table3.measure import write, sha


def main():
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = Path(__file__).resolve().parents[2] / "results/pi05_table3_414_20261028"
    target = data / "baseline_deployments/gptq_awq252"
    target.mkdir(exist_ok=False)
    original = data / "baseline_deployments/gptq_marlin252"
    manifest = json.loads((data / "baseline_capture/manifest.json").read_text())
    layers = {}
    provenance = []
    for item in manifest["layers"]:
        if "vision_tower" in item["name"]:
            continue
        cache = data / "baseline_build/gptq_marlin252/layers" / f"{item['index']:03d}.pt"
        entry = torch.load(cache, map_location="cpu", mmap=True, weights_only=False)
        q, s = entry["canonical_codes"], entry["canonical_scales"]
        record = kernels.pack_awq(q.int() + 8, s, torch.full_like(s, 8))
        record.update(
            K=item["K"],
            N=item["N"],
            original_K=item["K"],
            original_N=item["N"],
            bias=entry["record"].get("bias"),
            external_dtype=item["weight_dtype"],
        )
        layers[item["name"]] = record
        provenance.append(
            {"layer": item["name"], "canonical_cache_sha256": sha(cache), "quantizer_changed": False}
        )
        print("REPACK_GPTQ_AWQ", len(layers), "/252", flush=True)
    assert len(layers) == 252
    config = {
        "method": "GPTQ frozen G128 codes; official AWQ backend only, NOT AWQ quantization",
        "source_deployment_manifest_sha256": sha(original / "deployment.json"),
        "quantizer_changed": False,
        "weights": "Same signed16-level codes,FP16group128scales; affine encoding zero8; no requantization",
        "backend": "Already-present pinned official AWQ gemm/gemv kernels, current-stream wrapper; no dense fallback",
    }
    torch.save({"meta": config, "layers": layers}, target / "packed.pt")
    for file in ["model_nonquant.safetensors", "config.json"]:
        shutil.copy2(original / file, target / file)
    shutil.copytree(original / "assets", target / "assets")
    files = {
        str(f.relative_to(target)): {"bytes": f.stat().st_size, "sha256": sha(f)}
        for f in target.rglob("*")
        if f.is_file()
    }
    base = json.loads((original / "deployment.json").read_text())
    write(
        target / "deployment.json",
        {
            "method": "gptq_awq252",
            "configuration": config,
            "files": files,
            "excluded_state_keys": base["excluded_state_keys"],
            "model_artifact_bytes": sum(v["bytes"] for v in files.values()),
            "canonical_record_provenance": provenance,
            "source_builder_sha256": sha(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
