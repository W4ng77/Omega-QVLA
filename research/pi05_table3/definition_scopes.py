"""Subset export retaining identical definition8 codes/scales/rotations."""

from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from research.pi05_table3.measure import sha, write


def export_scope(output, scope):
    output = Path(output)
    if (output / "deployment.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    base = output.parents[1] / "definition8_deployment_fp64"
    full = torch.load(base / "packed_core.pt", map_location="cpu", mmap=True, weights_only=False)
    vision = torch.load(base / "vision_int4.pt", map_location="cpu", mmap=True, weights_only=False)
    records = {
        n: r
        for n, r in full["layers"].items()
        if scope == "both"
        or (scope == "head" and "gemma_expert" in n)
        or (scope == "backbone" and "language_model" in n)
    }
    vision = vision if scope != "head" else {}
    excluded = {n + ".weight" for n in records} | set(vision)
    assert len(excluded) == {"both": 414, "head": 126, "backbone": 288}[scope]
    torch.save({"meta": full["meta"], "layers": records}, output / "packed_core.pt")
    torch.save(vision, output / "vision_int4.pt")
    checkpoint = Path("/home/xinyu/models/pi05_libero_pytorch")
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as source:
        other = {k: source.get_tensor(k) for k in source.keys() if k not in excluded}
        save_file(other, output / "model_nonquant.safetensors", metadata=source.metadata())
    shutil.copy2(base / "config.json", output / "config.json")
    shutil.copytree(base / "assets", output / "assets", dirs_exist_ok=True)
    files = {
        str(f.relative_to(output)): {"bytes": f.stat().st_size, "sha256": sha(f)}
        for f in output.rglob("*")
        if f.is_file()
    }
    write(
        output / "deployment.json",
        {
            "scope": scope,
            "excluded_state_keys": sorted(excluded),
            "base_deployment_sha256": sha(base / "deployment.json"),
            "files": files,
            "model_artifact_bytes": sum(v["bytes"] for v in files.values()),
            "note": "Identical frozen grid retained for selected modules; unselected source weights restored; no success claim",
        },
    )
