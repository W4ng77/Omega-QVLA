"""Materialize a standard OpenPI checkpoint from packed Pi-0.5 INT4 storage."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from gr00t.quantization.int4_storage import unpack_int4
except ImportError:
    from int4_storage import unpack_int4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    manifest = json.loads((args.checkpoint / "manifest.json").read_text())
    if manifest["format"] != "omega_pi05_w4a4_int4_v1":
        raise ValueError("Unsupported packed checkpoint format")

    weights = {}
    with safe_open(args.checkpoint / "model_nonquant.safetensors", framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for name in source.keys():
            weights[name] = source.get_tensor(name)

    vision = torch.load(args.checkpoint / manifest["vision"]["path"], map_location="cpu", weights_only=True, mmap=True)
    for name, record in vision.items():
        weights[name] = unpack_int4(record).contiguous()

    for name, spec in manifest["lm_shapes"].items():
        dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
        weights[name] = torch.zeros(spec["shape"], dtype=dtype)

    if len(weights) != manifest["total_model_tensors"]:
        raise RuntimeError("Materialized model tensor count differs from the manifest")

    temporary = output / "model.safetensors.tmp"
    save_file(weights, temporary, metadata=metadata)
    temporary.replace(output / "model.safetensors")
    shutil.copy2(args.checkpoint / "config.json", output / "config.json")
    shutil.copytree(args.checkpoint / "assets", output / "assets")
    print(f"Materialized OpenPI checkpoint at {output}", flush=True)


if __name__ == "__main__":
    main()
