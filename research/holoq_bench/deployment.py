"""Self-contained quantized artifacts: no original checkpoint weights at load."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(pack_path, state_path, output):
    pack = torch.load(pack_path, map_location="cpu", mmap=True, weights_only=False)
    checkpoint = Path(pack["meta"]["checkpoint"])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint / "config.json", output / "config.json")
    shutil.copytree(checkpoint / "experiment_cfg", output / "experiment_cfg")
    shutil.copy2(pack_path, output / "packed.pt")
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=False)
    excluded = {
        name + "." + suffix
        for name, r in pack["layers"].items()
        for suffix in ("weight", "bias")
        if suffix == "weight" or r["bias"] is not None
    }
    assert excluded <= state.keys(), excluded - state.keys()
    keep = {k: v for k, v in state.items() if k not in excluded}
    torch.save(keep, output / "nonquantized.pt")
    # The tokenizer/processor + model architecture assets ship with the source
    # package. Account their actual files separately; never a model shard.
    assets = ROOT / "gr00t/model/backbone/eagle2_hg_model"
    asset_files = [f for f in assets.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    manifest = {
        "format": "holoq_bench_deployment_v1",
        "method": pack["meta"]["method"],
        "source_checkpoint": str(checkpoint),
        "source_pack": str(pack_path.resolve()),
        "source_pack_sha256": sha256(pack_path),
        "excluded_state_keys": sorted(excluded),
        "nonquantized_state_keys": sorted(keep),
        "tensor_bytes": sum(v.numel() * v.element_size() for v in keep.values()),
        "package_asset_bytes": sum(f.stat().st_size for f in asset_files),
        "package_assets": {str(f.relative_to(ROOT)): f.stat().st_size for f in asset_files},
        "files": {
            str(f.relative_to(output)): {"bytes": f.stat().st_size, "sha256": sha256(f)}
            for f in output.rglob("*")
            if f.is_file()
        },
        "storage_boundary": "model files + tokenizer/processor assets; excludes shared software/runtime and JIT kernel binaries",
    }
    manifest["deployment_bytes_excluding_manifest"] = (
        sum(v["bytes"] for v in manifest["files"].values()) + manifest["package_asset_bytes"]
    )
    (output / "deployment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DEPLOYMENT", pack["meta"]["method"], manifest["deployment_bytes_excluding_manifest"], flush=True)


def load(directory, device, horizon):
    from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
    from transformers.modeling_utils import no_init_weights
    from gr00t.atm import ensure_dit_attention_patch
    from research.holoq_bench.packed_runtime import enable

    config = GR00T_N1_5_Config.from_pretrained(directory, local_files_only=True)
    assert config.action_head_cfg["action_horizon"] == horizon
    # This constructor uses AutoModel.from_config, not from_pretrained.
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with no_init_weights():
            model = GR00T_N1_5(config, local_model_path=str(directory))
    finally:
        torch.set_default_dtype(old)
    model.half().eval()
    state = torch.load(directory / "nonquantized.pt", map_location="cpu", mmap=True, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    manifest = json.loads((directory / "deployment.json").read_text())
    assert set(missing) == set(manifest["excluded_state_keys"]), (missing, manifest["excluded_state_keys"])
    assert not unexpected, unexpected
    del state
    ensure_dit_attention_patch(model)
    metadata = enable(model, str(directory / "packed.pt"))
    model.to(device=device)
    return model, metadata


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--packs", nargs="+", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for pack in args.packs:
        directory = args.output / pack.stem
        if directory.exists():
            assert (directory / "deployment.json").exists(), f"Incomplete deployment: {directory}"
            print("REUSE", directory, flush=True)
        else:
            export(pack, args.state, directory)


if __name__ == "__main__":
    main()
