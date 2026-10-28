"""Avoid Marlin unscaled FP16 reduction overflow without changing any W grid."""
import fcntl
import json
from pathlib import Path
import torch
from research.holoq_bench.deployment import export
from research.holoq_bench.kernels import pack_marlin


def main():
    root = Path("results/holoq_object")
    lock = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    cache = {}
    for path in sorted((root / "paper_cfg_build/layer_cache").glob("*.pt")):
        entry = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        cache[entry["name"]] = entry["canonical"]
    assert len(cache) == 180
    dest = root / "packs_paper_v2"
    dest.mkdir(exist_ok=True)
    for converted in (False, True):
        for native in (False, True):
            suffix = ("_step_weight" if converted else "") + ("_llm_s4" if native else "")
            old = "holoq_paper" + suffix
            case = "holoq_paper_g128" + suffix
            target = dest / (case + ".pt")
            if not target.exists():
                pack = torch.load(root / "deployments" / old / "packed.pt", map_location="cpu", mmap=True, weights_only=False)
                count = 0
                for name, record in pack["layers"].items():
                    if record["backend"] != "marlin":
                        continue
                    canonical = cache[name]
                    assert canonical["scales"].shape == (record["N"], 1)
                    repeated = canonical["scales"].repeat(1, record["K"] // 128)
                    assert torch.equal(repeated, canonical["scales"].expand_as(repeated))
                    record.update(pack_marlin(canonical["q"], repeated))
                    record["scale_reencoding"] = "identical per-output coefficient repeated per128 weights; no weight requantization"
                    count += 1
                pack["meta"].update(method=case, original_paper_pack=old, weight_grid_reencoding="same per-output scales repeated in G128 to avoid unscaled FP16 reduction overflow", reencoded_layers=count)
                temp = target.with_suffix(".tmp")
                torch.save(pack, temp)
                temp.replace(target)
            deploy = root / "deployments" / case
            if not deploy.exists():
                export(target, root / "teacher_capture/fp16_model_state.pt", deploy)
            print("PAPER_G128_EXACT_GRID", case, flush=True)
    marker = root / "paper_cfg_reencoding/finished.json"
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps({"built": True, "weight_grid_changed": False, "reason": "Observed real-input Marlin per-output unscaled FP16 overflow; repeat same output scale in G128 metadata"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
