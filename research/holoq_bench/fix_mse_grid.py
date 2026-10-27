"""New exports using the released QuantVLA five-candidate MSE grid.

Preserve the first exploratory exports and their calibration logs. Reuse only
frozen rotations and teacher inputs; recompute weight codes from teacher W.
"""

import argparse
from pathlib import Path
import torch
from research.holoq_bench import kernels, quantizers
from research.holoq_bench.build_packs import export
import json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.capture / "manifest.json").read_text())
    weights = torch.load(a.capture / "weights.pt", mmap=True, weights_only=False)
    for method in ["duquant", "quantvla"]:
        cache = a.output / "layer_cache" / method
        cache.mkdir(parents=True, exist_ok=True)
        for path in sorted((a.old / "layer_cache" / method).glob("*.pt")):
            entry = torch.load(path, map_location="cpu", weights_only=False)
            rr = entry["record"]
            name = entry["name"]
            w = weights[name]["weight"].cuda().float()[:, rr["perm"].cuda()]
            n, k = w.shape
            w = (
                torch.bmm(w.reshape(n, k // 64, 64).transpose(0, 1), rr["rot_in"].cuda().float())
                .transpose(0, 1)
                .reshape(n, k)
            )
            w = torch.bmm(rr["rot_out"].cuda().float(), w.reshape(n // 64, 64, k)).reshape(n, k)
            q, s = quantizers.rtn(w, scale_half=True, mse=True)
            rr.update(kernels.pack_marlin(q.cpu(), s.cpu()))
            entry.update(canonical={"q": q.cpu(), "scales": s.cpu(), "group": k})
            entry["details"]["mse_candidates"] = (
                "base / [0.5, 0.75, 1.0, 1.25, 1.5]; FP16 deployed scales frozen before q"
            )
            torch.save(entry, cache / path.name)
        export(method, a.output, manifest, cache)
        print("REBUILT", method, flush=True)
    for path in a.old.glob("*.pt"):
        if path.stem.startswith(("duquant", "quantvla")):
            continue
        target = a.output / path.name
        if not target.exists():
            # Immutable hard-link for unchanged packs, not a second quantization.
            target.hardlink_to(path.resolve())


if __name__ == "__main__":
    main()
