"""Repeat identical row scales in Marlin G128: no weight requantization.

The per-output kernel's internal unscaled FP16 reductions overflow for some
SVD-rotated activations even when the mathematical output is finite. G128
applies the same coefficient earlier in reduction. Keep codes unchanged and
repeat each existing output scale; include extra scale storage in deployment.
"""

import argparse
from pathlib import Path
import torch
from research.holoq_bench.kernels import pack_marlin


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--packs", type=Path, nargs="+", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=True)
    for path in a.packs:
        method = "quantvla" if path.stem.startswith("quantvla") else "duquant"
        pack = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        entries = {
            e["name"]: e
            for f in (a.cache / method).glob("*.pt")
            for e in [torch.load(f, map_location="cpu", weights_only=False)]
        }
        for name, r in pack["layers"].items():
            c = entries[name]["canonical"]
            assert c["scales"].shape == (r["N"], 1)
            repeats = c["scales"].repeat(1, r["K"] // 128)
            assert torch.equal(repeats[:, 0], c["scales"][:, 0])
            r.update(pack_marlin(c["q"], repeats))
            r["scale_reencoding"] = "same row scale repeated per 128 weights; identical quantized grid"
        pack["meta"].update(
            weight_grid_reencoding="Marlin G128 with repeated identical per-output scale; zero requantization",
            original_pack=str(path.resolve()),
        )
        output = a.output / (path.stem + "_g128.pt")
        torch.save(pack, output)
        print("EXACT_GRID_REENCODE", output, len(pack["layers"]), flush=True)


if __name__ == "__main__":
    main()
