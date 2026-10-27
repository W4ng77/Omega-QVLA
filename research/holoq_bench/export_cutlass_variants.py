"""Backend-only exports for methods whose original scales are separable."""

import argparse
from pathlib import Path
import torch
from gr00t.quantization.perstep_backends import pack_int4


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=True)
    variants = [
        ("smoothquant_a4", "smoothquant_a4_cutlass", "smoothquant", 180),
        ("holoq", "holoq_llm_s4", "holoq", 84),
        ("holoq_step_weight", "holoq_step_weight_llm_s4", "holoq", 84),
    ]
    for original, label, method, expected in variants:
        pack = torch.load(a.packs / (original + ".pt"), map_location="cpu", mmap=True, weights_only=False)
        count = 0
        for file in sorted((a.cache / method).glob("*.pt")):
            e = torch.load(file, map_location="cpu", weights_only=False)
            name = e["name"]
            if method == "holoq" and "language_model" not in name:
                continue
            r = pack["layers"][name]
            assert r["act_bits"] == 4 and r["act_mode"] == "token"
            c = e["canonical"]
            assert c["scales"].shape == (r["N"], 1)
            r = {k: v for k, v in r.items() if k not in ("B", "s", "z", "z2", "s2")}
            r.update(B=pack_int4(c["q"]), s=c["scales"].half(), backend="cutlass_s4")
            pack["layers"][name] = r
            count += 1
        assert count == expected, (label, count)
        pack["meta"].update(
            method=label,
            runtime_plugin="cutlass_s4",
            exact_grid_backend_conversion="unchanged signed weight codes and FP16 scales, unchanged token quantization scale/clamp; integer dot and FP32 rescale",
            converted_layers=count,
            source_backend_pack=str((a.packs / (original + ".pt")).resolve()),
        )
        target = a.output / (label + ".pt")
        torch.save(pack, target)
        print("EXPORTED_BACKEND_ONLY", target, count, flush=True)


if __name__ == "__main__":
    main()
