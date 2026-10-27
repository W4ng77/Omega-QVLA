"""Backend-only variants: unchanged packed codes, scales, rotations and ATM."""

import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    a = p.parse_args()
    torch.set_num_threads(1)
    variants = [
        ("packs_v4/holoq_llm_s4.pt", "holoq_llm_s4_fusedrot"),
        ("packs_v4/holoq_step_weight_llm_s4.pt", "holoq_step_weight_llm_s4_fusedrot"),
        ("packs_v3/duquant_a8_g128.pt", "duquant_a8_g128_fusedrot"),
        ("packs_v3/duquant_a4_g128.pt", "duquant_a4_g128_fusedrot"),
        ("packs_v3/quantvla_selective_a4_atm_g128.pt", "quantvla_selective_a4_atm_g128_fusedrot"),
        ("packs_v3/quantvla_selective_a8_atm_g128.pt", "quantvla_selective_a8_atm_g128_fusedrot"),
    ]
    out = a.root / "packs_v5"
    out.mkdir(parents=True, exist_ok=True)
    for relative, label in variants:
        source = a.root / relative
        if (out / (label + ".pt")).exists():
            print("REUSE_IMMUTABLE_EXPORT", label, flush=True)
            continue
        pack = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
        pack["meta"].update(
            method=label,
            rotation_plugin="triton_block64",
            rotation_source_pack=str(source.resolve()),
            rotation_conversion="layout/permutation/block product fusion only; all R, codes and scales unchanged",
        )
        torch.save(pack, out / (label + ".pt"))
        print("EXPORTED_FUSED_ROTATION", label, flush=True)


if __name__ == "__main__":
    main()
