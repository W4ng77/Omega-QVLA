"""Export explicit Nunchaku candidates, preserving integer codes and all banks.

The step-weight variant additionally rounds FP32 bank scales to FP16 because
that is the official INT4 core's contract. This is recorded, never called an
unchanged-grid backend conversion. It needs its own task evaluation if selected.
"""

import argparse
from pathlib import Path
import torch
from gr00t.quantization.perstep_backends import unpack_int4
from research.holoq_bench.nunchaku_probe import pack


CANDIDATES = ["holoq_llm_nunchaku_fusedrot", "holoq_step_weight_nunchaku_f16sw_fusedrot", "smoothquant_a4_nunchaku"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.root / "packs_v6"
    output.mkdir(parents=True, exist_ok=True)
    for original, label in zip(["holoq", "holoq_step_weight", "smoothquant_a4"], CANDIDATES):
        target = output / (label + ".pt")
        if target.exists():
            print("REUSE_NUNCHAKU_EXPORT", label, flush=True)
            continue
        source = args.root / "packs_v2" / (original + ".pt")
        obj = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
        if original == "smoothquant_a4":
            cache = {
                e["name"]: e["canonical"]
                for file in sorted((args.root / "packs/layer_cache/smoothquant").glob("*.pt"))
                for e in [torch.load(file, map_location="cpu", weights_only=False)]
            }
        else:
            cache = {}
            for file in sorted((args.root / "packs/layer_cache/holoq").glob("*.pt")):
                e = torch.load(file, map_location="cpu", weights_only=False)
                if "language_model" in e["name"]:
                    cache[e["name"]] = e["canonical"]
        converted, relative = 0, 0.0
        for name, old in obj["layers"].items():
            if original == "holoq" and "language_model" not in name:
                continue
            record = {k: v for k, v in old.items() if k not in ("B", "s", "z", "s2", "z2", "integer")}
            n, k = old["N"], old["K"]
            if old["backend"] == "native_s4":
                integer = old["integer"]
                assert integer["mode"] == "step_weight" and integer["group_size"] == 128
                weights, scales = [], []
                for q, sw in zip(integer["packed_weight"], integer["weight_scales"]):
                    codes = unpack_int4(q.cuda(), k).to(torch.int8)
                    half_scales = sw.half()
                    assert (half_scales > 0).all() and torch.isfinite(half_scales).all()
                    relative = max(relative, ((half_scales.float() - sw).abs() / sw).max().item())
                    bp, sp = pack(codes, half_scales.cuda(), 128, True)
                    weights.append(bp.cpu())
                    scales.append(sp.cpu())
                record.update(B=torch.stack(weights), s=torch.stack(scales), logical_weight_group=128)
                assert record["B"].shape == (9, n, k // 8)
            else:
                canonical = cache[name]
                assert canonical["scales"].shape == (n, 1)
                bp, sp = pack(canonical["q"].cuda(), canonical["scales"].half().cuda(), k, True)
                record.update(B=bp.cpu(), s=sp.cpu(), logical_weight_group=k)
            record.update(backend="nunchaku_s4", physical_scale_group=64)
            obj["layers"][name] = record
            converted += 1
        assert converted == (84 if original == "holoq" else 180), (label, converted)
        obj["meta"].update(
            method=label,
            runtime_plugin="nunchaku_s4",
            rotation_plugin="nunchaku_and_triton_block64" if original != "smoothquant_a4" else None,
            native_backend="official Nunchaku INT4 core/default epilogue, no low-rank branch",
            source_backend_pack=str(source.resolve()),
            unchanged_integer_codes=True,
            weight_scale_change="FP32 step-bank scales rounded to FP16" if original == "holoq_step_weight" else "none",
            max_weight_scale_relative_cast_error=relative,
            native_int4_layers=converted,
            scale_layout="Repeat logical row/G128 scales into official G64 layout; no new groupwise code quantization",
        )
        torch.save(obj, target)
        print("EXPORTED_NUNCHAKU", label, converted, "scale_cast_max_relative", relative, flush=True)


if __name__ == "__main__":
    main()
