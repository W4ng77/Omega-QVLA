"""Keep exported integer banks unchanged and use proved block rotations."""

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--integer-pack", type=Path, required=True)
    p.add_argument("--block-pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = torch.load(args.integer_pack, map_location="cpu", mmap=True, weights_only=False)
    blocks = torch.load(args.block_pack, map_location="cpu", mmap=True, weights_only=False)
    count = 0
    for name, rec in base.items():
        if not isinstance(rec, dict) or name == "__meta__":
            continue
        src = blocks[name]
        for key in ("duquant_rotation", "duquant_rotation_out"):
            if key in rec:
                if key + "_blocks" not in src:
                    raise ValueError(f"Missing proved block structure: {name}/{key}")
                rec.pop(key)
                count += 1
        for key in ("duquant_rotation_blocks", "duquant_rotation_perm", "duquant_rotation_out_blocks"):
            if key in src:
                rec[key] = src[key]
    base["__meta__"] = dict(
        base.get("__meta__", {}),
        native_eval_rotation="exact_coefficients_block64",
        integer_source=str(args.integer_pack),
        rotation_source=str(args.block_pack),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print({"rotations_replaced": count, "output": str(args.output), "bytes": args.output.stat().st_size})


if __name__ == "__main__":
    main()
