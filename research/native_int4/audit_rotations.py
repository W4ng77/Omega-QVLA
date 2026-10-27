"""Losslessly replace permutation/block-diagonal dense rotation storage."""

import argparse
import json
from pathlib import Path

import torch


def factor(rotation, block=64):
    k = rotation.shape[0]
    if rotation.shape != (k, k) or k % block:
        raise ValueError("Expected a square 64-divisible rotation")
    nb = k // block
    energy = rotation.float().abs().reshape(k, nb, block).sum(-1)
    owner = energy.argmax(1)
    off = energy.clone()
    off.scatter_(1, owner[:, None], 0)
    if torch.count_nonzero(off) or not torch.equal(torch.bincount(owner, minlength=nb), torch.full((nb,), block)):
        raise ValueError("Rotation is not a row-permuted block diagonal matrix")
    order = torch.argsort(owner * k + torch.arange(k))
    blocks = torch.stack(
        [rotation[order[b * block : (b + 1) * block], b * block : (b + 1) * block] for b in range(nb)]
    ).contiguous()
    # Off-block elements are exactly zero; diagonal tiles and the row order
    # determine every original entry. Check selected blocks against dense R.
    for b in range(nb):
        if not torch.equal(blocks[b], rotation[order[b * block : (b + 1) * block], b * block : (b + 1) * block]):
            raise RuntimeError("Non-lossless decomposition")
    return order, blocks


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pack", default="/home/xinyu/models/omega-qvla-gr00t-w4a4/gr00t_object/quantized.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-pack", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    pack = torch.load(args.pack, map_location="cpu", mmap=True, weights_only=False)
    output = dict(pack)
    report = {
        "source_pack": args.pack,
        "block": 64,
        "rotations": [],
        "source_dense_bytes": 0,
        "block_representation_bytes": 0,
        "runtime_dense_fp32_bytes": 0,
        "runtime_block_fp32_bytes": 0,
    }
    for name, record in pack.items():
        if not isinstance(record, dict) or name == "__meta__":
            continue
        changed = dict(record)
        for key in ("duquant_rotation", "duquant_rotation_out"):
            if key not in record:
                continue
            r = record[key]
            order, blocks = factor(r)
            identity = torch.equal(order, torch.arange(r.shape[0]))
            if key.endswith("_out") and not identity:
                raise ValueError("Output permutation needs an explicit runtime implementation")
            changed.pop(key)
            if key.endswith("_out"):
                changed["duquant_rotation_out_blocks"] = blocks
            else:
                changed["duquant_rotation_blocks"] = blocks
                changed["duquant_rotation_perm"] = order
            dense_bytes = r.numel() * r.element_size()
            block_bytes = blocks.numel() * blocks.element_size() + (
                0 if key.endswith("_out") else order.numel() * 8
            )
            report["source_dense_bytes"] += dense_bytes
            report["block_representation_bytes"] += block_bytes
            report["runtime_dense_fp32_bytes"] += r.numel() * 4
            report["runtime_block_fp32_bytes"] += blocks.numel() * 4 + (
                0 if key.endswith("_out") else order.numel() * 8
            )
            report["rotations"].append(
                {
                    "layer": name,
                    "key": key,
                    "K": r.shape[0],
                    "exact_values": True,
                    "identity_permutation": identity,
                    "dense_bytes": dense_bytes,
                    "block_bytes": block_bytes,
                }
            )
        output[name] = changed
    report["count"] = len(report["rotations"])
    report["runtime_rotation_saving_bytes"] = (
        report["runtime_dense_fp32_bytes"] - report["runtime_block_fp32_bytes"]
    )
    if args.save_pack:
        args.save_pack.parent.mkdir(parents=True, exist_ok=True)
        if args.save_pack.exists():
            raise FileExistsError(args.save_pack)
        torch.save(output, args.save_pack)
        report["factored_pack"] = str(args.save_pack)
        report["factored_pack_bytes"] = args.save_pack.stat().st_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "rotations"}, flush=True)


if __name__ == "__main__":
    main()
