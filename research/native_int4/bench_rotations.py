"""Compare existing dense/block rotation paths on held-out real inputs."""

import argparse
import json
from pathlib import Path

import torch

from research.native_int4.bench import cold_us, graph_us, nmse


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    pack = torch.load(
        "/home/xinyu/models/omega-qvla-gr00t-w4a4/gr00t_object/quantized.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    factored = torch.load(
        "results/native_int4_20261027/object_block_rotations.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    inputs = torch.load(
        "/home/xinyu/omegaqvla-perstep-4080/results/perstep_full_object_20261027_captures/heldout.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )["inputs"]
    names = ["attn1.to_q", "attn1.to_k", "ff.net.0.proj", "ff.net.2"]
    report = {"gpu": torch.cuda.get_device_name(), "scope": "input rotation only; real step-0 inputs", "rows": []}
    for suffix in names:
        name = "action_head.model.transformer_blocks.0." + suffix
        record, small = pack[name], factored[name]
        r = record["duquant_rotation"].float().cuda()
        b = small["duquant_rotation_blocks"].float().cuda()
        perm = small["duquant_rotation_perm"].cuda()
        for m in (32, 49, 128):
            x = inputs[name][0][:m].cuda().contiguous()
            for autocast in (False, True):
                # Match GptqLinear: buffers FP32, cast to input dtype before matmul.
                rd, bd = r.to(x.dtype), b.to(x.dtype)

                def dense():
                    return x @ rd

                def block():
                    y = x.index_select(-1, perm).reshape(m, b.shape[0], 64)
                    y = torch.bmm(y.transpose(0, 1).contiguous(), bd)
                    return y.transpose(0, 1).contiguous().reshape(m, x.shape[-1])

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                    yd, yb = dense(), block()
                    row = {
                        "layer": name,
                        "M": m,
                        "K": x.shape[-1],
                        "input_dtype": str(x.dtype),
                        "autocast_bf16": autocast,
                        "output_dtype": str(yd.dtype),
                        "nmse": nmse(yb, yd),
                        "max_abs_diff": (yb.float() - yd.float()).abs().max().item(),
                        "dense_us": graph_us(dense),
                        "block_us": graph_us(block),
                        "dense_cold_us": cold_us(dense),
                        "block_cold_us": cold_us(block),
                    }
                # Compare source A4 grid over every step, following rotation/smoothing.
                if "smooth_scale" in record:
                    smooth = record["smooth_scale"].to(yd.device, yd.dtype)
                    yd, yb = yd / smooth, yb / smooth
                mismatch = []
                for scale in record["act_scale_table"]:
                    scale = scale.to(yd.device, yd.dtype)
                    qd = (yd / scale).round().clamp(-8, 7)
                    qb = (yb / scale).round().clamp(-8, 7)
                    mismatch.append((qd != qb).float().mean().item())
                row["max_a4_code_mismatch_fraction_all_steps"] = max(mismatch)
                row["warm_speedup"] = row["dense_us"] / row["block_us"]
                row["cold_speedup"] = row["dense_cold_us"] / row["block_cold_us"]
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(suffix, m, autocast, row, flush=True)


if __name__ == "__main__":
    main()
