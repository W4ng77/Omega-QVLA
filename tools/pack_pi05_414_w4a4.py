"""Export the Pi-0.5 W4A4 evaluation weights with real INT4 nibble storage."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gr00t.quantization.int4_storage import pack_codebook, pack_rtn, pack_symmetric, unpack_int4
from gr00t.quantization.rtn_layers import _quant_dequant_weight_groupwise


VISION_WEIGHT = re.compile(
    r"^paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))\.weight$"
)
LM_WEIGHT = re.compile(
    r"^paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.\d+\."
    r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)
SUITES = ("long", "spatial", "object", "goal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    nonquantized = {}
    vision_packed = {}
    lm_shapes = {}
    with safe_open(args.checkpoint / "model.safetensors", framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for name in source.keys():
            weight = source.get_tensor(name)
            if VISION_WEIGHT.fullmatch(name):
                group_size = 64
                while weight.shape[1] % group_size:
                    group_size //= 2
                quantized = _quant_dequant_weight_groupwise(weight, group_size, 4).to(weight.dtype)
                packed = pack_rtn(weight, group_size)
                if not torch.equal(quantized, unpack_int4(packed)):
                    raise RuntimeError(f"Vision INT4 round trip failed: {name}")
                vision_packed[name] = packed
                print(f"[vision] {len(vision_packed)}/162 {name}", flush=True)
            elif LM_WEIGHT.fullmatch(name):
                lm_shapes[name] = {"shape": list(weight.shape), "dtype": str(weight.dtype)}
            else:
                nonquantized[name] = weight.contiguous()

    if len(vision_packed) != 162 or len(lm_shapes) != 252:
        raise RuntimeError(f"Expected 162 vision and 252 LM/expert weights; got {len(vision_packed)} and {len(lm_shapes)}")

    save_file(nonquantized, output / "model_nonquant.safetensors", metadata=metadata)
    torch.save(vision_packed, output / "vision_int4.pt")
    shutil.copy2(args.checkpoint / "config.json", output / "config.json")
    shutil.copytree(args.checkpoint / "assets", output / "assets")

    for suite in SUITES:
        source_path = args.pack_root / f"pi05_{suite}" / "quantized.pt"
        source_pack = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        packed_records = {}
        count = 0
        for name, record in source_pack.items():
            if name == "__meta__":
                packed_records[name] = record
                continue
            if not isinstance(record, dict) or "weight_res_q" not in record:
                raise RuntimeError(f"Missing quantized weight record: {name}")
            weight = record["weight_res_q"]
            try:
                packed = pack_symmetric(weight)
            except ValueError:
                packed = pack_codebook(weight, 64)
            if not torch.equal(weight, unpack_int4(packed)):
                raise RuntimeError(f"LM/expert INT4 round trip failed: {name}")
            packed_records[name] = {key: value for key, value in record.items() if key != "weight_res_q"}
            packed_records[name]["weight_res_q_int4"] = packed
            count += 1
            print(f"[{suite}] {count}/252 {name}", flush=True)
        if count != 252 or set(packed_records) - {"__meta__"} != {name[:-7] for name in lm_shapes}:
            raise RuntimeError(f"Expected 252 matching records for {suite}, found {count}")
        target_dir = output / f"pi05_{suite}"
        target_dir.mkdir()
        torch.save(packed_records, target_dir / "quantized.pt")

    manifest = {
        "format": "omega_pi05_w4a4_int4_v1",
        "weight_storage": "real packed INT4 nibbles, with exact round-trip metadata",
        "runtime": "unpacks weights to BF16/FP16 and applies A4 fake quantization; not an INT4 GEMM",
        "vision": {"layers": 162, "method": "RTN", "weight_bits": 4, "activation_bits": 4, "path": "vision_int4.pt"},
        "lm_expert": {"layers": 252, "method": "Omega-QVLA", "weight_bits": 4, "activation_bits": 4, "packs": {suite: f"pi05_{suite}/quantized.pt" for suite in SUITES}},
        "lm_shapes": lm_shapes,
        "total_model_tensors": len(nonquantized) + len(vision_packed) + len(lm_shapes),
        "source_checkpoint": str(args.checkpoint),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved INT4 checkpoint to {output}", flush=True)


if __name__ == "__main__":
    main()
