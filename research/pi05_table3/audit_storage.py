"""Reproduce static414-target and complete-model accounting; never VRAM."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import torch
from safetensors import safe_open
from research.pi05_table3.measure import sha, write


def size(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(size(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(size(x) for x in value)
    return 0


def part(name):
    return "vision" if "vision_tower" in name else "expert" if "gemma_expert" in name else "language"


def main():
    root = Path(__file__).resolve().parents[2]
    source = Path("/home/xinyu/models/omega-qvla-pi05-414-w4a4-int4")
    checkpoint = Path("/home/xinyu/models/pi05_libero_pytorch/model.safetensors")
    layers = torch.load(source / "pi05_object/quantized.pt", map_location="cpu", mmap=True, weights_only=False)
    vision = torch.load(source / "vision_int4.pt", map_location="cpu", mmap=True, weights_only=False)
    expected = {n + ".weight" for n in layers if n != "__meta__"} | set(vision)
    raw = defaultdict(lambda: Counter())
    with safe_open(checkpoint, framework="pt", device="cpu") as file:
        for name in file.keys():
            t = file.get_tensor(name)
            bucket = part(name) if name in expected else "other_nonquantized"
            raw[bucket]["tensor_bytes"] += size(t)
            raw[bucket]["tensors"] += 1
    packed = defaultdict(lambda: Counter())
    for name, record in layers.items():
        if name == "__meta__":
            continue
        for key, value in record.items():
            if key == "weight_res_q_int4":
                for field, tensor in value.items():
                    packed[part(name)]["weight_" + field] += size(tensor)
            else:
                packed[part(name)][key] += size(value)
    for name, record in vision.items():
        for field, tensor in record.items():
            packed["vision"][field] += size(tensor)
    payload = sum(raw[p]["tensor_bytes"] for p in ["vision", "language", "expert"])
    qpayload = sum(sum(x.values()) for x in packed.values())
    result = {
        "source_checkpoint_sha256": sha(checkpoint),
        "source_pack_sha256": sha(source / "pi05_object/quantized.pt"),
        "raw_tensor_bytes_by_part": {k: dict(v) for k, v in raw.items()},
        "packed_tensor_bytes_by_part": {k: dict(v) for k, v in packed.items()},
        "original414_weight_bytes": payload,
        "packed414_with_metadata_bytes": qpayload,
        "target_saving_fraction": 1 - qpayload / payload,
        "nonquantized_original_bytes": raw["other_nonquantized"]["tensor_bytes"],
        "total_model_tensor_payload_bytes": qpayload + raw["other_nonquantized"]["tensor_bytes"],
        "scope_note": "Displayed Table3 FP16=5.41GB matches414 target weights to its rounding, not all model parameters. Current historical artifact with necessary metadata does NOT reproduce1.39GB/74.2%.",
        "VRAM_note": "This is static accounting; steady-state and peak runtime memory are independently measured.",
        "deployment_artifacts": {},
    }
    data = root / "results/pi05_table3_414_20261028"
    for family in ["holoq_deployments", "baseline_deployments"]:
        for path in (data / family).glob("*/deployment.json"):
            result["deployment_artifacts"][family + "/" + path.parent.name] = json.loads(path.read_text())
    write(data / "complete_storage_audit.json", result)
    print("STORAGE_AUDIT", payload, qpayload, result["target_saving_fraction"], flush=True)


if __name__ == "__main__":
    main()
