"""Independently prove every exported released weight/rotation/scale is preserved."""
import fcntl
import json
from pathlib import Path
import torch
from research.native_int4.audit_rotations import factor
from gr00t.quantization.perstep_backends import unpack_int4


def main():
    root = Path("results/holoq_object")
    lock = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    source = torch.load("/home/xinyu/models/omega-qvla-gr00t-w4a4/gr00t_object/quantized.pt", map_location="cpu", mmap=True, weights_only=False)
    pack = torch.load(root / "deployments/holoq_release_grid/packed.pt", map_location="cpu", mmap=True, weights_only=False)
    frozen = torch.load(root / "released_grid_build/frozen_llm_scales.pt", map_location="cpu", weights_only=False)
    rows = []
    for name, record in pack["layers"].items():
        old = source[name]
        w = old.get("baseline_q", old.get("weight_res_q"))
        q = unpack_int4(record["B"], record["K"]).float()
        recovered = (q * record["s"].repeat_interleave(record["group"], dim=1)).half()
        assert torch.equal(recovered, w.half()), name
        perm, blocks = factor(old["duquant_rotation"])
        assert torch.equal(perm, record["perm"]) and torch.equal(blocks, record["rot_in"]), name
        if record["rot_out"] is not None:
            out_perm, out_blocks = factor(old["duquant_rotation_out"])
            assert torch.equal(out_perm, torch.arange(record["N"])) and torch.equal(out_blocks, record["rot_out"]), name
        if "language_model" in name:
            assert torch.equal(record["a_scale"], frozen[name].float())
        else:
            assert torch.equal(record["a_scale"], old["act_scale_table"].float())
        rows.append({"layer": name, "weight_values_exact": True, "rotation_coefficients_exact": True,
                     "activation_scale_exact": True, "input_shape": [record["N"], record["K"]]})
    assert len(rows) == 180
    result = {"passed": True, "layers": rows, "count": 180, "weight_requantization": False,
              "LLM_scale_source": "Missing from release; immutable original first-warmup scales, recorded separately",
              "compute": "Packed W4 dequantization inside FP16-MMA tile; no dense inference weight cache"}
    target = root / "released_grid_build/export_integrity_audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print("RELEASED_ALL_WEIGHT_ROTATION_SCALE_VALUES_EXACT", len(rows), flush=True)


if __name__ == "__main__":
    main()
