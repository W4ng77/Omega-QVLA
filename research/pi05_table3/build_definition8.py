"""Freeze submitted-definition LLM/head with Table3's released vision RTN.

This is a disclosed reconstruction, not an assertion that10 initial snapshots
equal the submission's10 trajectories. Vision RTN retains the released grid.
"""

import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import torch
from safetensors import safe_open
from research.pi05_table3.measure import sha, write
from research.pi05_table3.paper_math import (
    direct_input_rotation,
    block_transform,
    positive_scale,
    signed15,
    rtn_per_output,
    pack_int4,
)

ROOT = Path(__file__).resolve().parents[2]


def gptq15(w, h, count):
    path = ROOT / "third_party/gptq"
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("pi05_definition_gptq", path / "gptq.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="cuda", dtype=torch.float32)
    layer.weight.data.copy_(w)
    solver = module.GPTQ(layer)
    solver.H = h
    solver.nsamples = count
    solver.quantizer = module.Quantizer()
    solver.quantizer.configure(4, perchannel=True, sym=True, mse=False)

    def parameters(values, weight=False):
        assert weight
        solver.quantizer.scale = positive_scale(values.abs().amax(1, keepdim=True))
        solver.quantizer.zero = torch.full_like(solver.quantizer.scale, 7)
        solver.quantizer.maxq = torch.tensor(14, device=values.device)

    solver.quantizer.find_params = parameters
    solver.fasterquant(blocksize=128, percdamp=0.01, groupsize=-1, actorder=False)
    scales = solver.quantizer.scale.detach().clone()
    q = signed15(layer.weight, scales)
    assert torch.equal(q.float() * scales, layer.weight)
    solver.free()
    return q, scales


def main():
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = ROOT / "results/pi05_table3_414_20261028"
    capture = data / "definition8_capture"
    manifest = json.loads((capture / "manifest.json").read_text())
    assert manifest["steps"] == 8
    out = data / "definition8_build"
    out.mkdir(exist_ok=True)
    (out / "layers").mkdir(exist_ok=True)
    config = {
        "quantization_source": "User-verified8-page submitted definition; see original rebuttal submission_contract.md",
        "capture_sha256": sha(capture / "manifest.json"),
        "math_sha256": sha(Path(__file__).with_name("paper_math.py")),
        "builder_sha256": sha(Path(__file__)),
        "core_scope": "126 language+126 expert; input-only block64 direct left-SVD U D H, no output rotation/lowrank",
        "permutation": "serpentine_descending_input_energy_v1; one disclosed interpretation, NOT uniquely specified by submission",
        "D_seed": "42+frozen capture layer index, not selected by scores",
        "core_grid": "[-7,7],FP32 maxabs/7 per-output weight scales acrossFULL K",
        "language": "Official full-H GPTQ,processing128,damping.01,no weight grouping,noactorder;dynamic per-tokenA4",
        "expert": "RTN W4,per-layer/step/input-channelA4;q999 explicitly a code-based robust estimator choice;8 steps",
        "vision": "162 released vision RTN records unchanged: asymmetricU4,G64/G16;A4 per-token minmax. Submission body did not uniquely specify vision recipe; explicitly separate from core.",
        "calibration": manifest["calibration"],
        "calibration_caveat": "10 initial Object observations, not10 full trajectories; this is deployment reconstruction, not paper experimental reproduction",
        "dtype": "Teacher BF16 operands;rotation/scales/quantizer mathFP32;runtime nativeLLM and packed floating-channel head with original grids",
    }
    path = out / "configuration.json"
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        write(path, config)
    selected = [x for x in manifest["layers"] if "vision_tower" not in x["name"]]
    assert len(selected) == 252
    records = {}
    audit = []
    source = Path(manifest["checkpoint"])
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as file:
        for item in selected:
            name, i = item["name"], item["index"]
            target = out / "layers" / f"{i:03d}.pt"
            if target.exists():
                entry = torch.load(target, map_location="cpu", mmap=True, weights_only=False)
            else:
                w = file.get_tensor(name + ".weight").float().cuda()
                wr, r, p, signs = direct_input_rotation(w, 42 + i)
                test = torch.randn(
                    3, w.shape[1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(42 + i)
                )
                expected = test @ w.t()
                actual = block_transform(test, r, p) @ wr.t()
                error = float((actual - expected).square().sum() / expected.square().sum())
                assert error < 1e-9, (name, error)
                llm = "language_model" in name
                count = 0
                scale_rows = {}
                if llm:
                    h = torch.zeros(w.shape[1], w.shape[1], device="cuda", dtype=torch.float32)
                    for frame in range(10):
                        bank = torch.load(
                            capture / f"{frame:02d}/{i:03d}.pt", map_location="cpu", mmap=True, weights_only=False
                        )
                        assert set(bank) == {-1}
                        for start in range(0, len(bank[-1]), 1024):
                            xr = block_transform(bank[-1][start : start + 1024].cuda(), r, p).reshape(
                                -1, w.shape[1]
                            )
                            h.addmm_(xr.t(), xr)
                            count += len(xr)
                    h *= 2 / count
                    q, sw = gptq15(wr, h, count)
                    sa = torch.empty(0)
                    del h, xr
                else:
                    q, sw = rtn_per_output(wr)
                    tables = []
                    for step in range(8):
                        chunks = []
                        for frame in range(10):
                            bank = torch.load(
                                capture / f"{frame:02d}/{i:03d}.pt",
                                map_location="cpu",
                                mmap=True,
                                weights_only=False,
                            )
                            chunks.append(block_transform(bank[step].cuda(), r, p).reshape(-1, w.shape[1]))
                        values = torch.cat(chunks)
                        peak = torch.quantile(values.abs(), 0.999, dim=0)
                        tables.append(positive_scale(peak).cpu())
                        scale_rows[str(step)] = len(values)
                    sa = torch.stack(tables)
                    count = sum(scale_rows.values())
                    del chunks, values, peak
                record = {
                    "K": w.shape[1],
                    "N": w.shape[0],
                    "B": pack_int4(q).cpu(),
                    "s": sw.cpu(),
                    "rot_in": r.cpu(),
                    "perm": p.cpu(),
                    "a_scale": sa,
                    "bias": file.get_tensor(name + ".bias") if name + ".bias" in file.keys() else None,
                    "act_mode": "token" if llm else "step",
                }
                entry = {
                    "name": name,
                    "record": record,
                    "D_signs": signs,
                    "audit": {
                        "name": name,
                        "unquantized_transform_nmse": error,
                        "calibration_rows": count,
                        "per_step_rows": scale_rows,
                        "qmin": int(q.min()),
                        "qmax": int(q.max()),
                        "scale_shape": list(sw.shape),
                        "scale_dtype": str(sw.dtype),
                        "zero_weight_scale_guards": int((wr.abs().amax(1) == 0).sum()),
                    },
                }
                torch.save(entry, target)
                del w, wr, r, p, q, sw, test, expected, actual
                torch.cuda.empty_cache()
            assert entry["name"] == name
            records[name] = entry["record"]
            audit.append(entry["audit"])
            print("DEFINITION8_FROZEN", len(records), "/252", name, flush=True)
    deployment = data / "definition8_deployment"
    deployment.mkdir(exist_ok=True)
    torch.save({"meta": config, "layers": records}, deployment / "packed_core.pt")
    released = Path("/home/xinyu/models/omega-qvla-pi05-414-w4a4-int4")
    shutil.copy2(released / "vision_int4.pt", deployment / "vision_int4.pt")
    shutil.copy2(released / "model_nonquant.safetensors", deployment / "model_nonquant.safetensors")
    shutil.copy2(source / "config.json", deployment / "config.json")
    shutil.copytree(source / "assets", deployment / "assets", dirs_exist_ok=True)
    inventory = {
        str(f.relative_to(deployment)): {"bytes": f.stat().st_size, "sha256": sha(f)}
        for f in deployment.rglob("*")
        if f.is_file()
    }
    write(
        deployment / "deployment.json",
        {
            "method": "definition8_core_with_released_vision_rtn",
            "scope": {"vision": 162, "language": 126, "expert": 126},
            "configuration": config,
            "files": inventory,
            "excluded_state_keys": sorted(
                {n + ".weight" for n in records}
                | set(torch.load(deployment / "vision_int4.pt", map_location="cpu", mmap=True, weights_only=False))
            ),
            "model_artifact_bytes": sum(x["bytes"] for x in inventory.values()),
            "calibration_caveat": config["calibration_caveat"],
        },
    )
    write(
        out / "audit.json",
        {"configuration": config, "layers": audit, "frozen_pack_sha256": sha(deployment / "packed_core.pt")},
    )


if __name__ == "__main__":
    main()
