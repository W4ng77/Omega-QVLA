"""Real GPTQ/Marlin252 and disclosed SmoothQuant-W4A8/QServe414 controls.

Available10 initial observations/all tokens, not submission10-trajectories.
No success claims; same real packed kernels are timed irrespective of scores.
"""

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from research.pi05_table3.measure import sha, write
from research.pi05_table3 import kernels

ROOT = Path(__file__).resolve().parents[2]


def gptq(w, h, count):
    path = ROOT / "third_party/gptq"
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("pi05_official_gptq", path / "gptq.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="cuda", dtype=torch.float32)
    layer.weight.data.copy_(w)
    solver = module.GPTQ(layer)
    solver.H = h
    solver.nsamples = count
    solver.quantizer = module.Quantizer()
    solver.quantizer.configure(4, perchannel=True, sym=True, mse=False)
    original = solver.quantizer.find_params
    history = []

    def parameters(x, weight=False):
        original(x, weight=weight)
        # Explicit Marlin FP16 scale contract; fixed BEFORE reconstruction.
        solver.quantizer.scale = solver.quantizer.scale.half().float().clamp_min(2**-24)
        history.append(solver.quantizer.scale.detach().clone())

    solver.quantizer.find_params = parameters
    solver.fasterquant(blocksize=128, percdamp=0.01, groupsize=128, actorder=False)
    assert len(history) == w.shape[1] // 128 + 1
    scales = torch.cat(history[1:], 1)
    expanded = scales.repeat_interleave(128, 1)
    q = (layer.weight / expanded).round().clamp(-8, 7).to(torch.int8)
    assert torch.equal(q.float() * expanded, layer.weight)
    solver.free()
    return q, scales.half()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--kind", choices=["gptq_marlin252", "smoothquant_qserve414"], required=True)
    a = p.parse_args()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = ROOT / "results/pi05_table3_414_20261028"
    capture = data / "baseline_capture"
    manifest = json.loads((capture / "manifest.json").read_text())
    out = data / "baseline_build" / a.kind
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "layers"
    cache.mkdir(exist_ok=True)
    selected = [x for x in manifest["layers"] if a.kind.endswith("414") or "vision_tower" not in x["name"]]
    assert len(selected) == (414 if a.kind.endswith("414") else 252)
    source = Path(manifest["checkpoint"])
    configuration = {
        "method": a.kind,
        "calibration_manifest_sha256": sha(capture / "manifest.json"),
        "source_builder_sha256": sha(Path(__file__)),
        "calibration": manifest["calibration"],
        "group_size": 128
        if a.kind.startswith("gptq")
        else "per-output weight quantization encoded with QServeG128 metadata",
        "gptq": {
            "processing_block": 128,
            "damping": 0.01,
            "actorder": False,
            "solver": "pinned official full-H GPTQ",
            "scale_dtype": "FP16 chosen before reconstruction",
        }
        if a.kind.startswith("gptq")
        else None,
        "smoothquant": {
            "alpha": 0.5,
            "activation_peak_cap": 1000,
            "activation_quantizer": "per-token symmetricA8,FP16 scales",
            "weight_quantizer": "per-output symmetricW4 RTN,FP16 scales; explicit SmoothQuant-W4 deployment extension, not originalW8 SmoothQuant",
        }
        if a.kind.startswith("smooth")
        else None,
    }
    configpath = out / "configuration.json"
    if configpath.exists():
        assert json.loads(configpath.read_text()) == configuration
    else:
        write(configpath, configuration)
    records = {}
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as file:
        for item in selected:
            name, index, k, n = item["name"], item["index"], item["K"], item["N"]
            target = cache / f"{index:03d}.pt"
            if target.exists():
                entry = torch.load(target, map_location="cpu", mmap=True, weights_only=False)
                assert entry["name"] == name
            else:
                w = file.get_tensor(name + ".weight").float().cuda()
                bias = file.get_tensor(name + ".bias") if name + ".bias" in file.keys() else None
                count = 0
                if a.kind.startswith("gptq"):
                    h = torch.zeros(k, k, device="cuda", dtype=torch.float32)
                    for frame in range(10):
                        banks = torch.load(
                            capture / f"{frame:02d}/{index:03d}.pt",
                            map_location="cpu",
                            mmap=True,
                            weights_only=False,
                        )
                        for values in banks.values():
                            for start in range(0, len(values), 1024):
                                x = values[start : start + 1024].cuda().float()
                                h.addmm_(x.t(), x)
                                count += len(x)
                    h *= 2 / count
                    q, s = gptq(w, h, count)
                    record = kernels.pack_marlin(q.cpu(), s.cpu())
                    record.update(
                        K=k, N=n, original_K=k, original_N=n, bias=bias, external_dtype=item["weight_dtype"]
                    )
                    smooth = torch.empty(0)
                    del h, x
                else:
                    peak = torch.zeros(k, device="cuda", dtype=torch.float32)
                    for frame in range(10):
                        banks = torch.load(
                            capture / f"{frame:02d}/{index:03d}.pt",
                            map_location="cpu",
                            mmap=True,
                            weights_only=False,
                        )
                        for values in banks.values():
                            peak = torch.maximum(peak, values.cuda().float().abs().amax(0))
                            count += len(values)
                    smooth = (peak.clamp(1e-5, 1000).sqrt() / w.abs().amax(0).clamp_min(1e-5).sqrt()).clamp_min(
                        1e-5
                    )
                    w = w * smooth[None]
                    s = (w.abs().amax(1, keepdim=True) / 7).clamp_min(2**-24).half()
                    q = (w / s.float()).round().clamp(-8, 7).to(torch.int8)
                    kp, npad = (k + 127) // 128 * 128, (n + 127) // 128 * 128
                    qp = torch.zeros(npad, kp, device="cuda", dtype=torch.int8)
                    qp[:n, :k] = q
                    sp = torch.ones(npad, 1, device="cuda", dtype=torch.float16)
                    sp[:n] = s
                    record = kernels.pack_qserve(qp.cpu(), sp.cpu())
                    record.update(
                        K=kp,
                        N=npad,
                        original_K=k,
                        original_N=n,
                        bias=bias,
                        smooth=smooth.cpu(),
                        external_dtype=item["weight_dtype"],
                    )
                    del peak, qp, sp
                entry = {
                    "name": name,
                    "record": record,
                    "canonical_codes": q.cpu(),
                    "canonical_scales": s.cpu(),
                    "calibration_rows": count,
                }
                torch.save(entry, target)
                del w, q, s, smooth
                torch.cuda.empty_cache()
            records[name] = entry["record"]
            print("BUILT_PACKED_BASELINE", a.kind, len(records), "/", len(selected), name, flush=True)
    deployment = data / "baseline_deployments" / a.kind
    deployment.mkdir(exist_ok=True)
    torch.save({"meta": configuration, "layers": records}, deployment / "packed.pt")
    keep = {}
    excluded = {n + ".weight" for n in records}
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as file:
        metadata = file.metadata()
        for name in file.keys():
            if name not in excluded:
                keep[name] = file.get_tensor(name)
    save_file(keep, deployment / "model_nonquant.safetensors", metadata=metadata)
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
            "method": a.kind,
            "source_checkpoint": str(source),
            "layers": len(records),
            "configuration": configuration,
            "files": inventory,
            "model_artifact_bytes": sum(x["bytes"] for x in inventory.values()),
            "storage_boundary": "One task-suite policy model artifacts, including nonquantized weights/config/norm assets; shared tokenizer/runtime software accounted separately.",
            "excluded_state_keys": sorted(excluded),
        },
    )


if __name__ == "__main__":
    main()
