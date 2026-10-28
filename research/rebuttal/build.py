"""Frozen submission-contract reconstruction; never selects settings by scores.

Four resumable stages: transform, capture ALL teacher tokens, quantize, export.
All artifacts carry a configuration identity; interrupted frames are discarded.
The 10x10-frame budget, q99.9, signs, zigzag and dtypes are disclosed choices.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.deployment import export, sha256
from research.rebuttal.method import (
    block_transform, direct_input_rotation, gptq_per_output_signed15,
    positive_scale, rtn_per_output,
)

ROOT = Path(__file__).resolve().parents[2]


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def tensor_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def source_identity(args):
    previous = json.loads((args.source_capture / "manifest.json").read_text())
    calibration = json.loads((args.calibration / "manifest.json").read_text())
    assert calibration["trajectory_count"] == 10 and calibration["observation_count"] == 100
    assert previous["checkpoint"] == calibration["checkpoint"]
    assert calibration["denoising_steps"] == 8 and calibration["action_horizon"] == 16
    sources = {"teacher_weights": args.source_capture / "weights.pt",
               "teacher_manifest": args.calibration / "manifest.json",
               "selected_observations": args.calibration / "observations.pt",
               "math": ROOT / "research/rebuttal/method.py",
               "builder": Path(__file__)}
    sources.update({f"trajectory_{i}": args.calibration / f"trajectory_{i:02d}.pt" for i in range(10)})
    sources.update({"checkpoint_" + p.name: p for p in Path(calibration["checkpoint"]).glob("*.safetensors")})
    config = {
        "format": "submission_contract_v2", "checkpoint": calibration["checkpoint"],
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "source_paths": {name: str(path.resolve()) for name, path in sources.items()},
        "rotation": "direct left SVD; canonical column sign; U D H64/8; input only",
        "permutation": "serpentine_descending_input_energy_v1 (one disclosed interpretation)",
        "sign_seed": "42 + frozen layer index", "grid": [-7, 7],
        "weight_scale": "one FP32 coefficient per output over FULL K; maxabs/7",
        "llm": "official GPTQ full-H, block128, damping.01, noactorder, NO weight grouping; per-token A",
        "dit": "RTN; per-step/layer/input-channel A; q99.9 from historical CODE, not specified by submission",
        "token_sampling": "ALL tokens in each of100 selected teacher calls; no52-token cap",
        "frame_sampling": "10 uniform frames from each of10 actual saved FP16 trajectories at state0",
        "calibration_identity_caveat": "Disclosed reconstruction; historical frame/token/source details remain unresolved",
        "teacher_successes": sum(bool(t["success"]) for t in calibration["trajectories"]),
        "teacher_trajectory_count": len(calibration["trajectories"]),
        "steps": 8, "action_horizon": 16, "batch": 1,
        "dtypes": "FP16 Linear operands from teacher; rotation/scales/quantizer construction FP32",
        "zero_peak_guard": "scale1 when peak0, separately count any nonzero values in guarded channels",
        "activation_order": "Capture teacher x, cast to FP16 like base Linear, then FP32 input transform",
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = args.output / "configuration.json"
    if path.exists():
        assert json.loads(path.read_text())["identity"] == identity, "Configuration changed; use a NEW output directory"
    else:
        json_write(path, {"identity": identity, "config": config, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "created": time.time()})
    return identity, previous, config


def transforms(args, identity, manifest):
    weights = load(args.source_capture / "weights.pt")
    audit = []
    for i, item in enumerate(manifest["layers"]):
        name = item["name"]
        path = args.output / "transforms" / f"{i:03d}.pt"
        if path.exists():
            entry = load(path)
            assert entry["identity"] == identity and entry["name"] == name
        else:
            w = weights[name]["weight"].cuda().float()
            wr, r, p, signs = direct_input_rotation(w, 42 + i)
            eye = torch.eye(64, device="cuda")[None]
            orth = float((r.transpose(1, 2) @ r - eye).abs().max())
            x = torch.randn(3, w.shape[1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(42 + i))
            expected = x @ w.t()
            actual = block_transform(x, r, p) @ wr.t()
            nmse = float((actual - expected).square().sum() / expected.square().sum().clamp_min(1e-30))
            assert orth < 2e-5 and nmse < 1e-9, (name, orth, nmse)
            entry = {"identity": identity, "name": name, "index": i, "K": w.shape[1], "N": w.shape[0],
                     "rotated_weight": wr.cpu(), "rot_in": r.cpu(), "perm": p.cpu(), "signs": signs,
                     "d_seed": 42 + i, "bias": weights[name]["bias"],
                     "unquantized_nmse": nmse, "orthogonality_max_abs": orth}
            tensor_write(path, entry)
            del w, wr, r, p, x, expected, actual
        audit.append({k: entry[k] for k in ("name", "K", "N", "d_seed", "unquantized_nmse", "orthogonality_max_abs")})
        if i % 10 == 9:
            print("TRANSFORMS", i + 1, "/", len(manifest["layers"]), flush=True)
    json_write(args.output / "transform_audit.json", {"identity": identity, "layers": audit})


def capture(args, identity, manifest, config):
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from gr00t.quantization.dit_step_context import get_current_dit_step

    c = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    seed(42)
    policy = Gr00tPolicy(model_path=config["checkpoint"], modality_config=c.modality_config(), modality_transform=c.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    actual = dict(policy.model.named_modules())
    weights = load(args.source_capture / "weights.pt")
    target_names = [item["name"] for item in manifest["layers"]]
    assert len(target_names) == 180 and len(set(target_names)) == 180
    for name in target_names:
        assert isinstance(actual[name], torch.nn.Linear)
        assert torch.equal(actual[name].weight.detach().cpu(), weights[name]["weight"]), name
        bias = actual[name].bias
        assert (bias is None and weights[name]["bias"] is None) or torch.equal(bias.detach().cpu(), weights[name]["bias"]), name
    state = args.output / "fp16_model_state.pt"
    if not state.exists():
        tensor_write(state, {k: v.detach().cpu() for k, v in policy.model.state_dict().items()})
    rotations = {i: load(args.output / "transforms" / f"{i:03d}.pt") for i in range(180)}
    rotations = {i: (r["rot_in"].cuda(), r["perm"].cuda()) for i, r in rotations.items()}
    current = {}
    dtype_seen = {name: set() for name in target_names}

    def hook(i, name):
        def record(module, inputs):
            step = get_current_dit_step()
            bank = -1 if step is None else int(step)
            x = inputs[0].detach()
            dtype_seen[name].add(str(x.dtype))
            bank_data = current.setdefault(i, {})
            assert bank not in bank_data, (name, bank, "unexpected repeated call")
            transformed = block_transform(x.half().float(), *rotations[i]).reshape(-1, module.in_features).cpu()
            assert bool(torch.isfinite(transformed).all()), (name, bank)
            bank_data[bank] = transformed
        return record

    handles = [actual[name].register_forward_pre_hook(hook(i, name)) for i, name in enumerate(target_names)]
    samples = load(args.calibration / "observations.pt")
    ledger = []
    try:
        with torch.inference_mode():
            for j, sample in enumerate(samples):
                folder = args.output / "capture" / f"{j:03d}"
                folder.mkdir(exist_ok=True)
                complete = folder / "complete.json"
                if complete.exists():
                    entry = json.loads(complete.read_text())
                    assert entry["identity"] == identity and entry["frame"] == j
                    assert all((folder / f"{i:03d}.pt").stat().st_size == layer["bytes"] for i, layer in enumerate(entry["layers"]))
                    for name, dtypes in entry["input_dtypes"].items():
                        dtype_seen[name].update(dtypes)
                    ledger.append(entry)
                    continue
                current.clear()
                seed(sample["seed"])
                result = policy.get_action(sample["obs"])
                assert all(__import__("numpy").isfinite(v).all() for v in result.values())
                assert set(current) == set(range(180))
                rows = []
                for i, name in enumerate(target_names):
                    expected = {-1} if "language_model" in name else set(range(8))
                    assert set(current[i]) == expected, (name, set(current[i]))
                    path = folder / f"{i:03d}.pt"
                    tensor_write(path, current[i])
                    rows.append({"name": name, "rows": {str(b): len(x) for b, x in current[i].items()}, "bytes": path.stat().st_size, "sha256": sha256(path)})
                entry = {"identity": identity, "frame": j, "task_id": sample["task_id"], "trajectory_step": sample["trajectory_step"], "seed": sample["seed"], "layers": rows, "input_dtypes": {n: sorted(v) for n, v in dtype_seen.items()}}
                json_write(complete, entry)
                ledger.append(entry)
                current.clear()
                print("ALL_TOKEN_CAPTURE", j + 1, "/", len(samples), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    json_write(args.output / "capture_ledger.json", {"identity": identity, "frames": ledger, "input_dtypes": {n: sorted(v) for n, v in dtype_seen.items()}, "all_teacher_weights_verified_against_loaded_checkpoint": True, "runtime": "Eager FP16 teacher; no graphs or quantized modules"})


def quantize(args, identity, manifest, config):
    from gr00t.quantization.perstep_backends import pack_int4

    ledger = json.loads((args.output / "capture_ledger.json").read_text())
    assert ledger["identity"] == identity and len(ledger["frames"]) == 100
    records, audit = {}, []
    for i, item in enumerate(manifest["layers"]):
        name = item["name"]
        path = args.output / "quantized" / f"{i:03d}.pt"
        if path.exists():
            entry = load(path)
            assert entry["identity"] == identity and entry["name"] == name
        else:
            transform = load(args.output / "transforms" / f"{i:03d}.pt")
            w = transform["rotated_weight"].cuda()
            k = w.shape[1]
            llm = "language_model" in name
            rows = {}
            if llm:
                h = torch.zeros((k, k), device="cuda", dtype=torch.float32)
                count = 0
                for j in range(100):
                    x = load(args.output / "capture" / f"{j:03d}/{i:03d}.pt")[-1].cuda()
                    h.addmm_(x.t(), x)
                    count += len(x)
                q, scale, solver = gptq_per_output_signed15(w, h / count, count)
                a_scale = torch.empty(0)
                rows = {"-1": count}
                extras = {"solver": solver}
                del h, x
            else:
                q, scale = rtn_per_output(w)
                per_step, clips = [], []
                for step in range(8):
                    values = torch.cat([load(args.output / "capture" / f"{j:03d}/{i:03d}.pt")[step] for j in range(100)]).cuda()
                    peak = torch.cat([torch.quantile(values[:, start:start + 128].abs().contiguous(), .999, dim=0) for start in range(0, k, 128)])
                    sa = positive_scale(peak)
                    clips.append({"step": step, "robust_peak_zero_channels": int((peak == 0).sum()), "guarded_channels_with_nonzero_values": int(((peak == 0) & (values.abs().amax(0) > 0)).sum()), "values_above_robust_peak": int((values.abs() > peak).sum()), "codes_clipped": int(((values / sa).round().abs() > 7).sum()), "total_values": values.numel()})
                    per_step.append(sa.cpu())
                    rows[str(step)] = len(values)
                    del values
                a_scale = torch.stack(per_step)
                extras = {"robust_estimator": "torch.quantile(abs(Xrot),q=.999,dim=tokens,interpolation=linear); code-based choice", "clipping": clips}
            record = {"K": w.shape[1], "N": w.shape[0], "B": pack_int4(q).cpu(), "s": scale.cpu(), "rot_in": transform["rot_in"], "perm": transform["perm"], "a_scale": a_scale, "bias": transform["bias"], "act_mode": "token" if llm else "step", "act_bits": 4}
            entry = {"identity": identity, "name": name, "record": record, "audit": {"name": name, "rows": rows, "qmin": int(q.min()), "qmax": int(q.max()), "minus8_count": int((q == -8).sum()), "scale_shape": list(scale.shape), "scale_dtype": str(scale.dtype), **extras}}
            tensor_write(path, entry)
            del w, q, scale
        records[name] = entry["record"]
        audit.append(entry["audit"])
        print("FROZEN_QUANTIZED", i + 1, "/180", name, flush=True)
    scope = [{"name": name, "K": r["K"], "N": r["N"], "act_mode": r["act_mode"]} for name, r in records.items()]
    meta = {"method": "holoq_submission_contract_v2", "runtime_plugin": "submission_contract_v2", "expected_layers": 180, "checkpoint": config["checkpoint"], "identity": identity, "configuration": config, "deployment_arithmetic": "packed W4 decoded on chip toFP16; numerical A4 reconstructed inFP16; existing FP16 MMA/FP32 accumulation, NOT nativeINT4xINT4", "reference": "Same packed codes/scales/R; corrected equation9 FP32 oracle", "quantized_scope": scope, "all_linear_layers": manifest["all_linear_layers"]}
    tensor_write(args.output / "frozen_packed.pt", {"meta": meta, "layers": records})
    json_write(args.output / "quantization_audit.json", {"identity": identity, "layers": audit, "pack_sha256": sha256(args.output / "frozen_packed.pt")})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--stage", choices=("all", "transform", "capture", "quantize", "export"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ("transforms", "capture", "quantized"):
        (args.output / folder).mkdir(exist_ok=True)
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    for key in list(os.environ):
        if key.startswith(("GR00T_GPTQ", "GR00T_DUQUANT", "GR00T_RTN", "GR00T_ATM", "GR00T_OHB", "HOLOQ_PACK", "HOLOQ_DEPLOY")):
            os.environ.pop(key)
    os.environ["HOLOQ_COMPUTE_DTYPE"] = "float16"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity, manifest, config = source_identity(args)
    for stage, function in (("transform", transforms), ("capture", capture), ("quantize", quantize)):
        if args.stage in ("all", stage):
            function(args, identity, manifest, *([config] if stage != "transform" else []))
            gc.collect()
            torch.cuda.empty_cache()
    if args.stage in ("all", "export"):
        assert args.deployment is not None, "--deployment required"
        if args.deployment.exists():
            saved = json.loads((args.deployment / "deployment.json").read_text())
            assert saved["source_pack_sha256"] == sha256(args.output / "frozen_packed.pt")
        else:
            export(args.output / "frozen_packed.pt", args.output / "fp16_model_state.pt", args.deployment)


if __name__ == "__main__":
    main()
