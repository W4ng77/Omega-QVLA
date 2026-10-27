"""Resumable shared-corpus quantization and explicit packed exports."""

import argparse
import json
from pathlib import Path
import time

import torch

from gr00t.quantization.perstep_backends import make_backend_record
from research.holoq_bench import kernels, quantizers
from research.holoq_bench.packed_runtime import apply_blocks

ROOT = Path(__file__).resolve().parents[2]


def recipe(method):
    return {
        "gptq": {"backend": "marlin", "bits": 16, "mode": "none", "group": 128},
        "awq": {"backend": "awq", "bits": 16, "mode": "none", "group": 128},
        "omniquant": {"backend": "awq", "bits": 16, "mode": "none", "group": 128},
        "smoothquant": {"backend": "qserve", "bits": 8, "mode": "token", "group": -1},
        "duquant": {"backend": "marlin", "bits": 8, "mode": "channel", "group": -1},
        "quantvla": {"backend": "marlin", "bits": 8, "mode": "channel", "group": -1},
        "holoq": {"backend": "marlin", "bits": 4, "mode": "step", "group": -1},
    }[method]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--methods", nargs="+", default=["gptq", "awq", "omniquant", "smoothquant", "duquant", "quantvla", "holoq"]
    )
    p.add_argument("--max-layers", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    weights = torch.load(args.capture / "weights.pt", map_location="cpu", mmap=True, weights_only=False)
    layers = manifest["layers"][: args.max_layers or None]
    for method in args.methods:
        cache = args.output / "layer_cache" / method
        cache.mkdir(parents=True, exist_ok=True)
        progress = {"method": method, "started": time.time(), "layers": []}
        for index, item in enumerate(layers):
            output = cache / f"{index:03d}.pt"
            if output.exists():
                print("REUSE", method, index, item["name"], flush=True)
                continue
            name = item["name"]
            # Paper's strong OmniQuant configuration excludes DiT attention.
            if method == "omniquant" and ".attn1." in name:
                continue
            w = weights[name]["weight"].cuda()
            n, k = w.shape
            data = torch.load(args.capture / item["file"], map_location="cpu", mmap=True, weights_only=False)
            config = recipe(method)
            record = {
                "K": k,
                "N": n,
                "bias": weights[name]["bias"],
                "act_bits": config["bits"],
                "act_mode": config["mode"],
                "act_qmin": -7 if config["bits"] == 4 else -127,
            }
            inputs = {bank: x.cuda() for bank, x in data.items()}
            if method in ("holoq", "duquant", "quantvla"):
                w, rin, rout, perm = quantizers.rotations(
                    w, composite=method == "holoq", permute=method != "quantvla"
                )
                record.update(rot_in=rin.cpu(), rot_out=rout.cpu(), perm=perm.cpu())
                inputs = {bank: apply_blocks(x, rin, perm) for bank, x in inputs.items()}
            xall = torch.cat(list(inputs.values()))
            generator = torch.Generator(device="cuda").manual_seed(42 + index)
            sampled = xall[torch.randperm(len(xall), generator=generator, device="cuda")[:512]]
            smooth = None
            if method == "gptq" or (method == "holoq" and "language_model" in name):
                q, scales, details = quantizers.gptq(w, xall, group=-1 if method == "holoq" else 128)
                packed = kernels.pack_marlin(q.cpu(), scales.cpu())
            elif method == "awq":
                q, scales, zero, smooth, details = quantizers.awq(w, sampled, name)
                packed = kernels.pack_awq(q.cpu(), scales.cpu(), zero.cpu())
            elif method == "omniquant":
                q, scales, zero, smooth, details = quantizers.omniquant(w, sampled)
                packed = kernels.pack_awq(q.cpu(), scales.cpu(), zero.cpu())
            elif method == "smoothquant":
                act_peak = xall.float().abs().amax(0).clamp(1e-5, 1000)
                weight_peak = w.float().abs().amax(0).clamp_min(1e-5)
                smooth = (act_peak / weight_peak).sqrt().clamp_min(1e-5).half()
                transformed = (w.float() * smooth.float()).half()
                q, scales = quantizers.rtn(transformed, scale_half=True)
                packed = kernels.pack_qserve(q.cpu(), scales.cpu())
                details = {"alpha": 0.5, "clipping_stat_threshold": 1000, "act_mode": "per_token"}
            else:
                q, scales = quantizers.rtn(w, scale_half=True, mse=method != "holoq")
                packed = kernels.pack_marlin(q.cpu(), scales.cpu())
                details = {
                    "rotation": "svd_hadamard" if method == "holoq" else "svd",
                    "block": 64,
                    "solver": "rtn",
                    "scale_dtype": "float16",
                }
            record.update(packed)
            if smooth is not None:
                record["smooth"] = smooth.cpu()
            if method == "holoq":
                if "language_model" in name:
                    record["act_mode"] = "token"
                else:
                    record["a_scale"] = (
                        torch.stack([torch.quantile(inputs[t].float().abs(), 0.999, dim=0) / 7 for t in range(8)])
                        .clamp_min(1e-8)
                        .cpu()
                    )
            elif method in ("duquant", "quantvla"):
                record["a_scale"] = (torch.quantile(xall.float().abs(), 0.999, dim=0) / 127).clamp_min(1e-8).cpu()
            # Retain canonical codes only in the offline layer cache. Inference
            # exports below contain packed weights, never this INT8 matrix.
            canonical = {"q": q.cpu(), "scales": scales.cpu(), "group": k // scales.shape[1]}
            torch.save({"name": name, "record": record, "details": details, "canonical": canonical}, output)
            progress["layers"].append({"name": name, "seconds_from_start": time.time() - progress["started"]})
            (cache / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print("BUILT", method, index + 1, "/", len(layers), name, flush=True)
            del inputs, xall, sampled, w, q, scales
            torch.cuda.empty_cache()
        if not args.max_layers:
            export(method, args.output, manifest, cache)


def export(method, root, capture_manifest, cache):
    entries = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(cache.glob("*.pt"))]
    full = {e["name"]: e["record"] for e in entries}
    variants = [(method, full)]
    if method in ("gptq", "awq", "omniquant", "quantvla"):
        selective = {n: r for n, r in full.items() if ".attn1." not in n}
        variants = [(method + "_selective", selective)]
        if len(full) == 180 and method in ("gptq", "awq"):
            variants.append((method + "_full", full))
    if method in ("smoothquant", "duquant", "quantvla"):
        key = variants[0][0]
        variants[0] = (key + "_a8", variants[0][1])
        a4 = {}
        for name, rec in variants[0][1].items():
            rr = dict(rec, act_bits=4, act_qmin=-7)
            if "a_scale" in rr:
                rr["a_scale"] = rr["a_scale"] * (127 / 7)
            a4[name] = rr
        variants.append((key + "_a4", a4))
    if method == "holoq":
        converted = {}
        for entry in entries:
            rr = dict(entry["record"])
            if ".transformer_blocks." in entry["name"]:
                c = entry["canonical"]
                w = (c["q"].float() * c["scales"].repeat_interleave(c["group"], 1)).half()
                integer = make_backend_record(w.cuda(), rr["a_scale"].cuda(), "step_weight", 128)
                integer["act_qmin"] = -7
                rr = {k: v for k, v in rr.items() if k not in ("B", "s", "group")}
                rr.update(backend="native_s4", integer=integer)
            converted[entry["name"]] = rr
        variants.append(("holoq_step_weight", converted))
    for label, records in variants:
        count = len(records)
        if count not in (116, 180):
            raise ValueError((label, count))
        obj = {
            "format": "holoq_bench_packed_v1",
            "meta": {
                "method": label,
                "expected_layers": count,
                "calibration": capture_manifest["calibration"],
                "compute_dtype": "float16",
                "checkpoint": capture_manifest["checkpoint"],
                "quantizer_adapter": "official linear reconstruction ports where model-specific full-block adapters are unavailable",
                "native_conversion": label == "holoq_step_weight",
            },
            "layers": records,
        }
        path = root / (label + ".pt")
        torch.save(obj, path)
        print("EXPORTED", label, "layers", count, "bytes", path.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
