"""Paper Eqs1–8: input-only U D H rotation, signed15-level W/A, T8 scales.

Keep prior two-sided reconstruction packs/results untouched. Use the same
teacher corpus and official GPTQ reconstruction solver, with the paper's
literal symmetric [-7,7] weight grid instead of the upstream16-level grid.
"""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

from gr00t.quantization.perstep_backends import make_backend_record, pack_int4
from gr00t.quantization.duquant_preprocess import zigzag_permutation
from research.holoq_bench import kernels, quantizers
from research.holoq_bench.packed_runtime import apply_blocks
from research.holoq_bench.deployment import export as export_deployment


@torch.no_grad()
def paper_rotation(w, seed):
    n, k = w.shape
    perm = torch.from_numpy(zigzag_permutation(w.float().square().mean(0).cpu().numpy())).to(w.device)
    wp = w.float()[:, perm]
    blocks = wp.reshape(n, k // 64, 64).permute(1, 2, 0).contiguous()
    _, u = torch.linalg.eigh(blocks @ blocks.transpose(1, 2))
    u = u.flip(-1)
    largest = u.abs().argmax(-2, keepdim=True)
    u = u * u.gather(-2, largest).sign()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (k // 64, 64), generator=generator) * 2 - 1).float().to(w.device)
    h = torch.ones(1, 1, device=w.device)
    while h.shape[0] < 64:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    rin = ((u * signs[:, None, :]) @ (h / 8)).half()
    wr = torch.bmm(wp.reshape(n, k // 64, 64).transpose(0, 1), rin.float()).transpose(0, 1).reshape(n, k)
    # Crucially, no output-side left multiplication or restore exists in Eq5.
    return wr, rin, perm


@torch.no_grad()
def gptq_signed15(weight, x):
    sys.path.insert(0, str(quantizers.ROOT / "third_party/gptq"))
    module = quantizers.load_file("holoq_paper_official_gptq", quantizers.ROOT / "third_party/gptq/gptq.py")
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="cuda", dtype=torch.float16)
    linear.weight.copy_(weight)
    solver = module.GPTQ(linear)
    solver.quantizer = module.Quantizer()
    solver.quantizer.configure(4, perchannel=True, sym=True, mse=False)

    def find(values, weight=False):
        assert weight and values.ndim == 2
        # Frozen FP16 scales are the actual mature packed backend contract.
        scale = (values.float().abs().amax(1, keepdim=True) / 7).clamp_min(2 ** -24).half().float()
        solver.quantizer.maxq = torch.tensor(14, device=values.device)
        solver.quantizer.scale = scale
        solver.quantizer.zero = torch.full_like(scale, 7)

    solver.quantizer.find_params = find
    solver.add_batch(x.unsqueeze(0), None)
    solver.fasterquant(blocksize=128, percdamp=.01, groupsize=-1, actorder=False)
    scale = solver.quantizer.scale.detach().clone()
    q = (linear.weight.float() / scale).round().clamp(-7, 7).to(torch.int8)
    assert torch.equal((q.float() * scale).half(), linear.weight), "GPTQ15 export changes the frozen grid"
    solver.free()
    return q, scale


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    a = p.parse_args()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = a.root
    out = root / "paper_cfg_build"
    cache = out / "layer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "teacher_capture/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    weights = torch.load(root / "teacher_capture/weights.pt", map_location="cpu", mmap=True, weights_only=False)
    config = {"paper": "arXiv2605.28803v3 Eqs1–8/Table5", "input_only": True, "output_rotation": False,
              "block": 64, "randomized_hadamard": "U @ D @ H; signs seed42+layer_index",
              "weight_codes": "[-7,7]", "activation_codes": "[-7,7]", "weight_scales": "FP16 backend contract",
              "LLM": "official GPTQ reconstruction, block128,damp.01,per-output signed15 quantizer; per-tokenA4",
              "DiT": "RTN per-outputW4; per-step/channel q999/7 A4;8 steps",
              "teacher_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "comparison_budget": "same10 teacher trajectories /100 frames /captured tokens as all current baselines"}
    (out / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    for index, item in enumerate(manifest["layers"]):
        path = cache / f"{index:03d}.pt"
        if path.exists():
            entry = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            assert entry["name"] == item["name"] and entry["paper_configuration"] == config
            continue
        name = item["name"]
        w = weights[name]["weight"].cuda()
        wr, rin, perm = paper_rotation(w, 42 + index)
        data = torch.load(root / "teacher_capture" / item["file"], map_location="cpu", mmap=True, weights_only=False)
        x = {step: apply_blocks(value.cuda(), rin, perm) for step, value in data.items()}
        llm = "language_model" in name
        if llm:
            q, scales = gptq_signed15(wr, torch.cat(list(x.values())))
        else:
            scales = (wr.float().abs().amax(1, keepdim=True) / 7).clamp_min(2 ** -24).half().float()
            q = (wr / scales).round().clamp(-7, 7).to(torch.int8)
        record = dict(kernels.pack_marlin(q.cpu(), scales.cpu()), K=w.shape[1], N=w.shape[0],
                      bias=weights[name]["bias"], act_bits=4, act_qmin=-7,
                      act_mode="token" if llm else "step", rot_in=rin.cpu(), rot_out=None, perm=perm.cpu())
        integer = None
        if not llm:
            record["a_scale"] = torch.stack([torch.quantile(x[t].float().abs(), .999, dim=0) / 7 for t in range(8)]).clamp_min(1e-8).cpu()
            wq = (q.float() * scales.float()).half()
            integer = make_backend_record(wq, record["a_scale"].cuda(), "step_weight", 128)
            integer.update(act_qmin=-7, weight_source="paper_input_only_frozen_quantized_grid")
        entry = {"name": name, "record": record, "canonical": {"q": q.cpu(), "scales": scales.half().cpu(), "group": w.shape[1]},
                 "integer": integer, "paper_configuration": config}
        temp = path.with_suffix(".tmp")
        torch.save(entry, temp)
        temp.replace(path)
        print("PAPER_LAYER", index + 1, name, flush=True)
        del w, wr, rin, perm, x, q, scales, entry
        torch.cuda.empty_cache()
    entries = [torch.load(f, map_location="cpu", mmap=True, weights_only=False) for f in sorted(cache.glob("*.pt"))]
    assert len(entries) == 180
    packs = root / "packs_paper_v1"
    packs.mkdir(parents=True, exist_ok=True)
    for converted in (False, True):
        for native_llm in (False, True):
            label = "holoq_paper" + ("_step_weight" if converted else "") + ("_llm_s4" if native_llm else "")
            records = {}
            for entry in entries:
                name, record, canonical = entry["name"], dict(entry["record"]), entry["canonical"]
                if "language_model" in name and native_llm:
                    record = {k: v for k, v in record.items() if k not in ("B", "s", "z", "s2", "z2")}
                    record.update(backend="cutlass_s4", B=pack_int4(canonical["q"]), s=canonical["scales"].half())
                elif "language_model" not in name and converted:
                    record = {k: v for k, v in record.items() if k not in ("B", "s", "z", "s2", "z2", "group")}
                    record.update(backend="native_s4", integer=entry["integer"])
                records[name] = record
            obj = {"format": "holoq_bench_packed_v1", "layers": records,
                   "meta": {"method": label, "expected_layers": 180, "checkpoint": manifest["checkpoint"],
                            "calibration": manifest["calibration"], "compute_dtype": "float16",
                            "paper_configuration": config, "native_conversion": converted,
                            "runtime_plugin": "cutlass_s4" if native_llm else None,
                            "rotation_plugin": "triton_block64_strided_v2"}}
            target = packs / (label + ".pt")
            if not target.exists():
                torch.save(obj, target)
            deploy = root / "deployments" / label
            if not deploy.exists():
                export_deployment(target, root / "teacher_capture/fp16_model_state.pt", deploy)
            print("PAPER_DEPLOYMENT", label, flush=True)
    (out / "finished.json").write_text(json.dumps({"built": True, "config": config}, indent=2) + "\n")


if __name__ == "__main__":
    main()
