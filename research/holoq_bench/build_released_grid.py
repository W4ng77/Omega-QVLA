"""Freeze published scales and losslessly encode actual released W4 values.

Use the original checkpoint and historical warmup behavior. Keep this distinct
from the paper-input-only rebuild. No new fastest-kernel search is performed.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch
from gr00t.quantization.perstep_backends import make_backend_record, pack_int4
from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.deployment import export, sha256
from research.holoq_bench.released_grid import recover_exact_grid


def main():
    root = Path("results/holoq_object")
    out = root / "released_grid_build"
    out.mkdir(parents=True, exist_ok=True)
    lock = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    original = Path("/home/xinyu/models/omega-qvla-gr00t-w4a4/gr00t_object/quantized.pt")
    factored = Path("/home/xinyu/omegaqvla-perstep-hp/results/native_int4_20261027/object_block_rotations.pt")
    manifest = json.loads((root / "teacher_capture/manifest.json").read_text())
    os.environ.pop("HOLOQ_DEPLOY", None)
    os.environ.pop("HOLOQ_PACK", None)
    os.environ.update(GR00T_GPTQ="1", GR00T_GPTQ_PATH=str(factored), GR00T_GPTQ_ABITS="4",
                      GR00T_GPTQ_SCOPE="", GR00T_GPTQ_MISSING="error", GR00T_GPTQ_CALIB_STEPS="32",
                      GR00T_CALIB_SUITE="libero_object", LIBERO_ROOT="/home/xinyu/LIBERO", HOLOQ_COMPUTE_DTYPE="float16")
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from gr00t.quantization.gptq_layers import GptqLinear

    seed(42)
    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=manifest["checkpoint"], modality_config=config.modality_config(), modality_transform=config.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    policy.warmup(num_steps=32)
    modules = {name: module for name, module in policy.model.named_modules() if isinstance(module, GptqLinear)}
    assert len(modules) == 180
    observed = {name: module._act_scale.detach().cpu() for name, module in modules.items() if "language_model" in name}
    frozen_path = out / "frozen_llm_scales.pt"
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=False) if frozen_path.exists() else observed
    assert len(frozen) == 84
    if not frozen_path.exists():
        torch.save(frozen, frozen_path)
    freeze_audit = {"frozen_scales_sha256": sha256(frozen_path), "warmup": 32,
                    "observed_calibration_batches": {name: module.calibrator._seen for name, module in modules.items() if name in frozen},
                    "replay_scale_nmse": {name: float((observed[name].float() - scale.float()).square().sum() / scale.float().square().sum().clamp_min(1e-30)) for name, scale in frozen.items()}}
    (out / "scale_freeze_audit.json").write_text(json.dumps(freeze_audit, indent=2) + "\n")
    source = torch.load(original, map_location="cpu", mmap=True, weights_only=False)
    factored_pack = torch.load(factored, map_location="cpu", mmap=True, weights_only=False)
    layers, converted, report = {}, {}, []
    cache = out / "layer_cache"
    cache.mkdir(exist_ok=True)
    for index, (name, module) in enumerate(modules.items()):
        old = source[name]
        stored = old.get("baseline_q", old.get("weight_res_q"))
        assert stored is not None and torch.equal(stored, factored_pack[name].get("baseline_q", factored_pack[name].get("weight_res_q")))
        w = module._weight_q.detach()
        assert torch.equal(w.cpu(), stored.half()), "Historical runtime weight differs from source"
        assert not module._lowrank_A.numel() and not module._lowrank_B.numel()
        q, sw = recover_exact_grid(w)
        record = {"K": module.in_features, "N": module.out_features, "backend": "tile_fp16",
                  "B": pack_int4(q).cpu(), "s": sw.cpu(), "group": 128, "act_bits": 4, "act_qmin": -8,
                  "act_mode": "step" if module._has_act_scale_table else "channel",
                  "a_scale": module._act_scale_table.detach().cpu() if module._has_act_scale_table else frozen[name].float(),
                  "rot_in": module._duquant_rotation_blocks.half().cpu(),
                  "rot_out": module._duquant_rotation_out_blocks.half().cpu() if module._has_duquant_rotation_out_blocks else None,
                  "perm": module._duquant_rotation_perm.cpu(), "bias": module.bias.detach().cpu() if module.bias is not None else None}
        if module._has_act_scale_table:
            assert torch.equal(record["a_scale"], old["act_scale_table"].float()), "Do not silently alter released activation rows"
        assert bool((record["a_scale"].half() > 0).all()), "Historical activation scale underflows FP16"
        layers[name] = record
        variation = dict(record)
        if "language_model" not in name:
            integer = make_backend_record(w, record["a_scale"].cuda(), "step_weight", 128)
            integer.update(act_qmin=-8, weight_source="released_FP16_weight_values_exact_no_rebuild")
            variation = {key: value for key, value in record.items() if key not in ("B", "s", "group")}
            variation.update(backend="native_s4", integer=integer)
        converted[name] = variation
        torch.save({"name": name, "canonical": {"q": q.cpu(), "scales": sw.cpu(), "group": 128}, "record": record}, cache / f"{index:03d}.pt")
        report.append({"name": name, "weight_values_exact": True, "source_weight_key": "baseline_q" if "baseline_q" in old else "weight_res_q",
                       "source_weight_bits": old["weight_bits"], "source_damp": old.get("gptq_damp_percent"),
                       "legacy_observed_calibration_batches": module.calibrator._seen,
                       "activation_mode": record["act_mode"], "source_a_table_preserved": bool(module._has_act_scale_table),
                       "source_llm_scale_missing": "language_model" in name})
        print("RELEASED_EXACT_GRID_LAYER", index + 1, name, flush=True)
    provenance = {"source_pack": str(original), "source_pack_sha256": sha256(original), "factored_pack": str(factored),
                  "factored_pack_sha256": sha256(factored), "checkpoint": manifest["checkpoint"], "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "historical_meta": source.get("__meta__"),
                  "LLM_scale_freeze": "Original runtime warmup32, black images/zero state, first Object task prompt; observed first-call freeze, not32-batch calibration. Seed42. Frozen, no evaluation adaptation.",
                  "original_calibration_overlap": "Historical artifact does not identify teacher state hashes; cannot verify its original calibration/evaluation disjointness",
                  "weight_encoding": "Signed W4 codes + FP32 G128 coefficients reconstruct every released FP16 weight exactly. Coefficients recover an equivalent encoding, since original codes/scales were not published; no weight values requantized.",
                  "activation_arithmetic": "Legacy FP16 division, nearest rounding,[-8,7] clamp, FP16 reconstruction; original DiT F32 tables stored without changes",
                  "deployment_compute": "Packed W4 read/dequantized inside existing tile_fp16 FP16 MMA; no complete floating weight cache. Native S4 only for separate DiT step_weight conversion.",
                  "layers": report}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for case, records in (("holoq_release_grid", layers), ("holoq_release_step_weight", converted)):
        pack = {"format": "holoq_bench_packed_v1", "layers": records, "meta": {"method": case, "expected_layers": 180, "checkpoint": manifest["checkpoint"], "calibration": provenance["LLM_scale_freeze"], "compute_dtype": "float16", "runtime_plugin": "released_exact_grid", "source_released_pack_sha256": provenance["source_pack_sha256"], "released_grid_provenance": provenance, "native_conversion": case.endswith("step_weight")}}
        target = out / (case + ".pt")
        assert not target.exists(), "Preserve prior exports"
        torch.save(pack, target)
        export(target, root / "teacher_capture/fp16_model_state.pt", root / "deployments" / case)
    (out / "finished.json").write_text(json.dumps({"built": True, "all180_weight_values_exact": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
