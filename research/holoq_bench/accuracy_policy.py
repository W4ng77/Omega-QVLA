"""Shared-protocol accuracy controls; dense mathematics NEVER a speed baseline.

Separate source-pack construction from execution arithmetic: published legacy
wrapper, frozen-grid mathematical oracle, and full-precision task reference.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.run_policy import stats, memory, write
from research.holoq_bench.rollout_resume import validate_resume_metadata
from research.holoq_bench.rollout_driver import run as rollout


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--kind", choices=("published_legacy", "released_grid_oracle", "paper_grid_oracle", "fp16_reference"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observations", type=Path, default=Path("results/holoq_object/heldout_init10_19/observations.pt"))
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--warmup", type=int, default=32)
    a = p.parse_args()
    a.mode, a.stage = "head_graph", "rollout"
    lock = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed(42)
    root = Path("results/holoq_object")
    manifest = json.loads((root / "teacher_capture/manifest.json").read_text())
    for key in list(os.environ):
        if key.startswith(("GR00T_GPTQ", "GR00T_DUQUANT", "GR00T_RTN", "GR00T_ATM", "GR00T_OHB", "GR00T_DIT_CORRECTION", "HOLOQ_PACK", "HOLOQ_DEPLOY")):
            os.environ.pop(key)
    os.environ["HOLOQ_COMPUTE_DTYPE"] = "float16"
    released = a.kind.startswith(("published", "released"))
    if released:
        provenance = json.loads((root / "released_grid_build/provenance.json").read_text())
        source_sha = provenance["source_pack_sha256"]
    elif a.kind == "paper_grid_oracle":
        source_sha = json.loads((root / "deployments/holoq_paper_g128/deployment.json").read_text())["source_pack_sha256"]
    else:
        source_sha = "checkpoint:" + Path(manifest["checkpoint"]).name
    model_path = manifest["checkpoint"]
    if a.kind == "published_legacy":
        os.environ.update(GR00T_GPTQ="1", GR00T_GPTQ_PATH=provenance["factored_pack"], GR00T_GPTQ_ABITS="4", GR00T_GPTQ_SCOPE="", GR00T_GPTQ_MISSING="error")
    elif a.kind.endswith("grid_oracle"):
        case = "holoq_release_grid" if released else "holoq_paper_g128"
        deployment = root / "deployments" / case
        os.environ["HOLOQ_DEPLOY"] = str(deployment.resolve())
        model_path = str(deployment.resolve())
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from research.holoq_bench.common_logits import install as install_logits
    from research.holoq_bench.graphs import install as install_graphs

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=model_path, modality_config=config.modality_config(), modality_transform=config.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    if a.kind == "published_legacy":
        from gr00t.quantization.gptq_layers import GptqLinear

        frozen = torch.load(root / "released_grid_build/frozen_llm_scales.pt", map_location="cpu", weights_only=False)
        modules = {name: m for name, m in policy.model.named_modules() if isinstance(m, GptqLinear)}
        assert len(modules) == 180
        for name, scale in frozen.items():
            modules[name]._act_scale = scale.cuda()
            modules[name]._act_scale_initialized = True
    elif a.kind.endswith("grid_oracle"):
        from research.holoq_bench.check_canonical_math import DenseCanonicalOracle
        from research.holoq_bench.packed_runtime import PackedLinear

        modules = {name: m for name, m in policy.model.named_modules() if isinstance(m, PackedLinear)}
        cache = root / ("released_grid_build/layer_cache" if released else "paper_cfg_build/layer_cache")
        parents = dict(policy.model.named_modules())
        for path in sorted(cache.glob("*.pt")):
            entry = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            grid = entry["canonical"]
            weight = (grid["q"].float() * grid["scales"].float().repeat_interleave(grid["group"], 1)).half().cuda()
            oracle = DenseCanonicalOracle(modules[entry["name"]], weight)
            parent, attr = entry["name"].rsplit(".", 1)
            setattr(parents[parent], attr, oracle)
        assert len(modules) == 180
    else:
        modules = {}
    install_logits(policy)
    graphs = install_graphs(policy)
    source_root = Path(__file__).resolve().parents[2]
    paths = [*source_root.glob("research/holoq_bench/*.py"), *source_root.glob("research/native_int4/*"), source_root / "gr00t/model/policy.py", source_root / "gr00t/model/action_head/flow_matching_action_head.py", source_root / "gr00t/quantization/perstep_backends.py", source_root / "gr00t/quantization/gptq_layers.py", source_root / "gr00t/quantization/duquant_preprocess.py"]
    report = {"method": "accuracy_" + a.kind, "mode": a.mode, "stage": a.stage, "command": subprocess.list2cmdline(__import__("sys").argv),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "source_pack_sha256": source_sha,
              "observation_file_sha256": hashlib.sha256(a.observations.read_bytes()).hexdigest(),
              "source_files_sha256": {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file() and path.suffix in (".py", ".cpp", ".cu")},
              "denoising_steps": 8, "action_horizon": 16, "batch": 1, "warmup": a.warmup,
              "noise_seed": "42 + task_id*10000 + initial_state_index*1000 + control_step", "init_indices": list(range(10, 10 + a.trials)),
              "common_optimizations": ["qwen3_last_token_logits"], "checkpoint": manifest["checkpoint"], "started": time.time(),
              "scope": "ACCURACY CONTROL ONLY: not a quantized deployment speed/storage/memory baseline. Uses the EXACT shared Object rollout driver.",
              "runtime": a.kind, "wrapped_target_layers": len(modules), "quantized_layers": 0 if a.kind == "fp16_reference" else 180,
              "calibration_overlap_note": "Historical calibration state provenance unknown" if released else "New shared teacher0 and heldout10..19 disjoint",
              "raw_image_shape_each_camera": [256, 256, 3], "transform_image_shape_each_camera": [224, 224, 3]}
    previous_memory, windows = validate_resume_metadata(a.output, report)
    report["resume_measurement_windows"] = windows
    samples = torch.load(a.observations, map_location="cpu", weights_only=False)
    for i in range(a.warmup):
        seed(2000 + i)
        policy.get_action(samples[(i % 10) * 10]["obs"])
    torch.cuda.synchronize()
    write(a.output / "metadata.json", report)

    def timed(obs, noise):
        seed(noise)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = policy.get_action(obs)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        assert all(np.isfinite(value).all() for value in result.values())
        return result, ms

    rollout(a, report, graphs, timed, previous_memory, stats, memory, write)


if __name__ == "__main__":
    main()
