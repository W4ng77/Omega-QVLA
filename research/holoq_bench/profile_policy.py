"""Whole get_action profiling of an explicit packed deployment, never a speed baseline.

Warmup/compilation precede capture. Actual uninstrumented timing is separate.
Stage annotations preserve the deployed head-only CUDA Graph implementation.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from research.holoq_bench.collect_calibration import seed
from research.holoq_bench.run_policy import stats


@contextmanager
def scope(name):
    with record_function(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


def wrap(obj, attr, label):
    original = getattr(obj, attr)

    def annotated(*args, **kwargs):
        name = label() if callable(label) else label
        with scope(name):
            return original(*args, **kwargs)

    setattr(obj, attr, annotated)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["eager", "head_graph"], default="head_graph")
    p.add_argument("--engine", choices=["torch", "nsys"], default="torch")
    p.add_argument("--calls", type=int, default=5)
    args = p.parse_args()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(HOLOQ_DEPLOY=str(args.deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed(42)
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from research.holoq_bench.packed_runtime import PackedLinear

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=str(args.deployment.resolve()), modality_config=config.modality_config(),
                         modality_transform=config.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    observations = torch.load(args.observations, map_location="cpu", weights_only=False)
    warm_samples = [observations[i * 10] for i in range(10)]
    sample = warm_samples[0]
    graphs = None
    if args.mode == "head_graph":
        from research.holoq_bench.graphs import install

        graphs = install(policy)
    for i in range(32):
        seed(2000 + i)
        policy.get_action(warm_samples[i % 10]["obs"])
    torch.cuda.synchronize()
    # No profiler, NVTX wrappers or recording hooks during formal timing.
    timings = []
    for i in range(20):
        seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = policy.get_action(sample["obs"])
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
        assert all(np.isfinite(v).all() for v in result.values())

    wrap(policy, "apply_transforms", "holoq/stage/preprocess")
    wrap(policy.model, "prepare_input", "holoq/stage/prepare_input")
    wrap(policy.model.backbone, "forward", "holoq/stage/backbone_other")
    wrap(policy.model.backbone.eagle_model.vision_model, "forward", "holoq/stage/vision")
    wrap(policy.model.backbone.eagle_model.language_model, "forward", "holoq/stage/language")
    wrap(policy.model.action_head, "get_action", "holoq/stage/action_head")
    wrap(policy, "_get_unnormalized_action", "holoq/stage/postprocess")
    if args.mode == "eager":
        from gr00t.quantization.dit_step_context import get_current_dit_step

        wrap(policy.model.action_head.model, "forward", lambda: f"holoq/step/{get_current_dit_step()}")
        for name, module in policy.model.named_modules():
            if isinstance(module, PackedLinear):
                wrap(module, "forward", "holoq/layer/" + name)
    manifest = json.loads((args.deployment / "deployment.json").read_text())
    metadata = {
        "command": sys.argv, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "deployment": str(args.deployment), "source_pack_sha256": manifest["source_pack_sha256"],
        "profiler_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "engine": args.engine, "mode": args.mode, "profiled_calls": args.calls,
        "warmup": 32, "denoising_steps": 8, "action_horizon": 16,
        "sample": {k: sample[k] for k in ("task_id", "init_index")},
        "unprofiled_fixed_input_latency": stats(timings),
        "unprofiled_boundary": "synchronize; perf_counter; complete get_action; synchronize; perf_counter; RNG reset outside timer",
        "profile_note": "Profiler timings diagnose work/overlap, not deployment latency. Fixed observation and identical per-call noise seed. RNG reset is outside each annotated policy range.",
        "graph_count_before": len(graphs.cache) if graphs else 0,
        "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used", "--format=csv"], text=True),
        "packed_layers": {name: {"backend": m.backend, "shapes": sorted(m.input_shapes)}
                          for name, m in policy.model.named_modules() if isinstance(m, PackedLinear)},
    }

    def calls():
        for i in range(args.calls):
            with scope("holoq/outside_rng_reset"):
                seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
            with scope(f"holoq/get_action/{i}"):
                policy.get_action(sample["obs"])

    torch.cuda.synchronize()
    if args.engine == "torch":
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            calls()
            with scope("holoq/final_sync"):
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / "policy_trace.json"))
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
        (args.output / "top_self_cuda.txt").write_text(table + "\n")
        (args.output / "top_shapes.txt").write_text(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=60) + "\n")
        print(table, flush=True)
    else:
        torch.cuda.cudart().cudaProfilerStart()
        with scope("holoq/profile_window"):
            calls()
            with scope("holoq/final_sync"):
                torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    metadata["graph_count_after"] = len(graphs.cache) if graphs else 0
    assert metadata["graph_count_after"] == metadata["graph_count_before"], "Profile includes cold capture"
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("WHOLE_POLICY_PROFILE_COMPLETE", args.engine, args.mode, "unprofiled", metadata["unprofiled_fixed_input_latency"], flush=True)


if __name__ == "__main__":
    main()
