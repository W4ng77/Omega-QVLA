"""Full-policy packed-path audit and common graph equivalence check, not timing."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.holoq_bench.collect_calibration import seed


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--pack", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ.update(HOLOQ_PACK=args.pack, HOLOQ_COMPUTE_DTYPE="float16")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed(42)
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy
    from research.holoq_bench.packed_runtime import PackedLinear
    from research.holoq_bench.graphs import install

    cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    pack = torch.load(args.pack, map_location="cpu", mmap=True, weights_only=False)
    policy = Gr00tPolicy(
        model_path=pack["meta"]["checkpoint"],
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )
    layers = {name: module for name, module in policy.model.named_modules() if isinstance(module, PackedLinear)}
    assert set(layers) == set(pack["layers"])
    samples = torch.load(args.observations, map_location="cpu", weights_only=False)
    samples = [samples[i * 10]["obs"] for i in range(10)]
    refs = []
    for i, obs in enumerate(samples):
        seed(1000 + i)
        action = policy.get_action(obs)
        assert all(np.isfinite(v).all() for v in action.values())
        refs.append(action)
    assert all(module.calls for module in layers.values())
    report = {
        "method": pack["meta"]["method"],
        "active_layers": len(layers),
        "backends": dict(Counter(module.backend for module in layers.values())),
        "per_layer_calls": {name: module.calls for name, module in layers.items()},
        "graph_status": "testing",
        "graph_checks": [],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    original = policy.model.action_head.get_action
    graphs = install(policy)
    try:
        # Exercise distinct shapes and arbitrary replay ordering in a shared
        # pool. No captured output is retained across subsequent policy calls.
        for i in list(range(10)) + list(reversed(range(10))):
            seed(1000 + i)
            out = policy.get_action(samples[i])
            num = sum(float(np.square(out[k].astype(float) - refs[i][k].astype(float)).sum()) for k in out)
            den = sum(float(np.square(refs[i][k].astype(float)).sum()) for k in out)
            nmse = num / max(den, 1e-30)
            exact = all(np.array_equal(out[k], refs[i][k]) for k in out)
            report["graph_checks"].append({"sample": i, "nmse": nmse, "exact": exact})
            if not all(np.isfinite(v).all() for v in out.values()) or nmse > 1e-5:
                raise RuntimeError(f"Graph action mismatch sample={i} nmse={nmse}")
        report.update(
            graph_status="supported_verified", graph_count=len(graphs.cache), graph_replays=graphs.replays
        )
    except Exception as exc:
        report.update(graph_status="unsupported", graph_error=f"{type(exc).__name__}: {exc}")
        policy.model.action_head.get_action = original
    report["allocated_bytes"] = torch.cuda.memory_allocated()
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k not in ("per_layer_calls", "graph_checks")}, flush=True)


if __name__ == "__main__":
    main()
