"""Accuracy-only audit of actual eager/Graph initial noise; no speed baseline."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from research.holoq_bench.collect_calibration import seed


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    os.environ.update(HOLOQ_DEPLOY=str(a.deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from research.holoq_bench.graphs import install

    c = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=str(a.deployment.resolve()), modality_config=c.modality_config(), modality_transform=c.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    head = policy.model.action_head
    target_shape = (1, head.config.action_horizon, head.config.action_dim)
    samples = torch.load(a.observations, map_location="cpu", weights_only=False)[::10]
    original_randn = torch.randn
    draws, context_dtypes = [], []
    process = head.process_backbone_output

    def record_context(*args, **kwargs):
        output = process(*args, **kwargs)
        context_dtypes.append(str(output["backbone_features"].dtype))
        return output

    def record_draw(*args, **kwargs):
        output = original_randn(*args, **kwargs)
        if tuple(output.shape) == target_shape and not torch.cuda.is_current_stream_capturing():
            draws.append(output.detach().cpu().clone())
        return output

    head.process_backbone_output = record_context
    torch.randn = record_draw
    eager = []
    try:
        with torch.inference_mode():
            for sample in samples:
                seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
                eager.append(policy.get_action(sample["obs"]))
            eager_noise = list(draws)
            eager_dtypes = list(context_dtypes)
            assert len(eager_noise) == 10
            draws.clear()
            context_dtypes.clear()
            graphs = install(policy)
            rows = []
            for i, sample in enumerate(samples):
                seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
                output = policy.get_action(sample["obs"])
                actual, reference = draws[-1].float(), eager_noise[i].float()
                noise_nmse = float((actual - reference).square().sum() / reference.square().sum())
                num = sum(np.square(output[k].astype(float) - eager[i][k].astype(float)).sum() for k in output)
                den = sum(np.square(eager[i][k].astype(float)).sum() for k in output)
                rows.append({"task_id": sample["task_id"], "init_index": sample["init_index"], "eager_noise_dtype": str(eager_noise[i].dtype), "graph_noise_dtype": str(draws[-1].dtype), "noise_bitwise_equal": torch.equal(draws[-1], eager_noise[i]), "noise_nmse": noise_nmse, "action_nmse": float(num / max(den, 1e-30))})
    finally:
        torch.randn = original_randn
        head.process_backbone_output = process
    result = {"purpose": "Only correctness audit; no deployment timing claim", "rows": rows,
              "eager_conditioning_dtypes": eager_dtypes, "graph_conditioning_dtypes": context_dtypes,
              "all_noise_values_equal": all(r["noise_bitwise_equal"] for r in rows), "graph_count": len(graphs.cache),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print("GRAPH_NOISE_AUDIT", result["all_noise_values_equal"], rows, flush=True)


if __name__ == "__main__":
    main()
