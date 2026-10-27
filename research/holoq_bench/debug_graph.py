"""Identify graph drift using fixed head inputs/noise and repeated eager calls."""

import json
import os
from pathlib import Path

import torch
from transformers.feature_extraction_utils import BatchFeature

from research.holoq_bench.collect_calibration import seed


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "results/holoq_object/packs/gptq_selective.pt"
    os.environ.update(HOLOQ_PACK=str(path), HOLOQ_COMPUTE_DTYPE="float16")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from research.holoq_bench.graphs import HeadGraphs

    cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["meta"]["checkpoint"]
    policy = Gr00tPolicy(
        model_path=ckpt,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )
    obs = torch.load(root / "results/holoq_object/calibration_fp16_init0_v2/observations.pt", weights_only=False)[
        0
    ]["obs"]
    head = policy.model.action_head
    original = head.get_action
    cache = []

    def capture(backbone, action):
        cache[:] = [
            {k: v.clone() for k, v in dict(backbone).items()},
            {k: v.clone() for k, v in dict(action).items()},
        ]
        return original(backbone, action)

    head.get_action = capture
    policy.get_action(obs)
    head.get_action = original
    graphs = HeadGraphs(head)
    results = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(5):
            seed(1000)
            a = original(BatchFeature(data=dict(cache[0])), BatchFeature(data=dict(cache[1])))[
                "action_pred"
            ].clone()
            seed(1000)
            b = original(BatchFeature(data=dict(cache[0])), BatchFeature(data=dict(cache[1])))[
                "action_pred"
            ].clone()
            seed(1000)
            c = graphs(BatchFeature(data=dict(cache[0])), BatchFeature(data=dict(cache[1])))["action_pred"].clone()

            def err(x, y):
                return ((x.float() - y.float()).square().sum() / y.float().square().sum()).item()

            results.append(
                {
                    "eager_repeat_nmse": err(a, b),
                    "graph_vs_eager_nmse": err(c, a),
                    "eager_exact": torch.equal(a, b),
                    "graph_exact": torch.equal(a, c),
                }
            )
    (root / "results/holoq_object/graph_drift_diagnosis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(results, flush=True)


if __name__ == "__main__":
    main()
