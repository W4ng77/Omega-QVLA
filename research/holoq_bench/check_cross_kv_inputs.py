"""Accuracy-only architecture invariant: cross K/V context is unchanged across8 steps."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import torch
from research.holoq_bench.collect_calibration import seed
from gr00t.quantization.dit_step_context import get_current_dit_step


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    lock = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    os.environ.update(HOLOQ_DEPLOY=str(a.deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16")
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(model_path=str(a.deployment.resolve()), modality_config=config.modality_config(), modality_transform=config.transform(), embodiment_tag="new_embodiment", denoising_steps=8)
    modules = {f"action_head.model.transformer_blocks.{i}.attn1.{key}": getattr(policy.model.action_head.model.transformer_blocks[i].attn1, key) for i in range(0, 16, 2) for key in ("to_k", "to_v")}
    references, rows = {}, []
    task = [None]

    def hook(name):
        def check(module, inputs):
            step = get_current_dit_step()
            x = inputs[0]
            if step == 0:
                references[name] = (x, x.clone())
            identity, frozen = references[name]
            rows.append({"task_id": task[0], "layer": name, "step": step, "shape": list(x.shape), "stride": list(x.stride()), "same_tensor_object": x is identity, "bitwise_equal": bool(torch.equal(x, frozen))})
        return check

    handles = [module.register_forward_pre_hook(hook(name)) for name, module in modules.items()]
    observations = torch.load(a.observations, map_location="cpu", weights_only=False)
    with torch.inference_mode():
        for sample in observations[::10]:
            task[0] = sample["task_id"]
            references.clear()
            seed(42 + sample["task_id"] * 10000 + sample["init_index"] * 1000)
            policy.get_action(sample["obs"])
    for handle in handles:
        handle.remove()
    assert len(rows) == 1280 and all(row["bitwise_equal"] for row in rows)
    output = {"purpose": "accuracy-only constant context invariant; no speed numbers", "deployment": str(a.deployment), "comparisons": len(rows), "all_inputs_bitwise_equal_across_steps": True, "all_same_tensor_object": all(row["same_tensor_object"] for row in rows), "rows": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + "\n")
    print("CROSS_KV_CONTEXT_INVARIANT_PASSED", len(rows), output["all_same_tensor_object"], flush=True)


if __name__ == "__main__":
    main()
