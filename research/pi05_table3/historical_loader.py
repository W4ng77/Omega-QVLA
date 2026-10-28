"""Self-contained loading and partial-scope controls of the frozen414 pack."""

import json
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_model
from research.pi05_table3.measure import sha, write

SOURCE = Path("/home/xinyu/models/omega-qvla-pi05-414-w4a4-int4")
CHECKPOINT = Path("/home/xinyu/models/pi05_libero_pytorch")


def export_scope(output, scope):
    """One selected suite, all required metadata and all nonquantized weights."""
    output = Path(output)
    if (output / "deployment.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    full = torch.load(SOURCE / "pi05_object/quantized.pt", map_location="cpu", mmap=True, weights_only=False)
    vision = torch.load(SOURCE / "vision_int4.pt", map_location="cpu", mmap=True, weights_only=False)
    records = {
        n: r
        for n, r in full.items()
        if n == "__meta__"
        or scope == "both"
        or (scope == "backbone" and "language_model" in n)
        or (scope == "head" and "gemma_expert" in n)
    }
    selected_vision = vision if scope != "head" else {}
    excluded = {n + ".weight" for n in records if n != "__meta__"} | set(selected_vision)
    expected = {"both": 414, "backbone": 288, "head": 126}[scope]
    assert len(excluded) == expected
    torch.save(records, output / "quantized.pt")
    torch.save(selected_vision, output / "vision_int4.pt")
    nonquant = {}
    with safe_open(CHECKPOINT / "model.safetensors", framework="pt", device="cpu") as source:
        for key in source.keys():
            if key not in excluded:
                nonquant[key] = source.get_tensor(key)
        save_file(nonquant, output / "model_nonquant.safetensors", metadata=source.metadata())
    shutil.copy2(CHECKPOINT / "config.json", output / "config.json")
    shutil.copytree(CHECKPOINT / "assets", output / "assets", dirs_exist_ok=True)
    inventory = {
        str(f.relative_to(output)): {"bytes": f.stat().st_size, "sha256": sha(f)}
        for f in output.rglob("*")
        if f.is_file()
    }
    write(
        output / "deployment.json",
        {
            "scope": scope,
            "quantized_layers": expected,
            "excluded_state_keys": sorted(excluded),
            "files": inventory,
            "model_artifact_bytes": sum(x["bytes"] for x in inventory.values()),
            "source_pack_sha256": sha(SOURCE / "pi05_object/quantized.pt"),
            "source_vision_sha256": sha(SOURCE / "vision_int4.pt"),
            "storage_boundary": "Single Object policy model files, all remaining weights and all quantization/rotation metadata; shared tokenizer/runtime software listed separately.",
        },
    )


def load_policy(directory, cfg):
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies import policy as policy_module
    from openpi.training import checkpoints
    from openpi import transforms
    from transformers.modeling_utils import no_init_weights
    from gr00t.quantization.gptq_layers import GptqConfig, wrap_gptq
    from gr00t.quantization.packed_w4a4 import enable_packed_w4a4
    from research.pi05_table3.vision_packed import PackedVision

    directory = Path(directory)
    manifest = json.loads((directory / "deployment.json").read_text())
    records = torch.load(directory / "quantized.pt", map_location="cpu", mmap=True, weights_only=False)
    vision = torch.load(directory / "vision_int4.pt", map_location="cpu", mmap=True, weights_only=False)
    with no_init_weights():
        model = PI0Pytorch(config=cfg.model)
    model.paligemma_with_expert.paligemma.tie_weights()
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    missing, unexpected = load_model(model, str(directory / "model_nonquant.safetensors"), strict=False)
    expected = set(manifest["excluded_state_keys"])
    assert set(missing) == expected, {
        "extra_missing": sorted(set(missing) - expected),
        "missing_not_reported": sorted(expected - set(missing)),
    }
    assert not unexpected
    lm_names = [n for n in records if n != "__meta__"]
    assert wrap_gptq(
        model,
        lm_names,
        GptqConfig(enabled=True, path=str(directory / "quantized.pt"), act_bits=4, weight_bits=4, missing="error"),
    ) == len(lm_names)
    enable_packed_w4a4(model, records, version=2)
    modules = dict(model.named_modules())
    for weight_name, record in vision.items():
        name = weight_name[:-7]
        parent, attr = name.rsplit(".", 1)
        setattr(modules[parent], attr, PackedVision(modules[name], record, name))
    del modules
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    stats = checkpoints.load_norm_stats(directory / "assets", data.asset_id)
    return policy_module.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": 10},
        metadata=cfg.policy_metadata,
        is_pytorch=True,
        pytorch_device="cuda",
    )
