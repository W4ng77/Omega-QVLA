"""Port official AWQ Qwen2 grouping/module loss/folding to GR00T Qwen3.

The DiT has no upstream AWQ architecture adapter; retain the explicit official
linear-inspection port there. Never represent that distinction as exact paper
reproduction. Searches use 512 loss/statistic tokens from the shared corpus,
with intact causal contexts for attention rather than shuffled token attention.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from torch import nn
import tree

from research.holoq_bench import kernels, quantizers
from research.holoq_bench.collect_calibration import seed


class AttentionInspection(nn.Module):
    def __init__(self, attention, contexts):
        super().__init__()
        self.attention = attention
        self.contexts = contexts

    def forward(self, unused):
        outputs = []
        for x, kwargs, indices in self.contexts:
            out = self.attention(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out.reshape(-1, out.shape[-1])[indices])
        return torch.cat(outputs)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["HOLOQ_COMPUTE_DTYPE"] = "float16"
    for key in ["HOLOQ_PACK", "HOLOQ_DEPLOY"]:
        os.environ.pop(key, None)
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=a.checkpoint,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )
    llm = policy.model.backbone.eagle_model.language_model
    assert type(llm).__name__ in ("Qwen2ForCausalLM", "Qwen3ForCausalLM"), type(llm)
    llm.config.use_cache = False
    blocks = llm.model.layers
    contexts = [[] for _ in blocks]

    def hook(index):
        def collect(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            assert kw.get("past_key_value") is None, type(kw.get("past_key_value"))
            assert kw.get("past_key_values") is None, type(kw.get("past_key_values"))

            def cpu(v):
                return v.detach().cpu() if isinstance(v, torch.Tensor) else v

            contexts[index].append((x.detach().cpu(), tree.map_structure(cpu, kw)))

        return collect

    handles = [b.self_attn.register_forward_pre_hook(hook(i), with_kwargs=True) for i, b in enumerate(blocks)]
    observations = torch.load(a.observations, weights_only=False)
    for i, sample in enumerate(observations):
        seed(sample["seed"])
        policy.get_action(sample["obs"])
        if i % 20 == 19:
            print("ATTENTION_CONTEXT", i + 1, flush=True)
    for h in handles:
        h.remove()
    policy.model.cpu()
    torch.cuda.empty_cache()
    qpath = quantizers.ROOT / "third_party/llm-awq/awq/quantize"
    ns = {"torch": torch, "gc": gc}
    pseudo = quantizers.official_function(qpath / "quantizer.py", "pseudo_quantize_tensor", ns)
    qc = {"q_group_size": 128, "zero_point": True}
    ns["w_quantize_func"] = lambda w: pseudo(w, n_bit=4, **qc).detach()
    ns["get_act_scale"] = lambda x: x.abs().reshape(-1, x.shape[-1]).mean(0)
    search = quantizers.official_function(qpath / "auto_scale.py", "_search_module_scale", ns)
    clip = quantizers.official_function(
        qpath / "auto_clip.py", "auto_clip_layer", dict(ns, pseudo_quantize_tensor=pseudo)
    )
    manifest = json.loads((a.capture / "manifest.json").read_text())
    input_files = {r["name"]: r["file"] for r in manifest["layers"]}
    records = {}
    overrides = {}
    details = []
    for index, block in enumerate(blocks):
        block.cuda()
        prefix = f"backbone.eagle_model.language_model.model.layers.{index}"
        linears = {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}
        features = {}
        for n in linears:
            raw = torch.load(
                a.capture / input_files[prefix + "." + n], map_location="cpu", mmap=True, weights_only=False
            )[-1]
            gen = torch.Generator().manual_seed(42 + index)
            features[n] = raw[torch.randperm(len(raw), generator=gen)[:512]].cuda()
        # The original search computes module-output loss. Keep complete causal
        # contexts, selecting 512 outputs evenly over all 100 shared frames.
        staged = []
        for frame, (x, kwargs) in enumerate(contexts[index]):
            count = 6 if frame < 12 else 5
            ids = torch.linspace(0, x.shape[-2] - 1, count, device="cuda").long()

            def gpu(v):
                return v.cuda() if isinstance(v, torch.Tensor) else v

            staged.append((x.cuda(), tree.map_structure(gpu, kwargs), ids))
        inspect = AttentionInspection(block.self_attn, staged)
        with torch.no_grad():
            s = search(
                inspect,
                [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj],
                features["self_attn.q_proj"],
            )
            block.input_layernorm.weight.div_(s)
            for name in ["q_proj", "k_proj", "v_proj"]:
                getattr(block.self_attn, name).weight.mul_(s[None, :])
                features["self_attn." + name].div_(s)
            del inspect, staged
            s = search(block.mlp, [block.mlp.gate_proj, block.mlp.up_proj], features["mlp.gate_proj"])
            block.post_attention_layernorm.weight.div_(s)
            for name in ["gate_proj", "up_proj"]:
                getattr(block.mlp, name).weight.mul_(s[None, :])
                features["mlp." + name].div_(s)
            s = search(block.mlp.down_proj, [block.mlp.down_proj], features["mlp.down_proj"])
            block.mlp.up_proj.weight.div_(s[:, None])
            if block.mlp.up_proj.bias is not None:
                block.mlp.up_proj.bias.div_(s)
            block.mlp.down_proj.weight.mul_(s[None, :])
            features["mlp.down_proj"].div_(s)
            for n, m in linears.items():
                w = m.weight.detach()
                if n not in ("self_attn.q_proj", "self_attn.k_proj"):
                    bounds = clip(
                        w, features[n], n_bit=4, q_config=qc, n_grid=20, max_shrink=0.5, n_sample_token=512
                    )
                    w = w.reshape(*bounds.shape[:2], -1).clamp(-bounds, bounds).reshape(m.weight.shape)
                quant, scales, zeros = pseudo(w, n_bit=4, get_scale_zp=True, **qc)
                q = (
                    (
                        quant.float() / scales.float().repeat_interleave(128, 1)
                        + zeros.float().repeat_interleave(128, 1)
                    )
                    .round()
                    .clamp(0, 15)
                    .int()
                )
                records[prefix + "." + n] = dict(
                    kernels.pack_awq(q.cpu(), scales.cpu(), zeros.cpu()),
                    K=m.in_features,
                    N=m.out_features,
                    act_bits=16,
                    act_mode="none",
                    act_qmin=-32768,
                    bias=m.bias.detach().cpu().clone() if m.bias is not None else None,
                )
            for name in ["input_layernorm", "post_attention_layernorm"]:
                for suffix, param in getattr(block, name).named_parameters():
                    overrides[prefix + "." + name + "." + suffix] = param.detach().cpu().clone()
        block.cpu()
        del features
        contexts[index].clear()
        gc.collect()
        torch.cuda.empty_cache()
        details.append(
            {
                "block": index,
                "attention_loss_tokens": 512,
                "full_causal_context_frames": 100,
                "scale_candidates": 20,
                "weight_group": 128,
                "gqa_output_scaling": "skip as upstream Qwen2 adapter",
            }
        )
        print("AWQ_QWEN_BLOCK", index + 1, "/", len(blocks), flush=True)
    for variant in ["selective", "full"]:
        pack = torch.load(a.packs / f"awq_{variant}.pt", map_location="cpu", mmap=True, weights_only=False)
        pack["layers"].update(records)
        pack["nonquantized_overrides"] = overrides
        pack["meta"].update(
            method=f"awq_qwen_{variant}",
            quantizer_adapter="upstream Qwen2 grouped attention/MLP search and folding ported to Qwen3 calls; GR00T DiT linear inspection",
            qwen_search_details=details,
        )
        path = a.output / f"awq_qwen_{variant}.pt"
        torch.save(pack, path)
        print("EXPORTED", path, flush=True)


if __name__ == "__main__":
    main()
