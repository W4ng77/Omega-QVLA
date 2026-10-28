"""Explicit common optimization: reuse identical cross-attention K/V within8 steps.

Only the verified GR00T interleaved cross-attention architecture is supported.
Never reuse between policy calls. Per-step activation tables and converted W
banks must be bitwise equal before two steps can share a projection result.
"""
import torch
from research.holoq_bench.packed_runtime import PackedLinear
from gr00t.quantization.dit_step_context import get_current_dit_step


def equivalent_steps(tables):
    """Partition8 steps only when every relevant frozen coefficient is equal."""
    assert tables and all(len(table) == 8 for table in tables)
    representatives, classes = [], []
    for step in range(8):
        found = next((r for r in representatives if all(torch.equal(table[step], table[r]) for table in tables)), None)
        if found is None:
            representatives.append(step)
            found = step
        classes.append(found)
    return classes


class ProjectionCache:
    def __init__(self, forward, classes):
        self.forward = forward
        self.classes = classes
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def __call__(self, x):
        step = get_current_dit_step()
        if step is None or not 0 <= step < 8:
            raise RuntimeError("Cross-K/V cache requires an explicit8-step scope")
        try:
            version = x._version
        except RuntimeError:
            # Inference tensors have no version counter. This optimization is
            # restricted to GR00T's unchanged, non-mutated encoder context.
            version = None
        group = self.classes[step]
        entry = self.cache.get(group)
        if entry is not None and entry[0] is x and entry[1] == version:
            return entry[2]
        output = self.forward(x)
        self.cache[group] = (x, version, output)
        return output


def install(policy):
    head = policy.model.action_head
    dit = head.model
    if type(dit).__name__ != "DiT" or not dit.config.interleave_self_attention or len(dit.transformer_blocks) != 16:
        raise ValueError("Only verified GR00T DiT16 interleaved architecture supported")
    if head.num_inference_timesteps != 8 or head.training:
        raise ValueError("Eight-step inference required")
    state = {"name": "qwen3_last_token_logits_and_identical_cross_KV_cache", "projections": {}, "caches": []}
    for index in range(0, 16, 2):
        block = dit.transformer_blocks[index]
        attn = block.attn1
        if block.cross_attention_dim != 2048 or attn.norm_cross:
            raise ValueError("Unexpected dynamic/normalized encoder-context path")
        for key in ("to_k", "to_v"):
            module = getattr(attn, key)
            if isinstance(module, PackedLinear) and getattr(module, "act_mode", None) == "step":
                if module.backend == "native_s4":
                    integer = module.integer
                    tables = [integer.act_scale_table.detach().cpu()]
                    if integer.mode == "step_weight":
                        tables += [integer.packed_weight[:8].detach().cpu(), integer.weight_scales[:8].detach().cpu()]
                    elif integer.mode != "group_int":
                        raise ValueError("Unknown integer scale dependence")
                    else:
                        tables += [integer.group_scale_table.detach().cpu()]
                else:
                    tables = [module.a_scale.detach().cpu()]
                classes = equivalent_steps(tables)
            else:
                classes = [0] * 8
            cached = ProjectionCache(module.forward, classes)
            module.forward = cached
            state["caches"].append(cached)
            state["projections"][f"action_head.model.transformer_blocks.{index}.attn1.{key}"] = {"step_classes": classes, "distinct_projections_per_policy": len(set(classes)), "backend": getattr(module, "backend", "unchanged_FP16_excluded_attention")}
    graphs = head.get_action
    if not hasattr(graphs, "original") or not hasattr(graphs, "cache"):
        raise ValueError("Explicit common HeadGraphs adapter required")
    original = graphs.original

    def invoke(*args, **kwargs):
        for cached in state["caches"]:
            cached.clear()
        return original(*args, **kwargs)

    graphs.original = invoke
    torch.cuda.synchronize()
    graphs.cache.clear()
    # Clear closures/caches before serializing the public configuration.
    public = {k: v for k, v in state.items() if k != "caches"}
    print("EXPLICIT_COMMON_IDENTICAL_CROSS_KV_CACHE", public, flush=True)
    return public
