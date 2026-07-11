"""CUDA-graph capture for the DiT denoising loop (GR00T flow-matching head).

The 8-step denoising loop issues ~768 quantized-linear calls per get_action,
each with ~30 python-dispatched CUDA ops — the workload is CPU-launch-bound
(FP16 78ms vs W4A8 ~190ms on H100). Capturing the whole loop as a CUDA graph
replaces thousands of kernel launches with one replay call.

Enabled via GR00T_DIT_CUDA_GRAPH=1. Graphs are cached per input-shape key
(prompt length varies per task); the first call for a key runs eager (also
serving as warmup that materializes all lazy buffers in GptqLinear), the
second call captures, subsequent calls replay.

Notes:
  * torch.randn inside the captured region is safe: PyTorch registers the
    CUDA generator with the graph and advances the Philox offset per replay.
  * The per-step contextvar (set_dit_quant_step) is python-level: during
    capture each iteration records ops referencing that step's scale row,
    which is exactly the desired per-step behavior baked into the graph.
  * Act-scale calibration must be finished before capture; DiT layers use
    static act_scale_table rows, and lazy buffers (_act_table_cast, ...)
    are created during the eager warmup call.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

import torch

_MAX_GRAPHS = int(os.environ.get("GR00T_DIT_CUDA_GRAPH_MAX", "16"))


def enabled() -> bool:
    return os.environ.get("GR00T_DIT_CUDA_GRAPH", "0") not in ("0", "false", "False")


def backbone_graph_enabled() -> bool:
    return os.environ.get("GR00T_BACKBONE_CUDA_GRAPH", "0") not in ("0", "false", "False")


_MASK_PATCHED = False


def _patch_hf_mask_shortcut() -> None:
    """transformers' SDPA fast path calls torch.all(mask == 1) -> CPU sync,
    which is illegal during CUDA-graph capture. Returning False just skips
    the shortcut (explicit 4D mask is built instead — pure tensor ops,
    semantically identical attention)."""
    global _MASK_PATCHED
    if _MASK_PATCHED:
        return
    try:
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter

        orig = AttentionMaskConverter._ignore_causal_mask_sdpa

        def safe_ignore(attention_mask, *a, **k):
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return False
            return orig(attention_mask, *a, **k)

        AttentionMaskConverter._ignore_causal_mask_sdpa = staticmethod(safe_ignore)
        _MASK_PATCHED = True
        print("[BB-graph] patched HF _ignore_causal_mask_sdpa for capture safety", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[BB-graph] mask patch failed: {e}", flush=True)


class DictGraphRunner:
    """Shape-keyed CUDA-graph cache around fn(feature_dict) -> feature_dict.

    Tensor entries are copied into static buffers; non-tensor entries must be
    call-invariant (they are baked at capture time)."""

    def __init__(self, fn):
        self.fn = fn
        self.cache = {}
        _patch_hf_mask_shortcut()

    @staticmethod
    def _tensors(d):
        return {k: v for k, v in d.items() if torch.is_tensor(v)}

    def _key(self, d):
        return tuple(sorted((k, tuple(v.shape), str(v.dtype)) for k, v in self._tensors(d).items()))

    def __call__(self, batch):
        data = dict(batch)
        # HF SDPA runs torch.all(mask == 1) which syncs and breaks capture.
        # For unpadded single-sample eval the mask is always all-ones — drop
        # it (None mask == pure causal attention, semantically identical).
        am = data.get("attention_mask")
        if torch.is_tensor(am) and bool((am == 1).all()):
            data["attention_mask"] = None
        key = self._key(data)
        entry = self.cache.get(key)
        if entry is None:
            if len(self.cache) >= _MAX_GRAPHS:
                return self.fn(type(batch)(data=data) if hasattr(batch, "data") else data)
            self.cache[key] = {"state": "warm"}
            return self.fn(type(batch)(data=data) if hasattr(batch, "data") else data)
        if entry.get("state") == "warm":
            static_in = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            cls = type(batch)
            with torch.cuda.graph(graph):
                static_out = self.fn(cls(data=static_in) if hasattr(batch, "data") else static_in)
            entry.update(state="ready", graph=graph, sin=static_in, out=static_out)
            print(f"[BB-graph] captured backbone graph ({len(self.cache)} cached)", flush=True)
            graph.replay()
            out = static_out
        else:
            for k, v in self._tensors(data).items():
                entry["sin"][k].copy_(v, non_blocking=True)
            entry["graph"].replay()
            out = entry["out"]
        # clone tensor outputs so later ops don't alias graph-owned memory
        od = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dict(out).items()}
        return type(out)(data=od) if hasattr(out, "data") else od


class DiTGraphRunner:
    """Shape-keyed CUDA-graph cache around a denoise(vl, state, emb) -> actions fn."""

    def __init__(self, fn: Callable[..., torch.Tensor]):
        self.fn = fn
        self.cache: Dict[Tuple, dict] = {}

    def _key(self, vl_embs, state_features, embodiment_id):
        return (
            tuple(vl_embs.shape), vl_embs.dtype,
            tuple(state_features.shape), state_features.dtype,
            tuple(embodiment_id.shape),
        )

    def __call__(self, vl_embs, state_features, embodiment_id):
        if not (enabled() and vl_embs.is_cuda):
            return self.fn(vl_embs, state_features, embodiment_id)

        key = self._key(vl_embs, state_features, embodiment_id)
        entry = self.cache.get(key)

        if entry is None:
            if len(self.cache) >= _MAX_GRAPHS:
                return self.fn(vl_embs, state_features, embodiment_id)
            # first sighting: eager (warmup — materializes lazy buffers)
            self.cache[key] = {"state": "warm"}
            return self.fn(vl_embs, state_features, embodiment_id)

        if entry.get("state") == "warm":
            # second sighting: capture
            static_vl = vl_embs.clone()
            static_state = state_features.clone()
            static_emb = embodiment_id.clone()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self.fn(static_vl, static_state, static_emb)
            entry.update(
                state="ready", graph=graph, vl=static_vl, st=static_state,
                emb=static_emb, out=static_out,
            )
            print(f"[DiT-graph] captured graph for key {key[0]} ({len(self.cache)} cached)", flush=True)
            # the capture pass itself already computed a valid result? NO —
            # capture does not execute; replay once to produce real output.
            entry["graph"].replay()
            return entry["out"].clone()

        # ready: replay
        entry["vl"].copy_(vl_embs, non_blocking=True)
        entry["st"].copy_(state_features, non_blocking=True)
        entry["emb"].copy_(embodiment_id, non_blocking=True)
        entry["graph"].replay()
        return entry["out"].clone()
