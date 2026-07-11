"""QVLA-style uniform-bit activation fake quantization (runtime hooks).

Env vars:
  QVLA_ACT_BITS     e.g. 4 or 8 (absent/>=16 -> disabled)
  QVLA_ACT_INCLUDE  regex over module names (default: eagle LLM + vision Linears)
  QVLA_ACT_EXCLUDE  regex over module names

Per-token symmetric fake quant on the inputs of matched nn.Linear modules,
matching QVLA's uniform-bit activation scheme.
"""

from __future__ import annotations

import os
import re

import torch
import torch.nn as nn

DEFAULT_INCLUDE = r".*backbone\.eagle_model\.(language_model|vision_model)\..*"
DEFAULT_EXCLUDE = r"(?:^|\.)(lm_head|embed|embeddings|norm|ln|layernorm|mlp1|rotary)(?:\.|$)"


@torch.no_grad()
def _fake_quant_act(x: torch.Tensor, bits: int) -> torch.Tensor:
    if not torch.is_floating_point(x):
        return x
    qmax = (1 << (bits - 1)) - 1
    scale = (x.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-8)
    return torch.clamp(torch.round(x / scale), -(qmax + 1), qmax) * scale


def enable_qvla_act_if_configured(model: nn.Module) -> bool:
    bits = int(os.environ.get("QVLA_ACT_BITS", "16"))
    if bits >= 16:
        return False
    inc = re.compile(os.environ.get("QVLA_ACT_INCLUDE", DEFAULT_INCLUDE))
    exc_pat = os.environ.get("QVLA_ACT_EXCLUDE", DEFAULT_EXCLUDE)
    exc = re.compile(exc_pat) if exc_pat else None

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not inc.search(name):
            continue
        if exc is not None and exc.search(name):
            continue

        def pre_hook(_m, inputs, _bits=bits):
            x = inputs[0]
            if torch.is_tensor(x):
                x = _fake_quant_act(x, _bits)
            return (x,) + tuple(inputs[1:])

        module.register_forward_pre_hook(pre_hook)
        count += 1
    print(f"[QVLA] A{bits} per-token act-quant hooks on {count} Linear modules")
    return count > 0
