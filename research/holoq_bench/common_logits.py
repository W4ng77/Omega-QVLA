"""Explicit common optimization: Qwen3 last-token logits, full hidden states.

GR00T consumes hidden_states, not the full vocabulary predictions. This uses
the installed model's own logits_to_keep API and leaves every decoder layer,
quantized projection, rotation and denoising step intact. Never auto-installed.
"""

import inspect


def install(policy):
    model = policy.model.backbone.eagle_model.language_model
    if type(model).__name__ != "Qwen3ForCausalLM" or model.training:
        raise RuntimeError("Validated only for evaluation-mode Qwen3 GR00T")
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        raise RuntimeError("Installed Qwen3 lacks its explicit logits_to_keep API")
    if hasattr(model, "_holoq_common_logits"):
        raise RuntimeError("Common logits optimization already installed")
    original = model.forward
    state = {"calls": 0, "logits_shapes": set(), "name": "qwen3_last_token_logits"}

    def forward(*args, **kwargs):
        if kwargs.get("labels") is not None or not kwargs.get("output_hidden_states"):
            raise RuntimeError("This optimization is only for GR00T hidden-state inference")
        if kwargs.get("logits_to_keep", 0) != 0 or kwargs.get("num_logits_to_keep", 0) != 0:
            raise RuntimeError("Caller already specifies a logits subset")
        kwargs.pop("num_logits_to_keep", None)
        kwargs["logits_to_keep"] = 1
        output = original(*args, **kwargs)
        if output.hidden_states is None or output.logits.shape[-2] != 1:
            raise RuntimeError("Unexpected hidden-state/logits contract")
        state["calls"] += 1
        state["logits_shapes"].add(tuple(output.logits.shape))
        return output

    model.forward = forward
    model._holoq_common_logits = state
    print("EXPLICIT_COMMON_QWEN3_LAST_TOKEN_LOGITS", flush=True)
    return state
