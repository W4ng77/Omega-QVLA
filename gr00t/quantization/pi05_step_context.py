"""Explicit timestep context for eager OpenPI sampling, without editing OpenPI."""

from __future__ import annotations

from .dit_step_context import set_dit_quant_step


def install_pi05_step_context(model, *, fixed_loop: bool = False) -> None:
    """Wrap the existing loop; check its actual call count on every inference.

    This adapter intentionally targets eager sampling. Compiling a Python
    ContextVar loop requires a different, tensor-valued step dispatch design.
    """
    if getattr(model, "_omega_step_adapter", False):
        raise ValueError("Step adapter already installed")
    if hasattr(model.sample_actions, "_torchdynamo_orig_callable"):
        raise ValueError("Disable pytorch_compile_mode before installing the eager step adapter")
    sample_actions = model.sample_actions
    if fixed_loop:
        from .pi05_cuda_graph import sample_actions_fixed_steps, embed_prefix_graph_safe, embed_suffix_graph_safe

        if not model.pi05:
            raise ValueError("The graph-safe adapter currently supports pi0.5 only")
        model.embed_prefix = embed_prefix_graph_safe.__get__(model)
        model.embed_suffix = embed_suffix_graph_safe.__get__(model)
        sample_actions = sample_actions_fixed_steps.__get__(model)
    denoise_step = model.denoise_step
    state = {"step": 0, "total": None}

    def denoise_with_step(*args, **kwargs):
        step, total = state["step"], state["total"]
        if total is None or step >= total:
            raise RuntimeError("denoise_step called outside the configured sampling loop")
        state["step"] += 1
        with set_dit_quant_step(step, total=total):
            return denoise_step(*args, **kwargs)

    def sample_with_steps(device, observation, noise=None, num_steps=10):
        if state["total"] is not None:
            raise RuntimeError("The eager sampling adapter is not reentrant")
        state.update(step=0, total=num_steps)
        try:
            with set_dit_quant_step(None, total=num_steps):
                result = sample_actions(device, observation, noise=noise, num_steps=num_steps)
            if state["step"] != num_steps:
                raise RuntimeError(f"Expected {num_steps} steps, observed {state['step']}")
            return result
        finally:
            state["total"] = None

    model.denoise_step = denoise_with_step
    model.sample_actions = sample_with_steps
    model._omega_step_adapter = True
