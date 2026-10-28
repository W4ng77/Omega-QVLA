"""Fixed-step eager pi0.5 sampling and CUDA Graph replay, with unchanged A4.

Unlike rotation folding, replay does not change any quantization scale, code,
or transform. Every step's selected scale pointer is captured in the graph.
"""

from __future__ import annotations

import torch


def embed_prefix_graph_safe(self, images, img_masks, lang_tokens, lang_masks):
    """Same embeddings/masks as OpenPI, without CPU-list tensor uploads."""
    import math

    embs, masks = [], []
    for image, mask in zip(images, img_masks, strict=True):
        embedded = self._apply_checkpoint(self.paligemma_with_expert.embed_image, image)
        embs.append(embedded)
        masks.append(mask[:, None].expand(embedded.shape[0], embedded.shape[1]))

    def language(tokens):
        embedded = self.paligemma_with_expert.embed_language_tokens(tokens)
        return embedded * math.sqrt(embedded.shape[-1])

    embs.append(self._apply_checkpoint(language, lang_tokens))
    masks.append(lang_masks)
    embs, masks = torch.cat(embs, dim=1), torch.cat(masks, dim=1)
    return embs, masks, torch.zeros_like(masks, dtype=torch.bool)


def embed_suffix_graph_safe(self, state, noisy_actions, timestep):
    """Pi0.5 suffix only; preserve the original timestep and action MLP math."""
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

    time_emb = create_sinusoidal_pos_embedding(
        timestep, self.action_in_proj.out_features, min_period=4e-3,
        max_period=4.0, device=timestep.device,
    ).type(dtype=timestep.dtype)
    action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

    def time_mlp(x):
        x = torch.nn.functional.silu(self.time_mlp_in(x))
        return torch.nn.functional.silu(self.time_mlp_out(x))

    condition = self._apply_checkpoint(time_mlp, time_emb)
    bsize, horizon = action_emb.shape[:2]
    mask = torch.ones((bsize, horizon), dtype=torch.bool, device=timestep.device)
    att_mask = (torch.arange(horizon, device=action_emb.device) == 0).to(action_emb.dtype)
    return action_emb, mask, att_mask[None, :].expand(bsize, horizon), condition


def sample_actions_fixed_steps(self, device, observation, noise=None, num_steps=10):
    """OpenPI's sampling math, replacing only its tensor-valued while test."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    bsize = observation.state.shape[0]
    if noise is None:
        noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)
    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
    )
    dt = torch.full((), -1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.ones((), dtype=torch.float32, device=device)
    for _ in range(num_steps):
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
        x_t = x_t + dt * v_t
        time += dt
    return x_t


class Pi05CudaGraphSampler:
    """Single-shape inference runner; copy new inputs into captured buffers.

    The caller must run the normal per-layer/step audit before constructing
    this runner. Python hooks do not execute during CUDA Graph replay.
    """

    def __init__(self, sample_actions):
        self.sample_actions = sample_actions
        self.graph = None
        self.signature = None

    def __call__(self, device, observation, noise=None, num_steps=10):
        import jax

        if noise is None:
            raise ValueError("Provide explicit noise when benchmarking CUDA Graph sampling")
        leaves, treedef = jax.tree.flatten((observation, noise))
        if not all(isinstance(leaf, torch.Tensor) and leaf.is_cuda for leaf in leaves):
            raise ValueError("CUDA Graph inputs must be CUDA tensor leaves")
        signature = (str(device), num_steps, treedef,
                     tuple((tuple(leaf.shape), leaf.dtype, leaf.device) for leaf in leaves))
        if self.graph is None:
            self.signature = signature
            self.static_leaves = [leaf.clone() for leaf in leaves]
            static_obs, static_noise = jax.tree.unflatten(treedef, self.static_leaves)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.sample_actions(device, static_obs, noise=static_noise, num_steps=num_steps)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.sample_actions(device, static_obs, noise=static_noise, num_steps=num_steps)
        elif signature != self.signature:
            raise ValueError("Input signature changed: create a new CUDA Graph runner")
        for source, target in zip(leaves, self.static_leaves, strict=True):
            target.copy_(source)
        self.graph.replay()
        return self.output
