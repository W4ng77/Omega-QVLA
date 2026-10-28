import pytest
import torch
from types import SimpleNamespace

from gr00t.quantization.dit_step_context import get_current_dit_step, set_dit_quant_step
from gr00t.quantization.pi05_cuda_graph import Pi05CudaGraphSampler


def test_graph_safe_embeddings_match_native_openpi():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from gr00t.quantization.pi05_cuda_graph import embed_prefix_graph_safe, embed_suffix_graph_safe

    torch.manual_seed(37)
    model = SimpleNamespace(
        pi05=True, config=SimpleNamespace(action_horizon=10),
        _apply_checkpoint=lambda fn, *args: fn(*args),
        action_in_proj=torch.nn.Linear(32, 16), time_mlp_in=torch.nn.Linear(16, 16),
        time_mlp_out=torch.nn.Linear(16, 16),
        paligemma_with_expert=SimpleNamespace(embed_image=lambda image: image,
                                            embed_language_tokens=torch.nn.Embedding(32, 16)),
    )
    images = [torch.randn(2, 4, 16), torch.randn(2, 4, 16)]
    masks = [torch.tensor([True, False]), torch.tensor([True, True])]
    tokens = torch.randint(0, 32, (2, 5))
    token_mask = torch.ones_like(tokens, dtype=torch.bool)
    native = PI0Pytorch.embed_prefix(model, images, masks, tokens, token_mask)
    safe = embed_prefix_graph_safe(model, images, masks, tokens, token_mask)
    for actual, expected in zip(safe, native, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    state, noise, time = torch.randn(2, 8), torch.randn(2, 10, 32), torch.tensor([0.9, 0.3])
    native = PI0Pytorch.embed_suffix(model, state, noise, time)
    safe = embed_suffix_graph_safe(model, state, noise, time)
    for actual, expected in zip(safe, native, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fixed_step_sampling_matches_native_while_loop():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from gr00t.quantization.pi05_cuda_graph import sample_actions_fixed_steps

    prefix_masks = torch.ones((2, 3), dtype=torch.bool)
    inner = SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace()))
    model = SimpleNamespace(
        config=SimpleNamespace(action_horizon=10, action_dim=32),
        _preprocess_observation=lambda obs, train: (None, None, None, None, obs),
        embed_prefix=lambda *args: (torch.zeros(2, 3, 16), prefix_masks, torch.zeros_like(prefix_masks)),
        _prepare_attention_masks_4d=lambda mask: mask,
        paligemma_with_expert=SimpleNamespace(paligemma=inner, forward=lambda **kwargs: (None, None)),
        denoise_step=lambda state, masks, past, x, time: x * 0.2 + time[:, None, None] * 0.1,
    )
    noise = torch.randn(2, 10, 32)
    observation = SimpleNamespace(state=torch.zeros(2, 8))
    native = PI0Pytorch.sample_actions(model, "cpu", observation, noise=noise, num_steps=10)
    safe = sample_actions_fixed_steps(model, "cpu", observation, noise=noise, num_steps=10)
    torch.testing.assert_close(safe, native, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_copies_inputs_and_keeps_each_steps_scale():
    scales = torch.tensor([0.1, 0.3, 0.7], device="cuda")

    def sample(device, observation, noise=None, num_steps=3):
        out = observation["x"].clone()
        for step in range(num_steps):
            with set_dit_quant_step(step, total=num_steps):
                out = out + noise * scales[get_current_dit_step()]
        return out

    runner = Pi05CudaGraphSampler(sample)
    x = torch.randn(2, 8, device="cuda")
    noise = torch.randn_like(x)
    with torch.inference_mode():
        torch.testing.assert_close(runner("cuda", {"x": x}, noise=noise, num_steps=3),
                                   sample("cuda", {"x": x}, noise=noise))
        x2, noise2 = x * 2, noise * -3
        torch.testing.assert_close(runner("cuda", {"x": x2}, noise=noise2, num_steps=3),
                                   sample("cuda", {"x": x2}, noise=noise2))
        scales.copy_(torch.tensor([0.8, 0.1, 0.05], device="cuda"))
        torch.testing.assert_close(runner("cuda", {"x": x}, noise=noise, num_steps=3),
                                   sample("cuda", {"x": x}, noise=noise))
        with pytest.raises(ValueError, match="signature"):
            runner("cuda", {"x": x[:1]}, noise=noise[:1], num_steps=3)
