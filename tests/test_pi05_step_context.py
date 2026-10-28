import pytest

from gr00t.quantization.dit_step_context import get_current_dit_step
from gr00t.quantization.pi05_step_context import install_pi05_step_context


def test_steps_reset_for_each_inference_and_context_is_restored():
    class Model:
        def __init__(self):
            self.seen = []

        def sample_actions(self, device, observation, noise=None, num_steps=10):
            assert get_current_dit_step() is None
            for _ in range(num_steps):
                self.denoise_step()
            return noise

        def denoise_step(self):
            self.seen.append(get_current_dit_step())

    model = Model()
    install_pi05_step_context(model)
    model.sample_actions(None, None, num_steps=3)
    model.sample_actions(None, None, num_steps=3)
    assert model.seen == [0, 1, 2, 0, 1, 2]
    assert get_current_dit_step() is None
    with pytest.raises(RuntimeError, match="outside"):
        model.denoise_step()
