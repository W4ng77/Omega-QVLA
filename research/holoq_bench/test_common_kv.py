"""Guard changing per-step scales/banks and policy/input cache invalidation."""
import torch
import pytest
from types import SimpleNamespace
from gr00t.quantization.dit_step_context import set_dit_quant_step
from research.holoq_bench.common_kv import ProjectionCache, equivalent_steps


def test_unequal_activation_row_or_weight_bank_cannot_share_step():
    scales = torch.ones(8, 4)
    codes = torch.zeros((8, 2, 2), dtype=torch.uint8)
    weights = torch.ones((8, 2, 1))
    scales[2, 0] = 2
    codes[3, 0, 0] = 1
    weights[4, 1, 0] = 3
    assert equivalent_steps([scales, codes, weights]) == [0, 0, 2, 3, 4, 0, 0, 0]


def test_projection_refreshes_for_new_input_mutation_and_new_policy():
    calls = []

    def projection(x):
        calls.append(1)
        return x * 2

    cache = ProjectionCache(projection, [0] * 8)
    x = torch.ones(2, 4)
    with set_dit_quant_step(0, total=8):
        first = cache(x)
    with set_dit_quant_step(1, total=8):
        assert cache(x) is first
        x.add_(1)
        assert torch.equal(cache(x), x * 2)
    assert len(calls) == 2
    with torch.inference_mode(), set_dit_quant_step(2, total=8):
        y = torch.full((2, 4), 3.)
        assert torch.equal(cache(y), y * 2)
        cache.clear()  # A new full policy must recompute even with same buffer.
        y.fill_(5.)
        assert torch.equal(cache(y), y * 2)
    assert len(calls) == 4


def test_nonidentical_step_group_recomputes_projection():
    calls = []
    cache = ProjectionCache(lambda x: calls.append(1) or x + len(calls), [0, 0, 2, 2, 4, 4, 6, 6])
    x = torch.ones(1)
    for step in range(8):
        with set_dit_quant_step(step, total=8):
            assert cache(x).item() == 2 + step // 2
    assert len(calls) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA Graph required")
def test_graph_recomputes_cached_projection_once_for_each_changed_policy_input():
    from research.holoq_bench.graphs import HeadGraphs

    class Head:
        config = SimpleNamespace(action_horizon=2, action_dim=4)

        def __init__(self):
            self.calls = 0
            self.projection = ProjectionCache(self.project, [0] * 8)

        def project(self, x):
            self.calls += 1
            return x * 2

        def get_action(self, backbone, action):
            self.projection.clear()
            x = backbone["backbone_features"]
            y = torch.zeros_like(x)
            for step in range(8):
                with set_dit_quant_step(step, total=8):
                    y = y + self.projection(x)
            return {"action_pred": y / 8}

    head = Head()
    graphs = HeadGraphs(head)
    x = torch.ones(1, 2, 4, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        a = graphs({"backbone_features": x}, {})["action_pred"].clone()
        b = graphs({"backbone_features": x * 3}, {})["action_pred"].clone()
    torch.testing.assert_close(a, x * 2, rtol=0, atol=0)
    torch.testing.assert_close(b, x * 6, rtol=0, atol=0)
    assert len(graphs.cache) == 1 and head.calls == 4  #3 warm invokes+1 capture
