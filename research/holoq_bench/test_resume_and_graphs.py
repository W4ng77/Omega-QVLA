"""Behavioral regression tests for silent graph constants and interrupted runs."""

import json
from types import SimpleNamespace

import pytest
import torch

from research.holoq_bench.graphs import HeadGraphs
from research.holoq_bench.rollout_resume import reconcile_calls, validate_resume_metadata


def test_interrupted_episode_is_rerun_without_duplicate_timing_rows(tmp_path):
    complete = {"task_id": 0, "init_index": 10, "control_steps": 2}
    rows = [{"task_id": 0, "init_index": 10, "control_step": i, "ms": 4.0} for i in range(2)]
    orphan = {"task_id": 0, "init_index": 11, "control_step": 0, "ms": 8.0}
    original = "".join(json.dumps(x) + "\n" for x in rows + [orphan]).encode() + b'{"task_id":'
    path = tmp_path / "rollout_calls.jsonl"
    path.write_bytes(original)
    reconcile_calls(tmp_path, [complete])
    assert [json.loads(x) for x in path.read_text().splitlines()] == rows
    backups = list(tmp_path.glob("interrupted_calls_*.jsonl"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    # A second restart preserves a valid journal byte-for-byte.
    saved = path.read_bytes()
    reconcile_calls(tmp_path, [complete])
    assert path.read_bytes() == saved


def test_completed_episode_with_missing_or_duplicate_calls_is_rejected(tmp_path):
    path = tmp_path / "rollout_calls.jsonl"
    row = {"task_id": 0, "init_index": 10, "control_step": 0, "ms": 4.0}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="missing"):
        reconcile_calls(tmp_path, [{"task_id": 0, "init_index": 10, "control_steps": 2}])
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(RuntimeError, match="Duplicate"):
        reconcile_calls(tmp_path, [{"task_id": 0, "init_index": 10, "control_steps": 1}])


def test_resume_rejects_mixed_pack_and_preserves_prior_peak_and_window(tmp_path):
    from research.holoq_bench.reuse_smoke import CORE

    current = dict(method="holoq", source_pack_sha256="pack-a", observation_file_sha256="observations-a",
                   mode="head_graph", stage="rollout", denoising_steps=8, action_horizon=16,
                   batch=1, warmup=32, noise_seed="fixed", started=10,
                   source_files_sha256={path: "same" for path in CORE + ["research/holoq_bench/deployment.py"]})
    (tmp_path / "metadata.json").write_text(json.dumps(current))
    (tmp_path / "summary.json").write_text(json.dumps(dict(episodes=1, updated=20,
         inference_memory=dict(peak_allocated_bytes=123, peak_reserved_bytes=456))))
    memory, windows = validate_resume_metadata(tmp_path, current)
    assert memory == dict(peak_allocated_bytes=123, peak_reserved_bytes=456)
    assert windows == [dict(started=10, finished=20)]
    with pytest.raises(RuntimeError, match="source_pack_sha256"):
        validate_resume_metadata(tmp_path, dict(current, source_pack_sha256="pack-b"))
    with pytest.raises(RuntimeError, match="observation_file_sha256"):
        validate_resume_metadata(tmp_path, dict(current, observation_file_sha256="observations-b"))
    changed = dict(current, source_files_sha256=dict(current["source_files_sha256"]))
    changed["source_files_sha256"][CORE[0]] = "changed"
    with pytest.raises(RuntimeError, match="math source changed"):
        validate_resume_metadata(tmp_path, changed)


def test_submission_resume_cannot_mix_reference_packed_or_episode_protocol(tmp_path):
    from research.holoq_bench.reuse_smoke import CORE

    extra = ["research/holoq_bench/deployment.py", "research/rebuttal/runtime.py",
             "research/rebuttal/method.py", "gr00t/quantization/dit_step_context.py"]
    current = dict(method="holoq_submission_contract_v2_packed", runtime_implementation="packed",
                   source_pack_sha256="same-frozen-grid", observation_file_sha256="same-observations",
                   mode="head_graph", stage="rollout", denoising_steps=8, action_horizon=16,
                   batch=1, warmup=32, noise_seed="fixed", started=10, suite="libero_object",
                   episode_env_seed=0, init_indices=list(range(20, 30)),
                   evaluation_manifest_sha256="states-frozen", validation_sha256="gate-passed",
                   source_files_sha256={path: "same" for path in CORE + extra})
    (tmp_path / "metadata.json").write_text(json.dumps(current))
    validate_resume_metadata(tmp_path, current)
    for key, value in (("runtime_implementation", "reference"), ("init_indices", list(range(10, 20))),
                       ("episode_env_seed", 1), ("validation_sha256", "different-gate")):
        with pytest.raises(RuntimeError, match="protocol/implementation changed"):
            validate_resume_metadata(tmp_path, dict(current, **{key: value}))
    altered = dict(current, source_files_sha256={**current["source_files_sha256"], "research/rebuttal/runtime.py": "changed"})
    with pytest.raises(RuntimeError, match="math source changed"):
        validate_resume_metadata(tmp_path, altered)


class ToyHead:
    config = SimpleNamespace(action_horizon=2, action_dim=4)

    def get_action(self, backbone, action):
        return {"action_pred": backbone["backbone_features"] * action["gain"]}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA Graph required")
def test_graph_changes_python_constants_without_reusing_stale_values():
    graphs = HeadGraphs(ToyHead())
    x = torch.ones((1, 2, 4), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        a = graphs({"backbone_features": x}, {"gain": 1.0})["action_pred"].clone()
        b = graphs({"backbone_features": x}, {"gain": 2.0})["action_pred"].clone()
    torch.testing.assert_close(a, x)
    torch.testing.assert_close(b, x * 2)
    assert len(graphs.cache) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA Graph required")
def test_graph_copies_changed_tensor_inputs_with_same_shape():
    graphs = HeadGraphs(ToyHead())
    x = torch.ones((1, 2, 4), device="cuda", dtype=torch.float16)
    gain = torch.ones((), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        a = graphs({"backbone_features": x}, {"gain": gain})["action_pred"].clone()
        gain.fill_(3)
        b = graphs({"backbone_features": x * 2}, {"gain": gain})["action_pred"].clone()
    torch.testing.assert_close(a, x)
    torch.testing.assert_close(b, x * 6)
    assert len(graphs.cache) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA Graph required")
def test_graph_preserves_noise_dtype_after_context_promotes_backbone_input():
    class PromotedHead:
        config = SimpleNamespace(action_horizon=2, action_dim=4)

        def get_action(self, backbone, action):
            self._holoq_noise_dtype = torch.float32
            context = backbone["backbone_features"].float()
            external = getattr(self, "_holoq_external_noise", None)
            noise = torch.randn(context.shape, dtype=context.dtype, device=context.device) if external is None else external.clone()
            return {"action_pred": context + noise}

    head = PromotedHead()
    graphs = HeadGraphs(head)
    x = torch.ones((1, 2, 4), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        torch.manual_seed(53)
        actual = graphs({"backbone_features": x}, {})["action_pred"].clone()
        torch.manual_seed(53)
        expected = head.get_action({"backbone_features": x}, {})["action_pred"]
    assert next(iter(graphs.cache.values()))[1].dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
