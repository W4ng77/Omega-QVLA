"""Recover interrupted rollouts without counting incomplete episode fragments.

Original orphaned calls are archived before rewriting the active journal.
Completed episodes must have exactly one record for every executed control step.
"""

import hashlib
import json
from pathlib import Path
import time


def validate_resume_metadata(directory: Path, current):
    """Refuse to mix different packs, inputs, math sources or protocols."""
    path = directory / "metadata.json"
    if not path.exists():
        return {}, []
    old = json.loads(path.read_text())
    identity = ("source_pack_sha256", "observation_file_sha256", "mode", "stage", "denoising_steps",
                "action_horizon", "batch", "warmup", "noise_seed", "common_optimizations")
    for key in identity:
        if old.get(key) != current.get(key):
            raise RuntimeError(f"Resume identity mismatch for {key}; use a new output directory")
    from research.holoq_bench.reuse_smoke import CORE

    checked = list(CORE) + ["research/holoq_bench/deployment.py"]
    if current["method"].endswith("_rotv2"):
        checked.append("research/holoq_bench/fused_rotation_strided.py")
    for key in checked:
        if old["source_files_sha256"].get(key) != current["source_files_sha256"].get(key):
            raise RuntimeError(f"Resume math source changed: {key}; use a new output directory")
    previous = directory / "summary.json"
    summary = json.loads(previous.read_text()) if previous.exists() else {}
    windows = list(old.get("resume_measurement_windows", []))
    if summary.get("episodes", 0):
        end = summary.get("finished") or summary.get("updated")
        if end is None:
            raise RuntimeError("Previous measured episodes lack a timing window")
        windows.append({"started": old["started"], "finished": end})
    # Keep the previous setup record before the normal runner writes metadata.
    directory.joinpath(f"resume_metadata_{time.time_ns()}.json").write_bytes(path.read_bytes())
    return summary.get("inference_memory", {}), windows


def reconcile_calls(directory: Path, episodes):
    path = directory / "rollout_calls.jsonl"
    if not path.exists():
        if episodes:
            raise RuntimeError("Completed episodes exist without their timing calls")
        return
    expected = {(e["task_id"], e["init_index"]): e["control_steps"] for e in episodes}
    if len(expected) != len(episodes):
        raise RuntimeError("Duplicate completed episode records")
    rows, orphaned = [], []
    original = path.read_bytes()
    for line in original.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A process may die midway through an append. Keep the exact raw
            # bytes in the backup, never fabricate a timing/episode completion.
            orphaned.append({"reason": "truncated JSON line"})
            continue
        if (row["task_id"], row["init_index"]) in expected:
            rows.append(row)
        else:
            orphaned.append(row)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["task_id"], row["init_index"]), []).append(row["control_step"])
    for key, count in expected.items():
        if sorted(grouped.get(key, [])) != list(range(count)):
            raise RuntimeError(f"Duplicate or missing timing calls for completed episode {key}")
    if orphaned:
        backup = directory / f"interrupted_calls_{time.time_ns()}.jsonl"
        backup.write_bytes(original)
        temp = path.with_suffix(".recovered.tmp")
        temp.write_text("".join(json.dumps(row) + "\n" for row in rows))
        temp.replace(path)
        note = {"original_journal": str(backup), "original_sha256": hashlib.sha256(original).hexdigest(),
                "orphaned_rows": len(orphaned), "kept_completed_calls": len(rows),
                "reason": "Resume only completed episodes; rerun interrupted episode from its original state/seed"}
        backup.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
