"""Resume from audited new smoke episodes; never reuse the historical 88/90.

A ten-task smoke already evaluates state 10. Copy those complete episodes only
when frozen operator hashes, pack, seed schedule and held-out state hashes match.
The normal runner then evaluates states 11..19, yielding 100 unique episodes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

CORE = [
    "research/holoq_bench/run_policy.py",
    "research/holoq_bench/packed_runtime.py",
    "research/holoq_bench/graphs.py",
    "research/holoq_bench/kernels.py",
    "research/holoq_bench/cutlass_s4.py",
    "research/holoq_bench/cutlass_bindings.cpp",
    "research/native_int4/backend.py",
    "research/native_int4/native.cpp",
    "research/native_int4/native.cu",
    "gr00t/model/policy.py",
    "gr00t/model/action_head/flow_matching_action_head.py",
    "gr00t/quantization/perstep_backends.py",
]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    a = p.parse_args()
    held = json.loads((a.root / "heldout_init10_19/manifest.json").read_text())
    hashes = {(v["task_id"], v["init_index"]): v["sha256"] for v in held["states"]}
    completed = set()
    while True:
        if not (a.runs / "queue.json").exists():
            time.sleep(5)
            continue
        queue = json.loads((a.runs / "queue.json").read_text())
        mode = queue.get("common_mode")
        if not mode:
            time.sleep(5)
            continue
        for case in queue["valid_cases"]:
            mode = queue.get("selected_modes", {}).get(case, queue.get("common_mode"))
            if case in completed:
                continue
            source = a.runs / case / mode / "rollout_smoke"
            marker = source / "completed.json"
            if not marker.exists() or json.loads(marker.read_text())["returncode"] != 0:
                continue
            dest = source.with_name("rollout")
            if (dest / "metadata.json").exists():
                completed.add(case)
                continue
            d = json.loads((source / "summary.json").read_text())
            assert d.get("finished") and d["episodes"] == 10
            assert (
                d["mode"] == mode
                and d["denoising_steps"] == 8
                and d["action_horizon"] == 16
                and d["batch"] == 1
                and d["warmup"] == 32
            )
            assert d["noise_seed"] == "42 + task_id*10000 + initial_state_index*1000 + control_step"
            manifest = json.loads((a.root / "deployments" / case / "deployment.json").read_text())
            assert d["source_pack_sha256"] == manifest["source_pack_sha256"]
            checked = CORE + (["research/holoq_bench/fused_rotation.py"] if "fusedrot" in case else [])
            if "nunchaku" in case:
                checked += [
                    "research/holoq_bench/nunchaku_runtime.py",
                    "research/holoq_bench/nunchaku_probe.py",
                    "research/native_int4/nunchaku_bindings.cpp",
                    "research/native_int4/nunchaku_gemm.cu",
                ]
            for file in checked:
                assert d["source_files_sha256"][file] == hashlib.sha256(Path(file).read_bytes()).hexdigest(), file
            episodes = [json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines()]
            assert {(e["task_id"], e["init_index"]) for e in episodes} == {(i, 10) for i in range(10)}
            assert all(e["initial_state_sha256"] == hashes[e["task_id"], 10] for e in episodes)
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "episodes.jsonl").exists():
                # Recover a completed preseed after a controller/server restart.
                # Never overwrite an episode or invent a second execution.
                assert (dest / "reused_smoke.json").exists()
                assert (dest / "episodes.jsonl").read_bytes() == (source / "episodes.jsonl").read_bytes()
                assert (dest / "rollout_calls.jsonl").read_bytes() == (source / "rollout_calls.jsonl").read_bytes()
                completed.add(case)
                print("VERIFIED_EXISTING_REUSE", case, flush=True)
                continue
            shutil.copy2(source / "episodes.jsonl", dest / "episodes.jsonl")
            shutil.copy2(source / "rollout_calls.jsonl", dest / "rollout_calls.jsonl")
            note = {
                "source": str(source),
                "reason": "same immutable operators, pack, seeds, held-out initial states and execution mode",
                "episodes": 10,
                "remaining_initial_indices": list(range(11, 20)),
                "source_started": d["started"],
                "source_finished": d["finished"],
                "source_code_commit": d["code_commit"],
                "source_pack_sha256": d["source_pack_sha256"],
                "source_inference_memory": d["inference_memory"],
                "episodes_sha256": hashlib.sha256((source / "episodes.jsonl").read_bytes()).hexdigest(),
            }
            (dest / "reused_smoke.json").write_text(json.dumps(note, indent=2) + "\n")
            completed.add(case)
            print("AUDITED_REUSE", case, "10 smoke episodes; remaining 90", flush=True)
        if queue.get("finished"):
            break
        time.sleep(5)
    # Preserve peaks over both actual measurement phases in resumed cases.
    for case in completed:
        mode = queue.get("selected_modes", {}).get(case, queue.get("common_mode"))
        path = a.runs / case / mode / "rollout/summary.json"
        note_path = path.with_name("reused_smoke.json")
        if not path.exists() or not note_path.exists():
            continue
        d = json.loads(path.read_text())
        note = json.loads(note_path.read_text())
        d["audited_reused_smoke"] = note
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            d["inference_memory"][key] = max(d["inference_memory"][key], note["source_inference_memory"][key])
        path.write_text(json.dumps(d, indent=2) + "\n")


if __name__ == "__main__":
    main()
