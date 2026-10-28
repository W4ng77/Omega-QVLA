"""Shared Object rollout protocol for packed deployments and accuracy diagnostics.

Extracted without changing the original rollout AST; seed/state/action/timing
rules are identical in both callers. No backend/performance policy is selected.
"""
import hashlib
import json
import time
import numpy as np
import torch
from research.holoq_bench.collect_calibration import seed


def run(args, report, graphs, timed, previous_memory, stats, memory, write):
    from tools.analyze_layerwise_quant_drift import ensure_libero_runtime, _convert_libero_observation

    ensure_libero_runtime()
    from libero.libero import benchmark
    from examples.Libero.eval.utils import get_libero_env, get_libero_dummy_action
    from examples.Libero.eval.run_libero_eval import ExternalPolicy, get_max_steps

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    converter = ExternalPolicy.__new__(ExternalPolicy)
    converter.policy_output_format = "gr00t_dict"
    converter.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    episodes_path = args.output / "episodes.jsonl"
    previous = (
        [json.loads(line) for line in episodes_path.read_text().splitlines()] if episodes_path.exists() else []
    )
    complete = {(e["task_id"], e["init_index"]) for e in previous}
    from research.holoq_bench.rollout_resume import reconcile_calls

    reconcile_calls(args.output, previous)
    episodes = list(previous)
    peak_allocated = previous_memory.get("peak_allocated_bytes", 0)
    peak_reserved = previous_memory.get("peak_reserved_bytes", 0)
    report["inference_memory"] = {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }
    for task_id in range(10):
        task = suite.get_task(task_id)
        env, description = get_libero_env(task, resolution=256)
        try:
            states = suite.get_task_init_states(task_id)
            for init in range(10, 10 + args.trials):
                if (task_id, init) in complete:
                    continue
                seed(42 + task_id * 10000 + init * 1000)
                env.reset()
                obs = env.set_init_state(states[init])
                for _ in range(10):
                    obs, _, _, _ = env.step(get_libero_dummy_action())
                torch.cuda.reset_peak_memory_stats()
                latencies = []
                done = False
                with (args.output / "rollout_calls.jsonl").open("a", buffering=1) as call_file:
                    for step in range(get_max_steps("libero_object", "groot")):
                        converted = _convert_libero_observation(obs, task.language)
                        before = len(graphs.cache) if graphs else 0
                        result, latency = timed(converted, 42 + task_id * 10000 + init * 1000 + step)
                        after = len(graphs.cache) if graphs else 0
                        call_file.write(
                            json.dumps(
                                {
                                    "task_id": task_id,
                                    "init_index": init,
                                    "control_step": step,
                                    "ms": latency,
                                    "cold_capture": after != before,
                                }
                            )
                            + "\n"
                        )
                        if before == after:
                            latencies.append(latency)
                        action = converter._convert_to_libero_action(result, 0)
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            break
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                episode = {
                    "task_id": task_id,
                    "task": description,
                    "init_index": init,
                    "success": bool(done),
                    "control_steps": step + 1,
                    "policy_latency": stats(latencies),
                    "initial_state_sha256": hashlib.sha256(np.asarray(states[init]).tobytes()).hexdigest(),
                }
                episodes.append(episode)
                with episodes_path.open("a") as file:
                    file.write(json.dumps(episode) + "\n")
                scores = [sum(e["success"] for e in episodes if e["task_id"] == t) for t in range(10)]
                report.update(
                    episodes=len(episodes),
                    successes=sum(scores),
                    per_task_successes=scores,
                    inference_memory={
                        "peak_allocated_bytes": peak_allocated,
                        "peak_reserved_bytes": peak_reserved,
                    },
                    updated=time.time(),
                )
                write(args.output / "summary.json", report)
                print(
                    "EPISODE",
                    report["method"],
                    args.mode,
                    task_id,
                    init,
                    bool(done),
                    "TOTAL",
                    sum(scores),
                    "/",
                    len(episodes),
                    flush=True,
                )
        finally:
            env.close()
    calls = [json.loads(line) for line in (args.output / "rollout_calls.jsonl").read_text().splitlines()]
    report.update(
        latency=stats([c["ms"] for c in calls if not c["cold_capture"]]),
        cold_capture_calls=sum(c["cold_capture"] for c in calls),
        finished=time.time(),
    )
    write(args.output / "summary.json", report)
    print(
        "ROLLOUT_COMPLETE",
        report["method"],
        args.mode,
        report["successes"],
        report["episodes"],
        report["latency"],
        flush=True,
    )

