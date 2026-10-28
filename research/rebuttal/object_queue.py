"""Run only the frozen Object work package, with fail-closed correctness gates.

Waits for an already-running build using the GPU lock. Does not optimize/tune,
select seeds from scores, or claim the entire rebuttal is complete. All policy
stages are serialized. Long/baseline follow-up remains a separately audited job.
"""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from research.holoq_bench.deployment import sha256
from research.rebuttal.build import json_write

ROOT = Path(__file__).resolve().parents[2]


def monitor(stop, path):
    queries = {"gpus": ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
               "processes": ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name", "--format=csv,noheader,nounits"]}
    with path.open("a", buffering=1) as output:
        while not stop.is_set():
            row = {"time": time.time()}
            for key, cmd in queries.items():
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    row[key] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                except (OSError, subprocess.TimeoutExpired) as error:
                    row[key] = {"error": str(error)}
            output.write(json.dumps(row) + "\n")
            stop.wait(5)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    data = a.data.resolve()
    output = data / "primary_queue"
    output.mkdir(exist_ok=True)
    deployment = data / "deployments/submission_contract_v2"
    heldout = data / "heldout_init20_29"
    validation = data / "validation_v2"
    python = sys.executable
    status = {"queue_pid": os.getpid(), "purpose": "Primary Object accuracy and validated same-version packed replay only; other rebuttal evidence remains pending", "stages": [], "started": time.time(), "source_sha256": sha256(Path(__file__)), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    json_write(output / "status.json", status)
    # Existing build is the only owner at launch. Release before spawning any
    # child because each model/renderer runner owns this same exclusive lock.
    with open("/tmp/holoq-vla-policy-timing.lock", "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        assert (deployment / "deployment.json").exists(), "Build did not complete; preserve logs and diagnose"
    stop = threading.Event()
    sampler = threading.Thread(target=monitor, args=(stop, output / "gpu_samples.jsonl"), daemon=True)
    sampler.start()

    def run(name, command, extra_env=None):
        metadata = {"name": name, "command": command, "env_delta": extra_env or {}, "started": time.time()}
        status["stages"].append(metadata)
        status["current"] = name
        json_write(output / "status.json", status)
        with (output / (name + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **(extra_env or {})}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            metadata["pid"] = child.pid
            json_write(output / "status.json", status)
            metadata["returncode"] = child.wait()
        metadata["finished"] = time.time()
        json_write(output / "status.json", status)
        assert metadata["returncode"] == 0, f"Stage {name} failed; no later scored stage is authorized by this gate"

    def completed(directory, implementation=None):
        path = directory / "summary.json"
        if not path.exists():
            return False
        summary = json.loads(path.read_text())
        if not summary.get("finished"):
            return False
        assert summary["episodes"] == 100 and summary["init_indices"] == list(range(20, 30))
        assert summary["observation_file_sha256"] == sha256(heldout / "observations.pt")
        assert summary["evaluation_manifest_sha256"] == sha256(heldout / "manifest.json")
        assert summary["mode"] == "head_graph" and summary["denoising_steps"] == 8
        if implementation is not None:
            assert summary["source_pack_sha256"] == sha256(deployment / "packed.pt")
            assert summary["runtime_implementation"] == implementation
            assert summary["validation_sha256"] == sha256(validation / "summary.json")
        for name, digest in summary["source_files_sha256"].items():
            # Execution identities only; unused collectors/builders do not
            # change an already frozen model's inference arithmetic.
            if name.startswith("research/rebuttal/") and name not in ("research/rebuttal/runtime.py", "research/rebuttal/method.py"):
                continue
            assert (ROOT / name).is_file() and sha256(ROOT / name) == digest, (name, "old result source changed")
        return True

    try:
        if not (heldout / "manifest.json").exists():
            run("prepare_states", [python, "-u", "-m", "research.rebuttal.prepare_evaluation", "--suite", "object", "--init-start", "20", "--trials", "10", "--calibration", str(data / "source_teacher_trajectories"), "--output", str(heldout)])
        if not (validation / "summary.json").exists():
            run("validate_frozen_policy", [python, "-u", "-m", "research.rebuttal.validate_policy", "--deployment", str(deployment), "--observations", str(heldout / "observations.pt"), "--output", str(validation)])
        gate = json.loads((validation / "summary.json").read_text())
        assert gate["passed"] and gate["pack_sha256"] == sha256(deployment / "packed.pt")
        assert gate["observation_sha256"] == sha256(heldout / "observations.pt")
        common = ["--observations", str(heldout / "observations.pt"), "--init-start", "20", "--trials", "10", "--suite", "libero_object", "--episode-env-seed", "0", "--evaluation-manifest", str(heldout / "manifest.json")]
        historical = data / "primary/published_legacy/head_graph/rollout"
        if not completed(historical):
            run("historical_100", [python, "-u", "-m", "research.holoq_bench.accuracy_policy", "--kind", "published_legacy", "--source-root", str(data / "source_previous_results"), "--output", str(historical), *common])
        for implementation in ("reference", "packed"):
            for stage in ("smoke", "rollout"):
                directory = data / "primary" / implementation / "head_graph" / stage
                if stage == "rollout" and completed(directory, implementation):
                    continue
                run(implementation + "_" + stage, [python, "-u", "-m", "research.holoq_bench.run_policy", "--deployment", str(deployment), "--mode", "head_graph", "--stage", stage, "--common-logits", "--validation", str(validation / "summary.json"), "--output", str(directory), *common], {"HOLOQ_REBUTTAL_IMPLEMENTATION": implementation})
        replay = data / "primary/packed/head_graph/replay"
        run("packed_replay", [python, "-u", "-m", "research.holoq_bench.run_policy", "--deployment", str(deployment), "--mode", "head_graph", "--stage", "replay", "--common-logits", "--validation", str(validation / "summary.json"), "--output", str(replay), *common], {"HOLOQ_REBUTTAL_IMPLEMENTATION": "packed"})
        status["primary_work_package_finished"] = time.time()
        status["rebuttal_overall_complete"] = False
        status["remaining"] = ["Interpret primary accuracy differences; no score-driven tuning", "Actual quantized-baseline table with disclosed calibration/scope differences", "Full-policy cumulative profiler and scope/memory/storage breakdown", "Long per-step controlled ablation after accuracy freeze", "Reviewer-specific final evidence review"]
    finally:
        stop.set()
        sampler.join(timeout=25)
        json_write(output / "status.json", status)
        archive = ROOT / "experiment_results/rebuttal_20261028/primary"
        archive.mkdir(parents=True, exist_ok=True)
        for relative in ("primary_queue", "validation_v2", "heldout_init20_29", "primary"):
            for path in (data / relative).rglob("*"):
                if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
                    target = archive / path.relative_to(data)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix == ".json":
                        target.write_bytes(path.read_bytes())
                    else:
                        with gzip.open(str(target) + ".gz", "wb") as file:
                            file.write(path.read_bytes())


if __name__ == "__main__":
    main()
