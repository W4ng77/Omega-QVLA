"""Explicit v2 regression recovery, correctness-gated pilots and Object100.

Never reuse legacy fused measurements/episodes. Preserve every prior artifact.
Hold the timing lock during heavy export/test work as well as policy execution.
"""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from research.holoq_bench.deployment import export


BASES = {
    "holoq": ["holoq", "holoq_llm_s4"],
    "holoq_step_weight": ["holoq_step_weight", "holoq_step_weight_llm_s4"],
    "quantvla_a8": ["quantvla_selective_a8_atm_g128"],
    "quantvla_a4": ["quantvla_selective_a4_atm_g128"],
    "duquant_a8": ["duquant_a8_g128"],
    "duquant_a4": ["duquant_a4_g128"],
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    args = parser.parse_args()
    root = args.root
    out = root / "evaluation_rotation_v2"
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "queue.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        "started": time.time(), "common_mode": "head_graph", "commands": [],
        "cases": {}, "valid_cases": [], "candidate_cases": [],
        "phase": "correctness_gated_v2", "supersedes": "evaluation_best_s4 legacy fused variants only",
    }

    def save():
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)

    save()
    shutil.copy2(root / "evaluation_v3/environment.txt", out / "environment.txt")
    stop = threading.Event()

    def monitor():
        with (out / "gpu_monitor.jsonl").open("a", buffering=1) as file:
            while not stop.is_set():
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
                file.write(json.dumps({"time": time.time(), "gpus": result.stdout.strip().splitlines()}) + "\n")
                stop.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def command(cmd, directory):
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "completed.json"
        if marker.exists():
            return json.loads(marker.read_text())["returncode"] == 0
        status["commands"].append({"cmd": cmd, "started": time.time(), "cwd": str(Path.cwd())})
        save()
        print("RECOVERY_START", directory, flush=True)
        with (directory / "run.log").open("a") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        marker.write_text(json.dumps({"returncode": result.returncode, "finished": time.time()}, indent=2) + "\n")
        print("RECOVERY_FINISH", directory, result.returncode, flush=True)
        return result.returncode == 0

    def policy(case, mode, stage, trials=10):
        directory = out / case / mode / stage
        ok = command([sys.executable, "-u", "-m", "research.holoq_bench.run_policy",
                      "--deployment", str(root / "deployments" / case), "--observations",
                      str(root / "heldout_init10_19/observations.pt"), "--output", str(directory),
                      "--mode", mode, "--stage", "rollout" if stage == "rollout_smoke" else stage,
                      "--trials", str(trials)], directory)
        status["cases"].setdefault(case, {})[mode + "_" + stage] = json.loads((directory / "completed.json").read_text())
        save()
        return ok

    try:
        # Unit tests include the exact transpose stride absent from the old suite.
        test = out / "strided_tests"
        if not (test / "completed.json").exists():
            guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
            fcntl.flock(guard, fcntl.LOCK_EX)
            try:
                assert command([sys.executable, "-m", "pytest", "-q",
                                "research/holoq_bench/test_fused_rotation_strided.py"], test)
            finally:
                guard.close()
        else:
            assert json.loads((test / "completed.json").read_text())["returncode"] == 0
        selection = {"frozen_by": "latency only, after real-input functional checks; legacy fused rows excluded",
                     "families": {}, "regression": str(root / "fused_rotation_regression.json")}
        for family, bases in BASES.items():
            choices = []
            for base in bases:
                # The reference is always an immutable, nonfused packed bundle.
                check = out / "functional_checks" / base
                oracle = check / "result.json"
                ok = command([sys.executable, "-u", "-m", "research.holoq_bench.check_rotation_policy",
                              "--deployment", str(root / "deployments" / base), "--observations",
                              str(root / "heldout_init10_19/observations.pt"), "--output", str(oracle)], check)
                if not ok or not json.loads(oracle.read_text()).get("passed"):
                    print("FUNCTIONAL_REJECT", base, flush=True)
                    continue
                case = base + "_rotv2"
                pack_path = root / "packs_v7" / (case + ".pt")
                directory = root / "deployments" / case
                if not directory.exists():
                    # Avoid heavy CPU/disk work during another policy's timer.
                    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
                    fcntl.flock(guard, fcntl.LOCK_EX)
                    try:
                        import torch

                        torch.set_num_threads(1)
                        source = root / "deployments" / base / "packed.pt"
                        pack = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
                        assert pack["meta"].get("rotation_plugin") is None
                        pack["meta"].update(method=case, rotation_plugin="triton_block64_strided_v2",
                                            rotation_source_pack=str(source.resolve()),
                                            rotation_conversion="stride-aware layout fusion only; unchanged codes, scales, R, bias and per-step behavior")
                        pack_path.parent.mkdir(parents=True, exist_ok=True)
                        if not pack_path.exists():
                            torch.save(pack, pack_path)
                        export(pack_path, root / "teacher_capture/fp16_model_state.pt", directory)
                        del pack
                    finally:
                        guard.close()
                shutil.copy2(oracle, directory / "functional_validation.json")
                if case not in status["candidate_cases"]:
                    status["candidate_cases"].append(case)
                if case not in status["valid_cases"]:
                    status["valid_cases"].append(case)
                save()
                for mode in ("eager", "head_graph"):
                    smoke_ok = policy(case, mode, "smoke")
                    if smoke_ok:
                        smoke_dir = out / case / mode / "smoke"
                        assert "EXPLICIT_STRIDED_BLOCK64_V2" in (smoke_dir / "run.log").read_text()
                        smoke_meta = json.loads((smoke_dir / "metadata.json").read_text())
                        assert smoke_meta["source_files_sha256"]["research/holoq_bench/fused_rotation_strided.py"] == json.loads(oracle.read_text())["plugin_source_sha256"]
                    if smoke_ok and policy(case, mode, "replay"):
                        path = out / case / mode / "replay/summary.json"
                        data = json.loads(path.read_text())
                        choices.append({"case": case, "mode": mode, "root": str(out),
                                        "mean_ms": data["latency"]["mean_ms"], "replay_summary": str(path),
                                        "source_pack_sha256": data["source_pack_sha256"], "code_commit": data["code_commit"]})
                # Correct nonfused alternatives remain eligible, independently
                # of task outcomes. Legacy fused candidates never reenter.
                for prior in (root / "evaluation_v3", root / "evaluation_best_s4"):
                    for mode in ("eager", "head_graph"):
                        path = prior / base / mode / "replay/summary.json"
                        if not path.exists():
                            continue
                        data = json.loads(path.read_text())
                        choices.append({"case": base, "mode": mode, "root": str(prior),
                                        "mean_ms": data["latency"]["mean_ms"], "replay_summary": str(path),
                                        "source_pack_sha256": data["source_pack_sha256"], "code_commit": data["code_commit"]})
            assert choices, (family, "no functional candidate")
            winner = min(choices, key=lambda row: row["mean_ms"])
            selection["families"][family] = {"selected": winner, "candidates": choices, "frozen_at": time.time()}
            (out / "backend_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
            print("FUNCTIONALLY_VALID_WINNER", family, winner, flush=True)
            case = winner["case"]
            prior_root = Path(winner["root"])
            if prior_root != out:
                # Preserve prior raw data byte-for-byte; don't copy a score to a
                # changed kernel. A nonfused winner is literally that same case.
                if not (out / case).exists():
                    shutil.copytree(prior_root / case, out / case)
                if case not in status["valid_cases"]:
                    status["valid_cases"].append(case)
                save()
            for mode in dict.fromkeys(["head_graph", winner["mode"]]):
                if policy(case, mode, "rollout_smoke", 1):
                    # A successful process alone does not establish correctness.
                    # Functional check above is mandatory; retain even a poor
                    # legitimate task score, never tune on the held-out score.
                    policy(case, mode, "rollout", 10)
            status["completed_families"] = list(selection["families"])
            save()
        # SmoothQuant-A4's rotation-free QServe choice is unaffected by the bug.
        old = json.loads((root / "evaluation_best_s4/backend_selection.json").read_text())
        selection["families"]["smoothquant_a4"] = old["families"]["smoothquant_a4"]
        selection["frozen_at"] = time.time()
        (out / "backend_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        status.update(phase="complete", finished=time.time(), backend_selection=selection)
        save()
    finally:
        stop.set()
        thread.join(timeout=10)
    print("ROTATION_RECOVERY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
