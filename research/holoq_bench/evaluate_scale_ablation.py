"""Freeze the single-factor LLM activation-scale ablation before reading scores."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    root = Path("results/holoq_object")
    out = root / "evaluation_scale_ablation"
    out.mkdir(parents=True, exist_ok=True)
    source = root / "deployments/holoq_release_grid/packed.pt"
    case = "holoq_release_token_ablation"
    deployment = root / "deployments" / case
    status = {"started": time.time(), "case": case, "commands": [], "scope": "One factor only: LLM activation scale granularity; not fastest-backend selection"}
    status_path = out / "queue.json"

    def save():
        temp = out / "queue.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)

    save()
    guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX)
    try:
        if not deployment.exists():
            import torch
            from research.holoq_bench.deployment import export

            torch.set_num_threads(1)
            pack = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
            count = 0
            for name, record in pack["layers"].items():
                if "language_model" in name:
                    assert record["act_mode"] == "channel"
                    record["act_mode"] = "token"
                    count += 1
            assert count == 84
            pack["meta"].update(method=case, runtime_plugin="released_token_ablation", source_grid="holoq_release_grid", activation_change="ONLY84 LLM layers: frozen per-column to dynamic per-token A4. Same[-8,7] FP16 rounding, all W/R, original8 DiT scale rows and same packed kernel.", role="Causal accuracy ablation; not the released method")
            target = out / (case + ".pt")
            assert not target.exists()
            torch.save(pack, target)
            export(target, root / "teacher_capture/fp16_model_state.pt", deployment)
    finally:
        guard.close()

    def command(cmd, directory, own_lock=False):
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "completed.json"
        if marker.exists():
            assert json.loads(marker.read_text())["returncode"] == 0
            return
        status["commands"].append({"cmd": cmd, "started": time.time()})
        save()
        guard = None
        if own_lock:
            guard = open("/tmp/holoq-vla-policy-timing.lock", "a")
            fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            with (directory / "run.log").open("a") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        finally:
            if guard:
                guard.close()
        marker.write_text(json.dumps({"returncode": result.returncode, "finished": time.time()}) + "\n")
        assert result.returncode == 0, directory

    diagnostic = out / "canonical_math"
    command([sys.executable, "-u", "-m", "research.holoq_bench.check_canonical_math", "--deployment", str(deployment), "--cache", str(root / "released_grid_build/layer_cache"), "--output", str(diagnostic / "result.json")], diagnostic, own_lock=True)
    assert not json.loads((diagnostic / "result.json").read_text())["review_required"]
    shutil.copy2(root / "evaluation_v3/environment.txt", out / "environment.txt")
    for stage in ("smoke", "rollout", "replay"):
        directory = out / case / "head_graph" / stage
        command([sys.executable, "-u", "-m", "research.holoq_bench.run_policy", "--deployment", str(deployment), "--observations", str(root / "heldout_init10_19/observations.pt"), "--output", str(directory), "--mode", "head_graph", "--stage", stage, "--trials", "10", "--common-logits"], directory)
    status["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
