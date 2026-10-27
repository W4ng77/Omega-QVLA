"""Serial single-GPU queue with preflight, common modes and resumable rollout.

Never run another GPU workload while this queue is timing. Failures are saved
per case and do not suppress the remaining required baselines.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

CASES = [
    "holoq",
    "holoq_step_weight",
    "awq_qwen_selective",
    "smoothquant_a8",
    "quantvla_selective_a8_atm_g128",
    "gptq_selective",
    "omniquant_selective",
    "duquant_a8_g128",
    "smoothquant_a4",
    "quantvla_selective_a4_atm_g128",
    "duquant_a4_g128",
    "awq_qwen_full",
    "gptq_full",
]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("results/holoq_object"))
    p.add_argument("--cases", nargs="+", default=CASES)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    status = {"started": time.time(), "cases": {}, "commands": []}
    (args.output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )

    def save():
        path = args.output / "queue.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(path)

    stop = threading.Event()

    def monitor():
        with (args.output / "gpu_monitor.jsonl").open("a", buffering=1) as file:
            while not stop.is_set():
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                file.write(json.dumps({"time": time.time(), "gpus": query.stdout.strip().splitlines()}) + "\n")
                stop.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def job(case, mode, stage, trials=10):
        out = args.output / case / mode / stage
        summary = out / ("metadata.json" if stage == "smoke" else "summary.json")
        done = out / "completed.json"
        if done.exists():
            return json.loads(done.read_text())["returncode"] == 0
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "research.holoq_bench.run_policy",
            "--deployment",
            str(args.root / "deployments" / case),
            "--observations",
            str(args.root / "heldout_init10_19/observations.pt"),
            "--output",
            str(out),
            "--mode",
            mode,
            "--stage",
            "rollout" if stage == "rollout_smoke" else stage,
            "--trials",
            str(trials),
        ]
        status["commands"].append({"cmd": cmd, "cwd": str(Path.cwd()), "started": time.time()})
        save()
        print("START", case, mode, stage, "trials", trials, flush=True)
        with (out / "run.log").open("a") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {"returncode": result.returncode, "finished": time.time(), "summary": str(summary)}
        done.write_text(json.dumps(record, indent=2) + "\n")
        status["cases"].setdefault(case, {})[mode + "_" + stage] = record
        save()
        print("FINISH", case, mode, stage, result.returncode, flush=True)
        return result.returncode == 0

    try:
        valid = []
        graphs = []
        # All-method correctness checks occur before any performance comparison.
        for case in args.cases:
            if not (args.root / "deployments" / case / "deployment.json").exists():
                status["cases"][case] = {"error": "missing complete deployment artifact"}
                save()
                continue
            if job(case, "eager", "smoke"):
                valid.append(case)
                if job(case, "head_graph", "smoke"):
                    graphs.append(case)
        common = "head_graph" if set(graphs) == set(valid) else "eager"
        status.update(common_mode=common, valid_cases=valid, graph_supported=graphs)
        save()
        if args.preflight_only:
            status["preflight_finished"] = time.time()
            save()
            return
        for case in valid:
            job(case, "eager", "replay")
            if case in graphs:
                job(case, "head_graph", "replay")
        # One episode per task first; every final case uses exactly the same
        # held-out 10..19 states and reset noise schedule.
        smoke_ok = []
        for case in valid:
            if job(case, common, "rollout_smoke", 1):
                smoke_ok.append(case)
        for case in smoke_ok:
            job(case, common, "rollout", args.trials)
        if common == "eager":
            for case in graphs:
                # Graph score is independently evaluated, never copied from eager.
                job(case, "head_graph", "rollout", args.trials)
        status["finished"] = time.time()
        save()
    finally:
        stop.set()
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
