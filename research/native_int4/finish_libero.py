"""After complete trials, verify results and run matched full-policy timings."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.native_int4.run_libero import PYTHON, ROOT


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()
    run = args.run.resolve()
    started = time.time()
    while True:
        manifest = json.loads((run / "run_manifest.json").read_text())
        if "exit_code" in manifest:
            if manifest["exit_code"]:
                raise RuntimeError(f"LIBERO failed with exit {manifest['exit_code']}")
            break
        if time.time() - started > 8 * 3600:
            raise TimeoutError("LIBERO did not finish within 8 hours")
        time.sleep(15)
    subprocess.run([PYTHON, "research/native_int4/summarize_libero.py", str(run)], cwd=ROOT, check=True)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{ROOT}:/home/xinyu/LIBERO",
        PYTHONNOUSERSITE="1",
        CUDA_VISIBLE_DEVICES=args.gpu,
        CUDA_HOME="/usr/local/cuda-12.4",
        TORCH_CUDA_ARCH_LIST="8.9",
        MAX_JOBS="2",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        HF_HUB_DISABLE_XET="1",
        NO_ALBUMENTATIONS_UPDATE="1",
        TOKENIZERS_PARALLELISM="false",
        TF_CPP_MIN_LOG_LEVEL="2",
        TORCH_COMPILE_DISABLE="1",
        TORCHDYNAMO_DISABLE="1",
        TORCH_CUDA_GRAPH_DISABLE="1",
        TORCHINDUCTOR_DISABLE_CUDAGRAPHS="1",
    )
    env["PATH"] = "/home/xinyu/.venv-vllm/bin:/usr/local/cuda-12.4/bin:" + env.get("PATH", "")
    parent = run.parent
    configs = [
        ("converted_paired", parent / "step_weight_blockrot.pt", True),
        ("original_block", parent / "object_block_rotations.pt", False),
        ("original_dense", Path("/home/xinyu/models/omega-qvla-gr00t-w4a4/gr00t_object/quantized.pt"), False),
    ]
    report = {
        "scope": "Full get_action on ten identical rollout observations, 10 measured repeats each; 32 warmups",
        "gpu": args.gpu,
        "runs": {},
    }
    for name, pack, paired in configs:
        output = run / f"replay_{name}.json"
        cmd = [
            PYTHON,
            "-u",
            "research/native_int4/replay_policy.py",
            "--pack",
            str(pack),
            "--engine",
            "native_s4" if paired else "triton",
            "--observations",
            str(run / "policy_latency"),
            "--output",
            str(output),
        ]
        if paired:
            cmd.append("--compare-engines")
        print("Starting timing control:", name, flush=True)
        with (run / f"replay_{name}.log").open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["runs"][name] = {"exit_code": proc.returncode}
        if not proc.returncode:
            data = json.loads(output.read_text())
            report["runs"][name].update(
                overall=data["overall"],
                peak_allocated_gib=data["peak_allocated_gib"],
                active_native_layers=data["active_native_layers"],
            )
            if paired:
                report["runs"][name]["max_action_nmse"] = max(row["action_nmse_max"] for row in data["rows"])
        (run / "timing_controls.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, report["runs"][name], flush=True)
    print("All evaluation and timing controls finished.", flush=True)


if __name__ == "__main__":
    main()
