"""Run only the two necessary full-task accuracy references; no GEMM benchmarks."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path("results/holoq_object")
    out = root / "accuracy_controls"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "queue.json"
    status = json.loads(path.read_text()) if path.exists() else {"started": time.time(), "commands": [], "cases": {}}

    def save():
        temp = out / "queue.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(path)

    save()
    while not (root / "released_grid_build/finished.json").exists():
        time.sleep(10)
    for kind in ("published_legacy", "fp16_reference"):
        directory = out / kind
        directory.mkdir(exist_ok=True)
        marker = directory / "completed.json"
        if marker.exists():
            assert json.loads(marker.read_text())["returncode"] == 0
            continue
        cmd = [sys.executable, "-u", "-m", "research.holoq_bench.accuracy_policy", "--kind", kind, "--output", str(directory), "--trials", "10"]
        status["commands"].append({"cmd": cmd, "started": time.time(), "cwd": str(Path.cwd())})
        save()
        print("ACCURACY_REFERENCE_START", kind, flush=True)
        with (directory / "run.log").open("a") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        record = {"returncode": result.returncode, "finished": time.time()}
        marker.write_text(json.dumps(record, indent=2) + "\n")
        status["cases"][kind] = record
        save()
        assert result.returncode == 0, kind
        print("ACCURACY_REFERENCE_FINISH", kind, flush=True)
    status["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
