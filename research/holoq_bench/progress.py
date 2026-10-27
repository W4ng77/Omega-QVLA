"""Read experiment progress without touching the GPU or running model code."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--runs", type=Path, default=Path("results/holoq_object/evaluation_v2"))
    a = p.parse_args()
    queue = json.loads((a.runs / "queue.json").read_text())
    print("common mode:", queue.get("common_mode", "preflight"))
    for case in queue.get("valid_cases", queue.get("cases", {})):
        summary = a.runs / case / queue.get("common_mode", "head_graph") / "rollout/summary.json"
        if summary.exists():
            data = json.loads(summary.read_text())
            print(case, f"{data['successes']}/{data['episodes']}", data.get("latency", {}))
        else:
            states = queue.get("cases", {}).get(case, {})
            print(case, [(name, d.get("returncode")) for name, d in states.items()])
    if queue.get("commands"):
        print("last command:", queue["commands"][-1])


if __name__ == "__main__":
    main()
