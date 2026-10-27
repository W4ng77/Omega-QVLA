"""Compact read-only view of ongoing LIBERO trials and inference timings."""

import json
from pathlib import Path
import statistics
import sys
import time

root = Path(sys.argv[1])
episodes = [
    json.loads(s) for f in (root / "summaries").glob("*_episodes.jsonl") for s in f.read_text().splitlines() if s
]
for task in sorted({x["task_id"] for x in episodes}):
    rows = [x for x in episodes if x["task_id"] == task]
    print(
        f"task={task} episodes={len(rows)}/10 successes={sum(x['success'] for x in rows)} "
        f"errors={sum(x['error'] is not None for x in rows)}"
    )
for f in sorted((root / "policy_latency").glob("*.jsonl")):
    rows = [json.loads(s) for s in f.read_text().splitlines() if s]
    print(
        f"{f.name} calls={len(rows)} mean_ms={statistics.mean(x['policy_ms'] for x in rows):.2f} "
        f"last_write_s={time.time() - f.stat().st_mtime:.1f}"
    )
manifest = json.loads((root / "run_manifest.json").read_text())
print(
    f"total={len(episodes)}/100 elapsed_min={(time.time() - manifest['start_unix']) / 60:.1f} "
    f"exit_code={manifest.get('exit_code', 'running')}"
)
