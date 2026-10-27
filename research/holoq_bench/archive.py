"""Archive small audit artifacts and compressed raw measurements, never packs."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for source in a.runs.rglob("*"):
        if not source.is_file() or source.suffix not in (".json", ".jsonl", ".csv", ".log", ".txt", ".md"):
            continue
        target = a.output / source.relative_to(a.runs)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in (".log", ".jsonl"):
            with source.open("rb") as raw, gzip.open(str(target) + ".gz", "wb") as compressed:
                shutil.copyfileobj(raw, compressed)
        elif source.name == "environment.txt":
            clean = re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", source.read_text())
            target.write_text(clean)
        else:
            shutil.copy2(source, target)
    manifest = {}
    for path in a.output.rglob("*"):
        if path.is_file() and path.name != "archive_manifest.json":
            manifest[str(path.relative_to(a.output))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (a.output / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Archived", len(manifest), "files,", sum(m["bytes"] for m in manifest.values()), "bytes", flush=True)


if __name__ == "__main__":
    main()
