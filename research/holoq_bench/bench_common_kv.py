"""Matched full-policy probe of shared logits/KV optimization; no borrowed score."""
import hashlib
import json
from pathlib import Path
import sys
from research.holoq_bench import bench_common_logits as bench
from research.holoq_bench.common_logits import install as logits
from research.holoq_bench.common_kv import install as kv

configuration = {}


def install(policy):
    a = logits(policy)
    b = kv(policy)
    configuration.update(b)
    a["name"] = b["name"]
    return a


def main():
    bench.install = install
    output = Path(sys.argv[sys.argv.index("--output") + 1]) / "result.json"
    try:
        bench.main()
    finally:
        if output.exists():
            data = json.loads(output.read_text())
            data["optimization_configuration"] = configuration
            data["optimization_source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("common_logits.py", "common_kv.py")}
            data["scope"] += " Cross-K/V results reused only within a policy and only for bitwise-equal per-step scales/converted banks; all8 denoising updates execute. Graphs recaptured."
            output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
