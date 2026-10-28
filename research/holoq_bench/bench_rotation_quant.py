"""Explicit paired complete-policy probe; never installed in baseline runners."""
import hashlib
import json
from pathlib import Path
import sys
from research.holoq_bench import bench_common_logits as bench
from research.holoq_bench.rotation_quant_fusion import install


if __name__ == "__main__":
    bench.install = install
    output = Path(sys.argv[sys.argv.index("--output") + 1]) / "result.json"
    try:
        bench.main()
    finally:
        if output.exists():
            data = json.loads(output.read_text())
            data["source_sha256"] = hashlib.sha256(Path(__file__).with_name("rotation_quant_fusion.py").read_bytes()).hexdigest()
            data["scope"] += " Explicit research prototype; no LIBERO success assigned. Graph cache cleared and all8 steps recaptured after installation. No original-grid code or scale changes."
            output.write_text(json.dumps(data, indent=2) + "\n")
