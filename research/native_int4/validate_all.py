"""Validate the native integer core on every real layer, step and token count."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.quantization.perstep_backends import PerStepBackend, transform_input  # noqa: E402
from research.native_int4.backend import extension, from_packed, pack_activation, pack_weight  # noqa: E402


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    extension()
    base = Path("/home/xinyu/omegaqvla-perstep-4080/results")
    source = torch.load(
        "/home/xinyu/models/omega-qvla-gr00t-w4a4/gr00t_object/quantized.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    captures = torch.load(
        base / "perstep_full_object_20261027_captures/heldout.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )["inputs"]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "rows": 0,
        "max_nmse_vs_existing": 0.0,
        "sk1_exact_value_matches": 0,
        "sk1_checks": 0,
        "modes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for mode in ("step_weight", "group_int"):
            pack = torch.load(
                base / f"perstep_full_object_20261027/{mode}.pt", map_location="cpu", mmap=True, weights_only=False
            )
            count = 0
            maximum = 0.0
            for name, record in pack.items():
                if not isinstance(record, dict) or "perstep_backend" not in record:
                    continue
                b = PerStepBackend(record["perstep_backend"]).cuda()
                n, k, g = b.out_features, b.in_features, b.group_size
                for step in range(b.act_scale_table.shape[0]):
                    p, s, a = b.selected(step)
                    pb = pack_weight(p, n, k)
                    st = s.t().contiguous()
                    for m in (32, 128):
                        x = transform_input(captures[name][step][:m].cuda(), source[name]).contiguous()
                        oracle = b(x, step)
                        pa = pack_activation(x, a, g, mode == "group_int")
                        for sk in (1, 4):
                            y = from_packed(pa, pb, st, a, m, n, k, g, 2 if mode == "group_int" else 1, 4, sk)
                            err = ((y - oracle).square().sum() / oracle.square().sum().clamp_min(1e-30)).item()
                            if not torch.isfinite(y).all() or err > 1e-6:
                                raise RuntimeError(f"Mismatch {mode}/{name}/{step}/{m}/{sk}: {err}")
                            if sk == 1:
                                report["sk1_checks"] += 1
                                report["sk1_exact_value_matches"] += int(torch.equal(y, oracle))
                            maximum = max(maximum, err)
                            count += 1
                print(mode, name, "validated", count, flush=True)
                report["modes"][mode] = {"checks": count, "max_nmse_vs_existing": maximum}
                report["rows"] = sum(v["checks"] for v in report["modes"].values())
                report["max_nmse_vs_existing"] = max(v["max_nmse_vs_existing"] for v in report["modes"].values())
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
