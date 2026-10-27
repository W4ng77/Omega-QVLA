"""Real-input research comparison: cached baseline, original kernels, native S4.

All operator timings include runtime activation quantization, packing and split
reduction. Weight reordering is offline. No full-policy speedup is inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl
import triton.testing
from triton.language.extra.cuda import libdevice

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gr00t.quantization.perstep_backends import (  # noqa: E402
    PerStepBackend,
    dequant_weight,
    record_weight,
    transform_input,
    unpack_int4,
)
from research.native_int4.backend import extension, from_packed, linear, pack_activation, pack_weight  # noqa: E402


def nmse(y, ref):
    return ((y.float() - ref.float()).square().sum() / ref.float().square().sum().clamp_min(1e-30)).item()


def graph_us(fn, rep=15):
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        result = triton.testing.do_bench_cudagraph(fn, rep=rep) * 1000
    current.wait_stream(stream)
    return result


def cold_us(fn, repeats=15):
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    values = []
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _output = fn()
        for _ in range(repeats):
            flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) * 1000)
    current.wait_stream(stream)
    return sorted(values)[len(values) // 2]


@triton.jit
def _prepare_marlin(
    X, A, Y, K: tl.constexpr, TOTAL: tl.constexpr, G: tl.constexpr, MODE: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    k = i % K
    if MODE == 2:
        ai = k // G
    else:
        ai = k
    a = tl.load(A + ai, i < TOTAL, 1)
    x = tl.load(X + i, i < TOTAL, 0).to(tl.float32)
    q = tl.minimum(tl.maximum(libdevice.nearbyint(tl.div_rn(x, a)), -8.0), 7.0)
    if MODE != 1:
        q *= a
    tl.store(Y + i, q, i < TOTAL)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", default="/home/xinyu/models/omega-qvla-gr00t-w4a4/gr00t_object/quantized.pt")
    parser.add_argument(
        "--exports",
        type=Path,
        default=Path("/home/xinyu/omegaqvla-perstep-4080/results/perstep_full_object_20261027"),
    )
    parser.add_argument(
        "--captures",
        default="/home/xinyu/omegaqvla-perstep-4080/results/perstep_full_object_20261027_captures/heldout.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 49, 128])
    parser.add_argument("--marlin", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    native = extension()
    if args.marlin:
        sys.path.insert(0, str(ROOT / "research/marlin"))
        import marlin
        from types import SimpleNamespace
    source = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    inputs = torch.load(args.captures, map_location="cpu", weights_only=False, mmap=True)["inputs"]
    names = [
        "action_head.model.transformer_blocks.0.attn1.to_q",
        "action_head.model.transformer_blocks.0.ff.net.0.proj",
        "action_head.model.transformer_blocks.0.ff.net.2",
    ]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "native_extension": native.__file__,
        "scope": "residual-only; step 0; three real-input layers",
        "timing": "warm matched CUDA Graph and L2-flushed single replay; includes activation preparation/reduction",
        "native_contract": "Same converted INT4 codes and FP32 scales; no new requantization",
        "marlin_contract": "A4 numerical codes in FP16 operands; FP16 weight scales; not S4xS4 MMA",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in ("step_weight", "group_int"):
        mode_id = 1 if mode == "step_weight" else 2
        converted = torch.load(args.exports / f"{mode}.pt", map_location="cpu", weights_only=False, mmap=True)
        for name in names:
            rec = source[name]
            b = PerStepBackend(converted[name]["perstep_backend"]).cuda()
            p, scales, act = b.selected(0)
            n, k, g = b.out_features, b.in_features, b.group_size
            bp = pack_weight(p, n, k)
            st = scales.t().contiguous()
            original_w = record_weight(rec).bfloat16().cuda()
            if args.marlin:
                sw16 = scales.half()
                if (sw16 <= 0).any():
                    raise ValueError(f"FP16 scales underflow: {name}")
                w16 = dequant_weight(p, scales, k, g).half()
                recovered = (w16 / sw16.repeat_interleave(g, dim=1)[:, :k]).round().clamp(-8, 7).to(torch.int8)
                if not torch.equal(recovered, unpack_int4(p, k)):
                    raise ValueError("Marlin packing would change INT4 codes")
                marlin_layer = marlin.Layer(k, n, g).cuda()
                marlin_layer.pack(SimpleNamespace(weight=w16), sw16)
            for m in args.tokens:
                x = transform_input(inputs[name][0][:m].cuda(), rec).contiguous()
                if x.shape[0] != m:
                    raise ValueError("Insufficient real capture tokens")
                old_scale = rec["act_scale_table"][0].cuda().to(x.dtype)

                def legacy():
                    xq = (x / old_scale).round().clamp(-8, 7) * old_scale
                    return torch.nn.functional.linear(xq.bfloat16(), original_w).to(x.dtype)

                oracle = b(x, 0, engine="torch_reference")
                a = pack_activation(x, act, g, mode_id == 2)
                row = {
                    "mode": mode,
                    "layer": name,
                    "M": m,
                    "N": n,
                    "K": k,
                    "input_dtype": str(x.dtype),
                    "legacy_cached_us": graph_us(legacy),
                    "original_kernel_us": graph_us(lambda: b(x, 0)),
                    "original_nmse_vs_legacy": nmse(oracle.to(x.dtype), legacy()),
                    "trials": [],
                }
                for wn in (1, 2, 4, 8):
                    for sk in (1, 2, 4, 8):
                        if sk > scales.shape[1]:
                            continue

                        def call():
                            return linear(x, bp, st, act, n, g, mode_id, wn, sk)

                        y = call()
                        err = nmse(y, oracle)
                        if not torch.isfinite(y).all() or err > 1e-6:
                            raise RuntimeError(f"Native mismatch: {mode}/{name}/{m}/{wn}/{sk}: {err}")
                        row["trials"].append(
                            {"warps_n": wn, "split_k": sk, "us": graph_us(call), "nmse_vs_oracle": err}
                        )
                best = min(row["trials"], key=lambda t: t["us"])
                row["best_native"] = best
                wn, sk = best["warps_n"], best["split_k"]

                def best_call():
                    return linear(x, bp, st, act, n, g, mode_id, wn, sk)

                row["native_dot_only_us"] = graph_us(
                    lambda: from_packed(a, bp, st, act, m, n, k, g, mode_id, wn, sk)
                )
                row["native_total_us_recheck"] = graph_us(best_call, rep=30)
                row["native_cold_us"] = cold_us(best_call)
                row["legacy_cold_us"] = cold_us(legacy)
                row["warm_speedup_vs_cached_legacy"] = row["legacy_cached_us"] / row["native_total_us_recheck"]
                row["cold_speedup_vs_cached_legacy"] = row["legacy_cold_us"] / row["native_cold_us"]
                if args.marlin:
                    ah = torch.empty((m, k), device="cuda", dtype=torch.float16)

                    def marlin_call():
                        _prepare_marlin[(triton.cdiv(m * k, 256),)](x, act, ah, k, m * k, g, mode_id, 256)
                        return marlin_layer(ah)

                    my = marlin_call()
                    row["marlin_us"] = graph_us(marlin_call)
                    row["marlin_nmse_vs_oracle"] = nmse(my, oracle)
                    row["marlin_cold_us"] = cold_us(marlin_call)
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    mode,
                    name.rsplit(".", 3)[-3:],
                    m,
                    "legacy",
                    round(row["legacy_cached_us"], 2),
                    "old",
                    round(row["original_kernel_us"], 2),
                    "native",
                    round(row["native_total_us_recheck"], 2),
                    "cold_gain",
                    round(row["cold_speedup_vs_cached_legacy"], 2),
                    "cfg",
                    (wn, sk),
                    "Marlin",
                    round(row.get("marlin_us", 0), 2),
                    flush=True,
                )


if __name__ == "__main__":
    main()
