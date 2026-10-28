"""Profile frozen paper original/conversion selections, without mixing timings."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from research.holoq_bench.analyze_policy_trace import sqlite_events


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def publish(source, destination, case):
    destination.mkdir(parents=True, exist_ok=True)
    raw = {}
    for file in sorted(source.rglob("*")):
        if not file.is_file():
            continue
        relative = file.relative_to(source)
        if file.suffix in (".sqlite", ".nsys-rep"):
            raw[str(relative)] = {"local_path": str(file.resolve()), "bytes": file.stat().st_size, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
            continue
        if file.suffix not in (".json", ".csv", ".txt", ".log"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if file.name == "policy_trace.json" or file.suffix == ".log":
            with file.open("rb") as src, gzip.open(str(target) + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy2(file, target)
    for mode in ("node", "graph"):
        with gzip.open(destination / ("nsys_" + mode + "_activities.json.gz"), "wt") as file:
            json.dump({"traceEvents": sqlite_events(source / ("nsys_" + mode + ".sqlite")), "scope": "Only NVTX/CUDA activity; environment tables omitted"}, file)
    (destination / "raw_artifacts.json").write_text(json.dumps(raw, indent=2) + "\n")
    graph = read_json(source / "torch_head_graph/analysis/breakdown.json")
    node = read_json(source / "nsys_node_analysis/breakdown.json")
    coarse = read_json(source / "nsys_graph_analysis/breakdown.json")
    meta = read_json(source / "torch_head_graph/metadata.json")
    rows = graph["phase_operation"]
    lines = ["# 完整 policy 实测剖析", "", f"目标：`{case}`。这是论文输入侧旋转配置的独立 trace；与旧双侧旋转对照分别保存。", "", "32 次 warmup 后采集 5 次完整 get_action，含 8 个去噪 step。部署延迟另用无 profiler 的 300 次重放统计；下面累计 GPU 工作时间来自 profiler，不能当作正式延迟。", "", "| 阶段 | GPU 总工作 ms/调用 | GEMM | 旋转 | 量化 | 拷贝/dtype |", "|---|---:|---:|---:|---:|---:|"]
    if "step_weight" in case:
        lines[2] += " 此 case 是独立 step_weight 转换：DiT 每步吸收 A scale 后重新量化 W 为 G128[-8,7]，存储9组权重，不是原方法的严格等价执行。"
    for phase in ("vision", "language", "action_head", "prepare_input", "backbone_other", "other_policy"):
        values = {row["operation"]: row["gpu_ms_per_call"] for row in rows if row["phase"] == phase}
        if values:
            gemm = values.get("gemm", 0) + values.get("gemm_scale_epilogue", 0)
            copy = sum(values.get(key, 0) for key in ("copy_cast_kernel", "gpu_memcpy", "gpu_memset"))
            lines.append(f"| {phase} | {sum(values.values()):.3f} | {gemm:.3f} | {values.get('rotation', 0):.3f} | {values.get('quantization_pack', 0):.3f} | {copy:.3f} |")
    mean = graph["means"]
    lines += ["", f"PyTorch profiled wall {mean['profiled_wall_ms']:.3f} ms；GPU 工作累计 {mean['gpu_activity_sum_ms']:.3f} ms；GPU busy union {mean['gpu_busy_union_ms']:.3f} ms；重叠 excess {mean['gpu_overlap_excess_ms']:.6f} ms。", f"Nsight node：{node['means']['gpu_activity_sum_ms']:.3f} ms GPU 工作，GPU coverage complete={node['gpu_coverage_complete']}。Nsight coarse Graph coverage complete={coarse['gpu_coverage_complete']}；Graph body 缺失时 idle 标为 null，不能解释为空等。", "", "`kernels_by_total_time.csv` 按全部调用累计 GPU 时间降序；`top_shapes.txt` 包含调用次数和形状。CPU 同步 API 的耗时与 GPU 执行有重叠，不能再加进总耗时；细分见 host_api_overlap.csv 和时间线。数据准备的 copy/cast 已在各阶段单元格内，不能重复计入。", "", "无 profiler 的同一固定输入 20 次均值：" + f"{meta['unprofiled_fixed_input_latency']['mean_ms']:.3f} ms。此数字只用来检查 profiler 开销；主表仍使用同一独占批次的 100 输入 ×3 重放。"]
    (destination / "REPORT.zh.md").write_text("\n".join(lines) + "\n")


def main():
    root = Path("results/holoq_object")
    out = root / "profiling_paper_selected"
    out.mkdir(parents=True, exist_ok=True)
    status = {"started": time.time(), "commands": [], "cases": {}}
    for family in ("holoq", "holoq_step_weight"):
        while True:
            selection = read_json(root / "evaluation_paper_holoq/backend_selection.json")
            if family in selection.get("families", {}):
                break
            time.sleep(10)
        choice = selection["families"][family]["selected"]
        case = choice["case"]
        directory = out / case
        marker = out / (case + ".completed.json")
        if not marker.exists():
            assert not directory.exists(), "Keep partial traces; explicitly choose new output after a failure"
            env = os.environ.copy()
            env["PROF_DEPLOY"] = str(root / "deployments" / case)
            cmd = ["bash", "scripts/profile_holoq_whole_policy.sh", str(directory)]
            status["commands"].append({"cmd": cmd, "PROF_DEPLOY": env["PROF_DEPLOY"], "started": time.time()})
            print("PAPER_PROFILE_START", case, flush=True)
            with (out / (case + ".log")).open("a") as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            marker.write_text(json.dumps({"returncode": result.returncode, "finished": time.time()}) + "\n")
        assert read_json(marker)["returncode"] == 0
        # Parsing/gzipping a large trace must not steal CPU from timed rollout.
        with open("/tmp/holoq-vla-policy-timing.lock", "a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            publish(directory, Path("experiment_results/holoq_object_20261027/profiling_paper") / case, case)
        status["cases"][family] = {"case": case, "path": str(directory)}
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print("PAPER_PROFILE_FINISHED", case, flush=True)
    status["finished"] = time.time()
    (out / "finished.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
