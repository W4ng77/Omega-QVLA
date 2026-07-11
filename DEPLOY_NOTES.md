# Ω-QVLA 速度优化部署说明（speed-opt 分支）

**成果**：π0.5 + LIBERO 真实观测，单次推理 **26.0ms vs FP16 32.3ms（快 24%）**，LIBERO-Long 端到端成功率与 FP16 持平（93/100，H100 实测）。官方原样运行时为 400~1800ms。

## 一、这套代码做了什么

| 组件 | 文件 | 作用 |
|---|---|---|
| fast-path 冻结 | `gr00t/quantization/gptq_layers.py` → `freeze_fast_path()` / `freeze_gptq_fast_paths()` | 把 GptqLinear 的全部运行时状态（act scale、旋转、权重、bias）固化成静态 buffer，forward 变成纯张量直线代码，torch.compile 可整图编译 |
| 旋转折叠 | 同上（`GR00T_GPTQ_FOLD_ROT=1`） | W_eq = R_outᵀ·W·Rᵀ 块级折叠（einsum，勿用稠密——16384 维层会爆），运行时零旋转成本 |
| QServe 真量化 | `gr00t/quantization/qserve_op.py` + freeze 内打包 | int4 打包权重 + per-token int8 activation，INT8 Tensor Core（`GR00T_GPTQ_QSERVE=1`，`GR00T_GPTQ_QSERVE_MAX_M` 门控小 M 层，默认 128） |
| A16 大 M 回退 | `GR00T_GPTQ_A16_LARGE_M=1` | prefill 等大 M 层跳过 act 量化（权重仍 W4 折叠烤入），GEMM 为原生 bf16 速度 |
| FP8 路径 | `GR00T_GPTQ_FP8=1` | rowwise `_scaled_mm`（备选，π0.5 上 A16 更优） |
| CUDA graph | `gr00t/quantization/dit_graph.py` + `flow_matching_action_head.py` + `gr00t_n1.py` | DiT 去噪循环与 backbone 的形状键控图捕获（`GR00T_DIT_CUDA_GRAPH=1` / `GR00T_BACKBONE_CUDA_GRAPH=1`，GR00T 用） |
| 服务修复 | `scripts/openpi_inference_service.py` | GPTQ 包装后 `model.to(device)`（修 CPU 权重灾难）；`GR00T_GPTQ_CALIB_OBS` 服务前预校准+冻结 |
| Triton 融合 kernel | `gr00t/quantization/fused_rot_quant.py` | 置换+块旋转+量化单 kernel（备用；折叠方案后主路径不再需要） |

## 二、π0.5 最优配置（26ms 冠军配置）

```bash
METHOD=gptq WBITS=4 ABITS=8 \
OPENPI_GPTQ_PATH=<pack>/pi05_<suite>/quantized.pt \
GR00T_GPTQ_QSERVE=1 GR00T_GPTQ_FOLD_ROT=1 GR00T_GPTQ_A16_LARGE_M=1 \
GR00T_GPTQ_CALIB_OBS=<obs.pt> GR00T_GPTQ_CALIB_STEPS=6 \
bash scripts/run_pi05_libero_benchmark.sh
```

速度基准一键复现：
```bash
PYTHONPATH=<本仓库> python tools/bench_pi05_speed.py \
  --mode fp16,w4a8_fold_a16,qvla_real,quantvla_real
```

## 三、在新机器上搭建（如 RTX 4080 = Ada / sm_89）

1. openpi venv（uv, py3.11）+ 本仓库 `pip install -e .`
2. **QServe kernel**：`git clone https://github.com/mit-han-lab/omniserve`，
   `mkdir kernels/omniserve_backend && touch kernels/omniserve_backend/__init__.py`（否则装不上），
   `TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=32 uv pip install -e kernels --no-build-isolation`
   （sm_89 在其支持列表；import 前必须先 `import torch`）
3. checkpoint：pi05_libero PyTorch 转换版 + `assets/` norm_stats 一起拷（转换脚本不带它，会报 FileNotFoundError）
4. calib obs：`tools/record_libero_obs.py` 生成（需 LIBERO+EGL）或直接拷 `pi05_calib_obs.pt`
5. 4080 注意：显存 16GB——π0.5 bf16 权重 ~7GB + 激活可以跑；INT8 TC 有（Ada 支持，还有 FP8）；绝对数字会不同但相对结论（真量化>fp16）应保持，因为 4080 带宽更小、int4 权重的访存优势更大

## 四、已知边界

- fake-quant 与真量化的 bin 边界舍入有 1-ulp 级差异，端到端成功率无损（93/100 = FP16）
- QServe 权重是 per-row 重量化（非 GPTQ 原 group 格点），更严谨可扩 per-group
- GR00T：backbone+DiT 双图已通，但其 GEMM 尺寸（DiT 权重 L2 常驻）物理上不适合真量化加速，最优=W4A16 与 fp16 等速
- 三方法（Ω/QVLA/QuantVLA）在同一真量化部署下速度趋同（26-27ms）——方法差异在精度，见 `out/pi05_speed_comparison.md`
