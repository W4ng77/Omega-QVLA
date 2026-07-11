# π0.5 速度对比实验指南（FP16 vs Ω-QVLA vs QVLA vs QuantVLA，真量化）

本文档指导复现单次推理速度对比与 LIBERO 端到端评测。参考结果（H100 80GB）：
FP16 32.3ms / **Ω-QVLA 26.0ms（0.80×）** / QuantVLA-real 26.9ms / QVLA-real 27.1ms。
新机器（如 RTX 4080）绝对数字会不同，先跑 FP16 定基线，看相对比值。

## 1. 前置条件

| 依赖 | 说明 |
|---|---|
| openpi venv | openpi 官方仓库 + `uv venv`（py3.11），并 `uv pip install -e <本仓库>` |
| QServe kernel | `omniserve_backend`，编译见下 §1.1；**没有它所有 `qserve/_real` 模式都跑不了** |
| π0.5 checkpoint | `pi05_libero` PyTorch 转换版，**必须带 `assets/` norm_stats** |
| 校准观测 | `deploy_extras/pi05_calib_obs.pt`（或用 `tools/record_libero_obs.py` 现录，需 LIBERO+EGL） |
| Ω-QVLA packs | `omega-qvla-pi05-w4a4/pi05_<suite>/quantized.pt`（GPTQ+旋转 pack） |
| 对手 checkpoints | QVLA / QuantVLA 烤制版，见 §4；只测我们自己时不需要 |

### 1.1 编译 QServe kernel

```bash
git clone https://github.com/mit-han-lab/omniserve
cd omniserve && mkdir -p kernels/omniserve_backend && touch kernels/omniserve_backend/__init__.py
# H100 用 9.0，RTX 4080/Ada 用 8.9
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 uv pip install -e kernels --no-build-isolation --python <openpi venv 的 python>
# 验证（必须先 import torch 再 import 扩展）
python -c "import torch, omniserve_backend.qgemm_w4a8_per_chn; print('ok')"
```

## 2. 单次推理速度对比（核心实验）

入口：[tools/bench_pi05_speed.py](../tools/bench_pi05_speed.py)。它对每个 mode 重建 policy、
预热 4 次、计时 12 次 `policy.infer`（真实 LIBERO-Long 观测）取平均。

```bash
cd <openpi 仓库根>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<本仓库> \
  .venv/bin/python <本仓库>/tools/bench_pi05_speed.py \
  --checkpoint      /path/to/pi05_libero_pytorch \
  --pack            /path/to/omega-qvla-pi05-w4a4/pi05_long/quantized.pt \
  --obs-path        <本仓库>/deploy_extras/pi05_calib_obs.pt \
  --qvla-checkpoint /path/to/qvla-pi05-w4-lmexp \
  --mode fp16,w4a8_fold_a16,qvla_real,quantvla_real
```

### mode 速查表

| mode | 含义 | kernel |
|---|---|---|
| `fp16` | openpi 原生 bf16 + torch.compile（基线） | cuBLAS bf16 |
| `w4a8_fold_a16` | **Ω-QVLA 冠军配置**：旋转折叠进权重 + prefill W4A16 + expert QServe 真量化 | QServe int4/int8 TC（小 M）+ bf16（大 M） |
| `qvla_real` / `quantvla_real` | 对手方法真量化：方法差异只在**烤好的权重值**里，部署管线与我们相同（不折叠旋转——他们的 checkpoint 已离线烤平） | 同上 |
| `qvla` / `quantvla` | 对手方法**原样运行时**（fake-quant / 运行时旋转），展示"不用快 kernel"的方法本来速度 | 无（bf16 假量化） |
| `rtn` | 每加一个模块的消融基线 | 无 |

可选开关：`--phase`（把延迟拆成 prefill 与 10 步去噪，用 2 步外推法）、`--profile`（torch profiler 按算子归类）。

### 环境变量（`_real`/`fold` 模式内部自动设置，手动跑服务时需要自己给）

```bash
GR00T_GPTQ_QSERVE=1            # expert 侧 QServe int4/int8 真量化（M≤GR00T_GPTQ_QSERVE_MAX_M=128 的调用）
GR00T_GPTQ_FOLD_ROT=1          # 旋转折叠 W_eq = R_out^T·W·R^T（只对 Ω pack 有意义）
GR00T_GPTQ_A16_LARGE_M=1       # 大 M（prefill）层跳过激活量化，权重仍 W4
GR00T_GPTQ_QSERVE_GPTQ_REQUANT=1  # 【推荐】打包时 GPTQ 误差补偿，精度 88→96/100，速度不变
GR00T_GPTQ_CALIB_OBS=<obs.pt>  GR00T_GPTQ_CALIB_STEPS=6   # 服务前预校准
```

### 读数注意

1. **freeze 是一切的前提**。bench 内部按 mode 自动触发 `freeze_gptq_fast_paths()`；
   自己写脚本时忘了 freeze 会测到假数字（历史教训：550ms 的"折叠"结果全是未 freeze 的）。
2. 第一次跑含 torch.compile 编译，几分钟正常；计时已在预热之后。
3. 运行间波动约 ±1ms；比较要同卡同电源状态，别跨卡比。
4. 单 prompt 计时（`--obs-slice` 里取第一条）是刻意的：换 prompt 会触发 dynamo 重编译，
   污染测量；真实部署一集内 prompt 不变。

## 3. LIBERO 端到端评测（速度 + 成功率）

入口：`deploy_extras/run_pi05_speed_eval.sh`（路径是硬编码的，换机器需改开头几个变量）。

```bash
#                   mode           suite gpu port
bash run_pi05_speed_eval.sh fp16          long 0 8400
bash run_pi05_speed_eval.sh omega_fold_gq long 1 8410   # 我们，含 GPTQ 补偿修复
bash run_pi05_speed_eval.sh qvla_real2    long 2 8420   # QVLA 统一口径
bash run_pi05_speed_eval.sh quantvla_real long 3 8430
```

每任务 10 集 × 10 任务 = 100 集，H100 上约 4-5 小时/场（MuJoCo 仿真占大头）。
结果在 `$OUT/<mode>_<suite>/merged_summary.json`（总成功率 + 分任务）。
参考精度（LIBERO-Long 100 集）：FP16 93 / Ω-QVLA(gq) ~96 / QuantVLA-real 96 / QVLA 统一口径 ~76。

## 4. 烤制对手方法的真量化 checkpoint

方法身份只体现在权重值上——离线把各自方法的量化烤进 checkpoint，部署统一走 QServe/A16：

```bash
# QuantVLA：DuQuant 旋转 + RTN group64，逐层 W_eq = RTN(W·R)·R^T
python tools/bake_quantvla_pi05.py   # 路径在脚本头部改

# QVLA：Hessian 代理 + {2,4,8,16} 贪心通道分配 avg4.0，per-row 对称烤入
python tools/build_qvla_pack_pi05.py \
  --checkpoint <pi05> --obs-path <calib_obs> --output-dir <out> \
  --include-regex '.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
```

⚠️ 两个坑：
- **口径**：QVLA 论文口径会量化 SigLIP 视觉塔且不动 expert——在 π0.5 上直接 0/100。
  对比实验必须用上面的 include-regex 统一到 252 层（LM+expert，不动视觉塔）。
- **tied weights**：烤完的 `model.safetensors` 会多出一个 `embed_tokens.weight`
  （state_dict 物化 tied 权重），需要删掉否则加载报错：对比原 checkpoint 的 key 集合，
  strip 多余键后重存（safetensors load→pop→save）。

## 5. 结果解读框架

- 真量化只在**小 M、大 K·N**（访存受限）的 GEMM 上赢：expert 调用（M≈50-60）QServe 1.42×；
  prefill（M≈800，计算受限）反而慢 → 混合部署是设计要点，不是妥协。
- 三方法统一部署后速度趋同（26-27ms）：运行时里没有"方法"，只有权重值。
  fake-quant 阶段的速度差异（如 QuantVLA 297ms）是实现工件，不是方法固有代价。
- 精度差异才是方法差异：GPTQ+旋转（我们）≈ DuQuant+RTN（QuantVLA）> 混合位宽贪心（QVLA）。
- 折叠后的权重不在任何量化格点上，QServe 打包的二次量化会掉精度；
  `GR00T_GPTQ_QSERVE_GPTQ_REQUANT=1` 用校准 Gram 做 GPTQ 误差补偿，免费拿回来。
