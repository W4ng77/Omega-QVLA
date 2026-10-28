# 完整 HoloQ-VLA get_action 剖析

配置：holoq_llm_s4_rotv2，84个语言层CUTLASS原生S4、96个DiT层Marlin，保留原始量化网格和8个per-step/channel scale；使用实际部署的head-only CUDA Graph。已完成模型加载、编译和32次warmup，捕获5次完整调用。固定输入为Object task0/init10。该配置完整Object成功率70/100。

## 正式延迟与 profiler 时间分开

不带profiler、不带NVTX/记录hook的20次固定输入调用：均值 **68.652ms**，p50 **68.336ms**，p95 **69.993ms**。此前100输入×3的正式重放均值68.878ms，真实rollout均值69.598ms。

下表时间来自真实CUDA活动，不包含与kernel重复的GPU阶段标记；按累计GPU时间排序。GEMM包含必需的尺度epilogue。

| 部分 | 五次累计GPU ms | 每次完整policy GPU ms | 每次相关kernel次数 |
|---|---:|---:|---:|
| Action head，全8步/GEMM及epilogue | 131.659 | 26.332 | 1122 |
| 视觉塔（全部GPU工作） | 40.504 | 8.101 | 821 |
| Action head，全8步/旋转 | 32.129 | 6.426 | 1536 |
| Action head，全8步/其他计算 | 31.528 | 6.306 | 2680 |
| Language/GEMM及epilogue | 31.124 | 6.225 | 170 |
| Language/旋转 | 12.591 | 2.518 | 168 |
| Language/其他计算 | 9.666 | 1.933 | 482 |
| Action head，全8步/拷贝/转换 | 9.638 | 1.928 | 864 |
| Action head，全8步/量化打包 | 9.370 | 1.874 | 768 |
| Language/拷贝/转换 | 3.219 | 0.644 | 151 |
| Language/量化打包 | 2.421 | 0.484 | 84 |

阶段GPU总计：action head **42.865ms**，language **11.804ms**，视觉 **8.101ms**。其余输入准备/投影/后处理GPU活动另见完整CSV。总GPU活动 **62.917ms/次**。

数据准备横跨上述阶段：GPU copy/cast/memcpy/memset 合计 **4.333ms/次**，已包含在上述统计中，不额外加总。CPU的to、copy、contiguous是嵌套调用，不能把inclusive时间相加。

| CPU操作 | 次数/policy | inclusive CPU ms/policy |
|---|---:|---:|
| aten::to | 705 | 45.221 |
| aten::_to_copy | 464 | 45.023 |
| aten::copy_ | 691 | 44.745 |
| aten::item | 20 | 0.019 |
| aten::_local_scalar_dense | 20 | 0.011 |
| aten::contiguous | 188 | 1.306 |
| aten::clone | 213 | 1.439 |

## CPU、同步与GPU空白

| 采集器 | 带profiler完整wall ms | GPU活动并集 ms | policy内GPU非活跃 ms |
|---|---:|---:|---:|
| PyTorch | 76.998 | 62.917 | 14.081 |
| Nsight node | 75.448 | 62.742 | 12.706 |

本次真实GPU活动基本串行，重叠小于0.001ms；程序仍按区间并集计算，未假设可直接相加。GPU空白包含CPU预处理/提交和profiler开销，不能全部称为生产环境同步等待。正式延迟采用上面的无profiler结果。

| CPU CUDA API | 次数/policy | API elapsed ms | 与GPU busy重叠 ms |
|---|---:|---:|---:|
| cudaGraphLaunch | 1 | 4.763 | 4.749 |
| cudaMemcpyAsync | 27 | 41.707 | 41.339 |
| cudaStreamSynchronize | 18 | 0.118 | 0.072 |

cudaMemcpyAsync的host时间包含pageable DtoH的隐式等待。约41.7ms的API时间大部分与GPU计算重叠，不能再加成41.7ms的CPU额外开销。时间线中存在CPU准备/提交间隙；最大的100个空白和当时CPU阶段保存在largest_gpu_idle_gaps.csv。

## 调用次数、形状、8个step

部署trace每次有1536个head旋转kernel、768个head量化kernel，即96层×8步×输入/输出旋转，以及96层×8步量化。Marlin可能对一个逻辑GEMM启动多个kernel，不能把kernel次数直接当层数。语言部分有84次量化和168次旋转。真实形状保存在metadata.json和top_shapes.txt；例如语言输入[1,558,2048]、head查询[1,49,1536]、交叉注意力context长度557–559。

补充eager采集只用于核对每个step和层的归属，不用其时间替代Graph部署结果。

| Step | 旋转kernel/policy | 量化kernel/policy |
|---|---:|---:|
| 0 | 192 | 96 |
| 1 | 192 | 96 |
| 2 | 192 | 96 |
| 3 | 192 | 96 |
| 4 | 192 | 96 |
| 5 | 192 | 96 |
| 6 | 192 | 96 |
| 7 | 192 | 96 |

## 结论与优化次序

1. 最大累计项是8步action head的GEMM，约26.3ms；其中packed Marlin本身约21.3ms。全policy旋转约8.9ms，量化打包约2.4ms。不能因为某个pack kernel单次很小/很慢就优先改它。
2. 原方法DiT的列scale处于K归约内部，不能直接替换成普通token/output scale的INT4 epilogue。应先完成独立命名的step_weight真实INT4分支的head剖析、质量和存储评测，再决定是否继续写定向PTX。
3. language中发现[558,2048]×[2048,151680]的完整词表输出投影，每次约3.70ms；backbone随后只取hidden_states。下一步可验证只算必要logits/跳过未消费输出的公共优化，必须同样提供给所有baseline并验证完整policy输出。这里只识别机会，未冒充已获得加速。
4. 拷贝/类型转换合计约4.3ms，适合检查布局和相邻算子融合。CPU同步的host阻塞与GPU工作重叠，不能把它重复计时，也不能只优化MMA来解决所有launch间隙。

Nsight2023.4的node模式成功核对实际Graph的kernel、拷贝和CPU提交关系，GPU累计时间与PyTorch相差约0.3%。该版本的粗粒度graph模式没有导出Graph GPU body，只有约20ms非Graph工作：该记录标为不完整，GPU idle字段置空，绝不把缺失的约43ms算成GPU空闲。

采用[PyTorch Profiler](https://docs.pytorch.org/docs/main/profiler)的CPU/CUDA和shape采集，按[Nsight Systems说明](https://docs.nvidia.com/nsight-systems/UserGuide/)使用NVTX和cudaProfilerStart/Stop限制采集窗口。原始Chrome时间线、Nsight报告、命令及逐项CSV保存在同目录；源码为research/holoq_bench/profile_policy.py和analyze_policy_trace.py。
