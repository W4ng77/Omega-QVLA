# 精度优先阶段：已证实事实与待完成项

此文件是进度记录，不是最终成功率或加速结论。新 kernel 融合、KV 性能探索和额外 profiler 已暂停，已完成数据保留。

- stride-aware 旋转已通过17项 GPU 测试及真实投影检查。Graph 的常量／Tensor 输入和断点日志恢复已有行为回归测试。8个 step 使用各自 scale 地址，历史／原网格部署不吸收 scale 后重新量化权重。
- 历史发布 pack 为 Hugging Face `ucmp137538/Omega-QVLA-GR00T-N1.5-LIBERO-W4A4` revision `01252878485fa9295b53ab833eb13d564f6911ad` 的 Object pack，SHA256见 provenance。
- 发布 pack 的 LLM **没有保存 activation scale**。原运行函数在首次调用后固定逐通道 scale。实际32次历史 warmup 后，84层的 calibrator 都只观察了1次；第二次独立初始化得到相同 scale（NMSE0）。部署将首次实际使用值显式冻结，不再依赖评测首个输入。
- 本轮70/100是另一套重建：LLM逐token A4、不同GPTQ求解与damping，且含额外LLM输出旋转。它是有效的构建／后端对照，不能充当发布 pack 的部署复现。尚不能仅凭这些差别断言哪项造成了全部掉分。
- 新发布网格部署保留全部180层原FP16权重值、全部旋转系数、96层原DiT八行scale；独立导出审计全部精确一致。缺失的LLM scale来自上述显式冻结过程，不冒充发布文件原有字段。
- 真实输入8,520层次比较全部有限，最大层NMSE8.90168e-8；完整动作NMSE0.000198–0.002431。允许小的数值误差，最终仍以完整任务成功率判断。
- 该保真版使用packed W4和已有tile_fp16：在GEMM块内解码后做FP16 MMA，不缓存整份浮点权重。单独的step_weight变体重新量化DiT每步吸收scale后的权重并使用原生S4 MMA，存储9份权重，不能作为原方法严格等价实现。

正在执行同checkpoint、8步、horizon16、seed／独立初始状态10–19的100集。发布原运行后端作为精度参考；FP16只补任务精度／内存参考，不开展浮点GEMM速度竞赛。原网格packed、step_weight与必要量化baseline将使用相同公共logits优化。额外单因素对照只把发布LLM逐通道scale改成逐token，W/R、DiT scale和后端全部固定，事先冻结后再读任务分数。

现有baseline及重建对照的有效性、原始路径和版本见existing_result_validity.json。历史88/90没有直接复用。发布pack原校准初始状态不可核验；其部署诊断与共享teacher校准重建必须分开报告。
