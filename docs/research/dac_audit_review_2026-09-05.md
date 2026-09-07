# DAC 评估审计复核（2026-09-05）

## 范围与结论

复核用户提供的评估总览，以及相关 workload、placement/NMP、GPU 能耗代码、配置与已有 canonical summary。此次不修改模型、物理假设或冻结求解器，不重跑热扫描。

总体判断：总览适合作为内部进度索引，但尚不能作为论文实验定稿依据。主动披露旧结果、条件性假设和未实现路径是正确方向；仍有基准混用、时序上界误当预测、算子记账不闭合及 provenance 表述过强的问题。应先修正 spec，再实现与重跑，避免把新问题固化进代码。

## P0：在论文结论或新实现前解决

### 1. 热优势对照混用了退役基准（基线入口清理已完成）

原总览第 5 节使用 M3D 83.85 °C 对比退役的固定功率基准，违反当前仓库明确的 Conventional 2x1 主基线口径。2026-09-05 按用户确认范围，已从当前文档、结果总表及运行示例移除该对照；历史配置仅保留为测试夹具，记录与图归档。83.85 °C 来自旧 NMP 工作点，不能直接与另一个功率工作点形成架构因果比较。同条件热评估仍待完成。

81.93349 °C 是当前 Conventional 2x1 回归锚点；83.85479 °C 数值上比它高约 1.92 °C，但由于工作点尚未匹配，这也不足以证明 M3D 热性能更差。

补证据：在相同冷却、边界条件和明确的封装约束下，分别报告等功率的几何比较与 workload 自洽功率的系统比较。给出 GPU Tmax、memory Tmax、各区材料适用温限、总功率、容量与吞吐。主张热墙需要给出满足温限时可支持的容量/性能或层数边界，不能只展示某个点求解成功。

### 2. Attention 算术强度与算子分工理由不成立

spec 给出 attention 68.719 GFLOP / KV 17.180 GB，即约 4 FLOP/byte；同一假设平台的 roofline 转折点为 100 TFLOP/s / 4.9 TB/s ≈ 20.4 FLOP/byte。因此在该模型下 attention 仍为带宽受限。FLOPs 占 82% 不等于计算密集，不能由此推导 Tensor Core 高利用率。

保留 attention 在 GPU 可以是合理的实现范围选择：涉及 softmax、归约、动态 KV 处理与更通用的执行能力，而当前 FEOL 单元只支持权重 GEMV。应改写理由，不必据此扩大 NMP 硬件功能。当前 workload attention FLOPs 公式是 QK 与 PV 的矩阵运算，未显式计算 softmax，表格应注明。

### 3. 1.93× 是理想流量上界，当前不足以当成吞吐预测

spec 用 max(GPU KV 时间, GPU compute 时间, MAC 时间) 得到 3.506 ms/token。但 B=1 的依赖包含 QKV 投影 → attention → O 投影 → FFN → 下一层；独立设备的总工作量不意味着可以完整重叠。

建议构建逐算子的依赖和资源时序，计入权重本地读取、KV 接口读取、GPU 计算、MAC、激活往返、部分和归约、同步以及同一 memory bank/service 资源争用。给出保守串行界与有依据的可重叠情形，不要求 cycle simulator。

示例：MAC 若只有 4.3 TFLOP/s，15.009 GFLOP 本身约需 3.49 ms。在 MAC 与 3.506 ms 的 GPU 阶段完全串行的简化情形，总时间约 7 ms，尚未计入额外开销。因此 4.3 TFLOP/s 只是理想重叠下的资源下界，不能叫“不拖后腿的可实现门槛”。这不是最终延迟预测，而是反例。

逐层串行执行还意味着不能默认全部 98 dies 在每个算子上都提供聚合算力/带宽。需依据算子 ownership 计算活跃 die 数与热点服务负载；集中放置降低通信，同时可能牺牲并行度。

### 4. Workload 与 placement 算子记账不闭合

从当前 canonical 配置与代码直接计算：

| 指标 | 数值 |
|---|---:|
| workload FLOPs/token | 83.728793600 G |
| placement units FLOPs/token 合计 | 82.678120448 G |
| 差值，即 vocab projection | 1.050673152 G |

`build_dense_decode_placement_units` 只生成逐层 Q/K/V/O/FFN 与 attention units，没有 vocab projection unit。总 traffic 又来自独立的 demand 总量，不能据此声称全部字节都漏算；确切问题是算子计算量、放置及通信缺少该输出投影的显式对应关系。

spec 的权重字节也需更正。FP16、B=1、每权重一次乘加计 2 FLOPs 时，投影的 FLOPs 数与权重 bytes 数相等：QKV 约 1.611 GB、O 约 1.074 GB、FFN 约 11.274 GB、vocab 约 1.051 GB。原表前三项约小了一半。

投影活跃权重合计约 15.009 GB，与 n_param=8e9 对应的 16 GB 驻留权重并非同一个量。必须解释 embedding、norm、参数近似与全量读取假设，区分驻留容量、活跃读取和 offload 字节，不能无解释地混用。

新增一致性检查应覆盖所有算子归属、总 FLOPs、驻留/活跃权重、local/external 流量、KV read/write，以及 FP16 MAC 与 FLOP 的二倍换算。不能仅替换旧 gain 的 expected 数字。

### 5. 文献实测锚点不等于项目已经校准或实现

GPU 代码明确将参数视为文献范围内的 nominal，且写明并非某 GPU 的逐 workload calibration；输出是 `ANALYTICAL_AFFINE_UTILIZATION_MODEL`。总览声称已升级为 `ANALYTICAL_CALIBRATED_BY_MEASURED_REFERENCE`，强于实现与证据，应降为“文献参考范围支持的参数化解析估算”。校准需要数据点、拟合方法和误差；跨设备找到数量级相近的结果只能做 sanity check。

同理，即使 Shiba 原文的 0.5 pJ/bit 已确认，`PAPER_REPORTED` 也只说明来源电路报告了该结果，不能证明本项目的工艺、coil 几何、距离、BER、时钟/SerDes 和 TX/RX 覆盖边界相同。应分别标注来源事实与迁移使用的假设。先明确 transceiver 是 silicon FEOL 还是 IGZO BEOL，再决定需要什么可行性证据，不能无端承担“IGZO 高速收发器”这一额外实现要求。

## P1：最终实验前补齐

### 6. 容量收益与小 batch 的 NMP 收益存在张力

33.18 GB 能装进 460.4 GB 只证明该点可行，未证明容量墙收益。已有 summary 包含更多 batch 点，但仍是旧全 offload 语义，不能直接承接新架构结论。

当前 tile_reuse 模型下，权重流量/token 为 W/B，KV 流量/token 约为 K(S)。忽略其他瓶颈时，去掉权重流量的理想增益是 A(B,S)=1+W/[B K(S)]。用旧 16 GB 与 128K 的 17.18 GB 口径，B=1 约 1.93×，B=8 约 1.116×，B=16 约 1.058×。这些是流量界，后续应使用闭合后的活跃权重数。

大容量可支持更大 batch，但更大 batch 会削弱权重卸载收益。应报告 B×S 的 regime 图，并区分等 B/S 的架构比较与相同延迟目标下各架构可支持的并发/吞吐。长于模型原生上下文的 1M 等点，应标明合成压力扫描，不能声称对应模型已验证的实际推理能力。

### 7. 功耗、能耗、热源的统一工作点（GPU 接线已修正）

当前 E8 只相加 GPU 与 memory dynamic energy，明确排除 host、cooling 等；还需说明 memory background/refresh、controller/PHY 的纳入与重叠边界。尤其 GPU 文献功耗的测量域是否包含 memory，需核实，避免再次加 memory 时重复记账。

原问题是 GPU 能耗与热路径工作点不一致。正式 runner 已将 E8 前移，E5/E6/E7 与 M3D logic-background 敏感性共用其 GPU 功率。2026-09-07 进一步冻结 bandwidth boundary：`B_actual=min(B_demand,4.8 TB/s)`；6.28–9.01 pJ/bit 参考派生范围的 nominal 中点为 7.645 pJ/bit，对应 canonical 饱和 decode 功率 367.568 W，超峰值需求不再外推功率。独立 memory energy 不从 GPU coefficient 中扣除。物理求解器未改；没有 E8 模型时正式评估直接拒绝运行，不存在固定功率兼容入口。该修正解决接线与边界语义，不升级模型的实测可信度。

最终每个结果行核对 P_accounted = E_accounted/token × tokens/s，未计入热源或能耗的项单列。1.2 J/token 目前只是 hand-check，不能进入论文摘要。

### 8. 公平基准与硬件可行性不能仅靠注明边界

同为 39.2 Tb/s 的 matched reference 可以隔离架构因素，但不是某款 GPU/HBM 产品的实现证明。需分别呈现 matched 对照与有明确产品参数的现实参考；不能把不同 GPU 的计算、带宽、功耗最有利数字拼成“实测平台”。

能耗比较要对齐功能边界；表格加 boundary 列不会自动使不等范围的数字可比。对未确认的部分给出范围或剔除对应优势主张。

FEOL 可行性至少闭合：数据精度、MAC/FLOP 定义、每 die 可用面积/频率/功率、每算子活跃 die、bank 并行度、有效读带宽、共享本地服务与 KV 服务争用、接口可实现带宽，以及存储容量扣除外围后的依据。不要只用聚合 TFLOPS 除以总 die 数证明可行。

敏感性优先扫描有效带宽、MAC 算力与局部并行性、执行重叠、接口能耗、材料/边界假设。以结论失效阈值为目标，而非只跑 ±30%。举例：在总览的同口径非 NMP 全路径近似下，M3D 非接口项约 0.355 pJ/bit，HBM 为 1.397 pJ/bit，则接口能耗约 1.042 pJ/bit 时访问能耗优势消失；1.0 档离边界很近。该阈值不是系统/NMP 能效的阈值。

## P2：文档一致性与科研证据组织

- 33.18/460.37≈7.21% 是总体容量利用率。summary 的 11.43% 字段是 `max_capacity_utilization`，表示最满 die 的利用率。
- “接触式接口”应改为“非接触式电感耦合接口”。
- “口径修正而非模型错误”应改为客观描述：旧结果采用全 offload 假设，不支持新权重-only 设计；实现存在遗漏时明确记录模型局限或缺陷，不预先排除错误。
- 分开标记冻结的求解器/回归基准与允许更新的 workload 结果；旧结果保留版本，不继续混入最新论文主表。
- 每个论文数据行记录 commit、配置与输入、结果路径、运行时间、模型版本、支持的 claim、剩余假设。旧结果和手算分别标记。
- 数值验证与物理验证分开：真 KCL 残差与测试 PASS 不等于热边界条件准确。主文/补充材料需索引既有网格、容差、边界、材料和功率分布敏感性及解析/文献对照；本次未穷尽扫描仓库，不能据此断言这些都没做。
- 题目中的 “Breaking ... Walls of LLM Inference” 比当前 decode-only 参数化证据更宽。先以“面向 LLM decode 的热约束容量与近存储计算协同设计”等范围组织贡献，再按最终证据决定是否采用更强题目。
- 增加明确的创新点和消融：HBM、正交存储不带 MAC、正交存储带 MAC、带放置优化；解释相对已有正交封装与既有 PIM 的新增机制。三项收益并列本身不足以证明新颖性。

## 建议执行顺序

1. 修正文档硬错误、claim 边界与算子分工 spec。
2. 统一算子/字节/FLOPs 账本和有依赖约束的时序，再实现 offload E8。
3. 做小规模语义与守恒测试，通过后重跑 canonical，保留冻结 Conventional 回归锚点。
4. 做 B×S 及等条件热对照、消融和影响结论最大的敏感性；若不存在稳健收益区域，收缩设计主张而非调参追数。
5. 用可追溯、同版本结果定稿摘要、主图和题目。

## 本次验证记录

- 环境：`C:\Users\Leslie\Miniconda3\envs\om3dthermal\python.exe`，win32。
- `python -m pytest tests/test_llm_decode_gpu_energy.py tests/test_llm_decode.py -q`：32 passed。
- 独立调用现有 workload 与 placement 函数核对 FLOPs、权重字节和 attention 算术强度；读取现有 summary 核对容量比率。
- 未复验总览所称的全部 37 项，未运行全量测试、热求解或 benchmark sweep。
- 本次未完成外部文献全文核验；以上文献相关结论是对证据适用范围的审计，不是对全文覆盖边界的认证。
