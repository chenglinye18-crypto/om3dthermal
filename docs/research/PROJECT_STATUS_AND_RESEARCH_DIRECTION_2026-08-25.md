# om3dthermal项目状态与下一阶段研究方向

日期：2026-08-25

## 1. 当前主线

项目已经从单纯steady-state thermal simulator扩展为workload-aware E2E framework：

```text
architecture
-> capacity / traffic / FLOPs
-> matched-reference performance
-> conditional memory energy
-> workload power
-> frozen GPU-PCG thermal
-> Tmax
```

当前论文主线建议冻结为：

> IGZO memory cell、monolithic 3D多层集成和orthogonal slab组织共同提供高local-resident capacity；在透明的matched-bandwidth功耗模型下，其memory energy和steady-state temperature保持可接受。

其中高容量是主创新和主要系统价值；低功耗与thermal feasibility是支撑该容量优势可用的第二层证据，不应建立在单个未经验证参数上。

## 2. 已经完成且可保留

- Conventional HBM、Orthogonal Si和Orthogonal M3D-IGZO三架构容量接口。
- LLM decode weight/KV footprint、traffic和FLOPs accounting。
- Aggregate capacity feasibility。
- 39.2 Tb/s matched-reference performance evaluator。
- Conditional memory J/token和workload-dependent memory power。
- Workload power到frozen FP64 matrix-free GPU-PCG thermal的完整连接。
- 3 architectures x 4 write-energy sensitivity的conditional E2E table。
- M3D bitcell/BEOL thermal merge regression和numerical baseline。
- Formal architecture/platform/workload/experiment config分层。

这些结果证明软件链路闭合，但不自动把conditional参数提升为实测architecture capability。

## 3. 当前M3D-IGZO nominal功耗

```text
memory internal       0.185758 pJ/bit   21.72%
MIV                   0.002446 pJ/bit    0.29%
FEOL route            0.167056 pJ/bit   19.53%
contactless interface 0.500000 pJ/bit   58.46%
total read energy     0.855261 pJ/bit
```

在39.2 Tb/s matched payload下：

```text
dynamic memory power  33.5262 W
refresh                0.0342 W
logic/background       unresolved
reported total         33.5604 W conditional lower bound
```

## 4. 建模精度决策

以下项目足以作为固定nominal，不再继续精细：

- 512x512 MAT；
- 当前MIV模型；
- 20 s retention nominal；
- 当前decode的write-energy sensitivity；
- 39.2 Tb/s iso-bandwidth scenario；
- frozen thermal solver和mesh/operator。

以下项目只需要小规模OFAT sensitivity，不需要电路级重建：

- FEOL energy scale：0.5x/1x/2x；
- M3D internal energy scale：0.5x/1x/2x；
- effective capacity：0.8x/0.9x/1.0x；
- interface/FEOL power的uniform与edge-concentrated thermal placement。

真正需要优先关闭的功耗边界：

1. contactless interface 0.5 pJ/bit是否包含完整TX/RX/coil/clock/serialization；
2. logic/background power的nominal或至少0/5/10/20 W sensitivity。

## 5. Conventional HBM窄语义问题已由Gate A关闭

canonical现按2个11x22 mm thermal-visible groups、每组2个11x11 mm physical stack equivalents解释，并以10.8x10.8 mm physical die做capacity packing：4 stacks x 12 dies/stack x 27 GiB/stack = 108 GiB/system。旧114.75 GiB与57.375 GiB/group口径已退役。

审计闭合字段：

```text
thermal visible groups
physical stack count
per-die area
per-stack capacity
system capacity
```

该问题不要求修改thermal solver，但会影响M3D/HBM capacity ratio的论文口径。

## 6. 推荐的添头创新：capacity-aware tensor placement

单纯“权重分配”已有大量相关工作，若作为独立算法很难形成创新。更适合本项目的是与orthogonal high-capacity architecture绑定的：

```text
capacity- and traffic-aware weight/KV placement
```

研究问题：当模型权重和KV总容量超过local HBM时，如何在local memory和外部DDR/host memory之间放置tensor，以最小化off-package traffic；同一workload在高容量M3D中是否可以全部local-resident。

最小可行模型不需要GPU cycle simulator：

1. 将weights和KV按layer/tensor group表示；
2. 为每组记录resident bytes和read/write bytes/token；
3. 在local capacity约束下优先放置traffic-per-byte最高的group；
4. 未放入的group进入显式external tier；
5. 使用外部tier的带宽和pJ/bit计算spill time与spill energy；
6. 比较HBM+DDR与M3D-local的tokens/s、J/token和Tmax。

这一方向的价值不是声称一种通用新offloading算法，而是把M3D容量优势转化成可观察的系统收益：

```text
larger local capacity
-> less external spill
-> less off-package traffic
-> higher tokens/s / lower J/token
```

若需要更强的architecture co-design，可在后续增加orthogonal slab内的traffic/thermal-aware placement：把高traffic tensor放到更接近接口的slab，同时平衡热源。但只有在简单tiered placement已经显示明显收益后才继续。

## 7. 进入该方向的Gate

开始实现前应先用现实workload做hand check：

- workload总容量应超过所选HBM reality anchor；
- workload应能够放入当前或0.8x容量的M3D；
- DDR spill应明显影响tokens/s或J/token；
- 改变placement确实改变至少一个论文主指标。

若上述条件不成立，权重分配不构成load-bearing贡献，应停止该支线。

## 8. 推荐执行顺序

1. 关闭Conventional HBM stack/capacity语义。
2. 完成M3D interface和logic-background的少量敏感度。
3. 选择一个现实高容量dense decode workload。
4. 写capacity-aware tensor placement analytical spec，不立即实现。
5. hand check确认HBM spill与M3D local-residency差异。
6. 通过Gate后再接入现有E2E evaluator。

## 9. 当前状态

```text
THERMAL_SOLVER                 FROZEN / PASS
E2E_SOFTWARE_CHAIN            PASS
M3D_CAPACITY                  ANALYTICAL / ROBUSTNESS_PENDING
M3D_READ_ENERGY               CONDITIONAL_ANALYTICAL_NOMINAL
M3D_INTERFACE_BOUNDARY        NEEDS_AUDIT
M3D_LOGIC_BACKGROUND          UNRESOLVED_LOWER_BOUND
BANDWIDTH_CAPABILITY          NOT_VALIDATED
HBM_PER_STACK_SEMANTICS       NEEDS_NARROW_AUDIT
TENSOR_PLACEMENT_EXTENSION    PROPOSED / NOT_STARTED
```

STOP
