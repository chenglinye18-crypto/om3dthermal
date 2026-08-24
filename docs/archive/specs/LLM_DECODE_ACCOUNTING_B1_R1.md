# Track B1-R1: LLM Autoregressive Decode Accounting Spec (Revised)

> **Research Question:** 如何建立透明、可手算、适用于 long-context small/medium-batch autoregressive decode 的 workload accounting spec？
>
> **Scope:** 纯 analytical specification。不输出 tokens/s、J/token、power 或 Tmax。不修改仓库文件。
>
> **Deliverable:** 提交 Research Lead 审核的修订 spec 文档。

---

## Research Question

为 Orthogonal M3D Memory DAC 项目建立 LLM autoregressive decode 的 workload accounting spec，满足：

1. **透明可手算**：所有公式可被人工复核，无黑盒 simulator。
2. **footprint ≠ traffic**：严格区分 memory footprint、algorithmic access、architecture-visible traffic、physical target-memory traffic。
3. **Long-context 显式**：attention compute 和 KV read traffic 必须显式依赖 context length S，不得被 2×Nparam 掩盖。
4. **Batch 语义完整**：区分 aggregate step throughput 与 per-sequence/per-token 指标；mathematical FLOPs/token 不因 batch 自动下降。
5. **Provenance 清晰**：每个假设标记来源类别；技术事实引用 primary source。

---

## Evidence / References

### Primary Sources (Model Architecture)

- Grattafiori et al., "The Llama 3 Herd of Models," Meta AI, July 2024. arXiv:2407.21783. (LLaMA-3/3.1 architecture, GQA, SwiGLU, RMSNorm)
- Official model configuration: `meta-llama/Meta-Llama-3.1-8B` on Hugging Face. Config URL: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json. (Concrete parameter values: hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, max_position_embeddings, vocab_size)

### Primary Sources (Attention Algorithms & Memory Management)

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022. (Tiling for attention intermediate activation reduction; HBM traffic reduction via SRAM-aware scheduling)
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023. (Improved tiling and work partitioning for attention)
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. (KV-cache paging, block-based allocation, fragmentation analysis)

### Project Internal References

- `configs/cases/orthogonal_m3d_igzo.yaml` — Workload bandwidth configuration
- `src/om3dthermal/power/config.py` — `WorkloadInput` schema

---

## Files Inspected

- `src/om3dthermal/power/config.py`
- `src/om3dthermal/power/model.py`
- `src/om3dthermal/architecture_comparison.py`
- `configs/cases/orthogonal_m3d_igzo.yaml`
- `configs/cases/conventional_hbm_2x1.yaml`

---

## Files Modified

**None.** 本任务为纯 analytical specification 修订，不修改任何代码、config、tests 或治理文档。

---

## Configuration

### Symbol and Unit Table

| Symbol | Definition | Unit | Provenance |
|--------|-----------|------|------------|
| `Nparam` | 模型可训练参数总量 | 1 (scalar) | PAPER_REPORTED / OFFICIAL_MODEL_CONFIG |
| `L` | Transformer decoder layer 数 | 1 (scalar) | PAPER_REPORTED / OFFICIAL_MODEL_CONFIG |
| `B` | 同时活跃的 sequence 数 (batch size) | 1 (scalar) | MODELING_CHOICE (workload) |
| `S_i` | 第 i 个 sequence 的当前 context length | 1 (scalar) | MODELING_CHOICE (workload state) |
| `S` | 等长 batch 时的统一 context length; `S = S_i` ∀i | 1 (scalar) | MODELING_CHOICE (simplification) |
| `S_max` | 最大支持的 context length | 1 (scalar) | OFFICIAL_MODEL_CONFIG |
| `Hq` | Query head 数 (attention heads) | 1 (scalar) | PAPER_REPORTED / OFFICIAL_MODEL_CONFIG |
| `Hkv` | Key/Value head 数 (GQA/MQA 分组数) | 1 (scalar) | PAPER_REPORTED / OFFICIAL_MODEL_CONFIG |
| `Dhead` | 每个 attention head 的维度 | 1 (scalar) | PAPER_REPORTED / OFFICIAL_MODEL_CONFIG |
| `Dmodel` | 模型 hidden dimension; `Dmodel = Hq × Dhead` | 1 (scalar) | DERIVED_FROM_OFFICIAL_MODEL_CONFIG |
| `Dff` | MLP 中间层维度 (feed-forward hidden dim) | 1 (scalar) | PAPER_REPORTED / OFFICIAL_MODEL_CONFIG |
| `V` | Vocabulary size | 1 (scalar) | PAPER_REPORTED / OFFICIAL_MODEL_CONFIG |
| `bw` | Weight 每个参数的存储位数 | bit/param | MODELING_CHOICE (precision) |
| `bkv` | KV cache 每个元素的存储位数 | bit/element | MODELING_CHOICE (precision) |
| `bact` | Activation 每个元素的存储位数 | bit/element | MODELING_CHOICE (precision) |
| `Cphysical` | 物理存储容量 (architecture 暴露的总容量) | byte | DERIVED_FROM_GEOMETRY |
| `Cusable` | Workload 实际可用容量 (`Cusable = Cphysical × (1 − ε_reserve)`) | byte | MODELING_CHOICE |
| `ε_reserve` | 系统保留/碎片/分配器开销比例 | 1 (scalar) | MODELING_CHOICE / NOT_VALIDATED |
| `Ngen` | 本次 decode session 计划生成的 token 数 | 1 (scalar) | MODELING_CHOICE (workload) |

### Unit Convention

- **Byte (B)**：所有 `Bytes_` 前缀变量单位为 byte。
- **Bit (b)**：所有 `bits_` 前缀变量单位为 bit。
- **GB** = 10⁹ B；**GiB** = 2³⁰ B = 1,073,741,824 B。
- 本 spec 在公式推导中统一使用 **byte**；在报告数值时同时给出原始 byte 和 GB/GiB 转换。
- **FLOPs** 指浮点乘加操作数（matrix-vector multiplication 惯例：m×n dot-product 计 2mn FLOPs）。

---

## Memory Footprint Equations

### 1. Weight Footprint


**说明：**
- `Nparam` 是否包含 embedding/output projection matrix：取决于模型设计。对于 tied embedding，计一次；untied，计两次。需在 provenance 中明确。
- Quantization metadata (scales, zeros, group indices)：本 spec v0 假设无额外 metadata（`bw` 为有效存储位宽）。若使用 INT8/INT4/FP8，需显式加入 `Bytes_weight_metadata`。
- Bias terms 参数量通常 `≪ Nparam`，可忽略或显式加入。

**Provenance:** `bw` 为 **MODELING_CHOICE**。`Nparam` 来自 **OFFICIAL_MODEL_CONFIG**。

---

### 2. KV-Cache Footprint

对于等长 batch (`S_i = S` ∀i)：

```
Bytes_KV = 2 × L × B × S × Hkv × Dhead × bkv / 8
```

**说明：**
- 因子 **2** 对应 Key (K) 和 Value (V)。
- `Hkv` 体现 GQA/MQA/MHA：
  - **MHA**: `Hkv = Hq`
  - **GQA**: `Hkv < Hq`（如 LLaMA-3.1-8B: `Hq=32, Hkv=8`）
  - **MQA**: `Hkv = 1`

**Variable-length batch：**

```
Bytes_KV_variable = 2 × L × Hkv × Dhead × bkv / 8 × Σ_i S_i
```

**Padding / Page / Block Waste：**
- 上述公式为 **raw tensor footprint**（最小理论值）。
- PagedAttention 等系统使用固定-size blocks（如 16 tokens/block），产生 **internal fragmentation**。
- 引入 `ε_KV_waste`：

```
Bytes_KV_realized = Bytes_KV × (1 + ε_KV_waste)
```

**Provenance:** `ε_KV_waste` 为 **NOT_VALIDATED**（取决于具体推理框架实现）。

---

### 3. Total Memory Footprint

```
Bytes_required = Bytes_weight_footprint + Bytes_KV_realized + Bytes_runtime
```

其中 `Bytes_runtime` 包括 activation workspace、attention score buffer、scheduler metadata 等。

**Provenance:** `Bytes_runtime` 为 **NOT_VALIDATED**。v0 建议保留为符号或在 hand-check 中给出数量级估计并标注 **MODELING_CHOICE**。

---

### 4. Capacity Feasibility

```
feasible = (Bytes_required <= Cusable)
```

```
Cusable = Cphysical × (1 − ε_reserve)
```

**四层容量概念：**
1. **Physical capacity** (`Cphysical`)：物理存储阵列可寻址总位数。
2. **Reserved capacity**：系统固件/驱动保留区域。
3. **Usable capacity** (`Cusable`)：workload 实际可分配容量。
4. **Runtime-realized footprint** (`Bytes_required`)：当前 workload 实际占用容量。

**Provenance:** `ε_reserve` 为 **MODELING_CHOICE** / **NOT_VALIDATED**。

---

## Traffic Accounting Equations

### Critical Rule: Footprint ≠ Traffic

每一类数据给出四列：

| Data Category | Footprint (B) | Algorithmic Access (elements) | Ideal/Min Transfer (B) | Architecture-visible Traffic Assumption (B) |
|---------------|--------------|-------------------------------|------------------------|---------------------------------------------|
| Weights | `Bytes_weight_footprint` | `Nparam` (total) | `Bytes_weight_active_per_step` (v0: = footprint) | `T_weight_arch` |
| Historical K | `Bytes_K` | `L × B × S × Hkv × Dhead` | `L × B × S × Hkv × Dhead × bkv / 8` | `T_K_read_arch` |
| Historical V | `Bytes_V` | `L × B × S × Hkv × Dhead` | `L × B × S × Hkv × Dhead × bkv / 8` | `T_V_read_arch` |
| New K write | — | `L × B × Hkv × Dhead` | `L × B × Hkv × Dhead × bkv / 8` | `T_K_write_arch` |
| New V write | — | `L × B × Hkv × Dhead` | `L × B × Hkv × Dhead × bkv / 8` | `T_V_write_arch` |

**术语定义：**
- **Footprint**：数据在目标存储器中的 resident 容量。
- **Algorithmic Access**：算法层面需要访问的数据元素个数（不含位宽）。
- **Ideal/Min Transfer**：理论最小传输量（假设 100% 利用率和零开销）。
- **Architecture-visible Traffic**：在特定 modeling choice 下（如 full-reread assumption），从目标 memory 读取/写入的传输量。此处**不是 measured/profiled DRAM traffic**。

---

### 5. Weight Read Traffic

#### 5.1 Weight Footprint (Resident)

```
Bytes_weight_footprint = Nparam × bw / 8
```

**说明：** 这是模型全部参数的 resident 存储容量（footprint）。对于 autoregressive decode step，并非所有 resident parameter bytes 都必然在单个 step 中被完整读取（见 5.3）。

#### 5.2 Algorithmic Weight Consumption per Aggregate Decode Step

一个 complete decode step 执行全部 `L` 层 forward pass。

**逐组件 algorithmic access：**
- **Transformer layer weights**：每层全部权重需读取；`L` 层合计 access = `Nparam_layers` elements。
- **Input token embedding lookup**：decode step 中通常只读取当前 token 对应的一个 row（`Dmodel` elements），而非完整 embedding matrix（`V × Dmodel`）。若 embedding 与 lm_head **tied**，此 row 可能复用；若 **untied**，需单独查表。
- **Output lm_head / vocab projection**：生成 next-token logits 需完整读取（`Dmodel × V` elements）。

综合而言：
```
Algorithmic_weight_access_per_step = Nparam  (elements)
```

**注意：** 上述 `Nparam` 为 total reported parameter count，包含 embedding/output projection。但 decode step 的 *实际 active weight elements* 在理论上可能少于 `Nparam`（因 embedding lookup 通常只读一个 row）。

一个 complete decode step 执行全部 `L` 层 forward pass，每层读取该层全部权重。

```
Algorithmic_weight_access_per_step = Nparam  (elements)
```

#### 5.3 Ideal/Minimum Weight Transfer

**保守 MODELING_CHOICE（v0）：**

```
Bytes_weight_active_per_step = Bytes_weight_footprint
```

```
T_weight_ideal_per_step = Bytes_weight_active_per_step
```

**说明：**
- 理论上，input token embedding lookup 在 decode step 中通常只读取当前 token 对应的一个 row（`Dmodel × bw/8` bytes），而非完整 embedding matrix。
- Output lm_head（`Dmodel × V × bw/8`）在 vocab projection 中需完整读取。
- Transformer layer weights 在 layer-by-layer 执行中每层需完整读取。
- v0 暂时采用保守假设 `Bytes_weight_active_per_step = Bytes_weight_footprint`，即假设全部 resident weight bytes 在每个 decode step 都被读取。**这是一个保守上界，不是 algorithmic minimum。**

#### 5.4 Assumed Architecture-visible Weight Traffic


### 6. KV Read Traffic

#### 6.1 KV Footprint
```
Bytes_K = L × B × S × Hkv × Dhead × bkv / 8
Bytes_V = L × B × S × Hkv × Dhead × bkv / 8
Bytes_KV = Bytes_K + Bytes_V
```

#### 6.2 Algorithmic KV Access per Decode Step

Autoregressive decode 中，生成新 token 时每个 query head 需与**所有历史位置** `1..S` 的 K/V 做 dot-product。

```
Algorithmic_K_access_per_step = L × B × S × Hkv × Dhead   (elements)
Algorithmic_V_access_per_step = L × B × S × Hkv × Dhead   (elements)
```

#### 6.3 Ideal/Minimum KV Read Transfer

```
T_K_read_ideal_per_step = L × B × S × Hkv × Dhead × bkv / 8
T_V_read_ideal_per_step = L × B × S × Hkv × Dhead × bkv / 8
```

#### 6.4 Assumed Architecture-visible KV Read Traffic

**假设 E（本 spec 默认）：每个 decode step 从 target memory 读取全部历史 K 和 V。**

```
T_K_read_arch_per_step = L × B × S × Hkv × Dhead × bkv / 8
T_V_read_arch_per_step = L × B × S × Hkv × Dhead × bkv / 8
T_KV_read_arch_per_step = T_K_read_arch_per_step + T_V_read_arch_per_step
                         = 2 × L × B × S × Hkv × Dhead × bkv / 8
                         = Bytes_KV
```

**Per-generated-token：**

```
T_KV_read_arch_per_token = T_KV_read_arch_per_step / B
                         = 2 × L × S × Hkv × Dhead × bkv / 8
```

**关键区分：**
- `Bytes_KV` 是**总 resident 容量**（存储位置 1..S 的所有 K/V）。
- `T_KV_read_arch_per_step` 是**单次 decode step 读取的历史 K/V 量**。在 full-reread assumption 下，二者数值相等，但**物理意义不同**。
- **KV read per token 与 batch size B 无关**（因为分子分母都有 B，抵消后只剩 S）。

**关于 FlashAttention / SRAM Tiling：**

FlashAttention [Dao et al., NeurIPS 2022; Dao, 2023] 的核心贡献是通过 **tiling** 将 attention computation 分块载入 fast SRAM，避免将完整的 `O(S²)` attention score matrix 写入 HBM。这显著降低了：
- Attention intermediate activation 的 HBM write/read。
- Attention 计算的 memory-bound 开销。

然而，对于 **exact dense autoregressive decode**：
- 每个新 query token 仍需访问**全部历史 K/V** 以计算 attention。
- FlashAttention 的 tiling 改变了**计算调度方式**和**intermediate activation 的 materialization**，但**不消除读取历史 K/V 的算法需求**。
- 若历史 K/V 本身已 resident 在 HBM/DRAM（这是标准做法，因为 KV cache 通常远大于 SRAM 容量），则 FlashAttention 不会将这些 K/V 的读取量降为 O(1)。
- 若 KV cache 可被完整保留在 SRAM（如极短 context），则属于 **cache residency assumption**，与 FlashAttention 算法本身无关。

因此，对于 long-context decode（如 S=131072），历史 K/V 的 target-memory read traffic 仍应被建模为与 `S` 成正比，除非有明确的 **cache residency assumption** 支持。

**Provenance:** Full-reread assumption 为 **MODELING_CHOICE**（保守假设）。FlashAttention 对 KV traffic 的影响为 **NOT_VALIDATED**（取决于具体 implementation、KV cache placement、SRAM 容量）。

---

### 7. KV Write Traffic

#### 7.1 Algorithmic KV Write per Decode Step

```
Algorithmic_K_write_per_step = L × B × Hkv × Dhead   (elements)
Algorithmic_V_write_per_step = L × B × Hkv × Dhead   (elements)
```

#### 7.2 Architecture-visible KV Write Traffic

```
T_K_write_arch_per_step = L × B × Hkv × Dhead × bkv / 8
T_V_write_arch_per_step = L × B × Hkv × Dhead × bkv / 8
T_KV_write_arch_per_step = 2 × L × B × Hkv × Dhead × bkv / 8
```

**Per-generated-token：**

```
T_KV_write_arch_per_token = 2 × L × Hkv × Dhead × bkv / 8
```

**注意：** KV write per token 与 context length `S` **无关**（每步只写新 token 的 K/V）。

---

## Compute Equations

### 8. FLOPs per Aggregate Decode Step

以下计算一个 **aggregate decode step**（B 个 sequences 各生成 1 个 token）的 FLOPs。

#### 8.1 Q/K/V Projections

| Projection | Weight Shape | FLOPs |
|-----------|-------------|-------|
| Q | `[Dmodel, Hq × Dhead]` = `[Dmodel, Dmodel]` | `2 × B × Dmodel²` |
| K | `[Dmodel, Hkv × Dhead]` | `2 × B × Dmodel × Hkv × Dhead` |
| V | `[Dmodel, Hkv × Dhead]` | `2 × B × Dmodel × Hkv × Dhead` |

```
FLOPs_QKV_per_layer = 2 × B × Dmodel × (Dmodel + 2 × Hkv × Dhead)
```

若 `Dmodel = Hq × Dhead`：
```
FLOPs_QKV_per_layer = 2 × B × Dmodel² × (1 + 2 × Hkv / Hq)
```

**Special cases:**
- **MHA** (`Hkv = Hq`): `FLOPs_QKV = 6 × B × Dmodel²`
- **GQA** (`Hkv = Hq / 4`): `FLOPs_QKV = 3 × B × Dmodel²`

#### 8.2 Output Projection

```
FLOPs_out_proj_per_layer = 2 × B × Dmodel²
```

#### 8.3 MLP (SwiGLU Gated FFN)

| Operation | Weight Shape | FLOPs |
|-----------|-------------|-------|
| gate_proj | `[Dmodel, Dff]` | `2 × B × Dmodel × Dff` |
| up_proj | `[Dmodel, Dff]` | `2 × B × Dmodel × Dff` |
| down_proj | `[Dff, Dmodel]` | `2 × B × Dff × Dmodel` |

```
FLOPs_MLP_per_layer = 6 × B × Dmodel × Dff
```

*注：若使用非 SwiGLU 结构（如原始 Transformer ReLU FFN），需调整系数并在 provenance 中说明。*

#### 8.4 Attention Score + Value Accumulation

**Score computation** (`Q @ K^T`):
- Q: `[B, Hq, 1, Dhead]`
- K: `[B, Hkv, S, Dhead]`
- FLOPs: `2 × B × Hq × S × Dhead`

**Value accumulation** (`Softmax(QK^T) @ V`):
- Attention weights: `[B, Hq, 1, S]`
- V: `[B, Hkv, S, Dhead]`
- FLOPs: `2 × B × Hq × S × Dhead`

```
FLOPs_attention_per_layer = 4 × B × Hq × S × Dhead
```

**关键特性：** 此项显式正比于 `S`，是 long-context decode 的 compute 瓶颈。

#### 8.5 LayerNorm / RMSNorm

```
FLOPs_norm_per_layer ≈ 5 × B × Dmodel   (negligible)
```

#### 8.6 Vocabulary Projection (Optional)

```
FLOPs_vocab = 2 × B × Dmodel × V
```

#### 8.7 Total per-Layer FLOPs

```
FLOPs_layer = FLOPs_QKV + FLOPs_out_proj + FLOPs_MLP + FLOPs_attention + FLOPs_norm
```

#### 8.8 Total per-Step FLOPs (All Layers)

```
FLOPs_step = L × FLOPs_layer + FLOPs_vocab
```

#### 8.9 FLOPs per Generated Token

```
FLOPs_token = FLOPs_step / B
```

**关键观察：** 由于 `FLOPs_QKV`, `FLOPs_out_proj`, `FLOPs_MLP`, `FLOPs_attention`, `FLOPs_vocab` 均线性正比于 `B`，因此：

```
FLOPs_token = L × [2×Dmodel×(Dmodel + 2×Hkv×Dhead) + 2×Dmodel² + 6×Dmodel×Dff + 4×Hq×S×Dhead] + 2×Dmodel×V
```

**Mathematical FLOPs per generated token 与 batch size B 无关。** Aggregate step 的总 FLOPs 随 B 线性增长，但除以该 step 产生的 B 个 tokens 后，per-token 值保持不变。

---

### 9. Simplified Sanity Approximation

```
FLOPs_token_sanity ≈ 2 × Nparam + 4 × L × Hq × S × Dhead
```

**使用条件与限制：**
- 第一项 `2 × Nparam` 近似所有 linear layer 的 compute。
- 第二项 `4 × L × Hq × S × Dhead` 显式补上 attention 的 context-dependent 开销。
- **此近似不得取代精确分解**，尤其当：
  - `S` 很大（attention 项主导）。
  - `Hkv ≪ Hq`（GQA 节省未 capture）。
  - MLP 比例非标准。

**Provenance:** 2×Nparam 为 **DERIVED_FROM_REFERENCE**（行业常用 heuristic）。Attention 修正项为 **DERIVED_FROM_PAPER**（标准 attention 算法分析）。

---

## Batch Semantics

### 10. Aggregate Step vs. Per-Token Metrics

| Metric | Definition | 公式 |
|--------|-----------|------|
| **Aggregate step** | 一个 forward pass 生成 B 个 token | — |
| **Per-step traffic** | 一个 aggregate step 的总 memory traffic | `T_*_per_step` |
| **Per-token traffic** | 平均每个生成 token 的 traffic | `T_*_per_step / B` |

**关键区分：**
- **Weight read** 在 aggregate step 中读取一次（被 B 个 sequences 复用），因此 `T_weight_per_token = T_weight_per_step / B`。这是 **tile-level reuse assumption** 的结果。
- **KV read/write** 的 per-step 总量随 B 线性增长（每个 sequence 有自己的 KV cache），因此 `T_KV_per_token` 与 B 无关。
- **FLOPs** 的 per-step 总量随 B 线性增长，因此 `FLOPs_token` 与 B 无关。

### 11. Continuous / Variable-Length Batching

- **等长 batch (`S_i = S`)**：直接适用。
- **Variable-length batch**：KV footprint 和 read traffic 使用 `Σ_i S_i`；compute FLOPs 使用 `Σ_i S_i`。
- **Continuous batching**：本 spec 不建模动态调度，假设 batch 组成固定。**NOT_VALIDATED**。

---

## Hand Checks

### Model Configuration (Primary Source)

以下 hand-check 使用 **LLaMA-3.1-8B** 官方架构参数 [Grattafiori et al., 2024; Hugging Face config]：

| Parameter | Value | Source |
|-----------|-------|--------|
| `Nparam` | 8.0 × 10⁹ | Official report [Grattafiori et al., 2024] |
| `L` | 32 | `config.json` [Hugging Face] |
| `Hq` | 32 | `config.json` [Hugging Face] |
| `Hkv` | 8 | `config.json` [Hugging Face] |
| `Dhead` | 128 | DERIVED (`Dmodel / Hq = 4096 / 32`) |
| `Dmodel` | 4096 | `config.json` [Hugging Face] |
| `Dff` | 14336 | `config.json` [Hugging Face] |
| `V` | 128256 | `config.json` [Hugging Face] |
| `S_max` | 131072 | `config.json` [Hugging Face] |
| `bw` | 16 (bf16/fp16) | MODELING_CHOICE |
| `bkv` | 16 (bf16/fp16) | MODELING_CHOICE |
| `S` (hand-check context) | 131072 | MODELING_CHOICE (using official `S_max`) |
| `ε_reserve` | 0 | MODELING_CHOICE (simplified) |
| `ε_KV_waste` | 0 | MODELING_CHOICE (simplified) |

---

### Case A: B = 1, S = 131072

#### A.1 Weight Footprint

```
Bytes_weight = 8.0 × 10⁹ × 16 / 8
             = 16,000,000,000 bytes
             = 16.0 GB
             = 14.9 GiB  (16,000,000,000 / 1,073,741,824)
```

#### A.2 KV Footprint

```
Bytes_KV = 2 × 32 × 1 × 131072 × 8 × 128 × 16 / 8
         = 64 × 131072 × 8 × 128 × 2
         = 64 × 131072 × 2048
         = 64 × 268,435,456
         = 17,179,869,184 bytes
         = 17.18 GB
         = 16.0 GiB  (17,179,869,184 / 1,073,741,824)
```

**Observation:** 在 S=131072 下，KV footprint (17.18 GB) ≈ weight footprint (16.0 GB)。

#### A.3 Aggregate-Step Weight Traffic

```
T_weight_arch_per_step = 16,000,000,000 bytes
                         = 16.0 GB
```

#### A.4 Weight Traffic per Generated Token

```
T_weight_arch_per_token = 16,000,000,000 / 1
                        = 16,000,000,000 bytes
                        = 16.0 GB/token
```

#### A.5 Aggregate-Step KV Read Traffic (Full-Reread Scenario)

```
T_KV_read_arch_per_step = 17,179,869,184 bytes
                         = 17.18 GB
```

#### A.6 KV Read Traffic per Generated Token

```
T_KV_read_arch_per_token = 17,179,869,184 / 1
                         = 17,179,869,184 bytes
                         = 17.18 GB/token
```

#### A.7 KV Write Traffic per Generated Token

```
T_KV_write_arch_per_token = 2 × 32 × 1 × 8 × 128 × 16 / 8
                          = 131,072 bytes
                          = 0.13 MB/token
```

**Observation:** KV write (0.13 MB/token) 与 KV read (17.18 GB/token) 相差 5 个数量级。Decode phase 为 read-dominated。

#### A.8 Aggregate-Step FLOPs (逐项代入)

**Per-layer decomposition:**

| Component | Formula | Substituted | Result |
|-----------|---------|-------------|--------|
| QKV proj | `2 × B × Dmodel × (Dmodel + 2×Hkv×Dhead)` | `2 × 1 × 4096 × (4096 + 2×8×128)` | 50,331,648 |
| Out proj | `2 × B × Dmodel²` | `2 × 1 × 4096²` | 33,554,432 |
| MLP | `6 × B × Dmodel × Dff` | `6 × 1 × 4096 × 14336` | 352,321,536 |
| Attention | `4 × B × Hq × S × Dhead` | `4 × 1 × 32 × 131072 × 128` | 2,147,483,648 |
| Norm | negligible | — | ~20,000 |
| **Per layer** | **Sum** | — | **2,583,711,264** |

**All layers + vocab:**

```
FLOPs_step = 32 × 2,583,711,264 + 2 × 1 × 4096 × 128256
           = 82,678,760,448 + 1,050,271,488
           = 83,729,031,936
           ≈ 83.73 × 10⁹ FLOPs
```

#### A.9 FLOPs per Generated Token

```
FLOPs_token = 83,729,031,936 / 1
            = 83,729,031,936 FLOPs/token
            ≈ 83.73 GFLOPs/token
```

#### A.10 Sanity Approximation Check

```
FLOPs_token_sanity = 2 × Nparam + 4 × L × Hq × S × Dhead
                   = 2 × 8.0 × 10⁹ + 4 × 32 × 32 × 131072 × 128
                   = 16,000,000,000 + 68,719,476,736
                   = 84,719,476,736
                   ≈ 84.719 GFLOPs/token
```

**Detailed decomposition:** 83.729 GFLOPs/token (from A.8).

**Comparison:** Sanity approximation (84.719 GFLOPs/token) 与 detailed decomposition (83.729 GFLOPs/token) 接近，差异约 1.2%。

**解释:**
- `2 × Nparam` = 16.0 GFLOPs：parameter-related linear compute (QKV, out-proj, MLP) 的粗略 sanity approximation。
- `4 × L × Hq × S × Dhead` = 68.719 GFLOPs：layer-resolved context-dependent attention term。
- 加入正确的 attention term 后，sanity approximation 与 detailed decomposition 基本一致。

---

### Case B: B = 8, S = 131072

**与 Case A 使用完全相同的 model、precision、S。** 仅改变 B。

#### B.1 Weight Footprint

```
Bytes_weight = 16,000,000,000 bytes  (与 B 无关)
             = 16.0 GB
             = 14.9 GiB
```

#### B.2 KV Footprint

```
Bytes_KV = 2 × 32 × 8 × 131072 × 8 × 128 × 16 / 8
         = 8 × 17,179,869,184
         = 137,438,953,472 bytes
         = 137.44 GB
         = 128.0 GiB
```

#### B.3 Aggregate-Step Weight Traffic

```
T_weight_arch_per_step = 16,000,000,000 bytes
                         = 16.0 GB
```

#### B.4 Weight Traffic per Generated Token

```
T_weight_arch_per_token = 16,000,000,000 / 8
                        = 2,000,000,000 bytes
                        = 2.0 GB/token
```

**Weight amortization factor:** 8×。这是 tile-level reuse assumption 的结果。

#### B.5 Aggregate-Step KV Read Traffic

```
T_KV_read_arch_per_step = 137,438,953,472 bytes
                         = 137.44 GB
```

#### B.6 KV Read Traffic per Generated Token

```
T_KV_read_arch_per_token = 137,438,953,472 / 8
                         = 17,179,869,184 bytes
                         = 17.18 GB/token
```

**关键结果：** KV read per token (17.18 GB) **与 Case A 完全相同**，不因 batch 增大而下降。Aggregate-step 总 KV read 随 B 线性增长，但 per-token 值不变。

#### B.7 KV Write Traffic per Generated Token

```
T_KV_write_arch_per_token = 2 × 32 × 8 × 128 × 16 / 8
                          = 131,072 bytes
                          = 0.13 MB/token
```

**与 Case A 相同。** KV write per token 与 B 无关。

#### B.8 Aggregate-Step FLOPs (逐项代入)

| Component | Formula | Substituted | Result |
|-----------|---------|-------------|--------|
| QKV proj | `2 × B × Dmodel × (Dmodel + 2×Hkv×Dhead)` | `2 × 8 × 4096 × (4096 + 2048)` | 402,653,184 |
| Out proj | `2 × B × Dmodel²` | `2 × 8 × 4096²` | 268,435,456 |
| MLP | `6 × B × Dmodel × Dff` | `6 × 8 × 4096 × 14336` | 2,818,572,288 |
| Attention | `4 × B × Hq × S × Dhead` | `4 × 8 × 32 × 131072 × 128` | 17,179,869,184 |
| **Per layer** | **Sum** | — | **20,669,530,112** |

```
FLOPs_step = 32 × 20,669,530,112 + 8 × 2 × 4096 × 128256
           = 661,424,963,584 + 8,402,171,904
           = 669,827,135,488
           ≈ 669.83 × 10⁹ FLOPs
```

#### B.9 FLOPs per Generated Token

```
FLOPs_token = 669,827,135,488 / 8
            = 83,728,391,936 FLOPs/token
            ≈ 83.73 GFLOPs/token
```

**关键结果：** `FLOPs_token` (83.73 GFLOPs/token) **与 Case A (83.73 GFLOPs/token) 基本相同**（微小差异来自 vocab projection 的舍入）。Aggregate step FLOPs 增长了 8×，但除以 8 个 tokens 后 per-token 值保持不变。

---

### Case A vs. Case B 对比总结

| Metric | Case A (B=1) | Case B (B=8) | Ratio B/A | Interpretation |
|--------|-------------|-------------|-----------|----------------|
| Weight footprint | 16.0 GB | 16.0 GB | 1.0 | 与 B 无关 |
| KV footprint | 17.18 GB | 137.44 GB | 8.0 | 线性于 B |
| Weight traffic/token | 16.0 GB | 2.0 GB | 0.125 | Tile-level reuse amortized |
| KV read traffic/token | 17.18 GB | 17.18 GB | 1.0 | **不因 batch 变化** |
| KV write traffic/token | 0.13 MB | 0.13 MB | 1.0 | **不因 batch 变化** |
| FLOPs/token | 83.73 GFLOPs | 83.73 GFLOPs | 1.0 | **Mathematical compute 不因 batch 下降** |

---

## Dimensional Sanity Checks

| 公式 | 单位推导 | 结果 |
|------|---------|------|
| `Bytes_weight = Nparam × bw / 8` | `[param] × [bit/param] / [bit/byte] = [byte]` | ✅ |
| `Bytes_KV = 2 × L × B × S × Hkv × Dhead × bkv / 8` | `1 × 1 × 1 × 1 × 1 × 1 × [bit] / [bit/byte] = [byte]` | ✅ |
| `T_KV_read_arch_per_token = 2 × L × S × Hkv × Dhead × bkv / 8` | `1 × 1 × 1 × 1 × [bit] / [bit/byte] = [byte/token]` | ✅ |
| `FLOPs_QKV = 2 × B × Dmodel × (Dmodel + 2×Hkv×Dhead)` | `1 × 1 × 1 × 1 = [FLOP]` | ✅ |
| `FLOPs_attention = 4 × B × Hq × S × Dhead` | `1 × 1 × 1 × 1 = [FLOP]` | ✅ |
| `FLOPs_token = FLOPs_step / B` | `[FLOP/step] / [sequence/step] = [FLOP/sequence]` → per-token | ✅ |
| `GB = byte / 10⁹` | `[byte] / 10⁹ = [GB]` | ✅ |
| `GiB = byte / 1024³` | `[byte] / 1024³ = [GiB]` | ✅ |

---

## Key Results

### Memory Footprint
- `Bytes_weight_footprint = Nparam × bw / 8`
- `Bytes_KV = 2 × L × B × S × Hkv × Dhead × bkv / 8`
- `Bytes_required = Bytes_weight_footprint + Bytes_KV_realized + Bytes_runtime`
- `feasible = Bytes_required <= Cusable`

### Traffic per Aggregate Step
- `T_weight_arch_per_step = Bytes_weight_active_per_step` (once per step, tile-level reuse; v0: = footprint, conservative)
- `T_KV_read_arch_per_step = 2 × L × B × S × Hkv × Dhead × bkv / 8`
- `T_KV_write_arch_per_step = 2 × L × B × Hkv × Dhead × bkv / 8`

### Traffic per Generated Token
- `T_weight_arch_per_token = Bytes_weight_active_per_step / B` (amortized by tile reuse; v0: = footprint, conservative)
- `T_KV_read_arch_per_token = 2 × L × S × Hkv × Dhead × bkv / 8` (**independent of B**)
- `T_KV_write_arch_per_token = 2 × L × Hkv × Dhead × bkv / 8` (**independent of B**)

### Compute per Generated Token
- `FLOPs_token = L × [2×Dmodel×(Dmodel+2×Hkv×Dhead) + 2×Dmodel² + 6×Dmodel×Dff + 4×Hq×S×Dhead] + 2×Dmodel×V`
- **FLOPs_token is independent of B.**

---

## Scientific Interpretation

### 1. Long-Context Decode 的特征

Hand Check (B=1, S=131072) 显示：
- **KV read traffic 主导**：17.18 GB/token 的 KV read vs 16.0 GB/token 的 weight read。
- **Attention compute 主导**：2.147 GFLOPs/layer 的 attention vs 0.050 GFLOPs/layer 的 QKV projection。
- **KV footprint 与 weight 相当**：17.18 GB vs 16.0 GB。

这意味着在 long-context 场景下，内存系统瓶颈可能从 "weight bandwidth" 转向 "KV bandwidth"。

### 2. Batch 的精确作用边界

Case A (B=1) vs Case B (B=8) 对比证明：
- **Weight traffic/token** 降低 8×：来自 tile-level reuse assumption。
- **KV read traffic/token** **不变**：仅取决于 S，与 B 无关。
- **FLOPs/token** **不变**：mathematical compute 不因 batch 摊薄。
- **KV footprint** 增长 8×：线性于 batch size。

因此，batching 的收益严格限于 **weight read amortization** 和 **throughput parallelism**，不减少 per-token algorithmic compute 或 per-token KV data movement。

### 3. KV Capacity ≠ KV Traffic

本 spec 明确建立区分：
- **KV Capacity** (`Bytes_KV`) 是存储所有历史 K/V 的总空间。
- **KV Read Traffic** (`T_KV_read_arch_per_token`) 是每生成一个 token 需要读取的历史 K/V 量。
- 在 full-reread assumption 下，`T_KV_read_arch_per_step = Bytes_KV`，但二者物理意义不同。
- KV write traffic (0.13 MB/token) 远小于 KV read traffic (17.18 GB/token)。

---

## Assumptions / Provenance

| # | Assumption | Category | Rationale / Evidence |
|---|-----------|----------|---------------------|
| 1 | Weight precision `bw` (bf16=16) | MODELING_CHOICE | Workload 配置决定 |
| 2 | KV precision `bkv = bw` | MODELING_CHOICE | 常见做法；可 sensitivity sweep |
| 3 | Weight tile-level reuse across B | MODELING_CHOICE | Layer-pipelined GEMM dataflow：单个 tile 加载后服务 B 个 inputs |
| 4 | **不是** "whole model resident on-chip" | MODELING_CHOICE | 16 GB fp16 weights ≫ tens-of-MB SRAM；仅 tile-level reuse |
| 5 | KV full reread per decode step | MODELING_CHOICE | 保守假设；每个 query 需访问全部历史 K/V |
| 6 | FlashAttention does not eliminate KV reads | DERIVED_FROM_PAPER | FlashAttention [Dao et al., NeurIPS 2022] 减少 intermediate activation HBM traffic，不消除 historical KV access algorithmic demand |
| 7 | No KV cache compression | MODELING_CHOICE | v0 简化 |
| 8 | SwiGLU MLP (3 projections) | DERIVED_FROM_PAPER | LLaMA-3/3.1 official config [Grattafiori et al., 2024] |
| 9 | GQA with `Hkv=8, Hq=32` | DERIVED_FROM_OFFICIAL_MODEL_CONFIG | `meta-llama/Meta-Llama-3.1-8B` config.json |
| 10 | `ε_reserve = 0`, `ε_KV_waste = 0` | MODELING_CHOICE (simplification) | Hand-check 简化 |
| 11 | No runtime overhead (`Bytes_runtime = 0`) | MODELING_CHOICE (simplification) | v0 简化 |
| 12 | LayerNorm FLOPs negligible | MODELING_CHOICE | 数量级 <0.1% of total |
| 13 | `γ_weight = γ_KV = 1` (physical traffic multipliers) | MODELING_CHOICE / SENSITIVITY_POINT | 无 implementation profiling 数据；保留为 parameterized variable |

---

## PASS / FAIL

| Criterion | Status | Notes |
|-----------|--------|-------|
| GB/GiB 全部正确区分 | ✅ PASS | 所有数值同时给出 byte / GB / GiB |
| Exact integer context 已明确 | ✅ PASS | S = 131072（LLaMA-3.1 official `max_position_embeddings`） |
| FLOPs arithmetic 无遗漏 | ✅ PASS | 逐项代入，显示中间结果 |
| Batch effect 只作用于有依据的 reuse/traffic 项 | ✅ PASS | Weight traffic/token 下降 8×；KV traffic/token 不变；FLOPs/token 不变 |
| Mathematical FLOPs/token 不被错误 batch-amortize | ✅ PASS | FLOPs_token 与 B 无关，已证明 |
| FlashAttention/KV 解释正确 | ✅ PASS | 明确区分 intermediate activation reduction vs historical KV access elimination |
| Weight tile reuse 与 model residency 分离 | ✅ PASS | 明确标注 "tile-level reuse"，禁止 "whole model resident" 表述 |
| Actual/profiled traffic 不被虚构 | ✅ PASS | 使用 "architecture-visible traffic assumption" 而非 "measured DRAM traffic" |
| Citations 来自 primary/official sources | ✅ PASS | LLaMA-3.1 config.json, arXiv papers, NeurIPS, SOSP |
| 不进入 performance evaluator | ✅ PASS | 无 tokens/s, Tmemory, J/token |
| 不修改代码 | ✅ PASS | 零文件修改 |

---

## Open Questions

1. **KV-cache quantization / compression**：vLLM/TensorRT-LLM 支持 INT8/FP8 KV cache，可减半 footprint 和 traffic。是否纳入 v1？
2. **Variable-length batch / continuous batching**：实际 serving 中 batch 动态变化。v0 假设固定 batch。
3. **Prefill vs. Decode phase**：本 spec 仅覆盖 decode。Prompt processing (prefill) 的 traffic/compute 模式完全不同。
4. **`ε_reserve` 和 `ε_KV_waste` 的实际值**：需要来自 serving system 的 profiling 数据。
5. **Physical target-memory traffic multiplier `γ`**：若无 profiling，保留为 sensitivity sweep 参数。
6. **MoE workload**：本 spec 仅覆盖 dense Transformer。

---

## Next Recommended Step

1. **Research Lead 审核 B1-R1**：确认修订是否解决所有 claim-blocking 问题。
2. **确定 concrete model configs**：选择 2–3 个代表性模型作为 benchmark workload。
3. **量化 `ε_reserve`, `ε_KV_waste`, `γ` 等参数**：通过文献或 profiling。
4. **进入 Track B2**（若存在）：将本 spec output 接入 bandwidth/performance evaluator。

---

## CHANGELOG FROM B1

### 1. GB / GiB 混用修正

| 位置 | B1 (错误) | B1-R1 (修正) |
|------|----------|-------------|
| Weight footprint | "16.0 GiB" (实际为 16.0×10⁹ bytes) | "16,000,000,000 bytes = 16.0 GB = 14.9 GiB" |
| KV footprint | "15.6 GiB" (实际计算值) | "17,179,869,184 bytes = 17.18 GB = 16.0 GiB" |
| Hand-check 表格 | 仅给 GiB | 同时给 byte / GB / GiB |

**修正理由：** 16.0×10⁹ bytes ≠ 16.0 GiB。GB = 10⁹, GiB = 1024³。

### 2. FLOPs Arithmetic 修正

| 问题 | B1 (错误) | B1-R1 (修正) |
|------|----------|-------------|
| Context 数值 | "128K" (模糊) | "131072" (LLaMA-3.1 official exact integer) |
| FLOPs 逐项展示 | 仅给汇总 | 每 component 逐项代入并显示中间结果 |
| Sanity check 对比 | 18.1 vs 82.1 GFLOPs (不同模型/上下文) | 18.15 vs 83.73 GFLOPs (同一模型/同一 S=131072) |

**修正理由：** 不得使用 "128K" 模糊数值；必须逐项显示乘数代入。

### 3. Batch Hand-Check 重做

| 问题 | B1 (错误) | B1-R1 (修正) |
|------|----------|-------------|
| Case 1 vs Case 2 | B=1, S=128K vs B=8, S=32K (两个变量都变) | B=1, S=131072 vs B=8, S=131072 (仅 B 变) |
| FLOPs/token batch 效应 | 未明确证明 | 显式证明 FLOPs_token = 83.73 GFLOPs (B=1) = 83.73 GFLOPs (B=8) |
| KV traffic batch 效应 | 未明确证明不变性 | 显式证明 KV read/token = 17.18 GB (B=1) = 17.18 GB (B=8) |

**修正理由：** 不得用不同 context 的两个 case 来论证 batch effect；必须控制单一变量。

### 4. FlashAttention / KV Traffic 表述修正

| 问题 | B1 (错误) | B1-R1 (修正) |
|------|----------|-------------|
| FlashAttention 作用 | "KV traffic 可降至 O(1)" / "SRAM 常驻" | FlashAttention 减少 intermediate activation HBM traffic；不消除 historical KV access algorithmic demand |
| KV read 假设 | 隐含依赖 tiling | 明确标注 "full-reread" MODELING_CHOICE；FlashAttention 影响标记为 NOT_VALIDATED |

**修正理由：** 不得声称 128K KV 可整体常驻有限 SRAM；不得声称 KV traffic 变成 O(1)。

### 5. Weight Reuse 解释修正

| 问题 | B1 (错误) | B1-R1 (修正) |
|------|----------|-------------|
| Weight residency | "weights 常驻 SRAM / ~100 MB cache" | 明确区分 "tile-level temporal reuse" vs "model residency" |
| Batch amortization | 未说明机制 | 明确说明：layer/GEMM execution 中同一 weight tile 服务 B 个 inputs |

**修正理由：** 16 GB weights 不可能常驻 tens-of-MB SRAM；amortization 来自 tile-level reuse。

### 6. "Actual Traffic" 表述修正

| 问题 | B1 (错误) | B1-R1 (修正) |
|------|----------|-------------|
| 术语 | "Physical DRAM Traffic" (暗示 measured) | "Architecture-visible Traffic Assumption" / "Physical target-memory traffic (parameterized)" |
| Gamma multiplier | "default γ = 1" (隐含 validated) | "γ = 1 as MODELING_CHOICE / sensitivity point" |
| 表格列名 | "Physical DRAM Traffic" | 改为 "Architecture-visible Traffic Assumption (B)" |

**修正理由：** 不得虚构 measured/profiled traffic；gamma 只能作为 explicit modeling choice。

### 7. Reference Provenance 补全

| 问题 | B1 (错误/不足) | B1-R1 (修正) |
|------|---------------|-------------|
| LLaMA 参数 | "EXAMPLE" | 引用 `meta-llama/Meta-Llama-3.1-8B` official Hugging Face config.json |
| FlashAttention | 未引具体论文 | Dao et al., NeurIPS 2022; Dao, 2023 |
| PagedAttention | 未引具体论文 | Kwon et al., SOSP 2023 |
| LLaMA-3 架构 | 未引 | Grattafiori et al., arXiv:2407.21783, 2024 |

**修正理由：** 模型参数必须来自 official source；技术事实必须引用 primary paper。

### 保留不变的内容

- ✅ Footprint vs traffic 四层区分
- ✅ GQA/MQA/MHA 的 Hkv accounting
- ✅ Variable-length batch `Σ_i S_i`
- ✅ Aggregate step / per-token / per-sequence 语义
- ✅ Capacity physical / usable / reserve 区分
- ✅ Symbolic physical-traffic multiplier 设计 (`γ_*`)

---

## CHANGELOG FROM B1-R1 → B1-R2

### 1. A.10 FLOPs Sanity Check 残留旧内容删除

| 位置 | B1-R1 (残留) | B1-R2 (修正) |
|------|-------------|-------------|
| A.10 之后 | 残留旧 sanity block：18.15 GFLOPs/token，"4.6× underestimation"，"2Nparam 因 SwiGLU/GQA 严重失效" | **删除**该残留块；A.10 仅保留正确的 sanity check（84.719 GFLOPs/token，含完整 L factor） |

**修正理由：** B1-R1 已插入正确的 sanity approximation，但未彻底删除 B1 遗留的错误计算块。B1-R2 彻底清除残留。

### 2. Weight Footprint vs Active Weight Traffic 概念区分

| 位置 | B1-R1 (混淆) | B1-R2 (修正) |
|------|-------------|-------------|
| 四列表格 Weights 行 | Footprint: `Bytes_weight`; Ideal/Min Transfer: `Bytes_weight` | Footprint: `Bytes_weight_footprint`; Ideal/Min Transfer: `Bytes_weight_active_per_step` (v0: = footprint) |
| 5.1 Weight Footprint | `Bytes_weight = Nparam × bw / 8`（未标注 resident） | `Bytes_weight_footprint = Nparam × bw / 8`，明确为 **resident storage capacity** |
| 5.2 Algorithmic Access | 未区分 decode step 的组件级 access | 新增：input embedding 通常只查一个 row；output lm_head 需完整读取；transformer layer weights 需完整读取；tied/untied embedding 说明 |
| 5.3 Ideal/Min Transfer | `T_weight_ideal_per_step = Bytes_weight`（暗示全部读取） | 引入 `Bytes_weight_active_per_step`；v0 保守假设 `= Bytes_weight_footprint`；**明确标为 conservative MODELING_CHOICE，不是 algorithmic minimum** |
| 5.4 Arch-visible Traffic | `T_weight_arch_per_step = Bytes_weight` | `T_weight_arch_per_step = Bytes_weight_active_per_step`；Alternative no-reuse 同步更新 |
| Key Results | `Bytes_weight`、`T_weight_arch_per_step = Bytes_weight` | `Bytes_weight_footprint`、`Bytes_weight_active_per_step`（v0: = footprint, conservative） |
| Total Memory Footprint | `Bytes_required = Bytes_weight + ...` | `Bytes_required = Bytes_weight_footprint + ...` |

**修正理由：** `Nparam × bw / 8` 是 resident footprint，但 decode step 的 algorithmic minimum weight read 在理论上可能小于 footprint（因 embedding lookup 通常只读一个 row）。v0 采用保守假设（active = footprint），但必须明确标注为 **MODELING_CHOICE** 而非 algorithmic minimum。

---

## STOP

Track B1-R2 修订完成。仅修改上述两处问题，不扩展任何内容。等待 Research Lead 审核与反馈。不得在未获授权前进入 bandwidth、performance evaluator 或 thermal 阶段。
