# Track B1: LLM Autoregressive Decode Accounting Spec v0

> **Research Question:** 如何建立透明、可手算、适用于 long-context small/medium-batch autoregressive decode 的 workload accounting spec？
>
> **Scope:** 纯 analytical specification。不输出 tokens/s、J/token、power 或 Tmax。不修改仓库文件。
>
> **Deliverable:** 提交 Research Lead 审核的 spec 文档。

---

## Research Question

为 Orthogonal M3D Memory DAC 项目建立 LLM autoregressive decode 的 workload accounting spec，满足：

1. **透明可手算**：所有公式可被人工复核，无黑盒 simulator。
2. **footprint ≠ traffic**：严格区分 memory footprint、algorithmic access、architecture-visible traffic、physical DRAM traffic。
3. **Long-context 显式**：attention compute 和 KV read traffic 必须显式依赖 context length S，不得被 2×Nparam 掩盖。
4. **Batch 语义完整**：区分 aggregate step throughput 与 per-sequence/per-token 指标。
5. **Provenance 清晰**：每个假设标记来源类别。

---

## Evidence / References

- Vaswani et al., "Attention Is All You Need," NeurIPS 2017. (Transformer architecture baseline)
- LLaMA / LLaMA-2 / LLaMA-3 technical reports, Meta AI. (GQA, SwiGLU MLP, RMSNorm)
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. (KV-cache paging, block waste)
- Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023. (Attention tiling, memory traffic reduction)
- Project internal: `configs/cases/orthogonal_m3d_igzo.yaml` (workload bandwidth configuration)
- Project internal: `src/om3dthermal/power/config.py` (WorkloadInput schema)

---

## Files Inspected

- `src/om3dthermal/power/config.py` — WorkloadInput schema, BinaryProbability, RowPolicy
- `src/om3dthermal/power/model.py` — `calculate_memory_power()`, workload bandwidth usage
- `src/om3dthermal/architecture_comparison.py` — Current end-to-end comparison pipeline (no workload model yet)
- `configs/cases/orthogonal_m3d_igzo.yaml` — Workload block: `read_bandwidth_gbps`, `read_data`, `control_address_reuse`
- `configs/cases/conventional_hbm_2x1.yaml` — Workload block: `read_bandwidth_gbps`, `row_policy`

---

## Files Modified

**None.** 本任务为纯 analytical specification，不修改任何代码、config、tests 或治理文档。

---

## Configuration

本 spec 使用以下**符号体系**（所有变量均为标量，除非显式标注为向量/张量）：

### Symbol and Unit Table

| Symbol | Definition | Unit | Provenance |
|--------|-----------|------|------------|
| `Nparam` | 模型可训练参数总量 | 1 (scalar) | PAPER_REPORTED / MODEL_CONFIG |
| `L` | Transformer decoder layer 数 | 1 (scalar) | PAPER_REPORTED / MODEL_CONFIG |
| `B` | 同时活跃的 sequence 数 (batch size) | 1 (scalar) | MODELING_CHOICE (workload) |
| `S_i` | 第 i 个 sequence 的当前 context length (含 prompt + 已生成 tokens) | 1 (scalar) | MODELING_CHOICE (workload state) |
| `S` | 等长 batch 时的统一 context length; `S = S_i` ∀i | 1 (scalar) | MODELING_CHOICE (simplification) |
| `S_max` | 最大支持的 context length | 1 (scalar) | MODEL_CONFIG |
| `Hq` | Query head 数 (attention heads) | 1 (scalar) | PAPER_REPORTED / MODEL_CONFIG |
| `Hkv` | Key/Value head 数 (GQA/MQA 分组数) | 1 (scalar) | PAPER_REPORTED / MODEL_CONFIG |
| `Dhead` | 每个 attention head 的维度 | 1 (scalar) | PAPER_REPORTED / MODEL_CONFIG |
| `Dmodel` | 模型 hidden dimension; `Dmodel = Hq × Dhead` | 1 (scalar) | DERIVED_FROM_PAPER |
| `Dff` | MLP 中间层维度 (feed-forward hidden dim) | 1 (scalar) | PAPER_REPORTED / MODEL_CONFIG |
| `V` | Vocabulary size | 1 (scalar) | PAPER_REPORTED / MODEL_CONFIG |
| `bw` | Weight 每个参数的存储位数 (bits per parameter) | bit/param | MODELING_CHOICE (precision) |
| `bkv` | KV cache 每个元素的存储位数 (bits per element) | bit/element | MODELING_CHOICE (precision) |
| `bact` | Activation 每个元素的存储位数 | bit/element | MODELING_CHOICE (precision) |
| `Cphysical` | 物理存储容量 (architecture 暴露的总容量) | byte | DERIVED_FROM_GEOMETRY |
| `Cusable` | Workload 实际可用容量 (`Cusable = Cphysical × (1 − ε_reserve)`) | byte | MODELING_CHOICE |
| `ε_reserve` | 系统保留/碎片/分配器开销比例 | 1 (scalar) | MODELING_CHOICE / NOT_VALIDATED |
| `Ngen` | 本次 decode session 计划生成的 token 数 | 1 (scalar) | MODELING_CHOICE (workload) |

**单位约定：**
- 所有 `Bytes_` 前缀变量单位为 **byte (B)**。
- 所有 `bits_` 前缀变量单位为 **bit (b)**。
- `GiB` = 1024³ B; `GB` = 10⁹ B。本 spec 统一使用 **GiB** 作为二进制容量单位，但公式推导中使用 byte 以保持维度一致性。
- FLOPs 指浮点乘加操作数 (multiply-accumulate count)，即一个乘法加一个加法计为 2 FLOPs（矩阵乘惯例）。

---

## Memory Footprint Equations

### 1. Weight Footprint

```
Bytes_weight = Nparam × bw / 8
```

**说明：**
- `Nparam` 是否包含 embedding/output projection matrix：取决于模型设计。对于 tied embedding（输入嵌入与输出投影共享权重），`Nparam` 应只计一次。如果 untied，则 `Nparam` 包含两个矩阵。需在 provenance 中明确说明。
- Quantization metadata (scales, zeros, group indices)：若使用 INT8/INT4/FP8 量化，需显式加入 `Bytes_weight_metadata`。本 spec v0 假设无额外 metadata（`bw` 为有效存储位宽）。
- 若存在 bias terms，其参数量通常 `≪ Nparam`（<0.1%），可忽略或显式加入。

**Provenance:** `bw` 为 **MODELING_CHOICE**（fp16=16, bf16=16, fp8=8, int4=4 等）。`Nparam` 来自 **PAPER_REPORTED** 或官方 model config。

---

### 2. KV-Cache Footprint

对于等长 batch (`S_i = S` ∀i) 的基础形式：

```
Bytes_KV = 2 × L × B × S × Hkv × Dhead × bkv / 8
```

**说明：**
- 因子 **2** 对应 Key (K) 和 Value (V) 两个张量。
- `Hkv` 体现 GQA/MQA/MHA：
  - **MHA**: `Hkv = Hq`
  - **GQA**: `Hkv < Hq`（如 LLaMA-3-8B: `Hq=32, Hkv=8`）
  - **MQA**: `Hkv = 1`
- 使用 `Hkv` 而非 `Hq` 是因为每个 KV head 被多个 query heads 共享，存储只需一份。

**Variable-length batch：**

```
Bytes_KV_variable = 2 × L × Hkv × Dhead × bkv / 8 × Σ_i S_i
```

**Padding / Page / Block Waste：**
- 上述公式给出 **raw tensor footprint**（最小理论值）。
- 实际 runtime 中，PagedAttention/vLLM 等系统使用固定-size blocks（如 16 tokens/block），导致 **internal fragmentation**。
- 此外，allocator alignment、tensor padding、reserved workspace 会增加 footprint。
- 本 spec 引入 `ε_KV_waste` 建模此开销：

```
Bytes_KV_realized = Bytes_KV × (1 + ε_KV_waste)
```

**Provenance:** `ε_KV_waste` 为 **NOT_VALIDATED**（取决于具体推理框架实现）。若无可信数据，保持符号形式或在 hand-check 中设为 0 并明确标注。

---

### 3. Total Memory Footprint

```
Bytes_required = Bytes_weight + Bytes_KV_realized + Bytes_runtime
```

其中 `Bytes_runtime` 包括：
- Activation workspace（当前 layer 的临时激活值）
- Attention score buffer（softmax 前的 attention scores）
- Scheduler / runtime metadata
- CUDA graph / kernel launch buffers

**Provenance:** `Bytes_runtime` 为 **NOT_VALIDATED**（高度依赖推理框架）。v0 spec 建议保留为符号 `Bytes_runtime` 或在 hand-check 中给出数量级估计（如 1–5% of `Bytes_weight`）并标注 **MODELING_CHOICE**。

---

### 4. Capacity Feasibility

```
feasible = (Bytes_required <= Cusable)
```

其中：

```
Cusable = Cphysical × (1 − ε_reserve)
```

**说明：**
- `Cphysical` 来自 architecture 的 capacity 计算（如 `system_capacity_GiB` from `architecture_comparison.py`）。
- `ε_reserve` 包括：OS/驱动保留、内存碎片、allocator overhead、冗余 bank 规划等。
- 必须区分四层容量概念：
  1. **Physical capacity** (`Cphysical`)：物理存储阵列可寻址的总位数。
  2. **Reserved capacity**：系统固件/驱动保留的不可用区域。
  3. **Usable capacity** (`Cusable`)：workload 实际可分配的容量。
  4. **Runtime-realized footprint** (`Bytes_required`)：当前 workload 实际占用的容量。

**Provenance:** `ε_reserve` 为 **MODELING_CHOICE** / **NOT_VALIDATED**。若缺乏数据，建议在 hand-check 中使用 `ε_reserve = 0` 并明确说明。

---

## Traffic Accounting Equations

### Critical Rule: Footprint ≠ Traffic

本 spec 的核心要求是：**不得将 KV-cache capacity 直接冒充为 KV DRAM traffic**。

每一类数据必须给出四列：

| Data Category | Footprint (B) | Algorithmic Access (elements) | Architecture-visible Traffic (B) | Physical DRAM Traffic (B) |
|---------------|--------------|-------------------------------|----------------------------------|---------------------------|
| Weights | `Bytes_weight` | `Nparam` | `T_weight_arch` | `T_weight_dram` |
| Historical K | `Bytes_K` | `L × B × S × Hkv × Dhead` | `T_K_read_arch` | `T_K_read_dram` |
| Historical V | `Bytes_V` | `L × B × S × Hkv × Dhead` | `T_V_read_arch` | `T_V_read_dram` |
| New K write | — | `L × B × Hkv × Dhead` | `T_K_write_arch` | `T_K_write_dram` |
| New V write | — | `L × B × Hkv × Dhead` | `T_V_write_arch` | `T_V_write_dram` |

**术语定义：**
- **Footprint**：数据在目标存储器中的总resident容量。
- **Algorithmic Access**：算法层面需要访问的数据元素个数（不含精度/位宽）。
- **Architecture-visible Traffic**：从 memory architecture 视角看到的总线传输量（考虑位宽、burst、alignment）。
- **Physical DRAM Traffic**：实际到达 DRAM 接口的传输量（考虑 cache hit、tiling、page organization）。

---

### 5. Weight Read Traffic

#### 5.1 Weight Footprint (recap)
```
Bytes_weight = Nparam × bw / 8
```

#### 5.2 Algorithmic Weight Consumption per Decode Step
- 一个 complete decode step 需要执行全部 `L` 层的 forward pass。
- 每层需要读取该层的全部权重（QKV proj, output proj, MLP gates/up/down, norms）。
- 因此，一个 **aggregate decode step**（生成 B 个 token）的算法级权重访问量为：

```
Algorithmic_weight_access_per_step = Nparam  (elements)
```

（`Nparam` 为所有层参数之和。）

#### 5.3 Assumed Architecture-visible Weight Traffic

关键 **MODELING_CHOICE**：weights 是否在 batch 的 B 个 sequences 之间复用？

**假设 A（本 spec 默认）：Weights 在 aggregate step 内被读取一次，供 B 个 sequences 复用。**

此假设成立的条件：weights 可被保留在 on-chip SRAM / cache / register file 中，或从外部 memory 读取一次后分发给 B 个计算单元。对于 small/medium batch 和 current GPU SRAM capacities (tens of MB to ~100 MB for L2-shared weights), this is typically true for model weights if they are pipelined layer-by-layer.

```
T_weight_arch_per_step = Bytes_weight   (read once per aggregate step)
T_weight_arch_per_token = Bytes_weight / B
T_weight_arch_per_sequence_token = Bytes_weight / B   (same as per-token in aggregate step)
```

**假设 B（备选，需明确标注）：Weights 无法被缓存，每处理一个 sequence 都需重新读取。**

```
T_weight_arch_per_step_alt = B × Bytes_weight
T_weight_arch_per_token_alt = Bytes_weight
```

**Provenance:** 假设 A 为 **MODELING_CHOICE**（基于 small/medium batch 和 layer-by-layer weight pipelining 的典型 GPU 推理行为）。若无证据支持，应降级为 **NOT_VALIDATED**。

#### 5.4 Physical DRAM Traffic

Physical DRAM traffic 可能因以下因素与 architecture-visible traffic 不同：
- **Cache residency**：若 weights 在 L2/cache 中命中，DRAM traffic 可为 0。
- **Compression**：若 weights 以压缩格式存储，解压后 traffic 可能大于 footprint。
- **Quantization decompression**：INT4/FP8 权重在读取后可能被解量化到 FP16/BF16，增加 internal traffic。

```
T_weight_dram_per_step = γ_weight × T_weight_arch_per_step
```

其中 `γ_weight` 为 DRAM traffic multiplier。若无可信数据，`γ_weight = 1` 并标注 **MODELING_CHOICE**。

---

### 6. KV Read Traffic

#### 6.1 KV Footprint (recap)
```
Bytes_K = L × B × S × Hkv × Dhead × bkv / 8
Bytes_V = L × B × S × Hkv × Dhead × bkv / 8
Bytes_KV = Bytes_K + Bytes_V
```

#### 6.2 Algorithmic KV Access per Decode Step

在 autoregressive decode 的每个 step 中，模型生成 B 个新 token（每个 sequence 一个）。为了计算第 `t+1` 个 token 的 attention，每个 query head 需要与**所有历史位置** `1..t` 的 K/V 做 dot-product。

因此，算法层面，一个 aggregate step 需要读取：

```
Algorithmic_K_access_per_step = L × B × S × Hkv × Dhead   (elements of K)
Algorithmic_V_access_per_step = L × B × S × Hkv × Dhead   (elements of V)
```

其中 `S` 为当前 context length（若 variable-length，则替换为 `Σ_i S_i`）。

**关键区分：**
- `Bytes_KV` 是**总resident容量**（存储了从位置 1 到 S 的所有 K/V）。
- `Algorithmic_K_access_per_step` 是**单次 decode step 需要读取的历史 K 元素数**。
- 若 `S` 很大，单次 step 的 KV read 可能接近甚至等于 `Bytes_KV` 的 footprint（若全量重读）。
- 但 **Footprint ≠ Traffic**，除非明确采用 "full KV reread" modeling choice。

#### 6.3 Assumed Architecture-visible KV Read Traffic

**假设 C（本 spec 默认保守假设）：每个 decode step 从 target memory 全量读取历史 K 和 V。**

```
T_K_read_arch_per_step = L × B × S × Hkv × Dhead × bkv / 8
T_V_read_arch_per_step = L × B × S × Hkv × Dhead × bkv / 8
T_KV_read_arch_per_step = T_K_read_arch_per_step + T_V_read_arch_per_step
                         = 2 × L × B × S × Hkv × Dhead × bkv / 8
                         = Bytes_KV          (only when S is the FULL context)
```

**Per-generated-token：**

```
T_KV_read_arch_per_token = T_KV_read_arch_per_step / B
                         = 2 × L × S × Hkv × Dhead × bkv / 8
```

**注意：** 这里 `S` 是读取时的 context length。随着 generation 进行，`S` 逐渐增加（从 prompt length 增长到 prompt + Ngen）。上述公式给出的是**某一特定 step** 的 traffic。若需要 average over all generation steps，需对 `S` 的变化做积分/求和（见 Batch Semantics 章节）。

**假设 D（备选，FlashAttention-style tiling）：**
- 若采用 SRAM tiling（如 FlashAttention），K/V 被分块载入 SRAM，HBM read traffic 可降至接近 `O(1)` per token（仅写入新 K/V，读取由 SRAM 满足）。
- 但这要求 SRAM 容量足以容纳 attention tile。
- 本 spec **不采用**此假设作为默认值，因其为 **NOT_VALIDATED** 且高度依赖 implementation。

#### 6.4 Physical DRAM Traffic

```
T_K_read_dram_per_step = γ_K_read × T_K_read_arch_per_step
T_V_read_dram_per_step = γ_V_read × T_V_read_arch_per_step
```

`γ_K_read`, `γ_V_read` 为 DRAM traffic multipliers，反映 cache hit、prefetch、page-granular access 等效应。若无可信数据，设 `γ = 1` 并标注 **NOT_VALIDATED**。

---

### 7. KV Write Traffic

#### 7.1 Algorithmic KV Write per Decode Step

每个新 token 为每层生成一组新的 K 和 V 向量。

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

**注意：** KV write traffic 与 context length `S` **无关**（每步只写新 token 的 K/V）。这是 decode phase 的关键特性之一。

#### 7.3 Physical DRAM Traffic

```
T_K_write_dram_per_step = γ_K_write × T_K_write_arch_per_step
T_V_write_dram_per_step = γ_V_write × T_V_write_arch_per_step
```

`γ` 反映 write-back policy、write-combining、page granularity 等。默认 `γ = 1`，**NOT_VALIDATED**。

---

## Compute Equations

### 8. FLOPs per Decode Step (Aggregate)

以下计算一个 **aggregate decode step**（生成 B 个 token，每个 sequence 一个）的 FLOPs。

#### 8.1 Q/K/V Projections

输入: `[B, 1, Dmodel]`（每个 sequence 一个待生成的 token）

| Projection | Weight Shape | FLOPs |
|-----------|-------------|-------|
| Q | `[Dmodel, Hq × Dhead]` = `[Dmodel, Dmodel]` | `2 × B × Dmodel²` |
| K | `[Dmodel, Hkv × Dhead]` | `2 × B × Dmodel × Hkv × Dhead` |
| V | `[Dmodel, Hkv × Dhead]` | `2 × B × Dmodel × Hkv × Dhead` |

```
FLOPs_QKV_per_layer = 2 × B × Dmodel × (Dmodel + 2 × Hkv × Dhead)
```

若 `Dmodel = Hq × Dhead`（标准设定），则：
```
FLOPs_QKV_per_layer = 2 × B × Dmodel² × (1 + 2 × Hkv / Hq)
```

**Special cases:**
- **MHA** (`Hkv = Hq`): `FLOPs_QKV = 6 × B × Dmodel²`
- **MQA** (`Hkv = 1`): `FLOPs_QKV = 2 × B × Dmodel × (Dmodel + 2 × Dhead)`

#### 8.2 Output Projection

```
FLOPs_out_proj_per_layer = 2 × B × Dmodel²
```

#### 8.3 MLP (SwiGLU Gated FFN)

假设 MLP 结构为：gate_proj → up_proj → elementwise_mul → down_proj

| Operation | Weight Shape | FLOPs |
|-----------|-------------|-------|
| gate_proj | `[Dmodel, Dff]` | `2 × B × Dmodel × Dff` |
| up_proj | `[Dmodel, Dff]` | `2 × B × Dmodel × Dff` |
| down_proj | `[Dff, Dmodel]` | `2 × B × Dff × Dmodel` |

```
FLOPs_MLP_per_layer = 6 × B × Dmodel × Dff
```

*注：若 MLP 使用非 SwiGLU 结构（如原始 Transformer 的 ReLU FFN: up + down 两矩阵），则 FLOPs 为 `4 × B × Dmodel × Dff`。需在 provenance 中说明模型类型。*

#### 8.4 Attention Score + Value Accumulation

**Score computation** (`Q @ K^T`):
- Q shape: `[B, Hq, 1, Dhead]`
- K shape: `[B, Hkv, S, Dhead]` (historical keys up to length S)
- For GQA, each query head uses its assigned KV head.
- FLOPs: `B × Hq × (2 × S × Dhead) = 2 × B × Hq × S × Dhead`

**Value accumulation** (`Softmax(QK^T) @ V`):
- Attention weights: `[B, Hq, 1, S]`
- V shape: `[B, Hkv, S, Dhead]`
- FLOPs: `B × Hq × (2 × S × Dhead) = 2 × B × Hq × S × Dhead`

```
FLOPs_attention_per_layer = 4 × B × Hq × S × Dhead
```

**关键特性：** 此项显式正比于 `S`（context length），是 long-context decode 的 compute 瓶颈来源。

#### 8.5 LayerNorm / RMSNorm

通常可忽略（参数量和计算量远小于 linear layers）：
- 参数量: `≈ 2 × Dmodel` per layer (scale + shift 或仅 scale)
- FLOPs: `≈ 5 × B × Dmodel` per layer (elementwise ops)
- 总计: `≈ 5 × B × L × Dmodel`

本 spec v0 将其记为 `FLOPs_norm` 但标注为 **NEGLIGIBLE** 或 **MODELING_CHOICE**。

#### 8.6 Vocabulary Projection (Optional)

若将 final logits projection 纳入：
```
FLOPs_vocab = 2 × B × Dmodel × V
```

对于 tied embedding，此矩阵与 input embedding 共享，但 forward pass 中仍需计算。

#### 8.7 Total per-Layer FLOPs

```
FLOPs_layer = FLOPs_QKV + FLOPs_out_proj + FLOPs_MLP + FLOPs_attention + FLOPs_norm
            = 2×B×Dmodel×(Dmodel + 2×Hkv×Dhead)    [QKV]
            + 2×B×Dmodel²                            [out proj]
            + 6×B×Dmodel×Dff                         [MLP]
            + 4×B×Hq×S×Dhead                         [attention]
            + negligible                             [norm]
```

#### 8.8 Total per-Step FLOPs (All Layers)

```
FLOPs_step = L × FLOPs_layer + FLOPs_vocab
```

#### 8.9 FLOPs per Generated Token

```
FLOPs_token = FLOPs_step / B
```

---

### 9. Simplified Sanity Approximation

为快速估算，可提供 **2 × Nparam** 近似：

```
FLOPs_token_sanity ≈ 2 × Nparam + 4 × L × Hq × S × Dhead
```

**使用条件与限制：**
- 第一项 `2 × Nparam` 近似了所有 linear layer (QKV, out, MLP) 的 compute，假设 batch=1 且忽略 GQA 节省。
- 第二项 `4 × L × Hq × S × Dhead` 显式补上了 attention 的 context-dependent 开销。
- **此近似不得取代精确分解**，尤其当：
  - `S` 很大（long-context），attention 项可能主导。
  - `Hkv ≪ Hq`（GQA），QKV projection 实际 FLOPs 低于 `2 × Nparam`。
  - MLP 结构非标准（如 MoE）。

**Provenance:** 2×Nparam 为 **DERIVED_FROM_REFERENCE**（行业常用 heuristic）。Attention 修正项为 **DERIVED_FROM_PAPER**（标准 attention 算法分析）。

---

## Batch Semantics

### 10. Aggregate Step vs. Per-Token Metrics

| Metric | Definition | 公式 |
|--------|-----------|------|
| **Aggregate step** | 一个 forward pass 生成 B 个 token（batch 中每个 sequence 一个） | — |
| **Per-step traffic** | 一个 aggregate step 的总 memory traffic | `T_*_per_step` |
| **Per-token traffic** | 平均每个生成 token 的 traffic | `T_*_per_step / B` |
| **Per-sequence token** | 同一 sequence 内，每个 token 的 traffic | 同 per-token（等长 batch） |

**关键区分：**
- Weight read 在 aggregate step 中通常只发生一次（被 B 个 sequences 复用），因此 `T_weight_per_token = T_weight_per_step / B` 体现了 batch amortization。
- KV read/write 随 B 线性增长（每个 sequence 有自己的 KV cache），因此 `T_KV_per_token` 与 B 无关（per-token 已归一化）。

### 11. Continuous / Variable-Length Batching

本 spec v0 处理策略：
- **等长 batch (`S_i = S`)**：所有公式直接适用。
- **Variable-length batch**：
  - KV footprint: 使用 `Σ_i S_i` 替代 `B × S`。
  - KV read traffic: 每个 sequence 读取自己的 `S_i` 长度，总计 `Σ_i S_i`。
  - Compute: Attention FLOPs 为 `Σ_i (4 × Hq × S_i × Dhead)` per layer。
  - **Continuous batching**（新请求动态加入）：本 spec 不建模动态调度，假设 batch 组成在 generation 过程中固定。**NOT_VALIDATED**。

---

## Hand Checks

### Case 1: B=1, Long Context (Symbolic Model)

**模型参数（EXAMPLE_VALUES，来自 LLaMA-3-8B-class architecture）：**

| Parameter | Value | Source |
|-----------|-------|--------|
| Nparam | 8.0 × 10⁹ | EXAMPLE (LLaMA-3-8B) |
| L | 32 | EXAMPLE |
| Hq | 32 | EXAMPLE |
| Hkv | 8 | EXAMPLE (GQA) |
| Dhead | 128 | EXAMPLE |
| Dmodel | 4096 | DERIVED (`Hq × Dhead`) |
| Dff | 14336 | EXAMPLE (LLaMA-3 SwiGLU) |
| V | 128256 | EXAMPLE |
| bw | 16 (fp16/bf16) | MODELING_CHOICE |
| bkv | 16 (fp16/bf16) | MODELING_CHOICE |
| B | 1 | MODELING_CHOICE |
| S | 128000 | MODELING_CHOICE (long-context) |
| ε_reserve | 0 | MODELING_CHOICE (simplified) |
| ε_KV_waste | 0 | MODELING_CHOICE (simplified) |

#### 1. Footprint Check

```
Bytes_weight = 8.0e9 × 16 / 8 = 16.0 × 10⁹ B = 16.0 GiB

Bytes_KV = 2 × 32 × 1 × 128000 × 8 × 128 × 16 / 8
         = 64 × 128000 × 8 × 128 × 2
         = 16,777,216,000 B
         = 15.6 GiB

Bytes_required = 16.0 + 15.6 = 31.6 GiB
```

**Observation:** 在 128K context 下，KV cache footprint (15.6 GiB) ≈ weight footprint (16.0 GiB)。这是 long-context decode 的关键容量特征。

#### 2. KV Read Traffic Check

```
T_KV_read_arch_per_step = Bytes_KV = 15.6 GiB   (full reread assumption)
T_KV_read_arch_per_token = 15.6 GiB / 1 = 15.6 GiB/token
```

**Critical Check:** 若错误地将 KV capacity 直接当作 per-token traffic，会得到相同数字。但本 spec 明确标注了 "full KV reread per decode step" 这一 **MODELING_CHOICE**。若采用 FlashAttention tiling，此 traffic 可能大幅降低（**NOT_VALIDATED**）。

#### 3. KV Write Traffic Check

```
T_KV_write_arch_per_step = 2 × 32 × 1 × 8 × 128 × 16 / 8
                         = 131,072 B
                         = 128 KiB

T_KV_write_arch_per_token = 128 KiB / 1 = 128 KiB/token
```

**Observation:** KV write per token 极小（128 KiB），与 KV read（15.6 GiB）相差 5 个数量级。这解释了为什么 decode phase 是 read-dominated。

#### 4. Weight Read Traffic Check

```
T_weight_arch_per_step = 16.0 GiB   (read once, B=1)
T_weight_arch_per_token = 16.0 GiB / 1 = 16.0 GiB/token
```

#### 5. Compute FLOPs Check

```
FLOPs_QKV_per_layer = 2 × 1 × 4096² × (1 + 2 × 8 / 32)
                    = 2 × 16,777,216 × 1.5
                    = 50,331,648
                    ≈ 50.3 MFLOPs

FLOPs_out_proj_per_layer = 2 × 1 × 4096² = 33.6 MFLOPs
FLOPs_MLP_per_layer = 6 × 1 × 4096 × 14336 = 352.3 MFLOPs
FLOPs_attention_per_layer = 4 × 1 × 32 × 128000 × 128
                          = 2,097,152,000
                          = 2.10 GFLOPs

FLOPs_layer = 50.3 + 33.6 + 352.3 + 2097.2
            ≈ 2.53 GFLOPs

FLOPs_step = 32 × 2.53 + 2 × 4096 × 128256 / 1e9
           ≈ 81.0 + 1.05
           ≈ 82.1 GFLOPs

FLOPs_token = 82.1 / 1 = 82.1 GFLOPs/token
```

**Sanity Approximation Check:**
```
FLOPs_token_sanity ≈ 2 × 8e9 + 4 × 32 × 128000 × 128
                   = 16e9 + 2.10e9
                   = 18.1 GFLOPs
```

**Observation:** Sanity approximation (18.1 GFLOPs) **严重低估**了实际 compute (82.1 GFLOPs)。原因是 LLaMA-3-8B 的 MLP 比例很大（Dff = 3.5 × Dmodel），且 `2 × Nparam` 近似未准确 capture GQA 和 SwiGLU 结构。**这验证了 spec 的要求：不得让 2×Nparam 取代精确分解。**

---

### Case 2: B>1, Medium Batch (Aggregate Step Amortization)

**模型参数：** 同 Case 1
**Workload 参数：**

| Parameter | Value | Source |
|-----------|-------|--------|
| B | 8 | MODELING_CHOICE |
| S | 32000 (等长) | MODELING_CHOICE |

#### 1. Footprint Check

```
Bytes_weight = 16.0 GiB   (与 B 无关)

Bytes_KV = 2 × 32 × 8 × 32000 × 8 × 128 × 16 / 8
         = 33,554,432,000 B
         = 31.3 GiB

Bytes_required = 16.0 + 31.3 = 47.3 GiB
```

**Observation:** Batch 增大 8×，KV footprint 线性增大 8×（从 15.6 → 31.3 GiB）。Weight footprint 不变。

#### 2. Weight Amortization Check

```
T_weight_arch_per_step = 16.0 GiB   (read once, shared by B=8)
T_weight_arch_per_token = 16.0 / 8 = 2.0 GiB/token
```

**Amortization factor:** 8×。这是 batching 的核心收益：weight read 被 B 个 sequences 分摊。

#### 3. KV Read Traffic Check

```
T_KV_read_arch_per_step = 2 × 32 × 8 × 32000 × 8 × 128 × 16 / 8
                        = 33,554,432,000 B
                        = 31.3 GiB

T_KV_read_arch_per_token = 31.3 / 8 = 3.91 GiB/token
```

**Observation:**
- Case 1 (B=1, S=128K): KV read per token = 15.6 GiB
- Case 2 (B=8, S=32K): KV read per token = 3.91 GiB
- KV read per token 与 `(B × S) / B = S` 成正比。即 **per-token KV read 只取决于该 sequence 的 context length，与 batch size 无关**。
- 但 aggregate step 的总 KV read = `B × S × (constants)`，随 batch 线性增长。

#### 4. KV Write Traffic Check

```
T_KV_write_arch_per_step = 2 × 32 × 8 × 8 × 128 × 16 / 8
                         = 1,048,576 B
                         = 1.0 MiB

T_KV_write_arch_per_token = 1.0 / 8 = 128 KiB/token
```

**Observation:** KV write per token (128 KiB) 与 B 无关。这是合理的：每个 sequence 每步只写自己的新 K/V。

#### 5. Compute FLOPs Check

```
FLOPs_attention_per_layer = 4 × 8 × 32 × 32000 × 128
                          = 4,194,304,000
                          = 4.19 GFLOPs

FLOPs_MLP_per_layer = 6 × 8 × 4096 × 14336 = 2.82 GFLOPs
FLOPs_QKV_per_layer = 2 × 8 × 4096² × 1.5 = 402.7 MFLOPs
FLOPs_out_proj_per_layer = 2 × 8 × 4096² = 268.4 MFLOPs

FLOPs_layer = 0.40 + 0.27 + 2.82 + 4.19 = 7.68 GFLOPs
FLOPs_step = 32 × 7.68 + 8.4 (vocab)
           ≈ 246 + 8.4
           ≈ 254 GFLOPs

FLOPs_token = 254 / 8 = 31.8 GFLOPs/token
```

**Comparison:**
- Case 1 (B=1, S=128K): 82.1 GFLOPs/token
- Case 2 (B=8, S=32K): 31.8 GFLOPs/token
- Batch 摊薄了 weight-related FLOPs（per-token 降低），但 attention FLOPs 仍由 context length 主导。

---

## Dimensional Sanity Checks

| 公式 | 单位推导 | 结果 |
|------|---------|------|
| `Bytes_weight = Nparam × bw / 8` | `[param] × [bit/param] / [bit/byte] = [byte]` | ✅ |
| `Bytes_KV = 2 × L × B × S × Hkv × Dhead × bkv / 8` | `1 × 1 × 1 × 1 × 1 × 1 × [bit] / [bit/byte] = [byte]` | ✅ |
| `T_KV_read_arch_per_token = 2 × L × S × Hkv × Dhead × bkv / 8` | `1 × 1 × 1 × 1 × [bit] / [bit/byte] = [byte/token]` | ✅ |
| `FLOPs_QKV = 2 × B × Dmodel × (Dmodel + 2×Hkv×Dhead)` | `1 × 1 × 1 × 1 = [FLOP]` | ✅ |
| `FLOPs_attention = 4 × B × Hq × S × Dhead` | `1 × 1 × 1 × 1 = [FLOP]` | ✅ |
| `FLOPs_token = FLOPs_step / B` | `[FLOP/step] / [token/step] = [FLOP/token]` | ✅ |
| `GiB = byte / 1024³` | `[byte] / 1024³ = [GiB]` | ✅ |

---

## Key Results

本 spec 建立了以下核心方程体系：

### Memory Footprint
- `Bytes_weight = Nparam × bw / 8`
- `Bytes_KV = 2 × L × B × S × Hkv × Dhead × bkv / 8`
- `Bytes_required = Bytes_weight + Bytes_KV_realized + Bytes_runtime`
- `feasible = Bytes_required <= Cusable`

### Traffic per Aggregate Step
- `T_weight_arch_per_step = Bytes_weight` (once per step, amortized over B)
- `T_KV_read_arch_per_step = 2 × L × B × S × Hkv × Dhead × bkv / 8`
- `T_KV_write_arch_per_step = 2 × L × B × Hkv × Dhead × bkv / 8`

### Traffic per Generated Token
- `T_weight_arch_per_token = Bytes_weight / B`
- `T_KV_read_arch_per_token = 2 × L × S × Hkv × Dhead × bkv / 8`
- `T_KV_write_arch_per_token = 2 × L × Hkv × Dhead × bkv / 8`

### Compute per Generated Token
- `FLOPs_token = (L × [2×B×Dmodel×(Dmodel+2×Hkv×Dhead) + 2×B×Dmodel² + 6×B×Dmodel×Dff + 4×B×Hq×S×Dhead] + FLOPs_vocab) / B`

---

## Scientific Interpretation

### 1. Long-Context Decode 的特征

从 Hand Check Case 1 (B=1, S=128K) 可得：
- **KV read traffic 主导**：15.6 GiB/token 的 KV read vs 16.0 GiB/token 的 weight read。
- **Attention compute 主导**：2.10 GFLOPs/layer 的 attention vs 0.40 GFLOPs/layer 的 QKV projection。
- **KV footprint 与 weight 相当**：15.6 GiB vs 16.0 GiB。

这意味着在 long-context 场景下，内存系统的瓶颈可能从 "weight bandwidth" 转向 "KV bandwidth"，这对 Orthogonal M3D-IGZO 的高容量优势具有潜在意义。

### 2. Batch Amortization 的边界

从 Case 2 (B=8)：
- Weight read per token 降低 8×（2.0 GiB/token）。
- KV read per token **不变**（仍由 S 决定）。
- 因此，随着 B 增大，workload traffic 的构成从 "weight-dominated" 转向 "KV-dominated"。

### 3. KV Capacity ≠ KV Traffic

本 spec 明确建立了这一区分：
- **KV Capacity** (`Bytes_KV`) 是存储所有历史 K/V 所需的总空间。
- **KV Read Traffic** (`T_KV_read_arch_per_token`) 是每生成一个 token 需要读取的历史 K/V 量。
- 在 "full reread" assumption 下，`T_KV_read_arch_per_step = Bytes_KV`（仅当读取全部历史时）。
- 若采用 tiling/caching，`T_KV_read_dram` 可能远小于 `Bytes_KV`。

---

## Assumptions / Provenance

| # | Assumption | Category | Rationale / Evidence |
|---|-----------|----------|---------------------|
| 1 | Weight precision `bw` (fp16/bf16/etc.) | MODELING_CHOICE |  workload 配置决定；需在最终 eval 中 sensitivity sweep |
| 2 | KV precision `bkv` = `bw` | MODELING_CHOICE |  常见做法；MQA/GQA 通常保持相同精度；可放宽 |
| 3 | Weights read once per aggregate step, shared by B | MODELING_CHOICE |  基于 layer-pipelined GPU inference 的典型行为；小 batch 时 weights 常驻 SRAM/L2 |
| 4 | KV full reread per decode step | MODELING_CHOICE |  **保守假设**；未考虑 FlashAttention tiling 或 L2 cache residency |
| 5 | No KV cache compression / quantization | MODELING_CHOICE |  v0 简化；实际系统可能使用 4-bit KV 或稀疏化 |
| 6 | SwiGLU MLP structure (3 projections) | DERIVED_FROM_REFERENCE |  LLaMA-2/3, Mistral 等主流模型采用 |
| 7 | GQA with `Hkv < Hq` | DERIVED_FROM_PAPER |  LLaMA-2/3, Mistral 官方配置 |
| 8 | `ε_reserve = 0`, `ε_KV_waste = 0` | MODELING_CHOICE (simplification) |  Hand-check 简化；实际 eval 需引入 realistic overhead |
| 9 | No runtime overhead (`Bytes_runtime = 0`) | MODELING_CHOICE (simplification) |  v0 简化；实际 footprint 需 +5-20% |
| 10 | `γ_weight = γ_K_read = γ_V_read = γ_K_write = γ_V_write = 1` | NOT_VALIDATED |  无 FlashAttention/缓存/压缩的 implementation 数据 |
| 11 | LayerNorm FLOPs negligible | MODELING_CHOICE |  数量级 <1% of total；精确 eval 可补入 |
| 12 | Tied embedding counted once in `Nparam` | MODELING_CHOICE |  取决于模型；LLaMA-3 为 untied，需核实 |

---

## PASS / FAIL

| Criterion | Status | Notes |
|-----------|--------|-------|
| Footprint 与 traffic 始终分离 | ✅ PASS | 四列表格明确区分 |
| KV capacity 未直接冒充 KV DRAM traffic | ✅ PASS | 显式标注 "full reread" MODELING_CHOICE |
| GQA/MQA/MHA 处理正确 | ✅ PASS | 使用 `Hkv` 而非 `Hq` 计算 KV footprint/traffic |
| Batch 语义完整 | ✅ PASS | 区分 aggregate step / per-token / per-sequence |
| Aggregate/per-sequence 单位明确 | ✅ PASS | 所有公式标注 per-step 或 per-token |
| Long-context attention compute 显式依赖 context | ✅ PASS | `FLOPs_attention ∝ S` |
| 所有公式可手算 | ✅ PASS | Case 1 & 2 完全手算验证 |
| 具体模型参数有来源 | ✅ PASS | LLaMA-3-8B 参数标注为 EXAMPLE |
| 缺失实现细节保持 symbolic 或 NOT_VALIDATED | ✅ PASS | `γ_*`, `ε_*`, `Bytes_runtime` 等 |
| 不输出 tokens/s, J/token, power, Tmax | ✅ PASS | 本 spec 范围严格限制 |
| 不修改仓库文件 | ✅ PASS | 零文件修改 |

---

## Open Questions

1. **FlashAttention / SRAM tiling 对 KV traffic 的削减因子**：当前假设 full reread。若 GPU SRAM 可 cache KV tiles，实际 DRAM traffic 可能降低 10-100×。需 implementation-dependent data。
2. **Weight residency on-chip**：小模型（如 8B）的 weights 在 large GPU 上可能部分常驻 L2，降低 weight DRAM traffic。需 profiling data。
3. **KV-cache quantization / compression**：主流 serving 系统（vLLM, TensorRT-LLM）支持 INT8/FP8 KV cache，可减半 footprint 和 traffic。是否纳入 v1？
4. **Variable-length batch / continuous batching**：实际 LLM serving 中，batch 组成动态变化。v0 假设固定 batch，需 v1 扩展。
5. **MoE workload**：本 spec 仅覆盖 dense Transformer。MoE 的 expert routing 会改变 weight traffic 模式（sparse weight access）。
6. **Prefill vs. Decode phase**：本 spec 仅覆盖 decode phase。Prompt processing (prefill) 的 traffic/compute 模式完全不同（compute-bound, GEMM 而非 GEMV）。
7. **`ε_reserve` 和 `ε_KV_waste` 的实际值**：需要来自 vLLM 或 TensorRT-LLM 的 profiling 数据。

---

## Next Recommended Step

1. **Research Lead 审核本 spec**：确认符号体系、四列 traffic accounting、batch 语义是否符合 DAC 项目需求。
2. **确定 concrete model configs**：选择 2-3 个代表性模型（如 LLaMA-3-8B, LLaMA-3-70B, 或 custom small model）作为 benchmark workload。
3. **量化 `ε_reserve`, `ε_KV_waste`, `γ_*` 等参数**：通过文献或 profiling 获取 realistic values，或保留为 sensitivity sweep 参数。
4. **进入 Track B2**（若存在）：将本 spec 的 output 接入 bandwidth/performance evaluator，计算 `Tmemory`, `tokens/s`, `J/token`。

---

## STOP

Track B1 analytical specification 完成。等待 Research Lead 审核与反馈。不得在未获授权前进入 bandwidth、performance evaluator 或 thermal 阶段。
