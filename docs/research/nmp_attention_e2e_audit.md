# Dense NMP attention E2E audit

Scope: LLaMA-3.1-8B-class decode, B=1, S=131072, FP16 weights/KV.
No architecture/MAC sweep, prefill, MoE, streaming attention, reduction network,
new bandwidth parameter, or thermal solver change.

## Before

```text
workload totals -> placement-built Q/K/V/O/FFN + ATTENTION_KV units
               -> incomplete named weights rescaled to full footprint
               -> per-die whole-step totals -> max_die(max(memory, compute))
               -> residual bytes / external link -> token interval
               -> nmp_die_power memory/MAC/refresh/external power
```

The indivisible attention unit hid GPU Softmax and both dependency barriers.
The old driver swept batches and arbitrary aggregate NMP throughput and compared
against an external rate that did not apply the canonical GPU sustained ceiling.

## After

```text
LLMDecodeInput -> workload/dense_decode_ledger.py
              -> exact weight/K/V read/write/FLOP/operator boundary ledger
              -> existing minimum-span performance-balanced ownership
              -> power/nmp_die_activity.py
                 per-layer Q,K,V -> QK -> score transfer -> GPU Softmax
                 -> probability transfer -> AV -> partial transfer
                 -> O -> FFN_GATE -> FFN_UP -> FFN_DOWN
                 -> LM_HEAD -> OTHER_WEIGHT read
              -> sum of dependent stage times
              -> nmp_die_power.py + GPU Softmax dynamic + GPU static once
```

Non-attention operators retain conservative GPU combine/forward boundary bytes;
these are separate from the three attention boundary terms. No implicit
die-to-die transfer is introduced. Reduction arithmetic, RoPE, RMSNorm, residual,
SiLU and sampling remain unmodeled. Boundary transfer events charge transport,
not a new GPU reduction kernel.

The existing allocator's minimum die-span rule is unchanged. Its whole-step
load-balancing objective remains a placement heuristic, not the new E2E timing
formula. With B=1, each AV layer has one owner, not all 106 dies. The resulting
serial execution leaves most dies idle at a given stage; the model does not
claim an NMP speedup.

## Equations and provenance

All workload ledger quantities are per aggregate decode step; per-token
reporting divides by B. Shared weight FLOPs in individual units are per request
and multiplied by B exactly once by consumers.

- `F_QK = F_AV = 2 B Hq S dh NL`.
- K and V each read `B S Hkv dh kv_bits/8` and write
  `B Hkv dh kv_bits/8` per layer. Their totals match the existing KV model.
- `D_score = B Hq S NL * 16/8`; probabilities use the same equation.
- Per layer, `D_partial = |union(AV owners)| * B d_model * 32/8`.
  Owner unions and partial vectors are exposed in `activity.attention_layers`.
- `D_attention = D_score + D_probability + D_partial`.
- `D_residual = D_attention + D_weight_operator_boundary`.
- The existing transfer resolver computes
  `BW_actual = min(BW_demand, BW_M3D_external, BW_GPU_sustained)`.
  GPU sustained bandwidth comes from the existing service resolver and canonical
  H200 YAML: `0.5 * 4.8 TB/s = 2.4 TB/s`, with no second derating.
- `t_boundary = D_residual / BW_actual`.
- `D_softmax = D_score + D_probability`; `t_softmax = D_softmax / BW_GPU_sustained`.
- `E_softmax = 8 D_softmax * e_GPU_decode`, with the canonical `15.29 pJ/actual-bit`.
- `t_stage = max_die(max(D_die/BW_local_die, F_die/F_peak_die))`;
  `t_token = sum(t_stage) + t_boundary + t_softmax` for B=1.
- `E_total = P_NMP_map * t_token + E_softmax + 74 W * t_token`.
  The existing NMP power map includes local memory, MAC, refresh, and residual
  long-FEOL/interface energy exactly once. Local reads exclude the last two.

Score/probability precision, FP32 partials, serial scheduling, and residual
OTHER_WEIGHT reads are **MODELING_CHOICE**. Hardware throughput and energies
retain their existing provenance. No new **PAPER_REPORTED** or
**DERIVED_FROM_PAPER_FIGURE** claim is made. The frozen **NUMERICAL_CHOICE** of
FP64 matrix-free GPU-PCG/Jacobi is unchanged.

Named Q/K/V/O/FFN weights occupy 13,958,643,712 bytes; LM_HEAD adds
1,050,673,152 bytes. OTHER_WEIGHT explicitly reads the remaining 990,683,136
bytes, giving exactly 16,000,000,000 bytes. OTHER_WEIGHT adds no invented MACs;
total FLOPs still match `evaluate_llm_decode`'s detailed dimensional formula.

## Nominal result

MB/GB use decimal units. Compute and memory times below are sums of realized
stage service times, not total work divided by the 106-die aggregate peak.

| Metric | Value per token unless specified |
|---|---:|
| QK FLOPs | 34.359738368 GFLOP |
| AV FLOPs | 34.359738368 GFLOP |
| Weight NMP FLOPs | 15.009316864 GFLOP |
| Total NMP FLOPs | 83.728793600 GFLOP |
| NMP local weight reads | 16,000,000,000 bytes |
| NMP local KV reads + writes | 17,180,000,256 bytes |
| Scores | 268.435456 MB |
| Probabilities | 268.435456 MB |
| AV partials | 0.524288 MB |
| Attention boundary | 537.395200 MB |
| Total residual boundary | 545.401344 MB |
| Effective boundary bandwidth | 2.4 TB/s |
| Boundary time | 0.227250560 ms |
| Softmax time | 0.223696213 ms |
| NMP compute service time | 81.766400000 ms |
| NMP local memory service time | 163.027177473 ms |
| Final decode latency | 163.478124247 ms |
| Throughput | 6.117026389 tokens/s |
| Energy | 12.257838480 J/token |
| Energy efficiency | 0.081580452 tokens/J |

Softmax dynamic energy is 0.065670050 J/token. GPU static energy is
12.097381194 J/token. Residual transport energy is 0.002910507 J/token;
it is not part of Softmax local processing energy.

The non-NMP baseline here is GPU-only compute with the same M3D-resident
workload and existing internal/external roofline path, capped at the same
2.4 TB/s GPU sustained ceiling. It is not the Conventional 2x1 thermal baseline.

| Metric | Non-NMP GPU baseline |
|---|---:|
| Decode latency | 13.825000107 ms/token |
| Throughput | 72.332730002 tokens/s |
| Energy | 5.309158478 J/token |
| Energy efficiency | 0.188353767 tokens/J |
| NMP speedup | 0.084567890x |
| NMP energy-efficiency gain | 0.433123547x |

## Removed semantics and changed files

- Removed the ATTENTION_KV primitive and incomplete-weight proportional filling.
- Removed dense NMP arbitrary-throughput sweep input and the driver's batch/MAC sweep.
- Removed whole-step max-service as NMP token timing and externally precomputed
  boundary times that could bypass the transfer resolver.
- Removed the thermal adapter's old full-bandwidth GPU power assignment; its
  GPU source now uses Softmax plus static energy over the full interval.

Implementation: `workload/dense_decode_ledger.py`,
`placement/nmp_load_balance.py`, `placement/nmp_locality_e2e.py`,
`power/nmp_die_activity.py`, `cli.py` (under `src/om3dthermal/`),
`scripts/evaluate_nmp_locality_placement.py`,
`scripts/evaluate_nmp_thermal_baseline.py`.

Tests: `tests/test_nmp_die_activity.py`, `tests/test_nmp_locality_e2e.py`,
`tests/test_nmp_die_thermal_mapping.py`. This document is the audit artifact.
`power/nmp_die_power.py` and the thermal solver are reused without modifications.

## Validation and reproduction

In native Windows Conda `om3dthermal`:

```powershell
conda activate om3dthermal
cd "E:\BaiduSyncdisk\study\PAPER\DAC 2026\Project"
python -m om3dthermal nmp-attention
python -m pytest tests/test_nmp_die_activity.py tests/test_nmp_locality_e2e.py tests/test_nmp_die_thermal_mapping.py tests/test_transfer_operating_point.py tests/test_llm_decode_workload_power.py tests/test_sweep.py -q
python -m pytest tests/test_thermal_relaxation.py -q -k gpu_pcg
```

Results: **90 passed**; GPU-PCG smoke **3 passed, 14 deselected**. Nominal CLI
completed. JSON, detailed ownership, stage timing, and component energy/power
remain untracked in `runs/nmp_attention_nominal/nmp_locality_placement.json`.
Tests include batch/precision/context formula checks, demand-limited resolver
use, AV owner unions, barrier ordering, and energy double-count gates.
No Conventional solver equation, tolerance, mapping or expected value changed;
the full 859596-cell Conventional solve was not rerun for this non-solver change.
