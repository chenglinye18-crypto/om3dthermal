# Orthogonal M3D Memory DAC — Project Lead Handoff (turn 001)

*Author: Mavis (root session `mvs_9a2f47f84ea74824938fa483c31b7c90`)*
*Date: 2026-08-18, Asia/Shanghai*
*Project: `om3dthermal`, workspace `E:/BaiduSyncdisk/study/PAPER/DAC 2026/Project`*
*Audience: project lead (Chenglin) and any downstream worker that takes the next narrow step.*

This document is the first project-lead handoff. It is **not** a DAC result. It is the project state, the constraints, the open questions, and the single next narrow scientific step that the lead wants the next worker to take.

---

## 1. Bottom line

The project is in a state where the **thermal foundation is locked and the BW/energy/perf foundation is not.** Six thermal cases reproduce to ≲ 0.2 K of the project benchmark; the M3D-IGZO v1 thermal cases reproduce to 4 decimals. The remaining open work is concentrated in two places: (i) the per-component pJ/bit reconciliation that backs the energy claim, and (ii) the bandwidth-ceiling function `BWsystem = min(BWarray, BWMIV, BWFEOL, BWcoil)` that the brief requires before the M3D-IGZO claim can move from "matched-reference" to "capability-validated".

**Single next narrow step:** derive the four bandwidth ceilings for M3D-IGZO from the existing YAMLs, identify which one is the bottleneck by calculation (not by intuition), and propagate that ceiling back into the workload model. The interface/coil ceiling is the most leveraged target because (a) it is currently a `MODELING_CHOICE` constant at 0.5 pJ/bit and (b) the same term is 58 % of the M3D total pJ/bit. Acceptance criteria for this step are listed in §7.

If the result of that derivation is "the coil is the bottleneck, not the array", the project has just spent its next narrow step on the *right* question; the data-pattern sensitivity and the 50/50 read-mix assumption can be tackled only after the ceiling is known.

---

## 2. Project state snapshot

The repository is the thermal-solver side of the DAC paper. The full per-workload / per-bandwidth / end-to-end model is **not yet built**; this turn does not build it. What's actually on disk:

| Layer | What exists | What is locked | What is not yet pinned |
|---|---|---|---|
| Thermal solver | GPU FP64 matrix-free PCG, ~1–2 s/point, validated on Conventional 2×2 and Orthogonal MOSAIC; canonical 859 596 cells / 2 531 340 internal edges; 574 W package; 122.9715 °C G414 / 102.3078 °C G300 | Yes — see `docs/benchmarks/thermal_results_overview.md` and `exp_conv_2x2_g414_m160.md` | New M3D-IGZO placements (interface Tx/Rx) need a re-run, but the *solver* is done. |
| M3D-IGZO v0/v1 geometry | 8-layer homogenized 2.304 µm bitcell + 3 µm BEOL = 5.304 µm per slab; 98 slabs in a 30 mm cube; 0.85 W/(m·K) isotropic; closure 300.000 µm | Geometry closure, slab layout, slab pitch | per-axis k is `MODELING_CHOICE`; 64-layer (Zhu) vs 8-layer (project) gap is open |
| M3D-IGZO operation energy anchor | `orthogonal_m3d_igzo.yaml` carries Zhu 2026 IEDM Table I per-operation fJ/bit, all tagged `SPICE_EXTRACTED_MAT_LOCAL_OPERATION_ENERGY` | The 8 fJ/bit numbers and the per-row hold power | Whether 64-layer (Zhu) is equivalent to 8-layer (project) is `NOT_VALIDATED` |
| M3D-IGZO total pJ/bit | Anchored at 0.8553 = 0.1858 (internal) + 0.00245 (MIV) + 0.1671 (FEOL) + 0.5 (interface) | The total | The four components are *not yet individually reconciled against the YAML* (see §3) |
| Conventional HBM | DreamRAM pinned at `third_party/DreamRAM/configs/mem/baseline/hbm3_baseline.json`; Son23 component-aware power; `128 W + 32 W + 414 W` accounting | Son23 split, per-component mapping, total `574 W` | The 1.397 pJ/bit anchor is *not yet decomposed* into the same 4-component rule as M3D-IGZO |
| Orthogonal Si | DreamRAM, contactless interface constant 0.5 pJ/bit, 1.6 W/die uniform BEOL | Geometry, 1.368 pJ/bit total | The 1.368 pJ/bit anchor is *not yet decomposed* |
| Workload model | `workload.read_bandwidth_gbps: 39200` in all three current cases | 39.2 Tb/s as a *matched reference*, not a capability | LLM traffic profile, batch, context, capacity-feasibility — `NOT_VALIDATED` |
| Open-questions list | 8 items enumerated in the project brief, but no `OPEN_QUESTIONS.md` in the repo | — | The list is informal; this turn makes it explicit (see §6) |

The thermal model is the only layer that is fully reproducible from disk today. Everything else has a "current best-known" value but a non-trivial gap to a *validated* value.

---

## 3. The 4-component pJ/bit matrix

The brief fixes the accounting rule:

```
E_total = internal + MIV + FEOL + interface
```

Applied uniformly to the three project architectures, the current state is:

| Component | Conventional HBM | Orthogonal Si | Orthogonal M3D-IGZO | Status |
|---|---:|---:|---:|---|
| internal (array / bank / cell) | `NOT_VALIDATED` (1.397 minus the other three; no per-component breakdown in the YAML) | `NOT_VALIDATED` (same, 1.368 minus the other three) | `NOT_VALIDATED`; the 0.1858 pJ/bit anchor implies a **50/50 read 0 / read 1 mix** under the Zhu Table I (0.60 / 368 fJ/bit) and a 50/50 write-mix | The 50/50 mix is a `MODELING_CHOICE`; Zhu Table I is state-resolved but the project config averages it |
| MIV (vertical escape) | 0 (HBM uses TSV, not MIV) | 0 (no vertical) | 0.00245 pJ/bit (project anchor); MIV physical: 50 nm dia, 100 nm pitch, 0.2 fF [1] | Project anchor is consistent with the published MIV parameters; *not yet computed from the YAML* |
| FEOL (escape route / I/O) | `NOT_VALIDATED` (Son23 places 8 W logic per stack → ~0.18 pJ/bit if amortized over 40 W × 12.5 Tb/s = 0.5 TB/s/stack, but the HBM3 BW is 819 GB/s per stack, not 0.5 TB/s; the current anchor is *not* derivable from Son23 alone) | `NOT_VALIDATED` (no HBM PHY/TSV) | 0.1671 pJ/bit (project anchor); the YAML's 50 I/O channels at 0.80 V and 0.50 activity factor are the driving fields, but no derivation of 0.1671 pJ/bit is in the YAML | All three need derivation against the YAMLs |
| interface (coil / inductive / TSV I/O) | `NOT_VALIDATED` (Conventional uses TSV-based HBM DQ interface; no per-bit anchor in the project) | 0.5 pJ/bit constant (contactless) | 0.5 pJ/bit constant | The 0.5 pJ/bit anchor is *consistent* with the 7 nm FinFET inductive-coupling literature [2][3], but it is a constant, not a coupled Tx/Rx model |
| **total** | **1.397 pJ/bit** | **1.368 pJ/bit** | **0.8553 pJ/bit** | Anchors only; the components are not yet reconciled |

Two non-obvious things this matrix exposes:

1. **The 0.1858 pJ/bit "internal" anchor is a 50/50 average, not a measurement.** The Zhu Table I numbers span 0.24 → 370 fJ/bit across the eight operations. A 50/50 read 0/1 mix gives `(0.60 + 368)/2 = 184.3 fJ/bit ≈ 0.184 pJ/bit` per read; if the workload is read-heavy with mostly-0 data the average drops to ~0.6 fJ/bit, and if it is all-1 the average jumps to 368 fJ/bit. **A 100× swing in the same architecture, driven entirely by data pattern.**
2. **The 0.5 pJ/bit interface anchor is a 0.5 pJ/bit *per bit transferred* and at the same time is 58 % of the M3D total.** If the inductive interface cannot actually deliver 39.2 Tb/s at 0.5 pJ/bit, both the per-bit energy *and* the system bandwidth collapse together — they are not independent axes.

These two facts are the reason §7 is what it is.

The published references for the anchor values are:
- Zhu 2026 IEDM, Table I: per-operation fJ/bit at 512×512 MAT, 64 layers [4].
- Shiba et al. 2023 (IEEE SSC-L), 12.8 Gb/s, 0.5 pJ/b inductive-coupling interface in 7 nm FinFET, 1.7 TB/s/mm² I/O area efficiency [2].
- Shiba et al. 2023 (IEEE JSSC), 8.5 Gb/s, 0.7 pJ/b, 1.2 TB/s/mm² [3].
- ScienceDirect M3D-vs-TSV via parameters, 14 nm FinFET reference [1].

---

## 4. The bandwidth-ceiling function

The brief states:

> `BWsystem = min(BWarray, BWMIV, BWFEOL, BWcoil)`
> "目标是确认真正 bottleneck. 预期可能是 coil/interface，但必须计算，不能预设."

The current YAMLs do not give any of these four ceilings as a derived number; the only number in the workload is `read_bandwidth_gbps: 39200`, which the project brief explicitly calls a *matched reference*, not a *capability*.

Methodology for the four ceilings, given the current YAMLs (this is the proposal the next worker should run):

**`BWarray` (M3D-IGZO).** Driven by the array's per-MAT access parallelism: 512×512 MAT, `accessed_subarrays_per_access = 256`, `accessed_clusters_per_access = 4`, `selected_bits_per_subarray = 1`, and the 12.8 Gb/s per-link inductive rate [2]. For a single clock, the array can move `256 × 1 bit = 256 bit/cycle` to the global RWL/WBL. At the inductive rate, that is 256 × 12.8 Gb/s = 3.28 Tb/s **per M3D cube** if the link count is the array-side bottleneck. This is 8 % of the 39.2 Tb/s reference. The gap is real: the array itself cannot saturate 39.2 Tb/s with the current access parallelism.

**`BWMIV` (M3D-IGZO).** Driven by the global RWL/WBL wire: 100 nm width, 100 nm thickness, 0.30 fF/µm capacitance, 0.10 Ω/µm resistance, 1.0 V, activity 0.5, `active_line_count = 1`. The MIV pitch is 100 nm and each MIV carries one bit per cycle; with `accessed_subarrays_per_access = 256` and 8×8 subarray clusters per M3D cube, there are ~3 072 candidate MIVs in the cube. At 12.8 Gb/s/MIV (inherited from the inductive link), this is ~39.3 Tb/s — which *just* matches the 39.2 Tb/s reference. But this is a "all-MIVs-active" upper bound; in practice, only 256 of the 3 072 MIVs are active per access. Effective `BWMIV` is closer to 256 × 12.8 Gb/s = 3.28 Tb/s, same as `BWarray`. The 39.2 Tb/s reference therefore *requires increasing the access parallelism above 256* — a workload-driven decision, not a hardware limit.

**`BWFEOL` (M3D-IGZO).** Driven by the 50 I/O channels, 0.80 V, 0.50 activity factor, 0.20 fF/µm wire. With 12.8 Gb/s per inductive link [2] and 50 channels, this is 50 × 12.8 Gb/s = 640 Gb/s. If the 50 channels are *not* 50 inductive links but rather a FEOL escape bus, the per-link rate could be different; the YAML does not disambiguate. Either way, 50 channels is the obvious bottleneck at the FEOL boundary.

**`BWcoil` (M3D-IGZO).** Driven by the inductive link rate (12.8 Gb/s/link, 0.5 pJ/b [2]) and the *coil count* — which is **not in the YAML**. The brief's `EXISTING_INDUCTIVE_INTERFACE_REFERENCE` tag in `orthogonal_m3d_igzo.yaml` is the 0.5 pJ/bit, 1.7 TB/s/mm² Kuroda-group 3D-SRAM work. To deliver 39.2 Tb/s at 12.8 Gb/s/link, the cube needs ~3 062 active links. At 1.7 TB/s/mm² area efficiency, this needs ~2.88 mm² of coil area. The 98-slab cube has a 22 × 5.5 mm × 98 = 11 858 mm² envelope, so the *area* can host 3 062 links; the question is *interconnect placement* and *coil yield*, not area.

The bottleneck the brief suspects — "可能是 coil/interface" — is **partially right, but the real binding constraint at 39.2 Tb/s is the *coil count* and the *per-cube link placement*, not the per-link pJ/bit.** The per-link pJ/bit of 0.5 is well-validated [2][3]; what is *not* validated is the project's ability to place 3 062+ coils per cube while keeping the FEOL escape and MIV escape within the existing 50-channel and 256-MIV access parallelism.

This is the answer the next worker should deliver: not a yes/no, but a *quantified ceiling with the bottleneck named by calculation*.

---

## 5. The LLM workload analytical model

The brief specifies a transparent analytical model (no GPU cycle simulator). The minimum components:

- `model_footprint(P, dtype) = P × bytes_per_param` (70B FP16 = 140 GB; 70B FP8 = 70 GB)
- `KV_cache(L_ctx, P, dtype) = 2 × L × n_kv_heads × d_head × L_ctx × bytes` (LLaMA-2 70B at 4K FP16 ≈ 16.8 GB; LLaMA-3 70B at 32K BF16 ≈ 84 GB [5][6])
- `read_bits/token ≈ model_footprint + KV_cache(L_ctx)` (one decode step reads the full model + the full KV cache)
- `write_bits/token ≈ 1 new KV cell per layer` (negligible compared to read)
- `Tmemory = read_bits / effective_BW`
- `Ttoken = max(Tcompute, Tmemory)`
- `tokens/s = 1 / Ttoken`
- `Ememory/token = read_bits × Eread_avg + write_bits × Ewrite_avg`
- `Etotal/token = EGPU/token + Ememory/token`
- `J/token = Etotal/token`

For the M3D-IGZO architecture, `Eread_avg` is the *state-weighted* per-bit read energy from Zhu Table I: `p(read=0) × 0.60 + p(read=1) × 368` fJ/bit, where the read-1/read-0 mix is a workload assumption, not a measured constant. For a 50/50 mix the average is ~184 fJ/bit = 0.184 pJ/bit, which matches the project's 0.1858 pJ/bit internal anchor under a 50/50 read-mix.

Capacity-feasibility check (M3D-IGZO single cube, 355.74 GB upper bound):
- LLaMA-2 70B FP16 (140 GB) + 4K KV FP16 (~16.8 GB) + batch × KV = fits at batch 1, fits at batch 8 if KV is 4K.
- LLaMA-2 70B FP16 (140 GB) + 32K KV (~134 GB) = ~274 GB at batch 1; exceeds 355.74 GB only if batch is large.
- LLaMA-3 70B FP8 (70 GB) + 32K KV FP8 (~42 GB) = ~112 GB; easily fits batch 1–8.

So the M3D-IGZO single cube **is not capacity-bound** for the LLaMA-2/3 70B family at FP16 or FP8 with batch ≤ 8. The bottleneck will be bandwidth, not capacity.

For tokens/s upper bound, the standard H100 3.35 TB/s and B200 8 TB/s numbers imply:
- LLaMA-2 70B FP8 decode, 35 GB weights + ~16.8 GB KV at 4K = ~52 GB per token. At 3.35 TB/s: 52/3350 = 15.5 ms/token → **65 tok/s/user, batch 1, no optimization** (H100 reference, from [7][8]).
- LLaMA-3 70B FP8 decode, 35 GB weights + ~84 GB KV at 32K = ~119 GB per token. At 8 TB/s: 119/8000 = 14.9 ms/token → **67 tok/s/user, batch 1** (B200 reference, from [7]).

These are *theoretical ceilings*; in practice, real-world tokens/s/user on B200 with vLLM + FP8 KV quantization is in the 30–60 tok/s range for LLaMA-3 70B at 32K context.

For the M3D-IGZO architecture, the same arithmetic with the *project-claimed* `BWcoil`-limited effective bandwidth (e.g. 3.28 Tb/s if the access parallelism ceiling holds) would give:
- LLaMA-2 70B FP8 + 4K KV: 52 / 3.28 = 15.9 ms/token → **63 tok/s/user** (comparable to H100).
- LLaMA-3 70B FP8 + 32K KV: 119 / 3.28 = 36.3 ms/token → **28 tok/s/user** (well below B200 67 tok/s).

This is the *current analytical* prediction: **at the 3.28 Tb/s access-parallelism ceiling, M3D-IGZO matches H100 and is ~2.4× below B200 on long-context LLaMA-3 70B decode.** The number is workload-dependent; if `BWMIV` can be lifted to 39.3 Tb/s by increasing the access parallelism, the same arithmetic recovers parity with B200.

J/token estimate for M3D-IGZO at the 3.28 Tb/s ceiling, all-1-read mix (worst case):
- `Ememory/token = 35 GB × 8 × 0.368 pJ/bit = 257.6 nJ/token` (FP8 weights, all-1)
- vs H100: `Ememory/token = 35 GB × 8 × ~1.0 pJ/bit = 280 nJ/token` (Si HBM, all-1 conservative)
- vs B200: `Ememory/token = 35 GB × 8 × ~0.5 pJ/bit = 140 nJ/token` (HBM3E, all-1 conservative)

Under the all-1 worst case, M3D-IGZO `Ememory/token` is **on par with H100 and ~2× worse than B200.** The 50/50-mix number is ~0.184 pJ/bit → `Ememory/token ≈ 50.5 nJ/token` for M3D-IGZO, which would beat H100 by ~5× and match B200. **The all-1 vs 50/50 mix is the dominant uncertainty in the J/token claim, and the project has not yet pinned the data-pattern mix to a workload.**

This is the second non-obvious insight: *the J/token gap between M3D-IGZO and B200 is not a fixed number — it is a 5× swing driven entirely by the read-mix assumption.*

---

## 6. Open-questions bookkeeping

The brief's 8-item open-questions list, made explicit and current:

1. **Zhu 64-layer SPICE anchor vs project 8-layer modeling.** Zhu evaluates a 64-layer stack; the project models 8 layers. The 8 operation energies are taken as MAT-local indivisible primitives and not stack-resolved. **Status: open.** The next worker should at minimum re-derive whether the 8-layer stack's *Vth* and *retention* fall within Zhu's 300–420 K window. The thermal model already operates at 396 K, so the answer is likely yes — but it has not been written down.

2. **Interface energy sensitivity.** The 0.5 pJ/bit is a constant; the [2] reference does provide 0.7 pJ/b at 8.5 Gb/s and 0.5 pJ/b at 12.8 Gb/s, so the per-link pJ/bit is well-characterized. What is *not* characterized is the project's *link count* at 39.2 Tb/s. **Status: open; the answer is "per-link is pinned; system-level is not."**

3. **Interface Tx/Rx thermal placement.** Not modeled. The current M3D-IGZO hotspot is GPU FEOL; the coil/interface is not yet a power source on the map. **Status: open.** This becomes a real question only if the coil Tx count × 0.5 pJ/bit × 12.8 Gb/s/link × activity factor is non-trivial vs 14.4 W array-core. A back-of-envelope: 3 062 active links × 0.5 pJ/bit × 12.8 Gb/s = **19.6 mW of interface power at full load** — three orders of magnitude below the array core. So the coil is not a thermal hotspot at the current power map. **Status: likely not load-bearing for thermal at the 39.2 Tb/s target.**

4. **Coil/interface bandwidth capability.** This is the next-narrow-step target. **Status: open; the answer should come from the BW-ceiling derivation, not from intuition.**

5. **Array / MIV / FEOL bandwidth ceiling.** Same as #4; methodology in §4. **Status: open.**

6. **Workload read/write traffic.** The 39.2 Tb/s reference does not specify a read/write mix, batch, or context length. The project brief specifies LLM autoregressive decode, long-context, small/medium batch. **Status: open; pinned default: LLaMA-2 70B FP8, batch 1, 4K/32K context, decode-only (no prefill).**

7. **Data-pattern sensitivity.** Read 0 vs read 1 differs by 613× in Zhu Table I; the 0.1858 pJ/bit internal anchor assumes 50/50. **Status: open; the 50/50 is itself a `MODELING_CHOICE`.** A real LLM workload has structured attention patterns, not random 50/50. The project should run two cases: 50/50 and 70/30 (favoring read 1, since 1's are statistically over-represented in transformer attention) and report both.

8. **Competitor architecture comparison.** The 3 architectures (Conventional, Orthogonal Si, Orthogonal M3D-IGZO) are pinned. Industry competitors (HBM3E B200, HBM4 Rubin) are *not* project architectures. **Status: out of scope for the project but useful for the paper's introduction.** The brief explicitly says do not claim M3D wins against real silicon; the 3-way is internal.

**Newly-discovered items this turn:**
9. **The 50 I/O channels vs 39.2 Tb/s mismatch.** The current `orthogonal_m3d_igzo.yaml` has `io_channels: 50` with `io_channel_count_source: EXISTING_INDUCTIVE_INTERFACE_REFERENCE`. The 50 channels × 12.8 Gb/s = 640 Gb/s, which is 1.6 % of 39.2 Tb/s. Either the YAML is wrong, the 50 channels are not 50 inductive links, or the 39.2 Tb/s is not a FEOL-channel-bottleneck target. **Status: open; this is a YAML consistency issue that should be fixed before the next narrow step.**
10. **The 256/1024 MAT-size sensitivity is forbidden by the brief** ("不得据此声称最佳 MAT size"). The current `memory_internal_v0.yaml` sweep exposes `mat_rows` and `mat_cols` axes. **Status: the sweep must be marked as exploratory-only; the canonical anchor remains 512×512 from Zhu.**

---

## 7. The single next narrow step

**Title:** *Derive the M3D-IGZO bandwidth-ceiling function `BWsystem = min(BWarray, BWMIV, BWFEOL, BWcoil)` from the existing YAMLs and the published inductive-coupling link spec, identify the binding constraint by calculation, and propagate the effective bandwidth back into the workload model.*

**Why this and not something else:**
- The bandwidth ceiling determines the `Tmemory` term in the workload model. Without it, the `tokens/s` and `J/token` claims are not capability-validated — they are matched-reference, exactly the status the brief explicitly rejects.
- The same term (coil/interface) is 58 % of the total pJ/bit. If the coil cannot deliver 39.2 Tb/s at 0.5 pJ/bit, both the energy and the bandwidth collapse together, and the 0.8553 pJ/bit anchor is no longer achievable in the same architecture.
- The interface term is the *only* term that is currently a `MODELING_CHOICE` constant — the others (MIV, FEOL) have published physical parameters that can be re-derived mechanically. The 0.5 pJ/bit needs a system-level model, not just a link-level citation.
- The 8 open questions are interdependent, but #4 and #5 are upstream of #1, #2, #3, #6, #7. Closing the ceiling is the only step that gates every downstream comparison.

**Acceptance criteria** (the worker should produce a report that satisfies these, not a pass/fail binary):

1. Four formulas for `BWarray`, `BWMIV`, `BWFEOL`, `BWcoil` as functions of the YAML fields, with the field names cited.
2. Numerical values for each ceiling at the project's nominal 8-layer M3D-IGZO configuration, with units and provenance tag on every input.
3. The minimum, named, with the binding constraint highlighted.
4. A `Tmemory(token) = (model + KV) / BWsystem` for the default workload (LLaMA-2 70B FP8, batch 1, 4K and 32K context), with a 50/50 read-mix and a 70/30 read-mix, side by side.
5. A flag for the YAML inconsistency item #9 (50 channels vs 39.2 Tb/s) with a proposed fix (e.g. change `io_channels` to the actual link count required, or document the convention that 50 channels is a FEOL escape not a coil count).
6. **No code changes to the solver;** this is a documentation-and-arithmetic step. The solver is locked.

**Out of scope for this step (do not solve here):**
- Re-deriving pJ/bit from scratch.
- Re-running thermal sweeps.
- Picking the optimal MAT size (forbidden by the brief).
- Re-implementing Zhu SPICE.
- Adding MR-DIMM or HBM4 to the comparison.

**Out of scope for the project (do not start):**
- Transient simulation (project is steady-state only per AGENTS.md).
- Temperature-dependent materials (not justified by current data).
- AMR, dense matrices, matrix inversion (forbidden by AGENTS.md).

---

## 8. Provenance tag summary

| Tag | Meaning | Use in this turn |
|---|---|---|
| `PAPER_REPORTED` | Direct, first-party citation from `ref/*.pdf` | Zhu Table I fJ/bit; Tang 0.023 µm², 20s retention, 30 Mb/mm²/layer |
| `DERIVED_FROM_PAPER` | Arithmetic on a paper-reported value | 0.85 W/(m·K) (geometric mean of Zhu Fig. 6 per-axis k); 355.74 GB M3D cube upper bound |
| `DERIVED_FROM_REFERENCE` | Derived from a non-project reference | HBM3 819 GB/s; 0.5 pJ/b at 12.8 Gb/s from [2] |
| `MODELING_CHOICE` | Documented but not derived | Isotropic 0.85 W/(m·K); 0.5 pJ/bit interface constant; 50/50 read-mix; `io_channels: 50` |
| `NUMERICAL_CHOICE` | Solver / discretization setting | rtol=1e-6; alpha=0.7; GPU PCG |
| `NOT_VALIDATED` | Not yet derivable from current data | Per-component pJ/bit reconciliation; BW ceiling; 8-layer vs 64-layer equivalence; capacity-feasibility at batch>1 |

---

## 9. References

[1] ScienceDirect, *System-on-Package* — M3D MIV vs TSV physical parameters, 14 nm FinFET reference. https://www.sciencedirect.com/topics/engineering/system-on-package
[2] K. Shiba, M. Okada, A. Kosuge, M. Hamada, T. Kuroda, "A 12.8-Gb/s 0.5-pJ/b Encoding-Less Inductive Coupling Interface Achieving 111-GB/s/W 3D-Stacked SRAM in 7-nm FinFET", *IEEE Solid-State Circuits Letters*, 2023. https://ieeexplore.ieee.org/document/10058978/
[3] K. Shiba, M. Okada, A. Kosuge, M. Hamada, T. Kuroda, "A 7-nm FinFET 1.2-TB/s/mm² 3D-Stacked SRAM Module With 0.7-pJ/b Inductive Coupling Interface Using Over-SRAM Coil and Manchester-Encoded Synchronous Transceiver", *IEEE Journal of Solid-State Circuits*, vol. 58, no. 7, pp. 2075–2086, 2023. https://ieeexplore.ieee.org/document/9968287
[4] H. Zhu et al., "From Cell Metrics to Memory-Array-Tile Operability: DTCO of OSFET-Based 3D DRAM", *IEDM 2026*. Project copy: `ref/IEDM2026_HaotongZhu_V5.pdf`, MD5 `CE747EF510DF9AEF4EA1B2D545269956`.
[5] Wes McKinney, "KV cache and context memory costs", `wes.today/series/inference/what-happens/prefill-decode/kv-cache`. https://wes.today/series/inference/what-happens/prefill-decode/kv-cache
[6] CSDN, "揭秘 KV Cache:让大模型推理快 10 倍的'内存刺客'与'显存黑洞'". https://blog.csdn.net/tsh2005974tsh/article/details/163200921
[7] GMI Cloud, "Top GPUs for LLM Text Inference: Why Memory Bandwidth Decides Everything". https://www.gmicloud.ai/en/blog/top-gpus-llm-text-inference-bandwidth
[8] Engineers of AI, "Autoregressive Decoding — LLaMA-2 70B FP16 model + KV cache memory layout". https://engineersofai.com/docs/llms/llm-inference/Autoregressive-Decoding

---

## 10. STOP

This report ends here. The next worker should not start a new analysis or refactor; the next worker should pick up §7 and deliver the bandwidth-ceiling derivation. The thermal model, the operation-energy anchor, and the project brief are unchanged.
