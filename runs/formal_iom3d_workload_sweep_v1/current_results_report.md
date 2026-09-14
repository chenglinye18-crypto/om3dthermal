# Formal B=1/8 architecture rebaseline

Scope: Llama-3.1-8B/70B/405B; W1=(2000,512,256), W2=(20000,512,256), W3=(126000,512,512); B=1,8. Throughput is aggregate generated B*G / (Prefill + Decode), tok/s.

## Root cause and claim boundary

BUG_FIXED: the old GPU-only path inherited the NMP whole-row resident mapping, GROUP_DIRECT nearest-port serialization, and a sum of per-operator GPU rooflines. HBM used the full-step active-weight/KV GPU ledger. This was an inconsistent execution abstraction; it was not a proven duplicate MAT/MIV-plus-bandwidth sum.

For 8B W1 B1, old Decode=1.6831122578856168 s and external component=1.6813510243409877 s, versus HBM Decode=1.1921522536921827 s. At context 2512, each layer K/V has a maximum per-port payload of 16,384 bytes on a 1 GB/s port under the whole-row mapping; each K/V family totals ~0.524451 ms over 32 layers instead of ~0.079050 ms at the thermal streaming cap. Q/O and FFN_DOWN have similar port concentration. Old stages use max(array,boundary,GPU); component sums overlap and are not additive latency.

The corrected GPU-only formal semantics use the existing first-order hierarchical GPU memory service model, not the physical NMP row/tile schedule. This is an aggregate streaming prediction, not a claim that concentrated GROUP_DIRECT traffic can attain the aggregate cap without changing the scheduling abstraction. The physical NMP route constraints remain valid and untouched.

Internal closure = 318 slabs * 50 lanes/slab * 32 bytes / (12.996014061205 ns * 1.0) = 39.150465489 TB/s. The existing full physical cycle includes MAT, MIV and FEOL routing. Eight layers share lanes. Boundary=15.9 TB/s; GPU=4.8 TB/s; thermal=3.395783558119 TB/s. Effective=min(all four). No additional independent startup is established by this closure; adding its access cycles again would duplicate service.

Both GPU backends use the same active weights/KV traffic, FLOPs, 989.5 TFLOP/s Decode ceiling and 700/700 TFLOP/s Prefill family ceilings. For HBM-local pairs the corrected M3D GPU time is no larger; host-offload differences retain the 416.34 GB/s Grace C2C model. All 18 existing M3D/NMP Prefills equal their GPU compute time, so no Prefill latency or NMP energy event was changed.

SEMANTIC_RENAME: formal systems are HBM_GPU, M3D_GPU, M3D_MAC_NMP. ADAPTIVE_POLICY: pre-dispatch minimum modeled Decode latency, tie to GPU, zero inference latency/energy. GPU-selected performance, traffic and energy are exact M3D_GPU; CPA and all NMP activity are inactive. NMP-selected results copy the frozen CPA checkpoint. UNCHANGED_VALID_MODEL: all NMP physical equations, events, resource limits and CPA objective.

## Primary performance

| Model | W | B | HBM_GPU | M3D_GPU | NMP_CPA candidate | M3D_MAC_NMP | Executor | Proposed/GPU |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Llama-3.1-8B | W1 | 1 | 212.761 | 219.050 | 736.944 | 736.944 | NMP_CPA | 3.364281 |
| Llama-3.1-8B | W1 | 8 | 1394.280 | 1433.306 | 1244.597 | 1433.306 | GPU | 1.000000 |
| Llama-3.1-8B | W2 | 1 | 183.736 | 189.146 | 660.068 | 660.068 | NMP_CPA | 3.489720 |
| Llama-3.1-8B | W2 | 8 | 685.080 | 704.511 | 1039.910 | 1039.910 | NMP_CPA | 1.476072 |
| Llama-3.1-8B | W3 | 1 | 103.035 | 106.072 | 490.966 | 490.966 | NMP_CPA | 4.628615 |
| Llama-3.1-8B | W3 | 8 | 178.692 | 179.854 | 682.145 | 682.145 | NMP_CPA | 3.792781 |
| Llama-3.1-70B | W1 | 1 | 23.350 | 24.040 | 128.734 | 128.734 | NMP_CPA | 5.354929 |
| Llama-3.1-70B | W1 | 8 | 170.171 | 173.021 | 220.916 | 220.916 | NMP_CPA | 1.276811 |
| Llama-3.1-70B | W2 | 1 | 22.484 | 23.005 | 117.399 | 117.399 | NMP_CPA | 5.103181 |
| Llama-3.1-70B | W2 | 8 | 65.735 | 130.695 | 188.861 | 188.861 | NMP_CPA | 1.445045 |
| Llama-3.1-70B | W3 | 1 | 11.580 | 18.575 | 90.994 | 90.994 | NMP_CPA | 4.898704 |
| Llama-3.1-70B | W3 | 8 | 10.115 | 55.501 | 131.314 | 131.314 | NMP_CPA | 2.365994 |
| Llama-3.1-405B | W1 | 1 | Capacity infeasible | 4.157 | 36.287 | 36.287 | NMP_CPA | 8.728691 |
| Llama-3.1-405B | W1 | 8 | Capacity infeasible | 30.794 | 63.369 | 63.369 | NMP_CPA | 2.057845 |
| Llama-3.1-405B | W2 | 1 | Capacity infeasible | 4.103 | 33.453 | 33.453 | NMP_CPA | 8.152582 |
| Llama-3.1-405B | W2 | 8 | Capacity infeasible | 28.063 | 54.598 | 54.598 | NMP_CPA | 1.945572 |
| Llama-3.1-405B | W3 | 1 | Capacity infeasible | 3.851 | 26.900 | 26.900 | NMP_CPA | 6.985832 |
| Llama-3.1-405B | W3 | 8 | Capacity infeasible | 19.369 | 39.843 | 39.843 | NMP_CPA | 2.057010 |

B8 decisions: NMP=8; GPU=1.

Selection minimizes latency only. It does not guarantee better tokens/J. Candidate and selected energy appear together in adaptive_selection_b1_b8.csv; HBM absolute tokens/J remains unresolved rather than synthesized.

## CPA vs Uniform placement ablation

| Model | W | B | CPA/Uniform E2E speed | Used by selected executor |
|---|---|---:|---:|---|
| Llama-3.1-8B | W1 | 1 | 1.182045 | True |
| Llama-3.1-8B | W1 | 8 | 1.335673 | False |
| Llama-3.1-8B | W2 | 1 | 1.123618 | True |
| Llama-3.1-8B | W2 | 8 | 1.214638 | True |
| Llama-3.1-8B | W3 | 1 | 1.052579 | True |
| Llama-3.1-8B | W3 | 8 | 1.100258 | True |
| Llama-3.1-70B | W1 | 1 | 1.193539 | True |
| Llama-3.1-70B | W1 | 8 | 1.268139 | True |
| Llama-3.1-70B | W2 | 1 | 1.135363 | True |
| Llama-3.1-70B | W2 | 8 | 1.159460 | True |
| Llama-3.1-70B | W3 | 1 | 1.066208 | True |
| Llama-3.1-70B | W3 | 8 | 1.073608 | True |
| Llama-3.1-405B | W1 | 1 | 1.185480 | True |
| Llama-3.1-405B | W1 | 8 | 1.226646 | True |
| Llama-3.1-405B | W2 | 1 | 1.131160 | True |
| Llama-3.1-405B | W2 | 8 | 1.126824 | True |
| Llama-3.1-405B | W3 | 1 | 1.072211 | True |
| Llama-3.1-405B | W3 | 8 | 1.059207 | True |

Uniform is an NMP placement ablation, never a fourth architecture. CPA optimizer runtime is preserved as an offline planning diagnostic; a GPU-selected case has zero CPA moves/runtime in the formal result.

## Reuse and validation

NMP physical Decode checkpoints reused: 36; complete Decode steps reused: 12288. Prefill recomposed: 0; GPU-only cases recomputed: 18. All original checkpoint files and legacy CSVs retain their original SHA-256. Source/config digest verification, ordered contexts, component totals, memory/interface/MIV/MAC event conservation, and coefficient-only energy reproduction pass for every primary physical checkpoint.

Regression tests cover identical GPU workload/compute semantics, internal and boundary bottleneck counterexamples, single-counted service latency, exact adaptive candidate copying, ties and GPU fallback, CPA eligibility, and immutable legacy/B32 artifacts. See primary_validation.json and the final Git/test report for execution evidence.

Independent correctness repair: thermal_sensitivity.no_nmp_read_energy now drops slab-indexed route maps inherited by a resized floorplan copy. Warm/cold read-stress events are identical; the canonical floorplan is not mutated. This diagnostic is not called by physical Decode/CPA. Its original and reviewed source digests are recorded separately; no saved thermal or primary result is invalidated. Legacy test goldens retain explicit legacy service/workload/layout fixtures instead of being applied to incompatible current defaults.

B32 untouched and frozen in this task.
No B32 result was recomputed, selected, extrapolated, or overwritten.

Verdict: B1/B8 GPU timing semantics are consistent within the stated aggregate streaming model; NMP physical execution remains unchanged; modeled adaptive latency is the candidate minimum; CPA applies only to NMP execution. These B1/B8 architecture semantics can be frozen. Workload thermal closure and local hardware validation remain outside this benchmark; HBM unresolved energy is retained.

## Final test validation

Full suite: **1060 passed, 0 failed**, 1084.47 s (18:04). Command: `python -u -m pytest -q --durations=10`, native Windows Conda `om3dthermal`, Python 3.11.15. Final targeted groups: 61 passed and 60 passed. Full log and SHA-256 are recorded in manifest.json and primary_validation.json. No numerical tolerances or production hardware/workload parameters were changed.
