# Attention-only NMP: 8B / W1 / B8 single-point probe

FAIL — GPU remains faster. `ATTENTION_NMP_DOES_NOT_RECOVER_SHORT_CONTEXT_B8`.

Scope: Llama-3.1-8B, H=2000, P=512, G=256, B=8; numerator B*G=2048.
Base frozen HEAD: `3e0604caf028d7f2006555973ff341e1a855e37c`. All 256 contexts (2512 through 2767) are explicitly executed.
Only Attention-NMP is newly simulated. No other workload, batch, model, reference execution, thermal sweep, scheduler, or hardware change.

## Performance and existing event energy

| system | E2E_s | Decode_s | E2E_tok_s | speedup_vs_M3D_GPU | speedup_vs_full_MAC_CPA | J_per_token | tokens_per_J |
|---|---|---|---|---|---|---|---|
| M3D_GPU | 1.4288645843899812 | 1.3402520031310097 | 1433.3058726305708 | 1.0 | 1.1516222144743602 | 0.2918336778769192 | 3.426609318276658 |
| M3D_NMP_FULL_CPA | 1.6455121968591766 | 1.556899615600205 | 1244.59727731527 | 0.8683403180585867 | 1.0 | 0.3146541263821507 | 3.1780927569514525 |
| M3D_ATTENTION_NMP | 1.694183196003308 | 1.6055706147443367 | 1208.842116266629 | 0.8433943789318469 | 0.9712717023407212 | 0.3431253686273129 | 2.9143866686410886 |

Prefill is copied exactly from the frozen GPU result: 0.08861258125897142 s. Event energy uses existing coefficients and GPU bit proxy; no new energy model. Workload-specific thermal closure remains pending.

## Semantics and implementation audit

Existing ATTENTION_NMP already maps exactly ATTENTION_QK and ATTENTION_AV to NMP. Embedding, Q/K/V projections, O projection, FFN Gate/Up/Down, LM Head, RMSNorm, Softmax, residual and other small operators stay on GPU. KV appends remain GPU-origin memory writes.

The existing ATTENTION_NMP GPU helper inherited legacy GROUP_DIRECT/per-operator physical timing. The isolated probe adapter replaces only GPU timing with the frozen M3D aggregate streaming closure, grouping the fixed graph's contiguous GPU operators between QK/AV handoffs. This is no dynamic scheduler. Small GPU operators retain their existing GPU-local service. Weight reads are shared across the batch; FLOPs and active weight/KV byte totals are independently checked against the canonical ledger for every context. No old GPU physical latency is added to the streaming closure.

GPU effective BW = min(thermal 3.395783558, internal 39.150465489, boundary 15.900000000, GPU 4.800000000) TB/s. MAT/MIV/FEOL are already in the internal service cycle; independent startup is zero.

CPA is the existing physical optimizer, enumerating only QK/AV entries while retaining complete residency/capacity lookup. Its objective, proposals, endpoint guards, request merging, and all equations are unchanged. No GPU weight operator is optimized. All 512 QK/AV CPA audit rows match the frozen Full-MAC attention trace exactly except wall-clock optimizer runtime. NMP remains 1 GHz with frozen MAC, Fabric, NoC and boundary capacities.

## Decode latency (seconds)

| system | gpu_compute_s | gpu_memory_s | nmp_array_s | nmp_fabric_s | nmp_mac_s | nmp_noc_s | nmp_reduction_s | boundary_transfer_s | total_decode_s |
|---|---|---|---|---|---|---|---|---|---|
| M3D_GPU | 0.033929482 | 1.340252 | 0 | 0 | 0 | 0 | 0 | 1.340252 | 1.340252 |
| M3D_NMP_FULL_CPA | 0 | 0.0787762398 | 0.344362142 | 0.524695194 | 0.482032192 | 0.260617472 | 0 | 0.638465753 | 1.55689962 |
| M3D_ATTENTION_NMP | 0.0310652662 | 1.21037673 | 0.0144277889 | 0.0715662165 | 0.05395104 | 0.151379968 | 0 | 0.157151936 | 1.60557061 |

Component demand sums are NOT additive wall time: GPU streaming overlaps compute; NMP core is max(array, Fabric, MAC). Physical NMP stage = boundary + NoC + core + local reduction. NoC includes the existing inter-region tree communication/add rounds; the separate reduction column is the model's local tile-to-region reduction term. Full-MAC ARRAY includes the original KV append stages. Its legacy GPU_COMPUTE field consists of small-op memory service and is correctly labelled GPU memory above.

The Full-MAC checkpoint retained NoC+local reduction together. The split above reuses local AV reduction from the identical deterministic QK/AV CPA trace; all other physical operators have zero local reduction. The frozen combined component and total Decode are unchanged. Full-MAC is not rerun.

Attention GPU segment wall time = 1.210376727 s; NMP core wall time = 0.086661984 s. See attention_operator_diagnostics.csv for QK/AV resource constraints and full_mac_saved_operator_diagnostics.csv for saved Full-MAC endpoint constraints (not extrapolated full-trace results).

An optimistic diagnostic aggregates ALL GPU work, removes ALL GPU small-op service, and retains the same physical QK/AV. Its Decode lower bound is 1.526794375 s, corresponding to at most 1267.791990 E2E tok/s under this serial GPU/NMP handoff model. This is algebra on the single run, not a second execution policy or simulation. It tests whether GPU segmentation/small-op penalties alone explain the result.

## Traffic and D_int

The GPU KV streaming service removed by offload is 0.208651516 s, while replacement physical QK/AV requires 0.395193887 s. Weight-heavy GPU reads remain. This compares the fixed partition's benefit and cost directly, without changing any parameter.

Decimal GB = 10^9 bytes; values are per generated token (divide aggregate step traffic by B). Weight/KV reads count physical memory reads regardless of consumer; internal_NMP counts NMP-local array payload, not replicated Fabric or hop traffic. Internal Fabric and NoC hop traffic are separately observable in checkpoints/CSV.

| system | weight_read_GB_per_token | KV_read_GB_per_token | boundary_GB_per_token | internal_NMP_GB_per_token | D_int_bytes_per_token | GPU_to_NMP_activation_bytes_per_token | NMP_to_GPU_activation_excluding_partial_bytes_per_token | partial_reduction_boundary_bytes_per_token | other_boundary_bytes_per_token |
|---|---|---|---|---|---|---|---|---|---|
| M3D_GPU | 1.8761728 | 0.345964544 | 2.22226842 | 0 | 0 | 0 | 0 | 0 | 2.22226842e+09 |
| M3D_NMP_FULL_CPA | 1.8761728 | 0.345964544 | 1.05858714 | 2.22213734 | 1.05845606e+09 | 883309568 | 8422912 | 166723584 | 131072 |
| M3D_ATTENTION_NMP | 1.8761728 | 0.345964544 | 2.13720064 | 0.345964544 | 260896768 | 88767488 | 5405696 | 166723584 | 1.87630387e+09 |

External D_int is GPU→NMP activation + NMP→GPU ordinary activation + AV partial output. These columns are disjoint. Other boundary bytes are bulk GPU weight reads and/or KV append, excluded from D_int. Intra-NMP NoC hop traffic is NOT added to external D_int.

Full-MAC D_int = 1058456064.000 bytes/token; Attention D_int = 260896768.000 bytes/token; reduction = 75.351195%.

Full-MAC directional bytes are reconstructed from actual resident slab ownership and the frozen transfer equations, and match saved interface event totals exactly. CPA resident migration stays within each slab; all non-embedding operators occupy every slab at both context endpoints, preserving ownership throughout this context range. This byte audit does not execute Full-MAC.

Full-MAC external boundary service is 0.638465753 s of 1.556899616 s Decode (41.01%). Its Fabric demand sum is 0.524695194 s, MAC demand sum 0.482032192 s, and NoC+local reduction 0.260617472 s. Thus D_int/boundary is a major cause and the largest reported serial component, not the sole cause: Fabric/MAC core service, reduction, and the GPU batching advantage also matter. A single D_int/B_link scalar does not capture the physical per-port and routing limits.

## Research decision

8B-W1-B8 is GPU-favorable under the current hardware model and existing fixed execution mechanisms. Do not introduce extra scheduling or overclocking solely to recover this point. The paper can restrict its primary scope to long-context LLM inference rather than introducing additional mechanisms solely to improve this short-context point.

If the title and central claims explicitly target long-context inference, W1 (H=2K) can move to a short-context boundary/control experiment; W2 (20K) and W3 (126K) cover two distinct long-context regimes, but alone do not prove every context length or application. Retain transparent scope and the W1 fallback evidence. This task does not remove W1 or change any benchmark.

## Integrity and regression gates

All frozen source/config and formal artifacts are SHA-256 unchanged, including B32 and STOPPED_BY_USER states. Original reference throughput and Prefill are reused. Only this case has new Attention-NMP execution. The probe checks every context for FLOP, weight/KV, interface, directional boundary and MAC-event conservation. GPU CPA optimization is absent. Targeted tests: 11 passed. Full pytest suite: 1071 passed, 0 failed, 0 skipped in 1087.76 seconds (native Windows Conda om3dthermal). All 81 test files were included; the artifact checks ran last after the single physical run completed. Raw step/CPA checkpoints stay local; their SHA-256 hashes are recorded in manifest.json.
