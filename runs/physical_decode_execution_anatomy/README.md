# Physical Decode Execution Anatomy

This is derived evidence from the frozen formal benchmark, not a new benchmark.
Workload: Llama-3.1-8B, LC20K, H=20000, P=128, G=32, B=1,
M3D_NMP_CPA, 1 GHz. The figure shows layer 0 of Decode step 1,
whose context is 20128. Full-step latency is 1.4410936841650986 ms;
the layer occupies 43.8436930580433 microseconds.

## Sources and replay

The canonical candidate and its fingerprint-selected JSONL checkpoint supply
the expected step latency, component totals, traffic, and energy events.
The existing F: frequency-study cache contains a CPA plan rebuilt at 1 GHz.
It was loaded read-only and replayed for this one unique checkpoint only.
The recovered step matches the canonical latency, physical component totals,
traffic totals, and event ledger at the existing numerical precision.
No CPA optimizer, Prefill, thermal solve, other workload, or frequency is run.
`provenance.json` records the candidate, checkpoint, cache and physical-source
SHA-256 values. The initial exploratory replay and extraction replay both
targeted this same single checkpoint. Future normal plot invocations reuse
`replayed_step.json` and do not replay execution.

## Timeline definition

The scheduler serializes operators in dependency order and sums their complete
stage latencies. Operator start/end are therefore exact cumulative sums within
this model. `timeline.csv` stores both step-relative and layer-relative times.
The plot is a reconstructed dependency-ordered operator timeline, **not a
resource occupancy Gantt**. A bar spans the complete operator duration and its
lane identifies the largest service component. A blank lane does not mean that
resource is inactive: e.g. boundary service is present but is not the largest
component for these operators. No arbitrary internal event ordering, overlap,
queue timestamps, or software synchronization cost is invented.

## Resource matrix definition

`operator_resource_matrix.csv` retains absolute service seconds for every
layer-0 operator, including small operators and both KV appends. Figure (b)
omits operators shorter than 0.1 microseconds solely for legibility. It shows
service(resource, operator) / max_resource_service(operator); black outlines
mark maxima. Repeated operators remain distinct via stage_index.

* ARRAY includes MAT and MIV service; layers serialize within a group.
* MAC is the maximum assigned tile load / tile FLOP rate.
* LOCAL_FABRIC is the maximum region payload service plus route startup.
* INTER_REGION_NOC includes structured NoC **and reduction** service.
* EXTERNAL_BOUNDARY sums the operator's input/output boundary service.
* GPU_COMPUTE is the existing field name; these small GPU stages use GPU-local
  bytes / bandwidth, not a separately measured GPU kernel compute duration.

NMP stage time = boundary + NoC/reduction + max(array, fabric, MAC).
GPU physical stage time = max(array, boundary, GPU service).
Thus heatmap columns must not simply be summed into operator latency.
Array dominates Q/K/V/O/FFN Down; Fabric dominates QK/Gate/Up;
NoC/reduction dominates AV. GPU Softmax and AV reduction are explicit stages.

## Capacity-controlled statistics

`subset_membership.csv` combines canonical HBM_only_fit with the capacity
audit's zero/nonzero Grace resident bytes. There are five resident cases:
8B LC20K B1/B8, LC64K B1/B8, and LC126K B1. The remaining 13 overflow.
`subset_statistics.csv` contains paired geometric means for both E2E tokens/s
and tokens/J. `representative_absolute_metrics.csv` contains existing absolute
metrics, not recomputed simulation results. TPOT is a batch-step interval;
one step emits B aggregate tokens.

| E2E throughput ratio | HBM-resident (5) | Overflow (13) | All (18) |
|---|---:|---:|---:|
| M3D-GPU / HBM-GPU | 1.0374688633 | 3.1449322058 | 2.3111250775 |
| DNS / HBM-GPU | 2.7724294064 | 9.3650640515 | 6.6782620874 |
| CPA / HBM-GPU | 3.0692074657 | 10.1454439406 | 7.2784078211 |
| DNS / M3D-GPU | 2.6723013138 | 2.9778270051 | 2.8896151716 |
| CPA / M3D-GPU | 2.9583610404 | 3.2259658641 | 3.1492920448 |

## Reproduce

From the repository root, in the required native Conda environment:

```powershell
conda run --no-capture-output -n om3dthermal python scripts/plot_physical_decode_execution_anatomy.py
```

`--extract` explicitly repeats the same single-step replay and requires the
recorded F: cache. The normal invocation uses saved extraction, regenerates
CSV statistics and deterministic vector SVG/PDF, and verifies canonical files
remain byte-identical. It does not rebuild a missing cache.

Formal results are unchanged. `canonical_preservation.json` records SHA-256
for all 390 existing formal JSON/JSONL/CSV files checked by this extraction.
No B32 result is read, recomputed, or rewritten.
