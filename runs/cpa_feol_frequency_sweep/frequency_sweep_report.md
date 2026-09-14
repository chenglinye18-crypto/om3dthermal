# CPA FEOL frequency sensitivity

Constant-voltage, measured discrete frequencies; no voltage or new clock/leakage model.
18 original 1 GHz CPA rows reused verbatim. 90 new physical executions; all 32 Decode contexts evaluated.
One CPA placement rebuilt at 1 GHz per workload and reused at all five new frequencies. Old placement audit comparison waived by the user.
Prefill latency/energy reused. Thermal uses Decode-only per-die BEOL uniform power plus GPU FEOL power, with the existing FP64 GPU-PCG operator.
MAC/Fabric/NoC/Reduction scale with clock. Array/MIV/GPU/external 3.4 TB/s and energy coefficients stay fixed.
NoC hop = fixed RC + (baseline hop - RC)/frequency ratio. Baseline pipeline register count and routing are retained; no new pipeline hardware.
This is model frequency sensitivity, not a demonstrated silicon timing/voltage operating guarantee.

| Model | Context | B | Best GHz | TPS gain % | Tmax °C | tokens/J change % | First infeasible GHz | Bottleneck |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | LC20K | 1 | 3 | 10.439 | 63.890 | 3.200 |  | ARRAY |
| Llama-3.1-8B | LC20K | 8 | 3 | 31.919 | 84.359 | 6.978 |  | EXTERNAL_BOUNDARY |
| Llama-3.1-8B | LC64K | 1 | 3 | 9.649 | 66.480 | 2.708 |  | ARRAY |
| Llama-3.1-8B | LC64K | 8 | 3 | 25.446 | 83.841 | 5.552 |  | EXTERNAL_BOUNDARY |
| Llama-3.1-8B | LC126K | 1 | 3 | 9.360 | 69.217 | 2.470 |  | ARRAY |
| Llama-3.1-8B | LC126K | 8 | 3 | 18.243 | 84.387 | 3.949 |  | EXTERNAL_BOUNDARY |
| Llama-3.1-70B | LC20K | 1 | 3 | 9.816 | 67.933 | 2.750 |  | ARRAY |
| Llama-3.1-70B | LC20K | 8 | 1.5 | 18.506 | 83.776 | 4.025 | 2.0 | EXTERNAL_BOUNDARY |
| Llama-3.1-70B | LC64K | 1 | 3 | 12.238 | 70.422 | 3.163 |  | ARRAY |
| Llama-3.1-70B | LC64K | 8 | 1.5 | 16.944 | 84.099 | 3.676 | 2.0 | EXTERNAL_BOUNDARY |
| Llama-3.1-70B | LC126K | 1 | 3 | 13.600 | 72.303 | 3.396 |  | ARRAY |
| Llama-3.1-70B | LC126K | 8 | 1.5 | 15.312 | 84.001 | 3.334 | 2.0 | EXTERNAL_BOUNDARY |
| Llama-3.1-405B | LC20K | 1 | 3 | 8.040 | 75.859 | 1.987 |  | ARRAY |
| Llama-3.1-405B | LC20K | 8 | 1 | 0.000 | 81.410 | 0.000 | 1.25 | MAC |
| Llama-3.1-405B | LC64K | 1 | 3 | 11.019 | 77.306 | 2.541 |  | ARRAY |
| Llama-3.1-405B | LC64K | 8 | 1 | 0.000 | 80.025 | 0.000 | 1.25 | MAC |
| Llama-3.1-405B | LC126K | 1 | 3 | 13.185 | 77.963 | 3.007 |  | ARRAY |
| Llama-3.1-405B | LC126K | 8 | 1.25 | 8.075 | 83.639 | 1.760 | 1.5 | MAC |

Aggregate statistics:
```json
{
  "throughput_headroom_ratio": {
    "min": 1.0,
    "max": 1.319192920548739,
    "geomean": 1.1263026198704287
  },
  "tokens_j_ratio": {
    "min": 1.0,
    "max": 1.0697779612562412,
    "geomean": 1.030151378535222
  },
  "f_opt_distribution": {
    "3.0": 12,
    "1.5": 3,
    "1.0": 2,
    "1.25": 1
  },
  "new_runs_complete": 90,
  "total_points": 108,
  "nonmonotonic_transitions": 0,
  "earliest_measured_thermal_crossing_ghz": 1.25,
  "earliest_crossing_workloads": [
    "Llama-3.1-405B_LC20K_B8",
    "Llama-3.1-405B_LC64K_B8"
  ],
  "unconstrained_3ghz_increment_percent": {
    "Llama-3.1-8B_LC20K_B1": 0.7328926288425075,
    "Llama-3.1-8B_LC20K_B8": 2.1180387495729702,
    "Llama-3.1-8B_LC64K_B1": 0.7336262901865842,
    "Llama-3.1-8B_LC64K_B8": 1.282333759846388,
    "Llama-3.1-8B_LC126K_B1": 0.6396928372611965,
    "Llama-3.1-8B_LC126K_B8": 0.9981184639916041,
    "Llama-3.1-70B_LC20K_B1": 0.9792252085029318,
    "Llama-3.1-70B_LC64K_B1": 0.7976047270364717,
    "Llama-3.1-70B_LC126K_B1": 0.6430751963505843,
    "Llama-3.1-405B_LC20K_B1": 0.803804034427813,
    "Llama-3.1-405B_LC64K_B1": 1.1017608132250167,
    "Llama-3.1-405B_LC126K_B1": 1.3183649227340943
  }
}
```

Service sums are resource diagnostics, not additive components of end-to-end latency: Array/Fabric/MAC overlap through max().
For reused 1 GHz rows, bottleneck labels use the largest stored service sum; new points use the existing per-stage bottleneck time classification.
MIV is embedded in Array group service; reduction_service_s separates local reduction from stored INTER_REGION_NOC. No invented utilization metric.
All infeasible samples retained. No saturation threshold or crossing interpolation applied.
Nonmonotonic transitions: 0; see nonmonotonic_diagnostics.json (values are not corrected).

Reproduce/resume in the om3dthermal Conda environment:
```powershell
python scripts/run_cpa_frequency_sweep.py --cases 2 --workers 5
python scripts/run_cpa_frequency_sweep.py --thermal
python scripts/analyze_cpa_frequency_sweep.py
```
Run thermal after performance to avoid concurrent operator/placement memory pressure. Completed points are reused.
