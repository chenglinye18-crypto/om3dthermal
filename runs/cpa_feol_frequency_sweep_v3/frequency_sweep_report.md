# CPA FEOL frequency sweep on formal long-context v3

Models: Llama-3.1-8B / Qwen2.5-32B; H=20000/64000/126000; P=128; G=32; B=1/8/32.
18 frozen nominal v3 CPA rows and 90 new physical frequency executions. Old sweep values are not reused.
The existing set_frequency implementation is reused unchanged: 1.25/1.5/2/2.5/3 GHz at constant voltage. Each workload has one rebuilt 1 GHz CPA placement, frozen across frequency; its nominal first physical step must match v3.
MAC, Fabric, NoC and reduction retain the existing clock scaling; array/MIV, GPU and external 3.4 TB/s are fixed. NoC hop = fixed RC + (baseline hop - RC)/frequency ratio. Routing and pipeline register counts stay fixed.
Prefill time/energy is copied exactly from v3. All 32 growing Decode contexts are evaluated at every new frequency. Physical per-step/per-die events are recorded directly (including Qwen GQA), avoiding unsupported spatial projection rounding; energy equations are unchanged.
Every new frequency point gets a fresh Decode steady-state solve using the canonical FP64 GPU-PCG operator and die-grouped uniform BEOL + GPU FEOL power. No duty-cycle averaging. Tmax >85 C is infeasible; all infeasible samples are retained.
The selected complete row maximizes throughput among sampled thermally feasible frequencies, including nominal 1 GHz; ties prefer lower frequency. Energy is not selected independently. No crossing interpolation or extrapolation.
This is constant-voltage model sensitivity, not demonstrated silicon timing/voltage closure at the selected GHz.
HBM_HOST_OFFLOAD and HBM_RESIDENT_WAVE both remain in the unchanged v3 benchmark. HBM_BEST still chooses one complete policy by E2E throughput; no unnecessary host traffic or waves when the batch fits; infeasible wave retains offload.

| Model | H | B | GHz | tok/s | Gain % | tok/J | Energy-efficiency change % | Tmax C | Bottleneck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | LC20K | 1 | 3 | 684.678 | 10.439 | 1.9839 | 3.200 | 63.890 | ARRAY |
| Llama-3.1-8B | LC20K | 8 | 3 | 1281.808 | 31.919 | 2.6485 | 6.978 | 84.359 | EXTERNAL_BOUNDARY |
| Llama-3.1-8B | LC20K | 32 | 1.5 | 1183.296 | 18.867 | 2.5925 | 4.373 | 80.019 | EXTERNAL_BOUNDARY |
| Llama-3.1-8B | LC64K | 1 | 3 | 569.185 | 9.649 | 1.5100 | 2.708 | 66.480 | ARRAY |
| Llama-3.1-8B | LC64K | 8 | 3 | 907.544 | 25.446 | 1.8715 | 5.552 | 83.841 | EXTERNAL_BOUNDARY |
| Llama-3.1-8B | LC64K | 32 | 1.5 | 866.214 | 16.855 | 1.8514 | 3.812 | 80.996 | EXTERNAL_BOUNDARY |
| Llama-3.1-8B | LC126K | 1 | 3 | 459.786 | 9.360 | 1.1470 | 2.470 | 69.217 | ARRAY |
| Llama-3.1-8B | LC126K | 8 | 3 | 654.644 | 18.243 | 1.3393 | 3.949 | 84.387 | EXTERNAL_BOUNDARY |
| Llama-3.1-8B | LC126K | 32 | 1.5 | 636.189 | 13.180 | 1.3314 | 2.918 | 82.394 | EXTERNAL_BOUNDARY |
| Qwen2.5-32B | LC20K | 1 | 3 | 209.231 | 8.977 | 0.5721 | 2.597 | 66.599 | ARRAY |
| Qwen2.5-32B | LC20K | 8 | 2 | 407.908 | 26.496 | 0.8376 | 5.756 | 84.295 | EXTERNAL_BOUNDARY |
| Qwen2.5-32B | LC20K | 32 | 1.5 | 393.553 | 18.729 | 0.8201 | 4.129 | 83.057 | EXTERNAL_BOUNDARY |
| Qwen2.5-32B | LC64K | 1 | 3 | 185.482 | 9.256 | 0.4745 | 2.505 | 68.103 | ARRAY |
| Qwen2.5-32B | LC64K | 8 | 1.5 | 295.600 | 15.827 | 0.6284 | 3.560 | 81.287 | EXTERNAL_BOUNDARY |
| Qwen2.5-32B | LC64K | 32 | 1.5 | 304.712 | 17.226 | 0.6272 | 3.752 | 83.930 | EXTERNAL_BOUNDARY |
| Qwen2.5-32B | LC126K | 1 | 3 | 156.618 | 9.626 | 0.3825 | 2.487 | 70.322 | ARRAY |
| Qwen2.5-32B | LC126K | 8 | 1.5 | 227.937 | 14.224 | 0.4764 | 3.146 | 82.537 | EXTERNAL_BOUNDARY |
| Qwen2.5-32B | LC126K | 32 | 1.25 | 221.935 | 8.752 | 0.4698 | 1.960 | 81.414 | EXTERNAL_BOUNDARY |

```json
{
  "Llama-3.1-8B": {
    "cases": 9,
    "throughput_headroom_geomean": 1.1689020459928472,
    "energy_efficiency_geomean": 1.0398645236153696,
    "selected_frequency_distribution": {
      "3.0": 6,
      "1.5": 3
    }
  },
  "Qwen2.5-32B": {
    "cases": 9,
    "throughput_headroom_geomean": 1.1421052651081156,
    "energy_efficiency_geomean": 1.0331564497309036,
    "selected_frequency_distribution": {
      "3.0": 3,
      "2.0": 1,
      "1.5": 4,
      "1.25": 1
    }
  },
  "ALL": {
    "cases": 18,
    "throughput_headroom_geomean": 1.15542597388326,
    "energy_efficiency_geomean": 1.0365050600067385,
    "selected_frequency_distribution": {
      "3.0": 9,
      "1.5": 7,
      "2.0": 1,
      "1.25": 1
    }
  },
  "total_points": 108,
  "new_runs": 90,
  "infeasible_points": 27,
  "nonmonotonic_transitions": 0
}
```

Service sums overlap and are not additive E2E latency. Nominal bottleneck labels use largest stored service sum; new samples use existing per-stage bottleneck time classifications.
Nominal validation: 6 targeted tests passed; the preceding related suites had 657 passed, 11 pre-existing failures and 8 skipped. The missing legacy files/old assertions were not changed to conceal failures.
Frequency-sweep targeted validation: 3 passed, 0 failed in 0.79 s (targeted_tests.xml). Each newly executed point also checks physical-event, traffic and energy conservation.

Run in the native Windows Conda environment `om3dthermal`:
```powershell
python scripts/run_cpa_frequency_sweep_v3.py --cases 3 --workers 2
python scripts/run_cpa_frequency_sweep_v3.py --thermal
python scripts/analyze_cpa_frequency_sweep_v3.py
python scripts/plot_cpa_frequency_response_v3.py
```
Run thermal after performance to avoid memory pressure. Persistent placement caches are on F: and are not committed; completed result files are reused.
