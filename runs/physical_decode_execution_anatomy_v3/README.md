# Physical Decode Execution Anatomy v3

Fixed workload: Qwen2.5-32B, LC64K, B=8, Decode step 0 (context 64128), layer 0, nominal 1.0 GHz CPA.

## HBM policy and provenance

Panel (a) is a system-level E2E timeline from the formal candidate rows. `HBM_BEST` selects `HBM_RESIDENT_WAVE` with safe resident batch 4 and waves `[4, 4]`. Wave 1 starts with resident historical KV. Wave 2 waits for Wave 1, admits historical KV once, then performs the same cached-history incremental Prefill and 32-step growing Decode. Historical H=64K is never recomputed as a full Prefill. The M3D rows use their exact canonical `prefill.latency_s`, `decode_s`, and `E2E_s` phase closure.

- HBM-GPU: aggregate-only; validation `PASS_AGGREGATE_STEP_CLOSURE`; optimizer runs 0.
- M3D-GPU: exact; validation `PASS`; optimizer runs 0.
- DNS: exact; validation `PASS`; optimizer runs 0.
- CPA: exact; validation `PASS`; optimizer runs 0.

Panel (b) is a single-layer physical execution zoom at Decode step 0, context 64128, layer 0, active B=8, and nominal 1.0 GHz. M3D-GPU and DNS are deterministic single-step replays using their canonical placement policies and are checked against frozen checkpoints. CPA is loaded from the existing serialized cache; the CPA optimizer is not rerun. The three rows align layer start to x=0 and retain absolute microsecond durations without per-row normalization.

## Representative layer latency

- M3D-GPU: 938.144 us, 1.000x relative to M3D-GPU and 0.468x relative to DNS.
- DNS: 439.173 us, 2.136x relative to M3D-GPU and 1.000x relative to DNS.
- CPA: 385.783 us, 2.432x relative to M3D-GPU and 1.138x relative to DNS.

These ratios are single-layer anatomy ratios, not end-to-end throughput speedups. Panel (a) and Panel (b) use different time scales; Panel (b) is a logical zoom rather than an equal-scale crop. The prior CPA operator-resource matrix and resource-diversity audit remain in this directory as provenance artifacts but are not shown in the default figure.

No formal benchmark, thermal solve, frequency sweep, or placement optimizer is run, and no canonical formal result is modified.
