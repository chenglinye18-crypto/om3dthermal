# Physical Decode Execution Anatomy v3

Fixed workload: Qwen2.5-32B, LC64K, B=8, Decode step 0 (context 64128), layer 0, nominal 1.0 GHz CPA.

## HBM policy and provenance

`e2e_timeline.pdf` and `e2e_timeline.svg` are clean system-level plots from the formal candidate rows. `HBM_BEST` selects `HBM_RESIDENT_WAVE` with safe resident batch 4 and waves `[4, 4]`. Wave 1 starts with resident historical KV. Wave 2 waits for Wave 1, admits historical KV once, then performs the same cached-history incremental Prefill and 32-step growing Decode. Historical H=64K is never recomputed as a full Prefill. The M3D rows use their exact canonical `prefill.latency_s`, `decode_s`, and `E2E_s` phase closure. Each row retains its canonical total E2E duration at the right edge.

- HBM-GPU: aggregate-only; validation `PASS_AGGREGATE_STEP_CLOSURE`; optimizer runs 0.
- M3D-GPU: exact; validation `PASS`; optimizer runs 0.
- DNS: exact; validation `PASS`; optimizer runs 0.
- CPA: exact; validation `PASS`; optimizer runs 0.

`decode_execution_timeline.pdf` and `decode_execution_timeline.svg` are a separate single-layer physical execution plot at Decode step 0, context 64128, layer 0, active B=8, and nominal 1.0 GHz. M3D-GPU and DNS are deterministic single-step replays using their canonical placement policies and are checked against frozen checkpoints. CPA is loaded from the existing serialized cache; the CPA optimizer is not rerun. The three rows align layer start to x=0 and retain absolute microsecond durations without per-row normalization. Each row reports `last operator end - first operator start` at the right edge.

## Representative layer latency

- M3D-GPU: 938.144 us, 1.000x relative to M3D-GPU and 0.468x relative to DNS.
- DNS: 439.173 us, 2.136x relative to M3D-GPU and 1.000x relative to DNS.
- CPA: 385.783 us, 2.432x relative to M3D-GPU and 1.138x relative to DNS.

The two formal plots are intentionally separate and contain no title, panel label, connector, handoff marker, or explanatory annotation. They use Times New Roman vector text. The prior combined figure and CPA operator-resource matrix remain in this directory as legacy/provenance artifacts but are not regenerated or included in the formal plots.

No formal benchmark, thermal solve, frequency sweep, or placement optimizer is run, and no canonical formal result is modified.
