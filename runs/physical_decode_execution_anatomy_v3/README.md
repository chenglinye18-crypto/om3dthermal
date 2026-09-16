# Physical Decode Execution Anatomy v3

Fixed workload: Qwen2.5-32B, LC64K, B=8, Decode step 0 (context 64128), layer 0, nominal 1.0 GHz CPA.

## HBM policy and provenance

`HBM_BEST` selects `HBM_RESIDENT_WAVE` with safe resident batch 4 and waves `[4, 4]`. The HBM evaluator has no operator scheduler. Its row is therefore an aggregate-only active-wave reference: the canonical first Decode-step latency for B=4 divided by 64 layers. It is not split into invented operator intervals. Request-level wave queuing remains represented only in the formal E2E metrics.

- HBM-GPU: aggregate-only; validation `PASS_AGGREGATE_STEP_CLOSURE`; optimizer runs 0.
- M3D-GPU: exact; validation `PASS`; optimizer runs 0.
- DNS: exact; validation `PASS`; optimizer runs 0.
- CPA: exact; validation `PASS`; optimizer runs 0.

M3D-GPU and DNS are deterministic single-step replays using their canonical placement policies and are checked against their frozen checkpoints. CPA is loaded from the existing nominal-frequency serialized cache; the CPA optimizer is not rerun. All four rows align their layer reference to x=0 and retain absolute microsecond durations without per-row normalization.

## Representative layer latency

- HBM-GPU: 664.232 us, 1.000x vs the HBM aggregate mean-layer reference.
- M3D-GPU: 938.144 us, 0.708x vs the HBM aggregate mean-layer reference.
- DNS: 439.173 us, 1.512x vs the HBM aggregate mean-layer reference.
- CPA: 385.783 us, 1.722x vs the HBM aggregate mean-layer reference.

These ratios are single-layer anatomy ratios, not end-to-end throughput speedups.

The right panel retains the v2 CPA matrix exactly: `T_norm(o,r)=T(o,r)/max_r T(o,r)`. Each row maximum is 1 and the black outline marks `argmax_r`. `AV_REDUCTION` remains `AV Reduce` because it is an independently scheduled GPU stage.

No formal benchmark, thermal solve, frequency sweep, or placement optimizer is run, and no canonical formal result is modified.
