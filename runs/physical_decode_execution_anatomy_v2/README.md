# Physical Decode Execution Anatomy v2

This directory is derived evidence from frozen `formal_long_context_v3` CPA results. It does not modify any formal result.

## Candidate audit

- Qwen2.5-32B / LC64K / B8: 5 dominant-resource types; Boundary-dominant=4, GPU-dominant=2.
- Qwen2.5-32B / LC64K / B32: 4 dominant-resource types; Boundary-dominant=4, GPU-dominant=2.
- Qwen2.5-32B / LC126K / B8: 4 dominant-resource types; Boundary-dominant=4, GPU-dominant=2.

The selected case is **Qwen2.5-32B / LC64K / B8**. The preferred non-extreme case already exposes memory-side, FEOL, and GPU-side bottlenecks with 5 distinct dominant resources, so no speedup- or temperature-based selection was used.

## Trace provenance

- Decode step: 0 (context 64128)
- Transformer layer: 0
- CPA frequency: 1.0 GHz
- Replay: deterministic, read-only replay of the existing serialized CPA engine
- Optimizer executions: 0
- Timeline: exact simulated start/end intervals from the sequential scheduler, measured relative to layer 0 start
- Checkpoint comparison: PASS for context, total latency, traffic, and every recorded component sum

The heatmap metric is `T_norm(o,r) = T(o,r) / max_r T(o,r)`. Every row therefore has a maximum of exactly 1. The dominant resource is `argmax_r T(o,r)` and is outlined in black. `Array` includes MIV service. `NoC` contains the modeled collective/reduction traffic where applicable.

`AV_REDUCTION` is retained as **AV Reduce** because the canonical Decode scheduler emits it as a distinct GPU stage with its own latency. It is not presented as an additional Transformer matrix operator.

The source checkpoint does not contain stage-level intervals. Stage detail is recovered by deterministic replay from the already placed nominal-frequency engine cache; the replayed step is then checked against the frozen checkpoint. No benchmark, placement optimization, frequency sweep, or thermal simulation is run.
