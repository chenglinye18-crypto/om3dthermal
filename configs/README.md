# Configuration

- `cases/`: geometry, memory power and thermal inputs.
- `architecture/`: canonical-case references and roles.
- `platform/`: GPU capability/power and host offload sources.
- `workload/`: decode/prefill descriptors, dense model registry and MoE profile.
- `experiment/`: workload, architecture and scenario composition.

Start with `m3d_igzo_llama31_8b_decode_conditional_v0.yaml` for the formal
experiment or `capacity_aware_serving_v0.yaml` for serving. Case-level workload
is a memory activity operating point, not an LLM descriptor or a capability.
See `docs/research/frozen_model_parameters.md` for units and model boundaries.
