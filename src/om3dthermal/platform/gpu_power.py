"""Affine GPU decode power model (E8 platform facts).

The model form is a first-order analytical choice: during the memory-bound
decode phase the GPU dynamic activity is parameterized by a single
memory-bandwidth utilization ``u``, so

```text
P_gpu = P_static + (P_peak_decode - P_static) * u
u     = gpu_side_bytes_per_token / (BW_peak * T_token)
```

At ``u = 1`` (the matched-bandwidth, memory-bottleneck scenario) the affine
model reproduces the existing fixed 300 W baseline exactly; the fixed
baseline is therefore the ``u = 1`` special case of this model.  The thermal
path is untouched: the fixed GPU power remains the thermal source, and this
model only produces energy-accounting outputs.

Parameter values are parametric nominals chosen inside the ranges reported
by the measured-reference anchors recorded in provenance; they are not a
per-workload calibration of a specific GPU.  See
``docs/research/gpu_power_model_spec_2026-09-05.md``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.provenance import ProvenanceRecord


class AffineGPUDecodePowerSpec(BaseModel):
    """Platform-level affine GPU decode power parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["AFFINE_UTILIZATION_MODEL"]
    static_power_W: float = Field(gt=0.0)
    peak_decode_power_W: float = Field(gt=0.0)
    peak_memory_bandwidth_bytes_per_s: float = Field(gt=0.0)
    static_power_status: Literal[
        "PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE"
    ]
    peak_power_status: Literal[
        "PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE"
    ]
    bandwidth_status: Literal[
        "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"
    ]
    model_form_status: Literal[
        "MODELING_CHOICE_AFFINE_FORM__LOCAL_MEASUREMENT_VALIDATION_PENDING"
    ]
    provenance: tuple[ProvenanceRecord, ...]

    @model_validator(mode="after")
    def _closure(self) -> "AffineGPUDecodePowerSpec":
        if self.peak_decode_power_W <= self.static_power_W:
            raise ValueError(
                "peak decode power must exceed static power")
        if not self.provenance:
            raise ValueError(
                "affine GPU decode power spec requires provenance records")
        return self
