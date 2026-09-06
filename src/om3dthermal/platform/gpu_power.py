"""Canonical bandwidth-bounded GPU decode power model (E8 platform facts).

The steady-state decode operating point is resolved from byte rate, not from
tokens or compute activity::

    B_actual = min(B_demand, B_gpu_peak)
    P_gpu = P_static + e_decode * 8 * B_actual

``e_decode`` is the GPU-side effective decode coefficient inferred from
measured decode dynamic power after subtracting memory energy accounted for
separately by E4. It is not a memory-I/O coefficient. The compatibility field
``peak_decode_power_W`` is a derived closure check, never an independent
power-model parameter.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from om3dthermal.provenance import ProvenanceRecord


class GPUDecodePowerOperatingPoint(BaseModel):
    """Resolved bandwidth and GPU power at one decode operating point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bandwidth_demand_bytes_per_s: float = Field(ge=0.0)
    bandwidth_actual_bytes_per_s: float = Field(ge=0.0)
    bandwidth_utilization: float = Field(ge=0.0, le=1.0)
    bandwidth_saturated: bool
    gpu_dynamic_power_W: float = Field(ge=0.0)
    gpu_power_W: float = Field(gt=0.0)


def _finite_real(name: str, value: float, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if positive and resolved <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    if not positive and resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def resolve_gpu_decode_power(
    *,
    static_power_W: float,
    e_decode_J_per_bit: float,
    bandwidth_demand_bytes_per_s: float,
    peak_bandwidth_bytes_per_s: float,
) -> GPUDecodePowerOperatingPoint:
    """Resolve the single canonical bandwidth-bounded decode power path.

    Saturation has a strict exceedance meaning: it is ``True`` only when
    ``bandwidth_demand_bytes_per_s > peak_bandwidth_bytes_per_s``. At the
    exact boundary utilization is one but ``bandwidth_saturated`` is false.
    """

    static_power = _finite_real(
        "static_power_W", static_power_W, positive=True)
    decode_energy = _finite_real(
        "e_decode_J_per_bit", e_decode_J_per_bit, positive=True)
    demand = _finite_real(
        "bandwidth_demand_bytes_per_s",
        bandwidth_demand_bytes_per_s,
        positive=False,
    )
    peak = _finite_real(
        "peak_bandwidth_bytes_per_s",
        peak_bandwidth_bytes_per_s,
        positive=True,
    )

    actual = min(demand, peak)
    dynamic_power = decode_energy * 8.0 * actual
    return GPUDecodePowerOperatingPoint(
        bandwidth_demand_bytes_per_s=demand,
        bandwidth_actual_bytes_per_s=actual,
        bandwidth_utilization=actual / peak,
        bandwidth_saturated=demand > peak,
        gpu_dynamic_power_W=dynamic_power,
        gpu_power_W=static_power + dynamic_power,
    )


class AffineGPUDecodePowerSpec(BaseModel):
    """Platform inputs for the bandwidth-bounded GPU decode power model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["AFFINE_UTILIZATION_MODEL"]
    static_power_W: float = Field(gt=0.0)
    e_decode_J_per_bit: float = Field(gt=0.0)
    peak_memory_bandwidth_bytes_per_s: float = Field(gt=0.0)
    # Compatibility/reporting field: validator enforces its derived value.
    peak_decode_power_W: float = Field(gt=0.0)
    static_power_status: Literal[
        "PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE",
        "MEASURED_REFERENCE_H200_SXM_IDLE_FLOOR",
    ]
    peak_power_status: Literal[
        "DERIVED_FROM_STATIC_E_DECODE_AND_PEAK_BANDWIDTH"
    ]
    bandwidth_status: Literal[
        "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
        "VENDOR_SPEC_H200_PEAK_HBM3E_BANDWIDTH",
    ]
    model_form_status: Literal[
        "MODELING_CHOICE_AFFINE_FORM__LOCAL_MEASUREMENT_VALIDATION_PENDING"
    ]
    provenance: tuple[ProvenanceRecord, ...]

    @property
    def derived_peak_decode_power_W(self) -> float:
        return (
            self.static_power_W
            + self.e_decode_J_per_bit
            * 8.0
            * self.peak_memory_bandwidth_bytes_per_s
        )

    @model_validator(mode="after")
    def _closure(self) -> "AffineGPUDecodePowerSpec":
        if not math.isclose(
            self.peak_decode_power_W,
            self.derived_peak_decode_power_W,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "peak_decode_power_W must equal the derived value "
                "static_power_W + e_decode_J_per_bit * 8 * "
                "peak_memory_bandwidth_bytes_per_s"
            )
        if not self.provenance:
            raise ValueError(
                "GPU decode power spec requires provenance records")
        return self
