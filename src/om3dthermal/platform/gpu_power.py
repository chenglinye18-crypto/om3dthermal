"""Canonical regime-bounded GPU decode power models (E8 platform facts).

The steady-state decode operating point is resolved from the selected
bottleneck's actual service rate, not directly from tokens::

    B_actual = min(B_demand, B_gpu_peak)
    P_gpu_memory = P_static + e_decode * 8 * B_actual

    F_actual = min(F_demand, F_effective)
    P_gpu_compute = P_static + e_compute_dynamic * F_actual

``e_decode`` is the project-level GPU decode dynamic energy-per-bit
coefficient used by the bandwidth-bounded GPU power model. Memory energy is
modeled independently elsewhere, without subtraction from this coefficient.
Peak decode power is a read-only derived value, never an input.

Both regimes share the 74 W static anchor. Their dynamic terms are alternatives,
not additive components.
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


class GPUComputePowerOperatingPoint(BaseModel):
    """Resolved compute rate and GPU power at one compute operating point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compute_demand_flops_per_s: float = Field(ge=0.0)
    compute_actual_flops_per_s: float = Field(ge=0.0)
    compute_utilization: float = Field(ge=0.0, le=1.0)
    compute_saturated: bool
    gpu_dynamic_compute_power_W: float = Field(ge=0.0)
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


def resolve_gpu_compute_power(
    *,
    static_power_W: float,
    compute_energy_dynamic_J_per_FLOP: float,
    compute_demand_FLOP_per_s: float,
    effective_compute_ceiling_FLOP_per_s: float,
) -> GPUComputePowerOperatingPoint:
    """Resolve the canonical compute-bounded GPU operating point.

    The supplied ceiling is the scenario/effective compute ceiling, which may
    be below the vendor peak. Saturation uses the same strict exceedance
    semantics as the bandwidth resolver; the exact boundary is not marked as
    saturated. Static power is added exactly once.
    """

    static_power = _finite_real(
        "static_power_W", static_power_W, positive=True)
    compute_energy = _finite_real(
        "compute_energy_dynamic_J_per_FLOP",
        compute_energy_dynamic_J_per_FLOP,
        positive=True,
    )
    demand = _finite_real(
        "compute_demand_FLOP_per_s",
        compute_demand_FLOP_per_s,
        positive=False,
    )
    ceiling = _finite_real(
        "effective_compute_ceiling_FLOP_per_s",
        effective_compute_ceiling_FLOP_per_s,
        positive=True,
    )

    actual = min(demand, ceiling)
    dynamic_power = compute_energy * actual
    return GPUComputePowerOperatingPoint(
        compute_demand_flops_per_s=demand,
        compute_actual_flops_per_s=actual,
        compute_utilization=actual / ceiling,
        compute_saturated=demand > ceiling,
        gpu_dynamic_compute_power_W=dynamic_power,
        gpu_power_W=static_power + dynamic_power,
    )


class AffineGPUDecodePowerSpec(BaseModel):
    """Platform inputs for the bandwidth-bounded GPU decode power model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["AFFINE_UTILIZATION_MODEL"]
    static_power_W: float = Field(gt=0.0)
    e_decode_J_per_bit: float = Field(gt=0.0)
    peak_memory_bandwidth_bytes_per_s: float = Field(gt=0.0)
    static_power_status: Literal[
        "PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE",
        "MEASURED_REFERENCE_H200_SXM_IDLE_FLOOR",
    ]
    bandwidth_status: Literal[
        "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
        "VENDOR_SPEC_H200_PEAK_HBM3E_BANDWIDTH",
    ]
    coefficient_range_status: Literal["REFERENCE_DERIVED_RANGE"]
    coefficient_nominal_status: Literal["MODELING_CHOICE_RANGE_MIDPOINT"]
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
        if not self.provenance:
            raise ValueError(
                "GPU decode power spec requires provenance records")
        return self


class AffineGPUComputePowerSpec(BaseModel):
    """H200 compute-power range with no implicit nominal selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["AFFINE_COMPUTE_RATE_MODEL"]
    static_power_W: float = Field(gt=0.0)
    peak_compute_BF16_dense_flops_per_s: float = Field(gt=0.0)
    compute_bound_power_W_min: float = Field(gt=0.0)
    compute_bound_power_W_max: float = Field(gt=0.0)
    e_compute_dynamic_J_per_FLOP_min: float = Field(gt=0.0)
    e_compute_dynamic_J_per_FLOP_max: float = Field(gt=0.0)
    static_power_status: Literal[
        "MEASURED_REFERENCE_H200_SXM_IDLE_FLOOR"
    ]
    peak_compute_status: Literal["VENDOR_SPEC_H200_BF16_DENSE"]
    compute_power_range_status: Literal[
        "DERIVED_FROM_MEASURED_REFERENCE"
    ]
    coefficient_status: Literal[
        "DYNAMIC_ONLY_DERIVED_AFTER_SUBTRACTING_STATIC_POWER"
    ]
    provenance: tuple[ProvenanceRecord, ...]

    @model_validator(mode="after")
    def _closure(self) -> "AffineGPUComputePowerSpec":
        if self.compute_bound_power_W_min <= self.static_power_W:
            raise ValueError(
                "compute-bound power minimum must exceed static power")
        if self.compute_bound_power_W_max < self.compute_bound_power_W_min:
            raise ValueError(
                "compute-bound power maximum must not be below minimum")
        derived_min = (
            self.compute_bound_power_W_min - self.static_power_W
        ) / self.peak_compute_BF16_dense_flops_per_s
        derived_max = (
            self.compute_bound_power_W_max - self.static_power_W
        ) / self.peak_compute_BF16_dense_flops_per_s
        for name, configured, derived in (
            ("e_compute_dynamic_J_per_FLOP_min",
             self.e_compute_dynamic_J_per_FLOP_min, derived_min),
            ("e_compute_dynamic_J_per_FLOP_max",
             self.e_compute_dynamic_J_per_FLOP_max, derived_max),
        ):
            if not math.isclose(
                configured, derived, rel_tol=1e-12, abs_tol=1e-18
            ):
                raise ValueError(
                    f"{name} must be derived from compute-bound power "
                    "after subtracting static_power_W"
                )
        if not self.provenance:
            raise ValueError(
                "GPU compute power spec requires provenance records")
        return self
