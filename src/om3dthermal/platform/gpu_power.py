"""Canonical regime-bounded GPU decode power models (E8 platform facts).

The steady-state decode operating point is resolved from the GPU-side
sustained service rate, not directly from tokens. The formal runner resolves
that rate as a modeled fraction of the physical transfer ceiling::

    B_ceiling = min(B_demand, B_memory, B_gpu_peak)
    B_sustained = eta_gpu_bandwidth * B_ceiling
    P_gpu_memory = P_static + e_decode * 8 * B_sustained

    F_actual = min(F_demand, F_effective)
    P_gpu_compute = P_static + e_compute_dynamic * F_actual

``e_decode`` is the project-level GPU decode dynamic energy-per-bit
coefficient used by the bandwidth-bounded GPU power model and is normalized
per sustained/actual bit. Memory energy is modeled independently elsewhere,
without subtraction from this coefficient. Peak-rate extrapolation remains a
read-only mathematical property, never an input or the formal nominal point.

Both regimes share the 74 W static anchor. Their dynamic terms are alternatives,
not additive components.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

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
    coefficient_nominal_status: Literal[
        "MODELING_CHOICE_RANGE_MIDPOINT",
        "USER_SPECIFIED_MODELING_CHOICE",
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
        "VENDOR_PEAK_DERIVED_DYNAMIC_ENERGY_REFERENCE"
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


class EffectiveThroughputReferenceRangeTFLOPS(BaseModel):
    """Reference envelope retained for provenance, not an active sweep."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: float = Field(gt=0.0)
    max: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> "EffectiveThroughputReferenceRangeTFLOPS":
        if self.max < self.min:
            raise ValueError("effective throughput range max must not be below min")
        return self


class ReferenceCalibratedGPUPrefillComputeSpec(BaseModel):
    """Two-family H200 Prefill effective-throughput calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal[
        "REFERENCE_CALIBRATED_TWO_FAMILY_EFFECTIVE_THROUGHPUT"
    ]
    peak_compute_reference_status: Literal[
        "VENDOR_REPORTED_BF16_DENSE_PEAK"
    ]
    large_gemm_effective_tflops: float = Field(gt=0.0)
    causal_attention_effective_tflops: float = Field(gt=0.0)
    large_gemm_reference_range_tflops: EffectiveThroughputReferenceRangeTFLOPS
    causal_attention_reference_range_tflops: EffectiveThroughputReferenceRangeTFLOPS
    large_gemm_effective_status: Literal[
        "REFERENCE_CALIBRATED_EFFECTIVE_THROUGHPUT"
    ]
    causal_attention_effective_status: Literal[
        "REFERENCE_CALIBRATED_EFFECTIVE_THROUGHPUT"
    ]
    provenance: tuple[ProvenanceRecord, ...]

    @model_validator(mode="after")
    def _calibration_closure(self) -> "ReferenceCalibratedGPUPrefillComputeSpec":
        for name, nominal, reference in (
            ("large_gemm", self.large_gemm_effective_tflops,
             self.large_gemm_reference_range_tflops),
            ("causal_attention", self.causal_attention_effective_tflops,
             self.causal_attention_reference_range_tflops),
        ):
            if not reference.min <= nominal <= reference.max:
                raise ValueError(
                    f"{name} effective throughput must lie within its reference range")
        if not self.provenance:
            raise ValueError("GPU Prefill compute calibration requires provenance")
        forbidden_statuses = {
            "MEASURED_H200_PREFILL", "VENDOR_REPORTED",
            "PAPER_REPORTED_H200_PREFILL",
        }
        if any(record.status in forbidden_statuses for record in self.provenance):
            raise ValueError("GPU Prefill calibration provenance overclaims its source")
        return self


class GPUPrefillComputeEnergyCalibration(BaseModel):
    """Platform-derived peak-reference and nominal Prefill coefficients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vendor_peak_tflops: float = Field(gt=0.0)
    large_gemm_effective_tflops: float = Field(gt=0.0)
    causal_attention_effective_tflops: float = Field(gt=0.0)
    static_power_W: float = Field(gt=0.0)
    compute_bound_total_power_W_min: float = Field(gt=0.0)
    compute_bound_total_power_W_max: float = Field(gt=0.0)
    compute_bound_dynamic_power_W_min: float = Field(gt=0.0)
    compute_bound_dynamic_power_W_max: float = Field(gt=0.0)
    peak_reference_dynamic_J_per_FLOP_min: float = Field(gt=0.0)
    peak_reference_dynamic_J_per_FLOP_max: float = Field(gt=0.0)
    nominal_gemm_dynamic_J_per_FLOP_min: float = Field(gt=0.0)
    nominal_gemm_dynamic_J_per_FLOP_max: float = Field(gt=0.0)
    nominal_attention_dynamic_J_per_FLOP_min: float = Field(gt=0.0)
    nominal_attention_dynamic_J_per_FLOP_max: float = Field(gt=0.0)
    peak_reference_coefficient_status: Literal[
        "VENDOR_PEAK_DERIVED_DYNAMIC_ENERGY_REFERENCE"
    ] = "VENDOR_PEAK_DERIVED_DYNAMIC_ENERGY_REFERENCE"
    nominal_coefficient_status: Literal[
        "REFERENCE_CALIBRATED_EFFECTIVE_THROUGHPUT_DERIVED_DYNAMIC_ENERGY"
    ] = "REFERENCE_CALIBRATED_EFFECTIVE_THROUGHPUT_DERIVED_DYNAMIC_ENERGY"
    derivation: Literal[
        "DYNAMIC_POWER_EQUALS_TOTAL_MINUS_STATIC__ENERGY_PER_FLOP_EQUALS_DYNAMIC_POWER_DIVIDED_BY_EFFECTIVE_THROUGHPUT"
    ] = (
        "DYNAMIC_POWER_EQUALS_TOTAL_MINUS_STATIC__ENERGY_PER_FLOP_EQUALS_"
        "DYNAMIC_POWER_DIVIDED_BY_EFFECTIVE_THROUGHPUT")

    @computed_field(return_type=float)
    @property
    def peak_reference_dynamic_pJ_per_FLOP_min(self) -> float:
        return self.peak_reference_dynamic_J_per_FLOP_min * 1e12

    @computed_field(return_type=float)
    @property
    def peak_reference_dynamic_pJ_per_FLOP_max(self) -> float:
        return self.peak_reference_dynamic_J_per_FLOP_max * 1e12

    @computed_field(return_type=float)
    @property
    def nominal_gemm_dynamic_pJ_per_FLOP_min(self) -> float:
        return self.nominal_gemm_dynamic_J_per_FLOP_min * 1e12

    @computed_field(return_type=float)
    @property
    def nominal_gemm_dynamic_pJ_per_FLOP_max(self) -> float:
        return self.nominal_gemm_dynamic_J_per_FLOP_max * 1e12

    @computed_field(return_type=float)
    @property
    def nominal_attention_dynamic_pJ_per_FLOP_min(self) -> float:
        return self.nominal_attention_dynamic_J_per_FLOP_min * 1e12

    @computed_field(return_type=float)
    @property
    def nominal_attention_dynamic_pJ_per_FLOP_max(self) -> float:
        return self.nominal_attention_dynamic_J_per_FLOP_max * 1e12


def resolve_gpu_prefill_compute_energy_calibration(
    compute: AffineGPUComputePowerSpec,
    prefill: ReferenceCalibratedGPUPrefillComputeSpec,
) -> GPUPrefillComputeEnergyCalibration:
    """Derive family-specific nominal pJ/FLOP from canonical power anchors."""
    dynamic_min = compute.compute_bound_power_W_min - compute.static_power_W
    dynamic_max = compute.compute_bound_power_W_max - compute.static_power_W
    gemm_flops = prefill.large_gemm_effective_tflops * 1e12
    attention_flops = prefill.causal_attention_effective_tflops * 1e12
    return GPUPrefillComputeEnergyCalibration(
        vendor_peak_tflops=(
            compute.peak_compute_BF16_dense_flops_per_s / 1e12),
        large_gemm_effective_tflops=prefill.large_gemm_effective_tflops,
        causal_attention_effective_tflops=(
            prefill.causal_attention_effective_tflops),
        static_power_W=compute.static_power_W,
        compute_bound_total_power_W_min=compute.compute_bound_power_W_min,
        compute_bound_total_power_W_max=compute.compute_bound_power_W_max,
        compute_bound_dynamic_power_W_min=dynamic_min,
        compute_bound_dynamic_power_W_max=dynamic_max,
        peak_reference_dynamic_J_per_FLOP_min=(
            compute.e_compute_dynamic_J_per_FLOP_min),
        peak_reference_dynamic_J_per_FLOP_max=(
            compute.e_compute_dynamic_J_per_FLOP_max),
        nominal_gemm_dynamic_J_per_FLOP_min=dynamic_min / gemm_flops,
        nominal_gemm_dynamic_J_per_FLOP_max=dynamic_max / gemm_flops,
        nominal_attention_dynamic_J_per_FLOP_min=(
            dynamic_min / attention_flops),
        nominal_attention_dynamic_J_per_FLOP_max=(
            dynamic_max / attention_flops))
