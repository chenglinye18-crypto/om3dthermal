"""Tests for the E8 affine GPU decode energy evaluator."""

import pytest

from om3dthermal.evaluator.llm_decode_architecture_energy import (
    ArchitectureDecodeMemoryEnergyMetrics,
)
from om3dthermal.evaluator.llm_decode_gpu_energy import (
    evaluate_gpu_decode_energy,
)
from om3dthermal.evaluator.llm_decode_performance import (
    LLMDecodePerformanceMetrics,
)
from om3dthermal.platform import (
    AffineGPUDecodePowerSpec,
    resolve_gpu_decode_power,
)
from om3dthermal.provenance import ProvenanceRecord


STATIC_POWER_W = 74.0
E_DECODE_J_PER_BIT = 7.645e-12
PEAK_BANDWIDTH_BYTES_PER_S = 4.8e12
PEAK_DECODE_POWER_W = 367.568


def _spec() -> AffineGPUDecodePowerSpec:
    return AffineGPUDecodePowerSpec(
        model="AFFINE_UTILIZATION_MODEL",
        static_power_W=STATIC_POWER_W,
        e_decode_J_per_bit=E_DECODE_J_PER_BIT,
        peak_memory_bandwidth_bytes_per_s=PEAK_BANDWIDTH_BYTES_PER_S,
        static_power_status="PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE",
        bandwidth_status="MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
        coefficient_range_status="REFERENCE_DERIVED_RANGE",
        coefficient_nominal_status="MODELING_CHOICE_RANGE_MIDPOINT",
        model_form_status=(
            "MODELING_CHOICE_AFFINE_FORM__LOCAL_MEASUREMENT_VALIDATION_PENDING"),
        provenance=(ProvenanceRecord(
            record_id="test_anchor",
            classification="MODELING_CHOICE",
            source="test",
            status="TEST_FIXTURE"),),
    )


def _performance(
    *,
    token_time_s: float | None = 1e-9,
    capacity_feasible: bool = True,
    bottleneck: str = "MEMORY",
) -> LLMDecodePerformanceMetrics:
    blocked = not capacity_feasible
    return LLMDecodePerformanceMetrics(
        architecture="orthogonal_m3d_igzo",
        batch_size=1,
        capacity_feasible=capacity_feasible,
        read_bytes_per_token=800.0,
        write_bytes_per_token=200.0,
        traffic_bits_per_token=8000.0,
        flops_per_token=10,
        matched_payload_bandwidth_bits_per_second=8e12,
        effective_compute_flops_per_second=1e14,
        memory_time_per_token_equivalent_s=None if blocked else 1e-9,
        compute_time_per_token_equivalent_s=None if blocked else 5e-10,
        token_equivalent_time_s=None if blocked else token_time_s,
        aggregate_step_time_s=None if blocked else token_time_s,
        aggregate_tokens_per_second=(
            None if blocked else 1.0 / token_time_s),
        per_sequence_tokens_per_second=(
            None if blocked else 1.0 / token_time_s),
        per_sequence_step_latency_s=None if blocked else token_time_s,
        compute_throughput_required_to_match_memory_flops_per_second=(
            None if blocked else 1e13),
        bottleneck=("NOT_EVALUATED_CAPACITY_INFEASIBLE" if blocked
                    else bottleneck),
        performance_status=("BLOCKED_BY_CAPACITY" if blocked
                            else "EVALUATED_MATCHED_REFERENCE_SCENARIO"),
        bandwidth_status="MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
        compute_throughput_status="NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED",
        memory_bandwidth_model="SHARED_READ_WRITE_PAYLOAD_BANDWIDTH",
        overlap_model="ROOFLINE_MAX",
    )


def _energy(*, capacity_feasible: bool = True,
            ) -> ArchitectureDecodeMemoryEnergyMetrics:
    return ArchitectureDecodeMemoryEnergyMetrics(
        architecture="orthogonal_m3d_igzo",
        rho=1.0,
        capacity_feasible=capacity_feasible,
        read_bytes_per_token=800.0,
        write_bytes_per_token=200.0,
        read_energy_pj_per_bit=0.855260575673,
        write_energy_pj_per_bit=0.855260575673,
        read_dynamic_energy_j_per_token=(
            1e-7 if capacity_feasible else None),
        write_dynamic_energy_j_per_token=0.0 if capacity_feasible else None,
        memory_dynamic_energy_j_per_token=(
            1e-7 if capacity_feasible else None),
        read_energy_status="CURRENT_NOMINAL_ANALYTICAL_MODEL",
        write_energy_status="RHO_SENSITIVITY_NOT_PHYSICAL_CLAIM",
        energy_scope_status="MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY",
        scenario_status="CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY",
        zhu_transferability_status="NOT_VALIDATED",
        interface_energy_pj_per_bit=0.5,
        interface_energy_status="PAPER_REPORTED_INDUCTIVE_LINK_ENERGY",
        evaluation_status=(
            "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY"
            if capacity_feasible else "CAPACITY_INFEASIBLE"),
    )


def _resolve(demand: float):
    return resolve_gpu_decode_power(
        static_power_W=STATIC_POWER_W,
        e_decode_J_per_bit=E_DECODE_J_PER_BIT,
        bandwidth_demand_bytes_per_s=demand,
        peak_bandwidth_bytes_per_s=PEAK_BANDWIDTH_BYTES_PER_S,
    )


def test_below_bandwidth_boundary() -> None:
    result = _resolve(0.5 * PEAK_BANDWIDTH_BYTES_PER_S)
    assert result.bandwidth_demand_bytes_per_s == 2.4e12
    assert result.bandwidth_actual_bytes_per_s == 2.4e12
    assert result.bandwidth_utilization == pytest.approx(0.5)
    assert result.bandwidth_saturated is False
    assert result.gpu_dynamic_power_W == pytest.approx(146.784)
    assert result.gpu_power_W == pytest.approx(220.784)


def test_exact_bandwidth_boundary_is_not_strictly_saturated() -> None:
    result = _resolve(PEAK_BANDWIDTH_BYTES_PER_S)
    assert result.bandwidth_actual_bytes_per_s == PEAK_BANDWIDTH_BYTES_PER_S
    assert result.bandwidth_utilization == pytest.approx(1.0)
    assert result.bandwidth_saturated is False
    assert result.gpu_power_W == pytest.approx(PEAK_DECODE_POWER_W)


def test_above_bandwidth_boundary_is_capped() -> None:
    result = _resolve(1.2 * PEAK_BANDWIDTH_BYTES_PER_S)
    assert result.bandwidth_demand_bytes_per_s == pytest.approx(5.76e12)
    assert result.bandwidth_actual_bytes_per_s == PEAK_BANDWIDTH_BYTES_PER_S
    assert result.bandwidth_utilization == pytest.approx(1.0)
    assert result.bandwidth_saturated is True
    assert result.gpu_power_W == pytest.approx(PEAK_DECODE_POWER_W)


def test_power_is_continuous_at_bandwidth_boundary() -> None:
    left = _resolve(PEAK_BANDWIDTH_BYTES_PER_S * (1.0 - 1e-9))
    boundary = _resolve(PEAK_BANDWIDTH_BYTES_PER_S)
    assert left.gpu_power_W == pytest.approx(
        boundary.gpu_power_W, rel=1e-9)


def test_power_is_monotonic_then_constant_across_bandwidth_boundary() -> None:
    demands = [
        0.0,
        0.25 * PEAK_BANDWIDTH_BYTES_PER_S,
        0.5 * PEAK_BANDWIDTH_BYTES_PER_S,
        PEAK_BANDWIDTH_BYTES_PER_S,
        1.2 * PEAK_BANDWIDTH_BYTES_PER_S,
        2.0 * PEAK_BANDWIDTH_BYTES_PER_S,
    ]
    powers = [_resolve(demand).gpu_power_W for demand in demands]
    assert powers[0] < powers[1] < powers[2] < powers[3]
    assert powers[3:] == pytest.approx([PEAK_DECODE_POWER_W] * 3)


def test_peak_decode_power_is_read_only_derived_value() -> None:
    assert _spec().derived_peak_decode_power_W == pytest.approx(367.568)


def test_decode_coefficient_is_the_runtime_nominal() -> None:
    spec = _spec()
    assert spec.e_decode_J_per_bit == 7.645e-12
    assert AffineGPUDecodePowerSpec.model_validate(
        spec.model_dump() | {"e_decode_J_per_bit": 7.0e-12}
    ).e_decode_J_per_bit == 7.0e-12


def test_decode_reference_range_peak_power_hand_checks() -> None:
    powers = [
        resolve_gpu_decode_power(
            static_power_W=STATIC_POWER_W,
            e_decode_J_per_bit=coefficient,
            bandwidth_demand_bytes_per_s=PEAK_BANDWIDTH_BYTES_PER_S,
            peak_bandwidth_bytes_per_s=PEAK_BANDWIDTH_BYTES_PER_S,
        ).gpu_power_W
        for coefficient in (6.28e-12, 7.645e-12, 9.01e-12)
    ]
    assert powers == pytest.approx([315.152, 367.568, 419.984])


def test_energy_evaluator_uses_canonical_boundary_operating_point() -> None:
    token_time = 1000.0 / PEAK_BANDWIDTH_BYTES_PER_S
    result = evaluate_gpu_decode_energy(
        _performance(token_time_s=token_time), _energy(), _spec())
    assert result.evaluation_status == "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY"
    assert result.memory_bandwidth_utilization == pytest.approx(1.0)
    assert result.utilization_clamped is False
    assert result.bandwidth_demand_bytes_per_s == pytest.approx(4.8e12)
    assert result.bandwidth_actual_bytes_per_s == pytest.approx(4.8e12)
    assert result.bandwidth_saturated is False
    assert result.gpu_dynamic_power_W == pytest.approx(293.568)
    assert result.gpu_decode_power_W == pytest.approx(367.568)
    assert result.gpu_energy_j_per_token == pytest.approx(367.568 * token_time)
    assert result.system_energy_j_per_token == pytest.approx(
        367.568 * token_time + 1e-7)


def test_longer_token_time_lowers_byte_rate_and_power() -> None:
    """A longer token time lowers byte rate and bandwidth-dependent power."""
    result = evaluate_gpu_decode_energy(
        _performance(token_time_s=2.0 * 1000.0 / PEAK_BANDWIDTH_BYTES_PER_S),
        _energy(), _spec())
    assert result.memory_bandwidth_utilization == pytest.approx(0.5)
    assert result.gpu_decode_power_W == pytest.approx(220.784)


def test_utilization_is_clamped_at_one() -> None:
    result = evaluate_gpu_decode_energy(
        _performance(token_time_s=1000.0 / (1.2 * PEAK_BANDWIDTH_BYTES_PER_S)),
        _energy(), _spec())
    assert result.memory_bandwidth_utilization == 1.0
    assert result.utilization_clamped is True
    assert result.bandwidth_saturated is True
    assert result.gpu_decode_power_W == pytest.approx(367.568)


def test_capacity_infeasible_blocks_all_numeric_outputs() -> None:
    result = evaluate_gpu_decode_energy(
        _performance(capacity_feasible=False),
        _energy(capacity_feasible=False), _spec())
    assert result.evaluation_status == "BLOCKED_BY_CAPACITY"
    assert result.gpu_decode_power_W is None
    assert result.gpu_energy_j_per_token is None
    assert result.system_energy_j_per_token is None
    assert result.memory_bandwidth_utilization is None
    assert result.bandwidth_demand_bytes_per_s is None
    assert result.bandwidth_actual_bytes_per_s is None
    assert result.bandwidth_saturated is None
    assert result.gpu_dynamic_power_W is None


def test_architecture_mismatch_is_rejected() -> None:
    performance = _performance()
    energy = _energy().model_copy(
        update={"architecture": "conventional_hbm_2x1"})
    with pytest.raises(ValueError, match="architecture identity mismatch"):
        evaluate_gpu_decode_energy(performance, energy, _spec())


def test_capacity_feasibility_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacity feasibility mismatch"):
        evaluate_gpu_decode_energy(
            _performance(capacity_feasible=True),
            _energy(capacity_feasible=False), _spec())
