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
from om3dthermal.platform import AffineGPUDecodePowerSpec
from om3dthermal.provenance import ProvenanceRecord


def _spec(bandwidth_bytes_per_s: float = 1e12) -> AffineGPUDecodePowerSpec:
    return AffineGPUDecodePowerSpec(
        model="AFFINE_UTILIZATION_MODEL",
        static_power_W=100.0,
        peak_decode_power_W=300.0,
        peak_memory_bandwidth_bytes_per_s=bandwidth_bytes_per_s,
        static_power_status="PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE",
        peak_power_status="PARAMETRIC_NOMINAL_WITHIN_MEASURED_REFERENCE_RANGE",
        bandwidth_status="MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
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
                    else ("MEMORY" if token_time_s == 1e-9 else "COMPUTE")),
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


def test_memory_bottleneck_recovers_fixed_baseline_power() -> None:
    """At u = 1 the affine model reproduces the fixed 300 W baseline."""
    result = evaluate_gpu_decode_energy(
        _performance(token_time_s=1e-9), _energy(), _spec())
    assert result.evaluation_status == "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY"
    assert result.memory_bandwidth_utilization == pytest.approx(1.0)
    assert result.utilization_clamped is False
    assert result.gpu_decode_power_W == pytest.approx(300.0)
    assert result.gpu_energy_j_per_token == pytest.approx(300.0 * 1e-9)
    assert result.system_energy_j_per_token == pytest.approx(
        300.0 * 1e-9 + 1e-7)


def test_compute_bound_lowers_utilization_and_power() -> None:
    """u < 1 pulls GPU power toward the static end of the affine range."""
    result = evaluate_gpu_decode_energy(
        _performance(token_time_s=2e-9), _energy(), _spec())
    assert result.memory_bandwidth_utilization == pytest.approx(0.5)
    assert result.gpu_decode_power_W == pytest.approx(200.0)
    assert result.gpu_energy_j_per_token == pytest.approx(200.0 * 2e-9)


def test_utilization_is_clamped_at_one() -> None:
    result = evaluate_gpu_decode_energy(
        _performance(token_time_s=1e-9), _energy(),
        _spec(bandwidth_bytes_per_s=5e11))
    assert result.memory_bandwidth_utilization == 1.0
    assert result.utilization_clamped is True
    assert result.gpu_decode_power_W == pytest.approx(300.0)


def test_capacity_infeasible_blocks_all_numeric_outputs() -> None:
    result = evaluate_gpu_decode_energy(
        _performance(capacity_feasible=False),
        _energy(capacity_feasible=False), _spec())
    assert result.evaluation_status == "BLOCKED_BY_CAPACITY"
    assert result.gpu_decode_power_W is None
    assert result.gpu_energy_j_per_token is None
    assert result.system_energy_j_per_token is None
    assert result.memory_bandwidth_utilization is None


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
