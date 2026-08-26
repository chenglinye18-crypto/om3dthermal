from pathlib import Path

import pytest

from om3dthermal.architecture_capacity import resolve_architecture_capacity
from om3dthermal.evaluation import evaluate_architecture_capacity_feasibility
from om3dthermal.evaluator import (
    LLMDecodeWorkloadThermalMetrics,
    evaluate_llm_decode_performance,
)
from om3dthermal.experiment import run_m3d_parameter_sensitivity
from om3dthermal.power import (
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.workload import LLMDecodeInput, evaluate_llm_decode


ROOT = Path(__file__).parents[1]


def _fake_thermal(mapping):
    temperature = 70.0 + 0.2 * mapping.expected_package_total_power_W
    return LLMDecodeWorkloadThermalMetrics(
        architecture=mapping.architecture, rho=mapping.rho,
        mapped_package_power_W=mapping.expected_package_total_power_W,
        expected_package_power_W=mapping.expected_package_total_power_W,
        source_power_breakdown_W={s.name: s.power_W for s in mapping.sources},
        power_closure_absolute_error_W=0.0,
        power_closure_relative_error=0.0,
        memory_Tmax_degC=temperature, gpu_Tmax_degC=temperature,
        package_Tmax_degC=temperature, converged=True, iterations=1,
        final_relative_residual=1e-6, max_temperature_update_K=1e-4,
        relative_power_imbalance=1e-6, cell_count=1,
        internal_edge_count=0, full_vector_d2h_during_iteration=0,
        thermal_backend="gpu_pcg", precision_status="FP64",
        preconditioner_status="JACOBI_DIAGONAL",
        initial_temperature_K=293.15, relative_residual_tolerance=0.001,
        max_temperature_update_tolerance_K=0.01, max_iterations=100000,
        check_interval=10, warm_start_status="FRESH_SOLVE_NO_WARM_START",
        write_spatial_distribution_status=(
            "WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY"),
        memory_total_completeness_status=(
            mapping.memory_total_completeness_status),
        scenario_status="CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY",
    )


def test_m3d_interface_and_logic_sensitivities_are_separate_and_close():
    case = load_case_config(ROOT / "configs/cases/orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(case, project_root=ROOT, geometry=geometry)
    workload = evaluate_llm_decode(LLMDecodeInput(
        n_param=8_000_000_000, n_layers=32, n_heads_q=32, n_heads_kv=8,
        d_model=4096, d_ff=14336, vocab_size=128_256,
        batch_size=1, context_length=131_072, weight_bits=16, kv_bits=16,
        runtime_bytes=0))
    capacity = evaluate_architecture_capacity_feasibility(
        workload, resolve_architecture_capacity(case, geometry, system),
        reserved_capacity_bytes=0)
    performance = evaluate_llm_decode_performance(
        workload, capacity, batch_size=1,
        matched_payload_bandwidth_bits_per_second=39.2e12,
        effective_compute_flops_per_second=100e12)
    result = run_m3d_parameter_sensitivity(
        case=case, system=system, workload=workload, capacity=capacity,
        performance=performance,
        interface_energy_values_pj_per_bit=(0.25, 0.5, 1.0),
        logic_background_values_W=(0, 5, 10, 20),
        thermal_runner=_fake_thermal)

    assert result.status == "PARAMETRIC_SENSITIVITY"
    assert [row.interface_power_at_matched_bandwidth_W
            for row in result.interface_rows] == pytest.approx((9.8, 19.6, 39.2))
    assert [row.read_total_energy_pj_per_bit
            for row in result.interface_rows] == pytest.approx(
                (0.6052605756733209, 0.8552605756733209,
                 1.355260575673321))
    logic = result.logic_background_rows
    assert [row.memory_total_power_W - logic[0].memory_total_power_W
            for row in logic] == pytest.approx((0, 5, 10, 20))
    assert all(row.status == "PARAMETRIC_SENSITIVITY" for row in logic)
