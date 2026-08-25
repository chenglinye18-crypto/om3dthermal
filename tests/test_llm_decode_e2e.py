"""E7 pure conditional E2E aggregation semantic tests (no thermal solves)."""

from __future__ import annotations

from pathlib import Path

import pytest

from om3dthermal.architecture_capacity import resolve_architecture_capacity
from om3dthermal.evaluator import (
    LLMDecodeWorkloadThermalMetrics,
    assemble_conditional_llm_decode_e2e_row,
    evaluate_architecture_decode_memory_energy,
    evaluate_llm_decode_performance,
    evaluate_llm_decode_workload_power,
    validate_conditional_llm_decode_e2e_table,
)
from om3dthermal.power import (
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.workload import (
    LLMDecodeInput,
    evaluate_architecture_capacity_feasibility,
    evaluate_llm_decode,
)


ROOT = Path(__file__).parents[1]
ARCHITECTURES = (
    "conventional_hbm_2x1", "orthogonal_si", "orthogonal_m3d_igzo")
RHOS = (0, 1, 100, 1000)
THERMAL = {
    "conventional_hbm_2x1": {
        0: (81.50569936158911, 81.93344900757944, 81.93344900757944, 930),
        1: (81.50573559888926, 81.93348523711035, 81.93348523711035, 930),
        100: (81.50932309158264, 81.93707196061627, 81.93707196061627, 930),
        1000: (81.54193666295293, 81.96967853961064, 81.96967853961064, 930),
    },
    "orthogonal_si": {
        0: (84.17709612183228, 84.62109873362812, 84.62109873362812, 820),
        1: (84.17711964517133, 84.6211221964034, 84.6211221964034, 820),
        100: (84.17944845637732, 84.62344501181036, 84.62344501181036, 820),
        1000: (84.20061952680021, 84.64456156336826, 84.64456156336826, 820),
    },
    "orthogonal_m3d_igzo": {
        0: (81.83992036973837, 82.290987332594, 82.290987332594, 1090),
        1: (81.83993509375131, 82.29100201678528, 82.29100201678528, 1090),
        100: (81.84139277093556, 82.29245575154397, 82.29245575154397, 1090),
        1000: (81.85464437979937, 82.30567150593589, 82.30567150593589, 1090),
    },
}
SIZES = {
    "conventional_hbm_2x1": (859596, 2531340),
    "orthogonal_si": (1518468, 4466056),
    "orthogonal_m3d_igzo": (1953392, 5748778),
}


def _thermal(name, rho, power):
    memory_t, gpu_t, package_t, iterations = THERMAL[name][rho]
    cells, edges = SIZES[name]
    completeness = power.memory_total_completeness_status
    return LLMDecodeWorkloadThermalMetrics(
        architecture=name, rho=rho,
        mapped_package_power_W=power.package_workload_total_W,
        expected_package_power_W=power.package_workload_total_W,
        source_power_breakdown_W={"committed_E6_evidence":
                                  power.package_workload_total_W},
        power_closure_absolute_error_W=0.0,
        power_closure_relative_error=0.0,
        memory_Tmax_degC=memory_t, gpu_Tmax_degC=gpu_t,
        package_Tmax_degC=package_t, converged=True,
        iterations=iterations, final_relative_residual=1e-4,
        max_temperature_update_K=0.009,
        relative_power_imbalance=1e-4, cell_count=cells,
        internal_edge_count=edges, full_vector_d2h_during_iteration=0,
        thermal_backend="gpu_pcg", precision_status="FP64",
        preconditioner_status="JACOBI_DIAGONAL",
        initial_temperature_K=293.15,
        relative_residual_tolerance=0.001,
        max_temperature_update_tolerance_K=0.01,
        max_iterations=100000, check_interval=10,
        warm_start_status="FRESH_SOLVE_NO_WARM_START",
        write_spatial_distribution_status=(
            "WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY"),
        memory_total_completeness_status=completeness,
        scenario_status="CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY")


@pytest.fixture(scope="module")
def frozen():
    inp = LLMDecodeInput(
        n_param=8_000_000_000, n_layers=32, n_heads_q=32, n_heads_kv=8,
        d_model=4096, d_ff=14336, vocab_size=128_256,
        batch_size=1, context_length=131_072, weight_bits=16, kv_bits=16,
        runtime_bytes=0)
    workload = evaluate_llm_decode(inp)
    data = {}
    for name in ARCHITECTURES:
        case = load_case_config(ROOT / "configs" / "cases" / f"{name}.yaml")
        geometry = resolve_case_geometry(case)
        system = resolve_system_power(case, project_root=ROOT, geometry=geometry)
        fit = evaluate_architecture_capacity_feasibility(
            workload, resolve_architecture_capacity(case, geometry, system),
            reserved_capacity_bytes=0)
        performance = evaluate_llm_decode_performance(
            workload, fit, batch_size=1,
            matched_payload_bandwidth_bits_per_second=39.2e12,
            effective_compute_flops_per_second=100e12)
        by_rho = {}
        for rho in RHOS:
            energy = evaluate_architecture_decode_memory_energy(
                workload, fit, system, rho=rho)
            policy = ("EXISTING_PLACEHOLDER_ZERO"
                      if name == "orthogonal_m3d_igzo" else "REQUIRE_RESOLVED")
            power = evaluate_llm_decode_workload_power(
                energy, performance, system,
                unresolved_logic_background_policy=policy)
            thermal = _thermal(name, rho, power)
            row = assemble_conditional_llm_decode_e2e_row(
                inp, workload, fit, performance, energy, power, thermal,
                workload_identifier="LLaMA-3.1-8B-class-B1-S131072-v0")
            by_rho[rho] = (energy, power, thermal, row)
        data[name] = (fit, performance, system, by_rho)
    return inp, workload, data


def _args(frozen, name="conventional_hbm_2x1", rho=1):
    inp, workload, data = frozen
    fit, performance, _, by_rho = data[name]
    energy, power, thermal, _ = by_rho[rho]
    return [inp, workload, fit, performance, energy, power, thermal]


def _assemble(args):
    return assemble_conditional_llm_decode_e2e_row(
        *args, workload_identifier="LLaMA-3.1-8B-class-B1-S131072-v0")


def test_valid_rows_all_architectures_and_claim_boundaries(frozen):
    for name in ARCHITECTURES:
        row = _assemble(_args(frozen, name))
        assert row.architecture == name
        assert row.bandwidth_status == "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"
        assert row.bandwidth_capability_status == "NOT_VALIDATED"
        assert row.write_energy_model_status == "NOT_VALIDATED"
        assert row.system_j_token_status == "NOT_AVAILABLE"


@pytest.mark.parametrize("index,update", [
    (2, {"architecture": "orthogonal_si"}),
    (3, {"architecture": "orthogonal_si"}),
    (4, {"architecture": "orthogonal_si"}),
    (5, {"architecture": "orthogonal_si"}),
    (6, {"architecture": "orthogonal_si"}),
])
def test_architecture_mismatch_rejected(frozen, index, update):
    args = _args(frozen)
    args[index] = args[index].model_copy(update=update)
    with pytest.raises(ValueError, match="architecture"):
        _assemble(args)


def test_workload_and_capacity_mismatch_rejected(frozen):
    args = _args(frozen)
    args[1] = args[1].model_copy(update={"read_bytes_per_token": 1.0})
    with pytest.raises(ValueError, match="traffic"):
        _assemble(args)
    args = _args(frozen)
    args[2] = args[2].model_copy(update={"required_capacity_bytes": 1.0})
    with pytest.raises(ValueError, match="required capacity"):
        _assemble(args)


def test_rho_energy_power_and_power_thermal_mismatch_rejected(frozen):
    args = _args(frozen)
    args[5] = args[5].model_copy(update={"rho": 100.0})
    with pytest.raises(ValueError, match="rho"):
        _assemble(args)
    args = _args(frozen)
    args[5] = args[5].model_copy(update={
        "memory_dynamic_energy_j_per_token":
            args[5].memory_dynamic_energy_j_per_token + 1.0})
    with pytest.raises(ValueError, match="dynamic energy"):
        _assemble(args)
    args = _args(frozen)
    args[6] = args[6].model_copy(update={
        "mapped_package_power_W": args[6].mapped_package_power_W + 1.0})
    with pytest.raises(ValueError, match="mapped package power"):
        _assemble(args)


def test_performance_energy_traffic_mismatch_rejected(frozen):
    args = _args(frozen)
    args[4] = args[4].model_copy(update={"read_bytes_per_token": 1.0})
    with pytest.raises(ValueError, match="energy read traffic"):
        _assemble(args)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_capacity_infeasible_propagates_without_thermal(frozen, name):
    inp, workload, data = frozen
    fit, _, system, _ = data[name]
    fit = fit.model_copy(update={
        "reserved_capacity_bytes": fit.physical_capacity_bytes,
        "usable_capacity_bytes": 0.0,
        "capacity_margin_bytes": -workload.required_capacity_bytes,
        "capacity_utilization": None,
        "capacity_feasible": False,
    })
    performance = evaluate_llm_decode_performance(
        workload, fit, batch_size=1,
        matched_payload_bandwidth_bits_per_second=39.2e12,
        effective_compute_flops_per_second=100e12)
    energy = evaluate_architecture_decode_memory_energy(
        workload, fit, system, rho=1)
    policy = ("EXISTING_PLACEHOLDER_ZERO"
              if name == "orthogonal_m3d_igzo" else "REQUIRE_RESOLVED")
    power = evaluate_llm_decode_workload_power(
        energy, performance, system,
        unresolved_logic_background_policy=policy)
    row = assemble_conditional_llm_decode_e2e_row(
        inp, workload, fit, performance, energy, power, None,
        workload_identifier="blocked-test")
    assert not row.capacity_feasible
    assert row.package_Tmax_degC is None
    assert row.thermal_status == "BLOCKED_BY_CAPACITY"


def test_m3d_and_rho_zero_statuses_are_preserved(frozen):
    zero = _assemble(_args(frozen, "orthogonal_m3d_igzo", 0))
    assert zero.rho_zero_status == "MATHEMATICAL_WRITE_ENERGY_LOWER_BOUND"
    assert zero.m3d_logic_background_status == "CONDITIONAL_LOWER_BOUND"
    assert zero.memory_total_completeness_status == (
        "CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND")


def test_forbidden_metric_names_and_claims_absent():
    fields = set(__import__(
        "om3dthermal.evaluator.llm_decode_e2e", fromlist=[
            "ConditionalLLMDecodeE2ERow"]).ConditionalLLMDecodeE2ERow.model_fields)
    assert "memory_dynamic_energy_j_per_token" in fields
    assert not fields.intersection({
        "system_energy_j_per_token", "system_j_per_token",
        "total_system_energy", "bandwidth_capability_bits_per_second"})


def test_exact_order_count_monotonic_and_rho_invariants(frozen):
    rows = [frozen[2][name][3][rho][3]
            for name in ARCHITECTURES for rho in RHOS]
    result = validate_conditional_llm_decode_e2e_table(rows)
    assert len(result) == 12
    assert all(sum(row.architecture == name for row in result) == 4
               for name in ARCHITECTURES)


def test_strict_rho_set_and_order_rejected(frozen):
    rows = [frozen[2][name][3][rho][3]
            for name in ARCHITECTURES for rho in RHOS]
    with pytest.raises(ValueError, match="ordered frozen"):
        validate_conditional_llm_decode_e2e_table(rows[:-1])
    with pytest.raises(ValueError, match="ordered frozen"):
        validate_conditional_llm_decode_e2e_table(rows[::-1])


def test_monotonic_and_invariant_failures_rejected(frozen):
    rows = [frozen[2][name][3][rho][3]
            for name in ARCHITECTURES for rho in RHOS]
    broken = list(rows)
    broken[2] = broken[2].model_copy(update={
        "package_Tmax_degC": broken[1].package_Tmax_degC - 1.0})
    with pytest.raises(ValueError, match="package_Tmax"):
        validate_conditional_llm_decode_e2e_table(broken)
    broken = list(rows)
    broken[1] = broken[1].model_copy(update={"read_bytes_per_token": 1.0})
    with pytest.raises(ValueError, match="read_bytes"):
        validate_conditional_llm_decode_e2e_table(broken)


def test_cross_architecture_workload_mismatch_is_rejected(frozen):
    rows = [frozen[2][name][3][rho][3]
            for name in ARCHITECTURES for rho in RHOS]
    broken = list(rows)
    for index in range(4, 8):
        broken[index] = broken[index].model_copy(update={
            "workload_identifier": "different-workload"})
    with pytest.raises(ValueError, match="workload_identifier"):
        validate_conditional_llm_decode_e2e_table(broken)
