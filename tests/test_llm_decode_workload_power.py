"""Semantic and frozen-scenario tests for E5 workload power accounting."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest

from om3dthermal.architecture_comparison import (
    _resolve_case_power_operating_point_kwargs,
)
from om3dthermal.architecture_capacity import resolve_architecture_capacity
from om3dthermal.evaluator import (
    ArchitectureDecodeMemoryEnergyMetrics,
    GPUDecodeEnergyMetrics,
    LLMDecodePerformanceMetrics,
    evaluate_architecture_decode_memory_energy,
    evaluate_gpu_decode_energy,
    evaluate_llm_decode_performance,
    evaluate_llm_decode_workload_power,
)
from om3dthermal.power import (
    EnergyDecomposition,
    MemoryPowerResult,
    ResolvedSystemPower,
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.workload import (
    ArchitectureCapacityFeasibility,
    LLMDecodeInput,
    evaluate_architecture_capacity_feasibility,
    evaluate_llm_decode,
)
from om3dthermal.experiment.config import load_platform_spec


ROOT = Path(__file__).parents[1]
CASES = ROOT / "configs" / "cases"
ARCHITECTURES = (
    "conventional_hbm_2x1", "orthogonal_si", "orthogonal_m3d_igzo")
RHOS = (0, 1, 100, 1000)


def _memory(*, logic=0.0, access=10.0, refresh=2.0, background=3.0):
    return MemoryPowerResult(
        technology="test", backend="test", architecture="test",
        E_memory_internal_pj_bit=0, E_vertical_pj_bit=0,
        E_feol_route_pj_bit=0, E_base_route_pj_bit=0,
        E_interface_pj_bit=0, E_access_total_pj_bit=0,
        P_read_W=access, P_write_W=0, P_access_W=access,
        P_refresh_W=refresh, P_memory_background_W=background,
        P_logic_background_W=logic,
        P_total_W=(None if logic is None else access+refresh+background+logic),
    )


def _system(*, name="test", logic=0.0, access=10.0, refresh=2.0,
            background=3.0, gpu=7.0):
    memory = _memory(
        logic=logic, access=access, refresh=refresh, background=background)
    old_total = access + refresh + background + (0 if logic is None else logic)
    return ResolvedSystemPower(
        case_name=name, architecture_type="test", gpu_power_W=gpu,
        memory_power_model="analytical", memory_power_status="VALIDATED",
        read_bandwidth_gbps=1, memory_access_energy_pJ_per_bit=1,
        memory_gpu_bandwidth_demand_bytes_per_s=None,
        memory_raw_bandwidth_capability_bytes_per_s=None,
        gpu_peak_bandwidth_bytes_per_s=None,
        memory_gpu_actual_bandwidth_bytes_per_s=None,
        memory_gpu_transfer_bottleneck=None,
        memory_dynamic_power_bandwidth_source="TEST_FIXTURE",
        memory_access_power_W=access, refresh_power_W=refresh,
        resolved_total_memory_power_W=old_total,
        memory_result=memory, diagnostics={},
    )


def _energy(*, name="test", rho=1.0, feasible=True, read=1.0, write=1.0,
            joules=2.0):
    status = (
        "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY" if feasible
        else "CAPACITY_INFEASIBLE")
    return ArchitectureDecodeMemoryEnergyMetrics(
        architecture=name, rho=rho, capacity_feasible=feasible,
        read_bytes_per_token=read, write_bytes_per_token=write,
        read_energy_pj_per_bit=1, write_energy_pj_per_bit=rho,
        read_dynamic_energy_j_per_token=(1 if feasible else None),
        write_dynamic_energy_j_per_token=(joules-1 if feasible else None),
        memory_dynamic_energy_j_per_token=(joules if feasible else None),
        read_energy_status="CURRENT_NOMINAL_ANALYTICAL_MODEL",
        write_energy_status="RHO_SENSITIVITY_NOT_PHYSICAL_CLAIM",
        energy_scope_status="MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY",
        scenario_status="CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY",
        zhu_transferability_status="NOT_VALIDATED",
        evaluation_status=status,
    )


def _performance(*, name="test", feasible=True, read=1.0, write=1.0,
                 aggregate=3.0, batch=1):
    none = None if not feasible else 1.0 / aggregate
    return LLMDecodePerformanceMetrics(
        architecture=name, batch_size=batch, capacity_feasible=feasible,
        read_bytes_per_token=read, write_bytes_per_token=write,
        traffic_bits_per_token=(read+write)*8, flops_per_token=1,
        matched_payload_bandwidth_bits_per_second=1,
        effective_compute_flops_per_second=1,
        memory_time_per_token_equivalent_s=none,
        compute_time_per_token_equivalent_s=none,
        token_equivalent_time_s=none,
        aggregate_step_time_s=none,
        aggregate_tokens_per_second=(aggregate if feasible else None),
        per_sequence_tokens_per_second=none,
        per_sequence_step_latency_s=none,
        compute_throughput_required_to_match_memory_flops_per_second=none,
        bottleneck=("MEMORY" if feasible
                    else "NOT_EVALUATED_CAPACITY_INFEASIBLE"),
        performance_status=("EVALUATED_MATCHED_REFERENCE_SCENARIO" if feasible
                            else "BLOCKED_BY_CAPACITY"),
        bandwidth_status="MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
        compute_throughput_status="NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED",
        memory_bandwidth_model="SHARED_READ_WRITE_PAYLOAD_BANDWIDTH",
        overlap_model="ROOFLINE_MAX",
    )


def _evaluate(energy=None, performance=None, system=None,
              policy="REQUIRE_RESOLVED"):
    resolved_energy = energy or _energy()
    resolved_performance = performance or _performance()
    resolved_system = system or _system()
    return evaluate_llm_decode_workload_power(
        resolved_energy, resolved_performance, resolved_system,
        unresolved_logic_background_policy=policy,
        gpu_decode_energy=_gpu_metrics(
            resolved_energy,
            (resolved_performance
             if resolved_performance.capacity_feasible
             == resolved_energy.capacity_feasible
             else _performance(
                 name=resolved_energy.architecture,
                 feasible=resolved_energy.capacity_feasible,
                 read=resolved_energy.read_bytes_per_token,
                 write=resolved_energy.write_bytes_per_token)),
            resolved_system))


def _gpu_metrics(energy, performance, system):
    if not energy.capacity_feasible:
        nominal = load_platform_spec(
            ROOT / "configs/platform/gpu_package_h200_reference.yaml",
            project_root=ROOT).gpu_decode_power
        return evaluate_gpu_decode_energy(performance, energy, nominal)
    token_time = performance.token_equivalent_time_s
    demand = (performance.read_bytes_per_token
              + performance.write_bytes_per_token) / token_time
    gpu = system.gpu_power_W
    return GPUDecodeEnergyMetrics(
        architecture=energy.architecture, rho=energy.rho,
        capacity_feasible=True, memory_bandwidth_utilization=0.5,
        utilization_clamped=False,
        bandwidth_demand_bytes_per_s=demand,
        bandwidth_actual_bytes_per_s=demand,
        bandwidth_saturated=False, gpu_dynamic_power_W=gpu - 1.0,
        compute_demand_flops_per_s=None, compute_actual_flops_per_s=None,
        compute_utilization=None, compute_saturated=None,
        gpu_dynamic_compute_power_W=None,
        compute_energy_dynamic_J_per_FLOP=None, gpu_power_regime="MEMORY",
        gpu_decode_power_W=gpu, token_time_s=token_time,
        gpu_energy_j_per_token=gpu * token_time,
        memory_dynamic_energy_j_per_token=(
            energy.memory_dynamic_energy_j_per_token),
        system_energy_j_per_token=(
            gpu * token_time + energy.memory_dynamic_energy_j_per_token),
        evaluation_status="EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY",
        gpu_power_model_status="ANALYTICAL_AFFINE_UTILIZATION_MODEL",
        parameter_provenance_status=(
            "MEASURED_REFERENCE_RANGE_WITH_EXPLICIT_COMPUTE_SELECTION"),
        bandwidth_status="MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED",
        system_energy_scope_status=(
            "GPU_PLUS_MEMORY_DYNAMIC_ONLY__EXCLUDES_HOST_CPU_DRAM_COOLING_NETWORK"),
        utilization_semantics_status=(
            "REGIME_ACTUAL_RATE_OVER_CEILING__STRICT_EXCEEDANCE_SATURATION"),
    )


def test_two_j_per_token_times_three_aggregate_tokens_is_six_watts() -> None:
    result = _evaluate()
    assert result.memory_dynamic_access_power_W == 6


def test_batch_factor_is_not_applied_again() -> None:
    result = _evaluate(performance=_performance(batch=8, aggregate=3))
    assert result.memory_dynamic_access_power_W == 6


def test_memory_and_package_power_close_without_old_access_double_counting() -> None:
    result = _evaluate()
    assert result.memory_workload_total_W == 6 + 2 + 3 + 0
    assert result.package_workload_total_W == 7 + result.memory_workload_total_W
    assert result.memory_workload_total_W != 10 + 6 + 2 + 3


@pytest.fixture
def gpu_operating_point():
    nominal = load_platform_spec(
        ROOT / "configs/platform/gpu_package_h200_reference.yaml",
        project_root=ROOT).gpu_decode_power
    # Dimensionally consistent small-number fixture: 2 B/s demand is half of
    # the 4 B/s peak.
    spec = type(nominal).model_validate(nominal.model_dump() | {
        "e_decode_J_per_bit": 1.0,
        "peak_memory_bandwidth_bytes_per_s": 4.0,
    })
    energy, performance = _energy(), _performance(aggregate=1.0)
    gpu = evaluate_gpu_decode_energy(performance, energy, spec)
    return energy, performance, gpu


def test_affine_power_is_consumed_once(gpu_operating_point):
    energy, performance, gpu = gpu_operating_point
    power = evaluate_llm_decode_workload_power(
        energy, performance, _system(gpu=90),
        unresolved_logic_background_policy="REQUIRE_RESOLVED", gpu_decode_energy=gpu)
    # The fixture's actual byte rate is half the peak, so power is
    # 74 W + 1 J/bit x 8 bit/byte x 2 B/s = 90 W.
    assert power.gpu_power_W == pytest.approx(90.0)
    assert power.package_workload_total_W == pytest.approx(
        90.0 + power.memory_workload_total_W)
    assert power.gpu_power_status == "WORKLOAD_AFFINE_GPU_DECODE_POWER_SHARED_WITH_ENERGY"


@pytest.mark.parametrize("update, message", [
    ({"rho": 2.0}, "identity mismatch"),
    ({"architecture": "wrong"}, "identity mismatch"),
    ({"token_time_s": 2.0}, "operating point mismatch"),
    ({"memory_dynamic_energy_j_per_token": 4.0}, "operating point mismatch"),
    ({"gpu_energy_j_per_token": 300.0}, "do not close"),
    ({"gpu_decode_power_W": float("nan")}, "finite"),
    ({"gpu_decode_power_W": -1.0}, "non-negative"),
    ({"evaluation_status": "BLOCKED_BY_CAPACITY"}, "requires evaluated GPU energy"),
])
def test_inconsistent_gpu_energy_cannot_feed_thermal_power(
        gpu_operating_point, update, message):
    energy, performance, gpu = gpu_operating_point
    with pytest.raises(ValueError, match=message):
        evaluate_llm_decode_workload_power(
            energy, performance, _system(gpu=90),
            unresolved_logic_background_policy="REQUIRE_RESOLVED",
            gpu_decode_energy=gpu.model_copy(update=update))


def test_inconsistent_token_time_and_throughput_are_rejected(gpu_operating_point):
    energy, performance, gpu = gpu_operating_point
    with pytest.raises(ValueError, match="token time and aggregate throughput"):
        evaluate_llm_decode_workload_power(
            energy, performance.model_copy(update={"aggregate_tokens_per_second": 2}),
            _system(gpu=90), unresolved_logic_background_policy="REQUIRE_RESOLVED",
            gpu_decode_energy=gpu)


def test_blocked_gpu_energy_keeps_all_power_outputs_absent():
    energy, performance = _energy(feasible=False), _performance(feasible=False)
    spec = load_platform_spec(
        ROOT / "configs/platform/gpu_package_h200_reference.yaml",
        project_root=ROOT).gpu_decode_power
    gpu = evaluate_gpu_decode_energy(performance, energy, spec)
    power = evaluate_llm_decode_workload_power(
        energy, performance, _system(gpu=90),
        unresolved_logic_background_policy="REQUIRE_RESOLVED", gpu_decode_energy=gpu)
    assert power.gpu_power_W is None
    assert power.package_workload_total_W is None


def test_m3d_placeholder_preserves_raw_none_and_marks_lower_bound() -> None:
    result = _evaluate(
        system=_system(logic=None), policy="EXISTING_PLACEHOLDER_ZERO")
    assert result.logic_background_raw_W is None
    assert result.logic_background_effective_W == 0
    assert result.logic_background_status == (
        "EXISTING_PLACEHOLDER_ZERO_NOT_SEPARATELY_MODELED")
    assert result.memory_total_completeness_status == (
        "CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND")


def test_require_resolved_returns_unresolved_for_none_logic() -> None:
    result = _evaluate(system=_system(logic=None))
    assert result.evaluation_status == "UNRESOLVED_STATIC_POWER"
    assert result.memory_workload_total_W is None


def test_placeholder_policy_rejects_numeric_logic() -> None:
    with pytest.raises(ValueError, match="already numeric"):
        _evaluate(policy="EXISTING_PLACEHOLDER_ZERO")


def test_placeholder_requires_old_total_closure() -> None:
    bad = replace(_system(logic=None), resolved_total_memory_power_W=999)
    with pytest.raises(ValueError, match="does not close"):
        _evaluate(system=bad, policy="EXISTING_PLACEHOLDER_ZERO")


def test_capacity_infeasible_returns_none_power_fields() -> None:
    result = _evaluate(
        energy=_energy(feasible=False), performance=_performance(feasible=False))
    assert result.evaluation_status == (
        "BLOCKED_BY_CAPACITY_OR_UPSTREAM_EVALUATION")
    assert result.memory_dynamic_access_power_W is None
    assert result.memory_workload_total_W is None
    assert result.package_workload_total_W is None


def test_architecture_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="architecture"):
        _evaluate(system=_system(name="wrong"))


@pytest.mark.parametrize("direction", ["read", "write"])
def test_traffic_mismatch_is_rejected(direction) -> None:
    kwargs = {"read": 1.0, "write": 1.0}
    kwargs[direction] = 2.0
    with pytest.raises(ValueError, match="traffic mismatch"):
        _evaluate(performance=_performance(**kwargs))


def test_energy_performance_feasibility_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="feasibility mismatch"):
        _evaluate(performance=_performance(feasible=False))


@pytest.mark.parametrize("component", ["refresh", "background"])
def test_none_static_component_is_not_silently_zero(component) -> None:
    memory = _memory()
    memory = replace(memory, **{
        "P_refresh_W" if component == "refresh"
        else "P_memory_background_W": None})
    system = replace(_system(), memory_result=memory)
    with pytest.raises((TypeError, ValueError)):
        _evaluate(system=system)


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf, -math.inf, True])
@pytest.mark.parametrize("target", ["energy", "throughput", "gpu"])
def test_invalid_numeric_inputs_are_rejected(bad, target) -> None:
    energy = _energy()
    performance = _performance()
    system = _system()
    if target == "energy":
        energy = energy.model_construct(
            **{**energy.model_dump(),
               "memory_dynamic_energy_j_per_token": bad})
    elif target == "throughput":
        performance = performance.model_construct(
            **{**performance.model_dump(),
               "aggregate_tokens_per_second": bad})
    else:
        system = replace(system, gpu_power_W=bad)
    with pytest.raises((TypeError, ValueError)):
        _evaluate(energy=energy, performance=performance, system=system)


def test_policy_is_mandatory_and_invalid_policy_rejected() -> None:
    with pytest.raises(TypeError, match="unresolved_logic_background_policy"):
        evaluate_llm_decode_workload_power(  # type: ignore[call-arg]
            _energy(), _performance(), _system())
    with pytest.raises(ValueError, match="unsupported"):
        _evaluate(policy="INVALID")


def test_output_has_no_thermal_or_system_energy_metrics() -> None:
    forbidden = {
        "thermal", "temperature", "tmax", "system_j_per_token",
        "gpu_energy_j_per_token", "memory_access_power_W",
        "resolved_total_memory_power_W", "P_access_W", "P_total_W",
    }
    fields = set(type(_evaluate()).model_fields)
    assert forbidden.isdisjoint(fields)


@pytest.fixture(scope="module")
def frozen():
    workload = evaluate_llm_decode(LLMDecodeInput(
        n_param=8_000_000_000, n_layers=32, n_heads_q=32, n_heads_kv=8,
        d_model=4096, d_ff=14336, vocab_size=128_256,
        batch_size=1, context_length=131_072, weight_bits=16, kv_bits=16,
        runtime_bytes=0))
    resolved = {}
    for name in ARCHITECTURES:
        case = load_case_config(CASES / f"{name}.yaml")
        geometry = resolve_case_geometry(case)
        system = resolve_system_power(
            case, project_root=ROOT, geometry=geometry,
            **_resolve_case_power_operating_point_kwargs(case, ROOT))
        capacity = resolve_architecture_capacity(case, geometry, system)
        feasibility = evaluate_architecture_capacity_feasibility(
            workload, capacity, reserved_capacity_bytes=0)
        performance = evaluate_llm_decode_performance(
            workload, feasibility, batch_size=1,
            matched_payload_bandwidth_bits_per_second=39.2e12,
            effective_compute_flops_per_second=100e12)
        policy = ("EXISTING_PLACEHOLDER_ZERO" if name == "orthogonal_m3d_igzo"
                  else "REQUIRE_RESOLVED")
        resolved[name] = (workload, feasibility, performance, system, policy)
    return resolved


def _frozen_rows(frozen):
    rows = []
    for name in ARCHITECTURES:
        workload, capacity, performance, system, policy = frozen[name]
        for rho in RHOS:
            energy = evaluate_architecture_decode_memory_energy(
                workload, capacity, system, rho=rho)
            rows.append(evaluate_llm_decode_workload_power(
                energy, performance, system,
                unresolved_logic_background_policy=policy,
                gpu_decode_energy=evaluate_gpu_decode_energy(
                    performance, energy,
                    load_platform_spec(
                        ROOT / "configs/platform/gpu_package_h200_reference.yaml",
                        project_root=ROOT).gpu_decode_power)))
    return rows


def test_frozen_table_has_exactly_twelve_rows_and_statuses(frozen) -> None:
    rows = _frozen_rows(frozen)
    assert len(rows) == 12
    assert {(row.architecture, row.rho) for row in rows} == {
        (name, rho) for name in ARCHITECTURES for rho in RHOS}
    assert all(row.evaluation_status == (
        "EVALUATED_WORKLOAD_DEPENDENT_MEMORY_POWER") for row in rows)
    for row in rows:
        assert row.dynamic_power_status == (
            "WORKLOAD_J_PER_TOKEN_TIMES_AGGREGATE_TOKENS_PER_SECOND")
        assert row.static_power_status == (
            "EXISTING_POWER_MODEL_COMPONENTS_ADDED_ONCE")
        assert row.gpu_power_status == (
            "WORKLOAD_AFFINE_GPU_DECODE_POWER_SHARED_WITH_ENERGY")
        assert row.system_energy_status == (
            "GPU_ENERGY_REPORTED_IN_GPU_DECODE_ENERGY_STAGE")


def test_rho_one_anchor_and_memory_total_close_for_three_architectures(frozen) -> None:
    rows = [row for row in _frozen_rows(frozen) if row.rho == 1]
    assert len(rows) == 3
    for row in rows:
        system = frozen[row.architecture][3]
        assert row.memory_dynamic_access_power_W == pytest.approx(
            system.memory_result.P_access_W, abs=1e-10)
        assert row.memory_workload_total_W == pytest.approx(
            system.resolved_total_memory_power_W, abs=1e-10)
    m3d = next(row for row in rows if row.architecture == "orthogonal_m3d_igzo")
    # Rev v2 re-frozen: refresh scales with capacity 428.75 -> 463.75 GiB
    # (was 33.5603645761 W).
    assert m3d.memory_workload_total_W == pytest.approx(32.8789438715)
    assert m3d.logic_background_raw_W is None
    assert m3d.logic_background_effective_W == 0


def test_conv_and_si_keep_resolved_explicit_zero(frozen) -> None:
    rows = [row for row in _frozen_rows(frozen)
            if row.rho == 1 and row.architecture != "orthogonal_m3d_igzo"]
    assert all(row.logic_background_raw_W == 0 for row in rows)
    assert all(row.logic_background_status == "RESOLVED_EXPLICIT_ZERO"
               for row in rows)
    assert all(row.memory_total_completeness_status ==
               "RESOLVED_EXISTING_STATIC_COMPONENTS" for row in rows)


def test_dynamic_power_monotonic_in_rho(frozen) -> None:
    rows = _frozen_rows(frozen)
    for name in ARCHITECTURES:
        values = [row.memory_dynamic_access_power_W for row in rows
                  if row.architecture == name]
        assert values == sorted(values)
