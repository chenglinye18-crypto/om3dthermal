"""E6 workload power-to-thermal mapping tests (no thermal solves)."""

from __future__ import annotations

from pathlib import Path

import pytest

import om3dthermal.evaluator.llm_decode_workload_thermal as thermal_module
from om3dthermal.architecture_capacity import resolve_architecture_capacity
from om3dthermal.evaluator import (
    WorkloadPowerBlockedError,
    evaluate_architecture_decode_memory_energy,
    evaluate_llm_decode_performance,
    evaluate_llm_decode_workload_power,
    map_workload_power_to_thermal,
)
from om3dthermal.power import (
    load_case_config,
    map_system_power_to_thermal,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.workload import (
    LLMDecodeInput,
    evaluate_architecture_capacity_feasibility,
    evaluate_llm_decode,
)


ROOT = Path(__file__).parents[1]
CASES = ROOT / "configs" / "cases"
ARCHITECTURES = (
    "conventional_hbm_2x1", "orthogonal_si", "orthogonal_m3d_igzo")


@pytest.fixture(scope="module")
def frozen():
    workload = evaluate_llm_decode(LLMDecodeInput(
        n_param=8_000_000_000, n_layers=32, n_heads_q=32, n_heads_kv=8,
        d_model=4096, d_ff=14336, vocab_size=128_256,
        batch_size=1, context_length=131_072, weight_bits=16, kv_bits=16,
        runtime_bytes=0))
    data = {}
    for name in ARCHITECTURES:
        case = load_case_config(CASES / f"{name}.yaml")
        geometry = resolve_case_geometry(case)
        system = resolve_system_power(
            case, project_root=ROOT, geometry=geometry)
        capacity = resolve_architecture_capacity(case, geometry, system)
        fit = evaluate_architecture_capacity_feasibility(
            workload, capacity, reserved_capacity_bytes=0)
        performance = evaluate_llm_decode_performance(
            workload, fit, batch_size=1,
            matched_payload_bandwidth_bits_per_second=39.2e12,
            effective_compute_flops_per_second=100e12)
        policy = ("EXISTING_PLACEHOLDER_ZERO"
                  if name == "orthogonal_m3d_igzo" else "REQUIRE_RESOLVED")
        powers = {}
        for rho in (0, 1, 100, 1000):
            energy = evaluate_architecture_decode_memory_energy(
                workload, fit, system, rho=rho)
            powers[rho] = evaluate_llm_decode_workload_power(
                energy, performance, system,
                unresolved_logic_background_policy=policy)
        data[name] = (case, system, powers)
    return data


def test_three_architecture_source_selection_and_gpu_once(frozen) -> None:
    expected = {
        "conventional_hbm_2x1": {
            "gpu", "dram_group_0", "base_route_group_0",
            "dram_group_1", "base_route_group_1"},
        "orthogonal_si": {"gpu", "orthogonal_si_memory"},
        "orthogonal_m3d_igzo": {"gpu", "m3d_memory_bitcell_beol"},
    }
    for name, (case, system, powers) in frozen.items():
        mapping = map_workload_power_to_thermal(case, system, powers[0])
        names = [source.name for source in mapping.sources]
        assert set(names) == expected[name]
        assert names.count("gpu") == 1
        assert next(source for source in mapping.sources
                    if source.name == "gpu").power_W == 300


def test_hbm_dynamic_decomposition_and_visible_group_split_close(frozen) -> None:
    case, system, powers = frozen["conventional_hbm_2x1"]
    power = powers[100]
    mapping = map_workload_power_to_thermal(case, system, power)
    by_name = {source.name: source.power_W for source in mapping.sources}
    assert by_name["dram_group_0"] == by_name["dram_group_1"]
    assert by_name["base_route_group_0"] == by_name["base_route_group_1"]
    memory_mapped = sum(value for name, value in by_name.items()
                        if name != "gpu")
    assert memory_mapped == pytest.approx(power.memory_workload_total_W)


def test_orthogonal_si_single_memory_source_closes(frozen) -> None:
    case, system, powers = frozen["orthogonal_si"]
    mapping = map_workload_power_to_thermal(case, system, powers[100])
    source = next(source for source in mapping.sources if source.name != "gpu")
    assert source.name == "orthogonal_si_memory"
    assert source.power_W == powers[100].memory_workload_total_W
    assert source.selector == {"material": "MOSAIC_BEOL", "tags": {}}


def test_m3d_uses_merged_region_and_preserves_lower_bound(frozen) -> None:
    case, system, powers = frozen["orthogonal_m3d_igzo"]
    mapping = map_workload_power_to_thermal(case, system, powers[100])
    source = next(source for source in mapping.sources if source.name != "gpu")
    assert source.name == "m3d_memory_bitcell_beol"
    assert source.selector == {"tags": {"role": "m3d_bitcell_beol_stack"}}
    assert mapping.memory_total_completeness_status == (
        "CONDITIONAL_LOWER_BOUND_UNRESOLVED_LOGIC_BACKGROUND")


def test_every_old_source_power_is_replaced_and_selectors_reused(frozen) -> None:
    for case, system, powers in frozen.values():
        old = thermal_module.compile_case_thermal(case, system)
        mapping = map_workload_power_to_thermal(case, system, powers[0])
        old_by_name = {source.name: source for source in
                       old.thermal_power_sources.sources}
        new_by_name = {source.name: source for source in
                       mapping.simulation.thermal_power_sources.sources}
        assert set(old_by_name) == set(new_by_name)
        for name in old_by_name:
            assert new_by_name[name].selector == old_by_name[name].selector
        assert sum(float(source.total_power) for source in new_by_name.values()) == (
            pytest.approx(powers[0].package_workload_total_W))
        assert any(float(new_by_name[name].total_power) !=
                   float(old_by_name[name].total_power) for name in old_by_name)


def test_mapping_total_closure_all_twelve_rows(frozen) -> None:
    mappings = []
    for case, system, powers in frozen.values():
        for power in powers.values():
            mappings.append(map_workload_power_to_thermal(case, system, power))
    assert len(mappings) == 12
    for mapping in mappings:
        assert mapping.absolute_closure_error_W <= 1e-9
        assert mapping.mapped_total_power_W == pytest.approx(
            mapping.expected_package_total_power_W)
        assert mapping.write_spatial_distribution_status == (
            "WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY")


def test_blocked_e5_does_not_compile_or_run_thermal(frozen, monkeypatch) -> None:
    case, system, powers = frozen["orthogonal_si"]
    valid = powers[0]
    blocked = valid.model_construct(**{
        **valid.model_dump(),
        "evaluation_status": "BLOCKED_BY_CAPACITY_OR_UPSTREAM_EVALUATION",
    })
    def forbidden(*args, **kwargs):
        raise AssertionError("thermal construction must not run")
    monkeypatch.setattr(thermal_module, "compile_case_thermal", forbidden)
    with pytest.raises(WorkloadPowerBlockedError):
        map_workload_power_to_thermal(case, system, blocked)


def test_rho_one_source_powers_reproduce_old_mapping(frozen) -> None:
    for case, system, powers in frozen.values():
        old = map_system_power_to_thermal(case, system)
        new = map_workload_power_to_thermal(case, system, powers[1])
        old_by_name = {source.name: source.power_W for source in old.sources}
        new_by_name = {source.name: source.power_W for source in new.sources}
        assert set(old_by_name) == set(new_by_name)
        for name in old_by_name:
            assert new_by_name[name] == pytest.approx(old_by_name[name], abs=1e-9)


def test_output_has_no_system_energy_or_bandwidth_capability_claim() -> None:
    forbidden = {
        "system_j_per_token", "gpu_energy_j_per_token",
        "bandwidth_capability", "validated_bandwidth_bits_per_second",
    }
    assert forbidden.isdisjoint(
        thermal_module.LLMDecodeWorkloadThermalMetrics.model_fields)
