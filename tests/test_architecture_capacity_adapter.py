"""Canonical capacity resolution and aggregate workload adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import om3dthermal.architecture_comparison as architecture_comparison
import om3dthermal.case_runner as case_runner
import om3dthermal.thermal.gpu_pcg as gpu_pcg
from om3dthermal.architecture_capacity import (
    ResolvedArchitectureCapacity,
    resolve_architecture_capacity,
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
CASES = ROOT / "configs" / "cases"
ARCHITECTURES = (
    "conventional_hbm_2x1",
    "orthogonal_si",
    "orthogonal_m3d_igzo",
)
EXPECTED_GIB = {
    "conventional_hbm_2x1": 114.75,
    "orthogonal_si": 234.28125,
    "orthogonal_m3d_igzo": 428.75,
}


def _resolve_capacity(name: str) -> ResolvedArchitectureCapacity:
    case = load_case_config(CASES / f"{name}.yaml")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(
        case, project_root=ROOT, geometry=geometry)
    return resolve_architecture_capacity(case, geometry, system)


def _frozen_workload():
    return evaluate_llm_decode(LLMDecodeInput(
        n_param=8_000_000_000,
        n_layers=32,
        n_heads_q=32,
        n_heads_kv=8,
        d_head=128,
        d_model=4096,
        d_ff=14336,
        vocab_size=128_256,
        batch_size=1,
        context_length=131_072,
        weight_bits=16,
        kv_bits=16,
        runtime_bytes=0,
    ))


@pytest.fixture(scope="module")
def capacities() -> dict[str, ResolvedArchitectureCapacity]:
    return {name: _resolve_capacity(name) for name in ARCHITECTURES}


def test_resolver_preserves_canonical_capacities(
    capacities: dict[str, ResolvedArchitectureCapacity],
) -> None:
    assert {
        name: capacity.system_capacity_GiB
        for name, capacity in capacities.items()
    } == EXPECTED_GIB


def test_exact_bit_closure_and_unit_conversion(
    capacities: dict[str, ResolvedArchitectureCapacity],
) -> None:
    for capacity in capacities.values():
        assert capacity.total_bits == (
            capacity.bits_per_instance * capacity.instance_count)
        assert capacity.system_capacity_bytes == capacity.total_bits / 8
        assert capacity.capacity_per_instance_bytes == (
            capacity.bits_per_instance / 8)
        assert capacity.system_capacity_GiB == (
            capacity.system_capacity_bytes / 2**30)
        assert capacity.capacity_per_instance_GiB == (
            capacity.capacity_per_instance_bytes / 2**30)


def test_architecture_comparison_compatibility_uses_public_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_case_config(CASES / "conventional_hbm_2x1.yaml")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(
        case, project_root=ROOT, geometry=geometry)
    called = False
    public_resolver = resolve_architecture_capacity

    def recording_resolver(*args):
        nonlocal called
        called = True
        return public_resolver(*args)

    monkeypatch.setattr(
        architecture_comparison,
        "resolve_architecture_capacity",
        recording_resolver,
    )
    compatibility = architecture_comparison._resolved_capacity(
        case, geometry, system)
    assert called is True
    assert compatibility["system_capacity_GiB"] == 114.75


def test_same_workload_and_first_table_are_feasible(
    capacities: dict[str, ResolvedArchitectureCapacity],
) -> None:
    workload = _frozen_workload()
    results = [
        evaluate_architecture_capacity_feasibility(
            workload,
            capacities[name],
            reserved_capacity_bytes=0,
        )
        for name in ARCHITECTURES
    ]

    assert workload.weight_footprint_bytes == 16_000_000_000
    assert workload.kv_footprint_bytes == 17_179_869_184
    assert workload.required_capacity_bytes == 33_179_869_184
    assert {result.required_capacity_bytes for result in results} == {
        workload.required_capacity_bytes}
    assert all(result.capacity_feasible for result in results)
    assert len({result.physical_capacity_bytes for result in results}) == 3
    assert len({result.usable_capacity_bytes for result in results}) == 3
    assert len({result.capacity_margin_bytes for result in results}) == 3
    assert len({result.capacity_utilization for result in results}) == 3


def test_adapter_requires_explicit_reserve(
    capacities: dict[str, ResolvedArchitectureCapacity],
) -> None:
    with pytest.raises(TypeError, match="reserved_capacity_bytes"):
        evaluate_architecture_capacity_feasibility(  # type: ignore[call-arg]
            _frozen_workload(), capacities[ARCHITECTURES[0]])


def test_scope_and_source_status_propagate(
    capacities: dict[str, ResolvedArchitectureCapacity],
) -> None:
    result = evaluate_architecture_capacity_feasibility(
        _frozen_workload(),
        capacities[ARCHITECTURES[0]],
        reserved_capacity_bytes=0,
    )
    assert result.capacity_scope_status == (
        "AGGREGATE_CAPACITY_FEASIBILITY_ONLY")
    assert result.capacity_source_status == (
        "ANALYTICAL_PACKING_DIAGNOSTICS_BIT_CLOSURE")


def test_capacity_path_does_not_invoke_thermal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("thermal path must not be invoked")

    monkeypatch.setattr(
        architecture_comparison, "compile_case_thermal", forbidden)
    monkeypatch.setattr(
        architecture_comparison, "run_steady_pipeline", forbidden)
    monkeypatch.setattr(case_runner, "run_steady_pipeline", forbidden)
    monkeypatch.setattr(gpu_pcg, "solve_pcg_gpu", forbidden)

    capacity = _resolve_capacity("orthogonal_m3d_igzo")
    result = evaluate_architecture_capacity_feasibility(
        _frozen_workload(), capacity, reserved_capacity_bytes=0)
    assert result.capacity_feasible is True
