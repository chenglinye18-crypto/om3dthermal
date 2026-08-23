"""Targeted tests for conditional architecture decode memory energy (E4)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from om3dthermal.architecture_capacity import (
    ResolvedArchitectureCapacity,
    resolve_architecture_capacity,
)
from om3dthermal.evaluator import (
    ArchitectureDecodeMemoryEnergyMetrics,
    evaluate_architecture_decode_memory_energy,
)
from om3dthermal.power import (
    ResolvedSystemPower,
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.workload import (
    ArchitectureCapacityFeasibility,
    LLMDecodeInput,
    LLMDecodeMetrics,
    evaluate_architecture_capacity_feasibility,
    evaluate_llm_decode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARCHITECTURES = (
    "conventional_hbm_2x1",
    "orthogonal_si",
    "orthogonal_m3d_igzo",
)

ROOT = Path(__file__).parents[1]
CASES = ROOT / "configs" / "cases"


def _build_workload(**overrides) -> LLMDecodeMetrics:
    """Build a frozen LLM-3.1-8B-class workload (B=1, S=131072, 16-bit)."""
    defaults = dict(
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
    )
    defaults.update(overrides)
    return evaluate_llm_decode(LLMDecodeInput(**defaults))


def _build_workload_metrics(
    *,
    read_bytes_per_token: float,
    write_bytes_per_token: float,
    required_capacity_bytes: float = 1.0,
) -> LLMDecodeMetrics:
    """Build a synthetic LLMDecodeMetrics for arithmetic / unit tests."""
    return LLMDecodeMetrics(
        weight_footprint_bytes=0.0,
        weight_active_per_step_bytes=0.0,
        kv_footprint_bytes=0.0,
        runtime_bytes=0.0,
        required_capacity_bytes=required_capacity_bytes,
        weight_read_bytes_per_token=0.0,
        kv_read_bytes_per_token=0.0,
        kv_write_bytes_per_token=0.0,
        read_bytes_per_token=read_bytes_per_token,
        write_bytes_per_token=write_bytes_per_token,
        flops_per_token=0,
        flops_sanity_per_token=0,
        weight_activity_model="full_footprint",
        weight_reuse_model="tile_reuse",
        kv_read_model="full_reread",
    )


def _capacity_feasibility(
    architecture: str,
    *,
    feasible: bool,
    required_capacity_bytes: float = 1.0,
    physical_capacity_bytes: float = 2.0,
) -> ArchitectureCapacityFeasibility:
    """Build a synthetic ArchitectureCapacityFeasibility for testing."""
    usable = physical_capacity_bytes
    return ArchitectureCapacityFeasibility(
        architecture=architecture,
        physical_capacity_bytes=physical_capacity_bytes,
        physical_capacity_GiB=physical_capacity_bytes / 2**30,
        reserved_capacity_bytes=0,
        usable_capacity_bytes=usable,
        required_capacity_bytes=required_capacity_bytes,
        capacity_margin_bytes=usable - required_capacity_bytes,
        capacity_utilization=(
            required_capacity_bytes / usable if usable > 0 else None),
        capacity_feasible=feasible,
        capacity_scope_status="AGGREGATE_CAPACITY_FEASIBILITY_ONLY",
        capacity_source_status="ANALYTICAL_PACKING_DIAGNOSTICS_BIT_CLOSURE",
    )


def _system(*, energy_pj_per_bit: float | None) -> ResolvedSystemPower:
    """Build a synthetic ResolvedSystemPower for testing."""
    return ResolvedSystemPower(
        case_name="test",
        architecture_type="test",
        gpu_power_W=100.0,
        memory_power_model="test",
        memory_power_status="test",
        read_bandwidth_gbps=100.0,
        memory_access_energy_pJ_per_bit=energy_pj_per_bit,
        memory_access_power_W=None,
        refresh_power_W=None,
        resolved_total_memory_power_W=None,
        memory_result=None,
        diagnostics={},
    )


def _resolve_architecture_capacity(name: str) -> ResolvedArchitectureCapacity:
    case = load_case_config(CASES / f"{name}.yaml")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(case, project_root=ROOT, geometry=geometry)
    return resolve_architecture_capacity(case, geometry, system)


def _resolve_system(name: str) -> ResolvedSystemPower:
    case = load_case_config(CASES / f"{name}.yaml")
    geometry = resolve_case_geometry(case)
    return resolve_system_power(case, project_root=ROOT, geometry=geometry)


# ---------------------------------------------------------------------------
# 1. Rho validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_rho", [
    -1.0, -1e-9, float("nan"), float("inf"), float("-inf"), True, False, "0.5",
])
def test_rejects_invalid_rho(bad_rho) -> None:
    """rho must be a finite non-negative real; bool/NaN/inf/negative rejected."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=2.0)
    with pytest.raises((TypeError, ValueError)):
        evaluate_architecture_decode_memory_energy(workload, cap, system, rho=bad_rho)


def test_accepts_zero_rho() -> None:
    """rho = 0 is valid (write energy becomes zero)."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=2.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.0)
    assert r.rho == 0.0
    assert r.write_energy_pj_per_bit == 0.0


# ---------------------------------------------------------------------------
# 2. System energy validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_energy", [
    -1.0, float("nan"), float("inf"), float("-inf"), True, False,
])
def test_rejects_invalid_system_energy(bad_energy) -> None:
    """system.memory_access_energy_pJ_per_bit must be finite non-negative."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=bad_energy)
    with pytest.raises((TypeError, ValueError)):
        evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)


# ---------------------------------------------------------------------------
# 3. Capacity infeasible blocks energy results
# ---------------------------------------------------------------------------

def test_capacity_infeasible_blocks_energy_results() -> None:
    """When capacity_feasible=False, all energy-result fields are None."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility(
        "test", feasible=False, required_capacity_bytes=10.0,
        physical_capacity_bytes=5.0,
    )
    system = _system(energy_pj_per_bit=2.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    assert r.capacity_feasible is False
    assert r.evaluation_status == "CAPACITY_INFEASIBLE"
    assert r.read_dynamic_energy_j_per_token is None
    assert r.write_dynamic_energy_j_per_token is None
    assert r.memory_dynamic_energy_j_per_token is None
    # Energy inputs are still populated so the auditor sees what the
    # architecture would have cost.
    assert r.read_energy_pj_per_bit == 2.0
    assert r.write_energy_pj_per_bit == 1.0


# ---------------------------------------------------------------------------
# 4. No architecture energy resolved
# ---------------------------------------------------------------------------

def test_no_architecture_energy_resolved() -> None:
    """When system.memory_access_energy_pJ_per_bit is None, energy results
    are None and status reflects the unresolved architecture energy."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=None)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    assert r.read_energy_pj_per_bit is None
    assert r.write_energy_pj_per_bit is None
    assert r.read_dynamic_energy_j_per_token is None
    assert r.write_dynamic_energy_j_per_token is None
    assert r.memory_dynamic_energy_j_per_token is None
    assert r.read_energy_status == "NO_ARCHITECTURE_ENERGY_RESOLVED"
    assert r.write_energy_status == "NO_ARCHITECTURE_ENERGY_RESOLVED"
    assert r.evaluation_status == "NO_ARCHITECTURE_ENERGY_RESOLVED"


# ---------------------------------------------------------------------------
# 5. Capacity feasible + energy available → evaluated
# ---------------------------------------------------------------------------

def test_capacity_feasible_and_energy_available_evaluates() -> None:
    """Happy path: capacity feasible and energy available produces evaluated
    per-token memory dynamic energy in Joules."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=2.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    assert r.capacity_feasible is True
    assert r.evaluation_status == "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY"
    # 1 byte = 8 bits; read = 8 * 2 = 16 pJ = 16e-12 J
    assert r.read_dynamic_energy_j_per_token == 16e-12
    # write = 8 * (0.5 * 2) = 8 pJ = 8e-12 J
    assert r.write_dynamic_energy_j_per_token == 8e-12
    assert r.memory_dynamic_energy_j_per_token == 24e-12


# ---------------------------------------------------------------------------
# 6. Read/write energy separation via rho
# ---------------------------------------------------------------------------

def test_rho_scales_write_energy_only() -> None:
    """Changing rho must affect only write energy, leaving read energy
    determined solely by system.memory_access_energy_pJ_per_bit."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=2.0)
    r_low = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.25)
    r_high = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=1.0)
    # Read energy unchanged
    assert r_low.read_energy_pj_per_bit == r_high.read_energy_pj_per_bit == 2.0
    assert r_low.read_dynamic_energy_j_per_token == r_high.read_dynamic_energy_j_per_token
    # Write energy scales with rho
    assert r_low.write_energy_pj_per_bit == 0.5
    assert r_high.write_energy_pj_per_bit == 2.0
    assert r_low.write_dynamic_energy_j_per_token == 4e-12
    assert r_high.write_dynamic_energy_j_per_token == 16e-12


# ---------------------------------------------------------------------------
# 7. Zero rho → zero write contribution
# ---------------------------------------------------------------------------

def test_zero_rho_zeroes_write_contribution() -> None:
    """rho = 0 must produce zero write dynamic energy."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=2.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.0)
    assert r.write_dynamic_energy_j_per_token == 0.0
    assert r.memory_dynamic_energy_j_per_token == r.read_dynamic_energy_j_per_token


# ---------------------------------------------------------------------------
# 8. Energy closure
# ---------------------------------------------------------------------------

def test_energy_closure_read_plus_write_equals_total() -> None:
    """read_dynamic_energy_j_per_token + write_dynamic_energy_j_per_token
    must equal memory_dynamic_energy_j_per_token (within floating-point
    tolerance; the underlying pJ accounting closes exactly)."""
    workload = _build_workload_metrics(read_bytes_per_token=2.5, write_bytes_per_token=1.5)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=3.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    expected_total = r.read_dynamic_energy_j_per_token + r.write_dynamic_energy_j_per_token
    assert math.isclose(r.memory_dynamic_energy_j_per_token, expected_total, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 9. Traffic echo
# ---------------------------------------------------------------------------

def test_traffic_echoed_from_workload() -> None:
    """read_bytes_per_token and write_bytes_per_token must be echoed
    verbatim from the workload for self-containment."""
    workload = _build_workload_metrics(read_bytes_per_token=7.5, write_bytes_per_token=3.5)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=2.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    assert r.read_bytes_per_token == 7.5
    assert r.write_bytes_per_token == 3.5


# ---------------------------------------------------------------------------
# 10. Status provenance preservation
# ---------------------------------------------------------------------------

def test_status_provenance_preserved_when_evaluated() -> None:
    """All v0 provenance labels must be present when evaluation succeeds."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=2.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    assert r.read_energy_status == "CURRENT_NOMINAL_ANALYTICAL_MODEL"
    assert r.write_energy_status == "RHO_SENSITIVITY_NOT_PHYSICAL_CLAIM"
    assert r.energy_scope_status == "MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY"
    assert r.scenario_status == "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"
    assert r.zhu_transferability_status == "NOT_VALIDATED"


def test_status_provenance_preserved_when_capacity_infeasible() -> None:
    """Provenance labels must persist even when capacity gate is closed."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=False)
    system = _system(energy_pj_per_bit=2.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    assert r.energy_scope_status == "MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY"
    assert r.scenario_status == "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"
    assert r.zhu_transferability_status == "NOT_VALIDATED"


def test_status_provenance_preserved_when_no_architecture_energy() -> None:
    """Provenance labels must persist even when architecture energy is unavailable."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("test", feasible=True)
    system = _system(energy_pj_per_bit=None)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    assert r.energy_scope_status == "MEMORY_DYNAMIC_TRAFFIC_ENERGY_ONLY"
    assert r.scenario_status == "CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY"
    assert r.zhu_transferability_status == "NOT_VALIDATED"


# ---------------------------------------------------------------------------
# 11. Forbidden metrics not in output model
# ---------------------------------------------------------------------------

def test_output_model_has_no_forbidden_metrics() -> None:
    """The output model must NOT carry bandwidth, time, tokens/s, power,
    Tmax, or thermal fields.  This pins the public surface."""
    forbidden = (
        "bandwidth",
        "memory_time",
        "compute_time",
        "tokens_per_second",
        "power",
        "temperature",
        "tmax",
        "tmax_c",
        "tmax_k",
        "thermal",
        "watts",
        "j_per_token",  # we use memory_dynamic_energy_j_per_token instead
    )
    field_names = set(ArchitectureDecodeMemoryEnergyMetrics.model_fields.keys())
    for name in forbidden:
        assert name not in field_names, (
            f"ArchitectureDecodeMemoryEnergyMetrics must not expose {name!r}")


# ---------------------------------------------------------------------------
# 12. Architecture identity preserved
# ---------------------------------------------------------------------------

def test_architecture_identity_preserved() -> None:
    """The architecture name from the capacity gate must flow through to
    the output model unchanged."""
    workload = _build_workload_metrics(read_bytes_per_token=1.0, write_bytes_per_token=1.0)
    cap = _capacity_feasibility("orthogonal_m3d_igzo", feasible=True)
    system = _system(energy_pj_per_bit=2.0)
    r = evaluate_architecture_decode_memory_energy(workload, cap, system, rho=0.5)
    assert r.architecture == "orthogonal_m3d_igzo"


# ---------------------------------------------------------------------------
# 13. Frozen LLaMA-3.1-8B-class scenario across three architectures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def capacities() -> dict[str, ResolvedArchitectureCapacity]:
    return {name: _resolve_architecture_capacity(name) for name in ARCHITECTURES}


@pytest.fixture(scope="module")
def systems() -> dict[str, ResolvedSystemPower]:
    return {name: _resolve_system(name) for name in ARCHITECTURES}


def test_frozen_llama_scenario_evaluates_for_all_three_architectures(
    capacities: dict[str, ResolvedArchitectureCapacity],
    systems: dict[str, ResolvedSystemPower],
) -> None:
    """The frozen 8B-class workload under matched-reference bandwidth must
    produce conditional memory dynamic energy for every architecture that
    has resolved energy, and None for those without.
    """
    workload = _build_workload()
    feasibilities = [
        evaluate_architecture_capacity_feasibility(
            workload, capacities[name], reserved_capacity_bytes=0,
        )
        for name in ARCHITECTURES
    ]
    # All three architectures must be capacity-feasible for the 8B workload
    assert all(f.capacity_feasible for f in feasibilities)

    results = [
        evaluate_architecture_decode_memory_energy(
            workload, feasibilities[i], systems[name], rho=0.5,
        )
        for i, name in enumerate(ARCHITECTURES)
    ]

    for r in results:
        assert r.capacity_feasible is True
        if r.evaluation_status == "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY":
            # Energy must close (within floating-point tolerance)
            expected = r.read_dynamic_energy_j_per_token + r.write_dynamic_energy_j_per_token
            assert math.isclose(r.memory_dynamic_energy_j_per_token, expected, rel_tol=1e-12)
            # Must be positive
            assert r.memory_dynamic_energy_j_per_token > 0
        elif r.evaluation_status == "NO_ARCHITECTURE_ENERGY_RESOLVED":
            # Architecture has no resolved energy (e.g., reference_fixed)
            assert r.read_energy_pj_per_bit is None
            assert r.memory_dynamic_energy_j_per_token is None
        else:
            raise AssertionError(f"unexpected status: {r.evaluation_status}")


def test_architecture_with_resolved_energy_produces_nonzero_j_per_token(
    capacities: dict[str, ResolvedArchitectureCapacity],
    systems: dict[str, ResolvedSystemPower],
) -> None:
    """Architectures with resolved memory_access_energy_pJ_per_bit must
    produce a concrete J/token number when capacity is feasible."""
    workload = _build_workload()
    for name in ARCHITECTURES:
        cap = evaluate_architecture_capacity_feasibility(
            workload, capacities[name], reserved_capacity_bytes=0)
        system = systems[name]
        r = evaluate_architecture_decode_memory_energy(
            workload, cap, system, rho=0.5)
        if system.memory_access_energy_pJ_per_bit is not None:
            assert r.evaluation_status == "EVALUATED_CONDITIONAL_ARCHITECTURE_MEMORY_ENERGY"
            assert r.memory_dynamic_energy_j_per_token is not None
            assert r.memory_dynamic_energy_j_per_token > 0
