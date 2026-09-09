"""Closure and isolation tests for the generated research ledger."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    resolve_case_geometry,
)
from scripts.generate_e2e_bandwidth_energy_ledger import (
    LEDGER_FILENAME,
    build_ledger_rows,
    write_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/research" / LEDGER_FILENAME
REQUIRED = {
    "HOST_MEMORY_SUBSYSTEM", "HOST_COHERENT_LINK", "HOST_OFFLOAD_EFFECTIVE",
    "HOST_OFFLOAD_DYNAMIC_PATH", "CONVENTIONAL_HBM_READ", "M3D_INTERNAL",
    "M3D_MAT_LOCAL_READ", "M3D_GLOBAL_ROUTING", "M3D_MIV",
    "M3D_FEOL_ROUTE", "M3D_CONTACTLESS_INTERFACE", "M3D_TOTAL_READ",
    "GPU_MEMORY_INTERFACE", "GPU_SUSTAINED_BANDWIDTH_SERVICE",
    "GPU_DECODE_DYNAMIC", "M3D_GPU_SHARED_TRANSFER",
}


@pytest.fixture(scope="module")
def rows():
    values = build_ledger_rows(ROOT)
    return {row.component_id: row for row in values}


def test_required_ids_are_unique(rows) -> None:
    assert set(rows) == REQUIRED
    assert len(rows) == len(REQUIRED)


def test_generator_is_deterministic_and_checked_in_csv_is_current(tmp_path) -> None:
    first = write_ledger(ROOT, tmp_path / "first.csv")
    second = write_ledger(ROOT, tmp_path / "second.csv")
    assert first.read_bytes() == second.read_bytes() == LEDGER.read_bytes()


def test_m3d_energy_and_bandwidth_closure(rows) -> None:
    component_ids = (
        "M3D_MAT_LOCAL_READ", "M3D_GLOBAL_ROUTING", "M3D_MIV",
        "M3D_FEOL_ROUTE", "M3D_CONTACTLESS_INTERFACE",
    )
    assert sum(rows[name].energy_nominal_pJ_per_bit for name in component_ids) == pytest.approx(
        rows["M3D_TOTAL_READ"].energy_nominal_pJ_per_bit, rel=1e-13)
    assert rows["M3D_TOTAL_READ"].bandwidth_nominal_GBps == pytest.approx(min(
        rows["M3D_INTERNAL"].bandwidth_nominal_GBps,
        rows["M3D_CONTACTLESS_INTERFACE"].bandwidth_nominal_GBps,
    ))


def test_shared_transfer_closure(rows) -> None:
    shared = rows["M3D_GPU_SHARED_TRANSFER"]
    assert shared.bandwidth_nominal_GBps == pytest.approx(min(
        shared.bandwidth_demand_GBps,
        rows["M3D_TOTAL_READ"].bandwidth_nominal_GBps,
        rows["GPU_MEMORY_INTERFACE"].bandwidth_nominal_GBps,
    ))
    assert shared.bandwidth_status == "GPU"


def test_host_bandwidth_and_energy_closure(rows) -> None:
    effective = rows["HOST_OFFLOAD_EFFECTIVE"]
    assert effective.bandwidth_nominal_GBps == pytest.approx(416.34)
    assert effective.bandwidth_efficiency is None
    assert effective.bandwidth_nominal_GBps < min(
        rows["HOST_MEMORY_SUBSYSTEM"].bandwidth_nominal_GBps,
        rows["HOST_COHERENT_LINK"].bandwidth_nominal_GBps)
    assert rows["HOST_OFFLOAD_DYNAMIC_PATH"].energy_nominal_pJ_per_bit == pytest.approx(
        rows["HOST_MEMORY_SUBSYSTEM"].energy_nominal_pJ_per_bit
        + rows["HOST_COHERENT_LINK"].energy_nominal_pJ_per_bit)
    assert rows["HOST_OFFLOAD_DYNAMIC_PATH"].energy_nominal_pJ_per_bit == pytest.approx(5.3)


def test_gpu_energy_and_direct_service_match_canonical_sources(rows) -> None:
    gpu = rows["GPU_DECODE_DYNAMIC"]
    assert gpu.energy_min_pJ_per_bit is None
    assert gpu.energy_nominal_pJ_per_bit == pytest.approx(11.68)
    assert gpu.energy_max_pJ_per_bit is None
    service = rows["GPU_SUSTAINED_BANDWIDTH_SERVICE"]
    assert service.bandwidth_max_GBps == pytest.approx(4800.0)
    assert service.bandwidth_efficiency == pytest.approx(1.0)
    assert service.bandwidth_nominal_GBps == pytest.approx(4800.0)


def test_blank_is_not_zero_and_hbm_is_resolver_derived(rows) -> None:
    with LEDGER.open("r", encoding="utf-8", newline="") as stream:
        csv_rows = {row["component_id"]: row for row in csv.DictReader(stream)}
    assert csv_rows["GPU_MEMORY_INTERFACE"]["energy_nominal_pJ_per_bit"] == ""
    assert csv_rows["M3D_INTERNAL"]["energy_nominal_pJ_per_bit"] == ""
    assert csv_rows["HOST_OFFLOAD_DYNAMIC_PATH"]["bandwidth_nominal_GBps"] == ""
    hbm = rows["CONVENTIONAL_HBM_READ"]
    assert math.isfinite(hbm.energy_nominal_pJ_per_bit)
    assert "calculate_memory_power" in hbm.runtime_source
    case = load_case_config(ROOT / "configs/cases/conventional_hbm_2x1.yaml")
    resolved = calculate_memory_power(
        case,
        project_root=ROOT,
        read_bandwidth_gbps=case.workload.read_bandwidth_gbps,
        geometry=resolve_case_geometry(case),
    )
    assert hbm.energy_nominal_pJ_per_bit == resolved.E_access_total_pj_bit


def test_generator_contains_no_handwritten_physics_values() -> None:
    source = (ROOT / "scripts/generate_e2e_bandwidth_energy_ledger.py").read_text(
        encoding="utf-8")
    for forbidden in (
        "5300", "4800", "0.855260", "11.68", "167.9", "24.7093",
        "56.2",
    ):
        assert forbidden not in source


def test_runtime_does_not_reference_generated_ledger() -> None:
    needle = LEDGER_FILENAME
    for directory in (ROOT / "src", ROOT / "configs"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                assert needle not in path.read_text(encoding="utf-8")
