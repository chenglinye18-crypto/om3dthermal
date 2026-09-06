"""Host DDR + PCIe transport-boundary power closure."""

from __future__ import annotations

import csv
import inspect
from pathlib import Path

import pytest

from om3dthermal.platform import (
    HostOffloadSpec,
    resolve_host_offload_power,
)
from om3dthermal.provenance import ProvenanceRecord


BW_EFF = 56.2e9
E_PCIE = 167.9e-12
E_DDR = 4.93 / (24.94e9 * 8.0)
ROOT = Path(__file__).parents[1]


def _resolve(demand: float):
    return resolve_host_offload_power(
        host_transfer_demand_bytes_per_second=demand,
        host_effective_bandwidth_bytes_per_second=BW_EFF,
        e_pcie_dynamic_J_per_bit=E_PCIE,
        e_ddr_dynamic_J_per_bit=E_DDR,
    )


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (ProvenanceRecord(
        record_id="host_power_test",
        classification="NUMERICAL_CHOICE",
        source="unit test",
        status="SYNTHETIC_TEST_ONLY",
    ),)


def test_zero_traffic_has_zero_dynamic_power() -> None:
    point = _resolve(0.0)
    assert point.host_bandwidth_actual_bytes_per_second == 0.0
    assert point.pcie_dynamic_power_W == 0.0
    assert point.ddr_dynamic_power_W == 0.0
    assert point.host_offload_dynamic_power_W == 0.0
    assert point.host_bandwidth_saturated is False


def test_below_boundary_is_linear() -> None:
    point = _resolve(0.5 * BW_EFF)
    assert point.host_bandwidth_actual_bytes_per_second == 0.5 * BW_EFF
    assert point.host_offload_dynamic_power_W == pytest.approx(
        (E_PCIE + E_DDR) * 8.0 * 0.5 * BW_EFF)


def test_exact_boundary_is_continuous_and_not_flagged_saturated() -> None:
    boundary = _resolve(BW_EFF)
    left = _resolve(BW_EFF * (1.0 - 1e-12))
    assert boundary.host_bandwidth_actual_bytes_per_second == BW_EFF
    assert boundary.host_bandwidth_saturated is False
    assert left.host_offload_dynamic_power_W == pytest.approx(
        boundary.host_offload_dynamic_power_W, rel=2e-12)


def test_above_boundary_clamps_rate_and_power() -> None:
    boundary = _resolve(BW_EFF)
    above = _resolve(1.2 * BW_EFF)
    assert above.host_bandwidth_actual_bytes_per_second == BW_EFF
    assert above.host_bandwidth_saturated is True
    assert above.host_offload_dynamic_power_W == boundary.host_offload_dynamic_power_W


def test_full_bandwidth_component_hand_check() -> None:
    point = _resolve(BW_EFF)
    assert E_DDR * 1e12 == pytest.approx(24.709302325581394)
    assert (E_PCIE + E_DDR) * 1e12 == pytest.approx(192.6093023255814)
    assert point.pcie_dynamic_power_W == pytest.approx(75.48784)
    assert point.ddr_dynamic_power_W == pytest.approx(11.109302325581394)
    assert point.host_offload_dynamic_power_W == pytest.approx(86.5971423255814)


def test_schema_preserves_components_and_has_no_host_static_or_compute_branch() -> None:
    spec = HostOffloadSpec(
        status="RESOLVED",
        host_memory_bandwidth_GBps=460.8,
        host_device_link_bandwidth_GBps=64.0,
        host_offload_efficiency=0.878125,
        power_model_status="INCREMENTAL_DYNAMIC_OFFLOAD_POWER",
        e_pcie_dynamic_J_per_bit=E_PCIE,
        e_pcie_dynamic_uncertainty_J_per_bit=10.5e-12,
        e_ddr_dynamic_J_per_bit=E_DDR,
        provenance=_provenance(),
    )
    assert spec.e_host_offload_dynamic_J_per_bit == pytest.approx(E_PCIE + E_DDR)
    names = set(HostOffloadSpec.model_fields)
    names.update(inspect.signature(resolve_host_offload_power).parameters)
    assert not any("flop" in name.lower() or "compute" in name.lower()
                   for name in names)
    assert "host_static_power_W" not in names
    assert spec.host_static_power_status == "UNRESOLVED"


def test_host_offload_csv_preserves_components_and_provenance() -> None:
    path = ROOT / "docs" / "research" / "host_offload_power_table_2026-09-07.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {row["parameter"]: row for row in csv.DictReader(stream)}
    assert float(rows["e_pcie_dynamic"]["value"]) == 167.9
    assert float(rows["e_pcie_dynamic_uncertainty"]["value"]) == 10.5
    assert float(rows["e_ddr_dynamic"]["value"]) == pytest.approx(E_DDR * 1e12)
    assert float(rows["e_host_offload_dynamic"]["value"]) == pytest.approx(
        (E_PCIE + E_DDR) * 1e12)
    assert rows["e_pcie_dynamic"]["provenance_status"] == "PAPER_REPORTED"
    assert rows["e_ddr_dynamic"]["provenance_status"] == (
        "SOFTWARE_DERIVED_FROM_PAPER_REPRESENTATIVE_RUN")
    assert rows["host_static_power"]["value"] == "UNRESOLVED"
    assert all(row["provenance_status"] and row["source"] for row in rows.values())
