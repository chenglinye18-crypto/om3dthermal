"""Tests for the per-edge thermal conductance computation.

These tests build small hand-crafted cell / edge pairs and check the
closed-form two-point face conductance against analytic expectations.
The HBM benchmark has its own dedicated tests in
``test_conductance_benchmark.py``.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pytest

from om3dthermal.config import (
    InterfaceResistanceConfig,
    ThermalConductanceConfig,
)
from om3dthermal.discretization.models import (
    AdjacencyEdge, BoundaryFace, ThermalCell,
)
from om3dthermal.materials import Material
from om3dthermal.thermal import build_conductance_table
from om3dthermal.thermal.errors import (
    MissingThermalConductivityError,
    UnsupportedMaterialRotationError,
)


IDENTITY = ((1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell(*, id: int, ix: int = 0, iy: int = 0, iz: int = 0,
          x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
          material: str, rotation=IDENTITY,
          parent_box_name: str | None = None,
          tags: dict | None = None) -> ThermalCell:
    return ThermalCell(
        id=id, ix=ix, iy=iy, iz=iz,
        x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
        material=material,
        parent_box_id=f"box-{id}", parent_box_name=parent_box_name or f"box-{id}",
        component=None, source_path="tests/test_conductance.py",
        rotation=rotation, tags=tags or {},
    )


def _edge(*, id: int, cell_a: int, cell_b: int, axis: str,
          face_area: float, half_d_a: float, half_d_b: float,
          material_a: str, material_b: str) -> AdjacencyEdge:
    if axis == "x":
        interface_coordinate = 0.0  # not used by conductance
    elif axis == "y":
        interface_coordinate = 0.0
    else:
        interface_coordinate = 0.0
    return AdjacencyEdge(
        id=id, cell_a=cell_a, cell_b=cell_b, axis=axis,
        interface_coordinate=interface_coordinate,
        face_area=face_area, center_distance=half_d_a + half_d_b,
        half_distance_a=half_d_a, half_distance_b=half_d_b,
        material_a=material_a, material_b=material_b,
        is_material_interface=(material_a != material_b),
    )


def _material(name: str, kx: float, ky: float, kz: float) -> Material:
    return Material(name=name, k_local=(kx, ky, kz))


# ---------------------------------------------------------------------------
# D. Single material, two cells
# ---------------------------------------------------------------------------

def test_single_material_two_cells_along_x():
    mat = _material("Cu", 400.0, 400.0, 400.0)
    # Two 0.5 mm x 0.5 mm x 0.5 mm cells along x; face area = 0.5 * 0.5.
    cell_a = _cell(id=0, x0=0.0, x1=0.5e-3, y0=0.0, y1=0.5e-3,
                    z0=0.0, z1=0.5e-3, material="Cu")
    cell_b = _cell(id=1, x0=0.5e-3, x1=1.0e-3, y0=0.0, y1=0.5e-3,
                    z0=0.0, z1=0.5e-3, material="Cu")
    # face area = 0.5e-3 * 0.5e-3 = 2.5e-7 m^2; half_d_a = half_d_b = 0.25e-3
    face_area = 0.5e-3 * 0.5e-3
    half_d = 0.25e-3
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=face_area, half_d_a=half_d, half_d_b=half_d,
                  material_a="Cu", material_b="Cu")
    cfg = ThermalConductanceConfig.model_validate({})
    table = build_conductance_table(
        cells=[cell_a, cell_b], adjacency_edges=[edge],
        materials={"Cu": mat}, config=cfg,
    )
    # G = A / R_bulk, with R_bulk = d_a/k + d_b/k = 2 * half_d / k (m^2 K/W).
    # The table stores the *divided-by-A* resistance, so it is
    # R_table = R_bulk / A in K/W.
    R_bulk_m2K_W = 2.0 * half_d / 400.0
    expected_R_K_W = R_bulk_m2K_W / face_area
    expected_G = 1.0 / expected_R_K_W
    assert table.edge_count == 1
    assert table.conductance_W_K[0] == pytest.approx(expected_G)
    assert table.k_normal_a_W_mK[0] == pytest.approx(400.0)
    assert table.k_normal_b_W_mK[0] == pytest.approx(400.0)
    assert table.material_interface[0] == False  # noqa: E712
    assert table.resistance_K_W[0] == pytest.approx(expected_R_K_W)
    assert table.conductance_W_K[0] * table.resistance_K_W[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# E. Two materials in series
# ---------------------------------------------------------------------------

def test_two_materials_in_series():
    mat_a = _material("A", 10.0, 10.0, 10.0)
    mat_b = _material("B", 40.0, 40.0, 40.0)
    # Both cells are 0.5 mm wide along x; face area 1e-6.
    cell_a = _cell(id=0, x0=0.0, x1=0.5e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="A")
    cell_b = _cell(id=1, x0=0.5e-3, x1=1.0e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="B")
    face_area = 1.0e-6
    half_d = 0.25e-3
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=face_area, half_d_a=half_d, half_d_b=half_d,
                  material_a="A", material_b="B")
    cfg = ThermalConductanceConfig.model_validate({})
    table = build_conductance_table(
        cells=[cell_a, cell_b], adjacency_edges=[edge],
        materials={"A": mat_a, "B": mat_b}, config=cfg,
    )
    # R = half_d/k_a + half_d/k_b = 0.25e-3/10 + 0.25e-3/40 = 3.125e-5
    # G = A / R = 1e-6 / 3.125e-5 = 0.032
    expected_R = half_d / 10.0 + half_d / 40.0
    expected_G = face_area / expected_R
    assert table.conductance_W_K[0] == pytest.approx(expected_G)
    assert table.conductance_W_K[0] == pytest.approx(0.032)
    assert table.material_interface[0] == True  # noqa: E712


# ---------------------------------------------------------------------------
# F. Interface areal resistance
# ---------------------------------------------------------------------------

def test_interface_resistance_in_series():
    mat_a = _material("A", 10.0, 10.0, 10.0)
    mat_b = _material("B", 10.0, 10.0, 10.0)
    cell_a = _cell(id=0, x0=0.0, x1=0.5e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="A")
    cell_b = _cell(id=1, x0=0.5e-3, x1=1.0e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="B")
    face_area = 1.0e-6
    half_d = 0.25e-3
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=face_area, half_d_a=half_d, half_d_b=half_d,
                  material_a="A", material_b="B")
    rule = InterfaceResistanceConfig.model_validate({
        "materials": ["A", "B"],
        "areal_resistance": "1e-6 m^2*K/W",
    })
    cfg = ThermalConductanceConfig.model_validate({
        "default_interface_areal_resistance": 0.0,
        "interfaces": [rule.model_dump()],
    })
    table = build_conductance_table(
        cells=[cell_a, cell_b], adjacency_edges=[edge],
        materials={"A": mat_a, "B": mat_b}, config=cfg,
    )
    # R_total (m^2*K/W) = d_a/k + R'' + d_b/k
    #                  = 2 * 0.25e-3 / 10 + 1e-6 = 5e-5 + 1e-6 = 5.1e-5
    # The table column is in K/W, so R_table = R_total / A.
    R_total_m2K_W = 2.0 * half_d / 10.0 + 1e-6
    expected_R_K_W = R_total_m2K_W / face_area
    expected_G = 1.0 / expected_R_K_W
    assert table.conductance_W_K[0] == pytest.approx(expected_G)
    assert table.interface_areal_resistance_m2K_W[0] == pytest.approx(1e-6)
    assert table.interface_rule_index[0] == 0
    # Confirm R_table is K/W. The R_total m^2*K/W splits as
    # (R'' + d_a/k_a + d_b/k_b); the table column is R_total / A.
    r_interface = table.interface_areal_resistance_m2K_W[0] / face_area
    r_bulk = (table.half_distance_a_m[0] / table.k_normal_a_W_mK[0]
              + table.half_distance_b_m[0] / table.k_normal_b_W_mK[0]) / face_area
    assert r_bulk + r_interface == pytest.approx(
        table.resistance_K_W[0])


# ---------------------------------------------------------------------------
# G. Symmetry
# ---------------------------------------------------------------------------

def test_swapping_a_b_does_not_change_G():
    mat_a = _material("A", 10.0, 10.0, 10.0)
    mat_b = _material("B", 40.0, 40.0, 40.0)
    cell_a = _cell(id=0, x0=0.0, x1=0.5e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="A")
    cell_b = _cell(id=1, x0=0.5e-3, x1=1.0e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="B")
    face_area = 1.0e-6
    half_d_a = 0.25e-3
    half_d_b = 0.5e-3  # asymmetric cell widths
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=face_area, half_d_a=half_d_a, half_d_b=half_d_b,
                  material_a="A", material_b="B")
    cfg = ThermalConductanceConfig.model_validate({})
    table1 = build_conductance_table(
        cells=[cell_a, cell_b], adjacency_edges=[edge],
        materials={"A": mat_a, "B": mat_b}, config=cfg,
    )
    # Swap: cell_a is now the bigger cell, cell_b is the smaller one.
    edge_swapped = _edge(id=0, cell_a=1, cell_b=0, axis="x",
                          face_area=face_area,
                          half_d_a=half_d_b, half_d_b=half_d_a,
                          material_a="B", material_b="A")
    table2 = build_conductance_table(
        cells=[cell_b, cell_a], adjacency_edges=[edge_swapped],
        materials={"A": mat_a, "B": mat_b}, config=cfg,
    )
    assert table1.conductance_W_K[0] == pytest.approx(
        table2.conductance_W_K[0])
    assert table1.resistance_K_W[0] == pytest.approx(
        table2.resistance_K_W[0])


# ---------------------------------------------------------------------------
# H. Extreme materials / thin cells / large R''
# ---------------------------------------------------------------------------

def test_extreme_conductivity_ratio_does_not_blow_up():
    mat_high = _material("H", 1000.0, 1000.0, 1000.0)
    mat_low = _material("L", 0.05, 0.05, 0.05)
    cell_h = _cell(id=0, x0=0.0, x1=0.5e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="H")
    cell_l = _cell(id=1, x0=0.5e-3, x1=1.0e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="L")
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=1.0e-6, half_d_a=0.25e-3, half_d_b=0.25e-3,
                  material_a="H", material_b="L")
    cfg = ThermalConductanceConfig.model_validate({})
    table = build_conductance_table(
        cells=[cell_h, cell_l], adjacency_edges=[edge],
        materials={"H": mat_high, "L": mat_low}, config=cfg,
    )
    G = table.conductance_W_K[0]
    assert math.isfinite(G) and G > 0


def test_very_thin_cell_remains_finite():
    mat = _material("M", 100.0, 100.0, 100.0)
    cell_thin = _cell(id=0, x0=0.0, x1=0.15e-6, y0=0.0, y1=1.0e-3,
                       z0=0.0, z1=1.0e-3, material="M")
    cell_fat = _cell(id=1, x0=0.15e-6, x1=1.0e-3, y0=0.0, y1=1.0e-3,
                      z0=0.0, z1=1.0e-3, material="M")
    face_area = 1.0e-6
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=face_area,
                  half_d_a=0.075e-6, half_d_b=(1.0e-3 - 0.15e-6) / 2,
                  material_a="M", material_b="M")
    cfg = ThermalConductanceConfig.model_validate({})
    table = build_conductance_table(
        cells=[cell_thin, cell_fat], adjacency_edges=[edge],
        materials={"M": mat}, config=cfg,
    )
    G = table.conductance_W_K[0]
    assert math.isfinite(G) and G > 0


def test_very_large_interface_resistance_remains_finite():
    mat = _material("M", 10.0, 10.0, 10.0)
    cell_a = _cell(id=0, x0=0.0, x1=0.5e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="M")
    cell_b = _cell(id=1, x0=0.5e-3, x1=1.0e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="M")
    rule = InterfaceResistanceConfig.model_validate({
        "materials": ["M", "M"],
        "areal_resistance": "1e-2 m^2*K/W",
    })
    cfg = ThermalConductanceConfig.model_validate({
        "default_interface_areal_resistance": 0.0,
        "interfaces": [rule.model_dump()],
    })
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=1.0e-6, half_d_a=0.25e-3, half_d_b=0.25e-3,
                  material_a="M", material_b="M")
    table = build_conductance_table(
        cells=[cell_a, cell_b], adjacency_edges=[edge],
        materials={"M": mat}, config=cfg,
    )
    G = table.conductance_W_K[0]
    R = table.resistance_K_W[0]
    assert math.isfinite(G) and G > 0
    assert math.isfinite(R) and R > 0
    # R is dominated by the interface; G should be tiny.
    assert G < 1e-3


# ---------------------------------------------------------------------------
# I. Invalid input
# ---------------------------------------------------------------------------

def test_missing_k_local_raises_missing_thermal_conductivity_error():
    mat_no_k = Material(name="M", k_local=None)
    cell_a = _cell(id=0, x0=0.0, x1=0.5e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="M")
    cell_b = _cell(id=1, x0=0.5e-3, x1=1.0e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="M")
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=1.0e-6, half_d_a=0.25e-3, half_d_b=0.25e-3,
                  material_a="M", material_b="M")
    cfg = ThermalConductanceConfig.model_validate({})
    with pytest.raises(MissingThermalConductivityError) as excinfo:
        build_conductance_table(
            cells=[cell_a, cell_b], adjacency_edges=[edge],
            materials={"M": mat_no_k}, config=cfg,
        )
    assert excinfo.value.material == "M"
    assert excinfo.value.cell_id in (0, 1)
    assert excinfo.value.edge_id == 0


def test_unsupported_rotation_raises():
    mat = _material("M", 1.0, 2.0, 3.0)
    a = math.radians(45)
    R_45 = ((1.0, 0.0, 0.0),
            (0.0, math.cos(a), -math.sin(a)),
            (0.0, math.sin(a),  math.cos(a)))
    cell_a = _cell(id=0, x0=0.0, x1=0.5e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="M", rotation=R_45)
    cell_b = _cell(id=1, x0=0.5e-3, x1=1.0e-3, y0=0.0, y1=1.0e-3,
                    z0=0.0, z1=1.0e-3, material="M")
    edge = _edge(id=0, cell_a=0, cell_b=1, axis="x",
                  face_area=1.0e-6, half_d_a=0.25e-3, half_d_b=0.25e-3,
                  material_a="M", material_b="M")
    cfg = ThermalConductanceConfig.model_validate({})
    with pytest.raises(UnsupportedMaterialRotationError):
        build_conductance_table(
            cells=[cell_a, cell_b], adjacency_edges=[edge],
            materials={"M": mat}, config=cfg,
        )


# ---------------------------------------------------------------------------
# J. 1-D mesh independence: serial resistance from first to last cell
# ---------------------------------------------------------------------------

def _1d_series(n_cells: int, *, length: float = 1.0e-3,
                area: float = 1.0e-6, k: float = 10.0
                ) -> tuple[list[ThermalCell], list[AdjacencyEdge]]:
    """Build a 1-D series of n_cells cells along x and the n_cells-1
    internal adjacency edges between consecutive cells.
    """
    cells: list[ThermalCell] = []
    edges: list[AdjacencyEdge] = []
    dx = length / n_cells
    for i in range(n_cells):
        x0 = i * dx
        x1 = (i + 1) * dx
        cells.append(_cell(
            id=i, ix=i, iy=0, iz=0,
            x0=x0, x1=x1, y0=0.0, y1=area, z0=0.0, z1=1.0,
            material="M",
        ))
    for i in range(n_cells - 1):
        edges.append(_edge(
            id=i, cell_a=i, cell_b=i + 1, axis="x",
            face_area=area, half_d_a=dx / 2, half_d_b=dx / 2,
            material_a="M", material_b="M",
        ))
    return cells, edges


def test_1d_series_internal_conductance_matches_centre_to_centre():
    """The series resistance of the internal edges is the bulk
    resistance between the first cell centre and the last cell centre.

    With ``n`` uniform cells of width ``L/n`` and material ``k``, the
    first cell centre sits at ``L/(2n)`` and the last cell centre
    sits at ``L - L/(2n)`` so the centre-to-centre distance is
    ``(n-1) L / n``. The expected internal resistance is

        R_expected = (n - 1) * L / n / (k * A)

    which grows with ``n`` and asymptotes to ``L / (k * A)`` for
    large ``n``. The conductance build must reproduce this exactly
    for every uniform subdivision.
    """
    mat = _material("M", 10.0, 10.0, 10.0)
    cfg = ThermalConductanceConfig.model_validate({})
    L = 1.0e-3
    A = 1.0e-6
    k = 10.0
    for n in (2, 5, 10, 20):
        cells, edges = _1d_series(n, length=L, area=A, k=k)
        table = build_conductance_table(
            cells=cells, adjacency_edges=edges,
            materials={"M": mat}, config=cfg,
        )
        expected_R = (n - 1) * L / n / (k * A)
        R_eq = float(np.sum(1.0 / table.conductance_W_K))
        assert R_eq == pytest.approx(expected_R, rel=1e-9, abs=1e-18), (
            f"subdivision {n} gave R_eq={R_eq} != {expected_R}")


def test_1d_series_non_uniform_subdivision_also_invariant():
    """A non-uniform subdivision must also preserve the centre-to-centre
    series resistance; the series path is exactly first cell centre
    to last cell centre.
    """
    mat = _material("M", 10.0, 10.0, 10.0)
    cfg = ThermalConductanceConfig.model_validate({})
    # 4 cells: 100, 200, 300, 400 um wide (sum = 1 mm).
    # First cell centre at x = 50 um, last cell centre at x = 800 um
    # (cell 3 spans 600..1000 um). Centre-to-centre distance is
    # therefore 750 um; R = 750e-6 / (10 * 1e-6) = 75 K/W.
    widths_um = [100, 200, 300, 400]
    cells: list[ThermalCell] = []
    edges: list[AdjacencyEdge] = []
    x = 0.0
    for i, w_um in enumerate(widths_um):
        w = w_um * 1e-6
        cells.append(_cell(
            id=i, ix=i, iy=0, iz=0,
            x0=x, x1=x + w, y0=0.0, y1=1.0e-6, z0=0.0, z1=1.0,
            material="M",
        ))
        x += w
    for i in range(len(widths_um) - 1):
        edges.append(_edge(
            id=i, cell_a=i, cell_b=i + 1, axis="x",
            face_area=1.0e-6,
            half_d_a=widths_um[i] * 1e-6 / 2,
            half_d_b=widths_um[i + 1] * 1e-6 / 2,
            material_a="M", material_b="M",
        ))
    table = build_conductance_table(
        cells=cells, adjacency_edges=edges,
        materials={"M": mat}, config=cfg,
    )
    R_eq = float(np.sum(1.0 / table.conductance_W_K))
    assert R_eq == pytest.approx(75.0, rel=1e-9, abs=1e-18)


# ---------------------------------------------------------------------------
# K. Columnar ConductanceTable invariants
# ---------------------------------------------------------------------------

def test_conductance_table_arrays_have_uniform_length():
    mat_a = _material("A", 1.0, 1.0, 1.0)
    mat_b = _material("B", 2.0, 2.0, 2.0)
    cells = []
    edges = []
    for i in range(3):
        cells.append(_cell(
            id=i, ix=i, iy=0, iz=0,
            x0=i * 1e-3, x1=(i + 1) * 1e-3, y0=0.0, y1=1.0e-3,
            z0=0.0, z1=1.0e-3,
            material="A" if i % 2 == 0 else "B",
        ))
    for i in range(2):
        edges.append(_edge(
            id=i, cell_a=i, cell_b=i + 1, axis="x",
            face_area=1.0e-6, half_d_a=0.5e-3, half_d_b=0.5e-3,
            material_a=cells[i].material, material_b=cells[i + 1].material,
        ))
    cfg = ThermalConductanceConfig.model_validate({})
    table = build_conductance_table(
        cells=cells, adjacency_edges=edges,
        materials={"A": mat_a, "B": mat_b}, config=cfg,
    )
    n = table.edge_count
    assert table.edge_id.shape == (n,)
    assert table.cell_a.shape == (n,)
    assert table.cell_b.shape == (n,)
    assert table.axis.shape == (n,)
    assert table.face_area_m2.shape == (n,)
    assert table.half_distance_a_m.shape == (n,)
    assert table.half_distance_b_m.shape == (n,)
    assert table.k_normal_a_W_mK.shape == (n,)
    assert table.k_normal_b_W_mK.shape == (n,)
    assert table.interface_areal_resistance_m2K_W.shape == (n,)
    assert table.resistance_K_W.shape == (n,)
    assert table.conductance_W_K.shape == (n,)
    assert table.material_interface.shape == (n,)
    assert table.interface_rule_index.shape == (n,)
    # dtypes
    assert table.edge_id.dtype == np.int64
    assert table.cell_a.dtype == np.int64
    assert table.cell_b.dtype == np.int64
    assert table.axis.dtype == np.int8
    assert table.face_area_m2.dtype == np.float64
    assert table.resistance_K_W.dtype == np.float64
    assert table.conductance_W_K.dtype == np.float64
    assert table.material_interface.dtype == bool
    assert table.interface_rule_index.dtype == np.int32
