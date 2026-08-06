"""Integration tests for the conductance CLI on the shipped HBM benchmark.

These tests load the shipped ``hbm_on_gpu_12hi.yaml``, run the
discretiser, and then build the :class:`ConductanceTable`. They
verify the geometric and physical invariants a future KCL solver
will rely on:

- the conductance edge count matches the adjacency edge count;
- every ``G`` is finite and strictly positive;
- all ``R''`` are ``0`` (the shipped benchmark does not introduce
  contact resistance);
- the anisotropic materials (BSPDN, GPU_HBM_uBump, Cu_Pillar_Bump)
  report the expected ``k_n`` per axis;
- the cache entry count is much smaller than the edge count;
- the NPZ round-trips with correct dtypes / lengths.

The tests intentionally do not hard-code exact ``G`` min / max values
— those depend on the discretiser and would couple the suite to
internal choices. They only require strictly positive / finite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from om3dthermal.config import load_config
from om3dthermal.discretization import (
    build_adjacency,
    build_boundary_faces,
    build_global_grid,
    generate_cells,
)
from om3dthermal.thermal import build_conductance_table
from om3dthermal.geometry.horizontal_columns import HorizontalColumnsBuilder


CONFIG = Path(__file__).parents[1] / "configs" / "hbm_on_gpu_12hi.yaml"


@pytest.fixture(scope="module")
def conductance_scene():
    """Discretise and build the conductance table once for the
    shipped benchmark.
    """
    cfg = load_config(CONFIG)
    assert cfg.discretization is not None
    assert cfg.thermal_conductance is not None
    scene = HorizontalColumnsBuilder(cfg).build()
    boxes = list(scene.boxes)
    grid = build_global_grid(boxes, cfg.discretization.max_cell_size)
    cells = generate_cells(boxes, grid)
    edges = build_adjacency(cells, grid)
    boundary_faces = build_boundary_faces(cells, grid)
    table = build_conductance_table(
        cells=cells, adjacency_edges=edges,
        materials=cfg.materials,
        config=cfg.thermal_conductance,
    )
    return {
        "config": cfg,
        "cells": cells,
        "edges": edges,
        "boundary_faces": boundary_faces,
        "table": table,
    }


# ---------------------------------------------------------------------------
# Edge count and positivity
# ---------------------------------------------------------------------------

def test_conductance_edge_count_matches_adjacency(conductance_scene):
    edges = conductance_scene["edges"]
    table = conductance_scene["table"]
    assert table.edge_count == len(edges)


def test_every_conductance_is_strictly_positive_and_finite(conductance_scene):
    table = conductance_scene["table"]
    G = table.conductance_W_K
    assert np.all(np.isfinite(G))
    assert np.all(G > 0)


def test_every_resistance_is_strictly_positive_and_finite(conductance_scene):
    table = conductance_scene["table"]
    R = table.resistance_K_W
    assert np.all(np.isfinite(R))
    assert np.all(R > 0)


def test_every_k_normal_is_strictly_positive_and_finite(conductance_scene):
    table = conductance_scene["table"]
    k_a = table.k_normal_a_W_mK
    k_b = table.k_normal_b_W_mK
    assert np.all(np.isfinite(k_a)) and np.all(k_a > 0)
    assert np.all(np.isfinite(k_b)) and np.all(k_b > 0)


# ---------------------------------------------------------------------------
# Default interface R'' is 0 for the shipped benchmark
# ---------------------------------------------------------------------------

def test_default_interface_resistance_is_zero(conductance_scene):
    cfg = conductance_scene["config"]
    assert cfg.thermal_conductance.default_interface_areal_resistance == 0.0


def test_no_nonzero_interface_resistance_edges_in_benchmark(conductance_scene):
    table = conductance_scene["table"]
    assert table.nonzero_interface_resistance_count == 0
    assert np.all(table.interface_areal_resistance_m2K_W == 0.0)
    # All rule indices are -1 (default rule).
    assert np.all(table.interface_rule_index == -1)


# ---------------------------------------------------------------------------
# Axis coverage and material-interface edge count
# ---------------------------------------------------------------------------

def test_all_three_axes_have_edges(conductance_scene):
    table = conductance_scene["table"]
    axis_codes = set(int(a) for a in table.axis)
    assert axis_codes == {0, 1, 2}


def test_material_interface_edge_count_matches_adjacency_summary(
        conductance_scene):
    table = conductance_scene["table"]
    iface = int(np.count_nonzero(table.material_interface))
    # The discretisation summary reported this number; we recompute it
    # from the same source data so the test is independent of any
    # JSON summary value.
    edges = conductance_scene["edges"]
    cell_by_id = {c.id: c for c in conductance_scene["cells"]}
    expected = sum(
        1 for e in edges if cell_by_id[e.cell_a].material
        != cell_by_id[e.cell_b].material)
    assert iface == expected
    assert iface > 0


# ---------------------------------------------------------------------------
# Material-specific k_n values
# ---------------------------------------------------------------------------

def test_silicon_is_isotropic_140_in_all_directions(conductance_scene):
    table = conductance_scene["table"]
    cell_by_id = {c.id: c for c in conductance_scene["cells"]}
    for i in range(table.edge_count):
        ca = cell_by_id[int(table.cell_a[i])]
        cb = cell_by_id[int(table.cell_b[i])]
        if ca.material == "Silicon":
            assert table.k_normal_a_W_mK[i] == pytest.approx(140.0)
        if cb.material == "Silicon":
            assert table.k_normal_b_W_mK[i] == pytest.approx(140.0)


def _check_anisotropic_material(conductance_scene, material: str,
                                expected_xy: float, expected_z: float
                                ) -> None:
    table = conductance_scene["table"]
    cell_by_id = {c.id: c for c in conductance_scene["cells"]}
    for i in range(table.edge_count):
        ca = cell_by_id[int(table.cell_a[i])]
        cb = cell_by_id[int(table.cell_b[i])]
        axis = int(table.axis[i])
        for cell_material, k_arr in (
            (ca.material, table.k_normal_a_W_mK),
            (cb.material, table.k_normal_b_W_mK),
        ):
            if cell_material != material:
                continue
            if axis == 2:  # z
                assert k_arr[i] == pytest.approx(expected_z)
            else:          # x or y
                assert k_arr[i] == pytest.approx(expected_xy)


def test_bspdn_k_n_83_xy_71_z(conductance_scene):
    _check_anisotropic_material(conductance_scene, "BSPDN", 83.0, 71.0)


def test_gpu_hbm_ubump_k_n_059_xy_1928_z(conductance_scene):
    _check_anisotropic_material(conductance_scene, "GPU_HBM_uBump", 0.59, 19.28)


def test_cu_pillar_bump_k_n_054_xy_1325_z(conductance_scene):
    _check_anisotropic_material(conductance_scene, "Cu_Pillar_Bump", 0.54, 13.25)


# ---------------------------------------------------------------------------
# Cache and material counts
# ---------------------------------------------------------------------------

def test_cache_entries_much_smaller_than_edge_count(conductance_scene):
    from om3dthermal.thermal.tensors import canonical_rotation_key
    cells = conductance_scene["cells"]
    cache_keys: set[tuple[str, tuple[int, ...], int]] = set()
    for cell in cells:
        rot_key = canonical_rotation_key(cell.rotation)
        for axis_int in (0, 1, 2):
            cache_keys.add((cell.material, rot_key, axis_int))
    # 15 materials * 3 axes = 45 entries max; well below edge count.
    assert len(cache_keys) < 100
    assert len(cache_keys) < conductance_scene["table"].edge_count // 100


def test_unique_material_count_is_15(conductance_scene):
    cells = conductance_scene["cells"]
    materials = {c.material for c in cells}
    assert len(materials) == 15


# ---------------------------------------------------------------------------
# NPZ round-trip
# ---------------------------------------------------------------------------

def test_npz_round_trip(tmp_path, conductance_scene):
    from om3dthermal.thermal.export import write_conductance_npz
    table = conductance_scene["table"]
    npz_path = tmp_path / "conductance.npz"
    write_conductance_npz(table, npz_path)
    data = np.load(npz_path)
    expected_keys = {
        "edge_id", "cell_a", "cell_b", "axis",
        "face_area_m2", "half_distance_a_m", "half_distance_b_m",
        "k_normal_a_W_mK", "k_normal_b_W_mK",
        "interface_areal_resistance_m2K_W",
        "resistance_K_W", "conductance_W_K",
        "material_interface", "interface_rule_index",
    }
    assert set(data.files) == expected_keys
    for key in expected_keys:
        arr = data[key]
        assert arr.shape[0] == table.edge_count
    assert data["edge_id"].dtype == np.int64
    assert data["axis"].dtype == np.int8
    assert data["face_area_m2"].dtype == np.float64
    assert data["material_interface"].dtype == bool
    assert data["interface_rule_index"].dtype == np.int32
    # Spot-check: conductance is finite and positive when reloaded.
    G = data["conductance_W_K"]
    assert np.all(np.isfinite(G))
    assert np.all(G > 0)
