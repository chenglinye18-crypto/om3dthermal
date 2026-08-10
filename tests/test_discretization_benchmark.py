"""Integration tests for the discretiser on the shipped HBM benchmark.

These tests build the full canonical conventional 2x2 scene and run the
discretiser end-to-end. They assert the geometric invariants that the
thermal solver will rely on: volume conservation, max cell sizes, all
material boundaries survive the cut merging, the face adjacency contains
at least one material interface, the boundary inventory contains both
classifications, and the surface area audit passes.

The tests intentionally do **not** hard-code the exact total cell or
edge counts; those depend on the implementation details of the
discretiser and would couple the test suite to internal choices. They
only require strictly positive counts and the conservation / max-size
invariants the solver needs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from om3dthermal.config import load_config
from om3dthermal.discretization import (
    build_adjacency,
    build_boundary_faces,
    build_global_grid,
    generate_cells,
    validate_cell_surface_partition,
    validate_volume_conservation,
)
from om3dthermal.geometry.horizontal_columns import HorizontalColumnsBuilder


CONFIG = Path(__file__).parents[1] / "configs" / "exp_conv_2x2_g414_m160.yaml"


# A reasonable tolerance for asserting "max cell size <= configured".
_SIZE_TOL = 1e-9


@pytest.fixture(scope="module")
def discretised_scene():
    """Discretise the shipped benchmark once and share across tests."""
    cfg = load_config(CONFIG)
    assert cfg.discretization is not None, (
        "shipped benchmark is expected to declare a 'discretization' block")
    scene = HorizontalColumnsBuilder(cfg).build()
    boxes = list(scene.boxes)
    grid = build_global_grid(boxes, cfg.discretization.max_cell_size)
    cells = generate_cells(boxes, grid)
    edges = build_adjacency(cells, grid)
    boundary_faces = build_boundary_faces(cells, grid)
    return {
        "config": cfg,
        "scene": scene,
        "boxes": boxes,
        "grid": grid,
        "cells": cells,
        "edges": edges,
        "boundary_faces": boundary_faces,
    }


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

def test_per_box_volume_conservation_holds(discretised_scene):
    boxes = discretised_scene["boxes"]
    cells = discretised_scene["cells"]
    # Should not raise.
    validate_volume_conservation(cells, boxes)


def test_total_cell_volume_matches_total_box_volume(discretised_scene):
    boxes = discretised_scene["boxes"]
    cells = discretised_scene["cells"]
    total_boxes = sum(
        (b.x1 - b.x0) * (b.y1 - b.y0) * (b.z1 - b.z0) for b in boxes)
    total_cells = sum(c.volume for c in cells)
    assert total_cells == pytest.approx(total_boxes, rel=1e-9, abs=1e-24)


def test_per_cell_surface_partition_audit_passes(discretised_scene):
    cells = discretised_scene["cells"]
    edges = discretised_scene["edges"]
    boundary_faces = discretised_scene["boundary_faces"]
    # Should not raise.
    validate_cell_surface_partition(cells, edges, boundary_faces)


# ---------------------------------------------------------------------------
# Grid index uniqueness
# ---------------------------------------------------------------------------

def test_no_duplicate_grid_indices(discretised_scene):
    cells = discretised_scene["cells"]
    keys = [(c.ix, c.iy, c.iz) for c in cells]
    assert len(set(keys)) == len(keys), "duplicate grid index in discretised cells"


# ---------------------------------------------------------------------------
# Maximum cell size constraints (relative to configured max_cell_size)
# ---------------------------------------------------------------------------

def test_x_and_y_max_cell_size_within_configured_limit(discretised_scene):
    cells = discretised_scene["cells"]
    cfg = discretised_scene["config"]
    max_x = cfg.discretization.max_cell_size.x
    max_y = cfg.discretization.max_cell_size.y
    # Use the global max_cell_size plus a small tolerance for accumulated
    # floating-point noise. A single box's max is enforced by
    # ``subdivide_interval``; the tolerance here is a sanity net, not a
    # fudge factor.
    assert max(c.size_x for c in cells) <= max_x + _SIZE_TOL
    assert max(c.size_y for c in cells) <= max_y + _SIZE_TOL


def test_z_max_cell_size_within_configured_limit(discretised_scene):
    cells = discretised_scene["cells"]
    cfg = discretised_scene["config"]
    max_z = cfg.discretization.max_cell_size.z
    assert max(c.size_z for c in cells) <= max_z + _SIZE_TOL


# ---------------------------------------------------------------------------
# z_cuts preserve every real material boundary
# ---------------------------------------------------------------------------

def test_z_cuts_include_every_real_box_boundary(discretised_scene):
    """Every real ``AxisAlignedBox`` z boundary must survive the global
    cut merger; otherwise two adjacent material layers would produce
    cells that straddle a material interface.
    """
    boxes = discretised_scene["boxes"]
    z_cuts = set(discretised_scene["grid"].z_cuts)
    missing = []
    for box in boxes:
        for coord in (box.z0, box.z1):
            if not any(abs(coord - c) <= 1e-12 for c in z_cuts):
                missing.append((box.name, coord))
    assert not missing, f"missing z cuts: {missing[:5]}"


def test_x_and_y_cuts_include_every_real_box_boundary(discretised_scene):
    boxes = discretised_scene["boxes"]
    x_cuts = set(discretised_scene["grid"].x_cuts)
    y_cuts = set(discretised_scene["grid"].y_cuts)
    for axis_name, cuts in (("x", x_cuts), ("y", y_cuts)):
        missing = []
        for box in boxes:
            for coord in ((box.x0, box.x1) if axis_name == "x" else (box.y0, box.y1)):
                if not any(abs(coord - c) <= 1e-12 for c in cuts):
                    missing.append((box.name, coord))
        assert not missing, f"missing {axis_name} cuts: {missing[:5]}"


# ---------------------------------------------------------------------------
# Cell provenance
# ---------------------------------------------------------------------------

def test_every_cell_has_a_real_box_parent(discretised_scene):
    cells = discretised_scene["cells"]
    box_names = {b.name for b in discretised_scene["boxes"]}
    assert all(c.parent_box_name in box_names for c in cells)
    assert all(c.parent_box_id for c in cells)


def test_cells_span_a_diverse_set_of_materials(discretised_scene):
    """The benchmark covers silicon, BEOL, hybrid bonding, TIM, lid, etc.;
    the discretised cells must reflect the same diversity.
    """
    cells = discretised_scene["cells"]
    materials = {c.material for c in cells}
    # 15 materials are declared in materials.py; the scene does not use
    # every one (e.g. substrate, solder), so we expect a strict subset
    # strictly greater than 5 and at most 15.
    assert 5 < len(materials) <= 15
    # The four canonical HBM material families must all be present.
    assert {"Silicon", "DRAM_BEOL", "Hybrid_Bonding", "Mold"} <= materials


# ---------------------------------------------------------------------------
# Adjacency edge diversity
# ---------------------------------------------------------------------------

def test_at_least_one_material_interface_edge(discretised_scene):
    edges = discretised_scene["edges"]
    material_edges = [e for e in edges if e.is_material_interface]
    assert material_edges, "expected at least one material-interface edge"


def test_edges_cover_all_three_axes(discretised_scene):
    edges = discretised_scene["edges"]
    axes = {e.axis for e in edges}
    assert axes == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# Boundary face classification
# ---------------------------------------------------------------------------

def test_both_boundary_classifications_are_present(discretised_scene):
    faces = discretised_scene["boundary_faces"]
    classes = {f.classification for f in faces}
    assert "scene_outer_boundary" in classes
    assert "exposed_internal_boundary" in classes


# ---------------------------------------------------------------------------
# Sanity counts
# ---------------------------------------------------------------------------

def test_all_counts_strictly_positive(discretised_scene):
    assert len(discretised_scene["cells"]) > 0
    assert len(discretised_scene["edges"]) > 0
    assert len(discretised_scene["boundary_faces"]) > 0


# ---------------------------------------------------------------------------
# Detailed size checks: thin layers are preserved
# ---------------------------------------------------------------------------

def test_min_cell_size_is_at_or_below_thin_layer_thickness(discretised_scene):
    """The benchmark has 0.15 um FEOL and 1 um oxide layers. The
    discretiser's min cell size must respect those.
    """
    cells = discretised_scene["cells"]
    # min cell size anywhere
    min_size = min(min(c.size_x, c.size_y, c.size_z) for c in cells)
    # Should be at or below the thinnest declared layer (0.15 um).
    assert min_size <= 0.15e-6 + _SIZE_TOL
