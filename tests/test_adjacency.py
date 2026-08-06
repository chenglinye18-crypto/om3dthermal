"""Tests for the face adjacency graph and boundary face inventory.

The algorithm lives in :mod:`om3dthermal.discretization.adjacency`; these
tests construct small, hand-built scenes so a regression points straight
at the adjacency bookkeeping rather than at any specific benchmark.
"""
from __future__ import annotations

import pytest

from om3dthermal.discretization import (
    build_adjacency,
    build_boundary_faces,
    build_global_grid,
    generate_cells,
    validate_cell_surface_partition,
)
from om3dthermal.geometry.primitives import AxisAlignedBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(name: str, x0: float, x1: float, y0: float, y1: float,
         z0: float, z1: float, material: str = "Silicon") -> AxisAlignedBox:
    return AxisAlignedBox(
        name=name, material=material,
        x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
        source_path="tests/test_adjacency.py",
    )


class _TinyCellSize:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


def _build(boxes, max_size):
    grid = build_global_grid(boxes, max_size)
    cells = generate_cells(boxes, grid)
    return grid, cells


# ---------------------------------------------------------------------------
# Single-cell topology
# ---------------------------------------------------------------------------

def test_single_cell_has_no_adjacency_edges_and_six_boundary_faces():
    box = _box("only", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    grid, cells = _build([box], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    edges = build_adjacency(cells, grid)
    assert edges == []
    faces = build_boundary_faces(cells, grid)
    # All six faces of the only cell.
    assert len(faces) == 6
    sides = {(f.axis, f.side) for f in faces}
    assert sides == {("x", "minus"), ("x", "plus"),
                     ("y", "minus"), ("y", "plus"),
                     ("z", "minus"), ("z", "plus")}
    # Every face is on the scene bounding box -> scene_outer_boundary.
    assert all(f.classification == "scene_outer_boundary" for f in faces)
    # Six identical 1 mm x 1 mm faces.
    assert all(f.area == pytest.approx(1e-6) for f in faces)


# ---------------------------------------------------------------------------
# Two cells along x
# ---------------------------------------------------------------------------

def test_two_cells_along_x_have_one_x_edge_and_face_areas_match():
    left  = _box("left",  0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    right = _box("right", 1.0e-3, 2.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si")
    grid, cells = _build([left, right], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    edges = build_adjacency(cells, grid)
    assert len(edges) == 1
    e = edges[0]
    assert e.axis == "x"
    assert e.face_area == pytest.approx(1.0e-6)  # 1 mm * 1 mm
    assert e.half_distance_a == pytest.approx(0.5e-3)
    assert e.half_distance_b == pytest.approx(0.5e-3)
    assert e.center_distance == pytest.approx(1.0e-3)
    assert e.interface_coordinate == pytest.approx(1.0e-3)
    assert e.is_material_interface is True
    assert {e.material_a, e.material_b} == {"Cu", "Si"}

    faces = build_boundary_faces(cells, grid)
    # 6 faces per cell minus 2 shared with the neighbour = 10.
    assert len(faces) == 10
    validate_cell_surface_partition(cells, edges, faces)


# ---------------------------------------------------------------------------
# 2x2x2 lattice - edge counts per axis
# ---------------------------------------------------------------------------

def test_2x2x2_lattice_has_exactly_twelve_edges_four_per_axis():
    box = _box("cube", 0.0, 2.0e-3, 0.0, 2.0e-3, 0.0, 2.0e-3, material="Cu")
    grid, cells = _build([box], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    assert len(cells) == 8
    edges = build_adjacency(cells, grid)
    by_axis = {}
    for e in edges:
        by_axis[e.axis] = by_axis.get(e.axis, 0) + 1
    # 2x2x2 has 4 internal faces along each axis: 2*1*2 in y-z direction etc.
    assert by_axis == {"x": 4, "y": 4, "z": 4}
    assert len(edges) == 12
    # No edge references the same cell twice.
    for e in edges:
        assert e.cell_a != e.cell_b
        assert e.is_material_interface is False  # all same material
    # Boundary faces for the lattice: 8 cells * 6 faces / 2 = 24
    # (every internal face is shared with a neighbour, so the remaining
    # faces are the 4 faces of the bounding box, each tiled into 4 = 16).
    faces = build_boundary_faces(cells, grid)
    assert len(faces) == 24
    assert all(f.classification == "scene_outer_boundary" for f in faces)


# ---------------------------------------------------------------------------
# No diagonal adjacency
# ---------------------------------------------------------------------------

def test_no_diagonal_or_edge_adjacency_is_built():
    # Two boxes that touch only at a single edge (the x1/y1 corner).
    # The only adjacency that should be built is the x face share and the
    # y face share; the edge-only contact should produce no edge.
    a = _box("a", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    b = _box("b", 1.0e-3, 2.0e-3, 1.0e-3, 2.0e-3, 0.0, 1.0e-3, material="Si")
    grid, cells = _build([a, b], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    edges = build_adjacency(cells, grid)
    # No shared face -> no edge.
    assert edges == []


# ---------------------------------------------------------------------------
# T-junction: one big face vs two small faces
# ---------------------------------------------------------------------------

def test_t_junction_global_cut_splits_big_face_into_smaller_edges():
    # A 0.5 x 0.5 x 0.5 mm "plate" sits under two 0.25 x 0.5 x 0.5 mm
    # "tops" that share the plate's top face. The shared boundary between
    # the two tops (x = 0.25 mm) is a global cut; the plate is therefore
    # split in x and the original "big" top face becomes two adjacency
    # edges whose face_areas sum to the original big face area.
    plate = _box("plate", 0.0, 0.5e-3, 0.0, 0.5e-3, 0.0, 0.5e-3, material="Cu")
    top_left  = _box("tl", 0.0, 0.25e-3, 0.0, 0.5e-3, 0.5e-3, 1.0e-3, material="Si")
    top_right = _box("tr", 0.25e-3, 0.5e-3, 0.0, 0.5e-3, 0.5e-3, 1.0e-3, material="Al")
    grid, cells = _build([plate, top_left, top_right],
                         _TinyCellSize(0.5e-3, 0.5e-3, 0.5e-3))
    # The top boxes' shared boundary must be a global x cut.
    assert 0.25e-3 in grid.x_cuts
    # The plate is now split into two cells (one per top box).
    plate_cells = [c for c in cells if c.parent_box_name == "plate"]
    assert len(plate_cells) == 2
    edges = build_adjacency(cells, grid)
    z_edges = [e for e in edges if e.axis == "z"]
    # Each plate cell meets exactly one top cell.
    assert len(z_edges) == 2
    # Both are material interfaces.
    assert all(e.is_material_interface for e in z_edges)
    # Their face_areas sum to the plate's top face area (0.5 x 0.5 mm).
    total_face_area = sum(e.face_area for e in z_edges)
    assert total_face_area == pytest.approx(0.5e-3 * 0.5e-3)
    # Each is 0.25 x 0.5 mm.
    for e in z_edges:
        assert e.face_area == pytest.approx(0.25e-3 * 0.5e-3)
    # Surface partition audit still passes.
    faces = build_boundary_faces(cells, grid)
    validate_cell_surface_partition(cells, edges, faces)


# ---------------------------------------------------------------------------
# Material interface flag
# ---------------------------------------------------------------------------

def test_material_interface_flag_is_set_only_when_materials_differ():
    # Two cells along x; left=Copper, right=Silicon. The single x edge
    # must be flagged as a material interface.
    left  = _box("left",  0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    right = _box("right", 1.0e-3, 2.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si")
    grid, cells = _build([left, right], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    edges = build_adjacency(cells, grid)
    assert len(edges) == 1
    assert edges[0].is_material_interface is True

    # Now same material on both sides.
    left2  = _box("left2",  0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    right2 = _box("right2", 1.0e-3, 2.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    grid2, cells2 = _build([left2, right2], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    edges2 = build_adjacency(cells2, grid2)
    assert len(edges2) == 1
    assert edges2[0].is_material_interface is False


# ---------------------------------------------------------------------------
# Boundary face classification: outer vs exposed internal
# ---------------------------------------------------------------------------

def test_void_in_middle_produces_exposed_internal_boundary_faces():
    # Two boxes separated by an empty region; the inner-facing faces must
    # be classified as exposed_internal_boundary, not scene_outer_boundary.
    left  = _box("left",  0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    right = _box("right", 3.0e-3, 4.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si")
    grid, cells = _build([left, right], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    faces = build_boundary_faces(cells, grid)
    by_class = {"scene_outer_boundary": [], "exposed_internal_boundary": []}
    for f in faces:
        by_class[f.classification].append(f)

    # Per cell: 5 outer (x_minus, y_minus, y_plus, z_minus, z_plus) +
    # 1 internal (x_plus on the left cell, x_minus on the right cell).
    # Two cells => 10 outer + 2 internal.
    assert len(by_class["scene_outer_boundary"]) == 10
    assert len(by_class["exposed_internal_boundary"]) == 2
    for f in by_class["exposed_internal_boundary"]:
        assert f.axis == "x"
        assert f.area == pytest.approx(1.0e-6)


# ---------------------------------------------------------------------------
# Per-cell surface area conservation
# ---------------------------------------------------------------------------

def test_cell_surface_partition_audit_passes_for_simple_scene():
    a = _box("a", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    b = _box("b", 1.0e-3, 2.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si")
    c = _box("c", 0.0, 1.0e-3, 1.0e-3, 2.0e-3, 0.0, 1.0e-3, material="Al")
    grid, cells = _build([a, b, c], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    edges = build_adjacency(cells, grid)
    faces = build_boundary_faces(cells, grid)
    # Should not raise.
    validate_cell_surface_partition(cells, edges, faces)


def test_cell_surface_partition_detects_missing_face_accounting():
    # Build a small scene, then drop one boundary face; the audit must fail.
    a = _box("a", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    grid, cells = _build([a], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    faces = build_boundary_faces(cells, grid)
    # Drop the last face; the audit should now complain.
    truncated = faces[:-1]
    with pytest.raises(ValueError, match="surface area mismatch"):
        validate_cell_surface_partition(cells, [], truncated)
