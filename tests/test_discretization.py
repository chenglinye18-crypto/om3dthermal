"""Unit tests for the block-structured thermal discretisation.

These tests exercise :mod:`om3dthermal.discretization` against small,
hand-built scenes so failures point straight at the algorithm rather
than at a specific benchmark. The shipped HBM-on-GPU benchmark has its
own dedicated tests in ``test_discretization_benchmark.py``.
"""
from __future__ import annotations

import pytest

from om3dthermal.discretization import (
    GlobalGrid,
    build_global_grid,
    generate_cells,
    subdivide_interval,
)
from om3dthermal.discretization.models import (
    GeometryOverlapError,
    ThermalCell,
)
from om3dthermal.geometry.primitives import AxisAlignedBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(name: str, x0: float, x1: float, y0: float, y1: float,
         z0: float, z1: float, material: str = "Silicon",
         **tags) -> AxisAlignedBox:
    return AxisAlignedBox(
        name=name, material=material,
        x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
        source_path="tests/test_discretization.py",
        tags=tags,
    )


class _TinyCellSize:
    """Minimal stand-in for ``CellSizeConfig`` exposing ``x/y/z``."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


# ---------------------------------------------------------------------------
# grid.subdivide_interval
# ---------------------------------------------------------------------------

def test_subdivide_interval_returns_endpoints_for_short_interval():
    cuts = subdivide_interval(0.0, 0.4e-3, 0.5e-3)
    assert cuts == [0.0, 0.4e-3]


def test_subdivide_interval_uniform_when_evenly_divisible():
    cuts = subdivide_interval(0.0, 2.0e-3, 0.5e-3)
    assert len(cuts) == 5
    assert cuts[0] == 0.0
    assert cuts[-1] == pytest.approx(2.0e-3)
    step = 0.5e-3
    for i in range(5):
        assert cuts[i] == pytest.approx(i * step)


def test_subdivide_interval_rounds_up_to_ceiling_when_not_evenly_divisible():
    # length 1.0, max 0.3 -> n = ceil(1.0/0.3) = 4
    cuts = subdivide_interval(0.0, 1.0, 0.3)
    assert cuts[0] == 0.0
    assert cuts[-1] == pytest.approx(1.0)
    assert len(cuts) == 5
    # max cell size should be < max_size (since we rounded up)
    max_step = max(b - a for a, b in zip(cuts, cuts[1:]))
    assert max_step <= 0.3 + 1e-12


def test_subdivide_interval_snaps_pint_100um_noise_to_integer_subdivisions():
    # Pint parses "100 um" to 9.999999999999999e-05; we don't want an
    # extra ghost subdivision because of that.
    a, b, max_size = 0.0, 100e-6, 9.999999999999999e-05
    cuts = subdivide_interval(a, b, max_size)
    # Without snapping, ratio = 1.000...0001 and ceil = 2 -> a ghost cell.
    # With snapping, n = 1 and we return [a, b].
    assert cuts == [pytest.approx(a), pytest.approx(b)]


def test_subdivide_interval_rejects_degenerate_interval():
    with pytest.raises(ValueError):
        subdivide_interval(0.0, 0.0, 1e-3)
    with pytest.raises(ValueError):
        subdivide_interval(0.0, -1e-3, 1e-3)


def test_subdivide_interval_rejects_non_positive_max_size():
    with pytest.raises(ValueError):
        subdivide_interval(0.0, 1e-3, 0.0)
    with pytest.raises(ValueError):
        subdivide_interval(0.0, 1e-3, -1e-3)


# ---------------------------------------------------------------------------
# build_global_grid / cell generation
# ---------------------------------------------------------------------------

def test_single_box_smaller_than_max_size_produces_exactly_one_cell():
    box = _box("only", 0.0, 0.2e-3, 0.0, 0.2e-3, 0.0, 0.2e-3, material="Cu")
    grid = build_global_grid([box], _TinyCellSize(0.5e-3, 0.5e-3, 0.5e-3))
    cells = generate_cells([box], grid)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.x0 == pytest.approx(0.0)
    assert cell.x1 == pytest.approx(0.2e-3)
    assert cell.material == "Cu"
    assert cell.parent_box_name == "only"
    assert cell.volume == pytest.approx((0.2e-3) ** 3)


def test_single_box_uniform_subdivision_2x2x2_yields_eight_cells():
    box = _box("cube", 0.0, 2.0e-3, 0.0, 2.0e-3, 0.0, 2.0e-3, material="Si")
    grid = build_global_grid([box], _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    cells = generate_cells([box], grid)
    assert len(cells) == 8
    centres = sorted((c.center_x, c.center_y, c.center_z) for c in cells)
    expected = [(0.5e-3, 0.5e-3, 0.5e-3),
                (0.5e-3, 0.5e-3, 1.5e-3),
                (0.5e-3, 1.5e-3, 0.5e-3),
                (0.5e-3, 1.5e-3, 1.5e-3),
                (1.5e-3, 0.5e-3, 0.5e-3),
                (1.5e-3, 0.5e-3, 1.5e-3),
                (1.5e-3, 1.5e-3, 0.5e-3),
                (1.5e-3, 1.5e-3, 1.5e-3)]
    assert centres == [(pytest.approx(x), pytest.approx(y), pytest.approx(z))
                       for (x, y, z) in expected]


def test_non_integer_length_subdivision_keeps_max_cell_size_below_config():
    # 1.7 mm long, max cell 0.5 mm -> n = ceil(3.4) = 4 -> step = 0.425 mm
    box = _box("bar", 0.0, 1.7e-3, 0.0, 0.5e-3, 0.0, 0.5e-3, material="Al")
    grid = build_global_grid([box], _TinyCellSize(0.5e-3, 0.5e-3, 0.5e-3))
    cells = generate_cells([box], grid)
    assert len(cells) == 4
    for cell in cells:
        assert cell.size_x <= 0.5e-3 + 1e-12
    # Endpoints must match the box exactly.
    assert cells[0].x0 == pytest.approx(0.0)
    assert cells[-1].x1 == pytest.approx(1.7e-3)


def test_stacked_multi_material_layers_share_z_cuts_at_material_boundaries():
    bottom = _box("bot", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 0.5e-3, material="Cu")
    middle = _box("mid", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.5e-3, 1.0e-3, material="Si")
    top    = _box("top", 0.0, 1.0e-3, 0.0, 1.0e-3, 1.0e-3, 1.5e-3, material="Al")
    grid = build_global_grid([bottom, middle, top],
                             _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    # Every material interface (0.5e-3, 1.0e-3) must be a global z cut.
    assert 0.5e-3 in grid.z_cuts
    assert 1.0e-3 in grid.z_cuts
    # Each layer should produce exactly one cell (size matches).
    cells = generate_cells([bottom, middle, top], grid)
    by_material = {c.material for c in cells}
    assert by_material == {"Cu", "Si", "Al"}
    # No cell crosses a material boundary.
    for cell in cells:
        assert cell.z0 < cell.z1
        assert (cell.material == "Cu" and cell.z1 == pytest.approx(0.5e-3)) \
            or (cell.material == "Si" and cell.z0 == pytest.approx(0.5e-3)
                and cell.z1 == pytest.approx(1.0e-3)) \
            or (cell.material == "Al" and cell.z0 == pytest.approx(1.0e-3))


def test_left_right_adjacent_boxes_share_a_cut_and_do_not_overlap():
    left  = _box("left",  0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    right = _box("right", 1.0e-3, 2.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si")
    grid = build_global_grid([left, right],
                             _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    assert 1.0e-3 in grid.x_cuts  # the shared boundary is a cut
    cells = generate_cells([left, right], grid)
    assert len(cells) == 2
    total_volume = sum(c.volume for c in cells)
    assert total_volume == pytest.approx(2.0 * (1.0e-3) ** 3)


def test_overlapping_boxes_raise_geometry_overlap_error():
    a = _box("a", 0.0, 1.5e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    b = _box("b", 1.0e-3, 2.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si")
    grid = build_global_grid([a, b],
                             _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    with pytest.raises(GeometryOverlapError) as excinfo:
        generate_cells([a, b], grid)
    # The error should point at both boxes and at a specific voxel.
    assert excinfo.value.box_a in {"a", "b"}
    assert excinfo.value.box_b in {"a", "b"}
    assert excinfo.value.box_a != excinfo.value.box_b
    assert excinfo.value.ix >= 0 and excinfo.value.iy >= 0 and excinfo.value.iz >= 0


def test_voxel_ownership_does_not_get_silently_overwritten_by_priority():
    # Two boxes claim exactly the same voxel; the second one must be a hard
    # error, not a silent priority override.
    a = _box("a", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu",
             priority=0)
    b = _box("b", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si",
             priority=10)
    grid = build_global_grid([a, b],
                             _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    with pytest.raises(GeometryOverlapError):
        generate_cells([a, b], grid)


def test_void_in_space_produces_no_cell_but_neighbouring_faces_are_internal():
    # Two boxes separated by empty space along x; the inner-facing faces
    # should later be classified as exposed_internal_boundary.
    left  = _box("left",  0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    right = _box("right", 3.0e-3, 4.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si")
    grid = build_global_grid([left, right],
                             _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    cells = generate_cells([left, right], grid)
    # No cell spans the 2 mm void.
    assert all(c.x1 <= 1.0e-3 + 1e-12 or c.x0 >= 3.0e-3 - 1e-12
               for c in cells)
    # The void is represented by an x_cut at 1.0e-3 and 3.0e-3 from the
    # two boxes, plus the internal subdivisions inside the void's extent.
    # What matters is that the inner-facing cells (the right face of "left"
    # and the left face of "right") sit on a cut that's not on the scene
    # bounding box, so they will be classified as exposed_internal_boundary
    # by build_boundary_faces. That's verified in test_adjacency.py.
    from om3dthermal.discretization import build_boundary_faces
    faces = build_boundary_faces(cells, grid)
    classifications = {f.classification for f in faces}
    assert "exposed_internal_boundary" in classifications


# ---------------------------------------------------------------------------
# ThermalCell invariants
# ---------------------------------------------------------------------------

def test_thermal_cell_rejects_zero_extent():
    with pytest.raises(Exception):
        ThermalCell(
            id=0, ix=0, iy=0, iz=0,
            x0=0.0, x1=0.0, y0=0.0, y1=1.0, z0=0.0, z1=1.0,
            material="Si", parent_box_id="b", parent_box_name="b",
            source_path="x",
        )


def test_thermal_cell_rejects_negative_grid_index():
    with pytest.raises(Exception):
        ThermalCell(
            id=0, ix=-1, iy=0, iz=0,
            x0=0.0, x1=1.0, y0=0.0, y1=1.0, z0=0.0, z1=1.0,
            material="Si", parent_box_id="b", parent_box_name="b",
            source_path="x",
        )


# ---------------------------------------------------------------------------
# GlobalGrid is a strict-ascending cut array
# ---------------------------------------------------------------------------

def test_global_grid_cuts_are_strictly_ascending_and_deduplicated():
    # Two boxes that touch along a plane produce a duplicate boundary cut
    # which the merger should drop, not double-list.
    a = _box("a", 0.0, 1.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Cu")
    b = _box("b", 1.0e-3, 2.0e-3, 0.0, 1.0e-3, 0.0, 1.0e-3, material="Si")
    grid = build_global_grid([a, b],
                             _TinyCellSize(1.0e-3, 1.0e-3, 1.0e-3))
    for cuts in (grid.x_cuts, grid.y_cuts, grid.z_cuts):
        # Strictly increasing
        for prev, cur in zip(cuts, cuts[1:]):
            assert cur > prev
        # The shared x boundary appears once, not twice.
        if cuts is grid.x_cuts:
            assert list(cuts).count(1.0e-3) == 1
