"""Cell generation, adjacency and boundary face construction.

The discretiser walks the global grid once per ``AxisAlignedBox`` to
emit one ``ThermalCell`` per voxel that lies entirely inside the box.
The same walk is used to detect overlapping boxes (a hard error) and
to record which cells were emitted from which box.

After all boxes have been processed, ``build_adjacency`` and
``build_boundary_faces`` walk the integer grid once more to build the
``AdjacencyEdge`` and ``BoundaryFace`` inventories in O(N_cells) time
via dict lookups.
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass

from ..geometry.horizontal_columns import _LENGTH_TOL
from ..geometry.primitives import AxisAlignedBox
from .grid import GlobalGrid
from .models import AdjacencyEdge, BoundaryFace, GeometryOverlapError, ThermalCell


@dataclass
class _CellBuilder:
    """Mutable accumulator used while emitting cells."""

    cells: list[ThermalCell]
    cell_by_grid: dict[tuple[int, int, int], int]
    voxel_to_box: dict[tuple[int, int, int], str]


def _snapped_lo(cuts: tuple[float, ...], v: float,
                tol: float = _LENGTH_TOL) -> int:
    """Lower index: the first cut whose value is at least ``v - tol``.

    Without the tolerance, a box's ``z0`` that is just slightly larger
    than a cut (float noise from the builder) would skip the cell
    starting at that cut. Snapping within tolerance restores the cell
    so the box's voxel coverage matches its geometric extent.
    """
    idx = bisect.bisect_left(cuts, v)
    while idx > 0 and cuts[idx - 1] >= v - tol:
        idx -= 1
    return idx


def _snapped_hi(cuts: tuple[float, ...], v: float,
                tol: float = _LENGTH_TOL) -> int:
    """Upper (inclusive) index: the last cell whose z1 is at most ``v + tol``.

    The cell at index ``k`` has z range ``[cuts[k], cuts[k+1]]``. The last
    cell whose z1 (``cuts[k+1]``) is at most ``v`` is the largest ``k``
    with ``cuts[k+1] <= v``, i.e. ``k <= bisect_right(cuts, v) - 2``.
    Snapping allows the next cut after the cell to exceed ``v`` by up
    to ``tol`` before we still consider the cell in range.
    """
    upper_limit = bisect.bisect_right(cuts, v)
    idx = upper_limit - 2
    while idx + 1 < len(cuts) - 1 and cuts[idx + 2] <= v + tol:
        idx += 1
    return idx


def _box_index_range(box: AxisAlignedBox, grid: GlobalGrid
                    ) -> tuple[int, int, int, int, int, int]:
    """Return ``(ix0, ix1, iy0, iy1, iz0, iz1)`` as half-open index
    intervals that span every voxel touching the interior of ``box``.

    The lower bound is the first cut whose value is at least
    ``box.a0 - tol`` (so float noise does not skip the boundary cell).
    The upper bound is the last cell whose z1 is at most
    ``box.a1 + tol`` (so float noise does not include the next cell).
    """
    ix0 = _snapped_lo(grid.x_cuts, box.x0)
    ix1 = _snapped_hi(grid.x_cuts, box.x1)
    iy0 = _snapped_lo(grid.y_cuts, box.y0)
    iy1 = _snapped_hi(grid.y_cuts, box.y1)
    iz0 = _snapped_lo(grid.z_cuts, box.z0)
    iz1 = _snapped_hi(grid.z_cuts, box.z1)
    return ix0, ix1, iy0, iy1, iz0, iz1


def generate_cells(boxes: list[AxisAlignedBox], grid: GlobalGrid) -> list[ThermalCell]:
    """Walk the global cut grid once per box and emit one ``ThermalCell``
    per voxel that lies inside the box. Raises ``GeometryOverlapError`` if
    two boxes claim the same voxel.
    """
    builder = _CellBuilder(cells=[], cell_by_grid={}, voxel_to_box={})
    for box in boxes:
        ix0, ix1, iy0, iy1, iz0, iz1 = _box_index_range(box, grid)
        for ix in range(ix0, ix1 + 1):
            x0 = grid.x_cuts[ix]; x1 = grid.x_cuts[ix + 1]
            for iy in range(iy0, iy1 + 1):
                y0 = grid.y_cuts[iy]; y1 = grid.y_cuts[iy + 1]
                for iz in range(iz0, iz1 + 1):
                    z0 = grid.z_cuts[iz]; z1 = grid.z_cuts[iz + 1]
                    key = (ix, iy, iz)
                    if key in builder.voxel_to_box:
                        prior = builder.voxel_to_box[key]
                        if prior != box.name:
                            raise GeometryOverlapError(
                                box_a=prior, box_b=box.name,
                                ix=ix, iy=iy, iz=iz,
                                x_range=(x0, x1), y_range=(y0, y1), z_range=(z0, z1))
                        # Same box visited twice -> skip.
                        continue
                    builder.voxel_to_box[key] = box.name
                    component = box.tags.get("component")
                    cell_id = len(builder.cells)
                    cell = ThermalCell(
                        id=cell_id, ix=ix, iy=iy, iz=iz,
                        x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                        material=box.material,
                        parent_box_id=box.id, parent_box_name=box.name,
                        component=component, source_path=box.source_path,
                        rotation=box.rotation, tags=dict(box.tags),
                    )
                    builder.cells.append(cell)
                    builder.cell_by_grid[key] = cell_id
    return builder.cells


def build_adjacency(cells: list[ThermalCell], grid: GlobalGrid
                    ) -> list[AdjacencyEdge]:
    """Build the face adjacency graph in O(N_cells) time via a
    ``cell_by_grid`` dict lookup. Each cell looks at its +x, +y, +z
    neighbours; if a neighbour exists, an edge is emitted.
    """
    if not cells:
        return []
    cell_by_grid: dict[tuple[int, int, int], int] = {
        (c.ix, c.iy, c.iz): c.id for c in cells
    }
    edges: list[AdjacencyEdge] = []
    for cell in cells:
        for axis, (di, dj, dk) in (("x", (1, 0, 0)),
                                   ("y", (0, 1, 0)),
                                   ("z", (0, 0, 1))):
            nkey = (cell.ix + di, cell.iy + dj, cell.iz + dk)
            if nkey not in cell_by_grid:
                continue
            neighbour = cells[cell_by_grid[nkey]]
            a, b = (cell, neighbour) if cell.id < neighbour.id else (neighbour, cell)
            interface_coordinate = (a.z1 if axis == "z" else a.y1 if axis == "y" else a.x1)
            if axis == "x":
                face_area = a.size_y * a.size_z
            elif axis == "y":
                face_area = a.size_x * a.size_z
            else:
                face_area = a.size_x * a.size_y
            half_a = a.size_x / 2 if axis == "x" else a.size_y / 2 if axis == "y" else a.size_z / 2
            half_b = b.size_x / 2 if axis == "x" else b.size_y / 2 if axis == "y" else b.size_z / 2
            edges.append(AdjacencyEdge(
                id=len(edges),
                cell_a=a.id, cell_b=b.id, axis=axis,
                interface_coordinate=interface_coordinate,
                face_area=face_area,
                center_distance=half_a + half_b,
                half_distance_a=half_a, half_distance_b=half_b,
                material_a=a.material, material_b=b.material,
                is_material_interface=(a.material != b.material),
            ))
    return edges


def build_boundary_faces(cells: list[ThermalCell], grid: GlobalGrid
                         ) -> list[BoundaryFace]:
    """For every cell, inspect the six neighbour voxels; for each
    missing neighbour, emit a ``BoundaryFace``. Classify by whether the
    face coordinate matches the scene's global bounding box.
    """
    if not cells:
        return []
    cell_by_grid: dict[tuple[int, int, int], int] = {
        (c.ix, c.iy, c.iz): c.id for c in cells
    }
    scene_x0, scene_x1 = grid.x0(), grid.x1()
    scene_y0, scene_y1 = grid.y0(), grid.y1()
    scene_z0, scene_z1 = grid.z0(), grid.z1()
    faces: list[BoundaryFace] = []
    for cell in cells:
        probes = (
            ("x", "minus", -1, 0, 0, cell.x0, (-1.0, 0.0, 0.0)),
            ("x", "plus",   1, 0, 0, cell.x1, ( 1.0, 0.0, 0.0)),
            ("y", "minus", 0, -1, 0, cell.y0, (0.0, -1.0, 0.0)),
            ("y", "plus",   0,  1, 0, cell.y1, (0.0,  1.0, 0.0)),
            ("z", "minus", 0, 0, -1, cell.z0, (0.0, 0.0, -1.0)),
            ("z", "plus",   0,  0,  1, cell.z1, (0.0, 0.0,  1.0)),
        )
        for axis, side, di, dj, dk, coord, normal in probes:
            nkey = (cell.ix + di, cell.iy + dj, cell.iz + dk)
            if nkey in cell_by_grid:
                continue
            area = (cell.size_y * cell.size_z if axis == "x"
                    else cell.size_x * cell.size_z if axis == "y"
                    else cell.size_x * cell.size_y)
            if axis == "x":
                on_outer = (abs(coord - scene_x0) <= _LENGTH_TOL
                             or abs(coord - scene_x1) <= _LENGTH_TOL)
            elif axis == "y":
                on_outer = (abs(coord - scene_y0) <= _LENGTH_TOL
                             or abs(coord - scene_y1) <= _LENGTH_TOL)
            else:
                on_outer = (abs(coord - scene_z0) <= _LENGTH_TOL
                             or abs(coord - scene_z1) <= _LENGTH_TOL)
            classification = ("scene_outer_boundary" if on_outer
                              else "exposed_internal_boundary")
            faces.append(BoundaryFace(
                id=len(faces),
                cell_id=cell.id, axis=axis, side=side,
                coordinate=coord, area=area, normal=normal,
                component=cell.component, material=cell.material,
                classification=classification,
            ))
    return faces


# ---------------------------------------------------------------------------
# Conservation audit helpers.
# ---------------------------------------------------------------------------

def validate_volume_conservation(cells: list[ThermalCell],
                                 boxes: list[AxisAlignedBox],
                                 rel_tol: float = 1e-9,
                                 abs_tol: float = 1e-24) -> None:
    """Per-box child-cell volume conservation. Each box's child cells
    must sum to the box volume within tolerance; total cell volume must
    equal total box volume.
    """
    cells_by_box: dict[str, list[ThermalCell]] = defaultdict(list)
    for cell in cells:
        cells_by_box[cell.parent_box_name].append(cell)
    for box in boxes:
        children = cells_by_box.get(box.name, [])
        if not children:
            raise ValueError(f"box {box.name!r} has no child cells")
        child_volume = sum(c.volume for c in children)
        box_volume = (box.x1 - box.x0) * (box.y1 - box.y0) * (box.z1 - box.z0)
        if not _volumes_close(child_volume, box_volume, rel_tol, abs_tol):
            raise ValueError(
                f"box {box.name!r} child cells sum to {child_volume} m^3 but "
                f"the box is {box_volume} m^3")
    total_cells = sum(c.volume for c in cells)
    total_boxes = sum((b.x1 - b.x0) * (b.y1 - b.y0) * (b.z1 - b.z0) for b in boxes)
    if not _volumes_close(total_cells, total_boxes, rel_tol, abs_tol):
        raise ValueError(
            f"total cell volume {total_cells} differs from total box volume "
            f"{total_boxes}")


def validate_cell_surface_partition(cells: list[ThermalCell],
                                   edges: list[AdjacencyEdge],
                                   boundary_faces: list[BoundaryFace],
                                   rel_tol: float = 1e-9,
                                   abs_tol: float = 1e-24) -> None:
    """For every cell, the shared face area (counted on this cell's
    side) plus the boundary face area equals the analytic surface area
    ``2 * (sx*sy + sx*sz + sy*sz)``.
    """
    if not cells:
        return
    edges_by_cell: dict[int, list[AdjacencyEdge]] = defaultdict(list)
    for edge in edges:
        edges_by_cell[edge.cell_a].append(edge)
        edges_by_cell[edge.cell_b].append(edge)
    faces_by_cell: dict[int, list[BoundaryFace]] = defaultdict(list)
    for face in boundary_faces:
        faces_by_cell[face.cell_id].append(face)
    for cell in cells:
        expected = 2.0 * (cell.size_x * cell.size_y
                          + cell.size_x * cell.size_z
                          + cell.size_y * cell.size_z)
        shared = sum(edge.face_area for edge in edges_by_cell[cell.id])
        boundary = sum(face.area for face in faces_by_cell[cell.id])
        accounted = shared + boundary
        if not _volumes_close(expected, accounted, rel_tol, abs_tol):
            raise ValueError(
                f"cell {cell.id} ({cell.parent_box_name}) surface area "
                f"mismatch: expected {expected}, accounted {accounted} "
                f"(shared {shared}, boundary {boundary})")


def _volumes_close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))
