"""Block-structured thermal discretisation.

The discretisation layer takes the non-overlapping ``AxisAlignedBox``
scene produced by ``HorizontalColumnsBuilder`` and partitions it into a
regular, conformally meshed set of ``ThermalCell`` nodes. From the
cells it builds a face adjacency graph (``AdjacencyEdge``) and an
inventory of candidate boundary faces (``BoundaryFace``). No thermal
physics is computed here; the output is the geometric input that a
future KCL / steady-state solver would consume.
"""
from .models import (
    AdjacencyEdge,
    BoundaryFace,
    CellSizeConfig,
    DiscretizationConfig,
    GeometryOverlapError,
    ThermalCell,
)
from .grid import GlobalGrid, build_global_grid, subdivide_interval
from .adjacency import (
    build_adjacency,
    build_boundary_faces,
    generate_cells,
    validate_cell_surface_partition,
    validate_volume_conservation,
)

__all__ = [
    "AdjacencyEdge",
    "BoundaryFace",
    "CellSizeConfig",
    "DiscretizationConfig",
    "GlobalGrid",
    "GeometryOverlapError",
    "ThermalCell",
    "build_adjacency",
    "build_boundary_faces",
    "build_global_grid",
    "generate_cells",
    "subdivide_interval",
    "validate_cell_surface_partition",
    "validate_volume_conservation",
]
