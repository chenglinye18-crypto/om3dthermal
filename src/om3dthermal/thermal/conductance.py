"""Per-edge thermal conductance for the discretised mesh.

For every :class:`AdjacencyEdge` produced by the discretiser we
compute the two-point face conductance

    G_ab = A / ( d_a / k_na + R''_ab + d_b / k_nb )   [W/K]

where

- ``A`` is the face area;
- ``d_a`` / ``d_b`` are the half-extents of the two cells along the
  shared-face normal;
- ``k_na`` / ``k_nb`` are the per-cell normal conductivities
  (``k_n = n^T K_global n``);
- ``R''_ab`` is the optional areal interface resistance
  (``m^2*K/W``), looked up from
  :class:`InterfaceResistanceRegistry`.

The result is stored as a **columnar** :class:`ConductanceTable` of
NumPy arrays (one entry per edge) so the 790 964-edge benchmark does
not allocate 790 964 Python objects. The arrays are the single source
of truth for the per-edge quantities; later stages (KCL assembly,
solver) consume them directly.

Performance notes:

- ``k_n`` is cached on the combination
  ``(material_name, canonical_rotation_key, axis)`` because the same
  material / rotation appears at ~hundreds of thousands of edges with
  the same ``k_n`` value; recomputing the projection per edge would
  be wasted work;
- the unordered ``R''`` lookup is a single dict access per edge;
- the conductance math is done in vectorised NumPy so the loop
  overhead is one Python iteration per edge (the heavy lifting is in
  C).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import ThermalConductanceConfig
from ..discretization.models import AdjacencyEdge, ThermalCell
from ..materials import Material
from .errors import (
    MissingThermalConductivityError,
    UnsupportedMaterialRotationError,
)
from .interfaces import InterfaceResistanceRegistry
from .tensors import (
    canonical_rotation_key,
    normal_conductivity,
)


# Per-axis integer code. The AdjacencyEdge stores the axis as a
# string; we encode it for cache keys and for the columnar array.
_AXIS_CODE = {"x": 0, "y": 1, "z": 2}
_AXIS_NORMAL = {
    0: (1.0, 0.0, 0.0),
    1: (0.0, 1.0, 0.0),
    2: (0.0, 0.0, 1.0),
}


@dataclass
class ConductanceTable:
    """Columnar per-edge thermal conductance.

    All arrays have length ``edge_count == len(adjacency_edges)``. The
    column order matches the input edge list; ``edge_id[i]`` is the
    index of the i-th edge in the source list.
    """

    edge_id: np.ndarray            # int64
    cell_a: np.ndarray             # int64
    cell_b: np.ndarray             # int64
    axis: np.ndarray               # int8 (0=x, 1=y, 2=z)
    face_area_m2: np.ndarray       # float64
    half_distance_a_m: np.ndarray  # float64
    half_distance_b_m: np.ndarray  # float64
    k_normal_a_W_mK: np.ndarray    # float64
    k_normal_b_W_mK: np.ndarray    # float64
    interface_areal_resistance_m2K_W: np.ndarray  # float64
    resistance_K_W: np.ndarray     # float64
    conductance_W_K: np.ndarray    # float64
    material_interface: np.ndarray # bool
    interface_rule_index: np.ndarray  # int32 (-1 = default rule)

    @property
    def edge_count(self) -> int:
        return int(self.edge_id.shape[0])

    @property
    def nonzero_interface_resistance_count(self) -> int:
        return int(np.count_nonzero(
            self.interface_areal_resistance_m2K_W > 0.0))

    @property
    def min_conductance(self) -> float:
        return float(self.conductance_W_K.min())

    @property
    def max_conductance(self) -> float:
        return float(self.conductance_W_K.max())

    @property
    def mean_conductance(self) -> float:
        return float(self.conductance_W_K.mean())

    @property
    def min_resistance(self) -> float:
        return float(self.resistance_K_W.min())

    @property
    def max_resistance(self) -> float:
        return float(self.resistance_K_W.max())

    @property
    def min_k_normal(self) -> float:
        return float(
            np.minimum.reduce([self.k_normal_a_W_mK.min(),
                               self.k_normal_b_W_mK.min()]))

    @property
    def max_k_normal(self) -> float:
        return float(
            np.maximum.reduce([self.k_normal_a_W_mK.max(),
                               self.k_normal_b_W_mK.max()]))


def _face_half_distances(cell: ThermalCell, axis_code: int
                         ) -> tuple[float, float, float]:
    """Return ``(face_area, half_distance_along_normal, half_distance_other_axis)``.

    The ``face_area`` is the area of the face perpendicular to ``axis``
    on this cell. The ``half_distance_along_normal`` is half the cell's
    extent along the normal axis (i.e. the distance from the cell
    centre to the shared interface). The third return value is unused
    but kept for symmetry with the formula.
    """
    if axis_code == 0:
        return (cell.size_y * cell.size_z, cell.size_x / 2, cell.size_x / 2)
    if axis_code == 1:
        return (cell.size_x * cell.size_z, cell.size_y / 2, cell.size_y / 2)
    return (cell.size_x * cell.size_y, cell.size_z / 2, cell.size_z / 2)


def build_conductance_table(
    cells: Sequence[ThermalCell],
    adjacency_edges: Sequence[AdjacencyEdge],
    materials: dict[str, Material],
    config: ThermalConductanceConfig,
) -> ConductanceTable:
    """Compute the per-edge conductance table.

    The edge order is the same as ``adjacency_edges``; the columnar
    arrays are populated in that order so callers can pair
    ``adjacency_edges[i]`` with ``table.edge_id[i]`` and the rest of
    the columns.

    Raises:
        MissingThermalConductivityError: a cell's material has
            ``k_local is None`` (e.g. a custom user material that did
            not declare a tensor). The error names the cell, parent
            box, and edge.
        UnsupportedMaterialRotationError: a cell carries a rotation
            that is not a signed axis permutation.
    """
    if config.rotation_policy != "axis_aligned_only":
        # Defensive: the Literal type already constrains this, but a
        # future policy string would slip past the schema check.
        raise ValueError(
            f"unsupported rotation_policy {config.rotation_policy!r}; "
            "this stage only implements 'axis_aligned_only'")
    registry = InterfaceResistanceRegistry(
        default_areal_resistance=config.default_interface_areal_resistance,
        rules=config.interfaces,
    )

    # Cell lookup. The discretiser assigns unique ids, so this is a
    # straight dict.
    cell_by_id: dict[int, ThermalCell] = {c.id: c for c in cells}

    # Per-cell k_n cache: key = (material_name, rotation_key, axis).
    # rotation_key is the canonical 9-int tuple from
    # tensors.canonical_rotation_key, so identity collapses with -I
    # and 90/180/270-degree rotations about any axis collapse to their
    # own distinct slots.
    k_n_cache: dict[tuple[str, tuple[int, ...], int], float] = {}

    edge_count = len(adjacency_edges)
    edge_id = np.arange(edge_count, dtype=np.int64)
    cell_a_arr = np.empty(edge_count, dtype=np.int64)
    cell_b_arr = np.empty(edge_count, dtype=np.int64)
    axis_arr = np.empty(edge_count, dtype=np.int8)
    face_area_arr = np.empty(edge_count, dtype=np.float64)
    half_d_a_arr = np.empty(edge_count, dtype=np.float64)
    half_d_b_arr = np.empty(edge_count, dtype=np.float64)
    k_n_a_arr = np.empty(edge_count, dtype=np.float64)
    k_n_b_arr = np.empty(edge_count, dtype=np.float64)
    rpp_arr = np.empty(edge_count, dtype=np.float64)
    r_arr = np.empty(edge_count, dtype=np.float64)
    g_arr = np.empty(edge_count, dtype=np.float64)
    mat_iface_arr = np.empty(edge_count, dtype=bool)
    rule_idx_arr = np.full(edge_count, -1, dtype=np.int32)

    for i, edge in enumerate(adjacency_edges):
        if edge.id != i:
            # Sanity: the discretiser assigns sequential ids. If a
            # caller passes a list with gaps / re-ordering we still
            # preserve the column order but record the actual id.
            pass
        cell_a = cell_by_id.get(edge.cell_a)
        cell_b = cell_by_id.get(edge.cell_b)
        if cell_a is None:
            raise ValueError(
                f"edge {i} references unknown cell_a {edge.cell_a}")
        if cell_b is None:
            raise ValueError(
                f"edge {i} references unknown cell_b {edge.cell_b}")
        axis_code = _AXIS_CODE[edge.axis]
        face_area_a, half_d_a, _ = _face_half_distances(cell_a, axis_code)
        _, half_d_b, _ = _face_half_distances(cell_b, axis_code)
        # The two cells share a face so face_area_a == face_area_b;
        # we use cell_a's value.
        face_area = face_area_a

        k_n_a = _resolve_k_n(cell_a, axis_code, materials, k_n_cache,
                             edge_id=i)
        k_n_b = _resolve_k_n(cell_b, axis_code, materials, k_n_cache,
                             edge_id=i)
        if not (face_area > 0):
            raise ValueError(
                f"edge {i} has non-positive face area {face_area}")
        if not (half_d_a > 0):
            raise ValueError(
                f"edge {i} cell_a {cell_a.id} has non-positive half "
                f"distance {half_d_a}")
        if not (half_d_b > 0):
            raise ValueError(
                f"edge {i} cell_b {cell_b.id} has non-positive half "
                f"distance {half_d_b}")
        if not (k_n_a > 0 and math.isfinite(k_n_a)):
            raise ValueError(
                f"edge {i} cell_a k_n must be positive finite, got {k_n_a}")
        if not (k_n_b > 0 and math.isfinite(k_n_b)):
            raise ValueError(
                f"edge {i} cell_b k_n must be positive finite, got {k_n_b}")

        query = registry.lookup(cell_a.material, cell_b.material)
        rpp = query.value
        if rpp < 0:
            raise ValueError(
                f"edge {i} areal interface resistance is negative: {rpp}")

        # Two-point face conductance (R is the total resistance K/W,
        # G is the conductance W/K).
        resistance = (half_d_a / k_n_a) + rpp + (half_d_b / k_n_b)
        resistance /= face_area
        conductance = 1.0 / resistance

        cell_a_arr[i] = cell_a.id
        cell_b_arr[i] = cell_b.id
        axis_arr[i] = axis_code
        face_area_arr[i] = face_area
        half_d_a_arr[i] = half_d_a
        half_d_b_arr[i] = half_d_b
        k_n_a_arr[i] = k_n_a
        k_n_b_arr[i] = k_n_b
        rpp_arr[i] = rpp
        r_arr[i] = resistance
        g_arr[i] = conductance
        mat_iface_arr[i] = (cell_a.material != cell_b.material)
        rule_idx_arr[i] = query.rule_index

        if not (conductance > 0) or not math.isfinite(conductance):
            raise ValueError(
                f"edge {i} conductance is not positive finite: "
                f"G={conductance} (face_area={face_area}, half_d_a={half_d_a},"
                f" half_d_b={half_d_b}, k_n_a={k_n_a}, k_n_b={k_n_b}, "
                f"R''={rpp})")
        if not (resistance > 0) or not math.isfinite(resistance):
            raise ValueError(
                f"edge {i} resistance is not positive finite: "
                f"R={resistance}")

    return ConductanceTable(
        edge_id=edge_id,
        cell_a=cell_a_arr,
        cell_b=cell_b_arr,
        axis=axis_arr,
        face_area_m2=face_area_arr,
        half_distance_a_m=half_d_a_arr,
        half_distance_b_m=half_d_b_arr,
        k_normal_a_W_mK=k_n_a_arr,
        k_normal_b_W_mK=k_n_b_arr,
        interface_areal_resistance_m2K_W=rpp_arr,
        resistance_K_W=r_arr,
        conductance_W_K=g_arr,
        material_interface=mat_iface_arr,
        interface_rule_index=rule_idx_arr,
    )


def _resolve_k_n(cell: ThermalCell, axis_code: int,
                 materials: dict[str, Material],
                 cache: dict[tuple[str, tuple[int, ...], int], float],
                 *, edge_id: int) -> float:
    """Resolve the normal conductivity of ``cell`` along ``axis_code``.

    Caches by ``(material_name, canonical_rotation_key, axis)`` so the
    same material / rotation / axis combination is computed once
    regardless of how many edges it appears on.
    """
    from .errors import InvalidRotationMatrixError
    material = materials.get(cell.material)
    if material is None:
        raise ValueError(
            f"cell {cell.id} (parent box {cell.parent_box_name!r}) "
            f"references unknown material {cell.material!r}")
    if material.k_local is None:
        raise MissingThermalConductivityError(
            material=cell.material, cell_id=cell.id,
            parent_box=cell.parent_box_name, edge_id=edge_id)
    try:
        rotation_key = canonical_rotation_key(cell.rotation)
    except InvalidRotationMatrixError as exc:
        # The rotation is not a signed axis permutation. Re-raise as
        # the more specific UnsupportedMaterialRotationError so
        # callers can distinguish it from a malformed input.
        raise UnsupportedMaterialRotationError(
            cell.rotation, cell_id=cell.id) from exc
    key = (cell.material, rotation_key, axis_code)
    cached = cache.get(key)
    if cached is not None:
        return cached
    k_n = normal_conductivity(material.k_local, cell.rotation, axis_code)
    cache[key] = k_n
    return k_n
