"""Boundary condition rules and the columnar ``BoundaryLinkTable``.

A boundary condition rule from the YAML is matched against every
``BoundaryFace`` produced by the discretiser; the highest-priority
match wins, ties are a config error, and unmatched faces fall back
to the block default (which must be ``adiabatic`` in this stage).
Adiabatic faces do not enter the active
``BoundaryLinkTable``; convection and fixed-temperature faces do,
and their per-link conductance includes the half-cell bulk
resistance so the unknown temperature lives at the cell centre.

The half-cell resistance is the same shape as the internal-edge
two-point conductance, but the external side is either an
ambient temperature with a heat transfer coefficient ``h`` or a
fixed surface temperature:

    R_cell_to_ambient = d_i / (k_n A) + R'' / A + 1 / (h A)   [convection]
    R_cell_to_face    = d_i / (k_n A) + R'' / A               [fixed T]
    G_i               = A / R                                  [W/K]

The "half distance" `d_i` is the cell's half-extent along the
face normal (matching what the discretiser stores on
``BoundaryFace``'s parent cell).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..config import (
    BoundaryConditionConfig,
    BoundarySelector,
    ThermalBoundaryConditionsConfig,
)
from ..discretization.models import BoundaryFace, ThermalCell


_BOUNDARY_KIND_INT = {
    "convection": 1,
    "fixed_temperature": 2,
}


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _face_matches_selector(face: BoundaryFace, cell: ThermalCell,
                           selector: BoundarySelector) -> bool:
    """Return True iff every non-``None`` selector field agrees
    with the face / cell.
    """
    if selector.component is not None and cell.component != selector.component:
        return False
    if selector.material is not None and cell.material != selector.material:
        return False
    if selector.layer is not None and cell.parent_box_name != selector.layer:
        return False
    if selector.axis is not None and face.axis != selector.axis:
        return False
    if selector.side is not None and face.side != selector.side:
        return False
    if (selector.classification is not None
            and face.classification != selector.classification):
        return False
    if selector.tags:
        for k, v in selector.tags.items():
            if cell.tags.get(k) != v:
                return False
    return True


def select_boundary_rule(
    face: BoundaryFace,
    cell: ThermalCell,
    rules: Sequence[BoundaryConditionConfig],
) -> tuple[int, BoundaryConditionConfig] | None:
    """Return the (index, rule) of the highest-priority matching
    rule, or ``None`` if no rule matches.

    A tie on priority between two distinct matching rules is a
    config error: it would silently depend on the order of the
    YAML list. The function raises ``ValueError`` instead.
    """
    matches: list[tuple[int, BoundaryConditionConfig]] = []
    for index, rule in enumerate(rules):
        if _face_matches_selector(face, cell, rule.selector):
            matches.append((index, rule))
    if not matches:
        return None
    matches.sort(key=lambda kv: (-kv[1].selector.priority, kv[0]))
    best_index, best_rule = matches[0]
    for other_index, other_rule in matches[1:]:
        if other_rule.selector.priority == best_rule.selector.priority:
            raise ValueError(
                f"boundary face {face.id} (cell {cell.id}, "
                f"parent box {cell.parent_box_name!r}) matches multiple "
                f"rules with the same priority "
                f"{best_rule.selector.priority}: rules "
                f"{best_index} ({best_rule.name!r}) and "
                f"{other_index} ({other_rule.name!r}). Disambiguate "
                "with a stricter selector or different priority.")
    return best_index, best_rule


# ---------------------------------------------------------------------------
# Cell k_n for the boundary face normal
# ---------------------------------------------------------------------------

def _cell_k_n(cell: ThermalCell, face: BoundaryFace, materials
              ) -> float:
    """Resolve the per-cell normal conductivity for a boundary face.

    The boundary face inherits the cell's normal axis. We re-use
    the same k_n cache key the conductance stage uses, so a
    boundary rule that references the same material / rotation /
    axis as an internal edge shares the same ``k_n`` value.
    """
    from .tensors import canonical_rotation_key, normal_conductivity
    material = materials[cell.material]
    if material.k_local is None:
        from .errors import MissingThermalConductivityError
        raise MissingThermalConductivityError(
            material=cell.material, cell_id=cell.id,
            parent_box=cell.parent_box_name)
    rotation_key = canonical_rotation_key(cell.rotation)
    axis_code = {"x": 0, "y": 1, "z": 2}[face.axis]
    return normal_conductivity(material.k_local, cell.rotation, axis_code)


def _face_half_distance(cell: ThermalCell, face: BoundaryFace) -> float:
    """Half-extent of ``cell`` along the face normal axis."""
    if face.axis == "x":
        return cell.size_x / 2
    if face.axis == "y":
        return cell.size_y / 2
    return cell.size_z / 2


# ---------------------------------------------------------------------------
# BoundaryLinkTable
# ---------------------------------------------------------------------------

@dataclass
class BoundaryLinkTable:
    """Columnar per-boundary-face thermal link inventory.

    Only non-adiabatic faces appear in the table. The columns are
    aligned with the parent ``BoundaryFace`` list via
    ``boundary_face_id``; ``cell_id`` is the face's parent cell.
    """

    boundary_face_id: np.ndarray   # int64
    cell_id: np.ndarray            # int64
    kind: np.ndarray               # int8 (1=convection, 2=fixed_temperature)
    axis: np.ndarray               # int8 (0=x, 1=y, 2=z)
    side: np.ndarray               # int8 (0=minus, 1=plus)
    face_area_m2: np.ndarray       # float64
    half_distance_m: np.ndarray    # float64
    k_normal_W_mK: np.ndarray      # float64
    areal_resistance_m2K_W: np.ndarray     # float64
    external_film_resistance_m2K_W: np.ndarray  # float64 (0 for fixed T)
    conductance_W_K: np.ndarray    # float64
    reference_temperature_K: np.ndarray  # float64
    rule_index: np.ndarray         # int32 (-1 for default; default is adiabatic so this is never -1 here)

    @property
    def link_count(self) -> int:
        return int(self.boundary_face_id.shape[0])


def build_boundary_link_table(
    boundary_faces: Iterable[BoundaryFace],
    cells: Sequence[ThermalCell],
    materials: dict,
    config: ThermalBoundaryConditionsConfig,
) -> BoundaryLinkTable:
    """Match every face against the rules and emit the active link
    table.

    Faces that match a ``convection`` or ``fixed_temperature`` rule
    become an active link. Faces that match nothing, or match an
    adiabatic rule, contribute nothing (they are not in the
    table). The anchored-component check (in
    :mod:`om3dthermal.thermal.steady_state`) is responsible for
    refusing to solve a network that has no active link in some
    component.
    """
    cell_by_id = {c.id: c for c in cells}
    bfid: list[int] = []
    cell_ids: list[int] = []
    kinds: list[int] = []
    axes: list[int] = []
    sides: list[int] = []
    areas: list[float] = []
    half_d: list[float] = []
    k_n_arr: list[float] = []
    rpp: list[float] = []
    rfilm: list[float] = []
    g_arr: list[float] = []
    tref: list[float] = []
    rule_idx: list[int] = []
    for face in boundary_faces:
        cell = cell_by_id[face.cell_id]
        match = select_boundary_rule(face, cell, config.rules)
        if match is None:
            # Default is always adiabatic, so unmatched faces are
            # silently skipped.
            continue
        rule_index, rule = match
        if rule.kind == "adiabatic":
            continue
        k_n = _cell_k_n(cell, face, materials)
        d = _face_half_distance(cell, face)
        # R_pp + R_external are the per-area contributions; the
        # half-cell bulk resistance lives in d/k_n which has units
        # of m^2*K/W already (m / (W/m/K) = m^2*K/W). R_pp is
        # m^2*K/W. R_external is 1/h for convection and 0 for fixed T.
        if rule.kind == "convection":
            h = rule.heat_transfer_coefficient
            r_ext = 1.0 / h
            t_ref = rule.ambient_temperature
        else:  # fixed_temperature
            r_ext = 0.0
            t_ref = rule.surface_temperature
        r_total = d / k_n + rule.areal_resistance + r_ext
        conductance = face.area / r_total
        if not (conductance > 0) or not np.isfinite(conductance):
            raise ValueError(
                f"boundary link {face.id} has non-positive or "
                f"non-finite conductance {conductance} (A={face.area}, "
                f"d={d}, k_n={k_n}, R''={rule.areal_resistance}, "
                f"R_ext={r_ext}, kind={rule.kind!r})")
        bfid.append(face.id)
        cell_ids.append(cell.id)
        kinds.append(_BOUNDARY_KIND_INT[rule.kind])
        axes.append({"x": 0, "y": 1, "z": 2}[face.axis])
        sides.append(0 if face.side == "minus" else 1)
        areas.append(face.area)
        half_d.append(d)
        k_n_arr.append(k_n)
        rpp.append(rule.areal_resistance)
        rfilm.append(r_ext)
        g_arr.append(conductance)
        tref.append(t_ref)
        rule_idx.append(rule_index)
    return BoundaryLinkTable(
        boundary_face_id=np.array(bfid, dtype=np.int64),
        cell_id=np.array(cell_ids, dtype=np.int64),
        kind=np.array(kinds, dtype=np.int8),
        axis=np.array(axes, dtype=np.int8),
        side=np.array(sides, dtype=np.int8),
        face_area_m2=np.array(areas, dtype=np.float64),
        half_distance_m=np.array(half_d, dtype=np.float64),
        k_normal_W_mK=np.array(k_n_arr, dtype=np.float64),
        areal_resistance_m2K_W=np.array(rpp, dtype=np.float64),
        external_film_resistance_m2K_W=np.array(rfilm, dtype=np.float64),
        conductance_W_K=np.array(g_arr, dtype=np.float64),
        reference_temperature_K=np.array(tref, dtype=np.float64),
        rule_index=np.array(rule_idx, dtype=np.int32),
    )
