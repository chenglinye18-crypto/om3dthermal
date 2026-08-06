"""Error types for the thermal conductance layer.

All errors raised by the conductance computation are subclasses of
:class:`ThermalError` so callers can catch the whole family without
swallowing unrelated exceptions.
"""
from __future__ import annotations


class ThermalError(Exception):
    """Base class for every error raised by :mod:`om3dthermal.thermal`."""


class InvalidRotationMatrixError(ThermalError, ValueError):
    """Raised when a rotation matrix is not orthogonal, is not
    determinant +1, contains NaN / inf, or is not 3x3.

    The original (un-validated) matrix is attached as ``rotation`` for
    diagnostics.
    """

    def __init__(self, message: str, rotation) -> None:
        self.rotation = rotation
        super().__init__(message)


class UnsupportedMaterialRotationError(ThermalError, NotImplementedError):
    """Raised when a cell carries a rotation that is not a signed axis
    permutation.

    The two-point face conductance implemented here is
    ``G = A / (d_a/k_na + R'' + d_b/k_nb)`` with
    ``k_n = n^T K_global n``. That ``k_n`` projection captures the
    normal flux of an anisotropic tensor but does not include the
    tangential coupling terms a full anisotropic finite-volume method
    would. Supporting arbitrary rotation angles therefore requires a
    non-orthogonal / multi-point flux scheme that is out of scope of
    this stage; we refuse the configuration explicitly rather than
    silently producing wrong numbers.
    """

    def __init__(self, rotation, cell_id: int | None = None) -> None:
        self.rotation = rotation
        self.cell_id = cell_id
        suffix = f" (cell id {cell_id})" if cell_id is not None else ""
        super().__init__(
            f"cell{suffix} carries a rotation matrix that is not a signed "
            "axis permutation; the two-point face conductance in this stage "
            "only supports 0/90/180/270-degree axis-aligned rotations. "
            "Arbitrary-angle anisotropic flux requires a non-orthogonal / "
            "multi-point flux scheme, which is not implemented.")


class MissingThermalConductivityError(ThermalError, ValueError):
    """Raised when a cell's material has ``k_local is None`` and is
    therefore not eligible for conductance computation.

    The material name, cell id, parent box and (when known) adjacency
    edge id are attached for diagnostics.
    """

    def __init__(self, *, material: str, cell_id: int, parent_box: str,
                 edge_id: int | None = None) -> None:
        self.material = material
        self.cell_id = cell_id
        self.parent_box = parent_box
        self.edge_id = edge_id
        edge_suffix = (
            f" on adjacency edge {edge_id}" if edge_id is not None else "")
        super().__init__(
            f"material {material!r} (cell {cell_id}, parent box "
            f"{parent_box!r}) has no k_local; cannot compute "
            f"conductance{edge_suffix}")
