"""Material local conductivity tensor, rotation matrix, and normal
conductivity for axis-aligned faces.

The two-point face conductance used in this stage assumes:

- the material's local conductivity tensor is diagonal
  ``K_local = diag(kx, ky, kz)``;
- the material is rotated to world coordinates by a 3x3 rotation
  matrix ``R`` so that ``K_global = R K_local R^T``;
- the face is axis-aligned, so the normal ``n`` is one of the standard
  basis vectors and the normal conductivity collapses to
  ``k_n = sum_m k_local[m] * R[n,m]^2``.

This last form is what makes the cache key
``(material, canonical_rotation, axis)`` cheap: the full 3x3 tensor
``K_global`` is never multiplied for a face whose normal is along a
global axis.

Arbitrary-angle rotations are rejected via
:class:`UnsupportedMaterialRotationError` because the
two-point face conductance does not include the tangential coupling
terms a full anisotropic finite-volume method would.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .errors import InvalidRotationMatrixError, UnsupportedMaterialRotationError


# Tolerances for the rotation checks. The matrices we accept are
# signed axis permutations (each row / column is exactly one +/-1 and
# the rest are 0) so a 1e-9 abs tolerance is more than enough.
_ROTATION_ABS_TOL = 1e-9


def validate_rotation_matrix(rotation: Sequence[Sequence[float]]
                             ) -> list[list[float]]:
    """Validate that ``rotation`` is a 3x3 signed axis permutation.

    A signed axis permutation is a 3x3 real matrix ``R`` such that
    every row and every column has exactly one entry of magnitude 1
    and the rest are 0, with ``det(R) = +1``. This is the class of
    rotation matrices the two-point face conductance in this stage
    supports (0/90/180/270-degree axis-aligned rotations); arbitrary
    continuous rotations are rejected because the ``k_n`` projection
    does not include the tangential coupling terms a full anisotropic
    FVM would.

    NaN, infinity, non-3x3 shapes, non-orthogonal matrices, negative
    determinants, and matrices with mixed-magnitude entries (e.g. a
    45-degree rotation) are all rejected.

    Raises:
        InvalidRotationMatrixError: with a message explaining the
            failure mode.
    """
    if rotation is None:
        raise InvalidRotationMatrixError("rotation is None", rotation)
    if not isinstance(rotation, (list, tuple)):
        raise InvalidRotationMatrixError(
            f"rotation must be a 3x3 sequence, got {type(rotation).__name__}",
            rotation)
    if len(rotation) != 3:
        raise InvalidRotationMatrixError(
            f"rotation must have 3 rows, got {len(rotation)}", rotation)
    matrix: list[list[float]] = []
    for i, row in enumerate(rotation):
        if not isinstance(row, (list, tuple)):
            raise InvalidRotationMatrixError(
                f"rotation row {i} must be a sequence, got "
                f"{type(row).__name__}", rotation)
        if len(row) != 3:
            raise InvalidRotationMatrixError(
                f"rotation row {i} must have 3 columns, got {len(row)}",
                rotation)
        floats: list[float] = []
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidRotationMatrixError(
                    f"rotation[{i}][{j}] must be a real number, got "
                    f"{type(value).__name__}", rotation)
            value_f = float(value)
            if math.isnan(value_f) or math.isinf(value_f):
                raise InvalidRotationMatrixError(
                    f"rotation[{i}][{j}] must be finite, got {value_f}",
                    rotation)
            floats.append(value_f)
        matrix.append(floats)

    # Row/column structure: each row must have exactly one entry
    # whose magnitude is 1 (within tolerance) and the rest must be 0.
    # This is what excludes arbitrary continuous rotations (e.g.
    # 45 degrees about x) and shapes the matrix into a signed axis
    # permutation.
    for i in range(3):
        near_one = 0
        near_zero = 0
        for v in matrix[i]:
            if abs(abs(v) - 1.0) <= _ROTATION_ABS_TOL:
                near_one += 1
            elif abs(v) <= _ROTATION_ABS_TOL:
                near_zero += 1
            else:
                raise InvalidRotationMatrixError(
                    f"rotation row {i} entry {v} is not 0 or ±1; "
                    "this stage only supports signed axis permutations",
                    rotation)
        if near_one != 1 or near_zero != 2:
            raise InvalidRotationMatrixError(
                f"rotation row {i} must have exactly one ±1 entry; "
                f"got {near_one} ±1 and {near_zero} 0", rotation)
    for j in range(3):
        near_one = 0
        near_zero = 0
        for i in range(3):
            v = matrix[i][j]
            if abs(abs(v) - 1.0) <= _ROTATION_ABS_TOL:
                near_one += 1
            elif abs(v) <= _ROTATION_ABS_TOL:
                near_zero += 1
            else:
                # Already caught in the row pass; keep this branch for
                # completeness.
                raise InvalidRotationMatrixError(
                    f"rotation column {j} entry {v} is not 0 or ±1",
                    rotation)
        if near_one != 1 or near_zero != 2:
            raise InvalidRotationMatrixError(
                f"rotation column {j} must have exactly one ±1 entry; "
                f"got {near_one} ±1 and {near_zero} 0", rotation)

    # Orthogonality (R^T R = I) is implied by the permutation check
    # above but we re-assert it for clarity.
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[i][k] * matrix[j][k] for k in range(3))
            target = 1.0 if i == j else 0.0
            if abs(dot - target) > _ROTATION_ABS_TOL:
                raise InvalidRotationMatrixError(
                    f"rotation is not orthogonal: R^T R[{i}][{j}] = {dot} "
                    f"(expected {target})", rotation)

    # det = +1. A signed axis permutation with a single -1 per row
    # has det = (-1)^(number_of_-1s); the row/column pass above
    # allows either sign, so we still check the determinant.
    det = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(det - 1.0) > _ROTATION_ABS_TOL:
        raise InvalidRotationMatrixError(
            f"rotation determinant must be +1, got {det}", rotation)

    return matrix


def is_signed_axis_permutation(rotation: Sequence[Sequence[float]]) -> bool:
    """Return ``True`` iff ``rotation`` is a signed axis permutation,
    i.e. every row and every column has exactly one entry of magnitude
    1 and the rest are exactly 0. Identity, 90 / 180 / 270 degree
    rotations about any axis, and any composition thereof, are signed
    axis permutations; arbitrary-angle rotations are not.
    """
    try:
        matrix = validate_rotation_matrix(rotation)
    except InvalidRotationMatrixError:
        return False
    for i in range(3):
        for j in range(3):
            value = matrix[i][j]
            if abs(value) < _ROTATION_ABS_TOL:
                continue
            if abs(abs(value) - 1.0) > _ROTATION_ABS_TOL:
                return False
    # Verify exactly one non-zero per row and per column.
    for i in range(3):
        nonzero = sum(1 for v in matrix[i] if abs(v) > _ROTATION_ABS_TOL)
        if nonzero != 1:
            return False
    for j in range(3):
        nonzero = sum(1 for i in range(3) if abs(matrix[i][j]) > _ROTATION_ABS_TOL)
        if nonzero != 1:
            return False
    return True


def canonical_rotation_key(rotation: Sequence[Sequence[float]]
                          ) -> tuple[int, ...]:
    """Encode a signed axis permutation as a flat 9-int tuple suitable
    for use as a dict key.

    Each entry is one of ``{-1, 0, 1}``. The encoding is *normalised*
    by the convention that the first non-zero entry of each row is
    positive when possible (so identity is the canonical form rather
    than ``-I``). This deduplicates ``R`` and ``-R`` (which produce the
    same ``k_n``) into a single cache slot.
    """
    matrix = validate_rotation_matrix(rotation)
    canonical: list[int] = []
    for row in matrix:
        # Force a sign on the row so R and -R collapse together.
        leading_sign = 0
        for v in row:
            if abs(v) > _ROTATION_ABS_TOL:
                leading_sign = 1 if v > 0 else -1
                break
        if leading_sign == 0:
            raise InvalidRotationMatrixError(
                "rotation has a zero row; cannot canonicalise", rotation)
        for v in row:
            if abs(v) <= _ROTATION_ABS_TOL:
                canonical.append(0)
            elif abs(v - 1.0) <= _ROTATION_ABS_TOL:
                canonical.append(leading_sign)
            elif abs(v + 1.0) <= _ROTATION_ABS_TOL:
                canonical.append(-leading_sign)
            else:
                raise InvalidRotationMatrixError(
                    f"non-axis-permutation entry {v} cannot be canonicalised",
                    rotation)
    return tuple(canonical)


def global_conductivity_tensor(k_local: Sequence[float],
                               rotation: Sequence[Sequence[float]]
                               ) -> list[list[float]]:
    """Return ``K_global = R K_local R^T`` as a 3x3 list of lists.

    ``k_local`` must be a 3-tuple of strictly positive real numbers
    (the unit is W/(m*K)). The rotation is validated; arbitrary-angle
    matrices are not (use :func:`is_signed_axis_permutation` first if
    you need the gating logic elsewhere).
    """
    if len(k_local) != 3:
        raise ValueError(f"k_local must have 3 components, got {len(k_local)}")
    for i, k in enumerate(k_local):
        if not (k > 0) or math.isnan(k) or math.isinf(k):
            raise ValueError(
                f"k_local[{i}] must be a strictly positive finite number, "
                f"got {k}")
    matrix = validate_rotation_matrix(rotation)
    k_global = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = 0.0
            for m in range(3):
                for n in range(3):
                    s += matrix[i][m] * (k_local[m] if m == n else 0.0) * matrix[j][n]
            k_global[i][j] = s
    return k_global


def normal_conductivity(k_local: Sequence[float],
                        rotation: Sequence[Sequence[float]],
                        axis: int | str) -> float:
    """Return ``k_n = n^T K_global n`` for an axis-aligned face.

    ``axis`` may be ``0`` / ``1`` / ``2`` or ``"x"`` / ``"y"`` / ``"z"``.
    The face normal is the corresponding standard basis vector. Only
    signed axis permutations are accepted; arbitrary-angle rotations
    raise :class:`UnsupportedMaterialRotationError`.
    """
    if not is_signed_axis_permutation(rotation):
        raise UnsupportedMaterialRotationError(rotation)
    if isinstance(axis, str):
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            raise ValueError(
                f"axis must be one of 'x'/'y'/'z' or 0/1/2, got {axis!r}")
        n_axis = axis_map[axis]
    elif isinstance(axis, int) and not isinstance(axis, bool):
        if axis not in (0, 1, 2):
            raise ValueError(
                f"axis must be one of 'x'/'y'/'z' or 0/1/2, got {axis!r}")
        n_axis = axis
    else:
        raise TypeError(
            f"axis must be str or int, got {type(axis).__name__}")
    if len(k_local) != 3:
        raise ValueError(f"k_local must have 3 components, got {len(k_local)}")
    matrix = validate_rotation_matrix(rotation)
    # k_n = sum_m k_local[m] * R[n, m]^2
    k_n = 0.0
    for m in range(3):
        k_n += k_local[m] * matrix[n_axis][m] ** 2
    if not (k_n > 0) or math.isnan(k_n) or math.isinf(k_n):
        raise ValueError(
            f"normal conductivity is not positive/finite: k_n={k_n} "
            f"(k_local={k_local}, axis={n_axis})")
    return k_n
