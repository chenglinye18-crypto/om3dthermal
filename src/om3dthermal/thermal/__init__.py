"""Internal face thermal conductance for the discretised mesh.

This module attaches material physics to the geometry produced by
:mod:`om3dthermal.discretization`. It does **not** solve a thermal
problem; it only computes the per-edge conductance numbers a future
KCL / steady-state solver would consume.

Scope:

- material local conductivity tensor ``k_local = (kx, ky, kz)``;
- material rotation matrix ``R`` (axis-aligned 0/90/180/270 only);
- normal conductivity ``k_n = n^T K_global n`` for axis-aligned faces;
- two-point face conductance
  ``G_ab = A / (d_a/k_na + R''_ab + d_b/k_nb)``;
- optional per-material-pair interface areal resistance ``R''``.

Explicit non-goals:

- boundary conditions, power mapping, KCL matrix assembly,
  temperature solving;
- arbitrary-angle anisotropic FVM with full off-diagonal flux
  coupling (only signed axis permutations are supported).
"""
from .conductance import ConductanceTable, build_conductance_table
from .errors import (
    InvalidRotationMatrixError,
    MissingThermalConductivityError,
    UnsupportedMaterialRotationError,
)
from .interfaces import InterfaceResistanceQuery, InterfaceResistanceRegistry
from .tensors import (
    canonical_rotation_key,
    global_conductivity_tensor,
    is_signed_axis_permutation,
    normal_conductivity,
    validate_rotation_matrix,
)

__all__ = [
    "ConductanceTable",
    "InterfaceResistanceQuery",
    "InterfaceResistanceRegistry",
    "InvalidRotationMatrixError",
    "MissingThermalConductivityError",
    "UnsupportedMaterialRotationError",
    "build_conductance_table",
    "canonical_rotation_key",
    "global_conductivity_tensor",
    "is_signed_axis_permutation",
    "normal_conductivity",
    "validate_rotation_matrix",
]
