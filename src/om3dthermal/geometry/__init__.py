"""Axis-aligned geometry model.

The builder is intentionally not imported here: configuration models depend on
the primitives, while the builder depends on configuration models.
"""

from .primitives import AxisAlignedBox, Footprint
from .scene import Scene

__all__ = ["AxisAlignedBox", "Footprint", "Scene"]
