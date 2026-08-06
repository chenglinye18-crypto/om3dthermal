"""Dedicated builder for footprint-plus-z-stack horizontal structures."""

from __future__ import annotations

from om3dthermal.config import ExpandedLayer, SimulationConfig

from .primitives import AxisAlignedBox, Footprint
from .scene import Scene


# ---------------------------------------------------------------------------
# Geometric tolerances.
#
# Length tolerances are absolute-only, intended to absorb the floating-point
# round-off introduced by SI-metre stack arithmetic (1 pm by default).
# Area tolerances combine an absolute floor with a relative ceiling so that
# very small regions (e.g. a 0.15 um FEOL layer x a 0.5 mm x-cell) still
# trigger the area checks but very large regions don't require unrealistic
# precision.
# ---------------------------------------------------------------------------
_LENGTH_TOL = 1e-12      # m
_AREA_ABS_TOL = 1e-24    # m^2
_REL_TOL = 1e-10


def _length_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(_LENGTH_TOL, _REL_TOL * max(abs(a), abs(b)))


def _area_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(_AREA_ABS_TOL, _REL_TOL * max(abs(a), abs(b)))


def _cluster_by(values: list[float], *, key, tol: float) -> list[list]:
    """Group items whose ``key(item)`` falls within ``tol`` of the first
    member of the current group. Returns the groups in ascending key order.
    """
    if not values:
        return []
    sorted_items = sorted(values, key=key)
    groups: list[list] = [[sorted_items[0]]]
    for item in sorted_items[1:]:
        if _length_close(key(item), key(groups[-1][0])):
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def validate_layer_partition(parent: Footprint, central: AxisAlignedBox,
                             fills: list[AxisAlignedBox]) -> None:
    """Verify that ``central`` plus ``fills`` partition ``parent`` in the
    layer's z slice, with no positive-area overlap and no uncovered gap.

    Raises ``ValueError`` on any violation. Fills are expected to share the
    central box's z range; the caller is responsible for any z sanity check.
    """
    if (central.z1 - central.z0) <= _LENGTH_TOL:
        raise ValueError("central box has zero z thickness")
    if ((central.x1 - central.x0) <= _LENGTH_TOL
            or (central.y1 - central.y0) <= _LENGTH_TOL):
        raise ValueError("central box has zero xy extent")
    z_lo, z_hi = central.z0, central.z1

    def _check_xy(name: str, box: AxisAlignedBox) -> None:
        if (box.x0 < parent.x0 - _LENGTH_TOL or box.x1 > parent.x1 + _LENGTH_TOL
                or box.y0 < parent.y0 - _LENGTH_TOL or box.y1 > parent.y1 + _LENGTH_TOL):
            raise ValueError(
                f"{name!r} extends outside parent footprint {parent.name!r}: "
                f"box=({box.x0}, {box.x1}, {box.y0}, {box.y1}) "
                f"parent=({parent.x0}, {parent.x1}, {parent.y0}, {parent.y1})")
        if not _length_close(box.z0, z_lo) or not _length_close(box.z1, z_hi):
            raise ValueError(
                f"{name!r} z range ({box.z0}, {box.z1}) does not match central z "
                f"({z_lo}, {z_hi})")
        if ((box.x1 - box.x0) <= _LENGTH_TOL
                or (box.y1 - box.y0) <= _LENGTH_TOL):
            raise ValueError(f"{name!r} has non-positive xy extent")

    _check_xy("central", central)
    for box in fills:
        _check_xy(f"fill {box.name!r}", box)

    def _overlap_xy(a: AxisAlignedBox, b: AxisAlignedBox) -> float:
        return (max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
                * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0)))

    if _overlap_xy(central, central) <= _AREA_ABS_TOL:
        raise ValueError("central box has zero area in the parent footprint")
    for i, a in enumerate(fills):
        if _overlap_xy(a, a) <= _AREA_ABS_TOL:
            raise ValueError(f"fill {a.name!r} has zero area")
        if _overlap_xy(central, a) > _AREA_ABS_TOL:
            raise ValueError(
                f"fill {a.name!r} overlaps central box by "
                f"{_overlap_xy(central, a)} m^2")
        for b in fills[i + 1:]:
            if _overlap_xy(a, b) > _AREA_ABS_TOL:
                raise ValueError(
                    f"fill {a.name!r} and fill {b.name!r} overlap by "
                    f"{_overlap_xy(a, b)} m^2")

    parent_area = (parent.x1 - parent.x0) * (parent.y1 - parent.y0)
    covered = (central.x1 - central.x0) * (central.y1 - central.y0)
    for box in fills:
        covered += (box.x1 - box.x0) * (box.y1 - box.y0)
    if not _area_close(parent_area, covered):
        raise ValueError(
            f"central + fills area {covered} does not match parent area "
            f"{parent_area} for {parent.name!r}")


class HorizontalColumnsBuilder:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.scene = Scene(stack_heights={
            name: stack.total_thickness for name, stack in config.stack_templates.items()
        })

    def _add_stack(self, stack_name: str, footprint: Footprint, z0: float,
                   component: str, source_path: str, *, priority: int = 10,
                   extra_tags: dict | None = None,
                   fill_material: str | None = None,
                   column_name: str | None = None) -> float:
        cursor = z0
        layers = self.config.stack_templates[stack_name].expand()
        for index, layer in enumerate(layers):
            top = cursor + layer.thickness
            self._emit_layer(
                layer=layer, footprint=footprint, z0=cursor, z1=top,
                component=component, source_path=f"{source_path}.{layer.source_suffix}",
                stack_name=stack_name, layer_index=index, priority=priority,
                extra_tags=extra_tags, fill_material=fill_material,
                column_name=column_name)
            cursor = top
        self._validate_continuity(component, z0, cursor)
        return cursor

    def _emit_layer(self, *, layer: ExpandedLayer, footprint: Footprint,
                    z0: float, z1: float, component: str, source_path: str,
                    stack_name: str, layer_index: int, priority: int,
                    extra_tags: dict | None,
                    fill_material: str | None,
                    column_name: str | None) -> None:
        inset = layer.lateral_inset
        base_tags = {**layer.tags, **(extra_tags or {}), "component": component,
                     "stack": stack_name, "layer": layer.name,
                     "layer_index": layer_index, "priority": priority}

        if inset is None or inset.is_zero():
            self.scene.add(AxisAlignedBox(
                name=f"{component}.{layer.name}", material=layer.material,
                x0=footprint.x0, x1=footprint.x1, y0=footprint.y0, y1=footprint.y1,
                z0=z0, z1=z1, tags=base_tags, source_path=source_path))
            return

        if fill_material is None:
            raise ValueError(
                f"layer {layer.name!r} has a non-zero lateral_inset but no "
                f"fill_material was provided for column {column_name!r}")

        inner_x0 = footprint.x0 + inset.x_minus
        inner_x1 = footprint.x1 - inset.x_plus
        inner_y0 = footprint.y0 + inset.y_minus
        inner_y1 = footprint.y1 - inset.y_plus
        if (inner_x1 - inner_x0) <= _LENGTH_TOL or (inner_y1 - inner_y0) <= _LENGTH_TOL:
            raise ValueError(
                f"lateral_inset on layer {layer.name!r} leaves no room for the "
                f"central entity in footprint {footprint.name!r}")

        central = AxisAlignedBox(
            name=f"{component}.{layer.name}", material=layer.material,
            x0=inner_x0, x1=inner_x1, y0=inner_y0, y1=inner_y1,
            z0=z0, z1=z1,
            tags={**base_tags, "lateral_inset_applied": True,
                  "parent_footprint": footprint.name},
            source_path=source_path)
        self.scene.add(central)

        fill_priority = priority - 1
        fill_base_tags = {
            "role": "lateral_fill", "fill_material": fill_material,
            "parent_layer": layer.name, "parent_column": column_name or "",
            "stack": stack_name, "layer": layer.name,
            "layer_index": layer_index, "priority": fill_priority,
            "component": component,
        }
        strip_specs: list[tuple[str, float, float, float, float]] = []
        if inset.x_minus > _LENGTH_TOL:
            strip_specs.append(("left", footprint.x0, inner_x0,
                                footprint.y0, footprint.y1))
        if inset.x_plus > _LENGTH_TOL:
            strip_specs.append(("right", inner_x1, footprint.x1,
                                footprint.y0, footprint.y1))
        if inset.y_minus > _LENGTH_TOL:
            strip_specs.append(("bottom", inner_x0, inner_x1,
                                footprint.y0, inner_y0))
        if inset.y_plus > _LENGTH_TOL:
            strip_specs.append(("top", inner_x0, inner_x1,
                                inner_y1, footprint.y1))

        fills: list[AxisAlignedBox] = []
        for side, sx0, sx1, sy0, sy1 in strip_specs:
            box = AxisAlignedBox(
                name=f"{component}.{layer.name}.fill_{side}",
                material=fill_material,
                x0=sx0, x1=sx1, y0=sy0, y1=sy1, z0=z0, z1=z1,
                tags={**fill_base_tags, "inset_side": side},
                source_path=source_path)
            self.scene.add(box)
            fills.append(box)

        validate_layer_partition(footprint, central, fills)

    def _validate_continuity(self, component: str, bottom: float, top: float) -> None:
        """Validate that the boxes assigned to ``component`` form a valid
        axis-aligned column: no 3D overlap between any two boxes, the first
        box starts at ``bottom`` and the last box ends at ``top``, and
        there is no z gap between adjacent z-levels.

        Per-layer lateral partitioning (a central box plus zero to four
        lateral fill boxes) is allowed to share a single z-level because
        the per-layer partition invariant is enforced separately by
        ``validate_layer_partition`` at emit time.

        z-level grouping uses a length-tolerance cluster rather than
        ``round()`` so the test never depends on the coordinate quantisation
        grid of the floating-point representation.
        """
        boxes = self.scene.filter(component=component)
        if not boxes:
            return
        z0_min = min(b.z0 for b in boxes)
        z1_max = max(b.z1 for b in boxes)
        if not _length_close(z0_min, bottom) or not _length_close(z1_max, top):
            raise ValueError(
                f"component {component!r} does not span its expected z range "
                f"[{bottom}, {top}] (got [{z0_min}, {z1_max}])")

        # No positive-volume 3D overlap between any two boxes in this component.
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                if _boxes_overlap_3d(a, b):
                    raise ValueError(
                        f"boxes {a.name!r} and {b.name!r} in component "
                        f"{component!r} overlap in 3D")

        # z-level continuity: cluster boxes by z0, every cluster must share
        # z1, and the next cluster's representative z0 must equal the
        # previous cluster's z1.
        groups = _cluster_by(boxes, key=lambda b: b.z0, tol=_LENGTH_TOL)
        prev_z1: float | None = None
        prev_repr: float | None = None
        for group in groups:
            z0 = group[0].z0
            z1_values = {b.z1 for b in group}
            if len(z1_values) != 1 or not _length_close(z1_values.pop(), group[0].z1):
                zs = sorted(b.z1 for b in group)
                raise ValueError(
                    f"component {component!r} has boxes at z0={z0} with "
                    f"inconsistent z1 values: {zs}")
            z1 = group[0].z1
            if prev_z1 is not None:
                if not _length_close(prev_z1, z0):
                    raise ValueError(
                        f"component {component!r} has a z gap: previous z-level "
                        f"ends at {prev_z1} (representative z0={prev_repr}), "
                        f"next z-level starts at {z0}")
            prev_z1 = z1
            prev_repr = z0

    def build(self) -> Scene:
        cfg = self.config
        horizontal = cfg.horizontal
        z = 0.0
        foundation_fp = cfg.footprints[horizontal.foundation.footprint]
        z = self._add_stack(horizontal.foundation.stack, foundation_fp, z, "foundation",
                            "horizontal.foundation.stack")
        gpu_fp = cfg.footprints[horizontal.gpu.footprint]
        z = self._add_stack(horizontal.gpu.stack, gpu_fp, z, "gpu", "horizontal.gpu.stack")

        memory = horizontal.memory_zone
        memory_bottom = z
        reference_height = cfg.stack_templates[memory.reference_stack].total_thickness
        memory_top = memory_bottom + reference_height
        # The memory_zone footprint is fully tiled by its columns (Fig. 3(a)):
        # 4 x 11x11 HBM corner footprints plus a central 8x22 thermal-silicon
        # footprint. The per-layer mold fill in each column now covers the
        # lateral cavity, so no memory_zone_background slab is emitted. The
        # mold material is taken from memory.background_material and applied
        # to every per-layer lateral fill.
        fill_material = memory.background_material

        for index, column in enumerate(memory.columns):
            footprint = cfg.footprints[column.footprint]
            component = f"memory_column:{column.name}"
            source = f"horizontal.memory_zone.columns[{index}]"
            if column.stack:
                column_top = self._add_stack(
                    column.stack, footprint, memory_bottom, component,
                    f"{source}.stack", priority=column.priority,
                    extra_tags={**column.tags, "column": column.name},
                    fill_material=fill_material, column_name=column.name)
            else:
                height = cfg.stack_templates[column.match_height_of].total_thickness
                column_top = memory_bottom + height
                self.scene.add(AxisAlignedBox(
                    name=column.name, material=column.material,
                    x0=footprint.x0, x1=footprint.x1, y0=footprint.y0, y1=footprint.y1,
                    z0=memory_bottom, z1=column_top,
                    tags={**column.tags, "component": component, "column": column.name,
                          "match_height_of": column.match_height_of, "priority": column.priority},
                    source_path=f"{source}.material"))
            if column_top < memory_top - _LENGTH_TOL:
                self.scene.add(AxisAlignedBox(
                    name=f"{column.name}.fill_above", material=column.fill_above,
                    x0=footprint.x0, x1=footprint.x1, y0=footprint.y0, y1=footprint.y1,
                    z0=column_top, z1=memory_top,
                    tags={"component": component, "column": column.name,
                          "fill_above": True, "priority": column.priority},
                    source_path=f"{source}.fill_above"))
                column_top = memory_top
            if not _length_close(column_top, memory_top):
                raise ValueError(f"column {column.name!r} does not match reference stack height")
            self._validate_continuity(component, memory_bottom, memory_top)

        z = memory_top
        top_fp = cfg.footprints[horizontal.top.footprint]
        self._add_stack(horizontal.top.stack, top_fp, z, "top", "horizontal.top.stack")
        return self.scene


def _boxes_overlap_3d(a: AxisAlignedBox, b: AxisAlignedBox) -> bool:
    """Return True iff ``a`` and ``b`` share a positive volume in 3D.

    The check is symmetric; uses the same length tolerance as the rest of
    the builder.
    """
    overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
    overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
    overlap_z = min(a.z1, b.z1) - max(a.z0, b.z0)
    return (overlap_x > _LENGTH_TOL
            and overlap_y > _LENGTH_TOL
            and overlap_z > _LENGTH_TOL)
