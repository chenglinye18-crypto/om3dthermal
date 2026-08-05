"""Dedicated builder for footprint-plus-z-stack horizontal structures."""

from __future__ import annotations

from om3dthermal.config import SimulationConfig

from .primitives import AxisAlignedBox, Footprint
from .scene import Scene


class HorizontalColumnsBuilder:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.scene = Scene(stack_heights={
            name: stack.total_thickness for name, stack in config.stack_templates.items()
        })

    def _add_stack(self, stack_name: str, footprint: Footprint, z0: float,
                   component: str, source_path: str, *, priority: int = 10,
                   extra_tags: dict | None = None) -> float:
        cursor = z0
        layers = self.config.stack_templates[stack_name].expand()
        for index, layer in enumerate(layers):
            top = cursor + layer.thickness
            self.scene.add(AxisAlignedBox(
                name=f"{component}.{layer.name}", material=layer.material,
                x0=footprint.x0, x1=footprint.x1, y0=footprint.y0, y1=footprint.y1,
                z0=cursor, z1=top,
                tags={**layer.tags, **(extra_tags or {}), "component": component,
                      "stack": stack_name, "layer": layer.name, "layer_index": index,
                      "priority": priority},
                source_path=f"{source_path}.{layer.source_suffix}"))
            cursor = top
        self._validate_continuity(component, z0, cursor)
        return cursor

    def _validate_continuity(self, component: str, bottom: float, top: float) -> None:
        boxes = sorted(self.scene.filter(component=component), key=lambda box: box.z0)
        if not boxes:
            return
        tolerance = 1e-15
        if abs(boxes[0].z0 - bottom) > tolerance or abs(boxes[-1].z1 - top) > tolerance:
            raise ValueError(f"component {component!r} does not span its expected z range")
        for lower, upper in zip(boxes, boxes[1:]):
            if abs(lower.z1 - upper.z0) > tolerance:
                raise ValueError(f"component {component!r} has a z gap or overlap")

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
        zone_fp = cfg.footprints[memory.footprint]
        self.scene.add(AxisAlignedBox(
            name="memory_zone.background", material=memory.background_material,
            x0=zone_fp.x0, x1=zone_fp.x1, y0=zone_fp.y0, y1=zone_fp.y1,
            z0=memory_bottom, z1=memory_top,
            tags={"component": "memory_zone_background", "priority": memory.background_priority},
            source_path="horizontal.memory_zone.background_material"))

        for index, column in enumerate(memory.columns):
            footprint = cfg.footprints[column.footprint]
            component = f"memory_column:{column.name}"
            source = f"horizontal.memory_zone.columns[{index}]"
            if column.stack:
                column_top = self._add_stack(column.stack, footprint, memory_bottom, component,
                                             f"{source}.stack", priority=column.priority,
                                             extra_tags={**column.tags, "column": column.name})
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
            if column_top < memory_top - 1e-15:
                self.scene.add(AxisAlignedBox(
                    name=f"{column.name}.fill_above", material=column.fill_above,
                    x0=footprint.x0, x1=footprint.x1, y0=footprint.y0, y1=footprint.y1,
                    z0=column_top, z1=memory_top,
                    tags={"component": component, "column": column.name,
                          "fill_above": True, "priority": column.priority},
                    source_path=f"{source}.fill_above"))
                column_top = memory_top
            if abs(column_top - memory_top) > 1e-15:
                raise ValueError(f"column {column.name!r} does not match reference stack height")
            self._validate_continuity(component, memory_bottom, memory_top)

        z = memory_top
        top_fp = cfg.footprints[horizontal.top.footprint]
        self._add_stack(horizontal.top.stack, top_fp, z, "top", "horizontal.top.stack")
        return self.scene
