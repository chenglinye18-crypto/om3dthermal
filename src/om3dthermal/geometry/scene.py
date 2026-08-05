"""Collection, querying and serialization of geometry regions."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from om3dthermal.units import format_length

from .primitives import AxisAlignedBox


class Scene:
    def __init__(self, boxes: list[AxisAlignedBox] | None = None, *, stack_heights: dict[str, float] | None = None):
        self.boxes = list(boxes or [])
        self.stack_heights = dict(stack_heights or {})

    def add(self, box: AxisAlignedBox) -> None:
        if any(existing.id == box.id for existing in self.boxes):
            raise ValueError(f"duplicate box id {box.id}")
        self.boxes.append(box)

    @property
    def bounds(self) -> dict[str, float]:
        if not self.boxes:
            raise ValueError("empty scene has no bounds")
        return {axis + edge: func(getattr(box, axis + edge) for box in self.boxes)
                for axis in "xyz" for edge, func in (("0", min), ("1", max))}

    def filter(self, *, material: str | None = None, component: str | None = None,
               layer: str | None = None) -> list[AxisAlignedBox]:
        return [box for box in self.boxes
                if (material is None or box.material == material)
                and (component is None or box.tags.get("component") == component)
                and (layer is None or box.tags.get("layer") == layer or box.name == layer)]

    def summary(self) -> dict[str, Any]:
        if not self.boxes:
            return {"total_boxes": 0, "boxes_by_material": {}, "stack_heights_m": {},
                    "component_bounds_m": {}, "minimum_dimension_m": None, "maximum_dimension_m": None}
        component_boxes: dict[str, list[AxisAlignedBox]] = defaultdict(list)
        for box in self.boxes:
            component_boxes[str(box.tags.get("component", "unassigned"))].append(box)
        component_bounds = {}
        for component, boxes in sorted(component_boxes.items()):
            component_bounds[component] = {
                "x": [min(box.x0 for box in boxes), max(box.x1 for box in boxes)],
                "y": [min(box.y0 for box in boxes), max(box.y1 for box in boxes)],
                "z": [min(box.z0 for box in boxes), max(box.z1 for box in boxes)],
            }
        dimensions = [dimension for box in self.boxes for dimension in box.dimensions]
        return {
            "total_boxes": len(self.boxes),
            "boxes_by_material": dict(sorted(Counter(box.material for box in self.boxes).items())),
            "stack_heights_m": dict(sorted(self.stack_heights.items())),
            "stack_heights_display": {name: format_length(height) for name, height in sorted(self.stack_heights.items())},
            "component_bounds_m": component_bounds,
            "minimum_dimension_m": min(dimensions),
            "maximum_dimension_m": max(dimensions),
            "minimum_dimension_display": format_length(min(dimensions)),
            "maximum_dimension_display": format_length(max(dimensions)),
            "scene_bounds_m": self.bounds,
        }

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["id", "name", "material", "x0", "x1", "y0", "y1", "z0", "z1",
                  "tags", "source_path", "rotation"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for box in self.boxes:
                row = box.model_dump()
                row["tags"] = json.dumps(row["tags"], ensure_ascii=False, sort_keys=True)
                row["rotation"] = json.dumps(row["rotation"])
                writer.writerow(row)

    def write_summary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(self.summary(), stream, ensure_ascii=False, indent=2)
