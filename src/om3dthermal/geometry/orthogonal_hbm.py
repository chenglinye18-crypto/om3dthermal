"""Parametric paper-aligned geometry for an orthogonal MOSAIC HBM cube."""

from __future__ import annotations

from om3dthermal.config import SimulationConfig

from .horizontal_columns import HorizontalColumnsBuilder, _LENGTH_TOL, _boxes_overlap_3d
from .primitives import AxisAlignedBox
from .scene import Scene


# Local die thickness is local z. This cyclic signed-axis permutation maps
# local z onto global +x (the 30 mm array direction), local x (die width)
# onto global +y, and local y (die height) onto global +z.
ORTHOGONAL_DIE_ROTATION = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


class OrthogonalHBMBuilder:
    """Build foundation/GPU, 98 vertical layered dies, mold, and TIM/Lid."""

    def __init__(self, config: SimulationConfig):
        if config.orthogonal_hbm is None:
            raise ValueError("OrthogonalHBMBuilder requires orthogonal_hbm config")
        self.config = config
        # Reuse the established stack emission path for canonical horizontal
        # package layers. Only the cube itself needs a dedicated template.
        self._horizontal = HorizontalColumnsBuilder(config)
        self.scene = self._horizontal.scene

    def _add_box(self, *, name: str, material: str,
                 x0: float, x1: float, y0: float, y1: float,
                 z0: float, z1: float, component: str, source_path: str,
                 tags: dict | None = None,
                 rotation=ORTHOGONAL_DIE_ROTATION) -> None:
        self.scene.add(AxisAlignedBox(
            name=name, material=material,
            x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
            tags={**(tags or {}), "component": component},
            source_path=source_path, rotation=rotation,
        ))

    def _emit_mold_partition(self, *, cube, array_x0: float, array_x1: float,
                             array_y0: float, array_y1: float,
                             array_z0: float, array_z1: float,
                             cube_z0: float, cube_z1: float) -> None:
        """Tile the complement of the die-array prism inside the cube."""
        material = self.config.orthogonal_hbm.background_material
        specs = [
            ("y_minus", cube.x0, cube.x1, cube.y0, array_y0, cube_z0, cube_z1),
            ("y_plus", cube.x0, cube.x1, array_y1, cube.y1, cube_z0, cube_z1),
            ("x_minus", cube.x0, array_x0, array_y0, array_y1, cube_z0, cube_z1),
            ("x_plus", array_x1, cube.x1, array_y0, array_y1, cube_z0, cube_z1),
            ("z_minus", array_x0, array_x1, array_y0, array_y1, cube_z0, array_z0),
            ("z_plus", array_x0, array_x1, array_y0, array_y1, array_z1, cube_z1),
        ]
        for side, x0, x1, y0, y1, z0, z1 in specs:
            if x1 - x0 <= _LENGTH_TOL or y1 - y0 <= _LENGTH_TOL or z1 - z0 <= _LENGTH_TOL:
                continue
            self._add_box(
                name=f"orthogonal_hbm.mold_{side}", material=material,
                x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                component="orthogonal_hbm:mold",
                source_path=f"orthogonal_hbm.background.{side}",
                tags={"role": "cube_background", "priority": 0},
                rotation=((1.0, 0.0, 0.0),
                          (0.0, 1.0, 0.0),
                          (0.0, 0.0, 1.0)),
            )

    def _validate_no_overlap(self) -> None:
        for index, first in enumerate(self.scene.boxes):
            for second in self.scene.boxes[index + 1:]:
                if _boxes_overlap_3d(first, second):
                    raise ValueError(
                        f"orthogonal geometry overlap: {first.name!r} and {second.name!r}")

    def build(self) -> Scene:
        cfg = self.config
        orthogonal = cfg.orthogonal_hbm
        die = orthogonal.memory_die
        cube = cfg.footprints[orthogonal.cube_footprint]

        z = 0.0
        foundation_fp = cfg.footprints[orthogonal.foundation.footprint]
        z = self._horizontal._add_stack(
            orthogonal.foundation.stack, foundation_fp, z, "foundation",
            "orthogonal_hbm.foundation.stack")
        gpu_fp = cfg.footprints[orthogonal.gpu.footprint]
        z = self._horizontal._add_stack(
            orthogonal.gpu.stack, gpu_fp, z, "gpu",
            "orthogonal_hbm.gpu.stack")

        adhesive_z0 = z
        adhesive_z1 = adhesive_z0 + orthogonal.adhesive.thickness
        self._add_box(
            name="orthogonal_hbm.adhesive",
            material=orthogonal.adhesive.material,
            x0=cube.x0, x1=cube.x1, y0=cube.y0, y1=cube.y1,
            z0=adhesive_z0, z1=adhesive_z1,
            component="orthogonal_hbm:adhesive",
            source_path="orthogonal_hbm.adhesive",
            tags={"role": "cube_adhesive", "priority": 9},
            rotation=((1.0, 0.0, 0.0),
                      (0.0, 1.0, 0.0),
                      (0.0, 0.0, 1.0)),
        )
        z = adhesive_z1

        cube_z0 = z
        cube_z1 = cube_z0 + orthogonal.cube_height
        array_length = die.count * die.thickness
        array_x0 = cube.center_x - array_length / 2.0
        array_x1 = array_x0 + array_length
        array_y0 = cube.center_y - die.width / 2.0
        array_y1 = array_y0 + die.width
        array_z0 = cube_z0 + (orthogonal.cube_height - die.height) / 2.0
        array_z1 = array_z0 + die.height

        self._emit_mold_partition(
            cube=cube, array_x0=array_x0, array_x1=array_x1,
            array_y0=array_y0, array_y1=array_y1,
            array_z0=array_z0, array_z1=array_z1,
            cube_z0=cube_z0, cube_z1=cube_z1)

        for die_index in range(1, die.count + 1):
            die_name = f"die_{die_index:03d}"
            component = f"orthogonal_hbm:{die_name}"
            cursor = array_x0 + (die_index - 1) * die.thickness
            die_x0 = cursor
            for layer_index, layer in enumerate(die.layers):
                layer_x1 = cursor + layer.thickness
                self._add_box(
                    name=f"{component}.{layer.name}", material=layer.material,
                    x0=cursor, x1=layer_x1, y0=array_y0, y1=array_y1,
                    z0=array_z0, z1=array_z1, component=component,
                    source_path=(f"orthogonal_hbm.memory_die.stack[{layer_index}]"
                                 f"#die={die_index}"),
                    tags={
                        "role": layer.role,
                        "die_index": die_index,
                        "layer": layer.name,
                        "layer_index": layer_index,
                        "orientation": "die_plane_perpendicular_to_gpu",
                        "priority": 10,
                    },
                )
                cursor = layer_x1
            expected_x1 = die_x0 + die.thickness
            if abs(cursor - expected_x1) > _LENGTH_TOL:
                raise ValueError(f"{die_name} layer stack does not equal die thickness")

        self.scene.stack_heights["orthogonal_memory_die_thickness"] = die.thickness
        self.scene.stack_heights["orthogonal_hbm_cube_height"] = orthogonal.cube_height

        top_fp = cfg.footprints[orthogonal.top.footprint]
        self._horizontal._add_stack(
            orthogonal.top.stack, top_fp, cube_z1, "top",
            "orthogonal_hbm.top.stack")
        self._validate_no_overlap()
        return self.scene
