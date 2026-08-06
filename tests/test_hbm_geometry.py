import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from om3dthermal.cli import build
from om3dthermal.config import SimulationConfig, load_config
from om3dthermal.geometry.horizontal_columns import HorizontalColumnsBuilder

CONFIG = Path(__file__).parents[1] / "configs" / "hbm_on_gpu_12hi.yaml"

HBM_COLUMN_NAMES = ("hbm_left_top", "hbm_left_bottom", "hbm_right_top", "hbm_right_bottom")


def _boxes_by_role(scene, *, component: str, role: str) -> list:
    return [b for b in scene.filter(component=component) if b.tags.get("role") == role]


def test_geometry_z_order_continuity_and_identity_rotation():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    tolerance = 1e-12
    for component in {box.tags.get("component") for box in scene.boxes
                      if str(box.tags.get("component", "")).startswith("memory_column:")}:
        boxes = list(scene.filter(component=component))
        # Total z extent must match the HBM reference height (775 um) for
        # HBM columns; thermal_silicon matches it by construction.
        z0_min = min(b.z0 for b in boxes)
        z1_max = max(b.z1 for b in boxes)
        assert z1_max - z0_min == pytest.approx(775e-6)
        # No 3D overlap between any two boxes in the same column.
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert not (
                    a.x1 - b.x0 > tolerance and a.x0 - b.x1 < -tolerance
                    and a.y1 - b.y0 > tolerance and a.y0 - b.y1 < -tolerance
                    and a.z1 - b.z0 > tolerance and a.z0 - b.z1 < -tolerance
                ), f"{a.name} and {b.name} overlap in 3D"
        # z contiguity: adjacent z-levels must meet exactly.
        z_levels = sorted({round(b.z0, 9) for b in boxes})
        for i in range(1, len(z_levels)):
            prev_z1 = max(b.z1 for b in boxes if round(b.z0, 9) == z_levels[i - 1])
            assert abs(prev_z1 - z_levels[i]) <= tolerance
    assert all(box.rotation == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
               for box in scene.boxes)


def test_each_hbm_column_has_twelve_dram_si_boxes():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMN_NAMES:
        assert len(_boxes_by_role(scene, component=f"memory_column:{name}", role="dram_si")) == 12
        assert len(_boxes_by_role(scene, component=f"memory_column:{name}", role="dram_beol")) == 12
        assert len(_boxes_by_role(scene, component=f"memory_column:{name}", role="hybrid_bonding")) == 12


def test_each_hbm_column_has_one_ubump_and_one_base_pair():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMN_NAMES:
        component = f"memory_column:{name}"
        ubump = [b for b in scene.filter(component=component) if b.tags.get("role") == "gpu_hbm_interface"]
        base  = [b for b in scene.filter(component=component) if b.tags.get("role") == "hbm_base"]
        assert len(ubump) == 1 and (ubump[0].z1 - ubump[0].z0) == pytest.approx(40e-6)
        assert len(base) == 2
        assert sorted(b.z1 - b.z0 for b in base) == pytest.approx(sorted([5e-6, 50e-6]))


def test_hbm_column_total_height_is_775_um():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMN_NAMES:
        boxes = scene.filter(component=f"memory_column:{name}")
        assert max(b.z1 for b in boxes) - min(b.z0 for b in boxes) == pytest.approx(775e-6)


def test_thermal_silicon_column_geometry():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    component = "memory_column:thermal_silicon"
    boxes = sorted(scene.filter(component=component), key=lambda b: b.z0)
    # Total height must match reference (hbm_12hi = 775 um).
    assert boxes[-1].z1 - boxes[0].z0 == pytest.approx(775e-6)
    # Bottom 1 um is Oxide.
    oxide = [b for b in boxes if b.material == "Oxide"]
    assert len(oxide) == 1
    assert oxide[0].z1 - oxide[0].z0 == pytest.approx(1e-6)
    # The remaining 774 um is the Thermal_Silicon entity itself.
    silicon = [b for b in boxes if b.material == "Thermal_Silicon"]
    assert len(silicon) == 1
    assert silicon[0].z1 - silicon[0].z0 == pytest.approx(774e-6)
    # Oxide must sit directly below Thermal_Silicon.
    assert oxide[0].z1 == pytest.approx(silicon[0].z0)


def test_foundation_gpu_memory_top_are_derived_in_order():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    foundation_top = max(box.z1 for box in scene.filter(component="foundation"))
    gpu = scene.filter(component="gpu")
    # The memory zone no longer emits a single background slab: its z range
    # is now defined by the union of the memory_column boxes.
    memory_boxes = [b for b in scene.boxes
                    if str(b.tags.get("component", "")).startswith("memory_column:")]
    top = scene.filter(component="top")
    assert min(box.z0 for box in gpu) == pytest.approx(foundation_top)
    assert min(b.z0 for b in memory_boxes) == pytest.approx(max(box.z1 for box in gpu))
    assert min(box.z0 for box in top) == pytest.approx(max(b.z1 for b in memory_boxes))


def test_summary_and_cli_outputs(tmp_path):
    scene = build(CONFIG, tmp_path)
    expected = {"regions.csv", "geometry_summary.json", "top_view.png",
                "xz_section.png", "yz_section.png"}
    assert expected == {path.name for path in tmp_path.iterdir()}
    summary = json.loads((tmp_path / "geometry_summary.json").read_text(encoding="utf-8"))
    assert summary["total_boxes"] == len(scene.boxes)
    # Per-column role counts aggregate to 4 * 12 = 48 for the four HBM columns;
    # the central thermal_silicon column does not contribute any of these layers.
    dram_si          = sum(1 for b in scene.boxes if b.tags.get("role") == "dram_si")
    dram_beol        = sum(1 for b in scene.boxes if b.tags.get("role") == "dram_beol")
    hybrid_bonding   = sum(1 for b in scene.boxes if b.tags.get("role") == "hybrid_bonding")
    lateral_fills    = sum(1 for b in scene.boxes if b.tags.get("role") == "lateral_fill")
    assert dram_si == 48
    assert dram_beol == 48
    assert hybrid_bonding == 48
    # 4 HBM columns * 36 inset layers * 4 fill sides = 576 mold fill boxes.
    assert lateral_fills == 4 * 36 * 4
    assert summary["boxes_by_material"]["Mold"] == lateral_fills
    assert summary["stack_heights_m"]["hbm_12hi"] == pytest.approx(775e-6)
    assert summary["stack_heights_m"]["thermal_silicon_stack"] == pytest.approx(775e-6)
    assert summary["minimum_dimension_m"] > 0
    assert summary["maximum_dimension_m"] >= summary["minimum_dimension_m"]
    assert {"foundation", "gpu", "top"} <= set(summary["component_bounds_m"])
    assert all(name.startswith("memory_column:") for name in summary["component_bounds_m"]
               if name not in {"foundation", "gpu", "top"})


def test_footprint_outside_package_is_rejected():
    data = load_config(CONFIG).model_dump()
    data["footprints"]["gpu"]["center_x"] = 1.0
    with pytest.raises(ValidationError, match="exceeds package bounds"):
        SimulationConfig.model_validate(data)


def test_missing_match_height_reference_is_rejected():
    data = load_config(CONFIG).model_dump()
    # Convert the thermal_silicon column from a stack column to a single-material
    # column that references a missing match_height_of, so we exercise the
    # "unknown stack reference" path.
    column = data["horizontal"]["memory_zone"]["columns"][4]
    column["stack"] = None
    column["material"] = "Thermal_Silicon"
    column["match_height_of"] = "missing"
    with pytest.raises(ValidationError, match="unknown stack reference"):
        SimulationConfig.model_validate(data)


def test_short_stack_without_fill_is_rejected():
    data = load_config(CONFIG).model_dump()
    # Inject a short test stack and assign it to one of the HBM columns
    # without providing a fill_above material.
    data["stack_templates"]["short_test_stack"] = {
        "items": [{"kind": "layer", "name": "thin",
                   "material": "Silicon", "thickness": "100 um"}]
    }
    column = data["horizontal"]["memory_zone"]["columns"][3]
    column["stack"] = "short_test_stack"
    column["fill_above"] = None
    with pytest.raises(ValidationError, match="requires fill_above"):
        SimulationConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Fig. 3(a) layout validation
# ---------------------------------------------------------------------------

def test_footprint_sizes():
    cfg = load_config(CONFIG)
    assert cfg.footprints["package"].size_x == pytest.approx(65e-3)
    assert cfg.footprints["package"].size_y == pytest.approx(65e-3)
    assert cfg.footprints["gpu"].size_x == pytest.approx(30e-3)
    assert cfg.footprints["gpu"].size_y == pytest.approx(22e-3)
    assert cfg.footprints["thermal_silicon"].size_x == pytest.approx(8e-3)
    assert cfg.footprints["thermal_silicon"].size_y == pytest.approx(22e-3)
    for name in HBM_COLUMN_NAMES:
        fp = cfg.footprints[name]
        assert fp.size_x == pytest.approx(11e-3)
        assert fp.size_y == pytest.approx(11e-3)


def test_hbm_and_thermal_silicon_form_nominal_envelope():
    """The four 11x11 HBM base footprints and the central 8x22 thermal-silicon
    footprint together form a 30x22 mm nominal placement envelope at the
    memory zone centre. This is a *placement* claim, not a zero-gap tiling
    claim: per-layer lateral insets and the HBM-base / DRAM-die footprint
    mismatch with mold-filled cavities (Fig. 3(a)) are not yet modelled.
    See README "Known limitations".
    """
    cfg = load_config(CONFIG)
    zone = cfg.footprints["memory_zone"]
    fps = [cfg.footprints[n] for n in HBM_COLUMN_NAMES + ("thermal_silicon",)]
    for fp in fps:
        assert zone.x0 <= fp.x0 and fp.x1 <= zone.x1
        assert zone.y0 <= fp.y0 and fp.y1 <= zone.y1
    x_min = min(fp.x0 for fp in fps)
    x_max = max(fp.x1 for fp in fps)
    y_min = min(fp.y0 for fp in fps)
    y_max = max(fp.y1 for fp in fps)
    assert (x_max - x_min) == pytest.approx(zone.size_x)
    assert (y_max - y_min) == pytest.approx(zone.size_y)
    assert x_min == pytest.approx(zone.x0)
    assert x_max == pytest.approx(zone.x1)
    assert y_min == pytest.approx(zone.y0)
    assert y_max == pytest.approx(zone.y1)


def test_top_footprint_is_memory_zone():
    """TIM and lid both use the 30x22 mm memory_zone footprint (Fig. 3(b)),
    not the full 65x65 mm package footprint."""
    cfg = load_config(CONFIG)
    assert cfg.horizontal.top.footprint == "memory_zone"
    zone = cfg.footprints["memory_zone"]
    assert zone.size_x == pytest.approx(30e-3)
    assert zone.size_y == pytest.approx(22e-3)


def test_laminate_foundation_keeps_package_footprint():
    """The laminate foundation still spans the full 65x65 mm package."""
    cfg = load_config(CONFIG)
    assert cfg.horizontal.foundation.footprint == "package"
    package = cfg.footprints["package"]
    assert package.size_x == pytest.approx(65e-3)
    assert package.size_y == pytest.approx(65e-3)


def test_top_component_bounds_match_memory_zone():
    """The top component (TIM + lid) must share the memory_zone x/y extents,
    so it covers the 30x22 mm device region only."""
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    zone = load_config(CONFIG).footprints["memory_zone"]
    top_boxes = scene.filter(component="top")
    assert min(b.x0 for b in top_boxes) == pytest.approx(zone.x0)
    assert max(b.x1 for b in top_boxes) == pytest.approx(zone.x1)
    assert min(b.y0 for b in top_boxes) == pytest.approx(zone.y0)
    assert max(b.y1 for b in top_boxes) == pytest.approx(zone.y1)


# ---------------------------------------------------------------------------
# Fig. 3(c) material conductivity sanity check.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "expected"), [
    ("Silicon",        (140.0, 140.0, 140.0)),
    ("GPU_HBM_uBump",  (0.59,  0.59,  19.28)),
    ("BSPDN",          (83.0,  83.0,  71.0)),
    ("Cu_Pillar_Bump", (0.54,  0.54,  13.25)),
    ("TIM",            (9.71,  9.71,  9.71)),
    ("Mold",           (3.0,   3.0,   3.0)),
])
def test_material_conductivities(name, expected):
    material = load_config(CONFIG).materials[name]
    assert material.k_local == pytest.approx(expected)
