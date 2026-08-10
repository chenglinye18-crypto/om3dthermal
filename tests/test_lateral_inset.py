"""Per-layer lateral inset and automatic mold-cavity generation tests.

The schema field is ``lateral_inset`` on ``Layer``. The shorthand
``{x: v, y: v}`` is normalised to ``{x_minus: v, x_plus: v, y_minus: v,
y_plus: v}``. A non-zero inset on a column layer is rendered as a single
central entity box plus up to four lateral fill boxes in the column
footprint's remaining cavity; the fill material is the
``memory_zone.background_material`` (default ``Mold``).
"""
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from pydantic import ValidationError

from om3dthermal.config import (
    LateralInset,
    Layer,
    SimulationConfig,
    StackTemplate,
    load_config,
)
from om3dthermal.geometry.horizontal_columns import (
    HorizontalColumnsBuilder,
    validate_layer_partition,
)
from om3dthermal.geometry.primitives import AxisAlignedBox, Footprint

CONFIG = Path(__file__).parents[1] / "configs" / "exp_conv_2x2_g414_m160.yaml"

HBM_COLUMNS = ("hbm_left_top", "hbm_left_bottom", "hbm_right_top", "hbm_right_bottom")


# ---------------------------------------------------------------------------
# A. Backward compatibility: no lateral_inset behaves exactly as before.
# ---------------------------------------------------------------------------

def test_layer_without_lateral_inset_keeps_full_footprint():
    cfg = _single_column_config(
        items=[{"kind": "layer", "name": "x", "material": "Silicon",
                "thickness": "1 um"}],
    )
    scene = HorizontalColumnsBuilder(cfg).build()
    column_boxes = scene.filter(component="memory_column:test")
    assert len(column_boxes) == 1
    fp = cfg.footprints["fp"]
    box = column_boxes[0]
    assert box.x0 == pytest.approx(fp.x0)
    assert box.x1 == pytest.approx(fp.x1)
    assert box.y0 == pytest.approx(fp.y0)
    assert box.y1 == pytest.approx(fp.y1)
    assert box.tags.get("lateral_inset_applied") is not True


def test_zero_inset_is_treated_as_no_inset():
    cfg = _single_column_config(
        items=[{"kind": "layer", "name": "x", "material": "Silicon",
                "thickness": "1 um",
                "lateral_inset": {"x_minus": "0 um", "x_plus": "0 um",
                                  "y_minus": "0 um", "y_plus": "0 um"}}],
    )
    scene = HorizontalColumnsBuilder(cfg).build()
    column_boxes = scene.filter(component="memory_column:test")
    assert len(column_boxes) == 1  # zero inset => no fill boxes
    inset = LateralInset.model_validate({})
    assert inset.is_zero()


# ---------------------------------------------------------------------------
# B. Symmetric inset: central entity + 4 fill strips, area conservation.
# ---------------------------------------------------------------------------

def test_symmetric_inset_produces_central_and_four_fills():
    cfg = _single_column_config(
        items=[{"kind": "layer", "name": "dram", "material": "Silicon",
                "thickness": "1 um",
                "lateral_inset": {"x": "1 mm", "y": "2 mm"}}],
        footprint_size=(10e-3, 8e-3),
    )
    scene = HorizontalColumnsBuilder(cfg).build()
    column_boxes = scene.filter(component="memory_column:test")
    # 1 central + 4 fills = 5 boxes for this single layer.
    assert len(column_boxes) == 5
    central = [b for b in column_boxes if b.tags.get("lateral_inset_applied") is True]
    fills   = [b for b in column_boxes if b.tags.get("role") == "lateral_fill"]
    assert len(central) == 1
    assert len(fills) == 4

    parent = cfg.footprints["fp"]
    inner_x0 = parent.x0 + 1e-3
    inner_x1 = parent.x1 - 1e-3
    inner_y0 = parent.y0 + 2e-3
    inner_y1 = parent.y1 - 2e-3
    c = central[0]
    assert (c.x0, c.x1, c.y0, c.y1) == (
        pytest.approx(inner_x0), pytest.approx(inner_x1),
        pytest.approx(inner_y0), pytest.approx(inner_y1),
    )

    by_side = {b.tags["inset_side"]: b for b in fills}
    assert set(by_side) == {"left", "right", "bottom", "top"}
    assert (by_side["left"].x0,  by_side["left"].x1)  == (pytest.approx(parent.x0),    pytest.approx(inner_x0))
    assert (by_side["right"].x0, by_side["right"].x1) == (pytest.approx(inner_x1),   pytest.approx(parent.x1))
    assert (by_side["bottom"].x0, by_side["bottom"].x1) == (pytest.approx(inner_x0), pytest.approx(inner_x1))
    assert (by_side["bottom"].y0, by_side["bottom"].y1) == (pytest.approx(parent.y0), pytest.approx(inner_y0))
    assert (by_side["top"].x0, by_side["top"].x1)    == (pytest.approx(inner_x0),    pytest.approx(inner_x1))
    assert (by_side["top"].y0, by_side["top"].y1)    == (pytest.approx(inner_y1),    pytest.approx(parent.y1))

    for box in fills:
        assert box.material == "Mold"
        assert box.z0 == pytest.approx(c.z0)
        assert box.z1 == pytest.approx(c.z1)
        assert box.tags["parent_layer"] == "dram"
        assert box.tags["parent_column"] == "test"

    total = sum((b.x1 - b.x0) * (b.y1 - b.y0) for b in column_boxes)
    assert total == pytest.approx(80e-6, rel=1e-6)


# ---------------------------------------------------------------------------
# C. Asymmetric inset: per-edge coordinates are honoured exactly.
# ---------------------------------------------------------------------------

def test_asymmetric_inset_per_edge_coordinates():
    cfg = _single_column_config(
        items=[{"kind": "layer", "name": "dram", "material": "Silicon",
                "thickness": "1 um",
                "lateral_inset": {"x_minus": "1 mm", "x_plus": "2 mm",
                                  "y_minus": "3 mm", "y_plus": "4 mm"}}],
        footprint_size=(20e-3, 20e-3),
    )
    scene = HorizontalColumnsBuilder(cfg).build()
    column_boxes = scene.filter(component="memory_column:test")
    central = [b for b in column_boxes if b.tags.get("lateral_inset_applied") is True][0]
    parent = cfg.footprints["fp"]
    assert central.x0 == pytest.approx(parent.x0 + 1e-3)
    assert central.x1 == pytest.approx(parent.x1 - 2e-3)
    assert central.y0 == pytest.approx(parent.y0 + 3e-3)
    assert central.y1 == pytest.approx(parent.y1 - 4e-3)
    assert central.material == "Silicon"

    by_side = {b.tags["inset_side"]: b for b in column_boxes
               if b.tags.get("role") == "lateral_fill"}
    assert by_side["left"].x1   - by_side["left"].x0   == pytest.approx(1e-3)
    assert by_side["right"].x1  - by_side["right"].x0  == pytest.approx(2e-3)
    assert by_side["bottom"].y1 - by_side["bottom"].y0 == pytest.approx(3e-3)
    assert by_side["top"].y1    - by_side["top"].y0    == pytest.approx(4e-3)


def test_partial_shorthand_uses_only_x_value():
    # Only "x" provided, no "y" => x_min and x_plus get the value, y_min/y_plus
    # default to 0. Therefore no top/bottom fills, only left/right.
    cfg = _single_column_config(
        items=[{"kind": "layer", "name": "dram", "material": "Silicon",
                "thickness": "1 um",
                "lateral_inset": {"x": "1 mm"}}],
    )
    scene = HorizontalColumnsBuilder(cfg).build()
    fills = [b for b in scene.filter(component="memory_column:test")
             if b.tags.get("role") == "lateral_fill"]
    assert {b.tags["inset_side"] for b in fills} == {"left", "right"}


# ---------------------------------------------------------------------------
# D. Invalid inset: rejected by schema (negative / unknown / bad unit) and
#    by the builder (inset that erases the central entity).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"x_minus": "-0.1 mm"},
    {"xx": "1 mm"},                           # unknown field
])
def test_invalid_lateral_inset_is_rejected_by_schema(bad):
    with pytest.raises(ValidationError):
        LateralInset.model_validate(bad)


def test_invalid_lateral_inset_unit_is_rejected_by_schema():
    with pytest.raises(ValidationError):
        LateralInset.model_validate({"x_minus": "3 seconds"})


# ---------------------------------------------------------------------------
# Hardening: the validator must not mutate the caller's mapping.
# ---------------------------------------------------------------------------

def test_lateral_inset_validator_does_not_mutate_input_mapping():
    original = {"x": "0.5 mm", "y_minus": "0.2 mm"}
    snapshot = {k: v for k, v in original.items()}
    inset = LateralInset.model_validate(original)
    # Original dict is left exactly as the caller passed it.
    assert original == snapshot
    # The normalised model is what we expect.
    assert inset.x_minus == pytest.approx(0.5e-3)
    assert inset.x_plus  == pytest.approx(0.5e-3)
    assert inset.y_minus == pytest.approx(0.2e-3)
    assert inset.y_plus  == 0.0
    # The returned model object does not alias the caller's mapping.
    assert inset.model_dump() != original


def test_lateral_inset_validator_does_not_mutate_explicit_input():
    original = {"x_minus": "1 mm", "x_plus": "2 mm",
                "y_minus": "3 mm", "y_plus": "4 mm"}
    snapshot = {k: v for k, v in original.items()}
    LateralInset.model_validate(original)
    assert original == snapshot


def test_builder_rejects_inset_that_erases_central():
    # x_min + x_plus == parent width (10 mm) -> no room for central.
    cfg = _single_column_config(
        items=[{"kind": "layer", "name": "dram", "material": "Silicon",
                "thickness": "1 um",
                "lateral_inset": {"x_minus": "5 mm", "x_plus": "5 mm",
                                  "y_minus": "0 um", "y_plus": "0 um"}}],
    )
    builder = HorizontalColumnsBuilder(cfg)
    with pytest.raises(ValueError, match="leaves no room"):
        builder.build()


def test_builder_rejects_inset_with_y_sum_exceeding_parent():
    cfg = _single_column_config(
        items=[{"kind": "layer", "name": "dram", "material": "Silicon",
                "thickness": "1 um",
                "lateral_inset": {"x_minus": "0 um", "x_plus": "0 um",
                                  "y_minus": "5 mm", "y_plus": "5 mm"}}],
    )
    builder = HorizontalColumnsBuilder(cfg)
    with pytest.raises(ValueError, match="leaves no room"):
        builder.build()


# ---------------------------------------------------------------------------
# E. Repeat-block expansion preserves the inset and keeps names unique.
# ---------------------------------------------------------------------------

def test_repeat_block_preserves_lateral_inset():
    template = StackTemplate.model_validate({
        "items": [
            {"kind": "repeat", "count": 3, "layers": [
                {"kind": "layer", "name": "dram", "material": "Silicon",
                 "thickness": "1 um",
                 "lateral_inset": {"x": "0.5 mm", "y": "0.5 mm"}},
            ]},
        ],
    })
    expanded = template.expand()
    assert len(expanded) == 3
    assert [layer.name for layer in expanded] == ["dram_01", "dram_02", "dram_03"]
    for layer in expanded:
        assert layer.lateral_inset is not None
        assert layer.lateral_inset.x_minus == pytest.approx(0.5e-3)
        assert layer.lateral_inset.x_plus  == pytest.approx(0.5e-3)
        assert layer.lateral_inset.y_minus == pytest.approx(0.5e-3)
        assert layer.lateral_inset.y_plus  == pytest.approx(0.5e-3)
        assert "layers[0]#repeat=" in layer.source_suffix


def test_repeat_block_unique_names_with_inset():
    template = StackTemplate.model_validate({
        "items": [
            {"kind": "repeat", "count": 5, "layers": [
                {"kind": "layer", "name": "dram", "material": "Silicon",
                 "thickness": "1 um",
                 "lateral_inset": {"x": "0.5 mm", "y": "0.5 mm"}},
            ]},
        ],
    })
    expanded = template.expand()
    names = [layer.name for layer in expanded]
    assert names == ["dram_01", "dram_02", "dram_03", "dram_04", "dram_05"]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# F. validate_layer_partition: independent unit-level invariant check.
# ---------------------------------------------------------------------------

def _make_footprint(name="fp", x0=0.0, x1=10e-3, y0=0.0, y1=10e-3) -> Footprint:
    return Footprint.model_validate({
        "name": name,
        "center_x": (x0 + x1) / 2, "center_y": (y0 + y1) / 2,
        "size_x": x1 - x0, "size_y": y1 - y0,
    })


def test_validate_layer_partition_accepts_well_formed_partition():
    parent = _make_footprint()
    central = AxisAlignedBox(
        name="c", material="Silicon",
        x0=parent.x0 + 1e-3, x1=parent.x1 - 1e-3,
        y0=parent.y0 + 2e-3, y1=parent.y1 - 2e-3,
        z0=0.0, z1=1e-6, tags={}, source_path="c",
    )
    fills = [
        AxisAlignedBox(name="L", material="Mold",
                       x0=parent.x0, x1=parent.x0 + 1e-3,
                       y0=parent.y0, y1=parent.y1, z0=0.0, z1=1e-6,
                       tags={}, source_path="L"),
        AxisAlignedBox(name="R", material="Mold",
                       x0=parent.x1 - 1e-3, x1=parent.x1,
                       y0=parent.y0, y1=parent.y1, z0=0.0, z1=1e-6,
                       tags={}, source_path="R"),
        AxisAlignedBox(name="B", material="Mold",
                       x0=parent.x0 + 1e-3, x1=parent.x1 - 1e-3,
                       y0=parent.y0, y1=parent.y0 + 2e-3, z0=0.0, z1=1e-6,
                       tags={}, source_path="B"),
        AxisAlignedBox(name="T", material="Mold",
                       x0=parent.x0 + 1e-3, x1=parent.x1 - 1e-3,
                       y0=parent.y1 - 2e-3, y1=parent.y1, z0=0.0, z1=1e-6,
                       tags={}, source_path="T"),
    ]
    validate_layer_partition(parent, central, fills)


def test_validate_layer_partition_rejects_overlap():
    parent = _make_footprint()
    central = AxisAlignedBox(
        name="c", material="Silicon",
        x0=parent.x0, x1=parent.x1, y0=parent.y0, y1=parent.y1,
        z0=0.0, z1=1e-6, tags={}, source_path="c",
    )
    overlapping = AxisAlignedBox(
        name="L", material="Mold",
        x0=parent.x0, x1=parent.x0 + 5e-3,
        y0=parent.y0, y1=parent.y1, z0=0.0, z1=1e-6,
        tags={}, source_path="L",
    )
    with pytest.raises(ValueError, match="overlaps central"):
        validate_layer_partition(parent, central, [overlapping])


def test_validate_layer_partition_rejects_area_mismatch():
    parent = _make_footprint()
    central = AxisAlignedBox(
        name="c", material="Silicon",
        x0=parent.x0, x1=parent.x1 - 1e-3,
        y0=parent.y0, y1=parent.y1, z0=0.0, z1=1e-6,
        tags={}, source_path="c",
    )
    with pytest.raises(ValueError, match="does not match parent area"):
        validate_layer_partition(parent, central, [])


def test_validate_layer_partition_rejects_fill_outside_parent():
    parent = _make_footprint()
    central = AxisAlignedBox(
        name="c", material="Silicon",
        x0=parent.x0, x1=parent.x1, y0=parent.y0, y1=parent.y1,
        z0=0.0, z1=1e-6, tags={}, source_path="c",
    )
    bad_fill = AxisAlignedBox(
        name="L", material="Mold",
        x0=parent.x0 - 1e-3, x1=parent.x0,
        y0=parent.y0, y1=parent.y1, z0=0.0, z1=1e-6,
        tags={}, source_path="L",
    )
    with pytest.raises(ValueError, match="extends outside parent footprint"):
        validate_layer_partition(parent, central, [bad_fill])


# ---------------------------------------------------------------------------
# G. The shipped benchmark: DRAM layers get 0.5 mm per-side inset and the
#    four HBM columns and the central thermal-silicon column behave as
#    documented.
# ---------------------------------------------------------------------------

def test_benchmark_hbm_base_layers_keep_full_11x11_footprint():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMNS:
        for layer_name in ("gpu_hbm_ubump", "hbm_base_beol", "hbm_base_si"):
            box = next(b for b in scene.filter(component=f"memory_column:{name}")
                       if b.name.endswith(layer_name))
            # 11x11 mm centred on the column footprint.
            assert (box.x1 - box.x0) == pytest.approx(11e-3)
            assert (box.y1 - box.y0) == pytest.approx(11e-3)
            assert box.tags.get("lateral_inset_applied") is not True
            assert box.tags.get("role") != "lateral_fill"


def test_benchmark_dram_layers_have_10_8x10_8_central_entity_and_4_fills():
    """The locked benchmark uses a 10.8 x 10.8 mm DRAM die
    (with 11 x 11 mm HBM base, 0.1 mm per-side mold ring); see
    the canonical conventional config and IEDM25 provenance document."""
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMNS:
        central = [b for b in scene.filter(component=f"memory_column:{name}")
                   if b.tags.get("lateral_inset_applied") is True
                   and b.tags.get("role") in {"dram_si", "dram_beol", "hybrid_bonding"}]
        # 11 regular dies * 3 layers + 3 top layers = 36 inset layers.
        assert len(central) == 36
        for box in central:
            assert (box.x1 - box.x0) == pytest.approx(10.8e-3)
            assert (box.y1 - box.y0) == pytest.approx(10.8e-3)
        fills = [b for b in scene.filter(component=f"memory_column:{name}")
                 if b.tags.get("role") == "lateral_fill"]
        # 36 central layers * 4 fill sides per layer.
        assert len(fills) == 36 * 4
        assert all(b.material == "Mold" for b in fills)


def test_benchmark_hbm_column_total_height_still_775_um():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMNS:
        boxes = scene.filter(component=f"memory_column:{name}")
        assert max(b.z1 for b in boxes) - min(b.z0 for b in boxes) == pytest.approx(775e-6)


def test_benchmark_thermal_silicon_column_has_no_lateral_inset_fills():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    fills = [b for b in scene.filter(component="memory_column:thermal_silicon")
             if b.tags.get("role") == "lateral_fill"]
    assert fills == []


def test_benchmark_hbm_partition_no_3d_overlap():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    tolerance = 1e-12
    for name in HBM_COLUMNS:
        boxes = list(scene.filter(component=f"memory_column:{name}"))
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                overlap = (
                    a.x1 - b.x0 > tolerance and a.x0 - b.x1 < -tolerance
                    and a.y1 - b.y0 > tolerance and a.y0 - b.y1 < -tolerance
                    and a.z1 - b.z0 > tolerance and a.z0 - b.z1 < -tolerance
                )
                assert not overlap, f"{a.name} and {b.name} overlap in column {name}"


def test_benchmark_each_hbm_column_box_count():
    """3 single-box layers + 36 inset layers * 5 boxes = 183 per column."""
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMNS:
        boxes = scene.filter(component=f"memory_column:{name}")
        assert len(boxes) == 3 + 36 * 5  # 183


def test_benchmark_each_hbm_column_inset_layer_names():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMNS:
        boxes = scene.filter(component=f"memory_column:{name}")
        inset_names = {b.name.split(".")[-1] for b in boxes
                       if b.tags.get("lateral_inset_applied") is True}
        for n in range(1, 12):
            for kind in ("hybrid_bonding", "dram_beol", "dram_si"):
                assert f"{kind}_{n:02d}" in inset_names
        for kind in ("top_hybrid_bonding", "top_dram_beol", "top_dram_si"):
            assert kind in inset_names


def test_benchmark_lateral_fills_share_z_with_their_central_box():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    for name in HBM_COLUMNS:
        boxes = scene.filter(component=f"memory_column:{name}")
        for fill in (b for b in boxes if b.tags.get("role") == "lateral_fill"):
            parent_layer = fill.tags["parent_layer"]
            central_name = f"memory_column:{name}.{parent_layer}"
            central = next(b for b in boxes if b.name == central_name)
            assert fill.z0 == pytest.approx(central.z0)
            assert fill.z1 == pytest.approx(central.z1)


def test_benchmark_total_scene_height_unchanged():
    scene = HorizontalColumnsBuilder(load_config(CONFIG)).build()
    z0_min = min(b.z0 for b in scene.boxes)
    z1_max = max(b.z1 for b in scene.boxes)
    # 300 (Laminate) + 73.265 (GPU) + 775 (memory) + 3200 (top) = 4348.265 um.
    assert z1_max - z0_min == pytest.approx(4348.265e-6)


# ---------------------------------------------------------------------------
# Hardening: tolerance helpers + z-level clustering do not depend on
# ``round()`` or any fixed quantisation grid. Tests use non-integer
# nanometre boundaries to make sure the partition and continuity logic
# survives arbitrary z values.
# ---------------------------------------------------------------------------

def test_length_close_respects_absolute_and_relative_tol():
    from om3dthermal.geometry.horizontal_columns import _length_close
    # Within 1 pm => close.
    assert _length_close(0.0, 5e-13)
    # 1 pm apart on a 1 mm quantity => close (relative tolerance 1e-10).
    assert _length_close(1e-3, 1e-3 + 1e-12)
    # 1 nm apart on a 1 mm quantity => not close (1e-9 > 1e-13 relative tol).
    assert not _length_close(1e-3, 1e-3 + 1e-9)
    # 1 mm apart on a 1 mm quantity => not close.
    assert not _length_close(0.0, 1e-3)


def test_area_close_respects_absolute_and_relative_tol():
    from om3dthermal.geometry.horizontal_columns import _area_close
    # Tiny absolute floor protects degenerate regions.
    assert _area_close(1e-30, 5e-31)
    # Relative ceiling kicks in for large regions.
    assert not _area_close(1e-6, 2e-6)


def test_boxes_overlap_3d_uses_length_tol():
    from om3dthermal.geometry.horizontal_columns import _boxes_overlap_3d, _LENGTH_TOL
    a = AxisAlignedBox(name="a", material="M", x0=0, x1=1, y0=0, y1=1,
                       z0=0, z1=1, tags={}, source_path="a")
    b = AxisAlignedBox(name="b", material="M", x0=0.5, x1=1.5, y0=0.5, y1=1.5,
                       z0=0.5, z1=1.5, tags={}, source_path="b")
    assert _boxes_overlap_3d(a, b)
    # Sliding b so the overlap is below the absolute length tolerance
    # makes the boxes no longer overlap.
    b_shifted = b.model_copy(update={"x0": 1.0 + 0.5 * _LENGTH_TOL,
                                     "x1": 1.5 + 0.5 * _LENGTH_TOL})
    assert not _boxes_overlap_3d(a, b_shifted)


def test_cluster_by_groups_within_length_tol():
    from om3dthermal.geometry.horizontal_columns import _cluster_by
    # Values within 1 pm of the cluster seed stay in the cluster; values
    # further than 1 pm start a new cluster.
    boxes = [
        AxisAlignedBox(name=f"b{i}", material="M",
                       x0=0, x1=1, y0=0, y1=1, z0=z, z1=z + 1e-6,
                       tags={}, source_path=f"b{i}")
        for i, z in enumerate([0.0, 5e-13, 9e-13, 1.0e-3, 1.0e-3 + 5e-13])
    ]
    groups = _cluster_by(boxes, key=lambda b: b.z0, tol=1e-12)
    z_per_group = [[b.z0 for b in group] for group in groups]
    assert z_per_group[0] == pytest.approx([0.0, 5e-13, 9e-13])
    assert z_per_group[1] == pytest.approx([1.0e-3, 1.0e-3 + 5e-13])
    # A box 1 mm above the second cluster must NOT join either of them.
    boxes.append(AxisAlignedBox(name="far", material="M",
                                x0=0, x1=1, y0=0, y1=1,
                                z0=1.0e-2, z1=1.0e-2 + 1e-6,
                                tags={}, source_path="far"))
    groups = _cluster_by(boxes, key=lambda b: b.z0, tol=1e-12)
    assert len(groups) == 3
    assert groups[0][0].z0 == pytest.approx(0.0)
    assert groups[1][0].z0 == pytest.approx(1.0e-3)
    assert groups[2][0].z0 == pytest.approx(1.0e-2)


def test_validate_continuity_accepts_non_integer_nm_z_boundaries():
    """Build a Scene with a single column whose z values are 0.3333,
    0.7777 and 1.2345 um above the foundation/gpu, none of which are
    integer nanometres. The continuity check must still treat them as
    four consecutive z-levels with no gap or overlap.
    """
    from om3dthermal.geometry.horizontal_columns import HorizontalColumnsBuilder, _length_close
    builder = HorizontalColumnsBuilder(_single_column_config(
        items=[
            {"kind": "layer", "name": "a", "material": "Silicon",
             "thickness": "0.3333 um"},
            {"kind": "layer", "name": "b", "material": "Silicon",
             "thickness": "0.4444 um"},  # 0.3333 + 0.4444 = 0.7777
            {"kind": "layer", "name": "c", "material": "Silicon",
             "thickness": "0.4568 um"},  # 0.7777 + 0.4568 = 1.2345
        ],
    ))
    scene = builder.build()
    column_boxes = scene.filter(component="memory_column:test")
    column_boxes.sort(key=lambda b: b.z0)
    # The column spans 0.3333 + 0.4444 + 0.4568 = 1.2345 um above the gpu.
    expected_height = 0.3333e-6 + 0.4444e-6 + 0.4568e-6
    assert column_boxes[-1].z1 - column_boxes[0].z0 == pytest.approx(expected_height, abs=1e-15)
    # The builder's z-level cluster should not have raised: every adjacent
    # pair of layer boundaries must agree within the length tolerance.
    for lower, upper in zip(column_boxes, column_boxes[1:]):
        assert _length_close(lower.z1, upper.z0), \
            f"z gap between {lower.name!r} (z1={lower.z1}) and {upper.name!r} (z0={upper.z0})"


def test_validate_continuity_rejects_real_z_gap_above_tol():
    """Two boxes whose z gap is well above the length tolerance must be
    rejected by the continuity check. We bypass the builder and feed
    a Scene directly to the validator to avoid column-height invariants
    in the rest of the pipeline.
    """
    from om3dthermal.geometry.horizontal_columns import HorizontalColumnsBuilder
    from om3dthermal.geometry.scene import Scene
    cfg = _single_column_config(
        items=[{"kind": "layer", "name": "a", "material": "Silicon",
                "thickness": "0.3333 um"}],
    )
    builder = HorizontalColumnsBuilder(cfg)
    # Add two well-separated boxes directly to the scene.
    builder.scene.add(AxisAlignedBox(
        name="memory_column:test.a", material="Silicon",
        x0=0, x1=10e-3, y0=0, y1=10e-3, z0=0.0, z1=0.3333e-6,
        tags={"component": "memory_column:test"}, source_path="a",
    ))
    builder.scene.add(AxisAlignedBox(
        name="memory_column:test.b", material="Silicon",
        x0=0, x1=10e-3, y0=0, y1=10e-3, z0=0.5e-6, z1=0.6e-6,
        tags={"component": "memory_column:test"}, source_path="b",
    ))
    with pytest.raises(ValueError, match="z gap"):
        builder._validate_continuity("memory_column:test", 0.0, 0.6e-6)


def test_validate_layer_partition_groups_boxes_at_same_z_with_float_drift():
    """Central + 4 fills at the same z slice, where the central's z range
    has a 1 pm float-rounding drift away from the fills' z range. The
    length tolerance must accept it as one consistent z slice; the
    partition invariant must still hold.
    """
    from om3dthermal.geometry.horizontal_columns import validate_layer_partition
    parent = Footprint.model_validate({
        "name": "fp", "center_x": "0 mm", "center_y": "0 mm",
        "size_x": "10 mm", "size_y": "10 mm",
    })
    central = AxisAlignedBox(
        name="c", material="Silicon",
        x0=parent.x0 + 1e-3, x1=parent.x1 - 1e-3,
        y0=parent.y0 + 1e-3, y1=parent.y1 - 1e-3,
        z0=0.0, z1=0.5e-6 + 5e-13, tags={}, source_path="c",
    )
    fills = [
        AxisAlignedBox(name="L", material="Mold",
                       x0=parent.x0, x1=parent.x0 + 1e-3,
                       y0=parent.y0, y1=parent.y1, z0=0.0, z1=0.5e-6,
                       tags={}, source_path="L"),
        AxisAlignedBox(name="R", material="Mold",
                       x0=parent.x1 - 1e-3, x1=parent.x1,
                       y0=parent.y0, y1=parent.y1, z0=0.0, z1=0.5e-6,
                       tags={}, source_path="R"),
        AxisAlignedBox(name="B", material="Mold",
                       x0=parent.x0 + 1e-3, x1=parent.x1 - 1e-3,
                       y0=parent.y0, y1=parent.y0 + 1e-3, z0=0.0, z1=0.5e-6,
                       tags={}, source_path="B"),
        AxisAlignedBox(name="T", material="Mold",
                       x0=parent.x0 + 1e-3, x1=parent.x1 - 1e-3,
                       y0=parent.y1 - 1e-3, y1=parent.y1, z0=0.0, z1=0.5e-6,
                       tags={}, source_path="T"),
    ]
    # The 1 pm drift on the central's z1 is within _LENGTH_TOL, so the
    # partition should still hold; the area overlap check uses the
    # central's effective footprint (0 .. 0.5 um + 5e-13 m), which is
    # essentially the same as the fills' (0 .. 0.5 um).
    validate_layer_partition(parent, central, fills)  # no exception


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_column_config(items, *, footprint_size=(10e-3, 10e-3)):
    """Build a minimal SimulationConfig that has a single column running
    through a stack template with the given items. The column's parent
    footprint is ``footprint_size`` centred at the origin; the package is
    20 x 20 mm.

    The foundation, gpu and top placements use a separate ``pad`` stack
    (a single no-inset Silicon layer) so that any inset on the column's
    stack only applies to the column itself.
    """
    sx, sy = footprint_size
    return SimulationConfig.model_validate({
        "name": "test",
        "package_footprint": "pkg",
        "materials": {
            "Silicon": {"name": "Silicon", "k_local": [140, 140, 140]},
            "Mold":    {"name": "Mold",    "k_local": [3, 3, 3]},
        },
        "footprints": {
            "fp":  {"name": "fp",
                    "center_x": "0 mm", "center_y": "0 mm",
                    "size_x": f"{sx * 1e3:g} mm", "size_y": f"{sy * 1e3:g} mm"},
            "pkg": {"name": "pkg",
                    "center_x": "0 mm", "center_y": "0 mm",
                    "size_x": "20 mm", "size_y": "20 mm"},
        },
        "stack_templates": {
            "s": {"items": items},
            "pad": {"items": [
                {"kind": "layer", "name": "pad", "material": "Silicon",
                 "thickness": "1 um"},
            ]},
        },
        "horizontal": {
            "foundation": {"footprint": "pkg", "stack": "pad"},
            "gpu":        {"footprint": "pkg", "stack": "pad"},
            "memory_zone": {
                "footprint": "fp",
                "reference_stack": "s",
                "background_material": "Mold",
                "columns": [
                    {"name": "test", "footprint": "fp", "stack": "s"}
                ],
            },
            "top": {"footprint": "pkg", "stack": "pad"},
        },
    })
