"""Targeted structural checks for the geometry-closed M3D template."""

from pathlib import Path

import numpy as np
import pytest

from om3dthermal.config import (
    UnresolvedPhysicalParametersError,
    load_config,
    load_orthogonal_m3d_template,
)
from om3dthermal.geometry.orthogonal_hbm import ORTHOGONAL_DIE_ROTATION
from om3dthermal.thermal import is_signed_axis_permutation


CONFIG = Path(__file__).parents[1] / "configs" / "orthogonal_m3d_edram_v0.yaml"
CONVENTIONAL = Path(__file__).parents[1] / "configs" / "exp_conv_2x2_g414_m160.yaml"
MOSAIC = Path(__file__).parents[1] / "configs" / "exp_orth_mosaic98_g414_m156p8_uniform.yaml"


@pytest.fixture(scope="module")
def template():
    return load_orthogonal_m3d_template(CONFIG)


def test_template_parses_known_structure_without_thermal_defaults(template):
    assert template.architecture.type == "orthogonal_m3d_edram"
    assert template.orthogonal.slab_count == 98
    assert template.orthogonal.slab_plane_y_mm == pytest.approx(22.0)
    assert template.orthogonal.slab_height_z_mm == pytest.approx(5.5)
    assert template.orthogonal.slab_pitch_x_um == pytest.approx(300.0)
    assert template.orthogonal.daa_um == pytest.approx(2.0)
    assert template.slab.total_pitch_um == pytest.approx(300.0)
    assert template.slab.region_order == (
        "si_substrate", "feol", "m3d_bitcell_stack",
        "beol_interconnect", "daa")


def test_slab_thickness_closes_exactly_to_300_um(template):
    assert template.slab.si_substrate_um == pytest.approx(292.546)
    assert template.slab.feol_um == pytest.approx(0.150)
    assert template.m3d_beol.bitcell_stack_um == pytest.approx(2.304)
    assert template.m3d_beol.interconnect_um == pytest.approx(3.0)
    assert template.orthogonal.daa_um == pytest.approx(2.0)
    closure = (
        template.slab.si_substrate_um + template.slab.feol_um
        + template.m3d_beol.bitcell_stack_um
        + template.m3d_beol.interconnect_um + template.orthogonal.daa_um)
    assert closure == pytest.approx(300.0, abs=1e-12)


def test_reuses_mosaic_orientation_and_array_placement_contract(template):
    assert template.orthogonal.slab_plane == "y-z"
    assert template.orthogonal.thickness_direction == "global_x"
    assert template.orthogonal.placement == "reuse_orthogonal_mosaic_array"
    rotation = np.asarray(ORTHOGONAL_DIE_ROTATION)
    assert tuple(rotation @ np.array([0.0, 0.0, 1.0])) == (1.0, 0.0, 0.0)
    assert is_signed_axis_permutation(ORTHOGONAL_DIE_ROTATION)
    array_length_mm = (
        template.orthogonal.slab_count
        * template.orthogonal.slab_pitch_x_um / 1000.0)
    assert array_length_mm == pytest.approx(29.4)
    assert array_length_mm <= template.orthogonal.cube_length_x_mm


def test_m3d_bitcell_stack_is_8_times_288_nm(template):
    memory = template.m3d_memory
    assert memory.technology == "CAA_IGZO_2T0C"
    assert memory.layers == 8
    assert template.m3d_beol.bitcell_layers == 8
    assert template.m3d_beol.bitcell_layer_pitch_nm == pytest.approx(288.0)
    assert 8 * 288.0 / 1000.0 == pytest.approx(2.304)
    assert template.m3d_beol.bitcell_stack_um == pytest.approx(2.304)
    assert template.m3d_beol.total_um == pytest.approx(5.304)


def test_daa_is_not_inserted_between_bitcell_layers(template):
    memory = template.m3d_memory
    assert memory.placement == "within_beol_above_feol"
    assert memory.daa_between_m3d_layers is False
    assert template.slab.region_order.count("daa") == 1
    assert template.m3d_beol.region_order == ("bitcell_stack", "interconnect")
    assert "daa" not in template.m3d_beol.region_order


def test_capacity_bookkeeping_uses_density_not_cim_metrics(template):
    result = template.capacity_bookkeeping()
    assert result["slab_area_mm2"] == pytest.approx(121.0)
    assert result["capacity_per_layer_Mb"] == pytest.approx(3630.0)
    assert result["capacity_per_slab_Mb"] == pytest.approx(29040.0)
    assert result["capacity_cube_Mb"] == pytest.approx(2_845_920.0)
    assert result["capacity_cube_Gb_decimal"] == pytest.approx(2845.92)
    assert result["capacity_cube_GB_decimal"] == pytest.approx(355.74)
    assert template.m3d_memory.slab_array_fill_factor == pytest.approx(1.0)
    assert template.power_models.iso_total.cim_metrics_used_as_memory_power is False


def test_unresolved_parameters_are_explicit_and_block_solver_config(template):
    expected = [
        "m3d_beol.thermal.k_in_plane_W_mK",
        "m3d_beol.thermal.k_cross_plane_W_mK",
    ]
    assert template.unresolved_physical_parameters() == expected
    with pytest.raises(UnresolvedPhysicalParametersError) as caught:
        load_config(CONFIG)
    message = str(caught.value)
    assert "geometry bookkeeping is valid" in message
    assert "cannot enter thermal material/operator/solve stages" in message
    assert all(parameter in message for parameter in expected)


def test_total_power_is_iso_total_only_and_not_per_bit_derived(template):
    assert template.power.default_mode == "iso_total"
    iso_total = template.power_models.iso_total
    assert iso_total.memory_total_W == pytest.approx(156.8)
    distribution = iso_total.distribution
    assert distribution.type == "uniform_m3d_layers"
    assert distribution.target_region == "m3d_bitcell_stack"
    assert distribution.direct_power_regions == ("m3d_bitcell_stack",)
    assert iso_total.memory_total_W / template.orthogonal.slab_count == pytest.approx(
        1.6)
    assert template.paper_metrics.energy_efficiency_TOPS_W == pytest.approx(256)
    assert template.paper_metrics.compute_density_TOPS_mm2 == pytest.approx(50)


def test_288nm_provenance_is_derived_not_paper_reported(template):
    assert template.provenance["DERIVED_FROM_PAPER_FIGURE"][
        "bitcell_layer_pitch_nm"] == pytest.approx(288.0)
    assert "bitcell_layer_pitch_nm" not in template.provenance["PAPER_REPORTED"]
    assert template.provenance["DERIVED_FROM_GEOMETRY_CLOSURE"][
        "si_substrate_um"] == pytest.approx(292.546)


def test_conventional_and_mosaic_configs_still_parse_unchanged():
    conventional = load_config(CONVENTIONAL)
    mosaic = load_config(MOSAIC)
    assert conventional.metadata["case_id"] == "exp_conv_2x2_g414_m160"
    assert mosaic.metadata["case_id"] == (
        "exp_orth_mosaic98_g414_m156p8_uniform")
