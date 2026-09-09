from pathlib import Path

import numpy as np

from om3dthermal.case_runner import run_steady_pipeline
from om3dthermal.config import CellSizeConfig, load_config
from om3dthermal.thermal.setup_cache import build_scene_and_signature


ROOT = Path(__file__).parents[1]


def test_signature_excludes_power_but_covers_mesh() -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "toy_1box.yaml")
    _, original = build_scene_and_signature(config)
    sources = config.thermal_power_sources
    assert sources is not None
    changed_source = sources.sources[0].model_copy(update={"total_power": 7.0})
    power_changed = config.model_copy(update={
        "thermal_power_sources": sources.model_copy(update={
            "sources": [changed_source]})})
    _, power_signature = build_scene_and_signature(power_changed)
    assert power_signature == original

    discretization = config.discretization
    assert discretization is not None
    mesh_changed = config.model_copy(update={
        "discretization": discretization.model_copy(update={
            "max_cell_size": CellSizeConfig(x=0.004, y=0.005, z=0.005)})})
    _, mesh_signature = build_scene_and_signature(mesh_changed)
    assert mesh_signature != original

    silicon = config.materials["Silicon"]
    material_changed = config.model_copy(update={
        "materials": {
            **config.materials,
            "Silicon": silicon.model_copy(update={
                "k_local": (141.0, 141.0, 141.0)}),
        }})
    _, material_signature = build_scene_and_signature(material_changed)
    assert material_signature != original

    boundary = config.thermal_boundary_conditions
    assert boundary is not None
    changed_rule = boundary.rules[0].model_copy(update={
        "ambient_temperature": 294.15})
    boundary_changed = config.model_copy(update={
        "thermal_boundary_conditions": boundary.model_copy(update={
            "rules": [changed_rule, *boundary.rules[1:]]})})
    _, boundary_signature = build_scene_and_signature(boundary_changed)
    assert boundary_signature != original


def test_pipeline_reloads_fixed_setup_and_rebuilds_only_rhs(tmp_path: Path) -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "toy_1box.yaml")
    cache = tmp_path / "thermal.pkl"
    first = run_steady_pipeline(
        config, backend="gpu_pcg", setup_cache_path=cache,
        rtol=1.0, max_delta_t_K=1.0, max_iterations=10,
    )
    assert first.cache_status == "BUILT"
    assert first.setup_build_seconds > 0.0
    assert first.cache_serialization_seconds > 0.0
    assert cache.stat().st_size == first.cache_size_bytes

    sources = config.thermal_power_sources
    assert sources is not None
    changed_source = sources.sources[0].model_copy(update={"total_power": 2.0})
    changed = config.model_copy(update={
        "thermal_power_sources": sources.model_copy(update={
            "sources": [changed_source]})})
    second = run_steady_pipeline(
        changed, backend="gpu_pcg", setup_cache_path=cache,
        rtol=1.0, max_delta_t_K=1.0, max_iterations=10,
    )
    assert second.cache_status == "HIT"
    assert second.setup_build_seconds == 0.0
    assert second.cache_load_seconds > 0.0
    assert second.power.total_power_W == 2.0
    assert np.allclose(
        second.operator.rhs_W - first.operator.rhs_W,
        second.operator.power_W - first.operator.power_W,
    )

    discretization = changed.discretization
    assert discretization is not None
    mesh_changed = changed.model_copy(update={
        "discretization": discretization.model_copy(update={
            "max_cell_size": CellSizeConfig(x=0.004, y=0.005, z=0.005)})})
    rebuilt = run_steady_pipeline(
        mesh_changed, backend="gpu_pcg", setup_cache_path=cache,
        rtol=1.0, max_delta_t_K=1.0, max_iterations=10,
    )
    assert rebuilt.cache_status == "REBUILT"
    assert rebuilt.setup_build_seconds > 0.0

    reused = run_steady_pipeline(
        mesh_changed, backend="gpu_pcg", reusable_setup=rebuilt.reusable_setup,
        rtol=1.0, max_delta_t_K=1.0, max_iterations=10,
    )
    assert reused.cache_status == "MEMORY_REUSE"
    assert reused.setup_build_seconds == 0.0
    assert reused.cache_load_seconds == 0.0
