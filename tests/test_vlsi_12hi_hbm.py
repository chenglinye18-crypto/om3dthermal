"""Power-only checks against the conventional canonical HBM baseline."""

from pathlib import Path

import pytest

from om3dthermal.cli import build_scene
from om3dthermal.config import load_config


ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "configs" / "exp_conv_2x2_g414_m160.yaml"
CHECK = ROOT / "configs" / "exp_conv_2x2_g300_m160.yaml"


@pytest.fixture(scope="module")
def cases():
    return load_config(BASELINE), load_config(CHECK)


def _physical_diff(first, second, path=""):
    if type(first) is not type(second):
        return [(path, first, second)]
    if isinstance(first, dict):
        out = []
        for key in sorted(set(first) | set(second)):
            child = f"{path}.{key}" if path else key
            if key not in first or key not in second:
                out.append((child, first.get(key), second.get(key)))
            else:
                out.extend(_physical_diff(first[key], second[key], child))
        return out
    if isinstance(first, (list, tuple)):
        out = []
        for index, (left, right) in enumerate(zip(first, second)):
            out.extend(_physical_diff(left, right, f"{path}[{index}]"))
        if len(first) != len(second):
            out.append((f"{path}.length", len(first), len(second)))
        return out
    return [] if first == second else [(path, first, second)]


def test_resolved_physical_diff_is_only_gpu_power(cases):
    baseline, check = cases
    differences = _physical_diff(baseline.model_dump(), check.model_dump())
    differences = [item for item in differences
                   if item[0] != "name" and not item[0].startswith("metadata")]
    assert differences == [
        ("thermal_power_sources.sources[0].total_power", 414.0, 300.0)
    ]


def test_geometry_is_identical(cases):
    baseline, check = cases
    scenes = [build_scene(config) for config in cases]
    def signature(scene):
        return sorted((box.name, box.material, box.x0, box.x1, box.y0, box.y1,
                       box.z0, box.z1, box.rotation, repr(sorted(box.tags.items())),
                       box.source_path) for box in scene.boxes)
    assert signature(scenes[0]) == signature(scenes[1])
    assert baseline.stack_templates["hbm_12hi"].total_thickness == pytest.approx(775e-6)
    assert check.stack_templates["hbm_12hi"].total_thickness == pytest.approx(775e-6)


def test_power_is_300_plus_four_times_40_W_with_son23_accounting(cases):
    _, check = cases
    sources = check.thermal_power_sources.sources
    assert sources[0].total_power == pytest.approx(300.0)
    hbm = sources[1:]
    stacks = {source.metadata["stack"] for source in hbm}
    assert len(stacks) == 4
    assert {stack: sum(source.total_power for source in hbm
                       if source.metadata["stack"] == stack)
            for stack in stacks} == pytest.approx({stack: 40.0 for stack in stacks})
    assert sum(source.total_power for source in hbm
               if source.metadata["component_class"] == "logic") == pytest.approx(32.0)
    assert sum(source.total_power for source in hbm
               if source.metadata["component_class"] == "dram") == pytest.approx(128.0)
    assert sum(source.total_power for source in sources) == pytest.approx(460.0)
