from pathlib import Path

import pytest

from om3dthermal.architecture_comparison import compile_case_thermal
from om3dthermal.power import (
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)
from om3dthermal.thermal.case_adapter import compile_canonical_thermal_case


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", (
    "conventional_hbm_2x1", "orthogonal_si", "orthogonal_m3d_igzo"))
def test_public_thermal_adapter_is_exact_legacy_compiler_facade(name: str) -> None:
    case = load_case_config(ROOT / "configs" / "cases" / f"{name}.yaml")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(case, project_root=ROOT, geometry=geometry)

    legacy = compile_case_thermal(case, system)
    public = compile_canonical_thermal_case(case, system)

    assert public.model_dump(mode="json") == legacy.model_dump(mode="json")
