from pathlib import Path

import pytest

from om3dthermal.bandwidth_thermal_sweep import (
    ARCHITECTURES,
    _base_case,
    _point_simulation,
)


ROOT = Path(__file__).parents[1]


def test_frozen_sweep_points_include_exact_endpoints() -> None:
    hbm, m3d = ARCHITECTURES
    assert hbm.bandwidths_TBps == pytest.approx(
        tuple(0.2 * index for index in range(25)))
    assert m3d.bandwidths_TBps[-2:] == pytest.approx((6.6, 6.7))
    assert len(m3d.bandwidths_TBps) == 35


@pytest.mark.parametrize("spec_index", (0, 1))
def test_2p4_power_closure_and_geometry(spec_index: int) -> None:
    spec = ARCHITECTURES[spec_index]
    case, simulation, memory = _base_case(ROOT, spec)
    point, gpu, memory_power = _point_simulation(
        simulation, spec.architecture, memory, 2.4)
    sources = point.thermal_power_sources
    assert sources is not None
    assert gpu == pytest.approx(298.256)
    assert sum(source.total_power for source in sources.sources) == (
        pytest.approx(gpu + memory_power))
    if spec_index == 0:
        assert memory.E_access_total_pj_bit == pytest.approx(1.9955)
        assert memory_power == pytest.approx(38.3136)
    else:
        assert case.thermal["edge_strip_material"] == "Cu"
        assert memory_power == pytest.approx(16.42100305292776)
