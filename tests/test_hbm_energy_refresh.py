"""Energy calibration and frozen-row protection contracts."""
from copy import deepcopy
from pathlib import Path

import pytest

from om3dthermal.hbm_energy_refresh import (
    COMPONENTS, M3D, preserve_rows, read_csv, write_csv,
)
from om3dthermal.thermal_sensitivity import SENSITIVITY_ARCHITECTURES, prepare_case

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('index,target', [(0, 3.0), (1, 4.04)])
def test_calibrated_components_and_row_states_close(index, target):
    case, _, memory, _, audit = prepare_case(ROOT, SENSITIVITY_ARCHITECTURES[index])
    nominal = case.memory.nominal_read_energy
    assert abs(sum(getattr(nominal, k+'_pj_per_bit') for k in COMPONENTS)-target) < 1e-12
    assert abs((nominal.full_row_pj_per_bit+nominal.closed_row_pj_per_bit)/2-target) < 1e-12
    assert abs(memory.E_access_total_pj_bit-target) < 1e-12
    if index == 0:
        assert nominal.full_row_pj_per_bit / nominal.closed_row_pj_per_bit == pytest.approx(0.978/3.013, rel=1e-14)
    else:
        assert audit['raw_DreamRAM_pj_per_bit'] == pytest.approx(2.6871792220414883, rel=1e-14)
        assert nominal.vertical_pj_per_bit / nominal.memory_internal_pj_per_bit == pytest.approx(
            audit['raw_mean']['vertical_pj_per_bit']/audit['raw_mean']['memory_internal_pj_per_bit'], rel=1e-14)


def test_frozen_rows_survive_csv_union_and_reject_any_change(tmp_path):
    old = [dict(architecture=arch, bandwidth_TBps=str(i/10), Tmax_C='69.44791686213455',
                hotspot='gpu/FEOL@(1,2,3)mm', residual='0.0007')
           for arch in M3D for i in range(36)]
    write_csv(tmp_path/'rows.csv', old + [dict(architecture='conventional_hbm_2x1', total_HBM_W=57.6)])
    new = read_csv(tmp_path/'rows.csv')
    preserve_rows(old, new)
    for field in old[0]:
        changed = deepcopy(new)
        changed[0][field] += '1'
        with pytest.raises(ValueError):
            preserve_rows(old, changed)
    with pytest.raises(ValueError):
        preserve_rows(old, new[1:])
    changed = deepcopy(new)
    changed[0]['total_HBM_W'] = '0'
    with pytest.raises(ValueError):
        preserve_rows(old, changed)
