import csv
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_cpa_frequency_sweep_v3 import set_frequency, BASE, read, key, POINTS
from analyze_cpa_frequency_sweep_v3 import select_feasible


def test_existing_clock_scaling_preserves_rc_and_pipeline_hardware():
    floor=SimpleNamespace(config={'clock_hz':1e9},links=[{'rc_ns':.35,'hop_ns':3.,'wire_pipeline_cycles':2}])
    engine=SimpleNamespace(floorplan=floor,static={'old':1},dynamic={'old':1},context=100)
    for frequency in (1.25,3.,1.5):
        set_frequency(engine,frequency)
        assert floor.config['clock_hz']==frequency*1e9
        assert floor.links[0]['hop_ns']==pytest.approx(.35+2.65/frequency)
        assert floor.links[0]['rc_ns']==.35 and floor.links[0]['wire_pipeline_cycles']==2
        assert engine.static==engine.dynamic=={} and engine.context is None


def test_best_frequency_selects_feasible_throughput_not_frequency_or_energy():
    rows=[dict(thermal_feasible=True,tokens_per_s=10,feol_frequency_ghz=1,tokens_per_j=9),
          dict(thermal_feasible=True,tokens_per_s=12,feol_frequency_ghz=2,tokens_per_j=7),
          dict(thermal_feasible=False,tokens_per_s=15,feol_frequency_ghz=3,tokens_per_j=10)]
    assert select_feasible(rows) is rows[1]
    rows[2]['thermal_feasible']=True;rows[2]['tokens_per_s']=11
    assert select_feasible(rows) is rows[1]


def test_both_hbm_policies_remain_for_all_18_v3_workloads():
    from om3dthermal.serving.cached_history_wave import select_hbm_best
    with (BASE/'capacity_audit.csv').open() as stream:capacity=list(csv.DictReader(stream))
    for index,(_,_,b) in enumerate(POINTS):
        get=lambda p:read(BASE/'candidates'/f'{key(index)}_{p}.json')
        host,wave,best=[get(p) for p in ('HBM_HOST_OFFLOAD','HBM_RESIDENT_WAVE','HBM_BEST')]
        assert host['status']=='EVALUATED'
        assert select_hbm_best(host,wave)==best
        if capacity[index]['HBM_full_batch_fit']=='True':
            assert host['traffic']['host_read_bytes']==host['traffic']['host_write_bytes']==0
            assert wave['wave_sizes']==[b] and wave['traffic']['admission_bytes']==0
        assert select_hbm_best(host,dict(status='CAPACITY_INFEASIBLE'))['selected_HBM_policy']=='HBM_HOST_OFFLOAD'
