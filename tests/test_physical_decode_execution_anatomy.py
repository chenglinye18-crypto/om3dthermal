"""Scientific checks for derived evidence, without simulation or optimization."""
import csv
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('anatomy',ROOT/'scripts/plot_physical_decode_execution_anatomy.py')
anatomy=importlib.util.module_from_spec(spec);spec.loader.exec_module(anatomy)


def test_serial_dependency_intervals_preserve_stage_latency():
    stages=[dict(layer=-1,operator='EMBED',executor='NMP',latency_s=2,components={'ARRAY':2}),
            dict(layer=0,operator='Q',executor='NMP',latency_s=7,components={'ARRAY':3,'EXTERNAL_BOUNDARY':4}),
            dict(layer=0,operator='SOFTMAX',executor='GPU',latency_s=1,components={'GPU_COMPUTE':1})]
    result=anatomy.make_timeline(stages)
    assert result[0]['start_s']==2 and result[0]['layer_start_s']==0
    assert result[0]['end_s']==result[1]['start_s']==9
    assert result[0]['dominant_resource']=='EXTERNAL_BOUNDARY'
    assert result[1]['layer_end_s']==8


def test_saved_resource_matrix_retains_nonadditive_core_equation():
    data=json.loads((anatomy.OUT/'replayed_step.json').read_text(encoding='utf-8'))
    assert math.isclose(sum(s['latency_s'] for s in data['stages']),data['latency_s'],rel_tol=1e-12)
    for s in data['stages']:
        c=s['components']
        if s['executor']=='NMP':
            expected=max(c[k] for k in ('ARRAY','LOCAL_FABRIC','MAC'))+c['EXTERNAL_BOUNDARY']+c['INTER_REGION_NOC']
        else: expected=max(c.values())
        assert math.isclose(s['latency_s'],expected,rel_tol=1e-12)
    for r in anatomy.rows(anatomy.OUT/'operator_resource_matrix.csv'):
        c=data['stages'][int(r['stage_index'])]['components']
        assert float(r['service_s'])==c.get(r['resource'],0)
        assert math.isclose(float(r['normalized_service']),c.get(r['resource'],0)/max(c.values()),rel_tol=1e-12)


def test_capacity_partition_and_paired_geomean():
    members=anatomy.rows(anatomy.OUT/'subset_membership.csv')
    assert len(members)==18 and sum(r['HBM_only_fit']=='True' for r in members)==5
    stats=anatomy.rows(anatomy.OUT/'subset_statistics.csv')
    for metric in ('tokens_per_s','tokens_per_J'):
        for numerator in ('M3D_GPU','M3D_NMP_UNIFORM','M3D_NMP_CPA'):
            q={r['subset']:float(r['geometric_mean_ratio']) for r in stats
               if r['metric']==metric and r['numerator']==numerator and r['denominator']=='HBM_GPU'}
            expected=math.exp((5*math.log(q['HBM_RESIDENT'])+13*math.log(q['HBM_OVERFLOW']))/18)
            assert math.isclose(q['ALL'],expected,rel_tol=1e-12)


def test_plot_exports_are_deterministic(tmp_path,monkeypatch):
    import shutil
    for name in ('timeline.csv','operator_resource_matrix.csv'):
        shutil.copyfile(anatomy.OUT/name,tmp_path/name)
    monkeypatch.setattr(anatomy,'OUT',tmp_path)
    anatomy.plot()
    before={ext:(tmp_path/f'figure.{ext}').read_bytes() for ext in ('svg','pdf')}
    anatomy.plot()
    assert before=={ext:(tmp_path/f'figure.{ext}').read_bytes() for ext in ('svg','pdf')}
