"""Constant-voltage sensitivity on one rebuilt, fixed 1 GHz CPA plan per workload.

Only the five requested frequencies are executed. Baseline and Prefill are read
from formal_long_context_v2. Large placement arrays stay in an F: mmap cache.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for _thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_setting, '1')

import numpy as np
import formal_long_context_v2_support as f
import formal_parallel_runtime as shared
from long_context_spatial_events import project_decode, audit_global
from om3dthermal.serving.decode_policy import DecodePolicyModel
from om3dthermal.power.feol_energy import FEOLEnergyModel, sum_events

ROOT = f.ROOT
BASE = ROOT / 'runs/formal_long_context_v2'
OUT = ROOT / 'runs/cpa_feol_frequency_sweep'
CACHE = Path('F:/om3dthermal_cache/cpa_feol_frequency_sweep')
FREQUENCIES = (1.25, 1.5, 2.0, 2.5, 3.0)
BASE_COMMIT = '31c9f3249f7d2ac1a8ad6ae506d83264b3223edf'
save = f.save


def key(m, c, b):
    return f'{m}_{c}_B{b}'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def baseline_rows():
    raw = (BASE / 'final_e2e_metrics.csv').read_bytes()
    expected = subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:runs/formal_long_context_v2/final_e2e_metrics.csv'], cwd=ROOT)
    assert raw == expected, 'Frozen baseline changed'
    rows = list(csv.DictReader(raw.decode('utf-8').splitlines()))
    assert len(rows) == 72
    return [r for r in rows if r['path'] == 'M3D_NMP_CPA']


def set_frequency(engine, ghz):
    """Keep physical wire delay and pipeline hardware; scale synchronous slack.

    Baseline hop = router cycles + RC rounded up to the baseline clock.
    Preserve the RC component, and scale only the remaining clock component.
    Do not add pipeline registers at higher frequencies (energy stays fixed).
    """
    floor = engine.floorplan
    if not hasattr(floor, '_frequency_baseline_links'):
        floor._frequency_baseline_links = deepcopy(floor.links)
    floor.config['clock_hz'] = ghz * 1e9
    for link, old in zip(floor.links, floor._frequency_baseline_links):
        link['hop_ns'] = old['rc_ns'] + (old['hop_ns'] - old['rc_ns']) / ghz
    # ingress_lookup is deliberately retained from the 1 GHz plan: no rerouting.
    engine.static.clear()
    engine.dynamic.clear()
    engine.context = None


def execute_frequency(task):
    m, c, b, ghz = task
    tag = key(m, c, b)
    target = OUT / 'points' / f'{tag}_f{ghz:g}.json'
    if target.exists():
        return str(target)
    started = time.perf_counter()
    engine = shared.ENGINE
    set_frequency(engine, ghz)
    _, cw, _ = f.legacy.inputs(m, c, b, ROOT)
    steps = []
    for context in cw.contexts:
        s = engine.step(context, 'MAC_NMP')
        steps.append({k: s[k] for k in ('context', 'latency_s', 'component_sums',
            'energy_events', 'bottlenecks', 'noc_s', 'NMP_average_utilization',
            'noc_max_link_utilization', 'max_external_port_utilization',
            'fabric_region_peak_Bps', 'local_array_bytes', 'boundary_bytes')})
        if len(steps) % 8 == 0:
            print('DECODE_PROGRESS', tag, ghz, f'{len(steps)}/{cw.generated}', flush=True)
    seconds = sum(s['latency_s'] for s in steps)
    events = sum_events(s['energy_events'] for s in steps)
    projection = read(CACHE / tag / 'projection.json')
    audit_global({k: np.array(v) for k, v in projection['events'].items()}, events)
    energy_model = FEOLEnergyModel(engine.floorplan, engine.platform)
    dec = energy_model.account(events, seconds, phase='decode', policy='MAC_NMP')
    old = read(BASE / 'candidates' / f'{tag}_M3D_NMP_CPA.json')
    pre_s = old['summary']['prefill_s']
    pre_j = old['energy']['prefill_J']
    n = b * cw.generated
    component = {k: sum(s['component_sums'][k] for s in steps) for k in steps[0]['component_sums']}
    noc = sum(s['noc_s'] for s in steps)
    bottlenecks = {}
    for s in steps:
        for k, v in s['bottlenecks'].items():
            bottlenecks[k] = bottlenecks.get(k, 0.) + v['time_s']
    die_j = np.array(projection['die_dynamic_J']) + dec['components']['feol_unresolved_J'] / engine.floorplan.layout.slab_count
    gpu_j = dec['components']['gpu_dynamic_J'] + dec['components']['gpu_static_J']
    np.testing.assert_allclose(die_j.sum() + gpu_j, dec['total_J'], rtol=1e-12, atol=1e-7)
    row = dict(model=m, context=c, cached_history=cw.history, batch=b,
        prefill_tokens=cw.prompt, decode_tokens=cw.generated, policy='M3D_NMP_CPA',
        feol_frequency_ghz=ghz, t_prefill=pre_s, t_decode=seconds,
        t_e2e=pre_s + seconds, tokens_per_s=n / (pre_s + seconds),
        e_prefill=pre_j, e_decode=dec['total_J'], e_e2e=pre_j + dec['total_J'],
        tokens_per_j=n / (pre_j + dec['total_J']),
        component_sums=component, noc_service_s=noc,
        reduction_service_s=component['INTER_REGION_NOC'] - noc,
        miv_service_status='INCLUDED_IN_ARRAY_GROUP_SERVICE_NOT_SEPARATELY_ADDITIVE',
        bottleneck_time_s=bottlenecks, dominant_bottleneck=max(bottlenecks, key=bottlenecks.get),
        NMP_average_utilization=sum(s['NMP_average_utilization'] * s['latency_s'] for s in steps) / seconds,
        peak_noc_utilization=max(s['noc_max_link_utilization'] for s in steps),
        peak_external_port_utilization=max(s['max_external_port_utilization'] for s in steps),
        fabric_region_peak_Bps=max(s['fabric_region_peak_Bps'] for s in steps),
        local_array_bytes=sum(s['local_array_bytes'] for s in steps),
        boundary_bytes=sum(s['boundary_bytes'] for s in steps),
        energy_components=dec['components'], GPU_power_W=gpu_j / seconds,
        die_power_W=(die_j / seconds).tolist(),
        hardware=dict(clock_hz=engine.floorplan.config['clock_hz'],
            fabric_Bps=engine.floorplan.fabric_Bps, noc_Bps=engine.floorplan.link_Bps,
            tile_flops=engine.floorplan.tile_flops, external_Bps=engine.floorplan.external_Bps,
            noc_links=engine.floorplan.links),
        event_conservation='PASS', energy_conservation='PASS',
        placement_status='REBUILT_AT_1GHZ_THEN_FROZEN_NO_OLD_AUDIT_COMPARISON',
        elapsed_s=time.perf_counter() - started)
    save(target, row)
    print('FREQUENCY_COMPLETE', tag, ghz, row['tokens_per_s'], flush=True)
    return str(target)


def build_plan(m, c, b):
    directory = CACHE / key(m, c, b)
    if (directory / 'ready.json').exists():
        return directory
    started = time.perf_counter()
    w, cw, _ = f.legacy.inputs(m, c, b, ROOT)
    engine = DecodePolicyModel(w, project_root=ROOT, placement_policy='CRITICAL_PATH_AWARE',
        record_energy=True, decode_start_context=cw.history + cw.prompt)
    assert engine.floorplan.config['clock_hz'] == 1e9
    assert engine.floorplan.external_Bps == 3.4e12
    print('PLACEMENT_REBUILT', key(m, c, b), time.perf_counter() - started, flush=True)
    slab = project_decode(engine.placement, cw, nmp=True)
    em = FEOLEnergyModel(engine.floorplan, engine.platform)
    die = []
    for i in range(engine.floorplan.layout.slab_count):
        ev = {k: v[i].tolist() for k, v in slab.items()}
        # Same formal per-die accounting: NMP events are already explicit;
        # suppress whole-stack background here and add it once at actual duration.
        components = em.account(ev, 1, phase='decode', policy='NO_NMP')['components']
        die.append(sum(v for k, v in components.items() if not k.startswith('gpu_')))
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / 'projection.json', dict(events=slab, die_dynamic_J=die))
    with (directory / 'arrays.bin').open('wb') as arrays, (directory / 'engine.pkl').open('wb') as stream:
        shared.ArrayWriter(stream, arrays).dump(engine)
    save(directory / 'ready.json', dict(model=m, context=c, batch=b,
        placement_frequency_ghz=1, optimizer_runs=1, old_audit_compared=False,
        elapsed_s=time.perf_counter() - started,
        array_service_sha256=hashlib.sha256(engine.floorplan.service_ns.tobytes()).hexdigest()))
    del engine, slab
    gc.collect()
    return directory


def run_workload(task):
    m, c, b, workers = task
    tasks = [(m, c, b, ghz) for ghz in FREQUENCIES
             if not (OUT / 'points' / f'{key(m,c,b)}_f{ghz:g}.json').exists()]
    if not tasks:
        return
    directory = build_plan(m, c, b)
    with ProcessPoolExecutor(max_workers=workers, initializer=shared.initialize,
                             initargs=(str(directory),)) as pool:
        for result in pool.map(execute_frequency, tasks):
            print('SAVED', result, flush=True)


def performance(workers, cases):
    baseline_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'manifest.json', dict(baseline_commit=BASE_COMMIT, old_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), frequencies_ghz=FREQUENCIES,
        new_runs=90, baseline_runs_reused=18, voltage_model='CONSTANT_VOLTAGE_EXISTING_EVENT_ENERGY',
        placement='REBUILD_ONCE_AT_1GHZ_THEN_FREEZE', cache=str(CACHE),
        concurrent_workloads=cases, frequency_workers_per_workload=workers,
        prefill='FROZEN_BASELINE_REUSED', thermal='DECODE_ONLY_STEADY_STATE',
        noc_hop='rc_ns + (baseline_hop_ns - rc_ns) / frequency_ratio; pipeline registers unchanged',
        tests='NO_PYTEST_USER_REQUEST'))
    tasks = [(m, c, b, workers) for m, c, b in f.points()]
    with ProcessPoolExecutor(max_workers=cases) as pool:
        for _ in pool.map(run_workload, tasks):
            pass
    baseline_rows()


def thermal(wait):
    from thermal_formal_long_context import (ARCHITECTURES, _base_case, build_scene_and_signature,
        load_setup_cache, GPUPCGOperator, require_cupy, SlabPowerMapper, solve_pcg_gpu, hot)
    _, simulation, _ = _base_case(ROOT, ARCHITECTURES[1])
    _, signature = build_scene_and_signature(simulation)
    cache = ROOT / f.CONFIG['thermal']['operator_cache']
    setup, load_s, status = load_setup_cache(cache, signature)
    assert setup is not None, 'Existing thermal operator required'
    gpu = GPUPCGOperator.from_cpu(setup.operator_template, require_cupy())
    mapper = SlabPowerMapper(setup.cells)
    for m, c, b in f.points():
        for ghz in FREQUENCIES:
            name = f'{key(m,c,b)}_f{ghz:g}.json'
            target = OUT / 'thermal' / name
            if target.exists():
                continue
            source = OUT / 'points' / name
            while wait and not source.exists():
                time.sleep(2)
            if not source.exists():
                continue
            x = read(source)
            power = mapper.power(x['GPU_power_W'], x['die_power_W'])
            np.testing.assert_allclose(power.sum(), x['e_decode']/x['t_decode'], rtol=1e-12, atol=1e-8)
            operator = setup.operator_template.with_power(power)
            result = solve_pcg_gpu(operator, np.full(operator.cell_count, 293.15), setup.boundary_table,
                relative_residual_tolerance=1e-3, max_temperature_update_tolerance=1e-2,
                max_iterations=100000, check_interval=10, gpu_operator=gpu)
            assert result.converged and result.solver_info['dtype'] == 'float64'
            assert result.solver_info['full_vector_d2h_during_iteration'] == 0
            row = dict(**hot(result, setup), thermal_feasible=result.max_temperature_K <= 358.15,
                operator_reused=True, physical_signature=signature, phase='DECODE_ONLY')
            save(target, row)
            print('THERMAL_COMPLETE', name, row['Tmax_C'], flush=True)
    save(OUT / 'thermal_operator_audit.json', dict(operator_reused=True, operator_builds=0,
        physical_signature=signature, cache_status=status, load_s=load_s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thermal', action='store_true')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--cases', type=int, default=2)
    args = parser.parse_args()
    thermal(args.wait) if args.thermal else performance(args.workers, args.cases)
