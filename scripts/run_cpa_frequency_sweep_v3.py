"""Existing constant-voltage CPA frequency experiment on the v3 workload matrix."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np
import formal_parallel_runtime as shared
from run_cpa_frequency_sweep import set_frequency, FREQUENCIES
from run_formal_long_context_v3 import ROOT, POINTS, inputs, read, save
from om3dthermal.serving.decode_policy import DecodePolicyModel
from om3dthermal.power.feol_energy import FEOLEnergyModel, sum_events, SCALARS, LAYERS, MAXIMA

BASE = ROOT / 'runs/formal_long_context_v3'
OUT = ROOT / 'runs/cpa_feol_frequency_sweep_v3'
CACHE = Path('F:/om3dthermal_cache/cpa_feol_frequency_sweep_v3')
BASE_COMMIT = 'b2586083763478103013dc15cbaaa4e8f4f4f376'


def key(index):
    m, c, b = POINTS[index]
    return f'{m}_{c}_B{b}'


def baseline(index):
    return read(BASE / 'candidates' / f'{key(index)}_M3D_NMP_CPA.json')


def frozen_files():
    return [p for directory in (BASE, ROOT/'runs/cpa_feol_frequency_sweep')
            for p in directory.rglob('*') if p.is_file()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_baseline():
    raw = (BASE/'final_e2e_metrics.csv').read_bytes()
    expected = subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:runs/formal_long_context_v3/final_e2e_metrics.csv'], cwd=ROOT)
    # Git stores LF while csv.writer creates CRLF in the Windows worktree.
    assert raw.replace(b'\r\n', b'\n') == expected.replace(b'\r\n', b'\n'), 'Nominal v3 benchmark changed'
    manifest = read(BASE/'manifest.json')
    assert manifest['formal_rows'] == 72 and manifest['M3D_capacity_infeasible'] == 0
    assert not manifest['thermal_over_85'] and read(BASE/'test_report.json')['new_targeted']['failed'] == 0


def build_plan(index):
    directory = CACHE/key(index)
    if (directory/'ready.json').exists():
        return directory
    start = time.perf_counter()
    m, c, b, spec, cw = inputs(index)
    engine = DecodePolicyModel(spec.decode_input(batch_size=b, context_length=cw.history+cw.prompt+cw.generated),
        project_root=ROOT, placement_policy='CRITICAL_PATH_AWARE', record_energy=True,
        record_slabs=True, decode_start_context=cw.history+cw.prompt)
    assert engine.floorplan.config['clock_hz'] == 1e9
    print('PLACED', key(index), round(time.perf_counter()-start, 1), flush=True)
    first = engine.step(cw.history+cw.prompt, 'MAC_NMP')
    old = baseline(index)['steps'][0]
    for field in ('latency_s', 'local_array_bytes', 'boundary_bytes'):
        np.testing.assert_allclose(first[field], old[field], rtol=1e-12, atol=1e-12)
    for field, value in old['component_sums'].items():
        np.testing.assert_allclose(first['component_sums'][field], value, rtol=1e-12, atol=1e-12)
    directory.mkdir(parents=True, exist_ok=True)
    engine.static.clear(); engine.dynamic.clear(); engine.context = None
    with (directory/'arrays.bin').open('wb') as arrays, (directory/'engine.pkl').open('wb') as stream:
        shared.ArrayWriter(stream, arrays).dump(engine)
    audit = dict(model=m, context=c, batch=b, placement_frequency_ghz=1., optimizer_runs=1,
        nominal_first_step_consistency='PASS', elapsed_s=time.perf_counter()-start,
        array_service_sha256=hashlib.sha256(engine.floorplan.service_ns.tobytes()).hexdigest())
    save(directory/'ready.json', audit)
    save(OUT/'placement'/f'{key(index)}.json', audit)
    del engine, first
    gc.collect()
    return directory


def execute_frequency(task):
    # Windows OS locks release automatically on exit, including interrupted runs.
    # Concurrent resumptions must not execute or overwrite the same sample.
    import msvcrt
    index, ghz = task
    target = OUT/'points'/f'{key(index)}_f{ghz:g}.json'
    lock = OUT/'execution_locks'/f'{target.name}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+b') as stream:
        stream.write(b'0'); stream.flush(); stream.seek(0)
        while True:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if target.exists() and (OUT/'checkpoints'/target.name).exists():
                    return str(target)
                time.sleep(1)
        try:
            return _execute_frequency(task)
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _execute_frequency(task):
    index, ghz = task
    target = OUT/'points'/f'{key(index)}_f{ghz:g}.json'
    if target.exists():
        return str(target)
    started = time.perf_counter()
    m, c, b, _, cw = inputs(index)
    old = baseline(index)
    engine = shared.ENGINE
    set_frequency(engine, ghz)
    steps, recorded, slab = [], [], None
    for context in cw.contexts:
        result = engine.step(context, 'MAC_NMP')
        recorded.append(result['energy_events'])
        if slab is None:
            slab = {k:np.asarray(v).copy() for k,v in result['slab_events'].items()}
        else:
            for k,v in result['slab_events'].items():
                if k in MAXIMA: slab[k] = np.maximum(slab[k], v)
                else: slab[k] += np.asarray(v)
        steps.append({k:result[k] for k in ('context','latency_s','component_sums','bottlenecks','noc_s',
            'NMP_average_utilization','noc_max_link_utilization','max_external_port_utilization',
            'fabric_region_peak_Bps','local_array_bytes','boundary_bytes')})
        if len(steps)%8 == 0:
            print('DECODE', key(index), ghz, len(steps), '/32', flush=True)
    duration = sum(s['latency_s'] for s in steps)
    events = sum_events(recorded)
    for field in SCALARS+list(LAYERS):
        np.testing.assert_allclose(events[field], old['events'][field], rtol=1e-12, atol=1e-7)
        np.testing.assert_allclose(np.asarray(slab[field]).sum(axis=0), events[field], rtol=1e-12, atol=1e-7)
    account = FEOLEnergyModel(engine.floorplan, engine.platform)
    dec = account.account(events, duration, phase='decode', policy='MAC_NMP')
    die = []
    for i in range(engine.floorplan.layout.slab_count):
        components = account.account({k:v[i].tolist() for k,v in slab.items()}, 1,
            phase='decode', policy='NO_NMP')['components']
        die.append(sum(v for k,v in components.items() if not k.startswith('gpu_')))
    die = np.asarray(die)+dec['components']['feol_unresolved_J']/len(die)
    gpu = dec['components']['gpu_dynamic_J']+dec['components']['gpu_static_J']
    np.testing.assert_allclose(die.sum()+gpu, dec['total_J'], rtol=1e-12, atol=1e-7)
    pre_s = old['prefill']['latency_s']
    pre_j = sum(v for k,v in old['energy'].items() if k.startswith('prefill_'))
    component = {k:sum(s['component_sums'][k] for s in steps) for k in steps[0]['component_sums']}
    bottlenecks = {}
    for s in steps:
        for k,v in s['bottlenecks'].items():
            bottlenecks[k] = bottlenecks.get(k, 0.)+v['time_s']
    row = dict(model=m, context=c, batch=b, H=cw.history, P=cw.prompt, G=cw.generated,
        feol_frequency_ghz=ghz, t_prefill=pre_s, t_decode=duration, t_e2e=pre_s+duration,
        e_prefill=pre_j, e_decode=dec['total_J'], e_e2e=pre_j+dec['total_J'],
        tokens_per_s=b*cw.generated/(pre_s+duration), tokens_per_j=b*cw.generated/(pre_j+dec['total_J']),
        component_sums=component, bottleneck_time_s=bottlenecks,
        dominant_bottleneck=max(bottlenecks, key=bottlenecks.get),
        noc_service_s=sum(s['noc_s'] for s in steps),
        energy_components=dec['components'], events=events,
        GPU_power_W=gpu/duration, die_power_W=(die/duration).tolist(),
        local_array_bytes=sum(s['local_array_bytes'] for s in steps), boundary_bytes=sum(s['boundary_bytes'] for s in steps),
        hardware=dict(clock_hz=engine.floorplan.config['clock_hz'], fabric_Bps=engine.floorplan.fabric_Bps,
            noc_Bps=engine.floorplan.link_Bps, tile_flops=engine.floorplan.tile_flops,
            external_Bps=engine.floorplan.external_Bps, noc_links=engine.floorplan.links),
        event_conservation='PASS', energy_conservation='PASS', elapsed_s=time.perf_counter()-started)
    row['reduction_service_s'] = component['INTER_REGION_NOC']-row['noc_service_s']
    save(target, row)
    save(OUT/'checkpoints'/target.name, dict(steps=steps))
    print('COMPLETE', key(index), ghz, row['tokens_per_s'], flush=True)
    return str(target)


def run_workload(task):
    index, workers = task
    tasks = [(index, f) for f in FREQUENCIES if not (OUT/'points'/f'{key(index)}_f{f:g}.json').exists()]
    if not tasks: return
    directory = build_plan(index)
    with ProcessPoolExecutor(max_workers=workers, initializer=shared.initialize, initargs=(str(directory),)) as pool:
        for _ in pool.map(execute_frequency, tasks): pass


def performance(args):
    validate_baseline()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUT/'preservation_sha256.json'
    if not snapshot.exists():
        save(snapshot, {str(p.relative_to(ROOT)):sha(p) for p in frozen_files()})
    save(OUT/'manifest.json', dict(baseline_commit=BASE_COMMIT, frequencies_ghz=[1., *FREQUENCIES],
        new_runs=90, nominal_rows_reused=18, workload_matrix='formal_long_context_v3',
        voltage_model='CONSTANT_VOLTAGE_EXISTING_EVENT_ENERGY', placement='REBUILD_ONCE_AT_1GHZ_THEN_FREEZE',
        frequency_model_source='scripts/run_cpa_frequency_sweep.py:set_frequency',
        cache=str(CACHE), prefill='EXACT_V3_NOMINAL_REUSE', thermal='DECODE_STEADY_STATE',
        old_sweep_values_reused=0, host_offload='PRESERVED_IN_V3_UNCHANGED'))
    order = [11,14,17,2,5,8]+[i for i in range(18) if i not in (11,14,17,2,5,8)]
    order = [i for i in order if POINTS[i][2] in args.batches]
    if args.case is not None: order=[args.case]
    with ProcessPoolExecutor(max_workers=args.cases) as pool:
        for _ in pool.map(run_workload, [(i,args.workers) for i in order]): pass


def thermal():
    from thermal_formal_long_context import load_setup_cache, GPUPCGOperator, require_cupy, SlabPowerMapper, solve_pcg_gpu, hot
    from thermal_m3d_gpu_formal_v2 import stable_source_power
    signature = read(BASE/'thermal_operator_reuse.json')['signature']
    setup, load_s, status = load_setup_cache(ROOT/'runs/formal_long_context_v2/thermal_setup.pkl', signature)
    assert setup is not None
    gpu = GPUPCGOperator.from_cpu(setup.operator_template, require_cupy())
    mapper = SlabPowerMapper(setup.cells)
    for index in range(18):
        for ghz in FREQUENCIES:
            name = f'{key(index)}_f{ghz:g}.json'
            target = OUT/'thermal'/name
            if target.exists(): continue
            # A completed checkpoint is written after its corresponding power row.
            # This permits thermal overlap once the large placement jobs release RAM.
            while not (OUT/'checkpoints'/name).exists():
                time.sleep(1)
            x = read(OUT/'points'/name)
            power = stable_source_power(mapper, x['GPU_power_W'], x['die_power_W'])
            np.testing.assert_allclose(power.sum(), x['e_decode']/x['t_decode'], rtol=1e-12, atol=1e-8)
            operator = setup.operator_template.with_power(power)
            result = solve_pcg_gpu(operator, np.full(operator.cell_count,293.15), setup.boundary_table,
                relative_residual_tolerance=1e-3, max_temperature_update_tolerance=1e-2,
                max_iterations=100000, check_interval=10, gpu_operator=gpu)
            assert result.converged and result.solver_info['dtype']=='float64'
            assert result.solver_info['full_vector_d2h_during_iteration']==0
            row = dict(**hot(result,setup), thermal_feasible=result.max_temperature_K<=358.15,
                operator_reused=True, physical_signature=signature, phase='DECODE_ONLY')
            save(target,row)
            print('THERMAL',name,row['Tmax_C'],flush=True)
    save(OUT/'thermal_operator_audit.json', dict(operator_reused=True, operator_builds=0,
        physical_signature=signature, cache_status=status, load_s=load_s, new_workload_solves=90))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--thermal',action='store_true')
    parser.add_argument('--case',type=int)
    parser.add_argument('--cases',type=int,default=3)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--batches',type=int,nargs='+',default=[1,8,32])
    args=parser.parse_args()
    thermal() if args.thermal else performance(args)
