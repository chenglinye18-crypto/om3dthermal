"""Transactional HBM-only refresh of the existing No-NMP v2 results.

Run with ``python -m om3dthermal.hbm_energy_refresh`` from the project root.
No M3D preparation or solver call is made here.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .bandwidth_thermal_sweep import _point_simulation, _row
from .case_runner import run_steady_pipeline
from .thermal.setup_cache import build_scene_and_signature, load_setup_cache
from .thermal_sensitivity import (
    SENSITIVITY_ARCHITECTURES, SOLVER_OPTIONS, crossing_85,
    plot_thermal_sensitivity, prepare_case, summarize_curve,
)

HBM = tuple(s.architecture for s in SENSITIVITY_ARCHITECTURES[:2])
M3D = tuple(s.architecture for s in SENSITIVITY_ARCHITECTURES[2:])
COMPONENTS = ('memory_internal', 'vertical', 'base_route', 'interface')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def numeric(rows):
    return [{k: v if k in ('architecture', 'hotspot') else float(v)
             for k, v in r.items()} for r in rows]


def preserve_rows(before, after):
    """Require identical old fields, order and row count; new columns stay empty."""
    old = [r for r in before if r['architecture'] in M3D]
    new = [r for r in after if r['architecture'] in M3D]
    if len(old) != 72 or len(new) != 72:
        raise ValueError('Expected exactly 72 frozen M3D rows')
    for a, b in zip(old, new):
        if any(b.get(k) != v for k, v in a.items()):
            raise ValueError('M3D field changed')
        if any(v != '' for k, v in b.items() if k not in a):
            raise ValueError('New M3D fields must be empty')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def refresh_hbm(root):
    root = Path(root).resolve()
    output = root / 'runs/no_nmp_geometry_sensitivity_v2'
    historical = root / 'runs/no_nmp_geometry_sensitivity'
    names = ('sweep.csv', 'power_breakdown.csv', 'thermal_limits.csv',
             'energy_parameter_audit.json', 'setup_audit.json', 'manifest.json',
             'tmax_vs_bandwidth.svg', 'tmax_vs_bandwidth.png')
    originals = {name: (output / name).read_bytes() for name in names}
    old = read_csv(output / 'sweep.csv')
    old_power = read_csv(output / 'power_breakdown.csv')
    preserve_rows(old, old)
    preserve_rows(old_power, old_power)
    for spec in SENSITIVITY_ARCHITECTURES[:2]:
        grid = [float(r['bandwidth_TBps']) for r in old if r['architecture'] == spec.architecture]
        if sorted(grid) != list(spec.bandwidths_TBps):
            raise ValueError('Existing HBM grid differs from frozen 25-point grid')
    energy = json.loads(originals['energy_parameter_audit.json'])
    setup = json.loads(originals['setup_audit.json'])
    manifest = json.loads(originals['manifest.json'])
    limits = [r for r in read_csv(output / 'thermal_limits.csv') if r['architecture'] in M3D]
    powers = [r for r in old_power if r['architecture'] in M3D]
    replacement = {}
    comparisons = {}
    for spec, target in zip(SENSITIVITY_ARCHITECTURES[:2], (3.0, 4.04)):
        case, simulation, memory, _, row_energy = prepare_case(root, spec)
        nominal = case.memory.nominal_read_energy
        components = {k: getattr(nominal, k + '_pj_per_bit') for k in COMPONENTS}
        closure = sum(components.values()) - target
        if abs(closure) >= 1e-12 or abs(nominal.nominal_pj_per_bit - target) >= 1e-12:
            raise ValueError('HBM read energy does not close')
        previous = numeric([r for r in old if r['architecture'] == spec.architecture])
        old_energy = next(r['memory_power_W'] / (8 * r['bandwidth_TBps'])
                          for r in previous if r['bandwidth_TBps'])
        metadata = json.loads((historical / (spec.architecture + '.json')).read_text())
        _, signature = build_scene_and_signature(simulation)
        if signature != metadata['physical_signature'] or manifest['solver'] != SOLVER_OPTIONS:
            raise ValueError('Frozen thermal physics or solver options changed')
        cache = Path(metadata['cache_path'])
        reusable, load_s, status = load_setup_cache(cache, signature)
        if status not in ('HIT', 'MISS'):
            raise RuntimeError(f'WHY_CACHE_INVALID: {cache}: {status}; no automatic rebuild')
        setup_missing = status == 'MISS'
        if setup_missing:
            cache = output / 'cache' / spec.cache_file
            print(f'MISSING_HBM_CACHE: rebuild identical validated physics at {cache}', flush=True)
        from .thermal.gpu_pcg import GPUPCGOperator, require_cupy
        gpu_operator = (None if reusable is None else
                        GPUPCGOperator.from_cpu(reusable.operator_template, require_cupy()))
        selected = []
        for bandwidth in spec.bandwidths_TBps:
            point, gpu, mem = _point_simulation(simulation, HBM[0], memory, bandwidth)
            mapped = sum(s.total_power for s in point.thermal_power_sources.sources)
            if abs(mapped - gpu - mem) > 1e-10 or abs(mem - 8 * bandwidth * target) > 1e-10:
                raise ValueError('Read power or source accounting does not close')
            pipeline = run_steady_pipeline(point, reusable_setup=reusable,
                                          setup_cache_path=cache if reusable is None else None,
                                          gpu_operator=gpu_operator, **SOLVER_OPTIONS)
            if reusable is not None and pipeline.setup_build_seconds != 0:
                raise RuntimeError('Unexpected thermal operator rebuild')
            row = _row(spec, bandwidth, gpu, mem, pipeline)
            selected.append(row)
            replacement[(spec.architecture, bandwidth)] = row
            power = dict(architecture=spec.architecture, bandwidth_TBps=bandwidth,
                         point_kind='ORIGINAL_COARSE_GRID', GPU_static_W=74.0,
                         GPU_dynamic_W=gpu - 74.0, total_memory_side_W=mem,
                         total_package_W=gpu + mem, total_HBM_W=mem)
            power.update({k + '_W': v * 8 * bandwidth for k, v in components.items()})
            power.update({k + '_W': 0.0 for k in ('MAC', 'SRAM', 'NMP_router',
                                                'NMP_NoC', 'reduction', 'NMP_unresolved')})
            powers.append(power)
            print(f"{spec.architecture} B={bandwidth}: {row['Tmax_C']:.8f} C; "
                  f"{row['solve_time_s']:.2f}s; residual={row['residual']:.6g}", flush=True)
            del pipeline
            if reusable is None:
                reusable, load_s, cache_status = load_setup_cache(cache, signature)
                if reusable is None or cache_status != 'HIT':
                    raise RuntimeError('Rebuilt HBM cache failed strict validation')
                gpu_operator = GPUPCGOperator.from_cpu(reusable.operator_template, require_cupy())
        before, after = summarize_curve(previous), summarize_curve(selected)
        old_limit, limit = crossing_85(previous), crossing_85(selected)
        indexed = {r['bandwidth_TBps']: r for r in selected}
        comparisons[spec.architecture] = dict(
            old_energy_pj_per_bit=old_energy, new_energy_pj_per_bit=target,
            temperatures_C={str(b): indexed[b]['Tmax_C'] for b in (0.0, 2.4, 4.8)},
            old_Bthermal_TBps=old_limit, Bthermal_TBps=limit,
            Bthermal_reduction_percent=100 * (1 - limit / old_limit),
            delta_Tmax_2p4_C=after['matched_2p4']['Tmax_C'] - before['matched_2p4']['Tmax_C'],
            old_slope_K_per_TBps=before['slope_K_per_TBps'],
            slope_K_per_TBps=after['slope_K_per_TBps'],
            local_2p4_slope_K_per_TBps=after['local_2p4_slope_K_per_TBps'])
        limits.append(dict(architecture=spec.architecture, Bthermal_TBps=limit,
            old_Bthermal_TBps=old_limit, delta_Bthermal_TBps=limit-old_limit,
            Tmax_2p4_C=indexed[2.4]['Tmax_C'], Tmax_4p8_C=indexed[4.8]['Tmax_C'],
            delta_Tmax_2p4_C=comparisons[spec.architecture]['delta_Tmax_2p4_C'],
            delta_Tmax_4p8_C=indexed[4.8]['Tmax_C']-next(r['Tmax_C'] for r in previous if r['bandwidth_TBps']==4.8),
            hotspot=indexed[2.4]['hotspot'], status='BRACKETED_INTERPOLATION_ORIGINAL_GRID'))
        energy[spec.architecture] = dict(target_pj_per_bit=target, old_nominal_pj_per_bit=old_energy,
            calibration_method='proportional component scaling',
            calibration_factor=row_energy['calibration_factor'] if row_energy else target/old_energy,
            calibrated_total_pj_per_bit=nominal.nominal_pj_per_bit,
            calibrated=nominal.model_dump(), component_sum_pj_per_bit=sum(components.values()),
            component_sum_closure_pj_per_bit=closure, raw_DreamRAM=row_energy,
            anchor='USER_SUPPLIED_LITERATURE_ANCHORED_TARGET; bibliography not supplied; not strict literature reproduction',
            accounting='read-only B_TBps * 8e12 bit/s * E_pJ_per_bit * 1e-12 J/pJ; zero memory static/refresh',
            status='HBM_ENERGY_REBASELINE_ONLY')
        setup[spec.architecture] = dict(physical_signature=signature,
            historical_signature=metadata['physical_signature'], setup_source=str(cache),
            setup_reused=not setup_missing, setup_load_s=load_s, solver_options=SOLVER_OPTIONS,
            cell_count=reusable.operator_template.cell_count, solves=25,
            production_setup_builds=int(setup_missing), validation='EXACT_PHYSICAL_SIGNATURE_MATCH',
            source_mapping=case.thermal['source_mapping'],
            converged=True, max_relative_residual=max(r['residual'] for r in selected))
        del gpu_operator, reusable
        gc.collect()
    rows = [replacement[(r['architecture'], float(r['bandwidth_TBps']))]
            if r['architecture'] in HBM else r for r in old]
    preserve_rows(old, rows)
    preserve_rows(old_power, powers)
    manifest['previous_provenance'] = dict(manifest)
    for key in ('HBM_source', 'HBM_source_sha256', 'regression_audit', 'source_worktree_sha256'):
        manifest.pop(key, None)
    manifest.update(source_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        status='HBM_ENERGY_REBASELINE_ONLY', HBM_status='50_HBM_POINTS_RECOMPUTED',
        M3D_status='M3D_ENERGY_AND_THERMAL_RESULTS_UNCHANGED',
        HBM_Bthermal_TBps={k:v['Bthermal_TBps'] for k,v in comparisons.items()},
        hbm_comparison=comparisons, HBM_solves=50, M3D_solves_this_refresh=0,
        M3D_preservation=dict(sweep_rows=72, power_rows=72, all_original_fields_equal=True,
            before_rows={name:[r for r in records if r['architecture'] in M3D]
                         for name,records in [('sweep.csv',old),('power_breakdown.csv',old_power)]}),
        input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root/'src/om3dthermal/hbm_energy_refresh.py',root/'src/om3dthermal/thermal_sensitivity.py',
                      *(root/'configs/cases'/s.case_file for s in SENSITIVITY_ARCHITECTURES)]})
    # Stage within v2, validate serialized rows, then replace; rollback on write failure.
    with tempfile.TemporaryDirectory(prefix='.hbm-refresh-', dir=output) as temp:
        stage = Path(temp)
        write_csv(stage/'sweep.csv', rows)
        write_csv(stage/'power_breakdown.csv', powers)
        write_csv(stage/'thermal_limits.csv', limits)
        preserve_rows(old, read_csv(stage/'sweep.csv'))
        preserve_rows(old_power, read_csv(stage/'power_breakdown.csv'))
        for name,obj in [('energy_parameter_audit.json',energy),('setup_audit.json',setup),('manifest.json',manifest)]:
            (stage/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
        plot_rows = numeric(read_csv(stage/'sweep.csv'))
        summary = {s.architecture:summarize_curve(sorted(
            [r for r in plot_rows if r['architecture']==s.architecture], key=lambda r:r['bandwidth_TBps']))
            for s in SENSITIVITY_ARCHITECTURES}
        plot_thermal_sensitivity(sorted(plot_rows,key=lambda r:(r['architecture'],r['bandwidth_TBps'])),summary,stage)
        if any((output/name).read_bytes()!=data for name,data in originals.items()):
            raise RuntimeError('Output changed concurrently; refusing replacement')
        try:
            for name in names:
                os.replace(stage/name, output/name)
        except BaseException:
            for name,data in originals.items():
                (output/name).write_bytes(data)
            raise
    return comparisons


if __name__ == '__main__':
    print(json.dumps(refresh_hbm(Path.cwd()), indent=2))
