"""No-NMP stack-depth/slab-thickness sensitivity using the frozen solver."""

from __future__ import annotations

import csv
import gc
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from copy import copy

import numpy as np

from .architecture import resolve_packing_from_legacy_power_result
from .architecture_comparison import (
    _resolve_case_power_operating_point_kwargs, compile_canonical_thermal_case,
)
from .bandwidth_thermal_sweep import (
    ARCHITECTURES, M3D_MESH_M, SweepArchitecture, _point_simulation, _row,
)
from .case_runner import run_steady_pipeline
from .config import CellSizeConfig
from .power import calculate_memory_power, load_case_config, resolve_case_geometry, resolve_system_power
from .power.config import HBMNominalReadEnergyInput, RowPolicy
from .thermal.setup_cache import load_setup_cache


def no_nmp_read_energy(root, case, memory):
    """One 32-B read at every physical slab/group/layer, uniformly stressed.

    The sweep prescribes traffic, not a resident model. Equal service activity
    matches its uniform BEOL source mapping; routes and MIV remain physical.
    """
    from .architecture.feol_floorplan import resolve_feol_floorplan
    from .power.feol_energy import FEOLEnergyModel, memory_events
    from .power.nmp_die_activity import external_service
    from .power.m3d_subarray import calculate_m3d_subarray
    from .power.feol_route import calculate_feol_route
    from .platform import load_platform_spec_file

    f = copy(resolve_feol_floorplan(root))
    topology = calculate_m3d_subarray(case.architecture.m3d_subarray,
                                    resolve_case_geometry(case).m3d)
    route = calculate_feol_route(case.architecture, topology)
    assert np.array_equal(route.feol_route_cluster_centers_um,
                          [g['center_um'] for g in f.clusters])
    assert np.array_equal(route.feol_io_channel_coordinates_um, f.ports)
    f.case, f.layout = case, memory.physical_capacity_layout
    d = memory.diagnostics
    f.miv_pj_per_bit_by_layer = (np.array(d['miv_effective_capacitance_per_layer_pF'])
        * d['miv_access_energy_pJ_per_bit'] / d['miv_average_effective_capacitance_pF'])
    b = np.full((f.layout.slab_count, len(f.groups), f.layout.layers_per_cluster), 32, dtype=np.int64)
    events = memory_events(b.reshape(-1, b.shape[-1]), write=False)
    transfer = external_service(f, b.sum(axis=2), mode='GROUP_DIRECT', record_events=True)
    events['sa_to_edge_bit_um'] = transfer['wire_bit_um']
    events['interface_bits'] = events['gpu_decode_proxy_bits'] = float(b.sum()*8)
    platform = load_platform_spec_file(root/'configs/platform/gpu_package_h200_reference.yaml')
    model = FEOLEnergyModel(f, platform)
    energy = model.account(events, 1, phase='decode', policy='NO_NMP')['components']
    fields = dict(array_read='array_read_J', read_peripheral='read_peripheral_J',
                  miv_effective='miv_J', feol_route_effective='feol_wire_J', interface='interface_J')
    effective = {k+'_pJ_per_bit': energy[v]*1e12/events['interface_bits'] for k,v in fields.items()}
    effective['total_memory_side_pJ_per_bit'] = sum(effective.values())
    assert all(energy[k+'_J'] == 0 for k in ('mac','sram_read','sram_write','router',
               'pipeline_register','reduction','feol_unresolved','row_select','column_select','write_driver'))
    return dict(effective=effective, events=events, parameters=model.audit(),
                slabs=f.layout.slab_count, capacity_GB=f.layout.total_capacity_bytes/1e9,
                activity='UNIFORM_READ_SERVICES_PER_PHYSICAL_SLAB_GROUP_LAYER',
                gpu_static_W=platform.gpu_decode_power.static_power_W,
                gpu_dynamic_pJ_per_bit=platform.gpu_decode_power.e_decode_J_per_bit*1e12)


SENSITIVITY_ARCHITECTURES = (
    ARCHITECTURES[0],
    SweepArchitecture("conventional_hbm_24hi_sensitivity",
                      "conventional_hbm_24hi_sensitivity.yaml",
                      ARCHITECTURES[0].bandwidths_TBps, "hbm_24hi.pkl"),
    ARCHITECTURES[1],
    SweepArchitecture("orthogonal_m3d_300um_sensitivity",
                      "orthogonal_m3d_300um_sensitivity.yaml",
                      ARCHITECTURES[1].bandwidths_TBps, "m3d_300um_cu.pkl"),
)
LABELS = ("HBM 12-high (nominal)", "HBM 24-high (hypothetical)",
          "M3D 100 µm / 318 slabs", "M3D 300 µm / 106 slabs")
SOLVER_OPTIONS = dict(backend="gpu_pcg", rtol=1e-3, max_delta_t_K=1e-2,
                      max_iterations=100_000, check_interval=10,
                      initial_temperature_K=293.15)


def resolve_hbm_row_mean(case, project_root: Path):
    """Run both row states without the nominal energy override; average components."""
    case = case.model_copy(update={"memory": case.memory.model_copy(
        update={"nominal_read_energy": None})})
    geometry = resolve_case_geometry(case)

    def row(utilization):
        workload = case.workload.model_copy(update={"row_policy": RowPolicy(
            activated_row_data_utilization=utilization)})
        return calculate_memory_power(case.model_copy(update={"workload": workload}),
                                      project_root=project_root, geometry=geometry,
                                      read_bandwidth_gbps=0.0)

    full = row(1.0)
    closed = row(1.0 / full.diagnostics["atoms_per_page"])
    if closed.diagnostics["effective_rd_per_act"] != 1.0:
        raise ValueError("closed row must contain exactly one RD per ACT/PRE")
    fields = {name: 0.5 * (getattr(full, field) + getattr(closed, field))
              for name, field in (
                  ("memory_internal_pj_per_bit", "E_memory_internal_pj_bit"),
                  ("vertical_pj_per_bit", "E_vertical_pj_bit"),
                  ("base_route_pj_per_bit", "E_base_route_pj_bit"),
                  ("interface_pj_per_bit", "E_interface_pj_bit"))}
    mean = HBMNominalReadEnergyInput(
        full_row_pj_per_bit=full.E_access_total_pj_bit,
        closed_row_pj_per_bit=closed.E_access_total_pj_bit,
        aggregation="ARITHMETIC_MEAN", **fields)
    resolved = case.model_copy(update={"memory": case.memory.model_copy(
        update={"nominal_read_energy": mean})})
    return resolved, {"full_row": full.as_dict(), "closed_row": closed.as_dict(),
                      "mean": mean.model_dump(), "mean_pj_per_bit": mean.nominal_pj_per_bit,
                      "status": "DREAMRAM_AVERAGE_CROSSED_LAYER_SCALING_NOT_PRODUCT_VALIDATED"}


def prepare_case(root: Path, spec: SweepArchitecture):
    case = load_case_config(root / "configs/cases" / spec.case_file)
    row_energy = None
    if spec.architecture == "conventional_hbm_24hi_sensitivity":
        case, row_energy = resolve_hbm_row_mean(case, root)
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(case, project_root=root, geometry=geometry,
                                 **_resolve_case_power_operating_point_kwargs(case, root))
    simulation = compile_canonical_thermal_case(case, system)
    if case.geometry.type == "orthogonal_m3d":
        simulation = simulation.model_copy(update={"discretization":
            simulation.discretization.model_copy(update={"max_cell_size":
                CellSizeConfig(x=M3D_MESH_M[0], y=M3D_MESH_M[1], z=M3D_MESH_M[2])})})
    memory = system.memory_result
    if memory is None:
        raise ValueError("sensitivity requires resolved memory power")
    packing = resolve_packing_from_legacy_power_result(case, geometry, memory)
    return case, simulation, memory, packing, row_energy


def summarize_curve(rows):
    """Common-domain OLS and centered 2.4 TB/s secant, in K/(TB/s)."""
    common = [row for row in rows if row["bandwidth_TBps"] <= 4.8]
    bandwidth = np.array([row["bandwidth_TBps"] for row in common])
    temperature = np.array([row["Tmax_C"] for row in common])
    slope, intercept = np.polyfit(bandwidth, temperature, 1)
    indexed = {row["bandwidth_TBps"]: row for row in rows}
    local = (indexed[2.6]["Tmax_C"] - indexed[2.2]["Tmax_C"]) / 0.4
    return {"matched_2p4": indexed[2.4], "slope_K_per_TBps": float(slope),
            "fit_domain_TBps": [0.0, 4.8], "local_2p4_slope_K_per_TBps": local,
            "fit_max_abs_error_K": float(np.max(np.abs(temperature - (slope * bandwidth + intercept)))),
            "segment_slopes_K_per_TBps": [
                (b["Tmax_C"] - a["Tmax_C"]) / (b["bandwidth_TBps"] - a["bandwidth_TBps"])
                for a, b in zip(rows, rows[1:])]}


def run_thermal_sensitivity(output_dir, *, project_root, nominal_cache_dir=None):
    root, output = Path(project_root).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite sensitivity output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "cache").mkdir()
    rows, summary = [], {}
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for spec in SENSITIVITY_ARCHITECTURES:
            case, simulation, memory, packing, row_energy = prepare_case(root, spec)
            family = ("conventional_hbm_2x1" if case.geometry.type == "dreamram_hbm"
                      else "orthogonal_m3d_igzo")
            cache = output / "cache" / spec.cache_file
            if nominal_cache_dir is not None and spec in ARCHITECTURES:
                cache = Path(nominal_cache_dir).resolve() / spec.cache_file
                if not cache.is_file():
                    raise FileNotFoundError(cache)
            selected, reusable = [], None
            metadata = {"capacity_GB": packing.total_bits / 8e9,
                        "capacity_GiB": packing.total_bits / 8 / 2**30,
                        "read_energy_pj_per_bit": memory.E_access_total_pj_bit,
                        "memory_static_refresh_W": 0.0, "row_energy": row_energy,
                        "case": case.model_dump(mode="json"), "cache_path": str(cache)}
            for bandwidth in spec.bandwidths_TBps:
                point, gpu_power, memory_power = _point_simulation(
                    simulation, family, memory, bandwidth)
                pipeline = run_steady_pipeline(
                    point, setup_cache_path=cache if reusable is None else None,
                    reusable_setup=reusable, **SOLVER_OPTIONS)
                row = _row(spec, bandwidth, gpu_power, memory_power, pipeline)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                selected.append(row)
                print(f"{spec.architecture} B={bandwidth:.1f}: {row['Tmax_C']:.6f} C [{pipeline.cache_status}]", flush=True)
                if reusable is None:
                    signature = pipeline.cache_physical_signature
                    metadata.update(cache_status=pipeline.cache_status,
                                    physical_signature=signature, cell_count=pipeline.cell_count,
                                    edge_count=pipeline.internal_edge_count)
                    del pipeline
                    gc.collect()
                    reusable, _, status = load_setup_cache(cache, signature)
                    if reusable is None or status != "HIT":
                        raise RuntimeError("new operator cache did not reload")
                else:
                    del pipeline
            rows.extend(selected)
            metadata.update(summarize_curve(selected))
            summary[spec.architecture] = metadata
            (output / f"{spec.architecture}.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8")
            del reusable
            gc.collect()
    slopes = [summary[spec.architecture]["slope_K_per_TBps"] for spec in SENSITIVITY_ARCHITECTURES]
    comparison = {"hbm_24hi_vs_nominal_slope_change_percent": 100 * (slopes[1] / slopes[0] - 1),
                  "m3d_100um_vs_300um_slope_change_percent": 100 * (slopes[2] / slopes[3] - 1)}
    payload = {"architectures": summary, "comparison": comparison,
               "solver_options": SOLVER_OPTIONS,
               "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "input_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                   for p in [*(root / "configs/cases" / s.case_file for s in SENSITIVITY_ARCHITECTURES),
                             root / "configs/platform/gpu_package_h200_reference.yaml"]}}
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_thermal_sensitivity(rows, summary, output)
    return {"output_dir": str(output), "rows": len(rows), **comparison}


def plot_thermal_sensitivity(rows, summary, output: Path, *, reused_hbm=False):
    """Plot stored results, with a zoom for the closely spaced nominal/M3D curves."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    zoom = axis.inset_axes([0.57, 0.15, 0.4, 0.38])
    for spec, label, color, style in zip(SENSITIVITY_ARCHITECTURES, LABELS,
                                       ("#245a81", "#245a81", "#bd5a36", "#bd5a36"),
                                       ("-", "--", "-", "--")):
        if reused_hbm and spec.architecture.startswith('conventional_hbm'):
            label += ' [reused]'
        curve = [row for row in rows if row["architecture"] == spec.architecture]
        axis.plot([r["bandwidth_TBps"] for r in curve], [r["Tmax_C"] for r in curve],
                  label=label, color=color, linestyle=style, linewidth=2)
        matched = summary[spec.architecture]["matched_2p4"]
        axis.scatter([2.4], [matched["Tmax_C"]], color=color, s=24)
        if spec.architecture != "conventional_hbm_24hi_sensitivity":
            detail = [row for row in curve if 2.2 <= row["bandwidth_TBps"] <= 2.6]
            zoom.plot([r["bandwidth_TBps"] for r in detail], [r["Tmax_C"] for r in detail],
                      color=color, linestyle=style, linewidth=1.5)
            zoom.scatter([2.4], [matched["Tmax_C"]], color=color, s=12)
    zoom.axvline(2.4, color="0.65", linestyle=":", linewidth=1)
    zoom.set(xlim=(2.2, 2.6), xticks=(2.2, 2.4, 2.6))
    zoom.set_title("2.4 TB/s detail: HBM 12-high / M3D", fontsize=8)
    zoom.set_ylabel("Tmax (°C)", fontsize=8)
    zoom.tick_params(labelsize=8)
    zoom.grid(alpha=0.2)
    axis.axvline(2.4, color="0.65", linestyle=":", linewidth=1)
    axis.axhline(85, color='0.5', linestyle=':', linewidth=1)
    axis.text(6.65, 86, '85°C', ha='right', color='0.4', fontsize=9)
    axis.set(xlabel="Prescribed external bandwidth (TB/s)", ylabel="Maximum temperature (°C)",
             title="No-NMP geometry sensitivity · zero memory static/refresh power")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "tmax_vs_bandwidth.png", dpi=200)
    figure.savefig(output / "tmax_vs_bandwidth.svg")
    svg = output / "tmax_vs_bandwidth.svg"
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(figure)


def crossing_85(rows):
    """Bracketed interpolation only; never extrapolate a thermal limit."""
    rows = sorted(rows, key=lambda r:r['bandwidth_TBps'])
    for a,b in zip(rows, rows[1:]):
        if a['Tmax_C'] <= 85 <= b['Tmax_C']:
            return a['bandwidth_TBps']+(85-a['Tmax_C'])*(b['bandwidth_TBps']-a['bandwidth_TBps'])/(b['Tmax_C']-a['Tmax_C'])
    raise ValueError('85 C crossing not bracketed by prescribed grid')


def run_read_peripheral_reclosure(output_dir, *, project_root, historical_dir,
                                  thin_cache, thick_cache, resume=False):
    """Reclose only M3D, requiring validated existing operators before solving."""
    from .thermal.setup_cache import build_scene_and_signature
    root, output, historical = map(lambda p:Path(p).resolve(),
                                  (project_root, output_dir, historical_dir))
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    def dump(name, obj):
        (output/name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    def csv_out(name, records):
        with (output/name).open('w', newline='', encoding='utf-8') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    old=[]
    with (historical/'curves.csv').open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            old.append({k: v if k in ('architecture','hotspot') else float(v) for k,v in row.items()})
    rows=[dict(r) for r in old if r['architecture'].startswith('conventional_hbm')]
    previous_audits={}
    previous_energy={}
    if resume:
        previous_audits=json.loads((output/'setup_audit.json').read_text(encoding='utf-8'))
        previous_energy=json.loads((output/'energy_parameter_audit.json').read_text(encoding='utf-8'))
        with (output/'sweep.csv').open(encoding='utf-8') as stream:
            rows=[{k:v if k in ('architecture','hotspot') else float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    summary={s.architecture:summarize_curve([r for r in rows if r['architecture']==s.architecture])
             for s in SENSITIVITY_ARCHITECTURES[:2]}
    audits, energy_audits, powers, limits = {}, {}, [], []
    if resume:
        with (output/'power_breakdown.csv').open(encoding='utf-8') as stream:
            powers=[{k:v if k in ('architecture','point_kind') else float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    for spec, cache_path in zip(SENSITIVITY_ARCHITECTURES[2:], (thin_cache, thick_cache)):
        case, simulation, memory, _, _ = prepare_case(root, spec)
        energy = no_nmp_read_energy(root,case,memory)
        energy['legacy_aggregate_model'] = {k:getattr(memory,k) for k in
            ('E_memory_internal_pj_bit','E_vertical_pj_bit','E_feol_route_pj_bit','E_base_route_pj_bit','E_interface_pj_bit','E_access_total_pj_bit')}
        if spec.architecture in previous_energy and previous_energy[spec.architecture] != energy:
            raise ValueError('Cannot resume after energy/path model change')
        energy_audits[spec.architecture]=energy
        dump('energy_parameter_audit.json',energy_audits)
        print('READ_ENERGY_AUDIT '+spec.architecture+' '+json.dumps(energy['effective']), flush=True)
        boxes, signature = build_scene_and_signature(simulation)
        cached_metadata=json.loads((historical/(spec.architecture+'.json')).read_text(encoding='utf-8'))
        if cached_metadata['physical_signature'] != signature:
            raise RuntimeError('WHY_CACHE_INVALID: current geometry/mesh/material/BC signature differs from historical canonical provenance')
        prior=previous_audits.get(spec.architecture)
        if prior and (prior['physical_signature']!=signature or prior['solver_options']!=SOLVER_OPTIONS):
            raise ValueError('Cannot resume after thermal physics or solver tolerance change')
        cache=Path(cache_path).resolve()
        print('LOAD_VALIDATED_SETUP '+str(cache),flush=True)
        reusable, load_s, status=load_setup_cache(cache,signature)
        if reusable is None or status!='HIT':
            raise RuntimeError(f'WHY_CACHE_INVALID: {cache}: {status}; refusing automatic rebuild')
        digest=lambda obj:sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
        audits[spec.architecture]=dict(setup_source=str(cache), setup_reused=True, setup_load_s=load_s,
            physical_signature=signature, historical_signature=cached_metadata['physical_signature'],
            validation='EXACT_COMBINED_GEOMETRY_MESH_MATERIAL_CONDUCTANCE_BC_SCHEMA_MATCH',
            geometry_hash=digest([(b.name,b.material,b.x0,b.x1,b.y0,b.y1,b.z0,b.z1,b.tags) for b in boxes]),
            mesh_hash=digest(simulation.discretization.model_dump(mode='json')),
            material_hash=digest({k:v.model_dump(mode='json') for k,v in simulation.materials.items()}),
            boundary_hash=digest(simulation.thermal_boundary_conditions.model_dump(mode='json')),
            thermal_conductance_hash=digest(simulation.thermal_conductance.model_dump(mode='json')),
            solver_options=SOLVER_OPTIONS,cell_count=reusable.operator_template.cell_count,
            operator_hash=signature, power_in_cache_signature=False)
        dump('setup_audit.json',audits)
        from .thermal.gpu_pcg import GPUPCGOperator, require_cupy
        import time
        started=time.perf_counter()
        gpu_operator=GPUPCGOperator.from_cpu(reusable.operator_template,require_cupy())
        audits[spec.architecture].update(gpu_operator_builds_this_session=1,
            gpu_operator_prepare_s=time.perf_counter()-started,
            resumed_completed_points=sum(r['architecture']==spec.architecture for r in rows),
            prior_setup_load_s=prior['setup_load_s'] if prior else None)
        selected=[r for r in rows if r['architecture']==spec.architecture]
        effective=energy['effective']
        proxy=SimpleNamespace(E_access_total_pj_bit=effective['total_memory_side_pJ_per_bit'])
        assert energy['gpu_static_W']==74 and energy['gpu_dynamic_pJ_per_bit']==11.68
        def solve(bandwidth, kind):
            point,gpu,mem=_point_simulation(simulation,'orthogonal_m3d_igzo',proxy,bandwidth)
            assert np.isclose(sum(s.total_power for s in point.thermal_power_sources.sources),gpu+mem,rtol=1e-14)
            pipeline=run_steady_pipeline(point,reusable_setup=reusable,gpu_operator=gpu_operator,**SOLVER_OPTIONS)
            assert pipeline.setup_build_seconds==0
            row=_row(spec,bandwidth,gpu,mem,pipeline)
            selected.append(row);rows.append(row)
            power=dict(architecture=spec.architecture,bandwidth_TBps=bandwidth,point_kind=kind,
                       GPU_static_W=energy['gpu_static_W'],GPU_dynamic_W=gpu-energy['gpu_static_W'])
            power.update({k.replace('_pJ_per_bit','_W'):v*8*bandwidth for k,v in effective.items()})
            power.update(MAC_W=0,SRAM_W=0,NMP_router_W=0,NMP_NoC_W=0,reduction_W=0,NMP_unresolved_W=0,
                         total_M3D_W=mem,total_package_W=gpu+mem)
            powers.append(power)
            csv_out('sweep.csv',rows);csv_out('power_breakdown.csv',powers)
            print(f"{spec.architecture} B={bandwidth:.8f}: {row['Tmax_C']:.8f} C; {row['solve_time_s']:.3f}s, {row['iterations']} iterations",flush=True)
            return row
        for bandwidth in spec.bandwidths_TBps:
            if not any(r['bandwidth_TBps']==bandwidth for r in selected):
                solve(bandwidth,'ORIGINAL_COARSE_GRID')
        coarse=[r for r in selected if r['bandwidth_TBps'] in spec.bandwidths_TBps]
        # Numerical crossing refinement, not a change to solver tolerances.
        for _ in range(4):
            check=solve(crossing_85(selected),'85C_LOCAL_REFINEMENT')
            if abs(check['Tmax_C']-85)<1e-4:break
        limit=crossing_85(selected)
        old_selected=[r for r in old if r['architecture']==spec.architecture]
        indexed={r['bandwidth_TBps']:r for r in coarse}
        old_indexed={r['bandwidth_TBps']:r for r in old_selected}
        limits.append(dict(architecture=spec.architecture,slabs=energy['slabs'],
            Bthermal_TBps=limit,old_Bthermal_TBps=crossing_85(old_selected),
            delta_Bthermal_TBps=limit-crossing_85(old_selected),
            Tmax_2p4_C=indexed[2.4]['Tmax_C'],Tmax_4p8_C=indexed[4.8]['Tmax_C'],
            delta_Tmax_2p4_C=indexed[2.4]['Tmax_C']-old_indexed[2.4]['Tmax_C'],
            delta_Tmax_4p8_C=indexed[4.8]['Tmax_C']-old_indexed[4.8]['Tmax_C'],
            hotspot=check['hotspot'],crossing_solved_Tmax_C=check['Tmax_C'],
            status='BRACKETED_INTERPOLATION_WITH_LOCAL_SOLVE'))
        audits[spec.architecture]['average_solve_time_s']=float(np.mean([r['solve_time_s'] for r in selected]))
        audits[spec.architecture]['solves']=len(selected)
        summary[spec.architecture]=summarize_curve(coarse)
        csv_out('thermal_limits.csv',limits);dump('setup_audit.json',audits)
        del gpu_operator, reusable
        gc.collect()
    dump('manifest.json',dict(source_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        status='NO_NMP_ARCHITECTURE_LEVEL_READ_STRESS_RECLOSURE',
        bandwidth_semantics='PRESCRIBED_STRESS_NOT_ACHIEVABLE_THROUGHPUT',
        thermal_mapping='EXISTING_GPU_FEOL_AND_UNIFORM_M3D_BEOL',
        HBM_status='REUSED_UNCHANGED_MODEL',HBM_source=str(historical/'curves.csv'),
        HBM_source_sha256=sha256((historical/'curves.csv').read_bytes()).hexdigest(),
        HBM_Bthermal_TBps={s.architecture:crossing_85([r for r in old if r['architecture']==s.architecture]) for s in SENSITIVITY_ARCHITECTURES[:2]},
        NMP_thermal='OUT_OF_SCOPE',performance_model='UNCHANGED',solver=SOLVER_OPTIONS,
        energy_config_sha256=sha256((root/'configs/architecture/m3d_feol_energy_v1.yaml').read_bytes()).hexdigest()))
    plot_thermal_sensitivity(sorted(rows,key=lambda r:(r['architecture'],r['bandwidth_TBps'])),summary,output,reused_hbm=True)
    return limits
