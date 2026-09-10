"""No-NMP stack-depth/slab-thickness sensitivity using the frozen solver."""

from __future__ import annotations

import csv
import gc
from hashlib import sha256
import json
from pathlib import Path
import subprocess

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


def plot_thermal_sensitivity(rows, summary, output: Path):
    """Plot stored results, with a zoom for the closely spaced nominal/M3D curves."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    zoom = axis.inset_axes([0.57, 0.15, 0.4, 0.38])
    for spec, label, color, style in zip(SENSITIVITY_ARCHITECTURES, LABELS,
                                       ("#245a81", "#245a81", "#bd5a36", "#bd5a36"),
                                       ("-", "--", "-", "--")):
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
    axis.set(xlabel="Memory bandwidth (TB/s)", ylabel="Maximum temperature (°C)",
             title="No-NMP geometry sensitivity · zero memory static/refresh power")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "tmax_vs_bandwidth.png", dpi=200)
    figure.savefig(output / "tmax_vs_bandwidth.svg")
    plt.close(figure)
