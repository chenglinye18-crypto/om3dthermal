"""Per-die NMP power carriers and steady-state thermal observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from ..architecture_comparison import compile_case_thermal
from ..config import PowerSelector, PowerSourceConfig, ThermalPowerSourcesConfig
from ..power.config import CanonicalCaseConfig
from ..power.nmp_die_power import NMPDiePowerMap
from ..power.system import ResolvedSystemPower


@dataclass(frozen=True)
class NMPDieThermalRegion:
    die_id: int
    region_id: str
    geometry_die_index: int
    center_x_m: float
    memory_role: str = "m3d_bitcell_beol_stack"
    nmp_role: str = "feol"


@dataclass(frozen=True)
class NMPDieThermalObservation:
    die_id: int
    physical_region: str
    center_x_m: float
    total_power_W: float
    memory_power_W: float
    nmp_power_W: float
    residual_external_power_W: float
    die_temperature_degC: float
    die_mean_temperature_degC: float


@dataclass(frozen=True)
class NMPDieThermalBaseline:
    requests: int
    dies: tuple[NMPDieThermalObservation, ...]
    global_Tmax_degC: float
    global_Tmax_region: str
    global_Tmax_material: str
    global_Tmax_xyz_m: tuple[float, float, float]
    hottest_m3d_die_id: int
    max_power_die_id: int
    power_temperature_correlation: float | None
    memory_power_temperature_correlation: float | None
    nmp_power_temperature_correlation: float | None
    correlation_status: str
    die_temperature_spread_degC: float
    thermal_power_mapping_closure: str
    residual_external_mapping_status: str
    solver_backend: str
    converged: bool
    iterations: int
    final_relative_residual: float

    def as_dict(self):
        return asdict(self)


def physical_nmp_die_regions(case: CanonicalCaseConfig) -> tuple[NMPDieThermalRegion, ...]:
    """Return the explicit zero-based architecture ID to geometry identity."""
    orth = case.geometry.orthogonal
    if orth is None:
        raise ValueError("NMP die mapping requires orthogonal geometry")
    count = orth.slab_count
    pitch_m = orth.slab_pitch_x_um * 1e-6
    array_x0_m = 0.5 * (orth.cube_length_x_mm * 1e-3 - count * pitch_m)
    return tuple(
        NMPDieThermalRegion(
            die_id=i,
            region_id=f"orthogonal_hbm:die_{i + 1:03d}",
            geometry_die_index=i + 1,
            center_x_m=array_x0_m + (i + 0.5) * pitch_m,
        )
        for i in range(count)
    )


def compile_nmp_die_thermal_config(
    case: CanonicalCaseConfig,
    system: ResolvedSystemPower,
    power_map: NMPDiePowerMap,
):
    """Compile frozen geometry with one memory and one FEOL carrier per die.

    Residual external power is a separate additive carrier on the owning
    die's FEOL.  This is a deterministic coarse interface-side proxy because
    the current geometry has no separately resolved per-die interface box.
    """
    config = compile_case_thermal(case, system)
    regions = physical_nmp_die_regions(case)
    rows = sorted(power_map.die_powers, key=lambda row: row.die_id)
    if len(rows) != len(regions) or tuple(row.die_id for row in rows) != tuple(range(len(regions))):
        raise ValueError("per-die power map must contain every die ID exactly once")

    sources = [PowerSourceConfig(
        name="gpu",
        total_power=system.gpu_power_W,
        selector=PowerSelector(component="gpu", material="FEOL"),
        metadata={"component_class": "gpu", "mapping_provenance": "CANONICAL_FROZEN_GPU_SOURCE"},
    )]
    for region, row in zip(regions, rows, strict=True):
        memory_W = float(row.thermal_memory_carrier_W or 0.0)
        nmp_W = float(row.thermal_nmp_carrier_W or 0.0)
        external_W = float(row.residual_external_W or 0.0)
        common = {"die_id": region.die_id, "region_id": region.region_id,
                  "geometry_die_index": region.geometry_die_index}
        sources.extend((
            PowerSourceConfig(
                name=f"nmp_die_{region.die_id:03d}_memory",
                total_power=memory_W,
                selector=PowerSelector(component=region.region_id,
                                       tags={"role": region.memory_role}),
                metadata={**common, "component_class": "m3d_memory",
                          "mapping_provenance": "WORKLOAD_DRIVEN_PER_DIE_MEMORY_ACTIVE_REGION"},
            ),
            PowerSourceConfig(
                name=f"nmp_die_{region.die_id:03d}_logic",
                total_power=nmp_W,
                selector=PowerSelector(component=region.region_id,
                                       tags={"role": region.nmp_role}),
                metadata={**common, "component_class": "nmp_logic",
                          "mapping_provenance": "MAC_DYNAMIC_ONLY_PER_DIE_FEOL"},
            ),
            PowerSourceConfig(
                name=f"nmp_die_{region.die_id:03d}_external",
                total_power=external_W,
                selector=PowerSelector(component=region.region_id,
                                       tags={"role": region.nmp_role}),
                metadata={**common, "component_class": "residual_external",
                          "mapping_provenance": "RESIDUAL_EXTERNAL_THERMAL_MAPPING_APPROXIMATION"},
            ),
        ))
    mapped_m3d = sum(source.total_power for source in sources[1:])
    if not math.isclose(mapped_m3d, power_map.aggregate_total_W,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("configured per-die thermal sources do not close")
    return config.model_copy(update={
        "thermal_power_sources": ThermalPowerSourcesConfig(sources=sources)
    }), regions


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if np.ptp(xa) == 0.0 or np.ptp(ya) == 0.0:
        return None
    value = float(np.corrcoef(xa, ya)[0, 1])
    return value if math.isfinite(value) else None


def analyze_nmp_die_thermal_pipeline(*, requests: int, power_map: NMPDiePowerMap,
                                    regions: tuple[NMPDieThermalRegion, ...], pipeline,
                                    solver_backend: str) -> NMPDieThermalBaseline:
    temperatures_C = np.asarray(pipeline.result.temperature_K, dtype=np.float64) - 273.15
    rows = sorted(power_map.die_powers, key=lambda row: row.die_id)
    observations = []
    for region, row in zip(regions, rows, strict=True):
        mask = np.array([
            cell.component == region.region_id
            and cell.tags.get("role") in {region.memory_role, region.nmp_role}
            for cell in pipeline.cells
        ], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"thermal region {region.region_id} selected no active cells")
        observations.append(NMPDieThermalObservation(
            die_id=region.die_id, physical_region=region.region_id,
            center_x_m=region.center_x_m, total_power_W=float(row.total_W or 0.0),
            memory_power_W=float(row.thermal_memory_carrier_W or 0.0),
            nmp_power_W=float(row.thermal_nmp_carrier_W or 0.0),
            residual_external_power_W=float(row.residual_external_W or 0.0),
            die_temperature_degC=float(np.max(temperatures_C[mask])),
            die_mean_temperature_degC=float(np.mean(temperatures_C[mask])),
        ))
    hottest = max(observations, key=lambda row: row.die_temperature_degC)
    max_power = max(observations, key=lambda row: row.total_power_W)
    hottest_index = int(np.argmax(temperatures_C))
    hottest_cell = pipeline.cells[hottest_index]
    power = [row.total_power_W for row in observations]
    temperature = [row.die_temperature_degC for row in observations]
    total_corr = _pearson(power, temperature)
    correlations = (
        total_corr,
        _pearson([row.memory_power_W for row in observations], temperature),
        _pearson([row.nmp_power_W for row in observations], temperature),
    )
    expected_package = 300.0 + power_map.aggregate_total_W
    closure = math.isclose(pipeline.power.total_power_W, expected_package,
                           rel_tol=1e-12, abs_tol=1e-9)
    if not closure:
        raise ValueError("cell-level package thermal source closure failed")
    return NMPDieThermalBaseline(
        requests=requests, dies=tuple(observations),
        global_Tmax_degC=float(temperatures_C[hottest_index]),
        global_Tmax_region=str(hottest_cell.component),
        global_Tmax_material=str(hottest_cell.material),
        global_Tmax_xyz_m=(float(hottest_cell.center_x), float(hottest_cell.center_y),
                           float(hottest_cell.center_z)),
        hottest_m3d_die_id=hottest.die_id, max_power_die_id=max_power.die_id,
        power_temperature_correlation=correlations[0],
        memory_power_temperature_correlation=correlations[1],
        nmp_power_temperature_correlation=correlations[2],
        correlation_status=("DEFINED" if correlations[0] is not None
                            else "UNDEFINED_ZERO_VARIANCE"),
        die_temperature_spread_degC=(max(temperature) - min(temperature)),
        thermal_power_mapping_closure="PASS",
        residual_external_mapping_status="RESIDUAL_EXTERNAL_THERMAL_MAPPING_APPROXIMATION",
        solver_backend=solver_backend, converged=bool(pipeline.result.converged),
        iterations=int(pipeline.result.iterations),
        final_relative_residual=float(pipeline.result.final_relative_residual),
    )
