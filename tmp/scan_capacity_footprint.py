"""Capacity scan: HBM stack footprint vs system capacity (DreamRAM integer packing).

Baseline: conventional_hbm_2x1, capacity_instance_region 10.8x10.8 mm.
Question: which footprint lands near 144 GB total?
"""
from pathlib import Path

from om3dthermal.power.config import (
    CaseMemoryRegionInput, find_project_root, load_case_config)
from om3dthermal.power.geometry import resolve_case_geometry
from om3dthermal.power.system import resolve_system_power
from om3dthermal.architecture_comparison import _resolve_case_gpu_operating_point

CASE = "configs/cases/conventional_hbm_2x1.yaml"
root = find_project_root(CASE)
base = load_case_config(CASE)

print(f"{'footprint':>10} {'GiB/stack':>10} {'GB/stack':>9} "
      f"{'total GiB':>10} {'total GB':>9} {'banks/die':>9}")
for side in [10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4]:
    new_region = CaseMemoryRegionInput(width_mm=side, height_mm=side)
    new_geo = base.geometry.model_copy(
        update={"capacity_instance_region": new_region})
    case = base.model_copy(update={"geometry": new_geo})
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(
        case, project_root=root, geometry=geometry,
        gpu_operating_point=_resolve_case_gpu_operating_point(case, root))
    diag = system.memory_result.diagnostics
    bits_stack = int(diag["bits_per_stack"])
    total_bits = int(diag["total_stored_bits"])
    banks = int(diag.get("packed_banks_per_memory_die", -1))
    gib_stack = bits_stack / 8 / 2**30
    print(f"{side:>9.1f} {gib_stack:>10.2f} {bits_stack/8/1e9:>9.2f} "
          f"{total_bits/8/2**30:>10.1f} {total_bits/8/1e9:>9.1f} {banks:>9}")
