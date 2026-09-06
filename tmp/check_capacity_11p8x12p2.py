"""Capacity check: rectangular capacity instance 11.8 x 12.2 mm, both orientations."""
from pathlib import Path

from om3dthermal.power.config import (
    CaseMemoryRegionInput, find_project_root, load_case_config)
from om3dthermal.power.geometry import resolve_case_geometry
from om3dthermal.power.system import resolve_system_power

CASE = "configs/cases/conventional_hbm_2x1.yaml"
root = find_project_root(CASE)
base = load_case_config(CASE)

print(f"{'x_mm':>6} {'y_mm':>6} {'GiB/stack':>10} {'GB/stack':>9} "
      f"{'total GiB':>10} {'total GB':>9}")
for x, y in [(11.8, 12.2), (12.2, 11.8)]:
    new_region = CaseMemoryRegionInput(width_mm=x, height_mm=y)
    new_geo = base.geometry.model_copy(
        update={"capacity_instance_region": new_region})
    case = base.model_copy(update={"geometry": new_geo})
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(
        case, project_root=root, geometry=geometry)
    diag = system.memory_result.diagnostics
    bits_stack = int(diag["bits_per_stack"])
    total_bits = int(diag["total_stored_bits"])
    print(f"{x:>6.1f} {y:>6.1f} {bits_stack/8/2**30:>10.2f} "
          f"{bits_stack/8/1e9:>9.2f} {total_bits/8/2**30:>10.1f} "
          f"{total_bits/8/1e9:>9.1f}")
