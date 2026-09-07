"""M3D capacity under the revised 32x24 mm GPU die.

Baseline: slab_count=98, plane 22x5.5 mm, cube x=30 mm (300 um pitch).
Revised:  y matched to 24 mm; x -> 32 mm; max slabs = floor(32000/300) = 106.
"""
from pathlib import Path

from om3dthermal.power.config import find_project_root, load_case_config
from om3dthermal.power.geometry import resolve_case_geometry
from om3dthermal.power.system import resolve_system_power
from om3dthermal.architecture_comparison import _resolve_case_gpu_operating_point

CASE = "configs/cases/orthogonal_m3d_igzo.yaml"
root = find_project_root(CASE)
base = load_case_config(CASE)


def capacity(case):
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(
        case, project_root=root, geometry=geometry,
        gpu_operating_point=_resolve_case_gpu_operating_point(case, root))
    return int(system.memory_result.diagnostics["total_stored_bits"])


def with_geometry(case, *, slabs, plane_y, cube_x):
    orth = case.geometry.orthogonal.model_copy(update={
        "slab_count": slabs,
        "slab_plane_y_mm": plane_y,
        "cube_length_x_mm": cube_x,
    })
    geo = case.geometry.model_copy(update={"orthogonal": orth})
    return case.model_copy(update={"geometry": geo})


base_bits = capacity(base)
print(f"baseline  : slabs= 98, plane 22.0x5.5, cube_x=30.0 -> "
      f"{base_bits/8/2**30:7.1f} GiB = {base_bits/8/1e9:6.1f} GB")

for slabs in (104, 105, 106):
    bits = capacity(with_geometry(
        base, slabs=slabs, plane_y=24.0, cube_x=32.0))
    slack_um = 32000 - slabs * 300
    print(f"revised   : slabs={slabs}, plane 24.0x5.5, cube_x=32.0 -> "
          f"{bits/8/2**30:7.1f} GiB = {bits/8/1e9:6.1f} GB "
          f"(x slack {slack_um} um)")
