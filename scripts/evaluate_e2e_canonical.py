"""One-command canonical workload-to-steady-state-thermal evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from om3dthermal.evaluator.canonical_e2e import (
    integrate_canonical_e2e, write_canonical_e2e_artifacts)

try:
    from evaluate_die_local_placement import ROOT, _architecture
    from evaluate_nmp_locality_placement import run as run_nmp
    from evaluate_nmp_thermal_baseline import run as run_thermal
except ModuleNotFoundError:
    from scripts.evaluate_die_local_placement import ROOT, _architecture
    from scripts.evaluate_nmp_locality_placement import run as run_nmp
    from scripts.evaluate_nmp_thermal_baseline import run as run_thermal


def run(output_dir: Path):
    """Recompute both frozen paths and join them with exact closure gates."""
    components = output_dir / "_components"
    nmp_payload = run_nmp(components / "nmp")
    thermal_payload = run_thermal(components / "thermal")
    layout, _ = _architecture()
    result = integrate_canonical_e2e(
        nmp_payload=nmp_payload, thermal_payload=thermal_payload,
        total_capacity_bytes=layout.total_capacity_bytes)
    write_canonical_e2e_artifacts(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/e2e_canonical")
    result = run(parser.parse_args().output_dir)
    for row in result.cases:
        print(
            f"N={row.requests} non={row.non_nmp_step_ms:.6f} ms "
            f"balanced={row.balanced_step_ms:.6f} ms "
            f"gain={row.balanced_gain:.6f}x "
            f"power={row.aggregate_m3d_nmp_power_W:.6f} W "
            f"M3D_Tmax={row.m3d_Tmax_degC:.6f} C")
    print(f"E2E_CANONICAL_GATE={result.gates['E2E_CANONICAL_GATE']}")


if __name__ == "__main__":
    main()
