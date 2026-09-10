"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .geometry.horizontal_columns import HorizontalColumnsBuilder
from .geometry.orthogonal_hbm import OrthogonalHBMBuilder
from .visualization import write_visualizations


def build_scene(config):
    """Select the geometry template while keeping all downstream stages shared."""
    if config.orthogonal_hbm is not None:
        return OrthogonalHBMBuilder(config).build()
    return HorizontalColumnsBuilder(config).build()


def build(config_path: str | Path, output_dir: str | Path):
    config = load_config(config_path)
    scene = build_scene(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.write_csv(output_dir / "regions.csv")
    scene.write_summary(output_dir / "geometry_summary.json")
    write_visualizations(scene, config, output_dir)
    return scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="om3dthermal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build horizontal geometry")
    build_parser.add_argument("config", type=Path)
    build_parser.add_argument("--out", type=Path, required=True)
    bandwidth_thermal_parser = subparsers.add_parser(
        "bandwidth-thermal-sweep",
        help="run the frozen No-NMP HBM/M3D bandwidth thermal sweep")
    bandwidth_thermal_parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/no_nmp_bandwidth_thermal_sweep"))
    experiment_parser = subparsers.add_parser(
        "experiment",
        help="run a formal workload-aware experiment and write a result bundle")
    experiment_parser.add_argument("config", type=Path)
    experiment_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="override the configured formal result-bundle directory")
    nmp_parser = subparsers.add_parser(
        "nmp-attention", help="run the nominal B=1 dense NMP attention E2E audit")
    nmp_parser.add_argument("--output-dir", type=Path, default=Path("runs/nmp_attention_nominal"))
    prefill_parser = subparsers.add_parser(
        "prefill", help="evaluate the independent dense prefill GPU roofline")
    prefill_parser.add_argument(
        "--config", type=Path,
        default=Path("configs/workload/llama31_8b_prefill_b1_s131072.yaml"))
    prefill_parser.add_argument(
        "--platform", type=Path,
        default=Path("configs/platform/gpu_package_h200_reference.yaml"))
    sensitivity_parser = subparsers.add_parser(
        "thermal-sensitivity", help="run the four-curve No-NMP geometry comparison")
    sensitivity_parser.add_argument("--output-dir", type=Path, required=True)
    sensitivity_parser.add_argument("--nominal-cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "thermal-sensitivity":
        from .thermal_sensitivity import run_thermal_sensitivity
        print(json.dumps(run_thermal_sensitivity(
            args.output_dir, project_root=Path.cwd(),
            nominal_cache_dir=args.nominal_cache_dir), indent=2))
    elif args.command == "prefill":
        from .experiment import load_platform_spec, load_prefill_workload_spec
        from .platform import (
            resolve_gpu_bandwidth_service,
            resolve_gpu_prefill_compute_energy_calibration,
        )
        from .workload import evaluate_gpu_prefill_roofline, evaluate_llm_prefill

        project_root = Path.cwd()
        workload_spec = load_prefill_workload_spec(
            args.config, project_root=project_root)
        platform = load_platform_spec(args.platform, project_root=project_root)
        if (platform.gpu_decode_power is None
                or platform.gpu_compute_power is None
                or platform.gpu_prefill_compute is None):
            raise ValueError(
                "prefill requires GPU bandwidth, peak, power, and calibrated capability data")
        bandwidth = resolve_gpu_bandwidth_service(
            transfer_ceiling_bytes_per_s=(
                platform.gpu_decode_power.peak_memory_bandwidth_bytes_per_s),
            service_status=platform.gpu_bandwidth_service.service_status,
            provenance=platform.gpu_bandwidth_service.provenance)
        metrics = evaluate_llm_prefill(workload_spec.prefill)
        compute = platform.gpu_compute_power
        prefill_compute = platform.gpu_prefill_compute
        energy_calibration = resolve_gpu_prefill_compute_energy_calibration(
            compute, prefill_compute)
        roofline = evaluate_gpu_prefill_roofline(
            metrics,
            peak_compute_flops_per_s=(
                compute.peak_compute_BF16_dense_flops_per_s),
            large_gemm_effective_flops_per_s=(
                prefill_compute.large_gemm_effective_tflops * 1e12),
            causal_attention_effective_flops_per_s=(
                prefill_compute.causal_attention_effective_tflops * 1e12),
            sustained_memory_bandwidth_bytes_per_s=(
                bandwidth.sustained_bandwidth_bytes_per_s),
            static_power_W=compute.static_power_W,
            peak_reference_dynamic_J_per_FLOP_min=(
                energy_calibration.peak_reference_dynamic_J_per_FLOP_min),
            peak_reference_dynamic_J_per_FLOP_max=(
                energy_calibration.peak_reference_dynamic_J_per_FLOP_max),
            nominal_gemm_dynamic_J_per_FLOP_min=(
                energy_calibration.nominal_gemm_dynamic_J_per_FLOP_min),
            nominal_gemm_dynamic_J_per_FLOP_max=(
                energy_calibration.nominal_gemm_dynamic_J_per_FLOP_max),
            nominal_attention_dynamic_J_per_FLOP_min=(
                energy_calibration.nominal_attention_dynamic_J_per_FLOP_min),
            nominal_attention_dynamic_J_per_FLOP_max=(
                energy_calibration.nominal_attention_dynamic_J_per_FLOP_max),
            compute_bound_total_power_W_min=(
                energy_calibration.compute_bound_total_power_W_min),
            compute_bound_total_power_W_max=(
                energy_calibration.compute_bound_total_power_W_max))
        print(json.dumps({
            "workload_id": workload_spec.workload_id,
            "input": workload_spec.prefill.model_dump(mode="json"),
            "metrics": metrics.model_dump(mode="json"),
            "gpu_prefill_compute_capability": (
                prefill_compute.model_dump(mode="json")),
            "gpu_prefill_energy_calibration": (
                energy_calibration.model_dump(mode="json")),
            "gpu_roofline": roofline.model_dump(mode="json"),
        }, indent=2))
    elif args.command == "nmp-attention":
        from scripts.evaluate_nmp_locality_placement import run
        print(json.dumps(run(args.output_dir)["summary"], indent=2))
    elif args.command == "build":
        scene = build(args.config, args.out)
        print(f"Built {len(scene.boxes)} boxes in {args.out}")
    elif args.command == "experiment":
        from .experiment import run_experiment
        result = run_experiment(
            args.config, output_dir_override=args.output_dir)
        assert result.output_dir is not None
        print(
            f"[experiment] {result.experiment.experiment_id}: PASS\n"
            f"[experiment] rows:       {len(result.rows)}\n"
            f"[experiment] output_dir: {result.output_dir}\n"
            f"[experiment] manifest:   {result.output_dir / 'manifest.json'}")
    elif args.command == "bandwidth-thermal-sweep":
        from .bandwidth_thermal_sweep import (
            run_no_nmp_bandwidth_thermal_sweep,
        )
        result = run_no_nmp_bandwidth_thermal_sweep(
            args.output_dir, project_root=Path.cwd())
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
