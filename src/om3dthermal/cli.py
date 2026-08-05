"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .geometry.horizontal_columns import HorizontalColumnsBuilder
from .visualization import write_visualizations


def build(config_path: str | Path, output_dir: str | Path):
    config = load_config(config_path)
    scene = HorizontalColumnsBuilder(config).build()
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
    args = parser.parse_args(argv)
    if args.command == "build":
        scene = build(args.config, args.out)
        print(f"Built {len(scene.boxes)} boxes in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
