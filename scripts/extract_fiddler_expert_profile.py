"""Deterministically digitize Fiddler ICLR 2025 Figure 8.

Input is the original ``img/expert-popularity.png`` from the official arXiv
v3 LaTeX source archive, not a PDF/web screenshot.  The fixed geometry and
SHA256 deliberately fail if a different raster is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from PIL import Image


SOURCE_SHA256 = "003b4c6ed2f20485b796cdb5591d871af8a1133900b5977bd3cb18b8c2ce0e29"
SOURCE_SIZE = (2700, 900)
HEATMAP_BOUNDS = (162, 194, 2229, 707)  # left, top, right-exclusive, bottom-exclusive
COLORBAR_X = 2290
COLORBAR_Y_RANGE = (249, 651)  # top-inclusive, bottom-exclusive
COLORBAR_TICKS = ((247.0, 1.0), (351.0, 0.8), (454.0, 0.6), (558.0, 0.4))

PAPER_STATS = {
    "mean": 0.71,
    "std": 0.08,
    "p25": 0.67,
    "p75": 0.76,
    "min": 0.22,
    "count_lt_0_6": 15,
    "count_gt_0_8": 27,
    "max": 1.0,
}
VALIDATION_TOLERANCES = {
    "mean": 0.015,
    "std": 0.015,
    "p25": 0.015,
    "p75": 0.015,
    "min": 0.015,
    "count_lt_0_6": 2,
    "count_gt_0_8": 2,
    "max": 0.015,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_fit(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / sum((x - mean_x) ** 2 for x in xs)
    return slope, mean_y - slope * mean_x


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics(values: tuple[float, ...]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "count_lt_0_6": sum(value < 0.6 for value in values),
        "count_gt_0_8": sum(value > 0.8 for value in values),
    }


def extract(source: Path) -> tuple[tuple[float, ...], dict[str, object]]:
    source_hash = _sha256(source)
    if source_hash != SOURCE_SHA256:
        raise ValueError(
            f"source image SHA256 mismatch: {source_hash} != {SOURCE_SHA256}")
    image = Image.open(source).convert("RGB")
    if image.size != SOURCE_SIZE:
        raise ValueError(f"source image size mismatch: {image.size} != {SOURCE_SIZE}")

    slope, intercept = _linear_fit(COLORBAR_TICKS)
    bar = tuple(
        (y, image.getpixel((COLORBAR_X, y)))
        for y in range(*COLORBAR_Y_RANGE)
    )
    left, top, right, bottom = HEATMAP_BOUNDS
    values: list[float] = []
    maximum_color_distance_squared = 0
    for layer in range(32):
        x = round(left + (layer + 0.5) * (right - left) / 32)
        for expert in range(8):
            y = round(top + (expert + 0.5) * (bottom - top) / 8)
            rgb = image.getpixel((x, y))
            distances = tuple(
                sum((rgb[channel] - color[channel]) ** 2 for channel in range(3))
                for _, color in bar
            )
            minimum_distance = min(distances)
            matching_y = tuple(
                y_value for (y_value, _), distance in zip(bar, distances)
                if distance == minimum_distance
            )
            inferred_y = statistics.fmean(matching_y)
            values.append(slope * inferred_y + intercept)
            maximum_color_distance_squared = max(
                maximum_color_distance_squared, minimum_distance)

    result = tuple(values)
    stats = _statistics(result)
    deviations = {
        name: stats[name] - expected
        for name, expected in PAPER_STATS.items()
    }
    checks = {
        name: abs(float(deviations[name])) <= tolerance
        for name, tolerance in VALIDATION_TOLERANCES.items()
    }
    if len(result) != 256 or not all(checks.values()):
        raise ValueError(
            "FIDDLER_PROFILE_EXTRACTION_GATE=FAIL: "
            f"shape={len(result)}, checks={checks}, stats={stats}")
    metadata = {
        "schema_version": 1,
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "source": "Fiddler, ICLR 2025, Appendix C, Figure 8",
        "paper_url": "https://arxiv.org/pdf/2402.07033",
        "paper_version": "arXiv:2402.07033v3, 1 May 2025",
        "source_archive_url": "https://export.arxiv.org/e-print/2402.07033",
        "source_archive_member": "img/expert-popularity.png",
        "source_image_sha256": SOURCE_SHA256,
        "source_workload": "random samples from ShareGPT",
        "profile_kind": "published_relative_expert_selection_frequency",
        "classification": "DERIVED_FROM_PUBLISHED_ROUTING_PROFILE",
        "measurement_status": "NOT_MEASURED_BY_THIS_WORK",
        "synthetic_status": "NOT_SYNTHETIC",
        "extraction": "DIGITIZED_FROM_OFFICIAL_ARXIV_SOURCE_FIGURE",
        "raw_matrix_search": {
            "status": "NO_RAW_32X8_ARTIFACT_FOUND",
            "official_repository": "https://github.com/efeslab/fiddler",
            "official_repository_commit": "227715bfd6e8c731b29548eab01d9919c4fe9564",
            "repository_recursive_tree_checked": True,
            "arxiv_source_archive_checked": True,
        },
        "geometry": {
            "source_size_pixels": SOURCE_SIZE,
            "heatmap_bounds_pixels": HEATMAP_BOUNDS,
            "cell_sampling": "CENTER_PIXEL",
            "colorbar_x_pixel": COLORBAR_X,
            "colorbar_y_range_pixels": COLORBAR_Y_RANGE,
            "colorbar_tick_y_value_pairs": COLORBAR_TICKS,
            "colorbar_linear_slope_per_pixel": slope,
            "colorbar_linear_intercept": intercept,
            "maximum_rgb_distance_squared": maximum_color_distance_squared,
            "estimated_colorbar_quantization_step": abs(slope),
        },
        "paper_reported_statistics": PAPER_STATS,
        "extracted_statistics": stats,
        "extraction_deviation": deviations,
        "validation_tolerances": VALIDATION_TOLERANCES,
        "validation_checks": checks,
        "extraction_gate": "PASS",
        "orientation": "CSV layer 0..31 maps x-axis Layer 1..32; expert 0..7 maps y-axis Expert 1..8",
    }
    return result, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    args = parser.parse_args()
    values, metadata = extract(args.source_image)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("layer", "expert", "relative_popularity"))
        for layer in range(32):
            for expert in range(8):
                writer.writerow((layer, expert, f"{values[layer * 8 + expert]:.9f}"))
    with args.output_metadata.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(metadata["extracted_statistics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
