"""Uniform, checksummed result bundle for formal experiments."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .serialization import to_jsonable


RESULT_FILES = (
    "architecture.json",
    "workload.json",
    "capacity.json",
    "performance.json",
    "energy.json",
    "power.json",
    "thermal.json",
    "provenance.json",
    "summary.json",
    "summary.csv",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_result_bundle(
    output_dir: Path,
    *,
    resolved_config: Mapping[str, Any],
    architecture: Any,
    workload: Any,
    capacity: Any,
    performance: Any,
    energy: Any,
    power: Any,
    thermal: Any,
    provenance: Any,
    summary: Any,
) -> Path:
    """Write stage-separated results; reporting performs no recomputation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(
            to_jsonable(resolved_config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    payloads = {
        "architecture.json": architecture,
        "workload.json": workload,
        "capacity.json": capacity,
        "performance.json": performance,
        "energy.json": energy,
        "power.json": power,
        "thermal.json": thermal,
        "provenance.json": provenance,
        "summary.json": summary,
    }
    for name, payload in payloads.items():
        _write_json(output_dir / name, payload)

    from om3dthermal.reporting.tables import write_e2e_summary_csv
    summary_rows = to_jsonable(summary).get("rows", [])
    write_e2e_summary_csv(summary_rows, output_dir / "summary.csv")

    checksummed = (resolved_path, *(output_dir / name for name in RESULT_FILES))
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "files": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in checksummed
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir
