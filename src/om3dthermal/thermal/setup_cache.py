"""Versioned, power-independent cache for steady-state thermal setup."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import SimulationConfig
from ..geometry.horizontal_columns import HorizontalColumnsBuilder
from ..geometry.orthogonal_hbm import OrthogonalHBMBuilder
from .boundary import BoundaryLinkTable
from .conductance import ConductanceTable
from .operator import MatrixFreeThermalOperator


CACHE_SCHEMA_VERSION = "thermal-setup-v1"


@dataclass
class ThermalSetupArtifacts:
    """All expensive fixed artifacts; workload power/RHS is excluded."""

    grid: Any
    cells: list
    edges: list
    boundary_faces: list
    conductance_table: ConductanceTable
    boundary_table: BoundaryLinkTable
    operator_template: MatrixFreeThermalOperator


@dataclass
class ThermalSetupCacheRecord:
    schema_version: str
    physical_signature: str
    artifacts: ThermalSetupArtifacts


def build_scene_and_signature(config: SimulationConfig) -> tuple[list, str]:
    """Build cheap geometry and hash every fixed physical input.

    Runtime-generated box UUIDs and thermal power sources are intentionally
    absent.  Geometry coordinates/material assignments/tags, mesh settings,
    the material registry, conductance rules, boundary conditions, and the
    cache/discretizer version all participate in the signature.
    """
    scene = (
        OrthogonalHBMBuilder(config).build()
        if config.orthogonal_hbm is not None
        else HorizontalColumnsBuilder(config).build())
    boxes = list(scene.boxes)
    geometry = [
        {
            "name": box.name,
            "material": box.material,
            "bounds": [box.x0, box.x1, box.y0, box.y1, box.z0, box.z1],
            "tags": box.tags,
            "source_path": box.source_path,
            "rotation": box.rotation,
        }
        for box in boxes
    ]
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "geometry": geometry,
        "mesh": config.discretization.model_dump(mode="json"),
        "materials": {
            key: value.model_dump(mode="json")
            for key, value in sorted(config.materials.items())
        },
        "thermal_conductance": config.thermal_conductance.model_dump(
            mode="json"),
        "thermal_boundary_conditions": (
            config.thermal_boundary_conditions.model_dump(mode="json")),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return boxes, hashlib.sha256(encoded).hexdigest()


def load_setup_cache(
    path: Path, expected_signature: str,
) -> tuple[ThermalSetupArtifacts | None, float, str]:
    """Load and strictly validate a cache, otherwise request a rebuild."""
    if not path.exists():
        return None, 0.0, "MISS"
    started = time.perf_counter()
    try:
        with path.open("rb") as stream:
            record = pickle.load(stream)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
        return None, time.perf_counter() - started, "INVALIDATED"
    elapsed = time.perf_counter() - started
    if (
        not isinstance(record, ThermalSetupCacheRecord)
        or record.schema_version != CACHE_SCHEMA_VERSION
        or record.physical_signature != expected_signature
    ):
        return None, elapsed, "INVALIDATED"
    template = record.artifacts.operator_template
    if (
        template.power_W.shape != (template.cell_count,)
        or bool(template.power_W.any())
    ):
        return None, elapsed, "INVALIDATED"
    return record.artifacts, elapsed, "HIT"


def save_setup_cache(
    path: Path, signature: str, artifacts: ThermalSetupArtifacts,
) -> float:
    """Atomically serialize a cache record and return wall-clock seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    started = time.perf_counter()
    record = ThermalSetupCacheRecord(
        schema_version=CACHE_SCHEMA_VERSION,
        physical_signature=signature,
        artifacts=artifacts,
    )
    try:
        with temporary.open("wb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return time.perf_counter() - started
