"""Tabular exports that never invoke scientific evaluators."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from om3dthermal.experiment.result_bundle import to_jsonable


def write_e2e_summary_csv(rows: Sequence[Any], path: Path) -> None:
    data = [to_jsonable(row) for row in rows]
    if not data:
        raise ValueError("E2E summary CSV requires at least one persisted row")
    fieldnames: list[str] = []
    for row in data:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
