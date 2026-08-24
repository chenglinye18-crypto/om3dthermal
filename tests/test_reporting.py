import csv
from pathlib import Path

from om3dthermal.reporting import write_e2e_summary_csv


def test_reporting_forwards_rows_without_scientific_calculation(tmp_path: Path) -> None:
    rows = [
        {"architecture": "a", "rho": 0, "package_Tmax_degC": 42.5},
        {"architecture": "a", "rho": 1, "package_Tmax_degC": 42.6},
    ]
    path = tmp_path / "summary.csv"
    write_e2e_summary_csv(rows, path)
    with path.open(newline="", encoding="utf-8") as stream:
        restored = list(csv.DictReader(stream))
    assert restored[0]["architecture"] == "a"
    assert restored[0]["package_Tmax_degC"] == "42.5"
    assert restored[1]["rho"] == "1"
