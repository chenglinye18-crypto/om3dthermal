"""Run the canonical aggregate-batch FEOL-NMP Decode evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from om3dthermal.experiment import load_workload_spec
from om3dthermal.serving import evaluate_nmp_decode_batch


ROOT = Path(__file__).resolve().parents[1]


def run(output_dir: Path, batch_size: int = 1) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    base = load_workload_spec(
        ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml",
        project_root=ROOT,
    ).decode
    workload = base.model_copy(update={"batch_size": batch_size})
    result = evaluate_nmp_decode_batch(workload, project_root=ROOT)
    payload = {
        "summary": result.model_dump(mode="json"),
        "workload": workload.model_dump(mode="json"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nmp_locality_placement.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/nmp_attention_nominal")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.batch_size)["summary"], indent=2))


if __name__ == "__main__":
    main()
