"""Evaluate and select PEACEXEC repository-level optimization candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd

from code_evaluation_pipeline import evaluate_candidate


CASE_COLUMNS = {
    "repo_path",
    "sha",
    "target_file",
    "target_class",
    "target_func",
    "venv_path",
    "test_cmd",
    "slow_code",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=25)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _metric_columns(label: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{label}_pass_rate": result["pass_rate"],
        f"{label}_runtime_ms": result["runtime_ms"],
        f"{label}_evaluation": json.dumps(result, ensure_ascii=True),
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {args.input}")
    frame = pd.read_csv(args.input)
    missing = sorted(CASE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {', '.join(missing)}")
    candidate_columns = sorted(
        (column for column in frame.columns if re.fullmatch(r"candidate_\d+", column)),
        key=lambda column: int(column.split("_")[-1]),
    )
    if not candidate_columns:
        raise ValueError("Input CSV contains no candidate_N columns.")

    for position, (_, row) in enumerate(frame.iterrows()):
        case = row.to_dict()
        base_code = _text(row.get("current_code")) or None
        base_result = evaluate_candidate(
            case,
            base_code,
            args.repository_root,
            args.venv_root,
            args.warmup_runs,
            args.measured_runs,
            args.timeout_seconds,
            args.temporary_root,
        )
        values = _metric_columns("base", base_result)
        choices: list[tuple[float, str, str]] = []
        if base_result["status"] == "ok":
            choices.append(
                (float(base_result["runtime_ms"]), "base", base_code or _text(row["slow_code"]))
            )

        for label in candidate_columns:
            candidate_code = _text(row[label]).strip()
            if candidate_code:
                candidate_result = evaluate_candidate(
                    case,
                    candidate_code,
                    args.repository_root,
                    args.venv_root,
                    args.warmup_runs,
                    args.measured_runs,
                    args.timeout_seconds,
                    args.temporary_root,
                )
            else:
                candidate_result = {
                    "status": "error",
                    "pass_rate": 0.0,
                    "runtime_ms": None,
                    "cpu_instructions": None,
                    "memory_mb": None,
                    "error": "Empty candidate.",
                }
            values.update(_metric_columns(label, candidate_result))
            if candidate_result["status"] == "ok":
                choices.append(
                    (float(candidate_result["runtime_ms"]), label, candidate_code)
                )

        if choices:
            runtime_ms, selected_source, selected_code = min(
                choices, key=lambda item: (item[0], item[1])
            )
            values.update(
                current_code=selected_code,
                selected_source=selected_source,
                selected_runtime_ms=runtime_ms,
            )
        else:
            values.update(
                current_code=base_code or _text(row["slow_code"]),
                selected_source="none",
                selected_runtime_ms=None,
            )

        row_index = frame.index[position]
        for column, value in values.items():
            frame.at[row_index, column] = value
        _atomic_write(frame, args.output)
        print(f"Evaluated {position + 1}/{len(frame)}")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
