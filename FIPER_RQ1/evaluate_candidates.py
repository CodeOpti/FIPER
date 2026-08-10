"""Evaluate optimization candidates using synthesized tests only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import statistics
from typing import Any

import pandas as pd

from pie_sandbox import FAILURE_TIME_MS, evaluate_code


FORBIDDEN_EVALUATION_TERMS = ("public", "private", "hidden", "hide_io")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("python", "cpp"), default="python")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def _drop_evaluation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in frame.columns
        if any(term in column.casefold() for term in FORBIDDEN_EVALUATION_TERMS)
    ]
    return frame.drop(columns=columns) if columns else frame


def _atomic_write(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def _language(row: pd.Series, default: str) -> str:
    value = str(row.get("language", default)).strip().casefold()
    aliases = {"py": "python", "python": "python", "c++": "cpp", "cpp": "cpp"}
    if value not in aliases:
        raise ValueError(f"Unsupported language: {value!r}")
    return aliases[value]


def _test_cases(row: pd.Series) -> dict[str, list[str]]:
    try:
        value = json.loads(str(row["generated_tests"]))
    except json.JSONDecodeError as error:
        raise ValueError("generated_tests must contain strict JSON.") from error
    if not isinstance(value, dict) or not isinstance(value.get("inputs"), list):
        raise ValueError("generated_tests must contain input and output lists.")
    if not isinstance(value.get("outputs"), list):
        raise ValueError("generated_tests must be oracle-verified.")
    return value


def _metrics(result: dict[str, Any]) -> tuple[float, float]:
    passes = [int(value) for value in result["test_passes"]]
    pass_rate = statistics.mean(passes) if passes else 0.0
    if pass_rate < 1.0:
        return pass_rate, FAILURE_TIME_MS
    per_test_times = [statistics.median(values) for values in result["runtime_ms"]]
    return pass_rate, statistics.mean(per_test_times)


def _evaluate_row(
    position: int,
    row: pd.Series,
    candidate_columns: list[str],
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    tests = _test_cases(row)
    language = _language(row, args.language)
    base_code = str(row.get("current_code", row["slow_code"]))
    code_columns = {"base": base_code}
    code_columns.update({column: str(row[column]) for column in candidate_columns})

    values: dict[str, Any] = {}
    for label, code in code_columns.items():
        result = evaluate_code(
            code,
            tests,
            language,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            timeout_seconds=args.timeout,
        )
        pass_rate, runtime_ms = _metrics(result)
        values[f"{label}_generated_pass_rate"] = pass_rate
        values[f"{label}_generated_runtime_ms"] = runtime_ms
        values[f"{label}_generated_evaluation"] = json.dumps(result, ensure_ascii=True)
    return position, values


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.measured_runs < 1 or args.warmup_runs < 0:
        raise ValueError("Run and thread counts are invalid.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {args.input}")
    frame = _drop_evaluation_columns(pd.read_csv(args.input))
    required = {"slow_code", "generated_tests"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    candidate_columns = sorted(
        (column for column in frame.columns if re.fullmatch(r"candidate_\d+", column)),
        key=lambda value: int(value.split("_")[-1]),
    )
    if not candidate_columns:
        raise ValueError("No candidate_N columns were found.")
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")
    for label in ["base", *candidate_columns]:
        frame[f"{label}_generated_pass_rate"] = 0.0
        frame[f"{label}_generated_runtime_ms"] = FAILURE_TIME_MS
        frame[f"{label}_generated_evaluation"] = ""

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(_evaluate_row, position, row, candidate_columns, args): position
            for position, (_, row) in enumerate(frame.iterrows())
        }
        for future in as_completed(futures):
            position, values = future.result()
            row_index = frame.index[position]
            for column, value in values.items():
                frame.at[row_index, column] = value
            _atomic_write(frame, args.output)
            print(f"Evaluated {position + 1}/{len(frame)}")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
