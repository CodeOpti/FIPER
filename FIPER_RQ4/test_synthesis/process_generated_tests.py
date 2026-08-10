"""Oracle verification and representative selection for synthesized tests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from pie_sandbox import capture_oracle_outputs


FORBIDDEN_EVALUATION_TERMS = ("public", "private", "hidden", "hide_io")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("python", "cpp"), default="python")
    parser.add_argument("--max-tests", type=int, default=10)
    parser.add_argument("--max-input-chars", type=int, default=4096)
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


def _candidate_inputs(value: Any, max_input_chars: int) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError("generated_test_inputs must contain strict JSON.") from error
    inputs = parsed.get("inputs") if isinstance(parsed, dict) else None
    if not isinstance(inputs, list):
        raise ValueError("generated_test_inputs must contain an 'inputs' list.")

    unique_inputs: list[str] = []
    seen: set[str] = set()
    for item in inputs:
        input_text = str(item)
        if not input_text.strip() or len(input_text) > max_input_chars or input_text in seen:
            continue
        seen.add(input_text)
        unique_inputs.append(input_text)
    return unique_inputs


def _boundary_scores(inputs: list[str]) -> list[float]:
    if len(inputs) < 2:
        return [1.0] * len(inputs)
    feature_sets = [
        {value[index : index + 2] for index in range(max(len(value) - 1, 1))}
        for value in inputs
    ]
    mean_length = sum(len(value) for value in inputs) / len(inputs)
    scores: list[float] = []
    for index, features in enumerate(feature_sets):
        distances = []
        for other_index, other_features in enumerate(feature_sets):
            if index == other_index:
                continue
            union = features | other_features
            similarity = len(features & other_features) / len(union) if union else 1.0
            distances.append(1.0 - similarity)
        structural_distance = sum(distances) / len(distances)
        length_distance = abs(len(inputs[index]) - mean_length) / max(mean_length, 1.0)
        scores.append(structural_distance + length_distance)
    return scores


def _representative_indices(
    inputs: list[str], runtime_ms: list[float], limit: int
) -> list[int]:
    if len(inputs) <= limit:
        return list(range(len(inputs)))

    boundary_scores = _boundary_scores(inputs)
    load_scores = [float(value) for value in runtime_ms]
    if not all(math.isfinite(value) for value in load_scores) or len(set(load_scores)) == 1:
        load_scores = [float(len(value)) for value in inputs]

    boundary_order = sorted(range(len(inputs)), key=lambda index: -boundary_scores[index])
    load_order = sorted(range(len(inputs)), key=lambda index: -load_scores[index])
    boundary_target = (limit + 1) // 2
    selected = boundary_order[:boundary_target]
    for index in load_order:
        if index not in selected:
            selected.append(index)
        if len(selected) == limit:
            break
    return selected


def _process_row(
    position: int, row: pd.Series, args: argparse.Namespace
) -> tuple[int, str, str]:
    slow_code = str(row["slow_code"])
    inputs = _candidate_inputs(row["generated_test_inputs"], args.max_input_chars)
    if not inputs:
        raise ValueError(f"Row {position} contains no usable synthesized inputs.")

    oracle_result = capture_oracle_outputs(
        slow_code,
        inputs,
        _language(row, args.language),
        timeout_seconds=args.timeout,
    )
    verified_inputs = [str(value) for value in oracle_result["inputs"]]
    verified_outputs = [str(value) for value in oracle_result["outputs"]]
    runtimes = [float(value) for value in oracle_result["runtime_ms"]]
    if not verified_inputs:
        raise RuntimeError(f"The slow-program oracle rejected every synthesized input in row {position}.")

    indices = _representative_indices(verified_inputs, runtimes, args.max_tests)
    tests = {
        "inputs": [verified_inputs[index] for index in indices],
        "outputs": [verified_outputs[index] for index in indices],
    }
    selected_runtimes = [runtimes[index] for index in indices]
    return (
        position,
        json.dumps(tests, ensure_ascii=True),
        json.dumps(selected_runtimes),
    )


def main() -> None:
    args = parse_args()
    if args.max_tests < 1 or args.max_input_chars < 1 or args.threads < 1:
        raise ValueError("Limits and thread count must be positive.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {args.input}")
    frame = _drop_evaluation_columns(pd.read_csv(args.input))
    required = {"slow_code", "generated_test_inputs"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")
    frame["generated_tests"] = ""
    frame["generated_test_oracle_runtime_ms"] = ""

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(_process_row, position, row, args): position
            for position, (_, row) in enumerate(frame.iterrows())
        }
        for future in as_completed(futures):
            position, tests, runtimes = future.result()
            row_index = frame.index[position]
            frame.at[row_index, "generated_tests"] = tests
            frame.at[row_index, "generated_test_oracle_runtime_ms"] = runtimes
            _atomic_write(frame, args.output)
            print(f"Oracle-verified {position + 1}/{len(frame)}")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
