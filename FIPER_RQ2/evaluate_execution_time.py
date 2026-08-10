"""Validate RQ2 candidates with synthesized tests plus limited public tests."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from pie_sandbox import run_code_io_tests_function


FORBIDDEN_EVALUATION_TERMS = ("private", "hidden", "hide_io")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-round", type=int, required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--language", choices=("python", "cpp"), default=None)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=25)
    return parser.parse_args()


def parse_io(value: Any) -> dict[str, list[str]]:
    if isinstance(value, dict):
        loaded = value
    else:
        text = "" if value is None else str(value).strip()
        if not text or text.casefold() == "nan":
            return {"inputs": [], "outputs": []}
        loaded = None
        for loader in (json.loads, ast.literal_eval):
            try:
                candidate = loader(text)
                if isinstance(candidate, list):
                    return merge_io(*(parse_io(item) for item in candidate))
                if isinstance(candidate, dict):
                    loaded = candidate
                    break
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
        if loaded is None:
            return {"inputs": [], "outputs": []}
    inputs = loaded.get("inputs", [])
    outputs = loaded.get("outputs", [])
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return {"inputs": [], "outputs": []}
    size = min(len(inputs), len(outputs))
    return {
        "inputs": [str(item) for item in inputs[:size]],
        "outputs": [str(item) for item in outputs[:size]],
    }


def merge_io(*records: dict[str, list[str]]) -> dict[str, list[str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        for pair in zip(record["inputs"], record["outputs"]):
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return {
        "inputs": [pair[0] for pair in pairs],
        "outputs": [pair[1] for pair in pairs],
    }


def _public_io_columns(frame: Any) -> list[str]:
    return [
        column
        for column in frame.columns
        if "public" in str(column).casefold()
        and ("io_unit_tests" in str(column).casefold() or "test_cases" in str(column).casefold())
    ]


def validation_io(row: Any, public_columns: list[str]) -> dict[str, list[str]]:
    generated_column = "Selected_IO" if "Selected_IO" in row.index else "Generated_IO"
    if generated_column not in row.index:
        raise KeyError("The input CSV must contain Selected_IO or Generated_IO.")
    records = [parse_io(row[generated_column])]
    records.extend(parse_io(row[column]) for column in public_columns)
    merged = merge_io(*records)
    if not merged["inputs"]:
        raise ValueError("The combined synthesized/public validation suite is empty.")
    return merged


def evaluate_candidate(
    code: str,
    io_data: dict[str, list[str]],
    language: str,
    timeout_seconds: float,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[float, float, str]:
    if not code.strip():
        return 0.0, math.inf, "missing_code"
    suffix = ".cpp" if language == "cpp" else ".py"
    with tempfile.TemporaryDirectory(prefix="fiper_rq2_candidate_") as temporary_dir:
        source_path = Path(temporary_dir) / f"candidate{suffix}"
        source_path.write_text(code.strip() + "\n", encoding="utf-8")
        try:
            result = run_code_io_tests_function(
                str(source_path),
                io_data,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                language=language,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            return 0.0, math.inf, f"failed: {error}"
    passes = [int(value) for value in result["io_passresult"]]
    pass_rate = sum(passes) / len(passes) if passes else 0.0
    elapsed_ms = float(result["mean_time_us"]) / 1000.0
    errors = [str(value) for value in result["error_type"] if value]
    status = "passed" if pass_rate == 1.0 else "failed: " + " | ".join(errors[:2])
    return pass_rate, elapsed_ms, status


def _language(row: Any, explicit: str | None, dataset_path: str) -> str:
    if explicit:
        return explicit
    value = str(row.get("language", "")).casefold()
    if "++" in value or "cpp" in value or "cpp" in dataset_path.casefold():
        return "cpp"
    return "python"


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0 or args.warmup_runs < 0 or args.measured_runs < 1:
        raise ValueError("Timeout and run counts are invalid.")

    import pandas as pd

    frame = pd.read_csv(args.dataset_path)
    forbidden = [
        column
        for column in frame.columns
        if any(term in str(column).casefold() for term in FORBIDDEN_EVALUATION_TERMS)
    ]
    if forbidden:
        frame = frame.drop(columns=forbidden)
    public_columns = _public_io_columns(frame)
    candidate_columns = sorted(
        (
            column
            for column in frame.columns
            if str(column).startswith("Optimized_Code_")
            and str(column).rsplit("_", 1)[-1].isdigit()
        ),
        key=lambda column: int(str(column).rsplit("_", 1)[1]),
    )
    if not candidate_columns:
        raise KeyError("No Optimized_Code_N columns were found in the input CSV.")

    metric_labels = ["Base_Code", *candidate_columns]
    for label in metric_labels:
        frame[f"{label}_PassRate"] = 0.0
        frame[f"{label}_TimeMs"] = math.inf
        frame[f"{label}_Evaluation"] = ""

    for row_index, row in frame.iterrows():
        language = _language(row, args.language, args.dataset_path)
        tests = validation_io(row, public_columns)
        base_code = str(
            row.get(
                "Selected_Optimized_Code",
                row.get("Slow_Code", row.get("slow_code", "")),
            )
        )
        code_by_label = {"Base_Code": base_code}
        code_by_label.update({column: str(row[column]) for column in candidate_columns})
        for label, code in code_by_label.items():
            pass_rate, elapsed, status = evaluate_candidate(
                code,
                tests,
                language,
                args.timeout_seconds,
                args.warmup_runs,
                args.measured_runs,
            )
            frame.at[row_index, f"{label}_PassRate"] = pass_rate
            frame.at[row_index, f"{label}_TimeMs"] = elapsed
            frame.at[row_index, f"{label}_Evaluation"] = status

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"Evaluated RQ2 round {args.iteration_round} with synthesized and public tests: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
