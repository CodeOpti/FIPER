"""Select the fastest valid candidate using synthesized tests only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

import pandas as pd


FORBIDDEN_EVALUATION_TERMS = ("public", "private", "hidden", "hide_io")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--pass-threshold", type=float, default=0.999999)
    return parser.parse_args()


def _atomic_write(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def _candidate_labels(frame: pd.DataFrame) -> list[str]:
    labels = [
        match.group(1)
        for column in frame.columns
        if (match := re.fullmatch(r"(candidate_\d+)_generated_pass_rate", column))
    ]
    return ["base", *sorted(labels, key=lambda value: int(value.split("_")[-1]))]


def _select_row(row: pd.Series, labels: list[str], threshold: float) -> tuple[str, str, float, float]:
    base_code = str(row.get("current_code", row["slow_code"]))
    code_by_label = {"base": base_code}
    code_by_label.update({label: str(row[label]) for label in labels if label != "base"})

    valid: list[tuple[float, str, float]] = []
    for label in labels:
        pass_rate = float(row[f"{label}_generated_pass_rate"])
        runtime_ms = float(row[f"{label}_generated_runtime_ms"])
        if pass_rate >= threshold:
            valid.append((runtime_ms, label, pass_rate))

    if not valid:
        label = "base"
        return (
            base_code,
            label,
            float(row["base_generated_pass_rate"]),
            float(row["base_generated_runtime_ms"]),
        )
    runtime_ms, label, pass_rate = min(valid, key=lambda item: (item[0], item[1]))
    return code_by_label[label], label, pass_rate, runtime_ms


def main() -> None:
    args = parse_args()
    if args.round < 1:
        raise ValueError("--round must be positive.")
    if not 0.0 <= args.pass_threshold <= 1.0:
        raise ValueError("--pass-threshold must be between zero and one.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {args.input}")
    frame = pd.read_csv(args.input)
    forbidden = [
        column
        for column in frame.columns
        if any(term in column.casefold() for term in FORBIDDEN_EVALUATION_TERMS)
    ]
    if forbidden:
        frame = frame.drop(columns=forbidden)
    if "slow_code" not in frame.columns:
        raise ValueError("Input CSV must contain slow_code.")

    labels = _candidate_labels(frame)
    if len(labels) == 1:
        raise ValueError("No evaluated candidate columns were found.")
    required_metrics = {
        f"{label}_generated_{metric}"
        for label in labels
        for metric in ("pass_rate", "runtime_ms")
    }
    missing = sorted(required_metrics.difference(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing evaluation columns: {', '.join(missing)}")

    selected = [
        _select_row(row, labels, args.pass_threshold) for _, row in frame.iterrows()
    ]
    round_prefix = f"round_{args.round}"
    frame["current_code"] = [value[0] for value in selected]
    frame[f"{round_prefix}_selected_source"] = [value[1] for value in selected]
    frame[f"{round_prefix}_generated_pass_rate"] = [value[2] for value in selected]
    frame[f"{round_prefix}_generated_runtime_ms"] = [value[3] for value in selected]

    transient_columns = [
        column
        for column in frame.columns
        if re.fullmatch(r"candidate_\d+", column)
        or re.fullmatch(r"(?:base|candidate_\d+)_generated_(?:pass_rate|runtime_ms|evaluation)", column)
    ]
    frame = frame.drop(columns=transient_columns)
    _atomic_write(frame, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
