"""Validate that generated I/O records contain at least one test case."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from typing import Any


def parse_record(value: Any) -> dict[str, list[str]]:
    """Return a normalized I/O record or an empty record for invalid input."""
    if isinstance(value, dict):
        record = value
    else:
        text = "" if value is None else str(value).strip()
        if not text or text.lower() == "nan":
            return {"inputs": [], "outputs": []}
        try:
            record = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                record = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return {"inputs": [], "outputs": []}

    inputs = record.get("inputs", []) if isinstance(record, dict) else []
    outputs = record.get("outputs", []) if isinstance(record, dict) else []
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return {"inputs": [], "outputs": []}
    return {"inputs": inputs, "outputs": outputs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="CSV file to inspect")
    parser.add_argument(
        "--column",
        default="Selected_IO",
        help="Column containing JSON I/O records (default: Selected_IO)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required; install it with 'python -m pip install pandas'") from exc

    frame = pd.read_csv(args.dataset_path)
    if args.column not in frame.columns:
        raise SystemExit(f"Missing required column: {args.column}")

    empty_count = sum(not parse_record(value)["inputs"] for value in frame[args.column])
    print(f"Checked {len(frame)} records; empty records: {empty_count}")
    return 20 if empty_count else 10


if __name__ == "__main__":
    sys.exit(main())
