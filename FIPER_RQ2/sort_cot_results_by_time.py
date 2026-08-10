"""Sort generated-code candidates by correctness and execution time."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def sort_candidates_by_time(
    dataset_path: str,
    output_path: str,
    code_prefix: str = "Optimized_Code",
    candidate_count: int = 5,
) -> None:
    """Add consistently ordered candidate columns to a CSV file.

    Candidates with a complete pass rate are preferred; ties are resolved by
    the smallest measured time. The original columns remain unchanged.
    """
    import pandas as pd

    frame = pd.read_csv(dataset_path)
    code_columns = [f"{code_prefix}_{index}" for index in range(1, candidate_count + 1)]
    available = [column for column in code_columns if column in frame.columns]
    if not available:
        raise ValueError(f"No candidate columns found for prefix {code_prefix!r}")

    sorted_codes: list[list[str]] = [[] for _ in available]
    sorted_times: list[list[float]] = [[] for _ in available]
    for _, row in frame.iterrows():
        candidates = []
        for column in available:
            suffix = column.rsplit("_", 1)[-1]
            pass_rate = float(row.get(f"{column}_PassRate", 0) or 0)
            time_ms = float(row.get(f"{column}_TimeMs", math.inf) or math.inf)
            candidates.append((pass_rate, time_ms, str(row.get(column, "")), suffix))
        candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
        for index, (_, time_ms, code, _) in enumerate(candidates):
            sorted_codes[index].append(code)
            sorted_times[index].append(time_ms)

    for index, values in enumerate(sorted_codes, start=1):
        frame[f"Sorted_{code_prefix}_{index}"] = values
        frame[f"Sorted_{code_prefix}_{index}_TimeMs"] = sorted_times[index - 1]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Sorted {len(available)} candidates for {len(frame)} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--code-prefix", default="Optimized_Code")
    parser.add_argument("--candidate-count", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sort_candidates_by_time(
        args.dataset_path,
        args.output_path,
        args.code_prefix,
        args.candidate_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
