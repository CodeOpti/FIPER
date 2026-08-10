"""Categorize RQ3 outcomes as NC, NO, NH, or FH.

Each input file must be strict JSON and use the name ``<dataset>__<method>.json``.
Supported dataset identifiers are ``python`` and ``cpp``. The output is a single
JSON object that can be consumed by ``plot_optimization_categories.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable


REQUIRED_KEYS = (
    "slow_pass_rates",
    "slow_runtimes_ms",
    "human_pass_rates",
    "human_runtimes_ms",
    "generated_pass_rates",
    "generated_runtimes_ms",
)
CATEGORIES = ("NC", "NO", "NH", "FH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimization-threshold", type=float, default=0.10)
    parser.add_argument("--pass-threshold", type=float, default=0.999)
    parser.add_argument("--failure-runtime", type=float, default=1_234_567_890.0)
    return parser.parse_args()


def _load_metrics(path: Path) -> dict[str, list[float]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    missing = sorted(set(REQUIRED_KEYS).difference(value))
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
    metrics = {key: [float(item) for item in value[key]] for key in REQUIRED_KEYS}
    lengths = {len(items) for items in metrics.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"{path} metric lists must have one shared non-zero length.")
    return metrics


def _valid_reference(
    pass_rate: float,
    runtime: float,
    pass_threshold: float,
    failure_runtime: float,
) -> bool:
    return (
        pass_rate >= pass_threshold
        and math.isfinite(runtime)
        and 0.0 < runtime < failure_runtime
    )


def categorize(
    metrics: dict[str, list[float]],
    optimization_threshold: float,
    pass_threshold: float,
    failure_runtime: float,
) -> dict[str, Any]:
    """Return counts and percentages using the paper's 10% OPT boundary."""

    counts: Counter[str] = Counter()
    evaluated = 0
    rows: Iterable[tuple[float, ...]] = zip(*(metrics[key] for key in REQUIRED_KEYS))
    for (
        slow_pass,
        slow_runtime,
        human_pass,
        human_runtime,
        generated_pass,
        generated_runtime,
    ) in rows:
        if not _valid_reference(
            slow_pass, slow_runtime, pass_threshold, failure_runtime
        ) or not _valid_reference(
            human_pass, human_runtime, pass_threshold, failure_runtime
        ):
            continue
        evaluated += 1
        if not _valid_reference(
            generated_pass, generated_runtime, pass_threshold, failure_runtime
        ):
            counts["NC"] += 1
        elif generated_runtime <= human_runtime * (1.0 - optimization_threshold):
            counts["FH"] += 1
        elif generated_runtime <= slow_runtime * (1.0 - optimization_threshold):
            counts["NH"] += 1
        else:
            counts["NO"] += 1

    if evaluated == 0:
        raise ValueError("No rows have valid slow and human reference measurements.")
    percentages = {
        category: round(100.0 * counts[category] / evaluated, 2)
        for category in CATEGORIES
    }
    return {
        "evaluated": evaluated,
        "counts": {category: counts[category] for category in CATEGORIES},
        "percentages": percentages,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.optimization_threshold < 1.0:
        raise ValueError("--optimization-threshold must be in [0, 1).")
    if not 0.0 <= args.pass_threshold <= 1.0:
        raise ValueError("--pass-threshold must be in [0, 1].")
    if not args.input_directory.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {args.input_directory}"
        )

    results: dict[str, dict[str, Any]] = {"python": {}, "cpp": {}}
    input_files = sorted(args.input_directory.glob("*.json"))
    if not input_files:
        raise ValueError("The input directory contains no JSON result files.")
    for path in input_files:
        parts = path.stem.split("__", maxsplit=1)
        if len(parts) != 2 or parts[0] not in results or not parts[1]:
            raise ValueError(
                f"Expected <python|cpp>__<method>.json, received {path.name!r}."
            )
        dataset, method = parts
        if method in results[dataset]:
            raise ValueError(f"Duplicate result for {dataset}/{method}.")
        results[dataset][method] = categorize(
            _load_metrics(path),
            args.optimization_threshold,
            args.pass_threshold,
            args.failure_runtime,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
