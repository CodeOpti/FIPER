"""Export Top-k code-generation results and speedup metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 0.999
DEFAULT_FALLBACK_THRESHOLD = 0.05
DEFAULT_LAST_RESORT_THRESHOLD = 0.0001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--column-prefix", required=True)
    parser.add_argument("--baseline-prefix", default="input")
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--submission-count", type=int, default=3)
    parser.add_argument("--mode", choices=("all", "top1", "topk"), default="all")
    parser.add_argument("--io-selection", choices=("public", "generated"), default="public")
    parser.add_argument("--save-selections-path")
    return parser.parse_args()


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _metric_columns(prefix: str, candidate_index: int) -> dict[str, str]:
    base = f"{prefix}__Predict_Fast_code_{candidate_index}"
    return {
        "public_pass": f"{base}__Public_IO_pass_rate_(%)",
        "public_time": f"{base}__Public_time(ms)",
        "generated_pass": f"{base}__Gen_IO_pass_rate_(%)",
        "generated_time": f"{base}__Gen_time(ms)",
        "private_pass": f"{base}__IO_pass_rate_(%)",
        "private_time": f"{base}__time(ms)",
    }


def load_metrics(frame: Any, prefix: str, candidate_count: int) -> dict[str, list[list[float]]]:
    metrics = {key: [] for key in ("public_pass", "public_time", "generated_pass", "generated_time", "private_pass", "private_time")}
    missing: list[str] = []
    for candidate_index in range(1, candidate_count + 1):
        columns = _metric_columns(prefix, candidate_index)
        for metric, column in columns.items():
            if column not in frame.columns:
                missing.append(column)
                metrics[metric].append([0.0] * len(frame))
            else:
                metrics[metric].append([_finite_float(value, math.inf if "time" in metric else 0.0) for value in frame[column]])
    if missing:
        preview = ", ".join(missing[:4])
        raise KeyError(f"Missing candidate metric columns: {preview}")
    return metrics


def _candidate_pool(values: list[float], threshold: float) -> list[int]:
    pool = [index for index, value in enumerate(values) if value >= threshold]
    if not pool:
        pool = [index for index, value in enumerate(values) if value >= DEFAULT_FALLBACK_THRESHOLD]
    return pool


def _select_one(
    metrics: dict[str, list[list[float]]],
    row_index: int,
    generated_count: int,
    submission_count: int,
    io_selection: str,
) -> tuple[int, float, float]:
    private_pass = [metrics["private_pass"][candidate][row_index] for candidate in range(generated_count)]
    private_time = [metrics["private_time"][candidate][row_index] for candidate in range(generated_count)]
    if generated_count == submission_count:
        candidates = _candidate_pool(private_pass, DEFAULT_THRESHOLD)
        if not candidates:
            candidates = _candidate_pool(private_pass, DEFAULT_LAST_RESORT_THRESHOLD)
        chosen = min(candidates or [0], key=lambda index: private_time[index])
        return chosen, private_pass[chosen], private_time[chosen]

    gate_pass_key = "public_pass" if io_selection == "public" else "generated_pass"
    gate_time_key = "public_time" if io_selection == "public" else "generated_time"
    gate_pass = [metrics[gate_pass_key][candidate][row_index] for candidate in range(generated_count)]
    gate_time = [metrics[gate_time_key][candidate][row_index] for candidate in range(generated_count)]
    candidates = _candidate_pool(gate_pass, DEFAULT_THRESHOLD)
    ranked = sorted(candidates, key=lambda index: gate_time[index])[:submission_count] if candidates else [0]
    private_candidates = [index for index in ranked if private_pass[index] >= DEFAULT_THRESHOLD]
    if not private_candidates:
        private_candidates = [index for index in ranked if private_pass[index] >= DEFAULT_FALLBACK_THRESHOLD]
    if not private_candidates:
        private_candidates = [index for index in ranked if private_pass[index] >= DEFAULT_LAST_RESORT_THRESHOLD]
    chosen = min(private_candidates or [0], key=lambda index: private_time[index])
    return chosen, private_pass[chosen], private_time[chosen]


def select_top_k(
    metrics: dict[str, list[list[float]]],
    generated_count: int,
    submission_count: int,
    io_selection: str,
) -> tuple[list[float], list[float], list[int]]:
    if not 1 <= submission_count <= generated_count:
        raise ValueError("submission_count must be between 1 and candidate_count")
    selected_pass: list[float] = []
    selected_time: list[float] = []
    selected_indices: list[int] = []
    row_count = len(metrics["private_pass"][0])
    for row_index in range(row_count):
        candidate, pass_rate, elapsed = _select_one(
            metrics,
            row_index,
            generated_count,
            submission_count,
            io_selection,
        )
        selected_indices.append(candidate + 1)
        selected_pass.append(pass_rate)
        selected_time.append(elapsed)
    return selected_pass, selected_time, selected_indices


def summarize_metrics(
    baseline_pass: list[float],
    baseline_time: list[float],
    candidate_pass: list[float],
    candidate_time: list[float],
    scenario: str,
) -> dict[str, Any]:
    if not (len(baseline_pass) == len(baseline_time) == len(candidate_pass) == len(candidate_time)):
        raise ValueError("Baseline and candidate metric lengths do not match")
    pass_threshold_count = sum(value >= DEFAULT_THRESHOLD for value in candidate_pass)
    speedups: list[float] = []
    optimization_flags: list[int] = []
    for base_pass, base_time, fast_pass, fast_time in zip(baseline_pass, baseline_time, candidate_pass, candidate_time):
        valid = base_pass > DEFAULT_THRESHOLD and fast_pass > DEFAULT_THRESHOLD
        valid = valid and base_time > 0 and fast_time > 0 and fast_time < 12_345_678
        if valid and fast_time < base_time:
            speedups.append(round(base_time / fast_time, 6))
            optimization_flags.append(int((base_time - fast_time) / base_time > 0.10))
        else:
            speedups.append(1.0)
            optimization_flags.append(0)
    return {
        "scenario": scenario,
        "rows": len(candidate_pass),
        "mean_pass_rate_pct": round(pass_threshold_count / len(candidate_pass) * 100, 2) if candidate_pass else 0.0,
        "optimization_rate_pct": round(sum(optimization_flags) / len(optimization_flags) * 100, 2) if optimization_flags else 0.0,
        "mean_speedup": round(sum(speedups) / len(speedups), 4) if speedups else 1.0,
    }


def _baseline_columns(frame: Any, prefix: str) -> tuple[str, str]:
    candidates = [
        (f"{prefix}__IO_pass_rate_(%)", f"{prefix}__time(ms)"),
        (f"{prefix}__IO_pass_rate_(%)", f"{prefix}__time(us)"),
    ]
    for pass_column, time_column in candidates:
        if pass_column in frame.columns and time_column in frame.columns:
            return pass_column, time_column
    raise KeyError(f"Could not find baseline metrics for prefix {prefix!r}")


def export_results(
    dataset_path: str,
    output_path: str,
    column_prefix: str,
    baseline_prefix: str = "input",
    candidate_count: int = 5,
    submission_count: int = 3,
    mode: str = "all",
    io_selection: str = "public",
    save_selections_path: str | None = None,
) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(dataset_path)
    baseline_pass_column, baseline_time_column = _baseline_columns(frame, baseline_prefix)
    baseline_pass = [_finite_float(value) for value in frame[baseline_pass_column]]
    baseline_time = [_finite_float(value) for value in frame[baseline_time_column]]
    metrics = load_metrics(frame, column_prefix, candidate_count)
    scenarios: list[tuple[str, int, int, str]] = []
    if mode in {"all", "top1"}:
        scenarios.extend((f"candidate_{index}", 1, 1, "private") for index in range(1, candidate_count + 1))
    if mode in {"all", "topk"}:
        if mode == "topk":
            scenarios.append((f"top_{candidate_count}_select_{submission_count}", candidate_count, submission_count, io_selection))
        else:
            scenarios.extend(
                (
                    f"top_{generated}_select_{submitted}",
                    generated,
                    submitted,
                    io_selection,
                )
                for generated, submitted in ((3, 3), (5, 5), (3, 1), (5, 1), (5, 3))
                if generated <= candidate_count
            )

    results: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for scenario, generated, submitted, ranking_metric in scenarios:
        if generated == 1:
            candidate_pass = metrics["private_pass"][0]
            candidate_time = metrics["private_time"][0]
            indices = [1] * len(candidate_pass)
        else:
            candidate_pass, candidate_time, indices = select_top_k(metrics, generated, submitted, ranking_metric)
        summary = summarize_metrics(baseline_pass, baseline_time, candidate_pass, candidate_time, scenario)
        summary.update(
            {
                "generated_count": generated,
                "submission_count": submitted,
                "ranking_metric": ranking_metric,
            }
        )
        results.append(summary)
        selection_rows.extend(
            {
                "row_index": row_index,
                "scenario": scenario,
                "selected_candidate": indices[row_index],
                "selected_pass_rate": candidate_pass[row_index],
                "selected_time_ms": candidate_time[row_index],
            }
            for row_index in range(len(indices))
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    if destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        result_frame.to_csv(destination, index=False, encoding="utf-8-sig")
    if save_selections_path:
        selection_destination = Path(save_selections_path)
        selection_destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(selection_rows).to_csv(selection_destination, index=False, encoding="utf-8-sig")
    print(f"Exported {len(results)} result scenarios to {destination}")
    return results


def main() -> int:
    args = parse_args()
    export_results(
        args.dataset_path,
        args.output_path,
        args.column_prefix,
        args.baseline_prefix,
        args.candidate_count,
        args.submission_count,
        args.mode,
        args.io_selection,
        args.save_selections_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
