"""Oracle-verify synthesized RQ2 inputs and select ten representative tests."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


RQ2_ROOT = Path(__file__).resolve().parents[1]
if str(RQ2_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ2_ROOT))

from pie_sandbox import capture_oracle_outputs


FORBIDDEN_EVALUATION_TERMS = ("private", "hidden", "hide_io")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--language", choices=("python", "cpp"), default="python")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--representative-count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stage",
        choices=("initial", "second", "final"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _load_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() == "nan":
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return None


def parse_candidate_inputs(value: Any) -> list[str]:
    """Extract inputs while deliberately ignoring model-proposed outputs."""

    loaded = _load_value(value)
    if isinstance(loaded, list):
        return [item for value in loaded for item in parse_candidate_inputs(value)]
    if isinstance(loaded, dict) and isinstance(loaded.get("inputs"), list):
        return [str(item) for item in loaded["inputs"]]

    text = "" if value is None else str(value).replace("\r", "")
    return [
        match.group(1).strip().strip("`'\"")
        for match in re.finditer(
            r"(?:^|\n)\s*(?:#+\s*)?(?:Test Case\s*\d*\s*)?Input\s*:\s*(.*?)(?=\n\s*(?:Output|#+\s*Test Case|Input)\s*:|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match.group(1).strip()
    ]


def candidate_columns(frame: Any) -> list[str]:
    numbered = sorted(
        (
            str(column)
            for column in frame.columns
            if re.fullmatch(r"Generated_IO_\d+", str(column))
        ),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    if numbered:
        return numbered
    return ["Generated_IO"] if "Generated_IO" in frame.columns else []


def _validated_inputs(values: list[Any], max_input_tokens: int) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        for input_text in parse_candidate_inputs(value):
            token_count = len(re.findall(r"\w+|[^\w\s]", input_text, flags=re.UNICODE))
            if (
                not input_text.strip()
                or token_count > max_input_tokens
                or input_text in seen
            ):
                continue
            seen.add(input_text)
            unique.append(input_text)
    return unique


def _input_features(inputs: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for value in inputs:
        characters = list(value)
        length = max(len(characters), 1)
        rows.append(
            [
                float(len(value)),
                float(value.count("\n") + 1),
                float(len(value.split())),
                float(sum(character.isdigit() for character in characters)),
                float(sum(character.isalpha() for character in characters)),
                float(sum(character.isspace() for character in characters)),
                float(len(set(characters)) / length),
                float(value.count("-")),
                float(value.count(".")),
                float(max((len(line) for line in value.splitlines()), default=0)),
            ]
        )
    return np.asarray(rows, dtype=float)


def _cluster_outlier_scores(inputs: list[str]) -> list[float]:
    if len(inputs) < 2:
        return [1.0] * len(inputs)
    features = _input_features(inputs)
    centered = features - features.mean(axis=0, keepdims=True)
    scales = centered.std(axis=0, keepdims=True)
    standardized = centered / np.where(scales == 0.0, 1.0, scales)
    _, _, right_vectors = np.linalg.svd(standardized, full_matrices=False)
    points = standardized @ right_vectors[: min(2, right_vectors.shape[0])].T
    cluster_count = min(5, max(1, round(math.sqrt(len(inputs)))))

    center_indices = [int(np.argmax(np.linalg.norm(points, axis=1)))]
    while len(center_indices) < cluster_count:
        distances = np.min(
            np.stack(
                [np.linalg.norm(points - points[index], axis=1) for index in center_indices]
            ),
            axis=0,
        )
        distances[center_indices] = -1.0
        center_indices.append(int(np.argmax(distances)))
    centers = points[center_indices].copy()
    for _ in range(25):
        distances = np.stack(
            [np.linalg.norm(points - center, axis=1) for center in centers], axis=1
        )
        assignments = np.argmin(distances, axis=1)
        updated = centers.copy()
        for cluster in range(cluster_count):
            members = points[assignments == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return [
        float(np.linalg.norm(point - centers[assignment]))
        for point, assignment in zip(points, assignments)
    ]


def representative_indices(
    inputs: list[str], runtimes_us: list[float], limit: int
) -> list[int]:
    if len(inputs) <= limit:
        return list(range(len(inputs)))
    boundary_scores = _cluster_outlier_scores(inputs)
    boundary_order = sorted(
        range(len(inputs)),
        key=lambda index: -boundary_scores[index],
    )
    load_order = sorted(range(len(inputs)), key=lambda index: -runtimes_us[index])
    boundary_target = min(5, (limit + 1) // 2)
    selected = boundary_order[:boundary_target]
    for index in load_order:
        if index not in selected:
            selected.append(index)
        if len(selected) == limit:
            break
    return selected


def _language(value: Any, default: str) -> str:
    normalized = str(value or default).casefold()
    if "++" in normalized or "cpp" in normalized:
        return "cpp"
    if "py" in normalized:
        return "python"
    return default


def _process_row(
    position: int,
    row: Any,
    columns: list[str],
    args: argparse.Namespace,
) -> tuple[int, str, str, str]:
    slow_column = "Slow_Code" if "Slow_Code" in row.index else "slow_code"
    if slow_column not in row.index:
        raise KeyError("The RQ2 input must contain Slow_Code or slow_code.")
    inputs = _validated_inputs(
        [row[column] for column in columns],
        args.max_input_tokens,
    )
    if not inputs:
        raise ValueError(f"Row {position} contains no usable synthesized inputs.")
    oracle = capture_oracle_outputs(
        str(row[slow_column]),
        inputs,
        _language(row.get("language"), args.language),
        args.timeout_seconds,
    )
    verified_inputs = [str(value) for value in oracle["inputs"]]
    verified_outputs = [str(value) for value in oracle["outputs"]]
    runtimes_us = [float(value) for value in oracle["runtime_us"]]
    if not verified_inputs:
        raise RuntimeError(f"The slow-program oracle rejected every input in row {position}.")
    selected = representative_indices(
        verified_inputs,
        runtimes_us,
        args.representative_count,
    )
    full_suite = {"inputs": verified_inputs, "outputs": verified_outputs}
    representatives = {
        "inputs": [verified_inputs[index] for index in selected],
        "outputs": [verified_outputs[index] for index in selected],
    }
    return (
        position,
        json.dumps(full_suite, ensure_ascii=True),
        json.dumps(representatives, ensure_ascii=True),
        json.dumps(runtimes_us),
    )


def main() -> None:
    args = parse_args()
    if (
        args.timeout_seconds <= 0
        or args.max_input_tokens < 1
        or args.representative_count < 1
        or args.workers < 1
    ):
        raise ValueError("Timeout and count options must be positive.")

    import pandas as pd

    frame = pd.read_csv(args.dataset_path)
    forbidden = [
        column
        for column in frame.columns
        if any(term in str(column).casefold() for term in FORBIDDEN_EVALUATION_TERMS)
    ]
    if forbidden:
        frame = frame.drop(columns=forbidden)
    columns = candidate_columns(frame)
    if not columns:
        raise KeyError("No Generated_IO or Generated_IO_N columns were found.")

    results: list[tuple[int, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_process_row, position, row, columns, args)
            for position, (_, row) in enumerate(frame.iterrows())
        ]
        for future in futures:
            results.append(future.result())

    frame["Selected_IO"] = ""
    frame["Representative_IO"] = ""
    frame["Generated_Oracle_Runtime_Us"] = ""
    for position, full_suite, representatives, runtimes in results:
        row_index = frame.index[position]
        frame.at[row_index, "Selected_IO"] = full_suite
        frame.at[row_index, "Representative_IO"] = representatives
        frame.at[row_index, "Generated_Oracle_Runtime_Us"] = runtimes

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Oracle-verified synthesized tests for {len(frame)} rows: {output_path}")


if __name__ == "__main__":
    main()
