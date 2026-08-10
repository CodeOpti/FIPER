"""Merge generated I/O records into a dataset without changing row order."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


def _as_record(value: Any) -> dict[str, list[str]]:
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
    if isinstance(record, list):
        records = [_as_record(item) for item in record]
        pairs = [
            pair
            for item in records
            for pair in zip(item["inputs"], item["outputs"])
        ]
        pairs = list(dict.fromkeys(pairs))
        return {"inputs": [item[0] for item in pairs], "outputs": [item[1] for item in pairs]}
    if not isinstance(record, dict):
        return {"inputs": [], "outputs": []}
    inputs = record.get("inputs", [])
    outputs = record.get("outputs", [])
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return {"inputs": [], "outputs": []}
    pairs = list(dict.fromkeys(zip(map(str, inputs), map(str, outputs))))
    return {"inputs": [item[0] for item in pairs], "outputs": [item[1] for item in pairs]}


def IOpostprocess_Cot_function(raw_IO_str: str, index: int = 0) -> dict[str, list[str]]:
    """Parse the ``Input:``/``Output:`` format used by older generation jobs."""
    text = str(raw_IO_str or "").replace("\\n", "\n").strip()
    if not text:
        return {"inputs": [], "outputs": []}

    records: list[tuple[str, str]] = []
    pattern = re.compile(
        r"(?:^|\n)\s*(?:#+\s*)?(?:Test Case\s*\d*\s*)?"
        r"Input\s*:\s*(.*?)\s*Output\s*:\s*(.*?)(?=\n\s*(?:#+\s*)?(?:Test Case|Input)\s*:?|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        input_text = match.group(1).strip().strip("`'\"")
        output_text = match.group(2).strip().strip("`'\"")
        if input_text and output_text:
            records.append((input_text + "\n", output_text + "\n"))

    unique_records = list(dict.fromkeys(records))
    return {
        "inputs": [item[0] for item in unique_records],
        "outputs": [item[1] for item in unique_records],
    }


def n5_combined_io_function(
    row_count: int, record_matrix: list[list[dict[str, list[str]]]]
) -> tuple[list[dict[str, list[str]]], float]:
    """Combine candidate records row by row and return the mean test count."""
    combined: list[dict[str, list[str]]] = []
    counts: list[int] = []
    for row_index in range(row_count):
        pairs: list[tuple[str, str]] = []
        for candidate_records in record_matrix:
            if row_index >= len(candidate_records):
                continue
            record = _as_record(candidate_records[row_index])
            pairs.extend(zip(record["inputs"], record["outputs"]))
        pairs = list(dict.fromkeys(pairs))
        combined.append({"inputs": [p[0] for p in pairs], "outputs": [p[1] for p in pairs]})
        counts.append(len(pairs))
    return combined, (sum(counts) / len(counts) if counts else 0.0)


def _read_source_records(source_path: Path, row_count: int, candidate_count: int) -> list[list[dict[str, list[str]]]]:
    if source_path.is_file():
        import pandas as pd

        generated = pd.read_csv(source_path)
        columns = [column for column in generated.columns if str(column).lower().startswith("generated_io")]
        columns = columns[:candidate_count] or ["Generated_IO"]
        matrix = []
        for column in columns:
            values = generated[column].tolist() if column in generated else [""] * row_count
            matrix.append([_as_record(value) for value in values[:row_count]])
        return matrix

    matrix: list[list[dict[str, list[str]]]] = []
    for candidate_index in range(candidate_count):
        records = []
        for row_index in range(row_count):
            candidates = [
                source_path / f"{row_index:04d}_{candidate_index}.txt",
                source_path / f"{row_index}_{candidate_index}.txt",
            ]
            file_path = next((path for path in candidates if path.exists()), None)
            records.append(
                IOpostprocess_Cot_function(file_path.read_text(encoding="utf-8"), row_index)
                if file_path
                else {"inputs": [], "outputs": []}
            )
        matrix.append(records)
    return matrix


def merge_generated_io_to_df_function(
    dataset_path: str,
    generated_path: str,
    output_path: str,
    column_name_prefix: str = "Generated_IO",
    candidate_count: int = 5,
) -> None:
    """Add candidate and combined I/O columns to ``dataset_path``."""
    import pandas as pd

    dataset = pd.read_csv(dataset_path)
    matrix = _read_source_records(Path(generated_path), len(dataset), candidate_count)
    for index, records in enumerate(matrix, start=1):
        dataset[f"{column_name_prefix}_{index}"] = records
    combined, mean_count = n5_combined_io_function(len(dataset), matrix)
    dataset[column_name_prefix] = combined
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Merged {len(dataset)} rows; mean test count: {mean_count:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--generated-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--column-prefix", default="Generated_IO")
    parser.add_argument("--candidate-count", type=int, default=5)
    args = parser.parse_args()
    merge_generated_io_to_df_function(
        args.dataset_path,
        args.generated_path,
        args.output_path,
        args.column_prefix,
        args.candidate_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
