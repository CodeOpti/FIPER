"""Merge generated optimized programs into a dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def CodeLlamacode_postprocess_function(raw_code_str: str, index: int = 0) -> str:
    """Extract the first fenced code block or return the raw model response."""
    text = str(raw_code_str or "").strip()
    match = re.search(r"```(?:python|py|cpp|c\+\+)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def checkcode_function(code: str) -> bool:
    """Return whether a candidate contains executable-looking source text."""
    stripped = CodeLlamacode_postprocess_function(code)
    return bool(stripped and not stripped.lower().startswith(("i cannot", "sorry", "unable")))


def _read_candidates(source_path: Path, row_count: int, candidate_count: int) -> list[list[str]]:
    if source_path.is_file():
        import pandas as pd

        generated = pd.read_csv(source_path)
        columns = [
            column
            for column in generated.columns
            if str(column).lower().startswith(("optimized_code", "generated_code", "fast_code"))
        ]
        columns = columns[:candidate_count]
        if not columns:
            raise ValueError(f"No generated-code columns found in {source_path}")
        return [
            [CodeLlamacode_postprocess_function(value) for value in generated[column].fillna("").tolist()[:row_count]]
            for column in columns
        ]

    matrix: list[list[str]] = []
    for candidate_index in range(candidate_count):
        records: list[str] = []
        for row_index in range(row_count):
            candidates = [
                source_path / f"{row_index:04d}_{candidate_index}.txt",
                source_path / f"{row_index}_{candidate_index}.txt",
            ]
            file_path = next((path for path in candidates if path.exists()), None)
            records.append(
                CodeLlamacode_postprocess_function(file_path.read_text(encoding="utf-8"))
                if file_path
                else ""
            )
        matrix.append(records)
    return matrix


def merge_to_df_fast_code_function(
    dataset_path: str,
    generated_path: str,
    output_path: str,
    column_name_prefix: str = "Optimized_Code",
    candidate_count: int = 5,
) -> None:
    """Add generated code candidates to a CSV and write a new CSV."""
    import pandas as pd

    dataset = pd.read_csv(dataset_path)
    matrix = _read_candidates(Path(generated_path), len(dataset), candidate_count)
    for index, candidates in enumerate(matrix, start=1):
        dataset[f"{column_name_prefix}_{index}"] = candidates
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Merged {len(matrix)} candidate columns for {len(dataset)} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--generated-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--column-prefix", default="Optimized_Code")
    parser.add_argument("--candidate-count", type=int, default=5)
    args = parser.parse_args()
    merge_to_df_fast_code_function(
        args.dataset_path,
        args.generated_path,
        args.output_path,
        args.column_prefix,
        args.candidate_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
