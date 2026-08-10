"""Merge generated natural-language problem descriptions into a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _read_descriptions(source_path: Path, row_count: int) -> list[str]:
    if source_path.is_file():
        import pandas as pd

        generated = pd.read_csv(source_path)
        candidates = [
            column
            for column in generated.columns
            if str(column).lower() in {"overview_description", "generated_overview", "description"}
        ]
        if not candidates:
            raise ValueError(f"No description column found in {source_path}")
        values = generated[candidates[0]].fillna("").astype(str).tolist()
        return values[:row_count] + [""] * max(0, row_count - len(values))

    descriptions: list[str] = []
    for row_index in range(row_count):
        candidates = [
            source_path / f"{row_index:04d}_0.txt",
            source_path / f"{row_index}.txt",
            source_path / f"{row_index:04d}.txt",
        ]
        file_path = next((path for path in candidates if path.exists()), None)
        descriptions.append(file_path.read_text(encoding="utf-8").strip() if file_path else "")
    return descriptions


def merge_generated_io_description_function(
    dataset_path: str,
    generated_path: str,
    output_path: str,
    column_name_prefix: str = "Overview_Description",
    _legacy_mode: Any = None,
) -> None:
    """Add one generated description column while preserving dataset row order."""
    import pandas as pd

    dataset = pd.read_csv(dataset_path)
    descriptions = _read_descriptions(Path(generated_path), len(dataset))
    dataset[column_name_prefix] = descriptions
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Merged descriptions for {len(dataset)} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--generated-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--column-name", default="Overview_Description")
    args = parser.parse_args()
    merge_generated_io_description_function(
        args.dataset_path,
        args.generated_path,
        args.output_path,
        args.column_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
