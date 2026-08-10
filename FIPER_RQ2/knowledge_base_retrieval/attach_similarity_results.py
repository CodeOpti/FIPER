"""Attach top graph-retrieval matches and private I/O data to a dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base-path", required=True)
    parser.add_argument("--query-dataset-path", required=True)
    parser.add_argument("--similarity-directory", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--code-column", default="Code_Uni")
    parser.add_argument("--graph-column", default="Code_Uni_Graph")
    parser.add_argument("--private-io-column", default="Representative_Private_IO")
    parser.add_argument("--code-weight", type=float, default=2.0)
    parser.add_argument("--graph-weight", type=float, default=1.0)
    return parser.parse_args()


def min_max_scale(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def attach_similarity_results(
    knowledge_base_path: str,
    query_dataset_path: str,
    similarity_directory: str,
    output_path: str,
    top_k: int = 4,
    code_column: str = "Code_Uni",
    graph_column: str = "Code_Uni_Graph",
    private_io_column: str = "Representative_Private_IO",
    code_weight: float = 2.0,
    graph_weight: float = 1.0,
) -> None:
    import pandas as pd

    if top_k < 1:
        raise ValueError("top_k must be positive")
    knowledge_base = pd.read_csv(knowledge_base_path)
    query_dataset = pd.read_csv(query_dataset_path)
    required = [code_column, graph_column, private_io_column]
    missing = [column for column in required if column not in knowledge_base.columns]
    if missing:
        raise KeyError(f"Missing knowledge-base columns: {missing}")

    similarity_dir = Path(similarity_directory)
    output_columns: dict[str, list[Any]] = {
        **{f"Max_Sim_idx__{rank}": [] for rank in range(top_k)},
        **{f"Max_Sim_Code__{rank}": [] for rank in range(top_k)},
        **{f"Max_Sim_Graph__{rank}": [] for rank in range(top_k)},
        **{f"Max_Sim_Top5__{rank}": [] for rank in range(top_k)},
        **{f"Max_Sim_Score__{rank}": [] for rank in range(top_k)},
    }

    for query_index in range(len(query_dataset)):
        score_path = similarity_dir / f"PIE_{query_index}_Sim.csv"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing similarity file: {score_path}")
        scores = pd.read_csv(score_path)
        for column in ("DB_idx", "Code_Uni_Sim", "Graph_Uni_Sim"):
            if column not in scores.columns:
                raise KeyError(f"Missing similarity column {column!r} in {score_path}")
        code_scores = [_safe_float(value) for value in scores["Code_Uni_Sim"]]
        graph_scores = [_safe_float(value) for value in scores["Graph_Uni_Sim"]]
        combined = [
            code_weight * code_score + graph_weight * graph_score
            for code_score, graph_score in zip(min_max_scale(code_scores), min_max_scale(graph_scores))
        ]
        ranking = sorted(range(len(combined)), key=lambda index: combined[index], reverse=True)[:top_k]
        ranking += [ranking[0]] * (top_k - len(ranking)) if ranking else [0] * top_k
        for rank, score_index in enumerate(ranking[:top_k]):
            db_index = int(scores.iloc[score_index]["DB_idx"])
            if not 0 <= db_index < len(knowledge_base):
                raise IndexError(f"DB_idx {db_index} is outside the knowledge base")
            output_columns[f"Max_Sim_idx__{rank}"].append(db_index)
            output_columns[f"Max_Sim_Code__{rank}"].append(knowledge_base.iloc[db_index][code_column])
            output_columns[f"Max_Sim_Graph__{rank}"].append(knowledge_base.iloc[db_index][graph_column])
            output_columns[f"Max_Sim_Top5__{rank}"].append(knowledge_base.iloc[db_index][private_io_column])
            output_columns[f"Max_Sim_Score__{rank}"].append(round(combined[score_index], 6))

    for column, values in output_columns.items():
        query_dataset[column] = values
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    query_dataset.to_csv(destination, index=False, encoding="utf-8-sig")
    print(f"Attached top-{top_k} retrieval results for {len(query_dataset)} rows: {destination}")


def main() -> int:
    args = parse_args()
    attach_similarity_results(
        args.knowledge_base_path,
        args.query_dataset_path,
        args.similarity_directory,
        args.output_path,
        args.top_k,
        args.code_column,
        args.graph_column,
        args.private_io_column,
        args.code_weight,
        args.graph_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
