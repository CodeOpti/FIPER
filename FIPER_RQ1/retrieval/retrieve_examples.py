"""Retrieve problem-disjoint algorithm/test examples for RQ1 test synthesis."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import pandas as pd


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]")


class BM25Index:
    """Small dependency-free BM25 implementation for reproducible retrieval."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.4) -> None:
        self.corpus = [Counter(document) for document in corpus]
        self.lengths = [sum(document.values()) for document in self.corpus]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.k1 = k1
        self.b = b
        document_frequency = Counter(
            token for document in self.corpus for token in document.keys()
        )
        size = len(self.corpus)
        self.idf = {
            token: math.log(1 + (size - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for token in set(query):
            inverse_frequency = self.idf.get(token, 0.0)
            for index, document in enumerate(self.corpus):
                frequency = document.get(token, 0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.average_length
                )
                if denominator:
                    scores[index] += inverse_frequency * frequency * (self.k1 + 1) / denominator
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--knowledge-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--allow-missing-problem-ids",
        action="store_true",
        help="Disable the problem-disjoint ID check (not recommended for paper runs).",
    )
    return parser.parse_args()


def tokenize(value: object) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(str(value))]


def _normalized(values: list[float]) -> list[float]:
    minimum, maximum = min(values), max(values)
    if maximum == minimum:
        return [0.0] * len(values)
    return [(value - minimum) / (maximum - minimum) for value in values]


def _validate_schema(
    queries: pd.DataFrame,
    knowledge_base: pd.DataFrame,
    allow_missing_ids: bool,
) -> None:
    missing_query = {"slow_code"}.difference(queries.columns)
    missing_knowledge = {"code", "reference_tests"}.difference(knowledge_base.columns)
    if missing_query or missing_knowledge:
        raise ValueError(
            "Missing columns: "
            f"queries={sorted(missing_query)}, knowledge_base={sorted(missing_knowledge)}"
        )
    if not allow_missing_ids:
        if "problem_id" not in queries.columns or "problem_id" not in knowledge_base.columns:
            raise ValueError(
                "Both CSVs require problem_id for a verifiable problem-disjoint run. "
                "Use --allow-missing-problem-ids only for non-paper exploration."
            )
        if queries["problem_id"].isna().any() or knowledge_base["problem_id"].isna().any():
            raise ValueError("problem_id values must be non-empty for paper runs.")
        overlap = set(queries["problem_id"].astype(str)) & set(
            knowledge_base["problem_id"].astype(str)
        )
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Query/knowledge-base problem IDs overlap: {preview}")


def _format_reference(row: pd.Series) -> str:
    return (
        "Reference program:\n"
        f"```\n{row['code']}\n```\n"
        "Reference tests:\n"
        f"{row['reference_tests']}"
    )


def retrieve(
    queries: pd.DataFrame,
    knowledge_base: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    code_corpus = [tokenize(value) for value in knowledge_base["code"]]
    if any(not tokens for tokens in code_corpus):
        raise ValueError("Knowledge-base code entries must be non-empty.")
    code_index = BM25Index(code_corpus, b=0.4)

    use_dfg = "dfg" in queries.columns and "dfg" in knowledge_base.columns
    dfg_index = None
    if use_dfg:
        dfg_corpus = [tokenize(value) for value in knowledge_base["dfg"]]
        if all(dfg_corpus):
            dfg_index = BM25Index(dfg_corpus, b=0.4)

    reference_examples: list[str] = []
    retrieved_indices: list[str] = []
    retrieved_scores: list[str] = []
    for _, query in queries.iterrows():
        code_scores = _normalized(code_index.get_scores(tokenize(query["slow_code"])))
        combined_scores = code_scores
        if dfg_index is not None:
            dfg_scores = _normalized(dfg_index.get_scores(tokenize(query["dfg"])))
            combined_scores = [
                (code_score + dfg_score) / 2
                for code_score, dfg_score in zip(code_scores, dfg_scores)
            ]

        order = sorted(
            range(len(knowledge_base)),
            key=lambda index: (-combined_scores[index], index),
        )[:top_k]
        references = [
            f"Example {rank}:\n{_format_reference(knowledge_base.iloc[index])}"
            for rank, index in enumerate(order, start=1)
        ]
        reference_examples.append("\n\n".join(references))
        retrieved_indices.append(json.dumps(order))
        retrieved_scores.append(json.dumps([combined_scores[index] for index in order]))

    result = queries.copy()
    result["reference_examples"] = reference_examples
    result["retrieved_knowledge_indices"] = retrieved_indices
    result["retrieved_scores"] = retrieved_scores
    return result


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    queries = pd.read_csv(args.queries)
    knowledge_base = pd.read_csv(args.knowledge_base)
    if len(knowledge_base) < args.top_k:
        raise ValueError("The knowledge base contains fewer rows than --top-k.")
    _validate_schema(queries, knowledge_base, args.allow_missing_problem_ids)
    result = retrieve(queries, knowledge_base, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
