"""Compute BM25 similarity scores for code and data-flow graph representations."""

from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base-path", required=True)
    parser.add_argument("--query-dataset-path", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--language", choices=("python", "cpp"), required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--code-column", default="Code_Uni")
    parser.add_argument("--graph-column", default="Code_Uni_Graph")
    parser.add_argument("--query-code-column", default="Slow_Code_Uni")
    parser.add_argument("--query-graph-column", default="Slow_Code_Uni_Graph")
    return parser.parse_args()


def _normalize_python_ast(source: str) -> str:
    class Abstractor(ast.NodeTransformer):
        def __init__(self) -> None:
            self.builtins = set(__builtins__ if isinstance(__builtins__, dict) else dir(__builtins__))

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id not in self.builtins:
                node.id = "VAR"
            return node

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = "VAR"
            return node

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                node.value = "STR"
            elif isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
                node.value = 0
            return node

    try:
        return ast.unparse(Abstractor().visit(ast.parse(str(source))))
    except (SyntaxError, ValueError, TypeError):
        return str(source)


def tokenize_python_code(source: str) -> list[str]:
    tokens: list[str] = []
    try:
        generator = tokenize.generate_tokens(io.StringIO(_normalize_python_ast(source)).readline)
        ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL}
        for token in generator:
            if token.type not in ignored and token.string.strip():
                tokens.append(token.string)
    except (IndentationError, tokenize.TokenError):
        tokens = re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|\S", str(source))
    return tokens or ["<empty>"]


def tokenize_cpp_code(source: str) -> list[str]:
    normalized = re.sub(r"\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'", "STR", str(source))
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "NUM", normalized)
    tokens = re.findall(r"::|->|==|!=|<=|>=|&&|\|\||[A-Za-z_]\w*|\S", normalized)
    return tokens or ["<empty>"]


def tokenize_code(source: Any, language: str) -> list[str]:
    return tokenize_python_code(str(source)) if language == "python" else tokenize_cpp_code(str(source))


def parse_graph(value: Any, language: str) -> list[str]:
    """Flatten serialized DFG tuples into BM25 tokens."""
    if value is None:
        return ["<empty>"]
    if isinstance(value, float) and value != value:
        return ["<empty>"]
    if isinstance(value, list):
        graph = value
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan" or text == "pass":
            return ["<empty>"]
        try:
            graph = ast.literal_eval(text) if language == "cpp" else text.splitlines()
        except (SyntaxError, ValueError):
            graph = text.splitlines()

    tokens: list[str] = []
    for item in graph:
        try:
            node = item if isinstance(item, tuple) else ast.literal_eval(f"({item})")
        except (SyntaxError, ValueError, TypeError):
            tokens.extend(re.findall(r"[A-Za-z_]\w*|\d+|\S", str(item)))
            continue
        if isinstance(node, (tuple, list)):
            for field in node:
                if isinstance(field, (tuple, list)):
                    tokens.extend(str(part) for part in field)
                else:
                    tokens.append(str(field))
    return tokens or ["<empty>"]


def build_corpus(values: Iterable[Any], language: str, graph: bool = False) -> list[list[str]]:
    parser = parse_graph if graph else tokenize_code
    return [parser(value, language) for value in values]


def compute_scores(query_tokens: list[str], corpus: list[list[str]]) -> list[float]:
    from rank_bm25 import BM25Okapi

    model = BM25Okapi(corpus, b=0.4)
    return [round(float(score), 2) for score in model.get_scores(query_tokens)]


def compute_similarity(
    knowledge_base_path: str,
    query_dataset_path: str,
    output_directory: str,
    language: str,
    start_index: int = 0,
    end_index: int | None = None,
    code_column: str = "Code_Uni",
    graph_column: str = "Code_Uni_Graph",
    query_code_column: str = "Slow_Code_Uni",
    query_graph_column: str = "Slow_Code_Uni_Graph",
) -> None:
    import pandas as pd

    knowledge_base = pd.read_csv(knowledge_base_path)
    query_dataset = pd.read_csv(query_dataset_path)
    required_columns = [code_column, graph_column]
    required_columns += [query_code_column, query_graph_column]
    missing = [column for column in required_columns if column not in knowledge_base.columns and column not in query_dataset.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    code_values = knowledge_base[code_column].tolist()
    graph_values = knowledge_base[graph_column].tolist()
    code_corpus = build_corpus(code_values, language)
    graph_corpus = build_corpus(graph_values, language, graph=True)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop = len(query_dataset) if end_index is None else min(end_index, len(query_dataset))
    if start_index < 0 or start_index >= len(query_dataset) and len(query_dataset) > 0:
        raise ValueError("start_index is outside the query dataset")
    for query_index in range(start_index, stop):
        code_query = tokenize_code(query_dataset.iloc[query_index][query_code_column], language)
        graph_query = parse_graph(query_dataset.iloc[query_index][query_graph_column], language)
        result = pd.DataFrame(
            {
                "DB_idx": list(range(len(knowledge_base))),
                "Code_Uni_Sim": compute_scores(code_query, code_corpus),
                "Graph_Uni_Sim": compute_scores(graph_query, graph_corpus),
            }
        )
        result.to_csv(output_dir / f"PIE_{query_index}_Sim.csv", index=False, encoding="utf-8-sig")
    print(f"Computed similarity files for rows {start_index} through {max(start_index, stop - 1)}")


def main() -> int:
    args = parse_args()
    compute_similarity(
        args.knowledge_base_path,
        args.query_dataset_path,
        args.output_directory,
        args.language,
        args.start_index,
        args.end_index,
        args.code_column,
        args.graph_column,
        args.query_code_column,
        args.query_graph_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
