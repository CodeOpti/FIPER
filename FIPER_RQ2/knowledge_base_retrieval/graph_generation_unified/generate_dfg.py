"""Generate data-flow graph columns for Python or C++ datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


PARSER_DIRECTORY = Path(__file__).resolve().parent
if str(PARSER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PARSER_DIRECTORY))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--language", choices=("python", "cpp"), required=True)
    parser.add_argument("--slow-column", default="Slow_Code")
    parser.add_argument("--fast-column", default="Fast_Code")
    parser.add_argument("--slow-unified-column", default="Slow_Code_Uni")
    parser.add_argument("--fast-unified-column", default="Fast_Code_Uni")
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep rows whose graph parser returns an empty graph.",
    )
    return parser.parse_args()


def build_parser(language: str) -> tuple[Any, Callable[..., Any], Callable[[str, str], str]]:
    from tree_sitter import Language, Parser
    from dfg_parser import DFG_csharp, DFG_python, index_to_code_token, remove_comments_and_docstrings, tree_to_token_index

    if language == "python":
        import tree_sitter_python

        language_object = Language(tree_sitter_python.language())
        dfg_function = DFG_python
    else:
        import tree_sitter_cpp

        language_object = Language(tree_sitter_cpp.language())
        dfg_function = DFG_csharp

    try:
        parser = Parser(language_object)
    except TypeError:
        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(language_object)
        else:
            parser.language = language_object

    def remove_comments(source: str, source_language: str) -> str:
        return remove_comments_and_docstrings(source, source_language)

    def graph_from_source(source: str, source_language: str) -> str:
        text = "" if source is None else str(source)
        if not text.strip() or text.strip().lower() == "nan":
            return ""
        try:
            cleaned = remove_comments(text, source_language)
            tree = parser.parse(cleaned.encode("utf-8"))
            root = tree.root_node
            token_indices = tree_to_token_index(root)
            code_lines = cleaned.splitlines()
            code_tokens = [index_to_code_token(index, code_lines) for index in token_indices]
            index_to_code = {
                index: (token_index, token)
                for token_index, (index, token) in enumerate(zip(token_indices, code_tokens))
            }
            data_flow, _ = dfg_function(root, index_to_code, {})
            data_flow = sorted(data_flow, key=lambda item: item[1])
            referenced_indices = set()
            for item in data_flow:
                if item[-1]:
                    referenced_indices.add(item[1])
                referenced_indices.update(item[-1])
            filtered = [item for item in data_flow if item[1] in referenced_indices]
            if source_language == "python":
                return "\n".join(str(item)[1:-1] for item in filtered).strip()
            return repr(filtered)
        except Exception:
            return ""

    return parser, graph_from_source, remove_comments


def generate_graph_columns(
    input_path: str,
    output_path: str,
    language: str,
    slow_column: str = "Slow_Code",
    fast_column: str = "Fast_Code",
    slow_unified_column: str = "Slow_Code_Uni",
    fast_unified_column: str = "Fast_Code_Uni",
    keep_failed: bool = False,
) -> None:
    import pandas as pd

    frame = pd.read_csv(input_path)
    source_columns = [slow_column, fast_column, slow_unified_column, fast_unified_column]
    missing = [column for column in source_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing source-code columns: {missing}")
    _, graph_from_source, _ = build_parser(language)
    for column in source_columns:
        frame[f"{column}_Graph"] = frame[column].map(lambda value: graph_from_source(value, language))
    graph_columns = [f"{column}_Graph" for column in source_columns]
    if not keep_failed:
        valid = frame[graph_columns].applymap(lambda value: bool(str(value).strip()))
        frame = frame[valid.all(axis=1)].copy()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    print(f"Generated graph columns for {len(frame)} rows: {destination}")


def main() -> int:
    args = parse_args()
    generate_graph_columns(
        args.input_path,
        args.output_path,
        args.language,
        args.slow_column,
        args.fast_column,
        args.slow_unified_column,
        args.fast_unified_column,
        args.keep_failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
