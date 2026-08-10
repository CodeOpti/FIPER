"""AST-guided replacement of a Python function body."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import textwrap


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _find_function(
    tree: ast.Module, function_name: str, class_name: str | None
) -> FunctionNode:
    if class_name and class_name != "Null":
        matching_classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(matching_classes) != 1:
            raise ValueError(f"Expected one class named {class_name!r}.")
        scope = matching_classes[0].body
    else:
        scope = tree.body

    matches = [
        node
        for node in scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one function named {function_name!r} in the requested scope."
        )
    return matches[0]


def _replacement_body(replacement_code: str, function_name: str) -> str:
    tree = ast.parse(replacement_code)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(
            "The candidate must contain exactly one top-level function definition "
            f"named {function_name!r}."
        )
    candidate = matches[0]
    if not candidate.body:
        raise ValueError("The candidate function has no body.")
    lines = replacement_code.splitlines()
    start = candidate.body[0].lineno - 1
    end = candidate.body[-1].end_lineno
    return textwrap.dedent("\n".join(lines[start:end])).rstrip()


def replace_function_body(
    file_path: str | Path,
    function_name: str,
    replacement_code: str,
    class_name: str | None = None,
) -> None:
    """Replace one function body while preserving the original signature.

    The candidate must define the same top-level function name. The target may be
    either module-level or a direct member of ``class_name``.
    """

    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = _find_function(tree, function_name, class_name)
    if not target.body:
        raise ValueError("The target function has no body.")

    body = _replacement_body(replacement_code, function_name)
    indentation = " " * (target.col_offset + 4)
    indented_body = textwrap.indent(body, indentation)

    lines = source.splitlines()
    start = target.body[0].lineno - 1
    end = target.end_lineno
    updated = "\n".join([*lines[:start], indented_body, *lines[end:]]) + "\n"
    ast.parse(updated)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(updated, encoding="utf-8", newline="\n")
    os.replace(temporary_path, path)
