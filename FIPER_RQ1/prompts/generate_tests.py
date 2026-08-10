"""Prompt construction for autonomous test-input synthesis."""

from __future__ import annotations


def system_prompt(language: str) -> str:
    return (
        "You are a senior software test engineer. Generate diverse contest-style "
        f"standard-input cases for the supplied {language} program. Do not use or "
        "infer any benchmark public, private, or hidden tests. Return strict JSON only."
    )


def build_prompt(
    slow_code: str,
    language: str,
    reference_examples: str = "",
    minimum_tests: int = 50,
) -> str:
    references = reference_examples.strip() or "No retrieval examples were supplied."
    return f"""Create at least {minimum_tests} independent test inputs for this program.

Use only the program and the problem-disjoint retrieval examples below. The examples
may demonstrate input shape, but they are not evaluation tests. Cover ordinary cases,
boundaries, degenerate inputs, and high-load inputs. Do not execute or reproduce any
benchmark test suite. Outputs are intentionally omitted because the original slow
program will serve as the oracle in a later stage.

Problem-disjoint retrieval examples:
{references}

Program ({language}):
```{language}
{slow_code}
```

Return exactly one JSON object with this schema:
{{"inputs": ["first complete stdin string", "second complete stdin string"]}}
"""
