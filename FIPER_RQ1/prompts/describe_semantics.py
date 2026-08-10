"""Prompt construction for the test-driven semantic specification."""

from __future__ import annotations

import json


def system_prompt(language: str) -> str:
    return (
        "You are a software specification writer. Infer behavior only from the given "
        f"{language} program and oracle-verified synthesized tests."
    )


def build_prompt(slow_code: str, language: str, generated_tests: dict) -> str:
    tests = json.dumps(generated_tests, ensure_ascii=True, indent=2)
    return f"""Write a concise semantic specification for the program below.

Describe its input contract, output contract, core computation, important boundary
behavior, and invariants demonstrated by the synthesized tests. Do not claim access
to public, private, or hidden benchmark tests.

Program ({language}):
```{language}
{slow_code}
```

Oracle-verified synthesized tests:
```json
{tests}
```

Return only the specification text, with no heading or code fence.
"""
