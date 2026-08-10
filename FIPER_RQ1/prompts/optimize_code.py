"""Prompt construction for progressive code optimization."""

from __future__ import annotations

import json


def system_prompt(language: str) -> str:
    return (
        "You are an expert performance engineer. Preserve observable behavior while "
        f"optimizing the supplied {language} program."
    )


def build_prompt(
    current_code: str,
    language: str,
    semantic_description: str,
    generated_tests: dict,
) -> str:
    tests = json.dumps(generated_tests, ensure_ascii=True, indent=2)
    return f"""Optimize the current program for execution speed.

The semantic specification and tests below were synthesized without benchmark public,
private, or hidden tests. Preserve the standard-input/standard-output interface and all
observed behavior. Avoid external services, filesystem dependencies, subprocesses, and
hard-coded answers. Return exactly one complete program in a single code fence.

Semantic specification:
{semantic_description}

Oracle-verified synthesized tests:
```json
{tests}
```

Current program ({language}):
```{language}
{current_code}
```
"""
