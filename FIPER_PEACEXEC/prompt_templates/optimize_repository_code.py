"""Prompt used for repository-level progressive code optimization."""

from __future__ import annotations


SUPPORTED_LANGUAGES = {"python": "Python", "cpp": "C++"}


def system_prompt(language: str) -> str:
    """Return the optimization system prompt for ``language``."""

    display_name = SUPPORTED_LANGUAGES[language]
    return (
        "You are an expert software performance engineer. Preserve observable "
        f"behavior while optimizing the supplied {display_name} function."
    )


def build_prompt(
    current_code: str,
    test_code: str,
    semantic_description: str,
    language: str,
) -> str:
    """Build the function-aware optimization prompt used in each round."""

    display_name = SUPPORTED_LANGUAGES[language]
    fence = "python" if language == "python" else "cpp"
    return f"""Optimize the current {display_name} function for execution speed.

First reason about the function using the semantic specification and project tests.
Then produce a faster implementation. Preserve the function name, signature,
exceptions, side effects, and all behavior exercised by the project tests. Avoid
hard-coded answers, external services, new processes, and unrelated edits.

Semantic specification:
{semantic_description}

Project test code:
```{fence}
{test_code}
```

Current function (the validated dynamic base):
```{fence}
{current_code}
```

Return exactly one complete function definition in a single code fence.
"""
