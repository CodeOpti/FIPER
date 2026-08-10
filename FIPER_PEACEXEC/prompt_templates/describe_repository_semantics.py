"""Prompt used to derive a test-driven repository-level specification."""

from __future__ import annotations


SUPPORTED_LANGUAGES = {"python": "Python", "cpp": "C++"}


def system_prompt(language: str) -> str:
    """Return the semantic-description system prompt for ``language``."""

    display_name = SUPPORTED_LANGUAGES[language]
    return (
        "You are a senior software test engineer and technical documentation "
        f"specialist. Infer the behavior of the supplied {display_name} function "
        "from its source and project test code."
    )


def build_prompt(slow_code: str, test_code: str, language: str) -> str:
    """Build the repository-level test-driven semantic prompt."""

    display_name = SUPPORTED_LANGUAGES[language]
    fence = "python" if language == "python" else "cpp"
    return f"""Describe the behavior of the {display_name} function below.

Summarize its input and output contract, core computation, relevant project state,
important boundary conditions, and behavior exercised by the supplied tests. Be
concise and do not propose an optimization.

Function under optimization:
```{fence}
{slow_code}
```

Project test code:
```{fence}
{test_code}
```

Return only the semantic specification, without a heading or code fence.
"""
