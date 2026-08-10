"""Isolated repository-level correctness and performance evaluation."""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any, Mapping

from function_replacer import replace_function_body
from git_checkout import detached_worktree
from test_performance_extractor import (
    extract_cpu_instructions,
    extract_elapsed_seconds,
    extract_memory_mb,
    run_test,
)


def _resolve_under(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def evaluate_candidate(
    case: Mapping[str, Any],
    candidate_code: str | None,
    repository_root: str | Path,
    virtual_environment_root: str | Path,
    warmup_runs: int = 1,
    measured_runs: int = 25,
    timeout_seconds: float = 120.0,
    temporary_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate in an isolated detached worktree."""

    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive.")
    repository_base = Path(repository_root).expanduser().resolve()
    environment_base = Path(virtual_environment_root).expanduser().resolve()
    repository = _resolve_under(repository_base, case["repo_path"])
    environment = _resolve_under(environment_base, case["venv_path"])

    result: dict[str, Any] = {
        "status": "error",
        "pass_rate": 0.0,
        "runtime_ms": None,
        "cpu_instructions": None,
        "memory_mb": None,
        "error": "",
    }
    try:
        with detached_worktree(repository, str(case["sha"]), temporary_root) as worktree:
            target_file = (worktree / str(case["target_file"])).resolve()
            if worktree not in target_file.parents:
                raise ValueError("target_file escapes the temporary worktree.")
            if candidate_code:
                replace_function_body(
                    target_file,
                    str(case["target_func"]),
                    candidate_code,
                    None
                    if case.get("target_class") in (None, "", "Null")
                    else str(case["target_class"]),
                )

            for _ in range(warmup_runs):
                warmup = run_test(
                    worktree, environment, str(case["test_cmd"]), timeout_seconds
                )
                if warmup.returncode != 0:
                    result["error"] = warmup.combined_output[-4000:]
                    return result

            elapsed_values: list[float] = []
            instruction_values: list[int] = []
            memory_values: list[float] = []
            for _ in range(measured_runs):
                run = run_test(
                    worktree, environment, str(case["test_cmd"]), timeout_seconds
                )
                if run.returncode != 0:
                    result["error"] = run.combined_output[-4000:]
                    return result
                elapsed = extract_elapsed_seconds(run.combined_output)
                instructions = extract_cpu_instructions(run.combined_output)
                memory = extract_memory_mb(run.combined_output)
                if elapsed is not None:
                    elapsed_values.append(elapsed * 1000.0)
                if instructions is not None:
                    instruction_values.append(instructions)
                if memory is not None:
                    memory_values.append(memory)

            if not elapsed_values:
                result["error"] = (
                    "The test passed but emitted no 'Elapsed time: <seconds> seconds' marker."
                )
                return result
            result.update(
                status="ok",
                pass_rate=1.0,
                runtime_ms=statistics.mean(elapsed_values),
                cpu_instructions=(
                    statistics.mean(instruction_values) if instruction_values else None
                ),
                memory_mb=statistics.mean(memory_values) if memory_values else None,
                error="",
            )
            return result
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result
