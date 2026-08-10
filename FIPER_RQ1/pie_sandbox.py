"""Subprocess runner for synthesized-test validation and profiling.

The runner is intentionally independent of benchmark public/private test storage. It
accepts an explicit in-memory test dictionary or JSON file supplied by the caller.
For untrusted generated code, run the repository inside an OS-level container as well.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


FAILURE_TIME_MS = 1_000_000_000.0


def _limit_posix_resources(timeout_seconds: float, memory_limit_mb: int) -> None:
    if os.name != "posix":
        return
    import resource

    cpu_seconds = max(1, math.ceil(timeout_seconds) + 1)
    memory_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def compile_cpp_code(
    code_path: str | Path,
    output_path: str | Path | None = None,
    cflags: str = "-std=c++20 -O3",
    pie_index: str = "0",
) -> str:
    """Compile a C++ source file and return the executable path."""

    source = Path(code_path)
    if output_path is None:
        suffix = ".exe" if os.name == "nt" else ""
        output_path = source.with_name(f"candidate_{pie_index}{suffix}")
    executable = Path(output_path)
    command = ["g++", *shlex.split(cflags), str(source), "-o", str(executable)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "C++ compilation failed.")
    return str(executable)


def run_program_once(
    command: Sequence[str],
    input_text: str,
    timeout_seconds: float,
    working_directory: str | Path,
    memory_limit_mb: int = 1536,
) -> dict[str, Any]:
    """Execute one standard-input case and capture output and wall-clock time."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            input=input_text,
            capture_output=True,
            text=True,
            cwd=working_directory,
            timeout=timeout_seconds,
            start_new_session=True,
            preexec_fn=(
                (lambda: _limit_posix_resources(timeout_seconds, memory_limit_mb))
                if os.name == "posix"
                else None
            ),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "runtime_ms": elapsed_ms,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "return_code": -1,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "execution timed out",
            "runtime_ms": FAILURE_TIME_MS,
            "timed_out": True,
        }


def _tokens_match(actual: str, expected: str) -> bool:
    actual_tokens = actual.split()
    expected_tokens = expected.split()
    if actual_tokens == expected_tokens:
        return True
    if len(actual_tokens) != len(expected_tokens):
        return False
    for actual_token, expected_token in zip(actual_tokens, expected_tokens):
        try:
            if not math.isclose(
                float(actual_token),
                float(expected_token),
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                return False
        except ValueError:
            return False
    return True


def _prepare_program(code: str, language: str, directory: Path) -> list[str]:
    if language == "python":
        code_path = directory / "candidate.py"
        code_path.write_text(code.strip() + "\n", encoding="utf-8")
        return [sys.executable, str(code_path)]
    if language == "cpp":
        code_path = directory / "candidate.cpp"
        code_path.write_text(code.strip() + "\n", encoding="utf-8")
        executable = compile_cpp_code(code_path, directory / ("candidate.exe" if os.name == "nt" else "candidate"))
        return [executable]
    raise ValueError(f"Unsupported language: {language}")


def evaluate_code(
    code: str,
    test_cases: dict[str, list[str]],
    language: str,
    warmup_runs: int = 1,
    measured_runs: int = 5,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Validate and profile code against oracle-verified synthesized tests."""

    inputs = test_cases.get("inputs")
    outputs = test_cases.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("Test cases must contain 'inputs' and 'outputs' lists.")
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("Synthesized test inputs and outputs must be non-empty and aligned.")
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive.")

    passes: list[int] = []
    runtime_ms: list[list[float]] = []
    captured_outputs: list[str] = []
    errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="fiper_eval_") as temporary_directory:
        directory = Path(temporary_directory)
        try:
            command = _prepare_program(code, language, directory)
        except Exception as error:
            return {
                "test_passes": [0] * len(inputs),
                "runtime_ms": [[FAILURE_TIME_MS]] * len(inputs),
                "captured_outputs": [""] * len(inputs),
                "errors": [f"preparation failed: {error}"],
            }

        for input_text, expected_output in zip(inputs, outputs):
            case_times: list[float] = []
            result: dict[str, Any] | None = None
            for _ in range(warmup_runs):
                result = run_program_once(command, str(input_text), timeout_seconds, directory)
                if result["return_code"] != 0:
                    break
            if result is None or result["return_code"] == 0:
                for _ in range(measured_runs):
                    result = run_program_once(command, str(input_text), timeout_seconds, directory)
                    case_times.append(float(result["runtime_ms"]))
                    if result["return_code"] != 0:
                        break

            assert result is not None
            passed = result["return_code"] == 0 and _tokens_match(
                str(result["stdout"]), str(expected_output)
            )
            passes.append(int(passed))
            runtime_ms.append(case_times or [FAILURE_TIME_MS])
            captured_outputs.append(str(result["stdout"]))
            if not passed:
                errors.append(str(result["stderr"]).strip() or "output mismatch")

    return {
        "test_passes": passes,
        "runtime_ms": runtime_ms,
        "captured_outputs": captured_outputs,
        "errors": errors,
    }


def capture_oracle_outputs(
    code: str,
    inputs: list[str],
    language: str,
    timeout_seconds: float = 2.0,
) -> dict[str, list[Any]]:
    """Execute the original slow program to label synthesized inputs."""

    valid_inputs: list[str] = []
    outputs: list[str] = []
    runtime_ms: list[float] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fiper_oracle_") as temporary_directory:
        directory = Path(temporary_directory)
        try:
            command = _prepare_program(code, language, directory)
        except Exception as error:
            return {"inputs": [], "outputs": [], "runtime_ms": [], "errors": [str(error)]}
        for input_text in inputs:
            result = run_program_once(command, str(input_text), timeout_seconds, directory)
            if result["return_code"] == 0:
                valid_inputs.append(str(input_text))
                outputs.append(str(result["stdout"]))
                runtime_ms.append(float(result["runtime_ms"]))
            else:
                errors.append(str(result["stderr"]).strip() or "oracle execution failed")
    return {
        "inputs": valid_inputs,
        "outputs": outputs,
        "runtime_ms": runtime_ms,
        "errors": errors,
    }


def run_python_cpp_io_tests_in_subprocess(
    candidate_path: str,
    test_data: str,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    language: str = "python",
    pie_index: str = "0",
    timeout_seconds: float = 2.0,
    measure_memory: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper used by earlier scripts."""

    del pie_index, measure_memory
    code = Path(candidate_path).read_text(encoding="utf-8")
    tests = json.loads(Path(test_data).read_text(encoding="utf-8"))
    return evaluate_code(code, tests, language, warmup_runs, measured_runs, timeout_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one program file.")
    parser.add_argument("candidate_path", type=Path)
    parser.add_argument("test_data", type=Path)
    parser.add_argument("--language", choices=("python", "cpp"), default="python")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.candidate_path.read_text(encoding="utf-8")
    tests = json.loads(args.test_data.read_text(encoding="utf-8"))
    result = evaluate_code(
        code,
        tests,
        args.language,
        args.warmup_runs,
        args.measured_runs,
        args.timeout,
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
