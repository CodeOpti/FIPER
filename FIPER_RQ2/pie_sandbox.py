"""Run generated Python or C++ programs against structured I/O cases."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _load_io(io_source: str | dict[str, Any]) -> dict[str, list[str]]:
    if isinstance(io_source, dict):
        record = io_source
    else:
        text = Path(io_source).read_text(encoding="utf-8")
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            record = ast.literal_eval(text)
    if not isinstance(record, dict):
        raise ValueError("I/O data must be a dictionary")
    inputs = record.get("inputs", [])
    outputs = record.get("outputs", [])
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("I/O data must contain list-valued 'inputs' and 'outputs'")
    if len(inputs) != len(outputs):
        raise ValueError("The input and output lists must have the same length")
    return {"inputs": [str(item) for item in inputs], "outputs": [str(item) for item in outputs]}


def _same_output(actual: str, expected: str) -> bool:
    return actual.strip() == expected.strip()


def _command_for_source(source_path: Path, language: str, build_dir: Path) -> list[str]:
    normalized_language = language.lower()
    if normalized_language in {"python", "py"}:
        return [sys.executable, str(source_path)]
    if normalized_language not in {"cpp", "c++", "cc"}:
        raise ValueError(f"Unsupported language: {language}")
    executable = build_dir / ("program.exe" if os.name == "nt" else "program")
    result = subprocess.run(
        ["g++", "-O3", "-std=c++20", str(source_path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "C++ compilation failed")
    return [str(executable)]


def capture_oracle_outputs(
    code: str,
    inputs: list[str],
    language: str = "python",
    timeout_seconds: float = 10.0,
) -> dict[str, list[Any]]:
    """Label synthesized inputs by executing the original slow program once."""

    valid_inputs: list[str] = []
    outputs: list[str] = []
    runtimes_us: list[float] = []
    errors: list[str] = []
    suffix = ".cpp" if language.lower() in {"cpp", "c++", "cc"} else ".py"
    with tempfile.TemporaryDirectory(prefix="fiper_rq2_oracle_") as temporary_dir:
        build_dir = Path(temporary_dir)
        source_path = build_dir / f"oracle{suffix}"
        source_path.write_text(code.strip() + "\n", encoding="utf-8")
        try:
            command = _command_for_source(source_path, language, build_dir)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            return {
                "inputs": [],
                "outputs": [],
                "runtime_us": [],
                "errors": [str(error)],
            }

        for input_text in inputs:
            started = time.perf_counter()
            try:
                result = subprocess.run(
                    command,
                    input=str(input_text),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                elapsed_us = (time.perf_counter() - started) * 1_000_000
            except subprocess.TimeoutExpired:
                errors.append("Oracle execution timed out.")
                continue
            if result.returncode != 0:
                errors.append(result.stderr.strip() or f"Oracle exited with code {result.returncode}.")
                continue
            valid_inputs.append(str(input_text))
            outputs.append(result.stdout)
            runtimes_us.append(elapsed_us)
    return {
        "inputs": valid_inputs,
        "outputs": outputs,
        "runtime_us": runtimes_us,
        "errors": errors,
    }


def run_code_io_tests_function(
    codepath: str,
    io_dict: str | dict[str, Any] | None = None,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    language: str = "python",
    timeout_seconds: float = 10.0,
    **legacy_options: Any,
) -> dict[str, Any]:
    """Execute a program for every I/O case and return pass rates and timings."""
    io_source = io_dict if io_dict is not None else legacy_options.get("io_file")
    if io_source is None:
        raise ValueError("An I/O dictionary or file path is required")
    warmup_runs = int(legacy_options.get("legacy_warmup_runs", warmup_runs))
    measured_runs = int(legacy_options.get("legacy_measured_runs", measured_runs))
    language = str(legacy_options.get("legacy_language", language))
    io_data = _load_io(io_source)
    if measured_runs < 1:
        raise ValueError("measured_runs must be at least 1")

    pass_results: list[int] = []
    timing_us: list[list[float]] = []
    outputs: list[str] = []
    errors: list[str | None] = []

    with tempfile.TemporaryDirectory(prefix="generated_program_") as temporary_dir:
        build_dir = Path(temporary_dir)
        command = _command_for_source(Path(codepath), language, build_dir)
        for input_text, expected_output in zip(io_data["inputs"], io_data["outputs"]):
            for _ in range(warmup_runs):
                subprocess.run(
                    command,
                    input=input_text,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )

            case_times: list[float] = []
            case_output = ""
            case_error: str | None = None
            passed = False
            for _ in range(measured_runs):
                start = time.perf_counter()
                try:
                    result = subprocess.run(
                        command,
                        input=input_text,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        check=False,
                    )
                    elapsed = (time.perf_counter() - start) * 1_000_000
                    case_times.append(round(elapsed, 2))
                    case_output = result.stdout
                    if result.returncode != 0:
                        case_error = result.stderr.strip() or f"Process exited with code {result.returncode}"
                    else:
                        passed = _same_output(case_output, expected_output)
                        if not passed:
                            case_error = "Output mismatch"
                except subprocess.TimeoutExpired:
                    case_times.append(round(timeout_seconds * 1_000_000, 2))
                    case_error = "Timeout"
            pass_results.append(int(passed))
            timing_us.append(case_times)
            outputs.append(case_output)
            errors.append(case_error)

    flat_times = [value for values in timing_us for value in values]
    return {
        "io_passresult": pass_results,
        "execution_times_us": timing_us,
        "mean_time_us": sum(flat_times) / len(flat_times) if flat_times else 0.0,
        "testcodeIOoutput": outputs,
        "error_type": errors,
        "timing_unit": "microseconds",
    }


def run_python_cpp_io_tests_function(
    codepath: str,
    io_file: str | dict[str, Any],
    warmup_runs: int = 0,
    measured_runs: int = 1,
    language: str = "python",
    pie_index: str = "0",
    timeout_seconds: float = 10.0,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Compatibility entry point for callers that use the historical name."""
    del pie_index
    result = run_code_io_tests_function(
        codepath,
        io_file,
        warmup_runs,
        measured_runs,
        language,
        timeout_seconds,
    )
    if not result["io_passresult"] and not allow_empty:
        result["error_type"] = ["No test cases supplied"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code_path")
    parser.add_argument("io_file")
    parser.add_argument("--language", choices=["python", "cpp"], default="python")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_code_io_tests_function(
        args.code_path,
        args.io_file,
        args.warmup_runs,
        args.measured_runs,
        args.language,
        args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
