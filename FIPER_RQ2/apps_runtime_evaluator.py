"""Compatibility wrapper for evaluating Python programs with structured I/O."""

from __future__ import annotations

import argparse
import json
from typing import Any

from pie_sandbox import run_code_io_tests_function


def run_python_io_tests_function(
    code_path: str,
    io_file: str,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    specified_cpu_core: int | None = None,
) -> dict[str, Any]:
    """Evaluate a Python source file; CPU pinning is intentionally not required."""
    del specified_cpu_core
    return run_code_io_tests_function(
        code_path,
        io_file,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        language="python",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code_path")
    parser.add_argument("io_file")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_python_io_tests_function(
        args.code_path,
        args.io_file,
        args.warmup_runs,
        args.measured_runs,
    )
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
