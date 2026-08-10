"""Run PEACEXEC project tests and parse their performance markers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


@dataclass(frozen=True)
class TestRun:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()


def extract_cpu_instructions(output: str) -> int | None:
    match = re.search(r"instruction_count=(\d+)", output)
    return int(match.group(1)) if match else None


def extract_elapsed_seconds(output: str) -> float | None:
    match = re.search(r"Elapsed\s*time:\s*([\d.]+)\s*seconds", output)
    return float(match.group(1)) if match else None


def extract_memory_mb(output: str) -> float | None:
    match = re.search(r"Memory usage \(MB\):\s*([\d.]+)", output)
    return float(match.group(1)) if match else None


def run_test(
    repository_path: str | Path,
    virtual_environment: str | Path,
    test_command: str,
    timeout_seconds: float,
) -> TestRun:
    """Run one trusted benchmark test command in an activated environment."""

    repository = Path(repository_path).resolve()
    environment = Path(virtual_environment).expanduser().resolve()
    activation_script = environment / "bin" / "activate"
    if not activation_script.is_file():
        raise FileNotFoundError(f"Virtual environment activation script missing: {activation_script}")
    bash = shutil.which("bash")
    if not bash:
        raise RuntimeError("PEACEXEC evaluation requires Bash on Linux.")

    script = (
        "set -o pipefail; "
        f"source {shlex.quote(str(activation_script))}; "
        f"export PYTHONPATH={shlex.quote(str(repository))}:\"${{PYTHONPATH:-}}\"; "
        f"{test_command}"
    )
    completed = subprocess.run(
        [bash, "-lc", script],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        env=os.environ.copy(),
        check=False,
    )
    return TestRun(completed.returncode, completed.stdout, completed.stderr)
