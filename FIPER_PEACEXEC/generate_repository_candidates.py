"""Generate PEACEXEC semantic descriptions or optimization candidates.

The input is a UTF-8 CSV with English column names. The script uses only the
official-provider adapters in ``single_generation.py`` and can perform a fully
offline dry run for artifact checks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import pandas as pd

from prompt_templates import describe_repository_semantics, optimize_repository_code
from single_generation import (
    ChatGPT_official_function,
    CodeLlama_serverstandard_inference_function,
    DeepSeek_official_function,
    Gemini_official_function,
    load_local_codellama,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("description", "optimize"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini", "deepseek", "codellama"),
        default="deepseek",
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--language", choices=("python", "cpp"), default="python")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _text(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _parse_code(response: str) -> str:
    blocks = re.findall(
        r"```(?:python|cpp|c\+\+|c)?\s*\n?(.*?)```",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    code = max(blocks, key=len).strip() if blocks else response.strip()
    if not code:
        raise ValueError("The model returned an empty candidate.")
    return code


def _with_retries(operation: Callable[[], str], retries: int) -> str:
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as current_error:
            error = current_error
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Generation failed after {retries} attempts: {error}") from error


class ModelClient:
    """Small uniform wrapper around the repository's provider adapters."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        if args.provider == "codellama" and not args.dry_run:
            if args.threads != 1:
                raise ValueError("Local CodeLlama generation requires --threads 1.")
            self.tokenizer, self.model = load_local_codellama(args.model)

    def generate(self, system: str, question: str, key_index: int) -> str:
        args = self.args
        if args.dry_run:
            if args.stage == "description":
                return "Dry-run semantic specification derived from the function and tests."
            return _text(self._dry_run_code(question))

        if args.provider == "openai":
            responses, _ = ChatGPT_official_function(
                key_index=key_index,
                model_name=args.model,
                system_prompt=system,
                question_text=question,
                num_candidates=1,
                temperature=args.temperature,
                max_length=args.max_tokens,
                output_prompt=args.show_prompt,
            )
        elif args.provider == "gemini":
            responses, _ = Gemini_official_function(
                key_index=key_index,
                model_name=args.model,
                system_prompt=system,
                question_text=question,
                num_candidates=1,
                temperature=args.temperature,
                max_length=args.max_tokens,
                output_prompt=args.show_prompt,
            )
        elif args.provider == "deepseek":
            responses, _, _, _ = DeepSeek_official_function(
                model_name=args.model,
                key_index=key_index,
                system_prompt=system,
                question_text=question,
                temperature=args.temperature,
                max_length=args.max_tokens,
                output_prompt=args.show_prompt,
            )
        else:
            responses = CodeLlama_serverstandard_inference_function(
                model=self.model,
                tokenizer=self.tokenizer,
                system_prompt=system,
                question_text=question,
                num_candidates=1,
                temperature=args.temperature,
                max_length=args.max_tokens,
                output_prompt=args.show_prompt,
            )
        if not responses:
            raise RuntimeError("The model provider returned no response.")
        return responses[0]

    @staticmethod
    def _dry_run_code(question: str) -> str:
        match = re.search(
            r"Current function \(the validated dynamic base\):\s*```[^\n]*\n(.*?)```",
            question,
            flags=re.DOTALL,
        )
        return match.group(1).strip() if match else "def dry_run_candidate():\n    pass"


def _validate_frame(frame: pd.DataFrame, stage: str) -> None:
    required = {"slow_code", "test_code"}
    if stage == "optimize":
        required.add("semantic_description")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")


def _generate_row(
    position: int,
    row: pd.Series,
    args: argparse.Namespace,
    client: ModelClient,
) -> dict[str, str]:
    language = _text(row.get("language"), args.language).strip().casefold()
    language = {"py": "python", "c++": "cpp"}.get(language, language)
    if language not in {"python", "cpp"}:
        raise ValueError(f"Unsupported language in row {position}: {language!r}")

    slow_code = _text(row["slow_code"]).strip()
    test_code = _text(row["test_code"]).strip()
    if not slow_code or not test_code:
        raise ValueError(f"Row {position} has empty slow_code or test_code.")
    key_index = position % max(args.threads, 1)

    if args.stage == "description":
        question = describe_repository_semantics.build_prompt(
            slow_code, test_code, language
        )
        response = _with_retries(
            lambda: client.generate(
                describe_repository_semantics.system_prompt(language),
                question,
                key_index,
            ),
            args.retries,
        )
        description = response.strip().strip("`").strip()
        if not description:
            raise ValueError("The model returned an empty semantic description.")
        return {"semantic_description": description}

    current_code = _text(row.get("current_code"), slow_code).strip() or slow_code
    question = optimize_repository_code.build_prompt(
        current_code=current_code,
        test_code=test_code,
        semantic_description=_text(row["semantic_description"]).strip(),
        language=language,
    )
    candidates: dict[str, str] = {}
    for candidate_number in range(1, args.num_candidates + 1):
        response = _with_retries(
            lambda: client.generate(
                optimize_repository_code.system_prompt(language),
                question,
                key_index,
            ),
            args.retries,
        )
        candidates[f"candidate_{candidate_number}"] = _parse_code(response)
    return candidates


def main() -> None:
    args = parse_args()
    if args.num_candidates < 1 or args.threads < 1 or args.retries < 1:
        raise ValueError("Candidate, thread, and retry counts must be positive.")
    if args.temperature is None:
        args.temperature = 0.01 if args.stage == "description" else 0.7
    if args.stage == "description":
        args.num_candidates = 1
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {args.input}")

    frame = pd.read_csv(args.input)
    _validate_frame(frame, args.stage)
    output_columns = (
        ["semantic_description"]
        if args.stage == "description"
        else [f"candidate_{index}" for index in range(1, args.num_candidates + 1)]
    )
    for column in output_columns:
        frame[column] = ""

    client = ModelClient(args)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(_generate_row, position, row, args, client): position
            for position, (_, row) in enumerate(frame.iterrows())
        }
        for future in as_completed(futures):
            position = futures[future]
            for column, value in future.result().items():
                frame.at[frame.index[position], column] = value
            _atomic_write(frame, args.output)
            print(f"Completed {position + 1}/{len(frame)}")

    metadata = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "stage": args.stage,
        "provider": "dry-run" if args.dry_run else args.provider,
        "model": args.model,
        "temperature": args.temperature,
        "num_candidates": args.num_candidates,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
