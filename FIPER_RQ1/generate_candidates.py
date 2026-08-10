"""LLM generation stages for the RQ1 zero-public-test pipeline."""

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

from prompts import describe_semantics, generate_tests, optimize_code
from single_generation import (
    ChatGPT_official_function,
    CodeLlama_serverstandard_inference_function,
    DeepSeek_official_function,
    Gemini_official_function,
    load_local_codellama,
)


FORBIDDEN_EVALUATION_TERMS = ("public", "private", "hidden", "hide_io")
SCHEMA_ALIASES = {
    "Slow_Code": "slow_code",
    "Language": "language",
    "Problem_ID": "problem_id",
    "Reference_Examples": "reference_examples",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tests", "description", "optimize"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini", "deepseek", "codellama"),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", choices=("python", "cpp"), default="python")
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--show-prompt", action="store_true")
    return parser.parse_args()


def _text(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _load_dataframe(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    frame = pd.read_csv(input_path)
    for legacy_name, canonical_name in SCHEMA_ALIASES.items():
        if legacy_name not in frame.columns:
            continue
        if canonical_name in frame.columns:
            frame = frame.drop(columns=[legacy_name])
        else:
            frame = frame.rename(columns={legacy_name: canonical_name})
    if "slow_code" not in frame.columns:
        raise ValueError("Input CSV must contain a 'slow_code' column.")

    forbidden_columns = [
        column
        for column in frame.columns
        if any(term in column.casefold() for term in FORBIDDEN_EVALUATION_TERMS)
    ]
    if forbidden_columns:
        frame = frame.drop(columns=forbidden_columns)
        print(
            "Dropped evaluation-only columns to enforce the zero-public-test protocol: "
            + ", ".join(forbidden_columns)
        )
    return frame


def _atomic_write(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def _parse_json_object(response: str) -> dict[str, Any]:
    cleaned = response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model response did not contain a JSON object.")
        cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object.")
    return value


def _parse_test_inputs(response: str) -> str:
    value = _parse_json_object(response)
    inputs = value.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("The generated test object must contain an 'inputs' list.")
    normalized = [str(item) for item in inputs if str(item).strip()]
    if not normalized:
        raise ValueError("The model returned no usable test inputs.")
    return json.dumps({"inputs": normalized}, ensure_ascii=True)


def _parse_code(response: str) -> str:
    blocks = re.findall(r"```(?:python|cpp|c\+\+|c)?\s*\n?(.*?)```", response, re.DOTALL | re.IGNORECASE)
    code = max(blocks, key=len).strip() if blocks else response.strip()
    if not code:
        raise ValueError("The model returned an empty candidate.")
    return code


class ModelClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.local_tokenizer: Any | None = None
        self.local_model: Any | None = None
        if args.provider == "codellama":
            if args.threads != 1:
                raise ValueError("Local CodeLlama generation requires --threads 1.")
            self.local_tokenizer, self.local_model = load_local_codellama(args.model)

    def generate(self, system: str, question: str, key_index: int) -> str:
        args = self.args
        if args.provider == "openai":
            responses, _ = ChatGPT_official_function(
                key_index=key_index,
                model_name=args.model,
                system_prompt=system,
                question_text=question,
                num_candidates=1,
                temperature=args.temperature,
                max_length=args.max_tokens,
                return_logprobs=False,
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
                model=self.local_model,
                tokenizer=self.local_tokenizer,
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


def _with_retries(operation: Callable[[], str], retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Generation failed after {retries} attempts: {last_error}") from last_error


def _row_language(row: pd.Series, default: str) -> str:
    language = _text(row.get("language"), default).strip().casefold()
    aliases = {"py": "python", "python": "python", "c++": "cpp", "cpp": "cpp"}
    if language not in aliases:
        raise ValueError(f"Unsupported language value: {language!r}")
    return aliases[language]


def _load_tests(
    row: pd.Series, column: str = "generated_tests"
) -> dict[str, list[str]]:
    try:
        tests = json.loads(_text(row.get(column)))
    except json.JSONDecodeError as error:
        raise ValueError(f"{column!r} must contain strict JSON.") from error
    if not isinstance(tests, dict) or not isinstance(tests.get("inputs"), list):
        raise ValueError(f"{column!r} must contain input and output lists.")
    if not isinstance(tests.get("outputs"), list):
        raise ValueError(f"{column!r} must be oracle-verified before this stage.")
    return tests


def _generate_row(
    position: int,
    row: pd.Series,
    args: argparse.Namespace,
    client: ModelClient,
) -> dict[str, str]:
    language = _row_language(row, args.language)
    slow_code = _text(row["slow_code"]).strip()
    if not slow_code:
        raise ValueError(f"Row {position} has empty slow_code.")
    key_index = position % max(args.threads, 1)

    if args.stage == "optimize" and _truthy(row.get("converged", False)):
        current_code = _text(row.get("current_code"), slow_code).strip() or slow_code
        return {
            f"candidate_{candidate_number}": current_code
            for candidate_number in range(1, args.num_candidates + 1)
        }

    if args.stage == "tests":
        question = generate_tests.build_prompt(
            slow_code=slow_code,
            language=language,
            reference_examples=_text(row.get("reference_examples")),
        )
        response = _with_retries(
            lambda: client.generate(generate_tests.system_prompt(language), question, key_index),
            args.retries,
        )
        return {"generated_test_inputs": _parse_test_inputs(response)}

    tests = _load_tests(row, "representative_tests")
    if args.stage == "description":
        question = describe_semantics.build_prompt(slow_code, language, tests)
        response = _with_retries(
            lambda: client.generate(
                describe_semantics.system_prompt(language), question, key_index
            ),
            args.retries,
        )
        description = response.strip().strip("`").strip()
        if not description:
            raise ValueError("The model returned an empty semantic description.")
        return {"semantic_description": description}

    current_code = _text(row.get("current_code"), slow_code).strip() or slow_code
    semantic_description = _text(row.get("semantic_description")).strip()
    if not semantic_description:
        raise ValueError("Optimization requires a 'semantic_description' column.")
    question = optimize_code.build_prompt(
        current_code=current_code,
        language=language,
        semantic_description=semantic_description,
        generated_tests=tests,
    )
    candidates: dict[str, str] = {}
    for candidate_number in range(1, args.num_candidates + 1):
        response = _with_retries(
            lambda: client.generate(optimize_code.system_prompt(language), question, key_index),
            args.retries,
        )
        candidates[f"candidate_{candidate_number}"] = _parse_code(response)
    return candidates


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.num_candidates < 1 or args.retries < 1:
        raise ValueError("Thread, candidate, and retry counts must be positive.")
    frame = _load_dataframe(args.input)
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")
    if args.stage == "tests":
        frame["generated_test_inputs"] = ""
    elif args.stage == "description":
        frame["semantic_description"] = ""
    else:
        for candidate_number in range(1, args.num_candidates + 1):
            frame[f"candidate_{candidate_number}"] = ""
    client = ModelClient(args)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(_generate_row, position, row, args, client): position
            for position, (_, row) in enumerate(frame.iterrows())
        }
        for future in as_completed(futures):
            position = futures[future]
            generated_values = future.result()
            for column, value in generated_values.items():
                frame.at[frame.index[position], column] = value
            _atomic_write(frame, args.output)
            print(f"Completed {position + 1}/{len(frame)}")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
