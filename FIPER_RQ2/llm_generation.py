"""Generate LLM candidates for the RQ2 limited-public-test pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FORBIDDEN_EVALUATION_TERMS = ("private", "hidden", "hide_io")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prompt", type=int, default=0)
    parser.add_argument("--baseline-df-path", required=True)
    parser.add_argument("--generated-df-path", required=True)
    parser.add_argument("--iteration-round", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--candidates-per-request", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--task-description", default="")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--language", choices=("python", "cpp"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_dataframe(path: str):
    import pandas as pd

    frame = pd.read_csv(path)
    forbidden_columns = [
        column
        for column in frame.columns
        if any(term in str(column).casefold() for term in FORBIDDEN_EVALUATION_TERMS)
    ]
    if forbidden_columns:
        frame = frame.drop(columns=forbidden_columns)
        print(
            "Dropped private evaluation columns before RQ2 generation: "
            + ", ".join(map(str, forbidden_columns))
        )
    return frame


def output_csv_path(value: str) -> Path:
    path = Path(value)
    return path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")


def choose_column(frame: Any, candidates: list[str], required: bool = False) -> str | None:
    columns = list(frame.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lowered = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise KeyError(f"None of these columns were found: {candidates}")
    return None


def find_io_columns(
    frame: Any,
    *,
    include_public: bool,
    representative_only: bool,
) -> list[str]:
    """Return only protocol-approved I/O columns for the requested stage."""

    columns = list(frame.columns)
    generated_preferences = (
        ("Representative_IO", "representative_tests")
        if representative_only
        else ("Selected_IO", "generated_tests", "Generated_IO")
    )
    selected = [column for column in generated_preferences if column in columns]
    if not selected and not representative_only:
        selected = [
            column
            for column in columns
            if str(column).startswith("Generated_IO_")
        ]
    if include_public:
        selected.extend(
            column
            for column in columns
            if "public" in str(column).casefold()
            and ("io_unit_tests" in str(column).casefold() or "test_cases" in str(column).casefold())
            and column not in selected
        )
    return selected


def infer_language(path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return "cpp" if "cpp" in path.lower() or "_c++" in path.lower() else "python"


def infer_model(path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    lower = path.lower()
    if "gemini" in lower:
        return "gemini-2.5-flash"
    if "gpt" in lower or "chatgpt" in lower:
        return "gpt-3.5-turbo-0125"
    if "codellama" in lower:
        return "codellama/CodeLlama-34b-Instruct-hf"
    return "deepseek-v3.2-exp"


def load_prompt_template(iteration_round: int) -> dict[str, str]:
    if iteration_round == -1:
        module_name = "generate_io_four_examples"
    elif iteration_round == 0:
        module_name = "generate_overview_description_cot"
    else:
        module_name = "generate_code_nl_cot_with_merged_io"
    module = importlib.import_module(f"prompt_templates.{module_name}")
    return module.prompt_dict


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def build_prompt(frame: Any, row_index: int, iteration_round: int, language: str, prompts: dict[str, str]) -> str:
    row = frame.iloc[row_index]
    slow_column = choose_column(
        frame,
        ["Selected_Optimized_Code", "Slow_Code", "slow_code", "Slow_program"],
        required=True,
    )
    slow_code = stringify(row[slow_column])
    code_label = "C++" if language == "cpp" else "Python"
    overview_column = choose_column(frame, ["Overview_Description", "overview_description", "Code_Function_Description"])
    overview = stringify(row[overview_column]) if overview_column else ""
    if iteration_round == -1:
        reference_column = choose_column(
            frame,
            ["4_Example_Prompt", "reference_examples", "Reference_Examples"],
        )
        reference_text = stringify(row[reference_column]) if reference_column else ""
        return prompts[f"instruction_{'Cpp' if language == 'cpp' else 'Python'}"].format(
            Slow_program=slow_code,
            four_example_prompt=reference_text or "No problem-disjoint retrieval examples are available.",
        )
    if iteration_round == 0:
        io_text = "\n\n".join(
            stringify(row[column])
            for column in find_io_columns(
                frame,
                include_public=False,
                representative_only=True,
            )
            if stringify(row[column])
        )
        return prompts[f"instruction_{'Cpp' if language == 'cpp' else 'Python'}"].format(
            Slow_program=slow_code,
            IO_examples=io_text or "No synthesized representative I/O examples are available.",
        )
    io_text = "\n\n".join(
        stringify(row[column])
        for column in find_io_columns(
            frame,
            include_public=True,
            representative_only=True,
        )
        if stringify(row[column])
    )
    return prompts[f"instruction_{'Cpp' if language == 'cpp' else 'Python'}"].format(
        Slow_program=slow_code,
        overview_description=overview or "No semantic overview is available.",
        IO_examples=io_text or "No I/O examples are available.",
        code_label=code_label,
    )


def call_model(
    model_name: str,
    system_prompt: str,
    prompt: str,
    output_prompt: bool,
    key_index: int,
    language: str,
    temperature: float,
) -> str:
    from single_generation import (
        ChatGPT_official_function,
        CodeLlama_serverstandard_inference_function,
        DeepSeek_official_function,
        Gemini_official_function,
        load_local_codellama,
    )

    normalized_model = model_name.lower()
    if normalized_model.startswith("gemini"):
        responses, _ = Gemini_official_function(
            key_index=key_index,
            model_name=model_name,
            system_prompt=system_prompt,
            question_text=prompt,
            output_prompt=output_prompt,
            temperature=temperature,
        )
        return responses[0]
    if normalized_model.startswith("gpt"):
        responses, _ = ChatGPT_official_function(
            key_index=key_index,
            model_name=model_name,
            system_prompt=system_prompt,
            question_text=prompt,
            output_prompt=output_prompt,
            temperature=temperature,
        )
        return responses[0]
    if "codellama" in normalized_model:
        tokenizer, model = _cached_codellama(model_name, load_local_codellama)
        responses = CodeLlama_serverstandard_inference_function(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            question_text=prompt,
            num_candidates=1,
            temperature=temperature,
            max_length=4096,
            output_prompt=output_prompt,
        )
        return responses[0]
    if normalized_model.startswith("deepseek"):
        responses, _, _, _ = DeepSeek_official_function(
            key_index=key_index,
            model_name=normalized_model,
            system_prompt=system_prompt,
            question_text=prompt,
            output_prompt=output_prompt,
            temperature=temperature,
        )
        return responses[0]
    raise ValueError(f"Unsupported model: {model_name}")


@lru_cache(maxsize=1)
def _cached_codellama(model_name: str, loader: Any) -> tuple[Any, Any]:
    return loader(model_name)


def dry_run_response(iteration_round: int, language: str) -> str:
    if iteration_round == -1:
        return json.dumps({"inputs": ["example input", "boundary input"]})
    if iteration_round == 0:
        return "The program transforms its input according to the supplied algorithm."
    return "pass" if language == "python" else "int main() { return 0; }"


def generate(args: argparse.Namespace) -> Path:
    frame = load_dataframe(args.baseline_df_path)
    language = infer_language(args.baseline_df_path, args.language)
    model_name = infer_model(args.baseline_df_path, args.model)
    if "codellama" in model_name.casefold() and args.workers != 1:
        raise ValueError("Local CodeLlama generation requires --workers 1.")
    prompts = load_prompt_template(args.iteration_round)
    total_candidates = args.candidates_per_request * args.repetitions
    if total_candidates != args.num_candidates:
        raise ValueError(
            "--num-candidates must equal --candidates-per-request * --repetitions."
        )
    def generate_row(row_index: int) -> list[str]:
        if args.iteration_round > 0 and _truthy(frame.iloc[row_index].get("Converged", False)):
            current_column = choose_column(
                frame,
                ["Selected_Optimized_Code", "Slow_Code", "slow_code", "Slow_program"],
                required=True,
            )
            current_code = stringify(frame.iloc[row_index][current_column])
            return [current_code] * args.num_candidates
        prompt = build_prompt(frame, row_index, args.iteration_round, language, prompts)
        system_prompt = prompts[f"system_prompt_{'Cpp' if language == 'cpp' else 'Python'}"]
        candidates = []
        for repetition in range(args.repetitions):
            candidates.append(
                dry_run_response(args.iteration_round, language)
                if args.dry_run
                else call_model(
                    model_name,
                    system_prompt,
                    prompt,
                    bool(args.output_prompt),
                    repetition,
                    language,
                    args.temperature,
                )
            )
        return candidates

    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        generated = [generate_row(row_index) for row_index in range(len(frame))]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            generated = list(executor.map(generate_row, range(len(frame))))

    if args.iteration_round == -1:
        for index in range(args.num_candidates):
            frame[f"Generated_IO_{index + 1}"] = [candidates[index] for candidates in generated]
        frame["Generated_IO"] = [json.dumps(candidates) for candidates in generated]
    elif args.iteration_round == 0:
        frame["Overview_Description"] = [candidates[0] for candidates in generated]
    else:
        for index in range(args.num_candidates):
            frame[f"Optimized_Code_{index + 1}"] = [candidates[index] for candidates in generated]
        frame["Optimized_Code"] = [candidates[0] for candidates in generated]

    destination = output_csv_path(args.generated_df_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    metadata = destination.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "baseline": str(Path(args.baseline_df_path).resolve()),
                "output": str(destination.resolve()),
                "iteration_round": args.iteration_round,
                "model": model_name,
                "language": language,
                "num_candidates": args.num_candidates,
                "temperature": args.temperature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    args = parse_args()
    if args.temperature is None:
        if args.iteration_round == -1:
            args.temperature = 1.0
        elif args.iteration_round == 0:
            args.temperature = 0.01
        else:
            args.temperature = 0.7 + 0.3 * (args.iteration_round - 1)
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    destination = generate(args)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
