"""Evaluate model-generated programs for functional correctness and runtime."""





import pathlib
import tempfile
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import os
import logging
import glob
import numpy as np
from collections import defaultdict


from evalconfig import EvaluationConfig
from sandbox_o import run_code_on_inputs

import pdb


logging.basicConfig(level=logging.CRITICAL)


lang2file_ending = {
    "python": "py",
    "cpp": "cpp"
}

def evaluate_generated_outputs(cfg: EvaluationConfig) -> None:
    """Evaluate generated programs for accuracy and runtime."""



    merged = read_inputs_and_prepare(cfg)
    print(f"Programs to evaluate: {len(merged)}")
    print(f"Trials per program (num_trials): {cfg.num_trials}")
    print(f"Discarded warm-up runs: {cfg.ignore_first_k}")
    print(f"Maximum time per run: {cfg.max_time_per_run}")
    print(f"Slow-code column: {cfg.slow_code_col}")
    print(f"Reference-code column: {cfg.reference_code_col}")
    print(f"Generated-code column: {cfg.model_generated_potentially_faster_code_col}")
    print(f"Input/output test base path: {cfg.inputs_outputs_basepath}")


    problem_id_to_ground_truths, output_code_location = write_programs_read_ground_truth(
        cfg, merged
    )




    lang_file_ending = lang2file_ending[cfg.language]



    tag_to_path = [
        ("input", f"_slow.{lang_file_ending}"),
        ("reference", f"_reference.{lang_file_ending}"),
    ]


    is_multigen = isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], list)
    if is_multigen:

        num_generations = len(merged[cfg.model_generated_potentially_faster_code_col].iloc[0])
        tag_to_path.extend([(f"{cfg.model_generated_potentially_faster_code_col}_{i}", f"_maybe_faster_{i}.{lang_file_ending}") for i in range(num_generations)])
    else:

        tag_to_path.append((cfg.model_generated_potentially_faster_code_col, f"_maybe_faster_0.{lang_file_ending}"))


    results = run_programs(cfg, merged, problem_id_to_ground_truths, output_code_location, tag_to_path)


    if is_multigen:
        results = get_best_generation_per_submission(results, gen_col=cfg.model_generated_potentially_faster_code_col)


    print_summary(cfg, merged, results, gen_col=cfg.model_generated_potentially_faster_code_col)


    if isinstance(cfg.temp_dir, tempfile.TemporaryDirectory):
        cfg.temp_dir.cleanup()


def read_inputs_and_prepare(cfg) -> pd.DataFrame:
    """Read generated outputs and reference data, then return their merged frame."""
    print(f"Reading {cfg.reference_file_path} as the reference file")
    print(f"Reading {cfg.model_generated_outputs_path} as generated outputs")

    print(
        f"Run each program {cfg.num_trials} times, discard the first {cfg.ignore_first_k} runs, and load input/output pairs from {cfg.inputs_outputs_basepath}"
    )


    gen_df = pd.read_json(
        cfg.model_generated_outputs_path, lines=True, orient="records"
    )

    if cfg.model_generated_outputs_path.endswith(".report"):
        return _prepare_for_rerun(gen_df, cfg)

    print(f"Read {len(gen_df)} rows from generated outputs")


    if cfg.is_prompt_based:
        gen_df["slower_program"] = gen_df.apply(
            lambda x: get_input_from_prompt(x), axis=1
        )
    else:

        gen_df["slower_program"] = gen_df[cfg.slow_code_col].apply(lambda x: x.strip())


    if cfg.reference_file_path is not None:

        ref_df = pd.read_json(cfg.reference_file_path, lines=True, orient="records")
        ref_df["slower_program"] = ref_df["input"].apply(
            lambda x: x.strip().replace("\n\n\n\n\n", "")

        )

        print(f"Unique inputs in the reference file: {len(ref_df['slower_program'].unique())}")
        gen_df["slower_program"] = gen_df[
            "slower_program"
        ].apply(lambda x: x.strip().replace("\n\n\n\n\n", ""))


        assert len(ref_df["submission_id_v0"].unique()) == len(
            ref_df
        ), "submission_id_v0 must be unique"


        merged = pd.merge(
            gen_df,
            ref_df,
            left_on="slower_program",
            right_on="slower_program",
            suffixes=("", "_ref"),
            how="inner",
        )


        merged = merged.drop_duplicates(subset=["slower_program"])


        assert abs(len(merged) - len(gen_df)) < 10, f"The merge dropped too many rows; verify that both inputs use the same code. Dropped {len(gen_df) - len(merged)} rows. len(gen_df)={len(gen_df)}, len(merged)={len(merged)}"
    else:

        assert (
            cfg.reference_code_col in gen_df.columns
        ), f"Column {cfg.model_generated_outputs_path} was not found in {cfg.reference_code_col}"
        merged = gen_df


        merged = merged[merged[cfg.slow_code_col] != merged[cfg.reference_code_col]]

    assert (
        len(merged) > 0
    ), f"No programs remain because {cfg.slow_code_col} and {cfg.reference_code_col} are identical for every row."


    if cfg.num_problems_to_evaluate != -1:
        merged = merged[: cfg.num_problems_to_evaluate]



    if isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], list):
        num_generations = len(merged[cfg.model_generated_potentially_faster_code_col].iloc[0])
        for i in range(num_generations):
            merged[f"{cfg.model_generated_potentially_faster_code_col}_{i}"] = merged[cfg.model_generated_potentially_faster_code_col].apply(lambda x: x[i])
    return merged


def _prepare_for_rerun(df: pd.DataFrame, cfg: EvaluationConfig) -> pd.DataFrame:
    """Resume an interrupted evaluation by excluding rows already evaluated successfully."""
    acc_columns = {"generated_answers_acc", "generated_answer_acc"}

    acc_column = list(acc_columns.intersection(set(df.columns)))[0]
    print("Warning: resuming a previous run")
    print("Preparing the resumed run...")
    print(f"Found accuracy column: {acc_column}, total {len(df)} rows from generated outputs")


    df = df[df[acc_column] > 0.99]

    df = df[[c for c in df.columns if not any(x in c for x in ["mean", "std", "acc"])]]
    print(f"Rows remaining after filtering: {len(df)} rows from generated outputs")
    if cfg.num_problems_to_evaluate != -1:
        df = df[: cfg.num_problems_to_evaluate]
    return df


def write_programs_read_ground_truth(
    cfg: EvaluationConfig, merged: pd.DataFrame
) -> Tuple[Dict[str, List[str]], str]:
    """Write programs to a temporary directory and load their ground-truth outputs."""
    problem_id_to_ground_truths = defaultdict(list)


    if cfg.temp_dir is None:
        cfg.temp_dir = tempfile.TemporaryDirectory()
        output_code_location = cfg.temp_dir.name
    else:
        output_code_location = cfg.temp_dir
        pathlib.Path(output_code_location).mkdir(parents=True, exist_ok=True)


    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="writing programs"):
        problem_id = row["problem_id"]


        if problem_id not in problem_id_to_ground_truths:
            num_test_cases = len(
                glob.glob(f"{cfg.inputs_outputs_basepath}/{problem_id}/output*.txt")
            )
            assert (
                num_test_cases > 0
            ), f"{cfg.inputs_outputs_basepath}/{problem_id} does not contain ground-truth files."


            for i in range(num_test_cases):
                with open(f"{cfg.inputs_outputs_basepath}/{problem_id}/output.{i}.txt") as f:
                    problem_id_to_ground_truths[problem_id].append(f.read().strip() + "\n")


        lang_file_ending = lang2file_ending[cfg.language]
        submission_id_v0 = row["submission_id_v0"]


        with open(
            os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_slow.{lang_file_ending}"), "w"
        ) as f:
            f.write(row["slower_program"])


        generated_programs = row[cfg.model_generated_potentially_faster_code_col]
        if isinstance(generated_programs, str):
            generated_programs = [generated_programs]

        for i, generated_program in enumerate(generated_programs):
            with open(
                os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_maybe_faster_{i}.{lang_file_ending}"),
                "w"
            ) as f:
                f.write(generated_program.strip())


        with open(
            os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_reference.{lang_file_ending}"), "w"
        ) as f:
            f.write(row[cfg.reference_code_col].strip())

    print(f"All programs were written to {output_code_location}")
    return problem_id_to_ground_truths, output_code_location


def run_programs(
    cfg: EvaluationConfig,
    merged: pd.DataFrame,
    problem_id_to_ground_truths: Dict,
    output_code_location: str,
    tag_to_path
):
    """Execute programs in the sandbox and collect correctness and runtime metrics."""

    results = dict()

    assert len(merged["submission_id_v0"].unique()) == len(
        merged
    ), f"Every row must have a unique submission_id_v0. Unique count: {len(merged['submission_id_v0'].unique())}, total rows: {len(merged)}"


    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="running programs"):
        problem_id = row["problem_id"]
        submission_id_v0 = row["submission_id_v0"]
        unit_test_data_basepath = f"{cfg.inputs_outputs_basepath}/{problem_id}"

        try:
            problem_execution_stats = dict()

            for (tag, suffix) in tag_to_path:
                code_path = os.path.join(
                    output_code_location, f"{submission_id_v0}_{problem_id}{suffix}"
                )


                avg_time, std_time, avg_acc = run_code_on_inputs(  # type: ignore
                    language=cfg.language,
                    code_path=code_path,
                    ground_truths=problem_id_to_ground_truths[problem_id],
                    unit_test_data_basepath=unit_test_data_basepath,
                    num_runs_per_test_case=cfg.num_trials,
                    ignore_first_k=cfg.ignore_first_k,
                    max_seconds_per_run=cfg.max_time_per_run,
                    cpu_number=cfg.cpu_number,
                    cflags=cfg.cflags,
                    return_if_acc_below=cfg.return_if_acc_below,
                )


                problem_execution_stats.update(
                    {
                        f"{tag}_time_mean": avg_time,
                        f"{tag}_time_std": std_time,
                        f"{tag}_acc": avg_acc,
                    }
                )
            results[submission_id_v0] = problem_execution_stats

        except Exception as e:

            logging.error(e)
            tmp = dict()
            for tag, suffix in tag_to_path:
                tmp[f"{tag}_time_mean"] = np.nan
                tmp[f"{tag}_time_std"] = np.nan
                tmp[f"{tag}_acc"] = 0.0
            results[submission_id_v0] = tmp
            continue

    print(f"Successfully evaluated {len(results)} problems")
    return results


def get_best_generation_per_submission(results: Dict, gen_col: str):
    """Select the fastest fully correct candidate for each submission."""
    best_per_sub = dict()
    for submission_id_v0, result_dict in results.items():

        gen_op_times = [(k, v) for k, v in result_dict.items() if gen_col in k and "time_mean" in k]

        gen_op_times = sorted(gen_op_times, key=lambda x: x[1])


        for gen_op_time in gen_op_times:

            if result_dict[f"{gen_op_time[0].replace('_time_mean', '')}_acc"] == 1.0:
                gen_op_times = [gen_op_time]
                break


        try:
            best_gen_key = gen_op_times[0][0].replace("_time_mean", "")
            best_per_sub[submission_id_v0] = result_dict
            best_per_sub[submission_id_v0][f"{gen_col}_time_mean"] = gen_op_times[0][1]
            best_per_sub[submission_id_v0][f"{gen_col}_time_std"] = result_dict[f"{best_gen_key}_time_std"]
            best_per_sub[submission_id_v0][f"{gen_col}_acc"] = result_dict[f"{best_gen_key}_acc"]
        except IndexError:

            pdb.set_trace()

    return best_per_sub

def print_summary(cfg, merged, results, gen_col: str):
    """Print summary metrics and save the detailed evaluation report."""
    report_rows = []


    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        submission_id_v0 = row["submission_id_v0"]

        if submission_id_v0 not in results:
            continue

        report_row = row.to_dict()
        report_row.update(results[submission_id_v0])
        report_rows.append(report_row)

    assert len(results) == len(report_rows)
    print(f"Writing a report with {len(report_rows)} rows to {cfg.output_report_file_path}")
    run_metrics = pd.DataFrame(report_rows)


    run_metrics.to_json(cfg.output_report_file_path, orient="records", lines=True)


    run_metrics = run_metrics[
        (run_metrics[f"{gen_col}_acc"] > 0.99) & (run_metrics["input_acc"] > 0.99)
    ]
    if run_metrics.empty:
        return


    print("--- Execution time ---")

    print(
        f"[CodeNet report] Slow-program time (ms): {mean_std(run_metrics, 'cpu_time_v0')}"
    )
    print(
        f"[CodeNet report] Reference-program time (ms): {mean_std(run_metrics, 'cpu_time_v1')}"
    )

    print("-" * 80)

    print(f"[Measured] Slow-program time (ms): {mean_std(run_metrics, 'input_time')}")
    print(
        f"[Measured] Reference-program time (ms): {mean_std(run_metrics, 'reference_time')}"
    )
    print(
        f"[Measured] Generated-program time (ms): {mean_std(run_metrics, f'{gen_col}_time')}"
    )


    run_metrics_improved = run_metrics[
        run_metrics[f"{gen_col}_time_mean"] < run_metrics["reference_time_mean"]
    ]
    if len(run_metrics_improved) > 0:
        print("--- Metrics for improved programs ---")
        print(
            f"Found {len(run_metrics_improved)} generated programs faster than the reference"
        )
        print(
            f"[Measured] Slow-program time (ms): {mean_std(run_metrics_improved, 'input_time')}"
        )
        print(
            f"[Measured] Reference-program time (ms): {mean_std(run_metrics_improved, 'reference_time')}"
        )
        print(
            f"[Measured] Generated-program time (ms): {mean_std(run_metrics_improved, f'{gen_col}_time')}"
        )


    print(
        f"Anomalies where the measured reference is slower: {len(get_anomalies(run_metrics))}"
    )
    print("--- Additional metrics ---")


    valid_samples = run_metrics[
        (run_metrics[f"{gen_col}_acc"] == 1.0) & (run_metrics["input_acc"] == 1.0)
        ]
    print(f"Valid samples (100% accuracy): {len(valid_samples)}")


    optimized_samples = valid_samples[
        valid_samples[f"{gen_col}_time_mean"] *1.1  < valid_samples["input_time_mean"]
        ]
    print(f"Optimized samples (faster than slow code): {len(optimized_samples)}")


def mean_std(df, col) -> str:
    """Return a formatted mean and standard deviation for a result column."""
    mean_col = f"{col}_mean"
    std_col = f"{col}_std"

    if mean_col not in df.columns or std_col not in df.columns:
        return f"{df[col].mean():.4f} ± {df[col].std():.4f}"


    return f"{df[mean_col].mean():.4f} ± {df[std_col].mean():.4f}"


def get_anomalies(run_metrics):
    """Return cases whose measured timing contradicts the dataset improvement label."""

    run_metrics["codenet_reported_rel_improvement"] = (
        run_metrics["cpu_time_v0"] - run_metrics["cpu_time_v1"]
    ) / run_metrics["cpu_time_v0"]
    run_metrics["codenet_reported_rel_improvement"] = run_metrics[
        "codenet_reported_rel_improvement"
    ].apply(lambda x: round(x * 100, 2))


    run_metrics["measured_rel_improvement"] = (
        run_metrics["input_time_mean"] - run_metrics["reference_time_mean"]
    ) / run_metrics["input_time_mean"]
    run_metrics["measured_rel_improvement"] = run_metrics["measured_rel_improvement"].apply(
        lambda x: round(x * 100, 2)
    )


    run_metrics["is_anomaly"] = run_metrics.apply(
        lambda x: x["codenet_reported_rel_improvement"] > 10 and x["measured_rel_improvement"] < 0,
        axis=1,
    )

    run_metrics_anomalies = run_metrics[run_metrics["is_anomaly"]]
    return run_metrics_anomalies


def get_input_from_prompt(
    row: pd.Series,
    question_sep: str = "# slower version:",
    answer_sep: str = "# optimized version of the same code:",
) -> str:
    """Extract source code from a prompt-based generation record."""


    if "entire_prompt" in row:
        prompt_str = row["entire_prompt"]
    else:
        prompt_str = row["prompt"] + row["question"]
    prompt_str = prompt_str.replace("\n\n\n\n\n", "")


    return prompt_str.split(question_sep)[-1].split(answer_sep)[0].strip()


if __name__ == "__main__":

    args = EvaluationConfig.get_args()
    args.add_argument("--eval_config", type=str, required=False)
    args = args.parse_args()


    if args.eval_config is not None:

        evaluation_config = EvaluationConfig.from_yaml(args.eval_config)
    else:

        evaluation_config = EvaluationConfig.from_args(args)


    evaluate_generated_outputs(evaluation_config)