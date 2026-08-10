"""Synthetic checks for the limited-public-test RQ2 protocol."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from evaluate_execution_time import merge_io, parse_io, validation_io
from export_top_results import summarize_metrics
from io_processing.filter_repair_remove_zero import parse_candidate_inputs
from llm_generation import build_prompt, load_dataframe, load_prompt_template
from pie_sandbox import capture_oracle_outputs


class RQ2ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            [
                {
                    "Slow_Code": "n = int(input())\nprint(n * 2)",
                    "4_Example_Prompt": "PROBLEM_DISJOINT_REFERENCE",
                    "Representative_IO": json.dumps(
                        {"inputs": ["2\n"], "outputs": ["4\n"]}
                    ),
                    "Selected_IO": json.dumps(
                        {"inputs": ["2\n", "3\n"], "outputs": ["4\n", "6\n"]}
                    ),
                    "Public_IO_unit_tests__Dedup": str(
                        {"inputs": ["5\n"], "outputs": ["10\n"]}
                    ),
                    "Hide_IO_unit_tests__Dedup": str(
                        {"inputs": ["SECRET_PRIVATE_INPUT\n"], "outputs": ["0\n"]}
                    ),
                    "Overview_Description": "Doubles an integer.",
                }
            ]
        )

    def test_private_columns_are_dropped_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            self.frame.to_csv(path, index=False)
            loaded = load_dataframe(str(path))
        self.assertIn("Public_IO_unit_tests__Dedup", loaded.columns)
        self.assertNotIn("Hide_IO_unit_tests__Dedup", loaded.columns)

    def test_stage_prompts_enforce_test_access_boundary(self) -> None:
        test_prompt = build_prompt(
            self.frame,
            0,
            -1,
            "python",
            load_prompt_template(-1),
        )
        self.assertIn("PROBLEM_DISJOINT_REFERENCE", test_prompt)
        self.assertNotIn("Public_IO_unit_tests", test_prompt)
        self.assertNotIn("SECRET_PRIVATE_INPUT", test_prompt)

        description_prompt = build_prompt(
            self.frame,
            0,
            0,
            "python",
            load_prompt_template(0),
        )
        self.assertIn('"2\\n"', description_prompt)
        self.assertNotIn('"5\\n"', description_prompt)
        self.assertNotIn("SECRET_PRIVATE_INPUT", description_prompt)

        optimization_prompt = build_prompt(
            self.frame,
            0,
            1,
            "python",
            load_prompt_template(1),
        )
        self.assertIn('"2\\n"', optimization_prompt)
        self.assertIn("'5\\n'", optimization_prompt)
        self.assertNotIn("SECRET_PRIVATE_INPUT", optimization_prompt)

    def test_oracle_overwrites_model_outputs(self) -> None:
        proposed = json.dumps(
            {"inputs": ["2\n", "7\n"], "outputs": ["wrong", "wrong"]}
        )
        inputs = parse_candidate_inputs(proposed)
        oracle = capture_oracle_outputs(
            "n = int(input())\nprint(n * 2)",
            inputs,
            "python",
        )
        self.assertEqual(oracle["outputs"], ["4\n", "14\n"])

    def test_candidate_validation_combines_generated_and_public_tests(self) -> None:
        public_columns = ["Public_IO_unit_tests__Dedup"]
        combined = validation_io(self.frame.iloc[0], public_columns)
        self.assertEqual(combined["inputs"], ["2\n", "3\n", "5\n"])
        self.assertEqual(
            merge_io(parse_io(self.frame.iloc[0]["Selected_IO"]), parse_io("")),
            parse_io(self.frame.iloc[0]["Selected_IO"]),
        )

    def test_opt_uses_original_runtime_as_denominator(self) -> None:
        result = summarize_metrics(
            [1.0],
            [100.0],
            [1.0],
            [90.5],
            "boundary",
        )
        self.assertEqual(result["optimization_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
