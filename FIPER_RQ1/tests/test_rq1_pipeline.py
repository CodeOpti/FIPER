"""Synthetic checks for the zero-public-test RQ1 pipeline."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from candidate_selection.select_best_candidate import _select_row
from generate_candidates import FORBIDDEN_EVALUATION_TERMS
from pie_sandbox import capture_oracle_outputs, evaluate_code
from retrieval.retrieve_examples import _validate_schema, retrieve


class RQ1PipelineTests(unittest.TestCase):
    def test_oracle_outputs_and_candidate_validation(self) -> None:
        slow_code = "n = int(input())\nprint(n * 2)\n"
        oracle = capture_oracle_outputs(slow_code, ["1\n", "7\n"], "python")
        tests = {"inputs": oracle["inputs"], "outputs": oracle["outputs"]}

        correct = evaluate_code(
            "n = int(input())\nprint(n << 1)\n",
            tests,
            "python",
            warmup_runs=0,
            measured_runs=1,
        )
        incorrect = evaluate_code(
            "n = int(input())\nprint(n * 3)\n",
            tests,
            "python",
            warmup_runs=0,
            measured_runs=1,
        )
        self.assertEqual(correct["test_passes"], [1, 1])
        self.assertEqual(incorrect["test_passes"], [0, 0])

    def test_selection_uses_generated_metrics(self) -> None:
        row = pd.Series(
            {
                "slow_code": "print(1)",
                "candidate_1": "print(1)",
                "candidate_2": "print(2)",
                "base_generated_pass_rate": 1.0,
                "base_generated_runtime_ms": 10.0,
                "candidate_1_generated_pass_rate": 1.0,
                "candidate_1_generated_runtime_ms": 5.0,
                "candidate_2_generated_pass_rate": 0.0,
                "candidate_2_generated_runtime_ms": 1.0,
            }
        )
        code, source, pass_rate, runtime = _select_row(
            row, ["base", "candidate_1", "candidate_2"], 0.999999
        )
        self.assertEqual((code, source, pass_rate, runtime), ("print(1)", "candidate_1", 1.0, 5.0))

    def test_forbidden_evaluation_terms_are_explicit(self) -> None:
        self.assertEqual(
            FORBIDDEN_EVALUATION_TERMS,
            ("public", "private", "hidden", "hide_io"),
        )

    def test_retrieval_is_problem_disjoint(self) -> None:
        queries = pd.DataFrame(
            [{"problem_id": "query-1", "slow_code": "n = int(input()); print(n * 2)"}]
        )
        knowledge_base = pd.DataFrame(
            [
                {
                    "problem_id": "kb-double",
                    "code": "x = int(input()); print(x * 2)",
                    "reference_tests": '{"inputs": ["2"], "outputs": ["4"]}',
                },
                {
                    "problem_id": "kb-sort",
                    "code": "print(*sorted(map(int, input().split())))",
                    "reference_tests": '{"inputs": ["2 1"], "outputs": ["1 2"]}',
                },
            ]
        )
        _validate_schema(queries, knowledge_base, allow_missing_ids=False)
        result = retrieve(queries, knowledge_base, top_k=1)
        retrieved_index = json.loads(result.loc[0, "retrieved_knowledge_indices"])[0]
        self.assertEqual(knowledge_base.iloc[retrieved_index]["problem_id"], "kb-double")

        overlapping = knowledge_base.copy()
        overlapping.loc[0, "problem_id"] = "query-1"
        with self.assertRaises(ValueError):
            _validate_schema(queries, overlapping, allow_missing_ids=False)


if __name__ == "__main__":
    unittest.main()
