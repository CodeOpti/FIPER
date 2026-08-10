"""Render the four-panel RQ3 outcome-distribution figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = ("NC", "NO", "NH", "FH")
COLORS = ("#FE7E0D", "#1BA1E2", "#8C564B", "#66CC66")
HATCHES = ("\\\\", "||||", "*", "////")
METHODS = {
    "python": [
        "Instruction",
        "ICL",
        "RAG",
        "COT",
        "FasterPy",
        "EffiSkill",
        "FIPER",
        "EffiLearner",
        "SBLLM",
        "FIPER",
    ],
    "cpp": [
        "Instruction",
        "ICL",
        "RAG",
        "COT",
        "AutoPatch",
        "EffiSkill",
        "FIPER",
        "EffiLearner",
        "SBLLM",
        "FIPER",
    ],
}
METHOD_IDS = [
    "instruction",
    "icl",
    "rag",
    "cot",
    "language_specific",
    "effiskill",
    "fiper_no_public",
    "effilearner",
    "sbllm",
    "fiper_with_public",
]
PAPER_RESULTS = {
    "python": {
        "NC": [14.02, 14.36, 17.23, 21.79, 16.55, 36.29, 15.03, 20.95, 33.78, 13.68],
        "NO": [12.50, 7.26, 14.86, 8.78, 9.29, 6.72, 1.69, 3.55, 4.22, 1.69],
        "NH": [42.74, 37.33, 45.10, 29.73, 30.07, 25.40, 9.12, 17.23, 22.13, 9.29],
        "FH": [30.74, 41.05, 22.80, 39.70, 44.09, 31.59, 74.16, 58.28, 39.86, 75.34],
    },
    "cpp": {
        "NC": [20.82, 20.25, 26.16, 30.94, 31.65, 49.37, 17.44, 19.97, 19.83, 7.88],
        "NO": [71.45, 21.24, 20.68, 21.10, 34.74, 24.47, 35.02, 41.63, 19.97, 41.49],
        "NH": [4.50, 40.65, 33.61, 30.66, 9.99, 15.19, 22.78, 22.50, 36.15, 24.05],
        "FH": [3.23, 17.86, 19.55, 17.30, 23.63, 10.97, 24.75, 15.89, 24.05, 26.58],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Optional JSON output from categorize_optimization_results.py.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("rq3_optimization_distribution.pdf")
    )
    return parser.parse_args()


def _load_results(path: Path | None) -> dict[str, dict[str, list[float]]]:
    if path is None:
        return PAPER_RESULTS
    raw = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, dict[str, list[float]]] = {}
    for dataset in ("python", "cpp"):
        language_specific = "fasterpy" if dataset == "python" else "autopatch"
        identifiers = [
            language_specific if item == "language_specific" else item
            for item in METHOD_IDS
        ]
        results[dataset] = {
            category: [
                float(raw[dataset][method]["percentages"][category])
                for method in identifiers
            ]
            for category in CATEGORIES
        }
    return results


def _validate(results: dict[str, dict[str, list[float]]]) -> None:
    for dataset in ("python", "cpp"):
        arrays = [np.asarray(results[dataset][category], dtype=float) for category in CATEGORIES]
        if any(len(array) != 10 for array in arrays):
            raise ValueError(f"{dataset} must contain ten values per category.")
        totals = sum(arrays)
        if not np.allclose(totals, 100.0, atol=0.15):
            raise ValueError(f"{dataset} category percentages do not sum to 100.")


def _stacked_bars(
    axis: Any,
    results: dict[str, list[float]],
    indices: slice,
    include_labels: bool,
) -> None:
    positions = np.arange(len(results["NC"][indices]))
    bottom = np.zeros(len(positions))
    for category, color, hatch in zip(CATEGORIES, COLORS, HATCHES):
        values = np.asarray(results[category][indices], dtype=float)
        axis.bar(
            positions,
            values,
            0.75,
            bottom=bottom,
            label=category if include_labels else None,
            color=color,
            hatch=hatch,
            linewidth=0.4,
            edgecolor="white",
        )
        bottom += values
    axis.set_xticks(positions)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)


def plot(results: dict[str, dict[str, list[float]]], output: Path) -> None:
    _validate(results)
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18, 3.1),
        sharey=True,
        gridspec_kw={"width_ratios": [7, 3, 7, 3]},
    )
    groups = (
        ("python", slice(0, 7)),
        ("python", slice(7, 10)),
        ("cpp", slice(0, 7)),
        ("cpp", slice(7, 10)),
    )
    for index, (axis, (dataset, selection)) in enumerate(zip(axes, groups)):
        _stacked_bars(axis, results[dataset], selection, index == 0)
        axis.set_xticklabels(METHODS[dataset][selection], fontsize=10)
        axis.tick_params(axis="x", rotation=20)
        axis.tick_params(axis="y", labelsize=11)
    axes[0].set_ylabel("Percentage (%)", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
    )
    captions = (
        "(a) Python w/o public test cases.",
        "(b) Python w/ public test cases.",
        "(c) C++ (-O3) w/o public test cases.",
        "(d) C++ (-O3) w/ public test cases.",
    )
    centers = (0.19, 0.43, 0.68, 0.91)
    for center, caption in zip(centers, captions):
        figure.text(center, 0.01, caption, ha="center", fontsize=11)
    figure.subplots_adjust(wspace=0.05, bottom=0.27, top=0.84)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plot(_load_results(args.data), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
