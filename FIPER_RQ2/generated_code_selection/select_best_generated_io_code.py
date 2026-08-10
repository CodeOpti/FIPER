"""Select the fastest functionally correct candidate for an RQ2 round."""

from __future__ import annotations

import argparse
import math


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--round", type=int, required=True)
    return parser.parse_args()


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def main() -> None:
    args = parse_args()
    import pandas as pd

    frame = pd.read_csv(args.dataset_path)
    candidate_columns = [
        column for column in frame.columns if str(column).startswith("Optimized_Code_")
        and str(column).split("_")[-1].isdigit()
    ]
    if not candidate_columns:
        raise KeyError("No Optimized_Code_N columns were found in the input CSV.")
    selected_code, selected_time, selected_rate = [], [], []
    selected_source, improved_flags, converged_flags = [], [], []
    for _, row in frame.iterrows():
        base_code = str(
            row.get(
                "Selected_Optimized_Code",
                row.get("Slow_Code", row.get("slow_code", "")),
            )
        )
        base_rate = float(row.get("Base_Code_PassRate", 0.0))
        base_time = float(row.get("Base_Code_TimeMs", math.inf))
        previous_converged = _truthy(row.get("Converged", False))
        candidates = []
        for column in candidate_columns:
            rate = float(row.get(f"{column}_PassRate", 0.0))
            elapsed = float(row.get(f"{column}_TimeMs", math.inf))
            if rate >= 1.0:
                candidates.append((elapsed, str(row[column]), rate, str(column)))

        best_candidate = min(candidates, key=lambda item: (item[0], item[3])) if candidates else None
        improved = bool(
            not previous_converged
            and best_candidate is not None
            and best_candidate[0] < base_time
        )
        if improved:
            elapsed, code, rate, source = best_candidate
        else:
            elapsed, code, rate, source = base_time, base_code, base_rate, "Base_Code"
        selected_code.append(code)
        selected_time.append(elapsed)
        selected_rate.append(rate)
        selected_source.append(source)
        improved_flags.append(improved)
        converged_flags.append(previous_converged or not improved)
    frame["Selected_Optimized_Code"] = selected_code
    frame["Selected_Optimized_Code_TimeMs"] = selected_time
    frame["Selected_Optimized_Code_PassRate"] = selected_rate
    frame[f"Round_{args.round}_Selected_Source"] = selected_source
    frame[f"Round_{args.round}_Improved"] = improved_flags
    frame["Converged"] = converged_flags
    frame.to_csv(args.output_path, index=False, encoding="utf-8")
    print(f"Selected round {args.round} candidates: {args.output_path}")


if __name__ == "__main__":
    main()
