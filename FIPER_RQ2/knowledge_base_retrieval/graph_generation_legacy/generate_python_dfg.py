"""Legacy Python graph-generation entry point.

The unified implementation is the canonical path. This wrapper keeps the
historical Python-only entry point available without duplicating parser logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


UNIFIED_DIRECTORY = Path(__file__).resolve().parents[1] / "graph_generation_unified"
if str(UNIFIED_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(UNIFIED_DIRECTORY))

from generate_dfg import generate_graph_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--slow-column", default="Slow_program")
    parser.add_argument("--fast-column", default="Fast_program")
    parser.add_argument("--slow-unified-column", default="Slow_program_Uni")
    parser.add_argument("--fast-unified-column", default="Fast_program_Uni")
    parser.add_argument("--keep-failed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_graph_columns(
        args.input_path,
        args.output_path,
        "python",
        args.slow_column,
        args.fast_column,
        args.slow_unified_column,
        args.fast_unified_column,
        args.keep_failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
