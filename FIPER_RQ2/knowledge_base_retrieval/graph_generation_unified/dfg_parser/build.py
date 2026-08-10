"""Check the grammar packages used by the unified graph generator."""

from __future__ import annotations

import importlib


def main() -> int:
    required_packages = ("tree_sitter", "tree_sitter_python", "tree_sitter_cpp")
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit("Install the missing packages: " + ", ".join(missing))
    print("Tree-sitter grammar packages are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
