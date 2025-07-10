# -*- coding: utf-8 -*-
"""Utility script to detect duplicate filenames in the repository.

Run as ``python scripts/check_duplicates.py``.  Exits with non-zero status
if duplicates are found so it can be wired into CI pipelines.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict


# Configure a minimal logger – avoids Ruff ``T201`` print-statement warnings
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def find_duplicates(root: str = ".") -> list[tuple[str, list[str]]]:
    """Scan *root* recursively and return duplicates by basename.

    Returns a list of tuples *(basename, [paths])* where the basename appears
    more than once in the tree (excluding hidden directories and venvs).
    """
    dup_map: defaultdict[str, list[str]] = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip common virtual-env / VCS dirs
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", "__pycache__", ".venv", "venv", "env"}
        ]
        for fname in filenames:
            dup_map[fname].append(os.path.join(dirpath, fname))

    duplicates = [(name, paths) for name, paths in dup_map.items() if len(paths) > 1]
    return duplicates


def main() -> None:  # noqa: D401
    """Entry point for CLI execution."""

    duplicates = find_duplicates()
    if not duplicates:
        logger.info("✅ No duplicate filenames detected.")
        return

    logger.error("❌ Duplicate filenames found:")
    for name, paths in duplicates:
        logger.error("  %s:", name)
        for p in paths:
            logger.error("    - %s", p)
    sys.exit(1)


if __name__ == "__main__":
    main()
