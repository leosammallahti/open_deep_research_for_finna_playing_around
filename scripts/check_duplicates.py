import os
import sys
from collections import defaultdict


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
    duplicates = find_duplicates()
    if not duplicates:
        print("✅ No duplicate filenames detected.")
        return

    print("❌ Duplicate filenames found:")
    for name, paths in duplicates:
        print(f"  {name}:")
        for p in paths:
            print(f"    - {p}")
    sys.exit(1)


if __name__ == "__main__":
    main()
