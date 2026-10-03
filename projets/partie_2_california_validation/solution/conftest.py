"""pytest configuration of the reference solution: makes `wb`, `mylearn` (the reference) and this folder importable.

    python -m pytest projets/partie_2_california_validation/solution -q
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "src" / "wb").exists())
for path in (ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import wb  # noqa: E402

wb.load_mylearn("ref")   # the reference implementation of mylearn
