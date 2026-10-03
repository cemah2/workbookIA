"""pytest configuration of the mini-project: makes `wb`, `mylearn` and this folder importable.

From the root of the repository:

    python -m pytest mon_travail/projets/partie_2_california_validation -q
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "src" / "wb").exists())
for path in (ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import wb  # noqa: E402

# Your own mylearn library; the modules of parts I and II that you have not written come from the reference.
wb.load_mylearn("learner", fallback="ref", chapter="CP2", notify=False)
