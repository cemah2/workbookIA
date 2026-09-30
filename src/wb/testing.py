"""Run pytest on test functions written in a notebook (🛠️ exercises, from chapter 0A on).

>>> result = wb.run_pytest([test_simple, test_raises], subject=my_function, name="min_max_scale")
>>> result.passed, result.failed

The source code of the tests (and of the function under test, renamed if needed)
is copied into a temporary ``test_notebook.py`` file, preceded by
``import math``, ``import numpy as np`` and ``import pytest``; then pytest runs
**for real** on this file, in a separate process. Tests can therefore use
``assert``, ``pytest.approx``, ``pytest.raises`` and ``@pytest.mark.parametrize``
exactly as in a ``tests/test_*.py`` file.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

HEADER = "import math\n\nimport numpy as np\nimport pytest\n\n\n"


@dataclass
class PytestResult:
    """Counts of a pytest run, with its summary line and full output."""

    passed: int
    failed: int
    errors: int
    summary: str
    output: str

    @property
    def ok(self) -> bool:
        """True when at least one test ran and none failed."""
        return self.passed > 0 and self.failed == 0 and self.errors == 0

    def __repr__(self) -> str:
        return f"PytestResult(passed={self.passed}, failed={self.failed}, errors={self.errors})"


def source_of(obj, name: str | None = None) -> str:
    """Source code of a function or class (decorators included), renamed to ``name`` if given."""
    try:
        source = textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"impossible de lire le code source de {obj!r} : définis-le avec `def` dans une cellule "
            "(une fonction créée par `lambda` ou dans une boucle n'a pas de source lisible)."
        ) from exc
    if name and name != obj.__name__:
        pattern = rf"^(\s*)(def|class)\s+{re.escape(obj.__name__)}\b"
        source = re.sub(pattern, rf"\1\2 {name}", source, count=1, flags=re.MULTILINE)
    return source


def _count(summary: str, word: str) -> int:
    match = re.search(rf"(\d+) {word}", summary)
    return int(match.group(1)) if match else 0


def run_pytest(tests, *, subject=None, name: str | None = None, extra=(), quiet: bool = False,
               timeout: float = 120.0) -> PytestResult:
    """Run the test functions ``tests`` with pytest, in a separate process.

    Parameters
    ----------
    tests : callable or list of callables
        Test functions (their names should start with ``test_``).
    subject : callable or class, optional
        The code under test, copied before the tests.
    name : str, optional
        Name given to ``subject`` in the test file (e.g. run the tests written for
        ``min_max_scale`` against a function called ``scale_bug_1``).
    extra : list of callables or classes
        Other helpers the tests need, copied with their own names.
    quiet : bool
        If False, print pytest's summary line (and the details when something fails).

    Returns
    -------
    PytestResult
        ``passed``, ``failed``, ``errors``, ``summary``, ``output`` and ``ok``.
    """
    if callable(tests) and not isinstance(tests, (list, tuple)):
        tests = [tests]
    parts = [HEADER]
    for obj in extra:
        parts.append(source_of(obj) + "\n\n")
    if subject is not None:
        parts.append(source_of(subject, name) + "\n\n")
    for test in tests:
        parts.append(source_of(test) + "\n\n")
    with tempfile.TemporaryDirectory(prefix="wb_pytest_") as tmp:   # removed afterwards
        folder = Path(tmp)
        path = folder / "test_notebook.py"
        path.write_text("".join(parts), encoding="utf-8")
        command = [sys.executable, "-m", "pytest", str(path), "-q", "-p", "no:cacheprovider", "--color=no",
                   "--rootdir", str(folder)]
        completed = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=timeout)
    output = (completed.stdout + completed.stderr).strip()
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    summary = lines[-1].strip("= ") if lines else output.splitlines()[-1] if output else "no output"
    result = PytestResult(_count(summary, "passed"), _count(summary, "failed"), _count(summary, "error"),
                          summary, output)
    if not quiet:
        if not result.ok:
            print(output)
        print(f"pytest : {summary}")
    return result
