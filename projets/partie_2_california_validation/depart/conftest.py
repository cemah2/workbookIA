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


import pytest  # noqa: E402


class _Pending:
    """A one-line report for a function that is not written yet (instead of a long traceback)."""

    def __init__(self, text):
        self.text = text
        self.reprcrash = self

    @property
    def message(self):
        return self.text

    def toterminal(self, tw):
        tw.line(self.text)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """« ⏳ pas encore écrit : LanguageDetector.fit » instead of a traceback, in a test or in one of its fixtures."""
    report = (yield).get_result()
    if report.when in ("setup", "call") and call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.longrepr = _Pending(f"⏳ pas encore écrit : {call.excinfo.value or 'une fonction de ton module'}")
