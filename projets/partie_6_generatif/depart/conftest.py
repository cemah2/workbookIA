"""pytest configuration of the mini-project: makes `wb`, `mylearn` and this folder importable.

From the root of the repository:

    python -m pytest mon_travail/projets/partie_6_generatif -q
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "src" / "wb").exists()), Path(os.environ.get("WB_ROOT", HERE)))
for path in (ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import wb  # noqa: E402

# Your own mylearn library; the modules of parts I to V that you have not written come from the reference.
wb.load_mylearn("learner", fallback="ref", chapter="CP6", notify=False)


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


def _not_written_yet(excinfo):
    """True for the NotImplementedError of a function still to write: raised in genbench/ or in your mylearn/,
    not by PyTorch (« Module [...] is missing the required forward function » is a real error)."""
    if not excinfo.errisinstance(NotImplementedError):
        return False
    return bool({"genbench", "mylearn"} & set(Path(str(excinfo.traceback[-1].path)).parts))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """« ⏳ pas encore écrit : vae_loss » instead of a traceback, in a test or in one of its fixtures."""
    report = (yield).get_result()
    if report.when in ("setup", "call") and call.excinfo is not None and _not_written_yet(call.excinfo):
        report.longrepr = _Pending(f"⏳ pas encore écrit : {call.excinfo.value or 'une fonction de ton module'}")
