"""pytest configuration of the project: ``src/`` importable without installation, and the ⏳ of the functions not
written yet.

    python -m pytest -q                 # from the root of the project
    PF_STRICT=1 python -m pytest -q     # the end of the project: a ⏳ (or a skipped « todo » test) fails

While a function of the package raises ``NotImplementedError("TODO PF.n: ...")``, the tests that call it are reported
as skipped with « ⏳ pas encore écrit : ... » instead of a failure: the CI stays green from the first commit, and the
skips list what remains to do. With ``PF_STRICT=1`` (the last weeks, PF.10), they fail, except the tests marked
``data`` that skip because the downloaded dataset is absent (in the CI, which never downloads it).
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
STRICT = os.environ.get("PF_STRICT") == "1"
# the name of the package, whatever you renamed it to (the only folder of src/ with an __init__.py)
PACKAGES = {p.name for p in SRC.iterdir() if (p / "__init__.py").is_file()} if SRC.is_dir() else set()


def _not_written_yet(excinfo) -> bool:
    """The NotImplementedError of a function of the package still to write (not one raised by a library)."""
    if not excinfo.errisinstance(NotImplementedError):
        return False
    return bool(PACKAGES & set(Path(str(excinfo.traceback[-1].path)).parts))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if report.when not in ("setup", "call"):
        return
    if STRICT:
        # a « data » test may still skip where the dataset is absent (the CI): it reads files that are not in git
        if report.skipped and item.get_closest_marker("todo") is not None and item.get_closest_marker("data") is None:
            report.outcome = "failed"
            report.longrepr = f"PF_STRICT=1: expected this « todo » test to be written and to pass, it was skipped"
        return
    if call.excinfo is not None and _not_written_yet(call.excinfo):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"⏳ pas encore écrit : {call.excinfo.value}")
