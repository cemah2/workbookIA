"""Shared pytest configuration.

Choose which implementation of mylearn the tests run against:

    pytest                      # your code: mon_travail/mylearn (default)
    pytest --impl=ref           # the reference: solutions/mylearn_ref
    pytest --impl=stubs         # the empty skeletons (all mylearn tests must fail)

Tests of mylearn use the ``mylearn`` fixture (the whole package) or the
``mylearn_module`` fixture (one module, skipped if you have not created it yet).
With your code (``--impl=learner``), a module you do not have yet is taken from
the reference when another module needs it (e.g. your ``ensemble.py`` imports
``tree.py``), but only your own modules are tested.
A function that still raises ``NotImplementedError`` is reported as a short
"⏳ pas encore implémenté" failure instead of a long traceback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# The repository (WB_ROOT when the tests of the mechanism copy this file elsewhere)
ROOT = Path(os.environ.get("WB_ROOT") or Path(__file__).resolve().parents[1])
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))  # `import wb` works without `pip install -e .`

_STATE: dict = {"impl": None, "path": None, "module": None, "error": None}


def pytest_addoption(parser):
    group = parser.getgroup("mylearn", "workbook: implementation of mylearn under test")
    group.addoption(
        "--impl",
        choices=["learner", "ref", "stubs"],
        default=os.environ.get("WB_IMPL", "learner"),
        help="learner = mon_travail/mylearn (default), ref = solutions/mylearn_ref, "
        "stubs = templates/mylearn_stubs",
    )
    group.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="also run the tests that download data (Fashion-MNIST, CIFAR-10...)",
    )
    group.addoption(
        "--learner-dir",
        default=None,
        help="use another directory as the learner's package (tests of the mechanism)",
    )


def pytest_configure(config):
    from wb.impl import load_mylearn, mylearn_path

    impl = config.getoption("--impl")
    learner_dir = config.getoption("--learner-dir")
    path = Path(learner_dir) if (impl == "learner" and learner_dir) else mylearn_path(impl, ROOT)
    _STATE.update(impl=impl, path=path)
    try:
        _STATE["module"] = load_mylearn(
            impl, path=path if learner_dir else None, root=ROOT,
            fallback="ref" if impl == "learner" else None, notify=False,
        )
    except ImportError as exc:  # includes MylearnNotFoundError
        _STATE["error"] = str(exc)
    os.environ["WB_IMPL"] = impl


def pytest_report_header(config):
    status = "OK" if _STATE["module"] is not None else f"absent ({_STATE['error']})"
    return f"mylearn : implémentation « {_STATE['impl']} » -> {_STATE['path']} [{status}]"


@pytest.fixture(scope="session")
def mylearn():
    """The mylearn package selected with --impl (skips if it does not exist).

    For the tests of the infrastructure only. Tests of mylearn modules use
    ``mylearn_module``: in learner mode, a module missing from the learner's library
    is served by the reference (fallback), so accessing it through this fixture would
    test the reference instead of the learner (guarded by tests/infra/test_syllabus.py).
    """
    if _STATE["module"] is None:
        pytest.skip(f"mylearn introuvable : {_STATE['error']}")
    return _STATE["module"]


@pytest.fixture(scope="session")
def mylearn_module(mylearn):
    """Import one mylearn module by name: ``metrics = mylearn_module("metrics")``.

    Skips the test (instead of failing) when the module has not been copied yet.
    """
    import importlib

    def _import(name: str):
        full = f"mylearn.{name}"
        if _STATE["impl"] == "learner":  # test only the modules the learner has
            rel = Path(*name.split("."))
            base = Path(_STATE["path"])
            present = (base / rel).with_suffix(".py").exists() or (
                base / rel / "__init__.py").exists()
            if not present:
                pytest.skip(
                    f"{full} n'existe pas encore : lance python tools/start_chapter.py <chapitre>"
                )
        try:
            return importlib.import_module(full)
        except ModuleNotFoundError as exc:
            if exc.name and (full == exc.name or full.startswith(exc.name + ".")):
                pytest.skip(
                    f"{full} n'existe pas encore : lance python tools/start_chapter.py <chapitre>"
                )
            raise

    return _import


def pytest_collection_modifyitems(config, items):
    run_network = config.getoption("--run-network")
    skip_network = pytest.mark.skip(reason="télécharge des données : relance avec --run-network")
    for item in items:
        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)
        if {"mylearn", "mylearn_module"} & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.mylearn)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and call.excinfo is not None
        and call.excinfo.errisinstance(NotImplementedError)
    ):
        message = str(call.excinfo.value) or "fonction pas encore écrite"
        report.longrepr = f"⏳ pas encore implémenté : {message}"
