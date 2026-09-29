"""Proves that the double-implementation mechanism works (--impl=learner|ref|stubs).

Each case runs pytest in a subprocess on tests/test_example_mylearn.py.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = "tests/test_example_mylearn.py"


def run_pytest(*args):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", EXAMPLE, "-q", "-p", "no:cacheprovider", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_ref_passes():
    code, out = run_pytest("--impl=ref")
    assert code == 0, out
    assert "5 passed" in out


def test_stubs_fail_with_friendly_message():
    code, out = run_pytest("--impl=stubs")
    assert code == 1, out
    assert "5 failed" in out and "pas encore implémenté" in out


def test_learner_copy_of_stubs_fails(tmp_path):
    shutil.copytree(ROOT / "templates" / "mylearn_stubs", tmp_path / "mylearn")
    code, out = run_pytest("--impl=learner", f"--learner-dir={tmp_path / 'mylearn'}")
    assert code == 1 and "5 failed" in out, out


def test_learner_finished_code_passes(tmp_path):
    shutil.copytree(ROOT / "solutions" / "mylearn_ref", tmp_path / "mylearn")
    code, out = run_pytest("--impl=learner", f"--learner-dir={tmp_path / 'mylearn'}")
    assert code == 0 and "5 passed" in out, out


def test_learner_missing_package_is_skipped(tmp_path):
    code, out = run_pytest("--impl=learner", f"--learner-dir={tmp_path / 'nothing'}", "-rs")
    assert code == 0 and "5 skipped" in out, out


def test_learner_missing_module_is_skipped(tmp_path):
    pkg = tmp_path / "mylearn"
    pkg.mkdir()
    shutil.copy(ROOT / "templates" / "mylearn_stubs" / "__init__.py", pkg / "__init__.py")
    code, out = run_pytest("--impl=learner", f"--learner-dir={pkg}", "-rs")
    assert code == 0 and "5 skipped" in out and "_example" in out, out


def test_load_mylearn_alias_and_lazy_submodules():
    import wb

    ref = wb.load_mylearn("ref")
    assert ref.__wb_impl__ == "ref"
    assert ref._example.mean([2, 4]) == 3.0  # submodule imported on attribute access
    from mylearn._example import mean  # noqa: PLC0415

    assert mean([1, 2]) == 1.5
    stubs = wb.load_mylearn("stubs")
    with pytest.raises(NotImplementedError):
        stubs._example.mean([1])
    with pytest.raises(AttributeError):
        stubs.does_not_exist  # noqa: B018
    wb.load_mylearn("ref")  # leave the reference loaded for other tests


def test_nested_subpackages_lazy_load(tmp_path):
    import wb

    pkg = tmp_path / "mylearn"
    (pkg / "nn").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "nn" / "__init__.py").write_text("")
    (pkg / "nn" / "activations.py").write_text("def relu(x):\n    return max(0, x)\n")
    module = wb.load_mylearn("learner", path=pkg)
    assert module.nn.activations.relu(-3) == 0
    with pytest.raises(AttributeError):
        module.nn.missing_module  # noqa: B018
    wb.load_mylearn("ref")


def test_missing_learner_forgets_previous_implementation(tmp_path, capsys):
    import sys as _sys

    import wb

    wb.load_mylearn("ref")
    assert "mylearn" in _sys.modules
    wb.load_mylearn("learner", path=tmp_path / "nothing", missing_ok=True)
    assert "mylearn" not in _sys.modules
    wb.load_mylearn("ref")


def test_nested_module_missing_is_skipped(tmp_path):
    import os

    pkg = tmp_path / "mylearn"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(ROOT / "tests" / "conftest.py", project / "conftest.py")
    (project / "test_nested_tmp.py").write_text(
        "def test_x(mylearn_module):\n    mylearn_module('nn.activations')\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "WB_ROOT": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
         "--impl=learner", f"--learner-dir={pkg}"],
        cwd=project, capture_output=True, text=True, timeout=120, env=env,
    )
    assert "1 skipped" in proc.stdout, proc.stdout + proc.stderr
