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


# ------------------------------------------------------------------ fallback on the reference
def _fake_repo(tmp_path):
    """A minimal repository: chapter 2 publishes a.py, chapter 3 publishes b.py and c.py."""
    import json as _json

    root = tmp_path / "repo"
    stubs = root / "templates" / "mylearn_stubs"
    ref = root / "solutions" / "mylearn_ref"
    learner = root / "mon_travail" / "mylearn"
    for folder in (stubs, ref, learner):
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("")
    (stubs / "MANIFEST.json").write_text(_json.dumps(
        {"base": ["__init__.py"], "chapters": {"2": ["a.py"], "3": ["b.py", "c.py"]}}))
    (ref / "a.py").write_text("def double(x):\n    return 2 * x\n")
    (ref / "b.py").write_text("def triple(x):\n    return 3 * x\n")
    (learner / "c.py").write_text(
        "from .a import double\n\ndef six_times(x):\n    return 3 * double(x)\n")
    return root


def test_fallback_takes_earlier_modules_from_the_reference(tmp_path, capsys):
    import wb

    root = _fake_repo(tmp_path)
    module = wb.load_mylearn("learner", root=root, fallback="ref", chapter="3")
    assert module.c.six_times(2) == 12  # c.py (learner) imports a.py, which only the ref has
    assert "mylearn.a" in module.__wb_fallback__.used
    assert "la référence est utilisée" in capsys.readouterr().out
    wb.load_mylearn("ref")


def test_fallback_never_replaces_a_module_of_the_current_chapter(tmp_path):
    import wb
    from wb.errors import MylearnMissingError

    root = _fake_repo(tmp_path)
    module = wb.load_mylearn("learner", root=root, fallback="ref", chapter="3", notify=False)
    with pytest.raises(MylearnMissingError):
        module.b  # noqa: B018 - chapter 3 module: the learner must write it
    module = wb.load_mylearn("learner", root=root, fallback="ref", notify=False)  # no chapter
    assert module.b.triple(2) == 6
    wb.load_mylearn("ref")


def test_learner_module_wins_over_the_reference(tmp_path):
    import wb

    root = _fake_repo(tmp_path)
    (root / "mon_travail" / "mylearn" / "a.py").write_text("def double(x):\n    return x + x + 0.5\n")
    module = wb.load_mylearn("learner", root=root, fallback="ref", chapter="3", notify=False)
    assert module.c.six_times(1) == 7.5
    assert module.__wb_fallback__.used == []
    wb.load_mylearn("ref")


def test_reloading_removes_the_fallback_finder(tmp_path):
    import sys as _sys

    import wb
    from wb.impl import _FallbackFinder

    root = _fake_repo(tmp_path)
    wb.load_mylearn("learner", root=root, fallback="ref", notify=False)
    wb.load_mylearn("learner", root=root, fallback="ref", notify=False)
    assert sum(isinstance(f, _FallbackFinder) for f in _sys.meta_path) == 1
    wb.load_mylearn("ref")
    assert not any(isinstance(f, _FallbackFinder) for f in _sys.meta_path)


def test_modules_before_uses_the_study_order():
    from wb.impl import modules_before

    before_14 = modules_before("14")
    assert "tree.py" in before_14 and "ensemble.py" not in before_14
    assert modules_before("0a") == set()  # base files never come from the reference
    assert "_example.py" not in modules_before("B8")
    assert modules_before(None) is None
    with pytest.raises(ValueError):
        modules_before("99")


def test_pytest_learner_mode_tests_only_the_learner_modules(tmp_path):
    """Fallback in the tests: a learner module may import a reference module, which is not tested."""
    import os

    pkg = tmp_path / "mylearn"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mine.py").write_text("from ._example import mean\n\ndef twice_mean(v):\n"
                                 "    return 2 * mean(v)\n")
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(ROOT / "tests" / "conftest.py", project / "conftest.py")
    (project / "test_fb_tmp.py").write_text(
        "def test_mine(mylearn_module):\n    assert mylearn_module('mine').twice_mean([1, 3]) == 4\n\n"
        "def test_example(mylearn_module):\n    mylearn_module('_example')\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "WB_ROOT": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
         "--impl=learner", f"--learner-dir={pkg}"],
        cwd=project, capture_output=True, text=True, timeout=120, env=env)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "1 passed" in out and "1 skipped" in out, out


def test_fallback_subpackage_respects_the_chapter(tmp_path):
    """A sub-package missing from the learner's library must not open the door to its modules."""
    import json as _json

    import wb
    from wb.errors import MylearnMissingError

    root = _fake_repo(tmp_path)
    ref = root / "solutions" / "mylearn_ref"
    (ref / "nn").mkdir()
    (ref / "nn" / "__init__.py").write_text("")
    (ref / "nn" / "layers.py").write_text("def dense(x):\n    return x\n")
    (ref / "nn" / "activations.py").write_text("def relu(x):\n    return max(0, x)\n")
    manifest = root / "templates" / "mylearn_stubs" / "MANIFEST.json"
    data = _json.loads(manifest.read_text())
    data["chapters"].update({"16": ["nn/__init__.py", "nn/layers.py"], "17": ["nn/activations.py"]})
    manifest.write_text(_json.dumps(data))
    # chapter 16 notebook without nn/: nothing of nn comes from the reference
    module = wb.load_mylearn("learner", root=root, fallback="ref", chapter="16", notify=False)
    with pytest.raises(MylearnMissingError):
        module.nn.layers  # noqa: B018
    # chapter 17 notebook: nn and nn.layers (ch. 16) come from the reference, not nn.activations
    module = wb.load_mylearn("learner", root=root, fallback="ref", chapter="17", notify=False)
    assert module.nn.layers.dense(3) == 3
    with pytest.raises(MylearnMissingError):
        module.nn.activations  # noqa: B018
    # the learner's own nn/activations.py is used even though nn/ itself comes from the reference
    (root / "mon_travail" / "mylearn" / "nn").mkdir()
    (root / "mon_travail" / "mylearn" / "nn" / "activations.py").write_text("def relu(x):\n    return 42\n")
    module = wb.load_mylearn("learner", root=root, fallback="ref", chapter="17", notify=False)
    assert module.nn.activations.relu(-1) == 42
    wb.load_mylearn("ref")


def test_short_summary_lines_carry_the_reason():
    # `pytest -rf` prints one line per failure: the notebooks show only these lines (with COLUMNS=200)
    import os

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", EXAMPLE, "-q", "-p", "no:cacheprovider", "--impl=stubs", "-rf", "--tb=line"],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env={**os.environ, "COLUMNS": "200"},
    )
    failed = [line for line in proc.stdout.splitlines() if line.startswith("FAILED")]
    assert failed and all("pas encore implémenté" in line for line in failed), proc.stdout


def test_numpy_assertion_messages_get_a_one_line_summary():
    import importlib.util

    import numpy as np

    spec = importlib.util.spec_from_file_location("wb_conftest", ROOT / "tests" / "conftest.py")
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    try:
        np.testing.assert_allclose(0.612903, 0.609926, rtol=1e-10, err_msg="average='weighted'")
    except AssertionError as exc:
        summary = conftest._numpy_summary(str(exc))
    assert "average='weighted'" in summary and "0.609926" in summary and "0.612903" in summary
    assert "\n" not in summary
    assert conftest._numpy_summary("a plain message") is None
    try:   # a 2-D array spreads over several lines in NumPy's message
        np.testing.assert_allclose([[1.0, 2.0], [3.0, 4.5]], [[1.0, 2.0], [3.0, 4.0]], err_msg="ddof=1")
    except AssertionError as exc:
        summary = conftest._numpy_summary(str(exc))
    assert "ddof=1" in summary and "\n" not in summary
    assert "expected array([[1., 2.], [3., 4.]])" in summary and "got array([[1. , 2. ], [3. , 4.5]])" in summary
