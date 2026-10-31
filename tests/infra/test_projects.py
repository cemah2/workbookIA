"""Guards on the mini-projects (projets/): the reference solution passes its tests, the starter kit mirrors it.

Each project has the same layout: ``solution/`` (the reference) and ``depart/`` (the
starter kit copied by ``tools/start_chapter.py``), each with ``data.py``, ``conftest.py``,
one module ``<name>.py`` or one package ``<name>/``, its tests ``test_<name>.py``, one
notebook and a ``README.md``. In the starter kit, every function raises NotImplementedError,
except ``__init__`` and the functions whose docstring starts with "Provided".
"""

import ast
import filecmp
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = sorted(p for p in (ROOT / "projets").glob("partie_*") if (p / "solution").is_dir())
SHARED = ("data.py", "conftest.py")


def _layout(project: Path) -> tuple[str, str, str]:
    """(module or package, its test file, notebook) of a project, read from its solution folder."""
    solution = project / "solution"
    modules = sorted(p.name for p in solution.glob("*.py") if p.name not in SHARED and not p.name.startswith("test_"))
    modules += sorted(p.name for p in solution.iterdir() if p.is_dir() and (p / "__init__.py").exists())
    notebooks = sorted(p.name for p in solution.glob("*.ipynb"))
    assert len(modules) == 1, f"{project.name}: expected one module or package in solution/, got {modules}"
    assert len(notebooks) == 1, f"{project.name}: expected one notebook in solution/, got {notebooks}"
    stem = modules[0].removesuffix(".py")
    return modules[0], f"test_{stem}.py", notebooks[0]


def _sources(folder: Path, module: str) -> list[Path]:
    """The Python files of a module (one file) or of a package (every file of the folder)."""
    path = folder / module
    return sorted(path.glob("*.py")) if path.is_dir() else [path]


def _public_api(path: Path) -> dict[str, str]:
    """Name -> signature of the public functions and methods of a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    api = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            api[node.name] = ast.unparse(node.args)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (item.name == "__init__" or not item.name.startswith("_")):
                    api[f"{node.name}.{item.name}"] = ast.unparse(item.args)
    return api


@pytest.mark.parametrize("project", PROJECTS, ids=lambda p: p.name)
def test_reference_solution_passes_its_tests(project):
    proc = subprocess.run([sys.executable, "-m", "pytest", str(project / "solution"), "-q", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]


@pytest.mark.parametrize("project", PROJECTS, ids=lambda p: p.name)
def test_notebook_is_named_after_the_project(project):
    """BIBLE §22: the notebook of MPn is mpn_<subject>.ipynb, a name that still says something in a portfolio."""
    part = re.match(r"partie_(\d+)_", project.name).group(1)
    notebook = _layout(project)[2]
    assert re.fullmatch(rf"mp{part}_[a-z0-9_]+\.ipynb", notebook), f"{project.name}: {notebook}"


@pytest.mark.parametrize("project", PROJECTS, ids=lambda p: p.name)
def test_starter_kit_mirrors_the_solution(project):
    starter, solution = project / "depart", project / "solution"
    module, tests, notebook = _layout(project)
    for name in (*SHARED, tests, module, notebook, "README.md"):
        assert (starter / name).exists(), f"depart/{name} missing"
    assert filecmp.cmp(starter / "data.py", solution / "data.py", shallow=False), "data.py differs between depart/ and solution/"
    assert [p.name for p in _sources(starter, module)] == [p.name for p in _sources(solution, module)], \
        f"depart/{module} and solution/{module} must have the same files"
    for start, ref in zip(_sources(starter, module), _sources(solution, module)):
        assert _public_api(start) == _public_api(ref), f"the API of depart/{module}/{start.name} differs"
        tree = ast.parse(start.read_text(encoding="utf-8"))
        given = {id(inner) for node in ast.walk(tree)            # helpers nested in a Provided function are given too
                 if isinstance(node, ast.FunctionDef) and (ast.get_docstring(node) or "").startswith("Provided")
                 for inner in ast.walk(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != "__init__" and id(node) not in given:
                assert "NotImplementedError" in ast.dump(node), \
                    f"depart/{start.name}: {node.name} must raise NotImplementedError"
    readme = (starter / "README.md").read_text(encoding="utf-8")
    figures = re.findall(r"\(figures/([\w.-]+\.png)\)", readme)
    assert "TODO" in readme and figures, "the starter README must hold TODO paragraphs and show the figures"
    for figure in figures:
        assert (solution / "figures" / figure).exists(), f"the starter README shows figures/{figure}, absent from solution/"
    assert "TODO" not in (solution / "README.md").read_text(encoding="utf-8")
    produced = {"test_indices.npy", "vault.json", "results.json", "results_rapide.json", "config.json",
                "config_rapide.json", "test_log.json", "runs"}                    # written by the learner's own notebook
    assert not produced & {p.name for p in starter.iterdir()}, "the starter kit must not ship the learner's outputs"
