"""Guards on the mini-projects (projets/): the reference solution passes its tests, the starter kit mirrors it."""

import ast
import filecmp
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = sorted(p for p in (ROOT / "projets").glob("partie_*") if (p / "solution").is_dir())


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
def test_starter_kit_mirrors_the_solution(project):
    starter, solution = project / "depart", project / "solution"
    for name in ("data.py", "conftest.py", "test_langid.py", "langid.py", "notebook.ipynb", "README.md"):
        assert (starter / name).exists(), f"depart/{name} missing"
    assert filecmp.cmp(starter / "data.py", solution / "data.py", shallow=False), "data.py differs between depart/ and solution/"
    assert _public_api(starter / "langid.py") == _public_api(solution / "langid.py"), "the starter API differs"
    tree = ast.parse((starter / "langid.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            assert "NotImplementedError" in ast.dump(node), f"depart/langid.py: {node.name} must raise NotImplementedError"
    readme = (starter / "README.md").read_text(encoding="utf-8")
    assert "TODO" in readme and "figures/accuracy_auc_longueur.png" in readme
    assert "TODO" not in (solution / "README.md").read_text(encoding="utf-8")
