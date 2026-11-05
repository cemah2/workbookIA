"""Guards on the final project (projets/projet_final/): the brief follows the contract, the starter kit works.

There is no reference solution: the learner chooses the dataset. What can be checked is the kit (its structure
tests pass, every function to write raises the NotImplementedError of its step, « Run all » on its notebooks ends
in ⏳), and that the documents follow the contract docs/syllabus/data/pf.json (13 steps, their headers, their
deliverables, three hints each, the bank of interview questions up to date).
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "projets" / "projet_final"
KIT = FINAL / "depart"
CONTRACT = json.loads((ROOT / "docs" / "syllabus" / "data" / "pf.json").read_text(encoding="utf-8"))
STEPS = CONTRACT["exercises"]
sys.path.insert(0, str(ROOT / "tools" / "chapters"))


def _header(step: dict) -> str:
    gpu = " 🚀" if step["gpu"] else ""
    return f"### {step['id']} — {step['title']} {step['type']} {'★' * step['stars']} ⏱️ {step['minutes']} min{gpu}"


def test_the_brief_has_the_thirteen_steps_of_the_contract_with_their_deliverables():
    readme = (FINAL / "README.md").read_text(encoding="utf-8")
    missing = [s["id"] for s in STEPS if _header(s) not in readme]
    assert not missing, f"expected the header « {_header(STEPS[0])} » (format of the contract) for: {missing}"
    for step in STEPS:
        section = readme.split(_header(step))[1].split("\n### ")[0]
        deliverable = re.sub(r"(\S+/\S*|\S+\.(?:md|py|ipynb|toml|yml|jsonl))", r"`\1`", step["deliverable"])
        assert step["deliverable"] in section.replace("`", "") or deliverable in section, \
            f"expected the deliverable of {step['id']} written as in the contract: {step['deliverable']}"
        points = f"{step['points']:g}".replace(".", ",")
        assert f"**Points :** {points}" in section, f"expected « **Points :** {points} » in {step['id']}"


def test_the_grid_gives_each_step_its_points_and_sums_to_twenty():
    grid = (FINAL / "GRILLE.md").read_text(encoding="utf-8")
    for step in STEPS:
        block = grid.split(f"## {step['id']} — ")[1].split("\n## ")[0]
        points = [float(m.replace(",", ".")) for m in re.findall(r"^\| \[ \] \|[^|]+\| ([0-9,]+) \|", block, re.M)]
        assert abs(sum(points) - step["points"]) < 1e-9, \
            f"expected the criteria of {step['id']} to sum to {step['points']} points, got {sum(points)}"
    assert sum(s["points"] for s in STEPS) == 20, "expected the contract to give 20 points"


def test_the_hints_give_three_levels_for_each_step():
    hints = (FINAL / "04_indices.md").read_text(encoding="utf-8")
    for step in STEPS:
        block = hints.split(f"### {step['id']} — {step['title']}\n")[1].split("\n### ")[0]
        levels = re.findall(r"<summary>Indice (\d)</summary>", block)
        assert levels == ["1", "2", "3"], f"expected the hints 1, 2 and 3 for {step['id']}, got {levels}"


def test_the_bank_of_questions_and_the_copy_of_the_grid_are_up_to_date(capsys):
    import build_pf

    assert build_pf.main(["--check"]) == 0, capsys.readouterr().out + " : run python tools/chapters/build_pf.py"
    bank = build_pf.bank()
    for cid in CONTRACT["chapters"]:
        assert re.search(rf"\*\*{re.escape(cid)}\.E\d+\*\*", bank), f"expected the 💼 questions of chapter {cid}"
    assert "Réponse modèle" not in bank, "the bank lists the questions, never the model answers"


def _kit_functions():
    """(module, function name, function node) of the kit's package, methods included."""
    for path in sorted((KIT / "src" / "projet").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        yield path.stem, f"{node.name}.{item.name}", item
            elif isinstance(node, ast.FunctionDef) and node in tree.body:
                yield path.stem, node.name, node


def test_every_function_to_write_raises_the_todo_of_its_step():
    wired = {"config", "cli", "__main__", "__init__"}                  # provided modules (the wiring)
    problems = []
    for module, name, node in _kit_functions():
        doc = ast.get_docstring(node) or ""
        if module in wired or "Provided" in doc or name.startswith("_") or name.endswith("__init__"):
            continue
        last = node.body[-1]
        text = ast.unparse(last) if isinstance(last, ast.Raise) else ""
        if not re.fullmatch(rf"raise NotImplementedError\('TODO PF\.\d+: {re.escape(name)}'\)", text):
            problems.append(f"{module}.{name}")
        if not doc:
            problems.append(f"{module}.{name} (no docstring)")
    assert not problems, f"expected raise NotImplementedError('TODO PF.n: <name>') and a docstring in: {problems}"


def _copy_of_the_kit(tmp_path: Path) -> Path:
    copy = tmp_path / "kit"
    shutil.copytree(KIT, copy, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info"))
    return copy


def test_the_structure_tests_of_the_kit_pass_and_the_others_wait(tmp_path):
    copy = _copy_of_the_kit(tmp_path)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "COLUMNS": "1000"}
    env.pop("PF_STRICT", None)
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=copy,
                         capture_output=True, text=True, env=env, timeout=300)
    assert run.returncode == 0, f"expected the tests of the empty kit to pass or skip:\n{run.stdout[-3000:]}"
    assert "failed" not in run.stdout and " passed" in run.stdout and "⏳ pas encore écrit" in run.stdout, \
        f"expected passed structure tests and ⏳ skips:\n{run.stdout[-2000:]}"
    strict = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=copy,
                            capture_output=True, text=True, env={**env, "PF_STRICT": "1"}, timeout=300)
    assert strict.returncode != 0, "expected PF_STRICT=1 to fail while functions are still to write"


@pytest.mark.slow
def test_run_all_on_the_empty_notebooks_ends_in_pending(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    copy = _copy_of_the_kit(tmp_path)
    for path in sorted((copy / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbclient.NotebookClient(nb, timeout=300, kernel_name="python3",
                                resources={"metadata": {"path": str(path.parent)}}).execute()
        outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
        errors = [f"{o['ename']}: {o['evalue']}" for o in outputs if o.get("output_type") == "error"]
        text = "".join(o.get("text", "") for o in outputs if o.get("output_type") == "stream")
        assert not errors, f"{path.name}: expected no error on the empty kit, got {errors}"
        assert "⏳" in text, f"{path.name}: expected ⏳ lines on the empty kit"
        assert "🔓" not in text, f"{path.name}: the sealed test must stay closed on the empty kit"
    assert not (copy / "experiments" / "test_access.jsonl").exists(), "the empty run must not open the test"
