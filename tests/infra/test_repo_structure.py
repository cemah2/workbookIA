"""Guards on the repository layout and conventions (BIBLE §7, §12, §13)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

REQUIRED = [
    "README.md", "requirements.txt", "pyproject.toml", ".gitignore",
    "docs/BIBLE.md", "docs/METHODE.md", "docs/SYLLABUS.md", "docs/PARCOURS.md",
    "00_setup/check_env.py", "00_setup/demo.ipynb", "00_setup/INSTALL_LOCAL.md", "00_setup/COLAB.md",
    "src/wb/__init__.py", "src/wb/answers.json",
    "templates/mylearn_stubs/__init__.py", "templates/mylearn_stubs/MANIFEST.json",
    "solutions/mylearn_ref/__init__.py", "tests/conftest.py",
    "data/cards/README.md", "chapitres", "checkpoints", "projets",
    "annexes/glossaire.md", "annexes/formulaire.md", "annexes/erreurs_frequentes.md",
    "annexes/metiers.md", "annexes/ressources.md",
    "annexes/cheatsheets/numpy.md", "annexes/cheatsheets/pandas.md", "annexes/cheatsheets/sklearn.md",
    "annexes/cheatsheets/pytorch.md", "annexes/cheatsheets/git.md",
    "suivi/PROGRESS.md", "suivi/tableau_de_bord.md", "suivi/journal.md",
    "suivi/auto_evaluation.md", "suivi/remediation.md",
    "tools/build_answers.py", "tools/start_chapter.py", "tools/run_all_notebooks.py",
    "tools/export_flashcards.py", "mon_travail",
]


@pytest.mark.parametrize("rel", REQUIRED)
def test_required_paths_exist(rel):
    assert (ROOT / rel).exists(), rel


def test_gitignore_blocks_pdfs_and_heavy_files():
    text = (ROOT / ".gitignore").read_text()
    for pattern in ("*.pdf", "data/downloads/", "*.pt", "__pycache__/", ".ipynb_checkpoints/"):
        assert pattern in text, pattern


def test_no_pdf_tracked():
    proc = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("not a git checkout")
    assert not [f for f in proc.stdout.splitlines() if f.lower().endswith(".pdf")]


def test_answers_json_is_valid_and_up_to_date():
    data = json.loads((ROOT / "src" / "wb" / "answers.json").read_text(encoding="utf-8"))
    for ex_id, entry in data["answers"].items():
        assert {"hash", "kind", "source"} <= set(entry), ex_id
        assert (ROOT / entry["source"]).exists(), f"{ex_id}: source notebook missing"
    import build_answers

    code = build_answers.build(build_answers.default_notebooks(), check_only=True, out=lambda *a: None)
    assert code == 0, "answers.json is outdated: run python tools/build_answers.py"


def test_published_flashcards_pass_the_check():
    import export_flashcards

    lines = []
    code = export_flashcards.export(ROOT / "exports" / "unused.csv", check_only=True, out=lines.append)
    problems = [str(line) for line in lines if str(line).startswith("⚠️")]
    assert code == 0, f"expected no flashcard problem, got {len(problems)}: " + " | ".join(problems)


def test_hammer_headers_name_their_mylearn_file():
    """BIBLE §11: the header of a 🔨 that completes mylearn names its file (« **mylearn :** `metrics.py` »)."""
    missing = []
    for data in sorted((ROOT / "docs" / "syllabus" / "data").glob("ch*.json")):
        chapter = json.loads(data.read_text(encoding="utf-8"))
        notebook = ROOT / "chapitres" / chapter["dir"] / "03_notebook.ipynb"
        if not notebook.exists() or (notebook.parent / "EN_COURS.md").exists():
            continue
        headers = {}
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
            text = "".join(cell["source"])
            if cell["cell_type"] == "markdown" and text.startswith("### Ex "):
                headers[text.split(" — ")[0].removeprefix("### Ex ")] = text.split("\n\n")[0]
        for ex in chapter["exercises"]:
            if ex.get("mylearn") and ex["file"] == "03":
                file = ex["mylearn"].split(":")[0]
                if f"**mylearn :** `{file}`" not in headers.get(ex["id"], ""):
                    missing.append(f"{ex['id']} ({file})")
    assert not missing, "expected « **mylearn :** `file` » in the header of: " + ", ".join(missing)


def _authored_notebooks():
    paths = list((ROOT / "00_setup").glob("*.ipynb"))
    paths += list((ROOT / "chapitres").glob("*/*.ipynb"))
    paths += list((ROOT / "checkpoints").glob("*/*.ipynb"))
    paths += list((ROOT / "projets").glob("*/*/*.ipynb"))      # mini-projects: depart/ and solution/
    return paths


def _expected_chapter(path):
    """The chapter a notebook belongs to: from its folder (ch18_backprop -> 18; partie_1... -> CP1)."""
    import re

    import start_chapter

    top, folder = path.relative_to(ROOT).parts[:2]
    if top in ("checkpoints", "projets"):
        return "CP" + re.match(r"partie_(\d)", folder).group(1)
    return start_chapter.canonical_id(path.parent.name.split("_")[0])


def test_notebooks_use_the_standard_setup_cell_and_no_magics():
    import nbbuild

    import re

    template = {"exercise": nbbuild.setup_cell("exercise", chapter="XX").source,
                "solution": nbbuild.setup_cell("solution").source,
                "demo": nbbuild.setup_cell("demo").source}
    for path in _authored_notebooks():
        nb = json.loads(path.read_text(encoding="utf-8"))
        setup = [c for c in nb["cells"] if "setup" in c.get("metadata", {}).get("tags", [])]
        assert len(setup) == 1, f"{path.name}: exactly one cell tagged 'setup' expected"
        raw = "".join(setup[0]["source"])
        source = re.sub(r'chapter="[^"]*"', 'chapter="XX"', raw)
        assert source in template.values(), f"{path.name}: setup cell differs from the template"
        if 'chapter="' in raw:  # the fallback must be limited to the chapters before this one
            expected = _expected_chapter(path)
            assert f'chapter="{expected}"' in raw, f"{path}: setup cell must say chapter=\"{expected}\""
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                lines = "".join(cell["source"]).splitlines()
                assert not any(l.lstrip().startswith(("!", "%")) for l in lines), f"{path.name}: magic"


def test_executed_notebooks_have_no_errors():
    for path in _authored_notebooks():
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", f"{path.name}: saved error output"


def published_stub_files() -> set[str]:
    """Stub files whose chapter is generated (its folder exists), plus the base files.

    The interface of every chapter is frozen from session 2 on (all stubs exist),
    but the reference implementation is written with the chapter.
    """
    import start_chapter

    manifest = start_chapter.load_manifest()
    files = set(manifest["base"])
    for cid, rels in manifest["chapters"].items():
        if start_chapter.find_chapter_dir(cid) is not None:
            files.update(rels)
    return files


def test_stubs_raise_not_implemented_and_ref_mirrors_them():
    import ast

    stubs = ROOT / "templates" / "mylearn_stubs"
    ref = ROOT / "solutions" / "mylearn_ref"
    published = published_stub_files()
    for stub in stubs.rglob("*.py"):
        rel = stub.relative_to(stubs)
        tree = ast.parse(stub.read_text(encoding="utf-8"))
        if rel.as_posix() in published:
            assert (ref / rel).exists(), f"no reference for {rel}"
            ref_tree = ast.parse((ref / rel).read_text(encoding="utf-8"))
            stub_funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            ref_funcs = {n.name for n in ast.walk(ref_tree) if isinstance(n, ast.FunctionDef)}
            assert stub_funcs <= ref_funcs, f"{rel}: functions missing in the reference"
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                if "Provided:" in (ast.get_docstring(node) or ""):
                    continue  # helper given to the learner, already implemented
                body = ast.dump(node)
                assert "NotImplementedError" in body, f"{rel}:{node.name} must raise NotImplementedError"
