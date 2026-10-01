"""Tests of the tools/ scripts on throw-away repositories (tmp_path)."""

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import build_answers  # noqa: E402
import export_flashcards  # noqa: E402
import run_all_notebooks  # noqa: E402
import start_chapter  # noqa: E402


def quiet(*args, **kwargs):
    pass


# ---------------------------------------------------------------- start_chapter
@pytest.mark.parametrize("raw, canonical, prefix", [
    ("18", "18", "ch18"), ("ch18", "18", "ch18"), ("3", "3", "ch03"), ("0A", "0A", "ch00a"),
    ("0b", "0B", "ch00b"), ("b3", "B3", "b3"), ("PF", "PF", "pf"), ("cp1", "CP1", "cp1"),
])
def test_canonical_ids(raw, canonical, prefix):
    assert start_chapter.canonical_id(raw) == canonical
    assert start_chapter.dir_prefix(canonical) == prefix


def test_canonical_id_invalid():
    with pytest.raises(ValueError):
        start_chapter.canonical_id("hello")


@pytest.fixture
def fake_repo(tmp_path):
    chapter = tmp_path / "chapitres" / "ch18_backprop"
    chapter.mkdir(parents=True)
    (chapter / "03_notebook.ipynb").write_text('{"cells": []}')
    (chapter / "06_mes_reponses.md").write_text("# Mes réponses\n")
    stubs = tmp_path / "templates" / "mylearn_stubs"
    (stubs / "nn").mkdir(parents=True)
    (stubs / "__init__.py").write_text("")
    (stubs / "_example.py").write_text("def mean(v): raise NotImplementedError\n")
    (stubs / "nn" / "__init__.py").write_text("")
    (stubs / "nn" / "backward.py").write_text("def backward(): raise NotImplementedError\n")
    (stubs / "MANIFEST.json").write_text(json.dumps({
        "base": ["__init__.py", "_example.py"], "chapters": {"18": ["nn/backward.py"]}}))
    return tmp_path


def test_start_chapter_copies_then_never_overwrites(fake_repo):
    assert start_chapter.start_chapter("18", root=fake_repo, out=quiet) == 0
    work = fake_repo / "mon_travail"
    expected = ["ch18_backprop/03_notebook.ipynb", "ch18_backprop/06_mes_reponses.md",
                "mylearn/__init__.py", "mylearn/_example.py", "mylearn/nn/__init__.py",
                "mylearn/nn/backward.py"]
    for rel in expected:
        assert (work / rel).exists(), rel
    # the learner works...
    (work / "mylearn" / "nn" / "backward.py").write_text("MY CODE")
    (work / "ch18_backprop" / "03_notebook.ipynb").write_text("MY NOTEBOOK")
    lines = []
    assert start_chapter.start_chapter("18", root=fake_repo, out=lines.append) == 0
    assert (work / "mylearn" / "nn" / "backward.py").read_text() == "MY CODE"
    assert (work / "ch18_backprop" / "03_notebook.ipynb").read_text() == "MY NOTEBOOK"
    assert any("conservé" in line for line in lines)


def test_start_chapter_dry_run_and_missing(fake_repo):
    assert start_chapter.start_chapter("18", root=fake_repo, dry_run=True, out=quiet) == 0
    assert not (fake_repo / "mon_travail").exists()
    lines = []
    assert start_chapter.start_chapter("7", root=fake_repo, out=lines.append) == 1
    assert any("n'existe pas encore" in line for line in lines)


def test_start_chapter_lists_earlier_modules_taken_from_the_reference(fake_repo):
    stubs = fake_repo / "templates" / "mylearn_stubs"
    (stubs / "tree.py").write_text("class DecisionTree: ...\n")
    manifest = json.loads((stubs / "MANIFEST.json").read_text())
    manifest["chapters"] = {"13": ["tree.py"], **manifest["chapters"]}
    (stubs / "MANIFEST.json").write_text(json.dumps(manifest))
    lines = []
    assert start_chapter.start_chapter("18", root=fake_repo, out=lines.append) == 0
    assert any("ch. 13 : tree.py" in line for line in lines)
    assert not (fake_repo / "mon_travail" / "mylearn" / "tree.py").exists()  # never copied
    (fake_repo / "mon_travail" / "mylearn" / "tree.py").write_text("MY TREE")
    lines = []
    start_chapter.start_chapter("18", root=fake_repo, out=lines.append)
    assert not any("ch. 13" in line for line in lines)


def test_start_chapter_in_progress_copies_everything_but_the_notebook(fake_repo):
    marker = fake_repo / "chapitres" / "ch18_backprop" / "EN_COURS.md"
    marker.write_text("in progress")
    work = fake_repo / "mon_travail" / "ch18_backprop"
    lines = []
    assert start_chapter.start_chapter("18", root=fake_repo, out=lines.append) == 0
    assert any("en cours de génération" in line for line in lines)
    assert (work / "06_mes_reponses.md").exists()
    assert not (work / "03_notebook.ipynb").exists()
    assert (fake_repo / "mon_travail" / "mylearn" / "nn" / "backward.py").exists()
    marker.unlink()  # the chapter is finished: running the same command copies the notebook
    assert start_chapter.start_chapter("18", root=fake_repo, out=quiet) == 0
    assert (work / "03_notebook.ipynb").exists()


def test_start_chapter_force_copies_a_notebook_in_progress(fake_repo):
    (fake_repo / "chapitres" / "ch18_backprop" / "EN_COURS.md").write_text("in progress")
    assert start_chapter.start_chapter("18", root=fake_repo, out=quiet, force=True) == 0
    assert (fake_repo / "mon_travail" / "ch18_backprop" / "03_notebook.ipynb").exists()


@pytest.fixture
def fake_checkpoint(fake_repo):
    exam = fake_repo / "checkpoints" / "partie_1"
    exam.mkdir(parents=True)
    for name in ("01_examen_sujet.md", "02_examen_notebook.ipynb", "03_examen_corrige.md", "04_mes_reponses.md"):
        (exam / name).write_text(name)
    project = fake_repo / "projets" / "partie_1_detecteur"
    for folder in ("depart", "depart/__pycache__", "solution"):
        (project / folder).mkdir(parents=True, exist_ok=True)
    (project / "README.md").write_text("brief")
    for name in ("notebook.ipynb", "langid.py", "test_langid.py", "README.md"):
        (project / "depart" / name).write_text(f"starter {name}")
        (project / "solution" / name).write_text(f"solution {name}")
    (project / "depart" / "__pycache__" / "langid.cpython-313.pyc").write_bytes(b"cache")
    return fake_repo


def test_start_chapter_checkpoint_copies_the_exam_and_the_starter_kit(fake_checkpoint):
    lines = []
    assert start_chapter.start_chapter("CP1", root=fake_checkpoint, out=lines.append) == 0
    work = fake_checkpoint / "mon_travail"
    for rel in ("checkpoints/partie_1/02_examen_notebook.ipynb", "checkpoints/partie_1/04_mes_reponses.md",
                "projets/partie_1_detecteur/notebook.ipynb", "projets/partie_1_detecteur/langid.py",
                "projets/partie_1_detecteur/test_langid.py", "projets/partie_1_detecteur/README.md",
                "mylearn/__init__.py", "mylearn/_example.py"):
        assert (work / rel).exists(), rel
    assert (work / "projets" / "partie_1_detecteur" / "langid.py").read_text() == "starter langid.py"
    assert not (work / "checkpoints" / "partie_1" / "03_examen_corrige.md").exists()   # the answers stay in the repo
    assert not (work / "projets" / "partie_1_detecteur" / "__pycache__").exists()
    assert not list(work.rglob("*solution*"))
    assert any(line.startswith("Checkpoint CP1") for line in lines)
    # the learner works, then runs the command again: nothing is overwritten
    (work / "projets" / "partie_1_detecteur" / "langid.py").write_text("MY MODULE")
    (work / "checkpoints" / "partie_1" / "04_mes_reponses.md").write_text("MY ANSWERS")
    assert start_chapter.start_chapter("cp1", root=fake_checkpoint, out=quiet) == 0
    assert (work / "projets" / "partie_1_detecteur" / "langid.py").read_text() == "MY MODULE"
    assert (work / "checkpoints" / "partie_1" / "04_mes_reponses.md").read_text() == "MY ANSWERS"


def test_start_chapter_checkpoint_not_generated_yet(fake_checkpoint):
    lines = []
    assert start_chapter.start_chapter("CP2", root=fake_checkpoint, out=lines.append) == 1
    assert any("n'existe pas encore" in line for line in lines)
    assert not (fake_checkpoint / "mon_travail").exists()


def test_start_chapter_init_only(fake_repo):
    assert start_chapter.start_chapter(None, root=fake_repo, out=quiet) == 0
    assert (fake_repo / "mon_travail" / "mylearn" / "_example.py").exists()
    assert not (fake_repo / "mon_travail" / "mylearn" / "nn").exists()


def test_real_manifest_is_consistent():
    manifest = start_chapter.load_manifest()
    stubs, ref = ROOT / "templates" / "mylearn_stubs", ROOT / "solutions" / "mylearn_ref"
    files = list(manifest["base"]) + [f for fs in manifest["chapters"].values() for f in fs]
    assert len(files) == len(set(files)), "a stub file is listed twice"
    for rel in files:
        assert (stubs / rel).exists(), f"stub {rel} missing"
    # the reference is written with the chapter: required once the chapter folder exists
    for cid, rels in manifest["chapters"].items():
        if start_chapter.find_chapter_dir(cid) is not None:
            for rel in rels:
                assert (ref / rel).exists(), f"reference {rel} missing (chapter {cid} is published)"
    for rel in manifest["base"]:
        assert (ref / rel).exists(), f"reference {rel} missing"


# ---------------------------------------------------------------- build_answers
def _answer_cell(ex_id, value, decimals=None, tagged=True):
    import wb

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        wb.record(ex_id, value, decimals)
    return {
        "cell_type": "code", "execution_count": 1, "source": ["wb.record(...)"],
        "metadata": {"tags": ["answer"] if tagged else []},
        "outputs": [{"output_type": "stream", "name": "stdout", "text": buffer.getvalue()}],
    }


def _write_nb(path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    return path


def test_build_answers_end_to_end(tmp_path):
    import wb

    nb = _write_nb(tmp_path / "chapitres" / "ch01_x" / "05_solutions.ipynb",
                   [_answer_cell("1.1", 42), _answer_cell("1.2", 0.5, 2),
                    _answer_cell("1.9", 7, tagged=False)])
    answers = tmp_path / "answers.json"
    lines = []
    assert build_answers.build([nb], answers, root=tmp_path, out=lines.append) == 0
    data = json.loads(answers.read_text())["answers"]
    assert set(data) == {"1.1", "1.2"}
    assert data["1.1"]["source"] == "chapitres/ch01_x/05_solutions.ipynb"
    assert any("non taguée" in line for line in lines)
    from wb import checker as check_module

    check_module.RECORDED.clear()  # force the use of the answers file
    assert wb.check("1.1", 42, answers_path=answers, quiet=True)
    assert not wb.check("1.1", 41, answers_path=answers, quiet=True)
    assert build_answers.build([nb], answers, root=tmp_path, check_only=True, out=quiet) == 0
    # removing an answer from the notebook removes it from answers.json
    _write_nb(nb, [_answer_cell("1.1", 42)])
    assert build_answers.build([nb], answers, root=tmp_path, check_only=True, out=quiet) == 1
    build_answers.build([nb], answers, root=tmp_path, out=quiet)
    assert set(json.loads(answers.read_text())["answers"]) == {"1.1"}


def test_build_answers_errors(tmp_path):
    cell = _answer_cell("2.1", 3)
    cell["outputs"] = []
    nb = _write_nb(tmp_path / "a.ipynb", [cell])
    with pytest.raises(build_answers.BuildError, match="exécuté"):
        build_answers.build([nb], tmp_path / "answers.json", root=tmp_path, out=quiet)
    nb1 = _write_nb(tmp_path / "b.ipynb", [_answer_cell("2.2", 3)])
    nb2 = _write_nb(tmp_path / "c.ipynb", [_answer_cell("2.2", 4)])
    with pytest.raises(build_answers.BuildError, match="deux"):
        build_answers.build([nb1, nb2], tmp_path / "answers.json", root=tmp_path, out=quiet)


def test_natural_sort():
    ids = ["10.2", "2.10", "2.9", "0A.1", "B3.1"]
    assert sorted(ids, key=build_answers.natural_key) == ["0A.1", "2.9", "2.10", "10.2", "B3.1"]


# ---------------------------------------------------------------- export_flashcards
def test_export_flashcards(tmp_path):
    for name, rows in {
        "ch02_stats": 'front;back;tags\n"Moyenne de 1, 2, 3 ?";2;dlwb::ch02::mean\n',
        "ch03_proba": "P(A) si A certain ?;1;dlwb::ch03::proba\nbad row without columns\n",
    }.items():
        folder = tmp_path / "chapitres" / name
        folder.mkdir(parents=True)
        (folder / "flashcards.csv").write_text(rows, encoding="utf-8")
    out_file = tmp_path / "deck.csv"
    lines = []
    code = export_flashcards.export(out_file, root=tmp_path, out=lines.append)
    assert code == 1 and any("colonne" in line for line in lines)
    text = out_file.read_text(encoding="utf-8")
    assert text.startswith("#separator:semicolon\n#html:true\n#tags column:3\n")
    assert "dlwb::ch02::mean" in text and "dlwb::ch03::proba" in text
    assert export_flashcards.export(out_file, root=tmp_path, check_only=True, out=quiet) == 1


# ---------------------------------------------------------------- run_all_notebooks
def _code(src):
    import uuid

    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {},
            "outputs": [], "source": src}


def _notebook(tmp_path, name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                      "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / name
    path.write_text(json.dumps(nb))
    return path


@pytest.mark.parametrize("engine", ["inprocess", "nbclient"])
def test_run_notebooks_engines(tmp_path, engine, monkeypatch):
    if engine == "nbclient":
        pytest.importorskip("nbclient")
        pytest.importorskip("ipykernel")
    # a backend exported in the shell (MPLBACKEND=Agg for scripts) must not stop figures being saved
    monkeypatch.setenv("MPLBACKEND", "Agg")
    good = _notebook(tmp_path, "good.ipynb", [
        _code("import os\nx = 21 * 2\nprint('fast', os.environ.get('WB_FAST_MODE'))"),
        _code("import matplotlib.pyplot as plt\nplt.plot([1, 2])\nx"),
    ])
    bad = _notebook(tmp_path, "bad.ipynb", [_code("y = 1"), _code("1 / 0"), _code("z = 2")])
    code = run_all_notebooks.main([str(good), str(bad), "--engine", engine, "--inplace",
                                   "--report", str(tmp_path / "report.md")])
    assert code == 1
    assert os.environ["MPLBACKEND"] == "Agg"  # the caller's environment is restored
    done = json.loads(good.read_text())
    outputs = [o for c in done["cells"] for o in c["outputs"]]
    assert any("fast 1" in "".join(o.get("text", "")) for o in outputs)
    assert any("image/png" in o.get("data", {}) for o in outputs)
    assert any("".join(o.get("data", {}).get("text/plain", "")) == "42" for o in outputs)
    report = (tmp_path / "report.md").read_text()
    assert "bad.ipynb" in report and "ZeroDivisionError" in report


def test_run_notebooks_skips_learner_space(tmp_path):
    folder = tmp_path / "mon_travail"
    folder.mkdir()
    _notebook(folder, "mine.ipynb", [_code("1")])
    assert run_all_notebooks.collect([str(folder)]) == []


def test_start_chapter_trackers_copy_then_append(fake_repo):
    suivi = fake_repo / "suivi"
    suivi.mkdir()
    (suivi / "tableau_de_bord.md").write_text(
        "# Tableau\n\n> **Modèle tenu par Claude : ne coche pas ici.**\n\n"
        "<!-- wb:section setup -->\n- [ ] demo\n<!-- wb:end setup -->\n", encoding="utf-8")
    (suivi / "journal.md").write_text("# Journal\n", encoding="utf-8")
    assert start_chapter.start_chapter(None, root=fake_repo, out=quiet) == 0
    mine = fake_repo / "mon_travail" / "suivi" / "tableau_de_bord.md"
    text = mine.read_text(encoding="utf-8")
    assert "Modèle tenu par Claude" not in text and "- [ ] demo" in text
    mine.write_text(text.replace("- [ ] demo", "- [x] demo"), encoding="utf-8")  # the learner ticks
    with (suivi / "tableau_de_bord.md").open("a", encoding="utf-8") as handle:  # Claude publishes ch 18
        handle.write("\n<!-- wb:section 18 -->\n## Ch. 18\n- [ ] Ex 18.1\n<!-- wb:end 18 -->\n")
    lines = []
    assert start_chapter.start_chapter("18", root=fake_repo, out=lines.append) == 0
    text = mine.read_text(encoding="utf-8")
    assert "- [x] demo" in text and "- [ ] Ex 18.1" in text and text.count("wb:section setup") == 1
    assert any("complété" in line for line in lines)
    start_chapter.start_chapter("18", root=fake_repo, out=quiet)  # idempotent
    assert mine.read_text(encoding="utf-8").count("wb:section 18") == 1


def test_copy_no_overwrite_cleans_up_on_failure(tmp_path, monkeypatch):
    src = tmp_path / "src.txt"
    src.write_text("data")
    dst = tmp_path / "out" / "dst.txt"
    real_open = Path.open

    def failing_open(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if self == dst and "x" in mode:
            handle.close()
            raise OSError("disk full")
        return handle

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError):
        start_chapter.copy_no_overwrite(src, dst, dry_run=False)
    assert not dst.exists()


def test_run_notebooks_collect_glob_matching_folders(tmp_path, monkeypatch):
    folder = tmp_path / "chapitres" / "ch03_proba"
    folder.mkdir(parents=True)
    _notebook(folder, "03_notebook.ipynb", [_code("1")])
    monkeypatch.chdir(tmp_path)
    found = run_all_notebooks.collect(["chapitres/ch03_*"])
    assert [p.name for p in found] == ["03_notebook.ipynb"]
