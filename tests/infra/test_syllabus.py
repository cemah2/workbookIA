"""The syllabus (docs/syllabus/data) is a contract: valid, and the documents derived from it are
up to date (BIBLE §22, session 2)."""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import syllabus  # noqa: E402


@pytest.fixture(scope="module")
def chapters():
    return syllabus.load()


def test_syllabus_is_valid(chapters):
    problems = syllabus.validate(chapters)
    assert problems == [], "\n".join(problems[:20])


def test_check_refuses_a_stale_check_field_and_a_prefixed_recall_title(chapters):
    import copy

    ch = copy.deepcopy(chapters["3"])
    all_ids = {ex["id"]: c for c in chapters.values() for ex in c["exercises"]}
    hammer = next(ex for ex in ch["exercises"] if ex["id"] == "3.16")
    hammer["check"] = "pytest"                       # 3.16 has answers in answers.json
    recall = next(ex for ex in ch["exercises"] if ex["type"] == "🔁")
    recall["title"] = "Ch. 2 : " + recall["title"]
    errors, _ = syllabus.validate_chapter(ch, all_ids)
    assert any(e.startswith("3.16:") and "wb.check" in e for e in errors), errors
    assert any(e.startswith(recall["id"] + ":") and "prefix" in e for e in errors), errors


def test_check_refuses_a_mini_project_function_that_a_track_does_not_write(chapters):
    import copy

    chs = copy.deepcopy(chapters)
    calibration = next(ex for ex in chs["3"]["exercises"] if ex["id"] == "3.28")
    calibration["tracks"] = ["maths", "code"]           # MP1 calls calibration_curve and brier_score (3.28)
    chs["CP1"]["miniproject"]["requires"].append("metrics.not_a_function")
    problems = syllabus.validate(chs, check_stubs=False)
    assert any(p.startswith("MP1 requires metrics.calibration_curve") and "3.28" in p and "rapide" in p
               for p in problems), problems
    assert any(p.startswith("MP1 requires metrics.not_a_function") and "no exercise" in p for p in problems), problems


def test_mylearn_owners_reads_every_format_of_the_field():
    fake = {"X": {"exercises": [
        {"id": "X.1", "mylearn": "metrics.py:accuracy,precision"},
        {"id": "X.2", "mylearn": "nn/activations.py:identity, step (+ _derivative)"},
        {"id": "X.3", "mylearn": "conv.py:conv_output_size, conv.py:pad2d"},
        {"id": "X.4", "mylearn": "assemblage sans nouvelle fonction : vae.py + optim.py (Adam)"},
        {"id": "X.5", "mylearn": "metrics.py:accuracy"},          # the first exercise that writes it wins
    ]}}
    assert syllabus.mylearn_owners(fake) == {
        "metrics.accuracy": "X.1", "metrics.precision": "X.1", "nn.activations.identity": "X.2",
        "nn.activations.step": "X.2", "conv.conv_output_size": "X.3", "conv.pad2d": "X.3"}


def test_reading_time_of_a_published_sheet_follows_its_length(chapters):
    ch3, ch12 = chapters["3"], chapters["12"]
    words = len((ROOT / "chapitres" / ch3["dir"] / "01_fiche.md").read_text(encoding="utf-8").split())
    assert syllabus.fiche_minutes(ch3) == 5 * round(words / syllabus.FICHE_WORDS_PER_MINUTE / 5)
    assert syllabus.reading_minutes(ch3) == (ch3["reading_minutes"] - syllabus.FICHE_MINUTES
                                             + syllabus.fiche_minutes(ch3))
    if not syllabus.published(ch12):                      # a sheet not written yet keeps the estimate
        assert syllabus.fiche_minutes(ch12) == syllabus.FICHE_MINUTES
        assert syllabus.reading_minutes(ch12) == ch12["reading_minutes"]
    assert syllabus.reading_minutes(chapters["0A"]) == chapters["0A"]["reading_minutes"]   # the sheet is the course


def test_published_fiches_mark_the_sections_of_the_fast_track(chapters):
    """A heading of the sheet ends with ⏩ exactly when its book section is read in the fast track (a
    track change must update the sheet). Sections without a heading of their own are not checked."""
    import re

    wrong = []
    for cid, ch in chapters.items():
        if syllabus.is_cp(cid) or not (ch.get("book") or {}).get("chapter") or not syllabus.published(ch):
            continue
        refs, fast = {s["ref"] for s in ch.get("sections", [])}, set(syllabus.rapide_sections(ch))
        fence = False
        for line in (ROOT / "chapitres" / ch["dir"] / "01_fiche.md").read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("```"):
                fence = not fence
                continue
            m = None if fence else re.match(r"^#{2,4} (\d+(?:\.\d+)+) · (.*)$", line)
            if not m:
                continue
            numbers = [m.group(1)] + re.findall(r", (\d+(?:\.\d+)+) ", m.group(2))    # « 2.5.1 · …, 2.5.2 … »
            marked = line.rstrip().endswith("⏩")
            wrong += [f"{cid}: §{n} {'marked' if marked else 'not marked'}" for n in numbers
                      if n in refs and marked != (n in fast)]
    assert not wrong, wrong


def test_chapter_order_is_shared_with_wb():
    from wb.impl import CHAPTER_ORDER

    assert tuple(syllabus.ORDER) == CHAPTER_ORDER


@pytest.mark.parametrize("rel, build", [
    ("docs/SYLLABUS.md", syllabus.build_syllabus),
    ("docs/PARCOURS.md", syllabus.build_parcours),
    ("suivi/tableau_de_bord.md", syllabus.build_dashboard),
])
def test_generated_documents_are_up_to_date(chapters, rel, build):
    on_disk = (ROOT / rel).read_text(encoding="utf-8")
    assert build(chapters) == on_disk, f"{rel} is outdated: run python tools/syllabus.py build"


def test_manifest_lists_every_planned_module(chapters):
    manifest = json.loads((syllabus.STUBS / "MANIFEST.json").read_text(encoding="utf-8"))
    listed = set(manifest["base"]) | {f for fs in manifest["chapters"].values() for f in fs}
    for cid, files in syllabus.manifest(chapters).items():
        for rel in files:
            assert rel in listed, f"{cid}: {rel} missing from MANIFEST.json"


def test_every_stub_module_imports(chapters):
    import wb

    wb.load_mylearn("stubs")
    try:
        for files in syllabus.manifest(chapters).values():
            for rel in files:
                name = "mylearn." + rel[:-3].replace("/", ".").removesuffix(".__init__")
                importlib.import_module(name)
    finally:
        wb.load_mylearn("ref")


def test_tracks_are_closed_on_code_prerequisites(chapters):
    exercises = {ex["id"]: ex for ch in chapters.values() for ex in ch["exercises"]}
    for ex in exercises.values():
        for p in ex["prereq"]:
            dep = exercises.get(p)
            if dep is not None and syllabus.is_code_prereq(dep, ex):
                assert set(ex["tracks"]) <= set(dep["tracks"]), f"{ex['id']} needs {p}"


def test_track_plans_add_up(chapters):
    totals = syllabus.track_totals(chapters)
    assert totals["complet"]["n"] == sum(len(ch["exercises"]) for ch in chapters.values())
    for key in syllabus.TRACKS:
        assert totals[key]["total"] < totals["complet"]["total"]
        assert totals[key]["n"] < totals["complet"]["n"]


def test_mylearn_tests_only_use_the_mylearn_module_fixture():
    """In learner mode a missing module comes from the reference: tests must go through
    ``mylearn_module`` (which skips modules the learner does not have), never through the
    ``mylearn`` fixture or a direct import."""
    import ast

    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = {a.arg for a in node.args.args}
                assert "mylearn" not in args, f"{path.name}:{node.name} uses the mylearn fixture"
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "mylearn" for a in node.names), path.name
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "mylearn", f"{path.name}: import from mylearn"
