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
