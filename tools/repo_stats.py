#!/usr/bin/env python
"""Key figures of the workbook, counted in the repository (the « En chiffres » block of README.md).

    python tools/repo_stats.py              # print the figures
    python tools/repo_stats.py --write      # also rewrite the block of README.md between its markers
    python tools/repo_stats.py --no-tests   # skip the count of the tests (pytest --collect-only, ≈ 10 s)

How each figure is obtained:
- exercises: the ``exercises`` of the contract (``docs/syllabus/data/*.json``), by type, as ``syllabus.py stats``;
- automatic checks: the answers of ``src/wb/answers.json`` (one per checked value, sub-ids included);
- flashcards: the cards of every ``flashcards.csv`` that ``export_flashcards.py --check`` accepts;
- tests: ``python -m pytest --collect-only -q --impl=ref`` in ``tests/`` (``tests/infra/`` = the tooling);
- notebooks: the ``.ipynb`` files of the repository, outside ``mon_travail/``, ``data/`` and the caches;
- mylearn: the stub files of ``templates/mylearn_stubs/MANIFEST.json`` and their public functions and classes;
- index: the notions and cross-references of ``annexes/index_notions.md``, the terms of the glossary.
The block is not rewritten by ``syllabus.py build`` (counting the tests takes a while):
run ``--write`` after a chapter, a checkpoint or the final project, and commit README.md with it.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_flashcards  # noqa: E402
import syllabus  # noqa: E402

ROOT = syllabus.ROOT
BEGIN, END = "<!-- wb:readme chiffres -->", "<!-- wb:end chiffres -->"
SKIP_PARTS = {"mon_travail", "data", ".ipynb_checkpoints", "exports", ".git", "__pycache__"}


def count_exercises(chapters: dict) -> dict:
    by_type = collections.Counter(ex["type"] for ch in chapters.values() for ex in ch["exercises"])
    kinds = collections.Counter()
    for cid, ch in chapters.items():
        kind = "pf" if cid == "PF" else "cp" if syllabus.is_cp(cid) else "chapter"
        kinds[kind] += len(ch["exercises"])
    return {"total": sum(by_type.values()), "by_type": {t: by_type.get(t, 0) for t in syllabus.TYPES},
            "chapters": sum(1 for c in chapters if not syllabus.is_cp(c)),
            "in_chapters": kinds["chapter"], "in_checkpoints": kinds["cp"], "in_final_project": kinds["pf"],
            "checkpoints": sum(1 for c in chapters if c.startswith("CP"))}


def count_flashcards(root: Path = ROOT) -> dict:
    files = [p for pattern in export_flashcards.GLOBS for p in root.glob(pattern)]
    cards, problems = export_flashcards.collect(files, root, out=lambda *a: None)
    return {"cards": len(cards), "decks": len({c[3] for c in cards}), "problems": len(problems)}


def count_notebooks(root: Path = ROOT) -> dict:
    found = collections.Counter()
    for path in root.rglob("*.ipynb"):
        rel = path.relative_to(root)
        if SKIP_PARTS & set(rel.parts):
            continue
        found[rel.parts[0]] += 1
    return {"total": sum(found.values()), "chapitres": found["chapitres"], "checkpoints": found["checkpoints"],
            "projets": found["projets"], "00_setup": found["00_setup"]}


def count_mylearn(chapters: dict) -> dict:
    manifest = json.loads((syllabus.STUBS / "MANIFEST.json").read_text(encoding="utf-8"))
    files = [f for fs in manifest["chapters"].values() for f in fs if not f.endswith("__init__.py")]
    items = sum(len(syllabus.stub_api(f)) for f in files)
    return {"modules": len(files), "api": items}


def count_answers(root: Path = ROOT) -> int:
    data = json.loads((root / "src" / "wb" / "answers.json").read_text(encoding="utf-8"))
    return len(data.get("answers", data))


def count_index(root: Path = ROOT) -> dict:
    text = (root / "annexes" / "index_notions.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*(\d+) notions\*\*, et \*\*(\d+) renvois\*\*", text)
    glossary = sum(1 for line in (root / "annexes" / "glossaire.md").read_text(encoding="utf-8").splitlines()
                   if line.startswith("| ") and not line.startswith(("| Retenu", "|---")))
    return {"notions": int(m.group(1)) if m else 0, "renvois": int(m.group(2)) if m else 0, "glossary": glossary}


def count_tests(root: Path = ROOT) -> dict | None:
    """Tests collected by pytest in tests/ (reference implementation), split between mylearn and the tooling."""
    try:
        run = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "--impl=ref", "tests"],
                             cwd=root, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    ids = [line for line in run.stdout.splitlines() if "::" in line]
    if not ids:
        return None
    infra = sum(1 for line in ids if line.startswith("tests/infra/"))
    return {"total": len(ids), "infra": infra, "mylearn": len(ids) - infra}


def cheap_stats(chapters: dict | None = None, root: Path = ROOT) -> dict:
    """Every figure except the tests (a fraction of a second)."""
    chapters = chapters if chapters is not None else syllabus.load()
    return {"exercises": count_exercises(chapters), "answers": count_answers(root),
            "flashcards": count_flashcards(root), "notebooks": count_notebooks(root),
            "mylearn": count_mylearn(chapters), "index": count_index(root)}


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def render(stats: dict, tests: dict | None, today: dt.date | None = None) -> str:
    e, f, nb, ml, ix = (stats[k] for k in ("exercises", "flashcards", "notebooks", "mylearn", "index"))
    types = " · ".join(f"{t} {fmt(n)}" for t, n in e["by_type"].items() if n)
    tests_txt = (f"**{fmt(tests['total'])} tests** pytest : {fmt(tests['mylearn'])} pour `mylearn` (ton code, comparé "
                 f"à NumPy, scikit-learn ou PyTorch) et {fmt(tests['infra'])} pour l'outillage du workbook"
                 if tests else "tests : non comptés (relance `python tools/repo_stats.py --write`)")
    day = (today or dt.date.today()).isoformat()
    return "\n".join([
        BEGIN,
        "| | |",
        "|---|---|",
        f"| **Exercices** | **{fmt(e['total'])}** : {fmt(e['in_chapters'])} dans les {e['chapters']} chapitres "
        f"(0A, 0B, 1 à 29, B1 à B8), {fmt(e['in_checkpoints'])} dans les {e['checkpoints']} examens blancs, "
        f"{fmt(e['in_final_project'])} étapes du projet final |",
        f"| **Par type** | {types} |",
        f"| **Vérifications automatiques** | {fmt(stats['answers'])} réponses vérifiées par `wb.check` (empreinte, "
        "jamais la réponse en clair) |",
        f"| **Tests** | {tests_txt} |",
        f"| **Notebooks** | {fmt(nb['total'])} : {nb['chapitres']} dans les chapitres (exercices et solutions "
        f"exécutées), {nb['checkpoints']} pour les examens blancs, {nb['projets']} pour les mini-projets et le "
        f"projet final, {nb['00_setup']} pour la mise en place |",
        f"| **Flashcards** | {fmt(f['cards'])} cartes en {f['decks']} sous-paquets Anki |",
        f"| **mylearn** | {ml['modules']} modules, {fmt(ml['api'])} fonctions et classes publiques aux signatures figées |",
        f"| **Index et glossaire** | {fmt(ix['notions'])} notions et {fmt(ix['renvois'])} renvois, "
        f"{fmt(ix['glossary'])} termes au glossaire |",
        "",
        f"*Compté le {day} par `python tools/repo_stats.py --write` : exercices du contrat "
        "(`docs/syllabus/data/*.json`), réponses de `src/wb/answers.json`, cartes acceptées par "
        "`export_flashcards.py --check`, tests de `pytest --collect-only --impl=ref`, fichiers `.ipynb` hors "
        "`mon_travail/`, stubs de `MANIFEST.json`, en-tête de l'index.*",
        END,
    ])


def write_block(readme: Path, block: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        return False
    readme.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    return True


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="rewrite the « En chiffres » block of README.md")
    parser.add_argument("--no-tests", action="store_true", help="do not count the tests")
    args = parser.parse_args(argv)
    stats = cheap_stats()
    tests = None if args.no_tests else count_tests()
    block = render(stats, tests)
    print(block)
    if args.write:
        if not write_block(ROOT / "README.md", block):
            print(f"❌ README.md n'a pas les marqueurs {BEGIN} … {END}.")
            return 1
        print("✅ README.md : bloc « En chiffres » mis à jour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
