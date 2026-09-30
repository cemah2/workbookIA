#!/usr/bin/env python
"""Validate the syllabus data and generate the documents that derive from it.

The syllabus is a contract (BIBLE §10-§11): one JSON file per chapter in
``docs/syllabus/data/`` (exercises, mylearn API, sections of the book...).
mylearn signatures are read from the stubs in ``templates/mylearn_stubs/``
(the single source of truth for the interface).

    python tools/syllabus.py check     # validate data + stubs, exit 1 on error
    python tools/syllabus.py build     # regenerate the files below
    python tools/syllabus.py stats     # totals per type and per part

Generated files: docs/SYLLABUS.md, docs/PARCOURS.md, suivi/tableau_de_bord.md,
templates/mylearn_stubs/MANIFEST.json (chapter -> stub files; additive only).
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "syllabus" / "data"
STUBS = ROOT / "templates" / "mylearn_stubs"

ORDER = ["0A", "0B", "1", "2", "3", "4", "5", "6", "CP1", "7", "8", "9", "10", "11", "CP2",
         "12", "13", "14", "15", "CP3", "16", "17", "18", "19", "20", "CP4",
         "21", "22", "23", "24", "CP5", "25", "26", "27", "28", "29", "CP6",
         "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "PF"]
PARTS = {"0": ["0A", "0B"], "I": ["1", "2", "3", "4", "5", "6", "CP1"],
         "II": ["7", "8", "9", "10", "11", "CP2"], "III": ["12", "13", "14", "15", "CP3"],
         "IV": ["16", "17", "18", "19", "20", "CP4"], "V": ["21", "22", "23", "24", "CP5"],
         "VI": ["25", "26", "27", "28", "29", "CP6"],
         "VII": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"], "PF": ["PF"]}
PART_TITLES = {"0": "Prérequis", "I": "Fondations", "II": "Concepts", "III": "ML classique",
               "IV": "Réseaux", "V": "Architectures", "VI": "Génératif et RL",
               "VII": "Bonus", "PF": "Projet final"}
TYPES = ["🧠", "🔁", "✏️", "∂", "🔨", "📦", "🔬", "🔮", "🐛", "📈", "🧮", "🗣️", "⚖️", "📄", "🎨",
         "🏆", "💼", "🛠️"]
TYPE_NAMES = {"🧠": "quiz", "🔁": "rappel", "✏️": "calcul", "∂": "démonstration",
              "🔨": "from scratch", "📦": "librairie", "🔬": "expérience", "🔮": "prédiction",
              "🐛": "bug", "📈": "graphique", "🧮": "Fermi", "🗣️": "Feynman", "⚖️": "éthique",
              "📄": "article", "🎨": "figure", "🏆": "défi", "💼": "entretien", "🛠️": "pro"}
FILE_OF = {"🧠": "02", "🔁": "02", "✏️": "02", "∂": "02", "🧮": "02", "🗣️": "02", "⚖️": "02",
           "📄": "02", "💼": "02", "🔨": "03", "📦": "03", "🔬": "03", "🔮": "03", "🐛": "03",
           "🎨": "03", "🏆": "03"}
TRACKS = ("rapide", "maths", "code")
CHECKS = {"wb.check", "pytest", "manual", "wb.check+pytest"}
FLASHCARD_MINUTES = 2      # first pass on each card (creation / import + first review)
SYNTHESIS_MINUTES = 90     # revision sheet + mind map of a checkpoint
STUDY_START = dt.date(2026, 10, 5)
HOURS_PER_WEEK = 10
SOLUTION_READING = 1 / 3   # reading the worked solution of a skipped prerequisite
FICHE_MINUTES = 30         # the chapter sheet (01_fiche.md), read in full by every track


def is_code_prereq(dep: dict, ex: dict) -> bool:
    """A prerequisite is *hard* when both exercises are notebook (03) exercises.

    The later exercise then reuses code, data or a mylearn function built by the
    earlier one, so a track that contains the later exercise must contain the earlier
    one (checked by ``validate``). Any other prerequisite is *soft* (knowledge): a
    learner who skipped it reads its worked solution (``05_solutions``) instead.
    """
    return dep["file"] == "03" and ex["file"] == "03"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(data_dir: Path = DATA) -> dict[str, dict]:
    chapters = {}
    for path in sorted(data_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_file"] = path.name
        chapters[data["id"]] = data
    return {cid: chapters[cid] for cid in ORDER if cid in chapters}


def is_cp(cid: str) -> bool:
    return cid.startswith("CP") or cid == "PF"


def part_of(cid: str) -> str:
    for part, ids in PARTS.items():
        if cid in ids:
            return part
    raise KeyError(cid)


def rank(cid: str) -> int:
    return ORDER.index(cid)


def exercise_chapter(ex_id: str) -> str:
    return ex_id.split(".")[0]


def minutes_of(ch: dict) -> dict:
    """Study time of a chapter, in minutes, by component."""
    ex = sum(e["minutes"] for e in ch["exercises"])
    if is_cp(ch["id"]):
        mp = (ch.get("miniproject") or {}).get("hours", 0) * 60
        synth = SYNTHESIS_MINUTES if ch.get("synthesis") else 0
        return {"reading": synth, "exercises": ex, "flashcards": 0, "project": mp,
                "total": ex + mp + synth}
    reading = ch.get("reading_minutes", 0)
    cards = ch.get("flashcards", 0) * FLASHCARD_MINUTES
    return {"reading": reading, "exercises": ex, "flashcards": cards, "project": 0,
            "total": reading + ex + cards}


def rapide_sections(ch: dict) -> list[str]:
    """Book sections read in the fast track.

    Those covered by one of its practice exercises (quizzes, recalls and interview
    questions do not count: the chapter's summary sheet is enough for them).
    """
    covered = {c for ex in ch["exercises"] if "rapide" in ex["tracks"]
               and ex["type"] not in ("🧠", "🔁", "💼") for c in ex["covers"]}
    return [s["ref"] for s in ch.get("sections", []) if s["ref"] in covered]


def rapide_reading(ch: dict) -> int:
    """Reading time of the fast track: the whole sheet, and the book sections of rapide_sections.

    Chapters without a book chapter (0A, 0B, bonus) are read in full: the sheet is the course.
    """
    reading = ch.get("reading_minutes", 0)
    secs = ch.get("sections", [])
    if not (ch.get("book") or {}).get("chapter") or not secs:
        return reading
    book = max(reading - FICHE_MINUTES, 0)
    return round(min(reading, FICHE_MINUTES) + book * len(rapide_sections(ch)) / len(secs))


def track_plan(chapters: dict, track: str) -> dict[str, dict]:
    """Per-chapter plan of a track ("complet", "rapide", "maths" or "code").

    ids: exercises of the track · read: soft prerequisites outside the track, whose
    worked solution is read (SOLUTION_READING of their time) · reading: the whole
    reading, except in the fast track, which reads only the book sections covered by
    its exercises · flashcards, synthesis and mini-projects are common to all tracks.
    """
    exercises = {ex["id"]: ex for ch in chapters.values() for ex in ch["exercises"]}
    selected = {eid for eid, ex in exercises.items() if track == "complet" or track in ex["tracks"]}
    to_read = collections.defaultdict(set)
    for eid in selected:
        for p in exercises[eid]["prereq"]:
            if p in exercises and p not in selected:
                to_read[exercise_chapter(p)].add(p)
    plan = {}
    for cid, ch in chapters.items():
        m = minutes_of(ch)
        ids = [ex["id"] for ex in ch["exercises"] if ex["id"] in selected]
        read = [ex["id"] for ex in ch["exercises"] if ex["id"] in to_read[cid]]
        reading, project = m["reading"], m["project"]
        if is_cp(cid):  # the synthesis of a checkpoint counts with the projects
            reading, project = 0, m["project"] + m["reading"]
        elif track == "rapide":
            reading = rapide_reading(ch)
        ex_min = sum(exercises[i]["minutes"] for i in ids)
        sol_min = round(SOLUTION_READING * sum(exercises[i]["minutes"] for i in read))
        plan[cid] = {"ids": ids, "read": read, "exercises": ex_min, "solutions": sol_min,
                     "reading": reading, "flashcards": m["flashcards"], "project": project,
                     "total": ex_min + sol_min + reading + m["flashcards"] + project}
    return plan


# ---------------------------------------------------------------------------
# Stubs (signatures)
# ---------------------------------------------------------------------------
def _signature(node: ast.FunctionDef) -> str:
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"def {node.name}({args}){ret}"


def _summary(node) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.strip().split("\n")[0] if doc else ""


def stub_api(rel: str) -> list[dict]:
    """Public API of a stub file: functions, classes and their public methods."""
    path = STUBS / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    items = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node) or ""
            items.append({"kind": "function", "name": node.name, "signature": _signature(node),
                          "summary": _summary(node), "provided": "Provided:" in doc})
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            methods = []
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and (
                        not sub.name.startswith("_") or sub.name in ("__init__", "__call__")
                        or (sub.name.startswith("__") and sub.name.endswith("__")
                            and sub.name not in ("__repr__",))):
                    methods.append({"name": sub.name, "signature": _signature(sub),
                                    "summary": _summary(sub)})
            items.append({"kind": "class", "name": node.name,
                          "signature": f"class {node.name}({bases})" if bases else f"class {node.name}",
                          "summary": _summary(node), "methods": methods})
    return items


def stub_imports(rel: str) -> list[str]:
    """Stub files imported (relative imports) by a stub file."""
    path = STUBS / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    here = Path(rel).parent
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            base = here
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                target = base / (node.module.replace(".", "/") + ".py")
                if (STUBS / target).exists():
                    out.append(target.as_posix())
                elif (STUBS / base / node.module.replace(".", "/") / "__init__.py").exists():
                    out.append((base / node.module.replace(".", "/") / "__init__.py").as_posix())
            else:
                for alias in node.names:
                    target = base / f"{alias.name}.py"
                    if (STUBS / target).exists():
                        out.append(target.as_posix())
    return sorted(set(out))


def manifest(chapters: dict) -> dict:
    """chapter id -> stub files, in chapter order (sub-package __init__ first)."""
    out = {}
    for cid, ch in chapters.items():
        files = [m["file"] for m in ch.get("mylearn", [])]
        if any(f.startswith("nn/") for f in files) and "nn/__init__.py" not in files:
            if not any("nn/__init__.py" in v for v in out.values()):
                files.insert(0, "nn/__init__.py")
        if files:
            out[cid] = files
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_chapter(ch: dict, all_ids: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    cid = ch["id"]
    cp = is_cp(cid)
    sections = {s["ref"] for s in ch.get("sections", [])}
    covered, seen, counts = set(), [], collections.Counter()
    last = 0
    for ex in ch["exercises"]:
        eid, t = ex["id"], ex["type"]
        counts[t] += 1
        if exercise_chapter(eid) != cid:
            errors.append(f"{eid}: wrong chapter prefix")
        if t not in TYPES:
            errors.append(f"{eid}: unknown type {t}")
        suffix = eid[len(cid) + 1:]
        if cp:
            if not re.fullmatch(r"\d+", suffix):
                errors.append(f"{eid}: checkpoint ids are {cid}.<k>")
        elif t == "🧠" and not re.fullmatch(r"Q\d+", suffix):
            errors.append(f"{eid}: quiz ids are {cid}.Q<k>")
        elif t == "🔁" and not re.fullmatch(r"R\d+", suffix):
            errors.append(f"{eid}: recall ids are {cid}.R<k>")
        elif t == "💼" and not re.fullmatch(r"E\d+", suffix):
            errors.append(f"{eid}: interview ids are {cid}.E<k>")
        elif t not in ("🧠", "🔁", "💼"):
            if not re.fullmatch(r"\d+", suffix) or int(suffix) != last + 1:
                errors.append(f"{eid}: numbered ids must be consecutive")
            else:
                last = int(suffix)
        if ex["stars"] not in (1, 2, 3, 4) or ex["minutes"] <= 0:
            errors.append(f"{eid}: bad stars/minutes")
        if not cp and t in FILE_OF and ex["file"] != FILE_OF[t]:
            errors.append(f"{eid}: type {t} goes in file {FILE_OF[t]}")
        if not set(ex["tracks"]) <= set(TRACKS):
            errors.append(f"{eid}: bad tracks")
        if ex.get("check") not in CHECKS:
            errors.append(f"{eid}: bad check")
        for p in ex["prereq"]:
            pc = exercise_chapter(p)
            if pc not in ORDER:
                errors.append(f"{eid}: prereq {p} unknown")
            elif rank(pc) > rank(cid):
                errors.append(f"{eid}: prereq {p} comes after {cid}")
            elif "." in p and p not in all_ids:
                errors.append(f"{eid}: prereq {p} does not exist")
            elif pc == cid and "." in p and p not in seen:
                errors.append(f"{eid}: prereq {p} must come earlier in the chapter")
        seen.append(eid)
        for c in ex["covers"]:
            if not cp and c not in sections:
                errors.append(f"{eid}: covers unknown section {c}")
            covered.add(c)
    if not cp:
        missing = sorted(sections - covered, key=lambda r: [int(x) for x in r.split(".")])
        for ref in missing:
            errors.append(f"section {ref} not covered")
        for r in ch.get("recall_from", []):
            if r not in ORDER or rank(r) >= rank(cid):
                errors.append(f"recall_from {r} is not an earlier chapter")
    return errors, warnings


def validate(chapters: dict, check_stubs: bool = True) -> list[str]:
    problems = []
    missing = [c for c in ORDER if c not in chapters]
    if missing:
        problems.append(f"missing chapters: {missing}")
    all_ids = {}
    for cid, ch in chapters.items():
        for ex in ch["exercises"]:
            if ex["id"] in all_ids:
                problems.append(f"duplicate exercise id {ex['id']}")
            all_ids[ex["id"]] = cid
    for cid, ch in chapters.items():
        errors, _ = validate_chapter(ch, all_ids)
        problems += [f"{cid}: {e}" for e in errors]
    # every track is closed on its code prerequisites (see is_code_prereq)
    exercises = {ex["id"]: ex for ch in chapters.values() for ex in ch["exercises"]}
    for ex in exercises.values():
        for p in ex["prereq"]:
            dep = exercises.get(p)
            if dep is None or not is_code_prereq(dep, ex):
                continue
            for track in ex["tracks"]:
                if track not in dep["tracks"]:
                    problems.append(f"{ex['id']}: track {track!r} lacks its code prerequisite {p}")
    for cid, ch in chapters.items():  # checkpoints and final project belong to every track
        if is_cp(cid):
            for ex in ch["exercises"]:
                if set(ex["tracks"]) != set(TRACKS):
                    problems.append(f"{ex['id']}: checkpoint/final project items belong to all tracks")
    if check_stubs:
        planned = set()
        for cid, ch in chapters.items():
            for m in ch.get("mylearn", []):
                rel = m["file"]
                planned.add(rel)
                if not (STUBS / rel).exists():
                    problems.append(f"{cid}: stub {rel} missing")
                    continue
                names = {item["name"] for item in stub_api(rel)}
                for api in m["api"]:
                    mm = re.match(r"\s*(?:def|class)\s+(\w+)", api["signature"])
                    if mm and api["kind"] != "method" and mm.group(1) not in names:
                        problems.append(f"{cid}: {rel} lacks {mm.group(1)}")
                for dep in stub_imports(rel):
                    owner = next((c for c, chh in chapters.items()
                                  if any(x["file"] == dep for x in chh.get("mylearn", []))), None)
                    if owner is None and dep not in ("nn/__init__.py", "_example.py"):
                        problems.append(f"{cid}: {rel} imports {dep}, owned by no chapter")
                    elif owner and rank(owner) > rank(cid):
                        problems.append(f"{cid}: {rel} imports {dep} from a later chapter ({owner})")
        for path in STUBS.rglob("*.py"):
            rel = path.relative_to(STUBS).as_posix()
            if rel not in planned | {"__init__.py", "_example.py", "nn/__init__.py"}:
                problems.append(f"stub {rel} belongs to no chapter")
    return problems


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def fmt_hours(minutes: float) -> str:
    hours = minutes / 60
    return f"{hours:.0f} h" if hours >= 10 else f"{hours:.1f} h".replace(".", ",")


def stars(n: int) -> str:
    return "★" * n


def tracks_code(ex: dict) -> str:
    return "".join(t[0].upper() for t in TRACKS if t in ex["tracks"]) or "–"


def md_cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def compress_ids(ids: list[str]) -> str:
    """'18.Q1, 18.Q2, 18.Q3, 18.5' -> '18.Q1–Q3, 18.5'."""
    out, run = [], []

    def key(eid):
        cid, suffix = eid.split(".", 1)
        m = re.fullmatch(r"([A-Z]?)(\d+)", suffix)
        return (cid, m.group(1), int(m.group(2))) if m else (cid, suffix, None)

    def flush():
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            last = run[-1].split(".", 1)[1]
            out.append(f"{run[0]}–{last}")
        run.clear()

    for eid in ids:
        k = key(eid)
        if run:
            pk = key(run[-1])
            if k[2] is not None and pk[:2] == k[:2] and pk[2] is not None and k[2] == pk[2] + 1:
                run.append(eid)
                continue
        flush()
        run.append(eid)
    flush()
    return ", ".join(out)


def anchor(cid: str) -> str:
    return "ch-" + cid.lower()


def chapter_label(ch: dict) -> str:
    cid = ch["id"]
    if cid.startswith("CP") or cid == "PF":
        return f"{cid} — {ch['title']}"
    return f"{cid} — {ch['title']}"


# ---------------------------------------------------------------------------
# Graphs and calendar
# ---------------------------------------------------------------------------
def chapter_edges(chapters: dict) -> set[tuple[str, str]]:
    edges = set()
    for cid, ch in chapters.items():
        for ex in ch["exercises"]:
            for p in ex["prereq"]:
                pc = exercise_chapter(p)
                if pc != cid and pc in chapters and not is_cp(pc) and not is_cp(cid):
                    edges.add((pc, cid))
    # transitive reduction (DAG in study order)
    succ = collections.defaultdict(set)
    for a, b in edges:
        succ[a].add(b)

    def reachable(a, b, skip_edge):
        stack, seen = [a], set()
        while stack:
            n = stack.pop()
            for m in succ[n]:
                if (n, m) == skip_edge:
                    continue
                if m == b:
                    return True
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return False

    return {(a, b) for a, b in edges if not reachable(a, b, (a, b))}


def mermaid_chapters(chapters: dict) -> str:
    lines = ["```mermaid", "flowchart TD"]
    for part, ids in PARTS.items():
        present = [c for c in ids if c in chapters and not is_cp(c)]
        if not present:
            continue
        lines.append(f'  subgraph P{part}["Partie {part} · {PART_TITLES[part]}"]')
        for cid in present:
            title = chapters[cid]["title"].replace('"', "'")
            lines.append(f'    n{cid}["{cid} · {title}"]')
        lines.append("  end")
    for a, b in sorted(chapter_edges(chapters), key=lambda e: (rank(e[0]), rank(e[1]))):
        lines.append(f"  n{a} --> n{b}")
    lines.append("```")
    return "\n".join(lines)


def mermaid_modules(chapters: dict) -> str:
    owner = {}
    for cid, ch in chapters.items():
        for m in ch.get("mylearn", []):
            owner[m["file"]] = cid
    lines = ["```mermaid", "flowchart LR"]
    node = {rel: "m" + re.sub(r"\W", "_", rel[:-3]) for rel in owner}
    for rel, cid in sorted(owner.items(), key=lambda kv: rank(kv[1])):
        if rel.endswith("__init__.py"):
            continue
        lines.append(f'  {node[rel]}["{rel[:-3]} ({cid})"]')
    edges = 0
    for rel in owner:
        if rel.endswith("__init__.py") or not (STUBS / rel).exists():
            continue
        for dep in stub_imports(rel):
            if dep in node and not dep.endswith("__init__.py"):
                lines.append(f"  {node[dep]} --> {node[rel]}")
                edges += 1
    lines.append("```")
    return "\n".join(lines)


def calendar(chapters: dict, hours_per_week: float = HOURS_PER_WEEK) -> list[dict]:
    rows, elapsed = [], 0.0
    for cid, ch in chapters.items():
        minutes = minutes_of(ch)["total"]
        start_week = int(elapsed // (hours_per_week * 60)) + 1
        elapsed += minutes
        end_week = int((elapsed - 1e-9) // (hours_per_week * 60)) + 1
        start = STUDY_START + dt.timedelta(weeks=start_week - 1)
        end = STUDY_START + dt.timedelta(weeks=end_week - 1, days=6)
        rows.append({"id": cid, "title": ch["title"], "hours": minutes / 60, "start_week": start_week,
                     "end_week": end_week, "start": start, "end": end})
    return rows


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.",
          "nov.", "déc."]


def fdate(d: dt.date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def render_api(rel: str) -> list[str]:
    lines = [f"**`{rel}`**", "", "```python"]
    for item in stub_api(rel):
        if item["kind"] == "function":
            tag = "  # provided" if item.get("provided") else ""
            lines.append(f"{item['signature']}{tag}")
        else:
            lines.append(f"{item['signature']}:")
            for m in item["methods"]:
                lines.append(f"    {m['signature'].replace('def ', 'def ', 1)}")
    lines += ["```", ""]
    return lines


def render_totals(chapters: dict) -> list[str]:
    lines = ["| Partie | " + " | ".join(TYPES) + " | Total | Temps d'étude |",
             "|---|" + "---|" * (len(TYPES) + 2)]
    grand = collections.Counter()
    grand_minutes = 0
    for part, ids in PARTS.items():
        c = collections.Counter()
        minutes = 0
        for cid in ids:
            if cid in chapters:
                c.update(ex["type"] for ex in chapters[cid]["exercises"])
                minutes += minutes_of(chapters[cid])["total"]
        grand.update(c)
        grand_minutes += minutes
        lines.append(f"| {part} · {PART_TITLES[part]} | " + " | ".join(str(c.get(t, 0)) for t in TYPES)
                     + f" | **{sum(c.values())}** | {fmt_hours(minutes)} |")
    lines.append("| **Total** | " + " | ".join(f"**{grand.get(t, 0)}**" for t in TYPES)
                 + f" | **{sum(grand.values())}** | **{fmt_hours(grand_minutes)}** |")
    return lines


def render_chapter(ch: dict, chapters: dict) -> list[str]:
    cid = ch["id"]
    t = minutes_of(ch)
    out = [f'<a id="{anchor(cid)}"></a>', "", f"### {chapter_label(ch)}", ""]
    if is_cp(cid):
        return out + render_checkpoint(ch, t)
    book = ch.get("book") or {}
    if book.get("chapter"):
        book_txt = f"vol. {book['volume']}, ch. {book['chapter']} « {book.get('title_en', '')} »" + (
            f", p. {book['pages']}" if book.get("pages") else "")
    else:
        book_txt = "— (chapitre propre au workbook)"
    counts = collections.Counter(ex["type"] for ex in ch["exercises"])
    out += [
        "| | |", "|---|---|",
        f"| **Partie** | {part_of(cid)} · {PART_TITLES[part_of(cid)]} |",
        f"| **Livre** | {book_txt} |",
        f"| **Dossier** | `chapitres/{ch['dir']}/` |",
        f"| **Exercices** | {len(ch['exercises'])} : " + " · ".join(
            f"{ty} {counts[ty]}" for ty in TYPES if counts.get(ty)) + " |",
        f"| **Temps d'étude** | **{fmt_hours(t['total'])}** (lecture {fmt_hours(t['reading'])}, "
        f"exercices {fmt_hours(t['exercises'])}, {ch.get('flashcards', 0)} flashcards "
        f"{fmt_hours(t['flashcards'])}) |",
        f"| **Génération** | {ch.get('generation_sessions', 1)} session(s) |",
        f"| **Rappels 🔁** | ch. {', '.join(ch.get('recall_from', [])) or '—'} |",
        f"| **Compétence 🛠️** | {md_cell(ch.get('pro_skill', '—'))} |",
        "",
        ch.get("summary", ""), "",
        "**Objectifs d'apprentissage**", "",
    ]
    out += [f"- {o}" for o in ch["objectives"]] + [""]
    secs = ch.get("sections", [])
    if secs and book.get("chapter"):
        tops = [s for s in secs if s["ref"].count(".") == 1]
        out += [f"**Sections du livre couvertes** : {len(secs)} sections et sous-sections, toutes "
                f"couvertes (§{tops[0]['ref']} à §{tops[-1]['ref']} ; détail dans la matrice de "
                f"couverture).", ""]
    elif secs:
        out += ["**Sections du chapitre** : " + " · ".join(
            f"{s['ref']} {s['title']}" for s in secs if s["ref"].count(".") == 1), ""]
    if secs and book.get("chapter"):
        fast = rapide_sections(ch)
        skipped = [s["ref"] for s in secs if s["ref"] not in fast]
        out += [f"**Lecture du parcours rapide** (fiche complète + sections ⏩ du livre, "
                f"≈ {fmt_hours(rapide_reading(ch))}) : "
                + ("toutes les sections." if not skipped else
                   f"{len(fast)} sections sur {len(secs)} ; sections laissées de côté : "
                   + ", ".join(f"§{r}" for r in skipped) + "."), ""]
    if ch.get("teaches"):
        out += ["**Notions enseignées** : " + " ; ".join(ch["teaches"]), ""]
    req = ch.get("requires", [])
    if req:
        out += ["**Notions mobilisées** : " + " ; ".join(
            f"{r['notion']} ({'introduite ici, encadré 🧮' if r.get('from') == 'local' else 'ch. ' + str(r.get('from'))})"
            for r in req), ""]
    # exercises
    out += ["**Exercices**", "",
            "| ID | Type | Titre | ★ | ⏱️ | Fil rouge | Fichier | Prérequis | Parcours | Vérif. |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for ex in ch["exercises"]:
        gpu = " 🚀" if ex.get("gpu") else ""
        prereq = ", ".join(ex["prereq"]) or "—"
        out.append(f"| {ex['id']} | {ex['type']} | {md_cell(ex['title'])}{gpu} | {stars(ex['stars'])} | "
                   f"{ex['minutes']} | {md_cell(ex.get('thread', '—'))} | {ex['file']} | {prereq} | "
                   f"{tracks_code(ex)} | {ex.get('check', '')} |")
    out.append("")
    if ch.get("mylearn"):
        out += ["**mylearn : signatures figées** (source : `templates/mylearn_stubs/`)", ""]
        for m in ch["mylearn"]:
            if (STUBS / m["file"]).exists() and not m["file"].endswith("__init__.py"):
                out += render_api(m["file"])
    if ch.get("updates"):
        out += ["**Points 🕰️ à traiter** (à vérifier par recherche web à la génération)", ""]
        for u in ch["updates"]:
            out.append(f"- **{u.get('topic', '')}** — livre : {u.get('book', '—')} · aujourd'hui : "
                       f"{u.get('today', '—')} · à vérifier : {u.get('verify', '—')}")
        out.append("")
    if ch.get("interview_themes"):
        out += ["**Thèmes 💼** : " + " · ".join(ch["interview_themes"]), ""]
    if ch.get("papers"):
        out += ["**Articles 📄** : " + " · ".join(
            f"{p.get('authors', '')} ({p.get('year', '')}), *{p.get('title', '')}*" for p in ch["papers"]), ""]
    if ch.get("notes"):
        out += ["<details><summary>Notes de planification</summary>", "", md_cell(ch["notes"]), "",
                "</details>", ""]
    return out


def render_checkpoint(ch: dict, t: dict) -> list[str]:
    cid = ch["id"]
    out = []
    if cid != "PF":
        points = sum(ex.get("points", 0) for ex in ch["exercises"])
        exam = sum(ex["minutes"] for ex in ch["exercises"])
        out += [f"**Dossier** : `{ch['dir']}/` · chapitres : {', '.join(ch.get('chapters', []))} · "
                f"examen blanc {exam} min sur {points} points · temps total {fmt_hours(t['total'])}", ""]
    else:
        out += [f"**Dossier** : `{ch['dir']}/` · chapitres mobilisés : {', '.join(ch.get('chapters', []))} "
                f"· temps total {fmt_hours(t['total'])}", ""]
    out += ["**Objectifs**", ""] + [f"- {o}" for o in ch["objectives"]] + [""]
    label = "Examen blanc" if cid != "PF" else "Étapes du projet"
    out += [f"**{label}**", "", "| ID | Type | Titre | ★ | ⏱️ | Points | Couvre | Parcours |",
            "|---|---|---|---|---|---|---|---|"]
    for ex in ch["exercises"]:
        out.append(f"| {ex['id']} | {ex['type']} | {md_cell(ex['title'])} | {stars(ex['stars'])} | "
                   f"{ex['minutes']} | {ex.get('points', '')} | {', '.join(ex['covers'])} | "
                   f"{tracks_code(ex)} |")
    out.append("")
    if ch.get("synthesis"):
        out += ["**Synthèse** (carte mentale et fiche d'une page) : " + " · ".join(ch["synthesis"]), ""]
    mp = ch.get("miniproject")
    if mp:
        out += [f"**Mini-projet {mp['id']} — {mp['title']}** (`{mp['dir']}/`, ≈ {mp.get('hours', '?')} h, "
                f"données : {mp.get('dataset', '—')})", "", mp.get("brief", ""), ""]
        if mp.get("steps"):
            out += ["| Étape | Titre | ⏱️ |", "|---|---|---|"]
            out += [f"| {s['id']} | {md_cell(s['title'])} | {s.get('minutes', '')} |" for s in mp["steps"]]
            out.append("")
        if mp.get("grading"):
            out += ["Grille : " + " · ".join(mp["grading"]), ""]
        if mp.get("extensions"):
            out += ["Extensions : " + " · ".join(mp["extensions"]), ""]
    if ch.get("study_notes"):
        out += [f"*{md_cell(ch['study_notes'])}*", ""]
    return out


def render_coverage(chapters: dict) -> list[str]:
    out = []
    for cid, ch in chapters.items():
        book = ch.get("book") or {}
        if is_cp(cid) or not book.get("chapter"):
            continue
        cov = collections.defaultdict(list)
        for ex in ch["exercises"]:
            for c in ex["covers"]:
                cov[c].append(ex["id"])
        out += [f"<details><summary><b>Ch. {cid} — {ch['title']}</b> ({len(ch['sections'])} sections)"
                "</summary>", "", "| § | Section du livre | Exercices |", "|---|---|---|"]
        for s in ch["sections"]:
            ids = cov.get(s["ref"], [])
            indent = "&nbsp;&nbsp;" * (s["ref"].count(".") - 1)
            out.append(f"| {s['ref']} | {indent}{md_cell(s['title'])} | {compress_ids(ids) or '**0**'} |")
        out += ["", "</details>", ""]
    return out


def build_syllabus(chapters: dict) -> str:
    total_ex = sum(len(ch["exercises"]) for ch in chapters.values())
    total_min = sum(minutes_of(ch)["total"] for ch in chapters.values())
    gen = sum(ch.get("generation_sessions", 1) for ch in chapters.values() if not is_cp(ch["id"]))
    n_cp = sum(1 for c in chapters if c.startswith("CP"))
    lines = [
        "# SYLLABUS — plan détaillé et contrat du workbook",
        "",
        "> **Statut : contrat.** Ce document fixe, pour chaque chapitre, les exercices (ID **stables**), "
        "les signatures de `mylearn` (**figées**), les sections du livre couvertes, les rappels, les "
        "points de modernisation et le temps d'étude. Il est **généré** à partir de "
        "`docs/syllabus/data/*.json` et des stubs `templates/mylearn_stubs/` par "
        "`python tools/syllabus.py build` : on ne l'édite pas à la main. Tout écart pendant la "
        "génération d'un chapitre est consigné dans `suivi/PROGRESS.md` (§ Écarts).",
        "",
        f"**En chiffres** : {len(chapters) - n_cp - 1} chapitres, {n_cp} checkpoints et le projet final · "
        f"**{total_ex} exercices** · **≈ {fmt_hours(total_min)} d'étude** · "
        f"{gen} sessions de génération de chapitres (+ checkpoints, audits et finalisation).",
        "",
        "## Sommaire",
        "",
        "1. [Conventions](#conventions)",
        "2. [Vue d'ensemble](#vue-densemble)",
        "3. [Totaux par type et par partie](#totaux)",
        "4. [Graphe de dépendances](#graphe-de-dépendances)",
        "5. [Calendrier indicatif](#calendrier-indicatif)",
        "6. [Plan détaillé des chapitres](#plan-détaillé)",
        "7. [Matrice de couverture du livre](#matrice-de-couverture)",
        "",
        '<a id="conventions"></a>',
        "",
        "## 1. Conventions",
        "",
        "- **ID des exercices** : `N.Qk` quiz 🧠 · `N.Rk` rappel 🔁 · `N.Ek` entretien 💼 · `N.k` "
        "(k consécutifs) pour tous les autres, dans l'ordre : papier (✏️, ∂) puis réflexion du "
        "fichier 02, puis notebook 03. Checkpoints : `CPk.n` et projet final : `PF.n`, pour tous les types "
        "(🧠 et 💼 compris) ; étapes des mini-projets : `MPk.n`. Un ID publié (chapitre généré) ne change jamais.",
        "- **Fichier** : `02` = `02_exercices.md` (papier, réflexion, entretien) ; `03` = "
        "`03_notebook.ipynb`.",
        "- **Difficulté** : ★ application directe (5–15 min) · ★★ standard (15–30) · ★★★ approfondi "
        "(30–90) · ★★★★ défi (> 90) · 🚀 GPU Colab conseillé en mode complet.",
        "- **Parcours** : R = rapide (l'essentiel pour l'employabilité) · M = orienté maths · C = "
        "orienté code ; le parcours complet contient tout (voir `docs/PARCOURS.md`).",
        "- **Vérification** : `wb.check` (réponse vérifiée par empreinte) · `pytest` (fonction mylearn "
        "comparée à un oracle) · `manual` (réponse rédigée, corrigée avec les solutions).",
        "- **Temps d'étude** = lecture (livre + fiche) + exercices + première passe des flashcards "
        f"({FLASHCARD_MINUTES} min/carte) ; checkpoints = examen + synthèse ({SYNTHESIS_MINUTES} min) + "
        "mini-projet.",
        "- **Prérequis** : un exercice ne dépend que d'exercices ou de chapitres antérieurs "
        "(vérifié automatiquement). Notions au-delà de 0A/0B : introduites dans le chapitre (encadré 🧮). "
        "Un prérequis *de code* (notebook → notebook) est toujours dans les mêmes parcours que l'exercice "
        "qui en dépend (vérifié) ; hors parcours, un autre prérequis se remplace par la lecture de son corrigé.",
        "- **mylearn** : NumPy + bibliothèque standard ; classes à la scikit-learn (`fit` renvoie `self`, "
        "attributs appris suffixés `_`) ; `random_state` pour les classes, `rng` pour les fonctions ; "
        "`X` de forme `(n_samples, n_features)`, images `(N, C, H, W)`, séquences `(N, T, D)` ; couches "
        "denses `z = x @ W + b` avec `W` de forme `(n_in, n_out)` ; imports relatifs vers les modules des "
        "chapitres antérieurs seulement. Les signatures ci-dessous sont extraites des stubs. Conventions "
        "de régularisation, noms du learning rate, orientation des poids et unités (bits ou nats) : "
        "`annexes/formulaire.md`. Un module d'un chapitre sauté est pris dans la référence (BIBLE §22).",
        "",
        '<a id="vue-densemble"></a>',
        "",
        "## 2. Vue d'ensemble",
        "",
        "| ID | Chapitre | Livre | Exercices | Temps | Génération | mylearn |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid, ch in chapters.items():
        book = ch.get("book") or {}
        bk = f"V{book['volume']} ch. {book['chapter']}" if book.get("chapter") else "—"
        mods = ", ".join(f"`{m['file'][:-3]}`" for m in ch.get("mylearn", []) if not m["file"].endswith("__init__.py")) or "—"
        gen = "" if is_cp(cid) else str(ch.get("generation_sessions", 1))
        lines.append(f"| [{cid}](#{anchor(cid)}) | {md_cell(ch['title'])} | {bk} | {len(ch['exercises'])} | "
                     f"{fmt_hours(minutes_of(ch)['total'])} | {gen} | {mods} |")
    lines += ["", '<a id="totaux"></a>', "", "## 3. Totaux par type et par partie", ""]
    lines += render_totals(chapters)
    lines += ["", "Légende des types : " + " · ".join(f"{t} {TYPE_NAMES[t]}" for t in TYPES), ""]
    totals = track_totals(chapters)
    lines += ["| Parcours | Exercices | Temps d'exercices | Temps total | Part du temps total |",
              "|---|---|---|---|---|"]
    for key in ("complet",) + TRACKS:
        t = totals[key]
        lines.append(f"| {key.capitalize()} | {t['n']} | {fmt_hours(t['exercises'])} | "
                     f"{fmt_hours(t['total'])} | {100 * t['total'] / totals['complet']['total']:.0f} % |")
    lines += ["", "Temps total d'un parcours = ses exercices + les corrigés de ses prérequis hors "
              "parcours (un tiers du temps) + la lecture (sélective pour le parcours rapide) + "
              "flashcards, synthèses, mini-projets et projet final (communs). Détail : "
              "`docs/PARCOURS.md`."]
    lines += ["", '<a id="graphe-de-dépendances"></a>', "", "## 4. Graphe de dépendances", "",
              "### Chapitres", "",
              "Arêtes déduites des prérequis des exercices (réduction transitive : une flèche A → B "
              "signifie qu'au moins un exercice de B s'appuie directement sur A).", "",
              mermaid_chapters(chapters), "", "### Modules mylearn", "",
              "Arêtes = imports relatifs entre stubs (le module de droite réutilise celui de gauche). "
              "Entre parenthèses : le chapitre qui publie le module.", "",
              mermaid_modules(chapters), "",
              '<a id="calendrier-indicatif"></a>', "", "## 5. Calendrier indicatif", "",
              f"Rythme régulier de **{HOURS_PER_WEEK} h par semaine** (≈ 1 h 30 par jour), parcours "
              f"complet, début le {fdate(STUDY_START)}. Pour un autre rythme, multiplie les numéros de "
              f"semaine par {HOURS_PER_WEEK}/(tes heures par semaine). Le parcours rapide prend environ "
              f"{100 * totals['rapide']['total'] / totals['complet']['total']:.0f} % de ce temps.", "",
              "| Semaines | Période | Chapitre | Temps |", "|---|---|---|---|"]
    for row in calendar(chapters):
        weeks = (f"{row['start_week']}" if row["start_week"] == row["end_week"]
                 else f"{row['start_week']}–{row['end_week']}")
        lines.append(f"| {weeks} | {fdate(row['start'])} → {fdate(row['end'])} | {row['id']} · "
                     f"{md_cell(row['title'])} | {fmt_hours(row['hours'] * 60)} |")
    last = calendar(chapters)[-1]
    years = last["end_week"] / 52
    years_txt = f"{years:.1f}".replace(".", ",") + (" an" if years < 2 else " ans")
    lines += ["", f"Fin estimée : semaine {last['end_week']} ({fdate(last['end'])}), soit environ "
              f"{years_txt} à ce rythme. La génération garde deux chapitres "
              "d'avance sur l'étude (METHODE §1).", "",
              '<a id="plan-détaillé"></a>', "", "## 6. Plan détaillé", ""]
    for part, ids in PARTS.items():
        lines += [f"## Partie {part} · {PART_TITLES[part]}", ""]
        for cid in ids:
            if cid in chapters:
                lines += render_chapter(chapters[cid], chapters)
    lines += ['<a id="matrice-de-couverture"></a>', "", "## 7. Matrice de couverture", "",
              "Chaque section et sous-section des chapitres 1 à 29 du livre, avec les exercices qui la "
              "couvrent. Aucune section n'est à zéro (vérifié par `python tools/syllabus.py check`). "
              "Le glossaire (ch. 30) sert à `annexes/glossaire.md` et n'a pas d'exercices.", ""]
    lines += render_coverage(chapters)
    return "\n".join(lines).rstrip() + "\n"


TRACK_TEXT = {
    "complet": ("Parcours complet", "Tout le workbook, dans l'ordre : l'objectif d'exhaustivité de la bible."),
    "rapide": ("Parcours rapide", "L'essentiel pour être employable (data scientist, ML engineer) : concepts "
               "centraux, pratique scikit-learn et PyTorch, toutes les questions d'entretien, les "
               "implémentations clés. On peut revenir plus tard sur le reste."),
    "maths": ("Parcours orienté maths", "Pour comprendre en profondeur : calculs à la main, démonstrations, "
              "estimations de Fermi et implémentations à forte composante mathématique."),
    "code": ("Parcours orienté code", "Pour devenir solide en implémentation : from scratch, bibliothèques, "
             "chasses au bug, défis, expériences et compétences pro."),
}


def track_totals(chapters: dict) -> dict[str, dict]:
    out = {}
    for key in ("complet",) + TRACKS:
        plan = track_plan(chapters, key)
        tot = collections.Counter()
        for row in plan.values():
            tot.update({k: v for k, v in row.items() if isinstance(v, int)})
            tot["n"] += len(row["ids"])
            tot["n_read"] += len(row["read"])
        out[key] = dict(tot)
    return out


def build_parcours(chapters: dict) -> str:
    totals = track_totals(chapters)
    lines = ["# PARCOURS", "",
             "> Généré par `python tools/syllabus.py build` à partir du syllabus (`docs/SYLLABUS.md`). "
             "Chaque exercice y porte ses parcours (colonne « Parcours » : R, M, C).", "",
             "Quel que soit le parcours : fiche du chapitre, flashcards chaque jour, **checkpoints, "
             "mini-projets et projet final en entier** (ils font partie des quatre parcours). Les "
             "listes ci-dessous donnent les **exercices** à faire ; les plages `18.Q1–Q12` se lisent "
             "« de 18.Q1 à 18.Q12 ».", "",
             "**Règles des prérequis.** Un exercice de notebook qui réutilise le code d'un exercice "
             "de notebook antérieur (fonction `mylearn`, modèle, données préparées) a toujours ce "
             "prérequis dans son parcours : c'est vérifié automatiquement. Les autres prérequis "
             "sont des connaissances : quand l'un d'eux n'est pas dans ton parcours, lis son corrigé "
             "(`05_solutions`), compté pour un tiers de son temps (lignes « corrigés à lire »). "
             "Si tu sautes un chapitre entier, ses modules `mylearn` sont pris dans la référence "
             "pour les chapitres suivants (voir le README, § « Ta librairie mylearn »).", "",
             "**Lecture.** Le parcours rapide lit toute la fiche, mais seulement les sections du "
             "livre couvertes par ses exercices pratiques (la liste figure dans le SYLLABUS, chapitre "
             "par chapitre, et le guide de lecture de la fiche les signale ⏩) ; ses 🧠 et 💼 se "
             "préparent avec la fiche. Les autres parcours lisent tout. Les chapitres sans livre (0A, "
             "0B, bonus) sont lus en entier par tous : la fiche y est le cours.",
             "", "| Parcours | Exercices | Exercices (temps) | Corrigés à lire | Lecture | "
             "Flashcards, synthèses, projets | **Total** |", "|---|---|---|---|---|---|---|"]
    for key in ("complet",) + TRACKS:
        t = totals[key]
        lines.append(f"| [{TRACK_TEXT[key][0]}](#{key}) | {t['n']} | {fmt_hours(t['exercises'])} | "
                     f"{t['n_read']} ({fmt_hours(t['solutions'])}) | {fmt_hours(t['reading'])} | "
                     f"{fmt_hours(t['flashcards'] + t['project'])} | **{fmt_hours(t['total'])}** |")
    lines += ["", f"À {HOURS_PER_WEEK} h par semaine : complet ≈ {totals['complet']['total'] / 60 / HOURS_PER_WEEK:.0f} "
              f"semaines, rapide ≈ {totals['rapide']['total'] / 60 / HOURS_PER_WEEK:.0f} semaines.", ""]
    for key in ("complet",) + TRACKS:
        title, text = TRACK_TEXT[key]
        plan = track_plan(chapters, key)
        t = totals[key]
        lines += [f'<a id="{key}"></a>', "", f"## {title}", "", text, "",
                  f"**{t['n']} exercices, {fmt_hours(t['exercises'])} d'exercices, "
                  f"≈ {fmt_hours(t['total'])} au total.**", ""]
        for part, ids in PARTS.items():
            lines += [f"### Partie {part} · {PART_TITLES[part]}", ""]
            for cid in ids:
                if cid not in chapters:
                    continue
                ch, row = chapters[cid], plan[cid]
                if not row["ids"]:
                    lines.append(f"- **{cid}** {ch['title']} : lecture de la fiche seulement")
                    continue
                lines.append(f"- **{cid}** {ch['title']} ({len(row['ids'])} ex., "
                             f"{fmt_hours(row['exercises'])}) : {compress_ids(row['ids'])}")
                if row["read"]:
                    lines.append(f"  - corrigés à lire : {compress_ids(row['read'])}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_dashboard(chapters: dict) -> str:
    head = (ROOT / "suivi" / "tableau_de_bord.md").read_text(encoding="utf-8")
    head = head.split("<!-- wb:section")[0].rstrip() + "\n\n"
    setup = re.search(r"<!-- wb:section setup -->.*?<!-- wb:end setup -->\n",
                      (ROOT / "suivi" / "tableau_de_bord.md").read_text(encoding="utf-8"), re.S)
    parts = [head, setup.group(0) if setup else ""]
    for cid, ch in chapters.items():
        t = minutes_of(ch)
        lines = [f"\n<!-- wb:section {cid} -->", f"## {chapter_label(ch)} ⏱️ {fmt_hours(t['total'])}", ""]
        groups = collections.OrderedDict()
        for ex in ch["exercises"]:
            key = ex["type"] if ex["type"] in ("🧠", "🔁", "💼") and not is_cp(cid) else None
            groups.setdefault(key, []).append(ex)
        for key, exs in groups.items():
            if key is not None:
                mins = sum(e["minutes"] for e in exs)
                label = {"🧠": "Quiz", "🔁": "Rappels", "💼": "Entretien"}[key]
                lines.append(f"- [ ] {compress_ids([e['id'] for e in exs])} {key} {label} "
                             f"({len(exs)} questions, {mins} min)")
            else:
                for e in exs:
                    gpu = " 🚀" if e.get("gpu") else ""
                    lines.append(f"- [ ] {e['id']} {e['type']} {e['title']} {stars(e['stars'])} "
                                 f"{e['minutes']} min{gpu}")
        mp = ch.get("miniproject")
        if mp:
            lines.append(f"- [ ] Mini-projet {mp['id']} — {mp['title']} (≈ {mp.get('hours', '?')} h)")
        if not is_cp(cid):
            lines.append(f"- [ ] Flashcards importées dans Anki ({ch.get('flashcards', 0)} cartes)")
        lines.append(f"<!-- wb:end {cid} -->")
        parts.append("\n".join(lines) + "\n")
    return "".join(parts)


def update_manifest(chapters: dict) -> dict:
    path = STUBS / "MANIFEST.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    new_chapters = dict(current.get("chapters", {}))
    base = set(current.get("base", []))
    for cid, files in manifest(chapters).items():
        files = [f for f in files if f not in base]  # base files are copied for every chapter
        if not files:
            continue
        old = new_chapters.get(cid, [])
        missing = [f for f in old if f not in files]
        if missing:
            raise SystemExit(f"MANIFEST is additive only: {cid} would lose {missing}")
        new_chapters[cid] = old + [f for f in files if f not in old]
    current["chapters"] = new_chapters
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return current


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "build", "stats"])
    args = parser.parse_args(argv)
    chapters = load()
    if args.command == "check":
        problems = validate(chapters)
        for p in problems:
            print("❌", p)
        print(f"{len(chapters)} chapitres, {sum(len(c['exercises']) for c in chapters.values())} exercices : "
              f"{len(problems)} problème(s).")
        return 1 if problems else 0
    if args.command == "stats":
        print("\n".join(render_totals(chapters)))
        return 0
    problems = validate(chapters)
    if problems:
        for p in problems:
            print("❌", p)
        return 1
    (ROOT / "docs" / "SYLLABUS.md").write_text(build_syllabus(chapters), encoding="utf-8")
    (ROOT / "docs" / "PARCOURS.md").write_text(build_parcours(chapters), encoding="utf-8")
    (ROOT / "suivi" / "tableau_de_bord.md").write_text(build_dashboard(chapters), encoding="utf-8")
    update_manifest(chapters)
    print("✅ docs/SYLLABUS.md, docs/PARCOURS.md, suivi/tableau_de_bord.md, MANIFEST.json générés.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
