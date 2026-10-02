"""Shared building blocks of the chapter builders (tools/chapters/build_chXX.py, used by Claude).

An exercise is described ONCE (statement, TODO skeleton, check, solution, recorded
answers) and produces the cells of both notebooks: the exercise notebook
(03_notebook.ipynb) and the executed solutions notebook (05_solutions.ipynb).

Paper exercises (✏️ of 02_exercices.md) are checked in a "Part 0" of the notebook:
the learner types the values found on paper (``answer_0B_1a = ...``) and
``wb.check`` compares them with the answers recorded from Python expressions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from nbbuild import badge, code, md, setup_cell, write_notebook  # noqa: E402,F401

STARS = {1: "★", 2: "★★", 3: "★★★", 4: "★★★★"}


@dataclass
class Ex:
    """One notebook exercise."""

    id: str
    type: str
    stars: int
    minutes: int
    title: str
    goal: str
    prereq: str
    body: str                      # the statement (Markdown, French)
    todo: str = ""                 # exercise notebook: code to complete (definitions only)
    check: str = ""                # exercise notebook: calls and wb.check
    solution: str = ""             # solutions notebook: the worked solution
    record: str = ""               # solutions notebook: wb.record (cell tagged "answer")
    given: str = ""                # code shared by both notebooks (data preparation)
    thread: str = "—"
    tracks: str = ""
    hypothesis: bool = False       # 🔮: a "my hypothesis" cell before running
    after: list = field(default_factory=list)   # extra (kind, text) cells after the check (see exercise_cells)
    note: str = ""                 # solutions notebook: short remark after the answer
    mylearn: str = ""              # 🔨: the mylearn file the exercise completes (header of BIBLE §11)

    def header(self) -> str:
        fil = f" · **Fil rouge :** {self.thread}" if self.thread != "—" else ""
        lib = f" · **mylearn :** `{self.mylearn}`" if self.mylearn else ""
        return (f"### Ex {self.id} — {self.title} {self.type} {STARS[self.stars]} ⏱️ {self.minutes} min\n"
                f"**Objectif :** {self.goal}  \n**Prérequis :** {self.prereq}{fil}{lib}"
                + (f" · **Parcours :** {self.tracks}" if self.tracks else ""))


@dataclass
class Part:
    key: str
    title: str
    intro: str
    given: str = ""
    exercises: list = field(default_factory=list)


@dataclass
class Paper:
    """A paper exercise checked in Part 0: sub-questions (letter, hint, expression, record options)."""

    id: str
    title: str
    subs: list  # [(letter, hint for the TODO comment, Python expression of the answer, wb.record options)]

    @property
    def letters(self) -> str:
        return "".join(letter for letter, *_ in self.subs)


def paper_cells(kind: str, papers: list[Paper], context: str, intro: str) -> list:
    """Part 0: the learner reports the values found on paper; wb.check compares them."""
    cells = [md(intro)]
    if kind == "solution":
        cells.append(code(context))
    for paper in papers:
        # quiz and recall questions (7.Q1, 7.R2) keep their bare ID; exercises read "Ex 7.3"
        label = paper.id if paper.id.split(".")[-1][:1] in ("Q", "R") else f"Ex {paper.id}"
        cells.append(md(f"**{label} — {paper.title}**"))
        stem = f"answer_{paper.id.replace('.', '_')}"
        if kind == "exercise":
            lines = [f"# {label}: the values you found on paper"]
            lines += [f"{stem}{letter} = ...  # {letter}) {hint}" for letter, hint, _, _ in paper.subs]
            names = ", ".join(f"{stem}{letter}" for letter, *_ in paper.subs)
            lines += ["", f"for letter, answer in zip(\"{paper.letters}\", [{names}]):",
                      f"    wb.check(f\"{paper.id}{{letter}}\", answer)"]
            cells.append(code("\n".join(lines)))
        else:
            lines = []
            for letter, _, expr, options in paper.subs:
                args = f", {options}" if options else ""
                lines.append(f'wb.record("{paper.id}{letter}", {expr}{args})')
            cells.append(code("\n".join(lines), tags=["answer"]))
    return cells


def exercise_cells(ex: Ex, kind: str) -> list:
    cells = [md(ex.header() + "\n\n" + ex.body)]
    if ex.given:
        cells.append(code(ex.given))
    if kind == "exercise":
        if ex.hypothesis:
            cells.append(md("📝 **Mon hypothèse** (à écrire **avant** d'exécuter quoi que ce soit) : …"))
        if ex.todo:
            cells.append(code(ex.todo))
        if ex.check:
            cells.append(code(ex.check))
    else:
        if ex.solution:
            cells.append(code(ex.solution))
        if ex.record:
            cells.append(code(ex.record, tags=["answer"]))
    for cell_kind, text in ex.after:   # "md" and "code": both notebooks; "todo"/"check"/"todo_md": exercise;
        if cell_kind == "md":          # "solution"/"record"/"solution_md": solutions
            cells.append(md(text))
        elif cell_kind == "code":
            cells.append(code(text))
        elif cell_kind in ("todo", "check") and kind == "exercise":
            cells.append(code(text))
        elif cell_kind == "todo_md" and kind == "exercise":   # a Markdown cell the learner fills in (written answers)
            cells.append(md(text))
        elif cell_kind == "solution" and kind == "solution":
            cells.append(code(text))
        elif cell_kind == "solution_md" and kind == "solution":
            cells.append(md(text))
        elif cell_kind == "record" and kind == "solution":
            cells.append(code(text, tags=["answer"]))
    if kind == "solution" and ex.note:
        cells.append(md(f"💡 {ex.note}"))
    return cells


def part_cells(part: Part, kind: str) -> list:
    cells = [md(f"## Partie {part.key} · {part.title}\n\n{part.intro}")]
    if part.given:
        cells.append(code(part.given))
    for ex in part.exercises:
        cells += exercise_cells(ex, kind)
    return cells


def guarded(code_text: str, names: list[str], message: str) -> str:
    """Wrap an experiment cell of a 🔮 so that it runs only once the learner's predictions are filled in.

    `names` are the prediction variables (left to `...` in the exercise notebook): while one of them is
    still `...` (or `None`, which `wb.check` also treats as "not done"), the cell prints `message` instead of
    running, so "Run all" on an empty notebook shows no result before the learner has predicted it. The
    identity tests also work for arrays.
    """
    body = "\n".join(("    " + line) if line.strip() else "" for line in code_text.splitlines())
    return (f"if any(answer is ... or answer is None for answer in [{', '.join(names)}]):\n"
            f"    print({message!r})\nelse:\n{body}")
