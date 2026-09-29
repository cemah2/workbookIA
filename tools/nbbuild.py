"""Small helpers to author workbook notebooks from Python (used by Claude).

    from nbbuild import md, code, setup_cell, badge, write_notebook
    cells = [md("# Titre\\n" + badge("chapitres/ch02_stats/03_notebook.ipynb")),
             setup_cell("exercise"),
             md("### Ex 2.1 ..."),
             code("answer = ...  # TODO"),
             code("wb.record('2.1', answer)", tags=["answer"])]
    write_notebook("chapitres/ch02_stats/03_notebook.ipynb", cells)

The setup cell comes from templates/notebook_setup_cell.py, so every notebook
uses exactly the same one.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
REPO = "cemah2/workbookIA"
BRANCH = "main"
SETUP_TEMPLATE = ROOT / "templates" / "notebook_setup_cell.py"

_MYLEARN_LINES = {
    "exercise": 'mylearn = wb.load_mylearn("learner", missing_ok=True)  # your own library',
    "solution": 'mylearn = wb.load_mylearn("ref")  # reference implementation',
    "demo": None,
}


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text.strip("\n"))


def code(source: str, tags: list[str] | None = None) -> nbformat.NotebookNode:
    cell = new_code_cell(source.strip("\n"))
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def setup_cell(kind: str = "exercise") -> nbformat.NotebookNode:
    """The standard setup cell; ``kind`` = exercise, solution or demo."""
    lines = SETUP_TEMPLATE.read_text(encoding="utf-8").rstrip("\n").splitlines()
    lines = [line for line in lines if not line.startswith("# Exercise notebooks add")
             and not line.startswith("# Solutions notebooks add")]
    extra = _MYLEARN_LINES[kind]
    if extra:
        lines.append(extra)
    return code("\n".join(lines), tags=["setup"])


def badge(notebook_path: str) -> str:
    """Markdown 'Open in Colab' badge for a notebook path relative to the repo root."""
    url = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{notebook_path}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def write_notebook(path: str | Path, cells: list) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nbformat.validate(nb)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    return path
