#!/usr/bin/env python
"""Harvest the answers of executed solutions notebooks into src/wb/answers.json.

In a solutions notebook, each code cell tagged ``answer`` calls
``wb.record(ex_id, value, decimals=...)``, which prints a line
``WB_ANSWER {...}`` (hashes only). This tool reads those lines from the
*saved outputs* of the notebooks (run them first, e.g. with
``python tools/run_all_notebooks.py --inplace``) and rewrites answers.json.

    python tools/build_answers.py                          # all solutions notebooks
    python tools/build_answers.py chapitres/ch18_backprop/05_solutions.ipynb
    python tools/build_answers.py --check                  # fail if answers.json is outdated

Entries coming from the processed notebooks are replaced entirely (an
answer removed from a notebook disappears); the others are kept.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "src" / "wb" / "answers.json"
PREFIX = "WB_ANSWER "
TAG = "answer"
DEFAULT_GLOBS = (
    "00_setup/*solutions*.ipynb",
    "chapitres/*/05_solutions.ipynb",
    "checkpoints/*/*solutions*.ipynb",
    "projets/*/*solutions*.ipynb",
)


class BuildError(Exception):
    pass


def natural_key(ex_id: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", ex_id)]


def _text(value) -> str:
    return "".join(value) if isinstance(value, list) else (value or "")


def harvest(notebook_path: Path, root: Path = ROOT) -> tuple[dict, list[str]]:
    """Return ({ex_id: entry}, warnings) for one notebook."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    rel = notebook_path.resolve().relative_to(root.resolve()).as_posix()
    entries, warnings = {}, []
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        tagged = TAG in cell.get("metadata", {}).get("tags", [])
        lines = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                lines += _text(output.get("text")).splitlines()
        found = [line[len(PREFIX):] for line in lines if line.startswith(PREFIX)]
        if not tagged:
            if found:
                warnings.append(f"{rel} cellule {index} : WB_ANSWER dans une cellule non taguée « answer » (ignorée)")
            continue
        if not found:
            raise BuildError(
                f"{rel} cellule {index} : cellule taguée « answer » sans ligne WB_ANSWER. "
                "Le notebook a-t-il été exécuté et sauvegardé avec ses sorties ?"
            )
        for payload in found:
            data = json.loads(payload)
            ex_id = str(data.pop("id"))
            if ex_id in entries:
                raise BuildError(f"{rel} : l'exercice {ex_id} est enregistré deux fois")
            data["source"] = rel
            entries[ex_id] = data
    return entries, warnings


def default_notebooks(root: Path = ROOT) -> list[Path]:
    paths = []
    for pattern in DEFAULT_GLOBS:
        paths += sorted(root.glob(pattern))
    return paths


def build(notebooks: list[Path], answers_path: Path = ANSWERS, root: Path = ROOT,
          check_only: bool = False, out=print) -> int:
    old = {"_meta": {}, "answers": {}}
    if answers_path.exists():
        old = json.loads(answers_path.read_text(encoding="utf-8"))
    old_answers = old.get("answers", {})

    harvested, sources, warnings = {}, set(), []
    for path in notebooks:
        entries, warns = harvest(path, root)
        warnings += warns
        sources.add(path.resolve().relative_to(root.resolve()).as_posix())
        for ex_id, entry in entries.items():
            if ex_id in harvested:
                raise BuildError(f"l'exercice {ex_id} est enregistré dans deux notebooks "
                                 f"({harvested[ex_id]['source']} et {entry['source']})")
            harvested[ex_id] = entry

    merged = {k: v for k, v in old_answers.items() if v.get("source") not in sources}
    for ex_id, entry in harvested.items():
        if ex_id in merged:
            raise BuildError(f"l'exercice {ex_id} existe déjà, venant de {merged[ex_id].get('source')}")
        merged[ex_id] = entry
    merged = {k: merged[k] for k in sorted(merged, key=natural_key)}

    added = sorted(set(merged) - set(old_answers), key=natural_key)
    removed = sorted(set(old_answers) - set(merged), key=natural_key)
    changed = sorted((k for k in set(merged) & set(old_answers) if merged[k] != old_answers[k]), key=natural_key)

    for warning in warnings:
        out("⚠️ " + warning)
    out(f"{len(harvested)} réponse(s) lue(s) dans {len(notebooks)} notebook(s) ; "
        f"answers.json : {len(merged)} réponse(s) au total.")
    for label, items in (("ajoutées", added), ("modifiées", changed), ("supprimées", removed)):
        if items:
            out(f"  {label} : {', '.join(items)}")

    new = {
        "_meta": {
            "format": 1,
            "generated_by": "tools/build_answers.py",
            "warning": "Never edit by hand: regenerate from the executed solutions notebooks.",
        },
        "answers": merged,
    }
    if check_only:
        outdated = bool(added or removed or changed)
        out("❌ answers.json n'est pas à jour." if outdated else "✅ answers.json est à jour.")
        return 1 if outdated else 0
    answers_path.write_text(json.dumps(new, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
                            encoding="utf-8")
    out(f"✅ {answers_path.relative_to(root) if answers_path.is_relative_to(root) else answers_path} écrit.")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # emoji on Windows consoles and pipes
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=Path, help="solutions notebooks (default: all)")
    parser.add_argument("--answers", type=Path, default=ANSWERS, help="answers.json path")
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if outdated")
    args = parser.parse_args(argv)
    notebooks = [p if p.is_absolute() else (Path.cwd() / p) for p in args.notebooks] or default_notebooks()
    if not notebooks:
        print("Aucun notebook de solutions trouvé.")
        return 0
    try:
        return build(notebooks, args.answers, check_only=args.check)
    except BuildError as exc:
        print(f"❌ {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
