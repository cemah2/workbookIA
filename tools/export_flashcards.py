#!/usr/bin/env python
"""Merge every chapter's flashcards.csv into one Anki deck.

    python tools/export_flashcards.py                   # -> exports/dlwb_anki.csv
    python tools/export_flashcards.py -o deck.csv
    python tools/export_flashcards.py --check           # only validate the files

Input format (BIBLE §12): ``;`` separator, columns ``front;back;tags``, HTML
allowed, maths in ``\\( ... \\)``, tags like ``dlwb::ch18::backprop``.
The output starts with Anki header lines (separator, html, tags column, deck)
so that Anki (File > Import) sets everything up by itself; choose the note type
« Basique » (« Basic » in English) in the import window.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "exports" / "dlwb_anki.csv"
DECK = "Deep Learning Workbook"
GLOBS = ("chapitres/*/flashcards.csv", "checkpoints/*/flashcards.csv")


def read_cards(path: Path, root: Path = ROOT) -> tuple[list[list[str]], list[str]]:
    """Return (cards, problems) for one flashcards.csv."""
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    cards, problems = [], []
    for number, row in enumerate(rows, start=1):
        if not row or all(not cell.strip() for cell in row):
            continue
        if number == 1 and [c.strip().lower() for c in row[:3]] == ["front", "back", "tags"]:
            continue  # header line
        if row[0].startswith("#"):
            continue  # Anki directive or comment
        if len(row) != 3:
            problems.append(f"{rel}:{number} : {len(row)} colonne(s) au lieu de 3 (front;back;tags)")
            continue
        front, back, tags = (cell.strip() for cell in row)
        if not front or not back:
            problems.append(f"{rel}:{number} : recto ou verso vide")
            continue
        if not all(tag.startswith("dlwb::") for tag in tags.split()):
            problems.append(f"{rel}:{number} : tags attendus sous la forme dlwb::chXX::concept")
        cards.append([front, back, tags])
    return cards, problems


def export(output: Path, root: Path = ROOT, check_only: bool = False, out=print) -> int:
    files = []
    for pattern in GLOBS:
        files += sorted(root.glob(pattern))
    all_cards, problems, seen, duplicates = [], [], {}, []
    for path in files:
        cards, issues = read_cards(path, root)
        problems += issues
        for card in cards:
            key = card[0].lower()
            if key in seen:
                duplicates.append(f"« {card[0][:60]} » ({seen[key]} et {path.parent.name})")
            else:
                seen[key] = path.parent.name
            all_cards.append(card)
    for problem in problems:
        out("⚠️ " + problem)
    for dup in duplicates:
        out("ℹ️ recto en double : " + dup)
    out(f"{len(all_cards)} carte(s) dans {len(files)} fichier(s).")
    if check_only:
        return 1 if problems else 0
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    buffer.write("#separator:semicolon\n#html:true\n#tags column:3\n")
    buffer.write(f"#deck:{DECK}\n")
    writer = csv.writer(buffer, delimiter=";", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerows(all_cards)
    output.write_text(buffer.getvalue(), encoding="utf-8")
    shown = output.relative_to(root) if output.is_relative_to(root) else output
    out(f"✅ Deck écrit : {shown}  (Anki : Fichier > Importer, type de note « Basique »)")
    return 1 if problems else 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # emoji on Windows consoles and pipes
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args(argv)
    return export(args.output, check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
