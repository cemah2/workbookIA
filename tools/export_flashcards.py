#!/usr/bin/env python
"""Merge every chapter's flashcards.csv into one Anki deck, with one sub-deck per part and per chapter.

    python tools/export_flashcards.py                   # -> exports/dlwb_anki.csv (every chapter)
    python tools/export_flashcards.py --chapter 3       # one chapter -> exports/dlwb_anki_ch03_probabilites.csv
    python tools/export_flashcards.py --apkg            # -> exports/dlwb_anki.apkg (needs genanki, see below)
    python tools/export_flashcards.py -o deck.csv
    python tools/export_flashcards.py --check           # only validate the files

Input format (BIBLE §12): ``;`` separator, columns ``front;back;tags``, HTML allowed, maths in ``\\( ... \\)``,
tags like ``dlwb::ch18::backprop``.

Output (CSV): the same three columns, then the deck of the card in a fourth column, e.g.
``Deep Learning Workbook::Partie IV::Ch. 18 Rétropropagation``; a chapter export goes to the same sub-deck as the
whole deck, so both can be imported one after the other. Each card also gets the tag of its part and chapter
(``dlwb::partie_IV::ch18``). The file starts with Anki header lines (separator, html, tags column, deck column),
so that Anki (File > Import) sets everything up by itself; choose the note type « Basique » (« Basic » in
English) in the import window. Procedure: ``annexes/anki.md``.

``--apkg`` writes an Anki package with genanki (``pip install genanki``, or the ``anki`` extra of pyproject.toml);
genanki is not a requirement of the workbook, and the command stops with a message when it is missing.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "exports" / "dlwb_anki.csv"
DECK = "Deep Learning Workbook"
GLOBS = ("chapitres/*/flashcards.csv", "checkpoints/*/flashcards.csv")
HEADER = "#separator:semicolon\n#html:true\n#tags column:3\n#deck column:4\n"
# genanki: fixed ids, so that importing a new package updates the cards already in the collection
MODEL_ID = 1_790_790_079
MODEL_NAME = "Deep Learning Workbook (recto/verso)"


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


def chapter_folder(chapter: str, root: Path = ROOT) -> Path | None:
    """The folder of a chapter (18, 0A, B3) or of a checkpoint (CP1) that has a flashcards.csv, or None."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from start_chapter import canonical_id, checkpoint_number, find_chapter_dir

    chapter_id = canonical_id(chapter)
    part = checkpoint_number(chapter_id)
    folder = (root / "checkpoints" / f"partie_{part}" if part is not None
              else find_chapter_dir(chapter_id, root / "chapitres"))
    return folder if folder is not None and (folder / "flashcards.csv").exists() else None


def chapter_of(folder: Path) -> str | None:
    """Id of the chapter of a folder: ch18_backprop -> 18, ch00a_python -> 0A, b3_transformers -> B3, partie_2 -> CP2."""
    name = folder.name
    m = re.fullmatch(r"ch(\d+)([ab]?)(?:_.*)?", name)
    if m:
        return f"{int(m.group(1))}{m.group(2).upper()}"
    m = re.fullmatch(r"(b\d+)(?:_.*)?", name)
    if m:
        return m.group(1).upper()
    m = re.fullmatch(r"partie_(\d)", name)
    return f"CP{m.group(1)}" if m else None


def _syllabus():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import syllabus

    return syllabus


@functools.cache
def _chapters() -> dict:
    return _syllabus().load()


def deck_of(folder: Path) -> str:
    """The sub-deck of a chapter: ``Deep Learning Workbook::Partie IV::Ch. 18 Rétropropagation``.

    Numbers have two digits (« Ch. 07 ») because Anki sorts decks by name; the parts sort by themselves (0, I, II,
    III, IV, V, VI, VII). A folder unknown to the syllabus keeps its own name, under the deck itself.
    """
    syllabus = _syllabus()
    cid = chapter_of(folder)
    chapters = _chapters()
    if cid is None or cid not in chapters:
        return f"{DECK}::{folder.name}"
    title = chapters[cid]["title"].replace("::", ":")
    if syllabus.is_cp(cid):
        label = f"Checkpoint {cid[2:]} {title}"
    else:
        label = f"Ch. {cid.zfill(2) if cid.isdigit() else cid} {title}"
    return f"{DECK}::Partie {syllabus.part_of(cid)}::{label}"


def part_tag(folder: Path) -> str | None:
    """The hierarchical tag of a chapter's cards: ``dlwb::partie_IV::ch18`` (``dlwb::partie_I::cp1``)."""
    syllabus = _syllabus()
    cid = chapter_of(folder)
    if cid is None or cid not in syllabus.ORDER:
        return None
    return f"dlwb::partie_{syllabus.part_of(cid)}::{folder.name.split('_')[0] if not syllabus.is_cp(cid) else cid.lower()}"


def collect(files: list[Path], root: Path = ROOT, out=print) -> tuple[list[list[str]], list[str]]:
    """Cards of the files as ``[front, back, tags, deck]``, and the problems found (printed with the duplicates)."""
    all_cards, problems, seen, duplicates = [], [], {}, []
    for path in files:
        cards, issues = read_cards(path, root)
        problems += issues
        deck, extra = deck_of(path.parent), part_tag(path.parent)
        for front, back, tags in cards:
            key = front.lower()
            if key in seen:
                duplicates.append(f"« {front[:60]} » ({seen[key]} et {path.parent.name})")
            else:
                seen[key] = path.parent.name
            if extra and extra not in tags.split():
                tags = f"{tags} {extra}".strip()
            all_cards.append([front, back, tags, deck])
    for problem in problems:
        out("⚠️ " + problem)
    for dup in duplicates:
        out("ℹ️ recto en double : " + dup)
    return all_cards, problems


def escape_text(field: str) -> str:
    """A field ready for Anki's HTML: a « < » or a « & » of the text (« \\( 0 < x \\) », « a & b ») becomes
    ``&lt;`` or ``&amp;``, as the Anki manual asks; the tags (``<code>``, ``</b>``, ``<br>``) and entities stay."""
    field = re.sub(r"&(?!(?:[A-Za-z]+|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", field)
    return re.sub(r"<(?![A-Za-z/!])", "&lt;", field)


def write_csv(cards: list[list[str]], output: Path) -> None:
    cards = [[escape_text(front), escape_text(back), tags, deck] for front, back, tags, deck in cards]
    buffer = io.StringIO()
    buffer.write(HEADER)
    writer = csv.writer(buffer, delimiter=";", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerows(cards)
    output.write_text(buffer.getvalue(), encoding="utf-8")


def _stable_id(text: str, base: int = 1_600_000_000) -> int:
    """A deck id that does not change from one export to the next (genanki needs an int)."""
    return base + int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 400_000_000


def write_apkg(cards: list[list[str]], output: Path, genanki) -> None:
    """An Anki package: one deck per chapter, a two-field note type, a GUID per (deck, front) for updates."""
    model = genanki.Model(
        MODEL_ID, MODEL_NAME,
        fields=[{"name": "Recto"}, {"name": "Verso"}],
        templates=[{"name": "Carte 1", "qfmt": "{{Recto}}",
                    "afmt": "{{FrontSide}}<hr id=answer>{{Verso}}"}],
        css=".card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }",
    )
    decks: dict[str, object] = {}
    for front, back, tags, deck in cards:
        if deck not in decks:
            decks[deck] = genanki.Deck(_stable_id(deck), deck)
        chapter = tags.split()[0].split("::")[1] if tags.count("::") >= 1 else deck    # « ch18 » of dlwb::ch18::…
        note = genanki.Note(model=model, fields=[escape_text(front), escape_text(back)], tags=tags.split(),
                            guid=genanki.guid_for(chapter, front))      # the GUID of the text as written
        decks[deck].add_note(note)
    genanki.Package(list(decks.values())).write_to_file(str(output))


def load_genanki():
    """genanki, or None when it is not installed."""
    try:
        import genanki
    except ImportError:
        return None
    return genanki


def export(output: Path, root: Path = ROOT, check_only: bool = False, out=print, chapter: str | None = None,
           apkg: bool = False) -> int:
    """Write the deck of every chapter, or of one ``chapter`` (its own sub-deck), to ``output``."""
    files = []
    if chapter is not None:
        folder = chapter_folder(chapter, root)
        if folder is None:
            out(f"❌ Pas de flashcards pour le chapitre {chapter} (dossier absent ou sans flashcards.csv).")
            return 2
        files = [folder / "flashcards.csv"]
    else:
        for pattern in GLOBS:
            files += root.glob(pattern)
        order = _syllabus().ORDER
        files.sort(key=lambda f: (order.index(chapter_of(f.parent)) if chapter_of(f.parent) in order else len(order),
                                  f.as_posix()))          # study order: 0A, 0B, 1… B8
    genanki = None
    if apkg and not check_only:
        genanki = load_genanki()
        if genanki is None:
            out("❌ L'export .apkg demande genanki, qui n'est pas installé. Installe-le avec "
                "`pip install genanki` (ou `pip install -e \".[anki]\"`), puis relance la commande ; sans lui, "
                "importe le fichier CSV, qui contient les mêmes cartes (annexes/anki.md).")
            return 2
    cards, problems = collect(files, root, out)
    decks = sorted({card[3] for card in cards})
    out(f"{len(cards)} carte(s) dans {len(files)} fichier(s), {len(decks)} sous-paquet(s).")
    if check_only:
        return 1 if problems else 0
    output.parent.mkdir(parents=True, exist_ok=True)
    if apkg:
        write_apkg(cards, output, genanki)
        how = "Anki : Fichier > Importer, ou double-clic sur le fichier"
    else:
        write_csv(cards, output)
        how = "Anki : Fichier > Importer, type de note « Basique »"
    shown = output.relative_to(root) if output.is_relative_to(root) else output
    out(f"✅ Deck écrit : {shown}  ({how} ; pas à pas : annexes/anki.md)")
    return 1 if problems else 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # emoji on Windows consoles and pipes
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output file (default: exports/dlwb_anki.csv, or exports/dlwb_anki_<chapter>.csv)")
    parser.add_argument("--chapter", help="only the flashcards of one chapter (3, 0A, B3, CP1), in its own sub-deck")
    parser.add_argument("--apkg", action="store_true", help="write an Anki package (.apkg) with genanki")
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args(argv)
    output = args.output
    if output is None:
        folder = chapter_folder(args.chapter) if args.chapter else None
        output = ROOT / "exports" / f"dlwb_anki_{folder.name}.csv" if folder is not None else DEFAULT_OUTPUT
        if args.apkg:
            output = output.with_suffix(".apkg")
    return export(output, check_only=args.check, chapter=args.chapter, apkg=args.apkg)


if __name__ == "__main__":
    sys.exit(main())
