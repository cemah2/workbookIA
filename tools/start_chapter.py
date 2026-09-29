#!/usr/bin/env python
"""Copy a chapter's exercise notebook and mylearn stubs into mon_travail/.

    python tools/start_chapter.py 18        # chapter 18
    python tools/start_chapter.py 0A        # prerequisites chapter 0A
    python tools/start_chapter.py B3        # bonus chapter B3
    python tools/start_chapter.py --init    # only create mon_travail/mylearn (base files)
    python tools/start_chapter.py 18 --dry-run

What is copied (NEVER overwriting an existing file):
  chapitres/<chapter>/03_notebook.ipynb   -> mon_travail/<chapter>/03_notebook.ipynb
  chapitres/<chapter>/06_mes_reponses.md  -> mon_travail/<chapter>/06_mes_reponses.md
  mylearn stubs listed in templates/mylearn_stubs/MANIFEST.json
      (the "base" files + the chapter's files) -> mon_travail/mylearn/
  your tracking files (models in suivi/)  -> mon_travail/suivi/
      tableau_de_bord.md, auto_evaluation.md, journal.md: copied once; later,
      sections published by Claude (<!-- wb:section ID --> ... <!-- wb:end ID -->)
      are APPENDED to your copy if missing. Your ticks and notes are never touched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapitres"
STUBS = ROOT / "templates" / "mylearn_stubs"
WORK = ROOT / "mon_travail"
CHAPTER_FILES = ("03_notebook.ipynb", "06_mes_reponses.md")
TRACKERS = ("tableau_de_bord.md", "auto_evaluation.md", "journal.md")
SECTION = re.compile(r"<!-- wb:section (\S+) -->.*?<!-- wb:end \1 -->\n?", re.S)


def canonical_id(raw: str) -> str:
    """'18' -> '18', '0a' -> '0A', 'b3' -> 'B3', 'ch18' -> '18', 'pf' -> 'PF'."""
    text = raw.strip().upper()
    text = re.sub(r"^CH", "", text)
    match = re.fullmatch(r"0*(\d+)([A-Z]?)", text)
    if match:
        number, letter = match.groups()
        return f"{int(number)}{letter}"
    if re.fullmatch(r"B\d+|PF", text):
        return text
    raise ValueError(f"identifiant de chapitre invalide : {raw!r} (exemples : 18, 0A, B3)")


def dir_prefix(chapter_id: str) -> str:
    """Folder prefix of a chapter: '18' -> 'ch18', '0A' -> 'ch00a', 'B3' -> 'b3'."""
    match = re.fullmatch(r"(\d+)([A-Z]?)", chapter_id)
    if match:
        number, letter = match.groups()
        return f"ch{int(number):02d}{letter.lower()}"
    return chapter_id.lower()


def find_chapter_dir(chapter_id: str, chapters: Path = CHAPTERS) -> Path | None:
    prefix = dir_prefix(chapter_id)
    if not chapters.exists():
        return None
    matches = sorted(p for p in chapters.iterdir()
                     if p.is_dir() and (p.name == prefix or p.name.startswith(prefix + "_")))
    return matches[0] if matches else None


def load_manifest(stubs: Path = STUBS) -> dict:
    path = stubs / "MANIFEST.json"
    if not path.exists():
        return {"base": [], "chapters": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def stub_files_for(chapter_id: str | None, manifest: dict, stubs: Path = STUBS) -> list[str]:
    """Base files + the chapter's files + the __init__.py of their sub-packages."""
    files = list(manifest.get("base", []))
    if chapter_id is not None:
        files += manifest.get("chapters", {}).get(chapter_id, [])
    extra = []
    for rel in files:
        parent = Path(rel).parent
        while str(parent) not in ("", "."):
            init = (parent / "__init__.py").as_posix()
            if (stubs / init).exists() and init not in files and init not in extra:
                extra.append(init)
            parent = parent.parent
    ordered = []
    for rel in extra + files:  # parents' __init__.py first
        if rel not in ordered:
            ordered.append(rel)
    return ordered


def copy_no_overwrite(src: Path, dst: Path, dry_run: bool) -> str:
    """Copy src to dst unless dst exists. Returns 'copied', 'kept' or 'missing'."""
    if not src.exists():
        return "missing"
    if dst.exists():
        return "kept"
    if dry_run:
        return "copied"
    data = src.read_bytes()  # read first: a read error leaves nothing behind
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dst.open("xb") as handle:  # "x": fails if the file appeared meanwhile
            handle.write(data)
    except FileExistsError:
        return "kept"
    except BaseException:  # disk full, Ctrl+C...: never leave a truncated file
        dst.unlink(missing_ok=True)
        raise
    return "copied"


def sync_trackers(model_dir: Path, work_dir: Path, dry_run: bool) -> list[tuple[str, str]]:
    """Copy the tracking files once, then append the sections missing from the learner's copy."""
    results = []
    for name in TRACKERS:
        model, mine = model_dir / name, work_dir / name
        if not model.exists():
            continue
        if mine.exists():
            status = "kept"
        elif dry_run:
            status = "copied"
        else:  # first copy, without the "model kept by Claude" banner
            lines = model.read_text(encoding="utf-8").splitlines(keepends=True)
            kept = [line for line in lines if not line.startswith("> **Modèle")]
            mine.parent.mkdir(parents=True, exist_ok=True)
            try:
                with mine.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write("".join(kept).replace("\n\n\n", "\n\n"))
                status = "copied"
            except FileExistsError:
                status = "kept"
        if status == "kept":
            text = mine.read_text(encoding="utf-8")
            missing = [m.group(0) for m in SECTION.finditer(model.read_text(encoding="utf-8"))
                       if f"<!-- wb:section {m.group(1)} -->" not in text]
            if missing:
                status = "appended"
                if not dry_run:
                    with mine.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(("" if text.endswith("\n") else "\n") + "\n" + "\n".join(missing))
        results.append((f"mon_travail/suivi/{name}", status))
    return results


def start_chapter(chapter: str | None, *, root: Path = ROOT, dry_run: bool = False,
                  out=print) -> int:
    chapters = root / "chapitres"
    stubs = root / "templates" / "mylearn_stubs"
    work = root / "mon_travail"
    manifest = load_manifest(stubs)
    results: list[tuple[str, str]] = []

    chapter_id = None
    if chapter is not None:
        chapter_id = canonical_id(chapter)
        chapter_dir = find_chapter_dir(chapter_id, chapters)
        if chapter_dir is None:
            available = sorted(p.name for p in chapters.iterdir() if p.is_dir()) if chapters.exists() else []
            out(f"❌ Le chapitre {chapter_id} n'existe pas encore dans chapitres/.")
            out("   Chapitres disponibles : " + (", ".join(available) or "aucun pour l'instant"))
            out("   Astuce : `git pull` pour récupérer les derniers chapitres.")
            return 1
        for name in CHAPTER_FILES:
            status = copy_no_overwrite(chapter_dir / name, work / chapter_dir.name / name, dry_run)
            results.append((f"mon_travail/{chapter_dir.name}/{name}", status))

    for rel in stub_files_for(chapter_id, manifest, stubs):
        status = copy_no_overwrite(stubs / rel, work / "mylearn" / rel, dry_run)
        results.append((f"mon_travail/mylearn/{rel}", status))
    results += sync_trackers(root / "suivi", work / "suivi", dry_run)

    icons = {"copied": "✅ copié  ", "kept": "🔒 conservé", "missing": "⚠️ absent ",
             "appended": "➕ complété"}
    title = f"Chapitre {chapter_id}" if chapter_id else "Initialisation de mylearn"
    out(f"{title}{' (simulation, rien n’est écrit)' if dry_run else ''} :")
    for path, status in results:
        out(f"  {icons[status]} {path}")
    if any(status == "kept" for _, status in results):
        out("  🔒 = fichier déjà présent : ton travail n'a pas été touché.")
    if any(status == "appended" for _, status in results):
        out("  ➕ = nouvelles sections ajoutées à la fin de ton fichier (le reste est intact).")
    if chapter_id:
        out(f"👉 Ouvre mon_travail/{find_chapter_dir(chapter_id, chapters).name}/03_notebook.ipynb")
    out("👉 Teste ta librairie : python -m pytest tests/ -q")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # emoji on Windows consoles and pipes
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chapter", nargs="?", help="chapter id: 18, 0A, B3...")
    parser.add_argument("--init", action="store_true", help="only create mon_travail/mylearn with the base files")
    parser.add_argument("--dry-run", action="store_true", help="show what would be copied")
    args = parser.parse_args(argv)
    if not args.chapter and not args.init:
        parser.error("indique un chapitre (ex. 18) ou --init")
    try:
        return start_chapter(None if args.init and not args.chapter else args.chapter, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
