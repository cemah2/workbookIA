#!/usr/bin/env python
"""Generate ``annexes/index_notions.md``, the index of the notions of the workbook (finalisation, session 79).

    python tools/syllabus.py build      # writes the index with the other generated documents
    python tools/build_index.py         # the index alone

The index is never written by hand. Its entries come from two sources:

- the glossary (``annexes/glossaire.md``): every retained term, with its equivalents in the other language and the
  chapter of its column « Ch. » (where it is introduced); an English equivalent of a term said in French gets a
  cross-reference (« *backpropagation* → voir rétropropagation »);
- the contract (``docs/syllabus/data/*.json``): every ``teaches`` notion of a chapter (merged with the glossary term
  of the same name), and the ``teaches`` and ``concepts`` entries that name a glossary term exactly, which add their
  chapter to the chapters that teach it.

For each entry: the section of the sheet that teaches it (a heading that names it, else its gloss or its bold
occurrence, else the section that names it most often), with a link and its anchor as GitHub computes it; the
exercises that practise it (those that cover the book section of that heading, then those whose title names it);
the chapters whose ``teaches`` name it inside a longer notion (« approfondie »); the chapters whose ``requires``
name it (« réutilisée »). A notion is found in a text as a whole word, without case, with an optional plural.
"""

from __future__ import annotations

import collections
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import syllabus  # noqa: E402

ROOT = syllabus.ROOT
OUTPUT = "annexes/index_notions.md"
GLOSSARY = "annexes/glossaire.md"
MAX_EXERCISES = 10              # exercises listed per entry (the count of the others follows)
MAX_TAUGHT = 3                  # chapters listed as « enseignée » (the others go to « approfondie »)
SKIPPED_HEADINGS = ("comment utiliser", "objectifs d'apprentissage", "l'essentiel en", "guide de lecture",
                    "pour aller plus loin", "liens avec les autres chapitres", "les pièges classiques",
                    "sommaire", "table des matières")
TERMINOLOGY = "tests/infra/test_terminology.py"     # its FORBIDDEN forms never become a cross-reference
# glossary sections: which language the second column holds
SECTION_KIND = {"Termes gardés en anglais": "fr", "Termes dits en français": "en"}


# ---------------------------------------------------------------------------------------------------- text helpers
def fold(text: str) -> str:
    """Sort key of French: no accents, no case, ligatures spelt out, leading symbols ignored."""
    text = unicodedata.normalize("NFD", text.replace("œ", "oe").replace("Œ", "Oe").replace("æ", "ae"))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return re.sub(r"^[^0-9a-z]+", "", text)


def norm(text: str) -> str:
    """The text searched: Markdown emphasis and code marks removed, apostrophes and spaces unified, lower case."""
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"[*`]{1,3}(?=\S)|(?<=\S)[*`]{1,3}", "", text)
    return re.sub(r"\s+", " ", text).lower()


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading: rendered text, lower case, punctuation and emoji removed, spaces → -."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)            # links keep their text
    text = re.sub(r"[*`]", "", text).replace("$", "").lower()
    text = "".join(c for c in text if c in " -_" or unicodedata.category(c)[0] in "LNM")
    return text.replace(" ", "-")


def pattern(variant: str) -> re.Pattern | None:
    """A whole-word, case-free pattern for a variant (plural allowed); None for a variant too short to search."""
    v = norm(variant).strip()
    if len(v) < 2:
        return None
    return re.compile(r"(?<![\w-])" + re.escape(v) + r"(?:s|es|x)?(?![\w-])")


def split_cells(line: str) -> list[str]:
    """Cells of a Markdown table row; a ``|`` inside ``$…$``, inside code or escaped (``\\|``) is not a separator."""
    cells, cur, math, code, i = [], "", False, False, 1
    s = line.strip()
    while i < len(s):
        c = s[i]
        if c == "\\" and s[i + 1:i + 2] == "|":
            cur += "\\|"
            i += 2
            continue
        if c == "`":
            code = not code
        elif c == "$" and not code:
            math = not math
        if c == "|" and not math and not code:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += c
        i += 1
    return cells


def variants_of(term: str) -> list[str]:
    """The names a term is searched under: its comma-separated forms, without parentheses, plus an acronym given
    in parentheses (« attention multi-têtes (MHA) » → « attention multi-têtes », « MHA »)."""
    acronyms = [a for a in re.findall(r"\(([^()]*)\)", term) if re.fullmatch(r"[A-Z][A-Za-z0-9-]{1,9}", a.strip())]
    bare = re.sub(r"\([^()]*\)", "", term)
    bare = re.sub(r"\*[^*]*\*", "", bare) if "(" in term else bare
    out = []
    for part in re.split(r",\s+|\s;\s", bare.replace("`", "")) + acronyms:
        part = re.sub(r"\s+", " ", part).strip(" .:;*")
        if part and part not in out and part != "—":
            out.append(part)
    return out


def search_variants(term: str) -> list[str]:
    """``variants_of``, plus each side of a slash: « on-policy / off-policy » is searched as both words."""
    out = variants_of(term)
    for v in list(out):
        out += [p.strip() for p in v.split(" / ") if " / " in v and p.strip() not in out]
    return out


# ---------------------------------------------------------------------------------------------------- sources
def read_glossary(root: Path = ROOT) -> list[dict]:
    """Rows of the glossary: term, the other-language forms, the kind of the second column, the chapters."""
    rows, kind = [], None
    for line in (root / GLOSSARY).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            kind = SECTION_KIND.get(line[3:].strip(), "other")
            continue
        if kind is None or not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = split_cells(line)
        if len(cells) < 4 or cells[0] == "Retenu":
            continue
        term, other, chapters = cells[0], cells[1], cells[-1] if cells[-1] else cells[-2]
        rows.append({"term": term, "other": [o for o in variants_of(other) if o != "—"], "kind": kind,
                     "chapters": [c.strip() for c in chapters.split(",") if c.strip()]})
    return rows


def doc_of(cid: str, chapters: dict, root: Path = ROOT) -> Path | None:
    """The page that teaches a chapter's notions: its sheet, the synthesis of a checkpoint, the brief of a project."""
    m = re.fullmatch(r"(CP|MP)(\d)", cid)
    if m:
        cp = chapters.get(f"CP{m.group(2)}")
        if cp is None:
            return None
        if m.group(1) == "CP":
            path = root / cp["dir"] / "05_synthese.md"
        else:
            path = root / (cp.get("miniproject") or {}).get("dir", "-") / "README.md"
    elif cid == "PF":
        path = root / chapters["PF"]["dir"] / "README.md" if "PF" in chapters else None
    elif cid in chapters:
        path = root / "chapitres" / chapters[cid]["dir"] / "01_fiche.md"
    else:
        return None
    return path if path is not None and path.exists() else None


def sections_of(path: Path) -> list[dict]:
    """Headings (levels 2 to 4) of a page with their anchor, the book section they belong to and their text."""
    out, seen, fence, ref = [], collections.Counter(), False, None
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,4}) (.+?)\s*$", line)
        if m:
            heading = m.group(2)
            base = slug(heading)
            anchor = base if not seen[base] else f"{base}-{seen[base]}"
            seen[base] += 1
            level = len(m.group(1))
            number = re.match(r"(\d+(?:\.\d+)+) · ", heading)
            if number:
                ref = number.group(1)
            elif level == 2:
                ref = None
            current = {"level": level, "heading": heading, "anchor": anchor, "ref": ref, "body": [],
                       "skip": level == 1 or norm(heading).startswith(SKIPPED_HEADINGS)}
            out.append(current)
        elif current is not None:
            current["body"].append(line)
    for s in out:
        raw = "\n".join(s["body"])
        s["text"] = norm(raw)
        s["head"] = norm(s["heading"])
        s["bold"] = norm(raw.replace("**", "\x00"))        # « **term** » becomes « \x00term\x00 »
        s["gloss"] = norm(raw.replace("(*", "\x01"))       # « term (*other* » becomes « term \x01other »
    return [s for s in out if not s["skip"]]


# ---------------------------------------------------------------------------------------------------- matching
def find_section(sections: list[dict], variants: list[str]) -> dict | None:
    """The section that teaches a notion: a heading that names it, else its gloss « term (*…* » or its bold
    occurrence, else the section that names it most often, else the most content words of a long notion."""
    pats = [p for p in map(pattern, variants) if p]
    if not pats or not sections:
        return None
    for s in sections:
        if any(p.search(s["head"]) for p in pats):
            return s
    keys = [norm(v).strip() for v in variants if pattern(v)]
    for s in sections:
        if any(f"\x00{k}\x00" in s["bold"] or f"{k} \x01" in s["gloss"] or f"{k}s \x01" in s["gloss"] for k in keys):
            return s
    counts = [(sum(len(p.findall(s["text"])) for p in pats), -i) for i, s in enumerate(sections)]
    best = max(counts)
    if best[0]:
        return sections[-best[1]]
    words = {w for v in variants for w in re.findall(r"\w{4,}|\b[A-Z][A-Z0-9]{1,}\b", v.replace("`", ""))}
    words = {w.lower() for w in words} - STOPWORDS
    if not words:
        return None
    for s in sections:
        if all(w in s["head"] for w in words):
            return s
    scored = [(sum(3 * (w in s["head"]) + (w in s["text"]) for w in words), -i) for i, s in enumerate(sections)]
    best = max(scored)
    return sections[-best[1]] if best[0] >= max(2, len(words)) else None


STOPWORDS = {"avec", "dans", "pour", "leur", "leurs", "entre", "sans", "sous", "comme", "plus", "moins", "d'un",
             "d'une", "des", "une", "est", "sont", "the", "and", "from", "with", "scratch", "pytorch", "numpy"}


class Corpus:
    """Short texts (exercise titles, notions) searched many times: one joined string, a substring test first."""

    def __init__(self, items: list[tuple[str, str]]):
        import bisect

        self._bisect = bisect.bisect_right
        self.keys, self.starts, parts, pos = [], [], [], 0
        for key, text in items:
            self.keys.append(key)
            self.starts.append(pos)
            parts.append(text)
            pos += len(text) + 1
        self.text = "\n".join(parts)

    def find(self, pats: list[re.Pattern], probes: list[str]) -> list[str]:
        """Keys of the texts that one of the patterns matches, in their order, once each."""
        hits = set()
        for probe in probes:                    # str.find, then the whole-word test of ``pattern`` by hand
            at = self.text.find(probe)
            while at >= 0:
                end = at + len(probe)
                for suffix in ("", "s", "es", "x"):
                    if self.text.startswith(suffix, end) and _boundary(self.text, at, end + len(suffix)):
                        hits.add(self._bisect(self.starts, at) - 1)
                        break
                at = self.text.find(probe, at + 1)
        return [self.keys[i] for i in sorted(hits)]


def _boundary(text: str, start: int, end: int) -> bool:
    """No letter, digit, ``_`` or ``-`` on either side of ``text[start:end]`` (the look-arounds of ``pattern``)."""
    def word(c):
        return c.isalnum() or c in "_-"
    return (start == 0 or not word(text[start - 1])) and (end >= len(text) or not word(text[end]))


# ---------------------------------------------------------------------------------------------------- entries
def collect_entries(chapters: dict, root: Path = ROOT) -> tuple[list[dict], list[tuple[str, str]]]:
    """Index entries (merged glossary terms and taught notions) and cross-references (other form, term)."""
    entries: dict[str, dict] = {}

    def entry(term: str) -> dict:
        key = norm(term)
        if key not in entries:
            entries[key] = {"term": term, "other": [], "kind": None, "taught": [], "variants": variants_of(term),
                            "search": search_variants(term)}
        return entries[key]

    cross = []
    for row in read_glossary(root):
        e = entry(row["term"])
        e["kind"] = e["kind"] or row["kind"]
        e["other"] += [o for o in row["other"] if o not in e["other"] and norm(o) != norm(row["term"])]
        e["taught"] += [c for c in row["chapters"] if c not in e["taught"]]
        if row["kind"] != "fr":                       # an English form leads to the retained term
            cross += [(o, row["term"], True) for o in row["other"]]
        cross += [(v, row["term"], False) for v in e["variants"][1:]]     # « PPO » leads to « objectif clippé, PPO »
    by_variant = {}
    for e in entries.values():
        for v in e["variants"]:
            by_variant.setdefault(norm(v), e)
    for cid, ch in chapters.items():
        for notion in ch.get("teaches", []):
            first = re.split(r",\s+|\s/\s", notion)[0]      # « softmax, log-softmax, température » → softmax
            e = (by_variant.get(norm(notion)) or by_variant.get(norm(re.sub(r"\s*\([^()]*\)$", "", notion)))
                 or (by_variant.get(norm(first)) if first != notion else None))
            if e is None:
                e = entry(notion)
                by_variant.setdefault(norm(notion), e)
            if cid not in e["taught"]:
                e["taught"].append(cid)
        for concept in ch.get("concepts", []):
            e = by_variant.get(norm(concept))
            if e is not None and cid not in e["taught"]:
                e["taught"].append(cid)
    for e in entries.values():          # study order, the glossary chapter first when it is a chapter
        order = {c: syllabus.ORDER.index(c) if c in syllabus.ORDER else 99 for c in e["taught"]}
        first = e["taught"][:1]
        e["taught"] = first + sorted((c for c in e["taught"][1:]), key=lambda c: order[c])
    names = {norm(e["term"]) for e in entries.values()}
    forbidden = forbidden_forms(root)
    refs = collections.defaultdict(list)
    for form, term, italic in cross:
        if forbidden and forbidden.search(form):        # a set-aside French form stays a gloss of its entry (§5)
            continue
        if norm(form) not in names and term not in refs[(form, italic)]:
            refs[(form, italic)].append(term)
    return list(entries.values()), sorted(refs.items(), key=lambda kv: (fold(kv[0][0]), kv[0][0]))


def forbidden_forms(root: Path = ROOT) -> re.Pattern | None:
    """The French forms that BIBLE §5 sets aside (``FORBIDDEN`` of the terminology test), as one pattern.

    A cross-reference from such a form would put it in the prose: it stays a gloss of its entry, as in
    « **early stopping** (*arrêt anticipé*) ». None if the test cannot be read.
    """
    import importlib.util

    path = root / TERMINOLOGY
    if not path.exists():
        path = ROOT / TERMINOLOGY
    try:
        spec = importlib.util.spec_from_file_location("_terminology", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:           # pytest missing, or the file moved: no filter
        return None
    return re.compile("|".join(f"(?:{pat})" for pat, _ in module.FORBIDDEN.values()))


def checkpoint_label(cid: str) -> str:
    return {"CP": "checkpoint", "MP": "mini-projet"}.get(cid[:2], "ch.") + (f" {cid[2:]}" if cid[:2] in ("CP", "MP")
                                                                          else f" {cid}")


def build_index(chapters: dict, root: Path = ROOT) -> str:
    entries, cross = collect_entries(chapters, root)
    exercises = [(cid, ex) for cid, ch in chapters.items() for ex in ch["exercises"]]
    titles = Corpus([(ex["id"], norm(ex["title"])) for _, ex in exercises])
    covers = collections.defaultdict(list)          # (chapter, book section) -> exercise ids
    for cid, ex in exercises:
        for ref in ex["covers"]:
            covers[(cid, ref)].append(ex["id"])
    requires = Corpus([(cid, norm(r["notion"])) for cid, ch in chapters.items() for r in ch.get("requires", [])])
    teaches = Corpus([(cid, norm(t)) for cid, ch in chapters.items() for t in ch.get("teaches", [])])
    pages: dict[str, tuple[Path | None, list[dict]]] = {}

    def page(cid):
        if cid not in pages:
            path = doc_of(cid, chapters, root)
            pages[cid] = (path, sections_of(path) if path else [])
        return pages[cid]

    out_dir = (root / OUTPUT).parent
    lines_by_letter = collections.defaultdict(list)
    links: dict[str, str] = {}              # link target -> label of its reference-style link (a smaller file)

    def ref(target: str) -> str:
        return links.setdefault(target, f"s{len(links) + 1}")
    for e in entries:
        pats = [p for p in map(pattern, e["search"]) if p]
        probes = [norm(v).strip() for v in e["search"] if pattern(v)]
        taught = e["taught"][:MAX_TAUGHT]
        parts, ids = [], []
        for cid in taught:
            path, secs = page(cid)
            label = checkpoint_label(cid)
            sec = find_section(secs, e["search"]) if path else None
            if path is None:
                parts.append(label)
                continue
            link = Path(_relpath(path, out_dir)).as_posix()
            if sec is None:
                parts.append(f"[{label}][{ref(link)}]")
                continue
            title = re.sub(r"\s*[⏩🕰️⚠️🔗📐💼🧮]+\s*$", "", sec["heading"]).strip()
            title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)
            if len(title) > 70:
                title = title[:67].rsplit(" ", 1)[0] + "…"
            title = title.replace("[", "(").replace("]", ")")
            parts.append(f"[{label}, {title}][{ref(link + '#' + sec['anchor'])}]")
            if sec["ref"] and cid in chapters:
                ids += covers.get((cid, sec["ref"]), [])
        ids = list(dict.fromkeys(ids + titles.find(pats, probes)))
        deeper = [cid for cid in teaches.find(pats, probes) if cid not in e["taught"]]
        reused = [cid for cid in requires.find(pats, probes) if cid not in e["taught"]]
        name = f"**{e['term']}**"
        if e["other"]:
            name += " (*" + ", ".join(o.replace("*", "") for o in e["other"]) + "*)"
        fields = []
        if parts:
            fields.append("enseignée : " + " ; ".join(parts))
        if ids:
            shown = sorted(ids[:MAX_EXERCISES], key=_exercise_key)
            more = f" (+ {len(ids) - MAX_EXERCISES})" if len(ids) > MAX_EXERCISES else ""
            fields.append("exercices : " + syllabus.compress_ids(shown) + more)
        extra = [c for c in e["taught"][MAX_TAUGHT:]] + deeper
        if extra:
            fields.append("approfondie : " + _chapter_list(extra))
        if reused:
            fields.append("réutilisée : " + _chapter_list(reused))
        lines_by_letter[_letter(e["term"])].append((fold(e["term"]), e["term"], f"- {name} — " + " · ".join(fields)))
    for (form, italic), terms in cross:
        targets = ", ".join(f"[{t}](#{_letter(t).lower() if _letter(t) != '#' else 'symboles'})" for t in terms)
        shown = f"*{form.replace('*', '')}*" if italic else form
        lines_by_letter[_letter(form)].append((fold(form), form, f"- {shown} → voir {targets}"))
    letters = sorted(lines_by_letter, key=lambda k: (k == "#", k))
    n_entries, n_cross = len(entries), len(cross)
    head = [
        "# Index des notions",
        "",
        "> **Généré** par `python tools/syllabus.py build` (`tools/build_index.py`) à partir du glossaire "
        "(`annexes/glossaire.md`), du contrat de chaque chapitre (`docs/syllabus/data/*.json` : notions enseignées, "
        "concepts, notions mobilisées, exercices) et des titres des fiches : on ne l'édite pas à la main.",
        "",
        f"**{n_entries} notions**, et **{n_cross} renvois** depuis l'autre langue ou un synonyme, dans l'ordre "
        "alphabétique (accents ignorés). Pour chaque notion :",
        "",
        "- **enseignée** : le chapitre qui l'introduit (colonne « Ch. » du glossaire), puis ceux qui l'enseignent "
        "(leurs « notions enseignées »), chacun avec un lien vers la section de sa fiche qui la présente : un titre "
        "qui la nomme, sinon sa première définition (en gras ou suivie de sa glose), sinon la section qui en parle "
        "le plus ;",
        f"- **exercices** : ceux qui couvrent cette section du livre, puis ceux dont le titre la nomme ({MAX_EXERCISES} "
        "au plus, le nombre des autres entre parenthèses) ;",
        "- **approfondie** : les chapitres dont une notion enseignée la contient ; **réutilisée** : les chapitres "
        "qui la mobilisent comme prérequis.",
        "",
        "Entre parenthèses et en italique, l'autre langue. Une ligne « *terme* → voir … » renvoie de l'autre langue "
        "(le plus souvent l'anglais) vers le terme retenu (BIBLE §5) ; une ligne « terme → voir … », d'un synonyme "
        "vers l'entrée qui le contient. "
        "Les définitions sont dans le [glossaire](glossaire.md), les formules dans le [formulaire](formulaire.md). "
        "La recherche est automatique : un lien qui tombe à côté se signale avec P6.",
        "",
        " · ".join(f"[{k if k != '#' else '0-9'}](#{k.lower() if k != '#' else 'symboles'})" for k in letters),
        "",
    ]
    body = []
    for k in letters:
        body += ['<a id="symboles"></a>', "", "## 0-9 et symboles"] if k == "#" else [f"## {k}"]
        body.append("")
        body += [line for _, _, line in sorted(lines_by_letter[k], key=lambda t: (t[0], t[1]))]
        body.append("")
    body += [f"[{label}]: {target}" for target, label in links.items()]
    return "\n".join(head + body).rstrip() + "\n"


def _letter(term: str) -> str:
    key = fold(term)
    return key[0].upper() if key and key[0].isalpha() else "#"


def _exercise_key(eid: str):
    cid, suffix = eid.split(".", 1)
    m = re.fullmatch(r"([A-Z]?)(\d+)([a-z]?)", suffix)
    return (syllabus.ORDER.index(cid) if cid in syllabus.ORDER else 99, m.group(1) if m else suffix,
            int(m.group(2)) if m else 0)


def _chapter_list(cids: list[str]) -> str:
    seen = list(dict.fromkeys(cids))
    seen.sort(key=lambda c: syllabus.ORDER.index(c) if c in syllabus.ORDER else 99)
    return ", ".join(checkpoint_label(c) if c[:2] in ("CP", "MP") else c for c in seen).join(
        ("ch. ", "")) if seen else ""


def _relpath(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    chapters = syllabus.load()
    (ROOT / OUTPUT).write_text(build_index(chapters), encoding="utf-8")
    print(f"✅ {OUTPUT} généré.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
