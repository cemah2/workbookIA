#!/usr/bin/env python
"""Generate ``annexes/index_notions.md``, the index of the notions of the workbook (finalisation, session 79).

    python tools/syllabus.py build      # writes the index with the other generated documents
    python tools/build_index.py         # the index alone

The index is never written by hand. Its entries come from two sources:

- the glossary (``annexes/glossaire.md``): every retained term, with its equivalents in the other language and the
  chapter of its column « Ch. » (where it first appears); an English equivalent of a term said in French gets a
  cross-reference (« *backpropagation* → voir rétropropagation »);
- the contract (``docs/syllabus/data/*.json``): every ``teaches`` notion of a chapter (merged with the glossary term
  of the same name, of the same first element, or of the same name without « from scratch », « en PyTorch »…), and
  the ``teaches`` and ``concepts`` entries that name a glossary term exactly, which add their chapter to the
  chapters that teach it.

For each entry:

- **enseignée**: the chapters whose sheet teaches it, the main lesson first. In each sheet, the section that teaches
  it is, by strength: a heading about it (« 20.4.3 · Le dropout »: 4), a heading that names it (3), its first
  definition, in bold or followed by its gloss (2), else the section that names it most often (1). The chapters are
  sorted by strength, then in study order; a chapter where the notion is only mentioned (strength 1) while another
  one defines it goes to « évoquée ». A notion found in none of its chapters is looked for (heading or definition)
  in every sheet;
- **exercices**: those whose title names it, then those whose statement (``02_exercices.md``, the notebooks, the
  exam subjects, the briefs of the projects) names it: first those that cover a section found above, then those of
  the chapters that teach it, then the others; a notion named by no statement (a long notion of the contract) keeps
  the exercises that cover the book section of its heading;
- **approfondie**: the chapters whose ``teaches`` name it inside a longer notion; **réutilisée**: the chapters
  whose ``requires`` name it.

A notion is found in a text as a whole word, without case, with an optional plural. Outside the sheets, a term is
searched under its main name, its acronyms and its forms of several words (« acteur-critique, acteur, critique »
is not searched as « critique »), and a term that shares its name with another entry (« noyau (d'un SVM) ») only in
the chapters that teach it. A word that starts three entries or more without being one (« attention ») gets a
cross-reference to them.
"""

from __future__ import annotations

import collections
import functools
import json
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
MAX_SAID = 100                  # a notion named by more statements is only looked for in its own chapters
MAX_GROUP = 12                  # entries listed by the cross-reference of a head word (« attention → voir … »)
SKIPPED_HEADINGS = ("comment utiliser", "objectifs d'apprentissage", "l'essentiel en", "guide de lecture",
                    "pour aller plus loin", "liens avec les autres chapitres", "les pièges classiques",
                    "sommaire", "table des matières")
TERMINOLOGY = "tests/infra/test_terminology.py"     # its FORBIDDEN forms never become a cross-reference
# glossary sections: which language the second column holds
SECTION_KIND = {"Termes gardés en anglais": "fr", "Termes dits en français": "en"}
# where the statements of the exercises are: Markdown files, and the Markdown cells of notebooks
STATEMENT_PAGES = ("chapitres/*/02_exercices.md", "checkpoints/*/01_examen_sujet.md", "projets/*/README.md")
STATEMENT_NOTEBOOKS = ("chapitres/*/03_notebook.ipynb", "checkpoints/*/02_examen_notebook.ipynb",
                       "projets/*/depart/*.ipynb")
EXERCISE_HEADING = re.compile(r"^#{2,4} (?:Ex )?((?:\d+[AB]?|B\d|CP\d|MP\d|PF)\.[A-Z]?\d+)[a-z]?\b")
# a teaches notion that only adds how it is practised is merged with the entry it names (« LoRA from scratch »)
QUALIFIER = re.compile(r"\s+(?:from scratch|en pytorch|en numpy|en pratique|avec (?:pytorch|scikit-learn|numpy))$"
                       r"|\s*\((?:usage|aperçu|intuition|bases?)\)$", re.I)
# what may precede the notion in a heading « about » it: a number, « Au-delà du livre (2) : », an article
HEADING_LEAD = re.compile(r"^(?:\d+(?:\.\d+)+ · )?(?:au-delà du livre(?: \(\d\))? : )?"
                          r"(?:le |la |les |l'|un |une |des |du |de la |de l')?")


# ---------------------------------------------------------------------------------------------------- text helpers
def fold(text: str) -> str:
    """Sort key of French: no accents, no case, ligatures spelt out, leading symbols ignored."""
    text = unicodedata.normalize("NFD", text.replace("œ", "oe").replace("Œ", "Oe").replace("æ", "ae"))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return re.sub(r"^[^0-9a-z]+", "", text)


def norm(text: str) -> str:
    """The text searched: Markdown emphasis and code marks removed, apostrophes and spaces unified, lower case."""
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"[*`]{1,3}(?=\S)|(?<=\S)[*`]{1,3}", "", text)
    return re.sub(r"\s+", " ", text).lower()


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading: its text in lower case, without the characters outside ``\\p{Word}``
    (letters, marks, decimal digits, connector punctuation), ``-`` and spaces; each space becomes ``-``.

    Emoji and punctuation go, a subscript or superscript digit too (« log₂ » → « log »); a variation selector
    (U+FE0F, a mark) stays, as on GitHub.
    """
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)            # links keep their text
    text = re.sub(r"[*`]", "", text).replace("$", "").lower()
    text = "".join(c for c in text if c in " -" or unicodedata.category(c)[0] in "LM"
                   or unicodedata.category(c) in ("Nd", "Pc"))
    return text.replace(" ", "-")


@functools.lru_cache(maxsize=None)
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


def strong_variants(variants: list[str]) -> list[str]:
    """The variants searched outside the sheets: the main name, acronyms, and the forms of several words.

    « acteur-critique, acteur, critique » is searched as « acteur-critique » only: « critique » alone also names the
    critical points of ch. 5.
    """
    first = variants[0].split(" / ") if variants else []
    return [v for i, v in enumerate(variants)
            if i == 0 or v in first or " " in v.strip()
            or re.fullmatch(r"[A-Z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9-]*", v.strip())]


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
    """Headings (levels 2 to 6) of a page with their anchor, the book section they belong to and their text.

    Every heading counts for the anchors (GitHub numbers a repeated anchor « -1 », « -2 »…), the lines of a fenced
    block (```` ``` ```` or ``~~~``) are not headings.
    """
    out, seen, fence, ref = [], collections.Counter(), None, None
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        mark = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if mark:
            if fence is None:
                fence = mark.group(1)[0]
            elif mark.group(1)[0] == fence:
                fence = None
        m = None if fence or mark else re.match(r"^ {0,3}(#{1,6}) (.+?)(?:\s+#+)?\s*$", line)
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
                       "numbered": bool(number),
                       "skip": level == 1 or norm(heading).startswith(SKIPPED_HEADINGS)}
            out.append(current)
        elif current is not None:
            current["body"].append(line)
    for s in out:
        raw = "\n".join(s.pop("body"))
        s["text"] = norm(raw)
        s["head"] = norm(s["heading"])
        s["core"] = HEADING_LEAD.sub("", s["head"])     # « 20.4.3 · le dropout ⏩ » → « dropout ⏩ »
        s["bold"] = norm(raw.replace("**", "\x00"))        # « **term** » becomes « \x00term\x00 »
        s["gloss"] = norm(raw.replace("(*", "\x01"))       # « term (*other* » becomes « term \x01other »
        s["bolds"] = [b for b in s["bold"].split("\x00")[1::2] if len(b) < 120]     # the phrases in bold
    return [s for s in out if not s["skip"]]


# ---------------------------------------------------------------------------------------------------- matching
def _about(core: str, key: str) -> bool:
    """A heading whose core (number and article removed) starts with the notion, then stops or goes on with
    « : », « , », « ( », « ; », « et », « ? »: « dropout », « attention multi-têtes : projections… »; the words
    are compared without their plural (« Réseaux convolutifs » is about « réseau convolutif »)."""
    words, heading = key.split(), core.split()
    if not words or len(heading) < len(words):
        return False
    for i, (w, h) in enumerate(zip(words, heading)):
        last = i == len(words) - 1
        bare = h.rstrip(":,;?!") if last else h
        if _singular(w) != _singular(bare):
            return False
    rest = " ".join(heading[len(words):])
    tail = heading[len(words) - 1][len(heading[len(words) - 1].rstrip(":,;?!")):]
    return bool(tail) or not rest.strip(" ⏩🕰️⚠️🔗📐💼🧮") or bool(re.match(r"[:,(;?!—]|et\s", rest))


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word[-1] in "sx" else word


def find_section(sections: list[dict], variants: list[str]) -> tuple[dict | None, int]:
    """The section that teaches a notion, and the strength of the match (``0`` when none).

    4: a heading about it; 3: a heading that names it (or, for a long notion, all its content words); 2: its gloss
    « term (*…* » or its bold occurrence; 1: the section that names it most often, or that shares the most content
    words with a long notion.
    """
    pats = [p for p in map(pattern, variants) if p]
    if not pats or not sections:
        return None, 0
    keys = [norm(v).strip() for v in variants if pattern(v)]
    for s in sections:
        if any(_about(s["core"], k) for k in keys):
            return s, 4
    for s in sections:
        if any(p.search(s["head"]) for p in pats):
            return s, 3
    for s in sections:
        if any(f"{k} \x01" in s["gloss"] or f"{k}s \x01" in s["gloss"] for k in keys) or _bold(s, keys):
            return s, 2
    counts = [(sum(len(p.findall(s["text"])) for p in pats), -i) for i, s in enumerate(sections)]
    best = max(counts)
    if best[0]:
        return sections[-best[1]], 1
    words = {w for v in variants for w in re.findall(r"\w{4,}|\b[A-Z][A-Z0-9]{1,}\b", v.replace("`", ""))}
    words = {w.lower() for w in words} - STOPWORDS
    if not words:
        return None, 0
    for s in sections:
        if all(w in s["head"] for w in words):
            return s, 3
    scored = [(sum(3 * (w in s["head"]) + (w in s["text"]) for w in words), -i) for i, s in enumerate(sections)]
    best = max(scored)
    return (sections[-best[1]], 1) if best[0] >= max(2, len(words)) else (None, 0)


def find_definition(sections: list[dict], variants: list[str]) -> tuple[dict | None, int]:
    """A heading about the notion (4), a heading that names it (3) or its bold occurrence (2) in a page, or
    ``(None, 0)``: the evidence asked of a page that does not list the notion (a gloss « mot (*…* » is not enough,
    « système d'exploitation (*PagedAttention*) » is no definition of « exploitation »)."""
    pats = [p for p in map(pattern, variants) if p]
    keys = [norm(v).strip() for v in variants if pattern(v)]
    for s in sections:
        if any(_about(s["core"], k) for k in keys):
            return s, 4
    for s in sections:
        if any(p.search(s["head"]) for p in pats):
            return s, 3
    for s in sections:
        if _bold(s, keys):
            return s, 2
    return None, 0


def _bold(section: dict, keys: list[str]) -> bool:
    """The notion, or its plural, is a phrase in bold of the section (« **data cards** »)."""
    return any(b in (k, k + "s", k + "es", k + "x") for b in section["bolds"] for k in keys)


HEAD_SKIP = {"mise", "deep", "learning"}     # first words of compounds (« mise à jour », « deep learning »)
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

    def find(self, probes: list[str]) -> list[str]:
        """Keys of the texts that one of the probes matches (whole word, optional plural), in order, once each."""
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


class Statements:
    """The statements of the exercises (long texts): an index of their words, then the whole-word test."""

    def __init__(self, texts: dict[str, str]):
        self.texts = texts
        self.words: dict[str, set[str]] = collections.defaultdict(set)
        for eid, text in texts.items():
            for word in set(re.findall(r"\w+", text)):
                self.words[word].add(eid)

    def find(self, probes: list[str]) -> set[str]:
        hits = set()
        for probe in probes:
            tokens = re.findall(r"\w+", probe)
            if not tokens:
                continue
            last = tokens[-1]
            ids = set().union(*(self.words.get(last + s, set()) for s in ("", "s", "es", "x")))
            for token in tokens[:-1]:
                ids &= self.words.get(token, set())
                if not ids:
                    break
            hits |= {eid for eid in ids if eid not in hits and _contains(self.texts[eid], probe)}
        return hits


def _contains(text: str, probe: str) -> bool:
    """``pattern(probe).search(text)``, with ``str.find`` (several times faster on long texts)."""
    at = text.find(probe)
    while at >= 0:
        end = at + len(probe)
        if any(text.startswith(suffix, end) and _boundary(text, at, end + len(suffix)) for suffix in ("", "s", "es", "x")):
            return True
        at = text.find(probe, at + 1)
    return False


def read_statements(ids: set[str], root: Path = ROOT) -> dict[str, str]:
    """The statement of each exercise (its heading and what follows, up to the next heading of level 1 to 3), in
    the pages and notebooks of ``STATEMENT_PAGES`` and ``STATEMENT_NOTEBOOKS``; a sub-item (« 0A.14a ») joins its
    exercise."""
    out: dict[str, list[str]] = collections.defaultdict(list)

    def scan(text: str) -> None:
        current = None
        for line in text.splitlines():
            m = EXERCISE_HEADING.match(line)
            if m:
                current = m.group(1) if m.group(1) in ids else None
                if current:
                    out[current].append(line)
            elif re.match(r"^#{1,3} ", line):
                current = None
            elif current:
                out[current].append(line)

    for pattern_ in STATEMENT_PAGES:
        for path in sorted(root.glob(pattern_)):
            scan(path.read_text(encoding="utf-8"))
    for pattern_ in STATEMENT_NOTEBOOKS:
        for path in sorted(root.glob(pattern_)):
            try:
                cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
            except (ValueError, KeyError):
                continue
            scan("\n".join("".join(c.get("source", "")) for c in cells if c.get("cell_type") == "markdown"))
    return {eid: norm("\n".join(lines)) for eid, lines in out.items()}


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
                            "search": search_variants(term), "teaches": []}
        return entries[key]

    cross = []
    words_of = {}                       # « exploration / exploitation » and « exploration, exploitation »: one entry
    for row in read_glossary(root):
        words = frozenset(norm(v) for v in search_variants(row["term"]) if " / " not in v)
        e = entries.get(norm(row["term"])) or (words_of.get(words) if len(words) > 1 else None) or entry(row["term"])
        words_of.setdefault(words, e)
        e["kind"] = e["kind"] or row["kind"]
        same = {norm(v) for v in [row["term"], *e["search"], *variants_of(row["term"])]}
        e["other"] += [o for o in row["other"] if o not in e["other"] and norm(o) not in same]
        e["taught"] += [c for c in row["chapters"] if c not in e["taught"]]
        if row["kind"] != "fr":                       # an English form leads to the retained term
            cross += [(o, e["term"], True) for o in row["other"] if norm(o) not in same]
        own = e["variants"][1:] if e["term"] == row["term"] else variants_of(row["term"])
        cross += [(v, e["term"], False) for v in own]     # « PPO » leads to « objectif clippé, PPO »
    by_variant = {}
    for e in entries.values():
        for v in e["variants"]:
            by_variant.setdefault(norm(v), e)
    for cid, ch in chapters.items():
        for notion in ch.get("teaches", []):
            first = re.split(r",\s+|\s/\s", notion)[0]      # « softmax, log-softmax, température » → softmax
            plain = QUALIFIER.sub("", notion)                # « LoRA from scratch » → LoRA
            e = (by_variant.get(norm(notion)) or by_variant.get(norm(re.sub(r"\s*\([^()]*\)$", "", notion)))
                 or (by_variant.get(norm(first)) if first != notion else None)
                 or (by_variant.get(norm(plain)) if plain != notion else None))
            if e is None:
                e = entry(notion)
                by_variant.setdefault(norm(notion), e)
            if cid not in e["taught"]:
                e["taught"].append(cid)
            e["teaches"].append(cid)
        for concept in ch.get("concepts", []):       # « token, vocabulaire, identifiant » adds B2 to « token »
            e = by_variant.get(norm(concept)) or by_variant.get(norm(re.split(r",\s+", concept)[0]))
            if e is not None and cid not in e["taught"]:
                e["taught"].append(cid)
    names = {norm(e["term"]) for e in entries.values()}
    forbidden = forbidden_forms(root)
    refs = collections.defaultdict(list)
    for form, term, italic in cross:
        if forbidden and forbidden.search(form):        # a set-aside French form stays a gloss of its entry (§5)
            continue
        if norm(form) not in names and term not in refs[(form, italic)]:
            refs[(form, italic)].append(term)
    # a word that starts several entries without being one: « attention → voir attention additive, … »
    forms = names | {norm(form) for form, _ in refs}
    heads = collections.defaultdict(list)
    for e in entries.values():
        words = e["variants"][0].split() if e["variants"] else []
        if len(words) > 1 and len(words[0]) >= 4 and words[0].isalpha():
            head = words[0].lower()
            heads[head[:-1] if head.endswith("s") and head[:-1] in forms | set(heads) else head].append(e["term"])
    plain = {norm(form): (form, italic) for form, italic in refs if not italic}
    for head, terms in heads.items():
        if len(terms) < 3 or norm(head) in names or head in STOPWORDS | HEAD_SKIP \
                or (forbidden and forbidden.search(head)):
            continue
        if norm(head) in plain:             # « valeur → voir requête, clé, valeur » gets « valeurs aberrantes »…
            key = plain[norm(head)]
            refs[key] += [t for t in sorted(terms, key=lambda t: (fold(t), t)) if t not in refs[key]]
        elif norm(head) not in forms:
            refs[(head, False)] = sorted(terms, key=lambda t: (fold(t), t))
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
    if cid == "PF":
        return "projet final"
    return {"CP": "checkpoint", "MP": "mini-projet"}.get(cid[:2], "ch.") + (f" {cid[2:]}" if cid[:2] in ("CP", "MP")
                                                                          else f" {cid}")


def build_index(chapters: dict, root: Path = ROOT) -> str:
    entries, cross = collect_entries(chapters, root)
    exercises = [(cid, ex) for cid, ch in chapters.items() for ex in ch["exercises"]]
    chapter_of = {ex["id"]: cid for cid, ex in exercises}
    titles = Corpus([(ex["id"], norm(ex["title"])) for _, ex in exercises])
    statements = Statements(read_statements(set(chapter_of), root))
    covers = collections.defaultdict(list)          # (chapter, book section) -> exercise ids
    for cid, ex in exercises:
        for ref in ex["covers"]:
            covers[(cid, ref)].append(ex["id"])
    requires = Corpus([(cid, norm(r["notion"])) for cid, ch in chapters.items() for r in ch.get("requires", [])])
    teaches = Corpus([(cid, norm(t)) for cid, ch in chapters.items() for t in ch.get("teaches", [])])
    pages: dict[str, tuple[Path | None, list[dict], str]] = {}
    bare_names = collections.Counter(norm(e["variants"][0]) for e in entries if e["variants"])

    def page(cid):
        if cid not in pages:
            path = doc_of(cid, chapters, root)
            pages[cid] = (path, sections_of(path) if path else [])
        return pages[cid]
    # the headings and the phrases in bold of every page: which pages may define a notion (a word index)
    digests = Statements({cid: "\n".join(s["head"] + "\n" + "\n".join(s["bolds"]) for s in page(cid)[1])
                          for cid in syllabus.ORDER if page(cid)[0]})

    titled = [(cid, HEADING_LEAD.sub("", norm(chapters[cid]["title"]))) for cid in syllabus.ORDER
              if cid in chapters and not syllabus.is_cp(cid)]
    out_dir = (root / OUTPUT).parent
    lines_by_letter = collections.defaultdict(list)
    links: dict[str, str] = {}              # link target -> label of its reference-style link (a smaller file)

    def ref(target: str) -> str:
        return links.setdefault(target, f"s{len(links) + 1}")

    def cite(cid: str, path: Path, sec: dict | None) -> str:
        label = checkpoint_label(cid)
        link = Path(_relpath(path, out_dir)).as_posix()
        if sec is None:
            return f"[{label}][{ref(link)}]"
        title = re.sub(r"\s*[⏩🕰️⚠️🔗📐💼🧮]+\s*$", "", sec["heading"]).strip()
        title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)
        if len(title) > 70:
            title = title[:67].rsplit(" ", 1)[0] + "…"
        title = title.replace("[", "(").replace("]", ")")
        return f"[{label}, {title}][{ref(link + '#' + sec['anchor'] if sec['anchor'] else link)}]"

    for e in entries:
        strong = strong_variants(e["search"])
        probes = [norm(v).strip() for v in strong if pattern(v)]
        homonym = bool(e["variants"]) and bare_names[norm(e["variants"][0])] > 1     # « noyau (d'un SVM) »
        # the chapters that teach it, and the strength of their section
        found = []                                   # (strength, study rank, cid, path, section)
        for cid in e["taught"]:
            path, secs = page(cid)
            sec, strength = find_section(secs, e["search"]) if path else (None, 0)
            found.append((strength, _rank(cid), cid, path, sec))
        deeper = list(dict.fromkeys(cid for cid in teaches.find(probes) if cid not in e["taught"]))
        for cid in list(deeper):                     # a longer notion of a chapter whose heading is about it
            path, secs = page(cid)
            sec, strength = find_section(secs, e["search"]) if path else (None, 0)
            if strength == 4 and not homonym:       # « biais (d'une pièce) » is not the heading « Biais : … »
                found.append((strength, _rank(cid), cid, path, sec))
                deeper.remove(cid)
        whole = []                                   # chapters whose title is about it: all their exercises
        if not homonym:                              # a chapter whose title is about it (« Modèles de diffusion »)
            for cid, title_core in titled:
                if any(_about(title_core, k) for k in probes):
                    whole.append(cid)
                    known = [f for f in found if f[2] == cid]
                    if known:
                        found = [(max(f[0], 4),) + f[1:] if f[2] == cid else f for f in found]
                    elif page(cid)[0] is not None:
                        sec, strength = find_section(page(cid)[1], e["search"])
                        sec = sec if strength >= 3 else {"heading": chapters[cid]["title"], "anchor": "", "ref": None}
                        found.append((4, _rank(cid), cid, page(cid)[0], sec))
                        deeper = [c for c in deeper if c != cid]
        if all(f[0] < 2 for f in found):             # defined nowhere it is listed: look in every page
            candidates = digests.find(probes)
            for cid in syllabus.ORDER:
                if cid not in candidates or cid in e["taught"] or cid in [f[2] for f in found]:
                    continue
                path, secs = page(cid)
                sec, strength = find_definition(secs, strong)
                if sec is not None:     # after the chapters of the entry: a bold word may be another sense
                    found.append((1, _rank(cid), cid, path, sec))
                    break
        found.sort(key=lambda f: (-f[0], f[1]))      # the main lesson first, then in study order
        best = max((f[0] for f in found), default=0)
        main = [f for f in found if f[0] >= 2 or best < 2 or f[2] in e["teaches"]]
        mentioned = sorted((f for f in found if f not in main), key=lambda f: f[1])
        taught = main[:MAX_TAUGHT]
        parts = [cite(cid, path, sec) if path else checkpoint_label(cid) for _, _, cid, path, sec in taught]
        aside = [cite(cid, path, sec) if path else checkpoint_label(cid) for _, _, cid, path, sec in mentioned]
        # the exercises that practise it
        shown_ids = {f[2] for f in taught}
        scope = {f[2] for f in found}
        said = statements.find(probes)
        if len(said) > MAX_SAID:        # a word too common to tell (« trace » is also a verb): the chapters only
            said = {i for i in said if chapter_of[i] in scope}
        named = [i for i in titles.find(probes) if not homonym or chapter_of[i] in scope]
        covering = [i for _, _, cid, _, sec in taught if sec is not None and sec["ref"] and cid in chapters
                    for i in covers.get((cid, sec["ref"]), [])]
        ids = named + [i for i in covering if i in said]
        ids += sorted((i for i in said if chapter_of[i] in shown_ids), key=_exercise_key)
        if not homonym:
            ids += sorted((i for i in said if chapter_of[i] not in shown_ids), key=_exercise_key)
        ids += [ex["id"] for cid in whole for ex in chapters[cid]["exercises"]]
        if not ids and e["teaches"]:    # a long notion of the contract, named by no statement: the section's
            ids = [i for (s, _, cid, _, sec) in taught if s >= 2 and sec is not None and sec["ref"]
                   for i in covers.get((cid, sec["ref"]), [])]          # exercises
        ids = list(dict.fromkeys(ids))
        reused = [cid for cid in requires.find(probes) if cid not in e["taught"]]
        if homonym:                     # « noyau » of another sense in their notions: only the entry's own chapters
            deeper, reused = [], []
        name = f"**{_escape(e['term'])}**"
        if e["other"]:
            name += " (*" + ", ".join(o.replace("*", "") for o in e["other"]) + "*)"
        fields = []
        if parts:
            fields.append("enseignée : " + " ; ".join(parts))
        if aside:
            fields.append("évoquée : " + " ; ".join(aside))
        if ids:
            shown = sorted(ids[:MAX_EXERCISES], key=_exercise_key)
            more = f" (+ {len(ids) - MAX_EXERCISES})" if len(ids) > MAX_EXERCISES else ""
            fields.append("exercices : " + syllabus.compress_ids(shown) + more)
        extra = [f[2] for f in main[MAX_TAUGHT:]] + deeper
        if extra:
            fields.append("approfondie : " + _chapter_list(extra))
        if reused:
            fields.append("réutilisée : " + _chapter_list(reused))
        lines_by_letter[_letter(e["term"])].append((fold(e["term"]), e["term"], f"- {name} — " + " · ".join(fields)))
    for (form, italic), terms in cross:
        targets = ", ".join(f"[{_escape(t)}](#{_letter(t).lower() if _letter(t) != '#' else 'symboles'})"
                            for t in terms[:MAX_GROUP])
        targets += ", …" if len(terms) > MAX_GROUP else ""
        shown = f"*{form.replace('*', '')}*" if italic else _escape(form)
        lines_by_letter[_letter(form)].append((fold(form), form, f"- {shown} → voir {targets}"))
    letters = sorted(lines_by_letter, key=lambda k: (k == "#", k))
    n_entries, n_cross = len(entries), len(cross)
    head = [
        "# Index des notions",
        "",
        "> **Généré** par `python tools/syllabus.py build` (`tools/build_index.py`) à partir du glossaire "
        "(`annexes/glossaire.md`), du contrat de chaque chapitre (`docs/syllabus/data/*.json` : notions enseignées, "
        "concepts, notions mobilisées, exercices), des fiches et des énoncés des exercices : on ne l'édite pas à la "
        "main.",
        "",
        f"**{n_entries} notions**, et **{n_cross} renvois** depuis l'autre langue, un synonyme ou un mot commun à "
        "plusieurs entrées, dans l'ordre alphabétique (accents ignorés). Pour chaque notion :",
        "",
        "- **enseignée** : les chapitres qui la présentent, la leçon principale d'abord, chacun avec un lien vers la "
        "section de sa fiche : un titre qui porte sur elle ou qui la nomme, sinon sa définition (en gras ou suivie "
        "de sa glose), sinon la section qui en parle le plus ; **évoquée** : un chapitre qui la mentionne "
        "seulement, souvent avant la leçon (le panorama du ch. 1, par exemple) ;",
        f"- **exercices** : ceux dont le titre la nomme, puis ceux dont l'énoncé la nomme, d'abord dans les "
        f"chapitres qui l'enseignent ({MAX_EXERCISES} au plus, le nombre des autres entre parenthèses) ;",
        "- **approfondie** : les autres chapitres dont une notion enseignée la contient ; **réutilisée** : les "
        "chapitres qui la mobilisent comme prérequis.",
        "",
        "Entre parenthèses et en italique, l'autre langue. Une ligne « *terme* → voir … » renvoie de l'autre langue "
        "(le plus souvent l'anglais) vers le terme retenu (BIBLE §5) ; une ligne « terme → voir … », d'un synonyme "
        "ou d'un mot commun vers les entrées qui le contiennent. "
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


def _rank(cid: str) -> int:
    return syllabus.ORDER.index(cid) if cid in syllabus.ORDER else 99


def _escape(text: str) -> str:
    """Asterisks of a term (« *args, **kwargs ») outside code: literal, not emphasis."""
    parts = text.split("`")
    return "`".join(p if i % 2 else p.replace("*", "\\*") for i, p in enumerate(parts))


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
    return ", ".join(checkpoint_label(c) if c[:2] in ("CP", "MP") or c == "PF" else c for c in seen).join(
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
