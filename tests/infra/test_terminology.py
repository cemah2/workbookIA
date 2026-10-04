"""BIBLE §5: the retained terms are used everywhere the learner reads (prose, titles, messages).

A French equivalent of a term kept in English may appear only
- as a gloss in italics right after an opening parenthesis: « **epoch** (*époque*) »;
- in the second column of the glossary tables (« En français », « Autre langue »);
- in an everyday sense, listed in ``FORBIDDEN`` as a phrase that contains it (« à l'époque »,
  « compression sans perte », « les étiquettes des axes » of a figure…).

« précision » is not checked: it is also the numerical precision of a float (ch. 5).
"""

import functools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SOURCES = ["chapitres/*/*.md", "chapitres/*/flashcards.csv", "checkpoints/*/*.md", "projets/*/README.md",
           "projets/*/*/README.md", "annexes/*.md", "annexes/cheatsheets/*.md", "README.md", "00_setup/*.md",
           "suivi/auto_evaluation.md", "suivi/journal.md", "suivi/remediation.md", "suivi/tableau_de_bord.md",
           "docs/METHODE.md", "docs/SYLLABUS.md", "docs/PARCOURS.md", "docs/syllabus/data/*.json",
           "tools/chapters/build_*.py", "tools/chapters/figures_*.py", "tools/chapters/chapter_kit.py",
           "data/cards/*.md", "src/wb/*.py", "tools/*.py", "templates/*.py", "projets/*/*/*.py"]

# retained term -> (pattern of the forbidden forms, phrases where the word keeps its everyday sense;
# a phrase is a regular expression when it starts with « re: »)
FORBIDDEN = {
    "epoch": (r"\b[ÉéE]poques?\b",
              ["re:de l'époque(?! \\d)", "à cette époque", "re:à l'époque(?! \\d)", "une autre époque", "même époque"]),
    "batch, mini-batch": (r"\b(?:[Mm]ini-)?[Ll]ots?\b",
                          ["au plus $n$, la taille du lot", "dans le lot complet", "un autre lot de", "même lot"]),
    "label": (r"\b[ÉéE]tiquettes?\b",
              ["re:ulti-étiquettes?", "étiquettes des axes", "étiquettes de l'axe", "étiquettes antivol",
               "étiquette de « malade »", "Pour les étiquettes : `ax.set_xticks",
               "les étiquettes : `[alphabet[i] for i in order]`", "oublier les étiquettes ou la légende",
               "le label (étiquette)", "label (étiquette, "]),
    "overfitting, underfitting": (r"\b[Ss]urapprentissage\b|\b[Ss]ous-apprentissage\b|\b[Ss]urajustement\b", []),
    "dataset": (r"\b[Jj]eux? de données\b", []),
    "loss": (r"\b[Pp]ertes?\b",
             ["sans perte", "avec perte", "perte de précision", "perte de chiffres", "perte de vitesse"]),
    "cross-entropy": (r"\b[Ee]ntropies? croisées?\b", []),
    "token": (r"\b[Jj]etons?\b", []),
    "batchnorm": (r"\b[Bb]atch [Nn]orm\b|\bnormalisation par (?:lots?|batch)\b", []),
    "learning rate": (r"\b[Tt]aux d'apprentissage\b", []),
}

GLOSS = re.compile(r"\(\*[^*\n]+\*")          # « (*époque*) », « (*taux d'apprentissage* : … »


@functools.cache
def _lines():
    """(path, line number, scannable line) for every line of the sources, read once."""
    out = []
    for pattern in SOURCES:
        for path in sorted(ROOT.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                out.append((path, number, _scannable(path, line)))
    return out


def _scannable(path, line):
    """The line, with the second column of a glossary table blanked out."""
    if path.name == "glossaire.md" and line.startswith("|"):
        cells = line.split("|")
        if len(cells) > 3:
            cells[2] = " " * len(cells[2])
            return "|".join(cells)
    return line


def _violations(line, pattern, allowed):
    spans = [m.span() for m in GLOSS.finditer(line)]
    for phrase in allowed:
        regex = phrase[3:] if phrase.startswith("re:") else re.escape(phrase)
        spans += [m.span() for m in re.finditer(regex, line)]
    for m in re.finditer(pattern, line):
        if not any(a <= m.start() and m.end() <= b for a, b in spans):
            yield m.group(0)


@pytest.mark.parametrize("term", sorted(FORBIDDEN))
def test_retained_terms_are_used_in_learner_facing_text(term):
    pattern, allowed = FORBIDDEN[term]
    rx = re.compile(pattern)
    problems = [f"{path.relative_to(ROOT)}:{number}: « {word} »"
                for path, number, line in _lines() if rx.search(line)
                for word in _violations(line, pattern, allowed)]
    assert not problems, (f"expected « {term} » (BIBLE §5), found {len(problems)} French form(s); a gloss goes in "
                          f"italics after « ( », an everyday sense goes in FORBIDDEN: " + " | ".join(problems[:20]))


def test_the_terminology_check_sees_a_forbidden_form_and_accepts_a_gloss():
    pattern, allowed = FORBIDDEN["epoch"]
    assert list(_violations("après chaque époque, on mélange", pattern, allowed)) == ["époque"]
    assert not list(_violations("une **epoch** (*époque*) ; à l'époque de Rosenblatt", pattern, allowed))
    assert list(_violations("elle remonte à l'époque 5", pattern, allowed)) == ["époque"]
    pattern, allowed = FORBIDDEN["batch, mini-batch"]
    assert list(_violations("des mini-lots de 64", pattern, allowed)) == ["mini-lots"]
