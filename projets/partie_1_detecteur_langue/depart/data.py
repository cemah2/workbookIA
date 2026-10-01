"""Data of the language detector: the chapters of Holmes (English) and Verne (French), and excerpts.

Provided with the mini-project: you do not need to change this file. It cuts the
two novels of the workbook into chapters, so that the chapters used for training
are never used for testing, and it draws excerpts of a given length from a list of
chapters, reproducibly (same generator, same excerpts).

    chapters = load_chapters()                  # {"en": 12 stories, "fr": 37 chapters}
    rng = np.random.default_rng(0)
    excerpts = sample_excerpts(chapters["fr"][31:], length=20, n=5, rng=rng)
"""

from __future__ import annotations

import re

import numpy as np

import wb

LANGUAGES = ("en", "fr")


def normalize(text: str) -> str:
    """Replace every run of whitespace (line breaks included) by one space, and strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


def holmes_stories(text: str | None = None) -> list[str]:
    """The 12 stories of *The Adventures of Sherlock Holmes*, in order (normalized text).

    A story starts after its title line, such as ``IX. THE ADVENTURE OF THE ENGINEER’S THUMB``,
    and ends where the next one starts.
    """
    text = wb.datasets.load_holmes() if text is None else text
    heads = list(re.finditer(r"^([IVX]+)\. ([A-Z][^a-z\n]+)$", text, re.M))
    ends = [m.start() for m in heads[1:]] + [len(text)]
    return [normalize(text[m.end():end]) for m, end in zip(heads, ends)]


def verne_chapters(text: str | None = None) -> list[str]:
    """The 37 chapters of *Le Tour du monde en quatre-vingts jours*, in order (normalized text).

    A chapter starts after its number, alone on its line (``XXXVII``), and ends where the next
    one starts; its title, in capitals, is the beginning of the chapter.
    """
    text = wb.datasets.load_verne() if text is None else text
    heads = list(re.finditer(r"^([IVXL]+)$", text, re.M))
    ends = [m.start() for m in heads[1:]] + [len(text)]
    return [normalize(text[m.end():end]) for m, end in zip(heads, ends)]


def load_chapters() -> dict[str, list[str]]:
    """The chapters of both books: ``{"en": [12 stories], "fr": [37 chapters]}``."""
    return {"en": holmes_stories(), "fr": verne_chapters()}


def sample_excerpts(chapters: list[str], length: int, n: int, rng: np.random.Generator) -> list[str]:
    """Draw ``n`` excerpts of exactly ``length`` characters from a list of chapters.

    For each excerpt, in order: a chapter is chosen with a probability proportional to its
    number of possible start positions (``len(chapter) - length + 1``), then a start position
    uniformly in it. Excerpts may overlap; they never cross from one chapter to another.

    Parameters
    ----------
    chapters : list of str
        The chapters to draw from (for example the test chapters of one language).
    length : int
        Number of characters of each excerpt, ``>= 1``.
    n : int
        Number of excerpts, ``>= 0``.
    rng : np.random.Generator
        The generator: the same generator state gives the same excerpts.

    Returns
    -------
    list of str
        ``n`` excerpts, each of exactly ``length`` characters.
    """
    if length < 1 or n < 0:
        raise ValueError(f"length must be >= 1 and n >= 0, got length={length}, n={n}")
    starts = np.array([len(c) - length + 1 for c in chapters], dtype=float)
    if not len(chapters) or starts.max() < 1:
        raise ValueError(f"no chapter has {length} characters")
    weights = np.clip(starts, 0, None) / np.clip(starts, 0, None).sum()
    excerpts = []
    for _ in range(n):
        k = int(rng.choice(len(chapters), p=weights))
        start = int(rng.integers(0, int(starts[k])))
        excerpts.append(chapters[k][start:start + length])
    return excerpts
