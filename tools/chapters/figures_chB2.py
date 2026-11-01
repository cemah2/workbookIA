#!/usr/bin/env python
"""Draw the figures of chapter B2 (used by Claude).

    python tools/chapters/figures_chB2.py              # draw the five figures (offline, a few seconds)
    python tools/chapters/figures_chB2.py --measure    # recompute the measured constants from the Hub (network)

Writes small PNG files into chapitres/b2_tokenisation/figures/:

- ``niveaux.png``: one French sentence cut into characters, words and the sub-words of two real tokenizers;
- ``fusions_bpe.png``: the six BPE merges learned on the toy corpus of section 202.3, step by step (computed here,
  with the tie rule of ``tokenizers.trainers.BpeTrainer``);
- ``fertilite.png``: tokens needed for the Universal Declaration of Human Rights in 15 languages, relative to
  English, for four tokenizers;
- ``skipgram.png``: the skip-gram with negative sampling as a diagram (one centre word, two positive contexts, three
  negatives drawn at random, two embedding tables);
- ``carte_embeddings.png``: a 2D PCA map of count-based embeddings (PPMI + truncated SVD) learned on Holmes in about
  two seconds, for five groups of words.

The sub-words and the token counts were read with transformers 5.21 on 2026-11-01 and are stored below as
constants, so that the figures redraw offline; ``--measure`` reads them again and prints any difference. None of the
figures uses the words, corpora or numbers of an exercise: the BPE corpus differs from those of 02_exercices.md, the
UDHR is not Holmes or Verne, and the map leaves out « Holmes », « Watson » and « letter » (📦 B2.24).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "b2_tokenisation" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED

# --- measured constants (transformers 5.21, tokenizers 0.23.3, 2026-11-01) -------------------------------------
SENTENCE = "Les chatons s'endorment paisiblement."
SUBWORDS = {
    "GPT-2 (BPE octet, 50 257)": ["Les", "Ġchat", "ons", "Ġs", "'", "endor", "ment", "Ġpa", "isible", "ment", "."],
    "CamemBERT (Unigram, 32 006)": ["▁Les", "▁chaton", "s", "▁s", "'", "end", "or", "ment", "▁paisible", "ment", "."],
}
TOKENIZERS = {"GPT-2": "openai-community/gpt2", "gpt-oss (o200k)": "openai/gpt-oss-20b",
              "Qwen3": "Qwen/Qwen3-0.6B", "XLM-R (e5)": "intfloat/multilingual-e5-small"}
SUBWORD_REPOS = {"GPT-2 (BPE octet, 50 257)": "openai-community/gpt2",
                 "CamemBERT (Unigram, 32 006)": "almanach/camembert-base"}
# UDHR in XML (github.com/unicode-org/udhr, files udhr_<code>.xml): every <para>, joined by newlines.
LANGS = {"eng": "anglais", "fra": "français", "deu_1996": "allemand", "spa": "espagnol", "fin": "finnois",
         "tur": "turc", "rus": "russe", "ell_monotonic": "grec", "arb": "arabe", "kor": "coréen",
         "jpn": "japonais", "cmn_hans": "chinois", "hin": "hindi", "tha": "thaï", "tam": "tamoul", "mya": "birman"}
UDHR_TOKENS = {  # tokens without special tokens: GPT-2, gpt-oss (o200k), Qwen3, XLM-R (e5)
    "eng": (1935, 1888, 1887, 2135), "fra": (3900, 2502, 2956, 2745), "deu_1996": (4408, 2420, 3102, 2393),
    "spa": (3839, 2321, 2754, 2416), "fin": (5330, 3448, 4517, 2758), "tur": (4860, 2824, 3262, 2367),
    "rus": (12551, 2653, 3296, 2470), "ell_monotonic": (13697, 4126, 10268, 3106), "arb": (7369, 2245, 2615, 2131),
    "kor": (9675, 2583, 2747, 2276), "jpn": (6317, 3424, 2761, 2338), "cmn_hans": (5732, 2192, 1708, 1867),
    "hin": (17123, 3133, 10168, 2764), "tha": (17815, 3779, 4958, 2367), "tam": (37077, 4554, 14538, 3452),
    "mya": (43027, 8061, 23363, 4742),
}

# --- the toy corpus of section 202.3 ------------------------------------------------------------------------------
BPE_CORPUS = {"chat": 5, "chats": 2, "chaton": 3, "rat": 4, "raton": 2}


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma, a thin space for thousands and a true minus sign."""
    return f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def arrow(ax, start, end, color=INK, lw=1.2, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def chips(ax, y, pieces, colors, x0=0.0, h=0.62, fontsize=10, char_w=0.105, pad=0.16, gap=0.07):
    """Draw a row of rounded boxes, one per piece; returns the right end."""
    x = x0
    for i, piece in enumerate(pieces):
        w = max(len(piece), 1) * char_w + pad
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=colors[i % len(colors)], ec=INK, lw=0.7))
        ax.text(x + w / 2, y, piece.replace(" ", "␣"), ha="center", va="center", fontsize=fontsize,
                family="DejaVu Sans Mono", color=INK)
        x += w + gap
    return x


# --- 1. three levels ----------------------------------------------------------------------------------------------
def figure_levels() -> None:
    chars = list(SENTENCE)
    words = re.findall(r"\w+|[^\w\s]+", SENTENCE)          # the 'whitespace' pre-tokenizer of the chapter
    rows = [(f"caractères ({len(chars)})", chars, 0.075, 0.07, 8.5),
            (f"mots et ponctuation ({len(words)})", words, 0.105, 0.16, 10)]
    rows += [(f"{name}\n{len(toks)} tokens", toks, 0.105, 0.16, 10) for name, toks in SUBWORDS.items()]
    light = ["#cde2fb", "#fde0cf", "#d3f0e4", "#fbe8b5"]
    fig, ax = plt.subplots(figsize=(11.2, 3.9))
    for k, (label, pieces, cw, pad, fs) in enumerate(rows):
        y = -k * 1.05
        ax.text(-0.25, y, label, ha="right", va="center", fontsize=9.5, color=INK)
        chips(ax, y, pieces, light, char_w=cw, pad=pad, fontsize=fs, gap=0.05 if k == 0 else 0.07)
    ax.set_xlim(-3.1, 7.9)
    ax.set_ylim(-3.6, 0.55)
    ax.axis("off")
    ax.set_title(f"« {SENTENCE} » à trois niveaux (Ġ et ▁ marquent l'espace qui précède le token)", fontsize=10.5)
    save(fig, "niveaux.png")


# --- 2. BPE merges step by step -------------------------------------------------------------------------------------
def learn_bpe(words: dict[str, int], n_merges: int):
    """Plain BPE at char level, ties broken like BpeTrainer: the smallest pair of ids (alphabet sorted by code point)."""
    vocab = sorted({ch for w in words for ch in w})
    index = {s: i for i, s in enumerate(vocab)}
    seqs = {tuple(index[ch] for ch in w): f for w, f in words.items()}
    steps = [(None, 0, [], {w: list(w) for w in words})]
    for _ in range(n_merges):
        counts = Counter()
        for seq, f in seqs.items():
            for pair in zip(seq, seq[1:]):
                counts[pair] += f
        best_count = max(counts.values())
        best = min(p for p, c in counts.items() if c == best_count)
        ties = sorted(p for p, c in counts.items() if c == best_count and p != best)
        new_id = len(vocab)
        vocab.append(vocab[best[0]] + vocab[best[1]])
        new = {}
        for seq, f in seqs.items():
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new[tuple(out)] = f
        seqs = new
        segm = {"".join(vocab[t] for t in s): [vocab[t] for t in s] for s in seqs}
        steps.append(((vocab[best[0]], vocab[best[1]]), best_count,
                      [(vocab[a], vocab[b]) for a, b in ties], segm))
    return vocab, steps


def figure_merges() -> None:
    vocab, steps = learn_bpe(BPE_CORPUS, 6)
    words = list(BPE_CORPUS)
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    col_x = [0.0] + [2.05 * (j + 1) for j in range(len(words))]
    for j, w in enumerate(words):
        ax.text(col_x[j + 1] + 0.75, 0.75, f"{w} × {BPE_CORPUS[w]}", ha="center", fontsize=10, weight="bold")
    for k, (pair, count, ties, segm) in enumerate(steps):
        y = -k * 0.86
        if pair is None:
            label = "départ : caractères\nalphabet a c h n o r s t (ids 0 à 7)"
        else:
            label = f"fusion {k} : {pair[0]} + {pair[1]} → {pair[0] + pair[1]}\n(compte {count})"
            if ties:
                label += "   égalité avec " + ", ".join(f"{a} + {b}" for a, b in ties)
        ax.text(col_x[0] + 1.75, y, label, ha="right", va="center", fontsize=8.6,
                color=C[7] if ties else INK)
        new_token = None if pair is None else pair[0] + pair[1]
        for j, w in enumerate(words):
            colors = ["#fbe8b5" if piece == new_token else "#e8eef8" for piece in segm[w]]
            chips(ax, y, segm[w], colors, x0=col_x[j + 1] + 0.05, h=0.5, fontsize=9.5, char_w=0.14, pad=0.1,
                  gap=0.05)
    ax.set_xlim(-2.2, col_x[-1] + 2.0)
    ax.set_ylim(-len(steps) * 0.86 + 0.3, 1.1)
    ax.axis("off")
    ax.set_title("BPE pas à pas : chaque fusion remplace la paire la plus fréquente (comptes pondérés par les "
                 "fréquences des mots)\nà égalité, BpeTrainer prend la plus petite paire d'identifiants "
                 "(ici c + h = (1, 2) avant h + at = (2, 8)) ; en jaune, le token créé à chaque étape", fontsize=9.8)
    save(fig, "fusions_bpe.png")


# --- 3. fertility by language ---------------------------------------------------------------------------------------
def figure_fertility() -> None:
    codes = [c for c in LANGS if c != "eng"]
    eng = np.array(UDHR_TOKENS["eng"], float)
    ratios = np.array([np.array(UDHR_TOKENS[c]) / eng for c in codes])
    order = np.argsort(ratios[:, 0])
    codes = [codes[i] for i in order]
    ratios = ratios[order]
    names = list(TOKENIZERS)
    fig, ax = plt.subplots(figsize=(9.6, 6.8))
    y = np.arange(len(codes))
    h = 0.2
    for k, name in enumerate(names):
        ax.barh(y + (k - 1.5) * h, ratios[:, k], height=h, color=C[k], label=name)
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ticks = [0.8, 1, 1.5, 2, 3, 5, 10, 20]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: french_number(v) + " ×"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.75, 26)
    ax.set_yticks(y, [LANGS[c] for c in codes])
    ax.invert_yaxis()
    ax.set_xlabel("tokens nécessaires, rapportés à ceux du texte anglais (échelle log)")
    ax.set_title("Déclaration universelle des droits de l'homme (préambule et 30 articles) : le même texte "
                 "dans 16 langues", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False, title="tokenizer", title_fontsize=9)
    ax.grid(axis="x", color=wbplot.GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fertilite.png")


# --- 4. skip-gram with negative sampling ---------------------------------------------------------------------------
def figure_skipgram() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    sentence = ["le", "vieux", "phare", "éclaire", "la", "côte"]
    xs = np.arange(len(sentence)) * 1.45
    centre = 3
    for i, (x, w) in enumerate(zip(xs, sentence)):
        fc = "#fbe8b5" if i == centre else ("#d3f0e4" if abs(i - centre) <= 2 and i != centre else "#f0efec")
        ax.add_patch(FancyBboxPatch((x - 0.55, 3.6), 1.1, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=INK, lw=0.8))
        ax.text(x, 3.875, w, ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(xs[1] - 0.6, 3.45), xytext=(xs[5] + 0.6, 3.45),
                arrowprops=dict(arrowstyle="|-|", color=MUTED, lw=1, mutation_scale=4))
    ax.text(xs[3], 3.2, "fenêtre de 2 : contextes positifs vieux, phare, la, côte (on en montre deux)",
            ha="center", fontsize=8.8, color=MUTED)
    # the two tables
    ax.add_patch(FancyBboxPatch((-0.6, 0.2), 2.3, 2.3, boxstyle="round,pad=0.02", fc="#cde2fb", ec=INK, lw=0.8))
    ax.text(0.55, 2.2, "table des mots $\\mathbf{E}$\nforme $(V, d)$, lignes $\\mathbf{v}_w$", ha="center", va="top", fontsize=9)
    ax.text(0.55, 1.05, "$\\mathbf{v}_{\\text{éclaire}}$", ha="center", fontsize=12)
    ax.add_patch(FancyBboxPatch((7.4, 0.2), 2.3, 2.3, boxstyle="round,pad=0.02", fc="#fde0cf", ec=INK, lw=0.8))
    ax.text(8.55, 2.2, "table des contextes $\\mathbf{E}'$\nforme $(V, d)$, lignes $\\mathbf{u}_c$", ha="center", va="top", fontsize=9)
    rows = [("phare", "+", C[5]), ("côte", "+", C[5]), ("fromage", "−", C[7]), ("lundi", "−", C[7]),
            ("parce", "−", C[7])]
    for k, (w, sign, col) in enumerate(rows):
        yy = 1.65 - k * 0.32
        ax.text(8.55, yy, f"$\\mathbf{{u}}_{{\\text{{{w}}}}}$", ha="center", va="center", fontsize=10, color=col)
        ax.text(5.3, yy, (f"$\\sigma(\\mathbf{{u}}_{{\\text{{{w}}}}}^\\top\\mathbf{{v}}) \\to 1$" if sign == "+"
                          else f"$\\sigma(\\mathbf{{u}}_{{\\text{{{w}}}}}^\\top\\mathbf{{v}}) \\to 0$"),
                ha="center", va="center", fontsize=9.5, color=col)
        arrow(ax, (1.8, 1.15), (4.2, yy), color=col, lw=0.9)
        arrow(ax, (7.35, yy), (6.45, yy), color=col, lw=0.9)
    arrow(ax, (xs[3], 3.55), (0.9, 2.55), color=INK, lw=1.1, rad=0.15)
    ax.text(10.2, 1.65, "contextes vrais :\nlabel 1", fontsize=8.8, color=C[5], va="center")
    ax.text(10.2, 0.6, "K = 3 négatifs\ntirés au hasard\n(fréquence$^{0,75}$) :\nlabel 0", fontsize=8.8,
            color=C[7], va="center")
    ax.set_xlim(-1.0, 12.3)
    ax.set_ylim(0.0, 4.4)
    ax.axis("off")
    ax.set_title("Skip-gram avec échantillonnage négatif : une régression logistique sur des produits scalaires "
                 "d'embeddings", fontsize=10)
    save(fig, "skipgram.png")


# --- 5. a 2D map of count-based embeddings --------------------------------------------------------------------------
MAP_GROUPS = {
    "nombres": ["two", "three", "four", "five", "six", "twenty", "hundred"],
    "famille": ["father", "mother", "sister", "brother", "husband", "daughter", "son", "uncle"],
    "corps": ["hand", "hands", "eyes", "head", "lips", "shoulder", "fingers"],
    "moments": ["morning", "evening", "night", "day", "week", "year", "minutes", "month"],
    "couleurs": ["red", "black", "white", "blue", "grey", "yellow", "green", "golden"],
}


# labels moved away from a neighbouring point or label (offset in points, horizontal alignment)
MAP_LABEL_OFFSETS = {"white": (4, 6, "left"), "grey": (4, -3, "left"), "yellow": (4, -11, "left"),
                     "fingers": (4, -11, "left"), "evening": (-4, 3, "right"), "minutes": (3, -11, "left")}


def ppmi_svd_embeddings(text: str, n_words: int = 4000, window: int = 2, dim: int = 100, alpha: float = 0.75):
    """PPMI (context smoothing alpha) of a symmetric window, then a truncated SVD weighted by sqrt of S."""
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD

    words = re.findall(r"[a-z]+", text.lower())
    vocab = [w for w, _ in Counter(words).most_common(n_words)]
    index = {w: i for i, w in enumerate(vocab)}
    ids = np.array([index.get(w, -1) for w in words])
    rows, cols = [], []
    for d in range(1, window + 1):
        a, b = ids[:-d], ids[d:]
        ok = (a >= 0) & (b >= 0)                  # words outside the vocabulary are skipped, not merged
        rows += [a[ok], b[ok]]
        cols += [b[ok], a[ok]]
    r, c = np.concatenate(rows), np.concatenate(cols)
    counts = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n_words, n_words)).tocsr().tocoo()
    total = counts.sum()
    p_w = np.asarray(counts.tocsr().sum(axis=1)).ravel() / total
    c_alpha = np.asarray(counts.tocsr().sum(axis=0)).ravel() ** alpha
    p_c = c_alpha / c_alpha.sum()
    pmi = np.log((counts.data / total) / (p_w[counts.row] * p_c[counts.col]))
    keep = pmi > 0
    M = sp.csr_matrix((pmi[keep], (counts.row[keep], counts.col[keep])), shape=(n_words, n_words))
    svd = TruncatedSVD(dim, algorithm="arpack", random_state=0)
    US = svd.fit_transform(M)                      # U S
    E = US / np.sqrt(svd.singular_values_)          # U S^(1/2): the « p = 0.5 » of Levy, Goldberg & Dagan (2015)
    return E / np.linalg.norm(E, axis=1, keepdims=True), index


def figure_map() -> None:
    from sklearn.decomposition import PCA
    from wb import datasets

    E, index = ppmi_svd_embeddings(datasets.load_holmes())
    words = [w for ws in MAP_GROUPS.values() for w in ws]
    P = PCA(2, svd_solver="full").fit_transform(E[[index[w] for w in words]])
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    k = 0
    for g, (name, ws) in enumerate(MAP_GROUPS.items()):
        pts = P[k:k + len(ws)]
        ax.scatter(pts[:, 0], pts[:, 1], s=28, color=C[g], label=name, zorder=3)
        for w, (x, y) in zip(ws, pts):
            dx, dy, ha = MAP_LABEL_OFFSETS.get(w, (3, 3, "left"))
            ax.annotate(w, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=C[g], ha=ha)
        k += len(ws)
    ax.margins(x=0.1, y=0.06)                       # room for the labels of the points on the edges
    ax.set_xlabel("composante principale 1")
    ax.set_ylabel("composante principale 2")
    ax.set_title("Holmes : embeddings PPMI + SVD (4 000 mots, fenêtre de 2, d = 100), projetés par PCA",
                 fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="best")
    ax.grid(color=wbplot.GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "carte_embeddings.png")


# --- measurement (network) ------------------------------------------------------------------------------------------
def measure() -> int:
    """Re-read the sub-words and the UDHR token counts (needs the Hub and a UDHR clone in data/downloads/udhr)."""
    import subprocess
    import xml.etree.ElementTree as ET

    from transformers import AutoTokenizer

    problems = 0
    for name, repo in SUBWORD_REPOS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        got = tok.convert_ids_to_tokens(tok.encode(SENTENCE, add_special_tokens=False))
        if got != SUBWORDS[name]:
            problems += 1
            print(f"❌ {name}: {got}")
    udhr = ROOT / "data" / "downloads" / "udhr"
    if not udhr.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/unicode-org/udhr", str(udhr)],
                       check=True)
    toks = [AutoTokenizer.from_pretrained(repo) for repo in TOKENIZERS.values()]
    for code in LANGS:
        root = ET.parse(udhr / "data" / "udhr" / f"udhr_{code}.xml").getroot()
        ns = root.tag.split("}")[0].strip("{")
        text = "\n".join(t for t in ("".join(p.itertext()).strip() for p in root.iter(f"{{{ns}}}para")) if t)
        got = tuple(len(t.encode(text, add_special_tokens=False)) for t in toks)
        if got != UDHR_TOKENS[code]:
            problems += 1
        print(("✅" if got == UDHR_TOKENS[code] else "❌"), code, got)
    _, steps = learn_bpe(BPE_CORPUS, 6)
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([" ".join(w for w, f in BPE_CORPUS.items() for _ in range(f))],
                            trainers.BpeTrainer(vocab_size=14, min_frequency=2, show_progress=False))
    import json

    ref = [tuple(m) if isinstance(m, list) else tuple(m.split()) for m in json.loads(tok.to_str())["model"]["merges"]]
    mine = [s[0] for s in steps[1:]]
    print(("✅" if ref == mine else "❌"), "BPE merges equal to BpeTrainer:", mine)
    return int(problems > 0 or ref != mine)


def main() -> int:
    if "--measure" in sys.argv:
        return measure()
    figure_levels()
    figure_merges()
    figure_fertility()
    figure_skipgram()
    figure_map()
    print(f"✅ figures written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
