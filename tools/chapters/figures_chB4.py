#!/usr/bin/env python
"""Draw the figures of chapter B4 (used by Claude).

    python tools/chapters/figures_chB4.py        # draw the six figures (offline, a few seconds)

Writes small PNG files into chapitres/b4_llm/figures/:

- ``pipeline_rag.png``: the two phases of a RAG system, indexing once (chunks, BM25 index, embeddings) and
  answering each question (retrieve, fuse, build the prompt with numbered passages, generate, check citations);
- ``decoupage.png``: chunk_text on the 13-word sentence of section 204.6, chunk_size = 5, overlap = 2;
- ``bm25.png``: the BM25 contribution of one term against its frequency in the document (saturation, several k1,
  next to a raw count), and against the document length (several b);
- ``rrf.png``: the reciprocal rank fusion example of section 204.6 (two rankings, k = 60), fused scores computed here;
- ``lora.png``: a LoRA layer, the frozen W next to the trainable B and A, and the parameter count r (d + k);
- ``memoire.png``: memory of the weights by dtype (4, 2, 1 and about 0.5 byte per parameter) for four models whose
  parameter counts were read from their config.json (meta device, transformers 5.21), against 8 and 16 GB.

Nothing here is trained, downloaded or timed. None of the figures uses the numbers of an exercise of
02_exercices.md (SmolLM2-135M and the 896 x 896 layer are left to B4.1 and B4.3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "b4_llm" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED
BLUE, ORANGE, GREEN = "#cde2fb", "#fde0cf", "#d5efd2"

# --- data shared with 01_fiche.md --------------------------------------------------------------------------------
SENTENCE = "Le docteur Watson prend son chapeau et suit Holmes dans la rue sombre".split()
CHUNK, OVERLAP = 5, 2
BM25_RANKING, DENSE_RANKING = [7, 2, 9, 4], [2, 5, 7, 1]
# parameters read with AutoModelForCausalLM.from_config on the meta device (transformers 5.21, 2 Nov 2026)
MODELS = [("Qwen3-0.6B", 596_049_920), ("SmolLM2-1.7B", 1_711_376_384), ("Llama 3.1 8B", 8_030_261_248),
          ("Llama 3.1 70B", 70_553_706_496)]


def french(x: float, digits: int = 2) -> str:
    """A number with a French decimal comma and a true minus sign."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fontsize=9.3, color=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)


def arrow(ax, start, end, color=INK, lw=1.0, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


# --- 1. the RAG pipeline -----------------------------------------------------------------------------------------
def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    # indexing, once
    ax.add_patch(Rectangle((-0.2, 2.35), 12.4, 1.75, fc="#f6f6f4", ec="none"))
    ax.text(0.0, 3.92, "indexation : une fois", fontsize=9.5, color=MUTED, ha="left", va="center")
    box(ax, 1.0, 3.1, 1.8, 0.75, "documents\n(Holmes,\n12 nouvelles)", fc=ORANGE, fontsize=8.6)
    box(ax, 3.4, 3.1, 1.9, 0.75, "découpage\nchunks de 200 mots,\nrecouvrement 50", fc="#ffffff", fontsize=8.6)
    box(ax, 6.2, 3.55, 2.2, 0.55, "index BM25 (mots)", fc=BLUE)
    box(ax, 6.2, 2.7, 2.2, 0.55, "embeddings normalisés", fc=BLUE)
    arrow(ax, (1.85, 3.1), (2.45, 3.1))
    arrow(ax, (4.35, 3.2), (5.1, 3.5))
    arrow(ax, (4.35, 3.0), (5.1, 2.75))
    ax.text(8.6, 3.12, "on garde aussi le texte\net l'identifiant de chaque chunk", fontsize=8.6, color=MUTED,
            ha="left", va="center")
    # answering, each question
    ax.text(0.0, 1.95, "réponse : à chaque question", fontsize=9.5, color=MUTED, ha="left", va="center")
    box(ax, 1.0, 1.05, 1.7, 0.75, "question", fc=ORANGE)
    box(ax, 3.4, 1.05, 1.9, 0.75, "récupérer\nBM25 et dense,\nfusion RRF, top-k", fontsize=8.6)
    box(ax, 6.2, 1.05, 2.3, 0.75, "augmenter le prompt :\npassages [1]…[k]\n+ consignes", fontsize=8.6)
    box(ax, 8.85, 1.05, 1.6, 0.75, "générer\n(LLM)", fc=GREEN)
    box(ax, 11.0, 1.05, 1.8, 0.75, "réponse + [n]\ncitations vérifiées", fontsize=8.6)
    for a, b in [((1.85, 1.05), (2.45, 1.05)), ((4.35, 1.05), (5.1, 1.05)), ((7.3, 1.05), (8.05, 1.05)),
                 ((9.65, 1.05), (10.1, 1.05))]:
        arrow(ax, a, b)
    arrow(ax, (5.3, 2.42), (3.75, 1.43), color=C[0], lw=1.2, rad=0.15)
    ax.text(5.6, 1.95, "lit les deux index", fontsize=8.4, color=C[0], ha="left")
    ax.text(3.4, 0.35, "évaluer ici : recall@k, MRR", fontsize=8.6, color=C[1], ha="center")
    ax.text(11.0, 0.35, "évaluer ici : exactitude,\ncitations, « je ne sais pas »", fontsize=8.6, color=C[1],
            ha="center")
    ax.set_xlim(-0.3, 12.3)
    ax.set_ylim(0.0, 4.15)
    ax.axis("off")
    save(fig, "pipeline_rag.png")


# --- 2. chunking with overlap ------------------------------------------------------------------------------------
def figure_chunks() -> None:
    n, step = len(SENTENCE), CHUNK - OVERLAP
    starts = []
    for s in range(0, n, step):
        starts.append(s)
        if s + CHUNK >= n:
            break
    fig, ax = plt.subplots(figsize=(12.5, 3.3))
    w = 0.92
    for i, word in enumerate(SENTENCE):
        ax.add_patch(Rectangle((i, 0), w, 0.6, fc="#ffffff", ec=MUTED, lw=0.7))
        ax.text(i + w / 2, 0.3, word, ha="center", va="center", fontsize=8.6)
        ax.text(i + w / 2, -0.25, str(i), ha="center", va="center", fontsize=7.5, color=MUTED)
    for c, s in enumerate(starts):
        e = min(s + CHUNK, n)
        y = 0.95 + 0.5 * c
        ax.add_patch(FancyBboxPatch((s + 0.02, y), e - s - 0.12, 0.36, boxstyle="round,pad=0.01,rounding_size=0.08",
                                    fc=C[c % len(C)], ec="none", alpha=0.85))
        ax.text(s + 0.1, y + 0.18, f"chunk {c} : mots {s} à {e - 1}", ha="left", va="center", fontsize=8.6,
                color="white")
        if c:
            ov = (s, min(s + OVERLAP, n))
            ax.add_patch(Rectangle((ov[0], 0.62), ov[1] - ov[0] - 0.08, 0.1, fc=C[(c - 1) % len(C)], ec="none"))
    ax.text(n + 0.2, 0.3, f"{n} mots", fontsize=8.6, color=MUTED, va="center")
    ax.text(n + 0.2, 0.68, "barres : mots partagés", fontsize=8.6, color=MUTED, va="center")
    ax.text(n + 0.2, 1.4, f"taille {CHUNK}, recouvrement {OVERLAP},\npas = {CHUNK} − {OVERLAP} = {step}",
            fontsize=8.8, color=INK, va="center")
    ax.set_xlim(-0.2, n + 3.6)
    ax.set_ylim(-0.45, 0.95 + 0.5 * len(starts))
    ax.axis("off")
    save(fig, "decoupage.png")


# --- 3. BM25: saturation and length normalisation ----------------------------------------------------------------
def figure_bm25() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    f = np.arange(0, 21)
    ax = axes[0]
    for c, k1 in enumerate([0.5, 1.2, 1.5, 3.0]):
        ax.plot(f, f * (k1 + 1) / (f + k1), marker="o", ms=3, color=C[c], label=f"$k_1$ = {french(k1, 1)}")
        ax.axhline(k1 + 1, color=C[c], lw=0.7, ls=":")
    ax.plot(f[:6], f[:6], color=MUTED, ls="--", lw=1.0, label="compte brut $f$")
    ax.set_xlabel("$f_{t,d}$ : occurrences du terme dans le document")
    ax.set_ylabel("facteur de $\\operatorname{idf}(t)$")
    ax.set_title("Saturation ($b = 0$) : plafond $k_1 + 1$ en pointillés", fontsize=10.5)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(color=wbplot.GRID, lw=0.6)
    ax = axes[1]
    ratio = np.linspace(0.2, 3.0, 200)
    k1 = 1.5
    for c, b in enumerate([0.0, 0.5, 0.75, 1.0]):
        ax.plot(ratio, 1 * (k1 + 1) / (1 + k1 * (1 - b + b * ratio)), color=C[c], label=f"$b$ = {french(b, 2)}")
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.text(1.03, 1.8, "longueur\nmoyenne", fontsize=8.4, color=MUTED)
    ax.set_xlabel("$|d| \\,/\\, \\overline{|d|}$ : longueur relative du document")
    ax.set_ylabel("facteur de $\\operatorname{idf}(t)$ pour $f_{t,d} = 1$")
    ax.set_title("Normalisation de la longueur ($k_1 = 1{,}5$)", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(color=wbplot.GRID, lw=0.6)
    fig.tight_layout()
    save(fig, "bm25.png")


# --- 4. reciprocal rank fusion -----------------------------------------------------------------------------------
def figure_rrf() -> None:
    k = 60
    scores: dict[int, float] = {}
    for ranking in (BM25_RANKING, DENSE_RANKING):
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank)
    fused = sorted(scores, key=lambda d: (-scores[d], d))
    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    colors = {d: C[i % len(C)] for i, d in enumerate(sorted(scores))}
    for col, (title, ranking) in enumerate([("BM25", BM25_RANKING), ("dense", DENSE_RANKING)]):
        x = 1.2 + 2.3 * col
        ax.text(x, 4.6, title, ha="center", fontsize=10, color=INK)
        for rank, doc in enumerate(ranking, 1):
            y = 4.6 - 0.75 * rank
            box(ax, x, y, 1.3, 0.5, f"doc {doc}", fc=colors[doc], color="white")
            ax.text(x - 0.95, y, f"rang {rank}", ha="right", va="center", fontsize=8.3, color=MUTED)
    ax.text(8.6, 4.6, "score RRF ($k$ = 60), puis classement fusionné", ha="center", fontsize=10, color=INK)
    for pos, doc in enumerate(fused):
        y = 4.15 - 0.6 * pos
        parts = [f"1/{k + r + 1}" for ranking in (BM25_RANKING, DENSE_RANKING)
                 for r, d in enumerate(ranking) if d == doc]
        box(ax, 6.4, y, 1.2, 0.45, f"doc {doc}", fc=colors[doc], color="white")
        ax.text(7.2, y, " + ".join(parts) + f" = {french(scores[doc], 4)}", ha="left", va="center", fontsize=8.8)
    ax.text(8.6, 0.25, "à égalité (docs 1 et 4), le plus petit identifiant d'abord", ha="center", fontsize=8.4,
            color=MUTED)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0.0, 4.9)
    ax.axis("off")
    save(fig, "rrf.png")


# --- 5. LoRA -----------------------------------------------------------------------------------------------------
def figure_lora() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.add_patch(Rectangle((0.6, 1.2), 2.6, 2.6, fc="#e9e8e4", ec=INK, lw=1.0))
    ax.text(1.9, 2.5, "$\\mathbf{W}$\n$d \\times k$\ngelée\n(pas de gradient)", ha="center", va="center", fontsize=10)
    ax.add_patch(Rectangle((4.4, 1.2), 0.45, 2.6, fc=ORANGE, ec=INK, lw=1.0))
    ax.text(4.62, 0.95, "$\\mathbf{B}$ : $d \\times r$\n(nulle au départ)", ha="center", va="top", fontsize=9)
    ax.add_patch(Rectangle((5.2, 3.35), 2.6, 0.45, fc=BLUE, ec=INK, lw=1.0))
    ax.text(6.5, 4.0, "$\\mathbf{A}$ : $r \\times k$ (tirage aléatoire)", ha="center", va="bottom", fontsize=9)
    ax.text(3.8, 2.5, "+", fontsize=18, ha="center", va="center")
    ax.text(5.6, 2.3, "$\\times\\ \\frac{\\alpha}{r}$", fontsize=13, ha="left", va="center")
    ax.text(4.0, 0.15, "$\\mathbf{W}' = \\mathbf{W} + \\frac{\\alpha}{r}\\,\\mathbf{B}\\mathbf{A}$ : "
            "entraînés : $r(d + k)$ paramètres au lieu de $dk$", ha="center", fontsize=10.5)
    # the forward pass
    box(ax, 9.6, 0.9, 1.4, 0.5, "entrée $\\mathbf{x}$ ($k$)", fc="#ffffff", fontsize=9)
    box(ax, 8.9, 2.5, 1.2, 0.5, "$\\mathbf{W}\\mathbf{x}$", fc="#e9e8e4", fontsize=9)
    box(ax, 10.3, 2.0, 1.2, 0.5, "$\\mathbf{A}\\mathbf{x}$ ($r$)", fc=BLUE, fontsize=9)
    box(ax, 10.3, 2.9, 1.2, 0.5, "$\\frac{\\alpha}{r}\\mathbf{B}(\\mathbf{A}\\mathbf{x})$", fc=ORANGE, fontsize=9)
    box(ax, 9.6, 4.0, 1.6, 0.5, "$\\mathbf{h}$ ($d$) : somme", fc=GREEN, fontsize=9)
    arrow(ax, (9.4, 1.15), (8.95, 2.25))
    arrow(ax, (9.8, 1.15), (10.25, 1.75))
    arrow(ax, (10.3, 2.25), (10.3, 2.65))
    arrow(ax, (8.95, 2.75), (9.4, 3.75))
    arrow(ax, (10.25, 3.15), (9.8, 3.75))
    ax.text(9.6, 4.5, "passe avant d'une couche LoRA", ha="center", fontsize=9.5, color=MUTED)
    ax.set_xlim(0.3, 11.2)
    ax.set_ylim(-0.1, 4.8)
    ax.axis("off")
    save(fig, "lora.png")


# --- 6. memory of the weights by dtype ---------------------------------------------------------------------------
def figure_memory() -> None:
    dtypes = [("float32", 4.0), ("bfloat16", 2.0), ("int8", 1.0), ("int4 (≈ 0,5)", 0.5)]
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    x = np.arange(len(MODELS))
    width = 0.2
    for c, (name, nbytes) in enumerate(dtypes):
        gb = [n * nbytes / 1e9 for _, n in MODELS]
        bars = ax.bar(x + (c - 1.5) * width, gb, width, color=C[c], label=name)
        for bar, v in zip(bars, gb):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.08, french(v, 1 if v < 10 else 0), ha="center",
                    va="bottom", fontsize=7.2, rotation=90)
    for ram in (8, 16):
        ax.axhline(ram, color=MUTED, lw=0.9, ls="--")
        ax.text(len(MODELS) - 0.45, ram * 1.06, f"{ram} Go", fontsize=8.5, color=MUTED, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 2000)
    ax.set_xticks(x, [f"{m}\n{french(n / 1e9, 2)} milliards" for m, n in MODELS], fontsize=9)
    ax.set_ylabel("mémoire des poids (Go, échelle log)")
    ax.set_title("Mémoire des poids seuls $\\approx N \\times$ octets par paramètre (sans KV cache ni activations)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper left")
    ax.grid(axis="y", color=wbplot.GRID, lw=0.6)
    save(fig, "memoire.png")


def main() -> None:
    figure_pipeline()
    figure_chunks()
    figure_bm25()
    figure_rrf()
    figure_lora()
    figure_memory()
    print("figures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
