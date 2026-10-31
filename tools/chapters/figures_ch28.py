#!/usr/bin/env python
"""Draw the figures of chapter 28 (used by Claude).

    python tools/chapters/figures_ch28.py

Writes small PNG files into chapitres/ch28_creatif/figures/. Every figure is computed or drawn here (never copied
from the book): gradient ascent on the input through a frozen network, the Gram matrix of a few activation maps and
its invariance when the positions are shuffled, the three losses of style transfer, the total variation of a clean,
a noisy and a smoothed photo (a free photo shipped with scikit-learn), and the next-word distribution under
temperature, top-k and top-p. None of them uses the numbers of an exercise: the maps, the photo crop and the
vocabulary differ from those of 02_exercices.md and of the notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch28_creatif" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma, a thin space for thousands and a true minus sign."""
    return f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")


def fr(value: float, digits: int) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def arrow(ax, start, end, color=INK, lw=1.4, rad=0.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def box(ax, xy, w, h, text, color, text_color="white", fontsize=11):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=color, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def figure_ascent() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.6, 3.6)
    ax.axis("off")
    box(ax, (0.2, 1.2), 1.9, 1.1, "image $\\mathbf{x}$\n(seule variable)", C[2], fontsize=10)
    box(ax, (3.2, 1.0), 2.8, 1.5, "réseau pré-entraîné\ngelé (poids fixes)", MUTED, fontsize=10)
    box(ax, (7.1, 1.2), 2.0, 1.1, "activation\n$a(\\mathbf{x})$ du filtre", C[0], fontsize=10)
    arrow(ax, (2.15, 1.75), (3.15, 1.75))
    arrow(ax, (6.05, 1.75), (7.05, 1.75))
    ax.text(4.6, 3.05, "passe avant", ha="center", fontsize=9.5, color=INK)
    arrow(ax, (8.1, 1.15), (1.15, 1.15), color=C[1], rad=-0.35)
    ax.text(4.6, -0.5, "rétropropagation jusqu'aux pixels : $\\nabla_{\\mathbf{x}}\\, a$", ha="center", fontsize=9.5,
            color=C[1])
    ax.text(9.3, 2.75, "$\\mathbf{x} \\leftarrow \\mathbf{x} + \\eta\\, \\nabla_{\\mathbf{x}}\\, a(\\mathbf{x})$",
            ha="center", fontsize=11, color=INK)
    save(fig, "montee_gradient.png")


def _maps(rng, size=6):
    """Three non-negative activation maps: a vertical-stripe detector, a blob detector, and one that fires with
    the first (so that the Gram matrix has a large off-diagonal entry)."""
    yy, xx = np.mgrid[0:size, 0:size]
    stripes = np.maximum(0, np.cos(np.pi * xx / 1.5)) * 1.2
    blob = np.exp(-((xx - 3.6) ** 2 + (yy - 2.2) ** 2) / 3.0) * 1.5
    partner = 0.8 * stripes + 0.15 * rng.random((size, size))
    return np.stack([stripes, blob, partner])


def figure_gram() -> None:
    rng = np.random.default_rng(28)
    A = _maps(rng)
    C_, H, W = A.shape
    perm = rng.permutation(H * W)
    B = A.reshape(C_, -1)[:, perm].reshape(C_, H, W)
    G = A.reshape(C_, -1) @ A.reshape(C_, -1).T / (C_ * H * W)
    G2 = B.reshape(C_, -1) @ B.reshape(C_, -1).T / (C_ * H * W)
    assert np.allclose(G, G2)
    fig = plt.figure(figsize=(9.4, 4.2))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.25, 1.7], wspace=0.25, hspace=0.35)
    vmax = A.max()
    for row, (maps, label) in enumerate(((A, "cartes d'activation"), (B, "mêmes cartes, positions mélangées"))):
        for c in range(C_):
            ax = fig.add_subplot(grid[row, c])
            ax.imshow(maps[c], cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"filtre {c + 1}", fontsize=10, color=INK)
        fig.text(0.37, 0.93 if row == 0 else 0.47, label, ha="center", fontsize=10, color=MUTED)
    ax = fig.add_subplot(grid[:, 4])
    ax.imshow(G, cmap="Blues", vmin=0)
    for i in range(C_):
        for j in range(C_):
            ax.text(j, i, fr(G[i, j], 2), ha="center", va="center", fontsize=10,
                    color="white" if G[i, j] > 0.6 * G.max() else INK)
    ticks = [f"{k + 1}" for k in range(C_)]
    ax.set_xticks(range(C_), ticks)
    ax.set_yticks(range(C_), ticks)
    ax.set_title("matrice de Gram $G$\n(la même pour les deux lignes)", fontsize=10, color=INK)
    save(fig, "gram.png")


def figure_transfer() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(-0.4, 5.4)
    ax.axis("off")
    rows = {"c": 4.3, "s": 2.55, "x": 0.8}
    box(ax, (0.1, rows["c"]), 2.1, 0.9, "contenu $\\mathbf{c}$\n(une photo)", C[3], fontsize=10)
    box(ax, (0.1, rows["s"]), 2.1, 0.9, "style $\\mathbf{s}$\n(un tableau)", C[4], fontsize=10)
    box(ax, (0.1, rows["x"]), 2.1, 0.9, "image générée $\\mathbf{x}$\n(seule variable)", C[2], fontsize=10)
    outputs = {"c": ("$F(\\mathbf{c})$ : activations\nd'une couche profonde", C[3]),
               "s": ("$G(\\mathbf{s})$ : matrices de\nGram de plusieurs couches", C[4]),
               "x": ("$F(\\mathbf{x})$ et $G(\\mathbf{x})$,\nmêmes couches", C[2])}
    for key, y in rows.items():
        arrow(ax, (2.25, y + 0.45), (2.95, y + 0.45))
        box(ax, (3.0, y), 2.2, 0.9, "réseau gelé", MUTED, fontsize=10)
        arrow(ax, (5.25, y + 0.45), (5.95, y + 0.45))
        box(ax, (6.0, y - 0.05), 3.2, 1.0, outputs[key][0], outputs[key][1], fontsize=9)
    ax.text(4.1, 5.3, "le même réseau, aux mêmes poids, pour les trois images", ha="center", fontsize=9,
            color=MUTED)
    ax.text(10.95, 4.55, "$L_c$ : écart entre\n$F(\\mathbf{x})$ et $F(\\mathbf{c})$", ha="center", fontsize=9.5,
            color=INK)
    ax.text(10.95, 2.8, "$L_s$ : écart entre\n$G(\\mathbf{x})$ et $G(\\mathbf{s})$", ha="center", fontsize=9.5,
            color=INK)
    ax.text(10.95, 1.25, "$L = \\alpha L_c + \\beta L_s + \\lambda L_{TV}$", ha="center", fontsize=11, color=INK)
    ax.text(10.95, 0.8, "($L_{TV}$ : lissage de $\\mathbf{x}$)", ha="center", fontsize=9, color=MUTED)
    arrow(ax, (10.95, 0.6), (1.15, 0.75), color=C[1], rad=-0.12)
    ax.text(5.9, -0.3, "rétropropagation jusqu'aux pixels de $\\mathbf{x}$, qui sont seuls mis à jour", ha="center",
            fontsize=9, color=C[1])
    save(fig, "transfert_style.png")


def _tv(img: np.ndarray) -> float:
    return float(((img[1:] - img[:-1]) ** 2).sum() + ((img[:, 1:] - img[:, :-1]) ** 2).sum())


def figure_tv() -> None:
    from scipy.ndimage import gaussian_filter
    from sklearn.datasets import load_sample_image

    photo = load_sample_image("flower.jpg").astype(float).mean(axis=2) / 255.0
    crop = photo[150:278, 200:328]
    rng = np.random.default_rng(280)
    noisy = np.clip(crop + rng.normal(0, 0.08, crop.shape), 0, 1)
    smooth = gaussian_filter(noisy, 1.2)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    for ax, img, title in zip(axes, (crop, noisy, smooth), ("photo", "photo + bruit", "photo bruitée, floutée")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\n$L_{{TV}}$ = {fr(_tv(img), 0)}", fontsize=10, color=INK)
    fig.text(0.5, -0.02, "photo flower.jpg fournie avec scikit-learn (Flickr, vultilion, licence CC BY 2.0), recadrée, en "
             "niveaux de gris, 128 × 128 pixels dans [0, 1]", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "variation_totale.png")


def figure_decoding() -> None:
    words = ["mer", "côte", "baleine", "nuit", "vague", "île", "tempête", "ciel", "brume", "sable"]
    logits = np.array([3.1, 2.6, 2.2, 1.6, 1.5, 1.1, 0.8, 0.5, 0.2, -0.3])
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), sharey=True)
    for ax, T in zip(axes, (0.5, 1.0, 2.0)):
        p = np.exp(logits / T - (logits / T).max())
        p /= p.sum()
        colors = [C[0]] * len(words)
        if T == 1.0:
            order = np.argsort(-p)
            keep = order[: np.searchsorted(np.cumsum(p[order]), 0.75) + 1]
            colors = [C[0] if i in keep else wbplot.GRID for i in range(len(words))]
            ax.set_title(f"$T$ = 1 ; top-p (p = 0,75) : {len(keep)} mots gardés", fontsize=10, color=INK)
        else:
            ax.set_title(f"$T$ = {fr(T, 1)}", fontsize=10, color=INK)
        ax.bar(range(len(words)), p, color=colors)
        ax.set_xticks(range(len(words)), words, rotation=60, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("probabilité du mot suivant", fontsize=9.5)
    fig.text(0.5, -0.12, "« … le navire disparut dans la ___ » : mêmes logits, trois températures ; en gris, les mots "
             "que le top-p écarte avant de tirer", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "decodage.png")


def main() -> int:
    figure_ascent()
    figure_gram()
    figure_transfer()
    figure_tv()
    figure_decoding()
    print(f"✅ figures written to {OUT.relative_to(ROOT)}: {', '.join(sorted(p.name for p in OUT.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
