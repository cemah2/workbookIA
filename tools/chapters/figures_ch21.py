#!/usr/bin/env python
"""Draw the figures of chapter 21 (used by Claude).

    python tools/chapters/figures_ch21.py

Writes small PNG files into chapitres/ch21_cnn/figures/. Every figure is computed or drawn here (never
copied from the book): three hand-made filters on a synthetic image, the "fly's-eye view" (im2col) of a
4 x 4 image, the positions of a 3 x 3 window with or without padding and stride, a multi-channel
convolution layer, the growth of the receptive field along a stack of layers, and the uneven overlaps of
a transposed convolution (the cause of checkerboard artifacts). None of them uses the numbers of an
exercise: the images, filters and sizes differ from those of 02_exercices.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch21_cnn" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
GREY = "#bdbdbd"
INK = wbplot.TEXT


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


def correlate_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Cross-correlation without padding (what nn.Conv2d computes), with plain loops."""
    kh, kw = kernel.shape
    out = np.zeros((image.shape[0] - kh + 1, image.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


# ------------------------------------------------------------------ 1. three hand-made filters
def fig_filtres() -> None:
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    image = np.zeros((n, n))
    image[10:30, 8:28] = 1.0                                          # a square
    image[((yy - 44) ** 2 + (xx - 44) ** 2) <= 12 ** 2] = 0.7         # a disk
    image[np.abs((yy - 50) + (xx - 14) * -1) <= 2] = 0.45             # a diagonal bar
    image[(yy > 34) & (xx > 26)] *= 1.0
    image = image + np.random.default_rng(21).normal(0, 0.03, image.shape)
    filters = {
        "bord vertical (clair à gauche)": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float),
        "bord horizontal (clair en haut)": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float),
        "flou (moyenne 5 × 5)": np.ones((5, 5)) / 25,
    }
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("image (64 × 64, niveaux de gris)", fontsize=9.5)
    for ax, (title, kernel) in zip(axes[1:], filters.items()):
        out = correlate_valid(image, kernel)
        if kernel.min() < 0:
            lim = np.abs(out).max()
            ax.imshow(out, cmap=wbplot.diverging_cmap(), vmin=-lim, vmax=lim)
        else:
            ax.imshow(out, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\nsortie {out.shape[0]} × {out.shape[1]}", fontsize=9.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Le même petit filtre, promené sur toute l'image : chaque sortie dit où il trouve son motif "
                 "(rouge positif, bleu négatif)", fontsize=10)
    save(fig, "filtres.png")


# ------------------------------------------------------------------ 2. im2col: the fly's-eye view
def draw_grid(ax, values, x0, y0, size=1.0, highlight=None, color=C[0], fmt="{:g}"):
    rows, cols = values.shape
    for i in range(rows):
        for j in range(cols):
            face = "white"
            if highlight is not None and highlight[i, j]:
                face = color
            ax.add_patch(Rectangle((x0 + j * size, y0 - (i + 1) * size), size, size, facecolor=face,
                                   edgecolor=INK, lw=0.8, alpha=0.9 if face != "white" else 1))
            ax.text(x0 + (j + 0.5) * size, y0 - (i + 0.5) * size, fmt.format(values[i, j]), ha="center",
                    va="center", fontsize=8.5, color=INK)


def fig_im2col() -> None:
    image = np.arange(1, 17, dtype=float).reshape(4, 4)
    k = 3
    patches = [(i, j) for i in range(2) for j in range(2)]
    columns = np.stack([image[i:i + k, j:j + k].ravel() for i, j in patches], axis=1)   # (9, 4)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    draw_grid(ax, image, 0, 4)
    ax.text(2, 4.35, "image 4 × 4", ha="center", fontsize=9.5)
    for p, (i, j) in enumerate(patches):
        mask = np.zeros((4, 4), bool)
        mask[i:i + k, j:j + k] = True
        draw_grid(ax, image, 5.2 + (p % 2) * 4.6, 4.3 - (p // 2) * 4.6, size=0.85, highlight=mask,
                  color=C[p % len(C)])
        ax.text(5.2 + (p % 2) * 4.6 + 1.7, 4.3 - (p // 2) * 4.6 + 0.25, f"morceau {p}, position ({i}, {j})",
                ha="center", fontsize=8.5)
    x0 = 15.2
    for p in range(4):
        col = columns[:, p].reshape(9, 1)
        hl = np.ones((9, 1), bool)
        draw_grid(ax, col, x0 + p * 0.75, 4.5, size=0.75, highlight=hl, color=C[p % len(C)])
    ax.text(x0 + 1.5, 4.85, "les 4 morceaux en colonnes\n(forme 9 × 4)", ha="center", fontsize=9)
    ax.text(x0 + 1.5, -3.2, "un filtre 3 × 3 aplati (1 × 9) × colonnes\n= les 4 sorties d'un coup", ha="center",
            va="top", fontsize=9)
    ax.set_xlim(-0.5, 19.6)
    ax.set_ylim(-4.6, 5.3)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "im2col.png")


# ------------------------------------------------------------------ 3. padding and stride
def fig_padding_stride() -> None:
    n, k = 7, 3
    cases = [("sans padding, stride 1", 0, 1), ("padding 1, stride 1", 1, 1), ("padding 1, stride 2", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (title, p, s) in zip(axes, cases):
        size = n + 2 * p
        out = (n + 2 * p - k) // s + 1
        for i in range(size):
            for j in range(size):
                inside = p <= i < p + n and p <= j < p + n
                ax.add_patch(Rectangle((j, -i - 1), 1, 1, facecolor="white" if inside else "#eeeeee",
                                       edgecolor=INK if inside else GREY, lw=0.6))
                if not inside:
                    ax.text(j + 0.5, -i - 0.5, "0", ha="center", va="center", fontsize=7, color=GREY)
        for a in range(out):
            for b in range(out):
                ci, cj = a * s + k // 2, b * s + k // 2         # the focal pixel under the anchor
                ax.add_patch(Rectangle((cj + 0.2, -ci - 0.8), 0.6, 0.6, facecolor=C[1], edgecolor="none"))
        ax.add_patch(Rectangle((0, -k), k, k, facecolor="none", edgecolor=C[0], lw=2.2))
        ax.set_title(f"{title}\nentrée {n} × {n} → sortie {out} × {out}", fontsize=9.5)
        ax.set_xlim(-0.3, size + 0.3)
        ax.set_ylim(-size - 0.3, 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Un filtre 3 × 3 (cadre bleu, à sa première position) : les carrés orange marquent les pixels "
                 "focaux où il se pose ; le gris, le zero-padding", fontsize=10)
    save(fig, "padding_stride.png")


# ------------------------------------------------------------------ 4. a convolution layer with several channels
def slab(ax, x, y, w, h, depth, color, label=None, n_lines=None):
    dx, dy = 0.35, 0.25
    n_lines = depth if n_lines is None else n_lines
    for t in reversed(range(n_lines)):
        ax.add_patch(Rectangle((x + t * dx, y + t * dy), w, h, facecolor=color, edgecolor=INK, lw=0.7,
                               alpha=0.85))
    if label:
        ax.text(x + w / 2 + n_lines * dx / 2, y - 0.45, label, ha="center", va="top", fontsize=9)


def fig_canaux() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    slab(ax, 0, 0, 2.6, 2.6, 5, "#dfe9f5", "entrée $(C_{in} = 5,\\ H,\\ W)$")
    for f in range(3):
        slab(ax, 5.2, 2.4 - f * 1.35, 0.8, 0.8, 5, C[f % len(C)])
    ax.text(6.2, -1.9, "3 filtres, chacun\n$(C_{in} = 5,\\ k,\\ k)$", ha="center", fontsize=9)
    for f in range(3):
        ax.add_patch(Rectangle((10.2 + f * 0.35, 0.0 + f * 0.25), 2.6, 2.6, facecolor=C[f % len(C)],
                               edgecolor=INK, lw=0.7, alpha=0.85))
    ax.text(11.8, -0.45, "sortie $(C_{out} = 3,\\ H',\\ W')$ :\nune carte par filtre", ha="center", va="top",
            fontsize=9)
    for f in range(3):
        ax.add_patch(FancyArrowPatch((7.3, 2.8 - f * 1.35), (10.0 + f * 0.35, 1.5 + f * 0.25),
                                     arrowstyle="-|>", mutation_scale=12, color=C[f % len(C)], lw=1.4))
    ax.add_patch(FancyArrowPatch((4.6, 1.6), (5.1, 1.6), arrowstyle="-|>", mutation_scale=12, color=INK))
    ax.set_title("Chaque filtre fait la somme sur TOUS les canaux d'entrée : "
                 "$\\#\\mathrm{param} = C_{out}\\,(C_{in}\\,k^2 + 1) = 3 \\times (5k^2 + 1)$", fontsize=10)
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-2.6, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "canaux.png")


# ------------------------------------------------------------------ 5. receptive field growth
def fig_champ_recepteur() -> None:
    layers = [("entrée", None, None), ("conv 5, stride 1", 5, 1), ("pool 2, stride 2", 2, 2),
              ("conv 3, stride 1", 3, 1)]
    width = 16
    fig, ax = plt.subplots(figsize=(11, 3.9))
    # walk backwards: the cells of each level seen by the output cell
    spans = []
    lo, hi = 1, 1                       # one output cell, index in the last level
    sizes = [width]
    for _, k, s in layers[1:]:
        sizes.append((sizes[-1] - k) // s + 1)
    for level in range(len(layers) - 1, 0, -1):
        spans.append((level, lo, hi))
        _, k, s = layers[level]
        lo, hi = lo * s, hi * s + k - 1
    spans.append((0, lo, hi))
    seen = {level: (a, b) for level, a, b in spans}
    for level, (name, k, s) in enumerate(layers):
        n = sizes[level]
        y = level * 1.2
        offset = (width - n) / 2
        a, b = seen[level]
        for i in range(n):
            on = a <= i <= b
            ax.add_patch(Rectangle((offset + i, y), 0.9, 0.9, facecolor=C[1] if on else "white", edgecolor=INK,
                                   lw=0.6))
        ax.text(-0.6, y + 0.45, name, ha="right", va="center", fontsize=9)
        ax.text(width + 0.4, y + 0.45, f"{b - a + 1} case(s)", ha="left", va="center", fontsize=9)
    ax.set_xlim(-5.2, width + 3.2)
    ax.set_ylim(-0.4, len(layers) * 1.2 + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("En orange, ce que « voit » une seule case de la dernière couche, niveau par niveau (une dimension)",
                 fontsize=10)
    save(fig, "champ_recepteur.png")


# ------------------------------------------------------------------ 6. transposed convolution: uneven overlaps
def overlaps(n_in: int, k: int, s: int) -> np.ndarray:
    size = (n_in - 1) * s + k
    count = np.zeros((size, size), int)
    for i in range(n_in):
        for j in range(n_in):
            count[i * s:i * s + k, j * s:j * s + k] += 1
    return count


def fig_transposee() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, k in zip(axes, (3, 4)):
        count = overlaps(5, k, 2)
        ax.imshow(count, cmap=wbplot.sequential_cmap(), vmin=0, vmax=4)
        for i in range(count.shape[0]):
            for j in range(count.shape[1]):
                ax.text(j, i, str(count[i, j]), ha="center", va="center", fontsize=7, color=INK)
        ax.set_title(f"noyau {k} × {k}, stride 2 : sortie {count.shape[0]} × {count.shape[1]}\n"
                     f"{'chevauchements inégaux : un damier' if k % 2 else 'chevauchements réguliers à l intérieur'}"
                     .replace("l intérieur", "l'intérieur"), fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Convolution transposée d'une entrée 5 × 5 : combien de pixels d'entrée contribuent à chaque "
                 "case de sortie", fontsize=10, y=1.04)
    save(fig, "transposee.png")


def main() -> int:
    wbplot.set_style()
    fig_filtres()
    fig_im2col()
    fig_padding_stride()
    fig_canaux()
    fig_champ_recepteur()
    fig_transposee()
    for path in sorted(OUT.glob("*.png")):
        print(f"✅ {path.relative_to(ROOT)} ({path.stat().st_size // 1024} ko)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
