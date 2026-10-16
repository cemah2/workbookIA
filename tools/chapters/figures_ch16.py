#!/usr/bin/env python
"""Draw the figures of chapter 16 (used by Claude).

    python tools/chapters/figures_ch16.py

Writes small PNG files into chapitres/ch16_feedforward/figures/. Every figure is computed or drawn
here (never copied from the book): a layered MLP and a general directed acyclic graph with its
evaluation order, the shapes of a forward pass on a mini-batch, the densities of a uniform and a
normal initialisation, and the spread of the activations through a deep LINEAR network for three
weight scales. None of them uses the numbers of an exercise: the exercises draw U(-0.05, 0.05) and
N(0, (0.05/3)**2) (the figure draws U(-0.2, 0.2) and N(0, 0.1**2)), use ReLU and tanh networks (the
figure has no activation), and other graphs (the figure's graph has nodes named i1, i2, h1...).
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
sys.path.insert(0, str(ROOT / "solutions"))

from mylearn_ref.nn import layers as ml  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch16_feedforward" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
GREY = "#9e9e9e"
INK = wbplot.TEXT


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma, a thin space for thousands and a true minus sign."""
    return f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")


def fr(value: float, decimals: int) -> str:
    """A number written the French way, for titles and legends."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:  # decimal commas on every numeric linear axis (not on named ticks)
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    path = OUT / name
    if path.stat().st_size > 115_000:   # many shades: a 256-colour palette is enough
        from PIL import Image

        with Image.open(path) as image:
            image.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"  {name} ({path.stat().st_size // 1000} ko)")


def arrow(ax, start, end, color=INK, lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw,
                                 shrinkA=11, shrinkB=11, connectionstyle=f"arc3,rad={rad}"))


def node(ax, xy, label, color, text_color="white"):
    ax.scatter(*xy, s=520, color=color, zorder=3, edgecolors="white", linewidths=1.5)
    ax.text(*xy, label, ha="center", va="center", fontsize=8, color=text_color, zorder=4, fontweight="bold")


# ---------------------------------------------------------------------------- graphs
def fig_graphs() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) a layered MLP 3-4-4-2: every neuron of a layer feeds every neuron of the next one
    sizes = [3, 4, 4, 2]
    names = ["entrée", "cachée 1", "cachée 2", "sortie"]
    colors = [C[0], C[1], C[1], C[2]]
    positions = []
    for layer, n in enumerate(sizes):
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
        positions.append([(layer * 2.0, y) for y in ys])
    for layer in range(len(sizes) - 1):
        for p in positions[layer]:
            for q in positions[layer + 1]:
                arrow(a, p, q, color=GREY, lw=0.7)
    for layer, pts in enumerate(positions):
        for k, p in enumerate(pts):
            node(a, p, "", colors[layer])
        a.text(layer * 2.0, -2.45, names[layer], ha="center", fontsize=9, color=INK)
        a.text(layer * 2.0, 2.25, f"{sizes[layer]}", ha="center", fontsize=9, color=INK)
    a.set_title(f"(a) MLP 3-4-4-2 : {ml.count_parameters(sizes)} paramètres", fontsize=10)
    a.set_xlim(-0.7, 6.7)
    a.set_ylim(-2.8, 2.6)
    a.axis("off")
    # (b) a general DAG: a skip connection and a node that feeds two layers; the numbers give the order
    graph = {"i1": ["h1", "h2"], "i2": ["h2"], "h1": ["h3", "s"], "h2": ["h3"], "h3": ["s"]}
    xy = {"i1": (0, 1.0), "i2": (0, -1.0), "h1": (1.6, 1.5), "h2": (1.6, -0.6), "h3": (3.2, 0.2), "s": (4.8, 0.6)}
    order = ml.topological_order(graph)
    for parent, kids in graph.items():
        for child in kids:
            arrow(b, xy[parent], xy[child], color=INK if (parent, child) != ("h1", "s") else C[3],
                  lw=1.2, rad=-0.25 if (parent, child) == ("h1", "s") else 0.0)
    kind = {"i1": C[0], "i2": C[0], "h1": C[1], "h2": C[1], "h3": C[1], "s": C[2]}
    for name, p in xy.items():
        node(b, p, name, kind[name])
        b.text(p[0], p[1] - 0.48, f"n° {order.index(name) + 1}", ha="center", fontsize=8, color=INK)
    b.text(3.2, 1.75, "connexion qui saute une couche", color=C[3], fontsize=8, ha="center")
    b.set_title("(b) un graphe orienté acyclique et son ordre d'évaluation", fontsize=10)
    b.set_xlim(-0.6, 5.4)
    b.set_ylim(-1.8, 2.2)
    b.axis("off")
    fig.tight_layout()
    save(fig, "graphes.png")


# ---------------------------------------------------------------------------- shapes of a forward pass
def block(ax, x, y, w, h, color, label, inside=""):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, alpha=0.35, edgecolor=color, lw=1.5))
    ax.text(x + w / 2, y + h + 0.15, label, ha="center", va="bottom", fontsize=10, color=INK)
    if inside:
        ax.text(x + w / 2, y + h / 2, inside, ha="center", va="center", fontsize=9, color=INK)


def fig_shapes() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    block(ax, 0.0, 0.0, 1.4, 3.0, C[0], r"$\mathbf{X}$ : $(n,\ n_{in})$", "un exemple\npar ligne")
    ax.text(1.75, 1.5, "@", fontsize=16, ha="center", va="center", color=INK)
    block(ax, 2.1, 0.8, 1.4, 1.4, C[1], r"$\mathbf{W}$ : $(n_{in},\ n_{out})$", "un neurone\npar colonne")
    ax.text(3.85, 1.5, "+", fontsize=16, ha="center", va="center", color=INK)
    block(ax, 4.2, 1.35, 1.4, 0.3, C[3], r"$\mathbf{b}$ : $(n_{out},)$")
    ax.annotate("", xy=(4.9, 0.1), xytext=(4.9, 1.3), arrowprops=dict(arrowstyle="->", color=C[3], lw=1.2))
    ax.text(5.1, 0.55, "recopié sur\nchaque ligne\n(broadcasting)", fontsize=8, color=C[3], va="center")
    ax.text(6.55, 1.5, "=", fontsize=16, ha="center", va="center", color=INK)
    block(ax, 7.0, 0.0, 1.4, 3.0, C[2], r"$\mathbf{Z}$ : $(n,\ n_{out})$", "une ligne\npar exemple")
    ax.set_xlim(-0.3, 8.8)
    ax.set_ylim(-0.4, 3.8)
    ax.axis("off")
    ax.set_title("La passe avant d'une couche dense sur un mini-batch de $n$ exemples", fontsize=10)
    fig.tight_layout()
    save(fig, "passe_avant.png")


# ---------------------------------------------------------------------------- uniform and normal laws
def fig_laws() -> None:
    rng = np.random.default_rng(1601)
    a, s = 0.2, 0.1
    u = ml.init_weights(200, 100, "uniform", rng, scale=a).ravel()
    g = ml.init_weights(200, 100, "normal", rng, scale=s).ravel()
    grid = np.linspace(-0.45, 0.45, 600)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, values, density, title in (
            (axes[0], u, np.where(np.abs(grid) <= a, 1 / (2 * a), 0.0), f"uniforme sur [−{fr(a, 1)} ; {fr(a, 1)}]"),
            (axes[1], g, np.exp(-grid ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi)),
             f"normale, écart-type {fr(s, 1)}")):
        ax.hist(values, bins=60, range=(-0.45, 0.45), density=True, color=C[0], alpha=0.45, label="20 000 tirages")
        ax.plot(grid, density, color=C[1], lw=2, label="densité théorique")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("valeur du poids")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("densité")
    fig.tight_layout()
    save(fig, "lois_initialisation.png")


# ---------------------------------------------------------------------------- variance through depth
def fig_depth() -> None:
    rng = np.random.default_rng(1602)
    width, depth = 200, 30
    x = rng.normal(size=(500, width))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for color, gain in zip((C[0], C[2], C[1]), (0.8, 1.0, 1.25)):
        params = [(rng.normal(0.0, np.sqrt(gain / width), size=(width, width)), np.zeros(width)) for _ in range(depth)]
        _, cache = ml.mlp_forward(x, params, return_cache=True)
        stds = [float(z.std()) for _, z in cache]
        layers_ = np.arange(1, depth + 1)
        ax.plot(layers_, stds, "o", color=color, ms=4, label=f"Var(w) = {fr(gain, 2)}/$n_{{in}}$ (mesuré)")
        ax.plot(layers_, gain ** (layers_ / 2), color=color, lw=1, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("couche")
    ax.set_ylabel("écart-type des sorties (échelle log)")
    ax.set_title("Un réseau linéaire de 30 couches de largeur 200 : l'écart-type multiplié\n"
                 "par $\\sqrt{n_{in}\\,\\mathrm{Var}(w)}$ à chaque couche (traits : la formule)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "variance_profondeur.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 16 -> {OUT.relative_to(ROOT)}")
    fig_graphs()
    fig_shapes()
    fig_laws()
    fig_depth()
    return 0


if __name__ == "__main__":
    sys.exit(main())
