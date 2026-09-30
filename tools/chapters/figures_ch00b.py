#!/usr/bin/env python
"""Draw the figures of the course sheet of chapter 0B (used by Claude).

    python tools/chapters/figures_ch00b.py

Writes small PNG files into chapitres/ch00b_maths/figures/. Every figure is
computed (never copied from the book) and uses the workbook style.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch00b_maths" / "figures"
C = wbplot.CATEGORICAL
DPI = 110


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma and a true minus sign: 0,5 and −2."""
    return f"{value:g}".replace(".", ",").replace("-", "\u2212")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:  # decimal commas on every linear axis (log axes keep their powers of 10)
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "linear":
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def axes_through_origin(ax) -> None:
    ax.axhline(0, color=wbplot.TEXT_MUTED, lw=0.8)
    ax.axvline(0, color=wbplot.TEXT_MUTED, lw=0.8)


def fig_functions() -> None:
    x = np.linspace(-4, 4, 400)
    xp = np.linspace(0.02, 4, 300)
    panels = [
        ("affine : 2x − 1", x, 2 * x - 1),
        ("parabole : x² − 2x − 3", x, x ** 2 - 2 * x - 3),
        ("exponentielle : eˣ", x, np.exp(x)),
        ("logarithme : ln x", xp, np.log(xp)),
        ("sigmoïde : σ(x)", x, 1 / (1 + np.exp(-x))),
        ("tanh(x)", x, np.tanh(x)),
        ("cos(x)", np.linspace(-2 * np.pi, 2 * np.pi, 400), np.cos(np.linspace(-2 * np.pi, 2 * np.pi, 400))),
        ("valeur absolue : |x|", x, np.abs(x)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, (title, xs, ys) in zip(axes.ravel(), panels):
        ax.plot(xs, ys, color=C[0], lw=2)
        axes_through_origin(ax)
        ax.set_title(title, fontsize=10.5)
        if "exponentielle" in title:
            ax.set_ylim(-1, 20)
        if "sigmoïde" in title:
            ax.axhline(1, color=C[1], lw=0.8, ls="--")
        if "tanh" in title:
            ax.axhline(1, color=C[1], lw=0.8, ls="--")
            ax.axhline(-1, color=C[1], lw=0.8, ls="--")
    fig.suptitle("Les fonctions usuelles du machine learning", fontsize=12)
    fig.tight_layout()
    save(fig, "fonctions_usuelles.png")


def fig_geometric() -> None:
    k = np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for q, color in zip([0.5, 0.9, 0.99, 1.05], C):
        ax.plot(k, q ** k, marker="o", ms=2.5, lw=1.4, color=color, label=f"q = {q}".replace(".", ","))
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("qᵏ (échelle log)")
    ax.set_title("Suites géométriques : qᵏ fond si |q| < 1, explose si |q| > 1")
    ax.legend()
    fig.tight_layout()
    save(fig, "suites_geometriques.png")


def fig_exp_log() -> None:
    x = np.linspace(-3, 3.5, 300)
    xp = np.linspace(0.05, 10, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(x, np.exp(x), color=C[0], lw=2, label="eˣ")
    ax.plot(xp, np.log(xp), color=C[1], lw=2, label="ln x")
    ax.plot(x, x, color=wbplot.TEXT_MUTED, lw=1, ls="--", label="y = x")
    axes_through_origin(ax)
    ax.set_xlim(-3, 6)
    ax.set_ylim(-3, 6)
    ax.set_aspect("equal")
    ax.set_title("eˣ et ln x : symétriques par rapport à y = x")
    ax.legend(loc="upper left")
    ax = axes[1]
    for base, fn, color in [("log₂ x", np.log2, C[0]), ("ln x", np.log, C[1]), ("log₁₀ x", np.log10, C[2])]:
        ax.plot(xp, fn(xp), lw=2, color=color, label=base)
    axes_through_origin(ax)
    ax.scatter([1, 2, np.e, 10], [0, 1, 1, 1], color=wbplot.TEXT, zorder=3, s=14)
    ax.set_title("Trois bases : même allure, toutes nulles en x = 1")
    ax.set_xlabel("x")
    ax.legend()
    fig.tight_layout()
    save(fig, "exp_log.png")


def fig_moving_average() -> None:
    rng = np.random.default_rng(7)
    t = np.arange(60)
    signal = 10 + 3 * np.sin(t / 8)
    x = signal + rng.normal(0, 1.2, size=t.size)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(t, x, color=wbplot.TEXT_MUTED, lw=1, marker="o", ms=2.5, label="série bruitée")
    for k, color in [(3, C[0]), (9, C[1])]:
        m = np.convolve(x, np.ones(k) / k, mode="valid")
        ax.plot(t[k - 1:], m, color=color, lw=2, label=f"moyenne mobile d'ordre {k}")
    ax.set_xlabel("t")
    ax.set_title("Une moyenne mobile lisse la série, d'autant plus que k est grand")
    ax.legend()
    fig.tight_layout()
    save(fig, "moyenne_mobile.png")


def fig_cosine() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=wbplot.TEXT_MUTED, lw=1)
    a = np.pi / 3
    ax.plot([0, np.cos(a)], [0, np.sin(a)], color=C[0], lw=2, label="θ = π/3")
    ax.plot([np.cos(a), np.cos(a)], [0, np.sin(a)], color=C[0], lw=1, ls="--")
    ax.plot([0, np.cos(a)], [0, 0], color=C[1], lw=3, label="cos θ = 0,5")
    arc = np.linspace(0, a, 40)
    ax.plot(0.22 * np.cos(arc), 0.22 * np.sin(arc), color=wbplot.TEXT, lw=1)
    ax.text(0.27, 0.1, "θ", fontsize=12)
    axes_through_origin(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Le cosinus : abscisse du point du cercle")
    ax.legend(loc="lower left")
    ax = axes[1]
    T = 100
    t = np.linspace(0, T, 200)
    lr = 0.5 * (1 + np.cos(np.pi * t / T))
    ax.plot(t, lr, color=C[0], lw=2)
    ax.scatter([0, T / 2, T], [1, 0.5, 0], color=C[1], zorder=3)
    ax.set_xlabel("étape t (sur T = 100)")
    ax.set_ylabel("facteur")
    ax.set_title("Planning en cosinus : ½ (1 + cos(πt / T))")
    fig.tight_layout()
    save(fig, "cosinus.png")


def label_beside(ax, start, vec, text, color, side=1.0, gap=0.3) -> None:
    """Write a vector's name beside the middle of its arrow (side = +1 left of it, -1 right of it)."""
    normal = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec) * gap * side
    mid = np.asarray(start, dtype=float) + 0.5 * np.asarray(vec, dtype=float)
    ax.text(mid[0] + normal[0], mid[1] + normal[1], text, color=color, fontsize=13, ha="center", va="center")


def fig_vectors() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    a, b = np.array([3.0, 1.0]), np.array([1.0, 2.0])
    for vec, color, start, dashed in [(a, C[0], (0, 0), False), (b, C[1], (0, 0), False),
                                      (b, C[1], tuple(a), True), (a + b, C[2], (0, 0), False)]:
        ax.annotate("", xy=(start[0] + vec[0], start[1] + vec[1]), xytext=start,
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, ls="--" if dashed else "-"))
    label_beside(ax, (0, 0), a, r"$\mathbf{a}$", C[0], side=-1)
    label_beside(ax, (0, 0), b, r"$\mathbf{b}$", C[1], side=1)
    label_beside(ax, a, b, r"$\mathbf{b}$", C[1], side=-1)
    label_beside(ax, (0, 0), a + b, r"$\mathbf{a} + \mathbf{b}$", C[2], side=1, gap=0.45)
    ax.set_xlim(-0.5, 4.8)
    ax.set_ylim(-0.5, 3.8)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Somme de deux vecteurs : les flèches bout à bout")
    fig.tight_layout()
    save(fig, "vecteurs.png")


def fig_dot_product() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    a, b = np.array([3.0, 1.0]), np.array([1.0, 2.0])
    for vec, color, text, side in [(a, C[0], r"$\mathbf{a}$", -1), (b, C[1], r"$\mathbf{b}$", 1)]:
        ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2))
        label_beside(ax, (0, 0), vec, text, color, side=side)
    ang_a, ang_b = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
    arc = np.linspace(ang_a, ang_b, 50)
    ax.plot(0.9 * np.cos(arc), 0.9 * np.sin(arc), color=wbplot.TEXT_MUTED)
    mid = (ang_a + ang_b) / 2
    ax.text(1.15 * np.cos(mid), 1.15 * np.sin(mid), "θ", fontsize=13, ha="center", va="center")
    ax.set_title(r"$\mathbf{a} \cdot \mathbf{b} = 5 = \|\mathbf{a}\|\,\|\mathbf{b}\| \cos\theta$, soit θ = 45°")
    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(-0.5, 2.8)
    ax.set_aspect("equal")
    ax.grid(True)
    fig.tight_layout()
    save(fig, "produit_scalaire.png")


def fig_matmul() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.axis("off")

    def block(x, y, w, h, label, color, sub=""):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.25, edgecolor=color, lw=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=14)
        if sub:
            ax.text(x + w / 2, y - 0.35, sub, ha="center", va="top", fontsize=11, color=wbplot.TEXT_MUTED)

    block(0, 0.5, 1.6, 2.4, "A", C[0], "(m, n) = (4, 3)")
    ax.text(1.95, 1.7, "×", fontsize=20, ha="center", va="center")
    block(2.7, 1.1, 2.4, 1.8, "B", C[1], "(n, p) = (3, 5)")
    ax.text(5.55, 1.7, "=", fontsize=20, ha="center", va="center")
    block(6.0, 0.5, 2.4, 2.4, "AB", C[2], "(m, p) = (4, 5)")
    ax.text(0.8, 3.0, "n = 3 colonnes", ha="center", fontsize=10, color=wbplot.BAD)
    ax.text(2.5, 2.0, "n = 3 lignes", rotation=90, ha="center", va="center", fontsize=10, color=wbplot.BAD)
    ax.text(4.3, 3.55, "colonnes de A = lignes de B", ha="center", fontsize=10.5, color=wbplot.BAD)
    ax.set_xlim(-0.3, 8.7)
    ax.set_ylim(-0.6, 3.9)
    ax.set_title("Produit matriciel : (m, n) × (n, p) → (m, p)")
    save(fig, "produit_matriciel.png")


def fig_tangent() -> None:
    f = lambda x: x ** 2  # noqa: E731
    a = 1.0
    x = np.linspace(-0.5, 3.2, 300)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(x, f(x), color=wbplot.TEXT, lw=2, label="f(x) = x²")
    for h, color in [(2.0, C[1]), (1.0, C[3]), (0.3, C[4])]:
        slope = (f(a + h) - f(a)) / h
        label = f"sécante, h = {h:g} : pente {slope:.1f}".replace(".", ",")
        ax.plot(x, f(a) + slope * (x - a), color=color, lw=1.2, ls="--", label=label)
        ax.scatter([a + h], [f(a + h)], color=color, zorder=3, s=18)
    ax.plot(x, f(a) + 2 * (x - a), color=C[0], lw=2, label="tangente en a = 1 : pente f′(1) = 2")
    ax.scatter([a], [f(a)], color=C[0], zorder=4)
    ax.set_ylim(-1, 9)
    ax.set_title("Quand h → 0, la sécante devient la tangente")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save(fig, "tangente.png")


def fig_extremum() -> None:
    x = np.linspace(-2.2, 3.2, 400)
    f = x ** 3 / 3 - x ** 2 / 2 - 2 * x + 1
    df = x ** 2 - x - 2
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    axes[0].plot(x, f, color=C[0], lw=2)
    axes[0].set_title("f(x) = x³/3 − x²/2 − 2x + 1")
    axes[1].plot(x, df, color=C[1], lw=2)
    axes[1].set_title("f′(x) = x² − x − 2 = (x + 1)(x − 2)")
    for ax in axes:
        ax.axhline(0, color=wbplot.TEXT_MUTED, lw=0.8)
        for x0 in (-1, 2):
            ax.axvline(x0, color=wbplot.TEXT_MUTED, lw=0.8, ls="--")
    axes[0].scatter([-1, 2], [(-1) ** 3 / 3 - 0.5 + 2 + 1, 8 / 3 - 2 - 4 + 1], color=C[3], zorder=3)
    axes[0].annotate("maximum local", xy=(-1, 2.17), xytext=(-0.55, 1.2), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=wbplot.TEXT_MUTED))
    axes[0].annotate("minimum local", xy=(2, -2.33), xytext=(0.6, -1.9), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=wbplot.TEXT_MUTED))
    axes[1].text(-2.15, -1.5, "f′ > 0 : f monte", fontsize=9)
    axes[1].text(-0.35, 1.2, "f′ < 0 : f descend", fontsize=9)
    axes[1].text(2.05, -1.5, "f′ > 0 : f monte", fontsize=9)
    axes[1].set_xlabel("x")
    fig.tight_layout()
    save(fig, "variations.png")


def fig_contour() -> None:
    from matplotlib.lines import Line2D

    f = lambda x, y: x ** 2 + 4 * y ** 2  # noqa: E731
    xs, ys = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-2, 2, 200))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    cs = ax.contour(xs, ys, f(xs, ys), levels=[4, 8, 12, 16], colors=wbplot.SEQUENTIAL[3:7])
    ax.clabel(cs, fontsize=8.5)
    gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 9), np.linspace(-1.6, 1.6, 7))
    ax.quiver(gx, gy, 2 * gx, 8 * gy, color=wbplot.TEXT_MUTED, alpha=0.6, angles="xy", scale=80, width=0.003)
    p = np.array([2.0, 1.0])
    grad = np.array([4.0, 8.0])
    q = p - 0.1 * grad
    ax.scatter(*p, color=C[1], zorder=3)
    ax.annotate("", xy=p + 0.12 * grad, xytext=p, arrowprops=dict(arrowstyle="->", color=C[1], lw=2.2))
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color=C[2], lw=2.2))
    handles = [Line2D([], [], color=C[1], lw=2.2, label="∇f(2, 1) = (4, 8), flèche réduite"),
               Line2D([], [], color=C[2], lw=2.2, label="un pas de descente (η = 0,1) : de (2, 1) à (1,6 ; 0,2)"),
               Line2D([], [], color=wbplot.TEXT_MUTED, lw=1, label="le gradient en chaque point (réduit)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=9, frameon=False)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-2, 2.3)
    ax.set_title("f(x, y) = x² + 4y² : lignes de niveau 4, 8, 12, 16 et gradient")
    fig.tight_layout()
    save(fig, "lignes_de_niveau.png")


def fig_chain_graph() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axis("off")
    nodes = {"x": (0.5, 1.5), "u": (3, 2.6), "v": (3, 0.4), "z": (5.5, 1.5)}
    for name, (x, y) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.38, facecolor=wbplot.SURFACE, edgecolor=C[0], lw=2))
        ax.text(x, y, name, ha="center", va="center", fontsize=15)
    edges = [("x", "u", "du/dx", C[1]), ("x", "v", "dv/dx", C[2]), ("u", "z", "∂z/∂u", C[1]), ("v", "z", "∂z/∂v", C[2])]
    for a, b, label, color in edges:
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        d = np.array([x1 - x0, y1 - y0])
        d = d / np.linalg.norm(d) * 0.42
        ax.annotate("", xy=(x1 - d[0], y1 - d[1]), xytext=(x0 + d[0], y0 + d[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        above = (y0 + y1) / 2 > 1.5  # upper edges: label above; lower edges: label below
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.28 if above else -0.38), label, color=color, ha="center", fontsize=11)
    ax.text(3, -0.55, "dz/dx = ∂z/∂u · du/dx  +  ∂z/∂v · dv/dx   (un terme par chemin)", ha="center", fontsize=11)
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.9, 3.2)
    ax.set_title("Règle de la chaîne à plusieurs variables : la somme sur les chemins")
    save(fig, "somme_sur_les_chemins.png")


def fig_law_of_large_numbers() -> None:
    rng = np.random.default_rng(3)
    n = 5000
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for run, color in zip(range(3), C):
        rolls = rng.integers(1, 7, size=n)
        freq = np.cumsum(rolls == 6) / np.arange(1, n + 1)
        ax.plot(np.arange(1, n + 1), freq, color=color, lw=1.4, label=f"série {run + 1}")
    ax.axhline(1 / 6, color=wbplot.TEXT, ls="--", lw=1, label="1/6")
    ax.set_xscale("log")
    ax.set_xlabel("nombre de lancers (échelle log)")
    ax.set_ylabel("fréquence des 6")
    ax.set_title("Loi des grands nombres : la fréquence se rapproche de la probabilité")
    ax.legend()
    fig.tight_layout()
    save(fig, "grands_nombres.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 0B -> {OUT.relative_to(ROOT)}")
    for draw in [fig_functions, fig_geometric, fig_exp_log, fig_moving_average, fig_cosine, fig_vectors,
                 fig_dot_product, fig_matmul, fig_tangent, fig_extremum, fig_contour, fig_chain_graph, fig_law_of_large_numbers]:
        draw()
    return 0


if __name__ == "__main__":
    sys.exit(main())
