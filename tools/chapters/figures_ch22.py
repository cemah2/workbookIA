#!/usr/bin/env python
"""Draw the figures of chapter 22 (used by Claude).

    python tools/chapters/figures_ch22.py

Writes small PNG files into chapitres/ch22_rnn/figures/. Every figure is computed or drawn here (never
copied from the book): the windows of a short series, a recurrent cell rolled and unrolled, the size of a
product of gradient factors, the LSTM cell, the five input/output structures and a softmax at four
temperatures. None of them uses the numbers of an exercise: the series, the sizes and the logits differ
from those of 02_exercices.md (the windows are those of the example of the sheet: T = 10, w = 3, h = 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch22_rnn" / "figures"
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


def arrow(ax, start, end, color=INK, style="-|>", lw=1.4, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def box(ax, xy, w, h, text, color, text_color="white", fontsize=11, round_=True):
    x, y = xy
    if round_:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=INK, lw=1)
    else:
        patch = Rectangle((x, y), w, h, fc=color, ec=INK, lw=1)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def figure_windows() -> None:
    """Ten values, windows of 3 values and a target 2 steps after the last value of each window."""
    t, w, h = 10, 3, 2
    n = t - w - h + 1
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for row in range(n):
        y = n - 1 - row
        for k in range(t):
            if row <= k < row + w:
                color, text = C[0], "w"
            elif k == row + w - 1 + h:
                color, text = C[1], "cible"
            else:
                color, text = "#f2f2f2", ""
            ax.add_patch(Rectangle((k, y), 0.92, 0.8, fc=color, ec=GREY, lw=0.8))
            if text == "cible":
                ax.text(k + 0.46, y + 0.4, "cible", ha="center", va="center", fontsize=7, color="white")
        ax.text(-0.3, y + 0.4, f"exemple {row}", ha="right", va="center", fontsize=9, color=INK)
    for k in range(t):
        ax.text(k + 0.46, n + 0.2, str(k), ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(t / 2, n + 0.9, "position dans la série (T = 10)", ha="center", fontsize=10, color=INK)
    ax.set_xlim(-2.4, t + 0.2)
    ax.set_ylim(-0.4, n + 1.4)
    ax.axis("off")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=C[0]), Rectangle((0, 0), 1, 1, fc=C[1])],
              labels=["fenêtre (w = 3 valeurs)", "cible (h = 2 pas plus loin)"], loc="lower right",
              bbox_to_anchor=(1.0, -0.22), ncol=2, frameon=False, fontsize=9)
    save(fig, "fenetres.png")


def figure_unrolled() -> None:
    """The rolled cell with its delay, and the same cell unrolled over four time steps."""
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    # rolled
    box(ax, (0.4, 1.3), 1.4, 1.0, "cellule\n(W_ih, W_hh)", C[0], fontsize=9)
    arrow(ax, (1.1, 0.35), (1.1, 1.28))
    ax.text(1.1, 0.15, "$x_t$", ha="center", va="center", fontsize=12)
    arrow(ax, (1.1, 2.32), (1.1, 3.15))
    ax.text(1.1, 3.35, "$h_t$", ha="center", va="center", fontsize=12)
    ax.add_patch(Rectangle((2.15, 1.65), 0.3, 0.3, fc=INK, ec=INK))
    ax.text(2.3, 1.38, "délai", ha="center", va="top", fontsize=8, color=INK)
    arrow(ax, (1.82, 2.05), (2.12, 1.86), color=C[2], rad=-0.3)
    arrow(ax, (2.3, 1.98), (1.82, 1.55), color=C[2], rad=-0.9)
    ax.text(1.1, -0.45, "enroulé", ha="center", fontsize=10, color=INK)
    # unrolled
    x0 = 3.6
    for k in range(4):
        x = x0 + 1.55 * k
        box(ax, (x, 1.3), 1.1, 1.0, "cellule", C[0], fontsize=9)
        arrow(ax, (x + 0.55, 0.35), (x + 0.55, 1.28))
        ax.text(x + 0.55, 0.15, f"$x_{k + 1}$", ha="center", va="center", fontsize=12)
        arrow(ax, (x + 0.55, 2.32), (x + 0.55, 3.15))
        ax.text(x + 0.55, 3.35, f"$h_{k + 1}$", ha="center", va="center", fontsize=12)
        start = (x - 0.45, 1.8) if k else (x - 0.45, 1.8)
        arrow(ax, start, (x - 0.02, 1.8), color=C[2], lw=2.0)
        label = "$h_0$" if k == 0 else f"$h_{k}$"
        ax.text(x - 0.25, 2.0, label, ha="center", va="bottom", fontsize=9, color=C[2])
    ax.text(x0 + 3.0, -0.45, "déroulé : une seule cellule, les mêmes poids à chaque pas de temps", ha="center",
            fontsize=10, color=INK)
    ax.set_xlim(0, x0 + 6.5)
    ax.set_ylim(-0.7, 3.7)
    ax.axis("off")
    save(fig, "deroulement.png")


def figure_gradient() -> None:
    """|q|^n for three factor sizes: the product of gradient factors along n steps back in time."""
    n = np.arange(0, 31)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for q, color, label in [(0.7, C[0], "facteurs ≈ 0,7 : le gradient s'évanouit"),
                            (1.0, C[3], "facteurs = 1"),
                            (1.2, C[1], "facteurs ≈ 1,2 : le gradient explose")]:
        ax.semilogy(n, q ** n, color=color, lw=2, label=label)
    ax.set_xlabel("nombre de pas remontés dans le temps")
    ax.set_ylabel("taille du produit (échelle log)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"$10^{{{int(round(np.log10(v)))}}}$"))
    ax.set_ylim(1e-5, 1e3)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3, which="major")
    save(fig, "gradient.png")


def figure_lstm() -> None:
    """The LSTM cell: the memory line at the top, the gates f, i, o and the candidate g below."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.add_patch(FancyBboxPatch((0.6, 0.2), 7.0, 3.6, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#f7f7f7",
                                ec=GREY, lw=1))
    # memory line
    arrow(ax, (0.0, 3.2), (2.05, 3.2), color=C[2], lw=2.2)
    ax.text(0.05, 3.42, "$c_{t-1}$", fontsize=12, color=C[2])
    for xc, sym in [(2.3, "×"), (4.3, "+")]:
        ax.add_patch(plt.Circle((xc, 3.2), 0.24, fc="white", ec=INK, lw=1.2))
        ax.text(xc, 3.2, sym, ha="center", va="center", fontsize=13)
    arrow(ax, (2.55, 3.2), (4.05, 3.2), color=C[2], lw=2.2)
    arrow(ax, (4.55, 3.2), (8.3, 3.2), color=C[2], lw=2.2)
    ax.text(7.95, 3.42, "$c_t$", fontsize=12, color=C[2])
    # gates
    gates = [(1.8, "σ", "$f_t$", "oubli"), (3.0, "σ", "$i_t$", "entrée"), (3.9, "tanh", "$g_t$", "candidat"),
             (5.3, "σ", "$o_t$", "sortie")]
    for x, fn, name, role in gates:
        box(ax, (x, 1.0), 0.9 if fn == "tanh" else 0.7, 0.5, fn, C[0], fontsize=10)
        ax.text(x + 0.35, 0.75, f"{name} ({role})", ha="left" if fn == "tanh" else "center", va="top", fontsize=8,
                color=INK)
    arrow(ax, (2.15, 1.52), (2.3, 2.95))                  # f -> x
    ax.add_patch(plt.Circle((3.75, 2.25), 0.22, fc="white", ec=INK, lw=1.2))
    ax.text(3.75, 2.25, "×", ha="center", va="center", fontsize=12)
    arrow(ax, (3.35, 1.52), (3.62, 2.08))                 # i -> x
    arrow(ax, (4.3, 1.52), (3.9, 2.08))                   # g -> x
    arrow(ax, (3.95, 2.38), (4.25, 2.96))                 # i*g -> +
    # output
    box(ax, (6.15, 2.45), 0.8, 0.42, "tanh", C[4] if len(C) > 4 else C[3], fontsize=9)
    arrow(ax, (6.55, 3.18), (6.55, 2.9))
    ax.add_patch(plt.Circle((6.55, 1.75), 0.22, fc="white", ec=INK, lw=1.2))
    ax.text(6.55, 1.75, "×", ha="center", va="center", fontsize=12)
    arrow(ax, (6.55, 2.43), (6.55, 1.98))
    arrow(ax, (6.0, 1.3), (6.33, 1.68))                   # o -> x
    arrow(ax, (6.78, 1.75), (8.3, 1.75), lw=2.0)
    ax.text(7.95, 1.95, "$h_t$", fontsize=12)
    # inputs
    ax.plot([0.0, 5.65], [0.45, 0.45], color=INK, lw=1.2)
    for x, fn, _n, _r in gates:
        arrow(ax, (x + 0.35 if fn != "tanh" else x + 0.45, 0.45), (x + 0.35 if fn != "tanh" else x + 0.45, 0.98))
    ax.text(0.05, 0.6, "$h_{t-1}$, $x_t$", fontsize=11)
    ax.text(4.2, 3.55, "$c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$", fontsize=10, ha="center", color=INK)
    ax.text(7.45, 1.25, "$h_t = o_t \\odot \\tanh(c_t)$", fontsize=10, ha="center", color=INK)
    ax.set_xlim(-0.1, 8.5)
    ax.set_ylim(0.0, 3.9)
    ax.axis("off")
    save(fig, "lstm.png")


def figure_structures() -> None:
    """One-to-one, one-to-many, many-to-one, many-to-many (delayed and synchronised)."""
    specs = [("one-to-one", [0], [0], 1), ("one-to-many", [0], [0, 1, 2], 3),
             ("many-to-one", [0, 1, 2], [2], 3), ("many-to-many\n(décalé)", [0, 1, 2], [3, 4], 5),
             ("many-to-many\n(synchronisé)", [0, 1, 2], [0, 1, 2], 3)]
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.2), gridspec_kw={"width_ratios": [1, 3, 3, 5, 3]})
    for ax, (title, ins, outs, steps) in zip(axes, specs):
        for k in range(steps):
            box(ax, (k, 1.0), 0.7, 0.6, "", C[0], round_=False)
            if k in ins:
                box(ax, (k, 0.0), 0.7, 0.5, "", C[3], round_=False)
                arrow(ax, (k + 0.35, 0.5), (k + 0.35, 1.0))
            if k in outs:
                box(ax, (k, 2.1), 0.7, 0.5, "", C[1], round_=False)
                arrow(ax, (k + 0.35, 1.6), (k + 0.35, 2.1))
            if k:
                arrow(ax, (k - 0.3, 1.3), (k, 1.3), color=C[2])
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlim(-0.2, max(steps, 1) + 0.0)
        ax.set_ylim(-0.2, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=C[3]), Rectangle((0, 0), 1, 1, fc=C[0]),
                        Rectangle((0, 0), 1, 1, fc=C[1])], labels=["entrée", "cellule (même poids)", "sortie"],
               loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    save(fig, "structures.png")


def figure_temperature() -> None:
    """A softmax over six characters at four temperatures."""
    chars = ["e", "a", "i", "o", "u", "y"]
    logits = np.array([2.0, 1.2, 0.5, 0.0, -0.5, -1.0])
    temps = [0.25, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    for ax, t, color in zip(axes, temps, [C[0], C[3], C[1], C[2]]):
        z = logits / t
        p = np.exp(z - z.max())
        p /= p.sum()
        ax.bar(chars, p, color=color)
        ax.set_title(f"T = {french_number(t)}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("probabilité")
    save(fig, "temperature.png")


def main() -> int:
    wbplot.set_style()
    figure_windows()
    figure_unrolled()
    figure_gradient()
    figure_lstm()
    figure_structures()
    figure_temperature()
    print(f"✅ figures written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
