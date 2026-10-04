#!/usr/bin/env python
"""Draw the figures of the course sheet of chapter 1 (used by Claude).

    python tools/chapters/figures_ch01.py

Writes small PNG files into chapitres/ch01_introduction/figures/. Every figure is
computed from the workbook's data or from synthetic data (never copied from the book)
and uses the workbook style.
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

from wb import datasets, synth  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch01_introduction" / "figures"
C = wbplot.CATEGORICAL
DPI = 110


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma and a true minus sign: 0,5 and −2."""
    return f"{value:g}".replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:  # decimal commas on every visible linear axis
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "linear" and axis.get_visible():
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig_penguins() -> None:
    """Classification: two features of the training penguins of the notebook, coloured by species (the label)."""
    complete = datasets.load_penguins(dropna=True)
    order = np.random.default_rng(42).permutation(len(complete))   # the same split as the notebook (part B)
    df = complete.iloc[order[100:]]                                 # training set only: the test set stays unseen
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for color, species in zip(C, ["Adelie", "Chinstrap", "Gentoo"]):
        part = df[df["species"] == species]
        ax.scatter(part["flipper_length_mm"], part["bill_length_mm"], s=16, color=color, alpha=0.85,
                   label=species)
    ax.set_xlabel("longueur de la nageoire (mm) : une feature")
    ax.set_ylabel("longueur du bec (mm) : une autre feature")
    ax.set_title("un point = un échantillon (un manchot) ; la couleur = le label (l'espèce)", fontsize=10.5)
    ax.legend(title="espèce", fontsize=9)
    save(fig, "manchots_features_label.png")


def fig_line_or_curve() -> None:
    """Regression: a straight line and a more flexible curve through the same noisy points."""
    x, y, _ = synth.make_polynomial(n=25, coefs=(1.0, 0.8, 0.0, -1.2), noise=0.12, x_range=(-1.2, 1.2), seed=3)
    grid = np.linspace(-1.25, 1.25, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, degree, title in [(axes[0], 1, "une droite : 2 paramètres"),
                              (axes[1], 3, "une courbe : 4 paramètres")]:
        coefs = np.polyfit(x, y, degree)
        ax.scatter(x, y, s=18, color=C[0], zorder=3, label="mesures")
        ax.plot(grid, np.polyval(coefs, grid), color=C[1], lw=2.2, label="modèle")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("x (entrée)")
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("y (valeur à prédire)")
    fig.suptitle("Régression : prédire une quantité à partir d'une entrée", fontsize=11.5)
    fig.tight_layout()
    save(fig, "regression_droite_courbe.png")


def fig_clustering() -> None:
    """Clustering: points without labels, then the groups found by an algorithm."""
    from sklearn.cluster import KMeans

    X, _ = synth.make_blobs(n=240, centers=3, std=0.9, seed=4)
    groups = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
    axes[0].scatter(X[:, 0], X[:, 1], s=14, color="0.45")
    axes[0].set_title("ce que reçoit l'algorithme : des points, sans labels", fontsize=10.5)
    for k in range(3):
        axes[1].scatter(X[groups == k, 0], X[groups == k, 1], s=14, color=C[k], label=f"groupe {k}")
    axes[1].set_title("ce qu'il renvoie : 3 groupes (sans noms)", fontsize=10.5)
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    fig.tight_layout()
    save(fig, "clustering.png")


def fig_mnist() -> None:
    """Two handwritten examples of each digit."""
    X, y = datasets.load_mnist("train")
    picks = np.concatenate([np.flatnonzero(y == d)[:2] for d in range(10)])
    order = np.concatenate([picks[0::2], picks[1::2]])
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.4))
    for ax, i in zip(axes.ravel(), order):
        ax.imshow(X[i], cmap="gray_r", vmin=0, vmax=255)
        ax.set_title(str(y[i]), fontsize=9)
        ax.axis("off")
    fig.suptitle("MNIST : des chiffres écrits à la main, 28 × 28 pixels, et leur label", fontsize=11)
    fig.tight_layout()
    save(fig, "mnist_exemples.png")


def fig_road() -> None:
    """Dimensionality reduction: cars on a winding one-lane road, 2 numbers -> 1 number."""
    t = np.linspace(0, 1, 400)
    road_x, road_y = 10 * t, 1.6 * np.sin(2.2 * np.pi * t) + 2 * t
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(road_x), np.diff(road_y)))])
    cars = [40, 120, 210, 300, 370]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].plot(road_x, road_y, color="0.7", lw=7, solid_capstyle="round")
    for k, i in enumerate(cars):
        axes[0].plot(road_x[i], road_y[i], "o", ms=9, color=C[k % len(C)])
        axes[0].annotate(f"({road_x[i]:.1f} ; {road_y[i]:.1f})".replace(".", ","), (road_x[i], road_y[i]),
                         textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8)
    axes[0].set_title("sur la carte : 2 nombres par voiture (x, y)", fontsize=10.5)
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    axes[0].set_ylim(-2.2, 4.6)
    axes[1].hlines(0, 0, arc[-1], color="0.7", lw=7)
    for k, i in enumerate(cars):
        axes[1].plot(arc[i], 0, "o", ms=9, color=C[k % len(C)])
        axes[1].annotate(f"{arc[i]:.1f}".replace(".", ","), (arc[i], 0), textcoords="offset points",
                         xytext=(0, 11), ha="center", fontsize=8)
    axes[1].set_title("le long de la route : 1 seul nombre (km parcourus)", fontsize=10.5)
    axes[1].set_xlabel("distance depuis le début de la route (km)")
    axes[1].set_ylim(-1, 1)
    axes[1].yaxis.set_visible(False)
    for side in ("left", "right", "top"):
        axes[1].spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "reduction_dimension.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    fig_penguins()
    fig_line_or_curve()
    fig_clustering()
    fig_mnist()
    fig_road()
    return 0


if __name__ == "__main__":
    sys.exit(main())
