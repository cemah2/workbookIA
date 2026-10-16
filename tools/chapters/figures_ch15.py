#!/usr/bin/env python
"""Draw the figures of chapter 15 (used by Claude).

    python tools/chapters/figures_ch15.py

Writes small PNG files into chapitres/ch15_sklearn/figures/. Every figure is computed or drawn here
(never copied from the book): the three outputs of a binary logistic regression (predict,
decision_function, predict_proba), a MinMaxScaler learned on training points and applied to test points,
three clusterings of two concentric circles, a grid search against a random search when only one
hyperparameter matters (with the probability that at least one random trial lands in the top q), the
folds of a cross-validation with and without a Pipeline, and the boundaries of a polynomial ridge
classifier. None of them uses the data or the numbers of an exercise: no penguins, no sine, no moons,
no seven blobs, no spirals (the notebook uses them), and the paper exercises 15.1 to 15.5 use other
grids, scores, ranges, degrees and quantiles (q = 5 %; the figure draws 2, 10 and 20 %).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch15_sklearn" / "figures"
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


def light(colors):
    """Pale versions of the class colours, for the background regions."""
    from matplotlib.colors import to_rgb

    return ListedColormap([tuple(0.72 + 0.28 * c for c in to_rgb(col)) for col in colors])


# ---------------------------------------------------------------------------- predict, scores, probabilities
def fig_outputs() -> None:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1501)
    x = np.r_[rng.normal(-1.0, 1.0, 80), rng.normal(1.5, 1.0, 80)]
    y = np.r_[np.zeros(80, int), np.ones(80, int)]
    model = LogisticRegression().fit(x[:, None], y)
    grid = np.linspace(-4.5, 5, 400)[:, None]
    score = model.decision_function(grid)
    proba = model.predict_proba(grid)[:, 1]
    cut = float(-model.intercept_[0] / model.coef_[0, 0])           # where the score is 0
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.1))
    for ax in (a, b):
        ax.axvspan(-4.5, cut, color=C[0], alpha=0.08)
        ax.axvspan(cut, 5, color=C[3], alpha=0.08)
        ax.axvline(cut, color=GREY, lw=0.9, ls="--")
        ax.set_xlim(-4.5, 5)
        ax.set_xlabel("feature x")
    a.plot(grid[:, 0], score, color=INK, lw=1.8, label="decision_function(x)")
    a.axhline(0, color=GREY, lw=0.8)
    a.text(cut - 0.2, 4.2, "predict → 0", ha="right", color=C[0], fontsize=9.5)
    a.text(cut + 0.2, 4.2, "predict → 1", ha="left", color=C[3], fontsize=9.5)
    a.set_ylabel("score (log-odds)")
    a.set_title("(a) un score sans borne : signe → classe", fontsize=10.5)
    a.legend(fontsize=8.5, loc="lower right")
    jitter = rng.uniform(-0.03, 0.03, len(x))
    b.scatter(x[y == 0], 0 + jitter[y == 0], s=9, color=C[0], label="exemples de la classe 0")
    b.scatter(x[y == 1], 1 + jitter[y == 1], s=9, color=C[3], label="exemples de la classe 1")
    b.plot(grid[:, 0], proba, color=INK, lw=1.8, label="predict_proba(x)[:, 1]")
    b.axhline(0.5, color=GREY, lw=0.8)
    b.set_ylim(-0.08, 1.08)
    b.set_ylabel("probabilité de la classe 1")
    b.set_title("(b) la sigmoïde du score : une probabilité", fontsize=10.5)
    b.legend(fontsize=8.5, loc="center right")
    fig.tight_layout()
    save(fig, "sorties_classifieur.png")


# ---------------------------------------------------------------------------- a scaler learned on the training set
def fig_scaler() -> None:
    from sklearn.preprocessing import MinMaxScaler

    rng = np.random.default_rng(1502)
    train = np.c_[rng.uniform(1.0, 4.0, 60), rng.uniform(-1.0, 3.0, 60)]
    test = np.c_[rng.uniform(0.0, 5.5, 25), rng.uniform(0.0, 1.6, 25)]
    scaler = MinMaxScaler().fit(train)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    a.scatter(*train.T, s=14, color=C[0], label="entraînement (sert au fit)")
    a.scatter(*test.T, s=18, color=C[1], marker="^", label="test (seulement transform)")
    a.add_patch(Rectangle(train.min(0), *(train.max(0) - train.min(0)), fill=False, ec=INK, lw=1, ls="--"))
    a.set_title("(a) données d'origine ; le cadre : étendue de l'entraînement", fontsize=10.5)
    a.set_xlabel("feature 1")
    a.set_ylabel("feature 2")
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    t_train, t_test = scaler.transform(train), scaler.transform(test)
    b.scatter(*t_train.T, s=14, color=C[0])
    b.scatter(*t_test.T, s=18, color=C[1], marker="^")
    b.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=1, ls="--"))
    b.set_title("(b) après le MinMaxScaler appris sur l'entraînement", fontsize=10.5)
    b.set_xlabel("feature 1 transformée")
    b.set_ylabel("feature 2 transformée")
    for ax in (a, b):
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    save(fig, "scaler_test.png")


# ---------------------------------------------------------------------------- three clusterings
def fig_clustering() -> None:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.datasets import make_circles

    X, _ = make_circles(n_samples=400, factor=0.45, noise=0.05, random_state=1503)
    models = [("KMeans(n_clusters=2)", KMeans(n_clusters=2, random_state=0)),
              ("DBSCAN(eps=0.2)", DBSCAN(eps=0.2, min_samples=5)),
              ("AgglomerativeClustering\n(n_clusters=2, linkage=\"single\")",
               AgglomerativeClustering(n_clusters=2, linkage="single"))]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, (title, model) in zip(axes, models):
        labels = model.fit_predict(X)
        for k in np.unique(labels):
            color = GREY if k == -1 else C[[0, 3, 2, 4][k % 4]]
            ax.scatter(*X[labels == k].T, s=9, color=color, label="bruit (−1)" if k == -1 else None)
        if hasattr(model, "cluster_centers_"):
            ax.scatter(*model.cluster_centers_.T, s=120, marker="X", color=INK, label="centres")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title, fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "clustering.png")


# ---------------------------------------------------------------------------- grid against random search
def fig_grid_random() -> None:
    rng = np.random.default_rng(1504)

    def score(h1, h2):                       # only h1 matters (a narrow peak), h2 barely
        return np.exp(-((h1 - 0.62) / 0.08) ** 2) + 0.03 * h2

    g = np.linspace(0, 1, 300)
    G1, G2 = np.meshgrid(g, g)
    grid = np.array([(a, b) for a in (1 / 6, 1 / 2, 5 / 6) for b in (1 / 6, 1 / 2, 5 / 6)])
    rand = rng.uniform(0, 1, (9, 2))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, pts, title in ((axes[0], grid, "(a) grille 3 × 3 : 3 valeurs de h₁"),
                           (axes[1], rand, "(b) 9 essais au hasard : 9 valeurs de h₁")):
        ax.contourf(G1, G2, score(G1, G2), levels=12, cmap=wbplot.sequential_cmap(), alpha=0.75)
        ax.scatter(*pts.T, s=55, color=C[3], edgecolor="white", zorder=3)
        for h1 in pts[:, 0]:
            ax.plot([h1, h1], [-0.06, -0.01], color=C[3], lw=2, clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("h₁ (important)")
        ax.set_ylabel("h₂ (sans effet ou presque)")
        ax.set_title(title, fontsize=10.5)
    n = np.arange(1, 101)
    for q, color in zip((0.02, 0.10, 0.20), (C[0], C[1], C[3])):
        axes[2].plot(n, 1 - (1 - q) ** n, lw=1.7, color=color, label=f"q = {fr(100 * q, 0)} %")
    axes[2].set_xlabel("nombre d'essais au hasard n")
    axes[2].set_ylabel("P(au moins un essai dans le meilleur q)")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("(c) 1 − (1 − q)ⁿ", fontsize=10.5)
    axes[2].legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save(fig, "grille_hasard.png")


# ---------------------------------------------------------------------------- folds with and without a Pipeline
def fig_pipeline_cv() -> None:
    k = 5
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, leak in zip(axes, (True, False)):
        ax.set_xlim(-0.2, k + 2.9)
        ax.set_ylim(-1.3, k + 0.7)
        ax.axis("off")
        if leak:
            ax.add_patch(FancyBboxPatch((0, k - 0.05), k, 0.6, boxstyle="round,pad=0.02", fc=C[3], ec="none",
                                        alpha=0.85))
            ax.text(k / 2, k + 0.25, "scaler.fit(X) sur TOUTES les lignes, une fois", ha="center", va="center",
                    color="white", fontsize=9.5, weight="bold")
        for r in range(k):
            yrow = k - 1 - r
            for c in range(k):
                val = c == r
                ax.add_patch(Rectangle((c, yrow + 0.08), 0.94, 0.7, fc=C[1] if val else C[0], ec="white",
                                       alpha=0.9))
            if leak:
                note = "le fold de validation a déjà servi au fit"
            else:
                note = "fit du pipeline sur les 4 autres folds ;\ntransform + score sur le fold coloré"
            ax.text(k + 0.15, yrow + 0.43, note, va="center", fontsize=8.2, color=INK)
            ax.text(-0.12, yrow + 0.43, f"tour {r + 1}", ha="right", va="center", fontsize=8.5, color=INK)
        ax.add_patch(Rectangle((0, -1.1), 0.5, 0.4, fc=C[0], ec="none", alpha=0.9))
        ax.text(0.6, -0.9, "entraînement du tour", va="center", fontsize=8.5)
        ax.add_patch(Rectangle((2.6, -1.1), 0.5, 0.4, fc=C[1], ec="none", alpha=0.9))
        ax.text(3.2, -0.9, "validation du tour", va="center", fontsize=8.5)
        ax.set_title("(a) prétraitement avant la validation croisée : fuite" if leak
                     else "(b) cross_val_score(Pipeline) : le scaler réappris à chaque tour", fontsize=10.5)
    fig.tight_layout()
    save(fig, "pipeline_cv.png")


# ---------------------------------------------------------------------------- polynomial ridge boundaries
def fig_polynomial() -> None:
    from sklearn.datasets import make_circles
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    X, y = make_circles(n_samples=300, factor=0.55, noise=0.12, random_state=1506)
    xx, yy = np.meshgrid(np.linspace(-1.6, 1.6, 300), np.linspace(-1.6, 1.6, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, degree in zip(axes, (1, 2, 8)):
        model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), RidgeClassifier(alpha=1e-3)).fit(X, y)
        z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5], cmap=light([C[0], C[3]]))
        ax.contour(xx, yy, model.decision_function(grid).reshape(xx.shape), levels=[0], colors=INK, linewidths=1)
        for c in (0, 1):
            ax.scatter(*X[y == c].T, s=16, color=(C[0], C[3])[c], edgecolor="white", linewidth=0.4)
        ax.set_title(f"PolynomialFeatures(degree={degree})", fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "features_polynomiales.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 15 → {OUT.relative_to(ROOT)}")
    fig_outputs()
    fig_scaler()
    fig_clustering()
    fig_grid_random()
    fig_pipeline_cv()
    fig_polynomial()
    return 0


if __name__ == "__main__":
    sys.exit(main())
