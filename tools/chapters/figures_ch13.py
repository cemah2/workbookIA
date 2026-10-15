#!/usr/bin/env python
"""Draw the figures of chapter 13 (used by Claude).

    python tools/chapters/figures_ch13.py

Writes small PNG files into chapitres/ch13_classifieurs/figures/. Every figure is computed or drawn
here (never copied from the book): k-nearest-neighbour boundaries for three values of k on three
overlapping blobs, the margin and the support vectors of a linear SVM (separable, then overlapping
classes), a one-dimensional problem lifted onto a parabola, the RBF kernel for three values of gamma,
the rectangles of a decision tree that approximates a diagonal boundary, a depth-2 tree drawn with
plot_tree, the impurity measures of a two-class node, the posterior of a one-dimensional Gaussian
naive Bayes with unequal variances, and the sigmoid with the log-loss. None of them uses the data of
an exercise (no penguins, no MNIST, no moons, no circles, no texts): the paper exercises 13.1 to
13.9 and the notebook exercises 13.12 to 13.36 use other data and other numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

from mylearn_ref import naive_bayes as nb  # noqa: E402
from mylearn_ref import neighbors as nn  # noqa: E402
from mylearn_ref import tree as mt  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch13_classifieurs" / "figures"
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


def grid_of(X, pad=0.6, n=300):
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


# ---------------------------------------------------------------------------- kNN and k
def three_blobs(seed=1301, n=70):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [2.2, 0.6], [1.0, 2.2]])
    X = np.vstack([rng.normal(c, 0.85, (n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y


def fig_knn() -> None:
    X, y = three_blobs()
    xx, yy, G = grid_of(X, n=220)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, k in zip(axes, (1, 15, 50)):
        pred = nn.KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(G).reshape(xx.shape)
        ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5, 2.5], cmap=light(C[:3]))
        ax.contour(xx, yy, pred, levels=[0.5, 1.5], colors=[INK], linewidths=0.7)
        for c in range(3):
            ax.scatter(*X[y == c].T, s=13, color=C[c], edgecolor="white", linewidth=0.3, label=f"classe {c}")
        acc = nn.KNeighborsClassifier(n_neighbors=k).fit(X, y).score(X, y)
        ax.set_title(f"k = {k} : accuracy d'entraînement {fr(acc, 2)}", fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    save(fig, "knn_k.png")


# ---------------------------------------------------------------------------- SVM margin
def fig_svm() -> None:
    from sklearn.svm import SVC

    rng = np.random.default_rng(1302)
    sep_X = np.vstack([rng.normal([0, 0], 0.55, (25, 2)), rng.normal([3.2, 2.4], 0.55, (25, 2))])
    over_X = np.vstack([rng.normal([0, 0], 1.0, (40, 2)), rng.normal([2.4, 1.8], 1.0, (40, 2))])
    y25, y40 = np.repeat([0, 1], 25), np.repeat([0, 1], 40)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    cases = [(axes[0], sep_X, y25, 1e6, "(a) Classes séparables, marge dure"),
             (axes[1], over_X, y40, 0.1, "(b) Classes qui se chevauchent, marge souple (C = 0,1)")]
    for ax, X, y, Cval, title in cases:
        svm = SVC(kernel="linear", C=Cval).fit(X, y)
        xx, yy, G = grid_of(X, pad=0.8, n=200)
        f = svm.decision_function(G).reshape(xx.shape)
        ax.contourf(xx, yy, (f > 0).astype(int), levels=[-0.5, 0.5, 1.5], cmap=light(C[:2]))
        ax.contour(xx, yy, f, levels=[-1, 0, 1], colors=[INK], linestyles=["--", "-", "--"], linewidths=[1, 1.6, 1])
        for c in range(2):
            ax.scatter(*X[y == c].T, s=18, color=C[c], edgecolor="white", linewidth=0.3)
        sv = svm.support_vectors_
        ax.scatter(*sv.T, s=110, facecolor="none", edgecolor=INK, linewidth=1.1, label=f"{len(sv)} vecteurs de support")
        w = svm.coef_[0]
        ax.set_title(f"{title}\nlargeur de la bande 2/‖w‖ = {fr(2 / np.linalg.norm(w), 2)}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.tight_layout()
    save(fig, "svm_marge.png")


# ---------------------------------------------------------------------------- lifting 1D -> 2D
def fig_lift() -> None:
    rng = np.random.default_rng(1303)
    x_in = rng.uniform(-1.3, 1.3, 14)
    x_out = np.concatenate([rng.uniform(-3, -1.9, 7), rng.uniform(1.9, 3, 7)])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ax.scatter(x_in, np.zeros_like(x_in), s=40, color=C[0], label="classe 0")
    ax.scatter(x_out, np.zeros_like(x_out), s=40, color=C[1], marker="s", label="classe 1")
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel("$x$")
    ax.set_title("(a) Sur une droite : aucun seuil unique ne sépare les classes", fontsize=10)
    ax.legend(loc="upper center", fontsize=8.5, ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    ax.set_ylim(-0.5, 1)
    ax = axes[1]
    xs = np.linspace(-3.2, 3.2, 200)
    ax.plot(xs, xs**2, color=GREY, lw=1, label="$z = x^2$")
    ax.scatter(x_in, x_in**2, s=40, color=C[0])
    ax.scatter(x_out, x_out**2, s=40, color=C[1], marker="s")
    ax.axhline(2.7, color=INK, lw=1.6, label="droite séparatrice $z = 2{,}7$")
    for s in (-1, 1):
        ax.axvline(s * np.sqrt(2.7), color=INK, lw=0.8, ls=":")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z = x^2$")
    ax.set_title("(b) Relevés sur $(x, x^2)$ : une droite suffit ;\nrevenue sur l'axe, elle donne deux seuils $\\pm 1{,}64$",
                 fontsize=10)
    ax.legend(loc="upper center", fontsize=8.5)
    fig.tight_layout()
    save(fig, "relevement.png")


def fig_rbf() -> None:
    d = np.linspace(0, 4, 300)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, g in enumerate((0.1, 1.0, 10.0)):
        ax.plot(d, np.exp(-g * d**2), color=C[i], lw=2, label=f"γ = {fr(g, 1)}")
    ax.set_xlabel("distance $\\Vert \\mathbf{x} - \\mathbf{z} \\Vert$")
    ax.set_ylabel("$K(\\mathbf{x}, \\mathbf{z}) = e^{-\\gamma \\Vert \\mathbf{x} - \\mathbf{z} \\Vert^2}$")
    ax.set_title("Noyau RBF : la ressemblance de deux points selon leur distance", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "noyau_rbf.png")


# ---------------------------------------------------------------------------- decision trees
def diagonal_data(seed=1304, n=240):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (n, 2))
    y = (X[:, 1] > 0.8 * X[:, 0] + 1.0).astype(int)
    flip = rng.random(n) < 0.06               # a little label noise
    y[flip] = 1 - y[flip]
    return X, y


def fig_tree_regions() -> None:
    X, y = diagonal_data()
    xx, yy, G = grid_of(X, pad=0.2, n=250)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    for ax, depth in zip(axes, (1, 2, 4, None)):
        tree = mt.DecisionTreeClassifier(max_depth=depth).fit(X, y)
        pred = tree.predict(G).reshape(xx.shape)
        ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], cmap=light(C[:2]))
        ax.contour(xx, yy, pred, levels=[0.5], colors=[INK], linewidths=0.8)
        for c in range(2):
            ax.scatter(*X[y == c].T, s=10, color=C[c], edgecolor="white", linewidth=0.3)
        xs = np.linspace(0, 10, 2)
        ax.plot(xs, 0.8 * xs + 1.0, color=GREY, lw=1, ls="--")
        name = "sans limite" if depth is None else f"profondeur max {depth}"
        ax.set_title(f"{name} : {tree.get_n_leaves()} feuilles,\naccuracy d'entraînement {fr(tree.score(X, y), 2)}",
                     fontsize=9.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "arbre_rectangles.png")


def fig_tree_plot() -> None:
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    X, y = diagonal_data()
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    plot_tree(tree, feature_names=["x1", "x2"], class_names=["classe 0", "classe 1"], filled=True,
              impurity=True, rounded=True, fontsize=9, ax=ax)
    save(fig, "arbre_schema.png")


def fig_impurities() -> None:
    p = np.linspace(0, 1, 401)
    q = 1 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nan_to_num(p * np.log2(p)) - np.nan_to_num(q * np.log2(q))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(p, h, color=C[0], lw=2, label="entropie (bits)")
    ax.plot(p, h / 2, color=C[0], lw=1.2, ls="--", label="entropie / 2")
    ax.plot(p, 1 - p**2 - q**2, color=C[1], lw=2, label="Gini $1 - p_1^2 - (1 - p_1)^2$")
    ax.plot(p, np.minimum(p, q), color=C[2], lw=2, label="taux d'erreur $\\min(p_1, 1 - p_1)$")
    ax.set_xlabel("$p_1$ : proportion de la classe 1 dans le nœud")
    ax.set_ylabel("impureté")
    ax.set_title("Trois mesures d'impureté d'un nœud à deux classes", fontsize=10)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "impuretes.png")


# ---------------------------------------------------------------------------- naive Bayes in 1D
def fig_nb_1d() -> None:
    rng = np.random.default_rng(1305)
    x0 = rng.normal(0.0, 1.0, 150)
    x1 = rng.normal(1.5, 2.6, 100)
    X = np.concatenate([x0, x1]).reshape(-1, 1)
    y = np.repeat([0, 1], [150, 100])
    model = nb.GaussianNB().fit(X, y)
    xs = np.linspace(-8, 9, 600)
    post = model.predict_proba(xs.reshape(-1, 1))[:, 1]
    dens = np.exp(nb.log_gaussian_pdf(xs[:, None], model.theta_[:, 0], model.var_[:, 0]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    for c in range(2):
        ax.plot(xs, model.class_prior_[c] * dens[:, c], color=C[c], lw=2,
                label=f"classe {c} : $P({c})\\,\\mathcal{{N}}(x;\\ \\mu_{c}, \\sigma_{c}^2)$, "
                      f"μ = {fr(model.theta_[c, 0], 2)}, σ = {fr(np.sqrt(model.var_[c, 0]), 2)}")
        ax.plot(X[y == c, 0], np.full((y == c).sum(), -0.008 - 0.012 * c), "|", color=C[c], ms=7, alpha=0.6)
    ax.set_ylabel("densité × prior")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Naïve Bayes gaussien à une feature : deux gaussiennes de variances différentes", fontsize=10)
    ax = axes[1]
    ax.plot(xs, post, color=C[1], lw=2)
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    cut = xs[np.flatnonzero(np.diff(np.sign(post - 0.5)))]
    for c in cut:
        ax.axvline(c, color=INK, lw=0.9, ls="--")
    ax.set_ylabel("$P(\\text{classe } 1 \\mid x)$")
    ax.set_xlabel("$x$")
    ax.set_title(f"Posterior : la classe 1 l'emporte des deux côtés, avant {fr(cut[0], 2)} et après {fr(cut[1], 2)}",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "nb_posterieur.png")


# ---------------------------------------------------------------------------- logistic regression
def fig_sigmoid() -> None:
    z = np.linspace(-7, 7, 400)
    s = 1 / (1 + np.exp(-z))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    ax.plot(z, s, color=C[0], lw=2, label="$\\sigma(z) = 1 / (1 + e^{-z})$")
    ax.plot(z, s * (1 - s), color=C[1], lw=1.6, ls="--", label="$\\sigma'(z) = \\sigma(z)\\,(1 - \\sigma(z))$")
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.axvline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("score $z = \\mathbf{w} \\cdot \\mathbf{x} + b$")
    ax.set_title("(a) La sigmoïde transforme un score en probabilité", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[1]
    p = np.linspace(0.005, 0.995, 400)
    ax.plot(p, -np.log(p), color=C[0], lw=2, label="label 1 : $-\\ln \\hat p$")
    ax.plot(p, -np.log(1 - p), color=C[1], lw=2, label="label 0 : $-\\ln(1 - \\hat p)$")
    ax.set_xlabel("probabilité prédite $\\hat p$ de la classe 1")
    ax.set_ylabel("log-loss d'un exemple (nats)")
    ax.set_ylim(0, 5.5)
    ax.set_title("(b) La log-loss punit fort une erreur sûre d'elle", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper center")
    fig.tight_layout()
    save(fig, "sigmoide_logloss.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 13 -> {OUT.relative_to(ROOT)}")
    fig_knn()
    fig_svm()
    fig_lift()
    fig_rbf()
    fig_tree_regions()
    fig_tree_plot()
    fig_impurities()
    fig_nb_1d()
    fig_sigmoid()
    return 0


if __name__ == "__main__":
    sys.exit(main())
