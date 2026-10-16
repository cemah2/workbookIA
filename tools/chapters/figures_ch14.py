#!/usr/bin/env python
"""Draw the figures of chapter 14 (used by Claude).

    python tools/chapters/figures_ch14.py

Writes small PNG files into chapitres/ch14_ensembles/figures/. Every figure is computed or drawn
here (never copied from the book): the probability that a majority of independent voters is right
and the variance of a mean of correlated models, the probability of never being drawn by a
bootstrap and the number of copies of a sample in a bootstrap, the boundaries of one tree, a bagging,
a random forest and ExtraTrees (the reference mylearn.ensemble), the sample weights of AdaBoost after
a few rounds, a gradient boosting on a noisy sine and its residuals, and the impurity and permutation
importances of a forest with an identifier column. None of them uses the data or the numbers of an
exercise (no penguins, no California, no moons): the paper exercises 14.1 to 14.7 use voters right with
the probabilities 0.7, 0.4, 0.6, 0.51 and 0.45 (the figure draws 0.42, 0.55 and 0.65), n = 8, 10 and 1000, and
other points.
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
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

from mylearn_ref import ensemble as me  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch14_ensembles" / "figures"
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


# ---------------------------------------------------------------------------- votes and variance
def fig_votes() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    M = np.arange(1, 102, 2)
    for p, color in zip((0.65, 0.55, 0.42), (C[0], C[1], C[3])):
        a.plot(M, stats.binom.sf(M // 2, M, p), "o-", ms=2.5, lw=1.4, color=color,
               label=f"chaque votant juste à {fr(100 * p, 0)} %")
    a.axhline(0.5, color=GREY, lw=0.8, ls="--")
    a.set_xlabel("nombre de votants M (impair)")
    a.set_ylabel("P(la majorité a raison)")
    a.set_ylim(0, 1.02)
    a.set_title("(a) votants indépendants, deux classes", fontsize=10.5)
    a.legend(fontsize=8.5, loc="center right")
    M = np.arange(1, 101)
    for rho, color in zip((0.0, 0.2, 0.6), (C[0], C[1], C[3])):
        b.plot(M, rho + (1 - rho) / M, lw=1.6, color=color, label=f"ρ = {fr(rho, 1)}")
        if rho > 0:
            b.axhline(rho, color=color, lw=0.8, ls=":")
    b.set_xscale("log")
    b.set_xlabel("nombre de modèles M (échelle log)")
    b.set_ylabel("variance de la moyenne (σ² = 1)")
    b.set_title("(b) moyenne de M modèles de corrélation ρ", fontsize=10.5)
    b.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "vote_majorite.png")


# ---------------------------------------------------------------------------- bootstrap
def fig_bootstrap() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    n = np.unique(np.round(np.logspace(0, 3.3, 60)).astype(int))
    n = n[n >= 2]
    a.plot(n, (1 - 1 / n) ** n, "o-", ms=2.5, lw=1.4, color=C[0], label="(1 − 1/n)ⁿ")
    a.axhline(np.exp(-1), color=C[3], lw=1.0, ls="--", label=f"e⁻¹ ≈ {fr(np.exp(-1), 3)}")
    a.set_xscale("log")
    a.set_xlabel("taille n du jeu d'entraînement (échelle log)")
    a.set_ylabel("P(un exemple n'est jamais tiré)")
    a.set_ylim(0.2, 0.42)
    a.set_title("(a) un bootstrap de n tirages parmi n", fontsize=10.5)
    a.legend(fontsize=8.5)
    rng = np.random.default_rng(1402)
    counts = np.bincount(me.bootstrap_indices(1000, rng=rng), minlength=1000)
    k = np.arange(0, 7)
    share = np.array([np.mean(counts == j) for j in k])
    b.bar(k, share, color=C[0], width=0.7, label="un bootstrap de 1 000 tirages parmi 1 000")
    b.plot(k, stats.poisson.pmf(k, 1), "o", color=C[3], ms=5, label="loi de Poisson de moyenne 1")
    b.set_xlabel("nombre de copies d'un exemple dans le bootstrap")
    b.set_ylabel("part des exemples")
    b.set_title("(b) combien de fois chaque exemple est tiré", fontsize=10.5)
    b.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "bootstrap.png")


# ---------------------------------------------------------------------------- boundaries
def noisy_two_classes(seed=1403, n=400):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    score = X[:, 1] - 0.8 * np.sin(1.6 * X[:, 0])
    y = (score + 0.35 * rng.normal(size=n) > 0).astype(int)
    return X, y


def fig_boundaries() -> None:
    X, y = noisy_two_classes()
    xx, yy = np.meshgrid(np.linspace(-2.1, 2.1, 160), np.linspace(-2.1, 2.1, 160))
    G = np.column_stack([xx.ravel(), yy.ravel()])
    models = [("un arbre sans limite", me.DecisionTreeClassifier(random_state=0)),
              ("bagging de 100 arbres", me.BaggingClassifier(n_estimators=100, random_state=0)),
              ("forêt aléatoire de 100 arbres", me.RandomForestClassifier(n_estimators=100, random_state=0)),
              ("ExtraTrees de 100 arbres", me.ExtraTreesClassifier(n_estimators=100, random_state=0))]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    for ax, (title, model) in zip(axes, models):
        model.fit(X, y)
        proba = model.predict_proba(G)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, proba, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
        ax.contour(xx, yy, proba, levels=[0.5], colors=[INK], linewidths=0.9)
        for c in range(2):
            ax.scatter(*X[y == c].T, s=9, color=(C[0], C[3])[c], edgecolor="white", linewidth=0.3)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "bagging_frontieres.png")


# ---------------------------------------------------------------------------- AdaBoost weights
def fig_adaboost_weights() -> None:
    rng = np.random.default_rng(1404)
    X = rng.uniform(0, 4, size=(40, 2))
    y = ((X[:, 0] > 1.6) ^ (X[:, 1] > 2.6)).astype(int)
    flip = rng.choice(40, size=3, replace=False)
    y[flip] = 1 - y[flip]
    ada = me.AdaBoostClassifier(n_estimators=3).fit(X, y)
    w = np.full(len(y), 1 / len(y))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for m, ax in enumerate(axes):
        for c in range(2):
            ax.scatter(*X[y == c].T, s=2200 * w[y == c], color=(C[0], C[3])[c], edgecolor=INK, linewidth=0.4,
                       alpha=0.85, label=f"classe {c}")
        stump = ada.estimators_[m]
        f, t = stump.root_.feature, stump.root_.threshold
        if f == 0:
            ax.axvline(t, color=INK, lw=1.4, ls="--")
        else:
            ax.axhline(t, color=INK, lw=1.4, ls="--")
        wrong = stump.predict(X) != y
        ax.set_title(f"tour {m + 1} : poids avant la souche {m + 1} ; erreur pondérée "
                     f"{fr(ada.estimator_errors_[m], 3)}", fontsize=9.5)
        ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(-0.2, 4.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        w = np.where(wrong, w * np.exp(ada.estimator_weights_[m]), w)
        w = w / w.sum()
    axes[0].legend(loc="lower left", fontsize=8, markerscale=0.4)
    fig.tight_layout()
    save(fig, "adaboost_poids.png")


# ---------------------------------------------------------------------------- gradient boosting
def fig_gradient_boosting() -> None:
    rng = np.random.default_rng(1405)
    x = np.sort(rng.uniform(0, 6, 120))
    y = np.sin(x) + 0.3 * x + 0.25 * rng.normal(size=120)
    X = x.reshape(-1, 1)
    grid = np.linspace(0, 6, 400).reshape(-1, 1)
    gb = me.GradientBoostingRegressor(n_estimators=50, learning_rate=0.3, max_depth=2).fit(X, y)
    stages = list(gb.staged_predict(grid))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.scatter(x, y, s=10, color=GREY, label="données")
    for m, color in zip((1, 5, 50), (C[1], C[0], C[3])):
        a.plot(grid[:, 0], stages[m - 1], lw=1.6, color=color, label=f"après {m} arbre{'s' if m > 1 else ''}")
    a.set_xlabel("x")
    a.set_ylabel("y")
    a.set_title("(a) la prédiction se construit arbre après arbre", fontsize=10.5)
    a.legend(fontsize=8.5)
    F5 = list(gb.staged_predict(X))[4]
    b.axhline(0, color=GREY, lw=0.8)
    b.scatter(x, y - F5, s=10, color=C[0])
    b.set_xlabel("x")
    b.set_ylabel("résidu y − F₅(x)")
    b.set_title("(b) les résidus après 5 arbres : la cible du sixième", fontsize=10.5)
    fig.tight_layout()
    save(fig, "gradient_boosting.png")


# ---------------------------------------------------------------------------- importances
def fig_importances() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(1406)
    n = 800
    X = rng.normal(size=(n, 5))
    y = (1.5 * X[:, 0] + X[:, 1] - 0.7 * X[:, 2] + 0.8 * rng.normal(size=n) > 0).astype(int)
    X[:, 3] = rng.integers(0, 3, size=n)                          # noise with 3 values (low cardinality)
    X[:, 4] = rng.permutation(n)                                  # an identifier: one value per row
    names = ["x₀", "x₁", "x₂", "bruit à 3 valeurs", "identifiant"]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.4, random_state=0)
    forest = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1).fit(X_tr, y_tr)
    perm = permutation_importance(forest, X_va, y_va, n_repeats=20, random_state=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    pos = np.arange(5)[::-1]
    a.barh(pos, forest.feature_importances_, color=C[0])
    a.set_title("(a) importance MDI (entraînement)", fontsize=10.5)
    a.set_yticks(pos)
    a.set_yticklabels(names)
    b.barh(pos, perm.importances_mean, xerr=perm.importances_std, color=C[3])
    b.axvline(0, color=GREY, lw=0.8)
    b.set_title("(b) importance par permutation (validation)", fontsize=10.5)
    a.set_xlabel("baisse d'impureté (normalisée)")
    b.set_xlabel("baisse de l'accuracy")
    fig.tight_layout()
    save(fig, "importances.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 14 -> {OUT.relative_to(ROOT)}")
    fig_votes()
    fig_bootstrap()
    fig_boundaries()
    fig_adaboost_weights()
    fig_gradient_boosting()
    fig_importances()
    return 0


if __name__ == "__main__":
    sys.exit(main())
