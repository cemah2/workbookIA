#!/usr/bin/env python
"""Draw the figures of the course sheet of chapter 7 (used by Claude).

    python tools/chapters/figures_ch07.py

Writes small PNG files into chapitres/ch07_classification/figures/. Every figure is computed
(never copied from the book): three kinds of class layouts (a line, a curve, an overlap), a map
of the probability of a class with three thresholds, one-versus-rest and one-versus-one with
linear binary classifiers on four classes, the regions of the nearest-centroid classifier,
Lloyd's algorithm step by step, three ways k-means fails, the silhouette of a point and of a
whole clustering, and the density of a fixed sample as the dimension grows. None of them uses
the data or the settings of an exercise (no two moons, no DBSCAN result, no ball-in-cube or
hyperorange curve, no nearest-neighbour distance: those are exercises 7.12, 7.18, 7.19 to 7.21).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch07_classification" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
FERT, UNFERT = C[1], C[0]          # orange circles: fertilised; blue squares: not fertilised


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


def sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)


def grid(xlim, ylim, n=300):
    xx, yy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def eggs(ax, X, y, size=30, legend=True) -> None:
    """Fertilised eggs as orange circles, the others as blue squares."""
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=size, marker="o", color=FERT, edgecolor="white", lw=0.8,
               label="fécondé", zorder=3)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=size * 0.85, marker="s", color=UNFERT, edgecolor="white", lw=0.8,
               label="non fécondé", zorder=3)
    if legend:
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, frameon=True)


# ---------------------------------------------------------------------------
def fig_boundaries() -> None:
    """A straight line, a curve, an overlap: three layouts of two classes of eggs."""
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), sharey=True)
    xlim, ylim = (49, 71), (51, 65)
    xx, yy, pts = grid(xlim, ylim, 250)
    region_cmap = ListedColormap(["#dceaf9", "#fbe0d2"])     # pale blue, pale orange

    # (a) a straight line separates the two clouds
    ax = axes[0]
    m1, m0 = np.array([56.5, 60.5]), np.array([63.5, 56.0])
    X1 = rng.normal(m1, [1.8, 1.2], (40, 2))
    X0 = rng.normal(m0, [1.8, 1.2], (40, 2))
    w, b = m1 - m0, -(m1 - m0) @ (m1 + m0) / 2          # perpendicular bisector of the two means
    X1, X0 = X1[X1 @ w + b > 1.5], X0[X0 @ w + b < -1.5]
    X, y = np.vstack([X1, X0]), np.r_[np.ones(len(X1)), np.zeros(len(X0))]
    margin = (pts @ w + b).reshape(xx.shape)
    ax.contourf(xx, yy, (margin > 0).astype(float), levels=[-0.5, 0.5, 1.5], cmap=region_cmap)
    ax.contour(xx, yy, margin, levels=[0], colors=[wbplot.TEXT], linewidths=1.8)
    eggs(ax, X, y)
    ax.set_title("(a) une droite suffit", fontsize=10.5, weight="normal")

    # (b) a curve: the class depends on which side of a wavy line the egg falls
    ax = axes[1]

    def wave(x):
        return 58 + 2.4 * np.sin((x - 50) / 2.6)

    P = np.column_stack([rng.uniform(*xlim, 400), rng.uniform(*ylim, 400)])
    gap = P[:, 1] - wave(P[:, 0])
    P, gap = P[np.abs(gap) > 0.9], gap[np.abs(gap) > 0.9]
    P, gap = P[:110], gap[:110]
    above = (yy > wave(xx)).astype(float)
    ax.contourf(xx, yy, above, levels=[-0.5, 0.5, 1.5], cmap=region_cmap)
    xs = np.linspace(*xlim, 400)
    ax.plot(xs, wave(xs), color=wbplot.TEXT, lw=1.8)
    eggs(ax, P, (gap > 0).astype(int), legend=False)
    ax.set_title("(b) il faut une courbe", fontsize=10.5, weight="normal")

    # (c) the two clouds overlap: no line and no curve separates them
    ax = axes[2]
    X1 = rng.normal([58.5, 59.0], [2.6, 1.9], (55, 2))
    X0 = rng.normal([61.5, 57.0], [2.6, 1.9], (55, 2))
    eggs(ax, np.vstack([X1, X0]), np.r_[np.ones(55), np.zeros(55)], legend=False)
    ax.set_title("(c) les classes se recouvrent :\naucune frontière n'est parfaite", fontsize=10.5,
                 weight="normal")
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("poids (g)")
        ax.grid(False)
    axes[0].set_ylabel("longueur (mm)")
    fig.suptitle("Trois situations pour deux classes d'œufs (données inventées) ; les régions colorées sont "
                 "celles où l'on répond « fécondé » (orange) ou « non fécondé » (bleu)", fontsize=10.5, y=1.02)
    save(fig, "frontieres.png")


# ---------------------------------------------------------------------------
def _gauss(points, mean, cov):
    diff = points - mean
    inv = np.linalg.inv(cov)
    return np.exp(-0.5 * np.einsum("ij,jk,ik->i", diff, inv, diff)) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def fig_threshold() -> None:
    """P(fertilised | weight, length) from two Gaussians and a prior; boundaries for three thresholds."""
    m1, S1 = np.array([59.0, 59.0]), np.array([[4.5, 1.2], [1.2, 3.0]])
    m0, S0 = np.array([62.5, 56.5]), np.array([[5.5, 0.6], [0.6, 3.6]])
    prior = 0.4
    rng = np.random.default_rng(3)
    n1, n0 = 48, 72
    X = np.vstack([rng.multivariate_normal(m1, S1, n1), rng.multivariate_normal(m0, S0, n0)])
    y = np.r_[np.ones(n1), np.zeros(n0)]
    xlim, ylim = (52, 70), (51, 64)
    xx, yy, pts = grid(xlim, ylim, 300)
    num = prior * _gauss(pts, m1, S1)
    post = (num / (num + (1 - prior) * _gauss(pts, m0, S0))).reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw=dict(width_ratios=[1, 1.18, 1]))
    ax = axes[0]
    eggs(ax, X, y)
    ax.set_title("(a) les œufs mirés : deux nuages qui se recouvrent", fontsize=10.2, weight="normal")
    ax = axes[1]
    cs = ax.contourf(xx, yy, post, levels=np.linspace(0, 1, 21), cmap=wbplot.diverging_cmap())
    fig.colorbar(cs, ax=ax, label="P(fécondé | poids, longueur)", ticks=[0, 0.25, 0.5, 0.75, 1],
                 format=FuncFormatter(french_number))
    ax.set_title("(b) la probabilité d'être fécondé :\nplus c'est rouge, plus c'est probable", fontsize=10.2,
                 weight="normal")
    ax = axes[2]
    ax.contourf(xx, yy, post, levels=np.linspace(0, 1, 21), cmap=wbplot.diverging_cmap(), alpha=0.35)
    styles = [(0.2, "--", "seuil 0,2"), (0.5, "-", "seuil 0,5"), (0.8, ":", "seuil 0,8")]
    for level, style, _ in styles:
        ax.contour(xx, yy, post, levels=[level], colors=[wbplot.TEXT], linewidths=1.8, linestyles=[style])
    eggs(ax, X, y, size=18, legend=False)
    handles = [Line2D([], [], color=wbplot.TEXT, lw=1.8, ls=s, label=lab) for _, s, lab in styles]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9)
    ax.set_title("(c) trois politiques : on déclare « fécondé »\nquand la probabilité dépasse le seuil",
                 fontsize=10.2, weight="normal")
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("poids (g)")
        ax.grid(False)
    axes[0].set_ylabel("longueur (mm)")
    fig.suptitle("Classes qui se recouvrent : une probabilité par point, puis une frontière choisie par un seuil "
                 f"(deux lois normales, a priori {fr(prior, 1)} pour « fécondé »)", fontsize=10.5, y=1.03)
    save(fig, "seuil.png")


# ---------------------------------------------------------------------------
def _four_classes():
    rng = np.random.default_rng(11)
    means = np.array([[-2.6, 1.4], [1.2, 2.7], [2.8, -1.2], [-0.8, -2.6]])
    X = np.vstack([rng.normal(m, 0.85, (45, 2)) for m in means])
    y = np.repeat(np.arange(4), 45)
    return X, y, means


def fig_ovr_ovo() -> None:
    """One-versus-rest and one-versus-one with four classes and linear binary classifiers."""
    from sklearn.linear_model import LogisticRegression

    X, y, means = _four_classes()
    names = "ABCD"
    colors = C[:4]
    cmap = ListedColormap(colors)
    xlim, ylim = (-5.5, 5.5), (-5.2, 5.2)
    xx, yy, pts = grid(xlim, ylim, 400)

    ovr = [LogisticRegression(C=1.0).fit(X, (y == k).astype(int)) for k in range(4)]
    scores = np.column_stack([m.decision_function(pts) for m in ovr])
    ovr_pred = scores.argmax(axis=1)
    n_yes = (scores > 0).sum(axis=1)

    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    votes = np.zeros((len(pts), 4), dtype=int)
    duels = []
    for i, j in pairs:
        keep = (y == i) | (y == j)
        model = LogisticRegression(C=1.0).fit(X[keep], (y[keep] == j).astype(int))
        duels.append(model)
        win_j = model.predict(pts) == 1
        votes[win_j, j] += 1
        votes[~win_j, i] += 1
    ovo_pred = votes.argmax(axis=1)
    # a test point where no one-versus-rest classifier says "yes"
    candidates = np.where(n_yes == 0)[0]
    probe = pts[candidates[np.argmin(np.linalg.norm(pts[candidates] - np.array([-1.3, -0.3]), axis=1))]]
    probe_scores = np.array([m.decision_function(probe[None])[0] for m in ovr])
    probe_votes = np.zeros(4, dtype=int)
    for (i, j), model in zip(pairs, duels):
        probe_votes[j if model.predict(probe[None])[0] == 1 else i] += 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    ax = axes[0]
    for k in range(4):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=22, color=colors[k], edgecolor="white", lw=0.7, zorder=3)
        ax.text(*means[k], names[k], fontsize=16, weight="bold", ha="center", va="center", color="white",
                zorder=4, bbox=dict(boxstyle="circle,pad=0.25", fc=colors[k], ec="white"))
    ax.set_title("(a) quatre classes, A à D", fontsize=10.5, weight="normal")

    ax = axes[1]
    ax.contourf(xx, yy, ovr_pred.reshape(xx.shape), levels=np.arange(5) - 0.5, cmap=cmap, alpha=0.22)
    nobody = (n_yes == 0).reshape(xx.shape)
    several = (n_yes >= 2).reshape(xx.shape)
    ax.contourf(xx, yy, nobody, levels=[0.5, 1.5], colors="none", hatches=["...."])
    ax.contourf(xx, yy, several, levels=[0.5, 1.5], colors="none", hatches=["////"])
    for k, model in enumerate(ovr):
        ax.contour(xx, yy, model.decision_function(pts).reshape(xx.shape), levels=[0], colors=[colors[k]],
                   linewidths=1.6, linestyles="--")
    for k in range(4):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=10, color=colors[k], edgecolor="none", zorder=3, alpha=0.8)
    handles = [Line2D([], [], color=wbplot.TEXT_MUTED, ls="--", lw=1.4, label="frontière « k contre le reste »"),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec=wbplot.TEXT_MUTED, hatch="....",
                             label="aucun classifieur ne dit « oui »"),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec=wbplot.TEXT_MUTED, hatch="////",
                             label="plusieurs disent « oui »")]
    ax.legend(handles=handles, loc="lower left", fontsize=7.8, frameon=True, framealpha=0.92)
    text = "\n".join(f"score {names[k]} : {fr(probe_scores[k], 1)}" for k in range(4))
    ax.scatter(*probe, s=170, marker="*", color=wbplot.TEXT, edgecolor="white", zorder=6)
    ax.annotate(text + f"\n→ {names[int(probe_scores.argmax())]}", xy=probe, xytext=(-5.2, 4.9), fontsize=8.3,
                va="top", bbox=dict(boxstyle="round", fc="white", ec=wbplot.GRID),
                arrowprops=dict(arrowstyle="->", color=wbplot.TEXT_MUTED))
    ax.set_title("(b) un-contre-tous : 4 classifieurs ;\nla couleur est celle du plus grand score", fontsize=10.5,
                 weight="normal")

    ax = axes[2]
    ax.contourf(xx, yy, ovo_pred.reshape(xx.shape), levels=np.arange(5) - 0.5, cmap=cmap, alpha=0.22)
    for model in duels:
        ax.contour(xx, yy, model.decision_function(pts).reshape(xx.shape), levels=[0], colors=[wbplot.TEXT_MUTED],
                   linewidths=0.9)
    for k in range(4):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=10, color=colors[k], edgecolor="none", zorder=3, alpha=0.8)
    handles = [Line2D([], [], color=wbplot.TEXT_MUTED, lw=1.0, label="les 6 frontières des duels")]
    ax.legend(handles=handles, loc="lower left", fontsize=7.8, frameon=True, framealpha=0.92)
    text = "\n".join(f"{names[k]} : {probe_votes[k]} voix" for k in range(4))
    ax.scatter(*probe, s=170, marker="*", color=wbplot.TEXT, edgecolor="white", zorder=6)
    ax.annotate(text + f"\n→ {names[int(probe_votes.argmax())]}", xy=probe, xytext=(-5.2, 4.9), fontsize=8.3,
                va="top", bbox=dict(boxstyle="round", fc="white", ec=wbplot.GRID),
                arrowprops=dict(arrowstyle="->", color=wbplot.TEXT_MUTED))
    ax.set_title("(c) un-contre-un : 6 duels ;\nla couleur est celle du vainqueur des votes", fontsize=10.5,
                 weight="normal")
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Deux façons de classer quatre classes avec des classifieurs binaires (ici, chacun trace une "
                 "droite)", fontsize=10.5, y=1.01)
    save(fig, "ovr_ovo.png")
    print(f"    point test {probe.round(2)} : scores {probe_scores.round(2)}, votes {probe_votes}")


# ---------------------------------------------------------------------------
def fig_centroid() -> None:
    """Nearest-centroid regions for five labelled groups; a spread-out class loses points to a tight one."""
    rng = np.random.default_rng(5)
    specs = [((-3.0, 2.6), 0.7, 35), ((1.0, 3.2), 0.6, 35), ((3.4, 0.2), 0.5, 30), ((-1.0, -0.6), 1.5, 50),
             ((2.4, -3.0), 0.7, 35)]
    X = np.vstack([rng.normal(m, s, (n, 2)) for m, s, n in specs])
    y = np.concatenate([np.full(n, k) for k, (_, _, n) in enumerate(specs)])
    centroids = np.array([X[y == k].mean(axis=0) for k in range(5)])
    colors = C[:5]
    xlim, ylim = (-6.2, 6.2), (-5.6, 5.8)
    xx, yy, pts = grid(xlim, ylim, 400)
    region = sq_dist(pts, centroids).argmin(axis=1).reshape(xx.shape)
    pred = sq_dist(X, centroids).argmin(axis=1)
    wrong = pred != y

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax in axes:
        if ax is axes[1]:
            ax.contourf(xx, yy, region, levels=np.arange(6) - 0.5, cmap=ListedColormap(colors), alpha=0.22)
            ax.contour(xx, yy, region, levels=np.arange(5) + 0.5, colors=[wbplot.TEXT_MUTED], linewidths=1.0)
        for k in range(5):
            ax.scatter(X[y == k, 0], X[y == k, 1], s=18, color=colors[k], edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(centroids[:, 0], centroids[:, 1], s=190, marker="X", color=wbplot.TEXT, edgecolor="white",
                   lw=1.5, zorder=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[1].scatter(X[wrong, 0], X[wrong, 1], s=110, facecolor="none", edgecolor=wbplot.BAD, lw=1.6, zorder=4)
    axes[0].set_title("(a) cinq groupes étiquetés et leurs centroïdes (×)", fontsize=10.5, weight="normal")
    axes[1].set_title(f"(b) chaque point du plan prend la classe du centroïde le plus proche ;\n"
                      f"{wrong.sum()} points d'entraînement entourés tombent dans une autre région",
                      fontsize=10.5, weight="normal")
    save(fig, "centroide.png")


# ---------------------------------------------------------------------------
def _lloyd_history(X, centres, n_steps=20):
    """Centres and labels after each Lloyd iteration (assignment then update), until nothing moves."""
    history = [(centres.copy(), sq_dist(X, centres).argmin(axis=1))]
    for _ in range(n_steps):
        labels = sq_dist(X, centres).argmin(axis=1)
        new = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centres[j]
                        for j in range(len(centres))])
        history.append((new, labels))
        if np.allclose(new, centres):
            break
        centres = new
    return history


def _inertia(X, centres):
    return float(sq_dist(X, centres).min(axis=1).sum())


def fig_lloyd() -> None:
    """Lloyd's algorithm on three groups, from three badly placed starting centres."""
    rng = np.random.default_rng(21)
    X = np.vstack([rng.normal(m, 0.75, (50, 2)) for m in [(0.0, 0.0), (4.6, 1.0), (1.8, 4.4)]])
    start = X[[np.argmin(X[:, 0] + X[:, 1]), np.argmin(X[:, 0] - 0.2 * X[:, 1]), np.argmin(X[:, 1] - 0.3 * X[:, 0])]]
    history = _lloyd_history(X, start)
    last = len(history) - 1
    steps = [0, 1, 2, last]
    colors = [C[0], C[1], C[2]]
    xlim, ylim = (-2.6, 7.0), (-2.6, 6.8)
    xx, yy, pts = grid(xlim, ylim, 300)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, step in zip(axes, steps):
        if step == 0:   # the starting centres and the regions they will use for the first assignment
            centres = history[0][0]
            ax.scatter(X[:, 0], X[:, 1], s=14, color=wbplot.TEXT_MUTED, edgecolor="none", alpha=0.6)
            title = "départ\n3 centres pris parmi les points"
            used = centres
        else:          # colours: assignment to the previous centres; arrows: move to the means
            used, (centres, labels) = history[step - 1][0], history[step]
            for j in range(3):
                ax.scatter(X[labels == j, 0], X[labels == j, 1], s=14, color=colors[j], edgecolor="none",
                           alpha=0.85)
            moved = False
            for a, b in zip(used, centres):
                if np.linalg.norm(b - a) > 0.05:
                    moved = True
                    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=wbplot.TEXT, lw=1.5))
            ax.scatter(used[:, 0], used[:, 1], s=120, marker="X", c=colors, alpha=0.3, edgecolor="none", zorder=4)
            what = "affecter, puis déplacer les centres" if moved else "plus rien ne bouge : c'est fini"
            j_assign = float(((X - used[labels]) ** 2).sum())          # after the assignment (old centres)
            j_update = float(((X - centres[labels]) ** 2).sum())       # after the update (same assignment)
            title = f"itération {step} : J = {fr(j_assign, 0)} → {fr(j_update, 0)}\n{what}"
        region = sq_dist(pts, used).argmin(axis=1).reshape(xx.shape)
        ax.contour(xx, yy, region, levels=[0.5, 1.5], colors=[wbplot.TEXT_MUTED], linewidths=0.9)
        ax.scatter(centres[:, 0], centres[:, 1], s=170, marker="X", c=colors, edgecolor=wbplot.TEXT, lw=1.3,
                   zorder=5)
        ax.set_title(title, fontsize=9.8, weight="normal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("L'algorithme de Lloyd (k = 3) : chaque point va au centre le plus proche (couleurs), puis chaque "
                 "centre va à la moyenne de ses points (flèches)\nLignes grises : les frontières entre les régions des "
                 "centres utilisés pour l'affectation (en pâle à partir de l'itération 1)\nJ, l'inertie : après "
                 "l'affectation → après la mise à jour ; elle ne remonte jamais", fontsize=10.2, y=1.12)
    save(fig, "kmeans_lloyd.png")


# ---------------------------------------------------------------------------
def fig_kmeans_limits() -> None:
    """Elongated groups, groups of very different spread, and a local minimum from a bad start."""
    from sklearn.cluster import KMeans

    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    def show(ax, X, truth, labels, centres, size=16):
        for j in range(labels.max() + 1):
            for t in range(truth.max() + 1):
                sel = (labels == j) & (truth == t)
                ax.scatter(X[sel, 0], X[sel, 1], s=size, marker=markers[t], color=C[j], edgecolor="none", alpha=0.85)
        ax.scatter(centres[:, 0], centres[:, 1], s=170, marker="X", c=[C[j] for j in range(len(centres))],
                   edgecolor=wbplot.TEXT, lw=1.3, zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect("equal", adjustable="datalim")

    # (a) three elongated, parallel groups
    rng = np.random.default_rng(2)
    base = np.vstack([rng.normal(m, 0.5, (90, 2)) for m in [(0, 0), (0, 3.0), (0, 6.0)]])   # gaps of 6 sigma
    base[:, 0] *= 5.0
    X = base @ np.array([[0.9, 0.35], [-0.35, 0.9]])
    truth = np.repeat(np.arange(3), 90)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    show(axes[0], X, truth, km.labels_, km.cluster_centers_, size=24)
    axes[0].set_title("(a) trois groupes allongés et parallèles :\nk-means les coupe en travers", fontsize=10.2,
                      weight="normal")

    # (b) one wide group and two tight ones next to it
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal((0, 0), 1.9, (240, 2)), rng.normal((4.6, 1.2), 0.35, (40, 2)),
                   rng.normal((4.4, -1.6), 0.35, (40, 2))])
    truth = np.r_[np.zeros(240, int), np.ones(40, int), np.full(40, 2)]
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    show(axes[1], X, truth, km.labels_, km.cluster_centers_)
    axes[1].set_title("(b) un groupe très étalé, deux groupes serrés :\nk-means coupe le grand et réunit les "
                      "petits", fontsize=10.2, weight="normal")

    # (c) four clear groups, but a start with two centres in the same group
    rng = np.random.default_rng(6)
    means = np.array([(0, 0), (5, 0), (0, 5), (5, 5)], dtype=float)
    X = np.vstack([rng.normal(m, 0.6, (60, 2)) for m in means])
    truth = np.repeat(np.arange(4), 60)
    init = np.array([[-0.4, -0.3], [0.5, 0.4], [5.2, 0.2], [2.5, 5.1]])
    bad = KMeans(n_clusters=4, init=init, n_init=1).fit(X)
    best = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    show(axes[2], X, truth, bad.labels_, bad.cluster_centers_)
    axes[2].set_title(f"(c) un mauvais départ : minimum local, J = {fr(bad.inertia_, 0)}\n"
                      f"(meilleur des 10 départs de n_init : J = {fr(best.inertia_, 0)})", fontsize=10.2,
                      weight="normal")
    fig.suptitle("Trois échecs de k-means : la couleur est le cluster trouvé, la forme du marqueur le vrai groupe",
                 fontsize=10.5, y=1.02)
    save(fig, "kmeans_limites.png")


# ---------------------------------------------------------------------------
def fig_silhouette() -> None:
    """a(i) and b(i) for one point, then the silhouette of every point of a clustering."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples

    rng = np.random.default_rng(9)
    means = np.array([(0.0, 0.0), (3.6, 0.6), (1.0, 4.2)])
    X = np.vstack([rng.normal(m, s, (n, 2)) for m, s, n in zip(means, [0.75, 0.65, 0.8], [30, 25, 28])])
    labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X).labels_
    s = silhouette_samples(X, labels)
    own = labels[np.argmin(np.linalg.norm(X - means[0], axis=1))]
    candidates = np.where(labels == own)[0]
    i = candidates[np.argmax(X[candidates, 0])]            # the point of the first group closest to the second
    dist = np.linalg.norm(X - X[i], axis=1)
    others = [j for j in range(3) if j != own]
    mean_to = {j: dist[labels == j].mean() for j in others}
    near = min(mean_to, key=mean_to.get)
    a = dist[(labels == own) & (np.arange(len(X)) != i)].mean()
    b = mean_to[near]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    for j in range(3):
        ax.scatter(X[labels == j, 0], X[labels == j, 1], s=20, color=C[j], edgecolor="white", lw=0.6, zorder=3)
    for k in np.where(labels == own)[0]:
        if k != i:
            ax.plot([X[i, 0], X[k, 0]], [X[i, 1], X[k, 1]], color=C[own], lw=0.6, alpha=0.6)
    for k in np.where(labels == near)[0]:
        ax.plot([X[i, 0], X[k, 0]], [X[i, 1], X[k, 1]], color=C[near], lw=0.6, alpha=0.6, ls="--")
    ax.scatter(*X[i], s=160, color=wbplot.TEXT, marker="*", zorder=5)
    ax.annotate(f"point i\na(i) = {fr(a, 2)} (son cluster)\nb(i) = {fr(b, 2)} (le cluster voisin)\n"
                f"s(i) = (b − a) / max(a, b) = {fr((b - a) / max(a, b), 2)}",
                xy=X[i], xytext=(0.99, 0.80), textcoords="axes fraction", fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round", fc="white", ec=wbplot.GRID),
                arrowprops=dict(arrowstyle="->", color=wbplot.TEXT_MUTED))
    ax.set_title("(a) distances moyennes du point i à son cluster (traits pleins)\net au cluster voisin le plus "
                 "proche (tirets)", fontsize=10.2, weight="normal")
    ax = axes[1]
    sc = ax.scatter(X[:, 0], X[:, 1], c=s, cmap=wbplot.diverging_cmap().reversed(), vmin=-1, vmax=1, s=46,
                    edgecolor=wbplot.TEXT_MUTED, lw=0.4, zorder=3)
    fig.colorbar(sc, ax=ax, label="silhouette s(i)", format=FuncFormatter(french_number))
    ax.set_title(f"(b) la silhouette de chaque point ; moyenne : {fr(s.mean(), 2)}\n(bleu foncé : bien placé ; "
                 "pâle : à la frontière ; rouge : mal placé)", fontsize=10.2, weight="normal")
    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, "silhouette.png")


# ---------------------------------------------------------------------------
def fig_density() -> None:
    """Density of 1000 samples as the dimension grows, and samples needed for one per bin."""
    d = np.arange(1, 13)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    for colour, b in zip([C[2], C[0], C[1]], [3, 5, 8]):
        ax.plot(d[:10], 1000 / b ** d[:10].astype(float), "o-", color=colour, ms=4,
                label=f"{b} cases par axe ({b}$^d$ cases)")
    ax.axhline(1, color=wbplot.TEXT_MUTED, ls="--", lw=1.1)
    ax.text(10.3, 1.5, "1 échantillon par case\nen moyenne", ha="right", va="bottom", fontsize=8.5,
            color=wbplot.TEXT_MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("nombre de features d")
    ax.set_ylabel("densité n / bᵈ (échelle log)")
    ax.legend(fontsize=8.5)
    ax.set_title("1 000 échantillons : la densité s'effondre\nquand on ajoute des features", fontsize=10.2,
                 weight="normal")
    ax = axes[1]
    for colour, b in zip([C[2], C[0], C[1]], [3, 5, 8]):
        ax.plot(d, b ** d.astype(float), "o-", color=colour, ms=4, label=f"{b} cases par axe")
    for value, text in [(7e4, "MNIST : 70 000 images"), (8e9, "≈ 8 milliards d'humains")]:
        ax.axhline(value, color=wbplot.TEXT_MUTED, ls=":", lw=1.1)
        ax.text(1, value * 2.2, text, fontsize=8.5, color=wbplot.TEXT_MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("nombre de features d")
    ax.set_ylabel("échantillons nécessaires (échelle log)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Échantillons nécessaires pour une densité de 1 :\nn = bᵈ croît exponentiellement", fontsize=10.2,
                 weight="normal")
    axes[0].set_xticks(range(1, 11))
    axes[0].set_xlim(0.5, 10.5)
    axes[1].set_xticks(range(1, 13))
    save(fig, "densite.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    fig_boundaries()
    fig_threshold()
    fig_ovr_ovo()
    fig_centroid()
    fig_lloyd()
    fig_kmeans_limits()
    fig_silhouette()
    fig_density()
    return 0


if __name__ == "__main__":
    sys.exit(main())
