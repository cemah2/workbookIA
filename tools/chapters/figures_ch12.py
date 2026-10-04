#!/usr/bin/env python
"""Draw the figures of chapter 12 (used by Claude).

    python tools/chapters/figures_ch12.py

Writes small PNG files into chapitres/ch12_preparation/figures/. Every figure is computed or drawn
here (never copied from the book): the golden rule of preparation as a flow chart, a crescent-shaped
cloud before and after min-max scaling and standardisation, one outlier against three scalers, the
univariate and the multivariate min-max of three feature ranges, the projection of a correlated
cloud on the horizontal axis and on its principal axis, the explained variance of a synthetic
dataset, whitening, the "eigen-digits" of scikit-learn's small 8x8 digits (load_digits, shipped with
scikit-learn) and their reconstructions, and the same digits seen by PCA, t-SNE and UMAP. None of them
uses the data of an exercise (no penguins, no California, no MNIST, no traffic): the notebook
exercises 12.11 to 12.33 and the paper exercises use other data and other numbers.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

from mylearn_ref import preprocessing as pp  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch12_preparation" / "figures"
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


# ---------------------------------------------------------------------------- the golden rule
def box(ax, xy, w, h, text, face, edge=INK, size=9.5, weight="normal"):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=face, ec=edge, lw=1.2)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight=weight, wrap=True)


def arrow(ax, start, end, text="", color=INK, style="-|>", ls="-", text_offset=(0, 0.07), size=9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, lw=1.4, color=color, ls=ls))
    if text:
        mid = ((start[0] + end[0]) / 2 + text_offset[0], (start[1] + end[1]) / 2 + text_offset[1])
        ax.text(*mid, text, ha="center", va="bottom", fontsize=size, color=color, style="italic")


def fig_golden_rule() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    light, param, model, new = "#e8f1fc", "#fdf1d6", "#e6f4e6", "#f6e5e5"
    box(ax, (0.2, 2.75), 2.2, 1.0, "Données\nd'entraînement", light, weight="bold")
    box(ax, (3.6, 2.75), 2.6, 1.0, "Paramètres appris\n(min, max, μ, σ,\ncatégories, axes…)", param)
    box(ax, (7.4, 2.75), 1.9, 1.0, "Entraînement\ntransformé", light)
    box(ax, (9.9, 2.75), 1.4, 1.0, "Entraîner\nle modèle", model)
    arrow(ax, (2.4, 3.25), (3.6, 3.25), "fit")
    arrow(ax, (6.2, 3.25), (7.4, 3.25), "transform")
    arrow(ax, (9.3, 3.25), (9.9, 3.25))
    box(ax, (0.2, 0.45), 2.2, 1.1, "Validation, test,\nproduction", new, weight="bold")
    box(ax, (7.4, 0.45), 1.9, 1.1, "Données\ntransformées", new)
    box(ax, (9.9, 0.45), 1.4, 1.1, "Prédire", model)
    arrow(ax, (2.4, 1.0), (7.4, 1.0), "transform avec les MÊMES paramètres (jamais de fit)", text_offset=(0, 0.06))
    arrow(ax, (4.9, 2.75), (4.9, 1.42), color=C[1], ls="--")
    arrow(ax, (9.3, 1.0), (9.9, 1.0))
    arrow(ax, (10.6, 2.75), (10.6, 1.55), "modèle\nentraîné", text_offset=(0.43, -0.25), size=8.5)
    ax.text(4.98, 2.0, "réutilisés", color=C[1], fontsize=9, style="italic", ha="left")
    save(fig, "regle_or.png")


# ---------------------------------------------------------------------------- scaling a 2-D cloud
def crescent(seed=1201, n=160):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.15 * np.pi, 0.85 * np.pi, n)
    radius = 1 + rng.normal(0, 0.09, n)
    x = 200 + 150 * radius * np.cos(theta)          # a duration in seconds, say
    y = 0.15 + 1.5 * radius * np.sin(theta) - 0.6   # a dimensionless index
    return np.column_stack([x, y]), theta


def fig_scaling() -> None:
    X, theta = crescent()
    Xm = pp.MinMaxScaler().fit_transform(X)
    Xs = pp.StandardScaler().fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    kw = dict(c=theta, cmap="viridis", s=14, edgecolor="none")
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], **kw)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("(a) Brutes, même échelle sur les deux axes", fontsize=10.5)
    ax.set_xlabel("feature 1 (d'environ 50 à 340)")
    ax.set_ylabel("feature 2 (d'environ 0,2 à 1,3)")
    axes[1].scatter(Xm[:, 0], Xm[:, 1], **kw)
    axes[1].set_title("(b) Min-max vers [0 ; 1] (par feature)", fontsize=10.5)
    axes[1].set_xlim(-0.15, 1.15)
    axes[1].set_ylim(-0.15, 1.15)
    for v in (0, 1):
        axes[1].axhline(v, color=GREY, lw=0.8, ls=":")
        axes[1].axvline(v, color=GREY, lw=0.8, ls=":")
    axes[2].scatter(Xs[:, 0], Xs[:, 1], **kw)
    axes[2].set_title("(c) Standardisée (moyenne 0, écart-type 1)", fontsize=10.5)
    axes[2].axhline(0, color=GREY, lw=0.8)
    axes[2].axvline(0, color=GREY, lw=0.8)
    axes[2].set_xlim(-2.4, 2.4)
    axes[2].set_ylim(-2.4, 2.4)
    for ax in axes[1:]:
        ax.set_aspect("equal")
        ax.set_xlabel("feature 1 transformée")
    axes[1].set_ylabel("feature 2 transformée")
    fig.tight_layout()
    save(fig, "mises_echelle.png")


# ---------------------------------------------------------------------------- one outlier
def fig_outlier() -> None:
    rng = np.random.default_rng(1202)
    x = np.append(rng.normal(50, 5, 40), 400.0)
    minmax = pp.MinMaxScaler().fit_transform(x.reshape(-1, 1)).ravel()
    standard = pp.StandardScaler().fit_transform(x.reshape(-1, 1)).ravel()
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    robust = (x - med) / (q3 - q1)
    rows = [("min-max", minmax), ("standardisation", standard), ("robuste : (x − médiane) / IQR", robust)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 3.9))
    for ax, (name, z) in zip(axes, rows):
        inl, out = z[:-1], z[-1]
        ax.scatter(inl, np.zeros_like(inl), s=22, color=C[0], alpha=0.7, edgecolor="none", label="40 valeurs ordinaires")
        ax.scatter([out], [0], s=60, color=C[1], marker="D", label="le point aberrant")
        ax.set_yticks([])
        ax.spines[["left", "top", "right"]].set_visible(False)
        span = f"[{fr(inl.min(), 2)} ; {fr(inl.max(), 2)}]"
        ax.set_title(f"{name} : les valeurs ordinaires occupent {span}, le point aberrant tombe à {fr(out, 1)}",
                     fontsize=9.5, loc="left")
    axes[0].legend(loc="center", fontsize=8.5, frameon=False)
    fig.tight_layout(h_pad=1.2)
    save(fig, "point_aberrant.png")


# ---------------------------------------------------------------------------- univariate vs multivariate
def fig_uni_multi() -> None:
    ranges = np.array([[2.0, 8.0], [-5.0, 1.0], [10.0, 30.0]])
    names = ["feature 1", "feature 2", "feature 3"]
    lo, hi = ranges.min(), ranges.max()
    panels = [("(a) Plages d'origine", ranges, (-8, 33)),
              ("(b) Univariée : chaque feature vers [0 ; 1]", np.tile([0.0, 1.0], (3, 1)), (-0.1, 1.1)),
              ("(c) Multivariée : toutes ensemble vers [0 ; 1]", (ranges - lo) / (hi - lo), (-0.1, 1.1))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 2.9))
    for ax, (title, bars, xlim) in zip(axes, panels):
        for i, ((a, b), color) in enumerate(zip(bars, C[:3])):
            ax.barh(i, b - a, left=a, height=0.5, color=color, alpha=0.85)
            ax.text(a, i + 0.33, fr(a, 2 if xlim[1] < 2 else 0), ha="center", fontsize=8.5)
            ax.text(b, i + 0.33, fr(b, 2 if xlim[1] < 2 else 0), ha="center", fontsize=8.5)
        ax.set_yticks(range(3), names)
        ax.set_ylim(-0.6, 2.7)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=10.5, loc="left")
    fig.tight_layout()
    save(fig, "univarie_multivarie.png")


# ---------------------------------------------------------------------------- projections
def correlated_cloud(seed, n=120, rho=0.8):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal([0, 0], cov, size=n)


def fig_projection() -> None:
    X = pp.StandardScaler().fit_transform(correlated_cloud(1203))
    order = np.argsort(X[:, 0] + X[:, 1])
    colors = plt.cm.viridis(np.linspace(0, 1, len(X)))[np.argsort(order)]
    pca = pp.PCA(n_components=1).fit(X)
    u = pca.components_[0]
    t_axis = X[:, 0]
    t_pc = (X - pca.mean_) @ u
    shown = np.random.default_rng(3).choice(len(X), 30, replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, title in zip(axes[:2], ["(a) Projection sur l'axe horizontal", "(b) Projection sur l'axe principal"]):
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=14, edgecolor="none")
        ax.set_aspect("equal")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("feature 1 standardisée")
    axes[0].set_ylabel("feature 2 standardisée")
    axes[0].axhline(0, color=INK, lw=1.6)
    for i in shown:
        axes[0].plot([X[i, 0], X[i, 0]], [X[i, 1], 0], color=GREY, lw=0.7)
    axes[0].scatter(t_axis, np.zeros_like(t_axis), c=colors, s=10, marker="|")
    line = np.outer([-4, 4], u)
    axes[1].plot(line[:, 0], line[:, 1], color=INK, lw=1.6)
    feet = np.outer(t_pc, u)
    for i in shown:
        axes[1].plot([X[i, 0], feet[i, 0]], [X[i, 1], feet[i, 1]], color=GREY, lw=0.7)
    axes[1].scatter(feet[:, 0], feet[:, 1], c=colors, s=10, marker="o")
    ax = axes[2]
    ax.scatter(t_axis, np.full_like(t_axis, 1.0), c=colors, s=22, marker="|")
    ax.scatter(t_pc, np.zeros_like(t_pc), c=colors, s=22, marker="|")
    ax.set_yticks([0, 1], [f"sur l'axe principal\nvariance {fr(np.var(t_pc), 2)}",
                           f"sur l'axe horizontal\nvariance {fr(np.var(t_axis), 2)}"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-3.6, 3.6)
    ax.set_title("(c) Les deux nuages à une dimension", fontsize=10.5)
    ax.set_xlabel("coordonnée sur la droite de projection")
    fig.tight_layout()
    save(fig, "projection.png")


# ---------------------------------------------------------------------------- explained variance
def fig_explained() -> None:
    rng = np.random.default_rng(1204)
    n, p = 400, 10
    latent = rng.normal(size=(n, 3)) * np.array([3.0, 2.0, 1.2])
    mixing, _ = np.linalg.qr(rng.normal(size=(p, 3)))
    X = latent @ mixing.T + rng.normal(0, 0.45, size=(n, p)) + rng.normal(0, 5, p)
    pca = pp.PCA().fit(X)
    ratio = pca.explained_variance_ratio_
    k = np.arange(1, p + 1)
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.bar(k, ratio, color=C[0], alpha=0.85, label="part de chaque composante")
    ax.plot(k, np.cumsum(ratio), color=C[1], marker="o", lw=1.6, label="part cumulée")
    ax.axhline(0.9, color=GREY, ls="--", lw=1)
    ax.text(10.4, 0.9, "90 %", va="center", fontsize=9, color=wbplot.TEXT_MUTED)
    ax.set_xticks(k)
    ax.set_xlabel("composante principale")
    ax.set_ylabel("part de la variance totale")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("10 features mesurées, 3 directions qui comptent", fontsize=10.5)
    fig.tight_layout()
    save(fig, "variance_expliquee.png")


# ---------------------------------------------------------------------------- whitening
def fig_whitening() -> None:
    rng = np.random.default_rng(1205)
    A = np.array([[2.2, 0.0], [1.3, 0.6]])
    X = rng.normal(size=(150, 2)) @ A.T + np.array([3.0, -1.0])
    pca = pp.PCA().fit(X)
    Z = pca.transform(X)
    W = pp.PCA(whiten=True).fit_transform(X)
    hue = plt.cm.viridis((Z[:, 0] - Z[:, 0].min()) / np.ptp(Z[:, 0]))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    axes[0].scatter(X[:, 0], X[:, 1], c=hue, s=14, edgecolor="none")
    for vec, lam, name in zip(pca.components_, pca.explained_variance_, ("1", "2")):
        end = pca.mean_ + 2 * np.sqrt(lam) * vec
        axes[0].add_patch(FancyArrowPatch(pca.mean_, end, arrowstyle="-|>", mutation_scale=14, lw=1.8, color=INK))
        axes[0].text(*(end + 0.25 * vec), f"axe {name}", fontsize=9)
    axes[0].set_title("(a) Données, et leurs deux axes principaux", fontsize=10.5)
    axes[1].scatter(Z[:, 0], Z[:, 1], c=hue, s=14, edgecolor="none")
    axes[1].set_title(f"(b) Coordonnées PCA : variances {fr(pca.explained_variance_[0], 2)} et "
                      f"{fr(pca.explained_variance_[1], 2)}", fontsize=10.5)
    axes[2].scatter(W[:, 0], W[:, 1], c=hue, s=14, edgecolor="none")
    axes[2].set_title("(c) Après whitening : variance 1 sur chaque axe", fontsize=10.5)
    lim = max(np.abs(Z).max(), np.abs(W).max()) * 1.1
    for ax in axes[1:]:
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("composante 1")
    axes[1].set_ylabel("composante 2")
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "blanchiment.png")


# ---------------------------------------------------------------------------- eigen-digits
def small_digits():
    from sklearn.datasets import load_digits

    digits = load_digits()
    return digits.data.astype(float), digits.target


def fig_digits() -> None:
    X, y = small_digits()
    train = np.arange(len(X)) < 1500                  # the PCA never sees the two digits it rebuilds
    pca = pp.PCA().fit(X[train])
    held = np.flatnonzero(~train)
    picks = [int(held[y[held] == 3][0]), int(held[y[held] == 7][0])]
    Z = pca.transform(X)
    ks = [1, 3, 5, 10, 20, 40]
    fig, axes = plt.subplots(3, 7, figsize=(10.5, 5.8))
    axes[0, 0].imshow(pca.mean_.reshape(8, 8), cmap="gray_r")
    axes[0, 0].set_title("moyenne", fontsize=9)
    for j in range(6):
        comp = pca.components_[j].reshape(8, 8)
        lim = np.abs(comp).max()
        axes[0, j + 1].imshow(comp, cmap=wbplot.diverging_cmap(), vmin=-lim, vmax=lim)
        axes[0, j + 1].set_title(f"composante {j + 1}", fontsize=9)
    for row, idx in zip((1, 2), picks):
        for col, k in enumerate(ks):
            back = Z[idx, :k] @ pca.components_[:k] + pca.mean_
            axes[row, col].imshow(back.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
            axes[row, col].set_title(f"k = {k}", fontsize=9)
        axes[row, 6].imshow(X[idx].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        axes[row, 6].set_title("original (k = 64)", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(h_pad=1.4, w_pad=0.4)
    save(fig, "chiffres_propres.png")


# ---------------------------------------------------------------------------- PCA, t-SNE, UMAP
def fig_embeddings() -> None:
    from sklearn.manifold import TSNE

    X, y = small_digits()
    X = X / 16.0
    views = [("PCA (2 composantes)", pp.PCA(n_components=2).fit_transform(X))]
    views.append(("t-SNE (perplexité 30)", TSNE(n_components=2, perplexity=30, init="pca",
                                                random_state=0).fit_transform(X)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import umap

        views.append(("UMAP (15 voisins)", umap.UMAP(n_neighbors=15, random_state=0).fit_transform(X)))
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (title, E) in zip(axes, views):
        for digit in range(10):
            pts = E[y == digit]
            ax.scatter(pts[:, 0], pts[:, 1], s=5, color=colors[digit], alpha=0.75, edgecolor="none")
            cx, cy = np.median(pts, axis=0)
            ax.text(cx, cy, str(digit), fontsize=12, weight="bold", ha="center", va="center", color=INK,
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=colors[digit], alpha=0.85))
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "pca_tsne_umap.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 12 -> {OUT.relative_to(ROOT)}")
    fig_golden_rule()
    fig_scaling()
    fig_outlier()
    fig_uni_multi()
    fig_projection()
    fig_explained()
    fig_whitening()
    fig_digits()
    fig_embeddings()
    return 0


if __name__ == "__main__":
    sys.exit(main())
