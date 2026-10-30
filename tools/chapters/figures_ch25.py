#!/usr/bin/env python
"""Draw the figures of chapter 25 (used by Claude).

    python tools/chapters/figures_ch25.py

Writes small PNG files into chapitres/ch25_autoencodeurs/figures/. Every figure is computed or drawn here
(never copied from the book): the shape of an autoencoder, a content blend against a parametric blend of two
rectangles, MNIST digits rebuilt from a few principal components, the projection that a linear autoencoder
learns, the norms of Gaussian draws in several dimensions, the reparameterization trick, and a straight line
against a great circle between two codes. None of them uses the numbers of an exercise: the shapes, the sizes,
the covariance and the dimensions differ from those of 02_exercices.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch25_autoencodeurs" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED


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


def arrow(ax, start, end, color=INK, style="-|>", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def box(ax, xy, w, h, text, color, text_color="white", fontsize=10):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=INK,
                                lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def figure_autoencoder() -> None:
    """The bottleneck: an encoder narrows the input down to a code, a decoder widens it back."""
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    widths = [3.0, 2.2, 1.4, 0.55, 1.4, 2.2, 3.0]
    labels = ["entrée x\n(n valeurs)", "", "", "code z\n(k valeurs)", "", "", "sortie x̂\n(n valeurs)"]
    colors = [C[0], C[0], C[0], C[3], C[1], C[1], C[1]]
    for i, (w, lab, col) in enumerate(zip(widths, labels, colors)):
        x = i * 1.25
        ax.add_patch(Rectangle((x - 0.18, -w / 2), 0.36, w, fc=col, ec=INK, lw=1, alpha=0.85))
        if lab:
            ax.text(x, -widths[0] / 2 - 0.35, lab, ha="center", va="top", fontsize=9, color=INK)
        if i < len(widths) - 1:
            nxt = widths[i + 1]
            ax.add_patch(Polygon([(x + 0.18, w / 2), (x + 1.07, nxt / 2), (x + 1.07, -nxt / 2), (x + 0.18, -w / 2)],
                                 closed=True, fc="#e9edf3", ec="none"))
    ax.annotate("", xy=(3 * 1.25 - 0.3, 1.95), xytext=(-0.2, 1.95),
                arrowprops=dict(arrowstyle="-|>", color=C[0], lw=1.6))
    ax.text(1.75, 2.1, "encodeur $f_\\theta$", ha="center", color=C[0], fontsize=10)
    ax.annotate("", xy=(6 * 1.25 + 0.2, 1.95), xytext=(3 * 1.25 + 0.3, 1.95),
                arrowprops=dict(arrowstyle="-|>", color=C[1], lw=1.6))
    ax.text(5.6, 2.1, "décodeur $g_\\phi$", ha="center", color=C[1], fontsize=10)
    ax.text(3 * 1.25, 0.6, "goulot", ha="center", color=C[3], fontsize=9)
    ax.text(3.75, -2.75, "loss : écart entre x̂ et x  (la cible est l'entrée elle-même)", ha="center", fontsize=9,
            color=MUTED)
    ax.set_xlim(-0.8, 8.3)
    ax.set_ylim(-3.0, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "autoencodeur.png")


def _rectangle_image(cx, cy, w, h, gray, size=64):
    """A size × size image (values in [0, 1]) with one filled rectangle."""
    ys, xs = np.mgrid[0:size, 0:size] + 0.5
    inside = (np.abs(xs - cx) <= w / 2) & (np.abs(ys - cy) <= h / 2)
    return np.where(inside, gray, 0.0)


def figure_blends() -> None:
    """Two rectangles: blending their pixels gives two faint rectangles, blending their parameters one in between."""
    a = dict(cx=16.0, cy=44.0, w=10.0, h=26.0, gray=1.0)
    b = dict(cx=46.0, cy=18.0, w=30.0, h=10.0, gray=0.45)
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, len(ts), figsize=(9.2, 4.1))
    img_a, img_b = _rectangle_image(**a), _rectangle_image(**b)
    for j, t in enumerate(ts):
        content = (1 - t) * img_a + t * img_b
        params = {k: (1 - t) * a[k] + t * b[k] for k in a}
        for row, img in enumerate((content, _rectangle_image(**params))):
            ax = axes[row, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, j].set_title(f"t = {french_number(t)}", fontsize=9)
    axes[0, 0].set_ylabel("contenu\n(pixels)", fontsize=9)
    axes[1, 0].set_ylabel("paramètres\n(centre, taille, gris)", fontsize=9)
    fig.suptitle("Mélanger deux rectangles : $(1 - t)\\,A + t\\,B$", fontsize=11)
    fig.tight_layout()
    save(fig, "melanges.png")


def figure_pca_digits() -> None:
    """MNIST digits rebuilt from their first k principal components: a lossy code of k numbers."""
    from sklearn.decomposition import PCA

    from wb import datasets

    x_train, _ = datasets.load_mnist("train", n=12000, flatten=True, normalize=True)
    x_test, y_test = datasets.load_mnist("test", flatten=True, normalize=True)
    picks = [int(np.flatnonzero(y_test == d)[0]) for d in (2, 4, 5, 8, 9)]
    x = x_test[picks]
    ks = [1, 4, 16, 64]
    fig, axes = plt.subplots(len(picks), len(ks) + 1, figsize=(6.4, 7.0))
    for col, k in enumerate(ks):
        pca = PCA(n_components=k, svd_solver="full").fit(x_train)
        rebuilt = pca.inverse_transform(pca.transform(x))
        for row in range(len(picks)):
            axes[row, col].imshow(rebuilt[row].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"k = {k}", fontsize=10)
    for row in range(len(picks)):
        axes[row, -1].imshow(x[row].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[0, -1].set_title("original\n(784 pixels)", fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Reconstruire à partir de k nombres (ACP, ch. 12)", fontsize=11)
    fig.tight_layout()
    save(fig, "acp_chiffres.png")


def figure_linear_ae() -> None:
    """A 2-D cloud, its first principal axis, and the reconstruction of a linear autoencoder with one latent."""
    rng = np.random.default_rng(25)
    cov = np.array([[3.0, 1.6], [1.6, 1.4]])
    pts = rng.multivariate_normal([0, 0], cov, size=220)
    pts -= pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    u = vecs[:, np.argmax(vals)]
    proj = (pts @ u)[:, None] * u
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=C[0], alpha=0.55, label="données x (centrées)")
    for p, q in zip(pts[::9], proj[::9]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.8)
    ax.scatter(proj[::9, 0], proj[::9, 1], s=16, color=C[1], zorder=3, label="reconstructions x̂ = (wᵀx) w")
    line = np.array([-5.0, 5.0])
    ax.plot(line * u[0], line * u[1], color=C[3], lw=2, label="droite engendrée par w")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Un code à 1 valeur : la meilleure droite est la 1ʳᵉ composante", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "ae_lineaire_acp.png")


def figure_gaussian_norms() -> None:
    """Norms of standard Gaussian draws: they gather around sqrt(d), not near 0."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for col, d in zip(C, (3, 12, 60)):
        norms = np.linalg.norm(rng.standard_normal((40000, d)), axis=1)
        ax.hist(norms, bins=120, density=True, histtype="stepfilled", alpha=0.35, color=col)
        ax.hist(norms, bins=120, density=True, histtype="step", color=col, lw=1.4, label=f"d = {d}")
        ax.axvline(np.sqrt(d), color=col, ls="--", lw=1)
    ax.set_xlabel("norme ‖z‖ d'un tirage z ~ N(0, I_d)")
    ax.set_ylabel("densité")
    ax.legend(fontsize=9)
    ax.set_title("En grande dimension, presque aucun tirage n'est près du centre (pointillés : √d)", fontsize=10)
    save(fig, "normes_gaussiennes.png")


def figure_reparameterization() -> None:
    """Sampling as a node of the graph (no gradient) against z = mu + sigma * eps (gradient through mu and sigma)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax in axes:
        ax.set_xlim(0, 10.4)
        ax.set_ylim(0, 5)
        ax.axis("off")
    ax = axes[0]
    box(ax, (0.3, 3.0), 2.0, 1.0, "encodeur", C[0])
    box(ax, (3.4, 3.6), 1.4, 0.8, "μ", C[0])
    box(ax, (3.4, 2.4), 1.4, 0.8, "log σ²", C[0])
    ax.add_patch(Circle((6.6, 3.4), 0.9, fc=C[3], ec=INK))
    ax.text(6.6, 3.4, "tirage\nz ~ N(μ, σ²)", ha="center", va="center", fontsize=8, color="white")
    box(ax, (8.5, 3.0), 1.8, 0.8, "décodeur", C[1])
    arrow(ax, (2.3, 3.6), (3.4, 4.0))
    arrow(ax, (2.3, 3.4), (3.4, 2.8))
    arrow(ax, (4.8, 4.0), (5.75, 3.65))
    arrow(ax, (4.8, 2.8), (5.75, 3.15))
    arrow(ax, (7.5, 3.4), (8.5, 3.4))
    arrow(ax, (6.0, 1.9), (4.4, 2.3), color=C[1], style="-|>", rad=0.2)
    ax.text(5.3, 1.25, "gradient bloqué :\nz n'est pas une fonction de μ et σ", ha="center", fontsize=8.5, color=C[1])
    ax.set_title("Un tirage dans le graphe", fontsize=10)
    ax = axes[1]
    box(ax, (0.3, 3.0), 2.0, 1.0, "encodeur", C[0])
    box(ax, (3.4, 3.6), 1.4, 0.8, "μ", C[0])
    box(ax, (3.4, 2.4), 1.4, 0.8, "log σ²", C[0])
    box(ax, (2.9, 0.6), 2.4, 0.8, "ε ~ N(0, 1)", C[3])
    box(ax, (5.7, 2.9), 2.2, 1.0, "z = μ + σ ε", C[2])
    box(ax, (8.5, 3.0), 1.8, 0.8, "décodeur", C[1])
    arrow(ax, (2.3, 3.6), (3.4, 4.0))
    arrow(ax, (2.3, 3.4), (3.4, 2.8))
    arrow(ax, (4.8, 4.0), (5.7, 3.6))
    arrow(ax, (4.8, 2.8), (5.7, 3.2))
    arrow(ax, (5.3, 1.0), (6.4, 2.9))
    arrow(ax, (7.9, 3.4), (8.5, 3.4))
    ax.text(8.2, 1.3, "ε est une entrée :\nle gradient passe par μ et σ", ha="center", fontsize=8.5, color=C[2])
    ax.set_title("L'astuce de reparamétrisation", fontsize=10)
    fig.tight_layout()
    save(fig, "reparametrisation.png")


def figure_slerp() -> None:
    """Between two codes of the same norm, the straight line cuts inside the circle; slerp follows the arc."""
    a0, a1, r = np.deg2rad(15), np.deg2rad(150), 2.0
    z0, z1 = r * np.array([np.cos(a0), np.sin(a0)]), r * np.array([np.cos(a1), np.sin(a1)])
    t = np.linspace(0, 1, 7)[:, None]
    lin = (1 - t) * z0 + t * z1
    omega = a1 - a0
    sl = np.sin((1 - t) * omega) / np.sin(omega) * z0 + np.sin(t * omega) / np.sin(omega) * z1
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.add_patch(Circle((0, 0), r, fill=False, ec=MUTED, ls="--"))
    ax.plot(lin[:, 0], lin[:, 1], "o-", color=C[1], label="linéaire")
    ax.plot(sl[:, 0], sl[:, 1], "s-", color=C[0], label="sphérique (slerp)")
    for name, z in (("z₀", z0), ("z₁", z1)):
        ax.annotate(name, z, xytext=(6, 6), textcoords="offset points", fontsize=11)
    ax.scatter([0], [0], color=INK, s=12)
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.8, 2.6)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Sept points entre deux codes de même norme", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "slerp.png")


def main() -> int:
    wbplot.set_style()
    figure_autoencoder()
    figure_blends()
    figure_pca_digits()
    figure_linear_ae()
    figure_gaussian_norms()
    figure_reparameterization()
    figure_slerp()
    print(f"Figures -> {OUT.relative_to(ROOT)}: {sorted(p.name for p in OUT.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
