#!/usr/bin/env python
"""Draw the figures of chapter 27 (used by Claude).

    python tools/chapters/figures_ch27.py

Writes small PNG files into chapitres/ch27_gan/figures/. Every figure is computed or drawn here (never copied from
the book): the game between the generator and the discriminator with the two losses, the four verdicts of the
discriminator, the optimal discriminator for two 1-D densities, the saturation of the two generator losses, the
full and partial mode collapse on a ring of gaussians, and the 1-D Wasserstein distance as the area between two
empirical CDFs. None of them uses the numbers of an exercise: the densities, the samples and the ring differ from
those of 02_exercices.md and of the notebook.
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

OUT = ROOT / "chapitres" / "ch27_gan" / "figures"
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


def arrow(ax, start, end, color=INK, lw=1.4, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def box(ax, xy, w, h, text, color, text_color="white", fontsize=11):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=color, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def figure_game() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    box(ax, (0.2, 0.55), 1.6, 0.9, "bruit $\\mathbf{z}$\n$\\sim p_z$", C[2], fontsize=10)
    box(ax, (2.5, 0.45), 2.0, 1.1, "générateur\n$G$", C[0])
    box(ax, (0.2, 3.0), 1.6, 0.9, "données\n$\\mathbf{x} \\sim p_{\\mathrm{data}}$", C[3], fontsize=10)
    box(ax, (6.0, 1.6), 2.0, 1.4, "discriminateur\n$D$", C[1])
    arrow(ax, (1.85, 1.0), (2.45, 1.0))
    arrow(ax, (4.55, 1.0), (5.95, 1.95))
    ax.text(4.0, 1.75, "$G(\\mathbf{z})$ (faux)", fontsize=9.5, color=INK)
    arrow(ax, (1.85, 3.45), (5.95, 2.65))
    ax.text(3.0, 3.35, "$\\mathbf{x}$ (vrai)", fontsize=9.5, color=INK)
    arrow(ax, (8.05, 2.3), (8.75, 2.3))
    ax.text(8.85, 2.3, "$D(\\cdot) \\in [0, 1]$\n« probabilité\nque ce soit vrai »", fontsize=9.5, va="center",
            color=INK)
    ax.text(7.0, 4.25, "étape de $D$ : vrais → label 1, faux (détachés) → label 0", ha="center", fontsize=9,
            color=C[1])
    ax.text(7.0, 3.85, "$L_D = -\\ln D(\\mathbf{x}) - \\ln(1 - D(G(\\mathbf{z})))$", ha="center", fontsize=9.5,
            color=INK)
    ax.text(7.9, 0.95, "étape de $G$ : faux → label 1\n$D$ gelé (seul l'optimiseur de $G$ avance)", ha="center",
            va="center", fontsize=9, color=C[0])
    ax.text(7.9, 0.3, "$L_G = -\\ln D(G(\\mathbf{z}))$ (loss non saturante)", ha="center", fontsize=9.5, color=INK)
    save(fig, "jeu_gan.png")


def figure_verdicts() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 3.7)
    ax.axis("off")
    cells = {(0, 0): ("vrai positif (TP)", "personne n'apprend\n(selon le livre)", C[2]),
             (0, 1): ("faux négatif (FN)", "$D$ apprend :\nun vrai pris pour un faux", C[1]),
             (1, 0): ("faux positif (FP)", "$D$ apprend :\nil s'est fait duper", C[1]),
             (1, 1): ("vrai négatif (TN)", "$G$ apprend :\nsa contrefaçon est repérée", C[0])}
    x0, y0, w, h = 1.9, 0.25, 2.6, 1.35
    for (row, col), (name, who, color) in cells.items():
        x, y = x0 + col * w, y0 + (1 - row) * h
        ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="white", lw=2, alpha=0.9))
        ax.text(x + w / 2, y + h * 0.68, name, ha="center", va="center", color="white", fontsize=10.5,
                fontweight="bold")
        ax.text(x + w / 2, y + h * 0.3, who, ha="center", va="center", color="white", fontsize=9)
    ax.text(x0 + w / 2, 3.15, "$D$ dit « vrai »", ha="center", fontsize=10, color=INK)
    ax.text(x0 + 1.5 * w, 3.15, "$D$ dit « faux »", ha="center", fontsize=10, color=INK)
    ax.text(x0 - 0.15, y0 + 1.5 * h, "échantillon\nvrai", ha="right", va="center", fontsize=10, color=INK)
    ax.text(x0 - 0.15, y0 + 0.5 * h, "échantillon\nfaux", ha="right", va="center", fontsize=10, color=INK)
    save(fig, "quatre_verdicts.png")


def figure_optimal_d() -> None:
    x = np.linspace(-5, 6, 600)
    p_data = 0.6 * gauss(x, 1.5, 0.8) + 0.4 * gauss(x, -1.5, 0.6)
    p_g = gauss(x, 0.3, 1.6)
    d_star = p_data / (p_data + p_g)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    axes[0].plot(x, p_data, color=C[3], lw=2, label="$p_{\\mathrm{data}}$")
    axes[0].plot(x, p_g, color=C[0], lw=2, label="$p_g$ (générateur)")
    axes[0].set_title("deux densités", fontsize=10)
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].set_xlabel("$x$")
    axes[1].plot(x, d_star, color=C[1], lw=2, label="$D^*(x) = p_{\\mathrm{data}} / (p_{\\mathrm{data}} + p_g)$")
    axes[1].axhline(0.5, color=MUTED, ls="--", lw=1)
    axes[1].text(-4.8, 0.53, "½ : là où $p_g = p_{\\mathrm{data}}$", fontsize=8.5, color=MUTED)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("le discriminateur optimal", fontsize=10)
    axes[1].legend(fontsize=8.5, frameon=False, loc="lower right")
    axes[1].set_xlabel("$x$")
    plt.tight_layout()
    save(fig, "discriminateur_optimal.png")


def figure_saturation() -> None:
    d = np.linspace(0.001, 0.999, 500)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    axes[0].plot(d, np.log(1 - d), color=C[1], lw=2, label="minimax : $G$ minimise $\\ln(1 - D(G(\\mathbf{z})))$")
    axes[0].plot(d, -np.log(d), color=C[0], lw=2, label="non saturante : $G$ minimise $-\\ln D(G(\\mathbf{z}))$")
    axes[0].set_ylim(-5, 5)
    axes[0].set_xlabel("$D(G(\\mathbf{z}))$")
    axes[0].set_title("ce que minimise le générateur", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc="upper center")
    # derivative with respect to the logit a of D = sigmoid(a)
    axes[1].plot(d, d, color=C[1], lw=2, label="minimax : $|\\partial/\\partial a| = D$")
    axes[1].plot(d, 1 - d, color=C[0], lw=2, label="non saturante : $|\\partial/\\partial a| = 1 - D$")
    axes[1].axvspan(0, 0.1, color=C[2], alpha=0.15)
    axes[1].text(0.11, 0.86, "début de l'entraînement :\n$D$ repère les faux", fontsize=8.5, color=INK)
    axes[1].set_xlabel("$D(G(\\mathbf{z})) = \\sigma(a)$")
    axes[1].set_title("gradient reçu par le logit $a$ du faux", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False, loc="center right")
    plt.tight_layout()
    save(fig, "saturation.png")


def _ring(rng, n, k=6, r=3.0, s=0.12, modes=None):
    modes = np.arange(k) if modes is None else np.asarray(modes)
    c = rng.choice(modes, size=n)
    angle = 2 * np.pi * c / k
    return np.stack([r * np.cos(angle), r * np.sin(angle)], 1) + s * rng.standard_normal((n, 2))


def figure_collapse() -> None:
    rng = np.random.default_rng(7)
    panels = [("données : 6 modes", _ring(rng, 600), C[3]),
              ("effondrement partiel : 2 modes", _ring(rng, 600, modes=[1, 4]), C[0]),
              ("effondrement total : 1 point", _ring(rng, 600, s=0.03, modes=[2]), C[1])]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharex=True, sharey=True)
    for ax, (title, pts, color) in zip(axes, panels):
        ax.scatter(*pts.T, s=4, color=color, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.2, 4.2)
    plt.tight_layout()
    save(fig, "effondrement.png")


def figure_wasserstein() -> None:
    u = np.array([0.4, 1.1, 1.7, 2.9])
    v = np.array([1.0, 2.2, 3.1, 3.6])
    grid = np.linspace(0, 4.2, 2000)
    cdf_u = np.searchsorted(np.sort(u), grid, side="right") / u.size
    cdf_v = np.searchsorted(np.sort(v), grid, side="right") / v.size
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.step(grid, cdf_u, where="post", color=C[0], lw=2, label="fonction de répartition de $u$")
    ax.step(grid, cdf_v, where="post", color=C[3], lw=2, label="fonction de répartition de $v$")
    ax.fill_between(grid, cdf_u, cdf_v, step="post", color=C[1], alpha=0.25, label="aire = $W_1(u, v)$")
    ax.set_xlabel("valeur")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_title("distance de Wasserstein 1D : l'aire entre les deux escaliers", fontsize=10)
    save(fig, "wasserstein.png")


def main() -> int:
    figure_game()
    figure_verdicts()
    figure_optimal_d()
    figure_saturation()
    figure_collapse()
    figure_wasserstein()
    print(f"✅ figures written to {OUT.relative_to(ROOT)}: {', '.join(sorted(p.name for p in OUT.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
