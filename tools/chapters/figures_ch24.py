#!/usr/bin/env python
"""Draw the figures of chapter 24 (used by Claude).

    python tools/chapters/figures_ch24.py

Writes small PNG files into chapitres/ch24_pytorch_2/figures/. Every figure is computed or drawn here (never
copied from the book): the max-norm projection, a grid search against a random search when only one
hyperparameter matters, the positions computed by a stride against a pooling, an augmented CIFAR-10 photo,
a sum of sines cut into windows, the outputs of a two-layer LSTM and a residual block. None of them uses the
numbers of an exercise: the series, the shapes and the networks differ from those of 02_exercices.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch24_pytorch_2" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED
RANDOM_SEED = 12          # chosen so that one random draw falls near the peak (see figure_search)


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


def figure_max_norm() -> None:
    """Three weight vectors and the ball of radius c: the outside ones are pulled back along their direction."""
    c = 2.0
    vectors = [np.array([1.1, 0.9]), np.array([2.75, 1.1]), np.array([-1.0, 3.0])]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.add_patch(Circle((0, 0), c, fc="#eef3fb", ec=C[0], lw=1.6))
    ax.text(c * np.cos(-0.6) + 0.1, c * np.sin(-0.6) - 0.25, "‖w‖ = c", color=C[0], fontsize=10)
    for k, w in enumerate(vectors):
        norm = np.linalg.norm(w)
        projected = w * min(1.0, c / norm)
        color = C[1] if norm > c else C[2]
        arrow(ax, (0, 0), tuple(w), color=color, lw=1.8)
        if norm > c:
            arrow(ax, (0, 0), tuple(projected), color=C[3], lw=2.6)
            ax.plot([projected[0], w[0]], [projected[1], w[1]], ls=":", color=MUTED, lw=1.2)
            ax.text(w[0] + 0.08, w[1] + 0.08, f"w (‖w‖ = {norm:.1f})".replace(".", ","), fontsize=9, color=color)
            ax.text(projected[0] + 0.12, projected[1] - 0.32, "projeté", fontsize=9, color=C[3])
        else:
            ax.text(w[0] + 0.08, w[1] - 0.25, "inchangé", fontsize=9, color=color)
    ax.set_aspect("equal")
    ax.set_xlim(-2.7, 3.6)
    ax.set_ylim(-2.4, 3.4)
    ax.axhline(0, color=wbplot.GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=wbplot.GRID, lw=0.8, zorder=0)
    ax.set_title("Max-norm, c = 2 : w ← w · min(1, c / ‖w‖)", fontsize=10, color=INK)
    save(fig, "max_norm.png")


def figure_search() -> None:
    """A 3 x 3 grid against 9 random draws on a score that depends on one hyperparameter only."""
    def score(a):                       # only the first hyperparameter matters: one narrow bump
        return 0.80 + 0.12 * np.exp(-((a - 0.62) / 0.09) ** 2)

    grid = np.array([(a, b) for a in (1 / 6, 1 / 2, 5 / 6) for b in (1 / 6, 1 / 2, 5 / 6)])
    rng = np.random.default_rng(RANDOM_SEED)
    rand = rng.uniform(0.04, 0.96, size=(9, 2))
    xs = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), gridspec_kw={"height_ratios": [1, 2.4]}, sharex=True)
    for col, (pts, title) in enumerate([(grid, "recherche en grille : 9 réglages"),
                                        (rand, "recherche au hasard : 9 réglages")]):
        top, ax = axes[0, col], axes[1, col]
        top.plot(xs, score(xs), color=MUTED, lw=1.4)
        top.scatter(pts[:, 0], score(pts[:, 0]), color=C[1], zorder=3, s=26)
        best = pts[np.argmax(score(pts[:, 0]))]
        top.scatter([best[0]], [score(best[0])], color=C[3], zorder=4, s=60, marker="*")
        top.set_ylabel("score", fontsize=9)
        top.set_title(title, fontsize=10, color=INK)
        top.set_ylim(0.78, 0.95)
        ax.imshow(np.tile(score(xs), (200, 1)), extent=(0, 1, 0, 1), origin="lower", cmap="Blues",
                  aspect="auto", vmin=0.78, vmax=0.95)
        ax.scatter(pts[:, 0], pts[:, 1], color=C[1], edgecolor="white", s=60, zorder=3)
        ax.scatter([best[0]], [best[1]], color=C[3], marker="*", s=180, zorder=4, edgecolor="white")
        for a in np.unique(np.round(pts[:, 0], 6)):
            ax.axvline(a, color=C[1], lw=0.6, ls=":", zorder=2)
        ax.set_xlabel("hyperparamètre important", fontsize=9)
        ax.set_ylabel("hyperparamètre sans effet", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.02, 0.03, f"{len(np.unique(np.round(pts[:, 0], 6)))} valeurs essayées", fontsize=9,
                color=INK, bbox=dict(fc="white", ec="none", alpha=0.8))
    fig.tight_layout()
    save(fig, "recherche.png")


def figure_stride_pooling() -> None:
    """Positions computed by a stride-1 convolution then a 2 x 2 max pooling, and by a stride-2 convolution."""
    n = 8
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), gridspec_kw={"width_ratios": [1, 0.55, 1]})
    ax = axes[0]
    for i in range(n):
        for j in range(n):
            kept = i % 2 == 0 and j % 2 == 0
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=C[0] if True else "white", ec="white", lw=1.2,
                                   alpha=0.9 if kept else 0.45))
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            ax.add_patch(Rectangle((j, n - 2 - i), 2, 2, fill=False, ec=INK, lw=1.4))
    ax.set_title("stride 1 : 64 positions calculées,\npuis un max pooling 2 × 2 en garde 16", fontsize=9.5,
                 color=INK)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, "même sortie\n4 × 4,\nquatre fois moins\nde positions\ncalculées à droite",
                 ha="center", va="center", fontsize=9.5, color=INK)
    ax = axes[2]
    for i in range(n):
        for j in range(n):
            computed = i % 2 == 0 and j % 2 == 0
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=C[1] if computed else "#f2f2f0", ec="white",
                                   lw=1.2))
    ax.set_title("stride 2 : seules 16 positions\nsont calculées", fontsize=9.5, color=INK)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "stride_pooling.png")


def figure_augmentation() -> None:
    """One CIFAR-10 photo and seven random transformations; the last one (upside down) is implausible."""
    import torch
    from torchvision.transforms import v2

    import wb.datasets as wbd

    X, y = wbd.load_cifar10("train", seed=None)
    index = int(np.flatnonzero(y == wbd.CIFAR10_CLASSES.index("horse"))[7])
    image = torch.from_numpy(X[index]).permute(2, 0, 1)            # (3, 32, 32) uint8
    torch.manual_seed(24)
    steps = [("original", v2.Identity()),
             ("miroir", v2.RandomHorizontalFlip(p=1.0)),
             ("recadrage décalé\n(marge de 4 pixels)", v2.RandomCrop(32, padding=4)),
             ("rotation de 12°", v2.RandomRotation((12, 12))),
             ("luminosité et\ncontraste", v2.ColorJitter(brightness=(1.35, 1.35), contrast=(0.7, 0.7))),
             ("recadrage\net miroir", v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(1.0)])),
             ("léger\nagrandissement", v2.RandomResizedCrop(32, scale=(0.7, 0.7), ratio=(1.0, 1.0))),
             ("haut et bas inversés :\nimplausible", v2.RandomVerticalFlip(p=1.0))]
    fig, axes = plt.subplots(1, len(steps), figsize=(13.5, 2.5))
    for ax, (title, tf) in zip(axes, steps):
        ax.imshow(tf(image).permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in axes[-1].spines.values():
        spine.set_color(wbplot.BAD)
        spine.set_linewidth(2.5)
    axes[-1].plot([0, 31], [0, 31], color=wbplot.BAD, lw=2.5)
    axes[-1].plot([0, 31], [31, 0], color=wbplot.BAD, lw=2.5)
    save(fig, "augmentation.png")


def figure_sines() -> None:
    """A sum of two sines (not the book's): train, test further on, and one window with its target."""
    d_theta, freqs, amps = 0.06, np.array([1.0, 3.0]), np.array([1.0, 0.5])

    def series(start, n_steps):
        n = np.arange(start, start + n_steps)
        return n, (amps * np.sin(freqs * (n * d_theta)[:, None])).sum(axis=1)

    n_tr, s_tr = series(0, 200)
    n_te, s_te = series(300, 200)
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    ax.plot(n_tr, s_tr, color=C[0], lw=1.6, label="train (points 0 à 199)")
    ax.plot(n_te, s_te, color=C[1], lw=1.6, label="test (points 300 à 499)")
    start, w = 60, 5
    ax.axvspan(start - 0.5, start + w - 0.5, color=C[2], alpha=0.18)
    ax.scatter(n_tr[start:start + w], s_tr[start:start + w], color=C[2], s=22, zorder=3,
               label=f"une fenêtre de {w} points (entrée)")
    ax.scatter([n_tr[start + w]], [s_tr[start + w]], color=C[3], s=60, marker="*", zorder=4,
               label="la valeur suivante (cible)")
    ax.set_xlabel("n (numéro du point)")
    ax.set_ylabel("s(n)")
    ax.set_title("s(n) = sin(n Δθ) + 0,5 sin(3 n Δθ), Δθ = 0,06", fontsize=10, color=INK)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    inset = ax.inset_axes([0.47, 0.56, 0.17, 0.4])          # zoom on the window and its target
    k = np.arange(start - 3, start + w + 4)
    inset.plot(n_tr[k], s_tr[k], color=C[0], lw=1.2)
    inset.scatter(n_tr[start:start + w], s_tr[start:start + w], color=C[2], s=26, zorder=3)
    inset.scatter([n_tr[start + w]], [s_tr[start + w]], color=C[3], s=90, marker="*", zorder=4)
    inset.axvspan(start - 0.5, start + w - 0.5, color=C[2], alpha=0.18)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("zoom", fontsize=8, color=MUTED)
    save(fig, "sinus.png")


def figure_lstm_outputs() -> None:
    """A two-layer LSTM unrolled over T steps: output (top layer, every step), h_n and c_n (last step, each layer)."""
    T = 4
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    xs = [1.4 * t for t in range(T)]
    for t, x in enumerate(xs):
        label = f"t = {t + 1}" if t < T - 1 else "t = T"
        box(ax, (x, 0.0), 0.9, 0.55, f"x_{t + 1}" if t < T - 1 else "x_T", C[7], fontsize=9)
        box(ax, (x, 1.2), 0.9, 0.7, "couche 1", C[0], fontsize=8.5)
        box(ax, (x, 2.6), 0.9, 0.7, "couche 2", C[1], fontsize=8.5)
        arrow(ax, (x + 0.45, 0.55), (x + 0.45, 1.2))
        arrow(ax, (x + 0.45, 1.9), (x + 0.45, 2.6))
        arrow(ax, (x + 0.45, 3.3), (x + 0.45, 3.95), color=C[3])
        ax.text(x + 0.45, -0.35, label, ha="center", fontsize=8.5, color=MUTED)
        if t:
            arrow(ax, (xs[t - 1] + 0.9, 1.55), (x, 1.55), color=MUTED)
            arrow(ax, (xs[t - 1] + 0.9, 2.95), (x, 2.95), color=MUTED)
    ax.add_patch(FancyBboxPatch((-0.1, 3.95), xs[-1] + 1.1, 0.55, boxstyle="round,pad=0.02", fc="#fdeedd",
                                ec=C[3], lw=1.4))
    ax.text(xs[-1] / 2 + 0.45, 4.22, "output : (N, T, H), la sortie h_t de la couche 2 à chaque pas",
            ha="center", va="center", fontsize=9.5, color=INK)
    x_end = xs[-1] + 1.6
    for k, (y0, name) in enumerate([(2.95, "couche 2"), (1.55, "couche 1")]):
        arrow(ax, (xs[-1] + 0.9, y0), (x_end, y0), color=C[4])
    ax.add_patch(FancyBboxPatch((x_end, 1.2), 3.3, 2.1, boxstyle="round,pad=0.02", fc="#eef6ee", ec=C[4],
                                lw=1.4))
    ax.text(x_end + 1.65, 2.25, "h_n et c_n : (L, N, H)\nl'état du dernier pas\nde chaque couche\n"
            "(ligne 0 : couche 1,\nligne −1 : couche 2)", ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(x_end + 1.65, 0.55, "output[:, −1] = h_n[−1]\n(un seul sens)", ha="center", fontsize=9.5,
            color=C[3])
    ax.set_xlim(-0.3, x_end + 3.5)
    ax.set_ylim(-0.6, 4.7)
    ax.axis("off")
    save(fig, "sorties_lstm.png")


def figure_residual() -> None:
    """A plain block of two convolutions, and the residual block that adds its input back."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8))
    for ax, residual in zip(axes, (False, True)):
        y = 0.2
        box(ax, (1.0, y), 2.0, 0.5, "x", C[7], fontsize=10)
        layers = [("conv 3 × 3, BN, ReLU", C[0]), ("conv 3 × 3, BN", C[0])]
        prev = y + 0.5
        for k, (text, color) in enumerate(layers):
            top = 1.2 + 1.0 * k
            arrow(ax, (2.0, prev), (2.0, top))
            box(ax, (0.6, top), 2.8, 0.6, text, color, fontsize=9)
            prev = top + 0.6
        if residual:
            ax.add_patch(Circle((2.0, 3.55), 0.22, fc="white", ec=INK, lw=1.4))
            ax.text(2.0, 3.55, "+", ha="center", va="center", fontsize=13, color=INK)
            arrow(ax, (2.0, prev), (2.0, 3.33))
            arrow(ax, (3.0, 0.45), (2.22, 3.55), color=C[3], rad=0.55, lw=1.8)
            ax.text(4.25, 1.6, "raccourci :\nx tel quel", color=C[3], fontsize=9, ha="center")
            ax.text(-0.15, 2.0, "F(x)", color=C[0], fontsize=10, ha="center")
            box(ax, (1.0, 4.05), 2.0, 0.5, "ReLU", C[2], fontsize=9)
            arrow(ax, (2.0, 3.77), (2.0, 4.05))
            ax.text(2.0, 4.8, "y = ReLU(F(x) + x)", ha="center", fontsize=10, color=INK)
            ax.set_title("bloc résiduel", fontsize=10, color=INK)
        else:
            box(ax, (1.0, 3.3), 2.0, 0.5, "ReLU", C[2], fontsize=9)
            arrow(ax, (2.0, prev), (2.0, 3.3))
            ax.text(2.0, 4.8, "y = ReLU(F(x))", ha="center", fontsize=10, color=INK)
            ax.set_title("bloc ordinaire", fontsize=10, color=INK)
        ax.set_xlim(-0.6, 4.8)
        ax.set_ylim(0, 5.2)
        ax.set_aspect("equal")
        ax.axis("off")
    save(fig, "bloc_residuel.png")


def main() -> int:
    figure_max_norm()
    figure_search()
    figure_stride_pooling()
    figure_augmentation()
    figure_sines()
    figure_lstm_outputs()
    figure_residual()
    print(f"✅ figures written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
