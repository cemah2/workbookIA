#!/usr/bin/env python
"""Draw the figures of chapter 23 (used by Claude).

    python tools/chapters/figures_ch23.py

Writes small PNG files into chapitres/ch23_pytorch_1/figures/. Every figure is computed or drawn here (never
copied from the book): the two memory layouts of a colour image, a tensor of twelve values seen through
several shapes, a sample of Fashion-MNIST, the pipeline of a PyTorch training, four learning-rate schedules
and an early stopping. None of them uses the numbers of an exercise: the shapes, schedules and losses differ
from those of 02_exercices.md.
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

import wb  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch23_pytorch_1" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
RGB = {"R": "#d1495b", "G": "#66a182", "B": "#3c78d8"}


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


def cell(ax, x, y, text, color, w=1.0, h=1.0, fontsize=9, text_color="white"):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="white", lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def figure_channels() -> None:
    """A 2 x 2 colour image: the twelve values in memory, channels first (NCHW) or channels last (NHWC)."""
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 4.4), gridspec_kw={"height_ratios": [1.6, 1, 1]})
    ax = axes[0]
    for k, ch in enumerate("RGB"):                         # the three channels, drawn side by side
        for i in range(2):
            for j in range(2):
                cell(ax, 3.2 * k + j, 1 - i, f"{ch}{i}{j}", RGB[ch])
        ax.text(3.2 * k + 1, 2.25, {"R": "canal rouge", "G": "canal vert", "B": "canal bleu"}[ch],
                ha="center", fontsize=9, color=INK)
    ax.text(9.6, 1, "une image de 2 × 2 pixels,\n3 canaux : 12 valeurs\n(« R10 » : rouge, ligne 1, colonne 0)",
            fontsize=9, va="center", color=INK)
    ax.set_xlim(-0.2, 14.5)
    ax.set_ylim(-0.2, 2.6)
    ax.axis("off")
    orders = {"NCHW (channels_first, PyTorch) : forme (1, 3, 2, 2)":
              [f"{c}{i}{j}" for c in "RGB" for i in range(2) for j in range(2)],
              "NHWC (channels_last, Keras, fichiers d'images) : forme (1, 2, 2, 3)":
              [f"{c}{i}{j}" for i in range(2) for j in range(2) for c in "RGB"]}
    for ax, (title, order) in zip(axes[1:], orders.items()):
        for pos, name in enumerate(order):
            cell(ax, pos, 0, name, RGB[name[0]])
            ax.text(pos + 0.5, -0.35, str(pos), ha="center", va="top", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_xlim(-0.2, 14.5)
        ax.set_ylim(-0.9, 1.1)
        ax.axis("off")
    axes[2].text(12.3, 0.5, "position\nen mémoire", fontsize=8, va="center", color=INK)
    save(fig, "canaux_nchw_nhwc.png")


def figure_reshape() -> None:
    """Twelve values seen as a list, a 2 x 6 grid, a 3 x 2 x 2 block, and the transposed grid."""
    fig = plt.figure(figsize=(9.2, 5.0))
    shade = [plt.cm.viridis(0.12 + 0.07 * v) for v in range(12)]
    ax = fig.add_axes([0.02, 0.80, 0.96, 0.16])
    for v in range(12):
        cell(ax, v, 0, str(v), shade[v])
    ax.set_title("t = torch.arange(12) : la mémoire, une liste de 12 valeurs", fontsize=10, loc="left", color=INK)
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    ax = fig.add_axes([0.02, 0.36, 0.40, 0.34])
    for v in range(12):
        i, j = divmod(v, 6)
        cell(ax, j, 1 - i, str(v), shade[v])
    ax.set_title("t.reshape(2, 6) : une vue, même ordre", fontsize=10, loc="left", color=INK)
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.1, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = fig.add_axes([0.47, 0.30, 0.51, 0.42])
    for v in range(12):
        a, rest = divmod(v, 4)
        i, j = divmod(rest, 2)
        cell(ax, 2.6 * a + j, 1 - i, str(v), shade[v])
    for a in range(3):
        ax.text(2.6 * a + 1, 2.2, f"[{a}]", ha="center", fontsize=9, color=INK)
    ax.set_title("t.reshape(3, 2, 2) : 3 « pages » de 2 × 2", fontsize=10, loc="left", color=INK)
    ax.set_xlim(-0.2, 7.6)
    ax.set_ylim(-0.1, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = fig.add_axes([0.02, -0.42, 0.30, 0.62])
    grid = np.arange(12).reshape(2, 6).T                 # 6 x 2, read row by row
    for i in range(6):
        for j in range(2):
            v = int(grid[i, j])
            cell(ax, j, 5 - i, str(v), shade[v])
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.1, 6.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax = fig.add_axes([0.22, -0.42, 0.76, 0.62])
    order = grid.reshape(12)
    ax.text(0.0, 0.5, "t.reshape(2, 6).T : forme (6, 2), une vue non contiguë.\n"
                      "Lue ligne par ligne, elle donne " + ", ".join(str(v) for v in order[:6]) + ", …,\n"
                      "un autre ordre que celui de la mémoire :\n"
                      "view(12) refuse, reshape(12) fait une copie dans ce nouvel ordre.",
            fontsize=9.5, va="center", color=INK, transform=ax.transAxes)
    ax.axis("off")
    save(fig, "remodeler.png")


def figure_fashion() -> None:
    """Eight random training images of each of the ten classes of Fashion-MNIST."""
    X, y = wb.datasets.load_fashion_mnist("train", seed=None)
    rng = np.random.default_rng(23)
    fig, axes = plt.subplots(10, 8, figsize=(6.4, 8.6))
    for label in range(10):
        picks = rng.choice(np.flatnonzero(y == label), 8, replace=False)
        for col, idx in enumerate(picks):
            ax = axes[label, col]
            ax.imshow(X[idx], cmap="gray_r", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[label, 0].set_ylabel(f"{label} · {wb.datasets.FASHION_CLASSES[label]}", rotation=0, ha="right",
                                  va="center", fontsize=8, color=INK)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    save(fig, "fashion_mnist_classes.png")


def figure_pipeline() -> None:
    """Data, preprocessing, Dataset and DataLoader, then the training loop and the end-of-epoch validation."""
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    w, h, gap = 2.2, 1.2, 0.5
    top, low = 3.3, 1.0
    steps = [("images uint8\n(N, 28, 28)", C[7]), ("prétraitement :\nfloat32, /255,\nμ et σ du train", C[0]),
             ("Dataset\n(image, label)", C[1]), ("DataLoader :\nmini-batches\nmélangés", C[2])]
    for k, (text, color) in enumerate(steps):
        box(ax, (k * (w + gap), top), w, h, text, color, fontsize=8.5)
        if k:
            arrow(ax, (k * (w + gap) - gap, top + h / 2), (k * (w + gap), top + h / 2))
    loop = [("passe avant\nlogits = model(xb)", C[3]), ("loss\nloss_fn(logits, yb)", C[4]),
            ("passe arrière\nloss.backward()", C[5]), ("mise à jour\noptimizer.step()", C[6])]
    for k, (text, color) in enumerate(loop):
        box(ax, (k * (w + gap), low), w, h, text, color, fontsize=8.5)
        if k:
            arrow(ax, (k * (w + gap) - gap, low + h / 2), (k * (w + gap), low + h / 2))
    arrow(ax, (3 * (w + gap) + w / 2, top), (w / 2, low + h))
    ax.text(2 * (w + gap) - 0.3, 2.45, "pour chaque mini-batch, en mode model.train()", ha="left",
            fontsize=9, color=INK)
    arrow(ax, (3 * (w + gap) + w / 2, low), (w / 2, low), rad=-0.25)
    ax.text(1.5 * (w + gap) + w / 2, -0.15, "optimizer.zero_grad(), puis le mini-batch suivant", ha="center",
            fontsize=9, color=INK)
    x_end = 4 * (w + gap) + 0.3
    box(ax, (x_end, 2.6), w, 1.5, "fin d'epoch :\nvalidation en mode\nmodel.eval(),\ninference_mode()", C[0],
        fontsize=8.5)
    box(ax, (x_end, 0.4), w, 1.3, "historique :\ntrain_loss,\nval_loss, val_acc", C[1], fontsize=8.5)
    arrow(ax, (x_end + w / 2, 2.6), (x_end + w / 2, 1.7))
    arrow(ax, (3 * (w + gap) + w, low + h), (x_end, 3.0), rad=-0.15)
    ax.set_xlim(-0.2, x_end + w + 0.2)
    ax.set_ylim(-0.6, 4.7)
    ax.axis("off")
    save(fig, "pipeline.png")


def figure_schedules() -> None:
    """Four learning-rate schedules over 60 epochs (none of them is the book's or an exercise's)."""
    epochs = np.arange(60)
    base = 0.1
    linear = base * np.maximum(0.25, 1 - 0.015 * epochs)
    step = base * 0.5 ** (epochs // 15)
    cosine = base * (1 + np.cos(np.pi * epochs / 60)) / 2
    rng = np.random.default_rng(5)
    val = 0.9 * np.exp(-epochs / 12) + 0.25 + 0.004 * np.maximum(epochs - 25, 0) + rng.normal(0, 0.012, 60)
    plateau, lr, best, wait = [], base, np.inf, 0
    for v in val:                                         # ReduceLROnPlateau(factor=0.5, patience=4)
        plateau.append(lr)
        if v < best:
            best, wait = v, 0
        else:
            wait += 1
            if wait > 4:
                lr, wait = lr * 0.5, 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(epochs, linear, color=C[0], lw=2, label="linéaire avec plancher (LambdaLR)")
    ax.plot(epochs, step, color=C[1], lw=2, drawstyle="steps-post", label="par paliers (StepLR)")
    ax.plot(epochs, cosine, color=C[2], lw=2, label="cosinus (CosineAnnealingLR)")
    ax.plot(epochs, plateau, color=C[3], lw=2, drawstyle="steps-post", label="sur plateau (ReduceLROnPlateau)")
    ax.set_xlabel("epoch (comptée à partir de 0)")
    ax.set_ylabel("learning rate")
    ax.legend(fontsize=8)
    ax.set_title("Quatre plannings, learning rate de départ 0,1", fontsize=10)
    ax = axes[1]
    ax.plot(epochs, val, color=C[3], lw=1.5)
    ax.set_xlabel("epoch (comptée à partir de 0)")
    ax.set_ylabel("loss de validation")
    ax.set_title("La loss qui pilote ReduceLROnPlateau", fontsize=10)
    fig.tight_layout()
    save(fig, "plannings.png")


def figure_early_stopping() -> None:
    """An invented validation loss: best epoch, patience window, stop and restored weights."""
    rng = np.random.default_rng(9)
    epochs = np.arange(1, 41)
    train = 0.85 * np.exp(-epochs / 6) + 0.04 + rng.normal(0, 0.006, 40)
    val = 0.7 * np.exp(-epochs / 6) + 0.32 + 0.006 * np.maximum(epochs - 14, 0) + rng.normal(0, 0.008, 40)
    patience, best, best_epoch, wait, stop = 8, np.inf, 0, 0, None
    for e, v in zip(epochs, val):
        if v < best:
            best, best_epoch, wait = v, e, 0
        else:
            wait += 1
            if wait == patience:
                stop = e
                break
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(epochs[:stop], train[:stop], color=C[0], lw=1.6, label="loss d'entraînement")
    ax.plot(epochs[stop - 1:], train[stop - 1:], color=C[0], lw=1, ls=":")
    ax.plot(epochs[:stop], val[:stop], color=C[1], lw=2, label="loss de validation (surveillée)")
    ax.plot(epochs[stop - 1:], val[stop - 1:], color=C[1], lw=1, ls=":", label="les epochs qu'on n'entraîne pas")
    ax.axvspan(best_epoch, stop, color=C[2], alpha=0.15, label=f"patience : {patience} epochs sans amélioration")
    ax.scatter([best_epoch], [best], color=C[3], zorder=3, s=40, label=f"meilleure epoch ({best_epoch}) : poids rechargés")
    ax.axvline(stop, color=INK, lw=1, ls="--")
    ax.text(stop + 0.4, ax.get_ylim()[1] * 0.92, f"arrêt après\nl'epoch {stop}", fontsize=8, va="top", color=INK)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "early_stopping.png")


def main() -> int:
    figure_channels()
    figure_reshape()
    figure_fashion()
    figure_pipeline()
    figure_schedules()
    figure_early_stopping()
    print(f"✅ figures written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
