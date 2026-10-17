#!/usr/bin/env python
"""Draw the figures of chapter 20 (used by Claude).

    python tools/chapters/figures_ch20.py

Writes small PNG files into chapitres/ch20_deep_learning/figures/. Every figure is computed or drawn here
(never copied from the book): a dropout mask drawn again at each mini-batch, the axes over which batch
normalization and layer normalization compute their statistics, the product of the derivatives along a
deep chain with and without residual connections, and the PyTorch training loop. None of them uses the
numbers of an exercise (the residual figure draws random derivatives, not those of ✏️ 20.7).
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

OUT = ROOT / "chapitres" / "ch20_deep_learning" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
GREY = "#bdbdbd"
INK = wbplot.TEXT


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


# ------------------------------------------------------------------ 1. dropout: a new mask at every mini-batch
def fig_dropout() -> None:
    sizes = [3, 6, 6, 2]
    rng = np.random.default_rng(2020)
    masks = [[np.ones(3, bool)] + [rng.random(6) >= 0.5 for _ in range(2)] + [np.ones(2, bool)] for _ in range(2)]
    masks.append([np.ones(s, bool) for s in sizes])
    titles = ["mini-batch 1 (entraînement)", "mini-batch 2 (entraînement)", "évaluation : aucun neurone coupé"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, mask, title in zip(axes, masks, titles):
        pos = [[(x, (y - (n - 1) / 2) * 0.9) for y in range(n)] for x, n in enumerate(sizes)]
        for layer in range(len(sizes) - 1):
            for i, (x0, y0) in enumerate(pos[layer]):
                for j, (x1, y1) in enumerate(pos[layer + 1]):
                    alive = mask[layer][i] and mask[layer + 1][j]
                    ax.plot([x0, x1], [y0, y1], color=INK if alive else GREY, lw=0.6 if alive else 0.4,
                            alpha=0.6 if alive else 0.35, zorder=1)
        for layer, column in enumerate(pos):
            for i, (x, y) in enumerate(column):
                alive = mask[layer][i]
                ax.scatter(x, y, s=170, color=C[layer % len(C)] if alive else "white",
                           edgecolors=INK if alive else GREY, linewidths=1.2, zorder=2)
                if not alive:
                    ax.plot(x, y, marker="x", color=GREY, ms=7, zorder=3)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-0.4, 3.4)
        ax.set_ylim(-2.9, 2.9)
        ax.axis("off")
    fig.suptitle("Dropout à 50 % sur les deux couches cachées : un nouveau tirage à chaque passe avant", fontsize=10.5)
    save(fig, "dropout.png")


# ------------------------------------------------------------------ 2. which axis: batchnorm or layer norm
def fig_normalisations() -> None:
    n, d = 6, 8
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, kind in zip(axes, ("batch", "layer")):
        for i in range(n):
            for j in range(d):
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor="white", edgecolor=GREY, lw=0.8))
        if kind == "batch":
            ax.add_patch(Rectangle((2, 0), 1, n, facecolor=C[0], alpha=0.35, edgecolor=C[0], lw=2))
            ax.set_title("BatchNorm : une moyenne et une variance\npar feature, sur les exemples du mini-batch", fontsize=9.5)
        else:
            ax.add_patch(Rectangle((0, n - 3), d, 1, facecolor=C[1], alpha=0.35, edgecolor=C[1], lw=2))
            ax.set_title("LayerNorm : une moyenne et une variance\npar exemple, sur ses features", fontsize=9.5)
        ax.set_xlim(-0.2, d + 0.2)
        ax.set_ylim(-0.6, n + 0.2)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(d) + 0.5, [f"f{j + 1}" for j in range(d)], fontsize=8)
        ax.set_yticks(np.arange(n) + 0.5, [f"ex. {n - k}" for k in range(n)], fontsize=8)
        ax.tick_params(length=0)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.text(d / 2, -0.45, "features (axe 1)" if kind == "batch" else "features (dernier axe)",
                ha="center", fontsize=8.5, color=INK)
    fig.suptitle("Un mini-batch de 6 exemples × 8 features : quelles cases partagent leurs statistiques ?",
                 fontsize=10.5, y=1.07)
    save(fig, "normalisations.png")


# ------------------------------------------------------------------ 3. residual connections and the gradient
def fig_residual() -> None:
    rng = np.random.default_rng(7)
    depth = 60
    plain = rng.uniform(0.5, 1.1, depth)               # derivative of each plain layer
    branch = rng.uniform(-0.15, 0.15, depth)           # derivative of each residual branch F
    layers = np.arange(1, depth + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(layers, np.cumprod(plain), color=C[3], lw=1.8, label="sans connexion résiduelle : $\\prod f'_\\ell$")
    ax.plot(layers, np.cumprod(1 + branch), color=C[0], lw=1.8,
            label="avec : $\\prod (1 + F'_\\ell)$, $|F'_\\ell| \\leq 0{,}15$")
    ax.set_yscale("log")
    ax.set_xlabel("nombre de couches traversées en remontant")
    ax.set_ylabel("facteur appliqué au gradient (log)")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("Le « + x » d'une connexion résiduelle garde une voie directe au gradient", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    save(fig, "residuel.png")


# ------------------------------------------------------------------ 4. the PyTorch training loop
def fig_loop() -> None:
    steps = [("model.train()", "mode entraînement\n(dropout, batchnorm)"),
             ("for X, y in loader:", "un mini-batch"),
             ("X, y = X.to(device), y.to(device)", "sur le bon appareil"),
             ("logits = model(X)", "passe avant"),
             ("loss = loss_fn(logits, y)", "loss sur les logits"),
             ("optimizer.zero_grad()", "remettre les .grad à 0"),
             ("loss.backward()", "autograd : les gradients"),
             ("optimizer.step()", "mise à jour (ch. 19)")]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for k, (code_text, note) in enumerate(steps):
        y = len(steps) - 1 - k
        box = FancyBboxPatch((0.2, y * 0.62 + 0.05), 4.2, 0.46, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor="#f4f6fb" if k > 1 else "#eef7ee", edgecolor=C[0] if k > 1 else C[2], lw=1.1)
        ax.add_patch(box)
        ax.text(0.35, y * 0.62 + 0.28, code_text, family="monospace", fontsize=9.5, va="center", color=INK)
        ax.text(4.6, y * 0.62 + 0.28, note, fontsize=8.5, va="center", color=INK)
        if k < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((1.2, y * 0.62 + 0.05), (1.2, (y - 1) * 0.62 + 0.51), arrowstyle="-|>",
                                         mutation_scale=9, color=GREY))
    ax.add_patch(FancyArrowPatch((4.45, 0.3), (4.45, 6 * 0.62 + 0.28), connectionstyle="arc3,rad=-0.6",
                                 arrowstyle="-|>", mutation_scale=10, color=C[1], lw=1.2))
    ax.text(6.9, 2.2, "mini-batch\nsuivant", fontsize=8.5, color=C[1], ha="center")
    ax.text(0.2, -0.45, "puis, après chaque epoch : model.eval() et torch.no_grad() pour mesurer la validation",
            fontsize=8.5, color=INK)
    ax.set_xlim(0, 8.2)
    ax.set_ylim(-0.7, len(steps) * 0.62 + 0.1)
    ax.axis("off")
    ax.set_title("Une epoch d'entraînement en PyTorch", fontsize=10.5)
    save(fig, "boucle.png")


def main() -> int:
    fig_dropout()
    fig_normalisations()
    fig_residual()
    fig_loop()
    for p in sorted(OUT.glob("*.png")):
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size // 1024} ko")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
