#!/usr/bin/env python
"""Draw the figures of chapter B1 (used by Claude).

    python tools/chapters/figures_chB1.py

Writes small PNG files into chapitres/b1_transfer/figures/. Every figure is drawn or computed here, without any
training (each runs in a second or two, on CPU, offline): the three transfer strategies on a backbone and its head,
a decision matrix (dataset size x domain distance), a warmup + cosine schedule with discriminative learning rates and
a gradual unfreezing, the zero-shot classification of CLIP, and the ImageNet accuracy of torchvision backbones against
their cost, read from ``weights.meta``. None of them uses the numbers or the models of an exercise: the learning
rates, cosines and backbones differ from those of 02_exercices.md (ResNet-18 and MobileNetV3-small are left out of
the backbone chart on purpose, the notebook measures them).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "b1_transfer" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED
FROZEN = "#9ec5f4"          # light blue: frozen weights
TRAINED = C[1]              # orange: trained weights
NEW = C[7]                  # red: a new layer, randomly initialised


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


def arrow(ax, start, end, color=INK, style="-|>", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def box(ax, xy, w, h, text, color, text_color=INK, fontsize=9, lw=1.0, ec=None, hatch=None):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color,
                                ec=ec or INK, lw=lw, hatch=hatch))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def figure_strategies() -> None:
    """One backbone (stem, four blocks), one new head: what is frozen and what is trained in each strategy."""
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    names = ["stem", "bloc 1", "bloc 2", "bloc 3", "bloc 4"]
    w, h, gap = 1.25, 0.9, 0.28
    head_w, in_w = 1.55, 1.15
    rows = [
        ("Extraction de features\n(linear probe, k-NN)", [False] * 5, "embeddings calculés une fois,\nmis en cache"),
        ("Fine-tuning partiel\n(dernier bloc dégelé)", [False, False, False, False, True],
         "bloc 4 : learning rate\nplus petit que la tête"),
        ("Fine-tuning complet", [True] * 5, "learning rates discriminatifs :\nplus petits vers l'entrée"),
    ]
    for r, (title, trained, note) in enumerate(rows):
        y = 2 * (len(rows) - 1 - r)
        ax.text(-in_w + 0.6, y + h / 2, title, ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold")
        box(ax, (-in_w + 0.8, y), in_w, h, "image\n(N, 3, H, W)", "white", fontsize=7.5, ec=MUTED)
        x = 0.8 + gap
        arrow(ax, (0.8, y + h / 2), (x, y + h / 2))
        for k, (name, on) in enumerate(zip(names, trained)):
            if on and r == 2:                       # full fine-tuning: shade from pale to deep orange
                alpha = 0.35 + 0.13 * k
                color = matplotlib.colors.to_hex(np.array(matplotlib.colors.to_rgb(TRAINED)) * alpha
                                                 + (1 - alpha) * np.ones(3))
            else:
                color = TRAINED if on else FROZEN
            box(ax, (x, y), w, h, name + ("" if on else "\ngelé"), color, fontsize=8.5)
            if k < len(names) - 1:
                arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2))
            x += w + gap
        ax.text(x - gap / 2, y + h + 0.12, "φ(x)", ha="center", fontsize=9, color=INK)
        arrow(ax, (x - gap, y + h / 2), (x + 0.15, y + h / 2))
        box(ax, (x + 0.15, y), head_w, h, "nouvelle\ntête\nd → K", NEW, text_color="white", fontsize=8)
        ax.text(x + head_w + 0.3, y + h / 2, note, ha="left", va="center", fontsize=8, color=MUTED)
    ax.text(0.8 + gap + 2.5 * (w + gap) - gap / 2, -0.55,
            "backbone pré-entraîné (poids ImageNet)", ha="center", fontsize=9, color=MUTED)
    ax.annotate("", xy=(0.8 + gap, -0.3), xytext=(0.8 + gap + 5 * (w + gap) - gap, -0.3),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.legend(handles=[Patch(fc=FROZEN, ec=INK, label="gelé (requires_grad = False)"),
                       Patch(fc=TRAINED, ec=INK, label="entraîné"),
                       Patch(fc=NEW, ec=INK, label="nouvelle couche, initialisée au hasard")],
              loc="upper center", bbox_to_anchor=(0.55, -0.02), ncol=3, fontsize=8.5)
    ax.set_xlim(-3.6, 14.2)
    ax.set_ylim(-0.9, 5.2)
    ax.axis("off")
    save(fig, "strategies.png")


def figure_decision() -> None:
    """Dataset size x domain distance: the usual first choice in each corner."""
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    cells = {
        (0, 1): ("Peu de données, domaine proche",
                 "extraction de features :\nlinear probe, k-NN, prototypes ;\npuis, si le temps le permet,\n"
                 "fine-tuning partiel prudent", C[2]),
        (1, 1): ("Beaucoup de données, domaine proche",
                 "fine-tuning complet,\npetit learning rate, warmup ;\nle pré-entraînement fait\n"
                 "surtout gagner du temps", C[0]),
        (0, 0): ("Peu de données, domaine lointain",
                 "le cas difficile : sonder les blocs\nintermédiaires, fine-tuning partiel\n"
                 "régularisé, augmentation ;\nchercher un modèle plus proche", C[7]),
        (1, 0): ("Beaucoup de données, domaine lointain",
                 "fine-tuning complet ;\ncomparer à un entraînement\nfrom scratch : l'écart\npeut disparaître", C[3]),
    }
    for (i, j), (title, text, color) in cells.items():
        x, y = 1.0 + 4.2 * i, 0.4 + 2.9 * j
        ax.add_patch(FancyBboxPatch((x, y), 3.9, 2.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=matplotlib.colors.to_rgba(color, 0.16), ec=color, lw=1.6))
        ax.text(x + 1.95, y + 2.3, title, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(x + 1.95, y + 1.1, text, ha="center", va="center", fontsize=8.6, color=INK)
    arrow(ax, (1.0, 0.1), (9.3, 0.1), color=MUTED)
    ax.text(5.15, -0.25, "taille du dataset cible (images étiquetées par classe) →", ha="center", fontsize=9,
            color=MUTED)
    arrow(ax, (0.7, 0.4), (0.7, 6.1), color=MUTED)
    ax.text(0.45, 3.25, "ressemblance avec les données du pré-entraînement →", rotation=90, ha="center",
            va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0.1, 9.5)
    ax.set_ylim(-0.5, 6.3)
    ax.axis("off")
    save(fig, "matrice_decision.png")


def warmup_cosine(t: np.ndarray, t_warmup: int, t_total: int) -> np.ndarray:
    """Multiplier of the learning rate: min(1, t / t_warmup), then a half cosine down to 0 at t_total."""
    up = np.minimum(1.0, t / t_warmup)
    progress = np.clip((t - t_warmup) / (t_total - t_warmup), 0, 1)
    return np.where(t < t_warmup, up, 0.5 * (1 + np.cos(np.pi * progress)))


def figure_schedule() -> None:
    """Left: discriminative learning rates (gamma = 0.5) with a warmup and a cosine. Right: gradual unfreezing."""
    head_lr, gamma = 2e-3, 0.5
    groups = ["tête", "bloc 4", "bloc 3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    t = np.arange(0, 61)
    factor = warmup_cosine(t, 6, 60)
    ax = axes[0]
    for k, name in enumerate(groups):
        lr = head_lr * gamma ** k
        ax.plot(t, lr * factor * 1e3, color=C[k], lw=2,
                label=f"{name} : η = {french_number(lr * 1e3)} × 10⁻³ au sommet")
    ax.axvspan(0, 6, color=wbplot.GRID, alpha=0.6, lw=0)
    ax.text(3, 2.08, "warmup", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xlabel("pas t de l'optimiseur")
    ax.set_ylabel("learning rate (× 10⁻³)")
    ax.set_ylim(0, 2.25)
    ax.set_title("Learning rates discriminatifs, warmup puis cosinus", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[1]
    epochs = np.arange(0, 9)
    start = {"tête": 0, "bloc 4": 2, "bloc 3": 4}
    for k, name in enumerate(groups):
        lr = np.where(epochs >= start[name], head_lr * gamma ** k, 0.0) * 1e3
        ax.step(np.append(epochs, 9), np.append(lr, lr[-1]), where="post", color=C[k], lw=2, label=name)
    for e, name in [(2, "dégel du bloc 4"), (4, "dégel du bloc 3")]:
        ax.axvline(e, color=MUTED, lw=0.8, ls=":")
        ax.text(e + 0.08, 2.12, name, fontsize=8, color=MUTED, va="top")
    ax.set_xlabel("epoch (comptée à partir de 0)")
    ax.set_ylabel("learning rate (× 10⁻³)")
    ax.set_ylim(0, 2.25)
    ax.set_xlim(0, 9)
    ax.set_title("Dégel progressif : un groupe de plus toutes les deux epochs", fontsize=10)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.01, 0.62))
    fig.tight_layout()
    save(fig, "planning_lr.png")


def figure_clip() -> None:
    """Zero-shot CLIP: one image, three prompts, cosines, scale s = 100, softmax (the toy numbers of 201.8)."""
    cos = np.array([0.31, 0.27, 0.22])
    s = 100.0
    z = s * cos
    probs = np.exp(z - z.max())
    probs /= probs.sum()
    labels = ["cat", "dog", "truck"]
    fig = plt.figure(figsize=(11, 4.4))
    ax = fig.add_axes([0, 0, 0.66, 1])
    box(ax, (0.0, 3.0), 1.5, 1.0, "image", "white", fontsize=9, ec=MUTED)
    arrow(ax, (1.5, 3.5), (2.0, 3.5))
    box(ax, (2.0, 3.0), 1.9, 1.0, "encodeur\nd'image", C[0], text_color="white", fontsize=9)
    arrow(ax, (3.9, 3.5), (4.4, 3.5))
    ax.text(4.75, 3.5, "u", fontsize=12, fontweight="bold", ha="center", va="center", color=INK)
    for k, lab in enumerate(labels):
        y = 1.6 - 0.75 * k
        box(ax, (0.0, y), 1.9, 0.55, f"« a photo of a {lab} »", "white", fontsize=8, ec=MUTED)
        arrow(ax, (1.9, y + 0.27), (2.4, 1.0))
        ax.text(4.75, y + 0.27, "v₀v₁v₂"[2 * k:2 * k + 2], fontsize=11, fontweight="bold", ha="center", va="center", color=INK)
        arrow(ax, (3.9, 1.0), (4.45, y + 0.27), rad=0.0)
    box(ax, (2.4, 0.6), 1.5, 0.8, "encodeur\nde texte", C[1], text_color="white", fontsize=9)
    box(ax, (5.4, 1.2), 2.2, 2.6, "cos(u, v₀) = 0,31\ncos(u, v₁) = 0,27\ncos(u, v₂) = 0,22\n\n× s = 100\n"
        "puis softmax", "#f0efec", fontsize=8.8, ec=MUTED)
    arrow(ax, (5.0, 3.5), (5.4, 3.0))
    arrow(ax, (5.0, 1.0), (5.4, 2.0))
    arrow(ax, (7.6, 2.5), (8.1, 2.5))
    ax.text(0.0, 4.35, "u et chaque v sont normalisés (norme 1) : leur produit scalaire est leur cosinus", fontsize=8.5,
            color=MUTED)
    ax.set_xlim(-0.1, 8.2)
    ax.set_ylim(-0.3, 4.6)
    ax.axis("off")
    bx = fig.add_axes([0.71, 0.2, 0.27, 0.62])
    bx.bar(range(3), probs, color=[C[0], C[1], C[2]])
    for k, p in enumerate(probs):
        bx.text(k, p + 0.02, french_number(round(float(p), 4)), ha="center", fontsize=8.5, color=INK)
    bx.set_xticks(range(3), labels)
    bx.set_ylim(0, 1.1)
    bx.set_ylabel("p(c | x)")
    bx.set_title("Probabilités (s = 100)", fontsize=9.5)
    save(fig, "clip_zero_shot.png")


def figure_backbones() -> None:
    """ImageNet top-1 accuracy against GFLOPS (log scale), read from weights.meta of torchvision 0.26."""
    from torchvision.models import get_model_weights

    names = ["mobilenet_v3_large", "efficientnet_b0", "regnet_y_400mf", "resnet34", "resnet50", "resnet101",
             "densenet121", "convnext_tiny", "convnext_base", "efficientnet_v2_s", "swin_t", "vit_b_16", "vgg16"]
    OFFSETS = {"mobilenet_v3_large": (-20, -22), "regnet_y_400mf": (7, 0), "resnet50": (-14, -24),
               "swin_t": (8, 2), "efficientnet_v2_s": (-78, -6), "convnext_base": (9, -6), "efficientnet_b0": (7, 2)}
    rows = []
    for name in names:
        meta = get_model_weights(name).DEFAULT.meta
        rows.append((name, meta["_ops"], meta["_metrics"]["ImageNet-1K"]["acc@1"], meta["num_params"] / 1e6))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for k, (name, ops, acc, params) in enumerate(rows):
        ax.scatter(ops, acc, s=12 + 3.2 * params, color=C[k % 8], alpha=0.75, edgecolor=INK, lw=0.6, zorder=3)
        ax.annotate(f"{name}\n{french_number(round(params, 1))} M", (ops, acc), textcoords="offset points",
                    xytext=OFFSETS.get(name, (7, -3)), fontsize=7.4, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20], ["0,2", "0,5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("GFLOPS par image (weights.meta['_ops'], échelle log)")
    ax.set_ylabel("accuracy top-1 ImageNet (%)")
    ax.set_title("Backbones de torchvision : accuracy et coût (aire ∝ paramètres)", fontsize=10)
    ax.set_xlim(0.12, 40)
    save(fig, "backbones.png")


def main() -> int:
    figure_strategies()
    figure_decision()
    figure_schedule()
    figure_clip()
    figure_backbones()
    print(f"✅ figures written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
