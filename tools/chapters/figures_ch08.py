#!/usr/bin/env python
"""Draw the figures of chapter 8 (used by Claude).

    python tools/chapters/figures_ch08.py

Writes small PNG files into chapitres/ch08_train_test/figures/. Every figure is computed (never
copied from the book): the training flow against the test flow, three ways to split the data
(hold-out, train-validation-test, 4-fold cross-validation), the optimism of the best of K
equivalent settings, three splitters on 24 dependent samples (shuffled k-fold, groups, time
order), and the box plots of the 📈 exercise 8.8 of 02_exercices.md. None of them uses the data
or the settings of a notebook exercise (no penguins, no California, no sunspots, no noise
features: those are 8.11 to 8.27), and the optimism figure uses other numbers than ∂ 8.6.
"""

from __future__ import annotations

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

OUT = ROOT / "chapitres" / "ch08_train_test" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
TRAIN, VAL, TEST, UNUSED = C[0], C[1], "#9e9e9e", "#ececec"


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


# ---------------------------------------------------------------------------- the two flows
def _box(ax, xy, text, color, width=2.3, height=0.9):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.08",
                                facecolor=color, edgecolor="#424242", lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5)


def _arrow(ax, start, end, color="#424242", style="-|>", rad=0.0, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_flows() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    light_train, light_test = "#dbe9f6", "#eeeeee"
    for ax, title in zip(axes, ["(a) Entraînement : l'erreur sert à corriger le modèle",
                                "(b) Test : l'erreur sert seulement à compter"]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6.2)
        ax.axis("off")
        ax.set_title(title, fontsize=11)
    ax = axes[0]
    _box(ax, (1.6, 5.0), "exemple\nd'entraînement", light_train)
    _box(ax, (5.0, 5.0), "modèle\n(paramètres)", light_train)
    _box(ax, (8.4, 5.0), "prédiction", light_train)
    _box(ax, (8.4, 2.6), "comparaison\navec le label", light_train)
    _box(ax, (5.0, 1.0), "optimiseur\n(mise à jour)", "#fde3c8")
    _box(ax, (1.6, 2.6), "label", light_train, width=1.6)
    _arrow(ax, (2.8, 5.0), (3.8, 5.0))
    _arrow(ax, (6.2, 5.0), (7.2, 5.0))
    _arrow(ax, (8.4, 4.5), (8.4, 3.1))
    _arrow(ax, (2.45, 2.6), (7.2, 2.6))
    _arrow(ax, (8.4, 2.1), (6.2, 1.15), rad=-0.15)
    _arrow(ax, (5.0, 1.5), (5.0, 4.5), color=VAL, lw=2.2)
    ax.text(5.25, 3.25, "les paramètres\nchangent", color=VAL, fontsize=9, va="center")
    ax = axes[1]
    _box(ax, (1.6, 5.0), "exemple\nde test", light_test)
    _box(ax, (5.0, 5.0), "modèle\n(figé)", light_test)
    _box(ax, (8.4, 5.0), "prédiction", light_test)
    _box(ax, (8.4, 2.6), "comparaison\navec le label", light_test)
    _box(ax, (5.0, 1.0), "compteur\n(accuracy)", "#e6f4e6")
    _box(ax, (1.6, 2.6), "label", light_test, width=1.6)
    _arrow(ax, (2.8, 5.0), (3.8, 5.0))
    _arrow(ax, (6.2, 5.0), (7.2, 5.0))
    _arrow(ax, (8.4, 4.5), (8.4, 3.1))
    _arrow(ax, (2.45, 2.6), (7.2, 2.6))
    _arrow(ax, (8.4, 2.1), (6.2, 1.15), rad=-0.15)
    ax.text(5.0, 3.25, "aucune flèche ne remonte\nvers le modèle", color="#616161", fontsize=9, ha="center",
            va="center", style="italic")
    save(fig, "boucle.png")


# ---------------------------------------------------------------------------- three ways to split
def _segment(ax, y, start, stop, color, text="", height=0.62, text_color="black"):
    ax.barh(y, stop - start, left=start, height=height, color=color, edgecolor="white", lw=1.5)
    if text:
        ax.text((start + stop) / 2, y, text, ha="center", va="center", fontsize=8.5, color=text_color)


def fig_splits() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    rows = []
    y = 7.0
    _segment(ax, y, 0, 75, TRAIN, "entraînement (75 %)", text_color="white")
    _segment(ax, y, 75, 100, TEST, "test (25 %)", text_color="white")
    rows.append((y, "hold-out"))
    y = 6.0
    _segment(ax, y, 0, 60, TRAIN, "entraînement (60 %)", text_color="white")
    _segment(ax, y, 60, 80, VAL, "validation (20 %)")
    _segment(ax, y, 80, 100, TEST, "test (20 %)", text_color="white")
    rows.append((y, "entraînement,\nvalidation, test"))
    for j in range(4):
        y = 4.6 - j
        for fold in range(4):
            start, stop = 20 * fold, 20 * (fold + 1)
            if fold == j:
                _segment(ax, y, start, stop, VAL, f"fold {fold + 1}")
            else:
                _segment(ax, y, start, stop, TRAIN, f"fold {fold + 1}", text_color="white")
        _segment(ax, y, 80, 100, TEST, "test (mis de côté)", text_color="white")
        rows.append((y, f"validation croisée\ntour {j + 1}"))
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("part des données (%)")
    ax.axhline(5.25, color="#bdbdbd", lw=0.8, ls="--")
    ax.set_title("Trois façons de découper les mêmes données", fontsize=11)
    ax.legend(handles=[Patch(color=TRAIN, label="entraînement"), Patch(color=VAL, label="validation"),
                       Patch(color=TEST, label="test")], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "decoupages.png")


# ---------------------------------------------------------------------------- optimism of the best of K
def fig_optimism() -> None:
    p_true, n_val, n_sim = 0.75, 150, 20_000
    rng = np.random.default_rng(81)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(0.62, 0.90, 1 / n_val) - 0.5 / n_val
    for K, color in zip([1, 5, 25], [C[0], C[1], C[3]]):
        scores = rng.binomial(n_val, p_true, size=(n_sim, K)) / n_val
        best = scores.max(axis=1)
        ax.hist(best, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"meilleur de K = {K} (moyenne {fr(best.mean(), 3)})")
    ax.axvline(p_true, color="black", lw=1.5, ls="--", label=f"vraie accuracy de chaque réglage : {fr(p_true, 2)}")
    ax.set_xlabel(f"score de validation retenu ({n_val} exemples de validation)")
    ax.set_ylabel("densité")
    ax.set_title("K réglages aussi bons les uns que les autres : on garde le meilleur score de validation", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "optimisme.png")


# ---------------------------------------------------------------------------- dependent samples
def _index_rows(ax, splits, n, title, groups=None):
    for j, (train, val) in enumerate(splits):
        colors = np.array([UNUSED] * n, dtype=object)
        colors[train] = TRAIN
        colors[val] = VAL
        ax.scatter(np.arange(n), np.full(n, j), c=list(colors), marker="s", s=70, edgecolor="white")
    if groups is not None:
        palette = ["#7e57c2", "#26a69a", "#ef5350", "#ffa726"]
        ax.scatter(np.arange(n), np.full(n, len(splits)), c=[palette[g] for g in groups], marker="s", s=70,
                   edgecolor="white")
    ax.set_yticks(list(range(len(splits))) + ([len(splits)] if groups is not None else []),
                  [f"tour {j + 1}" for j in range(len(splits))] + (["groupe"] if groups is not None else []))
    ax.set_ylim(len(splits) + 0.7, -0.7)
    ax.set_xlabel("indice de l'exemple (ordre chronologique)")
    ax.set_title(title, fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fig_dependent() -> None:
    from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

    n = 24
    groups = np.repeat(np.arange(4), 6)
    X = np.zeros((n, 1))
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.6))
    _index_rows(axes[0], list(KFold(4, shuffle=True, random_state=3).split(X)), n,
                "(a) k-fold mélangée : des voisins (et des membres d'un même groupe) des deux côtés", groups)
    _index_rows(axes[1], list(GroupKFold(4).split(X, groups=groups)), n,
                "(b) GroupKFold : chaque groupe entier dans un seul fold de validation", groups)
    _index_rows(axes[2], list(TimeSeriesSplit(4).split(X)), n,
                "(c) TimeSeriesSplit : l'entraînement est toujours avant la validation")
    fig.legend(handles=[Patch(color=TRAIN, label="entraînement"), Patch(color=VAL, label="validation"),
                        Patch(color=UNUSED, label="non utilisé à ce tour")], loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "donnees_dependantes.png")


# ---------------------------------------------------------------------------- 📈 8.8 (02_exercices.md)
def scores_88():
    """30 validation scores (10 folds repeated 3 times, the same folds for the three models)."""
    rng = np.random.default_rng(806)
    fold = rng.normal(0.0, 0.018, 30)
    fold[11] = -0.08                                        # a hard fold: A and B both drop
    a = 0.842 + fold + rng.normal(0.0, 0.005, 30)
    gain = np.concatenate([rng.uniform(0.005, 0.03, 25), rng.uniform(-0.02, -0.005, 5)])
    b = a + rng.permutation(gain)                           # B beats A on 25 folds, by at least 0.005
    c = 0.90 + rng.normal(0.0, 0.04, 30)                   # C: the best median and the most spread out...
    bad = rng.choice(30, size=5, replace=False)
    c[bad] -= rng.uniform(0.14, 0.20, 5)                   # ...with a few very bad folds, below A and B
    return a, b, c


def fig_boxes_88() -> None:
    a, b, c = scores_88()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.boxplot([a, b, c], tick_labels=["A", "B", "C"], widths=0.5, medianprops={"color": "black", "lw": 1.8},
               flierprops={"marker": "o", "markerfacecolor": C[3], "markeredgecolor": C[3], "markersize": 6})
    ax.set_ylabel("accuracy de validation")
    ax.set_title("(a) 30 scores par modèle (10 folds, 3 répétitions)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    diff = b - a
    ax.axhline(0, color="black", lw=1)
    ax.scatter(np.arange(1, 31), diff, color=C[0], s=28, zorder=3)
    ax.set_xlabel("fold (même découpage pour A et B)")
    ax.set_ylabel("score de B − score de A")
    ax.set_title("(b) B − A, fold par fold", fontsize=10)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(axis="y", alpha=0.3)
    save(fig, "boites_8_8.png")


def main() -> int:
    print(f"Figures of chapter 8 -> {OUT.relative_to(ROOT)}")
    fig_flows()
    fig_splits()
    fig_optimism()
    fig_dependent()
    fig_boxes_88()
    return 0


if __name__ == "__main__":
    sys.exit(main())
