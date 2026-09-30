#!/usr/bin/env python
"""Draw the figures of the course sheet of chapter 2 (used by Claude).

    python tools/chapters/figures_ch02.py

Writes small PNG files into chapitres/ch02_stats/figures/. Every figure is computed
from the workbook's data or from synthetic data (never copied from the book) and
uses the workbook style.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import datasets  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch02_stats" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma, a thin space for thousands and a true minus sign."""
    text = f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")
    return text


def fr(value: float, decimals: int) -> str:
    """A number written the French way, for titles and legends."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:  # decimal commas on every numeric linear axis (not on named ticks)
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig_mean_median() -> None:
    """Central tendencies on two real distributions: a skewed one and a bimodal one."""
    df = datasets.load_penguins().dropna(subset=MEASURES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, column, unit, title in [
        (axes[0], "body_mass_g", "g", "masse : une bosse penchée à droite"),
        (axes[1], "flipper_length_mm", "mm", "nageoire : deux bosses"),
    ]:
        x = df[column].to_numpy()
        ax.hist(x, bins=24, color=C[0], alpha=0.55, edgecolor="white")
        mean, median = x.mean(), np.median(x)
        ax.axvline(mean, color=C[1], lw=2.2, label="moyenne")          # no values: exercise 2.13 computes them
        ax.axvline(median, color=C[2], lw=2.2, ls="--", label="médiane")
        ax.set_xlabel(f"{column} ({unit})")
        ax.set_ylabel("nombre de manchots")
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=9)
    fig.suptitle(f"{len(df)} manchots : la moyenne et la médiane ne racontent pas la même chose", fontsize=11)
    save(fig, "moyenne_mediane.png")


def fig_discrete_distribution() -> None:
    """From counts to a probability distribution, then to a wheel (islands of the penguins)."""
    counts = datasets.load_penguins()["island"].value_counts()
    names, values = counts.index.tolist(), counts.to_numpy()
    probs = values / values.sum()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 0.85]})
    axes[0].bar(names, values, color=C[:3])
    for i, v in enumerate(values):
        axes[0].text(i, v + 3, str(v), ha="center", fontsize=9)
    axes[0].set_title(f"comptages : {values.sum()} manchots", fontsize=10.5)
    axes[0].set_ylabel("nombre de manchots")
    axes[1].bar(names, probs, color=C[:3])
    for i, v in enumerate(probs):
        axes[1].text(i, v + 0.01, fr(v, 3), ha="center", fontsize=9)
    axes[1].set_title("divisés par le total : la somme vaut 1", fontsize=10.5)
    axes[1].set_ylabel("probabilité")
    axes[2].pie(probs, labels=[f"{n}\n{fr(100 * p, 1)} %" for n, p in zip(names, probs)], colors=C[:3],
                startangle=90, counterclock=False, wedgeprops={"edgecolor": "white"}, textprops={"fontsize": 9})
    axes[2].set_title("la roue : chaque part suit sa probabilité", fontsize=10.5)
    fig.suptitle("« Je tire un manchot au hasard : de quelle île vient-il ? »", fontsize=11)
    save(fig, "distribution_discrete.png")


def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def fig_normal() -> None:
    """The 68-95-99.7 rule, and a density whose values exceed 1 (area 1 all the same)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    grid = np.linspace(-4, 4, 800)
    ax.plot(grid, normal_pdf(grid), color=wbplot.TEXT, lw=1.8)
    for k, alpha, share in [(3, 0.18, "99,7 %"), (2, 0.3, "95 %"), (1, 0.45, "68 %")]:
        band = (grid >= -k) & (grid <= k)
        ax.fill_between(grid[band], normal_pdf(grid[band]), color=C[0], alpha=alpha, lw=0)
    for k, y, share in [(1, 0.2, "68 %"), (2, 0.07, "95 %"), (3, 0.015, "99,7 %")]:
        ax.annotate("", xy=(-k, y), xytext=(k, y), arrowprops={"arrowstyle": "<->", "color": wbplot.TEXT_MUTED})
        ax.text(0, y + 0.012, share, ha="center", fontsize=9.5, color=wbplot.TEXT)
    ax.set_xticks(range(-3, 4))
    ax.set_xticklabels(["μ − 3σ", "μ − 2σ", "μ − σ", "μ", "μ + σ", "μ + 2σ", "μ + 3σ"])
    ax.set_ylabel("densité")
    ax.set_title("loi normale : part des tirages à moins de 1, 2 ou 3 écarts-types de la moyenne", fontsize=10)
    ax = axes[1]
    grid = np.linspace(-1, 1, 600)
    ax.plot(grid, normal_pdf(grid, 0, 0.2), color=C[1], lw=2, label="σ = 0,2 : densité jusqu'à ≈ 2")
    ax.plot(grid, normal_pdf(grid, 0, 0.5), color=C[0], lw=2, label="σ = 0,5 : densité ≤ 0,8")
    ax.fill_between(grid, normal_pdf(grid, 0, 0.2), color=C[1], alpha=0.15, lw=0)
    ax.axhline(1, color=wbplot.TEXT_MUTED, lw=1, ls=":")
    ax.set_title("une densité peut dépasser 1 ; c'est l'aire\nsous la courbe qui vaut 1", fontsize=10)
    ax.set_ylabel("densité")
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "densite_normale.png")


def fig_bootstrap() -> None:
    """Bootstrap distribution of a mean and its 95 % percentile interval (synthetic commute times)."""
    rng = np.random.default_rng(7)
    population_mean = 25.0
    sample = rng.gamma(shape=4.0, scale=population_mean / 4.0, size=40)     # 40 skewed commute times (min)
    boot = np.array([sample[rng.integers(0, len(sample), len(sample))].mean() for _ in range(5000)])
    low, high = np.percentile(boot, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.hist(boot, bins=40, color=C[0], alpha=0.55, edgecolor="white")
    ax.axvspan(low, high, color=C[2], alpha=0.12, label=f"intervalle à 95 % : [{fr(low, 1)} ; {fr(high, 1)}]")
    ax.axvline(sample.mean(), color=C[1], lw=2.2, label=f"moyenne de l'échantillon : {fr(sample.mean(), 1)}")
    ax.axvline(population_mean, color=wbplot.TEXT, lw=1.6, ls=":", label=f"vraie moyenne (inconnue en vrai) : {fr(population_mean, 0)}")
    ax.set_xlabel("moyenne d'un rééchantillon (minutes)")
    ax.set_ylabel("nombre de rééchantillons")
    ax.set_title("40 temps de trajet, 5 000 rééchantillons de 40 tirés avec remise", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "bootstrap.png")


def fig_correlations() -> None:
    """Clouds of points and their correlation; the last one depends on x but has r close to 0."""
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(0, 1, n)
    clouds = []
    for target in (0.95, 0.6, 0.0, -0.6, -0.95):
        y = target * x + np.sqrt(1 - target ** 2) * rng.normal(0, 1, n)
        clouds.append((x, y))
    t = rng.uniform(-1.6, 1.6, n)
    clouds.append((t, t ** 2 + rng.normal(0, 0.12, n)))
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.7))
    for i, (ax, (a, b)) in enumerate(zip(axes, clouds)):
        r = np.corrcoef(a, b)[0, 1]
        ax.scatter(a, b, s=6, color=C[1] if i == 5 else C[0], alpha=0.7)
        ax.set_title(f"r = {fr(r, 2)}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[5].set_title(f"r = {fr(np.corrcoef(*clouds[5])[0, 1], 2)}, et pourtant\ny dépend de x", fontsize=9.5)
    save(fig, "correlations.png")


ANSCOMBE_X = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ANSCOMBE = {
    "I": (ANSCOMBE_X, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (ANSCOMBE_X, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (ANSCOMBE_X, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}


def fig_anscombe() -> None:
    """Anscombe's quartet (1973): four datasets with the same summary statistics."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True, sharey=True)
    grid = np.array([2, 20])
    for ax, (name, (x, y)) in zip(axes, ANSCOMBE.items()):
        x, y = np.array(x, float), np.array(y, float)
        slope, intercept = np.polyfit(x, y, 1)
        ax.scatter(x, y, s=22, color=C[0], zorder=3)
        ax.plot(grid, intercept + slope * grid, color=C[1], lw=1.6)
        ax.set_title(f"{name} : r = {fr(np.corrcoef(x, y)[0, 1], 2)}", fontsize=10)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle("Quartet d'Anscombe : mêmes moyennes, mêmes écarts-types, même corrélation, même droite", fontsize=11, y=1.06)
    save(fig, "anscombe.png")


def fig_image_point() -> None:
    """An MNIST image is a list of 784 numbers: one point in a space with 784 dimensions."""
    X, y = datasets.load_mnist("train", n=None)
    image = X[0].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 2.9), gridspec_kw={"width_ratios": [1, 4.2]})
    axes[0].imshow(image, cmap="gray_r")
    axes[0].set_title(f"une image 28 × 28 (un {y[0]})", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for side in axes[0].spines.values():
        side.set_visible(False)
    axes[1].bar(np.arange(784), image.ravel(), width=1.0, color=C[0])
    axes[1].set_xlim(0, 784)
    axes[1].set_xlabel("numéro du pixel (ligne après ligne) : 784 coordonnées")
    axes[1].set_ylabel("intensité (0 à 255)")
    axes[1].set_title("la même image, vue comme un vecteur : un point de l'espace à 784 dimensions", fontsize=10)
    save(fig, "image_point.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    fig_mean_median()
    fig_discrete_distribution()
    fig_normal()
    fig_bootstrap()
    fig_correlations()
    fig_anscombe()
    fig_image_point()
    return 0


if __name__ == "__main__":
    sys.exit(main())
