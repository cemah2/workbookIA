#!/usr/bin/env python
"""Draw the figures of the course sheet of chapter 4 (used by Claude).

    python tools/chapters/figures_ch04.py

Writes small PNG files into chapitres/ch04_bayes/figures/. Every figure is computed
from the running examples of the sheet (never copied from the book): the fair coin and
the coin of bias 0.9, the posterior-prior loop, a run of 40 flips, five then 500
hypotheses on the bias of a coin, and the Beta posterior with its credible interval.
None of them uses the numbers of an exercise.
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
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch04_bayes" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
FAIR, RIGGED = C[0], C[1]          # blue for the fair coin, orange for the rigged one


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
    print(f"  {name}")


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
def fig_wall() -> None:
    """The painted wall of the sheet's example: fair coin or coin of bias 0.9, then heads is observed."""
    p_fair, bias = 0.5, 0.9
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    titles = ["(a) le choix de la pièce : le prior",
              "(b) chaque zone découpée en face / pile :\nles vraisemblances",
              "(c) on a vu face : seules les zones « face »\nrestent possibles"]
    for k, (ax, title) in enumerate(zip(axes, titles)):
        zones = [  # (x, y, width, height, colour, label, is_heads)
            (0, 0, p_fair, 1, FAIR, "équilibrée", None),
            (p_fair, 0, 1 - p_fair, 1, RIGGED, "truquée (biais 0,9)", None),
        ] if k == 0 else [
            (0, 0.5, p_fair, 0.5, FAIR, "équilibrée, face\naire 0,25", True),
            (0, 0, p_fair, 0.5, FAIR, "équilibrée, pile\naire 0,25", False),
            (p_fair, 1 - bias, 1 - p_fair, bias, RIGGED, "truquée, face\naire 0,45", True),
            (p_fair, 0, 1 - p_fair, 1 - bias, RIGGED, "truquée, pile : 0,05", False),
        ]
        for x, y, w, h, colour, label, heads in zones:
            faded = k == 2 and heads is False
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#efeeea" if faded else colour,
                                   alpha=1 if faded else (0.30 if heads is not False else 0.16),
                                   edgecolor="white", lw=3))
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5,
                    color=wbplot.TEXT_MUTED if faded else wbplot.TEXT)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        _bare(ax)
        ax.set_title(title, fontsize=10, weight="normal")
    axes[2].text(0.5, -0.14, "P(équilibrée | face) = 0,25 / (0,25 + 0,45) ≈ 0,357", ha="center", va="top",
                 fontsize=10, transform=axes[2].transAxes)
    axes[0].text(0.5, -0.14, "P(équilibrée) = P(truquée) = 0,5", ha="center", va="top", fontsize=10,
                 transform=axes[0].transAxes)
    axes[1].text(0.5, -0.14, "P(face | équilibrée) = 0,5 ; P(face | truquée) = 0,9", ha="center", va="top",
                 fontsize=10, transform=axes[1].transAxes)
    fig.suptitle("Le mur peint : la probabilité d'une zone est son aire ; observer « face », c'est ne garder "
                 "que les zones « face » et renormaliser", fontsize=10.5, y=1.02)
    save(fig, "mur_bayes.png")


def fig_loop() -> None:
    """The posterior-prior loop as a diagram."""
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    _bare(ax)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.75, 3.6)
    boxes = {
        "prior": (0.3, 1.45, "prior\nP(H)", C[0]),
        "joint": (3.75, 1.45, "produit\nP(o | H) · P(H)", C[3]),
        "post": (7.2, 1.45, "posterior\nP(H | o)", C[2]),
    }
    for x, y, text, colour in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), 2.7, 1.0, boxstyle="round,pad=0.08", facecolor=colour, alpha=0.18,
                                    edgecolor=colour, lw=1.6))
        ax.text(x + 1.35, y + 0.5, text, ha="center", va="center", fontsize=11)
    arrow = dict(arrowstyle="-|>", mutation_scale=16, color=wbplot.TEXT_MUTED, lw=1.5)
    ax.add_patch(FancyArrowPatch((3.05, 1.95), (3.7, 1.95), **arrow))
    ax.add_patch(FancyArrowPatch((6.5, 1.95), (7.15, 1.95), **arrow))
    ax.text(3.37, 2.55, "× vraisemblance\nde l'observation o", ha="center", va="bottom", fontsize=9.3,
            color=wbplot.TEXT_MUTED)
    ax.text(6.83, 2.55, "÷ évidence P(o)\n(la somme des produits)", ha="center", va="bottom", fontsize=9.3,
            color=wbplot.TEXT_MUTED)
    ax.add_patch(FancyArrowPatch((8.55, 1.4), (1.65, 1.4), connectionstyle="arc3,rad=-0.35", **arrow))
    ax.text(5.1, -0.55, "observation suivante : le posterior devient le nouveau prior", ha="center", va="bottom",
            fontsize=10)
    fig.suptitle("La boucle posterior → prior (une observation à chaque tour)", fontsize=10.5, y=0.98)
    save(fig, "boucle.png")


def _sequential(prior, table, observations):
    posterior = np.asarray(prior, dtype=float)
    rows = [posterior]
    for o in observations:
        joint = posterior * table[:, o]
        posterior = joint / joint.sum()
        rows.append(posterior)
    return np.array(rows)


def fig_bars() -> None:
    """40 flips of a coin of bias 0.65: P(fair) and P(rigged) after each flip, as stacked bars."""
    flips = (np.random.default_rng(2).random(40) < 0.65).astype(int)        # a typical run: 26 heads
    table = np.array([[0.5, 0.5], [0.35, 0.65]])          # rows: fair, rigged ; columns: tails, heads
    history = _sequential([0.5, 0.5], table, flips)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    x = np.arange(len(history))
    ax.bar(x, history[:, 0], color=FAIR, width=0.85, label="P(équilibrée | lancers)")
    ax.bar(x, history[:, 1], bottom=history[:, 0], color=RIGGED, width=0.85, label="P(truquée | lancers)")
    ax.axhline(0.5, color="white", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["avant"] + ["F" if f else "P" for f in flips], fontsize=8)
    ax.set_xlim(-0.6, len(history) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("probabilité")
    ax.set_xlabel("résultat de chaque lancer (F = face, P = pile)")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    ax.grid(False)
    ax.set_title(f"Équilibrée ou truquée (biais 0,65) ? {len(flips)} lancers d'une pièce qui est en fait la truquée "
                 f"({flips.sum()} faces)", fontsize=10.5, weight="normal")
    save(fig, "barres_lancers.png")


def fig_five() -> None:
    """Five hypotheses on the bias, uniform prior: the probability of each one after every flip (0 to 50)."""
    grid = np.array([0, 0.25, 0.5, 0.75, 1])
    flips = (np.random.default_rng(26).random(50) < 0.7).astype(int)    # starts with 3 heads, then a tail
    table = np.column_stack([1 - grid, grid])
    history = _sequential(np.full(5, 0.2), table, flips)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    n = np.arange(len(history))
    labels = ["θ = 0", "θ = 0,25", "θ = 0,5", "θ = 0,75", "θ = 1"]
    for k, (colour, label) in enumerate(zip([C[3], C[4], C[2], C[0], C[1]], labels)):
        ax.plot(n, history[:, k], color=colour, lw=1.8, marker="o", ms=2.8, label=label)
    first_tail = int(np.argmin(flips)) + 1
    ax.annotate("1ʳᵉ face : θ = 0 éliminée", xy=(1, history[1, 0]), xytext=(11, 0.56), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=wbplot.TEXT_MUTED))
    ax.annotate(f"1ʳᵉ pile (lancer n° {first_tail}) : θ = 1 éliminée", xy=(first_tail, history[first_tail, 4]),
                xytext=(11, 0.45), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=wbplot.TEXT_MUTED))
    ax.set_xlim(-0.5, len(history) - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("nombre de lancers")
    ax.set_ylabel("probabilité de l'hypothèse")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    ax.set_title(f"Cinq hypothèses sur le biais, prior uniforme : la probabilité de chacune au fil des lancers "
                 f"d'une pièce de biais 0,7 ({flips.sum()} faces sur {len(flips)})", fontsize=10.5, weight="normal")
    save(fig, "cinq_hypotheses.png")


def fig_concentration() -> None:
    """500 hypotheses, uniform prior, coin of bias 0.6: the posterior narrows like 1/sqrt(n)."""
    grid = np.linspace(0, 1, 500)
    flips = (np.random.default_rng(600).random(2000) < 0.6).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    shades = [C[3], C[1], C[4], C[6], C[0]]
    for colour, n in zip(shades, [0, 10, 50, 300, 2000]):
        h = flips[:n].sum()
        with np.errstate(divide="ignore"):
            log_post = h * np.log(grid) + (n - h) * np.log1p(-grid) if n else np.zeros(500)
        post = np.exp(log_post - log_post.max())
        post /= post.sum()
        ax.plot(grid, post, color=colour, lw=1.6, label=f"{n} lancers" if n else "prior uniforme")
    ax.set_xlabel("biais θ (500 hypothèses)")
    ax.set_ylabel("probabilité de chaque hypothèse")
    ax.legend(fontsize=8.5)
    ax.set_title("le posterior se resserre autour du vrai biais\net prend une forme en cloche", fontsize=10,
                 weight="normal")
    ax = axes[1]
    ns = np.arange(1, 2001)
    heads = np.cumsum(flips)
    sd = np.sqrt((heads + 1) * (ns - heads + 1) / ((ns + 2) ** 2 * (ns + 3)))   # sd of Beta(h + 1, t + 1)
    ax.loglog(ns, sd, color=C[0], lw=1.6, label="écart-type du posterior")
    ax.loglog(ns, 0.5 / np.sqrt(ns), color=wbplot.TEXT_MUTED, ls="--", lw=1.2, label="référence en 1/√n")
    ax.set_xlabel("nombre de lancers n (échelle log)")
    ax.set_ylabel("écart-type (échelle log)")
    ax.legend(fontsize=8.5)
    ax.set_title("sa largeur diminue comme 1/√n (ch. 2)", fontsize=10, weight="normal")
    fig.suptitle("500 hypothèses sur le biais, prior uniforme, une pièce de biais 0,6", fontsize=10.5, y=1.03)
    save(fig, "concentration.png")


def fig_beta() -> None:
    """Beta(h + 1, t + 1) posteriors, and a credible interval on a grid."""
    theta = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    for colour, (h, t) in zip([wbplot.TEXT_MUTED, C[1], C[4], C[0]], [(0, 0), (1, 0), (3, 1), (12, 4)]):
        heads = f"{h} face" + ("s" if h > 1 else "")
        tails = f"{t} pile" + ("s" if t > 1 else "")
        ax.plot(theta, stats.beta(h + 1, t + 1).pdf(theta), color=colour, lw=1.7,
                label=f"{heads}, {tails} : Beta({h + 1}, {t + 1})")
    ax.set_xlabel("biais θ")
    ax.set_ylabel("densité")
    ax.legend(fontsize=8.3, loc="upper left")
    ax.set_title("prior uniforme : le posterior est une loi Beta(h + 1, t + 1)", fontsize=10, weight="normal")
    ax = axes[1]
    h, t = 6, 4
    grid = np.linspace(0, 1, 101)
    pdf = stats.beta(h + 1, t + 1).pdf(grid)
    post = pdf / pdf.sum()
    low, high = stats.beta(h + 1, t + 1).ppf([0.025, 0.975])
    inside = (grid >= low) & (grid <= high)
    ax.bar(grid[inside], post[inside], width=0.0085, color=C[0], alpha=0.85, label="dans l'intervalle à 95 %")
    ax.bar(grid[~inside], post[~inside], width=0.0085, color=wbplot.TEXT_MUTED, alpha=0.5,
           label="2,5 % de la probabilité de chaque côté")
    ax.axvline(h / (h + t), color=C[1], lw=1.4, ls="--", label=f"MAP = h / n = {fr(h / (h + t), 1)}")
    ax.set_xlabel("biais θ (grille de 101 valeurs)")
    ax.set_ylabel("probabilité")
    ax.legend(fontsize=8.3, loc="upper left")
    ax.set_title(f"{h} faces et {t} piles : intervalle de crédibilité à 95 %, à queues égales", fontsize=10,
                 weight="normal")
    save(fig, "beta_credible.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    fig_wall()
    fig_loop()
    fig_bars()
    fig_five()
    fig_concentration()
    fig_beta()
    return 0


if __name__ == "__main__":
    sys.exit(main())
