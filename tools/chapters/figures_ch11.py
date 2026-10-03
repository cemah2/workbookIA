#!/usr/bin/env python
"""Draw the figures of chapter 11 (used by Claude).

    python tools/chapters/figures_ch11.py

Writes small PNG files into chapitres/ch11_raisonnement/figures/. Every figure is drawn or computed here
(never copied from the book): the eight regions of a three-circle Venn diagram and the check of two
syllogisms (one valid, one invalid), the Beta posteriors that Thompson sampling draws from, and the
curves of the 10-armed testbed read in the 📈 exercise 11.10 (our own simulation: 2,000 runs of 1,000
steps for ε = 0, 0.01 and 0.1). None of them uses the data of a notebook exercise (no penguins, no
Holmes, no bench of 11.26 and 11.27).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch11_raisonnement" / "figures"
C = wbplot.CATEGORICAL
DPI = 110


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma, a thin space for thousands and a true minus sign."""
    return f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")


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


# ---------------------------------------------------------------------------- 1. Venn diagrams
CENTRES = {"S": (-0.55, 0.32), "M": (0.55, 0.32), "P": (0.0, -0.62)}
RADIUS = 1.0
# one point inside each of the 7 bounded regions (s, m, p), and one outside (the 8th region)
LABEL_AT = {(1, 0, 0): (-1.05, 0.62), (0, 1, 0): (1.05, 0.62), (0, 0, 1): (0.0, -1.2),
            (1, 1, 0): (0.0, 0.78), (1, 0, 1): (-0.62, -0.38), (0, 1, 1): (0.62, -0.38),
            (1, 1, 1): (0.0, 0.0), (0, 0, 0): (1.55, -1.45)}


def _region_name(code) -> str:
    return " ".join(term if bit else rf"$\overline{{{term}}}$" for term, bit in zip("SMP", code))


def _shade(ax, empty, color="#9e9e9e"):
    """Grey out the regions in `empty` (codes (s, m, p)) by a fine grid of dots coloured by membership."""
    xs, ys = np.meshgrid(np.linspace(-1.75, 1.75, 400), np.linspace(-1.85, 1.55, 400))
    inside = {t: (xs - cx) ** 2 + (ys - cy) ** 2 <= RADIUS ** 2 for t, (cx, cy) in CENTRES.items()}
    mask = np.zeros_like(xs, dtype=bool)
    for code in empty:
        region = np.ones_like(xs, dtype=bool)
        for term, bit in zip("SMP", code):
            region &= inside[term] if bit else ~inside[term]
        mask |= region
    ax.contourf(xs, ys, mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.55)


def _circles(ax, names=("S", "M", "P")):
    for (term, (cx, cy)), color in zip(CENTRES.items(), (C[0], C[2], C[1])):
        ax.add_patch(Circle((cx, cy), RADIUS, fill=False, lw=2, color=color))
        dx = {"S": -0.75, "M": 0.75, "P": 0.0}[term]
        dy = {"S": 1.1, "M": 1.1, "P": -1.18}[term]       # above the top of the circles (cy + 1)
        ax.text(cx + dx, cy + dy, names["SMP".index(term)], fontsize=11, color=color,
                ha="center", va="bottom" if term != "P" else "top", fontweight="bold")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.05, 1.95)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_venn() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    # (a) the eight regions
    ax = axes[0]
    _circles(ax)
    for k, (code, (x, y)) in enumerate(sorted(LABEL_AT.items(), key=lambda kv: kv[0][::-1]), start=1):
        ax.text(x, y, _region_name(code), ha="center", va="center", fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdbdbd", lw=0.6))
    ax.add_patch(plt.Rectangle((-1.95, -2.0), 3.9, 3.7, fill=False, lw=1, color="#757575"))
    ax.set_title("(a) Trois catégories : 8 régions\n(la 8ᵉ est hors des trois cercles)", fontsize=10)
    # (b) a valid syllogism: All M are P, All S are M, so All S are P
    ax = axes[1]
    _shade(ax, [(1, 1, 0), (0, 1, 0),                        # All M are P: no M outside P
                (1, 0, 0), (1, 0, 1)])                       # All S are M: no S outside M
    _circles(ax, names=("S : chats", "M : félins", "P : carnivores"))
    ax.annotate("seule région de S\nencore possible :\nelle est dans P", xy=(0.0, -0.02), xytext=(1.25, -1.45),
                fontsize=8, ha="center", color="#424242",
                arrowprops=dict(arrowstyle="->", color="#424242", lw=1))
    ax.set_title("(b) Valide. « Tout M est P », « Tout S est M » :\nzones grisées vides, donc « Tout S est P »",
                 fontsize=10)
    # (c) an invalid one: All P are M, All S are M, so All S are P (undistributed middle)
    ax = axes[2]
    _shade(ax, [(1, 0, 1), (0, 0, 1),                        # All P are M
                (1, 0, 0)])                                  # All S are M (with (1, 0, 1), already grey)
    _circles(ax, names=("S : chats", "M : mammifères", "P : chiens"))
    ax.scatter([0.0], [0.78], s=120, color="#8e24aa", edgecolor="black", zorder=5)
    ax.annotate("un S qui n'est pas P\nreste possible", xy=(0.06, 0.72), xytext=(1.3, -1.5), fontsize=8,
                ha="center", color="#6a1b9a", arrowprops=dict(arrowstyle="->", color="#6a1b9a", lw=1))
    ax.set_title("(c) Non valide. « Tout P est M », « Tout S est M » :\n"
                 "le point violet contredit « Tout S est P »", fontsize=10)
    save(fig, "venn.png")


# ---------------------------------------------------------------------------- 2. Thompson sampling
def fig_thompson() -> None:
    arms = [("bras 0 : 6 succès, 4 échecs", 6, 4, C[0]), ("bras 1 : 1 succès, 1 échec", 1, 1, C[1]),
            ("bras 2 : 18 succès, 22 échecs", 18, 22, C[2])]   # numbered from 0, as in Python and in mylearn
    theta = np.linspace(0, 1, 501)
    rng = np.random.default_rng(1107)
    draws = np.column_stack([rng.beta(1 + s, 1 + f, size=200_000) for _, s, f, _ in arms])
    share = np.bincount(draws.argmax(axis=1), minlength=3) / len(draws)
    one = [rng.beta(1 + s, 1 + f) for _, s, f, _ in arms]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for (label, s, f, color), x in zip(arms, one):
        density = stats.beta(1 + s, 1 + f).pdf(theta)
        ax.plot(theta, density, color=color, lw=2,
                label=f"{label} : Beta({1 + s}, {1 + f})")
        ax.axvline(x, color=color, ls="--", lw=1.2)
    ax.set_xlabel(r"probabilité de succès $\theta$ du bras")
    ax.set_ylabel("densité du posterior")
    ax.set_ylim(0, 7.4)                            # room above the peaks for the legend
    ax.legend(fontsize=8.3, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_title("(a) Un tirage par bras (tirets) : on joue le plus grand\n"
                 f"(ici le bras {int(np.argmax(one))})", fontsize=10)
    ax = axes[1]
    ax.bar([0, 1, 2], share, color=[a[3] for a in arms], edgecolor="black")
    for k, value in enumerate(share):
        ax.text(k, value + 0.015, f"{value:.2f}".replace(".", ","), ha="center", fontsize=9)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["bras 0", "bras 1", "bras 2"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("probabilité d'être joué")
    ax.set_title("(b) Sur 200 000 tirages : chaque bras est joué\n"
                 "avec la probabilité qu'il soit le meilleur", fontsize=10)
    save(fig, "thompson.png")
    print(f"    (Thompson: one draw {np.round(one, 3).tolist()}, shares {np.round(share, 3).tolist()})")


# ---------------------------------------------------------------------------- 3. the 10-armed testbed (📈 11.10)
def _testbed(eps, runs=2000, steps=1000, k=10, seed=1110):
    """ε-greedy with sample averages on `runs` Gaussian 10-armed bandits (q* ~ N(0, 1), rewards N(q*, 1)),
    all the runs at once; random tie-breaking. Returns the mean reward and the share of optimal actions
    at each step, and the mean value of the best arm."""
    rng = np.random.default_rng(seed)
    qstar = rng.normal(0.0, 1.0, size=(runs, k))
    best = qstar.argmax(axis=1)
    Q, N = np.zeros((runs, k)), np.zeros((runs, k))
    rows = np.arange(runs)
    reward, optimal = np.zeros(steps), np.zeros(steps)
    for t in range(steps):
        ties = Q == Q.max(axis=1, keepdims=True)
        greedy = np.argmax(ties * rng.random((runs, k)), axis=1)
        explore = rng.random(runs) < eps
        a = np.where(explore, rng.integers(k, size=runs), greedy)
        r = rng.normal(qstar[rows, a], 1.0)
        N[rows, a] += 1
        Q[rows, a] += (r - Q[rows, a]) / N[rows, a]
        reward[t], optimal[t] = r.mean(), (a == best).mean()
    return reward, optimal, qstar.max(axis=1).mean()


def fig_epsilon() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True)
    steps = np.arange(1, 1001)
    summary = {}
    for eps, color in [(0.0, C[1]), (0.01, C[2]), (0.1, C[0])]:
        reward, optimal, best = _testbed(eps)
        label = "ε = " + f"{eps:g}".replace(".", ",") + (" (glouton)" if eps == 0 else "")
        axes[0].plot(steps, reward, color=color, lw=1.1, label=label)
        axes[1].plot(steps, 100 * optimal, color=color, lw=1.1, label=label)
        summary[eps] = (reward[-100:].mean(), optimal[-100:].mean(), reward[:20].mean(), optimal[9])
    axes[0].axhline(best, color="#424242", ls="--", lw=1)
    axes[0].text(1000, best + 0.04, "valeur moyenne du meilleur bras", ha="right", fontsize=8.5, color="#424242")
    axes[0].set_ylabel("récompense moyenne")
    axes[0].set_ylim(-0.1, 1.75)
    axes[1].set_ylabel("% d'action optimale")
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("pas")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_title("ε-greedy sur 2 000 bandits gaussiens à 10 bras (moyennes sur les 2 000 parties)", fontsize=10.5)
    save(fig, "bandit_epsilon.png")
    print(f"    (best arm mean {best:.3f}; per ε: reward last 100, optimal last 100, reward first 20, optimal at step 10: "
          + "; ".join(f"{eps}: {r:.3f} {o:.3f} {r20:.3f} {o10:.3f}" for eps, (r, o, r20, o10) in summary.items()) + ")")


def main() -> int:
    print(f"Figures of chapter 11 -> {OUT.relative_to(ROOT)}")
    fig_venn()
    fig_thompson()
    fig_epsilon()
    return 0


if __name__ == "__main__":
    sys.exit(main())
