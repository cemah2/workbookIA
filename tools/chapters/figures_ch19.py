#!/usr/bin/env python
"""Draw the figures of chapter 19 (used by Claude).

    python tools/chapters/figures_ch19.py

Writes small PNG files into chapitres/ch19_optimiseurs/figures/. Every figure is computed here (never
copied from the book): three landscapes where the gradient vanishes (a bowl, a saddle, a plateau), four
learning-rate schedules, three optimizers on a narrow quadratic valley, the exponential moving average
and its bias at the start, and gradient clipping by the global norm. None of them uses the numbers of an
exercise: the curves, the schedules and the valley are not those of 19.1 to 19.9, of the quizzes or of
the notebook (the notebook uses a negative Gaussian valley, a plateau curve and the Rosenbrock function).
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

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch19_optimiseurs" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
GREY = "#9e9e9e"
INK = wbplot.TEXT


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


# ------------------------------------------------------------------ 1. where the gradient vanishes
def fig_landscapes() -> None:
    x, y = np.meshgrid(np.linspace(-2, 2, 201), np.linspace(-2, 2, 201))
    surfaces = [
        ("Une cuvette : minimum", x**2 + 0.6 * y**2, [(0, 0)]),
        ("Une selle : $x^2 - y^2$", x**2 - y**2, [(0, 0)]),
        ("Un plateau entre deux pentes", np.tanh(3 * (x - 1)) + np.tanh(3 * (x + 1)) + 0.05 * y**2, []),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    gx, gy = np.meshgrid(np.linspace(-1.8, 1.8, 9), np.linspace(-1.8, 1.8, 9))
    for ax, (title, z, points) in zip(axes, surfaces):
        cs = ax.contourf(x, y, z, levels=18, cmap="Blues_r", alpha=0.85)
        ax.contour(x, y, z, levels=18, colors="white", linewidths=0.4)
        step = (x[0, 1] - x[0, 0])
        dzdy, dzdx = np.gradient(z, step)
        i = np.clip(((gy + 2) / step).round().astype(int), 0, 200)
        j = np.clip(((gx + 2) / step).round().astype(int), 0, 200)
        u, v = -dzdx[i, j], -dzdy[i, j]
        norm = np.hypot(u, v)
        scale = np.where(norm > 0, 1 / np.maximum(norm, 1e-12), 0) * np.minimum(norm, 1.5) / 1.5
        ax.quiver(gx, gy, u * scale, v * scale, color=C[1], angles="xy", scale_units="xy", scale=2.2,
                  width=0.006)
        for px, py in points:
            ax.plot(px, py, "o", color=C[2], ms=8, mec="white")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        del cs
    axes[2].text(0, -1.75, "gradient presque nul", ha="center", fontsize=9, color=INK,
                 bbox=dict(boxstyle="round", fc="white", ec=GREY))
    fig.suptitle("Trois endroits où la descente ralentit (flèches : −gradient, longueur bornée)", fontsize=11)
    fig.tight_layout()
    save(fig, "paysages.png")


# ------------------------------------------------------------------ 2. four schedules
def fig_schedules() -> None:
    steps = np.arange(0, 101)
    base = 0.2
    expo = base * 0.96**steps
    delayed = np.where(steps < 20, base, base * 0.96 ** np.maximum(steps - 20, 0))
    stair = base * 0.5 ** (steps // 25)
    warm = 8
    cos = np.where(steps < warm, base * steps / warm,
                   0.01 + (base - 0.01) * 0.5 * (1 + np.cos(np.pi * (steps - warm) / (100 - warm))))
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.0), sharey=True)
    data = [
        ("exponentiel ($\\gamma$ = " + fr(0.96, 2) + ")", expo, C[0]),
        ("exponentiel retardé (20 epochs)", delayed, C[1]),
        ("par paliers (× " + fr(0.5, 1) + " toutes les 25)", stair, C[2]),
        ("warmup (8) + cosinus", cos, C[3]),
    ]
    for ax, (title, values, color) in zip(axes, data):
        ax.plot(steps, values, color=color, lw=2)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("epoch (ou pas)")
        ax.set_ylim(0, 0.215)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("learning rate $\\eta$")
    fig.tight_layout()
    save(fig, "plannings.png")


# ------------------------------------------------------------------ 3. a narrow valley
def fig_narrow_valley() -> None:
    a, b = 1.0, 20.0                       # L = (a x^2 + b y^2) / 2: curvatures 1 and 20

    def grad(w):
        return np.array([a * w[0], b * w[1]])

    start = np.array([-4.5, 1.0])

    def run(update, n=40):
        w, path, state = start.copy(), [start.copy()], {}
        for t in range(1, n + 1):
            w = update(w, grad(w), state, t)
            path.append(w.copy())
        return np.array(path)

    def gd(w, g, state, t, lr=0.09):
        return w - lr * g

    def momentum(w, g, state, t, lr=0.03, mu=0.8):
        state["b"] = g if t == 1 else mu * state["b"] + g
        return w - lr * state["b"]

    def adam(w, g, state, t, lr=0.25, b1=0.9, b2=0.999):
        state["m"] = b1 * state.get("m", 0) + (1 - b1) * g
        state["v"] = b2 * state.get("v", 0) + (1 - b2) * g * g
        return w - lr * (state["m"] / (1 - b1**t)) / (np.sqrt(state["v"] / (1 - b2**t)) + 1e-8)

    x, y = np.meshgrid(np.linspace(-5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
    z = 0.5 * (a * x**2 + b * y**2)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
    runs = [("descente de gradient, $\\eta$ = " + fr(0.09, 2), gd, C[0]),
            ("momentum, $\\eta$ = " + fr(0.03, 2) + ", $\\mu$ = " + fr(0.8, 1), momentum, C[1]),
            ("Adam, $\\eta$ = " + fr(0.25, 2), adam, C[2])]
    for ax, (title, update, color) in zip(axes, runs):
        ax.contour(x, y, z, levels=[0.1, 0.5, 1, 2, 4, 7, 11, 16], colors=GREY, linewidths=0.6)
        path = run(update)
        ax.plot(path[:, 0], path[:, 1], "-o", color=color, ms=2.5, lw=1.2)
        ax.plot(0, 0, "*", color=INK, ms=11)
        ax.set_title(title + " : 40 pas", fontsize=9.5)
        ax.set_xlabel("$w_1$ (courbure 1)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("$w_2$ (courbure 20)")
    fig.tight_layout()
    save(fig, "vallee_etroite.png")


# ------------------------------------------------------------------ 4. the moving average and its bias
def fig_ema() -> None:
    rng = np.random.default_rng(1919)
    t = np.arange(1, 81)
    signal = 1.0 + 0.6 * rng.normal(size=t.size)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    ax = axes[0]
    ax.plot(t, signal, ".", color=GREY, ms=4, label="valeurs bruitées (moyenne 1)")
    for beta, color in [(0.8, C[0]), (0.95, C[1])]:
        m, ema = 0.0, []
        for k, g in enumerate(signal, start=1):
            m = beta * m + (1 - beta) * g
            ema.append(m)
        ema = np.array(ema)
        ax.plot(t, ema, color=color, lw=1.8, label=f"moyenne mobile, $\\beta$ = {fr(beta, 2)}")
        ax.plot(t, ema / (1 - beta**t), "--", color=color, lw=1.2,
                label=f"la même, divisée par $1 - \\beta^t$")
    ax.axhline(1.0, color=INK, lw=0.6)
    ax.set_xlabel("pas $t$")
    ax.set_title("Partie de 0, la moyenne mobile commence trop bas", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[1]
    k = np.arange(0, 40)
    for beta, color, off in [(0.8, C[0], -0.2), (0.95, C[1], 0.2)]:
        ax.bar(k + off, (1 - beta) * beta**k, width=0.4, color=color, label=f"$\\beta$ = {fr(beta, 2)}")
    ax.set_xlabel("âge $k$ de la valeur (0 : la plus récente)")
    ax.set_ylabel("poids $(1 - \\beta)\\,\\beta^k$")
    ax.set_title("Le poids de chaque valeur passée décroît géométriquement", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "moyenne_mobile.png")


# ------------------------------------------------------------------ 5. clipping by the global norm
def fig_clipping() -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    c = 1.0
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(c * np.cos(theta), c * np.sin(theta), color=C[0], alpha=0.12)
    ax.plot(c * np.cos(theta), c * np.sin(theta), color=C[0], lw=1.2)
    for g, color in [((2.4, 1.0), C[1]), ((-0.5, 0.6), C[2]), ((-1.0, -2.2), C[3])]:
        g = np.array(g)
        n = np.linalg.norm(g)
        ax.annotate("", xy=g, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.55))
        clipped = g * min(1.0, c / (n + 1e-6))
        ax.annotate("", xy=clipped, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
    ax.text(0.72, 0.85, "rayon $c$", color=C[0], fontsize=9)
    ax.set_xlim(-2.6, 2.8)
    ax.set_ylim(-2.6, 2.0)
    ax.set_aspect("equal")
    ax.axhline(0, color=GREY, lw=0.5)
    ax.axvline(0, color=GREY, lw=0.5)
    ax.set_title("Clipping par la norme : même direction,\nlongueur ramenée à $c$ au plus", fontsize=10)
    ax.set_xlabel("composante 1 du gradient")
    ax.set_ylabel("composante 2")
    save(fig, "clipping.png")


def main() -> int:
    fig_landscapes()
    fig_schedules()
    fig_narrow_valley()
    fig_ema()
    fig_clipping()
    for p in sorted(OUT.glob("*.png")):
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size // 1024} ko")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
