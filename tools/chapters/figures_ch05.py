#!/usr/bin/env python
"""Draw the figures of the course sheet of chapter 5 (used by Claude).

    python tools/chapters/figures_ch05.py

Writes small PNG files into chapitres/ch05_courbes/figures/. Every figure is computed
from the examples of the sheet (never copied from the book): curves that break the
book's three rules, local and global extrema with their zones of influence, symmetric
secants of exp, steps along a curve (fixed steps versus gradient descent), the four
kinds of critical points, a map with gradient arrows, and gradient descent in the
Rosenbrock valley. None of them uses the functions or the numbers of an exercise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402
from wb import synth  # noqa: E402

OUT = ROOT / "chapitres" / "ch05_courbes" / "figures"
C = wbplot.CATEGORICAL
DPI = 110


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma, a thin space for thousands and a true minus sign."""
    return f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")


def fr(value: float, decimals: int) -> str:
    """A number written the French way, for titles and legends."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:  # decimal commas on every numeric linear axis (not on 3-D axes or named ticks)
        if getattr(ax, "name", "") == "3d":
            continue
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    path = OUT / name
    if path.stat().st_size > 115_000:   # many shades (contour maps, 3-D surfaces): a 256-colour palette is enough
        from PIL import Image

        with Image.open(path) as image:
            image.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"  {name} ({path.stat().st_size // 1000} ko)")


# ---------------------------------------------------------------------------
def fig_rules() -> None:
    """Four curves that break the book's rules: a jump, a corner, two values, a vertical tangent."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    # (a) a jump
    ax = axes[0]
    x1, x2 = np.linspace(-2, 0.5, 100), np.linspace(0.5, 2, 100)
    ax.plot(x1, 0.4 * x1 + 1, color=C[0], lw=2)
    ax.plot(x2, 0.4 * x2 - 0.3, color=C[0], lw=2)
    ax.plot([0.5], [0.4 * 0.5 + 1], "o", color=C[0], ms=6)
    ax.plot([0.5], [0.4 * 0.5 - 0.3], "o", mfc="white", mec=C[0], ms=6)
    ax.annotate("", xy=(0.5, -0.05), xytext=(0.5, 1.15),
                arrowprops=dict(arrowstyle="->", ls="--", color=wbplot.TEXT_MUTED))
    ax.set_title("(a) un saut : pas continue", fontsize=10, weight="normal")
    # (b) a corner
    ax = axes[1]
    x = np.linspace(-2, 2, 401)
    y = 0.3 * np.sin(2 * x) + np.abs(x - 0.3)
    ax.plot(x, y, color=C[0], lw=2)
    ax.plot([0.3], [0.3 * np.sin(0.6)], "o", mfc="none", mec=C[1], ms=16, mew=1.8)
    ax.set_title("(b) un point anguleux : pas lisse", fontsize=10, weight="normal")
    # (c) two values for one x
    ax = axes[2]
    t = np.linspace(-1.6, 1.6, 300)
    ax.plot(t ** 3 - 2 * t, t, color=C[0], lw=2)
    ax.axvline(0.4, color=C[1], lw=1.2, ls="--")
    xs = 0.4
    roots = [r.real for r in np.roots([1, 0, -2, -xs]) if abs(r.imag) < 1e-9 and -1.6 <= r.real <= 1.6]
    ax.plot([xs] * len(roots), roots, "o", color=C[1], ms=5)
    ax.set_title(f"(c) {len(roots)} valeurs pour un même x : pas univoque", fontsize=10, weight="normal")
    # (d) a vertical tangent inside the curve (not the cube root of quiz Q3)
    ax = axes[3]
    x = np.linspace(-2, 2, 801)
    ax.plot(x, 0.7 * np.cbrt(x - 0.4) + 0.25 * x, color=C[0], lw=2)
    ax.plot([0.4, 0.4], [-0.8, 1.0], color=C[1], lw=1.4, ls="--")
    ax.plot([0.4], [0.1], "o", color=C[1], ms=5)
    ax.set_title("(d) une tangente verticale :\npente infinie", fontsize=10, weight="normal")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Les quatre défauts que le livre interdit : chacun empêche d'avoir, en chaque point, une pente "
                 "bien définie et finie", fontsize=10.5, y=1.03)
    save(fig, "courbes_interdites.png")


def _curve(x):
    return np.sin(x) + 0.55 * np.sin(2.3 * x + 0.4) + 0.06 * x


def fig_extrema() -> None:
    """Local and global extrema of a wavy curve, with the zone of influence of each local minimum."""
    x = np.linspace(0, 12, 4001)
    y = _curve(x)
    inner = np.arange(1, len(x) - 1)
    is_min = inner[(y[inner] < y[inner - 1]) & (y[inner] < y[inner + 1])]
    is_max = inner[(y[inner] > y[inner - 1]) & (y[inner] > y[inner + 1])]
    fig, ax = plt.subplots(figsize=(12, 3.9))
    borders = np.r_[0, is_max, len(x) - 1]        # between two maxima: the points that slide to one minimum
    shades = [C[0], C[2]]
    for k, m in enumerate(is_min):
        left = borders[borders < m].max()
        right = borders[borders > m].min()
        ax.axvspan(x[left], x[right], color=shades[k % 2], alpha=0.10, lw=0)
    ax.plot(x, y, color=wbplot.TEXT, lw=1.8)
    g_max, g_min = int(np.argmax(y)), int(np.argmin(y))
    ax.plot(x[is_max], y[is_max], "o", color=C[1], ms=7, label="maxima locaux")
    ax.plot(x[is_min], y[is_min], "s", color=C[0], ms=7, label="minima locaux")
    ax.plot(x[g_max], y[g_max], "o", mfc="none", mec=C[1], ms=17, mew=2, label="maximum global")
    ax.plot(x[g_min], y[g_min], "s", mfc="none", mec=C[0], ms=17, mew=2, label="minimum global")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    ax.set_title("Extrema locaux et globaux ; chaque bande colorée réunit les départs qui descendent vers le même "
                 "minimum local", fontsize=10.2, weight="normal")
    ax.set_xlim(0, 12)
    save(fig, "extrema.png")


def fig_secants() -> None:
    """Symmetric secants of exp around 0 close in on the tangent; the forward secant lags behind."""
    x = np.linspace(-1.3, 1.3, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
    ax = axes[0]
    ax.plot(x, np.exp(x), color=wbplot.TEXT, lw=2, label="f(x) = eˣ")
    for colour, h in zip([C[3], C[2], C[0]], [1.0, 0.5, 0.25]):
        slope = (np.exp(h) - np.exp(-h)) / (2 * h)
        ax.plot(x, 1 + slope * x + (np.exp(h) + np.exp(-h)) / 2 - 1, color=colour, lw=1.2,
                label=f"sécante symétrique, h = {fr(h, 2)} : pente {fr(slope, 3)}")
        ax.plot([-h, h], [np.exp(-h), np.exp(h)], "o", color=colour, ms=5)
    ax.plot(x, 1 + x, color=C[1], lw=1.6, ls="--", label="tangente en 0 : pente 1")
    ax.set_ylim(0, 3.8)
    ax.set_xlabel("x")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Deux points glissent vers 0, à égale distance :\nla sécante devient la tangente",
                 fontsize=10, weight="normal")
    ax = axes[1]
    hs = np.array([1.0, 0.5, 0.25, 0.1, 0.05])
    central = (np.exp(hs) - np.exp(-hs)) / (2 * hs)
    forward = (np.exp(hs) - 1) / hs
    ax.plot(hs, forward, "o-", color=C[3], label="pente avant (f(h) − f(0)) / h")
    ax.plot(hs, central, "s-", color=C[0], label="pente centrée (f(h) − f(−h)) / 2h")
    ax.axhline(1, color=C[1], ls="--", lw=1.2, label="dérivée exacte : 1")
    ax.invert_xaxis()
    ax.set_xlabel("h (de plus en plus petit vers la droite)")
    ax.set_ylabel("pente estimée en 0")
    ax.legend(fontsize=8.5)
    ax.set_title("Pour un même h, la pente centrée est\nbien plus proche de la dérivée", fontsize=10,
                 weight="normal")
    save(fig, "secantes.png")


def fig_steps() -> None:
    """Left: fixed steps in the direction of the sign of f' never settle. Right: gradient descent does."""
    def f(x):
        return (x - 2) ** 2 + 1

    def df(x):
        return 2 * (x - 2)

    xs = np.linspace(-0.3, 3.3, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), sharey=True)
    fixed = [0.0]
    for _ in range(7):
        fixed.append(fixed[-1] - 0.6 * np.sign(df(fixed[-1])))
    gd = [0.0]
    for _ in range(7):
        gd.append(gd[-1] - 0.2 * df(gd[-1]))
    for ax, pts, title in [(axes[0], fixed, "pas fixe de 0,6 dans le sens qui descend :\nça oscille autour du minimum"),
                           (axes[1], gd, "descente de gradient, η = 0,2 : pas proportionnel\nà la pente, qui "
                                         "rétrécit près du minimum")]:
        ax.plot(xs, f(xs), color=wbplot.TEXT, lw=1.8)
        pts = np.array(pts)
        ax.plot(pts, f(pts), "o", color=C[0], ms=6, zorder=3)
        for a, b in zip(pts[:-1], pts[1:]):
            ax.annotate("", xy=(b, f(b)), xytext=(a, f(a)),
                        arrowprops=dict(arrowstyle="->", color=C[1], lw=1.2, connectionstyle="arc3,rad=0.25"))
        ax.plot([pts[0]], [f(pts[0])], "o", mfc="white", mec=C[0], ms=10, mew=2, zorder=4)
        ax.set_xlabel("x")
        ax.set_title(title, fontsize=10, weight="normal")
    axes[0].set_ylabel("f(x) = (x − 2)² + 1")
    fig.suptitle("Chercher le minimum en suivant la dérivée, depuis x = 0", fontsize=10.5, y=1.03)
    save(fig, "pas_descente.png")


def fig_critical() -> None:
    """The four kinds of points where the gradient vanishes."""
    r, th = np.meshgrid(np.linspace(0, 1.5, 30), np.linspace(0, 2 * np.pi, 61))   # a round grid: no corners
    X, Y = r * np.cos(th), r * np.sin(th)
    surfaces = [
        ("creux : minimum", X ** 2 + Y ** 2, (0, 0, 0)),
        ("sommet : maximum", -(X ** 2 + Y ** 2), (0, 0, 0)),
        ("plateau : plat tout autour,\non ne peut pas conclure", np.sign(X) * np.clip(np.abs(X) - 0.6, 0, None) ** 2,
         (0, 0, 0)),
        ("col : point selle", X ** 2 - Y ** 2, (0, 0, 0)),
    ]
    fig = plt.figure(figsize=(13, 3.5))
    for k, (title, Z, point) in enumerate(surfaces):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d", computed_zorder=False)   # the point stays on top
        ax.plot_surface(X, Y, Z, cmap=wbplot.sequential_cmap().reversed(), linewidth=0, antialiased=True,
                        alpha=0.55, rcount=28, ccount=28)
        ax.scatter([point[0]], [point[1]], [point[2]], color=C[1], s=70, depthshade=False, edgecolor="white",
                   zorder=10)
        ax.set_title(title, fontsize=10, weight="normal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=28, azim=-55)
    fig.suptitle("Quatre points où le gradient s'annule (point orange) : seule la forme autour du point dit "
                 "lequel", fontsize=10.5, y=1.0)
    save(fig, "points_critiques.png")


def _bump(x, y):
    return x * y * np.exp(-(x ** 2 + y ** 2) / 2)


def _bump_grad(x, y):
    e = np.exp(-(x ** 2 + y ** 2) / 2)
    return y * (1 - x ** 2) * e, x * (1 - y ** 2) * e


def fig_map() -> None:
    """Level lines, gradient arrows and a path of steepest descent on f(x, y) = x y exp(-(x² + y²) / 2)."""
    g = np.linspace(-3, 3, 220)
    X, Y = np.meshgrid(g, g)
    Z = _bump(X, Y)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    cs = ax.contourf(X, Y, Z, levels=20, cmap=wbplot.diverging_cmap())
    ax.contour(X, Y, Z, levels=cs.levels, colors="white", linewidths=0.4)
    fig.colorbar(cs, ax=ax, shrink=0.85, label="f(x, y)", format=FuncFormatter(french_number))
    c = np.linspace(-2.6, 2.6, 14)
    Xc, Yc = np.meshgrid(c, c)
    Gx, Gy = _bump_grad(Xc, Yc)
    ax.quiver(Xc, Yc, Gx, Gy, color=wbplot.TEXT, width=0.004, scale=6)
    point = np.array([-0.4, 1.9])
    path = [point]
    for _ in range(400):
        point = point - 0.1 * np.array(_bump_grad(*point))
        path.append(point)
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=C[1], lw=2.2, label="descente depuis (−0,4 ; 1,9)")
    ax.plot(path[0, 0], path[0, 1], "o", color=C[1], ms=7)
    for (px, py), label in [((1, 1), "max"), ((-1, -1), "max"), ((1, -1), "min"), ((-1, 1), "min"),
                            ((0, 0), "col")]:
        ax.plot(px, py, "k+", ms=10, mew=1.6)
        ax.annotate(label, (px, py), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("f(x, y) = x y e^(−(x² + y²)/2) : les flèches du gradient montent,\nperpendiculaires aux lignes de "
                 "niveau ; la descente va contre elles", fontsize=10, weight="normal")
    save(fig, "carte_gradient.png")


def fig_rosenbrock() -> None:
    """Gradient descent in the narrow, curved valley of Rosenbrock (a = 1, b = 100)."""
    def grad(v):
        return np.array(synth.rosenbrock_grad(v[0], v[1]))

    point = np.array([-0.6, 2.4])
    path = [point]
    for _ in range(3000):
        point = point - 0.001 * grad(point)
        path.append(point)
    path = np.array(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    g1, g2 = np.linspace(-2, 2, 260), np.linspace(-1, 3, 260)
    X, Y = np.meshgrid(g1, g2)
    Z = synth.rosenbrock(X, Y) + 1e-3
    ax.contourf(X, Y, Z, levels=np.logspace(-3, 3.6, 22), norm=LogNorm(), cmap=wbplot.sequential_cmap().reversed())
    ax.contour(X, Y, Z, levels=np.logspace(-3, 3.6, 22), colors="white", linewidths=0.35)
    ax.plot(path[:, 0], path[:, 1], color=C[1], lw=1.6)
    ax.plot(path[::200, 0], path[::200, 1], "o", color=C[1], ms=4)
    ax.plot(path[0, 0], path[0, 1], "o", color=C[1], ms=8, mec="white")
    ax.plot(1, 1, "*", color=wbplot.TEXT, ms=14, mec="white")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("La vallée de Rosenbrock (lignes de niveau espacées en log) :\nchute rapide vers le fond, "
                 "puis lente marche vers (1, 1) ; un point tous les 200 pas", fontsize=10, weight="normal")
    ax = axes[1]
    values = synth.rosenbrock(path[:, 0], path[:, 1])
    ax.plot(np.arange(len(values)), values, color=C[0])
    ax.set_yscale("log")
    ax.set_xlabel("nombre de pas (η = 0,001)")
    ax.set_ylabel("f (échelle log)")
    ax.set_title("La valeur de f : une chute en quelques pas,\npuis une longue pente douce", fontsize=10,
                 weight="normal")
    save(fig, "rosenbrock.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    fig_rules()
    fig_extrema()
    fig_secants()
    fig_steps()
    fig_critical()
    fig_map()
    fig_rosenbrock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
