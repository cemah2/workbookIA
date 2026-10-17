#!/usr/bin/env python
"""Draw the figures of chapter 18 (used by Claude).

    python tools/chapters/figures_ch18.py

Writes small PNG files into chapitres/ch18_backprop/figures/. Every figure is computed here (never
copied from the book): a slice of the quadratic error and its tangent (the delta as a slope), the two
passes of one neuron (forward sum, backward sum), gradient descent on a parabola for four learning
rates, the gradient that reaches the logit of a saturated sigmoid output with the squared error and
with the cross-entropy, next to the pre-activations of a living and a dead ReLU neuron, and a small
computation graph with its values and gradients (the micro-autograd). None of them uses the numbers
of an exercise: the labels, points, curvatures and expressions are not those of 18.1 to 18.9, of the
quizzes or of the notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402
from scipy.special import expit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch18_backprop" / "figures"
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
    path = OUT / name
    if path.stat().st_size > 115_000:   # many shades: a 256-colour palette is enough
        from PIL import Image

        with Image.open(path) as image:
            image.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"  {name} ({path.stat().st_size // 1000} ko)")


# ---------------------------------------------------------------------------- the delta as a slope
def fig_slice() -> None:
    """L = ½((P1 − 1)² + (P2 − 0)²) for the label (1, 0): contours, then the slice P2 = 0,6 and its tangent at P1 = −0,3."""
    y1, y2 = 1.0, 0.0
    p1, p2 = -0.3, 0.6
    g = np.linspace(-1.5, 2.5, 241)
    P1, P2 = np.meshgrid(g, np.linspace(-1.5, 2.0, 211))
    L = 0.5 * ((P1 - y1) ** 2 + (P2 - y2) ** 2)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    cs = a.contourf(P1, P2, L, levels=14, cmap="Blues_r")
    a.contour(P1, P2, L, levels=14, colors="white", linewidths=0.5)
    fig.colorbar(cs, ax=a, label="loss L", format=FuncFormatter(french_number))
    a.plot([y1], [y2], marker="*", ms=14, color=C[1], label="le label (1 ; 0) : L = 0")
    a.plot([p1], [p2], "o", ms=8, color=C[3], label="les sorties actuelles")
    a.axhline(p2, color=C[3], ls="--", lw=1)
    a.set_xlabel("sortie P1")
    a.set_ylabel("sortie P2")
    a.set_title("(a) la loss en fonction des deux sorties", fontsize=10)
    a.legend(fontsize=8, loc="upper left", framealpha=0.9)
    x = np.linspace(-1.5, 2.5, 200)
    curve = 0.5 * ((x - y1) ** 2 + (p2 - y2) ** 2)
    slope = p1 - y1
    here = 0.5 * ((p1 - y1) ** 2 + (p2 - y2) ** 2)
    b.plot(x, curve, color=C[0], lw=2.5, label="la tranche P2 = 0,6 de la loss")
    b.plot(x, here + slope * (x - p1), color=C[3], lw=1.5, ls="--",
           label=f"la tangente en P1 = −0,3 : pente {fr(slope, 1)} = δ de la sortie P1")
    b.plot([p1], [here], "o", ms=8, color=C[3])
    b.set_ylim(-0.5, 4.5)
    b.set_xlabel("sortie P1 (P2 fixée à 0,6)")
    b.set_ylabel("loss L")
    b.set_title("(b) près du point, la tangente prédit bien la loss ; loin, non", fontsize=10)
    b.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "tranche_erreur.png")


# ---------------------------------------------------------------------------- forward and backward of one neuron
def _box(ax, xy, text, color, w=1.5, h=0.55, fs=10):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", fc="white",
                                ec=color, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, start, end, text="", color=INK, offset=(0, 0.12), fs=9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3))
    if text:
        mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=fs, color=color)


def fig_two_passes() -> None:
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.5, 6.4))
    for ax in (a, b):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
    # forward: three inputs feed the neuron H
    ins = [("$a_P$", 3.2), ("$a_Q$", 2.0), ("$a_R$", 0.8)]
    for name, y in ins:
        _box(a, (1.0, y), name, C[0], w=1.0)
        _arrow(a, (1.5, y), (3.6, 2.0), color=C[0])
    a.text(2.5, 3.25, "× $w_{PH}$, $w_{QH}$, $w_{RH}$", fontsize=9, color=C[0])
    _box(a, (4.3, 2.0), "$z = \\sum_i a_i\\, w_{iH} + b_H$", C[0], w=2.4)
    _arrow(a, (5.5, 2.0), (6.7, 2.0), color=C[0])
    _box(a, (7.6, 2.0), "$a_H = f(z)$", C[0], w=1.6)
    _arrow(a, (8.4, 2.0), (9.6, 2.0), color=C[0])
    a.text(9.0, 1.45, "vers la\ncouche suivante", ha="center", fontsize=8, color=C[0])
    a.set_title("(a) passe avant : on pondère les sorties d'avant, on somme, on applique f", fontsize=10)
    # backward: three deltas come back to H
    outs = [("$\\delta_J$", 3.2), ("$\\delta_K$", 2.0), ("$\\delta_M$", 0.8)]
    for name, y in outs:
        _box(b, (9.0, y), name, C[1], w=1.0)
        _arrow(b, (8.5, y), (6.4, 2.0), color=C[1])
    b.text(6.4, 3.25, "× $w_{HJ}$, $w_{HK}$, $w_{HM}$", fontsize=9, color=C[1])
    _box(b, (5.4, 2.0), "$\\sum_k w_{Hk}\\, \\delta_k$", C[1], w=1.9)
    _arrow(b, (4.45, 2.0), (3.3, 2.0), color=C[1])
    _box(b, (2.3, 2.0), "$\\delta_H = f'(z)\\times\\ldots$", C[1], w=2.0)
    b.text(2.3, 2.55, "le même z qu'à l'aller", ha="center", fontsize=8, color=C[1])
    _arrow(b, (1.3, 2.0), (0.2, 2.0), color=C[1])
    b.text(0.75, 1.3, "vers la\ncouche d'avant", ha="center", fontsize=8, color=C[1])
    b.set_title("(b) passe arrière : on pondère les deltas d'après, on somme, on multiplie par f′(z)", fontsize=10)
    fig.tight_layout()
    save(fig, "deux_passes.png")


# ---------------------------------------------------------------------------- learning rate on a parabola
def fig_parabola() -> None:
    """L = (a/2) w² with a = 2, from w0 = 1,6: four learning rates."""
    curvature, w0, steps = 2.0, 1.6, 12
    etas = [0.2, 0.5, 0.9, 1.05]
    labels = ["η = 0,2 : lent, régulier", "η = 0,5 : le minimum en un pas", "η = 0,9 : oscille, converge",
              "η = 1,05 : oscille, diverge"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    t = np.arange(steps + 1)
    for k, (eta, lab) in enumerate(zip(etas, labels, strict=True)):
        w = w0 * (1 - eta * curvature) ** t
        a.plot(t, w, marker="o", ms=4, color=C[k], lw=1.6, label=lab)
    a.axhline(0, color=GREY, lw=0.8)
    a.set_ylim(-3.2, 3.2)
    a.set_xlabel("pas t")
    a.set_ylabel("poids $w_t$ (le minimum est en 0)")
    a.set_title("(a) le poids au fil des pas, pour L = w²", fontsize=10)
    a.legend(fontsize=8, loc="lower left")
    x = np.linspace(-2.2, 2.2, 300)
    b.plot(x, 0.5 * curvature * x ** 2, color=INK, lw=1.5)
    w = w0 * (1 - 0.9 * curvature) ** np.arange(8)
    b.plot(w, 0.5 * curvature * w ** 2, color=C[2], marker="o", ms=5, lw=1.2, label="η = 0,9")
    w = w0 * (1 - 0.2 * curvature) ** np.arange(8)
    b.plot(w, 0.5 * curvature * w ** 2, color=C[0], marker="o", ms=5, lw=1.2, label="η = 0,2")
    b.set_xlabel("poids w")
    b.set_ylabel("loss L")
    b.set_title("(b) les mêmes pas sur la courbe de la loss", fontsize=10)
    b.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    save(fig, "pas_parabole.png")


# ---------------------------------------------------------------------------- saturated output and dead ReLU
def fig_saturation() -> None:
    z = np.linspace(-8, 6, 600)
    s = expit(z)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    a.plot(z, s - 1, color=C[0], lw=2.2, label="cross-entropy : σ(z) − y")
    a.plot(z, (s - 1) * s * (1 - s), color=C[1], lw=2.2, label="erreur quadratique :\n(σ(z) − y) σ′(z)")
    a.axhline(0, color=INK, lw=0.8)
    a.set_xlabel("logit z de la sortie (le label vaut y = 1)")
    a.set_ylabel("gradient ∂ℓ/∂z")
    a.set_title("(a) plus la sortie se trompe, plus le gradient... ?", fontsize=10)
    a.legend(fontsize=8, loc="center left")
    rng = np.random.default_rng(1818)
    alive = rng.normal(0.8, 1.0, 2000)
    dying = rng.normal(-0.9, 0.6, 2000)
    dead = rng.normal(-3.0, 0.5, 2000)
    dead = dead[dead < -0.2]
    bins = np.linspace(-5.5, 4.5, 60)
    for k, (vals, lab) in enumerate([(alive, "neurone vivant"), (dying, "neurone presque éteint"),
                                     (dead, "neurone mort : z < 0 pour tous les exemples")]):
        b.hist(vals, bins=bins, alpha=0.55, color=C[k + 2], label=lab)
    b.axvline(0, color=INK, lw=1)
    b.text(0.1, b.get_ylim()[1] * 0.92, "ReLU′(z) = 1 à droite\nde cette ligne, 0 à gauche", fontsize=8)
    b.set_xlabel("pré-activation z d'un neurone ReLU, sur un batch")
    b.set_ylabel("nombre d'exemples")
    b.set_title("(b) trois neurones ReLU sur le même batch", fontsize=10)
    b.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "sortie_saturee.png")


# ---------------------------------------------------------------------------- a computation graph
def fig_graph() -> None:
    """f = (a * b + c) * a with a = 3, b = −1, c = 4: values (forward) and gradients (backward)."""
    a_, b_, c_ = 3.0, -1.0, 4.0
    d_ = a_ * b_
    e_ = d_ + c_
    f_ = e_ * a_
    gf, ge = 1.0, a_
    gd = ge
    gc = ge
    ga = e_ + gd * b_
    gb = gd * a_
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.4, 4.4)
    ax.axis("off")
    nodes = {"a": (1.0, 3.5), "b": (1.0, 2.1), "c": (1.0, 0.7), "d": (4.4, 2.1), "e": (7.4, 1.3), "f": (10.6, 3.0)}
    texts = {"a": f"a = {fr(a_, 0)}\ngrad {fr(ga, 0)}", "b": f"b = {fr(b_, 0)}\ngrad {fr(gb, 0)}",
             "c": f"c = {fr(c_, 0)}\ngrad {fr(gc, 0)}", "d": f"d = a × b = {fr(d_, 0)}\ngrad {fr(gd, 0)}",
             "e": f"e = d + c = {fr(e_, 0)}\ngrad {fr(ge, 0)}", "f": f"f = e × a = {fr(f_, 0)}\ngrad {fr(gf, 0)}"}
    for key, xy in nodes.items():
        _box(ax, xy, texts[key], C[0] if key in "abc" else C[1], w=2.3 if key in "def" else 1.5, h=0.8, fs=9)
    for s, t in [("a", "d"), ("b", "d"), ("d", "e"), ("c", "e"), ("e", "f"), ("a", "f")]:
        (x0, y0), (x1, y1) = nodes[s], nodes[t]
        ws = 0.75 if s in "abc" else 1.15
        wt = 1.15
        _arrow(ax, (x0 + ws, y0), (x1 - wt, y1), color=GREY)
    ax.text(6.0, 4.15, "a sert deux fois : son gradient est la SOMME de ses deux contributions, e + b × grad(d) = 1 − 3 = −2",
            ha="center", fontsize=9, color=INK)
    ax.text(6.0, -0.3, "aller (valeurs) de gauche à droite ; retour (gradients df/d·) de droite à gauche, dans l'ordre topologique inverse",
            ha="center", fontsize=8.5, color=wbplot.TEXT_MUTED)
    save(fig, "graphe_calcul.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 18 -> {OUT.relative_to(ROOT)}")
    fig_slice()
    fig_two_passes()
    fig_parabola()
    fig_saturation()
    fig_graph()
    return 0


if __name__ == "__main__":
    sys.exit(main())
