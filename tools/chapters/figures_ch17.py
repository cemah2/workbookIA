#!/usr/bin/env python
"""Draw the figures of chapter 17 (used by Claude).

    python tools/chapters/figures_ch17.py

Writes small PNG files into chapitres/ch17_activations/figures/. Every figure is computed here with
the reference mylearn.nn.activations (never copied from the book): the collapse of a small linear
network and what one ReLU changes, the ReLU family (leaky, shifted, maxout, noisy), the smooth
functions next to the ReLU they round off, the saturation of the sigmoid and tanh, a gallery of the
ten activations of the module with their derivatives, the softmax at three temperatures, and the
unlabelled gallery of quiz 17.Q10. None of them uses the numbers of an exercise: the networks,
lines, scores and slopes of the figures are not those of 17.1, 17.5, 17.6, 17.19 or the quizzes.
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
sys.path.insert(0, str(ROOT / "solutions"))

from mylearn_ref.nn import activations as act  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch17_activations" / "figures"
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


def axes_cross(ax):
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    ax.axvline(0, color=GREY, lw=0.8, zorder=0)


# ---------------------------------------------------------------------------- collapse
def fig_collapse() -> None:
    """A 1 -> 3 -> 3 -> 1 network: without activation its output is one straight line; with ReLU it bends."""
    W1 = np.array([[1.0, -1.0, 0.5]])
    b1 = np.array([0.5, 1.0, -1.0])
    W2 = np.array([[1.0, -0.5, 0.0], [0.5, 1.0, -1.0], [-1.0, 0.5, 2.0]])
    b2 = np.array([0.0, -0.5, 0.5])
    W3 = np.array([[1.0], [-1.0], [0.8]])
    b3 = np.array([0.2])
    x = np.linspace(-4, 4, 801).reshape(-1, 1)

    def net(f):
        h = f(x @ W1 + b1)
        g = f(h @ W2 + b2)
        return (g @ W3 + b3).ravel()

    W_eq = W1 @ W2 @ W3
    b_eq = (b1 @ W2 + b2) @ W3 + b3
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    a.plot(x, net(act.identity), color=C[0], lw=2.5, label="le réseau 1-3-3-1, sans activation")
    a.plot(x, (x @ W_eq + b_eq).ravel(), color=C[1], lw=1.2, ls="--",
           label=f"un seul neurone : {fr(W_eq.item(), 2)} x + {fr(b_eq.item(), 2)}")
    a.set_title("(a) sans activation : 22 paramètres, une seule droite", fontsize=10)
    b.plot(x, net(act.relu), color=C[2], lw=2.5, label="le même réseau, avec ReLU dans les couches cachées")
    b.set_title("(b) avec ReLU : une ligne brisée", fontsize=10)
    for ax in (a, b):
        axes_cross(ax)
        ax.set_xlabel("entrée x")
        ax.legend(fontsize=8, loc="upper left")
    a.set_ylabel("sortie du réseau")
    fig.tight_layout()
    save(fig, "effondrement.png")


# ---------------------------------------------------------------------------- ReLU family
def fig_relu_family() -> None:
    z = np.linspace(-3, 3, 601)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    axs[0].plot(z, act.relu(z), color=C[0], lw=2.2, label="ReLU")
    axs[0].plot(z, act.leaky_relu(z, negative_slope=0.1), color=C[1], lw=1.6, ls="--", label="leaky, pente 0,1")
    axs[0].set_title("ReLU et leaky ReLU", fontsize=10)
    axs[1].plot(z, np.maximum(z, -1.0), color=C[2], lw=2.2, label="max(z, −1)")
    axs[1].set_title("ReLU décalée", fontsize=10)
    lines = [(-1.5, -1.0), (0.2, 0.0), (1.2, -1.5)]           # slopes and intercepts of three lines
    for k, (s, c) in enumerate(lines):
        axs[2].plot(z, s * z + c, color=GREY, lw=1, ls=":")
    axs[2].plot(z, np.max([s * z + c for s, c in lines], axis=0), color=C[3], lw=2.4, label="le maximum des trois")
    axs[2].set_title("maxout de trois droites", fontsize=10)
    rng = np.random.default_rng(17)
    zs = np.linspace(-3, 3, 121)
    axs[3].plot(z, act.relu(z), color=GREY, lw=1, label="ReLU")
    axs[3].scatter(zs, act.relu(zs + rng.normal(0, 0.3, zs.size)), s=8, color=C[4], label="ReLU bruitée, une passe")
    axs[3].set_title("ReLU bruitée (bruit d'écart-type 0,3)", fontsize=10)
    for ax in axs:
        axes_cross(ax)
        ax.set_xlabel("z")
        ax.set_ylim(-2.2, 3.2)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "famille_relu.png")


# ---------------------------------------------------------------------------- smooth functions
def fig_smooth() -> None:
    z = np.linspace(-5, 5, 801)
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.4))
    axs[0].plot(z, act.relu(z), color=GREY, lw=1.2, label="ReLU")
    axs[0].plot(z, act.softplus(z), color=C[0], lw=2.2, label="softplus")
    axs[0].set_title("softplus : une ReLU arrondie", fontsize=10)
    axs[1].plot(z, np.maximum(z, -1.0), color=GREY, lw=1.2, label="max(z, −1)")
    axs[1].plot(z, act.elu(z), color=C[1], lw=2.2, label="ELU (α = 1)")
    axs[1].set_title("ELU : une ReLU décalée arrondie", fontsize=10)
    axs[2].plot(z, act.sigmoid(z), color=C[2], lw=2.2, label="sigmoïde σ")
    axs[2].plot(z, act.tanh(z), color=C[6], lw=2.2, label="tanh")
    axs[2].set_title("deux courbes en S", fontsize=10)
    axs[3].plot(z, act.relu(z), color=GREY, lw=1.2, label="ReLU")
    axs[3].plot(z, act.silu(z), color=C[4], lw=2.2, label="SiLU (swish)")
    axs[3].plot(z, act.gelu(z), color=C[5], lw=1.6, ls="--", label="GELU")
    axs[3].set_title("SiLU et GELU : un petit creux sous 0", fontsize=10)
    for ax in axs:
        axes_cross(ax)
        ax.set_xlabel("z")
        ax.legend(fontsize=7.5, loc="upper left")
    for ax in (axs[0], axs[1], axs[3]):
        ax.set_ylim(-1.6, 5.2)
    axs[2].set_ylim(-1.2, 1.2)
    fig.tight_layout()
    save(fig, "lisses.png")


# ---------------------------------------------------------------------------- saturation
def fig_saturation() -> None:
    z = np.linspace(-8, 8, 1601)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.7))
    for ax, f, df, name, color in ((a, act.sigmoid, act.sigmoid_derivative, "sigmoïde", C[2]),
                                   (b, act.tanh, act.tanh_derivative, "tanh", C[6])):
        flat = df(z) < 0.05
        ax.fill_between(z, -1.1, 1.1, where=flat, color="#f4a3a2", alpha=0.35, lw=0,
                        label="pente < 0,05 : zone saturée")
        ax.plot(z, f(z), color=color, lw=2.2, label=name)
        ax.plot(z, df(z), color=INK, lw=1.4, ls="--", label=f"sa dérivée (maximum {fr(df(0.0).item(), 2)})")
        axes_cross(ax)
        ax.set_xlabel("pré-activation z")
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"{name} : la pente s'écrase loin de 0", fontsize=10)
    fig.tight_layout()
    save(fig, "saturation.png")


# ---------------------------------------------------------------------------- gallery
GALLERY = [("identity", "identité"), ("step", "marche (Heaviside)"), ("relu", "ReLU"),
           ("leaky_relu", "leaky ReLU (pente 0,01)"), ("elu", "ELU (α = 1)"), ("softplus", "softplus"),
           ("sigmoid", "sigmoïde"), ("tanh", "tanh"), ("silu", "SiLU (swish)"), ("gelu", "GELU")]


def fig_gallery() -> None:
    z = np.linspace(-4, 4, 801)
    fig, axs = plt.subplots(2, 5, figsize=(14, 5.6), sharex=True)
    for ax, (name, label) in zip(axs.ravel(), GALLERY):
        f, df = act.get_activation(name)
        y, dy = f(z), df(z)
        if name == "step":                       # no vertical segment at the jump
            for part in (z < 0, z >= 0):
                ax.plot(z[part], y[part], color=C[0], lw=2.2)
            ax.plot([0], [1], "o", color=C[0], ms=4)
            ax.plot([0], [0], "o", color=C[0], ms=4, mfc="white")
        else:
            ax.plot(z, y, color=C[0], lw=2.2, label="f(z)")
        ax.plot(z, dy, color=C[1], lw=1.4, ls="--", label="f′(z)")
        axes_cross(ax)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-1.5, 2.5)
    axs[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axs[1]:
        ax.set_xlabel("z")
    fig.suptitle("Les dix activations de mylearn.nn.activations (trait plein) et leurs dérivées (tirets)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "galerie.png")


# ---------------------------------------------------------------------------- softmax and temperature
def fig_temperature() -> None:
    scores = np.array([1.0, 0.5, -0.5, -2.0])
    names = ["A", "B", "C", "D"]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.2), sharey=False)
    axs[0].bar(names, scores, color=C[0])
    axs[0].axhline(0, color=GREY, lw=0.8)
    axs[0].set_title("scores (logits)", fontsize=10)
    for ax, T in zip(axs[1:], (0.5, 1.0, 3.0)):
        p = act.softmax(scores / T)
        ax.bar(names, p, color=C[2])
        for k, v in enumerate(p):
            ax.text(k, v + 0.02, fr(v, 2), ha="center", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"softmax(z / T), T = {fr(T, 1)}", fontsize=10)
    fig.tight_layout()
    save(fig, "temperature.png")


# ---------------------------------------------------------------------------- quiz 17.Q10
MYSTERY = [("sigmoid", {}), ("tanh", {}), ("softplus", {}), ("elu", {}), ("silu", {}),
           ("leaky_relu", {"negative_slope": 0.2})]


def fig_mystery() -> None:
    """Six unlabelled curves; the panel numbers follow a seeded permutation (quiz 17.Q10)."""
    order = np.random.default_rng(1710).permutation(len(MYSTERY))
    z = np.linspace(-4, 4, 801)
    fig, axs = plt.subplots(1, 6, figsize=(14, 2.7), sharey=True)
    for panel, k in enumerate(order):
        name, kwargs = MYSTERY[k]
        axs[panel].plot(z, getattr(act, name)(z, **kwargs), color=C[0], lw=2.2)
        axes_cross(axs[panel])
        axs[panel].set_title(f"courbe {panel + 1}", fontsize=10)
        axs[panel].set_ylim(-1.5, 4.0)
        axs[panel].set_xticks([-4, -2, 0, 2, 4])
        axs[panel].grid(True, color=wbplot.GRID, lw=0.6)
    fig.tight_layout()
    save(fig, "q10_mystere.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 17 -> {OUT.relative_to(ROOT)}")
    fig_collapse()
    fig_relu_family()
    fig_smooth()
    fig_saturation()
    fig_gallery()
    fig_temperature()
    fig_mystery()
    return 0


if __name__ == "__main__":
    sys.exit(main())
