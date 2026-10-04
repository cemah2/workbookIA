#!/usr/bin/env python
"""Draw the figures of chapter 10 (used by Claude).

    python tools/chapters/figures_ch10.py

Writes small PNG files into chapitres/ch10_neurones/figures/. Every figure is drawn or computed here
(never copied from the book): a simplified biological neuron next to a perceptron, the AND gate against
the XOR gate, the perceptron learning on a small synthetic data set (boundaries after each update,
mistakes per epoch), the margin and the radius of the convergence theorem, four activation functions and
their derivatives, a layer of two neurons with named weights and its two matrices, and a timeline. None
of them uses the data of a notebook exercise (no penguins, no MNIST, no sphere data of 10.23) nor the
weights asked in the paper exercises (the OR, NOT, NAND gates of 10.3 and the XOR network of 10.5 are
left to the learner).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Ellipse  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch10_neurones" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
POS, NEG = C[0], C[1]          # class +1 (blue), class -1 (orange)


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


def _arrow(ax, start, end, color="#424242", lw=1.3, style="-|>", rad=0.0, ls="-", scale=12):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def _node(ax, xy, text, r=0.32, face="white", edge="#424242", fontsize=10, lw=1.2):
    ax.add_patch(Circle(xy, r, facecolor=face, edgecolor=edge, lw=lw, zorder=3))
    ax.text(*xy, text, ha="center", va="center", fontsize=fontsize, zorder=4)


# ---------------------------------------------------------------------------- 1. neuron and perceptron
def fig_neuron() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    ax.set_xlim(-0.6, 10.7)
    ax.set_ylim(-0.6, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Un neurone biologique, très simplifié", fontsize=11)
    body, nucleus = (2.6, 2.6), (2.75, 2.7)
    rng = np.random.default_rng(3)
    for angle in np.linspace(100, 260, 6):                 # dendrites: branching lines on the left
        a = np.radians(angle + rng.uniform(-8, 8))
        start = np.array(body) + 0.75 * np.array([np.cos(a), np.sin(a)])
        end = np.array(body) + 2.0 * np.array([np.cos(a), np.sin(a)])
        ax.plot(*zip(start, end), color=NEG, lw=2.2, solid_capstyle="round")
        for da in (-25, 25):
            b = a + np.radians(da)
            tip = end + 0.55 * np.array([np.cos(b), np.sin(b)])
            ax.plot(*zip(end, tip), color=NEG, lw=1.4, solid_capstyle="round")
    ax.add_patch(Ellipse(body, 1.7, 1.45, facecolor="#fde3c8", edgecolor=NEG, lw=2, zorder=3))
    ax.add_patch(Circle(nucleus, 0.28, facecolor="#f4b183", edgecolor=NEG, lw=1, zorder=4))
    ax.plot([3.45, 7.9], [2.6, 2.6], color=NEG, lw=3, solid_capstyle="round", zorder=2)   # axon
    for dy in (-0.9, -0.3, 0.3, 0.9):                      # synaptic terminals
        ax.plot([7.9, 8.7], [2.6, 2.6 + dy], color=NEG, lw=1.6)
        ax.add_patch(Circle((8.78, 2.6 + dy * 1.03), 0.11, facecolor=NEG, edgecolor=NEG, zorder=3))
    ax.add_patch(Circle((9.85, 2.6), 0.62, facecolor="#dbe9f6", edgecolor=POS, lw=1.5, zorder=1))
    ax.text(9.85, 2.6, "neurone\nsuivant", ha="center", va="center", fontsize=6.8, color=POS)
    ax.plot([2.6, 3.6], [2.0, 0.55], color="#9e9e9e", lw=0.8)
    for xy, text, where in [((-0.55, 5.3), "dendrites : les signaux arrivent", "left"),
                            ((3.65, 0.3), "corps cellulaire : somme,\npuis comparaison au seuil", "left"),
                            ((2.85, 3.45), "noyau", "left"),
                            ((5.7, 3.0), "axone : la décharge", "center"),
                            ((8.2, 4.35), "terminaisons :\nneurotransmetteurs\nlibérés", "center")]:
        ax.text(*xy, text, ha=where, va="center", fontsize=8.5)
    _arrow(ax, (4.3, 2.15), (7.2, 2.15), color="#757575", lw=1.0)
    ax.text(5.75, 1.85, "sens du signal", ha="center", va="top", fontsize=8, color="#757575", style="italic")

    ax = axes[1]
    ax.set_xlim(-0.4, 10.2)
    ax.set_ylim(-0.6, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) Un perceptron à quatre entrées", fontsize=11)
    ys = [4.6, 3.3, 2.0, 0.7]
    total = (6.0, 2.65)
    for k, y in enumerate(ys, start=1):
        _node(ax, (0.4, y), f"$x_{k}$", r=0.36, face="#dbe9f6", edge=POS)
        _node(ax, (2.6, y), "×", r=0.28, face="#f5f5f5")
        ax.text(2.6, y + 0.45, f"$w_{k}$", ha="center", va="bottom", fontsize=10)
        _arrow(ax, (0.78, y), (2.31, y))
        _arrow(ax, (2.89, y), (total[0] - 0.45, total[1] + (y - 2.65) * 0.25))
    _node(ax, total, "Σ", r=0.45, face="#fde3c8", fontsize=13)
    ax.add_patch(FancyBboxPatch((7.15, 2.05), 1.5, 1.2, boxstyle="round,pad=0.05", facecolor="#f5f5f5",
                                edgecolor="#424242", lw=1.2, zorder=3))
    xs = np.array([7.3, 7.9, 7.9, 8.5])
    ax.plot(xs, [2.3, 2.3, 3.0, 3.0], color="#424242", lw=1.6, zorder=4)
    _arrow(ax, (6.46, 2.65), (7.12, 2.65))
    _arrow(ax, (8.7, 2.65), (9.7, 2.65))
    ax.text(9.75, 2.95, "$\\hat{y}$", ha="left", va="bottom", fontsize=12)
    ax.text(6.0, 1.75, "$z = \\sum_j w_j x_j$", ha="center", va="top", fontsize=10)
    ax.text(7.9, 1.75, "$+1$ si $z > 0$\n$-1$ sinon", ha="center", va="top", fontsize=9.5)
    save(fig, "neurone.png")


# ---------------------------------------------------------------------------- 2. AND against XOR
def fig_gates() -> None:
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    grid = np.linspace(-0.4, 1.4, 200)
    gx, gy = np.meshgrid(grid, grid)
    for ax, name, labels in [(axes[0], "AND", [0, 0, 0, 1]), (axes[1], "XOR", [0, 1, 1, 0])]:
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.4, 1.4)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        if name == "AND":
            ax.contourf(gx, gy, (gx + gy - 1.5 > 0).astype(float), levels=[-0.5, 0.5, 1.5],
                        colors=["#fdebd9", "#dbe9f6"])
            ax.plot(grid, 1.5 - grid, color="#424242", lw=1.6)
            ax.text(1.32, 0.05, "$x_1 + x_2 - 1{,}5 = 0$", ha="right", fontsize=9, rotation=-45,
                    rotation_mode="anchor")
            ax.set_title("(a) AND : une droite suffit\n$w = (1, 1)$, $b = -1{,}5$", fontsize=10.5)
        else:
            for (slope, icpt), ls in [((-1.0, 0.5), "--"), ((-1.0, 1.5), ":"), ((1.0, 0.5), "-.")]:
                ax.plot(grid, slope * grid + icpt, color="#9e9e9e", lw=1.2, ls=ls)
            ax.set_title("(b) XOR : aucune droite ne sépare\nles deux couleurs", fontsize=10.5)
        for (x1, x2), label in zip(corners, labels):
            ax.scatter(x1, x2, s=260, color=POS if label else NEG, edgecolor="black", zorder=3,
                       marker="o" if label else "s")
            ax.text(x1 + 0.09, x2 + 0.09, str(label), fontsize=11, zorder=4)
    save(fig, "portes.png")


# ---------------------------------------------------------------------------- 3. the learning rule at work
def _learning_data():
    """12 separable points in the plane (not a data set of the notebook) and their ±1 labels."""
    X = np.array([[0.4, 2.2], [1.0, 2.8], [1.6, 1.9], [2.4, 2.6], [0.6, 1.4], [2.9, 1.7],
                  [1.2, 0.3], [2.2, 0.6], [3.0, 0.4], [0.3, 0.2], [1.9, -0.4], [3.3, 1.0]])
    y = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1], dtype=float)
    return X, y


def _train(X, y, epochs):
    """The classic rule from zero, in order; returns the (w, b) after each update and the mistakes per epoch."""
    w, b = np.zeros(2), 0.0
    history, errors = [], []
    for _ in range(epochs):
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:
                w, b = w + yi * xi, b + yi
                history.append((w.copy(), b))
                mistakes += 1
        errors.append(mistakes)
        if mistakes == 0:
            break
    return history, errors


def fig_learning() -> None:
    X, y = _learning_data()
    history, errors = _train(X, y, 100)
    flipped = y.copy()
    flipped[6] = 1.0                                       # one label flipped: not separable any more
    _, errors_flipped = _train(X, flipped, 30)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    grid = np.linspace(-0.5, 3.8, 50)
    shades = plt.cm.Greys(np.linspace(0.3, 0.95, len(history)))
    for k, ((w, b), color) in enumerate(zip(history, shades), start=1):
        if abs(w[1]) > 1e-9:
            ax.plot(grid, -(w[0] * grid + b) / w[1], color=color, lw=2.2 if k == len(history) else 1.0,
                    ls="-" if k == len(history) else "--")
    ax.scatter(*X[y > 0].T, s=60, color=POS, edgecolor="black", label="classe $+1$", zorder=3)
    ax.scatter(*X[y < 0].T, s=60, color=NEG, edgecolor="black", marker="s", label="classe $-1$", zorder=3)
    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(-0.9, 3.4)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(f"(a) Les frontières après chacune des {len(history)} mises à jour\n"
                 "(de la plus claire à la plus foncée : la dernière, en trait plein)", fontsize=10)
    ax = axes[1]
    ax.plot(range(1, len(errors) + 1), errors, marker="o", color=POS, label="données séparables")
    ax.plot(range(1, len(errors_flipped) + 1), errors_flipped, marker="s", color=NEG, ms=4,
            label="un label inversé : plus séparables")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mises à jour dans l'epoch")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.set_title("(b) Erreurs par epoch : 0 signifie « séparé »", fontsize=10)
    save(fig, "apprentissage.png")
    print(f"    (learning figure: {len(history)} updates, errors {errors}; flipped: last {errors_flipped[-5:]})")


# ---------------------------------------------------------------------------- 4. margin and radius
def fig_margin() -> None:
    rng = np.random.default_rng(10)
    u = np.array([0.6, 0.8])                               # the separator, a unit vector
    gamma, R = 0.35, 2.0
    pts = []
    while len(pts) < 40:
        x = rng.uniform(-R, R, size=2)
        if np.linalg.norm(x) <= R * 0.97 and abs(x @ u) >= gamma:
            pts.append(x)
    pts = np.array(pts)
    pts[0] = u * gamma + np.array([-0.8, 0.6]) * 0.7       # one point exactly on each edge of the margin
    pts[1] = -u * gamma + np.array([0.8, -0.6]) * 0.4
    labels = np.sign(pts @ u)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    t = np.linspace(-2.6, 2.6, 2)
    normal = np.array([-0.8, 0.6])                         # direction of the boundary line
    for offset, style in [(0, "-"), (gamma, "--"), (-gamma, "--")]:
        line = np.outer(t, normal) + offset * u
        ax.plot(*line.T, color="#424242", lw=1.6 if offset == 0 else 1.0, ls=style)
    ax.add_patch(Circle((0, 0), R, fill=False, ls=":", color="#757575", lw=1.2))
    ax.scatter(*pts[labels > 0].T, color=POS, edgecolor="black", s=45, zorder=3, label="$y = +1$")
    ax.scatter(*pts[labels < 0].T, color=NEG, edgecolor="black", s=45, marker="s", zorder=3, label="$y = -1$")
    _arrow(ax, (0, 0), tuple(u * 1.2), color=C[2], lw=2)
    ax.text(*(u * 1.25 + np.array([0.05, 0.0])), "$\\mathbf{u}$ (norme 1)", color=C[2], fontsize=10)
    _arrow(ax, tuple(normal * 1.25), tuple(normal * 1.25 + u * gamma), color="black", style="<|-|>", lw=1.0, scale=9)
    ax.text(*(normal * 1.3 + u * gamma / 2 + np.array([-0.45, 0.0])), "$\\gamma$", fontsize=12)
    _arrow(ax, (0, 0), (-R * 0.6, -R * 0.8), color="#757575", style="-|>", lw=1.0, scale=9)
    ax.text(-R * 0.36, -R * 0.53, "$R$", fontsize=12, color="#616161")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Marge $\\gamma$ et rayon $R$ : au plus $(R/\\gamma)^2$ mises à jour", fontsize=10.5)
    save(fig, "marge.png")


# ---------------------------------------------------------------------------- 5. activation functions
def fig_activations() -> None:
    z = np.linspace(-4, 4, 801)
    sig = 1 / (1 + np.exp(-z))
    funcs = [("seuil (perceptron)", np.where(z > 0, 1.0, -1.0), np.zeros_like(z), "#424242", "-"),
             ("sigmoïde", sig, sig * (1 - sig), C[0], "-"),
             ("tanh", np.tanh(z), 1 - np.tanh(z) ** 2, C[2], "--"),
             ("ReLU", np.maximum(z, 0), (z > 0).astype(float), C[1], "-")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for name, f, df, color, ls in funcs:
        if name.startswith("seuil"):
            for part in (z < 0, z > 0):
                axes[0].plot(z[part], f[part], color=color, lw=2, label=name if part[0] else None)
                axes[1].plot(z[part], df[part], color=color, lw=2, label=name if part[0] else None)
            axes[0].scatter([0], [-1], color=color, s=25, zorder=3)
            axes[0].scatter([0], [1], facecolor="white", edgecolor=color, s=25, zorder=3)
        else:
            axes[0].plot(z, f, color=color, lw=2, ls=ls, label=name)
            axes[1].plot(z, df, color=color, lw=2, ls=ls, label=name)
    axes[0].set_ylim(-1.3, 2.3)
    axes[0].set_title("(a) Quelques fonctions d'activation $f(z)$", fontsize=10.5)
    axes[1].set_ylim(-0.1, 1.15)
    axes[1].set_title("(b) Leurs dérivées $f'(z)$ : nulle partout pour le seuil", fontsize=10.5)
    for ax in axes:
        ax.axhline(0, color="#bdbdbd", lw=0.8)
        ax.axvline(0, color="#bdbdbd", lw=0.8)
        ax.set_xlabel("$z = \\mathbf{w}\\cdot\\mathbf{x} + b$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "activations.png")


# ---------------------------------------------------------------------------- 6. named weights and matrices
def fig_layer() -> None:
    fig = plt.figure(figsize=(12, 4.4))
    ax = fig.add_axes([0.0, 0.0, 0.42, 1.0])
    ax.set_xlim(-0.5, 6.2)
    ax.set_ylim(-0.4, 5.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Trois entrées A, B, C, deux neurones D et E", fontsize=10.5)
    inputs = {"A": (0.6, 4.4), "B": (0.6, 2.5), "C": (0.6, 0.6)}
    neurons = {"D": (4.9, 3.6), "E": (4.9, 1.4)}
    for src, p in inputs.items():
        for dst, q in neurons.items():
            p0, q0 = np.array(p), np.array(q)
            direction = (q0 - p0) / np.linalg.norm(q0 - p0)
            _arrow(ax, tuple(p0 + direction * 0.4), tuple(q0 - direction * 0.45), color="#757575", lw=1.0)
            at = p0 + (q0 - p0) * 0.3                      # near the source: away from the crossings
            ax.text(*at, src + dst, fontsize=9, ha="center", va="center", color=C[0], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))
    for name, xy in inputs.items():
        _node(ax, xy, name, r=0.4, face="#dbe9f6", edge=POS, fontsize=11)
    for name, xy in neurons.items():
        _node(ax, xy, name, r=0.45, face="#fde3c8", edge=NEG, fontsize=11)
        _arrow(ax, (xy[0] + 0.47, xy[1]), (xy[0] + 1.1, xy[1]))
    ax.text(3.0, -0.25, "biais et activations implicites, comme sur la plupart des schémas",
            ha="center", fontsize=8, style="italic", color="#616161")

    def table(axt, rows, cols, cells, title, note):
        axt.axis("off")
        axt.set_title(title, fontsize=10.5)
        tab = axt.table(cellText=[[row] + list(cells_row) for row, cells_row in zip(rows, cells)],
                        colLabels=[""] + list(cols), loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(10.5)
        tab.scale(1.0, 1.9)
        for (r, c), cell in tab.get_celld().items():
            if r == 0 or c == 0:
                cell.set_facecolor("#f5f5f5")
                cell.set_text_props(fontweight="bold")
        axt.text(0.5, 0.06, note, transform=axt.transAxes, ha="center", fontsize=8.5, color="#424242")

    table(fig.add_axes([0.47, 0.12, 0.24, 0.72]), ["A", "B", "C"], ["D", "E"],
          [["AD", "AE"], ["BD", "BE"], ["CD", "CE"]],
          "(b) mylearn : W de forme (n_in, n_out)", "Z = X @ W + b : ligne = source, colonne = neurone")
    table(fig.add_axes([0.75, 0.2, 0.24, 0.56]), ["D", "E"], ["A", "B", "C"],
          [["AD", "BD", "CD"], ["AE", "BE", "CE"]],
          "(c) PyTorch : weight de forme (n_out, n_in)", "y = x @ weight.T + bias : la transposée de W")
    save(fig, "couche.png")


# ---------------------------------------------------------------------------- 7. timeline
def fig_timeline() -> None:
    # (year, label, side, x of the label): the labels are spread out, a leader line joins them to their year
    events = [(1943, "McCulloch et Pitts :\ndes neurones formels\ncalculent la logique", 1, 1943.0),
              (1949, "Hebb : une règle\nd'apprentissage des\nconnexions", -1, 1945.0),
              (1957, "Rosenblatt :\nrapport sur\nle perceptron", 1, 1951.5),
              (1958, "article de 1958,\ndémonstration\nsur IBM 704", -1, 1953.8),
              (1960, "Mark I : 400\nphotocellules,\nmoteurs sur\npotentiomètres", 1, 1960.5),
              (1962, "Novikoff : preuve\nde convergence", -1, 1962.8),
              (1969, "Minsky et Papert,\n« Perceptrons »", 1, 1969.5),
              (1973, "rapport Lighthill\n(Royaume-Uni) :\ncrédits coupés", -1, 1972.0),
              (1974, "Werbos : la\nrétropropagation\n(thèse)", 1, 1978.0),
              (1986, "Rumelhart, Hinton\net Williams :\nla rétropropagation\nse diffuse", -1, 1983.5),
              (1989, "LeCun : un réseau\nconvolutif lit\ndes codes postaux", 1, 1989.5),
              (2012, "AlexNet :\nle deep learning\ns'impose", -1, 1999.0)]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axhline(0, color="#424242", lw=1.5)
    ax.axvspan(1969, 1986, color="#eeeeee", zorder=0)
    ax.text(1977.5, 0.25, "les perceptrons passent de mode", ha="center", fontsize=8.5, style="italic",
            color="#616161")
    for year, text, side, label_x in events:
        x = year if year < 2000 else 1999      # compress the gap before 2012
        height = side * 0.95
        ax.plot([x, label_x], [0, height * 0.85], color="#9e9e9e", lw=0.9)
        ax.scatter([x], [0], color=C[0] if year < 1969 else (C[1] if year < 1986 else C[2]), s=40, zorder=3)
        ax.text(label_x, height, f"{year}\n{text}", ha="center", va="bottom" if side > 0 else "top", fontsize=7.6)
    ax.text(1996.8, 0.1, "≈", fontsize=14, ha="center")
    ax.set_xlim(1938, 2004)
    ax.set_ylim(-2.6, 2.7)
    ax.axis("off")
    ax.set_title("Du neurone formel au deep learning : les dates du chapitre", fontsize=11)
    save(fig, "chronologie.png")


def main() -> int:
    print(f"Figures of chapter 10 -> {OUT.relative_to(ROOT)}")
    fig_neuron()
    fig_gates()
    fig_learning()
    fig_margin()
    fig_activations()
    fig_layer()
    fig_timeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
