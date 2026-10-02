#!/usr/bin/env python
"""Draw the figures of chapter 9 (used by Claude).

    python tools/chapters/figures_ch09.py

Writes small PNG files into chapitres/ch09_overfitting/figures/. Every figure is computed (never
copied from the book): three polynomial fits of a cosine (too simple, about right, too wiggly),
the training and validation losses of a logistic regression trained too long (early stopping),
learning curves of a rigid and of a flexible model, the L1 and L2 penalties (1-D shrinkage and
the diamond against the disc), two families of fits of a sine (strong and weak regularisation)
with the U-shaped curve of the squared bias and of the variance, a sketch (not a measurement) of
the double descent, the likelihood of one point and a posterior in the slope-intercept plane, and
the four pairs of curves of the 📈 exercise 9.9 of 02_exercices.md. None of them uses the data of a
notebook exercise (no tempo of a shop, no wind speeds, no California, no random ReLU features:
those are 9.12 to 9.31), and the mini-examples use other numbers than the paper exercises.
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

from mylearn_ref import linear  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch09_overfitting" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
TRAIN, VAL, TRUTH, GREY = C[0], C[1], "#0b0b0b", "#9e9e9e"


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


def poly_fit(x, y, degree, alpha=0.0):
    """Polynomial of the given degree fitted by mylearn's LinearRegression (alpha = 0) or Ridge."""
    X = linear.polynomial_features(x, degree=degree)
    model = linear.LinearRegression() if alpha == 0 else linear.Ridge(alpha=alpha)
    model.fit(X, y)
    return lambda grid: model.predict(linear.polynomial_features(grid, degree=degree))


# ---------------------------------------------------------------------------- three fits
def cosine_data(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    return x, np.cos(1.5 * np.pi * x) + rng.normal(0.0, 0.25, n)


def fig_three_fits() -> None:
    x, y = cosine_data(901, 15)
    grid = np.linspace(-1.0, 1.0, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (degree, title) in zip(axes, [(1, "(a) degré 1 : trop rigide"), (4, "(b) degré 4 : à peu près juste"),
                                          (12, "(c) degré 12 : trop souple")]):
        ax.plot(grid, np.cos(1.5 * np.pi * grid), color=GREY, lw=1.5, ls="--", label="courbe idéale")
        ax.plot(grid, poly_fit(x, y, degree)(grid), color=VAL, lw=2.2, label="polynôme ajusté")
        ax.scatter(x, y, color=TRAIN, s=26, zorder=3, label="15 points d'entraînement")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(-2.0, 2.7)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].legend(loc="upper center", fontsize=8.5)
    save(fig, "trois_ajustements.png")


# ---------------------------------------------------------------------------- early stopping
def logistic_run(seed=0, n=200, n_val=100, p=150, k=8, lr=0.004, epochs=150, batch=4):
    """Logistic regression on p Gaussian features (k useful), mini-batch SGD, no penalty:
    the cross-entropy (in nats) on the training and validation sets after each epoch."""
    rng = np.random.default_rng(seed)
    w_true = np.zeros(p)
    w_true[:k] = rng.normal(size=k)
    X = rng.normal(size=(n + n_val, p))
    y = (rng.random(n + n_val) < 1.0 / (1.0 + np.exp(-X @ w_true))).astype(float)
    Xt, yt, Xv, yv = X[:n], y[:n], X[n:], y[n:]
    w, b = np.zeros(p), 0.0
    train, val = [], []
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            prob = 1.0 / (1.0 + np.exp(-(Xt[idx] @ w + b)))
            w -= lr * Xt[idx].T @ (prob - yt[idx]) / len(idx)
            b -= lr * float(np.mean(prob - yt[idx]))
        for data, labels, out in ((Xt, yt, train), (Xv, yv, val)):
            z = data @ w + b
            out.append(float(np.mean(np.logaddexp(0.0, z) - labels * z)))
    return np.array(train), np.array(val)


def fig_early_stopping() -> None:
    train, val = logistic_run()
    epochs = np.arange(1, len(train) + 1)
    best = int(np.argmin(val))
    patience = 20
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.axvspan(0, best + 1, color=TRAIN, alpha=0.06)
    ax.axvspan(best + 1, epochs[-1], color=VAL, alpha=0.06)
    ax.plot(epochs, train, color=TRAIN, label="entraînement (200 exemples)")
    ax.plot(epochs, val, color=VAL, label="validation (100 exemples)")
    ax.scatter([best + 1], [val[best]], color=VAL, s=60, zorder=4)
    ax.annotate(f"meilleure epoch : {best + 1}", (best + 1, val[best]), xytext=(best + 24, val[best] - 0.17),
                arrowprops={"arrowstyle": "->", "color": "#424242"}, fontsize=9.5)
    ax.axvline(best + 1 + patience, color="#424242", ls=":", lw=1.3)
    ax.text(best + 2 + patience, 0.9, f"arrêt avec une patience\nde {patience} epochs", fontsize=9, va="top")
    ax.text(3, 0.08, "les deux erreurs baissent", fontsize=9, color=TRAIN)
    ax.text(best + 40, 0.08, "l'entraînement baisse, la validation remonte", fontsize=9, color=VAL)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy moyenne (nats)")
    ax.set_xlim(0, epochs[-1])
    ax.set_ylim(0, 0.95)
    ax.legend(loc="center right")
    ax.set_title("Régression logistique, 150 features dont 8 utiles, sans pénalité", fontsize=10.5)
    save(fig, "courbes_erreur.png")


# ---------------------------------------------------------------------------- learning curves
def smooth_curve(x):
    return 1.5 * x - x ** 3 + 0.5 * np.sin(4.0 * x)


def learning_curve_errors(degree, sizes, repeats=100, noise=0.4, seed=902):
    """Median training and validation MSE of a polynomial fit, for each training size (the median:
    with few points, a flexible polynomial sometimes explodes at the edges and would swamp a mean)."""
    rng = np.random.default_rng(seed)
    x_val = rng.uniform(-1.5, 1.5, 2000)
    y_val = smooth_curve(x_val) + rng.normal(0.0, noise, 2000)
    train_mse, val_mse = [], []
    for n in sizes:
        tr, va = [], []
        for _ in range(repeats):
            x = rng.uniform(-1.5, 1.5, n)
            y = smooth_curve(x) + rng.normal(0.0, noise, n)
            predict = poly_fit(x, y, degree)
            tr.append(linear.mean_squared_error(y, predict(x)))
            va.append(linear.mean_squared_error(y_val, predict(x_val)))
        train_mse.append(np.median(tr))
        val_mse.append(np.median(va))
    return np.array(train_mse), np.array(val_mse)


def fig_learning_curves() -> None:
    sizes = np.array([20, 25, 35, 50, 70, 100, 150, 220, 320, 450])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, (degree, title) in zip(axes, [(1, "(a) droite : modèle trop rigide"),
                                          (9, "(b) polynôme de degré 9 : modèle souple")]):
        train, val = learning_curve_errors(degree, sizes)
        ax.plot(sizes, train, "o-", color=TRAIN, label="erreur d'entraînement")
        ax.plot(sizes, val, "o-", color=VAL, label="erreur de validation")
        ax.axhline(0.16, color=GREY, ls="--", lw=1.3, label="bruit : $\\sigma^2 = 0{,}16$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([20, 50, 100, 200, 450])
        ax.get_xaxis().set_major_formatter(FuncFormatter(french_number))
        ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
        ax.get_yaxis().set_major_formatter(FuncFormatter(french_number))
        ax.set_xlabel("nombre d'exemples d'entraînement $n$ (échelle log)")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0.04, 2.0)
    axes[0].set_ylabel("MSE médiane, 100 répétitions (échelle log)")
    axes[1].legend(loc="upper right")
    save(fig, "apprentissage.png")


# ---------------------------------------------------------------------------- L1 and L2
def fig_l1_l2() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    z = np.linspace(-3.0, 3.0, 601)
    ax.plot(z, z, color=GREY, ls="--", lw=1.3, label="sans pénalité : $w^* = z$")
    ax.plot(z, z / 2.0, color=TRAIN, label="L2 (Ridge) : $w^* = z / 2$")
    ax.plot(z, linear.soft_threshold(z, 1.0), color=VAL, label="L1 (Lasso) : $w^* = S(z, 1)$")
    ax.axvspan(-1.0, 1.0, color=VAL, alpha=0.08)
    ax.text(0, -2.55, "zone mise à 0\npar L1", ha="center", fontsize=9, color=VAL)
    ax.set_xlabel("$z$ : le poids qu'on obtiendrait sans pénalité")
    ax.set_ylabel("$w^*$ : le poids pénalisé")
    ax.set_title("(a) En dimension 1 : rétrécir contre seuiller", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    ax = axes[1]
    center = np.array([2.0, 0.9])                         # the least squares solution
    H = np.array([[1.0, 0.45], [0.45, 0.6]])              # the curvature of the loss around it
    g = np.linspace(-1.7, 3.3, 401)
    W1, W2 = np.meshgrid(g, g)
    D = np.stack([W1 - center[0], W2 - center[1]], axis=-1)
    Q = np.einsum("...i,ij,...j->...", D, H, D)
    budget = 1.0
    # constrained optima found on a fine set of boundary points
    angles = np.linspace(0, 2 * np.pi, 20001)
    disc = budget * np.column_stack([np.cos(angles), np.sin(angles)])
    t = np.linspace(-1, 1, 20001)
    diamond = np.concatenate([np.column_stack([t, budget - np.abs(t)]), np.column_stack([t, -(budget - np.abs(t))])])

    def best(points):
        d = points - center
        return points[np.argmin(np.einsum("ij,jk,ik->i", d, H, d))]

    w_l2, w_l1 = best(disc), best(diamond)
    levels = [np.einsum("i,ij,j->", w_l1 - center, H, w_l1 - center),
              np.einsum("i,ij,j->", w_l2 - center, H, w_l2 - center), 2.6, 4.2]
    ax.contour(W1, W2, Q, levels=sorted(levels), colors=[GREY], linewidths=1.0)
    ax.fill(disc[:, 0], disc[:, 1], color=TRAIN, alpha=0.18, label="L2 : $w_1^2 + w_2^2 \\leq 1$")
    ax.fill(diamond[::50, 0], diamond[::50, 1], color=VAL, alpha=0.18)
    ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=VAL, lw=1.6, label="L1 : $|w_1| + |w_2| \\leq 1$")
    ax.scatter(*center, color=TRUTH, s=30, zorder=4)
    ax.annotate("sans pénalité", center, xytext=(center[0] - 0.2, center[1] + 0.75), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#424242"})
    ax.scatter(*w_l2, color=TRAIN, s=45, zorder=5)
    ax.scatter(*w_l1, color=VAL, s=45, zorder=5)
    label_box = {"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85}
    ax.annotate(f"L2 : ({fr(w_l2[0], 2)} ; {fr(w_l2[1], 2)})", w_l2, xytext=(-1.6, 2.2), fontsize=9, color=TRAIN,
                arrowprops={"arrowstyle": "->", "color": TRAIN}, bbox=label_box)
    ax.annotate(f"L1 : ({fr(w_l1[0], 2)} ; {fr(abs(w_l1[1]), 2)})", w_l1, xytext=(0.9, -1.4), fontsize=9,
                color=VAL, arrowprops={"arrowstyle": "->", "color": VAL}, bbox=label_box)
    ax.axhline(0, color="#424242", lw=0.8)
    ax.axvline(0, color="#424242", lw=0.8)
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 3.3)
    ax.set_ylim(-1.7, 3.0)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("(b) En dimension 2 : le coin du losange", fontsize=10.5)
    save(fig, "l1_l2.png")


# ---------------------------------------------------------------------------- bias and variance
def sine_families(alpha, n_sets=25, n=20, degree=9, noise=0.3, seed=903, grid=None):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 200) if grid is None else grid
    curves = []
    for _ in range(n_sets):
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0.0, noise, n)
        curves.append(poly_fit(x, y, degree, alpha=alpha)(grid))
    return grid, np.array(curves)


def fig_bias_variance() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, (alpha, title) in zip(axes[:2], [(1.0, "(a) forte pénalité (alpha = 1)"),
                                             (1e-6, "(b) pénalité minuscule (alpha = 10⁻⁶)")]):
        # the same 200 sets as panel (c): the numbers agree; only the first 25 curves are drawn
        grid, curves = sine_families(alpha, n_sets=200, seed=904)
        for curve in curves[:25]:
            ax.plot(grid, curve, color=VAL, lw=0.8, alpha=0.45)
        ax.plot(grid, curves.mean(axis=0), color=TRAIN, lw=2.4, label="modèle moyen (200 jeux)")
        ax.plot(grid, np.sin(2 * np.pi * grid), color=TRUTH, lw=1.6, ls="--", label="courbe idéale")
        b2, var = linear.bias_variance_decomposition(curves, np.sin(2 * np.pi * grid))
        ax.set_title(f"{title}\nbiais² = {fr(b2, 4 if b2 < 0.01 else 3)}, variance = {fr(var, 3)}", fontsize=10)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].legend(loc="lower left", fontsize=8.5)
    ax = axes[2]
    alphas = np.logspace(-7, 1, 17)
    grid = np.linspace(0, 1, 200)
    stats = np.array([linear.bias_variance_decomposition(sine_families(a, n_sets=200, seed=904)[1],
                                                         np.sin(2 * np.pi * grid)) for a in alphas])
    ax.plot(alphas, stats[:, 0], "o-", color=TRAIN, ms=4, label="biais²")
    ax.plot(alphas, stats[:, 1], "o-", color=VAL, ms=4, label="variance")
    ax.plot(alphas, stats.sum(axis=1) + 0.09, "o-", color=TRUTH, ms=4, label="biais² + variance + bruit")
    ax.axhline(0.09, color=GREY, ls="--", lw=1.2, label="bruit : $\\sigma^2 = 0{,}09$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("alpha (échelle log) : plus à droite, plus de pénalité")
    ax.set_title("(c) Erreur attendue sur une nouvelle mesure (200 jeux)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "biais_variance.png")


# ---------------------------------------------------------------------------- double descent (sketch)
def fig_double_descent() -> None:
    p = np.linspace(0.05, 3.0, 600)                   # number of parameters, in units of n
    classical = 0.25 + 0.3 * (p - 0.45) ** 2 / (0.15 + p)
    at_one = 0.25 + 0.3 * 0.55 ** 2 / 1.15            # the classical curve at p = 1, joined continuously
    after = 0.18 + (at_one - 0.18) / (1.0 + 3.0 * (p - 1.0))
    peak = 0.55 / (1.0 + 80.0 * (p - 1.0) ** 2)
    modern = np.where(p < 1.0, classical, after) + peak
    train = 0.42 * np.clip(1.0 - p, 0.0, None) ** 1.5
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(p, modern, color=VAL, label="erreur de test")
    ax.plot(p, train, color=TRAIN, label="erreur d'entraînement")
    ax.axvline(1.0, color="#424242", ls=":", lw=1.3)
    ax.text(1.03, 1.05, "seuil d'interpolation\n($p \\approx n$)", fontsize=9, va="top")
    ax.text(0.3, 1.05, "régime classique :\nla courbe en U", fontsize=9, va="top", ha="center")
    ax.text(2.2, 1.05, "au-delà : l'erreur\npeut redescendre", fontsize=9, va="top", ha="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("capacité du modèle (nombre de paramètres $p$, pour $n$ exemples fixés)")
    ax.set_ylabel("erreur")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 3.0)
    ax.legend(loc="center right")
    ax.set_title("La double descente : un schéma, pas une mesure", fontsize=10.5)
    save(fig, "double_descente.png")


# ---------------------------------------------------------------------------- Bayes and lines
def fig_bayes_line() -> None:
    x0, y0 = -0.6, 0.5
    slopes = np.linspace(-2.0, 2.0, 201)
    intercepts = np.linspace(-2.0, 2.0, 201)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    grid = np.linspace(-2, 2, 50)
    lines = [(-1.5, y0 + 1.5 * x0), (0.0, y0), (1.0, y0 - x0), (1.8, y0 - 1.8 * x0 + 0.25), (-0.6, y0 + 0.6 * x0 - 0.3)]
    for k, (s, b) in enumerate(lines):
        ax.plot(grid, s * grid + b, color=C[(k + 2) % 8], lw=1.6)
    ax.scatter([x0], [y0], color=C[7], s=60, zorder=4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"(a) Le point ({fr(x0, 1)} ; {fr(y0, 1)}) et cinq droites", fontsize=10)
    ax = axes[1]
    like = linear.bayes_line_posterior([x0], [y0], slopes, intercepts, noise_std=0.25, prior_std=1e6)
    ax.imshow(like / like.max(), origin="lower", extent=(-2, 2, -2, 2), cmap="gray", vmin=0, vmax=1)
    ax.plot(slopes, y0 - slopes * x0, color=C[7], lw=0.8, ls="--")
    for k, (s, b) in enumerate(lines):                 # each line of (a) is one point of the diagram
        ax.scatter([s], [b], color=C[(k + 2) % 8], s=40, edgecolors="white", linewidths=0.8, zorder=4)
    ax.set_xlabel("pente")
    ax.set_ylabel("ordonnée à l'origine")
    ax.set_title("(b) Vraisemblance de chaque droite\n(ramenée à 1 au maximum)", fontsize=10)
    ax = axes[2]
    rng = np.random.default_rng(905)
    x = rng.uniform(-1.5, 1.5, 8)
    y = 0.7 * x - 0.4 + rng.normal(0.0, 0.25, 8)
    post = linear.bayes_line_posterior(x, y, slopes, intercepts, noise_std=0.25, prior_std=1.0)
    ax.imshow(post / post.max(), origin="lower", extent=(-2, 2, -2, 2), cmap="gray", vmin=0, vmax=1)
    ax.plot(0.7, -0.4, "+", color=C[7], ms=10, mew=2)  # the line that generated the 8 points
    ax.set_xlabel("pente")
    ax.set_ylabel("ordonnée à l'origine")
    ax.set_title("(c) Posterior après 8 points\n(prior gaussien centré en (0 ; 0))", fontsize=10)
    for a in axes[1:]:
        a.grid(False)
    save(fig, "droite_bayes.png")


# ---------------------------------------------------------------------------- 📈 exercise 9.9
def curves_99():
    """Four pairs of synthetic training and validation losses (epochs 1 to 100), with noise."""
    rng = np.random.default_rng(909)
    e = np.arange(1, 101)
    noise = lambda scale: rng.normal(0.0, scale, e.size)  # noqa: E731
    # panel 1: overfitting, the validation minimum at epoch 30
    train1 = 0.05 + 0.85 * np.exp(-e / 14.0) + noise(0.006)
    val1 = 0.33 + 0.55 * np.exp(-e / 8.0) + 0.00007 * np.maximum(0, e - 22) ** 2 + noise(0.006)
    # panel 2: the validation below the training loss (dropout or augmentation at training time)
    train2 = 0.30 + 0.5 * np.exp(-e / 18.0) + noise(0.02)
    val2 = 0.21 + 0.5 * np.exp(-e / 16.0) + noise(0.006)
    # panel 3: a good fit
    train3 = 0.22 + 0.7 * np.exp(-e / 12.0) + noise(0.006)
    val3 = 0.27 + 0.68 * np.exp(-e / 12.0) + noise(0.008)
    # panel 4: underfitting
    train4 = 0.62 + 0.3 * np.exp(-e / 5.0) + noise(0.006)
    val4 = 0.645 + 0.3 * np.exp(-e / 5.0) + noise(0.008)
    return e, [(train1, val1), (train2, val2), (train3, val3), (train4, val4)]


def fig_curves_99() -> None:
    e, pairs = curves_99()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for k, (ax, (train, val)) in enumerate(zip(axes, pairs), start=1):
        ax.plot(e, train, color=TRAIN, lw=1.6, label="entraînement")
        ax.plot(e, val, color=VAL, lw=1.6, label="validation")
        ax.set_title(f"Panneau {k}", fontsize=10.5)
        ax.set_xlabel("epoch")
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right", fontsize=8.5)
    save(fig, "courbes_9_9.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures of chapter 9 -> {OUT.relative_to(ROOT)}")
    fig_three_fits()
    fig_early_stopping()
    fig_learning_curves()
    fig_l1_l2()
    fig_bias_variance()
    fig_double_descent()
    fig_bayes_line()
    fig_curves_99()
    return 0


if __name__ == "__main__":
    sys.exit(main())
