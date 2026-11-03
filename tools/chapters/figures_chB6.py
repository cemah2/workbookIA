#!/usr/bin/env python
"""Draw the figures of chapter B6 (used by Claude).

    python tools/chapters/figures_chB6.py        # draw the six figures (offline, a few seconds)

Writes small PNG files into chapitres/b6_explicabilite/figures/:

- ``pdp_ice.png``: partial dependence and ICE curves of a gradient boosting fitted on a synthetic dataset where the
  effect of x1 flips with the sign of x2 (y = 0.2 x0 - 5 x1 + 10 x1 [x2 >= 0] + noise, after Goldstein et al. 2015):
  the average curve is almost flat, the individual curves go up or down;
- ``coalitions.png``: the 16 coalitions of M = 4 features, by size; the 8 edges along which feature 1 joins a
  coalition are highlighted with their Shapley weight (1/4, 1/12, 1/12, 1/4); no value of v(S) is shown;
- ``ig_chemin.png``: integrated gradients of f(x) = sigmoid(x1 x2 - 2) from the zero baseline to x = (2, 3): the
  straight path on the contour plot of f, then the two gradient components along the path (their areas, times
  the displacement, are the attributions; the gradient at x itself is almost zero);
- ``gradcam.png``: the Grad-CAM recipe on a toy layer of three 2 x 2 activation maps (the numbers of section
  206.6): maps, channel weights, weighted sum, ReLU and normalisation, then the bilinear upsampling to 8 x 8;
- ``audit_groupes.png``: the three confusion matrices of the age-group example of section 206.9, then the
  selection rate, TPR and FPR of each group with 95 % percentile bootstrap intervals (2 000 stratified draws);
- ``impossibilite.png``: Chouldechova's relation FPR = p / (1 - p) * (1 - PPV) / PPV * (1 - FNR) against the base
  rate p, for PPV = 0.7 and FNR = 0.25, with the two groups of the example (p = 0.2 and 0.4).

Nothing here is downloaded. None of these figures uses California, Penguins, CIFAR-10 or Adult, nor the numbers of
the paper exercises (B6.1: three features; B6.3: x1^2 + 3 x2; B6.4: two 3 x 3 maps; B6.5: two groups); no curve shows
the median income or the latitude (B6.13).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "b6_explicabilite" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
INK = wbplot.TEXT
MUTED = wbplot.TEXT_MUTED

# --- data shared with 01_fiche.md --------------------------------------------------------------------------------
GROUP_COUNTS = {"18-30": (20, 20, 10, 150), "31-50": (150, 50, 30, 270), "51+": (60, 30, 21, 189)}  # TP FN FP TN
TOY_A = np.array([[[1, 0], [2, 1]], [[0, 3], [1, 0]], [[2, 2], [0, 1]]], float)      # 3 channels, 2 x 2
TOY_G = np.array([np.full((2, 2), 0.5), [[-1, -1], [0, -2]], np.full((2, 2), 0.5)])
IG_X = np.array([2.0, 3.0])
PPV, FNR, BASE_RATES = 0.7, 0.25, (0.2, 0.4)


def french(x: float, digits: int = 2) -> str:
    """A number with a French decimal comma and a true minus sign."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --- 206.2 -------------------------------------------------------------------------------------------------------
def figure_pdp_ice() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import partial_dependence
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    n = 3000
    X = rng.uniform(-1, 1, size=(n, 3))
    y = 0.2 * X[:, 0] - 5 * X[:, 1] + 10 * X[:, 1] * (X[:, 2] >= 0) + rng.normal(0, 0.3, n)
    X_tr, X_va, y_tr, _ = train_test_split(X, y, random_state=0)         # the split of the code in 206.2
    model = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
    res = partial_dependence(model, X_va, features=[1], kind="both", grid_resolution=25,
                             percentiles=(0.02, 0.98))
    grid, ice, avg = res["grid_values"][0], res["individual"][0], res["average"][0]
    up = X_va[:, 2] >= 0
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    show = rng.choice(len(X_va), 60, replace=False)
    for i in show:
        ax.plot(grid, ice[i], color=C[0] if up[i] else C[1], lw=0.7, alpha=0.45)
    ax.plot(grid, avg, color=INK, lw=2.6, label="dépendance partielle (moyenne)")
    ax.plot([], [], color=C[0], lw=1.2, label="courbes ICE : exemples où x₃ ≥ 0")
    ax.plot([], [], color=C[1], lw=1.2, label="courbes ICE : exemples où x₃ < 0")
    ax.set_xlabel("valeur imposée à x₂ (colonne 1 du code)")
    ax.set_ylabel("prédiction du modèle")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("La moyenne est plate, les courbes individuelles ne le sont pas", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "pdp_ice.png")


# --- 206.3 -------------------------------------------------------------------------------------------------------
def figure_coalitions() -> None:
    from itertools import combinations
    from math import factorial

    M = 4
    levels = [list(combinations(range(1, M + 1), k)) for k in range(M + 1)]
    pos = {}
    for k, subsets in enumerate(levels):
        xs = np.linspace(-(len(subsets) - 1) / 2, (len(subsets) - 1) / 2, len(subsets)) * 1.45
        for s, xx in zip(subsets, xs):
            pos[s] = (xx, -k * 1.25)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for k in range(M):
        for s in levels[k]:
            for t in levels[k + 1]:
                if set(s) <= set(t):
                    added = (set(t) - set(s)).pop()
                    (x0, y0), (x1, y1) = pos[s], pos[t]
                    if added == 1:
                        ax.plot([x0, x1], [y0, y1], color=C[1], lw=2.2, zorder=1)
                        w = factorial(len(s)) * factorial(M - len(s) - 1) / factorial(M)
                        frac = {0.25: "1/4", 1 / 12: "1/12"}[min((0.25, 1 / 12), key=lambda v: abs(v - w))]
                        ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), frac, fontsize=8, color=C[1], ha="center",
                                va="center", bbox=dict(fc="white", ec="none", pad=0.6), zorder=3)
                    else:
                        ax.plot([x0, x1], [y0, y1], color="#d6d5d0", lw=0.8, zorder=0)
    for s, (xx, yy) in pos.items():
        label = "∅" if not s else "{" + ", ".join(str(j) for j in s) + "}"
        has1 = 1 in s
        ax.text(xx, yy, label, ha="center", va="center", fontsize=9, zorder=4,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fde0cf" if has1 else "#cde2fb", ec=INK, lw=0.7))
    for k in range(M + 1):
        ax.text(-5.4, -k * 1.25, f"|S| = {k}\n{len(levels[k])} coalition{'s' if len(levels[k]) > 1 else ''}",
                ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(-5.6, 5.0)
    ax.set_ylim(-M * 1.25 - 0.6, 0.6)
    ax.axis("off")
    ax.set_title("M = 4 features, 16 coalitions ; en orange, les 8 arêtes où la feature 1 entre dans S,\n"
                 "avec le poids de Shapley de son apport", fontsize=9.5)
    save(fig, "coalitions.png")


# --- 206.5 -------------------------------------------------------------------------------------------------------
def _sig(z):
    return 1 / (1 + np.exp(-z))


def figure_ig_path() -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    u = np.linspace(-0.3, 3.3, 200)
    U, V = np.meshgrid(u, u)
    cs = ax0.contourf(U, V, _sig(U * V - 2), levels=np.linspace(0, 1, 11), cmap="Blues", alpha=0.85)
    fig.colorbar(cs, ax=ax0, fraction=0.046, pad=0.03, label="f(x)")
    alphas = np.linspace(0, 1, 11)
    pts = alphas[:, None] * IG_X
    ax0.plot(pts[:, 0], pts[:, 1], "-o", color=C[1], ms=3.5, lw=1.6)
    ax0.annotate("ligne de référence x′ = (0, 0)", (0, 0), (0.15, -0.25), fontsize=8, color=INK)
    ax0.annotate("x = (2, 3)", IG_X, (2.15, 2.95), fontsize=8, color=INK,
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
    ax0.set_xlabel("x₁")
    ax0.set_ylabel("x₂")
    ax0.set_title("Le chemin droit x′ + α (x − x′)", fontsize=9.5)
    a = np.linspace(0, 1, 401)
    z = (a * IG_X[0]) * (a * IG_X[1]) - 2
    s = _sig(z) * (1 - _sig(z))
    g1, g2 = s * a * IG_X[1], s * a * IG_X[0]                        # df/dx1 = s x2, df/dx2 = s x1 on the path
    ax1.plot(a, g1, color=C[0], lw=2, label="∂f/∂x₁ le long du chemin")
    ax1.plot(a, g2, color=C[2], lw=2, label="∂f/∂x₂ le long du chemin")
    ax1.fill_between(a, 0, g1, color=C[0], alpha=0.15)
    ax1.plot(a, _sig(z), color=MUTED, lw=1.2, ls="--", label="f le long du chemin")
    ax1.scatter([1, 1], [g1[-1], g2[-1]], color=INK, zorder=3, s=14)
    ax1.annotate("au point x lui-même,\nle gradient est presque nul", (1, g1[-1]), (0.62, 0.62), fontsize=8,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax1.set_xlabel("α")
    ax1.set_title("IG₁ = (x₁ − x′₁) × aire bleue", fontsize=9.5)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    for s_ in ("top", "right"):
        ax1.spines[s_].set_visible(False)
    fig.tight_layout()
    save(fig, "ig_chemin.png")


# --- 206.6 -------------------------------------------------------------------------------------------------------
def figure_gradcam() -> None:
    import torch
    import torch.nn.functional as F

    w = TOY_G.mean(axis=(1, 2))
    cam = np.tensordot(w, TOY_A, axes=1)
    pos = np.maximum(cam, 0)
    norm = pos / np.abs(pos).max()
    up = F.interpolate(torch.tensor(norm)[None, None], size=(8, 8), mode="bilinear",
                       align_corners=False)[0, 0].numpy()
    fig, axes = plt.subplots(1, 6, figsize=(12.4, 2.6))

    def grid(ax, M, title, cmap="Greys", vmin=None, vmax=None, annotate=True, fmt=1):
        ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        if annotate:
            for (i, j), v in np.ndenumerate(M):
                ax.text(j, i, french(v, fmt) if v != int(v) else french(v, 0), ha="center", va="center",
                        fontsize=10, color=INK, bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)

    for c in range(3):
        grid(axes[c], TOY_A[c], f"carte A{c + 1}  ·  α{c + 1} = {french(w[c], 1)}", cmap="Blues", vmin=0, vmax=3)
    grid(axes[3], cam, "Σ αc Ac", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    grid(axes[4], norm, "ReLU puis ÷ max", cmap="Oranges", vmin=0, vmax=1, fmt=2)
    grid(axes[5], up, "agrandie en 8 × 8\n(bilinéaire)", cmap="Oranges", vmin=0, vmax=1, annotate=False)
    fig.tight_layout()
    save(fig, "gradcam.png")


# --- 206.9 -------------------------------------------------------------------------------------------------------
def _arrays():
    yt, yp, gr = [], [], []
    for g, (tp, fn, fp, tn) in GROUP_COUNTS.items():
        for t, p, k in [(1, 1, tp), (1, 0, fn), (0, 1, fp), (0, 0, tn)]:
            yt += [t] * k
            yp += [p] * k
            gr += [g] * k
    return np.array(yt), np.array(yp), np.array(gr)


def _rates(yt, yp):
    return {"taux de sélection": yp.mean(), "TPR": yp[yt == 1].mean(), "FPR": yp[yt == 0].mean()}


def figure_audit() -> None:
    yt, yp, gr = _arrays()
    groups = list(GROUP_COUNTS)
    fig = plt.figure(figsize=(11.2, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.3], wspace=0.35)
    for k, g in enumerate(groups):
        ax = fig.add_subplot(gs[0, k])
        tp, fn, fp, tn = GROUP_COUNTS[g]
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm / cm.sum(), cmap="Blues", vmin=0, vmax=0.8)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11, color=INK)
        ax.set_xticks([0, 1], ["ŷ = 0", "ŷ = 1"])
        ax.set_yticks([0, 1], ["y = 0", "y = 1"] if k == 0 else ["", ""])
        ax.set_title(f"{g} ans (n = {cm.sum()})", fontsize=9.5)
    ax = fig.add_subplot(gs[0, 3])
    rng = np.random.default_rng(0)
    idx = {g: np.flatnonzero(gr == g) for g in groups}
    draws = {g: [] for g in groups}
    for _ in range(2000):
        for g in groups:
            s = rng.choice(idx[g], len(idx[g]), replace=True)
            draws[g].append(list(_rates(yt[s], yp[s]).values()))
    names = list(_rates(yt, yp))
    width = 0.26
    for k, g in enumerate(groups):
        m = gr == g
        point = np.array(list(_rates(yt[m], yp[m]).values()))
        lo, hi = np.percentile(np.array(draws[g]), [2.5, 97.5], axis=0)
        xs = np.arange(len(names)) + (k - 1) * width
        ax.bar(xs, point, width * 0.92, color=C[k], label=f"{g} ans")
        ax.errorbar(xs, point, yerr=[point - lo, hi - point], fmt="none", ecolor=INK, capsize=3, lw=0.9)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylim(0, 0.9)
    ax.set_title("Taux par groupe, intervalles bootstrap à 95 %", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "audit_groupes.png")


def figure_impossibility() -> None:
    p = np.linspace(0.02, 0.66, 300)                                  # FPR reaches 0.6 near p = 0.65
    fpr = p / (1 - p) * (1 - PPV) / PPV * (1 - FNR)
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.plot(p, fpr, color=C[0], lw=2)
    for k, b in enumerate(BASE_RATES):
        v = b / (1 - b) * (1 - PPV) / PPV * (1 - FNR)
        ax.scatter([b], [v], color=C[1 + k], zorder=3, s=36)
        ax.annotate(f"groupe {'AB'[k]} : p = {french(b, 1)}, FPR = {french(v, 3)}", (b, v), (b + 0.04, v - 0.03),
                    fontsize=8.5)
    ax.set_xlabel("taux de base p du groupe")
    ax.set_ylabel("FPR imposé")
    ax.set_ylim(0, 0.6)
    ax.set_xlim(0, 0.7)
    ax.set_title(f"Même PPV ({french(PPV, 1)}) et même FNR ({french(FNR, 2)}) : le FPR suit le taux de base",
                 fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "impossibilite.png")


def main() -> None:
    figure_pdp_ice()
    figure_coalitions()
    figure_ig_path()
    figure_gradcam()
    figure_audit()
    figure_impossibility()
    print("figures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
