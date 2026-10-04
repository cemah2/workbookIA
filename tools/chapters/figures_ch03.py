#!/usr/bin/env python
"""Draw the figures of the course sheet of chapter 3 (used by Claude).

    python tools/chapters/figures_ch03.py

Writes small PNG files into chapitres/ch03_probabilites/figures/. Every figure is
computed from synthetic data or from the running examples of the sheet (never copied
from the book) and uses the workbook style.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn import metrics as skm  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "chapitres" / "ch03_probabilites" / "figures"
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
    for ax in fig.axes:  # decimal commas on every numeric linear axis (not on named ticks)
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------------------
def fig_darts() -> None:
    """Uniform darts on a wall with two overlapping blobs: counts estimate P(A|B) and P(B|A)."""
    rng = np.random.default_rng(30)
    width, height, n = 5.0, 3.0, 500
    pts = rng.random((n, 2)) * [width, height]
    a_center, a_radius = np.array([2.1, 1.5]), 0.8                  # A: a small disc
    b_center, b_axes = np.array([3.1, 1.45]), np.array([1.45, 1.0])   # B: a larger ellipse (semi-axes)
    in_a = np.sum((pts - a_center) ** 2, axis=1) <= a_radius**2
    in_b = np.sum(((pts - b_center) / b_axes) ** 2, axis=1) <= 1
    both = in_a & in_b
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    panels = [
        (axes[0], np.ones(n, bool), f"(a) {n} fléchettes sur le mur"),
        (axes[1], in_b, f"(b) dans B : {in_b.sum()}, dont {both.sum()} aussi dans A"),
        (axes[2], in_a, f"(c) dans A : {in_a.sum()}, dont {both.sum()} aussi dans B"),
    ]
    for ax, keep, title in panels:
        ax.add_patch(Rectangle((0, 0), width, height, facecolor="#f6f5f2", edgecolor=wbplot.TEXT_MUTED, lw=1.2))
        ax.add_patch(Ellipse(a_center, 2 * a_radius, 2 * a_radius, facecolor=C[0], alpha=0.22, edgecolor=C[0], lw=1.5))
        ax.add_patch(Ellipse(b_center, 2 * b_axes[0], 2 * b_axes[1], facecolor=C[1], alpha=0.22, edgecolor=C[1], lw=1.5))
        ax.text(a_center[0] - 0.85, a_center[1] + 0.55, "A", fontsize=13, color=C[0], weight="bold")
        ax.text(b_center[0] + 1.1, b_center[1] + 0.8, "B", fontsize=13, color=C[1], weight="bold")
        other = keep & ~both
        ax.scatter(pts[other, 0], pts[other, 1], s=7, color=wbplot.TEXT_MUTED, alpha=0.8, lw=0)
        ax.scatter(pts[keep & both, 0], pts[keep & both, 1], s=11, color=C[6], lw=0)
        ax.set_xlim(-0.1, width + 0.1)
        ax.set_ylim(-0.1, height + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=10, weight="normal")
    fig.suptitle("Chaque point du mur a la même chance d'être touché : les comptages estiment des rapports d'aires "
                 "(en violet : les fléchettes dans A et B)", fontsize=10.5)
    save(fig, "flechettes.png")


def _draw_matrix(ax, cells, row_labels, col_labels, title):
    """A 2 x 2 table: green diagonal (right answers), orange elsewhere (errors)."""
    for i in range(2):
        for j in range(2):
            good = i == j
            ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=C[2] if good else C[1], alpha=0.28 if good else 0.22,
                                   edgecolor="white", lw=3))
            name, value = cells[i][j]
            ax.text(j + 0.5, 1.5 - i + 0.12, name, ha="center", va="center", fontsize=13, weight="bold",
                    color=wbplot.TEXT)
            ax.text(j + 0.5, 1.5 - i - 0.2, value, ha="center", va="center", fontsize=11, color=wbplot.TEXT_MUTED)
    for k, label in enumerate(row_labels):
        ax.text(-0.08, 1.5 - k, label, ha="right", va="center", fontsize=10)
    for k, label in enumerate(col_labels):
        ax.text(k + 0.5, 2.08, label, ha="center", va="bottom", fontsize=10)
    ax.text(-0.95, 1.0, "vérité", rotation=90, ha="center", va="center", fontsize=10.5, color=wbplot.TEXT_MUTED)
    ax.text(1.0, 2.42, "prédiction", ha="center", va="bottom", fontsize=10.5, color=wbplot.TEXT_MUTED)
    ax.set_xlim(-1.15, 2.05)
    ax.set_ylim(-0.1, 2.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, weight="normal", y=-0.12)


def fig_confusion_layouts() -> None:
    """The same spam results in the book's layout and in scikit-learn's layout."""
    tp, fn, fp, tn = 8, 4, 2, 36
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _draw_matrix(axes[0], [[("TP", tp), ("FN", fn)], [("FP", fp), ("TN", tn)]],
                 ["spam", "normal"], ["spam", "normal"],
                 "disposition du livre (fig. 3.19) :\npositifs d'abord, TP en haut à gauche")
    _draw_matrix(axes[1], [[("TN", tn), ("FP", fp)], [("FN", fn), ("TP", tp)]],
                 ["0 (normal)", "1 (spam)"], ["0 (normal)", "1 (spam)"],
                 "scikit-learn : labels triés (0 puis 1),\nconfusion_matrix renvoie [[TN, FP], [FN, TP]]")
    fig.suptitle("50 e-mails, 12 spams : les bonnes réponses sont toujours sur la diagonale, "
                 "mais TP et TN changent de coin", fontsize=10.5)
    save(fig, "matrice_confusion.png")


def fig_threshold() -> None:
    """Scores of two classes, one threshold: the four counts, then precision and recall for every threshold."""
    share_pos, mu_pos = 0.1, 2.2              # 10 % of positives; negative scores ~ N(0, 1), positive ~ N(2.2, 1)
    grid = np.linspace(-3.5, 5.5, 900)
    neg = (1 - share_pos) * norm.pdf(grid)
    pos = share_pos * norm.pdf(grid, loc=mu_pos)
    t = 1.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.plot(grid, neg, color=wbplot.TEXT_MUTED, lw=1.6, label="négatifs (90 %)")
    ax.plot(grid, pos, color=C[0], lw=1.8, label="positifs (10 %)")
    right, left = grid >= t, grid <= t
    ax.fill_between(grid[left], neg[left], color=wbplot.TEXT_MUTED, alpha=0.14, lw=0, label="TN : négatifs < t")
    ax.fill_between(grid[right], neg[right], color=C[7], alpha=0.35, lw=0, label="FP : négatifs ≥ t")
    ax.fill_between(grid[right], pos[right], color=C[2], alpha=0.45, lw=0, label="TP : positifs ≥ t")
    ax.fill_between(grid[left], pos[left], color=C[6], alpha=0.40, lw=0, label="FN : positifs < t")
    ax.axvline(t, color=wbplot.TEXT, lw=1.6, ls="--")
    ax.text(t + 0.08, 0.345, "seuil t", fontsize=9.5)
    ax.set_xlabel("score donné par le classifieur")
    ax.set_ylabel("densité (pondérée par la part de chaque classe)")
    ax.set_title("prédire « positif » quand score ≥ t", fontsize=10.5, weight="normal")
    ax.legend(fontsize=8.3, loc="upper right")
    ax = axes[1]
    ts = np.linspace(-1.5, 4.5, 400)
    tpr = norm.sf(ts, loc=mu_pos)
    fpr = norm.sf(ts)
    precision = share_pos * tpr / (share_pos * tpr + (1 - share_pos) * fpr)
    ax.plot(ts, tpr, color=C[0], label="recall")
    ax.plot(ts, precision, color=C[1], label="precision")
    ax.axvline(t, color=wbplot.TEXT, lw=1.4, ls="--")
    ax.set_xlabel("seuil t")
    ax.set_ylim(0, 1.03)
    ax.set_title("monter le seuil : moins de positifs annoncés,\nrecall en baisse, precision en hausse (en général)",
                 fontsize=10, weight="normal")
    ax.legend(fontsize=9, loc="center right")
    save(fig, "seuil.png")


def fig_f1() -> None:
    """F1 = harmonic mean of precision and recall, compared with the arithmetic mean."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    grid = np.linspace(0.001, 1, 300)
    P, R = np.meshgrid(grid, grid)
    F1 = 2 * P * R / (P + R)
    levels = np.arange(0.1, 1.0, 0.1)
    cs = ax.contour(P, R, F1, levels=levels, colors=[C[0]], linewidths=1.3)
    ax.clabel(cs, fmt=lambda v: fr(v, 1), fontsize=8.5)
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.set_aspect("equal")
    ax.set_title("lignes de même F1 : il faut être bon\nsur les deux mesures pour un F1 élevé", fontsize=10, weight="normal")
    ax = axes[1]
    r = np.linspace(0, 1, 300)
    p = 0.9
    ax.plot(r, (p + r) / 2, color=wbplot.TEXT_MUTED, ls="--", label="moyenne arithmétique (P + R) / 2")
    ax.plot(r, 2 * p * r / (p + r), color=C[0], label="F1 (moyenne harmonique)")
    ax.plot(r, np.minimum(p, r), color=C[1], ls=":", label="min(P, R), le maillon faible")
    ax.set_xlabel("recall (precision fixée à 0,9)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("F1 colle au maillon faible", fontsize=10, weight="normal")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "f1.png")


def fig_prevalence() -> None:
    """Precision and NPV of one screening test as the prevalence changes (sensitivity 0.90, specificity 0.95)."""
    sens, spec = 0.90, 0.95
    prev = np.logspace(-3, np.log10(0.5), 300)
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    ax.plot(prev * 100, ppv, color=C[1], label="precision = P(malade | test positif)")
    ax.plot(prev * 100, npv, color=C[0], label="NPV = P(sain | test négatif)")
    p0 = 0.02
    ppv0 = sens * p0 / (sens * p0 + (1 - spec) * (1 - p0))
    ax.scatter([p0 * 100], [ppv0], color=C[1], zorder=3)
    ax.annotate(f"prévalence 2 % : precision {fr(ppv0, 2)}", (p0 * 100, ppv0), xytext=(4, 0.12),
                arrowprops={"arrowstyle": "->", "color": wbplot.TEXT_MUTED}, fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: fr(v, 1 if v < 1 else 0) + " %"))
    ax.set_xlabel("prévalence (part de malades dans la population, échelle logarithmique)")
    ax.set_ylim(0, 1.03)
    ax.set_title("le même test (sensibilité 0,90, spécificité 0,95) dans des populations différentes",
                 fontsize=10.5, weight="normal")
    ax.legend(fontsize=9, loc="center right")
    save(fig, "prevalence.png")


def fig_roc_pr() -> None:
    """Same score laws, two prevalences: nearly the same ROC curve, very different precision-recall curves."""
    rng = np.random.default_rng(12)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for (n_pos, n_neg), color, name in [((2500, 2500), C[0], "50 % de positifs"), ((100, 4900), C[1], "2 % de positifs")]:
        y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
        s = np.r_[rng.normal(1.5, 1, n_pos), rng.normal(0, 1, n_neg)]
        fpr, tpr, _ = skm.roc_curve(y, s)
        prec, rec, _ = skm.precision_recall_curve(y, s)
        auc, ap = skm.roc_auc_score(y, s), skm.average_precision_score(y, s)
        axes[0].plot(fpr, tpr, color=color, label=f"{name} : AUC = {fr(auc, 2)}")
        axes[1].step(rec, prec, where="post", color=color, label=f"{name} : AP = {fr(ap, 2)}")
        axes[1].axhline(n_pos / (n_pos + n_neg), color=color, lw=1, ls=":")
    axes[0].plot([0, 1], [0, 1], color=wbplot.TEXT_MUTED, ls="--", lw=1, label="hasard (AUC = 0,5)")
    axes[0].set_xlabel("taux de faux positifs FPR = FP / (FP + TN)")
    axes[0].set_ylabel("recall = TPR = TP / (TP + FN)")
    axes[0].set_title("courbes ROC : presque identiques", fontsize=10.5, weight="normal")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("courbes precision-recall : très différentes\n(pointillés : precision d'un classifieur au hasard)",
                      fontsize=10.5, weight="normal")
    for ax in axes:
        ax.legend(fontsize=8.8, loc="lower right" if ax is axes[0] else "upper right")
        ax.set_xlim(0, 1)
    fig.suptitle("Mêmes lois de scores pour les positifs et pour les négatifs, seule la part de positifs change",
                 fontsize=10.5, y=1.04)
    save(fig, "roc_pr.png")


def fig_calibration() -> None:
    """Reliability diagram: a calibrated forecaster and an overconfident one (synthetic).

    The overconfident model is an increasing transform of the calibrated one: same ranking,
    hence the same AUC, but different probabilities. Seed 9 keeps every point of the calibrated
    model within about one standard error of the diagonal (seed 5 left one at 3.5).
    """
    rng = np.random.default_rng(9)
    n = 4000
    q = rng.beta(2, 2, n)                              # the true probability of each case
    y = (rng.random(n) < q).astype(int)
    logit = np.log(q / (1 - q))
    over = 1 / (1 + np.exp(-2.5 * logit))              # pushes every probability towards 0 or 1
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=wbplot.TEXT_MUTED, ls="--", lw=1, label="calibration parfaite")
    for p, color, name in [(q, C[0], "modèle calibré"), (over, C[1], "modèle trop sûr de lui")]:
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
        brier = skm.brier_score_loss(y, p)
        ax.plot(prob_pred, prob_true, "o-", color=color, label=f"{name} (Brier {fr(brier, 3)})")
    ax.set_xlabel("probabilité annoncée (moyenne de chaque intervalle)")
    ax.set_ylabel("fréquence observée des positifs")
    ax.set_aspect("equal")
    ax.set_title("diagramme de fiabilité, 10 intervalles", fontsize=10.5, weight="normal")
    ax.legend(fontsize=8.6, loc="upper left")
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(q, bins=bins, color=C[0], alpha=0.5, label="modèle calibré")
    ax.hist(over, bins=bins, color=C[1], alpha=0.5, label="modèle trop sûr de lui")
    ax.set_xlabel("probabilité annoncée")
    ax.set_ylabel("nombre de cas")
    ax.set_title("le modèle trop sûr de lui annonce souvent\ndes probabilités proches de 0 ou de 1",
                 fontsize=10, weight="normal")
    ax.legend(fontsize=8.6)
    fig.suptitle(f"{fr(n, 0)} prévisions : parmi les cas annoncés à 0,8, environ 80 % devraient être positifs",
                 fontsize=10.5, y=1.04)
    save(fig, "calibration.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    fig_darts()
    fig_confusion_layouts()
    fig_threshold()
    fig_f1()
    fig_prevalence()
    fig_roc_pr()
    fig_calibration()
    return 0


if __name__ == "__main__":
    sys.exit(main())
