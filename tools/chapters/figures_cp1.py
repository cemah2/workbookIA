#!/usr/bin/env python
"""Draw the figure of the mock exam of part I (question CP1.7), used by Claude.

    python tools/chapters/figures_cp1.py

Writes checkpoints/partie_1/figures/cp1_7_roc_pr.png. The curves come from synthetic
scores (never from the book): the negatives follow N(0, 1) and the positives N(2.5, 1),
in two populations of 400 000 cases, one with 10 % of positives and one with 1 %.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "checkpoints" / "partie_1" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
SHIFT = 2.5          # mean score of the positives (the negatives have mean 0), both with std 1
N_CASES = 400_000
SHARES = (0.10, 0.01)
SEED = 71


def french_number(value: float, _pos=None) -> str:
    """Tick label with a French decimal comma and a true minus sign."""
    return f"{value:,.10g}".replace(",", " ").replace(".", ",").replace("-", "−")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if (axis.get_scale() == "linear" and axis.get_visible()
                    and isinstance(axis.get_major_formatter(), ScalarFormatter)):
                axis.set_major_formatter(FuncFormatter(french_number))
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def curves(share: float, rng: np.random.Generator):
    """ROC and precision-recall curves of one simulated population."""
    y = rng.random(N_CASES) < share
    score = rng.normal(0.0, 1.0, N_CASES) + SHIFT * y
    fpr, tpr, _ = skm.roc_curve(y, score)
    precision, recall, _ = skm.precision_recall_curve(y, score)
    return fpr, tpr, precision, recall


def fig_roc_pr() -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for share, colour in zip(SHARES, (C[0], C[1])):
        fpr, tpr, precision, recall = curves(share, rng)
        label = f"{share * 100:.0f} % de fraudes"
        axes[0].plot(fpr, tpr, color=colour, lw=2, label=label)
        axes[1].plot(fpr, tpr, color=colour, lw=2, label=label)
        axes[2].plot(recall, precision, color=colour, lw=2, label=label)
    axes[0].plot([0, 1], [0, 1], ls="--", color="0.5", label="hasard")
    axes[0].set(title="Courbe ROC", xlabel="taux de faux positifs (FPR)", ylabel="recall (TPR)",
                xlim=(0, 1), ylim=(0, 1.02), xticks=np.arange(0, 1.01, 0.2), yticks=np.arange(0, 1.01, 0.1))
    axes[1].set(title="Courbe ROC : zoom sur les petits FPR", xlabel="taux de faux positifs (FPR)",
                ylabel="recall (TPR)", xlim=(0, 0.10), ylim=(0, 1.02),
                xticks=np.arange(0, 0.101, 0.01), yticks=np.arange(0, 1.01, 0.1))
    axes[1].axhline(0.8, color="0.35", ls=":", lw=1.5)
    axes[2].set(title="Courbe precision-recall", xlabel="recall", ylabel="precision",
                xlim=(0, 1), ylim=(0, 1.02), xticks=np.arange(0, 1.01, 0.1), yticks=np.arange(0, 1.01, 0.1))
    axes[2].axvline(0.8, color="0.35", ls=":", lw=1.5)
    for ax in axes:
        ax.grid(True, alpha=0.45)
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower right")
    axes[2].legend(loc="lower left")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.suptitle("Question CP1.7 : un même modèle évalué sur deux populations", y=1.02)
    save(fig, "cp1_7_roc_pr.png")


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    fig_roc_pr()
    return 0


if __name__ == "__main__":
    sys.exit(main())
