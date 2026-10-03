#!/usr/bin/env python
"""Draw the figure of the mock exam of part II (question CP2.7), used by Claude.

    python tools/chapters/figures_cp2.py

Writes checkpoints/partie_2/figures/cp2_7_courbes_apprentissage.png: the learning curves
(training and validation RMSE against the number of training examples) of three models
on the same synthetic regression task, y = 1.5 sin(2 pi x) + 0.5 x + noise, with x uniform
in [0, 1] and a Gaussian noise of standard deviation 0.5 (the floor of the RMSE). Each point
is the median over 200 training sets; the validation RMSE is measured on 20 000 fresh points.
The three models are least squares on Gaussian bumps (with a column of ones):
- panel 1: 8 bumps of width 0.12, alpha = 1e-3 (about the right capacity);
- panel 2: a straight line (too rigid);
- panel 3: 150 bumps of width 0.006, alpha = 0.1 (very flexible, lightly regularized).
Nothing comes from the book.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "checkpoints" / "partie_2" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
SIGMA = 0.5
SIZES = (30, 50, 100, 200, 300, 500, 1000, 2000)
N_REPEATS = 200
SEED = 2607
MODELS = {   # panel -> (number of bumps (0: a straight line), width, alpha)
    1: (8, 0.12, 1e-3),
    2: (0, None, 0.0),
    3: (150, 0.006, 0.1),
}


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


def f_true(x):
    return 1.5 * np.sin(2 * np.pi * x) + 0.5 * x


def design(x, n_bumps, width):
    """A column of ones, then the inputs (a line) or Gaussian bumps evenly spaced on [0, 1]."""
    if n_bumps == 0:
        return np.column_stack([np.ones_like(x), x])
    centres = np.linspace(0, 1, n_bumps)
    return np.column_stack([np.ones_like(x), np.exp(-((x[:, None] - centres) ** 2) / (2 * width ** 2))])


def fit(A, y, alpha):
    """Least squares with a ridge penalty that leaves the column of ones alone."""
    penalty = alpha * np.eye(A.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + penalty, A.T @ y)


def learning_curves():
    """Median training and validation RMSE of the three models, for every size of SIZES."""
    rng = np.random.default_rng(SEED)
    x_val = rng.random(20_000)
    y_val = f_true(x_val) + rng.normal(0, SIGMA, x_val.size)
    curves = {}
    for panel, (n_bumps, width, alpha) in MODELS.items():
        A_val = design(x_val, n_bumps, width)
        train, valid = [], []
        for n in SIZES:
            tr, va = [], []
            for rep in range(N_REPEATS):
                r = np.random.default_rng(1000 * n + rep)
                x = r.random(n)
                y = f_true(x) + r.normal(0, SIGMA, n)
                A = design(x, n_bumps, width)
                w = fit(A, y, alpha)
                tr.append(np.sqrt(np.mean((A @ w - y) ** 2)))
                va.append(np.sqrt(np.mean((A_val @ w - y_val) ** 2)))
            train.append(np.median(tr))
            valid.append(np.median(va))
        curves[panel] = (np.array(train), np.array(valid))
    return curves


def fig_learning_curves() -> dict:
    curves = learning_curves()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (panel, (train, valid)) in zip(axes, curves.items()):
        ax.plot(SIZES, train, marker="o", color=C[0], lw=2, label="entraînement")
        ax.plot(SIZES, valid, marker="s", color=C[1], lw=2, label="validation")
        ax.axhline(SIGMA, color="0.35", ls="--", lw=1.5, label="bruit (plancher)")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(SIZES))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}".replace(",", " ")))
        ax.set(title=f"Panneau {panel}", xlabel="nombre d'exemples d'entraînement", ylim=(0, 1.0),
               yticks=np.arange(0, 1.01, 0.1))
        ax.grid(True, alpha=0.45)
    axes[0].set_ylabel("RMSE")
    axes[0].legend(loc="lower right")
    fig.suptitle("Question CP2.7 : trois modèles, une même tâche, leurs courbes d'apprentissage", y=1.02)
    save(fig, "cp2_7_courbes_apprentissage.png")
    return curves


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    curves = fig_learning_curves()
    for panel, (train, valid) in curves.items():
        print(f"  panel {panel}: train {np.round(train, 3).tolist()}")
        print(f"           valid {np.round(valid, 3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
