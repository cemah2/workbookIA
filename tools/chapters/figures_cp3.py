#!/usr/bin/env python
"""Draw the figure of the mock exam of part III (question CP3.8), used by Claude.

    python tools/chapters/figures_cp3.py

Writes checkpoints/partie_3/figures/cp3_8_courbes.png, computed on 6 000 California districts
(rows drawn with np.random.default_rng(2608), the eight measures, the target MedHouseVal):
- panel (a): a DecisionTreeRegressor(random_state=0), mean training and validation RMSE of a
  5-fold cross-validation (KFold(5, shuffle=True, random_state=0)) for max_depth = 1 to 20; the
  two curves are numbered 1 and 2 (curve 1 = validation, curve 2 = training), the question asks which
  one is the training curve;
- panel (b): two GradientBoostingRegressor(max_depth=3, random_state=0) with learning rates 1 and
  0.1, 400 trees, trained on 75 % of the districts (train_test_split(random_state=0)); training and
  validation RMSE after each tree (staged_predict).
Nothing comes from the book.
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

import wb  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "checkpoints" / "partie_3" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
DEPTHS = np.arange(1, 21)
N_TREES = 400
RATES = (1.0, 0.1)


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


def data():
    """The 6 000 districts of the figure: (X, y)."""
    california = wb.datasets.load_california()
    features = california.columns[:8].tolist()
    rows = np.sort(np.random.default_rng(2608).choice(len(california), 6000, replace=False))
    return california[features].to_numpy(dtype=float)[rows], california["MedHouseVal"].to_numpy(dtype=float)[rows]


def tree_curves(X, y):
    from sklearn.model_selection import KFold, validation_curve
    from sklearn.tree import DecisionTreeRegressor
    train, valid = validation_curve(DecisionTreeRegressor(random_state=0), X, y, param_name="max_depth",
                                    param_range=DEPTHS, cv=KFold(5, shuffle=True, random_state=0),
                                    scoring="neg_root_mean_squared_error")
    return -train.mean(axis=1), -valid.mean(axis=1)


def boosting_curves(X, y):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=0)
    curves = {}
    for rate in RATES:
        model = GradientBoostingRegressor(n_estimators=N_TREES, learning_rate=rate, max_depth=3,
                                          random_state=0).fit(X_tr, y_tr)
        train = [np.sqrt(np.mean((y_tr - p) ** 2)) for p in model.staged_predict(X_tr)]
        valid = [np.sqrt(np.mean((y_va - p) ** 2)) for p in model.staged_predict(X_va)]
        curves[rate] = (np.array(train), np.array(valid))
    return curves


def fig_curves() -> tuple:
    X, y = data()
    train, valid = tree_curves(X, y)
    boosting = boosting_curves(X, y)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax_a.plot(DEPTHS, valid, marker="o", color=C[0], lw=2, label="courbe ①")
    ax_a.plot(DEPTHS, train, marker="s", color=C[1], lw=2, label="courbe ②")
    ax_a.set(title="(a) Arbre de régression, validation croisée à 5 folds", xlabel="max_depth", ylabel="RMSE",
             xticks=DEPTHS, ylim=(0, 1.0), yticks=np.arange(0, 1.01, 0.1))
    ax_a.grid(True, alpha=0.45)
    ax_a.legend(loc="lower left")
    trees = np.arange(1, N_TREES + 1)
    for (rate, (tr, va)), colour in zip(boosting.items(), (C[2], C[3])):
        label = french_number(rate)
        ax_b.plot(trees, va, color=colour, lw=2.2, label=f"learning rate {label}, validation")
        ax_b.plot(trees, tr, color=colour, lw=1.6, ls="--", label=f"learning rate {label}, entraînement")
    ax_b.set(title="(b) Gradient boosting (arbres de profondeur 3)", xlabel="nombre d'arbres", ylabel="RMSE",
             xlim=(0, N_TREES), xticks=np.arange(0, N_TREES + 1, 50), ylim=(0, 1.0), yticks=np.arange(0, 1.01, 0.1))
    ax_b.grid(True, alpha=0.45)
    ax_b.legend(loc="upper right", fontsize=9)
    fig.suptitle("Question CP3.8 : 6 000 districts de Californie", y=1.02)
    save(fig, "cp3_8_courbes.png")
    return train, valid, boosting


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    train, valid, boosting = fig_curves()
    print("  tree train:", np.round(train, 3).tolist())
    print("  tree valid:", np.round(valid, 3).tolist(), "best depth", int(DEPTHS[np.argmin(valid)]))
    for rate, (tr, va) in boosting.items():
        print(f"  boosting {rate}: best trees {int(np.argmin(va)) + 1} (valid {va.min():.3f}), "
              f"at {N_TREES}: valid {va[-1]:.3f}, train {tr[-1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
