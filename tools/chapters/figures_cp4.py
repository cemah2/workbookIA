#!/usr/bin/env python
"""Draw the figure of the mock exam of part IV (question CP4.8), used by Claude.

    python tools/chapters/figures_cp4.py

Writes checkpoints/partie_4/figures/cp4_8_courbes.png: four real trainings of a MLP 784-256-10
(ReLU, PyTorch, CPU, torch.manual_seed(48)) on MNIST, 30 epochs, mini-batches of 64, training loss
measured during the epoch and validation loss after it:
- panel 1: SGD with a learning rate far too large (1.5), 3 000 training images;
- panel 2: Adam (3e-3) on 300 training images, no regularization;
- panel 3: SGD with a tiny learning rate (2e-4), 3 000 images;
- panel 4: Adam (3e-4) with a dropout of 0.5, 6 000 images.
The validation set is the same 2 000 other images for the four panels. The panel numbers are what the
question asks to diagnose; nothing comes from the book.
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

OUT = ROOT / "checkpoints" / "partie_4" / "figures"
C = wbplot.CATEGORICAL
DPI = 110
EPOCHS = 30
PANELS = [  # (optimizer, learning rate, training images, dropout)
    ("sgd", 1.5, 3000, 0.0),
    ("adam", 3e-3, 300, 0.0),
    ("sgd", 2e-4, 3000, 0.0),
    ("adam", 3e-4, 6000, 0.5),
]


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


def train(optimizer: str, lr: float, n_train: int, dropout: float):
    """(train losses, validation losses) per epoch of one training."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
    X, y = wb.datasets.load_mnist("train", n=8000, flatten=True, normalize=True)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    X_tr, y_tr, X_va, y_va = X[:n_train], y[:n_train], X[6000:8000], y[6000:8000]
    torch.manual_seed(48)
    layers = [nn.Linear(784, 256), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else []) + [nn.Linear(256, 10)]
    model = nn.Sequential(*layers)
    opt = (torch.optim.SGD(model.parameters(), lr=lr) if optimizer == "sgd"
           else torch.optim.Adam(model.parameters(), lr=lr))
    generator = torch.Generator().manual_seed(48)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    for _ in range(EPOCHS):
        model.train()
        total = 0.0
        for idx in torch.randperm(n_train, generator=generator).split(64):
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_va), y_va).item()
        train_losses.append(min(total / n_train, 50.0))         # a diverging loss is drawn at the top of the panel
        val_losses.append(min(val, 50.0) if np.isfinite(val) else 50.0)
    return np.array(train_losses), np.array(val_losses)


def fig_curves():
    curves = [train(*panel) for panel in PANELS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    epochs = np.arange(1, EPOCHS + 1)
    for k, (ax, (tr, va)) in enumerate(zip(axes.ravel(), curves), start=1):
        ax.plot(epochs, tr, color=C[1], lw=2, ls="--", marker="s", ms=3, label="entraînement")
        ax.plot(epochs, va, color=C[0], lw=2, marker="o", ms=3, label="validation")
        ax.set(title=f"Courbes {'①②③④'[k - 1]}", xlabel="epoch", ylabel="loss (cross-entropy)", xlim=(0, EPOCHS + 1))
        top = max(3.0, float(np.nanmax(np.r_[tr, va])) * 1.05) if k == 1 else 2.6
        ax.set_ylim(0, top)
        ax.grid(True, alpha=0.45)
        ax.legend(loc="center right" if k == 3 else "upper right")
    fig.suptitle("Question CP4.8 : quatre entraînements d'un MLP sur MNIST", y=1.0)
    fig.tight_layout()
    save(fig, "cp4_8_courbes.png")
    return curves


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    for k, (tr, va) in enumerate(fig_curves(), start=1):
        print(f"  panel {k}: train {np.round(tr[[0, 4, 9, 19, 29]], 3).tolist()} · val {np.round(va[[0, 4, 9, 19, 29]], 3).tolist()}"
              f" · best val epoch {int(np.argmin(va)) + 1} ({va.min():.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
