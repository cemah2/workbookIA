#!/usr/bin/env python
"""Draw the figure of the mock exam of part V (question CP5.7), used by Claude.

    python tools/chapters/figures_cp5.py

Writes checkpoints/partie_5/figures/cp5_7_courbes.png: four real trainings, the training loss (measured
during the epoch, as Keras and the workbook's train_classifier print it) and the validation loss (measured
after the epoch, in eval mode):
- panel 1: the MLP 784-32-32-10 with dropout 0.2 and max-norm 3 of chapter 24 (📈 24.14), Adam(1e-3),
  30 epochs on 54 000 Fashion-MNIST images (tools/chapters/history_ch24.json, entry mlp14);
- panel 2: the MLP 784-32-32-10 without dropout, Adam with a learning rate far too large (0.2), 30 epochs on
  6 000 Fashion-MNIST images (trained here, torch.manual_seed(57));
- panel 3: the deep CNN of 📦 24.20 with batchnorm, Adam(1e-3), 20 epochs on 20 000 CIFAR-10 images
  (history_ch24.json, entry cnn21, variant "batchnorm");
- panel 4: the same MLP as panel 2 with a learning rate far too small (Adam, 3e-6), 30 epochs (trained here).
The panel numbers are what the question asks to diagnose; nothing comes from the book.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import wb  # noqa: E402
from wb import plot as wbplot  # noqa: E402

OUT = ROOT / "checkpoints" / "partie_5" / "figures"
HISTORY = json.loads((ROOT / "tools" / "chapters" / "history_ch24.json").read_text(encoding="utf-8"))
C = wbplot.CATEGORICAL
DPI = 110


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


def train_mlp(lr: float, epochs: int = 30):
    """(train losses during the epoch, validation losses) of the MLP 784-32-32-10, Adam(lr), on Fashion-MNIST."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    torch.set_num_threads(2)
    X, y = wb.datasets.load_fashion_mnist("train", seed=None)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
    X_tr, y_tr = X_tr[:6000], y_tr[:6000]
    mean, std = float((X_tr / 255.0).mean()), float((X_tr / 255.0).std())

    def prep(images):
        return torch.from_numpy(((images.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)).flatten(1)

    x_tr, t_tr, x_va, t_va = prep(X_tr), torch.from_numpy(y_tr), prep(X_va), torch.from_numpy(y_va)
    torch.manual_seed(57)
    model = nn.Sequential(nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 10))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(57)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x_tr), generator=generator).split(256):
            loss = loss_fn(model(x_tr[idx]), t_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(x_va), t_va).item())
        train_losses.append(total / len(x_tr))
    return np.array(train_losses), np.array(val_losses)


def fig_curves():
    mlp14, cnn = HISTORY["mlp14"], HISTORY["cnn21"]["batchnorm"]
    curves = [(np.array(mlp14["running_loss"]), np.array(mlp14["val_loss"])),
              train_mlp(0.2),
              (np.array(cnn["train_loss"]), np.array(cnn["val_loss"])),
              train_mlp(3e-6)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    for k, (ax, (tr, va)) in enumerate(zip(axes.ravel(), curves), start=1):
        epochs = np.arange(1, len(tr) + 1)
        ax.plot(epochs, tr, color=C[1], lw=2, ls="--", marker="s", ms=3, label="entraînement")
        ax.plot(epochs, va, color=C[0], lw=2, marker="o", ms=3, label="validation")
        ax.set(title=f"Courbes {'①②③④'[k - 1]}", xlabel="epoch", ylabel="loss (cross-entropy)", xlim=(0, len(tr) + 1))
        ax.set_ylim(0, max(1.0, float(np.nanmax(np.r_[tr, va])) * 1.08))
        ax.grid(True, alpha=0.45)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="lower left" if k == 4 else "upper right")
    fig.suptitle("Question CP5.7 : quatre entraînements réels (MLP sur Fashion-MNIST, CNN sur CIFAR-10)", y=1.0)
    fig.tight_layout()
    save(fig, "cp5_7_courbes.png")
    return curves


def main() -> int:
    wbplot.set_style()
    print(f"Figures -> {OUT.relative_to(ROOT)}")
    for k, (tr, va) in enumerate(fig_curves(), start=1):
        pick = [0, 4, 9, len(tr) - 1]
        print(f"  panel {k}: train {np.round(tr[pick], 3).tolist()} · val {np.round(va[pick], 3).tolist()}"
              f" · best val epoch {int(np.argmin(va)) + 1} ({va.min():.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
