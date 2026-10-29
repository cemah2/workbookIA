"""The 50-epoch training history of 📈 Ex 23.19 and 🔬 Ex 23.27 (chapter 23), too long to run in the notebook.

The book's MLP (Flatten, Linear(784, 784), ReLU, Linear(784, 10)) on Fashion-MNIST, as in the book's long run
(100 epochs there, 50 here): Adam(lr=1e-3), mini-batches of 256, cross-entropy. Training on the 54 000 images of
the stratified split of the notebook (test_size=0.1, stratify=y, random_state=0), validation on the 6 000 others;
pixels divided by 255, then centred and reduced with the mean and standard deviation of the training images.
After each epoch: the mean training loss and accuracy during the epoch (what Keras prints), and the validation
loss and accuracy (evaluation mode). Seed 2319 for the weights and the order of the mini-batches. About 1.5 s per
epoch on 2 CPUs.

    python tools/chapters/history_ch23.py          # writes tools/chapters/history_ch23.json
"""

import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import wb  # noqa: E402

OUT = pathlib.Path(__file__).with_name("history_ch23.json")
EPOCHS = 50
SEED = 2319


def preprocess(images, mean, std):
    x = (images.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1)


def main() -> int:
    torch.set_num_threads(2)
    X, y = wb.datasets.load_fashion_mnist("train", seed=None)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
    mean, std = float((X_tr / 255.0).mean()), float((X_tr / 255.0).std())
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 784), nn.ReLU(), nn.Linear(784, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(preprocess(X_tr, mean, std), torch.from_numpy(y_tr)), batch_size=256,
                              shuffle=True, generator=torch.Generator().manual_seed(SEED))
    x_val, t_val = preprocess(X_val, mean, std), torch.from_numpy(y_val)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, count = 0.0, 0, 0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            count += len(xb)
        model.eval()
        with torch.inference_mode():
            logits = model(x_val)
            val_loss = loss_fn(logits, t_val).item()
            val_acc = (logits.argmax(dim=1) == t_val).float().mean().item()
        history["train_loss"].append(round(total / count, 6))
        history["train_acc"].append(round(correct / count, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        print(f"epoch {epoch:2d} · train {total / count:.4f} {correct / count:.4f} · val {val_loss:.4f} {val_acc:.4f} · "
              f"{time.perf_counter() - start:.0f} s", flush=True)
    meta = {"model": "Flatten, Linear(784, 784), ReLU, Linear(784, 10)", "optimizer": "Adam(lr=1e-3)",
            "batch_size": 256, "epochs": EPOCHS, "seed": SEED, "n_train": len(X_tr), "n_val": len(X_val),
            "torch": torch.__version__, "generated_by": "tools/chapters/history_ch23.py"}
    OUT.write_text(json.dumps({"meta": meta, **history}, indent=1) + "\n", encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
