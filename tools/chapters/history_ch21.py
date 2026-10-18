"""The two training histories of 📈 Ex 21.23 (chapter 21), too long to run in the notebook.

The convnet of the book (Ex 21.22), with and without its two dropouts, trained 15 epochs with Adam(1e-3) and mini-batches
of 128 on 50 000 Fashion-MNIST training images; after each epoch, the loss and the accuracy are measured in evaluation
mode on 10 000 of these training images and on the 10 000 other training images (the validation), with the confusion
matrix of the validation. About 15 minutes per run on 2 CPUs.

    python tools/chapters/history_ch21.py          # writes tools/chapters/history_ch21.json
"""

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

import wb

OUT = pathlib.Path(__file__).with_name("history_ch21.json")


def make(dropout: bool) -> nn.Sequential:
    drop = (lambda p: [nn.Dropout(p)]) if dropout else (lambda p: [])
    return nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), *drop(0.25),
                         nn.Flatten(), nn.Linear(9216, 128), nn.ReLU(), *drop(0.5), nn.Linear(128, 10))


def run(dropout: bool) -> list:
    torch.manual_seed(2123)
    X, y = wb.datasets.load_fashion_mnist("train", normalize=True, seed=None)
    X, y = torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)
    perm = torch.from_numpy(np.random.default_rng(2123).permutation(60000))
    train, val = perm[:50000], perm[50000:]
    model = make(dropout)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(2123)

    def evaluate(idx):
        model.eval()
        total, correct, preds = 0.0, 0, []
        with torch.no_grad():
            for i in range(0, len(idx), 1000):
                b = idx[i:i + 1000]
                out = model(X[b])
                total += loss_fn(out, y[b]).item() * len(b)
                p = out.argmax(1)
                correct += (p == y[b]).sum().item()
                preds.append(p)
        return total / len(idx), correct / len(idx), torch.cat(preds)

    history, start = [], time.time()
    for epoch in range(15):
        model.train()
        order = train[torch.randperm(len(train), generator=generator)]
        for i in range(0, len(order), 128):
            b = order[i:i + 128]
            optimizer.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            optimizer.step()
        train_loss, train_acc, _ = evaluate(train[:10000])
        val_loss, val_acc, pred = evaluate(val)
        cm = np.zeros((10, 10), int)
        np.add.at(cm, (y[val].numpy(), pred.numpy()), 1)
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                        "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4), "cm": cm.tolist()})
        print(dropout, epoch + 1, round(val_loss, 4), round(val_acc, 4), f"{time.time() - start:.0f} s", flush=True)
    return history


def main() -> None:
    data = {"dropout": run(True), "no_dropout": run(False)}
    OUT.write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
