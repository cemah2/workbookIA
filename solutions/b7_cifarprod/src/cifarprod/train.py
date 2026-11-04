"""The training of cifarprod: the seeds, one evaluation, and train(), which returns the model with its normalization.

    result = train(X, y, X_val, y_val, arch="cnn", widths=(16, 32), epochs=1, lr=0.05, batch_size=128, seed=0)
    result["model"], result["mean"], result["std"], result["history"]["val_acc"]

SGD with momentum 0.9, the cross-entropy, mini-batches in the order of the generator of seed_everything; one INFO line
per epoch, one DEBUG line per mini-batch (logger "cifarprod.train"). Nothing is printed.
"""

from __future__ import annotations

import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import channel_stats, make_loader
from .model import build_model

log = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed Python, NumPy and PyTorch, switch the deterministic algorithms on or off, and return a generator seeded
    with ``seed`` for the DataLoader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(bool(deterministic))
    return torch.Generator().manual_seed(seed)


def evaluate(model: nn.Module, loader) -> tuple[float, float]:
    """(mean cross-entropy, accuracy) of the model on all the examples of the loader, in evaluation mode and without a
    graph; the mode of the model is restored afterwards."""
    was_training = model.training
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for x, t in loader:
            out = model(x)
            total_loss += F.cross_entropy(out, t, reduction="sum").item()
            correct += int((out.argmax(dim=1) == t).sum())
            n += len(t)
    model.train(was_training)
    return total_loss / max(n, 1), correct / max(n, 1)


def train(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, *, arch: str = "cnn",
          widths=(16, 32), epochs: int = 1, lr: float = 0.05, batch_size: int = 128, seed: int = 0,
          deterministic: bool = True, num_workers: int = 0) -> dict:
    """Train ``build_model(arch, widths)`` on the uint8 images X (N, 32, 32, 3) and their labels y.

    The normalization (mean, std) is learned on X only (channel_stats). Returns {"model": the trained model in
    evaluation mode, "mean", "std", "history": {"train_loss", "val_loss", "val_acc"} (one value per epoch),
    "seconds"}."""
    start = time.perf_counter()
    seed_everything(seed, deterministic)               # the initialization of the weights
    mean, std = channel_stats(X)
    model = build_model(arch, widths)
    # the order of the mini-batches: a generator of its own, seeded with the same seed
    loader = make_loader(X, y, mean, std, batch_size, shuffle=True, seed=seed, num_workers=num_workers)
    val_loader = make_loader(X_val, y_val, mean, std, batch_size=500)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    log.info("training %s %s on %d images, %d epoch(s)", arch, list(widths), len(X), epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for step, (x, t) in enumerate(loader, start=1):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), t)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(t)
            n += len(t)
            log.debug("epoch %d batch %d/%d loss %.4f", epoch, step, len(loader), loss.item())
        val_loss, val_acc = evaluate(model, val_loader)
        history["train_loss"].append(total / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        log.info("epoch %d/%d train_loss %.4f val_loss %.4f val_acc %.4f", epoch, epochs, total / n, val_loss, val_acc)
    model.eval()
    return {"model": model, "mean": mean, "std": std, "history": history, "seconds": time.perf_counter() - start}
