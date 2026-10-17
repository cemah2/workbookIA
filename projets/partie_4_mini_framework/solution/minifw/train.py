"""The training loop of minifw: mini-batches, AdamW (or any optimizer of mylearn), warmup + cosine, clipping, early
stopping, and the measure of a model (chapters 19 and 20).

    history = fit(model, SoftmaxCrossEntropy(), AdamW(model.parameters(), lr=1e-3), X_train, y_train, X_val, y_val,
                  epochs=10, batch_size=128, base_lr=1e-3, warmup_steps=100, max_norm=1.0, patience=3,
                  rng=np.random.default_rng(0))
"""

from __future__ import annotations

import math
import platform
from collections.abc import Iterator

import numpy as np

from mylearn.nn.regularization import EarlyStopping
from mylearn.optim import clip_grad_norm, cosine_lr

from .model import Sequential


def iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """The index arrays of one epoch: ``rng.permutation(n)`` cut into consecutive slices of batch_size (the last one
    may be smaller)."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def evaluate(model: Sequential, loss_fn, X: np.ndarray, y: np.ndarray, batch_size: int = 1000) -> tuple[float, float]:
    """(mean loss per example, accuracy) of the model on (X, y), in evaluation mode and by slices of batch_size;
    the model goes back to its previous mode afterwards. The accuracy compares the argmax of the outputs to y."""
    was_training = model.training
    model.eval()
    total, correct = 0.0, 0
    try:
        for start in range(0, len(X), batch_size):
            out = model(X[start:start + batch_size])
            loss, _ = loss_fn(out, y[start:start + batch_size])
            total += loss * len(out)
            correct += int((np.argmax(out, axis=1) == y[start:start + batch_size]).sum())
    finally:
        if was_training:
            model.train()
    return float(total / len(X)), correct / len(X)


def fit(model: Sequential, loss_fn, optimizer, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, *,
        epochs: int, batch_size: int = 128, base_lr: float = 1e-3, warmup_steps: int = 0, min_lr: float = 0.0,
        max_norm: float | None = None, patience: int | None = None, rng: np.random.Generator | None = None) -> dict:
    """Train the model, one mini-batch at a time, and return its history.

    At each step t (counted from 0 over the whole training): ``optimizer.lr = cosine_lr(t, total_steps, base_lr,
    warmup_steps, min_lr)`` with ``total_steps = epochs * ceil(len(X) / batch_size)``; the forward pass in training
    mode, the loss, the backward pass; the gradients clipped by ``mylearn.optim.clip_grad_norm`` when max_norm is
    given; ``optimizer.step(model.gradients())``. The mini-batches of each epoch come from
    ``iterate_minibatches(len(X), batch_size, rng)``. After each epoch, ``evaluate`` on (X_val, y_val); with a
    patience, ``mylearn.nn.regularization.EarlyStopping(patience)`` on the validation loss, a ``state_dict()`` kept at
    each improvement, and the best one loaded back at the end.

    Returns a dict: "train_loss", "val_loss", "val_acc" (one value per epoch done; the training loss is the mean per
    example of the losses measured during the epoch), "lr" and "grad_norm" (one value per step; the norm BEFORE
    clipping, None for every step when max_norm is None), and "best_epoch" (1-based, the epoch whose weights the
    model has at the end: the last one without early stopping).
    """
    rng = np.random.default_rng() if rng is None else rng
    total_steps = epochs * math.ceil(len(X) / batch_size)
    stopper = EarlyStopping(patience=patience) if patience else None
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": [], "grad_norm": [], "best_epoch": epochs}
    best_state, step = None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for idx in iterate_minibatches(len(X), batch_size, rng):
            optimizer.lr = cosine_lr(step, total_steps, base_lr, warmup_steps, min_lr)
            out = model(X[idx])
            loss, dout = loss_fn(out, y[idx])
            model.backward(dout)
            grads = model.gradients()
            history["grad_norm"].append(float(clip_grad_norm(grads, max_norm)) if max_norm else None)
            history["lr"].append(optimizer.lr)
            optimizer.step(grads)
            total += loss * len(idx)
            step += 1
        val_loss, val_acc = evaluate(model, loss_fn, X_val, y_val)
        history["train_loss"].append(total / len(X))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if stopper is not None:
            stop = stopper.step(val_loss)
            if stopper.improved_:
                best_state = model.state_dict()
                history["best_epoch"] = epoch
            if stop:
                break
        else:
            history["best_epoch"] = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def library_versions() -> dict[str, str]:
    """The versions that make a result reproducible: Python, NumPy, and PyTorch when it is installed."""
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "not installed"
    return versions
