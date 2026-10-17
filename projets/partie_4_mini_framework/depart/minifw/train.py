"""The training loop of minifw: mini-batches, AdamW (or any optimizer of mylearn), warmup + cosine, clipping, early
stopping, and the measure of a model (chapters 19 and 20).

    history = fit(model, SoftmaxCrossEntropy(), AdamW(model.parameters(), lr=1e-3), X_train, y_train, X_val, y_val,
                  epochs=10, batch_size=128, base_lr=1e-3, warmup_steps=100, max_norm=1.0, patience=3,
                  rng=np.random.default_rng(0))

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
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
    raise NotImplementedError("iterate_minibatches")  # TODO MP4.5


def evaluate(model: Sequential, loss_fn, X: np.ndarray, y: np.ndarray, batch_size: int = 1000) -> tuple[float, float]:
    """(mean loss per example, accuracy) of the model on (X, y), in evaluation mode and by slices of batch_size;
    the model goes back to its previous mode afterwards. The accuracy compares the argmax of the outputs to y."""
    raise NotImplementedError("evaluate")  # TODO MP4.5


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
    raise NotImplementedError("fit")  # TODO MP4.5


def library_versions() -> dict[str, str]:
    """The versions that make a result reproducible: Python, NumPy, and PyTorch when it is installed."""
    raise NotImplementedError("library_versions")  # TODO MP4.1
