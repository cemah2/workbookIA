"""The losses of minifw, computed on the logits like PyTorch (mylearn, chapter 18).

A loss is called on the output of the model and the targets, and returns the pair
``(loss, dL/d(output))``: a Python float (the MEAN over the mini-batch, in nats) and an
array of the shape of the output, that ``Sequential.backward`` takes.
"""

from __future__ import annotations

import numpy as np

from mylearn.nn.backward import mse_loss, softmax_cross_entropy


class SoftmaxCrossEntropy:
    """The cross-entropy of the softmax of the logits, like ``torch.nn.CrossEntropyLoss()``: logits of shape (n, K),
    integer labels in [0, K) of shape (n,)."""

    def __call__(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """(mean loss, dL/dlogits), with ``mylearn.nn.backward.softmax_cross_entropy``."""
        loss, grad = softmax_cross_entropy(np.asarray(logits, dtype=np.float64), np.asarray(y))
        return float(loss), grad


class MSELoss:
    """The mean squared error, like ``torch.nn.MSELoss()``: predictions and targets of the same shape."""

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
        """(mean loss, dL/dy_pred), with ``mylearn.nn.backward.mse_loss``."""
        loss, grad = mse_loss(np.asarray(y_pred, dtype=np.float64), np.asarray(y_true, dtype=np.float64))
        return float(loss), grad
