"""The losses of minifw, computed on the logits like PyTorch (mylearn, chapter 18).

A loss is called on the output of the model and the targets, and returns the pair
``(loss, dL/d(output))``: a Python float (the MEAN over the mini-batch, in nats) and an
array of the shape of the output, that ``Sequential.backward`` takes.

Your version: write every function that raises NotImplementedError (the step of the
notebook is given in the TODO). Keep the classes, the signatures and the docstrings: the
notebook and the tests rely on them. The __init__ methods and the functions marked
"Provided" are already written. You may add private helpers (names starting with _).
"""

from __future__ import annotations

import numpy as np

from mylearn.nn.backward import mse_loss, softmax_cross_entropy


class SoftmaxCrossEntropy:
    """The cross-entropy of the softmax of the logits, like ``torch.nn.CrossEntropyLoss()``: logits of shape (n, K),
    integer labels in [0, K) of shape (n,)."""

    def __call__(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """(mean loss, dL/dlogits), with ``mylearn.nn.backward.softmax_cross_entropy``."""
        raise NotImplementedError("SoftmaxCrossEntropy.__call__")  # TODO MP4.3


class MSELoss:
    """The mean squared error, like ``torch.nn.MSELoss()``: predictions and targets of the same shape."""

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
        """(mean loss, dL/dy_pred), with ``mylearn.nn.backward.mse_loss``."""
        raise NotImplementedError("MSELoss.__call__")  # TODO MP4.3
