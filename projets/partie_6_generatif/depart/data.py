"""Data of the mini-project MP6: Fashion-MNIST, cut once for all into a training set and a validation set.

Provided (do not change it): your notebook and your tests use it.

    X_train, y_train, X_val, y_val = data.load_split()   # uint8 images (N, 28, 28) and their labels 0-9

The 60 000 training images of Fashion-MNIST are shuffled by a permutation of fixed seed (SPLIT_SEED): the first
10 000 are the validation set, the other 50 000 the training set. The generators and the reference classifier learn
on the training set only; the validation images are the real images the generated ones are compared with. The
10 000 test images of Fashion-MNIST are not used by this project (nothing is chosen on them, nothing is reported on
them): they stay available for the final project.
"""

from __future__ import annotations

import numpy as np

import wb

CLASSES = list(wb.datasets.FASHION_CLASSES)
CLASSES_FR = ["T-shirt", "pantalon", "pull", "robe", "manteau", "sandale", "chemise", "basket", "sac", "bottine"]
N_VAL = 10_000
SPLIT_SEED = 2026


def load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X_train, y_train, X_val, y_val): the 50 000 training and 10 000 validation images of the project, uint8
    (N, 28, 28), and their labels (int64, 0-9). Always the same split. The first call downloads Fashion-MNIST
    (about 30 MB; then: the cache of the workbook)."""
    X, y = wb.datasets.load_fashion_mnist("train", seed=None)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(y))
    val, train = order[:N_VAL], order[N_VAL:]
    X, y = np.ascontiguousarray(X, dtype=np.uint8), np.asarray(y, dtype=np.int64)
    return X[train], y[train], X[val], y[val]
