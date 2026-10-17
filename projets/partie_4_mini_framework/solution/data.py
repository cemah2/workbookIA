"""Data of the mini-project MP4: Fashion-MNIST (training, validation, and a test set opened once), and two moons.

Provided (do not change it): your notebook and your tests use it.

    X_train, y_train, X_val, y_val = data.load_fashion()   # 50 000 + 10 000 images, flattened, in [0, 1], float64
    X, y = data.load_moons()                                # 1 000 points in 2-D, two classes
    X_test, y_test = data.open_test(PROJECT, "minifw 784-256-128-10 ...")   # once, at the very end (test_log.json)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import wb

N_VALIDATION = 10_000            # the last 10 000 training images of Fashion-MNIST


def load_fashion(n_train: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X_train, y_train, X_val, y_val) of Fashion-MNIST: the first 50 000 training images (or the first n_train of
    them) and the last 10 000 for the validation, flattened to 784 values in [0, 1] (float64); labels in 0-9."""
    X, y = wb.datasets.load_fashion_mnist("train", flatten=True, normalize=True)
    X = X.astype(np.float64)
    split = len(X) - N_VALIDATION
    end = split if n_train is None else min(int(n_train), split)
    return X[:end], y[:end], X[split:], y[split:]


def load_moons(n: int = 1000, noise: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n points of two half-moons (``wb.synth.make_moons``), float64, and their labels 0 or 1."""
    X, y = wb.synth.make_moons(n, noise=noise, seed=seed)
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def test_openings(folder) -> list[dict]:
    """The openings of the test set logged so far in folder / "test_log.json", oldest first."""
    path = Path(folder) / "test_log.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


test_openings.__test__ = False   # not a test for pytest, despite its name


def open_test(folder, description: str) -> tuple[np.ndarray, np.ndarray]:
    """(X_test, y_test): the 10 000 test images of Fashion-MNIST, flattened, in [0, 1] (float64).

    Every opening is logged in folder / "test_log.json" with its date and the description of the model evaluated.
    Opening again for the same description is allowed (the notebook is run again); for another description, it
    still opens, but warns that the test score is no longer an honest measure.
    """
    folder = Path(folder)
    log = test_openings(folder)
    description = str(description)
    if log and all(entry["description"] != description for entry in log):
        print("⚠️ The test set was already opened for another model: its score is no longer an honest measure "
              "(say so in your README).")
    if all(entry["description"] != description for entry in log):
        log.append({"date": dt.date.today().isoformat(), "description": description})
        (folder / "test_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    X, y = wb.datasets.load_fashion_mnist("test", flatten=True, normalize=True)
    return X.astype(np.float64), y


open_test.__test__ = False
