"""Fixtures of the tests of cifarprod: a fresh model, the data file of the project, a small synthetic data file."""

from pathlib import Path

import numpy as np
import pytest
import torch

from cifarprod.data import synthetic
from cifarprod.model import build_model

DATA = Path(__file__).resolve().parents[1] / "data" / "cifar10_2000.npz"


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model(widths=(12, 24))


@pytest.fixture
def data_file():
    """The CIFAR-10 subset of the project; the tests that need it are skipped without it (in the CI, for example)."""
    if not DATA.is_file():
        pytest.skip(f"no data file {DATA}")
    return DATA


@pytest.fixture
def synthetic_file(tmp_path):
    """A small .npz file with the schema of the data (96 training and 32 validation random images)."""
    X, y = synthetic(96, seed=0)
    X_val, y_val = synthetic(32, seed=1)
    path = tmp_path / "synthetic.npz"
    np.savez(path, x_train=X, y_train=y, x_val=X_val, y_val=y_val)
    return path
