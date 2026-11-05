"""Tests of models.py and train.py (PF.5 to PF.7), marked « todo ».

Two of them are generic (the naive model is constant, a seed fixes the weights). The others need a small batch of
YOUR inputs: write the fixture ``batch`` (a few examples built in memory, not the dataset), then the shape test and
the overfitting test (a model that cannot learn 8 examples by heart has a bug: B7.15).
"""

import numpy as np
import pandas as pd
import pytest

from conftest import PACKAGES

pytestmark = pytest.mark.todo
PACKAGE = sorted(PACKAGES)[0]
models = pytest.importorskip(f"{PACKAGE}.models")
train = pytest.importorskip(f"{PACKAGE}.train")
Config = pytest.importorskip(f"{PACKAGE}.config").Config


def test_the_naive_model_predicts_one_value():
    X = pd.DataFrame({"id": range(40), "x": np.linspace(0, 1, 40)})
    y = np.array(["a"] * 30 + ["b"] * 10)
    predictions = models.build_naive(Config(task="classification")).fit(X, y).predict(X)
    assert set(predictions) == {"a"}, f"expected the majority class « a » everywhere, got {sorted(set(predictions))}"


def test_the_same_seed_gives_the_same_weights():
    import torch

    train.set_seed(0)
    first = models.build_model(Config())
    train.set_seed(0)
    second = models.build_model(Config())
    different = [k for k, v in first.state_dict().items() if not torch.equal(v, second.state_dict()[k])]
    assert not different, f"expected the same initial weights after set_seed(0), different: {different[:5]}"


@pytest.fixture
def batch():
    """A mini-batch (inputs, targets) of YOUR data, built in memory: 8 examples, the shapes of the real ones."""
    pytest.skip("⏳ à écrire : la fixture batch (PF.6)")


def test_the_output_has_one_row_per_example(batch):
    inputs, targets = batch
    out = models.build_model(Config())(inputs)
    assert out.shape[0] == len(targets), f"expected {len(targets)} rows of output, got shape {tuple(out.shape)}"


def test_the_model_overfits_a_tiny_batch(batch):
    pytest.skip("⏳ à écrire : 100 pas d'AdamW sur le batch, la loss doit tomber sous 10 % de sa valeur (PF.6)")
