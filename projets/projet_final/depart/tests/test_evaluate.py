"""Tests of evaluate.py (PF.8, PF.9), marked « todo »: the bootstrap and the metrics, on toy predictions."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from conftest import PACKAGES

pytestmark = pytest.mark.todo
PACKAGE = sorted(PACKAGES)[0]
evaluate = pytest.importorskip(f"{PACKAGE}.evaluate")
Config = pytest.importorskip(f"{PACKAGE}.config").Config


@pytest.fixture
def predictions():
    """400 binary labels and predictions that are right 80 % of the time."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    wrong = rng.random(400) < 0.2
    return y, np.where(wrong, 1 - y, y)


def test_the_main_metric_is_the_one_of_sklearn(predictions):
    y, p = predictions
    metrics = evaluate.compute_metrics(y, p, Config(metric="f1_macro"))
    expected = f1_score(y, p, average="macro")
    assert abs(metrics["f1_macro"] - expected) < 1e-12, f"expected f1_macro = {expected:.6f}, got {metrics['f1_macro']}"


def test_the_bootstrap_interval_holds_the_estimate_and_is_reproducible(predictions):
    y, p = predictions
    low, high = evaluate.bootstrap_ci(y, p, accuracy_score, n_boot=500, seed=0)
    again = evaluate.bootstrap_ci(y, p, accuracy_score, n_boot=500, seed=0)
    estimate = accuracy_score(y, p)
    assert low < estimate < high, f"expected {estimate:.3f} inside the interval, got [{low:.3f}, {high:.3f}]"
    assert (low, high) == tuple(again), "expected the same interval for the same seed"
    assert 0.05 < high - low < 0.12, f"expected a width near 0.08 for 400 examples at 80 %, got {high - low:.3f}"


def test_the_interval_narrows_with_more_examples(predictions):
    y, p = predictions
    small = evaluate.bootstrap_ci(y[:100], p[:100], accuracy_score, n_boot=500, seed=0)
    large = evaluate.bootstrap_ci(np.tile(y, 4), np.tile(p, 4), accuracy_score, n_boot=500, seed=0)
    assert large[1] - large[0] < small[1] - small[0], f"expected a narrower interval with 16 times more examples"


def test_a_model_compared_with_itself_differs_by_zero(predictions):
    y, p = predictions
    result = evaluate.paired_bootstrap(y, p, p, accuracy_score, n_boot=200, seed=0)
    assert result["diff"] == 0 and result["low"] == 0 and result["high"] == 0, \
        f"expected a difference of exactly 0 with itself, got {result}"
