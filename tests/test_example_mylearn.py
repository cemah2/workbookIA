"""Example of a mylearn test (session 1): oracle + property tests.

The oracle is NumPy: ``mylearn._example.mean`` must agree with ``np.mean``.
These tests do not reveal the solution; they only describe the expected behaviour.

    pytest tests/test_example_mylearn.py              # your code
    pytest tests/test_example_mylearn.py --impl=ref   # the reference
"""

import numpy as np
import pytest


@pytest.fixture
def example(mylearn_module):
    return mylearn_module("_example")


def test_mean_matches_numpy(example):
    rng = np.random.default_rng(0)
    for _ in range(20):
        values = rng.normal(size=int(rng.integers(1, 50)))
        assert example.mean(values) == pytest.approx(np.mean(values))


def test_mean_small_list(example):
    assert example.mean([1, 2, 3, 4]) == pytest.approx(2.5)


def test_mean_single_value(example):
    assert example.mean([7]) == pytest.approx(7.0)


def test_mean_returns_float(example):
    assert isinstance(example.mean([1, 2]), float)


def test_mean_empty_raises_value_error(example):
    with pytest.raises(ValueError):
        example.mean([])
