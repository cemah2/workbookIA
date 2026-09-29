"""Tests of wb.synth: shapes, dtypes, reproducibility, known values."""

import numpy as np
import pytest

from wb import synth


@pytest.mark.parametrize(
    "make, kwargs, n_classes",
    [
        (synth.make_moons, {"n": 101}, 2),
        (synth.make_circles, {"n": 101}, 2),
        (synth.make_blobs, {"n": 101, "centers": 4}, 4),
        (synth.make_spirals, {"n": 101, "n_classes": 3}, 3),
        (synth.make_xor, {"n": 101}, 2),
    ],
)
def test_classification_generators(make, kwargs, n_classes):
    X, y = make(**kwargs, seed=3)
    assert X.shape == (101, 2) and y.shape == (101,)
    assert X.dtype == np.float64 and y.dtype == np.int64
    assert set(np.unique(y)) == set(range(n_classes))
    X2, y2 = make(**kwargs, seed=3)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)
    X3, _ = make(**kwargs, seed=4)
    assert not np.array_equal(X, X3)


def test_blobs_with_explicit_centres():
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    X, y = synth.make_blobs(n=400, centers=centers, std=0.5, seed=0)
    for k in range(2):
        np.testing.assert_allclose(X[y == k].mean(axis=0), centers[k], atol=0.15)


@pytest.mark.parametrize("gate, expected", [("and", [0, 0, 0, 1]), ("or", [0, 1, 1, 1]),
                                            ("xor", [0, 1, 1, 0]), ("nand", [1, 1, 1, 0])])
def test_logic_gates(gate, expected):
    X, y = synth.logic_gate(gate)
    assert X.shape == (4, 2)
    assert y.tolist() == expected


def test_logic_gate_unknown():
    with pytest.raises(ValueError):
        synth.logic_gate("maybe")


def test_linear_and_polynomial():
    X, y = synth.make_linear(n=500, w=(2.0, -1.0), b=0.5, noise=0.0, seed=0)
    np.testing.assert_allclose(y, X @ np.array([2.0, -1.0]) + 0.5)
    x, y, f = synth.make_polynomial(n=20, coefs=(1.0, 0.0, 2.0), noise=0.0, seed=0)
    assert np.all(np.diff(x) >= 0)
    np.testing.assert_allclose(y, 1 + 2 * x**2)
    np.testing.assert_allclose(f(np.array([0.0, 1.0])), [1.0, 3.0])


def test_noisy_sine_without_noise_is_a_sine():
    t, y = synth.noisy_sine(n=101, freq=0.5, noise=0.0, t_max=4.0)
    np.testing.assert_allclose(y, np.sin(np.pi * t), atol=1e-12)


def test_rosenbrock_minimum_and_gradient():
    assert synth.rosenbrock(1.0, 1.0) == 0.0
    dx, dy = synth.rosenbrock_grad(1.0, 1.0)
    assert dx == 0.0 and dy == 0.0
    x0, y0, h = -0.7, 1.3, 1e-6
    num_dx = (synth.rosenbrock(x0 + h, y0) - synth.rosenbrock(x0 - h, y0)) / (2 * h)
    num_dy = (synth.rosenbrock(x0, y0 + h) - synth.rosenbrock(x0, y0 - h)) / (2 * h)
    dx, dy = synth.rosenbrock_grad(x0, y0)
    assert dx == pytest.approx(num_dx, rel=1e-5) and dy == pytest.approx(num_dy, rel=1e-5)


def test_coin_flips_and_gaussian():
    flips = synth.coin_flips(20000, p=0.3, seed=1)
    assert set(np.unique(flips)) <= {0, 1}
    assert flips.mean() == pytest.approx(0.3, abs=0.02)
    with pytest.raises(ValueError):
        synth.coin_flips(10, p=1.5)
    g = synth.gaussian_1d(20000, mu=4.0, sigma=1.25, seed=0)
    assert g.mean() == pytest.approx(4.0, abs=0.05) and g.std() == pytest.approx(1.25, abs=0.05)
