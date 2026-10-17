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


def test_loss_curves_are_reproducible_and_start_together():
    curves = synth.loss_curves()
    assert sorted(curves) == list("ABCDEF")
    assert all(np.shape(c) == (60,) for c in curves.values())
    again = synth.loss_curves()
    assert all(np.array_equal(curves[k], again[k], equal_nan=True) for k in curves)
    assert np.isnan(curves["E"][-1]) and np.all(curves["F"] == curves["F"][0])


def test_optimizer_curves_are_reproducible_and_start_near_log_10():
    curves = synth.optimizer_curves()
    assert sorted(curves) == list("ABCDEF")
    assert all(np.shape(c) == (1200,) and np.all(np.isfinite(c)) and np.all(c > 0) for c in curves.values())
    again = synth.optimizer_curves()
    assert all(np.array_equal(curves[k], again[k]) for k in curves)
    assert all(abs(np.median(c[:5]) - np.log(10)) < 0.6 for c in curves.values())
    assert np.allclose(curves["B"][-300:], np.log(10), atol=0.01)        # curve B: flat at log 10 after its burst
    assert np.std(curves["E"][-300:]) > 3 * np.std(curves["C"][-300:])   # curve E: the noisiest at the end
    assert len(synth.optimizer_curves(n_updates=600)["A"]) == 600
    assert len(synth.optimizer_curves(n_updates=100)["B"]) == 100
    for bad in (99, 20, 600.0, True):
        with pytest.raises(ValueError, match="n_updates"):
            synth.optimizer_curves(n_updates=bad)


def test_training_curves_are_reproducible_and_have_their_shapes():
    curves = synth.training_curves()
    assert sorted(curves) == list("ABCDE")
    for history in curves.values():
        assert sorted(history) == ["train_loss", "val_acc", "val_loss"]
        assert all(np.shape(v) == (30,) and np.all(np.isfinite(v)) and np.all(v > 0) for v in history.values())
        assert np.all(history["val_acc"] <= 1.0)
    again = synth.training_curves()
    assert all(np.array_equal(curves[k][m], again[k][m]) for k in curves for m in curves[k])
    val_b = curves["B"]["val_loss"]
    best = int(val_b.argmin())
    assert best + 1 == 7 and min(val_b[best - 1], val_b[best + 1]) - val_b[best] > 0.01    # a clear minimum to read
    assert val_b[-1] > 2 * val_b[best] and curves["B"]["train_loss"][-1] < 0.1             # B: the gap widens
    assert np.all(curves["D"]["val_loss"][3:] < curves["D"]["train_loss"][3:])            # D: val_loss < train_loss
    assert curves["E"]["val_acc"][-1] > 0.99 and np.allclose(curves["E"]["val_loss"], curves["E"]["train_loss"], rtol=0.06)
    rough = lambda v: np.std(np.diff(v[10:]))
    assert rough(curves["A"]["val_loss"]) > 4 * rough(curves["D"]["val_loss"])               # A: the roughest val_loss
    assert curves["C"]["train_loss"][-1] > 0.75 and curves["C"]["val_acc"][-1] < 0.8         # C: both losses high
    assert len(synth.training_curves(n_epochs=12)["A"]["val_loss"]) == 12
