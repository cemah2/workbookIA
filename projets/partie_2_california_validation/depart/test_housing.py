"""Tests of my module housing.py (mini-project of part II).

    python -m pytest mon_travail/projets/partie_2_california_validation -q

Two tests are given as examples. Write at least four more (step MP2.7 of the notebook
checks that there are six or more), each with a clear name and a message that says what
was expected. NumPy and scikit-learn may serve as oracles: they check, they never build
the model. Ideas, from the deliverables of the brief:
- make_test_indices follows np.random.default_rng(seed).permutation, and refuses
  test_size = 0 or 1;
- rmse equals the square root of scikit-learn's mean_squared_error;
- the bounds low_ and high_ are the percentiles of the training rows only, and a value
  beyond a bound is predicted like the bound itself;
- the prediction of a district does not depend on the other districts predicted with it
  (transform reuses the statistics learnt by fit);
- degree 2 + Ridge gives the predictions of the same chain built with scikit-learn
  (np.percentile and np.clip, StandardScaler, PolynomialFeatures, StandardScaler, Ridge);
- a strong Lasso sets some weights exactly to 0;
- the zone of a district is its nearest k-means centre (latitude and longitude
  standardized with the training statistics), and transform adds one column per zone;
- invalid hyperparameters (degree 0, an unknown penalty, alpha = 0 with Ridge) raise ValueError;
- cross_validate leaves the model it receives unfitted (it fits clones);
- on each validation fold, out_of_fold_predictions gives the predictions of the model of that fold;
- validation_curve and learning_curve return arrays of shape (number of values, number of folds);
- cross_validate, out_of_fold_predictions, validation_curve and learning_curve leave the folds
  they receive unchanged (give them copies, then compare).
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import data
import housing


@pytest.fixture(scope="module")
def districts():
    """3 000 training districts drawn once (never the frozen test districts): the tests stay fast."""
    X, y = data.load_housing()
    n = len(y)
    test = np.random.default_rng(2026).permutation(n)[: int(np.ceil(0.2 * n))]   # the split of MP2.1
    rows = np.random.default_rng(7).choice(np.setdiff1d(np.arange(n), test), 3000, replace=False)
    return X[rows], y[rows]


def test_test_indices_are_sorted_distinct_and_reproducible():
    first = housing.make_test_indices(20640, test_size=0.2, seed=2026)
    assert len(first) == 4128, f"expected ceil(0.2 * 20640) = 4128 test indices, got {len(first)}"
    assert np.all(np.diff(first) > 0), "expected indices sorted in increasing order, without repetition"
    again = housing.make_test_indices(20640, test_size=0.2, seed=2026)
    assert np.array_equal(first, again), "expected the same indices for the same seed"


def test_least_squares_without_bounds_matches_scikit_learn(districts):
    X, y = districts
    mine = housing.HousingModel(clip=None).fit(X[:2000], y[:2000]).predict(X[2000:])
    expected = LinearRegression().fit(X[:2000], y[:2000]).predict(X[2000:])
    assert np.allclose(mine, expected, atol=1e-8), (
        f"expected the predictions of scikit-learn's LinearRegression on the raw features (standardizing does not "
        f"change least squares); largest gap {np.abs(np.asarray(mine) - expected).max():.3g}")


# TODO MP2.7: your tests (at least four more)
