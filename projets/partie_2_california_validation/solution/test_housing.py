"""Tests of the module housing.py (mini-project of part II, reference solution).

    python -m pytest projets/partie_2_california_validation/solution -q

The oracles come from NumPy and scikit-learn (used only to check, never to build the model).
The tests run on 3 000 districts drawn once, to stay fast.
"""

import numpy as np
import pytest
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import data
import housing
from mylearn import model_selection


@pytest.fixture(scope="module")
def districts():
    """3 000 training districts drawn once (never the frozen test districts): the tests stay fast."""
    X, y = data.load_housing()
    n = len(y)
    test = np.random.default_rng(2026).permutation(n)[: int(np.ceil(0.2 * n))]   # the split of MP2.1
    rows = np.random.default_rng(7).choice(np.setdiff1d(np.arange(n), test), 3000, replace=False)
    return X[rows], y[rows]


@pytest.fixture(scope="module")
def folds(districts):
    return model_selection.kfold_indices(len(districts[1]), 3, shuffle=True, rng=np.random.default_rng(0))


def copied(folds):
    """Copies of the folds: a function that changed them could not spoil the fixture shared by the other tests."""
    return [(np.array(train), np.array(val)) for train, val in folds]


def sklearn_twin(X_train, y_train, X_new, degree, alpha, clip=(1.0, 99.0), penalty="ridge"):
    """The same chain built with NumPy and scikit-learn: bounds, StandardScaler, PolynomialFeatures, StandardScaler, model."""
    low, high = np.percentile(X_train, list(clip), axis=0) if clip else (-np.inf, np.inf)
    first = StandardScaler().fit(np.clip(X_train, low, high))
    poly = PolynomialFeatures(degree, include_bias=False)
    second = StandardScaler().fit(poly.fit_transform(first.transform(np.clip(X_train, low, high))))

    def design(X):
        return second.transform(poly.transform(first.transform(np.clip(X, low, high))))

    model = {"ridge": Ridge(alpha=alpha), "lasso": Lasso(alpha=alpha, tol=1e-10, max_iter=100_000),
             "none": LinearRegression()}[penalty]
    return model.fit(design(X_train), y_train).predict(design(X_new))


# --- make_test_indices -------------------------------------------------------------------------------------------

def test_test_indices_have_the_right_size_and_no_repetition():
    test = housing.make_test_indices(20640, test_size=0.2, seed=2026)
    assert len(test) == 4128, f"expected ceil(0.2 * 20640) = 4128 test indices, got {len(test)}"
    assert len(np.unique(test)) == len(test), "a test index appears twice"
    assert test.min() >= 0 and test.max() < 20640, "test indices must lie between 0 and n - 1"
    assert np.all(np.diff(test) > 0), "the test indices must be sorted in increasing order"


def test_test_indices_follow_the_permutation_of_the_seed():
    expected = np.sort(np.random.default_rng(11).permutation(103)[:21])
    got = housing.make_test_indices(103, test_size=0.2, seed=11)
    assert np.array_equal(got, expected), (
        f"expected the first ceil(0.2 * 103) = 21 values of np.random.default_rng(11).permutation(103), sorted; "
        f"got {got.tolist()}")


@pytest.mark.parametrize("n, size, why", [(1, 0.2, "one sample cannot be split"), (100, 0.0, "an empty test set"),
                                          (100, 1.0, "no training sample left")],
                         ids=["one-sample", "test-size-0", "test-size-1"])
def test_test_indices_reject_invalid_inputs(n, size, why):
    with pytest.raises(ValueError):
        housing.make_test_indices(n, test_size=size)


# --- rmse ---------------------------------------------------------------------------------------------------------

def test_rmse_matches_scikit_learn():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=50), rng.normal(size=50)
    expected = float(np.sqrt(mean_squared_error(a, b)))
    assert housing.rmse(a, b) == pytest.approx(expected), f"expected sqrt(mean_squared_error) = {expected}"


# --- HousingModel -------------------------------------------------------------------------------------------------

def test_clone_of_a_model_has_nothing_learnt(districts):
    X, y = districts
    model = housing.HousingModel(degree=2, penalty="ridge", alpha=3.0, n_zones=4).fit(X, y)
    fresh = model_selection.clone(model)
    learnt = [name for name in vars(fresh) if name.endswith("_")]
    assert learnt == [], f"expected a clone without learnt attributes, got {learnt}"
    assert (fresh.degree, fresh.penalty, fresh.alpha, fresh.n_zones) == (2, "ridge", 3.0, 4), \
        "expected the clone to keep the hyperparameters"


def test_least_squares_without_bounds_matches_scikit_learn(districts):
    X, y = districts
    predictions = housing.HousingModel(degree=1, penalty="none", clip=None).fit(X[:2000], y[:2000]).predict(X[2000:])
    expected = LinearRegression().fit(X[:2000], y[:2000]).predict(X[2000:])
    assert np.allclose(predictions, expected, atol=1e-8), (
        f"expected the predictions of LinearRegression on the raw features (standardizing does not change least "
        f"squares); largest gap {np.abs(predictions - expected).max():.3g}")


@pytest.mark.parametrize("degree, alpha", [(2, 1.0), (3, 30.0)], ids=["degree-2", "degree-3"])
def test_ridge_chain_matches_scikit_learn(districts, degree, alpha):
    X, y = districts
    model = housing.HousingModel(degree=degree, penalty="ridge", alpha=alpha).fit(X[:2000], y[:2000])
    expected = sklearn_twin(X[:2000], y[:2000], X[2000:], degree, alpha)
    gap = np.abs(model.predict(X[2000:]) - expected).max()
    assert gap < 1e-6, (f"expected the predictions of the same chain built with scikit-learn (percentiles 1 and 99, "
                        f"StandardScaler, PolynomialFeatures, StandardScaler, Ridge); largest gap {gap:.3g}")


def test_lasso_chain_matches_scikit_learn_and_zeroes_weights(districts):
    X, y = districts
    model = housing.HousingModel(degree=2, penalty="lasso", alpha=0.05).fit(X[:2000], y[:2000])
    expected = sklearn_twin(X[:2000], y[:2000], X[2000:], 2, 0.05, penalty="lasso")
    gap = np.abs(model.predict(X[2000:]) - expected).max()
    assert gap < 1e-4, f"expected the predictions of scikit-learn's Lasso on the same design; largest gap {gap:.3g}"
    zeros = int(np.sum(np.asarray(model.regressor_.coef_) == 0))
    assert zeros > 0, "expected some weights exactly 0 with alpha = 0.05 (the Lasso selects features)"


def test_bounds_are_the_percentiles_of_the_training_rows(districts):
    X, y = districts
    model = housing.HousingModel(clip=(2.0, 98.0)).fit(X[:1000], y[:1000])
    low, high = np.percentile(X[:1000], [2.0, 98.0], axis=0)
    assert np.allclose(model.low_, low) and np.allclose(model.high_, high), \
        "expected low_ and high_ = the percentiles 2 and 98 of the training rows only"
    extreme = X[1000:1001].copy()
    extreme[0, 5] = 1e6                                    # an absurd mean occupancy
    bounded = np.clip(extreme, low, high)
    assert model.predict(extreme)[0] == pytest.approx(model.predict(bounded)[0]), \
        "expected a value beyond the bounds to be predicted like the bound itself"


def test_preprocessing_is_learnt_on_the_training_rows_only(districts):
    X, y = districts
    model = housing.HousingModel(degree=2, penalty="ridge", alpha=1.0).fit(X[:1500], y[:1500])
    other = housing.HousingModel(degree=2, penalty="ridge", alpha=1.0).fit(X[:1500], y[:1500])
    changed = X[1500:].copy()
    changed[:, 0] *= 10                                    # districts never seen by fit
    assert np.allclose(model.predict(X[1500:1510]), other.predict(X[1500:1510])), "fit must be deterministic"
    one_by_one = np.array([model.predict(X[i:i + 1])[0] for i in range(1500, 1510)])
    assert np.allclose(model.predict(X[1500:1510]), one_by_one), (
        "expected the prediction of a district not to depend on the other districts predicted with it: transform "
        "must reuse the statistics learnt by fit, never recompute them")
    bounded = np.clip(X[:1500], model.low_, model.high_)
    assert np.allclose(model.mean_, bounded.mean(axis=0)), "expected mean_ = the mean of the bounded training rows"


def test_zones_come_from_a_kmeans_on_the_training_coordinates(districts):
    X, y = districts
    model = housing.HousingModel(degree=1, n_zones=5, random_state=0).fit(X[:2000], y[:2000])
    zones = model.zones(X)
    assert zones.shape == (3000,) and set(np.unique(zones)) <= set(range(5)), \
        "expected one zone between 0 and 4 per district"
    geo = X[:2000, [6, 7]]
    standardized = (X[:, [6, 7]] - geo.mean(axis=0)) / geo.std(axis=0)
    distances = ((standardized[:, None, :] - np.asarray(model.kmeans_.cluster_centers_)[None]) ** 2).sum(axis=2)
    assert np.array_equal(zones, distances.argmin(axis=1)), (
        "expected the zone of each district to be its nearest k-means centre, coordinates standardized with the "
        "means and standard deviations of the training rows")
    assert model.transform(X[:10]).shape == (10, 8 + 5), "expected 8 features + one column per zone"


def test_model_without_zones_refuses_zones(districts):
    X, y = districts
    model = housing.HousingModel().fit(X[:500], y[:500])
    with pytest.raises(ValueError):
        model.zones(X[:5])


@pytest.mark.parametrize("params, why", [({"degree": 0}, "degree 0"), ({"penalty": "elastic"}, "unknown penalty"),
                                         ({"penalty": "ridge", "alpha": 0.0}, "Ridge without strength"),
                                         ({"n_zones": -1}, "negative zones"), ({"clip": (99.0, 1.0)}, "reversed bounds")],
                         ids=["degree-0", "penalty", "alpha-0", "zones", "clip"])
def test_invalid_hyperparameters_are_refused_by_fit(districts, params, why):
    X, y = districts
    with pytest.raises(ValueError):
        housing.HousingModel(**params).fit(X[:200], y[:200])


# --- cross-validation helpers -------------------------------------------------------------------------------------

def test_cross_validate_fits_a_fresh_clone_per_fold(districts, folds):
    X, y = districts
    model = housing.HousingModel(degree=2, penalty="ridge", alpha=1.0)
    scores = housing.cross_validate(model, X, y, copied(folds))
    assert not any(name.endswith("_") for name in vars(model)), "the model passed must stay unfitted"
    expected = [housing.rmse(y[val], housing.HousingModel(degree=2, penalty="ridge", alpha=1.0)
                             .fit(X[train], y[train]).predict(X[val])) for train, val in folds]
    assert np.allclose(scores["val_rmse"], expected), f"expected the validation RMSE of each fold, {np.round(expected, 4)}"
    assert len(scores["train_rmse"]) == len(folds), "expected one training RMSE per fold"


def test_out_of_fold_predictions_come_from_the_model_that_did_not_see_them(districts, folds):
    X, y = districts
    model = housing.HousingModel()
    predictions = housing.out_of_fold_predictions(model, X, y, copied(folds))
    train, val = folds[1]
    expected = housing.HousingModel().fit(X[train], y[train]).predict(X[val])
    assert np.allclose(predictions[val], expected), "expected the predictions of the model of the fold they validate"
    with pytest.raises(ValueError):
        housing.out_of_fold_predictions(model, X, y, copied(folds)[:2])


def test_validation_curve_has_one_row_per_value(districts, folds):
    X, y = districts
    train, val = housing.validation_curve(housing.HousingModel(degree=2, penalty="ridge"), "alpha", [0.1, 10.0, 1e4],
                                          X, y, copied(folds))
    assert train.shape == val.shape == (3, len(folds)), f"expected two arrays (3, {len(folds)}), got {train.shape}"
    expected = housing.cross_validate(housing.HousingModel(degree=2, penalty="ridge", alpha=1e4), X, y, copied(folds))
    assert np.allclose(val[2], expected["val_rmse"]), "expected row 2 = cross_validate with alpha = 1e4"
    assert train[2].mean() > train[0].mean(), "a much stronger penalty must raise the training error"


def test_learning_curve_uses_the_first_rows_of_the_shuffled_training_part(districts, folds):
    X, y = districts
    sizes = [100, 400]
    train, val = housing.learning_curve(housing.HousingModel(), X, y, sizes, copied(folds), seed=3)
    assert train.shape == val.shape == (2, len(folds)), f"expected arrays of shape (2, {len(folds)})"
    train_idx, val_idx = folds[0]
    rows = np.asarray(train_idx)[np.random.default_rng(3).permutation(len(train_idx))][:100]
    expected = housing.rmse(y[val_idx], housing.HousingModel().fit(X[rows], y[rows]).predict(X[val_idx]))
    assert val[0, 0] == pytest.approx(expected), (
        "expected the model of size 100 in fold 0 to learn from the first 100 training rows shuffled by "
        "np.random.default_rng(seed)")
    with pytest.raises(ValueError):
        housing.learning_curve(housing.HousingModel(), X, y, [len(y)], copied(folds))


def test_helpers_do_not_modify_the_folds_they_receive(districts, folds):
    X, y = districts
    calls = {"cross_validate": lambda f: housing.cross_validate(housing.HousingModel(), X, y, f),
             "out_of_fold_predictions": lambda f: housing.out_of_fold_predictions(housing.HousingModel(), X, y, f),
             "validation_curve": lambda f: housing.validation_curve(housing.HousingModel(penalty="ridge"), "alpha",
                                                                    [1.0], X, y, f),
             "learning_curve": lambda f: housing.learning_curve(housing.HousingModel(), X, y, [200], f, seed=3)}
    for name, call in calls.items():
        given = copied(folds)
        call(given)
        changed = [k for k, (pair, fold) in enumerate(zip(given, folds))
                   if not all(np.array_equal(mine, original) for mine, original in zip(pair, fold))]
        assert changed == [], (f"expected {name} to leave the folds it receives unchanged (every model of the project "
                               f"shares them); folds {changed} were modified")
