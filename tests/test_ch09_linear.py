"""Tests of mylearn.linear (chapter 9): oracle tests and properties.

    pytest tests/test_ch09_linear.py              # your code (mon_travail/mylearn/linear.py)
    pytest tests/test_ch09_linear.py --impl=ref   # the reference

Oracles: scikit-learn (``PolynomialFeatures``, ``mean_squared_error``,
``mean_absolute_error``, ``r2_score``, ``LinearRegression``, ``Ridge``, ``Lasso``), NumPy
(the formulas of the bias-variance decomposition, a fine grid search for the
soft-thresholding operator) and SciPy (``scipy.stats.norm`` for the prior and the
likelihood of the lines, ``scipy.special.logsumexp``), plus the closed form of the
Bayesian linear regression with a Gaussian prior (Bishop 2006, eq. 3.53-3.54) for the
mean and the covariance of the posterior of the lines.
Every test name starts with the name of what it tests, so that each exercise runs its
own group: ``-k "test_mean_squared_error_ or test_mean_absolute_error_ or test_r2_score_"``
(9.14), ``"test_polynomial_features_"`` (9.15), ``"test_linear_regression_"`` (9.16),
``"test_ridge_"`` (9.17), ``"test_soft_threshold_ or test_lasso_"`` (9.23),
``"test_bias_variance_decomposition_"`` (9.24), ``"test_bayes_line_posterior_"`` (9.26).
The tests of one exercise never call the functions of another one (your models may call
your ``r2_score`` in ``score``, and your ``Lasso`` your ``soft_threshold``, as their
docstrings say).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import math
import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures


@pytest.fixture
def lin(mylearn_module):
    return mylearn_module("linear")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for the first line of a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    if arr.dtype.kind in "biuf":
        text = np.array2string(arr, precision=digits, separator=", ", threshold=12,
                               edgeitems=2 if arr.ndim > 1 else 3, max_line_width=10**6)
    else:
        text = repr(arr.tolist())
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = "", details: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data and details."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    if details:
        lines.append(details)
    raise AssertionError("\n".join(lines)) from None


def _first_difference(got, want, rtol, atol, largest=False) -> str:
    """' (first difference at index i: expected a, got b)' for two arrays of the same shape, '' otherwise.

    largest=True reports the value that is the furthest off instead of the first one (big tables: the first
    difference may sit in a corner where every value is negligible).
    """
    if got.ndim == 0 or got.shape != want.shape:
        return ""
    bad = np.flatnonzero(~np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True))
    if bad.size == 0:
        return ""
    i = int(bad[np.argmax(np.abs(got.ravel()[bad] - want.ravel()[bad]))]) if largest else int(bad[0])
    index = np.unravel_index(i, got.shape)
    where = int(index[0]) if got.ndim == 1 else tuple(int(k) for k in index)
    a, b = want.ravel()[i], got.ravel()[i]
    digits = 6 if f"{a:.6g}" != f"{b:.6g}" else 17
    kind = "largest" if largest else "first"
    return f" ({kind} difference at index {where}: expected {a:.{digits}g}, got {b:.{digits}g})"


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg="", data=""):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'.

    For a big array (more than 12 values), the first line names the value that is the furthest off and how many
    values differ; both arrays go to the next line.
    """
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected {_short(expected)}, got an object of type {type(result).__name__}", data)
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {_short(expected)}, "
                   f"got {_short(result)}", data)
    if np.isnan(got).any() and not np.isnan(want).any():
        _fail(msg, f"expected {_short(expected)}, got NaN values: {_short(result)}", data)
    try:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    except AssertionError as exc:
        have, need = _short(result), _short(expected)
        if have == need and want.size <= 12:
            have, need = _short(result, 17), _short(expected, 17)
        if want.size > 12:
            n_bad = int(np.sum(~np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True)))
            diff = _first_difference(got, want, rtol, atol, largest=True)
            head = (f"{diff.strip()[1:-1]}; " if diff else "") + f"{n_bad} of {want.size} values differ"
            _fail(msg, head, "\n".join(filter(None, [data, f"expected {need}", f"got {have}"])), str(exc))
        head = f"expected {need}, got {have}{_first_difference(got, want, rtol, atol)}"
        _fail(msg, head, data, str(exc))


def assert_python_float(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, float):
        raise AssertionError(f"expected {name} to be a float (a Python float or np.float64), got "
                             f"{type(value).__name__}" + (f" of shape {np.shape(value)}"
                                                          if isinstance(value, np.ndarray) else "")
                             + " (use float(...) on a 0-d array)")


def assert_raises_value_error(function, *args, why="", **kwargs):
    """The call must raise ValueError (or a subclass); `why` names the case in the failure message."""
    name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
    if name == "<lambda>":
        name = "the call"
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    except NotImplementedError:
        raise                  # shown as "⏳ pas encore implémenté" by conftest.py
    except Exception as exc:  # noqa: BLE001 - say which error was raised instead
        raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else "")
                             + f" (it raised {type(exc).__name__}: {exc})") from None
    raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else ""))


def _copied(value):
    return value.copy() if isinstance(value, np.ndarray) else value


def call(function, *args, expected="a result", hint="", hint_when=None, copy_inputs=True, **kwargs):
    """Call the learner's function; an unexpected exception becomes a failure that says what was expected.

    The function receives copies of the array arguments, so that an array changed in place cannot spoil an oracle
    computed afterwards: only the tests named ..._does_not_modify_... (copy_inputs=False) pass the originals.
    `hint` is added to the message of any exception, or only to those for which `hint_when(exc)` is true.
    """
    if copy_inputs:
        args = tuple(_copied(value) for value in args)
        kwargs = {key: _copied(value) for key, value in kwargs.items()}
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
        show_hint = hint and (hint_when is None or hint_when(exc))
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}"
                             + (f" ({hint})" if show_hint else "")) from None


def _division_problem(exc) -> bool:
    """True for the errors of a division by zero (0 / 0 or x / 0), as an exception or a RuntimeWarning raised as one."""
    text = str(exc).lower()
    return isinstance(exc, (ZeroDivisionError, FloatingPointError)) or "divide" in text or "invalid value" in text


def fit(model, X, y, **kwargs):
    """model.fit(X, y) without chaining: a fit that forgets `return self` is reported by its own test."""
    call(model.fit, X, y, expected="fit to succeed", **kwargs)
    return model


def learned(model, name):
    """A learned attribute (coef_, intercept_, n_iter_), with a clear message when fit did not create it."""
    if not hasattr(model, name):
        raise AssertionError(f"expected fit to create the attribute {name}, but the fitted "
                             f"{type(model).__name__} has no {name} (attributes: {sorted(vars(model))})")
    return getattr(model, name)


def regression_data(seed, n=60, p=4, noise=0.5, offset=3.0, scale=1.0):
    """Independent Gaussian features (well conditioned), a linear target with an intercept, Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * scale
    w = rng.normal(size=p) * 2.0
    y = X @ w + offset + noise * rng.normal(size=n)
    return X, y


def sklearn_fit(cls, X, y, **params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(**params).fit(X, y)


# ================================================================== metrics (9.14)
METRIC_CASES = [(0, 5), (1, 30), (2, 200), (3, 2)]


@pytest.mark.parametrize("seed, n", METRIC_CASES, ids=[f"n{n}" for _, n in METRIC_CASES])
def test_mean_squared_error_matches_sklearn(lin, seed, n):
    rng = np.random.default_rng(seed)
    y_true, y_pred = rng.normal(size=n) * 3, rng.normal(size=n) * 3
    result = call(lin.mean_squared_error, y_true, y_pred, expected="a float")
    assert_close(result, metrics.mean_squared_error(y_true, y_pred), msg="MSE = mean of the squared residuals")


@pytest.mark.parametrize("seed, n", METRIC_CASES, ids=[f"n{n}" for _, n in METRIC_CASES])
def test_mean_absolute_error_matches_sklearn(lin, seed, n):
    rng = np.random.default_rng(seed + 10)
    y_true, y_pred = rng.normal(size=n) * 3, rng.normal(size=n) * 3
    result = call(lin.mean_absolute_error, y_true, y_pred, expected="a float")
    assert_close(result, metrics.mean_absolute_error(y_true, y_pred),
                 msg="MAE = mean of the ABSOLUTE residuals |y - ŷ|")


@pytest.mark.parametrize("seed, n", METRIC_CASES, ids=[f"n{n}" for _, n in METRIC_CASES])
def test_r2_score_matches_sklearn(lin, seed, n):
    rng = np.random.default_rng(seed + 20)
    y_true = rng.normal(size=n) * 3
    y_pred = y_true + rng.normal(size=n) * 1.5
    result = call(lin.r2_score, y_true, y_pred, expected="a float")
    assert_close(result, metrics.r2_score(y_true, y_pred),
                 msg="R² = 1 - SS_res / SS_tot, SS_tot computed around the mean of y_true")


def test_mean_squared_error_docstring_example(lin):
    assert_close(call(lin.mean_squared_error, [3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), 0.375,
                 msg="mean_squared_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) (lists must be accepted)")


def test_mean_absolute_error_docstring_example(lin):
    assert_close(call(lin.mean_absolute_error, [3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), 0.5,
                 msg="mean_absolute_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) (lists must be accepted)")


def test_r2_score_docstring_example(lin):
    assert_close(call(lin.r2_score, [3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), 0.9486081370449679, rtol=1e-12,
                 msg="r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) (lists must be accepted)")


def test_mean_squared_error_and_mean_absolute_error_return_floats(lin):
    for name in ("mean_squared_error", "mean_absolute_error"):
        value = call(getattr(lin, name), np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.0, 3.0]))
        assert_python_float(value, f"{name}(...)")


def test_r2_score_returns_a_float(lin):
    assert_python_float(call(lin.r2_score, np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.0, 3.0])), "r2_score(...)")


def test_mean_squared_error_is_a_mean_not_a_sum(lin):
    y_true, y_pred = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 3.0, 6.0])
    once = call(lin.mean_squared_error, y_true, y_pred)
    twice = call(lin.mean_squared_error, np.tile(y_true, 2), np.tile(y_pred, 2))
    assert_close(once, 1.25, msg="residuals 1, 0, 0, 2: (1 + 0 + 0 + 4) / 4")
    assert_close(twice, 1.25, msg="the same residuals twice: a mean does not change (divide by n, do not sum)")


def test_mean_absolute_error_takes_absolute_values(lin):
    assert_close(call(lin.mean_absolute_error, [0.0, 0.0], [2.0, -2.0]), 2.0,
                 msg="residuals -2 and +2 must not cancel out: mean of |y - ŷ|")


def test_mean_squared_error_and_mean_absolute_error_of_a_perfect_prediction_are_zero(lin):
    y = np.array([0.5, -1.25, 3.0])
    assert_close(call(lin.mean_squared_error, y, y.copy()), 0.0, atol=0, msg="MSE of a perfect prediction")
    assert_close(call(lin.mean_absolute_error, y, y.copy()), 0.0, atol=0, msg="MAE of a perfect prediction")


def test_r2_score_of_the_mean_is_zero_and_of_a_perfect_prediction_is_one(lin):
    y = np.array([1.0, 2.0, 4.0, 9.0])
    assert_close(call(lin.r2_score, y, np.full(4, 4.0)), 0.0, atol=1e-15,
                 msg="always predicting the mean of y_true gives R² = 0")
    assert_close(call(lin.r2_score, y, y.copy()), 1.0, atol=0, msg="a perfect prediction gives R² = 1")


def test_r2_score_can_be_negative(lin):
    y_true, y_pred = np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    assert_close(call(lin.r2_score, y_true, y_pred), -3.0,
                 msg="a prediction worse than the mean gives R² < 0 (SS_res = 8, SS_tot = 2: never clip at 0)")


def test_r2_score_uses_the_mean_of_y_true_not_of_y_pred(lin):
    y_true, y_pred = np.array([0.0, 1.0, 2.0, 7.0]), np.array([1.0, 1.0, 3.0, 5.0])
    assert_close(call(lin.r2_score, y_true, y_pred), metrics.r2_score(y_true, y_pred),
                 msg="R² is not symmetric: SS_tot is the spread of y_true around ITS mean (check the order of "
                     "the arguments and which mean you subtract)")


def test_r2_score_constant_target_follows_sklearn(lin):
    y = np.array([2.0, 2.0, 2.0])
    assert_close(call(lin.r2_score, y, y.copy()), 1.0, atol=0,
                 msg="constant y_true and a perfect prediction (SS_res = SS_tot = 0): R² = 1.0, as scikit-learn")
    assert_close(call(lin.r2_score, y, np.array([2.0, 2.0, 3.0])), 0.0, atol=0,
                 msg="constant y_true and an imperfect prediction (only SS_tot = 0): R² = 0.0, as scikit-learn "
                     "(no division by zero)")


def test_mean_squared_error_does_not_modify_its_inputs(lin):
    y_true, y_pred = np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 2.0])
    keep = (y_true.copy(), y_pred.copy())
    for name in ("mean_squared_error", "mean_absolute_error", "r2_score"):
        call(getattr(lin, name), y_true, y_pred, copy_inputs=False)
        assert np.array_equal(y_true, keep[0]) and np.array_equal(y_pred, keep[1]), \
            f"expected {name} to leave y_true and y_pred unchanged, but it modified them"


BAD_METRIC_INPUTS = [
    ([1.0, 2.0, 3.0], [1.0, 2.0], "lengths 3 and 2"),
    ([1.0, 2.0, 3.0], [2.0], "lengths 3 and 1 (NumPy would broadcast the single value: compare the lengths "
                             "yourself)"),
    ([], [], "two empty arrays"),
]
BAD_METRIC_IDS = ["lengths-differ", "length-one-broadcast", "empty"]


@pytest.mark.parametrize("y_true, y_pred, why", BAD_METRIC_INPUTS, ids=BAD_METRIC_IDS)
def test_mean_squared_error_rejects_invalid_inputs(lin, y_true, y_pred, why):
    assert_raises_value_error(lin.mean_squared_error, y_true, y_pred, why=why)


@pytest.mark.parametrize("y_true, y_pred, why", BAD_METRIC_INPUTS, ids=BAD_METRIC_IDS)
def test_mean_absolute_error_rejects_invalid_inputs(lin, y_true, y_pred, why):
    assert_raises_value_error(lin.mean_absolute_error, y_true, y_pred, why=why)


@pytest.mark.parametrize("y_true, y_pred, why",
                         BAD_METRIC_INPUTS + [([1.5], [1.0], "a single sample (R² needs at least 2)")],
                         ids=BAD_METRIC_IDS + ["one-sample"])
def test_r2_score_rejects_invalid_inputs(lin, y_true, y_pred, why):
    assert_raises_value_error(lin.r2_score, y_true, y_pred, why=why)


# ================================================================== polynomial_features (9.15)
POLY_CASES = [(1, 1, False), (1, 3, True), (2, 2, False), (2, 2, True), (3, 2, False), (3, 3, True),
              (4, 3, False), (2, 5, False), (5, 1, True)]


@pytest.mark.parametrize("p, degree, include_bias", POLY_CASES,
                         ids=[f"{p}features-degree{d}" + ("-bias" if b else "") for p, d, b in POLY_CASES])
def test_polynomial_features_matches_sklearn(lin, p, degree, include_bias):
    X = np.random.default_rng(p * 10 + degree).normal(size=(6, p)) * 2
    want = PolynomialFeatures(degree, include_bias=include_bias).fit_transform(X)
    got = call(lin.polynomial_features, X, degree=degree, include_bias=include_bias, expected="a 2-D array")
    names = PolynomialFeatures(degree, include_bias=include_bias).fit(X).get_feature_names_out()
    order = ", ".join(names[:7]) + (", ..." if len(names) > 7 else "")
    assert_close(got, want, rtol=1e-12, atol=1e-12, msg=f"{p} feature(s), degree {degree}, include_bias="
                                                        f"{include_bias} (columns in scikit-learn's order: {order})")


def test_polynomial_features_docstring_examples(lin):
    got = call(lin.polynomial_features, np.array([[2.0, 3.0], [1.0, -1.0]]), degree=2)
    assert_close(got, [[2, 3, 4, 6, 9], [1, -1, 1, -1, 1]], msg="first docstring example (columns a, b, a², ab, b²)")
    got = call(lin.polynomial_features, np.array([2.0, 3.0]), degree=3, include_bias=True)
    assert_close(got, [[1, 2, 4, 8], [1, 3, 9, 27]],
                 msg="second docstring example (a 1-D array is ONE feature: 1, x, x², x³)")


def test_polynomial_features_keeps_squares_and_every_interaction_once(lin):
    X = np.array([[2.0, 3.0, 5.0]])
    got = call(lin.polynomial_features, X, degree=2)
    assert_close(got, [[2, 3, 5, 4, 6, 10, 9, 15, 25]],
                 msg="a, b, c, then a², ab, ac, b², bc, c² (combinations WITH replacement: squares included, "
                     "ba never repeats ab)")


def test_polynomial_features_1d_input_is_a_single_feature(lin):
    x = np.array([1.0, 2.0, -3.0, 0.5])
    got = call(lin.polynomial_features, x, degree=4, expected="an array of shape (4, 4)")
    assert_close(got, np.column_stack([x, x ** 2, x ** 3, x ** 4]),
                 msg="a 1-D array of 4 values is 4 samples of one feature: columns x, x², x³, x⁴")


def test_polynomial_features_returns_floats_and_keeps_X(lin):
    X = np.array([[1, 2], [3, 4]])
    X0 = X.copy()
    got = call(lin.polynomial_features, X, degree=2, copy_inputs=False)
    assert isinstance(got, np.ndarray) and got.dtype.kind == "f", \
        f"expected a float NumPy array, got {type(got).__name__} of dtype {getattr(got, 'dtype', None)}"
    assert np.array_equal(X, X0), "polynomial_features must not modify X"


def test_polynomial_features_bias_column_comes_first(lin):
    X = np.array([[2.0, 3.0], [4.0, 5.0], [0.0, 1.0]])
    got = np.asarray(call(lin.polynomial_features, X, degree=1, include_bias=True))
    assert_close(got, [[1, 2, 3], [1, 4, 5], [1, 0, 1]],
                 msg="include_bias=True: a column of ones FIRST, then the degree-1 columns")


def test_polynomial_features_counts_the_columns(lin):
    for p, degree in [(2, 3), (3, 4), (6, 2)]:
        got = np.asarray(call(lin.polynomial_features, np.ones((2, p)), degree=degree))
        expected = math.comb(p + degree, degree) - 1
        if got.shape != (2, expected):
            raise AssertionError(f"expected shape (2, {expected}) for {p} features up to degree {degree} "
                                 f"(C({p} + {degree}, {degree}) - 1 monomials), got shape {got.shape}")


@pytest.mark.parametrize("X, degree, why", [
    (np.ones((3, 2)), 0, "degree = 0"),
    (np.ones((3, 2)), -1, "degree = -1"),
    (np.ones((2, 2, 2)), 2, "X with 3 dimensions"),
], ids=["degree-0", "degree-negative", "X-3d"])
def test_polynomial_features_rejects_invalid_inputs(lin, X, degree, why):
    assert_raises_value_error(lin.polynomial_features, X, degree=degree, why=why)


# ================================================================== LinearRegression (9.16)
OLS_CASES = [(0, 40, 1), (1, 50, 3), (2, 80, 8), (3, 12, 2)]


@pytest.mark.parametrize("seed, n, p", OLS_CASES, ids=[f"n{n}-p{p}" for _, n, p in OLS_CASES])
def test_linear_regression_matches_sklearn(lin, seed, n, p):
    X, y = regression_data(seed, n=n, p=p)
    model = fit(lin.LinearRegression(), X, y)
    oracle = sklearn_fit(linear_model.LinearRegression, X, y)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=1e-8, atol=1e-10,
                 msg="coef_ (fit_intercept=True: centre X and y, solve on the centred data, then intercept_ = "
                     "mean(y) - mean(X) . coef_)")
    assert_close(learned(model, "intercept_"), oracle.intercept_, rtol=1e-8, atol=1e-10,
                 msg="intercept_ = mean(y) - mean(X) . coef_")


@pytest.mark.parametrize("seed, n, p", OLS_CASES[:3], ids=[f"n{n}-p{p}" for _, n, p in OLS_CASES[:3]])
def test_linear_regression_without_intercept_matches_sklearn(lin, seed, n, p):
    X, y = regression_data(seed + 5, n=n, p=p, offset=4.0)
    model = fit(lin.LinearRegression(fit_intercept=False), X, y)
    oracle = sklearn_fit(linear_model.LinearRegression, X, y, fit_intercept=False)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=1e-8, atol=1e-10,
                 msg="fit_intercept=False: no centring, the line goes through the origin")
    assert_close(learned(model, "intercept_"), 0.0, atol=0, msg="fit_intercept=False: intercept_ = 0.0")


def test_linear_regression_docstring_example(lin):
    model = fit(lin.LinearRegression(), np.array([[1.0], [2.0], [3.0]]), np.array([3.0, 5.0, 7.0]))
    assert_close(learned(model, "coef_"), [2.0], msg="coef_ of the docstring example (y = 2x + 1)")
    assert_close(learned(model, "intercept_"), 1.0, msg="intercept_ of the docstring example")
    assert_close(call(model.predict, np.array([[10.0]])), [21.0], msg="predict([[10.0]]) of the docstring example")


def test_linear_regression_recovers_an_exact_line(lin):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 3))
    y = X @ np.array([1.5, -2.0, 0.25]) - 7.0
    model = fit(lin.LinearRegression(), X, y)
    assert_close(learned(model, "coef_"), [1.5, -2.0, 0.25], rtol=1e-9, atol=1e-10,
                 msg="noise-free data y = 1.5 x0 - 2 x1 + 0.25 x2 - 7")
    assert_close(learned(model, "intercept_"), -7.0, rtol=1e-9, atol=1e-10, msg="intercept of the noise-free data")


def test_linear_regression_collinear_features_give_the_minimum_norm_solution(lin):
    rng = np.random.default_rng(5)
    x = rng.normal(size=40)
    X = np.column_stack([x, x, rng.normal(size=40)])          # the first two columns are identical
    y = 3.0 * x - X[:, 2] + 0.1 * rng.normal(size=40)
    model = fit(lin.LinearRegression(), X, y, hint="identical columns make XᵀX singular: np.linalg.lstsq still "
                                                    "works, np.linalg.solve on the normal equations does not",
                hint_when=lambda exc: isinstance(exc, np.linalg.LinAlgError))
    oracle = sklearn_fit(linear_model.LinearRegression, X, y)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=1e-6, atol=1e-8,
                 msg="two identical columns: infinitely many solutions, the expected one is the minimum-norm one, "
                     "found by np.linalg.lstsq on the CENTRED data (the weight is shared equally; the normal "
                     "equations fail here)")


def test_linear_regression_more_features_than_samples_gives_the_minimum_norm_solution(lin):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(8, 20))
    y = rng.normal(size=8)
    model = fit(lin.LinearRegression(), X, y, hint="with 20 features and 8 samples XᵀX is singular: use "
                                                    "np.linalg.lstsq", hint_when=lambda exc: isinstance(exc, np.linalg.LinAlgError))
    oracle = sklearn_fit(linear_model.LinearRegression, X, y)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=1e-6, atol=1e-8,
                 msg="20 features, 8 samples: the minimum-norm coef_, computed by np.linalg.lstsq on the CENTRED "
                     "data (the intercept is not part of the norm: lstsq on [1, X] gives another solution)")
    assert_close(call(model.predict, X), y, rtol=1e-8, atol=1e-8,
                 msg="with more features than samples, the training points are fitted exactly (interpolation)")


def test_linear_regression_predict_and_score_match_sklearn(lin):
    X, y = regression_data(7, n=70, p=3, noise=2.0)
    model = fit(lin.LinearRegression(), X[:50], y[:50])
    oracle = sklearn_fit(linear_model.LinearRegression, X[:50], y[:50])
    pred = call(model.predict, X[50:], expected="an array of 20 predictions")
    assert_close(pred, oracle.predict(X[50:]), rtol=1e-8, atol=1e-10,
                 msg="predict(X) = X @ coef_ + intercept_, shape (n_samples,)")
    score = call(model.score, X[50:], y[50:], expected="a float")
    assert_python_float(score, "score(X, y)")
    assert_close(score, oracle.score(X[50:], y[50:]), rtol=1e-8, msg="score(X, y) = R² of predict(X) against y")


def test_linear_regression_attributes_have_the_documented_types(lin):
    X, y = regression_data(8, n=20, p=3)
    model = fit(lin.LinearRegression(), X, y)
    coef = learned(model, "coef_")
    assert isinstance(coef, np.ndarray) and coef.shape == (3,), \
        f"expected coef_ to be a NumPy array of shape (3,) for 3 features, got {type(coef).__name__} of shape " \
        f"{np.shape(coef)}"
    assert_python_float(learned(model, "intercept_"), "intercept_")


def test_linear_regression_fit_returns_self_and_does_not_modify_X_and_y(lin):
    X, y = regression_data(9, n=20, p=2)
    X0, y0 = X.copy(), y.copy()
    model = lin.LinearRegression()
    assert call(model.fit, X, y, copy_inputs=False) is model, "fit must return the fitted estimator itself (return self)"
    assert np.array_equal(X, X0) and np.array_equal(y, y0), \
        "expected fit to leave X and y unchanged (centre copies: X - X.mean(axis=0) creates a new array)"


def _public(model):
    """The attributes that clone (ch. 8) takes for hyperparameters: names that neither start nor end with _."""
    return {key: value for key, value in vars(model).items() if not key.startswith("_") and not key.endswith("_")}


def check_init_then_fit(model, want, X, y):
    """__init__ only stores the hyperparameters; fit only adds learned (name_) or private (_name) attributes."""
    name = type(model).__name__
    params = _public(model)
    assert params == want, (f"expected the hyperparameters of {name}(...) to be exactly {want} right after __init__ "
                            f"(it only stores its arguments; learned attributes, ending with _, are created by fit), "
                            f"got {params}")
    early = sorted(key for key in vars(model) if key.endswith("_") and not key.startswith("_"))
    assert not early, (f"expected no learned attribute right after __init__ (fit creates them), got {early} "
                       f"(an attribute like coef_ = None makes an unfitted model look fitted)")
    fit(model, X, y)
    added = sorted(set(_public(model)) - set(want))
    assert not added, (f"expected fit to add only learned attributes (names ending with _, like coef_) or private "
                       f"ones (starting with _), got {added}: clone (ch. 8) would take them for hyperparameters")
    changed = {key: vars(model)[key] for key in want if vars(model)[key] != want[key]}
    assert not changed, f"expected fit to leave the hyperparameters unchanged, got {changed} instead of {want}"


def test_linear_regression_init_only_stores_the_hyperparameter(lin):
    X, y = regression_data(12, n=15, p=2)
    check_init_then_fit(lin.LinearRegression(fit_intercept=False), {"fit_intercept": False}, X, y)


def test_linear_regression_second_fit_forgets_the_first(lin):
    X1, y1 = regression_data(10, n=30, p=2)
    X2, y2 = regression_data(11, n=25, p=2, offset=-5.0)
    model = fit(lin.LinearRegression(), X1, y1)
    fit(model, X2, y2)
    oracle = sklearn_fit(linear_model.LinearRegression, X2, y2)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=1e-8, atol=1e-10,
                 msg="a second fit must start from scratch (coef_ of the second data only)")


@pytest.mark.parametrize("X, y, why", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "X is 1-D (use X.reshape(-1, 1))"),
    (np.ones((4, 2)), np.ones(3), "4 rows in X but 3 values in y"),
], ids=["X-1d", "lengths-differ"])
def test_linear_regression_rejects_invalid_inputs(lin, X, y, why):
    assert_raises_value_error(lin.LinearRegression().fit, X, y, why=why)


# ================================================================== Ridge (9.17)
RIDGE_CASES = [(0, 50, 3, 0.01), (1, 50, 3, 1.0), (2, 50, 3, 100.0), (3, 30, 8, 10.0), (4, 10, 25, 1.0),
               (5, 200, 5, 1000.0)]


@pytest.mark.parametrize("seed, n, p, alpha", RIDGE_CASES,
                         ids=[f"n{n}-p{p}-alpha{a:g}" for _, n, p, a in RIDGE_CASES])
def test_ridge_matches_sklearn(lin, seed, n, p, alpha):
    X, y = regression_data(seed + 30, n=n, p=p)
    model = fit(lin.Ridge(alpha=alpha), X, y)
    oracle = sklearn_fit(linear_model.Ridge, X, y, alpha=alpha)
    convention = "objective ||y - Xw - b||² + alpha ||w||²: a SUM of squares, no 1/n, intercept not penalised"
    hint = "on the centred data, solve (XcᵀXc + alpha I) w = Xcᵀyc, then b = mean(y) - mean(X) . w"
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=1e-8, atol=1e-8,
                 msg=f"coef_ with alpha = {alpha:g} ({convention})", data=hint)
    assert_close(learned(model, "intercept_"), oracle.intercept_, rtol=1e-8, atol=1e-8,
                 msg=f"intercept_ with alpha = {alpha:g} (b = mean(y) - mean(X) . coef_)", data=hint)


@pytest.mark.parametrize("alpha", [0.1, 5.0])
def test_ridge_without_intercept_matches_sklearn(lin, alpha):
    X, y = regression_data(40, n=40, p=4, offset=2.0)
    model = fit(lin.Ridge(alpha=alpha, fit_intercept=False), X, y)
    oracle = sklearn_fit(linear_model.Ridge, X, y, alpha=alpha, fit_intercept=False)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=1e-8, atol=1e-8,
                 msg=f"fit_intercept=False, alpha = {alpha:g}: no centring")
    assert_close(learned(model, "intercept_"), 0.0, atol=0, msg="fit_intercept=False: intercept_ = 0.0")


def test_ridge_docstring_example(lin):
    model = fit(lin.Ridge(alpha=1.0), np.array([[1.0], [2.0], [3.0]]), np.array([3.0, 5.0, 7.0]))
    assert_close(learned(model, "coef_"), [4.0 / 3.0], rtol=1e-12, msg="coef_ of the docstring example (4/3)")
    assert_close(learned(model, "intercept_"), 7.0 / 3.0, rtol=1e-12, msg="intercept_ of the docstring example (7/3)")


def test_ridge_does_not_penalise_the_intercept(lin):
    X, y = regression_data(41, n=30, p=2, offset=0.0)
    base = fit(lin.Ridge(alpha=50.0), X, y)
    shifted = fit(lin.Ridge(alpha=50.0), X, y + 1000.0)
    assert_close(learned(shifted, "coef_"), learned(base, "coef_"), rtol=1e-8, atol=1e-8,
                 msg="adding 1000 to every target must not change coef_ (the intercept absorbs it)")
    assert_close(learned(shifted, "intercept_") - learned(base, "intercept_"), 1000.0, rtol=1e-10,
                 msg="adding 1000 to every target must add exactly 1000 to intercept_ (the intercept is not "
                     "penalised: centre the data instead of adding a column of ones to X)")


def test_ridge_with_alpha_zero_is_ordinary_least_squares(lin):
    X, y = regression_data(42, n=40, p=3)
    model = fit(lin.Ridge(alpha=0.0), X, y)
    oracle = sklearn_fit(linear_model.LinearRegression, X, y)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=1e-8, atol=1e-10,
                 msg="alpha = 0: the least squares solution (scikit-learn's LinearRegression)")


def test_ridge_coefficients_shrink_when_alpha_grows(lin):
    X, y = regression_data(43, n=40, p=5)
    norms = []
    for alpha in [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        norms.append(float(np.linalg.norm(learned(fit(lin.Ridge(alpha=alpha), X, y), "coef_"))))
    if not all(a > b for a, b in zip(norms, norms[1:])):
        raise AssertionError(f"expected ||coef_|| to decrease strictly as alpha grows (0, 0.1, 1, 10, 100, 1000), "
                             f"got {[round(v, 4) for v in norms]}")


def test_ridge_huge_alpha_predicts_the_mean(lin):
    X, y = regression_data(44, n=30, p=3, offset=6.0)
    model = fit(lin.Ridge(alpha=1e12), X, y)
    assert_close(learned(model, "coef_"), np.zeros(3), atol=1e-9, msg="alpha = 1e12: every weight is crushed to 0")
    assert_close(learned(model, "intercept_"), y.mean(), rtol=1e-9,
                 msg="alpha = 1e12: the model predicts the mean of y (the unpenalised intercept)")


def test_ridge_predict_and_score_match_sklearn(lin):
    X, y = regression_data(45, n=60, p=4, noise=2.0)
    model = fit(lin.Ridge(alpha=3.0), X[:40], y[:40])
    oracle = sklearn_fit(linear_model.Ridge, X[:40], y[:40], alpha=3.0)
    assert_close(call(model.predict, X[40:]), oracle.predict(X[40:]), rtol=1e-8, atol=1e-8,
                 msg="predict(X) = X @ coef_ + intercept_")
    score = call(model.score, X[40:], y[40:], expected="a float")
    assert_python_float(score, "score(X, y)")
    assert_close(score, oracle.score(X[40:], y[40:]), rtol=1e-8, msg="score(X, y) = R² of predict(X) against y")


def test_ridge_fit_returns_self_and_does_not_modify_X_and_y(lin):
    X, y = regression_data(46, n=20, p=2)
    X0, y0 = X.copy(), y.copy()
    model = lin.Ridge(alpha=2.0)
    assert call(model.fit, X, y, copy_inputs=False) is model, "fit must return the fitted estimator itself (return self)"
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "expected fit to leave X and y unchanged"


def test_ridge_init_only_stores_the_hyperparameters(lin):
    X, y = regression_data(48, n=15, p=2)
    check_init_then_fit(lin.Ridge(alpha=0.5, fit_intercept=False), {"alpha": 0.5, "fit_intercept": False}, X, y)


def test_ridge_checks_alpha_in_fit_not_in_init(lin):
    X, y = regression_data(47, n=10, p=2)
    try:
        model = lin.Ridge(alpha=-1.0)
    except ValueError:
        raise AssertionError("expected Ridge(alpha=-1.0) to be created without error and fit to raise ValueError "
                             "(__init__ only stores; the checks happen in fit, like scikit-learn)") from None
    assert_raises_value_error(model.fit, X, y, why="alpha = -1")


@pytest.mark.parametrize("X, y, why", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "X is 1-D (use X.reshape(-1, 1))"),
    (np.ones((4, 2)), np.ones(3), "4 rows in X but 3 values in y"),
], ids=["X-1d", "lengths-differ"])
def test_ridge_rejects_invalid_inputs(lin, X, y, why):
    assert_raises_value_error(lin.Ridge(alpha=1.0).fit, X, y, why=why)


# ================================================================== soft_threshold (9.23)
def test_soft_threshold_docstring_example(lin):
    result = call(lin.soft_threshold, np.array([-3.0, 0.5, 2.0]), 1.0, expected="an array")
    assert isinstance(result, np.ndarray), f"expected a NumPy array, got {type(result).__name__}"
    assert_close(result, [-2.0, 0.0, 1.0], msg="soft_threshold([-3, 0.5, 2], 1)")


def test_soft_threshold_is_zero_on_the_whole_interval(lin):
    z = np.array([-0.75, -0.5, -0.125, 0.0, 0.25, 0.75])     # -0.75 and 0.75 are exactly ±gamma
    got = call(lin.soft_threshold, z, 0.75)
    assert_close(got, np.zeros(6), atol=0, msg="every value of [-0.75, 0.75] (bounds included) becomes exactly 0",
                 data="input: [-0.75, -0.5, -0.125, 0, 0.25, 0.75], gamma = 0.75")


def test_soft_threshold_shifts_the_other_values_towards_zero(lin):
    z = np.array([-5.0, -1.0, 1.0, 2.5, 100.0])
    assert_close(call(lin.soft_threshold, z, 0.5), [-4.5, -0.5, 0.5, 2.0, 99.5],
                 msg="outside [-gamma, gamma], z moves TOWARDS 0 by gamma (slope 1): z - gamma if z > gamma, "
                     "z + gamma if z < -gamma")


def test_soft_threshold_is_an_odd_function(lin):
    z = np.random.default_rng(0).normal(size=50) * 3
    plus = np.asarray(call(lin.soft_threshold, z, 1.25), dtype=float)
    minus = np.asarray(call(lin.soft_threshold, -z, 1.25), dtype=float)
    assert_close(minus, -plus, atol=0, msg="soft_threshold(-z) = -soft_threshold(z)")


def test_soft_threshold_matches_a_grid_search_of_the_1d_lasso(lin):
    gamma = 0.8
    grid = np.linspace(-6.0, 6.0, 120_001)                     # step 1e-4
    zs = np.array([-4.2, -1.3, -0.8, -0.3, 0.0, 0.6, 0.95, 2.0, 3.7])
    want = np.array([grid[np.argmin(0.5 * (grid - z) ** 2 + gamma * np.abs(grid))] for z in zs])
    assert_close(call(lin.soft_threshold, zs, gamma), want, atol=2e-4,
                 msg="soft_threshold(z, gamma) must minimise 0.5 (w - z)² + gamma |w| (here found by a grid search)")


def test_soft_threshold_keeps_the_shape(lin):
    z = np.arange(-6.0, 6.0).reshape(3, 4)
    got = call(lin.soft_threshold, z, 2.0, expected="an array of shape (3, 4)")
    assert_close(got, np.sign(z) * np.maximum(np.abs(z) - 2.0, 0), msg="a 2-D input keeps its shape (3, 4)")
    scalar = call(lin.soft_threshold, -3.5, 1.0, expected="a 0-d result")
    if np.shape(scalar) != ():
        raise AssertionError(f"expected a 0-d result (shape ()) for a scalar input, got shape {np.shape(scalar)}")
    assert_close(scalar, -2.5, msg="soft_threshold(-3.5, 1.0)")


def test_soft_threshold_with_gamma_zero_is_the_identity(lin):
    z = np.array([-2.0, -0.1, 0.0, 0.3, 7.0])
    assert_close(call(lin.soft_threshold, z, 0.0), z, atol=0, msg="gamma = 0 leaves every value unchanged")


def test_soft_threshold_does_not_modify_z(lin):
    z = np.array([-3.0, 0.5, 2.0])
    call(lin.soft_threshold, z, 1.0, copy_inputs=False)
    assert np.array_equal(z, [-3.0, 0.5, 2.0]), "soft_threshold must not modify its input array"


def test_soft_threshold_accepts_a_list(lin):
    assert_close(call(lin.soft_threshold, [-3.0, 0.5, 2.0], 1.0), [-2.0, 0.0, 1.0],
                 msg="a list input (convert with np.asarray(z, dtype=float))")


def test_soft_threshold_rejects_a_negative_gamma(lin):
    assert_raises_value_error(lin.soft_threshold, np.array([1.0, 2.0]), -0.5, why="gamma = -0.5")


# ================================================================== Lasso (9.23)
LASSO_HINT = ("objective (1 / 2n) ||y - Xw - b||² + alpha ||w||₁: rho_j = x_jᵀ r_j / n (r_j: the residual WITHOUT "
              "feature j), z_j = x_jᵀ x_j / n, w_j = soft_threshold(rho_j, alpha) / z_j, residual updated after "
              "each coordinate")
LASSO_SHORT = "objective (1/2n) ||y - Xw - b||² + alpha ||w||₁: w_j = soft_threshold(rho_j, alpha) / z_j"
LASSO_CASES = [(0, 60, 4, 0.05), (1, 60, 4, 0.5), (2, 80, 10, 0.2), (3, 40, 6, 1.5), (4, 100, 3, 0.01),
               (5, 30, 12, 0.3)]


def _sklearn_lasso(X, y, alpha, fit_intercept=True):
    return sklearn_fit(linear_model.Lasso, X, y, alpha=alpha, fit_intercept=fit_intercept, tol=1e-12,
                       max_iter=100_000)


@pytest.mark.parametrize("seed, n, p, alpha", LASSO_CASES,
                         ids=[f"n{n}-p{p}-alpha{a:g}" for _, n, p, a in LASSO_CASES])
def test_lasso_matches_sklearn(lin, seed, n, p, alpha):
    X, y = regression_data(seed + 60, n=n, p=p, noise=1.0)
    model = fit(lin.Lasso(alpha=alpha, tol=1e-10, max_iter=2_000), X, y)
    oracle = _sklearn_lasso(X, y, alpha)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=0, atol=1e-6,
                 msg=f"coef_ with alpha = {alpha:g}, tol = 1e-10 ({LASSO_SHORT})", data=LASSO_HINT)
    assert_close(learned(model, "intercept_"), oracle.intercept_, rtol=0, atol=1e-6,
                 msg=f"intercept_ with alpha = {alpha:g}", data="intercept_ = mean(y) - mean(X) . coef_")


@pytest.mark.parametrize("alpha", [0.1, 0.8])
def test_lasso_without_intercept_matches_sklearn(lin, alpha):
    X, y = regression_data(70, n=50, p=5, noise=1.0, offset=1.5)
    model = fit(lin.Lasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=2_000), X, y)
    oracle = _sklearn_lasso(X, y, alpha, fit_intercept=False)
    assert_close(learned(model, "coef_"), oracle.coef_, rtol=0, atol=1e-6,
                 msg=f"fit_intercept=False, alpha = {alpha:g}: no centring ({LASSO_SHORT})", data=LASSO_HINT)
    assert_close(learned(model, "intercept_"), 0.0, atol=0, msg="fit_intercept=False: intercept_ = 0.0")


def test_lasso_docstring_example(lin):
    model = fit(lin.Lasso(alpha=0.5), np.array([[1.0], [2.0], [3.0]]), np.array([3.0, 5.0, 7.0]))
    assert_close(learned(model, "coef_"), [1.25], rtol=1e-9,
                 msg="coef_ of the docstring example (centred: rho = 4/3, z = 2/3, so soft_threshold(4/3, 0.5) / "
                     "(2/3) = 1.25)", data=LASSO_HINT)
    assert_close(learned(model, "intercept_"), 2.5, rtol=1e-9, msg="intercept_ of the docstring example")


def test_lasso_large_alpha_sets_every_weight_to_exactly_zero(lin):
    X, y = regression_data(71, n=50, p=6)
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    alpha_max = float(np.max(np.abs(Xc.T @ yc)) / len(y))
    model = fit(lin.Lasso(alpha=alpha_max * 1.001), X, y)
    coef = np.asarray(learned(model, "coef_"))
    if not np.all(coef == 0.0):
        raise AssertionError(f"expected every weight to be EXACTLY 0 when alpha >= max|Xcᵀyc| / n "
                             f"(= {alpha_max:.4g}), got {_short(coef)}")
    assert_close(learned(model, "intercept_"), y.mean(), rtol=1e-12, msg="all weights at 0: intercept_ = mean(y)")
    n_iter = learned(model, "n_iter_")
    assert n_iter == 1, (f"expected n_iter_ == 1: the weights start at 0 and the first pass changes nothing, so "
                         f"the descent stops after it, got n_iter_ = {n_iter}")


def test_lasso_just_below_alpha_max_keeps_one_weight(lin):
    X, y = regression_data(72, n=50, p=6)
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    scores = np.abs(Xc.T @ yc) / len(y)
    model = fit(lin.Lasso(alpha=float(scores.max()) * 0.95, tol=1e-10, max_iter=2_000), X, y)
    nonzero = np.flatnonzero(np.asarray(learned(model, "coef_")) != 0.0)
    assert nonzero.tolist() == [int(np.argmax(scores))], \
        (f"expected a single non-zero weight, on feature {int(np.argmax(scores))} (the most correlated with y), "
         f"just below alpha_max, got non-zero weights on features {nonzero.tolist()}")


def test_lasso_sets_the_useless_features_to_exactly_zero(lin):
    rng = np.random.default_rng(73)
    X = rng.normal(size=(120, 10))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 3] + 1.5 * X[:, 7] + 0.5 * rng.normal(size=120)
    model = fit(lin.Lasso(alpha=0.3, tol=1e-10, max_iter=2_000), X, y)
    oracle = _sklearn_lasso(X, y, 0.3)
    got = np.flatnonzero(np.asarray(learned(model, "coef_")) != 0.0).tolist()
    want = np.flatnonzero(oracle.coef_ != 0.0).tolist()
    assert got == want, (f"expected exact zeros on the useless features: non-zero weights on features {want} "
                         f"(scikit-learn), got {got} (soft_threshold must return exactly 0 inside [-alpha, alpha])")


def test_lasso_solution_satisfies_the_optimality_conditions(lin):
    X, y = regression_data(74, n=80, p=8, noise=2.0)
    alpha = 0.4
    model = fit(lin.Lasso(alpha=alpha, tol=1e-12, max_iter=2_000), X, y)
    w = np.asarray(learned(model, "coef_"))
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    grad = Xc.T @ (yc - Xc @ w) / len(y)                     # x_jᵀ r / n for every feature
    for j in range(len(w)):
        if w[j] != 0.0 and abs(grad[j] - alpha * np.sign(w[j])) > 1e-6:
            raise AssertionError(f"expected x_jᵀ r / n = alpha * sign(w_j) = {alpha * np.sign(w[j]):g} for the "
                                 f"non-zero weight w_{j} = {w[j]:.6g} (optimum of the lasso), got {grad[j]:.6g}")
        if w[j] == 0.0 and abs(grad[j]) > alpha + 1e-6:
            raise AssertionError(f"expected |x_jᵀ r| / n <= alpha = {alpha:g} for the zero weight w_{j} (else "
                                 f"moving it away from 0 would lower the objective), got {abs(grad[j]):.6g}")


def test_lasso_column_of_zeros_gets_a_zero_weight(lin):
    X, y = regression_data(75, n=40, p=3)
    X = np.column_stack([X[:, 0], np.zeros(40), X[:, 1:]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # judged on the result: a warning alone is not an error
        model = fit(lin.Lasso(alpha=0.1, tol=1e-10, max_iter=2_000), X, y,
                    hint="a column of zeros has z_j = 0: give it w_j = 0 without dividing", hint_when=_division_problem)
    coef = np.asarray(learned(model, "coef_"))
    assert np.isfinite(coef).all() and coef[1] == 0.0, \
        (f"expected a finite coef_ with an exact 0 for the column of zeros (feature 1), got {_short(coef)} "
         "(a column of zeros has z_j = 0: give it w_j = 0 without dividing)")


def test_lasso_counts_its_passes(lin):
    X, y = regression_data(76, n=60, p=6, noise=1.0)
    capped = fit(lin.Lasso(alpha=0.05, max_iter=3, tol=1e-12), X, y)
    n_iter = learned(capped, "n_iter_")
    assert n_iter == 3, (f"expected n_iter_ == 3 with max_iter=3 on a problem that needs more passes (n_iter_ "
                         f"counts the full passes done), got {n_iter}")
    free = fit(lin.Lasso(alpha=0.05, tol=1e-6, max_iter=10_000), X, y)
    n_free = learned(free, "n_iter_")
    if isinstance(n_free, (int, np.integer)) and n_free == 10_000:
        raise AssertionError("expected the descent to stop long before max_iter = 10 000 passes (this problem needs "
                             "a few dozen), got n_iter_ = 10000: it never met the tol criterion (is the residual "
                             "updated after each coordinate? is the largest change of the pass compared with tol?)")
    if not (isinstance(n_free, (int, np.integer)) and 1 <= n_free < 10_000):
        raise AssertionError(f"expected n_iter_ to be an integer between 1 and max_iter (the descent stops after the "
                             f"first pass whose largest change is < tol), got {n_free!r}")


def test_lasso_does_not_penalise_the_intercept(lin):
    X, y = regression_data(77, n=40, p=3, offset=0.0)
    base = fit(lin.Lasso(alpha=0.2, tol=1e-10, max_iter=2_000), X, y)
    shifted = fit(lin.Lasso(alpha=0.2, tol=1e-10, max_iter=2_000), X, y + 500.0)
    assert_close(learned(shifted, "coef_"), learned(base, "coef_"), rtol=0, atol=1e-8,
                 msg="adding 500 to every target must not change coef_ (centre X and y first)")
    assert_close(learned(shifted, "intercept_") - learned(base, "intercept_"), 500.0, rtol=1e-10,
                 msg="adding 500 to every target must add exactly 500 to intercept_")


def test_lasso_predict_and_score_match_sklearn(lin):
    X, y = regression_data(78, n=70, p=5, noise=2.0)
    model = fit(lin.Lasso(alpha=0.3, tol=1e-10, max_iter=2_000), X[:50], y[:50])
    oracle = _sklearn_lasso(X[:50], y[:50], 0.3)
    assert_close(call(model.predict, X[50:]), oracle.predict(X[50:]), rtol=0, atol=1e-5,
                 msg="predict(X) = X @ coef_ + intercept_")
    score = call(model.score, X[50:], y[50:], expected="a float")
    assert_python_float(score, "score(X, y)")
    assert_close(score, oracle.score(X[50:], y[50:]), rtol=1e-6, msg="score(X, y) = R² of predict(X) against y")


def test_lasso_fit_returns_self_and_does_not_modify_X_and_y(lin):
    X, y = regression_data(79, n=20, p=3)
    X0, y0 = X.copy(), y.copy()
    model = lin.Lasso(alpha=0.1)
    assert call(model.fit, X, y, copy_inputs=False) is model, "fit must return the fitted estimator itself (return self)"
    assert np.array_equal(X, X0) and np.array_equal(y, y0), \
        "expected fit to leave X and y unchanged (work on centred copies, keep the residual in a new array)"


def test_lasso_init_only_stores_the_hyperparameters(lin):
    X, y = regression_data(81, n=15, p=2)
    check_init_then_fit(lin.Lasso(alpha=0.5, fit_intercept=False, max_iter=50, tol=1e-3),
                        {"alpha": 0.5, "fit_intercept": False, "max_iter": 50, "tol": 1e-3}, X, y)


def test_lasso_checks_alpha_in_fit_not_in_init(lin):
    X, y = regression_data(80, n=10, p=2)
    for alpha in (0.0, -1.0):
        try:
            model = lin.Lasso(alpha=alpha)
        except ValueError:
            raise AssertionError(f"expected Lasso(alpha={alpha}) to be created without error and fit to raise "
                                 "ValueError (__init__ only stores; the checks happen in fit)") from None
        assert_raises_value_error(model.fit, X, y, why=f"alpha = {alpha} (use LinearRegression for alpha = 0)")


@pytest.mark.parametrize("X, y, why", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "X is 1-D (use X.reshape(-1, 1))"),
    (np.ones((4, 2)), np.ones(3), "4 rows in X but 3 values in y"),
], ids=["X-1d", "lengths-differ"])
def test_lasso_rejects_invalid_inputs(lin, X, y, why):
    assert_raises_value_error(lin.Lasso(alpha=0.1).fit, X, y, why=why)


# ================================================================== bias_variance_decomposition (9.24)
def test_bias_variance_decomposition_docstring_example(lin):
    result = call(lin.bias_variance_decomposition, [[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0], expected="a pair of floats")
    assert_close(result, (1.0, 1.0), msg="docstring example: (bias2, variance)")


@pytest.mark.parametrize("seed, n_models, n_points", [(0, 2, 5), (1, 10, 30), (2, 50, 100), (3, 7, 1)],
                         ids=["2-models", "10-models", "50-models", "1-point"])
def test_bias_variance_decomposition_matches_the_numpy_formulas(lin, seed, n_models, n_points):
    rng = np.random.default_rng(seed)
    f_true = np.sin(np.linspace(0, 3, n_points))
    predictions = f_true + 0.3 + rng.normal(size=(n_models, n_points)) * 0.5
    result = call(lin.bias_variance_decomposition, predictions, f_true, expected="a pair (bias2, variance)")
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError(f"expected a tuple (bias2, variance), got {type(result).__name__}"
                             + (f" of length {len(result)}" if hasattr(result, "__len__") else ""))
    bias2 = np.mean((predictions.mean(axis=0) - f_true) ** 2)
    variance = np.mean(predictions.var(axis=0))
    assert_close(result[0], bias2, rtol=1e-12,
                 msg="bias2 = mean over the points of (average model - f)², the average model being the mean of "
                     "the ROWS (axis=0)")
    assert_close(result[1], variance, rtol=1e-12,
                 msg="variance = mean over the points of the variance of the models (axis=0, ddof=0)")


def test_bias_variance_decomposition_returns_two_floats(lin):
    result = call(lin.bias_variance_decomposition, np.array([[0.0, 1.0], [2.0, 5.0], [1.0, 0.0]]),
                  np.array([0.5, 0.5]))
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError(f"expected a tuple of two floats, got {type(result).__name__}")
    assert_python_float(result[0], "bias2")
    assert_python_float(result[1], "variance")


def test_bias_variance_decomposition_sums_to_the_mean_squared_error_of_the_models(lin):
    rng = np.random.default_rng(5)
    f_true = rng.normal(size=40)
    predictions = f_true + rng.normal(size=(25, 40)) + rng.normal(size=40) * 0.7
    bias2, variance = call(lin.bias_variance_decomposition, predictions, f_true)
    mse = np.mean((predictions - f_true) ** 2)
    assert_close(bias2 + variance, mse, rtol=1e-12,
                 msg="bias2 + variance must equal mean_m mean_x (f_m(x) - f(x))² exactly (population variance)")


def test_bias_variance_decomposition_uses_the_population_variance(lin):
    _, variance = call(lin.bias_variance_decomposition, [[0.0, 0.0], [2.0, 4.0]], [1.0, 2.0])
    assert_close(variance, 2.5, msg="two models 0 and 2 (variance 1), 0 and 4 (variance 4): mean 2.5 with ddof=0 "
                                    "(ddof=1 would give 5)")


def test_bias_variance_decomposition_does_not_swap_its_results(lin):
    bias2, variance = call(lin.bias_variance_decomposition, [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]], [1.0, 2.0])
    assert_close((bias2, variance), (2.5, 0.0), atol=1e-15,
                 msg="identical models (no variance) that miss f by 2 and 1: (bias2, variance) = (2.5, 0)")


def test_bias_variance_decomposition_of_unbiased_models(lin):
    predictions = np.array([[1.0, 5.0, -2.0], [3.0, 7.0, 0.0]])     # their average is exactly f_true
    bias2, variance = call(lin.bias_variance_decomposition, predictions, [2.0, 6.0, -1.0])
    assert_close(bias2, 0.0, atol=1e-15, msg="the average model equals f: bias2 = 0")
    assert_close(variance, 1.0, msg="each point varies by ±1 around the average: variance = 1")


def test_bias_variance_decomposition_does_not_modify_its_inputs(lin):
    predictions = np.array([[1.0, 2.0], [3.0, 5.0]])
    f_true = np.array([1.0, 2.0])
    call(lin.bias_variance_decomposition, predictions, f_true, copy_inputs=False)
    assert np.array_equal(predictions, [[1.0, 2.0], [3.0, 5.0]]) and np.array_equal(f_true, [1.0, 2.0]), \
        "bias_variance_decomposition must not modify its inputs"


@pytest.mark.parametrize("predictions, f_true, why", [
    (np.ones((1, 4)), np.ones(4), "a single model (no variance can be measured)"),
    (np.ones((3, 4)), np.ones(5), "4 points per model but 5 values in f_true"),
    (np.ones(4), np.ones(4), "predictions is 1-D (expected (n_models, n_points))"),
], ids=["one-model", "points-differ", "predictions-1d"])
def test_bias_variance_decomposition_rejects_invalid_inputs(lin, predictions, f_true, why):
    assert_raises_value_error(lin.bias_variance_decomposition, predictions, f_true, why=why)


# ================================================================== bayes_line_posterior (9.26)
def oracle_posterior(x, y, slopes, intercepts, noise_std, prior_std):
    """Prior × likelihood with scipy.stats.norm, normalised in log space (scipy.special.logsumexp)."""
    S, B = np.meshgrid(np.asarray(slopes, float), np.asarray(intercepts, float))
    log_post = stats.norm.logpdf(S, 0.0, prior_std) + stats.norm.logpdf(B, 0.0, prior_std)
    for xi, yi in zip(np.asarray(x, float), np.asarray(y, float)):
        log_post = log_post + stats.norm.logpdf(yi, loc=S * xi + B, scale=noise_std)
    return np.exp(log_post - logsumexp(log_post))


def posterior(lin, x, y, slopes, intercepts, **kwargs):
    """The learner's posterior as a float array, checked to have the shape (n_b, n_s) and to be finite."""
    result = call(lin.bayes_line_posterior, x, y, slopes, intercepts, expected="an array of probabilities", **kwargs)
    post = np.asarray(result, dtype=float)
    shape = (len(intercepts), len(slopes))
    if post.shape != shape:
        raise AssertionError(f"expected shape (n_intercepts, n_slopes) = {shape} (row = intercept, column = slope: "
                             f"np.meshgrid(slopes, intercepts)), got {post.shape}")
    if not np.all(np.isfinite(post)):
        raise AssertionError(f"expected finite probabilities, got {int(np.sum(~np.isfinite(post)))} NaN or inf "
                             "value(s) (work in log space and subtract the maximum before np.exp)")
    return post


GRID_S = np.linspace(-2.0, 2.0, 41)
GRID_B = np.linspace(-1.5, 1.5, 31)


def test_bayes_line_posterior_docstring_example(lin):
    post = posterior(lin, [2.0], [2.0], slopes=[0.0, 1.0], intercepts=[0.0, 1.0], noise_std=1.0, prior_std=1.0)
    assert_close(post, [[0.1015, 0.4551], [0.2760, 0.1674]], rtol=0, atol=5e-5,
                 msg="docstring example (rows = intercepts 0 and 1, columns = slopes 0 and 1)")


@pytest.mark.parametrize("seed, n, noise_std, prior_std", [(0, 1, 0.3, 1.0), (1, 5, 0.2, 1.0), (2, 12, 0.5, 0.7),
                                                         (3, 3, 1.0, 2.0)],
                         ids=["1-point", "5-points", "12-points", "3-points-wide-prior"])
def test_bayes_line_posterior_matches_prior_times_likelihood(lin, seed, n, noise_std, prior_std):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=n)
    y = 0.8 * x - 0.3 + rng.normal(size=n) * noise_std
    post = posterior(lin, x, y, GRID_S, GRID_B, noise_std=noise_std, prior_std=prior_std)
    want = oracle_posterior(x, y, GRID_S, GRID_B, noise_std, prior_std)
    assert_close(post, want, rtol=1e-9, atol=1e-14,
                 msg=f"posterior ∝ N(slope; 0, {prior_std}) N(intercept; 0, {prior_std}) × Π N(y_i; slope x_i + "
                     f"intercept, {noise_std}), normalised (standard deviations: the exponents divide by 2 std²)")


def test_bayes_line_posterior_rows_are_intercepts_and_columns_are_slopes(lin):
    rng = np.random.default_rng(10)
    x = rng.uniform(-1, 1, size=200)
    y = 1.5 * x - 0.5 + rng.normal(size=200) * 0.05
    post = posterior(lin, x, y, GRID_S, GRID_B, noise_std=0.05, prior_std=1.0)
    if post.shape != (len(GRID_B), len(GRID_S)):
        raise AssertionError(f"expected shape (n_intercepts, n_slopes) = ({len(GRID_B)}, {len(GRID_S)}), got "
                             f"{post.shape} (np.meshgrid(slopes, intercepts) gives this orientation)")
    row, col = np.unravel_index(int(np.argmax(post)), post.shape)
    best = (round(float(GRID_B[row]), 3), round(float(GRID_S[col]), 3))
    assert (row, col) == (int(np.argmin(np.abs(GRID_B + 0.5))), int(np.argmin(np.abs(GRID_S - 1.5)))), \
        (f"expected the most probable line at intercept -0.5 (row) and slope 1.5 (column) for data from "
         f"y = 1.5 x - 0.5, got intercept {best[0]} and slope {best[1]}")


def test_bayes_line_posterior_is_a_probability_table(lin):
    post = posterior(lin, [0.3, -0.5], [0.6, 0.1], GRID_S, GRID_B, noise_std=0.2)
    assert post.min() >= 0.0, f"expected non-negative probabilities, got a minimum of {post.min():.3g}"
    assert_close(post.sum(), 1.0, rtol=1e-12, msg="the probabilities of the grid must sum to 1")


def test_bayes_line_posterior_without_points_is_the_prior(lin):
    post = posterior(lin, [], [], GRID_S, GRID_B, noise_std=0.1, prior_std=0.8)
    want = oracle_posterior([], [], GRID_S, GRID_B, 0.1, 0.8)
    assert_close(post, want, rtol=1e-12, atol=1e-15,
                 msg="no point: only the prior is left, an isotropic Gaussian centred on (0, 0) with std prior_std")
    row, col = np.unravel_index(int(np.argmax(post)), post.shape)
    assert (GRID_B[row], GRID_S[col]) == (GRID_B[15], GRID_S[20]), \
        "expected the prior to peak at slope 0 and intercept 0 (the horizontal line y = 0)"


def test_bayes_line_posterior_does_not_underflow_with_many_points(lin):
    rng = np.random.default_rng(11)
    x = rng.uniform(-1, 1, size=3000)
    y = -0.6 * x + 0.9 + rng.normal(size=3000) * 0.05
    post = posterior(lin, x, y, GRID_S, GRID_B, noise_std=0.05, prior_std=1.0)
    want = oracle_posterior(x, y, GRID_S, GRID_B, 0.05, 1.0)
    assert_close(post, want, rtol=1e-6, atol=1e-12,
                 msg="3 000 points: posterior ∝ prior × the 3 000 likelihoods, normalised (computed in log space)")


def test_bayes_line_posterior_one_point_at_a_time_gives_the_batch_posterior(lin):
    rng = np.random.default_rng(12)
    x = rng.uniform(-1, 1, size=6)
    y = 0.4 * x + 0.2 + rng.normal(size=6) * 0.3
    batch = posterior(lin, x, y, GRID_S, GRID_B, noise_std=0.3)
    S, B = np.meshgrid(GRID_S, GRID_B)
    for k in (1, 3, 5):
        first = posterior(lin, x[:k], y[:k], GRID_S, GRID_B, noise_std=0.3)
        log_rest = sum(stats.norm.logpdf(yi, loc=S * xi + B, scale=0.3) for xi, yi in zip(x[k:], y[k:]))
        log_seq = np.log(first) + log_rest
        assert_close(batch, np.exp(log_seq - logsumexp(log_seq)), rtol=1e-8, atol=1e-14,
                     msg=f"the posterior after {k} point(s), used as the prior of the {6 - k} others, must give the "
                         f"posterior of the 6 points at once")


def test_bayes_line_posterior_does_not_depend_on_the_order_of_the_points(lin):
    rng = np.random.default_rng(13)
    x, y = rng.uniform(-1, 1, size=8), rng.normal(size=8)
    order = rng.permutation(8)
    a = posterior(lin, x, y, GRID_S, GRID_B, noise_std=0.4)
    b = posterior(lin, x[order], y[order], GRID_S, GRID_B, noise_std=0.4)
    assert_close(b, a, rtol=1e-9, atol=1e-15, msg="shuffling the points must not change the posterior")


def test_bayes_line_posterior_mean_and_covariance_match_the_closed_form(lin):
    rng = np.random.default_rng(14)
    x = rng.uniform(-1, 1, size=10)
    y = 0.7 * x + 0.4 + rng.normal(size=10) * 0.5
    noise_std, prior_std = 0.5, 0.8
    Phi = np.column_stack([x, np.ones_like(x)])                   # columns: slope, intercept
    A = np.eye(2) / prior_std ** 2 + Phi.T @ Phi / noise_std ** 2  # posterior precision (Bishop eq. 3.54)
    cov = np.linalg.inv(A)
    mean = cov @ Phi.T @ y / noise_std ** 2                       # posterior mean (Bishop eq. 3.53)
    sd = np.sqrt(np.diag(cov))
    slopes = np.linspace(mean[0] - 8 * sd[0], mean[0] + 8 * sd[0], 301)
    intercepts = np.linspace(mean[1] - 8 * sd[1], mean[1] + 8 * sd[1], 291)
    post = posterior(lin, x, y, slopes, intercepts, noise_std=noise_std, prior_std=prior_std)
    S, B = np.meshgrid(slopes, intercepts)
    m_s, m_b = float(np.sum(post * S)), float(np.sum(post * B))
    assert_close((m_s, m_b), mean, rtol=0, atol=1e-6,
                 msg="posterior mean of (slope, intercept) on a fine grid against the closed form (Bishop 2006, "
                     "eq. 3.53)")
    grid_cov = [[np.sum(post * (S - m_s) ** 2), np.sum(post * (S - m_s) * (B - m_b))],
                [np.sum(post * (S - m_s) * (B - m_b)), np.sum(post * (B - m_b) ** 2)]]
    assert_close(grid_cov, cov, rtol=1e-4, atol=1e-8,
                 msg="posterior covariance of (slope, intercept) against the closed form (Bishop 2006, eq. 3.54): "
                     "check that noise_std and prior_std are squared in the exponents")


def test_bayes_line_posterior_does_not_modify_its_inputs(lin):
    x, y = np.array([0.1, 0.5]), np.array([0.3, 0.2])
    slopes, intercepts = GRID_S.copy(), GRID_B.copy()
    posterior(lin, x, y, slopes, intercepts, copy_inputs=False)
    assert np.array_equal(x, [0.1, 0.5]) and np.array_equal(y, [0.3, 0.2]), "x and y must not be modified"
    assert np.array_equal(slopes, GRID_S) and np.array_equal(intercepts, GRID_B), "the grids must not be modified"


@pytest.mark.parametrize("kwargs, why", [
    (dict(x=[0.1, 0.2], y=[0.3]), "2 abscissas but 1 ordinate"),
    (dict(x=[0.1], y=[0.3], noise_std=0.0), "noise_std = 0"),
    (dict(x=[0.1], y=[0.3], noise_std=-0.5), "noise_std = -0.5"),
    (dict(x=[0.1], y=[0.3], prior_std=0.0), "prior_std = 0"),
], ids=["lengths-differ", "noise-std-zero", "noise-std-negative", "prior-std-zero"])
def test_bayes_line_posterior_rejects_invalid_inputs(lin, kwargs, why):
    assert_raises_value_error(lin.bayes_line_posterior, slopes=GRID_S, intercepts=GRID_B, why=why, **kwargs)
