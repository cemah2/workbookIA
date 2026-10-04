"""Tests of mylearn.preprocessing (chapter 12): oracle tests and properties.

    pytest tests/test_ch12_preprocessing.py              # your code (mon_travail/mylearn/preprocessing.py)
    pytest tests/test_ch12_preprocessing.py --impl=ref   # the reference

Oracles: scikit-learn 1.6 (``StandardScaler``, ``MinMaxScaler``, ``SimpleImputer``,
``OrdinalEncoder``, ``OneHotEncoder(sparse_output=False)``, ``PCA(svd_solver="full")``)
and NumPy (means, variances, ``np.linalg.norm``), plus properties: a transformer learns
its statistics in ``fit`` only and reapplies them unchanged, ``inverse_transform`` goes
back to the original units, the inputs are never modified.
Every test name starts with the name of the class it tests, so that each exercise runs
its own group: ``-k "test_standard_scaler_"`` (12.16), ``"test_min_max_scaler_"``
(12.17), ``"test_simple_imputer_"`` (12.18), ``"test_ordinal_encoder_ or
test_one_hot_encoder_"`` (12.26), ``"test_pca_"`` (12.27). The tests of one class never
call another class.
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn import decomposition, impute
from sklearn import preprocessing as skp


@pytest.fixture
def pp(mylearn_module):
    return mylearn_module("preprocessing")


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
    """' (first difference at index i: expected a, got b)' for two arrays of the same shape, '' otherwise."""
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
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
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


def assert_same_objects(result, expected, msg=""):
    """Two tables of categories (strings, numbers or None) equal value by value."""
    got = np.asarray(result, dtype=object)
    want = np.asarray(expected, dtype=object)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {want.tolist()!r}, "
                   f"got {got.tolist()!r}")
    if got.tolist() != want.tolist():
        bad = next(i for i, (a, b) in enumerate(zip(got.ravel().tolist(), want.ravel().tolist())) if a != b)
        index = tuple(int(k) for k in np.unravel_index(bad, want.shape))
        _fail(msg, f"expected {want.ravel()[bad]!r} at index {index}, got {got.ravel()[bad]!r}",
              f"expected {want.tolist()!r}\ngot      {got.tolist()!r}")


def assert_float64_array(value, shape, what):
    if not isinstance(value, np.ndarray):
        raise AssertionError(f"expected {what} to return a NumPy array of shape {shape}, got {type(value).__name__} "
                             "(use np.asarray)")
    if value.shape != shape:
        raise AssertionError(f"expected {what} to return an array of shape {shape}, got shape {value.shape}")
    if value.dtype != np.float64:
        raise AssertionError(f"expected {what} to return float64 values, got dtype {value.dtype} "
                             "(np.asarray(..., dtype=float))")


def _copied(value):
    return value.copy() if isinstance(value, (np.ndarray, pd.DataFrame)) else value


def call(function, *args, expected="a result", copy_inputs=True, **kwargs):
    """Call the learner's method; an unexpected exception becomes a failure that says what was expected.

    The method receives copies of the array arguments, so that an array changed in place cannot spoil an oracle
    computed afterwards: only the tests named ..._does_not_modify_... (copy_inputs=False) pass the originals.
    """
    if copy_inputs:
        args = tuple(_copied(value) for value in args)
        kwargs = {key: _copied(value) for key, value in kwargs.items()}
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__qualname__", getattr(function, "__name__", "the method"))
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error` (or a subclass); `why` names the case in the failure message."""
    name = getattr(function, "__qualname__", getattr(function, "__name__", "the method"))
    try:
        function(*args, **kwargs)
    except NotImplementedError:
        raise                  # before `except error`: NotImplementedError is a RuntimeError; shown as "⏳"
    except error:
        return
    except Exception as exc:  # noqa: BLE001 - say which error was raised instead
        raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else "")
                             + f" (it raised {type(exc).__name__}: {exc})") from None
    raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else ""))


def fit(model, X, *args, **kwargs):
    """model.fit(X) without chaining: a fit that forgets `return self` is reported by its own test."""
    call(model.fit, X, *args, expected="fit to succeed", **kwargs)
    return model


def learned(model, name):
    """A learned attribute (mean_, scale_...), with a clear message when fit did not create it."""
    if not hasattr(model, name):
        raise AssertionError(f"expected fit to create the attribute {name}, but the fitted "
                             f"{type(model).__name__} has no {name} (attributes: {sorted(vars(model))})")
    return getattr(model, name)


def _public(model) -> dict:
    return {key: value for key, value in vars(model).items() if not key.startswith("_") and not key.endswith("_")}


def _stored_as_given(got, want) -> bool:
    """A hyperparameter kept exactly as passed: same type and same value (never a converted copy, as an array)."""
    if type(got) is not type(want):
        return False
    try:
        return bool(np.array_equal(got, want)) if isinstance(want, np.ndarray) else bool(got == want)
    except (TypeError, ValueError):
        return False


def check_init_then_fit(model, want, X):
    """__init__ only stores the hyperparameters; fit only adds learned (name_) or private (_name) attributes."""
    name = type(model).__name__
    params = _public(model)
    assert set(params) == set(want), (
        f"expected the hyperparameters of {name}(...) to be exactly {sorted(want)} right after __init__ (it only "
        f"stores its arguments; learned attributes, ending with _, are created by fit), got {sorted(params)}")
    for key, value in want.items():
        assert _stored_as_given(params[key], value), (
            f"expected __init__ to store {key} as given ({value!r}), got {params[key]!r} (keep the argument "
            "unchanged: conversions and checks belong in fit)")
    early = sorted(key for key in vars(model) if key.endswith("_") and not key.startswith("_"))
    assert not early, (f"expected no learned attribute right after __init__ (fit creates them), got {early} "
                       f"(an attribute like mean_ = None makes an unfitted transformer look fitted)")
    fit(model, X)
    added = sorted(set(_public(model)) - set(want))
    assert not added, (f"expected fit to add only learned attributes (names ending with _, like mean_) or private "
                       f"ones (starting with _), got {added}: clone (ch. 8) would take them for hyperparameters")
    changed = {key: vars(model)[key] for key in want if not _stored_as_given(vars(model)[key], want[key])}
    assert not changed, (f"expected fit to leave the hyperparameters unchanged ({want}), got {changed} (write what "
                         "fit computes in a learned attribute, like n_components_)")


def check_returns_self(model, X):
    result = call(model.fit, X, expected="fit to return the transformer itself")
    assert result is model, (f"expected {type(model).__name__}.fit(X) to return the transformer itself (return self), "
                             f"got {type(result).__name__}: fit_transform and pipelines chain fit(X).transform(X)")


def check_unfitted(model, methods, X):
    """Every listed method of an unfitted model must raise RuntimeError."""
    for method in methods:
        assert_raises(RuntimeError, getattr(model, method), X.copy() if isinstance(X, np.ndarray) else X,
                      why=f"{method} called before fit (the transformer has learnt nothing yet)")


def sk(cls, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(*args, **kwargs)


def numeric_data(seed, n=40, p=4, constant=None):
    """Features of very different scales and offsets (the elephants of §12.5), optionally one constant column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * np.array([1.0, 30.0, 0.05, 400.0, 2.0, 7.0])[:p] \
        + np.array([0.0, -50.0, 2.0, 1e4, 3.0, -1.0])[:p]
    if constant is not None:
        X[:, constant] = 3.25
    return X


# ================================================================== StandardScaler (12.16)
STD_CASES = [(0, 40, 4, None), (1, 7, 2, 1), (2, 200, 1, None), (3, 25, 5, 0)]
STD_IDS = ["n40_p4", "n7_p2_constant", "n200_p1", "n25_p5_constant"]


@pytest.mark.parametrize("seed, n, p, constant", STD_CASES, ids=STD_IDS)
@pytest.mark.parametrize("with_mean, with_std", [(True, True), (False, True), (True, False), (False, False)],
                         ids=["default", "no_mean", "no_std", "neither"])
def test_standard_scaler_matches_sklearn(pp, seed, n, p, constant, with_mean, with_std):
    X = numeric_data(seed, n, p, constant)
    X_new = numeric_data(seed + 100, 6, p)
    model = fit(pp.StandardScaler(with_mean=with_mean, with_std=with_std), X)
    oracle = sk(skp.StandardScaler, with_mean=with_mean, with_std=with_std).fit(X)
    options = f"StandardScaler(with_mean={with_mean}, with_std={with_std})"
    assert_close(learned(model, "mean_"), X.mean(axis=0), msg=f"{options}: mean_ = per-feature mean of the "
                 "training set (always computed)")
    assert_close(learned(model, "var_"), X.var(axis=0), rtol=1e-9, msg=f"{options}: var_ = per-feature POPULATION "
                 "variance (ddof=0, np.var(X, axis=0)), always computed")
    if with_std:
        assert_close(learned(model, "scale_"), oracle.scale_, msg=f"{options}: scale_ = sqrt(var_), with a 0 "
                     "replaced by 1")
    else:
        assert_close(learned(model, "scale_"), np.ones(p), msg=f"{options}: scale_ is all ones when with_std=False")
    assert learned(model, "n_features_in_") == p, f"expected n_features_in_ = {p}, got {model.n_features_in_!r}"
    assert learned(model, "n_samples_seen_") == n, f"expected n_samples_seen_ = {n}, got {model.n_samples_seen_!r}"
    assert_close(call(model.transform, X_new), oracle.transform(X_new), rtol=1e-9, atol=1e-9,
                 msg=f"{options}.transform(new data) uses the statistics of fit")
    assert_close(call(model.inverse_transform, X_new), oracle.inverse_transform(X_new), rtol=1e-9, atol=1e-9,
                 msg=f"{options}.inverse_transform(Z) = Z * scale_ + mean_ (without mean_ if with_mean=False)")


def test_standard_scaler_docstring_example(pp):
    X = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    model = fit(pp.StandardScaler(), X)
    assert_close(learned(model, "mean_"), [3.0, 10.0], msg="mean_ of the docstring example")
    assert_close(learned(model, "scale_"), [np.sqrt(8 / 3), 1.0], msg="scale_ of the docstring example (the "
                 "constant feature keeps scale 1)")
    assert_close(call(model.transform, X), [[-np.sqrt(1.5), 0.0], [0.0, 0.0], [np.sqrt(1.5), 0.0]], atol=1e-12,
                 msg="transform(X) of the docstring example")


def test_standard_scaler_uses_the_population_variance(pp):
    X = np.array([[0.0], [2.0], [4.0], [6.0]])
    model = fit(pp.StandardScaler(), X)
    assert_close(learned(model, "var_"), [5.0], msg="var_ of [0, 2, 4, 6] with ddof=0 (divide by n = 4, not n - 1)")
    assert_close(learned(model, "scale_"), [np.sqrt(5.0)], msg="scale_ = sqrt(var_) with ddof=0")


def test_standard_scaler_transformed_train_has_mean_0_and_std_1(pp):
    X = numeric_data(4, n=50, p=4)
    Z = call(fit(pp.StandardScaler(), X).transform, X)
    assert_close(Z.mean(axis=0), np.zeros(4), atol=1e-9, msg="per-feature mean of the standardised TRAINING set")
    assert_close(Z.std(axis=0), np.ones(4), rtol=1e-9, msg="per-feature std (ddof=0) of the standardised TRAINING "
                 "set")


def test_standard_scaler_constant_feature_is_centred_not_divided_by_zero(pp):
    X = numeric_data(5, n=12, p=3, constant=2)
    model = fit(pp.StandardScaler(), X)
    assert_close(learned(model, "scale_")[2], 1.0, msg="scale_ of a constant feature (its std is 0: 1 instead, no "
                 "division by 0)")
    Z = call(model.transform, X)
    assert_close(Z[:, 2], np.zeros(12), msg="a constant feature becomes 0 after centring (and is not divided)")


def test_standard_scaler_constant_column_whose_value_is_not_exact_in_binary(pp):
    # 0.1 has no exact binary form: the computed variance of seven 0.1 is about 1e-34, not 0
    X = np.column_stack([np.linspace(0.0, 6.0, 7), np.full(7, 0.1), [0.0, 1e-5, 2e-5, 0.0, 1e-5, 2e-5, 0.0]])
    model = fit(pp.StandardScaler(), X)
    oracle = sk(skp.StandardScaler).fit(X)
    assert_close(learned(model, "scale_"), oracle.scale_, rtol=1e-9, atol=0.0,
                 msg="scale_ of [ordinary, constant 0.1, tiny spread 1e-5] (column 1: its variance is 0, but the "
                     "computed one is about 1e-34; detect a constant column exactly, np.ptp(X, axis=0) == 0, and do "
                     "not treat the tiny but real spread of column 2 as 0)")
    assert_close(call(model.transform, np.array([[3.0, 0.11, 1e-5]])), oracle.transform([[3.0, 0.11, 1e-5]]),
                 rtol=1e-9, atol=1e-9, msg="transform([[3, 0.11, 1e-5]]): the constant column is only centred "
                 "(0.11 - 0.1 = 0.01)")


def test_standard_scaler_transform_reuses_the_statistics_of_fit(pp):
    X_train = numeric_data(6, n=30, p=3)
    X_test = numeric_data(7, n=8, p=3) * 2.0 + 5.0         # another scale: refitting would change the result
    model = fit(pp.StandardScaler(), X_train)
    want = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
    assert_close(call(model.transform, X_test), want, rtol=1e-9, atol=1e-9,
                 msg="transform(X_test) must use the mean and std of the TRAINING set (never recompute them on the "
                     "new data)")


def test_standard_scaler_second_fit_forgets_the_first(pp):
    model = fit(pp.StandardScaler(), numeric_data(8, n=20, p=2))
    X2 = numeric_data(9, n=15, p=3)
    fit(model, X2)
    assert_close(learned(model, "mean_"), X2.mean(axis=0), msg="after a second fit, mean_ comes from the second X only")
    assert model.n_features_in_ == 3, f"expected n_features_in_ = 3 after the second fit, got {model.n_features_in_!r}"


def test_standard_scaler_fit_transform_equals_fit_then_transform(pp):
    X = numeric_data(10, n=20, p=3)
    once = call(pp.StandardScaler().fit_transform, X)
    twice = call(fit(pp.StandardScaler(), X).transform, X)
    assert_close(once, twice, msg="fit_transform(X) = fit(X).transform(X)")


def test_standard_scaler_fit_transform_also_fits(pp):
    X = numeric_data(18, n=12, p=3)
    model = pp.StandardScaler()
    call(model.fit_transform, X)
    assert_close(learned(model, "mean_"), X.mean(axis=0), msg="mean_ after fit_transform(X) (fit_transform = fit(X), "
                 "then transform(X): the scaler keeps what it learnt)")


def test_standard_scaler_inverse_transform_goes_back_to_the_original_units(pp):
    X = numeric_data(11, n=25, p=4)
    model = fit(pp.StandardScaler(), X)
    back = call(model.inverse_transform, call(model.transform, X))
    assert_close(back, X, rtol=1e-9, atol=1e-9, msg="inverse_transform(transform(X)) gives X back")


def test_standard_scaler_returns_float64_arrays(pp):
    X = np.array([[1, 2], [3, 5], [6, 4]])                  # integers in, floats out
    model = fit(pp.StandardScaler(), X)
    assert_float64_array(call(model.transform, X), (3, 2), "transform")
    assert_float64_array(call(model.fit_transform, X), (3, 2), "fit_transform")
    assert_float64_array(call(model.inverse_transform, np.zeros((2, 2))), (2, 2), "inverse_transform")


def test_standard_scaler_accepts_lists_and_dataframes(pp):
    X = [[1.0, 10.0], [2.0, 30.0], [6.0, 20.0]]
    want = sk(skp.StandardScaler).fit_transform(np.array(X))
    assert_close(call(pp.StandardScaler().fit_transform, X), want, msg="fit_transform on a list of lists")
    frame = pd.DataFrame(X, columns=["a", "b"])
    assert_close(call(pp.StandardScaler().fit_transform, frame), want, msg="fit_transform on a pandas DataFrame "
                 "(np.asarray converts it)")


def test_standard_scaler_does_not_modify_its_input(pp):
    X = numeric_data(12, n=10, p=2)
    X0 = X.copy()
    model = pp.StandardScaler()
    call(model.fit, X, copy_inputs=False)
    call(model.transform, X, copy_inputs=False)
    call(model.inverse_transform, X, copy_inputs=False)
    assert np.array_equal(X, X0), "expected fit, transform and inverse_transform to leave X unchanged (work on a copy)"


def test_standard_scaler_fit_returns_self(pp):
    check_returns_self(pp.StandardScaler(), numeric_data(13, n=6, p=2))


def test_standard_scaler_init_only_stores_the_hyperparameters(pp):
    check_init_then_fit(pp.StandardScaler(with_mean=False, with_std=True), {"with_mean": False, "with_std": True},
                        numeric_data(14, n=6, p=2))


def test_standard_scaler_before_fit_raises_runtime_error(pp):
    check_unfitted(pp.StandardScaler(), ["transform", "inverse_transform"], numeric_data(15, n=4, p=2))


BAD_NUMERIC = [
    pytest.param(np.array([1.0, 2.0, 3.0]), "X is 1-D (use X.reshape(-1, 1))", id="1d"),
    pytest.param(np.ones((2, 3, 4)), "X is 3-D", id="3d"),
    pytest.param(np.array([[1.0, np.nan], [2.0, 3.0]]), "X contains NaN", id="nan"),
    pytest.param(np.array([[1.0, np.inf], [2.0, 3.0]]), "X contains an infinite value", id="inf"),
]


@pytest.mark.parametrize("X, why", BAD_NUMERIC)
def test_standard_scaler_fit_rejects_bad_inputs(pp, X, why):
    assert_raises(ValueError, pp.StandardScaler().fit, X.copy(), why=why)


@pytest.mark.parametrize("X, why", [
    pytest.param(np.ones((3, 1)), "1 feature, but fitted with 2 (NumPy would broadcast it silently)",
                 id="wrong_n_features"),
    pytest.param(np.array([1.0, 2.0]), "X is 1-D", id="1d"),
    pytest.param(np.array([[1.0, np.nan]]), "X contains NaN", id="nan"),
    pytest.param(np.array([[1.0, np.inf]]), "X contains an infinite value", id="inf"),
])
def test_standard_scaler_transform_rejects_bad_inputs(pp, X, why):
    model = fit(pp.StandardScaler(), numeric_data(16, n=5, p=2))
    assert_raises(ValueError, model.transform, X.copy(), why=why)


@pytest.mark.parametrize("Z, why", [
    pytest.param(np.ones((2, 1)), "Z has 1 feature, but fitted with 2 (NumPy would broadcast it silently)",
                 id="wrong_n_features"),
    pytest.param(np.array([1.0, 2.0]), "Z is 1-D (2 values for 2 features would broadcast into a 1-D result)",
                 id="1d"),
])
def test_standard_scaler_inverse_transform_rejects_bad_inputs(pp, Z, why):
    model = fit(pp.StandardScaler(), numeric_data(17, n=5, p=2))
    assert_raises(ValueError, model.inverse_transform, Z.copy(), why=why)


# ================================================================== MinMaxScaler (12.17)
RANGES = [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)]


@pytest.mark.parametrize("feature_range", RANGES, ids=["0_1", "m1_1", "2_5"])
@pytest.mark.parametrize("clip", [False, True], ids=["noclip", "clip"])
@pytest.mark.parametrize("seed, constant", [(20, None), (21, 1)], ids=["varied", "constant"])
def test_min_max_scaler_matches_sklearn(pp, feature_range, clip, seed, constant):
    X = numeric_data(seed, n=30, p=3, constant=constant)
    X_new = numeric_data(seed + 50, n=8, p=3) * 3.0           # new data, partly outside the training range
    model = fit(pp.MinMaxScaler(feature_range=feature_range, clip=clip), X)
    oracle = sk(skp.MinMaxScaler, feature_range=feature_range, clip=clip).fit(X)
    options = f"MinMaxScaler(feature_range={feature_range}, clip={clip})"
    for name, why in [("data_min_", "per-feature minimum of the training set"),
                      ("data_max_", "per-feature maximum of the training set"),
                      ("data_range_", "data_max_ - data_min_ (0 stays 0 for a constant feature)"),
                      ("scale_", "(b - a) / data_range_, with a range of 0 treated as 1"),
                      ("min_", "a - data_min_ * scale_")]:
        assert_close(learned(model, name), getattr(oracle, name), msg=f"{options}: {name} = {why}")
    assert learned(model, "n_features_in_") == 3, f"expected n_features_in_ = 3, got {model.n_features_in_!r}"
    assert_close(call(model.transform, X_new), oracle.transform(X_new), rtol=1e-9, atol=1e-9,
                 msg=f"{options}.transform(new data) = X * scale_ + min_" + (", then clipped to (a, b)" if clip else
                                                                           " (no clipping)"))
    Z = oracle.transform(X_new)
    assert_close(call(model.inverse_transform, Z), oracle.inverse_transform(Z), rtol=1e-9, atol=1e-9,
                 msg=f"{options}.inverse_transform(Z) = (Z - min_) / scale_ (never clipped)")


def test_min_max_scaler_docstring_example(pp):
    X = np.array([[-1.0, 2.0], [-0.5, 6.0], [0.0, 10.0], [1.0, 18.0]])
    model = fit(pp.MinMaxScaler(), X)
    assert_close(call(model.transform, X), [[0, 0], [0.25, 0.25], [0.5, 0.5], [1, 1]], msg="transform(X) of the "
                 "docstring example")
    assert_close(call(model.transform, np.array([[2.0, 2.0]])), [[1.5, 0.0]], msg="transform([[2, 2]]): a new value "
                 "above the training maximum goes above 1 (no clipping by default)")
    clipped = fit(pp.MinMaxScaler(clip=True), X)
    assert_close(call(clipped.transform, np.array([[2.0, 2.0]])), [[1.0, 0.0]], msg="MinMaxScaler(clip=True)"
                 ".transform([[2, 2]]) is clipped to [0, 1]")


@pytest.mark.parametrize("feature_range", RANGES, ids=["0_1", "m1_1", "2_5"])
def test_min_max_scaler_training_set_spans_exactly_the_feature_range(pp, feature_range):
    X = numeric_data(22, n=40, p=3)
    Z = call(fit(pp.MinMaxScaler(feature_range=feature_range), X).transform, X)
    a, b = feature_range
    assert_close(Z.min(axis=0), np.full(3, a), atol=1e-12, msg=f"per-feature minimum of the transformed TRAINING set "
                 f"(feature_range={feature_range})")
    assert_close(Z.max(axis=0), np.full(3, b), atol=1e-12, msg=f"per-feature maximum of the transformed TRAINING set "
                 f"(feature_range={feature_range})")


def test_min_max_scaler_new_data_can_leave_the_range_unless_clip(pp):
    X = np.array([[0.0], [10.0]])
    model = fit(pp.MinMaxScaler(), X)
    assert_close(call(model.transform, np.array([[-5.0], [20.0]])), [[-0.5], [2.0]],
                 msg="training range [0, 10]: -5 and 20 transform to -0.5 and 2 (outside [0, 1]; that is normal)")
    clipped = fit(pp.MinMaxScaler(clip=True), X)
    assert_close(call(clipped.transform, np.array([[-5.0], [20.0]])), [[0.0], [1.0]],
                 msg="with clip=True the same values are clipped to [0, 1]")


def test_min_max_scaler_constant_feature_maps_to_the_lower_bound(pp):
    X = numeric_data(23, n=10, p=2, constant=0)
    model = fit(pp.MinMaxScaler(feature_range=(-1.0, 1.0)), X)
    assert_close(learned(model, "data_range_")[0], 0.0, msg="data_range_ of a constant feature")
    assert_close(call(model.transform, X)[:, 0], np.full(10, -1.0), msg="a constant feature transforms to a = -1 "
                 "(its range is treated as 1: no division by 0)")


def test_min_max_scaler_inverse_transform_goes_back_to_the_original_units(pp):
    X = numeric_data(24, n=20, p=3)
    model = fit(pp.MinMaxScaler(feature_range=(-1.0, 1.0)), X)
    assert_close(call(model.inverse_transform, call(model.transform, X)), X, rtol=1e-9, atol=1e-9,
                 msg="inverse_transform(transform(X)) gives X back")


def test_min_max_scaler_second_fit_forgets_the_first(pp):
    model = fit(pp.MinMaxScaler(), numeric_data(32, n=20, p=2))
    X2 = numeric_data(33, n=15, p=3)
    fit(model, X2)
    assert_close(learned(model, "data_min_"), X2.min(axis=0), msg="after a second fit, data_min_ comes from the "
                 "second X only")
    assert_close(learned(model, "data_max_"), X2.max(axis=0), msg="after a second fit, data_max_ comes from the "
                 "second X only")
    assert model.n_features_in_ == 3, f"expected n_features_in_ = 3 after the second fit, got {model.n_features_in_!r}"


def test_min_max_scaler_inverse_transform_is_never_clipped(pp):
    model = fit(pp.MinMaxScaler(clip=True), np.array([[0.0], [10.0]]))
    assert_close(call(model.inverse_transform, np.array([[-0.5], [2.0]])), [[-5.0], [20.0]],
                 msg="MinMaxScaler(clip=True).inverse_transform([[-0.5], [2]]) on the range [0, 10]: (Z - min_) / "
                     "scale_, never clipped (clip only concerns transform)")


def test_min_max_scaler_fit_transform_equals_fit_then_transform(pp):
    X = numeric_data(25, n=15, p=2)
    assert_close(call(pp.MinMaxScaler().fit_transform, X), call(fit(pp.MinMaxScaler(), X).transform, X),
                 msg="fit_transform(X) = fit(X).transform(X)")


def test_min_max_scaler_fit_transform_also_fits(pp):
    X = numeric_data(34, n=12, p=3)
    model = pp.MinMaxScaler()
    call(model.fit_transform, X)
    assert_close(learned(model, "data_min_"), X.min(axis=0), msg="data_min_ after fit_transform(X) (fit_transform = "
                 "fit(X), then transform(X): the scaler keeps what it learnt)")


def test_min_max_scaler_returns_float64_arrays(pp):
    X = np.array([[1, 2], [3, 5], [6, 4]])
    model = fit(pp.MinMaxScaler(), X)
    assert_float64_array(call(model.transform, X), (3, 2), "transform")
    assert_float64_array(call(model.inverse_transform, np.zeros((2, 2))), (2, 2), "inverse_transform")


def test_min_max_scaler_does_not_modify_its_input(pp):
    X = numeric_data(26, n=10, p=2)
    X0 = X.copy()
    model = pp.MinMaxScaler(clip=True)
    call(model.fit, X, copy_inputs=False)
    call(model.transform, X, copy_inputs=False)
    call(model.inverse_transform, X, copy_inputs=False)
    assert np.array_equal(X, X0), "expected fit, transform and inverse_transform to leave X unchanged (work on a copy)"


def test_min_max_scaler_fit_returns_self(pp):
    check_returns_self(pp.MinMaxScaler(), numeric_data(27, n=6, p=2))


def test_min_max_scaler_init_only_stores_the_hyperparameters(pp):
    check_init_then_fit(pp.MinMaxScaler(feature_range=(-1.0, 1.0), clip=True),
                        {"feature_range": (-1.0, 1.0), "clip": True}, numeric_data(28, n=6, p=2))


def test_min_max_scaler_before_fit_raises_runtime_error(pp):
    check_unfitted(pp.MinMaxScaler(), ["transform", "inverse_transform"], numeric_data(29, n=4, p=2))


@pytest.mark.parametrize("feature_range, why", [
    pytest.param((1.0, 0.0), "feature_range (1, 0): a > b", id="reversed"),
    pytest.param((0.5, 0.5), "feature_range (0.5, 0.5): a = b", id="empty"),
])
def test_min_max_scaler_fit_rejects_a_bad_feature_range(pp, feature_range, why):
    assert_raises(ValueError, pp.MinMaxScaler(feature_range=feature_range).fit, numeric_data(30, n=5, p=2), why=why)


@pytest.mark.parametrize("X, why", BAD_NUMERIC)
def test_min_max_scaler_fit_rejects_bad_inputs(pp, X, why):
    assert_raises(ValueError, pp.MinMaxScaler().fit, X.copy(), why=why)


@pytest.mark.parametrize("X, why", [
    pytest.param(np.ones((2, 1)), "X has 1 feature, but fitted with 2 (NumPy would broadcast it silently)",
                 id="wrong_n_features"),
    pytest.param(np.array([1.0, 2.0]), "X is 1-D", id="1d"),
    pytest.param(np.array([[np.inf, 1.0]]), "X contains an infinite value (invalid, as in fit)", id="inf"),
])
def test_min_max_scaler_transform_rejects_bad_inputs(pp, X, why):
    model = fit(pp.MinMaxScaler(), numeric_data(31, n=5, p=2))
    assert_raises(ValueError, model.transform, X.copy(), why=why)


@pytest.mark.parametrize("Z, why", [
    pytest.param(np.ones((2, 1)), "Z has 1 feature, but fitted with 2 (NumPy would broadcast it silently)",
                 id="wrong_n_features"),
    pytest.param(np.array([0.5, 0.5]), "Z is 1-D (2 values for 2 features would broadcast into a 1-D result)",
                 id="1d"),
])
def test_min_max_scaler_inverse_transform_rejects_bad_inputs(pp, Z, why):
    model = fit(pp.MinMaxScaler(), numeric_data(31, n=5, p=2))
    assert_raises(ValueError, model.inverse_transform, Z.copy(), why=why)


# ================================================================== SimpleImputer (12.18)
def missing_data(seed, n=30, p=3, rate=0.3, integers=True):
    """Small integer values (ties for most_frequent) with NaN everywhere, each column keeping observed values."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, size=(n, p)).astype(float) if integers else rng.normal(size=(n, p)) * 10
    holes = rng.random((n, p)) < rate
    holes[0] = False                                       # every column keeps at least one observed value
    X[holes] = np.nan
    return X


@pytest.mark.parametrize("strategy, fill_value", [("mean", None), ("median", None), ("most_frequent", None),
                                                  ("constant", None), ("constant", -7.5)],
                         ids=["mean", "median", "most_frequent", "constant_default", "constant_value"])
@pytest.mark.parametrize("seed, integers", [(40, True), (41, False), (42, True)], ids=["ints", "floats", "ints2"])
def test_simple_imputer_matches_sklearn(pp, strategy, fill_value, seed, integers):
    X = missing_data(seed, integers=integers)
    X_new = missing_data(seed + 10, n=8, integers=integers)
    model = fit(pp.SimpleImputer(strategy=strategy, fill_value=fill_value), X)
    oracle = sk(impute.SimpleImputer, strategy=strategy, fill_value=fill_value).fit(X)
    options = f"SimpleImputer(strategy={strategy!r}, fill_value={fill_value!r})"
    hint = {"mean": "mean of the OBSERVED values of each column (np.nanmean, or a mask)",
            "median": "median of the observed values of each column",
            "most_frequent": "most frequent observed value of each column, the SMALLEST one among ties",
            "constant": "fill_value for every column (None means 0.0)"}[strategy]
    assert_close(learned(model, "statistics_"), oracle.statistics_, msg=f"{options}: statistics_ = {hint}")
    assert learned(model, "n_features_in_") == 3, f"expected n_features_in_ = 3, got {model.n_features_in_!r}"
    assert_close(call(model.transform, X_new), oracle.transform(X_new), msg=f"{options}.transform(new data): every "
                 "NaN of column j replaced by statistics_[j], the other values unchanged")


def test_simple_imputer_docstring_example(pp):
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
    model = fit(pp.SimpleImputer(strategy="mean"), X)
    assert_close(learned(model, "statistics_"), [2.0, 6.0], msg="statistics_ of the docstring example")
    assert_close(call(model.transform, X), [[1, 6], [3, 4], [2, 8]], msg="transform(X) of the docstring example")


def test_simple_imputer_most_frequent_takes_the_smallest_value_among_ties(pp):
    X = np.array([[5.0], [2.0], [5.0], [2.0], [9.0], [np.nan]])
    model = fit(pp.SimpleImputer(strategy="most_frequent"), X)
    assert_close(learned(model, "statistics_"), [2.0], msg="2 and 5 both appear twice: most_frequent keeps the "
                 "SMALLEST of the tied values")


def test_simple_imputer_median_of_an_even_number_of_values(pp):
    X = np.array([[1.0], [2.0], [10.0], [20.0], [np.nan]])
    model = fit(pp.SimpleImputer(strategy="median"), X)
    assert_close(learned(model, "statistics_"), [np.median([1.0, 2.0, 10.0, 20.0])],
                 msg="median of the observed 1, 2, 10, 20 (an even count): the mean of the two middle values, "
                     "(2 + 10) / 2, as np.median")


def test_simple_imputer_constant_default_fill_value_is_zero(pp):
    X = np.array([[1.0, np.nan], [np.nan, 4.0]])
    model = fit(pp.SimpleImputer(strategy="constant"), X)
    assert_close(learned(model, "statistics_"), [0.0, 0.0], msg="strategy='constant' with fill_value=None fills "
                 "with 0.0")


def test_simple_imputer_transform_reuses_the_statistics_of_fit(pp):
    X_train = np.array([[1.0], [3.0], [np.nan]])
    X_test = np.array([[np.nan], [100.0], [200.0]])
    model = fit(pp.SimpleImputer(strategy="mean"), X_train)
    assert_close(call(model.transform, X_test), [[2.0], [100.0], [200.0]],
                 msg="the NaN of the test set is replaced by the mean of the TRAINING set (2), never by a statistic "
                     "of the test set")


def test_simple_imputer_constant_fills_a_column_that_is_entirely_nan(pp):
    X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    model = fit(pp.SimpleImputer(strategy="constant", fill_value=-1.0), X)
    assert_close(call(model.transform, X), [[-1.0, 1.0], [-1.0, 2.0]], msg="strategy='constant' fills an entirely "
                 "NaN column too")


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_imputer_rejects_a_column_that_is_entirely_nan(pp, strategy):
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    assert_raises(ValueError, pp.SimpleImputer(strategy=strategy).fit, X,
                  why=f"column 1 is entirely NaN: strategy={strategy!r} has nothing to learn (documented difference "
                      "with scikit-learn, which drops the column)")


def test_simple_imputer_rejects_an_unknown_strategy(pp):
    assert_raises(ValueError, pp.SimpleImputer(strategy="average").fit, missing_data(43),
                  why="strategy='average' is not one of 'mean', 'median', 'most_frequent', 'constant'")


@pytest.mark.parametrize("X, why", [
    pytest.param(np.array([1.0, np.nan, 3.0]), "X is 1-D (use X.reshape(-1, 1))", id="1d"),
    pytest.param(np.ones((2, 2, 2)), "X is 3-D", id="3d"),
])
def test_simple_imputer_fit_rejects_bad_inputs(pp, X, why):
    assert_raises(ValueError, pp.SimpleImputer().fit, X.copy(), why=why)


@pytest.mark.parametrize("X, why", [
    pytest.param(np.ones((2, 1)), "X has 1 feature, but fitted with 3 (NumPy would broadcast it silently)",
                 id="wrong_n_features"),
    pytest.param(np.array([np.nan, 1.0, 2.0]), "X is 1-D (3 values for 3 features would broadcast silently)",
                 id="1d"),
])
def test_simple_imputer_transform_rejects_bad_inputs(pp, X, why):
    model = fit(pp.SimpleImputer(), missing_data(44))
    assert_raises(ValueError, model.transform, X.copy(), why=why)


def test_simple_imputer_returns_float64_without_nan(pp):
    X = missing_data(45)
    Z = call(fit(pp.SimpleImputer(strategy="median"), X).transform, X)
    assert_float64_array(Z, X.shape, "transform")
    assert not np.isnan(Z).any(), "expected no NaN left after transform"


def test_simple_imputer_second_fit_forgets_the_first(pp):
    model = fit(pp.SimpleImputer(), missing_data(51))
    X2 = missing_data(52, p=4)
    fit(model, X2)
    assert_close(learned(model, "statistics_"), np.nanmean(X2, axis=0), msg="after a second fit, statistics_ come "
                 "from the second X only")


def test_simple_imputer_fit_transform_equals_fit_then_transform(pp):
    X = missing_data(46)
    assert_close(call(pp.SimpleImputer().fit_transform, X), call(fit(pp.SimpleImputer(), X).transform, X),
                 msg="fit_transform(X) = fit(X).transform(X)")


def test_simple_imputer_fit_transform_also_fits(pp):
    X = missing_data(53)
    model = pp.SimpleImputer()
    call(model.fit_transform, X)
    assert_close(learned(model, "statistics_"), np.nanmean(X, axis=0), msg="statistics_ after fit_transform(X) "
                 "(fit_transform = fit(X), then transform(X): the imputer keeps what it learnt)")


def test_simple_imputer_does_not_modify_its_input(pp):
    X = missing_data(47)
    X0 = X.copy()
    model = pp.SimpleImputer()
    call(model.fit, X, copy_inputs=False)
    call(model.transform, X, copy_inputs=False)
    assert np.array_equal(X, X0, equal_nan=True), ("expected fit and transform to leave X unchanged, NaN included "
                                                   "(fill a copy: np.where returns a new array)")


def test_simple_imputer_fit_returns_self(pp):
    check_returns_self(pp.SimpleImputer(), missing_data(48))


def test_simple_imputer_init_only_stores_the_hyperparameters(pp):
    check_init_then_fit(pp.SimpleImputer(strategy="constant", fill_value=3.0),
                        {"strategy": "constant", "fill_value": 3.0}, missing_data(49))


def test_simple_imputer_before_fit_raises_runtime_error(pp):
    check_unfitted(pp.SimpleImputer(), ["transform"], missing_data(50))


# ================================================================== OrdinalEncoder (12.26)
PENGUINS = np.array([["Biscoe", "male", "Adelie"], ["Dream", "female", "Gentoo"], ["Torgersen", "female", "Adelie"],
                     ["Biscoe", "male", "Chinstrap"], ["Dream", "male", "Adelie"], ["Biscoe", "female", "Gentoo"]],
                    dtype=object)
NEW_PENGUINS = np.array([["Dream", "female", "Chinstrap"], ["Torgersen", "male", "Gentoo"]], dtype=object)
NUMBERS = np.array([[3.0, 10.0], [1.0, 20.0], [3.0, 10.0], [2.0, 30.0]])
# numbers whose text order differs from their numeric order ("-1" < "10" < "9"), and mixed-case strings
SIGNED_NUMBERS = np.array([[10.0, 2.0], [9.0, 2.0], [-1.0, 30.0], [10.0, 4.0]])
MIXED_CASE = np.array([["b"], ["B"], ["a"], ["A"]], dtype=object)


def categories_of(model):
    cats = learned(model, "categories_")
    return [np.asarray(c, dtype=object).tolist() for c in cats]


def check_sorted_categories(model, X, rule):
    """categories_ learnt on X equal those of scikit-learn (the sorted unique values of each column)."""
    oracle = sk(skp.OneHotEncoder, sparse_output=False) if "OneHot" in type(model).__name__ else sk(skp.OrdinalEncoder)
    want = [c.tolist() for c in oracle.fit(X).categories_]
    assert categories_of(model) == want, (f"expected categories_ = {want} ({rule}), got {categories_of(model)}")


def check_numpy_categories(model):
    kinds = sorted({type(c).__name__ for c in learned(model, "categories_")})
    assert kinds == ["ndarray"], (f"expected categories_ to be a list of NumPy arrays (one per column), got elements "
                                  f"of type {kinds}")


@pytest.mark.parametrize("X, X_new", [pytest.param(PENGUINS, NEW_PENGUINS, id="strings"),
                                      pytest.param(NUMBERS, NUMBERS[::-1], id="numbers")])
def test_ordinal_encoder_matches_sklearn(pp, X, X_new):
    model = fit(pp.OrdinalEncoder(), X)
    oracle = sk(skp.OrdinalEncoder).fit(X)
    want = [c.tolist() for c in oracle.categories_]
    assert categories_of(model) == want, (f"expected categories_ = {want} (the SORTED unique values of each column), "
                                          f"got {categories_of(model)}")
    assert learned(model, "n_features_in_") == X.shape[1], f"expected n_features_in_ = {X.shape[1]}"
    assert_close(call(model.transform, X_new), oracle.transform(X_new), msg="transform(X): code of a category = its "
                 "position in categories_[j]")
    codes = oracle.transform(X_new)
    assert_same_objects(call(model.inverse_transform, codes), oracle.inverse_transform(codes),
                        msg="inverse_transform(codes): the category at each position")


def test_ordinal_encoder_docstring_example(pp):
    X = [["red", "S"], ["blue", "M"], ["red", "L"]]
    model = fit(pp.OrdinalEncoder(), X)
    assert categories_of(model) == [["blue", "red"], ["L", "M", "S"]], (
        f"expected categories_ = [['blue', 'red'], ['L', 'M', 'S']] (alphabetical order), got {categories_of(model)}")
    assert_close(call(model.transform, X), [[1, 2], [0, 1], [1, 0]], msg="transform(X) of the docstring example")


def test_ordinal_encoder_follows_the_given_order(pp):
    X = [["S"], ["L"], ["M"], ["XL"]]
    order = [["S", "M", "L", "XL"]]
    model = fit(pp.OrdinalEncoder(categories=order), X)
    oracle = sk(skp.OrdinalEncoder, categories=order).fit(X)
    assert categories_of(model) == order, f"expected categories_ = {order} (the given order), got {categories_of(model)}"
    assert_close(call(model.transform, X), oracle.transform(X), msg="transform with categories=[['S', 'M', 'L', 'XL']]"
                 ": S < M < L < XL get the codes 0, 1, 2, 3")


def test_ordinal_encoder_sorts_numbers_numerically(pp):
    check_sorted_categories(fit(pp.OrdinalEncoder(), SIGNED_NUMBERS), SIGNED_NUMBERS,
                            "numbers in numeric order: -1 < 9 < 10, not in the order of their text")


def test_ordinal_encoder_sorts_strings_like_python(pp):
    check_sorted_categories(fit(pp.OrdinalEncoder(), MIXED_CASE), MIXED_CASE,
                            "the order of sorted() and np.unique: upper case before lower case")


def test_ordinal_encoder_transform_returns_float64_codes(pp):
    model = fit(pp.OrdinalEncoder(), PENGUINS)
    assert_float64_array(call(model.transform, PENGUINS), PENGUINS.shape, "transform")
    check_numpy_categories(model)


def test_ordinal_encoder_fit_transform_also_fits(pp):
    model = pp.OrdinalEncoder()
    call(model.fit_transform, PENGUINS)
    check_sorted_categories(model, PENGUINS, "after fit_transform(X): fit_transform = fit(X), then transform(X)")


def test_ordinal_encoder_inverse_transform_returns_the_categories(pp):
    model = fit(pp.OrdinalEncoder(), PENGUINS)
    back = call(model.inverse_transform, call(model.transform, PENGUINS))
    assert isinstance(back, np.ndarray), f"expected inverse_transform to return a NumPy array, got {type(back).__name__}"
    assert_same_objects(back, PENGUINS, msg="inverse_transform(transform(X)) gives X back")


def test_ordinal_encoder_rejects_a_category_unseen_in_fit(pp):
    model = fit(pp.OrdinalEncoder(), PENGUINS)
    assert_raises(ValueError, model.transform, np.array([["Biscoe", "male", "Emperor"]], dtype=object),
                  why="'Emperor' was not seen in fit (column 2)")


@pytest.mark.parametrize("categories, why", [
    pytest.param([["Biscoe", "Dream"], ["female", "male"], ["Adelie", "Chinstrap", "Gentoo"]],
                 "'Torgersen' is absent from the list given for column 0", id="value_not_in_list"),
    pytest.param([["Biscoe", "Dream", "Torgersen"], ["female", "male"]], "2 lists for 3 columns", id="too_few_lists"),
])
def test_ordinal_encoder_fit_rejects_bad_categories(pp, categories, why):
    assert_raises(ValueError, pp.OrdinalEncoder(categories=categories).fit, PENGUINS, why=why)


@pytest.mark.parametrize("codes, why", [
    pytest.param([[3.0, 0.0, 0.0]], "code 3 in column 0, which has only 3 categories (0 to 2)", id="too_big"),
    pytest.param([[-1.0, 0.0, 0.0]], "code -1", id="negative"),
    pytest.param([[0.5, 0.0, 0.0]], "code 0.5 is not an integer", id="not_integer"),
    pytest.param([[0.0, 0.0]], "2 columns, fitted with 3", id="wrong_n_features"),
])
def test_ordinal_encoder_inverse_transform_rejects_bad_codes(pp, codes, why):
    model = fit(pp.OrdinalEncoder(), PENGUINS)
    assert_raises(ValueError, model.inverse_transform, np.array(codes), why=why)


def test_ordinal_encoder_rejects_1d_input_and_a_wrong_number_of_features(pp):
    assert_raises(ValueError, pp.OrdinalEncoder().fit, np.array(["a", "b"], dtype=object),
                  why="X is 1-D (use a list of lists, or X.reshape(-1, 1))")
    model = fit(pp.OrdinalEncoder(), PENGUINS)
    assert_raises(ValueError, model.transform, PENGUINS[:, :2], why="X has 2 columns, fitted with 3")


def test_ordinal_encoder_does_not_modify_its_input(pp):
    X = PENGUINS.copy()
    model = pp.OrdinalEncoder()
    call(model.fit, X, copy_inputs=False)
    call(model.transform, X, copy_inputs=False)
    assert X.tolist() == PENGUINS.tolist(), "expected fit and transform to leave X unchanged"


def test_ordinal_encoder_fit_returns_self(pp):
    check_returns_self(pp.OrdinalEncoder(), PENGUINS)


def test_ordinal_encoder_init_only_stores_the_hyperparameter(pp):
    check_init_then_fit(pp.OrdinalEncoder(), {"categories": "auto"}, PENGUINS)


def test_ordinal_encoder_before_fit_raises_runtime_error(pp):
    check_unfitted(pp.OrdinalEncoder(), ["transform"], PENGUINS)
    assert_raises(RuntimeError, pp.OrdinalEncoder().inverse_transform, np.zeros((1, 3)),
                  why="inverse_transform called before fit")


# ================================================================== OneHotEncoder (12.26)
@pytest.mark.parametrize("drop", [None, "first", "if_binary"], ids=["drop_none", "drop_first", "drop_if_binary"])
@pytest.mark.parametrize("handle_unknown", ["error", "ignore"])
def test_one_hot_encoder_matches_sklearn(pp, drop, handle_unknown):
    model = fit(pp.OneHotEncoder(drop=drop, handle_unknown=handle_unknown), PENGUINS)
    oracle = sk(skp.OneHotEncoder, drop=drop, handle_unknown=handle_unknown, sparse_output=False).fit(PENGUINS)
    options = f"OneHotEncoder(drop={drop!r}, handle_unknown={handle_unknown!r})"
    want = [c.tolist() for c in oracle.categories_]
    assert categories_of(model) == want, (f"{options}: expected categories_ = {want} (sorted, dropped ones included), "
                                          f"got {categories_of(model)}")
    drop_idx = learned(model, "drop_idx_")
    if oracle.drop_idx_ is None:
        assert drop_idx is None, f"{options}: expected drop_idx_ = None (nothing dropped), got {drop_idx!r}"
    else:
        got = _index_list(drop_idx)
        assert got == list(oracle.drop_idx_), (f"{options}: expected drop_idx_ = {list(oracle.drop_idx_)} (an object "
                                               f"array: the index of the dropped category of each column, None where "
                                               f"nothing is dropped), got {drop_idx!r}")
    assert_close(call(model.transform, NEW_PENGUINS), oracle.transform(NEW_PENGUINS),
                 msg=f"{options}.transform(X): one block of 0/1 per column, the dropped columns removed")
    codes = oracle.transform(PENGUINS)
    assert_same_objects(call(model.inverse_transform, codes), _sk_inverse(oracle, codes),
                        msg=f"{options}.inverse_transform(encoding)")
    names = call(model.get_feature_names_out, ["island", "sex", "species"])
    assert np.asarray(names, dtype=object).tolist() == oracle.get_feature_names_out(["island", "sex", "species"]).tolist(), (
        f"{options}: expected get_feature_names_out(['island', 'sex', 'species']) = "
        f"{oracle.get_feature_names_out(['island', 'sex', 'species']).tolist()}, got {np.asarray(names).tolist()}")


def _index_list(values):
    """drop_idx_ read as a list of int or None; None when it cannot be read so (a number, NaN...)."""
    try:
        return [None if v is None else int(v) for v in values]
    except (TypeError, ValueError):
        return None


def _sk_inverse(oracle, codes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return oracle.inverse_transform(codes)


def test_one_hot_encoder_matches_sklearn_on_numbers(pp):
    model = fit(pp.OneHotEncoder(), SIGNED_NUMBERS)
    oracle = sk(skp.OneHotEncoder, sparse_output=False).fit(SIGNED_NUMBERS)
    check_sorted_categories(model, SIGNED_NUMBERS, "numbers in numeric order: -1 < 9 < 10, not in the order of their "
                            "text")
    assert_close(call(model.transform, SIGNED_NUMBERS), oracle.transform(SIGNED_NUMBERS),
                 msg="transform(X) of numeric columns: one 0/1 block per column, in the numeric order of the categories")
    names = np.asarray(call(model.get_feature_names_out), dtype=object).tolist()
    assert names == oracle.get_feature_names_out().tolist(), (
        f"expected get_feature_names_out() = {oracle.get_feature_names_out().tolist()} (str() of each category), got "
        f"{names}")


def test_one_hot_encoder_sorts_strings_like_python(pp):
    check_sorted_categories(fit(pp.OneHotEncoder(), MIXED_CASE), MIXED_CASE,
                            "the order of sorted() and np.unique: upper case before lower case")


def test_one_hot_encoder_docstring_example(pp):
    X = [["Biscoe", "male"], ["Dream", "female"], ["Biscoe", "female"]]
    model = fit(pp.OneHotEncoder(), X)
    assert_close(call(model.transform, X), [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]],
                 msg="transform(X) of the docstring example")
    names = call(model.get_feature_names_out, ["island", "sex"])
    assert np.asarray(names, dtype=object).tolist() == ["island_Biscoe", "island_Dream", "sex_female", "sex_male"], (
        f"expected ['island_Biscoe', 'island_Dream', 'sex_female', 'sex_male'], got {np.asarray(names).tolist()}")
    assert_close(call(pp.OneHotEncoder(drop="if_binary").fit_transform, X), [[0, 1], [1, 0], [0, 0]],
                 msg="OneHotEncoder(drop='if_binary').fit_transform(X): both columns are binary, each keeps one column")


def test_one_hot_encoder_default_names_are_x0_x1(pp):
    model = fit(pp.OneHotEncoder(), PENGUINS)
    oracle = sk(skp.OneHotEncoder, sparse_output=False).fit(PENGUINS)
    names = np.asarray(call(model.get_feature_names_out), dtype=object).tolist()
    assert names == oracle.get_feature_names_out().tolist(), (
        f"expected get_feature_names_out() = {oracle.get_feature_names_out().tolist()} (x0, x1... when no names are "
        f"given), got {names}")


def test_one_hot_encoder_each_row_has_one_1_per_column(pp):
    Z = call(fit(pp.OneHotEncoder(), PENGUINS).transform, PENGUINS)
    assert_float64_array(Z, (6, 8), "transform (3 + 2 + 3 categories)")
    assert set(np.unique(Z).tolist()) <= {0.0, 1.0}, f"expected only 0 and 1, got {np.unique(Z).tolist()}"
    assert_close(Z.sum(axis=1), np.full(6, 3.0), msg="each row has exactly one 1 in each of its 3 blocks")


def test_one_hot_encoder_unknown_category_raises_or_gives_zeros(pp):
    new = np.array([["Biscoe", "female", "Emperor"]], dtype=object)
    strict = fit(pp.OneHotEncoder(), PENGUINS)
    assert_raises(ValueError, strict.transform, new, why="'Emperor' unseen in fit with handle_unknown='error'")
    lenient = fit(pp.OneHotEncoder(handle_unknown="ignore"), PENGUINS)
    Z = call(lenient.transform, new)
    assert_close(Z, [[1, 0, 0, 1, 0, 0, 0, 0]], msg="handle_unknown='ignore': the block of the unknown 'Emperor' is "
                 "all zeros, the other blocks are encoded normally")
    back = call(lenient.inverse_transform, Z)
    assert_same_objects(back, [["Biscoe", "female", None]], msg="inverse_transform of an all-zero block without drop "
                        "gives None")


def test_one_hot_encoder_inverse_of_a_zero_block_is_the_dropped_category(pp):
    model = fit(pp.OneHotEncoder(drop="first"), PENGUINS)
    Z = call(model.transform, PENGUINS)
    assert_same_objects(call(model.inverse_transform, Z), PENGUINS, msg="drop='first': inverse_transform gives the "
                        "categories back, an all-zero block being the dropped (first) category")


def test_one_hot_encoder_inverse_of_a_zero_block_where_nothing_is_dropped(pp):
    model = fit(pp.OneHotEncoder(drop="if_binary", handle_unknown="ignore"), PENGUINS)
    Z = call(model.transform, np.array([["Biscoe", "female", "Emperor"]], dtype=object))
    assert_same_objects(call(model.inverse_transform, Z), [["Biscoe", "female", None]],
                        msg="drop='if_binary', handle_unknown='ignore': the species column (3 categories) drops "
                            "nothing, so the all-zero block of the unknown 'Emperor' gives None (look at drop_idx_ of "
                            "that column, not at the option drop)")


def test_one_hot_encoder_returns_numpy_arrays(pp):
    model = fit(pp.OneHotEncoder(), PENGUINS)
    names = call(model.get_feature_names_out)
    assert isinstance(names, np.ndarray), (f"expected get_feature_names_out to return a NumPy array, got "
                                           f"{type(names).__name__} (np.array(names, dtype=object))")
    check_numpy_categories(model)


def test_one_hot_encoder_fit_transform_also_fits(pp):
    model = pp.OneHotEncoder()
    call(model.fit_transform, PENGUINS)
    check_sorted_categories(model, PENGUINS, "after fit_transform(X): fit_transform = fit(X), then transform(X)")


@pytest.mark.parametrize("params, why", [
    pytest.param({"drop": "last"}, "drop='last' is not None, 'first' or 'if_binary'", id="drop"),
    pytest.param({"handle_unknown": "warn"}, "handle_unknown='warn' is not 'error' or 'ignore'", id="handle_unknown"),
])
def test_one_hot_encoder_fit_rejects_bad_options(pp, params, why):
    assert_raises(ValueError, pp.OneHotEncoder(**params).fit, PENGUINS, why=why)


def test_one_hot_encoder_fit_rejects_a_value_absent_from_the_given_categories(pp):
    categories = [["Biscoe", "Dream"], ["female", "male"], ["Adelie", "Chinstrap", "Gentoo"]]
    assert_raises(ValueError, pp.OneHotEncoder(categories=categories).fit, PENGUINS,
                  why="'Torgersen' is absent from the list given for column 0")


def test_one_hot_encoder_rejects_bad_shapes(pp):
    assert_raises(ValueError, pp.OneHotEncoder().fit, np.array(["a", "b"], dtype=object),
                  why="X is 1-D (use a list of lists, or X.reshape(-1, 1))")
    model = fit(pp.OneHotEncoder(), PENGUINS)
    assert_raises(ValueError, model.inverse_transform, np.zeros((1, 7)), why="7 columns, but the encoding has 8")
    assert_raises(ValueError, model.get_feature_names_out, ["island", "sex"], why="2 names for 3 input columns")
    assert_raises(ValueError, model.transform, PENGUINS[:, :2], why="X has 2 columns, fitted with 3")


def test_one_hot_encoder_follows_the_given_categories(pp):
    order = [["Torgersen", "Dream", "Biscoe"], ["male", "female"], ["Gentoo", "Chinstrap", "Adelie"]]
    model = fit(pp.OneHotEncoder(categories=order), PENGUINS)
    oracle = sk(skp.OneHotEncoder, categories=order, sparse_output=False).fit(PENGUINS)
    assert categories_of(model) == order, f"expected categories_ = {order} (the given lists), got {categories_of(model)}"
    assert_close(call(model.transform, PENGUINS), oracle.transform(PENGUINS), msg="transform with given categories: "
                 "the blocks follow the given order")


def test_one_hot_encoder_does_not_modify_its_input(pp):
    X = PENGUINS.copy()
    model = pp.OneHotEncoder()
    call(model.fit, X, copy_inputs=False)
    call(model.transform, X, copy_inputs=False)
    assert X.tolist() == PENGUINS.tolist(), "expected fit and transform to leave X unchanged"


def test_one_hot_encoder_fit_returns_self(pp):
    check_returns_self(pp.OneHotEncoder(), PENGUINS)


def test_one_hot_encoder_init_only_stores_the_hyperparameters(pp):
    check_init_then_fit(pp.OneHotEncoder(drop="first", handle_unknown="ignore"),
                        {"categories": "auto", "drop": "first", "handle_unknown": "ignore"}, PENGUINS)


def test_one_hot_encoder_before_fit_raises_runtime_error(pp):
    check_unfitted(pp.OneHotEncoder(), ["transform"], PENGUINS)
    assert_raises(RuntimeError, pp.OneHotEncoder().inverse_transform, np.zeros((1, 8)),
                  why="inverse_transform called before fit")
    assert_raises(RuntimeError, pp.OneHotEncoder().get_feature_names_out, why="get_feature_names_out called before fit")


# ================================================================== PCA (12.27)
def correlated_data(seed, n=60, p=5):
    """Correlated features of different variances (no direction ties), shifted away from 0."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p)) * np.linspace(3.0, 0.3, p)
    return rng.normal(size=(n, p)) @ A + rng.normal(size=p) * 10


def _signs_aligned(got, want):
    """Flip the rows of `got` to the sign of the matching rows of `want` (components are defined up to sign)."""
    signs = np.sign(np.sum(got * want, axis=1))
    return got * np.where(signs == 0, 1.0, signs)[:, None]


# the wide case keeps k = 4 < 5 = rank of the centred data: a direction of zero variance is not unique
PCA_CASES = [(60, 61, 5, None), (61, 40, 6, 3), (62, 80, 4, 0.9), (63, 30, 5, 0.5), (64, 6, 9, 4)]
PCA_IDS = ["n61_p5_all", "n40_p6_k3", "n80_p4_ratio0.9", "n30_p5_ratio0.5", "n6_p9_wide_k4"]


@pytest.mark.parametrize("seed, n, p, n_components", PCA_CASES, ids=PCA_IDS)
@pytest.mark.parametrize("whiten", [False, True], ids=["plain", "whiten"])
def test_pca_matches_sklearn(pp, seed, n, p, n_components, whiten):
    X = correlated_data(seed, n, p)
    X_new = correlated_data(seed + 100, 7, p)
    model = fit(pp.PCA(n_components=n_components, whiten=whiten), X)
    oracle = sk(decomposition.PCA, n_components=n_components, whiten=whiten, svd_solver="full").fit(X)
    options = f"PCA(n_components={n_components!r}, whiten={whiten})"
    k = oracle.n_components_
    assert learned(model, "n_components_") == k, (f"{options}: expected n_components_ = {k}"
                                                   + (" (smallest k whose cumulative explained variance ratio is "
                                                      "STRICTLY greater than the float)" if isinstance(n_components,
                                                                                                      float) else "")
                                                   + f", got {model.n_components_!r}")
    assert learned(model, "n_features_in_") == p, f"expected n_features_in_ = {p}, got {model.n_features_in_!r}"
    assert_close(learned(model, "mean_"), X.mean(axis=0), msg=f"{options}: mean_ = per-feature mean")
    assert_close(learned(model, "explained_variance_"), oracle.explained_variance_, rtol=1e-8, atol=1e-10,
                 msg=f"{options}: explained_variance_ = S² / (n_samples - 1) (ddof=1), largest first")
    assert_close(learned(model, "explained_variance_ratio_"), oracle.explained_variance_ratio_, rtol=1e-8,
                 atol=1e-12, msg=f"{options}: explained_variance_ratio_ = variance / TOTAL variance of all the "
                                 "directions (not only the kept ones)")
    assert_close(learned(model, "singular_values_"), oracle.singular_values_, rtol=1e-8, atol=1e-10,
                 msg=f"{options}: singular_values_ = S of the CENTRED data")
    comps = np.asarray(learned(model, "components_"), dtype=float)
    if comps.shape != oracle.components_.shape:
        raise AssertionError(f"{options}: expected components_ of shape {oracle.components_.shape} (one unit row per "
                             f"component), got {comps.shape}")
    assert_close(_signs_aligned(comps, oracle.components_), oracle.components_, rtol=1e-6, atol=1e-8,
                 msg=f"{options}: components_ (compared up to the sign of each row) = the first rows of Vᵀ")
    Z = np.asarray(call(model.transform, X_new), dtype=float)
    Z_want = oracle.transform(X_new)
    if Z.shape != Z_want.shape:
        raise AssertionError(f"{options}.transform(new data): expected shape {Z_want.shape}, got {Z.shape}")
    signs = np.sign(np.sum(comps * oracle.components_, axis=1))
    assert_close(Z * np.where(signs == 0, 1.0, signs), Z_want, rtol=1e-6, atol=1e-8,
                 msg=f"{options}.transform(new data) = (X - mean_) @ components_.T"
                     + (", divided by sqrt(explained_variance_)" if whiten else ""))
    assert_close(call(model.inverse_transform, Z), oracle.inverse_transform(Z_want), rtol=1e-6, atol=1e-8,
                 msg=f"{options}.inverse_transform(Z) = Z @ components_ + mean_"
                     + (" (whitening undone first)" if whiten else ""))


def test_pca_docstring_example(pp):
    X = np.array([[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
    model = fit(pp.PCA(n_components=2), X)
    assert_close(learned(model, "explained_variance_ratio_"), [0.99244289, 0.00755711], rtol=1e-7,
                 msg="explained_variance_ratio_ of the docstring example")
    assert_close(learned(model, "components_"), [[0.83849224, 0.54491354], [-0.54491354, 0.83849224]], rtol=1e-7,
                 msg="components_ of the docstring example (signs included: largest |coordinate| of each row positive)")
    small = fit(pp.PCA(n_components=0.95), X)
    assert learned(small, "n_components_") == 1, f"expected PCA(n_components=0.95).n_components_ = 1, got {small.n_components_!r}"


def test_pca_sign_convention_largest_coordinate_is_positive(pp):
    X = correlated_data(65, n=50, p=5)
    comps = np.asarray(learned(fit(pp.PCA(), X), "components_"), dtype=float)
    largest = comps[np.arange(len(comps)), np.argmax(np.abs(comps), axis=1)]
    assert (largest > 0).all(), (f"expected the largest-magnitude coordinate of each component to be positive (flip "
                                 f"the row otherwise, as scikit-learn), got these coordinates: {_short(largest)}")


def test_pca_sign_convention_does_not_depend_on_the_sum_of_the_coordinates(pp):
    rng = np.random.default_rng(80)
    u = np.array([0.8, -0.4, -0.4, -0.2])          # unit; its largest coordinate is positive, but its sum is -0.2
    X = rng.normal(size=(60, 1)) * 5.0 * u + rng.normal(size=(60, 4)) * 0.05
    first = np.asarray(learned(fit(pp.PCA(n_components=1), X), "components_"), dtype=float)[0]
    assert first[np.argmax(np.abs(first))] > 0, (
        f"expected the largest-magnitude coordinate of the first component to be positive (about +0.8 here, although "
        f"the coordinates sum to -0.2), got {_short(first, 3)}")


def test_pca_components_are_orthonormal(pp):
    X = correlated_data(66, n=50, p=5)
    comps = np.asarray(learned(fit(pp.PCA(n_components=3), X), "components_"), dtype=float)
    assert_close(comps @ comps.T, np.eye(3), atol=1e-10, msg="components_ @ components_.T: unit rows, orthogonal to "
                 "each other")


def test_pca_explained_variance_uses_ddof_1(pp):
    X = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    model = fit(pp.PCA(n_components=1), X)
    assert_close(learned(model, "explained_variance_"), [np.var(X[:, 0], ddof=1)], msg="explained_variance_ of points "
                 "on the x axis = variance of x with ddof=1 (divide S² by n - 1)")


def test_pca_whitened_projection_has_unit_variance(pp):
    X = correlated_data(67, n=80, p=4)
    Z = call(fit(pp.PCA(whiten=True), X).transform, X)
    assert_close(np.var(Z, axis=0, ddof=1), np.ones(4), rtol=1e-8, msg="variance (ddof=1) of each column of the "
                 "whitened training projection")


def test_pca_keeping_all_components_reconstructs_the_data(pp):
    X = correlated_data(68, n=30, p=4)
    for whiten in (False, True):
        model = fit(pp.PCA(whiten=whiten), X)
        back = call(model.inverse_transform, call(model.transform, X))
        assert_close(back, X, rtol=1e-8, atol=1e-8, msg=f"PCA(whiten={whiten}): inverse_transform(transform(X)) = X "
                     "when every component is kept")


def test_pca_float_n_components_is_the_smallest_k_strictly_above(pp):
    X = correlated_data(69, n=50, p=5)
    full = sk(decomposition.PCA, svd_solver="full").fit(X)
    cumsum = np.cumsum(full.explained_variance_ratio_)
    for target in (cumsum[1] - 1e-9, cumsum[1] + 1e-9, 0.3, 0.999):
        if not 0 < target < 1:
            continue
        want = sk(decomposition.PCA, n_components=target, svd_solver="full").fit(X).n_components_
        got = learned(fit(pp.PCA(n_components=float(target)), X), "n_components_")
        assert got == want, (f"PCA(n_components={target:.10f}): expected n_components_ = {want} (the smallest k whose "
                             f"cumulative ratio {_short(cumsum, 10)} is STRICTLY greater), got {got!r}")


def test_pca_float_n_components_equal_to_a_cumulative_ratio_takes_one_more(pp):
    # orthogonal centred columns of norms 10, 8 and 6: ratios exactly 0.5, 0.32, 0.18 (no rounding anywhere)
    X = np.column_stack([[5, -5, 5, -5, 0], [4, 4, -4, -4, 0], [3, -3, -3, 3, 0]]) + np.array([7.0, -3.0, 0.5])
    got = learned(fit(pp.PCA(n_components=0.5), X), "n_components_")
    assert got == 2, ("PCA(n_components=0.5) on data whose first component explains exactly 50 %: expected "
                      f"n_components_ = 2 (the cumulative ratio must be STRICTLY greater than 0.5), got {got!r}")


def test_pca_on_wide_data_keeps_min_n_samples_n_features_components(pp):
    X = correlated_data(79, n=6, p=9)
    model = fit(pp.PCA(), X)
    assert learned(model, "n_components_") == 6, ("PCA() on 6 samples of 9 features: expected n_components_ = "
                                                  f"min(n_samples, n_features) = 6, got {model.n_components_!r}")
    comps = np.asarray(learned(model, "components_"), dtype=float)
    assert comps.shape == (6, 9), f"expected components_ of shape (6, 9), got {comps.shape}"
    variance = np.asarray(learned(model, "explained_variance_"), dtype=float)
    assert_close(variance[5], 0.0, atol=1e-8 * variance[0], msg="the 6th variance: 6 centred points span only 5 "
                 "directions, so the last one carries no variance")


def test_pca_n_components_cannot_exceed_the_number_of_samples(pp):
    assert_raises(ValueError, pp.PCA(n_components=5).fit, correlated_data(78, n=4, p=6),
                  why="n_components = 5 > min(n_samples, n_features) = 4 (only 4 samples)")


def test_pca_fit_transform_equals_fit_then_transform(pp):
    X = correlated_data(70, n=40, p=4)
    once = np.asarray(call(pp.PCA(n_components=2).fit_transform, X), dtype=float)
    model = fit(pp.PCA(n_components=2), X)
    assert_close(once, call(model.transform, X), rtol=1e-8, atol=1e-8, msg="fit_transform(X) = fit(X).transform(X)")


def test_pca_fit_transform_also_fits(pp):
    X = correlated_data(81, n=30, p=4)
    model = pp.PCA(n_components=2)
    call(model.fit_transform, X)
    assert_close(learned(model, "mean_"), X.mean(axis=0), msg="mean_ after fit_transform(X) (fit_transform = fit(X), "
                 "then transform(X): the PCA keeps what it learnt)")


def test_pca_returns_float64_arrays(pp):
    X = correlated_data(71, n=20, p=4)
    model = fit(pp.PCA(n_components=2), X)
    assert_float64_array(call(model.transform, X), (20, 2), "transform with n_components=2")
    assert_float64_array(call(model.inverse_transform, np.zeros((3, 2))), (3, 4), "inverse_transform")


@pytest.mark.parametrize("n_components, why", [
    pytest.param(0, "n_components = 0", id="zero"),
    pytest.param(-1, "n_components = -1", id="negative"),
    pytest.param(6, "n_components = 6 > min(n_samples, n_features) = 5", id="too_big"),
    pytest.param(1.0, "n_components = 1.0: a float must be in (0, 1)", id="float_one"),
    pytest.param(1.5, "n_components = 1.5", id="float_above_one"),
    pytest.param(0.0, "n_components = 0.0", id="float_zero"),
    pytest.param("2", "n_components = '2' (a string)", id="string"),
])
def test_pca_fit_rejects_bad_n_components(pp, n_components, why):
    assert_raises(ValueError, pp.PCA(n_components=n_components).fit, correlated_data(72, n=20, p=5), why=why)


def test_pca_rejects_bad_inputs(pp):
    assert_raises(ValueError, pp.PCA().fit, np.array([1.0, 2.0, 3.0]), why="X is 1-D")
    model = fit(pp.PCA(n_components=2, whiten=True), correlated_data(73, n=20, p=4))
    assert_raises(ValueError, model.transform, np.ones((3, 1)),
                  why="X has 1 feature, fitted with 4 (NumPy would broadcast it silently)")
    assert_raises(ValueError, model.inverse_transform, np.ones((3, 1)),
                  why="Z has 1 column, but n_components_ = 2 (NumPy would broadcast it silently)")


def test_pca_does_not_modify_its_input(pp):
    X = correlated_data(74, n=20, p=4)
    X0 = X.copy()
    model = pp.PCA(n_components=2, whiten=True)
    call(model.fit, X, copy_inputs=False)
    Z = call(model.transform, X, copy_inputs=False)
    Z0 = np.array(Z, dtype=float, copy=True)
    call(model.inverse_transform, Z, copy_inputs=False)
    assert np.array_equal(X, X0), "expected fit and transform to leave X unchanged (centre a copy: X - mean_)"
    assert np.array_equal(np.asarray(Z, dtype=float), Z0), "expected inverse_transform to leave Z unchanged"


def test_pca_fit_returns_self(pp):
    check_returns_self(pp.PCA(), correlated_data(75, n=10, p=3))


@pytest.mark.parametrize("n_components", [2, 0.9, None], ids=["int", "float", "none"])
def test_pca_init_only_stores_the_hyperparameters(pp, n_components):
    check_init_then_fit(pp.PCA(n_components=n_components, whiten=True),
                        {"n_components": n_components, "whiten": True}, correlated_data(76, n=30, p=4))


def test_pca_before_fit_raises_runtime_error(pp):
    check_unfitted(pp.PCA(), ["transform", "inverse_transform"], correlated_data(77, n=5, p=3))
