"""Tests of mylearn.naive_bayes (chapter 13): oracle tests and properties.

    pytest tests/test_ch13_naive_bayes.py              # your code (mon_travail/mylearn/naive_bayes.py)
    pytest tests/test_ch13_naive_bayes.py --impl=ref   # the reference

Oracles: SciPy (``scipy.stats.norm.logpdf``, ``scipy.special.logsumexp``), scikit-learn
1.6 (``GaussianNB``, ``MultinomialNB(alpha, force_alpha=True)``: learned attributes,
``predict_joint_log_proba``, ``predict_log_proba``, ``predict_proba``, ``predict``)
and NumPy (population variances, counts), plus properties: log-probabilities never
underflow (far-away points, long documents), ``logsumexp`` of every row of
``predict_log_proba`` is 0, the inputs are never modified.
Every test name starts with the name of the function or class it tests, so that each
exercise runs its own group: ``-k "test_log_gaussian_pdf_ or test_logsumexp_ or
test_gaussian_nb_"`` (13.28), ``"test_multinomial_nb_"`` (13.29). The tests of
MultinomialNB never call GaussianNB, and the other way round.
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest
from scipy import special, stats
from sklearn import naive_bayes as sknb


@pytest.fixture
def nbm(mylearn_module):
    return mylearn_module("naive_bayes")


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


def assert_same_labels(result, expected, msg=""):
    """Two arrays of labels (numbers or strings) equal value by value."""
    got = np.asarray(result, dtype=object)
    want = np.asarray(expected, dtype=object)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {want.tolist()!r}, "
                   f"got {got.tolist()!r}")
    if got.tolist() != want.tolist():
        bad = next(i for i, (a, b) in enumerate(zip(got.ravel().tolist(), want.ravel().tolist())) if a != b)
        index = tuple(int(k) for k in np.unravel_index(bad, want.shape))
        where = index[0] if len(index) == 1 else index
        _fail(msg, f"expected {want.ravel()[bad]!r} at index {where}, got {got.ravel()[bad]!r}",
              f"expected {want.tolist()!r}\ngot      {got.tolist()!r}")


def assert_indices(result, expected, msg=""):
    """Integer index arrays equal (same shape, same values)."""
    got = np.asarray(result)
    want = np.asarray(expected)
    if got.shape != want.shape:
        _fail(msg, f"expected indices of shape {want.shape}, got shape {got.shape}: expected {_short(want)}, "
                   f"got {_short(got)}")
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)[0]
        where = tuple(int(k) for k in bad)
        _fail(msg, f"expected {_short(want)}, got {_short(got)} (first difference at index {where}: expected "
                   f"{int(want[tuple(bad)])}, got {got[tuple(bad)]!r})")


def _copied(value):
    return value.copy() if isinstance(value, np.ndarray) else value


def call(function, *args, expected="a result", copy_inputs=True, **kwargs):
    """Call the learner's function; an unexpected exception becomes a failure that says what was expected.

    The function receives copies of the array arguments, so that an array changed in place cannot spoil an oracle
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


def build(cls, **params):
    """cls(**params), the constructor only storing its arguments: a check done in __init__ instead of fit is reported
    with a first line that says what was expected."""
    try:
        return cls(**params)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"expected {cls.__name__}({', '.join(f'{k}={v!r}' for k, v in params.items())}) to only "
                             f"store its arguments (the check belongs in fit), but __init__ raised "
                             f"{type(exc).__name__}: {exc}") from None


def fit(model, X, y, **kwargs):
    """model.fit(X, y) without chaining: a fit that forgets `return self` is reported by its own test."""
    call(model.fit, X, y, expected="fit to succeed", **kwargs)
    return model


def learned(model, name):
    """A learned attribute (classes_...), with a clear message when fit did not create it."""
    if not hasattr(model, name):
        raise AssertionError(f"expected fit to create the attribute {name}, but the fitted "
                             f"{type(model).__name__} has no {name} (attributes: {sorted(vars(model))})")
    return getattr(model, name)


def _public(model) -> dict:
    return {key: value for key, value in vars(model).items() if not key.startswith("_") and not key.endswith("_")}


def _stored_as_given(got, want) -> bool:
    """A hyperparameter kept exactly as passed: same type and same value (never a converted copy)."""
    if type(got) is not type(want):
        return False
    try:
        return bool(np.array_equal(got, want)) if isinstance(want, np.ndarray) else bool(got == want)
    except (TypeError, ValueError):
        return False


def check_init_then_fit(model, want, X, y):
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
                       f"(an attribute like classes_ = None makes an unfitted classifier look fitted)")
    fit(model, X, y)
    added = sorted(set(_public(model)) - set(want))
    assert not added, (f"expected fit to add only learned attributes (names ending with _, like theta_) or private "
                       f"ones (starting with _), got {added}: clone (ch. 8) would take them for hyperparameters")
    changed = {key: vars(model)[key] for key in want if not _stored_as_given(vars(model)[key], want[key])}
    assert not changed, (f"expected fit to leave the hyperparameters unchanged ({want}), got {changed}")


def check_returns_self(model, X, y):
    result = call(model.fit, X, y, expected="fit to return the classifier itself")
    assert result is model, (f"expected {type(model).__name__}.fit(X, y) to return the classifier itself (return "
                             f"self), got {type(result).__name__}: {type(model).__name__}(...).fit(X, y).predict(X) "
                             "chains the calls")


def sk(cls, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(*args, **kwargs)


def check_unfitted(model, methods, X):
    """Every listed method of an unfitted model must raise RuntimeError."""
    for method in methods:
        assert_raises(RuntimeError, getattr(model, method), X.copy(),
                      why=f"{method} called before fit (the classifier has learnt nothing yet)")


def gaussian_data(seed, n=60, p=3, labels=(0, 1, 2)):
    """Overlapping classes, features of different scales, every class present (at least 2 samples)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    codes = rng.integers(0, len(labels), size=n)
    m = min(n, 2 * len(labels))
    codes[:m] = np.arange(m) % len(labels)
    scales = np.array([1.0, 20.0, 0.1, 3.0])[:p]
    centres = rng.normal(size=(len(labels), p)) * 1.2 * scales
    X = centres[codes] + rng.normal(size=(n, p)) * scales * rng.uniform(0.5, 1.5, size=(len(labels), p))[codes]
    return X, labels[codes]


def count_data(seed, n=40, d=6, labels=("en", "fr", "de"), length=20):
    """Word counts (non-negative integers, many zeros), unbalanced classes with class-specific word rates."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    weights = np.arange(1, len(labels) + 1) / np.sum(np.arange(1, len(labels) + 1))
    codes = rng.choice(len(labels), size=n, p=weights)
    codes[: len(labels)] = np.arange(len(labels))
    rates = rng.dirichlet(np.full(d, 0.6), size=len(labels))
    X = np.array([rng.multinomial(length, rates[c]) for c in codes])
    return X, labels[codes]


# ================================================================== log_gaussian_pdf (13.28)
LGP_CASES = [
    pytest.param(np.linspace(-3.0, 4.0, 7), 0.5, 2.0, id="1d_scalar_params"),
    pytest.param(np.arange(15.0).reshape(5, 1, 3) / 3, np.arange(12.0).reshape(1, 4, 3) / 4,
                 np.arange(1.0, 13.0).reshape(1, 4, 3) / 5, id="samples_classes_features"),
    pytest.param(np.arange(12.0).reshape(4, 3) - 5.0, np.array([0.0, -1.0, 2.0]), np.array([0.5, 4.0, 9.0]),
                 id="2d_per_feature"),
    pytest.param(2.0, np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[1.0, 0.25], [9.0, 4.0]]), id="scalar_x"),
]


@pytest.mark.parametrize("x, mean, var", LGP_CASES)
def test_log_gaussian_pdf_matches_scipy(nbm, x, mean, var):
    want = stats.norm.logpdf(x, loc=mean, scale=np.sqrt(var))
    got = call(nbm.log_gaussian_pdf, x, mean, var)
    assert_close(got, want, rtol=1e-12, atol=1e-12,
                 msg=f"log_gaussian_pdf(x {np.shape(x)}, mean {np.shape(mean)}, var {np.shape(var)}) = "
                     "-0.5 log(2 pi var) - (x - mean)^2 / (2 var), with the broadcast shape "
                     f"{np.shape(want)} (var is the VARIANCE, scale = sqrt(var))")


def test_log_gaussian_pdf_docstring_example(nbm):
    assert_close(call(nbm.log_gaussian_pdf, np.array([0.0, 1.0]), 0.0, 1.0), [-0.5 * np.log(2 * np.pi),
                                                                               -0.5 * np.log(2 * np.pi) - 0.5],
                 msg="log_gaussian_pdf([0, 1], 0, 1) of the docstring example")


def test_log_gaussian_pdf_far_from_the_mean_stays_finite(nbm):
    got = call(nbm.log_gaussian_pdf, np.array([1000.0, -60.0]), 0.0, 1.0)
    assert_close(got, stats.norm.logpdf([1000.0, -60.0]), rtol=1e-12,
                 msg="log_gaussian_pdf([1000, -60], 0, 1): about -500000 and -1800, never -inf (use the formula of "
                     "the log-density: np.log of the density underflows to log(0) = -inf)")


@pytest.mark.parametrize("var, why", [
    pytest.param(0.0, "var = 0", id="zero"),
    pytest.param(-1.0, "var = -1", id="negative"),
    pytest.param(np.array([1.0, 0.0, 2.0]), "var contains a 0", id="array_with_zero"),
])
def test_log_gaussian_pdf_rejects_a_variance_that_is_not_positive(nbm, var, why):
    assert_raises(ValueError, nbm.log_gaussian_pdf, np.array([0.0, 1.0, 2.0]), 0.0, var, why=why)


def test_log_gaussian_pdf_does_not_modify_its_inputs(nbm):
    x, mean, var = np.array([0.5, 1.0, 3.0]), np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 0.5])
    x0, mean0, var0 = x.copy(), mean.copy(), var.copy()
    call(nbm.log_gaussian_pdf, x, mean, var, copy_inputs=False)
    assert np.array_equal(x, x0) and np.array_equal(mean, mean0) and np.array_equal(var, var0), (
        "expected log_gaussian_pdf to leave x, mean and var unchanged")


# ================================================================== logsumexp (13.28)
@pytest.mark.parametrize("shape, axis", [((5, 4), 0), ((5, 4), 1), ((5, 4), -1), ((3, 4, 5), 0), ((3, 4, 5), 1),
                                         ((3, 4, 5), 2), ((3, 4, 5), -1), ((6,), 0), ((6,), -1)],
                         ids=["2d_axis0", "2d_axis1", "2d_axis-1", "3d_axis0", "3d_axis1", "3d_axis2", "3d_axis-1",
                              "1d_axis0", "1d_axis-1"])
def test_logsumexp_matches_scipy(nbm, shape, axis):
    a = np.random.default_rng(len(shape) * 10 + axis + 3).normal(size=shape) * 5.0
    assert_close(call(nbm.logsumexp, a, axis=axis), special.logsumexp(a, axis=axis), rtol=1e-12, atol=1e-12,
                 msg=f"logsumexp(a of shape {shape}, axis={axis}) = log(sum(exp(a))) along axis {axis} (that axis "
                     "removed)")


def test_logsumexp_default_axis_is_the_last_one(nbm):
    a = np.array([[0.0, 1.0, 2.0], [3.0, -1.0, 0.5]])
    assert_close(call(nbm.logsumexp, a), special.logsumexp(a, axis=-1), rtol=1e-12,
                 msg="logsumexp(a) without axis: one value per ROW (axis=-1 by default)")


def test_logsumexp_docstring_example(nbm):
    assert_close(call(nbm.logsumexp, np.array([-1000.0, -1000.0])), -1000.0 + np.log(2.0), rtol=1e-12,
                 msg="logsumexp([-1000, -1000]) of the docstring example = -1000 + ln 2")
    assert_close(call(nbm.logsumexp, np.array([[0.0, 0.0], [1.0, 2.0]]), axis=1), [np.log(2.0), 2.0 + np.log1p(np.exp(-1.0))],
                 rtol=1e-12, msg="logsumexp([[0, 0], [1, 2]], axis=1) of the docstring example")


@pytest.mark.parametrize("a, what", [
    pytest.param([[-1000.0, -1000.0], [-1000.0, -1001.0], [-1000.5, -1200.0]], "values around -1000 (exp underflows "
                 "to 0, log(0) = -inf: subtract the maximum first)", id="minus_1000"),
    pytest.param([[1000.0, 1000.0], [1000.0, 999.0], [1000.5, 700.0]], "values around +1000 (exp overflows to inf: "
                 "subtract the maximum first)", id="plus_1000"),
    pytest.param([[-1000.0, 0.0, 1000.0], [-5.0, -1000.0, -3.0], [1e3, 1e3 - 1e-9, -1e3]], "values of very "
                 "different sizes in the same row", id="mixed"),
])
def test_logsumexp_never_overflows_or_underflows(nbm, a, what):
    a = np.array(a)
    assert_close(call(nbm.logsumexp, a, axis=1), special.logsumexp(a, axis=1), rtol=1e-12, atol=1e-12,
                 msg=f"logsumexp(a, axis=1) with {what}")


def test_logsumexp_ignores_minus_infinity_among_finite_values(nbm):
    a = np.array([[-np.inf, 0.0, np.log(3.0)], [2.0, -np.inf, -np.inf]])
    assert_close(call(nbm.logsumexp, a, axis=1), [np.log(4.0), 2.0], rtol=1e-12,
                 msg="logsumexp with -inf terms (probability 0: exp(-inf) = 0 adds nothing): rows log(1 + 3) and 2")


def test_logsumexp_of_a_1d_array_is_a_single_value(nbm):
    got = call(nbm.logsumexp, np.array([0.0, np.log(2.0), np.log(5.0)]))
    assert np.ndim(got) == 0, (f"expected logsumexp of a 1-D array to return a single value (a 0-d array or a float), "
                               f"got shape {np.shape(got)} (remove the reduced axis: keepdims only for the maximum)")
    assert_close(got, np.log(8.0), rtol=1e-12, msg="logsumexp([0, ln 2, ln 5]) = ln 8")


def test_logsumexp_does_not_modify_its_input(nbm):
    a = np.array([[1.0, 2.0, 3.0], [-1000.0, -1001.0, -999.0]])
    a0 = a.copy()
    call(nbm.logsumexp, a, axis=1, copy_inputs=False)
    assert np.array_equal(a, a0), "expected logsumexp to leave a unchanged (a - m is a new array; a -= m is not)"


# ================================================================== GaussianNB (13.28)
GNB_CASES = [
    pytest.param(0, 60, 3, (0, 1, 2), 1e-9, None, id="3_classes"),
    pytest.param(1, 40, 2, (5, -1), 1e-9, None, id="labels_5_-1"),
    pytest.param(2, 50, 4, (0, 1, 2), 0.05, None, id="var_smoothing_0.05"),
    pytest.param(3, 45, 3, (0, 1, 2), 1e-9, [0.2, 0.5, 0.3], id="priors"),
    pytest.param(4, 30, 1, ("b", "a"), 0.1, [0.9, 0.1], id="strings_1_feature_priors"),
]


@pytest.mark.parametrize("seed, n, p, labels, var_smoothing, priors", GNB_CASES)
def test_gaussian_nb_matches_sklearn(nbm, seed, n, p, labels, var_smoothing, priors):
    X, y = gaussian_data(seed, n, p, labels)
    X_new = gaussian_data(seed + 50, 15, p, labels)[0]
    model = fit(nbm.GaussianNB(var_smoothing=var_smoothing, priors=priors), X, y)
    oracle = sk(sknb.GaussianNB, var_smoothing=var_smoothing, priors=priors).fit(X, y)
    options = f"GaussianNB(var_smoothing={var_smoothing}, priors={priors})"
    assert_same_labels(learned(model, "classes_"), oracle.classes_, msg=f"{options}: classes_ = sorted distinct labels")
    assert_close(learned(model, "class_count_"), oracle.class_count_, msg=f"{options}: class_count_ = number of "
                 "training samples of each class")
    assert_close(learned(model, "class_prior_"), oracle.class_prior_, msg=f"{options}: class_prior_ = "
                 + ("the given priors" if priors is not None else "class frequencies (class_count_ / n_samples)"))
    assert_close(learned(model, "theta_"), oracle.theta_, rtol=1e-9, atol=1e-12,
                 msg=f"{options}: theta_[c, j] = mean of feature j over the samples of class c")
    assert_close(learned(model, "epsilon_"), oracle.epsilon_, rtol=1e-9, atol=1e-15,
                 msg=f"{options}: epsilon_ = var_smoothing * largest per-feature variance of the WHOLE training set")
    assert_close(learned(model, "var_"), oracle.var_, rtol=1e-9, atol=1e-15,
                 msg=f"{options}: var_[c, j] = POPULATION variance (ddof=0) of feature j in class c, plus epsilon_")
    assert_close(call(model.predict_joint_log_proba, X_new), oracle.predict_joint_log_proba(X_new), rtol=1e-9,
                 atol=1e-9, msg=f"{options}: predict_joint_log_proba(X) = log P(c) + sum over j of log N(x_j; "
                                "theta_cj, var_cj)")
    assert_close(call(model.predict_log_proba, X_new), oracle.predict_log_proba(X_new), rtol=1e-9, atol=1e-9,
                 msg=f"{options}: predict_log_proba(X) = joint log-probabilities minus the logsumexp of their row")
    assert_close(call(model.predict_proba, X_new), oracle.predict_proba(X_new), rtol=1e-8, atol=1e-12,
                 msg=f"{options}: predict_proba(X) = exp(predict_log_proba(X))")
    assert_same_labels(call(model.predict, X_new), oracle.predict(X_new), msg=f"{options}: predict(X) = classes_ of "
                       "the largest joint log-probability (labels, not indices)")


def test_gaussian_nb_docstring_example(nbm):
    X = np.array([[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
    y = np.array([1, 1, 1, 2, 2, 2])
    model = fit(nbm.GaussianNB(), X, y)
    assert_close(learned(model, "theta_"), [[-2.0, -4 / 3], [2.0, 4 / 3]], msg="theta_ of the docstring example")
    assert_same_labels(call(model.predict, np.array([[-0.8, -1.0]])), [1], msg="predict([[-0.8, -1]]) of the "
                       "docstring example")


def test_gaussian_nb_var_is_the_population_variance_plus_epsilon(nbm):
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0], [14.0, 3.0]])
    y = np.array([0, 0, 1, 1])
    model = fit(nbm.GaussianNB(var_smoothing=0.1), X, y)
    epsilon = 0.1 * np.var(X, axis=0).max()
    assert_close(learned(model, "epsilon_"), epsilon, msg="GaussianNB(var_smoothing=0.1): epsilon_ = 0.1 * the "
                 "largest variance of the features over ALL the training samples (np.var(X, axis=0).max() = 31.75)")
    assert_close(learned(model, "theta_"), [[1.0, 0.0], [12.0, 2.0]], msg="theta_ = per-class means")
    assert_close(learned(model, "var_"), np.array([[1.0, 0.0], [4.0, 1.0]]) + epsilon,
                 msg="var_ = per-class POPULATION variances (divide by the class count, ddof=0) plus epsilon_")


def test_gaussian_nb_feature_constant_in_a_class_does_not_divide_by_zero(nbm):
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 7.0], [6.0, 7.0], [5.0, 7.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = fit(nbm.GaussianNB(), X, y)
    oracle = sk(sknb.GaussianNB).fit(X, y)
    assert_close(learned(model, "var_")[:, 1], oracle.var_[:, 1], rtol=1e-9, atol=0.0,
                 msg="feature 1 is constant in each class: its var_ is epsilon_ (> 0), not 0")
    Q = np.array([[2.5, 5.0], [4.5, 6.0]])
    assert_close(call(model.predict_proba, Q), oracle.predict_proba(Q), rtol=1e-8, atol=1e-12,
                 msg="predict_proba with a feature constant in each class (finite, thanks to epsilon_)")


def test_gaussian_nb_predict_log_proba_rows_are_normalised(nbm):
    X, y = gaussian_data(5)
    Q = gaussian_data(55, n=10)[0]
    model = fit(nbm.GaussianNB(), X, y)
    log_proba = np.asarray(call(model.predict_log_proba, Q), dtype=float)
    assert_close(special.logsumexp(log_proba, axis=1), np.zeros(10), atol=1e-9, msg="logsumexp of every row of "
                 "predict_log_proba(X) (normalised log-probabilities: the probabilities of a row sum to 1)")
    assert_close(np.sum(call(model.predict_proba, Q), axis=1), np.ones(10), atol=1e-9, msg="sum of every row of "
                 "predict_proba(X)")


def test_gaussian_nb_far_away_points_do_not_underflow(nbm):
    # feature 1 has the same values in both classes: far along it, both joint log-probabilities are about -1e6
    X = np.array([[-0.2, -1.0], [0.0, 0.0], [0.2, 1.0], [0.8, -1.0], [1.0, 0.0], [1.2, 1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    Q = np.array([[0.45, 1500.0], [0.6, -900.0], [-3.0, 1200.0]])
    model = fit(nbm.GaussianNB(), X, y)
    oracle = sk(sknb.GaussianNB).fit(X, y)
    joint = oracle.predict_joint_log_proba(Q)
    proba = np.asarray(call(model.predict_proba, Q), dtype=float)
    assert_close(proba, oracle.predict_proba(Q), rtol=1e-6, atol=1e-12,
                 msg=f"predict_proba of points far away (joint log-probabilities about {joint[0, 0]:.3g}: exp() of them "
                     "underflows to 0 and 0 / 0 = NaN; normalise with logsumexp, in log space)")
    assert_same_labels(call(model.predict, Q), oracle.predict(Q), msg="predict of the far-away points")


def test_gaussian_nb_uses_the_given_priors(nbm):
    X, y = gaussian_data(6, n=40, p=2, labels=(0, 1))
    model = fit(nbm.GaussianNB(priors=[0.999, 0.001]), X, y)
    assert_close(learned(model, "class_prior_"), [0.999, 0.001], msg="class_prior_ with priors=[0.999, 0.001] (the "
                 "given priors, not the class frequencies)")
    Q = gaussian_data(66, n=10, p=2, labels=(0, 1))[0]
    oracle = sk(sknb.GaussianNB, priors=[0.999, 0.001]).fit(X, y)
    assert_close(call(model.predict_proba, Q), oracle.predict_proba(Q), rtol=1e-8, atol=1e-12,
                 msg="predict_proba with priors=[0.999, 0.001] (log P(c) uses the given priors)")


def test_gaussian_nb_priors_summing_to_1_up_to_1e_6_are_accepted(nbm):
    X, y = gaussian_data(7, n=20, p=2, labels=(0, 1))
    model = fit(nbm.GaussianNB(priors=[0.3, 0.7 + 5e-7]), X, y)
    assert_close(learned(model, "class_prior_"), [0.3, 0.7 + 5e-7], msg="priors=[0.3, 0.7 + 5e-7] sum to 1 within the "
                 "tolerance 1e-6: accepted and kept as given")


@pytest.mark.parametrize("priors, why", [
    pytest.param([0.5, 0.3, 0.2], "3 priors for 2 classes", id="length"),
    pytest.param([1.2, -0.2], "a negative prior (they sum to 1)", id="negative"),
    pytest.param([0.5, 0.51], "priors sum to 1.01", id="sum_above"),
    pytest.param([0.5, 0.4], "priors sum to 0.9", id="sum_below"),
])
def test_gaussian_nb_fit_rejects_bad_priors(nbm, priors, why):
    X, y = gaussian_data(8, n=20, p=2, labels=(0, 1))
    assert_raises(ValueError, build(nbm.GaussianNB, priors=priors).fit, X, y, why=why)


def test_gaussian_nb_score_is_the_accuracy(nbm):
    X, y = gaussian_data(9)
    Q, yq = gaussian_data(59, n=25)
    model = fit(nbm.GaussianNB(), X, y)
    want = float(np.mean(sk(sknb.GaussianNB).fit(X, y).predict(Q) == yq))
    got = call(model.score, Q, yq)
    assert isinstance(got, float), f"expected score to return a float, got {type(got).__name__}"
    assert_close(got, want, msg="score(X, y) = fraction of correct predictions")


def test_gaussian_nb_second_fit_forgets_the_first(nbm):
    model = fit(nbm.GaussianNB(), *gaussian_data(10))
    X2, y2 = gaussian_data(11, n=20, p=2, labels=("x", "y"))
    fit(model, X2, y2)
    oracle = sk(sknb.GaussianNB).fit(X2, y2)
    assert_same_labels(learned(model, "classes_"), ["x", "y"], msg="classes_ after a second fit")
    assert_close(learned(model, "theta_"), oracle.theta_, msg="theta_ after a second fit comes from the second data only")


def test_gaussian_nb_does_not_modify_its_inputs(nbm):
    X, y = gaussian_data(12)
    Q = gaussian_data(62, n=5)[0]
    X0, y0, Q0 = X.copy(), y.copy(), Q.copy()
    model = nbm.GaussianNB()
    call(model.fit, X, y, copy_inputs=False)
    for method in ("predict_joint_log_proba", "predict_log_proba", "predict_proba", "predict"):
        call(getattr(model, method), Q, copy_inputs=False)
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "expected fit to leave X and y unchanged"
    assert np.array_equal(Q, Q0), "expected the prediction methods to leave X unchanged"


def test_gaussian_nb_fit_returns_self(nbm):
    check_returns_self(nbm.GaussianNB(), *gaussian_data(13, n=10))


def test_gaussian_nb_init_only_stores_the_hyperparameters(nbm):
    check_init_then_fit(nbm.GaussianNB(var_smoothing=1e-3, priors=[0.25, 0.75]),
                        {"var_smoothing": 1e-3, "priors": [0.25, 0.75]}, *gaussian_data(14, n=12, labels=(0, 1)))


def test_gaussian_nb_before_fit_raises_runtime_error(nbm):
    check_unfitted(nbm.GaussianNB(), ["predict_joint_log_proba"], gaussian_data(15, n=4)[0])


@pytest.mark.parametrize("X, y, why", [
    pytest.param(np.arange(6.0), np.arange(6) % 2, "X is 1-D (use X.reshape(-1, 1))", id="1d"),
    pytest.param(np.ones((6, 2, 2)), np.arange(6) % 2, "X is 3-D", id="3d"),
    pytest.param(np.arange(12.0).reshape(6, 2), np.arange(5) % 2, "6 rows in X, 5 labels in y", id="length"),
    pytest.param(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.5], [3.0, 1.5], [np.nan, 1.0], [5.0, 2.0]]),
                 np.arange(6) % 2, "X contains NaN", id="nan"),
])
def test_gaussian_nb_fit_rejects_bad_inputs(nbm, X, y, why):
    assert_raises(ValueError, nbm.GaussianNB().fit, X.copy(), y.copy(), why=why)


# ================================================================== MultinomialNB (13.29)
MNB_CASES = [
    pytest.param(20, 1.0, True, ("en", "fr", "de"), id="laplace_3_languages"),
    pytest.param(21, 0.5, False, ("en", "fr", "de"), id="alpha0.5_uniform_prior"),
    pytest.param(22, 0.01, True, (0, 1), id="alpha0.01"),
    pytest.param(23, 2.5, True, (4, 9, 1, 7), id="alpha2.5_4_classes"),
    pytest.param(24, 1.0, False, (1, 0), id="uniform_prior_2_classes"),
]


@pytest.mark.parametrize("seed, alpha, fit_prior, labels", MNB_CASES)
def test_multinomial_nb_matches_sklearn(nbm, seed, alpha, fit_prior, labels):
    X, y = count_data(seed, labels=labels)
    X_new = count_data(seed + 50, n=12, labels=labels)[0]
    model = fit(nbm.MultinomialNB(alpha=alpha, fit_prior=fit_prior), X, y)
    oracle = sk(sknb.MultinomialNB, alpha=alpha, fit_prior=fit_prior, force_alpha=True).fit(X, y)
    options = f"MultinomialNB(alpha={alpha}, fit_prior={fit_prior})"
    assert_same_labels(learned(model, "classes_"), oracle.classes_, msg=f"{options}: classes_ = sorted distinct labels")
    assert_close(learned(model, "class_count_"), oracle.class_count_, msg=f"{options}: class_count_ = number of "
                 "training samples (rows) of each class")
    assert_close(learned(model, "feature_count_"), oracle.feature_count_, msg=f"{options}: feature_count_[c, j] = "
                 "total count of feature j over the rows of class c")
    assert_close(learned(model, "class_log_prior_"), oracle.class_log_prior_, rtol=1e-12, atol=1e-12,
                 msg=f"{options}: class_log_prior_ = " + ("log of the class frequencies" if fit_prior else
                                                          "log(1 / n_classes) for every class (fit_prior=False)"))
    assert_close(learned(model, "feature_log_prob_"), oracle.feature_log_prob_, rtol=1e-12, atol=1e-12,
                 msg=f"{options}: feature_log_prob_[c, j] = log((N_cj + alpha) / (N_c + alpha * n_features))")
    assert_close(call(model.predict_joint_log_proba, X_new), oracle.predict_joint_log_proba(X_new), rtol=1e-10,
                 atol=1e-10, msg=f"{options}: predict_joint_log_proba(X) = X @ feature_log_prob_.T + class_log_prior_")
    assert_close(call(model.predict_log_proba, X_new), oracle.predict_log_proba(X_new), rtol=1e-9, atol=1e-9,
                 msg=f"{options}: predict_log_proba(X) = joint log-probabilities minus the logsumexp of their row")
    assert_close(call(model.predict_proba, X_new), oracle.predict_proba(X_new), rtol=1e-8, atol=1e-12,
                 msg=f"{options}: predict_proba(X) = exp(predict_log_proba(X))")
    assert_same_labels(call(model.predict, X_new), oracle.predict(X_new), msg=f"{options}: predict(X) = classes_ of "
                       "the largest joint log-probability (labels, not indices)")


def test_multinomial_nb_docstring_example(nbm):
    X = np.array([[2, 1, 0], [3, 0, 1], [0, 2, 3], [1, 0, 4]])
    y = np.array(["en", "en", "fr", "fr"])
    model = fit(nbm.MultinomialNB(alpha=1.0), X, y)
    assert_close(np.exp(learned(model, "feature_log_prob_")), [[6 / 10, 2 / 10, 2 / 10], [2 / 13, 3 / 13, 8 / 13]],
                 msg="exp(feature_log_prob_) of the docstring example: en counts 5, 1, 1 -> (5+1, 1+1, 1+1) / (7+3)")
    assert_same_labels(call(model.predict, np.array([[4, 0, 0], [0, 1, 5]])), ["en", "fr"],
                       msg="predict([[4, 0, 0], [0, 1, 5]]) of the docstring example")


def test_multinomial_nb_laplace_smoothing_by_hand(nbm):
    X = np.array([[3, 0, 1], [1, 0, 0], [0, 4, 2]])
    y = np.array([0, 0, 1])
    model = fit(nbm.MultinomialNB(alpha=1.0), X, y)
    # class 0: counts 4, 0, 1 (N_0 = 5); class 1: counts 0, 4, 2 (N_1 = 6); 3 features
    assert_close(learned(model, "feature_log_prob_"), np.log([[5 / 8, 1 / 8, 2 / 8], [1 / 9, 5 / 9, 3 / 9]]),
                 rtol=1e-12, msg="alpha=1: feature_log_prob_ = log((N_cj + 1) / (N_c + 3)); a word never seen in a "
                                 "class gets 1 / (N_c + 3), not 0")
    assert_close(learned(model, "class_log_prior_"), np.log([2 / 3, 1 / 3]), rtol=1e-12,
                 msg="class_log_prior_ = log of the class frequencies (2 rows of class 0, 1 of class 1)")


def test_multinomial_nb_alpha_zero_gives_the_raw_frequencies(nbm):
    X = np.array([[3, 1, 2], [1, 2, 2], [2, 4, 1], [5, 1, 1]])
    y = np.array([1, 1, 2, 2])
    model = fit(nbm.MultinomialNB(alpha=0.0), X, y)
    counts = np.array([[4.0, 3.0, 4.0], [7.0, 5.0, 2.0]])
    assert_close(learned(model, "feature_log_prob_"), np.log(counts / counts.sum(axis=1, keepdims=True)), rtol=1e-12,
                 msg="alpha=0 (allowed, no smoothing): feature_log_prob_ = log(N_cj / N_c)")


def test_multinomial_nb_uniform_prior_when_fit_prior_is_false(nbm):
    X, y = count_data(25, n=30, labels=("a", "b", "c"))
    model = fit(nbm.MultinomialNB(fit_prior=False), X, y)
    assert_close(learned(model, "class_log_prior_"), np.full(3, np.log(1 / 3)), rtol=1e-12,
                 msg="fit_prior=False: class_log_prior_ = log(1/3) for each of the 3 classes, whatever their "
                     "frequencies")


def test_multinomial_nb_long_documents_do_not_underflow(nbm):
    X, y = count_data(26, n=30, d=8, labels=("en", "fr"), length=3000)
    Q = count_data(76, n=8, d=8, labels=("en", "fr"), length=3000)[0]
    model = fit(nbm.MultinomialNB(), X, y)
    oracle = sk(sknb.MultinomialNB, force_alpha=True).fit(X, y)
    joint = oracle.predict_joint_log_proba(Q)
    proba = np.asarray(call(model.predict_proba, Q), dtype=float)
    assert_close(proba, oracle.predict_proba(Q), rtol=1e-6, atol=1e-12,
                 msg=f"predict_proba of documents of 3000 words (joint log-probabilities about {joint[0, 0]:.3g}: "
                     "exp() of them underflows to 0 and 0 / 0 = NaN; normalise with logsumexp, in log space)")


def test_multinomial_nb_score_is_the_accuracy(nbm):
    X, y = count_data(27)
    Q, yq = count_data(77, n=25)
    model = fit(nbm.MultinomialNB(), X, y)
    want = float(np.mean(sk(sknb.MultinomialNB, force_alpha=True).fit(X, y).predict(Q) == yq))
    got = call(model.score, Q, yq)
    assert isinstance(got, float), f"expected score to return a float, got {type(got).__name__}"
    assert_close(got, want, msg="score(X, y) = fraction of correct predictions")


def test_multinomial_nb_second_fit_forgets_the_first(nbm):
    model = fit(nbm.MultinomialNB(), *count_data(28))
    X2, y2 = count_data(29, n=15, d=4, labels=(0, 1))
    fit(model, X2, y2)
    oracle = sk(sknb.MultinomialNB, force_alpha=True).fit(X2, y2)
    assert_close(learned(model, "feature_count_"), oracle.feature_count_, msg="feature_count_ after a second fit comes "
                 "from the second data only")


def test_multinomial_nb_does_not_modify_its_inputs(nbm):
    X, y = count_data(30)
    Q = count_data(80, n=5)[0]
    X0, y0, Q0 = X.copy(), y.copy(), Q.copy()
    model = nbm.MultinomialNB()
    call(model.fit, X, y, copy_inputs=False)
    for method in ("predict_joint_log_proba", "predict_log_proba", "predict_proba", "predict"):
        call(getattr(model, method), Q, copy_inputs=False)
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "expected fit to leave X and y unchanged"
    assert np.array_equal(Q, Q0), "expected the prediction methods to leave X unchanged"


def test_multinomial_nb_fit_returns_self(nbm):
    check_returns_self(nbm.MultinomialNB(), *count_data(31, n=10))


def test_multinomial_nb_init_only_stores_the_hyperparameters(nbm):
    check_init_then_fit(nbm.MultinomialNB(alpha=0.5, fit_prior=False), {"alpha": 0.5, "fit_prior": False},
                        *count_data(32, n=12))


def test_multinomial_nb_before_fit_raises_runtime_error(nbm):
    check_unfitted(nbm.MultinomialNB(), ["predict_joint_log_proba"], count_data(33, n=4)[0])


@pytest.mark.parametrize("X, y, alpha, why", [
    pytest.param(np.array([[1, 0], [2, -1], [0, 3]]), np.array([0, 1, 1]), 1.0, "X contains a negative count",
                 id="negative_count"),
    pytest.param(np.array([[1, 0], [2, 1], [0, 3]]), np.array([0, 1, 1]), -0.5, "alpha = -0.5 < 0", id="alpha"),
    pytest.param(np.array([1, 0, 2]), np.array([0, 1, 1]), 1.0, "X is 1-D (use X.reshape(-1, 1))", id="1d"),
    pytest.param(np.array([[1, 0], [2, 1], [0, 3]]), np.array([0, 1]), 1.0, "3 rows in X, 2 labels in y",
                 id="length"),
    pytest.param(np.array([[1.0, 0.0], [2.0, np.nan], [0.0, 3.0]]), np.array([0, 1, 1]), 1.0, "X contains NaN",
                 id="nan"),
])
def test_multinomial_nb_fit_rejects_bad_inputs(nbm, X, y, alpha, why):
    assert_raises(ValueError, build(nbm.MultinomialNB, alpha=alpha).fit, X.copy(), y.copy(), why=why)
