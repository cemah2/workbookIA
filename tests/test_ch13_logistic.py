"""Tests of mylearn.logistic (chapter 13): oracle tests and properties.

    pytest tests/test_ch13_logistic.py              # your code (mon_travail/mylearn/logistic.py)
    pytest tests/test_ch13_logistic.py --impl=ref   # the reference

Oracles: SciPy (``scipy.special.expit`` for the provided ``sigmoid``), scikit-learn 1.6
(``LogisticRegression(C=1 / (alpha * n_samples))``, ``penalty=None`` when alpha = 0, on
standardised non-separable data: ``coef_``, ``intercept_`` and ``predict_proba`` within
1e-2 after enough gradient steps) and a few gradient steps written with NumPy in this
file (they pin the gradient ``Xᵀ (p - y) / n + alpha * w``, the unpenalised intercept,
the objective ``mean log-loss + (alpha / 2) ||w||²`` and the stopping rule).
Every test name starts with the name of the function or class it tests; exercise 13.32
runs ``-k "test_logistic_regression_"`` (``sigmoid`` is provided: its test is
``test_sigmoid_``).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import numbers
import warnings

import numpy as np
import pytest
from scipy import special
from sklearn import linear_model


@pytest.fixture
def lg(mylearn_module):
    return mylearn_module("logistic")


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
    assert not added, (f"expected fit to add only learned attributes (names ending with _, like coef_) or private "
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


def binary_data(seed, n=200, p=3, labels=(0, 1)):
    """Standardised features, labels drawn from a logistic model (classes overlap: not separable)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 3.0, size=p) + rng.normal(size=p) * 5
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    z = X @ np.array([1.5, -1.0, 0.5, 0.25, -0.75])[:p] + 0.4
    t = (rng.random(n) < special.expit(z)).astype(int)
    t[:2] = [0, 1]                                         # both classes present
    return X, np.asarray(labels)[t]


def numpy_gradient_descent(X, t, alpha, learning_rate, steps, fit_intercept=True):
    """w, b and the objective before training and after each of `steps` gradient steps (labels t coded 0/1)."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0

    def objective(w, b):
        p = special.expit(X @ w + b)
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)) + 0.5 * alpha * (w @ w)

    history = [objective(w, b)]
    for _ in range(steps):
        r = special.expit(X @ w + b) - t
        w, b = w - learning_rate * (X.T @ r / n + alpha * w), (b - learning_rate * r.mean()) if fit_intercept else b
        history.append(objective(w, b))
    return w, b, np.array(history)


def sk_logistic(alpha, n, fit_intercept=True):
    params = {"penalty": None} if alpha == 0 else {"C": 1.0 / (alpha * n)}
    return sk(linear_model.LogisticRegression, tol=1e-12, max_iter=100000, fit_intercept=fit_intercept, **params)


# ================================================================== sigmoid (provided)
def test_sigmoid_matches_scipy_expit(lg):
    z = np.array([[-1000.0, -30.0, -1.0, 0.0], [0.5, 2.0, 40.0, 1000.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # no overflow warning
        got = call(lg.sigmoid, z)
    assert_close(got, special.expit(z), rtol=1e-12, atol=1e-300, msg="sigmoid(z) = 1 / (1 + exp(-z)) "
                 "(provided function: it must not have been changed)")


# ================================================================== LogisticRegression (13.32)
@pytest.mark.parametrize("alpha", [0.0, 0.01, 0.1, 1.0])
@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no_intercept"])
def test_logistic_regression_matches_sklearn(lg, alpha, fit_intercept):
    X, y = binary_data(1)
    X_new = binary_data(51, n=20)[0]
    model = fit(lg.LogisticRegression(alpha=alpha, learning_rate=1.0, max_iter=5000, tol=1e-12,
                                      fit_intercept=fit_intercept), X, y)
    oracle = sk_logistic(alpha, len(X), fit_intercept).fit(X, y)
    options = f"LogisticRegression(alpha={alpha}, learning_rate=1.0, max_iter=5000, tol=1e-12, " \
              f"fit_intercept={fit_intercept})"
    sk_options = "penalty=None" if alpha == 0 else f"C = 1 / (alpha * n_samples) = {1 / (alpha * len(X)):.4g}"
    assert_same_labels(learned(model, "classes_"), [0, 1], msg=f"{options}: classes_ = the two sorted labels")
    assert_close(learned(model, "coef_"), oracle.coef_[0], rtol=0.0, atol=1e-2,
                 msg=f"{options}: coef_ = the weights w of the optimum, as scikit-learn's LogisticRegression("
                     f"{sk_options}) (gradient Xᵀ (p - y) / n + alpha * w)")
    assert_close(learned(model, "intercept_"), oracle.intercept_[0] if fit_intercept else 0.0, rtol=0.0, atol=1e-2,
                 msg=f"{options}: intercept_ = the bias b of the optimum (gradient mean(p - y), never penalised)")
    assert_close(call(model.predict_proba, X_new), oracle.predict_proba(X_new), rtol=0.0, atol=1e-2,
                 msg=f"{options}: predict_proba(new data) = [1 - p, p] with p = sigmoid(X @ coef_ + intercept_)")
    assert_same_labels(call(model.predict, X_new), oracle.predict(X_new), msg=f"{options}: predict(new data)")


@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no_intercept"])
def test_logistic_regression_first_steps_match_a_hand_computation(lg, fit_intercept):
    X, y = binary_data(2, n=40)
    w, b, history = numpy_gradient_descent(X, y, alpha=0.5, learning_rate=0.3, steps=2, fit_intercept=fit_intercept)
    model = fit(lg.LogisticRegression(alpha=0.5, learning_rate=0.3, max_iter=2, tol=0.0, fit_intercept=fit_intercept),
                X, y)
    options = f"LogisticRegression(alpha=0.5, learning_rate=0.3, max_iter=2, tol=0, fit_intercept={fit_intercept})"
    assert_close(learned(model, "coef_"), w, rtol=1e-10, atol=1e-12,
                 msg=f"{options}: coef_ after 2 steps w -= learning_rate * (Xᵀ (p - y) / n + alpha * w)")
    assert_close(learned(model, "intercept_"), b, rtol=1e-10, atol=1e-12,
                 msg=f"{options}: intercept_ after 2 steps " + ("b -= learning_rate * mean(p - y) (no penalty on b)"
                                                                if fit_intercept else "stays 0 (fit_intercept=False)"))
    assert_close(learned(model, "loss_history_"), history, rtol=1e-10, atol=1e-12,
                 msg=f"{options}: loss_history_ = objective before training (ln 2) and after each step, J = mean "
                     "log-loss + (alpha / 2) ||w||² (intercept not penalised)")
    assert learned(model, "n_iter_") == 2, f"{options}: expected n_iter_ = 2 (two gradient steps), got {model.n_iter_!r}"


@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_logistic_regression_loss_history_starts_at_ln_2(lg, alpha):
    X, y = binary_data(3, n=30)
    history = learned(fit(lg.LogisticRegression(alpha=alpha, max_iter=3), X, y), "loss_history_")
    assert_close(history[0], np.log(2.0), rtol=1e-12, msg=f"alpha={alpha}: loss_history_[0] = ln 2 (w = 0 and b = 0: "
                 "every probability is 0.5, and the penalty of w = 0 is 0)")


def test_logistic_regression_loss_history_never_increases_for_a_small_step(lg):
    X, y = binary_data(4)
    history = np.asarray(learned(fit(lg.LogisticRegression(alpha=0.05, learning_rate=0.05, max_iter=300, tol=0.0),
                                     X, y), "loss_history_"), dtype=float)
    rises = np.flatnonzero(np.diff(history) > 1e-12)
    assert rises.size == 0, (f"expected loss_history_ never to increase with learning_rate=0.05, but it rises at step "
                             f"{int(rises[0]) + 1} ({history[rises[0]]:.10g} -> {history[rises[0] + 1]:.10g}): check "
                             "the sign of the step (w -= learning_rate * gradient) and the 1/n of the gradient")
    _, _, want = numpy_gradient_descent(X, y, 0.05, 0.05, 300)
    assert_close(history, want, rtol=1e-9, atol=1e-12, msg="loss_history_ of 300 steps with alpha=0.05, "
                 "learning_rate=0.05 (NumPy gradient descent)")


def test_logistic_regression_runs_max_iter_steps_when_the_loss_keeps_changing(lg):
    X, y = binary_data(5, n=50)
    model = fit(lg.LogisticRegression(max_iter=25, tol=0.0), X, y)
    n_iter = learned(model, "n_iter_")
    assert isinstance(n_iter, numbers.Integral), f"expected n_iter_ to be an int, got {type(n_iter).__name__}"
    assert n_iter == 25, f"max_iter=25, tol=0: expected n_iter_ = 25 gradient steps, got {n_iter!r}"
    history = learned(model, "loss_history_")
    assert isinstance(history, list), f"expected loss_history_ to be a list of floats, got {type(history).__name__}"
    assert len(history) == 26, (f"expected len(loss_history_) = n_iter_ + 1 = 26 (the objective before training, then "
                                f"one value per step), got {len(history)}")


def test_logistic_regression_stops_at_the_first_step_that_changes_the_loss_by_less_than_tol(lg):
    X, y = binary_data(6, n=60)
    _, _, history = numpy_gradient_descent(X, y, alpha=0.05, learning_rate=0.5, steps=30)
    changes = np.abs(np.diff(history))                     # changes[i]: during step i + 1
    tol = float(np.sqrt(changes[5] * changes[6]))          # steps 1-6 change more, step 7 changes less
    assert changes[:6].min() > tol > changes[6], "test data: the 7th step must be the first small one"
    model = fit(lg.LogisticRegression(alpha=0.05, learning_rate=0.5, max_iter=100, tol=tol), X, y)
    options = f"LogisticRegression(alpha=0.05, learning_rate=0.5, tol={tol:.3g})"
    assert learned(model, "n_iter_") == 7, (f"{options}: expected n_iter_ = 7 (step 7 is the first one with |J_new - "
                                            f"J_old| < tol; it is done and counted, then training stops), got "
                                            f"{model.n_iter_!r}")
    assert_close(learned(model, "loss_history_"), history[:8], rtol=1e-10, atol=1e-12,
                 msg=f"{options}: loss_history_ = J before training and after each of the 7 steps (8 values)")


def test_logistic_regression_tol_zero_never_stops_early(lg):
    # nothing to learn: features all 0 and balanced classes, so every step leaves the objective exactly at ln 2
    X = np.zeros((6, 2))
    y = np.array([0, 1, 0, 1, 1, 0])
    model = fit(lg.LogisticRegression(max_iter=5, tol=0.0), X, y)
    assert learned(model, "n_iter_") == 5, (f"tol=0 on data where the objective never changes: expected n_iter_ = "
                                            f"max_iter = 5 (stop only when |J_new - J_old| < tol, strictly: 0 < 0 is "
                                            f"false), got {model.n_iter_!r}")


def test_logistic_regression_large_tol_stops_after_one_step(lg):
    X, y = binary_data(7, n=30)
    model = fit(lg.LogisticRegression(tol=10.0), X, y)
    assert learned(model, "n_iter_") == 1, (f"tol=10: expected n_iter_ = 1 (the first step changes the loss by less "
                                            f"than 10: stop after it), got {model.n_iter_!r}")
    assert len(learned(model, "loss_history_")) == 2, (f"tol=10: expected 2 values in loss_history_ (before training "
                                                       f"and after the step), got {len(model.loss_history_)}")


def test_logistic_regression_learned_attribute_types(lg):
    X, y = binary_data(8, n=40, p=4)
    model = fit(lg.LogisticRegression(max_iter=20), X, y)
    coef = learned(model, "coef_")
    assert isinstance(coef, np.ndarray) and coef.shape == (4,), (
        f"expected coef_ to be an array of shape (n_features,) = (4,) (scikit-learn's is (1, 4)), got "
        f"{type(coef).__name__} of shape {np.shape(coef)}")
    intercept = learned(model, "intercept_")
    assert isinstance(intercept, float), (f"expected intercept_ to be a float, got {type(intercept).__name__} of shape "
                                          f"{np.shape(intercept)} (convert with float(...))")
    history = learned(model, "loss_history_")
    kinds = sorted({type(v).__name__ for v in history} - {"float", "float64"})
    assert isinstance(history, list) and not kinds, (f"expected loss_history_ to be a list of floats, got "
                                                     f"{type(history).__name__} with elements of type "
                                                     f"{sorted({type(v).__name__ for v in history})}")


def test_logistic_regression_string_labels(lg):
    X, t = binary_data(9, n=80)
    y = np.where(t == 1, "spam", "ham")
    model = fit(lg.LogisticRegression(max_iter=50, tol=0.0), X, y)
    coded = fit(lg.LogisticRegression(max_iter=50, tol=0.0), X, t)
    assert_same_labels(learned(model, "classes_"), ["ham", "spam"], msg="classes_ = the two sorted labels")
    assert_close(learned(model, "coef_"), learned(coded, "coef_"), rtol=1e-12, atol=1e-14,
                 msg="coef_ with labels 'ham'/'spam' = coef_ with the same labels coded 0/1 (classes_[1] = 'spam' -> 1)")
    Q = binary_data(59, n=10)[0]
    assert_close(call(model.predict_proba, Q), call(coded.predict_proba, Q), rtol=1e-12, atol=1e-14,
                 msg="predict_proba with string labels: column 1 is P('spam'), the probability of classes_[1]")
    want = np.where(np.asarray(call(coded.predict, Q)) == 1, "spam", "ham")
    assert_same_labels(call(model.predict, Q), want, msg="predict with string labels returns the labels")


@pytest.mark.parametrize("labels", [(-1, 1), (5, 2)], ids=["minus1_1", "5_2"])
def test_logistic_regression_codes_the_labels_through_classes(lg, labels):
    X, t = binary_data(10, n=60)
    y = np.asarray(labels)[t]                              # t = 1 -> labels[1]
    model = fit(lg.LogisticRegression(max_iter=40, tol=0.0), X, y)
    positive = max(labels)
    w, b, _ = numpy_gradient_descent(X, (y == positive).astype(int), 0.0, 0.1, 40)
    assert_close(learned(model, "coef_"), w, rtol=1e-9, atol=1e-12,
                 msg=f"labels {sorted(labels)}: the classes are coded classes_[0] = {min(labels)} -> 0 and "
                     f"classes_[1] = {positive} -> 1 before the gradient steps (never use the labels as numbers)")
    assert_close(learned(model, "intercept_"), b, rtol=1e-9, atol=1e-12, msg=f"intercept_ with labels {sorted(labels)}")


def test_logistic_regression_predict_at_score_zero_is_the_first_class(lg):
    X, t = binary_data(11, n=50)
    model = fit(lg.LogisticRegression(fit_intercept=False, max_iter=30), X, np.where(t == 1, "yes", "no"))
    origin = np.zeros((2, 3))
    assert_close(call(model.decision_function, origin), [0.0, 0.0], msg="decision_function at the origin with "
                 "fit_intercept=False: z = 0")
    assert_close(call(model.predict_proba, origin), [[0.5, 0.5], [0.5, 0.5]], msg="predict_proba where z = 0")
    assert_same_labels(call(model.predict, origin), ["no", "no"], msg="predict where z = 0: classes_[1] only where z "
                       "> 0 (strictly), so classes_[0] = 'no' here")


def test_logistic_regression_scores_and_probabilities(lg):
    X, y = binary_data(12, n=60, p=2)
    model = fit(lg.LogisticRegression(max_iter=30), X, y)
    Q = binary_data(62, n=7, p=2)[0]
    z = Q @ np.asarray(learned(model, "coef_"), dtype=float) + float(learned(model, "intercept_"))
    assert_close(call(model.decision_function, Q), z, rtol=1e-12, atol=1e-14,
                 msg="decision_function(X) = X @ coef_ + intercept_, shape (n_samples,)")
    p = special.expit(z)
    assert_close(call(model.predict_proba, Q), np.column_stack([1 - p, p]), rtol=1e-12, atol=1e-14,
                 msg="predict_proba(X) = [1 - p, p] with p = sigmoid(z), shape (n_samples, 2)")
    assert_same_labels(call(model.predict, Q), np.where(z > 0, 1, 0), msg="predict(X) = classes_[1] where z > 0")


def test_logistic_regression_fit_intercept_false_keeps_the_intercept_at_zero(lg):
    X, y = binary_data(13, n=50)
    model = fit(lg.LogisticRegression(fit_intercept=False, max_iter=200), X, y)
    assert float(learned(model, "intercept_")) == 0.0, (f"fit_intercept=False: expected intercept_ = 0 (b is not "
                                                        f"learnt), got {model.intercept_!r}")


def test_logistic_regression_score_is_the_accuracy(lg):
    X, y = binary_data(14)
    Q, yq = binary_data(64, n=40)
    model = fit(lg.LogisticRegression(learning_rate=1.0, max_iter=3000, tol=1e-12), X, y)
    want = float(np.mean(sk_logistic(0.0, len(X)).fit(X, y).predict(Q) == yq))
    got = call(model.score, Q, yq)
    assert isinstance(got, float), f"expected score to return a float, got {type(got).__name__}"
    assert_close(got, want, msg="score(X, y) = fraction of correct predictions")


def test_logistic_regression_second_fit_forgets_the_first(lg):
    model = fit(lg.LogisticRegression(max_iter=10), *binary_data(15, n=30, p=3))
    X2, y2 = binary_data(16, n=30, p=2, labels=("a", "b"))
    fit(model, X2, y2)
    assert np.shape(learned(model, "coef_")) == (2,), f"expected coef_ of shape (2,) after a second fit on 2 " \
                                                      f"features, got {np.shape(model.coef_)}"
    assert_same_labels(learned(model, "classes_"), ["a", "b"], msg="classes_ after a second fit")
    assert len(learned(model, "loss_history_")) == model.n_iter_ + 1, (
        "expected loss_history_ to restart at each fit (n_iter_ + 1 values), got "
        f"{len(model.loss_history_)} values for n_iter_ = {model.n_iter_}")


def test_logistic_regression_does_not_modify_its_inputs(lg):
    X, y = binary_data(17, n=40, labels=(3, 7))      # labels other than 0/1: recoding y in place would show
    Q = binary_data(67, n=5)[0]
    X0, y0, Q0 = X.copy(), y.copy(), Q.copy()
    model = lg.LogisticRegression(max_iter=20)
    call(model.fit, X, y, copy_inputs=False)
    for method in ("decision_function", "predict_proba", "predict"):
        call(getattr(model, method), Q, copy_inputs=False)
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "expected fit to leave X and y unchanged"
    assert np.array_equal(Q, Q0), "expected decision_function, predict_proba and predict to leave X unchanged"


def test_logistic_regression_fit_returns_self(lg):
    check_returns_self(lg.LogisticRegression(max_iter=5), *binary_data(18, n=20))


def test_logistic_regression_init_only_stores_the_hyperparameters(lg):
    want = {"alpha": 0.1, "learning_rate": 0.5, "max_iter": 7, "tol": 1e-4, "fit_intercept": False}
    check_init_then_fit(lg.LogisticRegression(**want), want, *binary_data(19, n=20))


def test_logistic_regression_before_fit_raises_runtime_error(lg):
    Q = binary_data(20, n=3)[0]
    for method in ("decision_function", "predict_proba", "predict"):
        assert_raises(RuntimeError, getattr(lg.LogisticRegression(), method), Q.copy(),
                      why=f"{method} called before fit (no weights learnt yet)")


@pytest.mark.parametrize("params, why", [
    pytest.param({"learning_rate": 0.0}, "learning_rate = 0 (it must be > 0)", id="lr_zero"),
    pytest.param({"learning_rate": -0.1}, "learning_rate = -0.1", id="lr_negative"),
    pytest.param({"alpha": -1.0}, "alpha = -1 (it must be >= 0)", id="alpha_negative"),
    pytest.param({"max_iter": 0}, "max_iter = 0 (at least one step)", id="max_iter_zero"),
])
def test_logistic_regression_fit_rejects_bad_hyperparameters(lg, params, why):
    assert_raises(ValueError, build(lg.LogisticRegression, **params).fit, *binary_data(21, n=20), why=why)


@pytest.mark.parametrize("X, y, why", [
    pytest.param(np.arange(12.0).reshape(6, 2), np.ones(6, dtype=int), "y has a single class", id="one_class"),
    pytest.param(np.arange(12.0).reshape(6, 2), np.arange(6) % 3, "y has three classes (binary model)",
                 id="three_classes"),
    pytest.param(np.arange(6.0), np.arange(6) % 2, "X is 1-D (use X.reshape(-1, 1))", id="1d"),
    pytest.param(np.arange(12.0).reshape(6, 2), np.arange(5) % 2, "6 rows in X, 5 labels in y", id="length"),
    pytest.param(np.array([[0.0, 1.0], [1.0, 2.0], [np.nan, 0.5], [3.0, 1.5]]), np.array([0, 1, 0, 1]),
                 "X contains NaN", id="nan"),
])
def test_logistic_regression_fit_rejects_bad_inputs(lg, X, y, why):
    assert_raises(ValueError, lg.LogisticRegression().fit, X.copy(), y.copy(), why=why)
