"""Tests of mylearn.ensemble (chapter 14): oracle tests and properties.

    pytest tests/test_ch14_ensemble.py              # your code (mon_travail/mylearn/ensemble.py)
    pytest tests/test_ch14_ensemble.py --impl=ref   # the reference

Oracles: ``scipy.stats.mode`` and a brute-force loop (votes), NumPy (bootstrap frequencies),
scikit-learn 1.6+ (``AdaBoostClassifier``, ``GradientBoostingRegressor``, ``BaggingClassifier``,
``RandomForestClassifier``, ``ExtraTreesClassifier``, ``DecisionTreeClassifier``,
``DecisionTreeRegressor``, ``r2_score``) and properties.
Exact comparisons: AdaBoost and gradient boosting against scikit-learn on continuous
float32-representable data checked to have no tie (the same scikit-learn result for several
``random_state`` values); bagging against scikit-learn trees refitted on your
``estimators_samples_`` with your members' ``random_state`` (a scikit-learn template makes the
members exactly reproducible, whatever your random stream). Statistical comparisons (your random
stream differs from scikit-learn's): random forests and ExtraTrees, accuracy averaged over three
seeds.
In learner mode, your ensembles are built on YOUR ``tree.py`` (chapter 13) and ``model_selection.py``
(chapter 8, ``clone``), or on the reference when you do not have them.
Every test name starts with the name of the function or class it tests, so that each exercise runs
its own group: ``-k "test_plurality_vote_ or test_bootstrap_indices_"`` (14.12),
``"test_bagging_classifier_"`` (14.17), ``"test_random_forest_classifier_ or
test_extra_trees_classifier_"`` (14.18), ``"test_ada_boost_classifier_"`` (14.24) and
``"test_gradient_boosting_regressor_"`` (14.25).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import collections.abc
import inspect
import warnings

import numpy as np
import pytest
import scipy.stats
from sklearn import ensemble as ske
from sklearn import metrics as skm
from sklearn import naive_bayes as sknb
from sklearn import tree as sktree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_moons


@pytest.fixture
def en(mylearn_module):
    return mylearn_module("ensemble")


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
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {_short(want.tolist())}, "
                   f"got {_short(got.tolist())}")
    if got.tolist() != want.tolist():
        bad = next(i for i, (a, b) in enumerate(zip(got.tolist(), want.tolist())) if a != b)
        n_bad = sum(a != b for a, b in zip(got.tolist(), want.tolist()))
        _fail(msg, f"expected {want[bad]!r} at index {bad}, got {got[bad]!r} ({n_bad} of {len(want)} differ)",
              f"expected {_short(want.tolist())}\ngot      {_short(got.tolist())}")


def assert_python_float(value, what):
    if not isinstance(value, float):
        raise AssertionError(f"expected {what} to return a float, got {type(value).__name__} {value!r} "
                             "(float(...) converts a NumPy number or a 0-d array)")


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
        name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error` (or a subclass); `why` names the case in the failure message."""
    name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
    try:
        result = function(*args, **kwargs)
        if inspect.isgenerator(result):          # a generator function only runs at the first next()
            next(result, None)
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
    """A learned attribute (estimators_, classes_...), with a clear message when fit did not create it."""
    if not hasattr(model, name):
        raise AssertionError(f"expected fit to create the attribute {name}, but the fitted "
                             f"{type(model).__name__} has no {name} (attributes: {sorted(vars(model))})")
    return getattr(model, name)


def members(model, n=None):
    """estimators_ as a list (of n members when n is given)."""
    value = learned(model, "estimators_")
    if not isinstance(value, (list, tuple)):
        raise AssertionError(f"expected estimators_ to be a list of fitted members, got {type(value).__name__}")
    if n is not None and len(value) != n:
        raise AssertionError(f"expected {n} members in estimators_ (one per round of the loop), got {len(value)}")
    return list(value)


def staged(model, X, n=None):
    """The list of the arrays yielded by staged_predict (a generator)."""
    result = call(model.staged_predict, X, expected="staged_predict to yield arrays")
    if not isinstance(result, collections.abc.Iterator):
        raise AssertionError(f"expected staged_predict to yield one array per member (a generator function, with "
                             f"yield), got a {type(result).__name__}")
    out = [np.asarray(value) for value in call(list, result, expected="staged_predict to yield arrays")]
    if n is not None and len(out) != n:
        raise AssertionError(f"expected staged_predict to yield {n} arrays (one per member), got {len(out)}")
    return out


def _public(model) -> dict:
    return {key: value for key, value in vars(model).items() if not key.startswith("_") and not key.endswith("_")}


def _stored_as_given(got, want) -> bool:
    """A hyperparameter kept exactly as passed: same type and same value (the same object for an estimator)."""
    if want is None or isinstance(want, (bool, int, float, str)):
        return type(got) is type(want) and got == want
    return got is want


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
                       f"(an attribute like estimators_ = [] makes an unfitted ensemble look fitted)")
    fit(model, X, y)
    added = sorted(set(_public(model)) - set(want))
    assert not added, (f"expected fit to add only learned attributes (names ending with _, like estimators_) or "
                       f"private ones (starting with _), got {added}: clone (ch. 8) would take them for "
                       "hyperparameters")
    changed = {key: vars(model)[key] for key in want if not _stored_as_given(vars(model)[key], want[key])}
    assert not changed, (f"expected fit to leave the hyperparameters unchanged ({want}), got {changed} (write what "
                         "fit computes in a learned attribute, and never fit the template itself: clone it)")


def check_returns_self(model, X, y):
    result = call(model.fit, X, y, expected="fit to return the ensemble itself")
    assert result is model, (f"expected {type(model).__name__}.fit(X, y) to return the ensemble itself (return self), "
                             f"got {type(result).__name__}: {type(model).__name__}().fit(X, y).predict(X) chains")


def check_does_not_modify(model, X, y):
    X0, y0 = X.copy(), y.copy()
    call(model.fit, X, y, expected="fit to succeed", copy_inputs=False)
    assert np.array_equal(X, X0) and np.array_equal(y, y0), (
        f"expected {type(model).__name__}.fit(X, y) to leave X and y unchanged, got modified arrays (work on copies "
        "or on new arrays: X[indices] is already a copy)")
    call(model.predict, X, expected="predict to succeed", copy_inputs=False)
    assert np.array_equal(X, X0), f"expected {type(model).__name__}.predict(X) to leave X unchanged, got a modified X"


def sk(cls, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(*args, **kwargs)


def sk_fit(model, X, y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(X, y, **kwargs)


def _f32(values):
    """Values exactly representable in float32 (scikit-learn converts X to float32)."""
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def classification_data(seed, n, d, n_classes):
    """Continuous float32-representable features; classes = quantile bins of a noisy, slightly curved score."""
    rng = np.random.default_rng(seed)
    X = _f32(rng.normal(size=(n, d)))
    score = X @ rng.normal(size=d) + 0.8 * rng.normal(size=n) + np.sin(3.0 * X[:, 0])
    y = np.digitize(score, np.quantile(score, np.arange(1, n_classes) / n_classes))
    return X, y


def regression_data(seed, n, d):
    rng = np.random.default_rng(seed)
    X = _f32(rng.normal(size=(n, d)))
    y = np.sin(2.0 * X[:, 0]) + X[:, 1 % d] ** 2 + 0.3 * rng.normal(size=n)
    return X, y


def moons(seed, n, noise=0.3):
    X, y = make_moons(n, noise=noise, random_state=seed)
    return _f32(X), y


def penguins():
    import wb

    df = wb.datasets.load_penguins(dropna=True)
    X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].to_numpy(dtype=float)
    y = df["species"].to_numpy()
    rng = np.random.default_rng(14)
    order = rng.permutation(len(y))
    return X[order], y[order]


def int_seed(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def tree_class(en, name):
    """The DecisionTreeClassifier / DecisionTreeRegressor that ensemble.py imports from .tree."""
    cls = getattr(en, name, None)
    if cls is None:
        raise AssertionError(f"expected ensemble.py to import {name} from .tree (from .tree import {name}), got no "
                             f"{name} in the module")
    return cls


def accuracy_mean(make, X_train, y_train, X_test, y_test, seeds=(0, 1, 2)):
    scores = []
    for seed in seeds:
        model = make(seed)
        fit(model, X_train, y_train)
        scores.append(np.mean(np.asarray(call(model.predict, X_test)) == y_test))
    return float(np.mean(scores))


# ================================================================== plurality_vote (14.12)
def _brute_force_vote(predictions, weights):
    """The book's rule written as a loop: every label of predictions starts at 0, each voter adds its weight."""
    predictions = np.asarray(predictions)
    labels = sorted(set(predictions.ravel().tolist()))
    out = []
    for j in range(predictions.shape[1]):
        score = {label: 0.0 for label in labels}
        for v in range(predictions.shape[0]):
            score[predictions[v, j].item()] += weights[v]
        best = max(score.values())
        out.append(min(label for label in labels if score[label] == best))
    return np.array(out)


def test_plurality_vote_docstring_examples(en):
    predictions = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 1]])
    assert_same_labels(call(en.plurality_vote, predictions), [0, 0, 1], msg="plurality_vote(predictions)")
    assert_same_labels(call(en.plurality_vote, predictions, weights=[1.0, 1.0, 3.0]), [1, 0, 1],
                       msg="plurality_vote(predictions, weights=[1, 1, 3]) (the third voter outweighs the two others)")


@pytest.mark.parametrize("n_voters, n_samples, n_labels, seed",
                         [(3, 40, 2, 0), (5, 60, 3, 1), (4, 80, 3, 2), (10, 50, 5, 3), (1, 20, 4, 4)])
def test_plurality_vote_unweighted_matches_scipy_mode(en, n_voters, n_samples, n_labels, seed):
    predictions = np.random.default_rng(seed).integers(0, n_labels, size=(n_voters, n_samples))
    want = scipy.stats.mode(predictions, axis=0).mode               # ties: the smallest label, as scipy
    assert_same_labels(call(en.plurality_vote, predictions), want,
                       msg=f"plurality_vote of {n_voters} voters (as scipy.stats.mode(predictions, axis=0); ties: the "
                           "smallest label)")


@pytest.mark.parametrize("n_voters, n_samples, n_labels, seed", [(3, 40, 2, 10), (5, 60, 3, 11), (7, 50, 4, 12)])
def test_plurality_vote_weighted_matches_brute_force(en, n_voters, n_samples, n_labels, seed):
    rng = np.random.default_rng(seed)
    predictions = rng.integers(0, n_labels, size=(n_voters, n_samples))
    weights = rng.integers(-8, 17, size=n_voters) / 4.0             # exact sums, some negative weights
    assert_same_labels(call(en.plurality_vote, predictions, weights=weights), _brute_force_vote(predictions, weights),
                       msg=f"plurality_vote with weights {weights.tolist()} (each label gets the sum of the weights of "
                           "its voters; the largest total wins, ties: the smallest label)")


def test_plurality_vote_book_ballots(en):
    # the book's example: a ballot of +4 for A then one of -1.5 for A, and one of +2 for B: A has 2.5, B has 2
    predictions = np.array([["A"], ["A"], ["B"]])
    got = call(en.plurality_vote, predictions, weights=[4.0, -1.5, 2.0])
    assert_same_labels(got, ["A"], msg="plurality_vote of the ballots A (+4), A (-1.5), B (+2): A totals 2.5, B 2")
    got = call(en.plurality_vote, predictions, weights=[4.0, -3.0, 2.0])
    assert_same_labels(got, ["B"], msg="plurality_vote of the ballots A (+4), A (-3), B (+2): A totals 1, B 2")


def test_plurality_vote_a_label_without_votes_starts_at_zero(en):
    # sample 1: both voters say 1, with negative weights, so label 1 totals -3 and label 0 (predicted elsewhere) keeps 0
    predictions = np.array([[0, 1], [0, 1]])
    got = call(en.plurality_vote, predictions, weights=[-1.0, -2.0])
    assert_same_labels(got, [1, 0], msg="plurality_vote([[0, 1], [0, 1]], weights=[-1, -2]): every label of "
                                        "predictions starts with a score of 0, so for sample 1 label 0 (score 0) beats "
                                        "label 1 (score -3); for sample 0, label 1 (score 0) beats label 0 (score -3)")


def test_plurality_vote_ties_give_the_smallest_label(en):
    predictions = np.array([[2, 0, 5], [1, 3, 4]])
    assert_same_labels(call(en.plurality_vote, predictions), [1, 0, 4],
                       msg="plurality_vote([[2, 0, 5], [1, 3, 4]]): one vote each, the smallest label wins")
    got = call(en.plurality_vote, np.array([[1, 1], [0, 0]]), weights=[2.5, 2.5])
    assert_same_labels(got, [0, 0], msg="plurality_vote of two voters of equal weight 2.5: the smallest label wins")


def test_plurality_vote_string_labels(en):
    predictions = np.array([["chat", "chien", "chat"], ["chien", "chien", "oiseau"], ["chat", "oiseau", "oiseau"]])
    assert_same_labels(call(en.plurality_vote, predictions), ["chat", "chien", "oiseau"],
                       msg="plurality_vote of string labels")


def test_plurality_vote_returns_one_label_per_sample(en):
    predictions = np.random.default_rng(5).integers(0, 3, size=(4, 17))
    got = np.asarray(call(en.plurality_vote, predictions))
    assert got.shape == (17,), (f"expected an array of shape (17,) for predictions of shape (4, 17) (one label per "
                                f"sample: votes are counted down each column), got shape {got.shape}")


@pytest.mark.parametrize("predictions, weights, why", [
    (np.array([0, 1, 1]), None, "1-D predictions (a single voter is a row: shape (1, n_samples))"),
    (np.zeros((2, 3, 1), dtype=int), None, "3-D predictions"),
    (np.array([[0, 1], [1, 1]]), [1.0, 2.0, 3.0], "3 weights for 2 voters"),
    (np.array([[0, 1], [1, 1]]), [1.0], "1 weight for 2 voters"),
    (np.array([[0, 1], [1, 1]]), [1.0, np.nan], "a NaN weight"),
], ids=["1d", "3d", "too_many_weights", "too_few_weights", "nan_weight"])
def test_plurality_vote_rejects_bad_inputs(en, predictions, weights, why):
    assert_raises(ValueError, en.plurality_vote, predictions, weights, why=why)


def test_plurality_vote_does_not_modify_its_inputs(en):
    predictions = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 1]])
    weights = np.array([1.0, 1.0, 3.0])
    p0, w0 = predictions.copy(), weights.copy()
    call(en.plurality_vote, predictions, weights, copy_inputs=False)
    assert np.array_equal(predictions, p0) and np.array_equal(weights, w0), (
        "expected plurality_vote to leave predictions and weights unchanged, got modified arrays")


# ================================================================== bootstrap_indices (14.12)
def _indices(en, *args, **kwargs):
    got = call(en.bootstrap_indices, *args, **kwargs)
    if not isinstance(got, np.ndarray):
        raise AssertionError(f"expected bootstrap_indices to return a NumPy array, got {type(got).__name__}")
    return got


@pytest.mark.parametrize("n_samples, n_draws", [(5, 5), (8, 4), (10, 30), (1, 3), (100, 1)])
def test_bootstrap_indices_shape_range_and_dtype(en, n_samples, n_draws):
    got = _indices(en, n_samples, n_draws, rng=np.random.default_rng(0))
    assert got.shape == (n_draws,), f"expected shape ({n_draws},) for n_draws={n_draws}, got {got.shape}"
    assert got.dtype.kind in "iu", (f"expected integer indices (usable as X[indices]), got dtype {got.dtype} "
                                    "(rng.integers draws integers)")
    assert got.min() >= 0 and got.max() < n_samples, (
        f"expected indices in [0, {n_samples}) for n_samples={n_samples}, got values from {got.min()} to {got.max()}")


def test_bootstrap_indices_default_size_is_n_samples(en):
    got = _indices(en, 12, rng=np.random.default_rng(1))
    assert got.shape == (12,), f"expected n_draws=None to draw n_samples = 12 indices, got shape {got.shape}"


def test_bootstrap_indices_docstring_example(en):
    idx = _indices(en, 5, rng=np.random.default_rng(0))
    assert idx.shape == (5,) and idx.min() >= 0 and idx.max() < 5, f"expected 5 indices in [0, 5), got {idx}"


def test_bootstrap_indices_same_generator_same_indices(en):
    a = _indices(en, 50, 30, rng=np.random.default_rng(7))
    b = _indices(en, 50, 30, rng=np.random.default_rng(7))
    assert np.array_equal(a, b), ("expected the same indices from two generators np.random.default_rng(7), got "
                                  f"{_short(a)} and {_short(b)} (draw with the generator rng, never with np.random)")


def test_bootstrap_indices_accepts_an_int_seed(en):
    a = _indices(en, 50, 30, rng=7)
    b = _indices(en, 50, 30, rng=7)
    assert np.array_equal(a, b), ("expected the same indices for rng=7 twice (np.random.default_rng(rng) accepts None, "
                                  "an int or a Generator)")


def test_bootstrap_indices_uses_the_given_generator(en):
    rng = np.random.default_rng(3)
    a = _indices(en, 1000, 20, rng=rng)
    b = _indices(en, 1000, 20, rng=rng)
    assert not np.array_equal(a, b), (
        "expected two successive calls with the same generator to give different indices (the generator moves on), "
        "got twice the same: do not create a new generator from a fixed seed inside the function")
    c = _indices(en, 1000, 20, rng=np.random.default_rng(4))
    assert not np.array_equal(a, c), ("expected different generators (seeds 3 and 4) to give different indices, got "
                                      "the same: draw with rng")


def test_bootstrap_indices_draws_with_replacement(en):
    got = _indices(en, 100, 100, rng=np.random.default_rng(11))
    assert len(np.unique(got)) < 100, (
        "expected repeated indices in a bootstrap sample of 100 among 100 (with replacement: about 63 distinct), got "
        "100 distinct indices (a permutation draws WITHOUT replacement)")
    got = _indices(en, 3, 50, rng=np.random.default_rng(12))
    assert got.shape == (50,), "expected 50 draws among 3 indices to be possible (with replacement)"


def test_bootstrap_indices_fraction_of_distinct_indices(en):
    n = 50
    rng = np.random.default_rng(13)
    fractions = [len(np.unique(_indices(en, n, rng=rng))) / n for _ in range(400)]
    want = 1 - (1 - 1 / n) ** n
    assert abs(np.mean(fractions) - want) < 0.01, (
        f"expected a mean fraction of distinct indices close to 1 - (1 - 1/n)^n = {want:.4f} for n = {n} (400 "
        f"samples), got {np.mean(fractions):.4f} (draw n indices uniformly in 0..n-1, with replacement)")


def test_bootstrap_indices_are_uniform(en):
    counts = np.bincount(_indices(en, 10, 40_000, rng=np.random.default_rng(14)), minlength=10)
    assert counts.min() > 3_700 and counts.max() < 4_300, (
        f"expected each of the 10 indices about 4000 times in 40000 draws (uniform), got counts {counts.tolist()} "
        "(rng.integers(0, n_samples, size=n_draws): the upper bound is excluded)")


@pytest.mark.parametrize("args, why", [((0,), "n_samples=0"), ((-3,), "n_samples=-3"), ((5, 0), "n_draws=0"),
                                       ((5, -2), "n_draws=-2")], ids=["n0", "n_neg", "draws0", "draws_neg"])
def test_bootstrap_indices_rejects_bad_sizes(en, args, why):
    assert_raises(ValueError, en.bootstrap_indices, *args, rng=np.random.default_rng(0), why=why)


# ================================================================== BaggingClassifier (14.17)
def _template_tree(**params):
    """A scikit-learn tree as the template: its members are exactly reproducible from their random_state."""
    return sktree.DecisionTreeClassifier(**params)


def _samples(model, n):
    value = learned(model, "estimators_samples_")
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise AssertionError(f"expected estimators_samples_ to be a list of {n} index arrays (one per member), got "
                             f"{type(value).__name__}" + (f" of length {len(value)}" if hasattr(value, "__len__")
                                                          else ""))
    return [np.asarray(v) for v in value]


def _refit_probas(model, X_train, y_train, X, classes, **params):
    """Each member refitted by scikit-learn on its own sample, with its own random_state; probabilities on classes."""
    out = []
    for member, rows in zip(members(model), _samples(model, len(members(model)))):
        seed = getattr(member, "random_state", None)
        oracle = sk_fit(_template_tree(random_state=seed, **params), X_train[rows], y_train[rows])
        proba = np.zeros((len(X), len(classes)))
        proba[:, np.searchsorted(classes, oracle.classes_)] = oracle.predict_proba(X)
        out.append(proba)
    return out


def test_bagging_classifier_docstring_example(en):
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bag = fit(en.BaggingClassifier(n_estimators=5, random_state=0), X, y)
    assert len(members(bag)) == 5, f"expected 5 members, got {len(bag.estimators_)}"
    assert_same_labels(call(bag.predict, np.array([[1.5], [11.5]])), [0, 1], msg="predict([[1.5], [11.5]])")


def test_bagging_classifier_fit_returns_self(en):
    check_returns_self(en.BaggingClassifier(n_estimators=3, random_state=0), *moons(1, 60))


def test_bagging_classifier_init_only_stores_the_hyperparameters(en):
    template = _template_tree(max_depth=2)
    params = {"estimator": template, "n_estimators": 4, "max_samples": 0.5, "bootstrap": True, "oob_score": True,
              "random_state": 3}
    check_init_then_fit(en.BaggingClassifier(**params), params, *moons(2, 80))


@pytest.mark.parametrize("max_samples, size", [(1.0, 90), (0.5, 45), (0.33, 29), (20, 20), (1, 1), (90, 90)])
def test_bagging_classifier_sample_sizes(en, max_samples, size):
    X, y = moons(3, 90)
    model = fit(en.BaggingClassifier(_template_tree(), n_estimators=4, max_samples=max_samples, random_state=0), X, y)
    for k, rows in enumerate(_samples(model, 4)):
        assert rows.shape == (size,), (
            f"expected every member's sample to have {size} indices with max_samples={max_samples!r} and n = 90 (an "
            f"int is a number of samples, a float f gives int(f * n)), got {rows.shape} for member {k}")
        assert rows.dtype.kind in "iu" and rows.min() >= 0 and rows.max() < 90, (
            f"expected integer indices in [0, 90) in estimators_samples_[{k}], got {_short(rows)}")


def test_bagging_classifier_bootstrap_draws_with_replacement(en):
    X, y = moons(4, 200)
    model = fit(en.BaggingClassifier(_template_tree(max_depth=2), n_estimators=6, random_state=1), X, y)
    for k, rows in enumerate(_samples(model, 6)):
        distinct = len(np.unique(rows)) / 200
        assert 0.5 < distinct < 0.75, (
            f"expected about 63 % of distinct indices in a bootstrap sample of 200 among 200, got {distinct:.0%} for "
            f"member {k} (bootstrap=True: bootstrap_indices, with replacement)")


def test_bagging_classifier_without_bootstrap_draws_distinct_indices(en):
    X, y = moons(5, 100)
    model = fit(en.BaggingClassifier(_template_tree(max_depth=2), n_estimators=5, max_samples=0.6, bootstrap=False,
                                     random_state=1), X, y)
    for k, rows in enumerate(_samples(model, 5)):
        assert len(np.unique(rows)) == 60, (
            f"expected 60 DISTINCT indices with bootstrap=False and max_samples=0.6 (without replacement), got "
            f"{len(np.unique(rows))} distinct among {len(rows)} for member {k}")
    full = fit(en.BaggingClassifier(_template_tree(max_depth=2), n_estimators=3, bootstrap=False, random_state=1), X, y)
    for k, rows in enumerate(_samples(full, 3)):
        assert sorted(rows.tolist()) == list(range(100)), (
            f"expected every index once with bootstrap=False and max_samples=1.0 (the whole training set), got "
            f"{len(np.unique(rows))} distinct indices for member {k}")


def test_bagging_classifier_members_get_different_samples(en):
    X, y = moons(6, 80)
    model = fit(en.BaggingClassifier(_template_tree(max_depth=2), n_estimators=5, random_state=2), X, y)
    rows = [tuple(r.tolist()) for r in _samples(model, 5)]
    assert len(set(rows)) == 5, ("expected 5 different bootstrap samples (one draw per member, with the ensemble "
                                 "generator created ONCE in fit), got repeated samples")


def test_bagging_classifier_members_are_fresh_clones(en):
    X, y = moons(7, 60)
    template = _template_tree(max_depth=3, min_samples_leaf=2)
    model = fit(en.BaggingClassifier(template, n_estimators=4, random_state=0), X, y)
    assert not hasattr(template, "tree_"), ("expected the template to stay unfitted (fit clone(template), never the "
                                            "template itself), got a fitted template")
    group = members(model, 4)
    assert len({id(m) for m in group}) == 4 and all(m is not template for m in group), (
        "expected 4 distinct member objects, none of them the template (clone(template) for every member)")
    for k, member in enumerate(group):
        assert type(member) is type(template) and member.max_depth == 3 and member.min_samples_leaf == 2, (
            f"expected member {k} to be a {type(template).__name__} with the template's hyperparameters (max_depth=3, "
            f"min_samples_leaf=2), got {type(member).__name__} {getattr(member, 'max_depth', None)}, "
            f"{getattr(member, 'min_samples_leaf', None)}")
        assert hasattr(member, "tree_"), f"expected member {k} to be fitted, got an unfitted member"


def test_bagging_classifier_default_member_is_your_decision_tree(en):
    X, y = moons(8, 60)
    model = fit(en.BaggingClassifier(n_estimators=3, random_state=0), X, y)
    cls = tree_class(en, "DecisionTreeClassifier")
    for k, member in enumerate(members(model, 3)):
        assert isinstance(member, cls), (f"expected member {k} to be a DecisionTreeClassifier of tree.py (estimator=None), "
                                         f"got {type(member).__module__}.{type(member).__name__}")
        assert member.max_depth is None and member.max_features is None, (
            f"expected member {k} to be DecisionTreeClassifier() with its defaults, got max_depth={member.max_depth!r}, "
            f"max_features={member.max_features!r}")


def test_bagging_classifier_seeds_the_members(en):
    X, y = moons(9, 60)
    model = fit(en.BaggingClassifier(_template_tree(max_features=1), n_estimators=6, random_state=4), X, y)
    seeds = [m.random_state for m in members(model, 6)]
    assert all(int_seed(s) for s in seeds), (
        f"expected every member's random_state to be an int drawn from the ensemble generator (the template has a "
        f"random_state attribute), got {seeds}")
    assert len(set(seeds)) == 6, f"expected 6 different seeds for 6 members, got {seeds}"


def test_bagging_classifier_member_without_random_state(en):
    X, y = moons(10, 80)
    template = sknb.GaussianNB()
    model = fit(en.BaggingClassifier(template, n_estimators=4, random_state=0), X, y)
    want = np.mean([sk_fit(sknb.GaussianNB(), X[r], y[r]).predict_proba(X) for r in _samples(model, 4)], axis=0)
    assert_close(call(model.predict_proba, X), want, rtol=1e-12, atol=1e-12,
                 msg="predict_proba of a bagging of GaussianNB() (a template without random_state: no seed to set; the "
                     "mean of the members' predict_proba)")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bagging_classifier_predict_proba_is_the_mean_of_the_members(en, seed):
    X, y = classification_data(20 + seed, 120, 3, 3)
    model = fit(en.BaggingClassifier(_template_tree(max_depth=3), n_estimators=7, random_state=seed), X, y)
    want = np.mean(_refit_probas(model, X, y, X, np.unique(y), max_depth=3), axis=0)
    got = call(model.predict_proba, X)
    assert_close(got, want, rtol=1e-12, atol=1e-12,
                 msg="predict_proba (the mean of the members' predict_proba, each member refitted by scikit-learn on "
                     "its estimators_samples_ with its random_state)")
    assert_close(np.sum(got, axis=1), np.ones(len(X)), rtol=0, atol=1e-9, msg="sum of every row of predict_proba")
    assert_same_labels(call(model.predict, X), np.unique(y)[np.argmax(want, axis=1)],
                       msg="predict (classes_[argmax of predict_proba])")


def test_bagging_classifier_aligns_members_that_miss_a_class(en):
    rng = np.random.default_rng(31)
    X = _f32(rng.normal(size=(60, 2)))
    y = np.array([0] * 4 + [1] * 30 + [2] * 26)                     # class 0 is rare: small samples often miss it
    X[y == 0] += 3.0
    model = fit(en.BaggingClassifier(_template_tree(max_depth=2), n_estimators=12, max_samples=10, random_state=5),
                X, y)
    missing = [k for k, r in enumerate(_samples(model, 12)) if 0 not in set(y[r].tolist())]
    assert missing, "test setup: no member missed class 0"
    assert_same_labels(learned(model, "classes_"), [0, 1, 2], msg="classes_ (all the labels of y, sorted)")
    want = np.mean(_refit_probas(model, X, y, X, np.array([0, 1, 2]), max_depth=2), axis=0)
    got = call(model.predict_proba, X)
    assert_close(got, want, rtol=1e-12, atol=1e-12,
                 msg=f"predict_proba when members {missing} never saw class 0 (place each member's columns on classes_ "
                     "with its own classes_: a member without class 0 has two columns, for the classes 1 and 2)")


def test_bagging_classifier_string_labels(en):
    X, y = moons(11, 80)
    labels = np.array(["rond", "carré"])[y]
    model = fit(en.BaggingClassifier(_template_tree(max_depth=3), n_estimators=5, random_state=0), X, labels)
    assert_same_labels(learned(model, "classes_"), ["carré", "rond"], msg="classes_ (sorted labels)")
    want = np.array(["carré", "rond"])[np.argmax(np.mean(_refit_probas(model, X, labels, X, np.array(["carré", "rond"]),
                                                                       max_depth=3), axis=0), axis=1)]
    assert_same_labels(call(model.predict, X), want, msg="predict with string labels")


@pytest.mark.parametrize("seed", [0, 1])
def test_bagging_classifier_oob_score_is_the_out_of_bag_accuracy(en, seed):
    X, y = classification_data(40 + seed, 100, 3, 2)
    model = fit(en.BaggingClassifier(_template_tree(max_depth=3), n_estimators=8, oob_score=True, random_state=seed),
                X, y)
    classes = np.unique(y)
    probas = _refit_probas(model, X, y, X, classes, max_depth=3)
    total, count = np.zeros((len(X), len(classes))), np.zeros(len(X))
    for proba, rows in zip(probas, _samples(model, 8)):
        out = ~np.isin(np.arange(len(X)), rows)
        total[out] += proba[out]
        count[out] += 1
    kept = count > 0
    want = float(np.mean(classes[np.argmax(total[kept] / count[kept, None], axis=1)] == y[kept]))
    got = learned(model, "oob_score_")
    assert_python_float(got, "oob_score_")
    assert_close(got, want, rtol=1e-12, atol=1e-12,
                 msg=f"oob_score_ ({kept.sum()} samples have out-of-bag members: each is predicted by the mean "
                     "predict_proba of the members that did NOT see it; samples seen by every member are left out)")


def test_bagging_classifier_no_oob_score_by_default(en):
    model = fit(en.BaggingClassifier(_template_tree(max_depth=2), n_estimators=3, random_state=0), *moons(12, 40))
    assert not hasattr(model, "oob_score_"), "expected no oob_score_ attribute with oob_score=False (the default)"


def test_bagging_classifier_random_state_controls_the_ensemble(en):
    X, y = moons(13, 80)
    a = fit(en.BaggingClassifier(_template_tree(max_features=1), n_estimators=5, random_state=3), X, y)
    b = fit(en.BaggingClassifier(_template_tree(max_features=1), n_estimators=5, random_state=3), X, y)
    c = fit(en.BaggingClassifier(_template_tree(max_features=1), n_estimators=5, random_state=4), X, y)
    same = all(np.array_equal(r, s) for r, s in zip(_samples(a, 5), _samples(b, 5)))
    assert same and np.array_equal(call(a.predict_proba, X), call(b.predict_proba, X)), (
        "expected the same samples and the same predict_proba with random_state=3 twice (create the generator "
        "np.random.default_rng(random_state) in fit and draw everything from it)")
    assert not all(np.array_equal(r, s) for r, s in zip(_samples(a, 5), _samples(c, 5))), (
        "expected other samples with random_state=4 than with random_state=3, got the same")


def test_bagging_classifier_second_fit_of_the_same_object_gives_the_same_ensemble(en):
    X, y = moons(14, 80)
    model = en.BaggingClassifier(_template_tree(max_depth=3), n_estimators=4, random_state=8)
    first = [r.copy() for r in _samples(fit(model, X, y), 4)]
    second = _samples(fit(model, X, y), 4)
    assert all(np.array_equal(r, s) for r, s in zip(first, second)), (
        "expected the same samples when the same object is fitted twice with random_state=8 (create the generator in "
        "fit, not in __init__)")


def test_bagging_classifier_score(en):
    X, y = moons(15, 100)
    X_test, y_test = moons(16, 100)
    model = fit(en.BaggingClassifier(_template_tree(max_depth=3), n_estimators=5, random_state=0), X, y)
    got = call(model.score, X_test, y_test)
    assert_python_float(got, "score")
    want = np.mean(np.unique(y)[np.argmax(np.mean(_refit_probas(model, X, y, X_test, np.unique(y), max_depth=3),
                                                  axis=0), axis=1)] == y_test)
    assert_close(got, want, rtol=1e-12, msg="score(X_test, y_test) (accuracy of predict)")


def test_bagging_classifier_close_to_sklearn(en):
    X, y = moons(17, 300)
    X_test, y_test = moons(18, 600)
    mine = accuracy_mean(lambda s: en.BaggingClassifier(n_estimators=10, random_state=s), X, y, X_test, y_test)
    theirs = np.mean([sk_fit(sk(ske.BaggingClassifier, sktree.DecisionTreeClassifier(), n_estimators=10,
                                random_state=s), X, y).score(X_test, y_test) for s in range(3)])
    assert abs(mine - theirs) < 0.03, (
        f"expected a test accuracy within 0.03 of scikit-learn's BaggingClassifier(DecisionTreeClassifier(), "
        f"n_estimators=10) ({theirs:.3f}, mean of 3 seeds), got {mine:.3f}")
    model = fit(en.BaggingClassifier(n_estimators=25, oob_score=True, random_state=0), X, y)
    oob = sk_fit(sk(ske.BaggingClassifier, sktree.DecisionTreeClassifier(), n_estimators=25, oob_score=True,
                    random_state=0), X, y).oob_score_
    assert abs(learned(model, "oob_score_") - oob) < 0.05, (
        f"expected oob_score_ within 0.05 of scikit-learn's ({oob:.3f}, 25 trees), got {model.oob_score_:.3f}")


@pytest.mark.parametrize("params, why", [
    ({"n_estimators": 0}, "n_estimators=0"),
    ({"n_estimators": 2.5}, "n_estimators=2.5 (not an int)"),
    ({"max_samples": 0}, "max_samples=0"),
    ({"max_samples": 0.0}, "max_samples=0.0"),
    ({"max_samples": 1.5}, "max_samples=1.5 (a float must be in (0, 1])"),
    ({"max_samples": 41}, "max_samples=41 with 40 samples"),
    ({"bootstrap": False, "oob_score": True, "max_samples": 0.5}, "oob_score=True with bootstrap=False"),
], ids=["n0", "n_float", "ms0", "ms0f", "ms_float_big", "ms_int_big", "oob_without_bootstrap"])
def test_bagging_classifier_fit_rejects_bad_hyperparameters(en, params, why):
    model = build(en.BaggingClassifier, estimator=_template_tree(max_depth=2), **params)
    assert_raises(ValueError, model.fit, *moons(19, 40), why=why)


def test_bagging_classifier_fit_rejects_bad_data(en):
    X, y = moons(20, 40)
    model = en.BaggingClassifier(_template_tree(max_depth=2), n_estimators=2)
    assert_raises(ValueError, model.fit, X, y[:-1], why="39 labels for 40 samples")
    X_nan = X.copy()
    X_nan[3, 1] = np.nan
    assert_raises(ValueError, model.fit, X_nan, y, why="X contains NaN (as scikit-learn)")


def test_bagging_classifier_not_fitted(en):
    model = en.BaggingClassifier()
    X, _ = moons(21, 10)
    assert_raises(RuntimeError, model.predict_proba, X, why="predict_proba before fit")
    assert_raises(RuntimeError, model.predict, X, why="predict before fit")


def test_bagging_classifier_does_not_modify_its_inputs(en):
    X, y = moons(22, 60)
    check_does_not_modify(en.BaggingClassifier(_template_tree(max_depth=2), n_estimators=3, random_state=0), X, y)


# ================================================================== RandomForest and ExtraTrees (14.18)
FORESTS = ["RandomForestClassifier", "ExtraTreesClassifier"]
SK_FORESTS = {"RandomForestClassifier": ske.RandomForestClassifier, "ExtraTreesClassifier": ske.ExtraTreesClassifier}


def _forest(en, name):
    return getattr(en, name)


def _sk_tree_like_root(tree):
    return learned(tree, "root_")


def _trees(model, n, en):
    cls = tree_class(en, "DecisionTreeClassifier")
    out = members(model, n)
    for k, t in enumerate(out):
        assert isinstance(t, cls), (f"expected tree {k} of estimators_ to be a DecisionTreeClassifier of tree.py, got "
                                    f"{type(t).__module__}.{type(t).__name__}")
    return out


def test_random_forest_classifier_docstring_example(en):
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    forest = fit(en.RandomForestClassifier(n_estimators=5, random_state=0), X, y)
    assert_same_labels(call(forest.predict, np.array([[1.5], [11.5]])), [0, 1], msg="predict([[1.5], [11.5]])")
    assert_close(learned(forest, "feature_importances_"), [1.0], rtol=1e-12,
                 msg="feature_importances_ with a single feature (trees whose bootstrap sample has one class only are "
                     "leaves without importances: average the trees that split, then renormalise to sum 1, as "
                     "scikit-learn)")


def test_extra_trees_classifier_docstring_example(en):
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    extra = fit(en.ExtraTreesClassifier(n_estimators=5, random_state=0), X, y)
    assert_same_labels(call(extra.predict, np.array([[1.5], [11.5]])), [0, 1], msg="predict([[1.5], [11.5]])")


def test_random_forest_classifier_fit_returns_self(en):
    check_returns_self(en.RandomForestClassifier(n_estimators=3, random_state=0), *moons(30, 60))


def test_extra_trees_classifier_fit_returns_self(en):
    check_returns_self(en.ExtraTreesClassifier(n_estimators=3, random_state=0), *moons(30, 60))


def test_random_forest_classifier_init_only_stores_the_hyperparameters(en):
    params = {"n_estimators": 4, "max_depth": 3, "min_samples_split": 4, "min_samples_leaf": 2, "max_features": 1,
              "bootstrap": True, "oob_score": True, "random_state": 7}
    check_init_then_fit(en.RandomForestClassifier(**params), params, *moons(31, 80))


def test_extra_trees_classifier_init_only_stores_the_hyperparameters(en):
    params = {"n_estimators": 4, "max_depth": 3, "min_samples_split": 4, "min_samples_leaf": 2, "max_features": 1,
              "bootstrap": False, "random_state": 7}
    check_init_then_fit(en.ExtraTreesClassifier(**params), params, *moons(31, 80))


def _check_tree_hyperparameters(model, en, splitter):
    for k, t in enumerate(_trees(model, 5, en)):
        got = (t.max_depth, t.min_samples_split, t.min_samples_leaf, t.max_features, t.splitter)
        want = (3, 5, 2, 2, splitter)
        assert got == want, (
            f"expected tree {k} to have (max_depth, min_samples_split, min_samples_leaf, max_features, splitter) = "
            f"{want}, the forest's hyperparameters, got {got}")
        assert int_seed(t.random_state), (f"expected tree {k} to have an int random_state drawn from the forest's "
                                          f"generator, got {t.random_state!r}")
    seeds = [t.random_state for t in members(model)]
    assert len(set(seeds)) == 5, f"expected 5 different seeds for the 5 trees, got {seeds}"


def test_random_forest_classifier_trees_get_the_hyperparameters(en):
    X, y = classification_data(32, 120, 4, 3)
    model = fit(en.RandomForestClassifier(n_estimators=5, max_depth=3, min_samples_split=5, min_samples_leaf=2,
                                          max_features=2, random_state=0), X, y)
    _check_tree_hyperparameters(model, en, "best")


def test_extra_trees_classifier_trees_get_the_hyperparameters(en):
    X, y = classification_data(32, 120, 4, 3)
    model = fit(en.ExtraTreesClassifier(n_estimators=5, max_depth=3, min_samples_split=5, min_samples_leaf=2,
                                        max_features=2, random_state=0), X, y)
    _check_tree_hyperparameters(model, en, "random")


def test_random_forest_classifier_without_randomness_is_one_tree(en):
    X, y = classification_data(33, 150, 4, 3)
    X_test, _ = classification_data(34, 200, 4, 3)
    oracle = sk_fit(sk(sktree.DecisionTreeClassifier, max_depth=4, min_samples_leaf=5, random_state=0), X, y)
    for seed in range(1, 6):
        other = sk_fit(sk(sktree.DecisionTreeClassifier, max_depth=4, min_samples_leaf=5, random_state=seed), X, y)
        assert np.array_equal(other.tree_.threshold, oracle.tree_.threshold), "test setup: tie between features"
    model = fit(en.RandomForestClassifier(n_estimators=4, max_depth=4, min_samples_leaf=5, max_features=None,
                                          bootstrap=False, random_state=0), X, y)
    assert_close(call(model.predict_proba, X_test), oracle.predict_proba(X_test), rtol=1e-12, atol=1e-12,
                 msg="predict_proba with max_features=None and bootstrap=False (every tree sees all the samples and "
                     "all the features: each one is scikit-learn's DecisionTreeClassifier(max_depth=4, "
                     "min_samples_leaf=5))")
    assert_close(learned(model, "feature_importances_"), oracle.feature_importances_, rtol=1e-9, atol=1e-12,
                 msg="feature_importances_ with max_features=None and bootstrap=False (those of the single tree)")


def test_random_forest_classifier_bootstrap_changes_the_trees_samples(en):
    X, y = classification_data(35, 90, 3, 3)
    proportions = np.bincount(y) / len(y)
    boot = fit(en.RandomForestClassifier(n_estimators=6, max_depth=2, random_state=0), X, y)
    roots = [np.asarray(_sk_tree_like_root(t).value, dtype=float) for t in _trees(boot, 6, en)]
    assert any(not np.allclose(r, proportions) for r in roots), (
        "expected some trees of a forest with bootstrap=True to see other class proportions at their root than the "
        "whole training set, got the proportions of the whole set in every root (train each tree on X[indices], with "
        "bootstrap_indices)")
    full = fit(en.RandomForestClassifier(n_estimators=4, max_depth=2, bootstrap=False, random_state=0), X, y)
    for k, t in enumerate(_trees(full, 4, en)):
        assert_close(_sk_tree_like_root(t).value, proportions, rtol=1e-12,
                     msg=f"root value of tree {k} with bootstrap=False (the class proportions of the whole training set)")


def test_extra_trees_classifier_sees_the_whole_training_set(en):
    X, y = classification_data(36, 90, 3, 3)
    proportions = np.bincount(y) / len(y)
    model = fit(en.ExtraTreesClassifier(n_estimators=4, max_depth=2, random_state=0), X, y)
    for k, t in enumerate(_trees(model, 4, en)):
        root = _sk_tree_like_root(t)
        assert root.n_samples == 90, f"expected tree {k} to see the 90 training samples, got {root.n_samples}"
        assert_close(root.value, proportions, rtol=1e-12,
                     msg=f"root value of tree {k} (bootstrap=False by default: the whole training set)")


def test_random_forest_classifier_feature_bagging_decorrelates_the_trees(en):
    X, y = classification_data(37, 120, 6, 2)
    model = fit(en.RandomForestClassifier(n_estimators=6, max_depth=3, max_features=1, bootstrap=False,
                                          random_state=1), X, y)
    roots = {(_sk_tree_like_root(t).feature, float(_sk_tree_like_root(t).threshold)) for t in _trees(model, 6, en)}
    assert len(roots) > 1, ("expected different root splits with max_features=1 and bootstrap=False (a new random "
                            "feature at every node, a different seed per tree), got the same root in all 6 trees")


def test_extra_trees_classifier_thresholds_are_random(en):
    X, y = classification_data(38, 80, 1, 2)
    values = np.unique(X[:, 0])
    midpoints = (values[:-1] + values[1:]) / 2
    model = fit(en.ExtraTreesClassifier(n_estimators=5, max_depth=1, max_features=None, random_state=2), X, y)
    thresholds = [float(_sk_tree_like_root(t).threshold) for t in _trees(model, 5, en)]
    assert len(set(thresholds)) == 5 and not any(np.isclose(midpoints, t, rtol=0, atol=1e-12).any()
                                                for t in thresholds), (
        f"expected 5 different random root thresholds (splitter='random'), none of them a midpoint between two "
        f"values, got {thresholds}")


@pytest.mark.parametrize("name", FORESTS)
def test_random_forest_classifier_and_extra_trees_predict_proba(en, name):
    X, y = classification_data(39, 100, 3, 3)
    model = fit(_forest(en, name)(n_estimators=5, max_depth=3, random_state=0), X, y)
    proba = np.asarray(call(model.predict_proba, X), dtype=float)
    assert proba.shape == (100, 3), f"expected predict_proba of shape (100, 3), got {proba.shape}"
    assert_close(proba.sum(axis=1), np.ones(100), rtol=0, atol=1e-9, msg=f"{name}: sum of every row of predict_proba")
    assert_same_labels(call(model.predict, X), learned(model, "classes_")[np.argmax(proba, axis=1)],
                       msg=f"{name}: predict (classes_[argmax of predict_proba])")
    assert_python_float(call(model.score, X, y), f"{name}.score")


@pytest.mark.parametrize("name", FORESTS)
def test_random_forest_classifier_and_extra_trees_feature_importances(en, name):
    X, y = classification_data(40, 150, 4, 2)
    model = fit(_forest(en, name)(n_estimators=10, max_depth=4, random_state=0), X, y)
    imp = np.asarray(learned(model, "feature_importances_"), dtype=float)
    assert imp.shape == (4,), f"expected feature_importances_ of shape (4,), got {imp.shape}"
    assert_close(imp.sum(), 1.0, rtol=1e-9, msg=f"{name}: sum of feature_importances_")
    assert (imp >= 0).all(), f"expected non-negative importances, got {imp}"


@pytest.mark.parametrize("name", FORESTS)
def test_random_forest_classifier_and_extra_trees_importances_skip_the_trees_that_never_split(en, name):
    X = _f32(np.array([[0.1, 2.0], [0.4, 1.0], [0.7, 3.0], [0.9, 0.5], [1.3, 2.5], [1.6, 1.5]]))
    y = np.array([0, 0, 0, 0, 0, 1])                        # small bootstrap samples often miss the only class-1 sample
    for seed in range(20):
        model = fit(_forest(en, name)(n_estimators=6, bootstrap=True, random_state=seed), X, y)
        trees = _trees(model, 6, en)
        imps = [np.asarray(t.feature_importances_, dtype=float) for t in trees]
        split = [imp for imp in imps if imp.sum() > 0]
        if 0 < len(split) < len(imps):
            want = np.mean(split, axis=0) / np.mean(split, axis=0).sum()
            assert_close(learned(model, "feature_importances_"), want, rtol=1e-9, atol=1e-12,
                         msg=f"{name}.feature_importances_ when {len(imps) - len(split)} of the 6 trees are a single leaf "
                             "(a bootstrap sample of one class): average the importances of the trees that split, then "
                             "renormalise to sum 1, as scikit-learn")
            return
    raise AssertionError("test setup: no forest mixed trees that split and single-leaf trees; choose other data")


@pytest.mark.parametrize("name", FORESTS)
def test_random_forest_classifier_and_extra_trees_random_state(en, name):
    X, y = moons(41, 80)
    a = fit(_forest(en, name)(n_estimators=5, random_state=3), X, y)
    b = fit(_forest(en, name)(n_estimators=5, random_state=3), X, y)
    c = fit(_forest(en, name)(n_estimators=5, random_state=4), X, y)
    grid = _f32(np.random.default_rng(0).uniform(-1.5, 2.5, size=(300, 2)))
    assert np.array_equal(call(a.predict_proba, grid), call(b.predict_proba, grid)), (
        f"expected the same {name} with random_state=3 twice (one generator np.random.default_rng(random_state) "
        "created in fit)")
    assert not np.array_equal(call(a.predict_proba, grid), call(c.predict_proba, grid)), (
        f"expected another {name} with random_state=4 than with random_state=3, got the same predict_proba")


@pytest.mark.parametrize("name", FORESTS)
@pytest.mark.parametrize("dataset", ["moons", "penguins"])
def test_random_forest_classifier_and_extra_trees_close_to_sklearn(en, name, dataset):
    if dataset == "moons":
        X, y = moons(42, 300)
        X_test, y_test = moons(43, 600)
    else:
        X_all, y_all = penguins()
        X, y, X_test, y_test = X_all[:200], y_all[:200], X_all[200:], y_all[200:]
    mine = accuracy_mean(lambda s: _forest(en, name)(n_estimators=25, random_state=s), X, y, X_test, y_test)
    theirs = np.mean([sk_fit(sk(SK_FORESTS[name], n_estimators=25, random_state=s), X, y).score(X_test, y_test)
                      for s in range(3)])
    assert abs(mine - theirs) < 0.03, (
        f"expected a test accuracy within 0.03 of scikit-learn's {name}(n_estimators=25) on {dataset} "
        f"({theirs:.3f}, mean of 3 seeds), got {mine:.3f}")


def test_random_forest_classifier_top_importances_like_sklearn(en):
    rng = np.random.default_rng(44)
    X = _f32(rng.normal(size=(300, 5)))
    y = (2.0 * X[:, 2] + 1.0 * X[:, 0] + 0.3 * X[:, 4] + 0.5 * rng.normal(size=300) > 0).astype(int)
    model = fit(en.RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0), X, y)
    oracle = sk_fit(sk(ske.RandomForestClassifier, n_estimators=100, max_depth=5, random_state=0), X, y)
    got = np.argsort(-np.asarray(learned(model, "feature_importances_")))[:2].tolist()
    want = np.argsort(-oracle.feature_importances_)[:2].tolist()
    assert got == want, (f"expected the two most important features {want} (as scikit-learn's forest), got {got} "
                         f"(importances {np.round(model.feature_importances_, 3).tolist()})")


def test_random_forest_classifier_oob_score_close_to_sklearn(en):
    X, y = moons(45, 300)
    model = fit(en.RandomForestClassifier(n_estimators=30, oob_score=True, random_state=0), X, y)
    oob = sk_fit(sk(ske.RandomForestClassifier, n_estimators=30, oob_score=True, random_state=0), X, y).oob_score_
    got = learned(model, "oob_score_")
    assert_python_float(got, "oob_score_")
    assert abs(got - oob) < 0.05, f"expected oob_score_ within 0.05 of scikit-learn's ({oob:.3f}), got {got:.3f}"
    plain = fit(en.RandomForestClassifier(n_estimators=3, random_state=0), X, y)
    assert not hasattr(plain, "oob_score_"), "expected no oob_score_ with oob_score=False (the default)"


@pytest.mark.parametrize("name", FORESTS)
@pytest.mark.parametrize("params, why", [({"n_estimators": 0}, "n_estimators=0"),
                                         ({"max_features": 0}, "max_features=0"),
                                         ({"max_features": "auto"}, "max_features='auto' (removed in scikit-learn 1.3)"),
                                         ({"max_depth": 0}, "max_depth=0")],
                         ids=["n0", "mf0", "mf_auto", "depth0"])
def test_random_forest_classifier_and_extra_trees_reject_bad_hyperparameters(en, name, params, why):
    model = build(_forest(en, name), **params)
    assert_raises(ValueError, model.fit, *moons(46, 40), why=why)


@pytest.mark.parametrize("name", FORESTS)
def test_random_forest_classifier_and_extra_trees_reject_nan(en, name):
    X, y = moons(50, 40)
    X[::2, 0] = np.nan                                       # many rows: every bootstrap sample sees a NaN
    assert_raises(ValueError, _forest(en, name)(n_estimators=2).fit, X, y, why="X contains NaN (as scikit-learn)")


def test_random_forest_classifier_oob_without_bootstrap_is_rejected(en):
    model = build(en.RandomForestClassifier, bootstrap=False, oob_score=True, n_estimators=3)  # forest: max_samples=1
    assert_raises(ValueError, model.fit, *moons(47, 40), why="oob_score=True with bootstrap=False")


@pytest.mark.parametrize("name", FORESTS)
def test_random_forest_classifier_and_extra_trees_not_fitted(en, name):
    model = _forest(en, name)()
    X, _ = moons(48, 10)
    assert_raises(RuntimeError, model.predict_proba, X, why=f"{name}.predict_proba before fit")
    assert_raises(RuntimeError, model.predict, X, why=f"{name}.predict before fit")


@pytest.mark.parametrize("name", FORESTS)
def test_random_forest_classifier_and_extra_trees_do_not_modify_their_inputs(en, name):
    X, y = moons(49, 60)
    check_does_not_modify(_forest(en, name)(n_estimators=3, random_state=0), X, y)


# ================================================================== AdaBoostClassifier (14.24)
def _ada_oracle(X, y, n_estimators, learning_rate, estimator=None):
    """scikit-learn's AdaBoost with stumps, checked to be the same for five random_state values (no tie)."""
    estimator = sktree.DecisionTreeClassifier(max_depth=1) if estimator is None else estimator
    runs = [sk_fit(sk(ske.AdaBoostClassifier, estimator, n_estimators=n_estimators, learning_rate=learning_rate,
                      random_state=seed), X, y) for seed in range(5)]
    for other in runs[1:]:
        if not np.array_equal(other.estimator_weights_, runs[0].estimator_weights_):
            raise AssertionError("test setup: tie between two stumps (scikit-learn breaks it at random); choose "
                                 "other test data")
    return runs[0]


ADA_CASES = [(50, 150, 3, 2, 30, 1.0), (51, 150, 4, 3, 30, 1.0), (52, 120, 3, 2, 20, 0.5), (53, 160, 4, 4, 25, 0.7)]
ADA_IDS = ["binary", "3_classes", "binary_lr0.5", "4_classes_lr0.7"]


def test_ada_boost_classifier_docstring_example(en):
    X = np.arange(1.0, 8.0).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 0, 1])
    ada = fit(en.AdaBoostClassifier(n_estimators=3), X, y)
    assert_close(learned(ada, "estimator_errors_"), [1 / 7, 1 / 6, 1 / 5], rtol=1e-9,
                 msg="estimator_errors_ (weighted error of each stump, the weights normalised to sum 1)")
    assert_close(learned(ada, "estimator_weights_"), [np.log(6), np.log(5), np.log(4)], rtol=1e-9,
                 msg="estimator_weights_ (alpha = ln((1 - err) / err) + ln(K - 1), with K = 2)")
    stages = staged(ada, X, 3)
    for m, (got, want) in enumerate(zip(stages, [[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1],
                                                 [0, 0, 0, 1, 1, 0, 1]])):
        assert_same_labels(got, want, msg=f"staged_predict after {m + 1} stump(s)")


def test_ada_boost_classifier_fit_returns_self(en):
    check_returns_self(en.AdaBoostClassifier(n_estimators=3), *classification_data(54, 40, 2, 2))


def test_ada_boost_classifier_init_only_stores_the_hyperparameters(en):
    template = sktree.DecisionTreeClassifier(max_depth=1)
    params = {"estimator": template, "n_estimators": 4, "learning_rate": 0.5, "random_state": 2}
    check_init_then_fit(en.AdaBoostClassifier(**params), params, *classification_data(55, 40, 2, 2))


@pytest.mark.parametrize("seed, n, d, K, M, lr", ADA_CASES, ids=ADA_IDS)
def test_ada_boost_classifier_weights_and_errors_match_sklearn(en, seed, n, d, K, M, lr):
    X, y = classification_data(seed, n, d, K)
    oracle = _ada_oracle(X, y, M, lr)
    m = len(oracle.estimators_)
    model = fit(en.AdaBoostClassifier(n_estimators=M, learning_rate=lr), X, y)
    errors = np.asarray(learned(model, "estimator_errors_"), dtype=float)
    weights = np.asarray(learned(model, "estimator_weights_"), dtype=float)
    assert errors.shape == (m,) and weights.shape == (m,), (
        f"expected {m} kept members, so estimator_errors_ and estimator_weights_ of shape ({m},), got {errors.shape} "
        f"and {weights.shape}")
    assert_close(weights[:1], oracle.estimator_weights_[:1], rtol=1e-7, atol=1e-12,
                 msg=f"estimator_weights_[0] with K = {K}, learning_rate = {lr} (alpha = learning_rate * (ln((1 - err) / "
                     "err) + ln(K - 1)), the first err computed with uniform weights 1/n)")
    assert_close(errors, oracle.estimator_errors_[:m], rtol=1e-7, atol=1e-12,
                 msg=f"estimator_errors_ with K = {K} (as scikit-learn; first error: the weighted error of the first "
                     "stump with uniform weights 1/n)")
    assert_close(weights, oracle.estimator_weights_[:m], rtol=1e-7, atol=1e-12,
                 msg=f"estimator_weights_ with K = {K}, learning_rate = {lr} (alpha = learning_rate * (ln((1 - err) / "
                     "err) + ln(K - 1)))")


@pytest.mark.parametrize("seed, n, d, K, M, lr", ADA_CASES, ids=ADA_IDS)
def test_ada_boost_classifier_predict_and_staged_predict_match_sklearn(en, seed, n, d, K, M, lr):
    X, y = classification_data(seed, n, d, K)
    X_test, _ = classification_data(seed + 100, 200, d, K)
    oracle = _ada_oracle(X, y, M, lr)
    model = fit(en.AdaBoostClassifier(n_estimators=M, learning_rate=lr), X, y)
    assert_same_labels(call(model.predict, X_test), oracle.predict(X_test), msg=f"predict with K = {K} (as scikit-learn)")
    want = list(oracle.staged_predict(X_test))
    got = staged(model, X_test, len(want))
    for m, (g, w) in enumerate(zip(got, want)):
        assert_same_labels(g, w, msg=f"staged_predict after {m + 1} member(s) (as scikit-learn)")


@pytest.mark.parametrize("seed, n, d, K, M, lr", ADA_CASES[:2], ids=ADA_IDS[:2])
def test_ada_boost_classifier_decision_function(en, seed, n, d, K, M, lr):
    X, y = classification_data(seed, n, d, K)
    oracle = _ada_oracle(X, y, M, lr)
    m = len(oracle.estimators_)
    alphas = oracle.estimator_weights_[:m]
    classes = oracle.classes_
    want = sum(a * (stump.predict(X)[:, None] == classes[None, :]) for stump, a in zip(oracle.estimators_, alphas))
    want = want / alphas.sum()
    model = fit(en.AdaBoostClassifier(n_estimators=M, learning_rate=lr), X, y)
    got = call(model.decision_function, X)
    assert_close(got, want, rtol=1e-9, atol=1e-12,
                 msg=f"decision_function with K = {K} (shape (n_samples, K): sum of the alpha of the members that vote "
                     "for each class, divided by the sum of all the alpha)")


def test_ada_boost_classifier_default_member_is_your_stump(en):
    X, y = classification_data(56, 80, 2, 2)
    model = fit(en.AdaBoostClassifier(n_estimators=4), X, y)
    cls = tree_class(en, "DecisionTreeClassifier")
    for k, member in enumerate(members(model)):
        assert isinstance(member, cls) and member.max_depth == 1, (
            f"expected member {k} to be DecisionTreeClassifier(max_depth=1) of tree.py (a stump, estimator=None), got "
            f"{type(member).__name__} with max_depth={getattr(member, 'max_depth', None)!r}")


def test_ada_boost_classifier_template_stays_unfitted(en):
    X, y = classification_data(57, 80, 2, 2)
    template = sktree.DecisionTreeClassifier(max_depth=1)
    model = fit(en.AdaBoostClassifier(template, n_estimators=5), X, y)
    assert not hasattr(template, "tree_"), "expected the template to stay unfitted (clone it for every round)"
    group = members(model)
    assert len({id(m) for m in group}) == len(group) and all(m is not template for m in group), (
        "expected one new member object per round, none of them the template")


def test_ada_boost_classifier_stops_when_a_member_is_perfect(en):
    X = _f32(np.array([[0.3], [1.1], [1.7], [2.9], [3.4], [4.8]]))
    y = np.array([0, 0, 0, 1, 1, 1])
    model = fit(en.AdaBoostClassifier(n_estimators=10), X, y)
    assert len(members(model)) == 1, (
        f"expected 1 member: the first stump makes no error (err = 0), so the loop keeps it and stops, got "
        f"{len(model.estimators_)} members")
    assert_close(learned(model, "estimator_weights_"), [1.0], rtol=0, msg="estimator_weights_ after a perfect member "
                                                                          "(weight 1.0, ln((1 - 0) / 0) is infinite)")
    assert_close(learned(model, "estimator_errors_"), [0.0], rtol=0, atol=0, msg="estimator_errors_ after a perfect "
                                                                                 "member")
    assert_same_labels(call(model.predict, X), y, msg="predict after a perfect member")


class _WorseThanChanceOnceWeighted(ClassifierMixin, BaseEstimator):
    """A deliberately bad member (to test the stop rule): with uniform weights, the sign of feature 0; with any other
    weights, the class of SMALLEST total weight everywhere, whose weighted error is at least 1 - 1/K."""

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        totals = np.array([w[y == c].sum() for c in self.classes_])
        self.constant_ = None if np.allclose(w, w[0]) else self.classes_[np.argmin(totals)]
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.constant_ is not None:
            return np.full(len(X), self.constant_)
        return self.classes_[(X[:, 0] > 0).astype(int)]


@pytest.mark.parametrize("K", [2, 3])
def test_ada_boost_classifier_discards_a_member_no_better_than_chance(en, K):
    rng = np.random.default_rng(58 + K)
    X = _f32(rng.normal(size=(60, 2)))
    y = (X[:, 0] > 0).astype(int)
    flip = rng.choice(60, size=12, replace=False)
    y[flip] = 1 - y[flip]
    if K == 3:
        y[rng.choice(60, size=6, replace=False)] = 2
    oracle = _ada_oracle(X, y, 10, 1.0, estimator=_WorseThanChanceOnceWeighted())
    model = fit(en.AdaBoostClassifier(_WorseThanChanceOnceWeighted(), n_estimators=10), X, y)
    assert len(oracle.estimators_) == 1, "test setup: scikit-learn kept more than one member"
    assert len(members(model)) == 1, (
        f"expected 1 kept member (as scikit-learn): the second member has a weighted error above 1 - 1/K = "
        f"{1 - 1 / K:.4g}, so it is discarded and the loop stops, got {len(model.estimators_)} members")
    assert_close(learned(model, "estimator_weights_"), oracle.estimator_weights_[:1], rtol=1e-9,
                 msg="estimator_weights_ of the kept member")
    assert_close(learned(model, "estimator_errors_"), oracle.estimator_errors_[:1], rtol=1e-9,
                 msg="estimator_errors_ of the kept member")


def test_ada_boost_classifier_first_member_no_better_than_chance(en):
    X = _f32(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    y = np.array([0, 1, 1, 0])                                       # an "exclusive or": no stump beats chance
    assert_raises(ValueError, en.AdaBoostClassifier(n_estimators=5).fit, X, y,
                  why="the first stump has a weighted error of 0.5 = 1 - 1/K: the ensemble would be empty")


class _NoSampleWeight(ClassifierMixin, BaseEstimator):
    """A member whose fit has no sample_weight parameter: AdaBoost cannot use it."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_ada_boost_classifier_requires_sample_weight(en):
    X, y = classification_data(59, 40, 2, 2)
    assert_raises(ValueError, en.AdaBoostClassifier(_NoSampleWeight(), n_estimators=3).fit, X, y,
                  why="the template's fit(X, y) has no sample_weight parameter (inspect.signature(template.fit)"
                      ".parameters)")


@pytest.mark.parametrize("params, why", [({"n_estimators": 0}, "n_estimators=0"),
                                         ({"learning_rate": 0.0}, "learning_rate=0"),
                                         ({"learning_rate": -0.5}, "learning_rate=-0.5")],
                         ids=["n0", "lr0", "lr_neg"])
def test_ada_boost_classifier_fit_rejects_bad_hyperparameters(en, params, why):
    model = build(en.AdaBoostClassifier, **params)
    assert_raises(ValueError, model.fit, *classification_data(60, 40, 2, 2), why=why)


def test_ada_boost_classifier_fit_rejects_bad_data(en):
    X, y = classification_data(61, 40, 2, 2)
    assert_raises(ValueError, en.AdaBoostClassifier(n_estimators=3).fit, X, np.zeros(40, dtype=int),
                  why="y has a single class")
    assert_raises(ValueError, en.AdaBoostClassifier(n_estimators=3).fit, X, y[:-1], why="39 labels for 40 samples")


def test_ada_boost_classifier_rejects_nan(en):
    X, y = classification_data(68, 40, 2, 2)
    X[3, 1] = np.nan
    assert_raises(ValueError, en.AdaBoostClassifier(n_estimators=2).fit, X, y, why="X contains NaN (as scikit-learn)")


class _LoudNoSampleWeight(ClassifierMixin, BaseEstimator):
    """A member without sample_weight whose fit must never be called: AdaBoost checks the signature first."""

    def fit(self, X, y):
        raise AssertionError("expected AdaBoostClassifier.fit to check inspect.signature(template.fit) for a "
                             "sample_weight parameter BEFORE fitting any member, but it called the template's fit")

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_ada_boost_classifier_checks_sample_weight_before_fitting(en):
    X, y = classification_data(69, 30, 2, 2)
    assert_raises(ValueError, en.AdaBoostClassifier(_LoudNoSampleWeight(), n_estimators=3).fit, X, y,
                  why="the template's fit(X, y) has no sample_weight parameter: check the signature first")


def test_ada_boost_classifier_string_labels(en):
    X, y = classification_data(62, 120, 3, 3)
    names = np.array(["Adelie", "Chinstrap", "Gentoo"])
    oracle = _ada_oracle(X, names[y], 15, 1.0)
    model = fit(en.AdaBoostClassifier(n_estimators=15), X, names[y])
    assert_same_labels(learned(model, "classes_"), names, msg="classes_ (sorted labels)")
    assert_same_labels(call(model.predict, X), oracle.predict(X), msg="predict with string labels (as scikit-learn)")


def test_ada_boost_classifier_seeds_the_members(en):
    X, y = classification_data(63, 80, 3, 2)
    model = fit(en.AdaBoostClassifier(n_estimators=5, random_state=1), X, y)
    seeds = [m.random_state for m in members(model)]
    assert all(int_seed(s) for s in seeds), (f"expected every stump's random_state to be an int drawn from the "
                                             f"generator np.random.default_rng(random_state), got {seeds}")


def test_ada_boost_classifier_staged_predict_ends_with_predict(en):
    X, y = classification_data(64, 100, 3, 3)
    model = fit(en.AdaBoostClassifier(n_estimators=8), X, y)
    stages = staged(model, X, len(members(model)))
    assert_same_labels(stages[-1], call(model.predict, X), msg="the last array of staged_predict (all the members)")


def test_ada_boost_classifier_score(en):
    X, y = classification_data(65, 120, 3, 2)
    oracle = _ada_oracle(X, y, 10, 1.0)
    model = fit(en.AdaBoostClassifier(n_estimators=10), X, y)
    got = call(model.score, X, y)
    assert_python_float(got, "score")
    assert_close(got, oracle.score(X, y), rtol=1e-12, msg="score(X, y) (accuracy of predict, as scikit-learn)")


def test_ada_boost_classifier_not_fitted(en):
    model = en.AdaBoostClassifier()
    X, _ = classification_data(66, 10, 2, 2)
    assert_raises(RuntimeError, model.decision_function, X, why="decision_function before fit")
    assert_raises(RuntimeError, model.predict, X, why="predict before fit")
    assert_raises(RuntimeError, model.staged_predict, X, why="staged_predict before fit (at the first next())")


def test_ada_boost_classifier_does_not_modify_its_inputs(en):
    X, y = classification_data(67, 60, 2, 2)
    check_does_not_modify(en.AdaBoostClassifier(n_estimators=4), X, y)


# ================================================================== GradientBoostingRegressor (14.25)
def _gb_oracle(X, y, **params):
    """scikit-learn's gradient boosting (squared error), checked to give the same trees for five random_state values."""
    runs = [sk_fit(sk(ske.GradientBoostingRegressor, criterion="squared_error", random_state=seed, **params), X, y)
            for seed in range(5)]
    for other in runs[1:]:
        for a, b in zip(other.estimators_[:, 0], runs[0].estimators_[:, 0]):
            if not (np.array_equal(a.tree_.feature, b.tree_.feature)
                    and np.array_equal(a.tree_.threshold, b.tree_.threshold)):
                raise AssertionError("test setup: tie between two features in a tree (scikit-learn breaks it at "
                                     "random); choose other test data")
    return runs[0]


def _gb_queries(oracle, seed, d, m=200):
    """Query points farther than 1e-6 from every threshold of every tree of the oracle."""
    X = _f32(np.random.default_rng(seed + 700).normal(size=(m, d)) * 1.2)
    keep = np.ones(m, dtype=bool)
    for t in oracle.estimators_[:, 0]:
        for i in np.flatnonzero(t.tree_.children_left != -1):
            keep &= np.abs(X[:, t.tree_.feature[i]] - t.tree_.threshold[i]) > 1e-6
    return X[keep]


GB_CASES = [(70, 150, 3, {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3, "min_samples_leaf": 8}),
            (71, 120, 2, {"n_estimators": 15, "learning_rate": 0.5, "max_depth": 2, "min_samples_leaf": 5}),
            (72, 100, 3, {"n_estimators": 10, "learning_rate": 1.0, "max_depth": 1}),
            (73, 160, 4, {"n_estimators": 12, "learning_rate": 0.3, "max_depth": 4, "min_samples_leaf": 10})]
GB_IDS = ["depth3", "depth2_lr0.5", "stumps_lr1", "depth4"]


def test_gradient_boosting_regressor_docstring_example(en):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 1.2, 3.0, 3.4])
    gb = fit(en.GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, max_depth=1), X, y)
    assert_close(learned(gb, "init_"), 2.15, rtol=1e-12, msg="init_ (the mean of y)")
    stages = staged(gb, np.array([[0.0], [5.0]]), 2)
    assert_close(stages[0], [1.625, 2.675], rtol=1e-9, msg="staged_predict after the first tree (2.15 ± 0.5 × 1.05)")
    assert_close(stages[1], [1.3625, 2.9375], rtol=1e-9, msg="staged_predict after the second tree")
    assert_close(learned(gb, "train_score_"), [0.300625, 0.09390625], rtol=1e-6,
                 msg="train_score_ (training MSE after each tree)")


def test_gradient_boosting_regressor_fit_returns_self(en):
    check_returns_self(en.GradientBoostingRegressor(n_estimators=3), *regression_data(74, 40, 2))


def test_gradient_boosting_regressor_init_only_stores_the_hyperparameters(en):
    params = {"n_estimators": 4, "learning_rate": 0.2, "max_depth": 2, "min_samples_leaf": 3, "subsample": 0.8,
              "random_state": 5}
    check_init_then_fit(en.GradientBoostingRegressor(**params), params, *regression_data(75, 50, 2))


def test_gradient_boosting_regressor_init_is_the_mean(en):
    X, y = regression_data(76, 60, 2)
    model = fit(en.GradientBoostingRegressor(n_estimators=2), X, y)
    got = learned(model, "init_")
    assert_python_float(got, "init_")
    assert_close(got, np.mean(y), rtol=1e-12, msg="init_ (the mean of y, the best constant for the squared error)")


@pytest.mark.parametrize("seed, n, d, params", GB_CASES, ids=GB_IDS)
def test_gradient_boosting_regressor_predict_matches_sklearn(en, seed, n, d, params):
    X, y = regression_data(seed, n, d)
    oracle = _gb_oracle(X, y, **params)
    model = fit(en.GradientBoostingRegressor(**params), X, y)
    assert_close(call(model.predict, X), oracle.predict(X), rtol=1e-9, atol=1e-9,
                 msg=f"predict on the training set with {params} (as GradientBoostingRegressor(criterion="
                     "'squared_error'))")
    X_test = _gb_queries(oracle, seed, d)
    assert_close(call(model.predict, X_test), oracle.predict(X_test), rtol=1e-9, atol=1e-9,
                 msg=f"predict on new points with {params}")


@pytest.mark.parametrize("seed, n, d, params", GB_CASES, ids=GB_IDS)
def test_gradient_boosting_regressor_staged_predict_and_train_score_match_sklearn(en, seed, n, d, params):
    X, y = regression_data(seed, n, d)
    oracle = _gb_oracle(X, y, **params)
    model = fit(en.GradientBoostingRegressor(**params), X, y)
    X_test = _gb_queries(oracle, seed, d)
    want = list(oracle.staged_predict(X_test))
    got = staged(model, X_test, len(want))
    for m, (g, w) in enumerate(zip(got, want)):
        assert_close(g, w, rtol=1e-9, atol=1e-9, msg=f"staged_predict after {m + 1} tree(s) (init_ + learning_rate × "
                                                     "the sum of the first trees)")
    assert_close(learned(model, "train_score_"), oracle.train_score_, rtol=1e-9, atol=1e-12,
                 msg="train_score_ (the training MSE after each stage, as scikit-learn)")


def test_gradient_boosting_regressor_first_tree_fits_the_residuals(en):
    X, y = regression_data(77, 120, 3)
    model = fit(en.GradientBoostingRegressor(n_estimators=3, max_depth=2, min_samples_leaf=10), X, y)
    trees = members(model, 3)
    cls = tree_class(en, "DecisionTreeRegressor")
    for k, t in enumerate(trees):
        assert isinstance(t, cls), (f"expected tree {k} to be a DecisionTreeRegressor of tree.py, got "
                                    f"{type(t).__module__}.{type(t).__name__}")
        assert (t.max_depth, t.min_samples_leaf) == (2, 10), (
            f"expected tree {k} to have max_depth=2 and min_samples_leaf=10 (the hyperparameters of the boosting), got "
            f"({t.max_depth!r}, {t.min_samples_leaf!r})")
    oracle = sk_fit(sk(sktree.DecisionTreeRegressor, max_depth=2, min_samples_leaf=10, random_state=0), X,
                    y - np.mean(y))
    assert_close(call(trees[0].predict, X), oracle.predict(X), rtol=1e-9, atol=1e-12,
                 msg="predictions of the first tree (fitted to the residuals y - mean(y), not to y)")


def test_gradient_boosting_regressor_subsample(en):
    X, y = regression_data(78, 100, 3)
    model = fit(en.GradientBoostingRegressor(n_estimators=6, subsample=0.5, random_state=3), X, y)
    for k, t in enumerate(members(model, 6)):
        n_root = learned(t, "root_").n_samples
        assert n_root == 50, (f"expected tree {k} to be fitted on max(1, int(0.5 × 100)) = 50 samples (subsample=0.5), "
                              f"got {n_root} samples at its root")
    score = np.asarray(learned(model, "train_score_"))
    assert score.shape == (6,), f"expected train_score_ of shape (6,), got {score.shape}"
    again = fit(en.GradientBoostingRegressor(n_estimators=6, subsample=0.5, random_state=3), X, y)
    other = fit(en.GradientBoostingRegressor(n_estimators=6, subsample=0.5, random_state=4), X, y)
    assert np.array_equal(call(model.predict, X), call(again.predict, X)), (
        "expected the same model with subsample=0.5 and random_state=3 twice (one generator created in fit)")
    assert not np.array_equal(call(model.predict, X), call(other.predict, X)), (
        "expected another model with random_state=4 than with random_state=3 (the subsamples are random)")


def test_gradient_boosting_regressor_subsample_draws_without_replacement(en):
    X, y = regression_data(79, 40, 1)
    X[:, 0] = np.arange(40.0)                                     # one distinct value per sample
    y = np.arange(40.0) ** 2
    model = fit(en.GradientBoostingRegressor(n_estimators=4, subsample=0.75, max_depth=8, random_state=0), X, y)
    for k, t in enumerate(members(model, 4)):
        assert t.get_n_leaves() == 30, (
            f"expected tree {k} (max_depth=8, 30 samples of distinct x and distinct residuals) to have 30 leaves, got "
            f"{t.get_n_leaves()}: draw the subsample WITHOUT replacement (rng.choice(n, size, replace=False))")


def test_gradient_boosting_regressor_subsample_train_score_is_in_bag(en):
    X = np.arange(40.0).reshape(-1, 1)                            # one distinct value per sample
    y = np.arange(40.0) ** 2
    model = fit(en.GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, subsample=0.5, max_depth=8,
                                             random_state=1), X, y)
    first = members(model, 2)[0]
    r0 = y - np.mean(y)
    fitted_rows = np.flatnonzero(np.asarray(call(first.predict, X)) == r0)   # each in-bag sample has its own leaf
    assert len(fitted_rows) == 20, (f"expected the first tree (max_depth=8, 20 in-bag samples of distinct x) to "
                                    f"predict exactly the residual of its 20 in-bag samples, got {len(fitted_rows)}")
    F1 = np.mean(y) + 0.5 * np.asarray(call(first.predict, X))
    want = float(np.mean((y[fitted_rows] - F1[fitted_rows]) ** 2))
    assert_close(learned(model, "train_score_")[0], want, rtol=1e-9,
                 msg="train_score_[0] with subsample=0.5 (the MSE on the 20 in-bag samples of the first stage only, "
                     "not on all the samples)")


def test_gradient_boosting_regressor_more_stages_lower_training_error(en):
    X, y = regression_data(80, 120, 2)
    model = fit(en.GradientBoostingRegressor(n_estimators=30, learning_rate=0.2, max_depth=2), X, y)
    score = np.asarray(learned(model, "train_score_"), dtype=float)
    assert np.all(np.diff(score) <= 1e-12), (
        f"expected the training MSE to decrease at every stage (subsample=1), got {_short(score)}")


def test_gradient_boosting_regressor_score(en):
    X, y = regression_data(81, 120, 3)
    X_test, y_test = regression_data(82, 80, 3)
    oracle = _gb_oracle(X, y, n_estimators=10, max_depth=2, min_samples_leaf=8)
    model = fit(en.GradientBoostingRegressor(n_estimators=10, max_depth=2, min_samples_leaf=8), X, y)
    got = call(model.score, X_test, y_test)
    assert_python_float(got, "score")
    assert_close(got, skm.r2_score(y_test, oracle.predict(X_test)), rtol=1e-6,
                 msg="score(X_test, y_test) (the R² of predict, as sklearn.metrics.r2_score)")


@pytest.mark.parametrize("params, why", [({"n_estimators": 0}, "n_estimators=0"),
                                         ({"learning_rate": 0.0}, "learning_rate=0"),
                                         ({"learning_rate": -0.1}, "learning_rate=-0.1"),
                                         ({"subsample": 0.0}, "subsample=0"),
                                         ({"subsample": 1.5}, "subsample=1.5")],
                         ids=["n0", "lr0", "lr_neg", "sub0", "sub_big"])
def test_gradient_boosting_regressor_fit_rejects_bad_hyperparameters(en, params, why):
    model = build(en.GradientBoostingRegressor, **params)
    assert_raises(ValueError, model.fit, *regression_data(83, 30, 2), why=why)


def test_gradient_boosting_regressor_fit_rejects_bad_data(en):
    X, y = regression_data(84, 30, 2)
    assert_raises(ValueError, en.GradientBoostingRegressor(n_estimators=2).fit, X, y[:-1],
                  why="29 targets for 30 samples")
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    assert_raises(ValueError, en.GradientBoostingRegressor(n_estimators=2).fit, X_nan, y, why="X contains NaN")


def test_gradient_boosting_regressor_not_fitted(en):
    model = en.GradientBoostingRegressor()
    X, _ = regression_data(85, 10, 2)
    assert_raises(RuntimeError, model.predict, X, why="predict before fit")
    assert_raises(RuntimeError, model.staged_predict, X, why="staged_predict before fit (at the first next())")


def test_gradient_boosting_regressor_does_not_modify_its_inputs(en):
    X, y = regression_data(86, 40, 2)
    check_does_not_modify(en.GradientBoostingRegressor(n_estimators=3), X, y)
