"""Tests of mylearn.neighbors (chapter 13): oracle tests and properties.

    pytest tests/test_ch13_neighbors.py              # your code (mon_travail/mylearn/neighbors.py)
    pytest tests/test_ch13_neighbors.py --impl=ref   # the reference

Oracles: SciPy (``scipy.spatial.distance.cdist`` with "euclidean" and "cityblock"),
scikit-learn 1.6 (``sklearn.metrics.pairwise_distances``,
``KNeighborsClassifier(algorithm="brute")`` with the same metric and weights, on data
without distance ties) and NumPy (``np.argsort(kind="stable")`` for the ties), plus
small cases computed by hand: equal distances keep the lower training index first, a
2-2 vote goes to the first class of ``classes_``, neighbours at distance 0 take all
the weight with ``weights="distance"``.
Every test name starts with the name of the function or class it tests; exercise 13.14
runs ``-k "test_pairwise_distances_ or test_kneighbors_classifier_"``.
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn import metrics as skm
from sklearn import neighbors as skn


@pytest.fixture
def nb(mylearn_module):
    return mylearn_module("neighbors")


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
    assert not added, (f"expected fit to add only learned attributes (names ending with _, like X_fit_) or private "
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


def blobs(seed, n=30, p=3, labels=(3, 7, 10)):
    """Overlapping clusters of continuous points (no two distances equal), labels given in a shuffled order."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    y = labels[rng.integers(0, len(labels), size=n)]
    y[: len(labels)] = labels                             # every class present
    centres = rng.normal(size=(len(labels), p)) * 1.5
    codes = np.searchsorted(np.sort(labels), y)
    X = centres[codes] + rng.normal(size=(n, p))
    return X, y


def queries(seed, m=12, p=3):
    return np.random.default_rng(seed).normal(size=(m, p)) * 1.8


# ================================================================== pairwise_distances (13.14)
SHAPES = [((5, 3), (7, 3)), ((1, 1), (4, 1)), ((6, 2), (6, 2)), ((3, 4), (1, 4)), ((8, 5), (2, 5))]
SHAPE_IDS = ["5x3_7x3", "1x1_4x1", "6x2_6x2", "3x4_1x4", "8x5_2x5"]


@pytest.mark.parametrize("shape_a, shape_b", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("metric, scipy_name", [("euclidean", "euclidean"), ("manhattan", "cityblock")],
                         ids=["euclidean", "manhattan"])
def test_pairwise_distances_matches_scipy_and_sklearn(nb, shape_a, shape_b, metric, scipy_name):
    rng = np.random.default_rng(sum(shape_a) * 10 + sum(shape_b))
    A = rng.normal(size=shape_a) * 2.0
    B = rng.normal(size=shape_b) * 2.0
    D = call(nb.pairwise_distances, A, B, metric=metric)
    rule = {"euclidean": "sqrt of the sum over the features of the squared differences",
            "manhattan": "sum over the features of the absolute differences"}[metric]
    assert_close(D, cdist(A, B, scipy_name), rtol=1e-9, atol=1e-12,
                 msg=f"pairwise_distances(A {shape_a}, B {shape_b}, metric={metric!r}) = cdist(A, B, {scipy_name!r}): "
                     f"D[i, j] = {rule}")
    assert_close(D, skm.pairwise_distances(A, B, metric=metric), rtol=1e-7, atol=1e-7,
                 msg=f"pairwise_distances(A {shape_a}, B {shape_b}, metric={metric!r}) against "
                     "sklearn.metrics.pairwise_distances")


def test_pairwise_distances_docstring_example(nb):
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    B = np.array([[3.0, 4.0]])
    assert_close(call(nb.pairwise_distances, A, B), [[5.0], [np.sqrt(13.0)]],
                 msg="pairwise_distances(A, B) of the docstring example (euclidean by default: 3-4-5 triangle)")
    assert_close(call(nb.pairwise_distances, A, B, metric="manhattan"), [[7.0], [5.0]],
                 msg="pairwise_distances(A, B, metric='manhattan') of the docstring example (|3| + |4| = 7)")


def test_pairwise_distances_rows_of_a_against_rows_of_b(nb):
    A = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
    B = np.array([[0.0, 3.0], [6.0, 8.0]])
    assert_close(call(nb.pairwise_distances, A, B), [[3.0, 10.0], [np.sqrt(109.0), np.sqrt(80.0)], [2.0, np.sqrt(85.0)]],
                 msg="D[i, j] is the distance between row i of A and row j of B (shape (n_a, n_b) = (3, 2))")


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_pairwise_distances_of_a_set_with_itself_is_symmetric_with_zero_diagonal(nb, metric):
    A = np.random.default_rng(3).normal(size=(9, 4)) * 5.0 + 20.0
    D = call(nb.pairwise_distances, A, A, metric=metric)
    assert_close(np.diag(D), np.zeros(9), atol=1e-9, msg=f"pairwise_distances(A, A, metric={metric!r}): the diagonal "
                 "(distance of a point to itself) is 0")
    assert_close(D, np.asarray(D, dtype=float).T, rtol=1e-12, atol=1e-12,
                 msg=f"pairwise_distances(A, A, metric={metric!r}) is symmetric")


def test_pairwise_distances_returns_a_float64_array(nb):
    A = np.array([[0, 0], [1, 2], [3, 1]])                 # integers in, floats out
    B = np.array([[1, 1], [2, 2]])
    D = call(nb.pairwise_distances, A, B)
    assert isinstance(D, np.ndarray), f"expected a NumPy array of shape (3, 2), got {type(D).__name__}"
    assert D.shape == (3, 2), f"expected shape (n_a, n_b) = (3, 2), got {D.shape}"
    assert D.dtype == np.float64, f"expected float64 distances, got dtype {D.dtype} (np.asarray(A, dtype=float))"
    assert_close(D, cdist(A, B), msg="pairwise_distances on integer coordinates")


def test_pairwise_distances_accepts_lists(nb):
    A = [[0.0, 0.0], [1.0, 1.0]]
    B = [[3.0, 4.0], [0.0, 1.0], [2.0, 2.0]]
    assert_close(call(nb.pairwise_distances, A, B, metric="manhattan"), cdist(A, B, "cityblock"),
                 msg="pairwise_distances on lists of lists (np.asarray converts them)")


def test_pairwise_distances_does_not_modify_its_inputs(nb):
    rng = np.random.default_rng(4)
    A, B = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    A0, B0 = A.copy(), B.copy()
    call(nb.pairwise_distances, A, B, copy_inputs=False)
    call(nb.pairwise_distances, A, B, metric="manhattan", copy_inputs=False)
    assert np.array_equal(A, A0) and np.array_equal(B, B0), "expected pairwise_distances to leave A and B unchanged"


@pytest.mark.parametrize("A, B, metric, why", [
    pytest.param(np.array([1.0, 2.0]), np.ones((3, 2)), "euclidean", "A is 1-D (one point is A.reshape(1, -1))",
                 id="A_1d"),
    pytest.param(np.ones((3, 2)), np.array([1.0, 2.0]), "euclidean", "B is 1-D", id="B_1d"),
    pytest.param(np.ones((2, 2, 2)), np.ones((3, 2)), "euclidean", "A is 3-D", id="A_3d"),
    pytest.param(np.ones((3, 2)), np.ones((4, 3)), "euclidean", "A has 2 features, B has 3", id="n_features"),
    pytest.param(np.ones((3, 1)), np.ones((4, 3)), "manhattan",
                 "A has 1 feature, B has 3 (the broadcast difference would silently repeat A)", id="n_features_1"),
    pytest.param(np.ones((3, 2)), np.ones((4, 2)), "cosine", "metric='cosine' is not 'euclidean' or 'manhattan'",
                 id="unknown_metric"),
    pytest.param(np.array([[1.0, np.nan]]), np.ones((4, 2)), "euclidean", "A contains NaN", id="nan_A"),
    pytest.param(np.ones((3, 2)), np.array([[np.nan, 1.0]]), "manhattan", "B contains NaN", id="nan_B"),
])
def test_pairwise_distances_rejects_bad_inputs(nb, A, B, metric, why):
    assert_raises(ValueError, nb.pairwise_distances, A.copy(), B.copy(), metric=metric, why=why)


# ================================================================== KNeighborsClassifier (13.14)
KNN_CASES = [(1, "uniform", "euclidean"), (3, "uniform", "euclidean"), (7, "uniform", "manhattan"),
             (1, "distance", "manhattan"), (4, "distance", "euclidean"), (9, "distance", "manhattan"),
             (5, "uniform", "manhattan"), (6, "distance", "euclidean")]
KNN_IDS = [f"k{k}_{w}_{m}" for k, w, m in KNN_CASES]


@pytest.mark.parametrize("k, weights, metric", KNN_CASES, ids=KNN_IDS)
@pytest.mark.parametrize("seed", [0, 1])
def test_kneighbors_classifier_matches_sklearn(nb, k, weights, metric, seed):
    X, y = blobs(seed)
    Q = queries(seed + 100)
    model = fit(nb.KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric), X, y)
    oracle = sk(skn.KNeighborsClassifier, n_neighbors=k, weights=weights, metric=metric, algorithm="brute").fit(X, y)
    options = f"KNeighborsClassifier(n_neighbors={k}, weights={weights!r}, metric={metric!r})"
    assert_same_labels(learned(model, "classes_"), oracle.classes_, msg=f"{options}: classes_ = sorted distinct labels")
    assert learned(model, "n_features_in_") == 3, f"expected n_features_in_ = 3, got {model.n_features_in_!r}"
    assert learned(model, "n_samples_fit_") == 30, f"expected n_samples_fit_ = 30, got {model.n_samples_fit_!r}"
    result = call(model.kneighbors, Q, expected="kneighbors(X) to return (dist, ind)")
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError(f"{options}: expected kneighbors(X) to return a tuple (dist, ind), got "
                             f"{type(result).__name__}")
    dist, ind = result
    want_dist, want_ind = oracle.kneighbors(Q)
    assert_indices(ind, want_ind, msg=f"{options}: kneighbors(X)[1] = indices of the {k} closest training samples, "
                   "closest first")
    assert_close(dist, want_dist, rtol=1e-9, atol=1e-12, msg=f"{options}: kneighbors(X)[0] = their distances, in "
                 "increasing order")
    assert_close(call(model.predict_proba, Q), oracle.predict_proba(Q), rtol=1e-9, atol=1e-12,
                 msg=f"{options}: predict_proba(X) = " + ("share of the k votes of each class" if weights == "uniform"
                                                          else "share of the weights 1/d of each class"))
    assert_same_labels(call(model.predict, Q), oracle.predict(Q), msg=f"{options}: predict(X) = classes_[argmax of "
                       "predict_proba] (labels, not indices; ties: the first class)")


def test_kneighbors_classifier_docstring_example(nb):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = fit(nb.KNeighborsClassifier(n_neighbors=3), X, y)
    assert_same_labels(call(model.predict, np.array([[1.1]])), [0], msg="predict([[1.1]]) of the docstring example")
    assert_close(call(model.predict_proba, np.array([[0.9]])), [[2 / 3, 1 / 3]],
                 msg="predict_proba([[0.9]]) of the docstring example (neighbours 1, 0, 2: two votes for 0)")
    dist, ind = call(model.kneighbors, np.array([[1.1]]))
    assert_indices(ind, [[1, 2, 0]], msg="kneighbors([[1.1]])[1] of the docstring example")
    assert_close(dist, [[0.1, 0.9, 1.1]], rtol=1e-9, atol=1e-12,
                 msg="kneighbors([[1.1]])[0] of the docstring example")


def test_kneighbors_classifier_equal_distances_keep_the_lower_training_index_first(nb):
    X = np.array([[3.0], [1.0], [5.0], [-1.0]])
    model = fit(nb.KNeighborsClassifier(n_neighbors=4), X, np.array([0, 1, 0, 1]))
    dist, ind = call(model.kneighbors, np.array([[2.0]]))
    assert_indices(ind, [[0, 1, 2, 3]], msg="kneighbors([[2.0]])[1] with training points 3, 1, 5, -1 (distances 1, 1, "
                   "3, 3): equal distances must keep the lower training index first (stable sort)")
    assert_close(dist, [[1.0, 1.0, 3.0, 3.0]], msg="kneighbors([[2.0]])[0]")


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_kneighbors_classifier_ties_on_a_grid_follow_a_stable_sort(nb, metric):
    # 49 training points on an integer grid: many exactly equal distances (a non-stable sort scrambles them)
    g = np.arange(7.0)
    X = np.array([[a, b] for a in g for b in g])[np.random.default_rng(5).permutation(49)]
    y = np.arange(49) % 3
    Q = np.array([[3.0, 3.0], [2.5, 4.0], [0.0, 6.0], [4.0, 1.5]])
    model = fit(nb.KNeighborsClassifier(n_neighbors=20, metric=metric), X, y)
    D = cdist(Q, X, {"euclidean": "euclidean", "manhattan": "cityblock"}[metric])
    want = np.argsort(D, axis=1, kind="stable")[:, :20]
    dist, ind = call(model.kneighbors, Q)
    assert_indices(ind, want, msg=f"kneighbors on an integer grid (metric={metric!r}), many equal distances: equal "
                   "distances keep the lower training index first (np.argsort(D, axis=1, kind='stable'))")
    assert_close(dist, np.take_along_axis(D, want, axis=1), msg="kneighbors(X)[0] on the integer grid")


def test_kneighbors_classifier_kneighbors_n_neighbors_overrides_the_default(nb):
    X, y = blobs(2)
    Q = queries(102)
    model = fit(nb.KNeighborsClassifier(n_neighbors=5), X, y)
    oracle = sk(skn.KNeighborsClassifier, n_neighbors=5, algorithm="brute").fit(X, y)
    for k in (2, 30):
        dist, ind = call(model.kneighbors, Q, n_neighbors=k)
        want_dist, want_ind = oracle.kneighbors(Q, n_neighbors=k)
        assert_indices(ind, want_ind, msg=f"kneighbors(X, n_neighbors={k}) on a model fitted with n_neighbors=5: the "
                       f"argument wins ({k} columns)")
        assert_close(dist, want_dist, msg=f"kneighbors(X, n_neighbors={k})[0]")
    assert model.n_neighbors == 5, (f"expected kneighbors(X, n_neighbors=...) to leave the hyperparameter n_neighbors "
                                    f"= 5 unchanged, got {model.n_neighbors!r}")


def test_kneighbors_classifier_kneighbors_without_distances_returns_the_indices(nb):
    X, y = blobs(3)
    Q = queries(103)
    model = fit(nb.KNeighborsClassifier(n_neighbors=4), X, y)
    ind = call(model.kneighbors, Q, return_distance=False)
    assert isinstance(ind, np.ndarray), (f"expected kneighbors(X, return_distance=False) to return the index array "
                                         f"alone, got {type(ind).__name__}")
    want = sk(skn.KNeighborsClassifier, n_neighbors=4, algorithm="brute").fit(X, y).kneighbors(Q,
                                                                                               return_distance=False)
    assert_indices(ind, want, msg="kneighbors(X, return_distance=False)")
    assert ind.dtype.kind in "iu", (f"expected integer indices (they index the training set), got dtype {ind.dtype}")


def test_kneighbors_classifier_two_two_vote_goes_to_the_first_class(nb):
    # neighbours of 2.0, closest first: 1.9 ("b"), 3.0 ("a"), 0.0 ("a"), 4.5 ("b"): 2 votes each
    X = np.array([[1.9], [0.0], [3.0], [4.5]])
    model = fit(nb.KNeighborsClassifier(n_neighbors=4), X, np.array(["b", "a", "a", "b"]))
    assert_close(call(model.predict_proba, np.array([[2.0]])), [[0.5, 0.5]], msg="predict_proba([[2.0]]): 2 votes of "
                 "4 for each class")
    assert_same_labels(call(model.predict, np.array([[2.0]])), ["a"], msg="predict([[2.0]]) with a 2-2 vote: the FIRST "
                       "class of classes_ ('a'), not the class of the closest neighbour ('b')")
    ints = fit(nb.KNeighborsClassifier(n_neighbors=4), X, np.array([5, 2, 2, 5]))
    assert_same_labels(call(ints.predict, np.array([[2.0]])), [2], msg="the same 2-2 vote with labels 5 and 2: the "
                       "first class of classes_ = [2, 5], i.e. 2")


def test_kneighbors_classifier_distance_weights_by_hand(nb):
    X = np.array([[0.0], [1.0], [3.0]])
    y = np.array([0, 1, 1])
    model = fit(nb.KNeighborsClassifier(n_neighbors=3, weights="distance"), X, y)
    # distances 0.5, 0.5, 2.5 -> weights 2, 2, 0.4
    assert_close(call(model.predict_proba, np.array([[0.5]])), [[2 / 4.4, 2.4 / 4.4]],
                 msg="weights='distance', query 0.5, training points 0 (class 0), 1 and 3 (class 1): weights 1/d = 2, "
                     "2, 0.4, so P(class 0) = 2 / 4.4")
    assert_same_labels(call(model.predict, np.array([[0.5]])), [1], msg="predict([[0.5]]) with weights='distance'")


def test_kneighbors_classifier_distance_weights_with_a_duplicate_of_a_training_point(nb):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    y = np.array([1, 0, 0, 0])
    model = fit(nb.KNeighborsClassifier(n_neighbors=3, weights="distance"), X, y)
    Q = np.array([[0.0, 0.0], [0.5, 0.5]])
    oracle = sk(skn.KNeighborsClassifier, n_neighbors=3, weights="distance", algorithm="brute").fit(X, y)
    assert_close(call(model.predict_proba, Q), oracle.predict_proba(Q),
                 msg="weights='distance', the query [0, 0] is a training point of class 1: the neighbour at distance 0 "
                     "takes all the weight (row [0, 1]), the other query [0.5, 0.5] is weighted 1/d as usual")
    assert_same_labels(call(model.predict, Q), oracle.predict(Q), msg="predict with a query equal to a training point")


def test_kneighbors_classifier_neighbours_at_distance_zero_share_the_weight(nb):
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    y = np.array(["x", "y", "y", "x"])
    model = fit(nb.KNeighborsClassifier(n_neighbors=4, weights="distance"), X, y)
    assert_close(call(model.predict_proba, np.array([[0.0, 0.0]])), [[1 / 3, 2 / 3]],
                 msg="weights='distance', three training points at distance 0 (one 'x', two 'y') and one 'x' at "
                     "distance 1: the three at distance 0 share all the weight equally, the fourth gets 0")


def test_kneighbors_classifier_string_labels(nb):
    X, codes = blobs(6, labels=(0, 1, 2))
    names = np.array(["gentoo", "adelie", "chinstrap"])
    y = names[codes]
    Q = queries(106)
    model = fit(nb.KNeighborsClassifier(n_neighbors=5), X, y)
    oracle = sk(skn.KNeighborsClassifier, n_neighbors=5, algorithm="brute").fit(X, y)
    assert_same_labels(learned(model, "classes_"), ["adelie", "chinstrap", "gentoo"],
                       msg="classes_ = sorted distinct labels (np.unique)")
    assert_close(call(model.predict_proba, Q), oracle.predict_proba(Q), msg="predict_proba with string labels: column "
                 "k is the share of classes_[k]")
    assert_same_labels(call(model.predict, Q), oracle.predict(Q), msg="predict with string labels returns the labels")


def test_kneighbors_classifier_with_k_equal_to_n_samples_gives_the_class_frequencies(nb):
    X, y = blobs(7, n=12)
    model = fit(nb.KNeighborsClassifier(n_neighbors=12), X, y)
    freq = np.array([np.mean(y == c) for c in np.unique(y)])
    assert_close(call(model.predict_proba, queries(107, m=3)), np.tile(freq, (3, 1)),
                 msg="n_neighbors = n_samples = 12 (allowed): every training sample votes, predict_proba = the class "
                     "frequencies")


def test_kneighbors_classifier_one_neighbour_predicts_the_training_labels(nb):
    X, y = blobs(8)
    model = fit(nb.KNeighborsClassifier(n_neighbors=1), X, y)
    assert_same_labels(call(model.predict, X), y, msg="n_neighbors=1 on the training set: each point is its own "
                       "nearest neighbour (distance 0)")


def test_kneighbors_classifier_score_is_the_accuracy(nb):
    X, y = blobs(9)
    Q, yq = blobs(19, n=15)
    model = fit(nb.KNeighborsClassifier(n_neighbors=3), X, y)
    want = float(np.mean(sk(skn.KNeighborsClassifier, n_neighbors=3, algorithm="brute").fit(X, y).predict(Q) == yq))
    got = call(model.score, Q, yq)
    assert isinstance(got, float), f"expected score to return a float, got {type(got).__name__}"
    assert_close(got, want, msg="score(X, y) = fraction of correct predictions")


def test_kneighbors_classifier_fit_stores_a_copy_of_the_training_set(nb):
    X, y = blobs(10)
    Q = queries(110)
    model = nb.KNeighborsClassifier(n_neighbors=3)
    call(model.fit, X, y, copy_inputs=False)
    before = np.asarray(call(model.predict_proba, Q), dtype=float).copy()
    X[:] = 0.0                                             # the caller reuses its arrays
    y[:] = y[0]
    assert_close(call(model.predict_proba, Q), before, msg="predict_proba after the caller changed X and y in place: "
                 "fit must store COPIES (np.array(X, dtype=float)), not references")


def test_kneighbors_classifier_second_fit_forgets_the_first(nb):
    model = fit(nb.KNeighborsClassifier(n_neighbors=2), *blobs(11))
    X2 = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    fit(model, X2, np.array(["u", "v", "v"]))
    assert model.n_features_in_ == 2, f"expected n_features_in_ = 2 after the second fit, got {model.n_features_in_!r}"
    assert model.n_samples_fit_ == 3, f"expected n_samples_fit_ = 3 after the second fit, got {model.n_samples_fit_!r}"
    assert_same_labels(learned(model, "classes_"), ["u", "v"], msg="classes_ after a second fit")
    assert_same_labels(call(model.predict, np.array([[4.0, 4.0]])), ["v"], msg="predict after a second fit")


def test_kneighbors_classifier_does_not_modify_its_inputs(nb):
    X, y = blobs(12)
    Q = queries(112)
    X0, y0, Q0 = X.copy(), y.copy(), Q.copy()
    model = nb.KNeighborsClassifier(n_neighbors=4, weights="distance")
    call(model.fit, X, y, copy_inputs=False)
    call(model.kneighbors, Q, copy_inputs=False)
    call(model.predict_proba, Q, copy_inputs=False)
    call(model.predict, Q, copy_inputs=False)
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "expected fit to leave X and y unchanged"
    assert np.array_equal(Q, Q0), "expected kneighbors, predict_proba and predict to leave X unchanged"


def test_kneighbors_classifier_fit_returns_self(nb):
    check_returns_self(nb.KNeighborsClassifier(n_neighbors=2), *blobs(13, n=6))


def test_kneighbors_classifier_init_only_stores_the_hyperparameters(nb):
    check_init_then_fit(nb.KNeighborsClassifier(n_neighbors=3, weights="distance", metric="manhattan"),
                        {"n_neighbors": 3, "weights": "distance", "metric": "manhattan"}, *blobs(14, n=8))


def test_kneighbors_classifier_before_fit_raises_runtime_error(nb):
    Q = queries(115, m=2)
    for method in ("kneighbors", "predict_proba", "predict"):
        assert_raises(RuntimeError, getattr(nb.KNeighborsClassifier(), method), Q.copy(),
                      why=f"{method} called before fit (no training set stored yet)")


@pytest.mark.parametrize("params, why", [
    pytest.param({"n_neighbors": 0}, "n_neighbors = 0 (k >= 1)", id="k_zero"),
    pytest.param({"n_neighbors": -2}, "n_neighbors = -2", id="k_negative"),
    pytest.param({"n_neighbors": 7}, "n_neighbors = 7 > n_samples = 6", id="k_too_big"),
    pytest.param({"weights": "gaussian"}, "weights='gaussian' is not 'uniform' or 'distance'", id="weights"),
    pytest.param({"metric": "cosine"}, "metric='cosine' is not 'euclidean' or 'manhattan'", id="metric"),
])
def test_kneighbors_classifier_fit_rejects_bad_hyperparameters(nb, params, why):
    X, y = blobs(16, n=6)
    assert_raises(ValueError, build(nb.KNeighborsClassifier, **params).fit, X, y, why=why)


@pytest.mark.parametrize("X, y, why", [
    pytest.param(np.arange(6.0), np.arange(6) % 2, "X is 1-D (use X.reshape(-1, 1))", id="1d"),
    pytest.param(np.ones((6, 2, 2)), np.arange(6) % 2, "X is 3-D", id="3d"),
    pytest.param(np.ones((6, 2)), np.arange(5) % 2, "6 rows in X, 5 labels in y", id="length"),
    pytest.param(np.array([[0.0, 1.0]] * 5 + [[np.nan, 1.0]]), np.arange(6) % 2, "X contains NaN", id="nan"),
])
def test_kneighbors_classifier_fit_rejects_bad_inputs(nb, X, y, why):
    assert_raises(ValueError, nb.KNeighborsClassifier(n_neighbors=1).fit, X.copy(), y.copy(), why=why)


@pytest.mark.parametrize("Q, k, why", [
    pytest.param(np.ones((2, 2)), None, "X has 2 features, but fitted with 3", id="n_features"),
    pytest.param(np.ones((2, 1)), None, "X has 1 feature, but fitted with 3 (broadcasting would hide it)",
                 id="n_features_1"),
    pytest.param(np.ones(3), None, "X is 1-D (one query is X.reshape(1, -1))", id="1d"),
    pytest.param(np.array([[np.nan, 0.0, 0.0]]), None, "X contains NaN", id="nan"),
    pytest.param(np.ones((2, 3)), 0, "n_neighbors = 0", id="k_zero"),
    pytest.param(np.ones((2, 3)), 11, "n_neighbors = 11 > n_samples_fit_ = 10", id="k_too_big"),
])
def test_kneighbors_classifier_kneighbors_rejects_bad_inputs(nb, Q, k, why):
    model = fit(nb.KNeighborsClassifier(n_neighbors=3), *blobs(17, n=10))
    assert_raises(ValueError, model.kneighbors, Q.copy(), n_neighbors=k, why=why)
