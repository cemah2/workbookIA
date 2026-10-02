"""Tests of mylearn.cluster (chapter 7): oracle tests and properties.

    pytest tests/test_ch07_cluster.py              # your code (mon_travail/mylearn/cluster.py)
    pytest tests/test_ch07_cluster.py --impl=ref   # the reference

Oracles: SciPy (``scipy.spatial.distance.cdist``, ``scipy.stats.chisquare``) and
scikit-learn (``NearestCentroid``, ``KMeans`` with given initial centres and
``n_init=1``, ``silhouette_samples``, ``silhouette_score``, ``accuracy_score``).
Every test name starts with the name of what it tests, so that each exercise runs
its own group: ``-k "test_pairwise_sq_distances_"`` (7.13), ``"test_nearest_centroid_"``
(7.14), ``"test_kmeans_plusplus_"`` (7.25), ``"test_kmeans_ and not plusplus"`` (7.26),
``"test_silhouette_"`` (7.28). The tests of one exercise never call the functions of
another one (your ``KMeans`` may call your ``pairwise_sq_distances`` and
``kmeans_plusplus``, as their docstrings say).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestCentroid as SklearnNearestCentroid


@pytest.fixture
def cl(mylearn_module):
    return mylearn_module("cluster")


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


def _py(value):
    """A NumPy scalar as a Python one, so that messages show 0 and 'beta', not np.int64(0) and np.str_('beta')."""
    return value.item() if isinstance(value, np.generic) else value


def _fail(msg: str, head: str, data: str = "", details: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data and details."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    if details:
        lines.append(details)
    raise AssertionError("\n".join(lines)) from None


def _first_difference(got, want, rtol, atol) -> str:
    """' (first difference at index i: expected a, got b)' for two arrays of the same shape, '' otherwise."""
    if got.ndim == 0 or got.shape != want.shape:
        return ""
    bad = np.flatnonzero(~np.isclose(got, want, rtol=rtol, atol=atol))
    if bad.size == 0:
        return ""
    i = int(bad[0])
    index = np.unravel_index(i, got.shape)
    where = int(index[0]) if got.ndim == 1 else tuple(int(k) for k in index)
    a, b = want.ravel()[i], got.ravel()[i]
    digits = 6 if f"{a:.6g}" != f"{b:.6g}" else 17      # 0.410609 and 0.410609 differ further on
    return f" (first difference at index {where}: expected {a:.{digits}g}, got {b:.{digits}g})"


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
        if have == need and want.size <= 12:   # a short array that differs beyond the 6th digit: every digit
            have, need = _short(result, 17), _short(expected, 17)
        diff = _first_difference(got, want, rtol, atol)
        if diff and want.size > 12:   # a long array: the first difference first, where a narrow screen cannot cut it
            head = f"{diff.strip()[1:-1]}; expected {need}, got {have}"
        else:
            head = f"expected {need}, got {have}{diff}"
        _fail(msg, head, data, str(exc))


def assert_same_labels(result, expected, msg="", data=""):
    """Two label arrays equal element by element; the message gives the first difference."""
    got = np.asarray(result)
    want = np.asarray(expected)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {_short(want)}, got {_short(got)}",
              data)
    bad = np.flatnonzero(got != want)
    if bad.size:
        index = np.unravel_index(int(bad[0]), want.shape)     # a row and a column for 2-D arrays (votes)
        where = int(index[0]) if want.ndim == 1 else tuple(int(k) for k in index)
        _fail(msg, f"expected {_short(want)}, got {_short(got)} ({bad.size} difference(s), the first at index "
                   f"{where}: expected {_py(want[index])!r}, got {_py(got[index])!r})", data)


def assert_python_float(value, name):
    assert isinstance(value, float), f"{name} must be a Python float, got {type(value).__name__}"


def assert_raises_value_error(function, *args, why="", **kwargs):
    """The call must raise ValueError (or a subclass); `why` names the case in the failure message."""
    name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
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


def blobs(seed, n=150, centers=3, d=2, std=1.0, scale=1.0, shift=0.0):
    """Gaussian blobs (data only: make_blobs is not an oracle) and their true blob of origin."""
    X, y = make_blobs(n_samples=n, centers=centers, n_features=d, cluster_std=std, random_state=seed)
    return X * scale + shift, y


def sq_dist(A, B):
    """Oracle: squared Euclidean distances computed by SciPy."""
    return cdist(np.asarray(A, dtype=float), np.asarray(B, dtype=float), "sqeuclidean")


def sklearn_lloyd(X, init, tol=1e-4, max_iter=300):
    """Oracle: scikit-learn's Lloyd algorithm from the given centres, one run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SklearnKMeans(n_clusters=len(init), init=init, n_init=1, algorithm="lloyd", tol=tol,
                             max_iter=max_iter).fit(X)


# ================================================================== pairwise_sq_distances (7.13)
@pytest.mark.parametrize("n_a, n_b, d, seed", [(5, 3, 2, 0), (1, 7, 4, 1), (20, 20, 10, 2), (6, 4, 1, 3),
                                               (3, 1, 3, 4), (50, 40, 7, 5)])
def test_pairwise_sq_distances_matches_scipy_cdist(cl, n_a, n_b, d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_a, d)) * 3
    B = rng.normal(size=(n_b, d)) * 3 + 1
    scale = max(1.0, (A ** 2).sum(axis=1).max(), (B ** 2).sum(axis=1).max())
    assert_close(cl.pairwise_sq_distances(A, B), sq_dist(A, B), rtol=1e-9, atol=1e-11 * scale,
                 msg=f"squared distances between {n_a} and {n_b} points in {d}-D (scipy cdist 'sqeuclidean')")


def test_pairwise_sq_distances_docstring_example(cl):
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    assert_close(cl.pairwise_sq_distances(A, B), [[1.0, 4.0, 25.0], [1.0, 2.0, 13.0]], atol=1e-12,
                 msg="the example of the docstring")


def test_pairwise_sq_distances_of_a_set_with_itself_has_a_zero_diagonal(cl):
    A = np.random.default_rng(6).normal(size=(8, 3))
    D = np.asarray(cl.pairwise_sq_distances(A, A), dtype=float)
    assert_close(np.diag(D), np.zeros(8), atol=1e-12, msg="the distance from a point to itself is 0 (diagonal)")
    assert_close(D, D.T, atol=1e-12, msg="D(A, A) must be symmetric")


def test_pairwise_sq_distances_returns_floats_of_shape_n_a_by_n_b(cl):
    D = cl.pairwise_sq_distances([[0, 0], [1, 2], [3, 1]], [[1, 1], [0, 0]])   # lists of ints
    assert isinstance(D, np.ndarray), f"expected a NumPy array, got {type(D).__name__}"
    assert D.shape == (3, 2), f"expected shape (3, 2), one row per point of A and one column per point of B, got {D.shape}"
    assert D.dtype.kind == "f", f"expected float values (np.asarray(A, dtype=float)), got dtype {D.dtype}"
    assert_close(D, [[2, 0], [1, 5], [4, 10]], msg="squared distances of integer points")


def test_pairwise_sq_distances_is_never_negative_far_from_the_origin(cl):
    # with ||a||^2 - 2 a.b + ||b||^2, points far from 0 give tiny negative values by rounding: clip them
    rng = np.random.default_rng(7)
    A = 1e4 + rng.normal(scale=0.01, size=(30, 3))
    D = np.asarray(cl.pairwise_sq_distances(A, A), dtype=float)
    assert (D >= 0).all(), (f"expected only values >= 0, got {int((D < 0).sum())} negative value(s), the smallest "
                            f"{D.min():.3g}: clip the rounding errors to 0 (np.maximum(D, 0))")
    assert_close(D, sq_dist(A, A), atol=1e-6, msg="squared distances between points close to (1e4, 1e4, 1e4)")


def test_pairwise_sq_distances_does_not_modify_its_inputs(cl):
    rng = np.random.default_rng(8)
    A, B = rng.normal(size=(4, 2)), rng.normal(size=(5, 2))
    A0, B0 = A.copy(), B.copy()
    cl.pairwise_sq_distances(A, B)
    assert np.array_equal(A, A0) and np.array_equal(B, B0), "expected A and B unchanged after the call"


BAD_SHAPES = [
    (np.zeros(3), np.zeros((2, 3)), "A is 1-D (one point must be written [[x, y, z]])"),
    (np.zeros((2, 3)), np.zeros((2, 3, 1)), "B is 3-D"),
    (np.zeros((2, 3)), np.zeros((4, 2)), "A has 3 features and B has 2"),
]


@pytest.mark.parametrize("A, B, why", BAD_SHAPES, ids=["A-1-D", "B-3-D", "different-features"])
def test_pairwise_sq_distances_rejects_invalid_shapes(cl, A, B, why):
    assert_raises_value_error(cl.pairwise_sq_distances, A, B, why=why)


# ================================================================== NearestCentroid (7.14)
LABELINGS = {"ints": None, "strings": np.array(["setosa", "beta", "zeta", "alpha", "gamma"]),
             "spaced ints": np.array([10, 3, 42, 7, 99])}


@pytest.mark.parametrize("labels", list(LABELINGS), ids=list(LABELINGS))
@pytest.mark.parametrize("n_classes, d, seed", [(2, 2, 0), (3, 2, 1), (5, 4, 2)])
def test_nearest_centroid_matches_sklearn(cl, labels, n_classes, d, seed):
    X, y = blobs(seed, n=90, centers=n_classes, d=d, std=2.5)
    if LABELINGS[labels] is not None:
        y = LABELINGS[labels][y]
    X_new, _ = blobs(seed + 100, n=40, centers=n_classes, d=d, std=4.0)
    oracle = SklearnNearestCentroid().fit(X, y)
    model = cl.NearestCentroid().fit(X, y)
    assert_same_labels(model.classes_, np.unique(y), msg="classes_ (the sorted distinct labels)")
    assert_close(model.centroids_, oracle.centroids_, atol=1e-12,
                 msg="centroids_[k] is the mean of the training rows of class classes_[k]")
    assert_same_labels(model.predict(X_new), oracle.predict(X_new), msg="predict (scikit-learn NearestCentroid)")


def test_nearest_centroid_fit_returns_self_and_does_not_modify_X(cl):
    X, y = blobs(3, n=30)
    X0 = X.copy()
    model = cl.NearestCentroid()
    returned = model.fit(X, y)
    assert returned is model, "fit must return the fitted classifier itself (return self)"
    assert np.array_equal(X, X0), "fit must not modify X"


def test_nearest_centroid_classes_are_sorted(cl):
    X = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0], [0.0, 1.0], [5.0, 6.0], [10.0, 1.0]])
    y = np.array(["c", "a", "b", "c", "a", "b"])
    model = cl.NearestCentroid().fit(X, y)
    assert_same_labels(model.classes_, ["a", "b", "c"], msg="classes_ must be sorted (np.unique)")
    assert_close(model.centroids_, [[5.0, 5.5], [10.0, 0.5], [0.0, 0.5]],
                 msg="centroids_ in the order of classes_")


def test_nearest_centroid_decision_function_two_classes(cl):
    X, y = blobs(4, n=60, centers=2, std=3.0)
    X_new, _ = blobs(5, n=25, centers=2, std=5.0)
    model = cl.NearestCentroid().fit(X, y)
    centroids = np.array([X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)])
    D = sq_dist(X_new, centroids)
    scores = model.decision_function(X_new)
    assert_close(scores, D[:, 0] - D[:, 1], rtol=1e-9, atol=1e-9,
                 msg="with 2 classes, decision_function is d²(x, c0) - d²(x, c1), shape (n_samples,)")
    assert_same_labels(np.where(np.asarray(scores) > 0, 1, 0), model.predict(X_new),
                       msg="a positive score must mean classes_[1]")


def test_nearest_centroid_decision_function_several_classes(cl):
    X, y = blobs(6, n=80, centers=4, std=2.0)
    X_new, _ = blobs(7, n=20, centers=4, std=3.0)
    model = cl.NearestCentroid().fit(X, y)
    centroids = np.array([X[y == k].mean(axis=0) for k in range(4)])
    scores = model.decision_function(X_new)
    assert_close(scores, -sq_dist(X_new, centroids), rtol=1e-9, atol=1e-9,
                 msg="with K > 2 classes, column k of decision_function is -d²(x, c_k), shape (n_samples, K)")
    assert_same_labels(model.classes_[np.argmax(scores, axis=1)], model.predict(X_new),
                       msg="the largest score must be the predicted class")


def test_nearest_centroid_ties_go_to_the_first_class(cl):
    X = np.array([[0.0, 0.0], [0.0, 2.0], [4.0, 0.0], [4.0, 2.0], [2.0, 9.0], [2.0, 11.0]])
    y = np.array([0, 0, 1, 1, 2, 2])          # centroids (0, 1), (4, 1) and (2, 10)
    model = cl.NearestCentroid().fit(X, y)
    tie = np.array([[2.0, 1.0], [2.0, 1.5]])   # as far from (0, 1) as from (4, 1)
    assert_same_labels(model.predict(tie), [0, 0], msg="equal distances to classes 0 and 1: the first class wins")


def test_nearest_centroid_score_is_the_accuracy(cl):
    X, y = blobs(8, n=100, centers=3, std=3.5)
    X_new, y_new = blobs(9, n=60, centers=3, std=3.5)
    model = cl.NearestCentroid().fit(X, y)
    score = model.score(X_new, y_new)
    assert_python_float(score, "score")
    assert_close(score, metrics.accuracy_score(y_new, model.predict(X_new)), msg="score is the accuracy of predict")


def test_nearest_centroid_takes_no_hyperparameter(cl):
    model = cl.NearestCentroid()
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    assert_same_labels(model.fit(X, [0, 0, 1, 1]).predict([[2.0], [9.0]]), [0, 1],
                       msg="NearestCentroid() with no argument, on four 1-D points")


def test_nearest_centroid_docstring_example(cl):
    X = np.array([[0.0, 0.0], [0.0, 2.0], [4.0, 0.0], [4.0, 2.0]])
    clf = cl.NearestCentroid().fit(X, np.array([0, 0, 1, 1]))
    X_new = np.array([[1.0, 1.0], [3.5, 0.0]])
    assert_close(clf.centroids_, [[0.0, 1.0], [4.0, 1.0]], msg="centroids_ of the docstring example")
    assert_same_labels(clf.predict(X_new), [0, 1], msg="predict of the docstring example")
    assert_close(clf.decision_function(X_new), [-8.0, 12.0], msg="decision_function of the docstring example")
    assert_close(clf.score(X, np.array([0, 0, 1, 1])), 1.0, msg="score of the docstring example")


@pytest.mark.parametrize("X, y, why", [
    (np.zeros((4, 2)), np.zeros(4), "a single class"),
    (np.zeros((4, 2)), np.array([0, 1, 0]), "4 rows in X but 3 labels"),
], ids=["one-class", "length-mismatch"])
def test_nearest_centroid_rejects_invalid_inputs(cl, X, y, why):
    assert_raises_value_error(cl.NearestCentroid().fit, X, y, why=why)


# ================================================================== kmeans_plusplus (7.25)
def test_kmeans_plusplus_returns_distinct_rows_of_X(cl):
    X, _ = blobs(10, n=60, centers=4, d=3)
    for seed in range(20):
        centres = np.asarray(cl.kmeans_plusplus(X, 4, rng=np.random.default_rng(seed)), dtype=float)
        assert centres.shape == (4, 3), f"expected an array of shape (4, 3) (n_clusters, n_features), got {centres.shape}"
        is_row = (sq_dist(centres, X) == 0).any(axis=1)
        assert is_row.all(), f"expected every centre to be a row of X, got {_short(centres[~is_row])} (seed {seed})"
        assert len(np.unique(centres, axis=0)) == 4, (
            f"expected 4 different rows, got {_short(centres)} (seed {seed}): D(x) is the distance from x to the "
            f"NEAREST of all the centres chosen so far, so a chosen row (D = 0) cannot be drawn again")


def test_kmeans_plusplus_never_picks_a_centre_twice_with_duplicate_rows(cl):
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])   # only 2 distinct rows
    for seed in range(200):
        centres = np.asarray(cl.kmeans_plusplus(X, 2, rng=np.random.default_rng(seed)), dtype=float)
        got = sorted(map(tuple, centres.tolist()))
        assert got == [(0.0, 0.0), (5.0, 5.0)], (
            f"expected the two distinct rows (0, 0) and (5, 5), got {got} (seed {seed}): a row at distance D = 0 "
            f"from a chosen centre must have probability 0")


def test_kmeans_plusplus_is_reproducible_and_keeps_X(cl):
    X, _ = blobs(11, n=50, centers=3)
    X0 = X.copy()
    a = cl.kmeans_plusplus(X, 3, rng=np.random.default_rng(5))
    b = cl.kmeans_plusplus(X, 3, rng=np.random.default_rng(5))
    assert_close(a, b, atol=0, msg="two generators with the same seed must give the same centres")
    assert np.array_equal(X, X0), "kmeans_plusplus must not modify X"


def test_kmeans_plusplus_accepts_rng_none(cl):
    X, _ = blobs(12, n=20, centers=2)
    centres = np.asarray(cl.kmeans_plusplus(X, 2), dtype=float)
    assert centres.shape == (2, 2), f"expected shape (2, 2) with rng=None, got {centres.shape}"


def test_kmeans_plusplus_first_centre_is_drawn_uniformly(cl):
    X = np.arange(6, dtype=float).reshape(-1, 1) ** 2        # 6 rows: 0, 1, 4, 9, 16, 25
    rng = np.random.default_rng(2024)
    n_draws = 3000
    picks = [int(np.asarray(cl.kmeans_plusplus(X, 1, rng=rng)).ravel()[0]) for _ in range(n_draws)]
    counts = np.array([picks.count(int(v)) for v in X.ravel()])
    p = stats.chisquare(counts, np.full(6, n_draws / 6)).pvalue
    assert p > 1e-6, (f"expected the first centre to be a uniform draw among the 6 rows (about {n_draws // 6} times "
                      f"each), got the counts {counts.tolist()} (chi-square p-value {p:.2g})")


def test_kmeans_plusplus_next_centre_is_drawn_with_probability_d_squared(cl):
    values = np.array([0.0, 1.0, 3.0, 7.0, 15.0])
    X = values.reshape(-1, 1)
    rng = np.random.default_rng(7)
    n_draws = 6000
    counts = np.zeros((5, 5))                      # counts[first, second]
    for _ in range(n_draws):
        centres = np.asarray(cl.kmeans_plusplus(X, 2, rng=rng), dtype=float).ravel()
        first, second = (int(np.flatnonzero(values == c)[0]) for c in centres)
        counts[first, second] += 1
    assert counts.trace() == 0, "expected the second centre never equal to the first one (D = 0 means probability 0)"
    for first in range(5):
        if counts[first].sum() == 0:
            continue                 # never drawn first: test_kmeans_plusplus_first_centre_is_drawn_uniformly says why
        d2 = (values - values[first]) ** 2
        others = np.flatnonzero(d2 > 0)
        expected = counts[first].sum() * d2[others] / d2.sum()
        observed = counts[first, others]
        p = stats.chisquare(observed, expected).pvalue
        assert p > 1e-6, (
            f"after the first centre {values[first]:g}, expected the next one to be drawn with probability "
            f"D(x)²/ΣD²: about {np.round(expected).astype(int).tolist()} times for {values[others].tolist()}, "
            f"got {observed.astype(int).tolist()} (chi-square p-value {p:.2g})")


def test_kmeans_plusplus_beats_random_rows_after_lloyd(cl):
    # one big blob and seven small ones: random rows often put two centres in the big blob
    sizes = [400, 25, 25, 25, 25, 25, 25, 25]
    X, _ = make_blobs(n_samples=sizes, centers=None, cluster_std=0.6, random_state=13,
                      center_box=(-30.0, 30.0))
    rng = np.random.default_rng(13)
    inertia_pp, inertia_random = [], []
    for _ in range(30):
        pp = np.asarray(cl.kmeans_plusplus(X, 8, rng=rng), dtype=float)
        rows = X[rng.choice(len(X), size=8, replace=False)]
        inertia_pp.append(sklearn_lloyd(X, pp).inertia_)
        inertia_random.append(sklearn_lloyd(X, rows).inertia_)
    assert np.mean(inertia_pp) < 0.8 * np.mean(inertia_random), (
        f"expected k-means++ centres to give a clearly lower mean inertia after Lloyd's algorithm than random rows, "
        f"got {np.mean(inertia_pp):.1f} against {np.mean(inertia_random):.1f} (30 runs): draw the next centre "
        f"with probability D(x)²/ΣD²")


@pytest.mark.parametrize("X, k, why", [
    (np.zeros((5, 2)) + np.arange(5)[:, None], 0, "n_clusters = 0"),
    (np.zeros((5, 2)) + np.arange(5)[:, None], -1, "n_clusters = -1"),
    (np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]), 3, "3 centres asked but only 2 distinct rows"),
], ids=["zero", "negative", "too-many"])
def test_kmeans_plusplus_rejects_invalid_n_clusters(cl, X, k, why):
    assert_raises_value_error(cl.kmeans_plusplus, X, k, rng=np.random.default_rng(0), why=why)


# ================================================================== KMeans (7.26)
LLOYD_CASES = [
    # seed, n, centers, d, std, scale, tol, max_iter, why
    (0, 150, 3, 2, 1.0, 1.0, 1e-4, 300, "3 blobs in 2-D"),
    (1, 200, 4, 3, 2.0, 10.0, 1e-4, 300, "4 blobs in 3-D, scaled by 10"),
    (2, 120, 5, 4, 3.0, 0.1, 0.0, 300, "tol = 0: only an unchanged assignment stops"),
    (3, 300, 3, 2, 4.0, 1.0, 1e-2, 300, "large tol: the centre-shift rule stops early"),
    (4, 100, 3, 2, 2.0, 1.0, 1e-4, 1, "max_iter = 1"),
    (5, 100, 4, 5, 3.0, 1.0, 1e-4, 2, "max_iter = 2"),
    (6, 250, 2, 1, 1.5, 100.0, 1e-4, 300, "2 blobs in 1-D, scaled by 100"),
    (0, 200, 4, 2, 3.0, 1.0, 5e-2, 300, "tol = 0.05: the sum of the SQUARED centre shifts decides"),
]


@pytest.mark.parametrize("seed, n, centers, d, std, scale, tol, max_iter, why", LLOYD_CASES,
                         ids=[case[-1] for case in LLOYD_CASES])
def test_kmeans_matches_sklearn_with_given_init(cl, seed, n, centers, d, std, scale, tol, max_iter, why):
    X, origin = blobs(seed, n=n, centers=centers, d=d, std=std, scale=scale)
    init = X[[int(np.flatnonzero(origin == k)[0]) for k in range(centers)]]   # one row of each blob
    oracle = sklearn_lloyd(X, init, tol=tol, max_iter=max_iter)
    model = cl.KMeans(n_clusters=centers, init=init, n_init=1, tol=tol, max_iter=max_iter).fit(X)
    case = f"[{why}]"
    iterations = ("" if model.n_iter_ == oracle.n_iter_ else
                  f" (and n_iter_ = {_py(model.n_iter_)} instead of {oracle.n_iter_}: check the stopping rules)")
    assert_close(model.cluster_centers_, oracle.cluster_centers_, rtol=1e-7, atol=1e-9 * scale,
                 msg=f"cluster_centers_ (scikit-learn KMeans, same init) {case}{iterations}")
    assert_same_labels(model.labels_, oracle.labels_, msg=f"labels_ (scikit-learn KMeans, same init) {case}")
    assert_close(model.inertia_, oracle.inertia_, rtol=1e-7, msg=f"inertia_ (scikit-learn KMeans, same init) {case}")
    assert model.n_iter_ == oracle.n_iter_, (
        f"n_iter_: expected {oracle.n_iter_} iteration(s) like scikit-learn, got {model.n_iter_} {case}: stop when "
        f"the assignment does not change any more (that last iteration counts), or when the sum of the squared "
        f"centre shifts is <= tol * mean(np.var(X, axis=0)), or after max_iter iterations")


def test_kmeans_attributes_have_the_documented_types(cl):
    X, origin = blobs(20, n=60, centers=3, d=2)
    init = X[[int(np.flatnonzero(origin == k)[0]) for k in range(3)]]
    model = cl.KMeans(n_clusters=3, init=init).fit(X)
    assert isinstance(model.cluster_centers_, np.ndarray) and model.cluster_centers_.shape == (3, 2), (
        f"expected cluster_centers_ of shape (3, 2), got {np.shape(model.cluster_centers_)}")
    labels = model.labels_
    assert isinstance(labels, np.ndarray) and labels.shape == (60,) and labels.dtype.kind in "iu", (
        f"expected labels_ as an integer array of shape (60,), got {type(labels).__name__} "
        f"{getattr(labels, 'dtype', '')} {np.shape(labels)}")
    assert_python_float(model.inertia_, "inertia_")
    assert isinstance(model.n_iter_, (int, np.integer)) and not isinstance(model.n_iter_, bool), (
        f"expected n_iter_ to be an int, got {type(model.n_iter_).__name__}")


def test_kmeans_inertia_and_labels_match_the_final_centres(cl):
    X, _ = blobs(21, n=200, centers=4, d=3, std=2.0)
    model = cl.KMeans(n_clusters=4, init="random", n_init=3, random_state=0).fit(X)
    D = sq_dist(X, model.cluster_centers_)
    assert_same_labels(model.labels_, D.argmin(axis=1), msg="labels_: the nearest final centre of every sample")
    assert_close(model.inertia_, D.min(axis=1).sum(), rtol=1e-9,
                 msg="inertia_: the sum of the squared distances to the nearest final centre")


def test_kmeans_predict_transform_and_score(cl):
    X, _ = blobs(22, n=120, centers=3, d=2)
    X_new, _ = blobs(23, n=30, centers=3, d=2, std=3.0)
    model = cl.KMeans(n_clusters=3, init="random", n_init=2, random_state=1).fit(X)
    D = sq_dist(X_new, model.cluster_centers_)
    assert_same_labels(model.predict(X_new), D.argmin(axis=1), msg="predict: index of the nearest learnt centre")
    assert_close(model.transform(X_new), np.sqrt(D), rtol=1e-9, atol=1e-9,
                 msg="transform: Euclidean distances (not squared) to every centre, shape (n_samples, n_clusters)")
    score = model.score(X_new)
    assert_python_float(score, "score")
    assert_close(score, -D.min(axis=1).sum(), rtol=1e-9, msg="score: minus the inertia of X_new")


def test_kmeans_fit_predict_returns_labels_(cl):
    X, _ = blobs(24, n=80, centers=3)
    model = cl.KMeans(n_clusters=3, random_state=4, n_init=2)
    labels = model.fit_predict(X)
    assert_same_labels(labels, model.labels_, msg="fit_predict(X) must return labels_")


def test_kmeans_fit_returns_self_and_does_not_modify_X(cl):
    X, _ = blobs(25, n=40, centers=2)
    X0 = X.copy()
    model = cl.KMeans(n_clusters=2, random_state=0, n_init=1)
    assert model.fit(X) is model, "fit must return the fitted estimator itself (return self)"
    assert np.array_equal(X, X0), "fit must not modify X"


def test_kmeans_ignores_y(cl):
    X, origin = blobs(26, n=60, centers=3)
    a = cl.KMeans(n_clusters=3, random_state=2, n_init=2).fit(X)
    b = cl.KMeans(n_clusters=3, random_state=2, n_init=2).fit(X, origin)
    assert_close(b.cluster_centers_, a.cluster_centers_, atol=0, msg="fit(X, y) must give the same result as fit(X)")


def test_kmeans_given_init_does_a_single_run_and_keeps_the_array(cl):
    X, origin = blobs(27, n=90, centers=3, std=2.5)
    init = X[[int(np.flatnonzero(origin == k)[0]) for k in range(3)]]
    init0 = init.copy()
    one = cl.KMeans(n_clusters=3, init=init, n_init=1).fit(X)
    ten = cl.KMeans(n_clusters=3, init=init, n_init=10, random_state=3).fit(X)
    assert np.array_equal(init, init0), "fit must not modify the init array (work on a copy)"
    assert_close(ten.cluster_centers_, one.cluster_centers_, atol=0,
                 msg="with given centres, a single run is done whatever n_init")
    assert ten.n_iter_ == one.n_iter_, f"expected the same n_iter_ ({one.n_iter_}), got {ten.n_iter_}"


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_kmeans_random_state_makes_fit_reproducible(cl, init):
    X, _ = blobs(28, n=100, centers=4, d=2, std=2.0)
    a = cl.KMeans(n_clusters=4, init=init, n_init=3, random_state=11).fit(X)
    b = cl.KMeans(n_clusters=4, init=init, n_init=3, random_state=11).fit(X)
    assert_close(b.cluster_centers_, a.cluster_centers_, atol=0,
                 msg=f"init={init!r}: the same random_state must give the same centres")


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_kmeans_keeps_the_best_of_n_init_runs(cl, init):
    X, _ = make_blobs(n_samples=[300, 20, 20, 20, 20, 20], centers=None, cluster_std=0.8, random_state=29,
                      center_box=(-20.0, 20.0))
    improved = 0
    for seed in range(8):
        single = cl.KMeans(n_clusters=6, init=init, n_init=1, random_state=seed).fit(X)
        best = cl.KMeans(n_clusters=6, init=init, n_init=10, random_state=seed).fit(X)
        assert best.inertia_ <= single.inertia_ + 1e-9, (
            f"init={init!r}, random_state={seed}: expected the best of 10 runs to have an inertia <= the first run "
            f"alone ({single.inertia_:.3f}), got {best.inertia_:.3f}: create the generator once in fit, use it for "
            f"every run in order, and keep the run with the lowest inertia")
        improved += best.inertia_ < single.inertia_ - 1e-6
    assert improved > 0, (
        f"init={init!r}: expected 10 runs to beat a single run at least once in 8 tries, got the same inertia every "
        f"time: the runs must start from different initialisations (one generator, created once in fit)")


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_kmeans_with_as_many_clusters_as_samples_puts_one_point_per_cluster(cl, init):
    X = np.random.default_rng(30).normal(size=(7, 2))
    model = cl.KMeans(n_clusters=7, init=init, n_init=1, random_state=0).fit(X)
    assert_close(np.unique(np.asarray(model.cluster_centers_, dtype=float), axis=0), np.unique(X, axis=0), atol=1e-12,
                 msg=f"init={init!r}, 7 clusters for 7 points: the centres must be the 7 points (the initial centres "
                     f"must be 7 different rows of X)")
    assert_close(model.inertia_, 0.0, atol=1e-12,
                 msg=f"init={init!r}, one point per cluster: inertia_ is 0 (each point is its own centre)")


def test_kmeans_empty_cluster_keeps_its_previous_centre(cl):
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    init = np.array([[0.5], [10.5], [100.0]])       # nobody is close to 100: that cluster stays empty
    model = cl.KMeans(n_clusters=3, init=init).fit(X)
    assert_close(model.cluster_centers_, [[0.5], [10.5], [100.0]],
                 msg="an empty cluster keeps its previous centre (the mean of no point is undefined)")
    assert_same_labels(model.labels_, [0, 0, 1, 1], msg="labels_ with an empty third cluster")
    assert_close(model.inertia_, 1.0, msg="inertia_ with an empty third cluster")


def test_kmeans_docstring_example(cl):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
    km = cl.KMeans(n_clusters=2, init=np.array([[0.0, 0.0], [5.0, 5.0]])).fit(X)
    assert_close(km.cluster_centers_, [[0.0, 0.5], [5.0, 5.5]], msg="cluster_centers_ of the docstring example")
    assert_same_labels(km.labels_, [0, 0, 1, 1], msg="labels_ of the docstring example")
    assert_close(km.inertia_, 1.0, msg="inertia_ of the docstring example")
    assert_same_labels(km.predict(np.array([[1.0, 1.0], [4.0, 4.0]])), [0, 1], msg="predict of the docstring example")
    assert_close(km.score(X), -1.0, msg="score of the docstring example")


BAD_KMEANS = [
    (dict(n_clusters=0), "n_clusters = 0"),
    (dict(n_clusters=11), "n_clusters = 11 for 10 samples"),
    (dict(n_clusters=2, n_init=0), "n_init = 0"),
    (dict(n_clusters=2, max_iter=0), "max_iter = 0"),
    (dict(n_clusters=2, init="kmeans++"), "init = 'kmeans++' (unknown string: 'k-means++' or 'random')"),
    (dict(n_clusters=3, init=np.zeros((2, 2))), "an init array with 2 rows for 3 clusters"),
    (dict(n_clusters=2, init=np.zeros((2, 3))), "an init array with 3 features for data with 2"),
]


@pytest.mark.parametrize("params, why", BAD_KMEANS,
                         ids=["zero-clusters", "too-many-clusters", "n_init-0", "max_iter-0", "unknown-init",
                              "init-rows", "init-features"])
def test_kmeans_rejects_invalid_parameters(cl, params, why):
    X = np.random.default_rng(31).normal(size=(10, 2))
    assert_raises_value_error(cl.KMeans(**params).fit, X, why=why)


# ================================================================== silhouette (7.28)
def _labelled(seed, n, d, n_labels, kind):
    rng = np.random.default_rng(seed)
    X, origin = blobs(seed, n=n, centers=n_labels, d=d, std=2.0)
    labels = origin.copy()
    flip = rng.random(n) < 0.15                   # some samples in the wrong cluster: negative silhouettes
    labels[flip] = rng.integers(0, n_labels, size=int(flip.sum()))
    if kind == "strings":
        labels = np.array(["red", "green", "blue", "cyan", "teal"])[labels]
    elif kind == "spaced ints":
        labels = np.array([3, 17, 8, 42, 5])[labels]
    elif kind == "singleton":
        labels[0] = 99                            # a cluster with a single sample
    return X, labels


SIL_CASES = [(0, 40, 2, 2, "ints"), (1, 60, 3, 3, "strings"), (2, 50, 1, 4, "spaced ints"),
             (3, 45, 4, 3, "singleton"), (4, 80, 2, 5, "ints")]
SIL_HINT = ("a(i): mean distance to the OTHER samples of its cluster (divide by its size - 1); b(i): the smallest of the "
            "mean distances to the samples of each other cluster; 0 for a sample alone in its cluster. If you build "
            "the distances with ||a||² - 2a.b + ||b||², set the diagonal to 0 (np.fill_diagonal(D, 0.0)).")


@pytest.mark.parametrize("seed, n, d, n_labels, kind", SIL_CASES, ids=[f"{c[3]}-labels-{c[4]}" for c in SIL_CASES])
def test_silhouette_samples_matches_sklearn(cl, seed, n, d, n_labels, kind):
    X, labels = _labelled(seed, n, d, n_labels, kind)
    # rtol = atol = 1e-7: distances built with ||a||² - 2a.b + ||b||² have a diagonal of about 1e-14 instead of 0,
    # i.e. about 1e-7 after the square root (scikit-learn sets that diagonal to 0); wrong versions miss by > 1e-3
    assert_close(cl.silhouette_samples(X, labels), metrics.silhouette_samples(X, labels), rtol=1e-7, atol=1e-7,
                 msg=f"silhouette of every sample (scikit-learn silhouette_samples) [{kind} labels]",
                 data=SIL_HINT)


@pytest.mark.parametrize("seed, n, d, n_labels, kind", SIL_CASES, ids=[f"{c[3]}-labels-{c[4]}" for c in SIL_CASES])
def test_silhouette_score_matches_sklearn(cl, seed, n, d, n_labels, kind):
    X, labels = _labelled(seed, n, d, n_labels, kind)
    score = cl.silhouette_score(X, labels)
    assert_python_float(score, "silhouette_score")
    assert_close(score, metrics.silhouette_score(X, labels), rtol=1e-7, atol=1e-7,
                 msg=f"mean silhouette (scikit-learn silhouette_score) [{kind} labels]", data=SIL_HINT)


def test_silhouette_samples_docstring_example(cl):
    X = np.array([[0.0], [1.0], [4.0], [6.0]])
    assert_close(cl.silhouette_samples(X, [0, 0, 1, 1]), [0.8, 0.75, 1.5 / 3.5, 3.5 / 5.5], rtol=1e-9,
                 msg="silhouettes of the docstring example")
    assert_close(cl.silhouette_score(X, [0, 0, 1, 1]), np.mean([0.8, 0.75, 1.5 / 3.5, 3.5 / 5.5]), rtol=1e-9,
                 msg="silhouette_score of the docstring example")


def test_silhouette_samples_a_sample_alone_in_its_cluster_gets_zero(cl):
    X = np.array([[0.0], [1.0], [2.0], [10.0], [30.0]])
    s = np.asarray(cl.silhouette_samples(X, [0, 0, 0, 1, 2]), dtype=float)
    assert_close(s[3:], [0.0, 0.0], msg="the two samples alone in their cluster get 0 (and no NaN)")


def test_silhouette_samples_identical_points_give_zero_not_nan(cl):
    X = np.ones((4, 2))                           # a = b = 0 everywhere: 0/0 must become 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = np.asarray(cl.silhouette_samples(X, [0, 0, 1, 1]), dtype=float)
    assert_close(s, np.zeros(4), msg="identical points (a = b = 0) get 0, not NaN")


def test_silhouette_samples_are_between_minus_one_and_one(cl):
    X, labels = _labelled(5, 70, 3, 4, "ints")
    s = np.asarray(cl.silhouette_samples(X, labels), dtype=float)
    assert s.shape == (70,), f"expected one value per sample, shape (70,), got {s.shape}"
    assert ((s >= -1) & (s <= 1)).all(), f"expected values in [-1, 1], got min {s.min():.3g} and max {s.max():.3g}"


@pytest.mark.parametrize("labels, why", [
    ([0, 0, 0, 0, 0], "a single cluster"),
    ([0, 1, 2, 3, 4], "5 clusters for 5 samples"),
], ids=["one-label", "n-labels"])
def test_silhouette_rejects_invalid_labels(cl, labels, why):
    X = np.arange(10, dtype=float).reshape(5, 2)
    assert_raises_value_error(cl.silhouette_samples, X, labels, why=why)
    assert_raises_value_error(cl.silhouette_score, X, labels, why=why)
