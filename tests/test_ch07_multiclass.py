"""Tests of mylearn.multiclass (chapter 7): oracle tests and properties.

    pytest tests/test_ch07_multiclass.py              # your code (mon_travail/mylearn/multiclass.py)
    pytest tests/test_ch07_multiclass.py --impl=ref   # the reference

Oracles: scikit-learn (``sklearn.multiclass.OneVsRestClassifier`` and
``OneVsOneClassifier`` wrapping the same base estimator: ``LinearSVC``,
``LogisticRegression`` or ``GaussianNB``; ``accuracy_score``) and small binary models
written here, which record what they were trained on. The base estimators never come
from your mylearn. Every test name starts with the name of the class it tests:
``-k "test_one_vs_rest_"`` (7.22), ``-k "test_one_vs_one_"`` (7.23).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import numpy as np
import pytest
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier as SklearnOvO
from sklearn.multiclass import OneVsRestClassifier as SklearnOvR
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


@pytest.fixture
def mc(mylearn_module):
    return mylearn_module("multiclass")


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
    digits = 6 if f"{a:.6g}" != f"{b:.6g}" else 17
    return f" (first difference at index {where}: expected {a:.{digits}g}, got {b:.{digits}g})"


def _fail(msg: str, head: str, data: str = "", details: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data and details."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    if details:
        lines.append(details)
    raise AssertionError("\n".join(lines)) from None


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg=""):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected {_short(expected)}, got an object of type {type(result).__name__}")
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {_short(expected)}, "
                   f"got {_short(result)}")
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
        _fail(msg, head, "", str(exc))


def assert_same_labels(result, expected, msg=""):
    """Two label arrays equal element by element; the message gives the first difference."""
    got = np.asarray(result)
    want = np.asarray(expected)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {_short(want)}, got {_short(got)}")
    bad = np.flatnonzero(got != want)
    if bad.size:
        index = np.unravel_index(int(bad[0]), want.shape)     # a row and a column for 2-D arrays (votes)
        where = int(index[0]) if want.ndim == 1 else tuple(int(k) for k in index)
        _fail(msg, f"expected {_short(want)}, got {_short(got)} ({bad.size} difference(s), the first at index "
                   f"{where}: expected {_py(want[index])!r}, got {_py(got[index])!r})")


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


NAMES = np.array(["setosa", "beta", "zeta", "alpha", "gamma"])


def dataset(seed, n_classes, strings=False, n=150, d=3, std=2.5):
    """Blobs of n_classes classes, with integer or string labels (data only: make_blobs is not an oracle)."""
    X, y = make_blobs(n_samples=n, centers=n_classes, n_features=d, cluster_std=std, random_state=seed)
    return X, (NAMES[y] if strings else y)


def linear_svc():
    return LinearSVC(random_state=0, max_iter=20_000)


def logistic():
    return LogisticRegression(max_iter=2_000)


class Recorder:
    """A tiny binary model written here: it remembers its training data and scores by the nearest class mean."""

    def fit(self, X, y):
        self.X_seen_ = np.asarray(X, dtype=float).copy()
        self.y_seen_ = np.asarray(y).copy()
        self.means_ = np.array([self.X_seen_[self.y_seen_ == c].mean(axis=0) for c in (0, 1)])
        return self

    def decision_function(self, X):
        D = ((np.asarray(X, dtype=float)[:, None, :] - self.means_[None, :, :]) ** 2).sum(axis=2)
        return D[:, 0] - D[:, 1]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


class PredictOnly:
    """A binary model with fit and predict only (no score to compare the classes)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class Flat:
    """A binary model whose score is 0 for every sample: in a one-versus-rest, every class gets the same score."""

    def fit(self, X, y):
        return self

    def decision_function(self, X):
        return np.zeros(len(X))

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class Cyclic:
    """Binary model whose vote depends on its pair of classes, read from the first feature of its rows:
    pair (0, 1) votes for 0, pair (0, 2) for 2 and pair (1, 2) for 1: one vote each, a three-way tie."""

    CODED_WINNER = {(0, 1): 0, (0, 2): 1, (1, 2): 0}     # 0 = class i, 1 = class j of the pair (i, j)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        pair = (int(X[y == 0, 0][0]), int(X[y == 1, 0][0]))
        if pair not in self.CODED_WINNER:
            raise AssertionError(f"for the pair (i, j), i < j, class i must be coded 0 and class j coded 1; "
                                 f"this model received class {pair[0]} coded 0 and class {pair[1]} coded 1")
        self.out_ = self.CODED_WINNER[pair]
        return self

    def predict(self, X):
        return np.full(len(X), self.out_)


# ================================================================== OneVsRestClassifier (7.22)
@pytest.mark.parametrize("n_classes, strings, seed", [(3, False, 0), (4, True, 1), (5, False, 2)],
                         ids=["3-classes", "4-classes-strings", "5-classes"])
@pytest.mark.parametrize("make", [linear_svc, logistic], ids=["LinearSVC", "LogisticRegression"])
def test_one_vs_rest_matches_sklearn(mc, n_classes, strings, seed, make):
    X, y = dataset(seed, n_classes, strings)
    X_new, _ = dataset(seed + 50, n_classes, std=4.0, n=40)
    oracle = SklearnOvR(make()).fit(X, y)
    model = mc.OneVsRestClassifier(make()).fit(X, y)
    assert_same_labels(model.classes_, oracle.classes_, msg="classes_ (the sorted distinct labels)")
    assert_close(model.decision_function(X_new), oracle.decision_function(X_new), rtol=1e-7, atol=1e-9,
                 msg=f"decision_function: column k is the score of the model 'class k against the rest' "
                     f"(scikit-learn OneVsRestClassifier, {make.__name__})")
    assert_same_labels(model.predict(X_new), oracle.predict(X_new),
                       msg="predict: the class with the highest score (scikit-learn OneVsRestClassifier)")


def test_one_vs_rest_uses_predict_proba_when_there_is_no_decision_function(mc):
    X, y = dataset(3, 4)
    X_new, _ = dataset(53, 4, std=4.0, n=30)
    model = mc.OneVsRestClassifier(GaussianNB()).fit(X, y)
    expected = np.column_stack([GaussianNB().fit(X, (y == c).astype(int)).predict_proba(X_new)[:, 1]
                                for c in np.unique(y)])
    assert_close(model.decision_function(X_new), expected, rtol=1e-9, atol=1e-12,
                 msg="without decision_function, column k is predict_proba(X)[:, 1] of the model of class k")
    # probabilities can saturate at 1.0 for two classes: scikit-learn breaks those ties differently (last class)
    unique_max = (expected == expected.max(axis=1, keepdims=True)).sum(axis=1) == 1
    assert_same_labels(np.asarray(model.predict(X_new))[unique_max],
                       SklearnOvR(GaussianNB()).fit(X, y).predict(X_new)[unique_max],
                       msg="predict with GaussianNB where the highest probability is unique "
                           "(scikit-learn OneVsRestClassifier)")


def test_one_vs_rest_ties_go_to_the_first_class(mc):
    X, y = dataset(8, 4, strings=True)
    model = mc.OneVsRestClassifier(Flat()).fit(X, y)
    first = np.unique(y)[0]
    assert_same_labels(model.predict(X[:6]), np.full(6, first),
                       msg=f"every class has the same score: the first class of classes_ ({first!s}) wins, as with "
                           f"np.argmax (scikit-learn's OneVsRestClassifier keeps the last one instead)")


def test_one_vs_rest_trains_one_model_per_class_against_the_rest(mc):
    X, y = dataset(4, 4, strings=True)
    model = mc.OneVsRestClassifier(Recorder()).fit(X, y)
    classes = np.unique(y).tolist()
    assert len(model.estimators_) == 4, f"expected 4 models (one per class), got {len(model.estimators_)}"
    for k, (c, est) in enumerate(zip(classes, model.estimators_)):
        assert np.array_equal(est.X_seen_, X), f"model {k} (class {c!r}) must be trained on every sample of X"
        assert_same_labels(est.y_seen_, (y == c).astype(int),
                           msg=f"model {k} must learn class {c!r} coded 1 against all the other classes coded 0")


def test_one_vs_rest_copies_the_estimator(mc):
    X, y = dataset(5, 3)
    base = linear_svc()
    model = mc.OneVsRestClassifier(base).fit(X, y)
    assert not hasattr(base, "coef_"), "the estimator passed to __init__ must never be fitted: fit deep copies of it"
    ids = {id(est) for est in model.estimators_}
    assert len(ids) == 3 and id(base) not in ids, (
        "expected 3 different fitted copies in estimators_ (copy.deepcopy), not the same object reused")


def test_one_vs_rest_fit_returns_self_and_score_is_the_accuracy(mc):
    X, y = dataset(6, 3, strings=True)
    X_new, y_new = dataset(56, 3, strings=True, std=4.0, n=60)
    model = mc.OneVsRestClassifier(linear_svc())
    assert model.fit(X, y) is model, "fit must return the fitted classifier itself (return self)"
    score = model.score(X_new, y_new)
    assert isinstance(score, float), f"score must be a Python float, got {type(score).__name__}"
    assert_close(score, metrics.accuracy_score(y_new, model.predict(X_new)), msg="score is the accuracy of predict")


def test_one_vs_rest_predict_returns_labels_of_classes_(mc):
    X, y = dataset(7, 5, strings=True)
    pred = mc.OneVsRestClassifier(logistic()).fit(X, y).predict(X)
    assert set(np.asarray(pred).tolist()) <= set(y.tolist()), (
        f"expected predictions taken from classes_ (names such as 'setosa'), got {_short(np.unique(pred))}")


@pytest.mark.parametrize("estimator, labels, why", [
    (linear_svc(), [0, 1, 0, 1, 0, 1], "only 2 classes (use the binary classifier directly)"),
    (PredictOnly(), [0, 1, 2, 0, 1, 2], "an estimator with neither decision_function nor predict_proba"),
], ids=["two-classes", "no-scores"])
def test_one_vs_rest_rejects_invalid_inputs(mc, estimator, labels, why):
    X = np.arange(12, dtype=float).reshape(6, 2)
    assert_raises_value_error(mc.OneVsRestClassifier(estimator).fit, X, np.array(labels), why=why)


# ================================================================== OneVsOneClassifier (7.23)
@pytest.mark.parametrize("n_classes, strings, seed", [(3, False, 10), (4, True, 11), (5, False, 12)],
                         ids=["3-classes", "4-classes-strings", "5-classes"])
@pytest.mark.parametrize("make", [linear_svc, logistic], ids=["LinearSVC", "LogisticRegression"])
def test_one_vs_one_matches_sklearn(mc, n_classes, strings, seed, make):
    X, y = dataset(seed, n_classes, strings)
    X_new, _ = dataset(seed + 50, n_classes, std=4.0, n=60)
    oracle = SklearnOvO(make()).fit(X, y)
    model = mc.OneVsOneClassifier(make()).fit(X, y)
    assert_same_labels(model.classes_, oracle.classes_, msg="classes_ (the sorted distinct labels)")
    votes = np.asarray(model.votes(X_new))
    assert_same_labels(votes, np.round(oracle.decision_function(X_new)).astype(int),
                       msg=f"votes (scikit-learn OneVsOneClassifier with {make.__name__}: decision_function rounded)")
    unique_max = (votes == votes.max(axis=1, keepdims=True)).sum(axis=1) == 1
    assert_same_labels(np.asarray(model.predict(X_new))[unique_max], oracle.predict(X_new)[unique_max],
                       msg="predict where the maximum vote is unique (scikit-learn OneVsOneClassifier)")


def test_one_vs_one_pairs_are_in_lexicographic_order(mc):
    X, y = dataset(13, 5)
    model = mc.OneVsOneClassifier(Recorder()).fit(X, y)
    expected = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    pairs = [tuple(int(v) for v in pair) for pair in model.pairs_]
    assert pairs == expected, f"expected pairs_ = {expected} (K(K-1)/2 = 10 pairs, i < j), got {pairs}"
    assert len(model.estimators_) == 10, f"expected 10 models (one per pair), got {len(model.estimators_)}"


def test_one_vs_one_each_model_sees_only_its_two_classes(mc):
    X, y = dataset(14, 4, strings=True)
    model = mc.OneVsOneClassifier(Recorder()).fit(X, y)
    classes = np.unique(y).tolist()
    for (i, j), est in zip(model.pairs_, model.estimators_):
        keep = (y == classes[i]) | (y == classes[j])
        assert np.array_equal(est.X_seen_, X[keep]), (
            f"the model of the pair ({classes[i]!r}, {classes[j]!r}) must be trained on the {int(keep.sum())} samples "
            f"of these two classes only, got {len(est.X_seen_)} samples")
        assert_same_labels(est.y_seen_, (y[keep] == classes[j]).astype(int),
                           msg=f"pair ({classes[i]!r}, {classes[j]!r}): class j = {classes[j]!r} coded 1, "
                               f"class i = {classes[i]!r} coded 0")


def test_one_vs_one_votes_are_integers_summing_to_the_number_of_duels(mc):
    X, y = dataset(15, 4)
    votes = np.asarray(mc.OneVsOneClassifier(linear_svc()).fit(X, y).votes(X))
    assert votes.shape == (len(X), 4), f"expected votes of shape ({len(X)}, 4), got {votes.shape}"
    assert votes.dtype.kind in "iu", f"expected integer vote counts, got dtype {votes.dtype}"
    assert (votes.sum(axis=1) == 6).all(), (
        f"expected every row to sum to K(K-1)/2 = 6 duels, got sums {_short(np.unique(votes.sum(axis=1)))}")


def test_one_vs_one_ties_go_to_the_smallest_class_index(mc):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]])   # 1st feature = class
    y = np.array([0, 0, 1, 1, 2, 2])
    model = mc.OneVsOneClassifier(Cyclic()).fit(X, y)
    assert_same_labels(np.asarray(model.votes(X)), np.ones((6, 3), dtype=int),
                       msg="votes of the three duels (0, 1) → 0, (0, 2) → 2, (1, 2) → 1")
    assert_same_labels(model.predict(X), np.zeros(6, dtype=int),
                       msg="a three-way tie must go to the smallest class index (np.argmax keeps the first)")


def test_one_vs_one_copies_the_estimator(mc):
    X, y = dataset(16, 3)
    base = linear_svc()
    model = mc.OneVsOneClassifier(base).fit(X, y)
    assert not hasattr(base, "coef_"), "the estimator passed to __init__ must never be fitted: fit deep copies of it"
    ids = {id(est) for est in model.estimators_}
    assert len(ids) == 3 and id(base) not in ids, (
        "expected 3 different fitted copies in estimators_ (copy.deepcopy), not the same object reused")


def test_one_vs_one_fit_returns_self_and_score_is_the_accuracy(mc):
    X, y = dataset(17, 4, strings=True)
    X_new, y_new = dataset(67, 4, strings=True, std=4.0, n=60)
    model = mc.OneVsOneClassifier(logistic())
    assert model.fit(X, y) is model, "fit must return the fitted classifier itself (return self)"
    score = model.score(X_new, y_new)
    assert isinstance(score, float), f"score must be a Python float, got {type(score).__name__}"
    assert_close(score, metrics.accuracy_score(y_new, model.predict(X_new)), msg="score is the accuracy of predict")


def test_one_vs_one_rejects_two_classes(mc):
    X = np.arange(12, dtype=float).reshape(6, 2)
    assert_raises_value_error(mc.OneVsOneClassifier(linear_svc()).fit, X, np.array([0, 1, 0, 1, 0, 1]),
                              why="only 2 classes (use the binary classifier directly)")
