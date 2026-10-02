"""Tests of mylearn.model_selection (chapter 8): oracle tests and properties.

    pytest tests/test_ch08_model_selection.py              # your code (mon_travail/mylearn/model_selection.py)
    pytest tests/test_ch08_model_selection.py --impl=ref   # the reference

Oracles: scikit-learn (``train_test_split`` for the sizes and the stratified counts,
``KFold`` and ``StratifiedKFold`` for the folds, ``cross_val_score`` with the same
splits and the same scorer, ``LinearRegression``, ``NearestCentroid`` and
``LogisticRegression`` as estimators) and NumPy (the round-robin rule written in the
docstring of ``stratified_kfold_indices``). Small estimators written in this file
record what they see (rows used by ``fit``, rows scored, fitted or not).
Every test name starts with the name of what it tests, so that each exercise runs its
own group: ``-k "test_train_test_split_"`` (8.13), ``"test_kfold_indices_"`` (8.14),
``"test_stratified_kfold_indices_"`` (8.21), ``"test_clone_ or test_cross_val_score_"``
(8.22). The tests of one exercise never call the functions of another one (your
``cross_val_score`` may call your ``kfold_indices`` and ``clone``, as its docstring says).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import copy
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn import model_selection as skms
from sklearn.base import clone as sklearn_clone
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestCentroid as SklearnNearestCentroid


@pytest.fixture
def ms(mylearn_module):
    return mylearn_module("model_selection")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for the first line of a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    if arr.dtype.kind in "biuf":
        text = np.array2string(arr, precision=digits, separator=", ", threshold=14,
                               edgeitems=3, max_line_width=10**6)
    elif arr.ndim == 1 and arr.size > 12:      # a long list of labels: its first and last elements only
        text = (repr(arr[:3].tolist())[:-1] + ", ..., " + repr(arr[-3:].tolist())[1:]).replace("np.str_(", "(")
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
    if not np.allclose(got, want, rtol=rtol, atol=atol):
        have, need = _short(result), _short(expected)
        if have == need:
            have, need = _short(result, 17), _short(expected, 17)
        bad = np.flatnonzero(~np.isclose(got, want, rtol=rtol, atol=atol))
        where = f" (first difference at index {int(bad[0])})" if got.ndim == 1 and bad.size else ""
        _fail(msg, f"expected {need}, got {have}{where}", data)


def assert_same(result, expected, msg="", data=""):
    """Two integer or label arrays equal element by element; the message gives the first difference."""
    got = np.asarray(result)
    want = np.asarray(expected)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {_short(want)}, got {_short(got)}",
              data)
    bad = np.flatnonzero(got != want)
    if bad.size:
        i = int(bad[0])
        _fail(msg, f"expected {_short(want)}, got {_short(got)} ({bad.size} difference(s), the first at index "
                   f"{i}: expected {want.ravel()[i].item()!r}, got {got.ravel()[i].item()!r})", data)


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


def call(function, *args, expected="a result", hint="", **kwargs):
    """Call the learner's function; an unexpected exception becomes a failure that says what was expected."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}"
                             + (f" ({hint})" if hint else "")) from None


def split_parts(result, n_arrays, name="train_test_split"):
    """The 2 * n_arrays parts returned by train_test_split, checked to be NumPy arrays."""
    if not isinstance(result, (list, tuple)):
        raise AssertionError(f"expected a list [a_train, a_test, ...], got an object of type {type(result).__name__}")
    if len(result) != 2 * n_arrays:
        raise AssertionError(f"expected {2 * n_arrays} arrays for {n_arrays} input array(s) "
                             f"([a_train, a_test, b_train, b_test, ...]), got {len(result)}")
    for k, part in enumerate(result):
        if not isinstance(part, np.ndarray):
            raise AssertionError(f"expected NumPy arrays, got {type(part).__name__} at position {k} "
                                 f"(convert every input with np.asarray)")
    return list(result)


def pairs(result, n_splits, name):
    """The n_splits (train_idx, val_idx) pairs of a k-fold function, checked to be integer arrays."""
    if not isinstance(result, (list, tuple)):
        raise AssertionError(f"expected a list of {n_splits} pairs (train_idx, val_idx), got an object of type "
                             f"{type(result).__name__} (return a list, not a generator)")
    if len(result) != n_splits:
        raise AssertionError(f"expected {n_splits} pairs (train_idx, val_idx), one per fold, got {len(result)}")
    out = []
    for j, pair in enumerate(result):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise AssertionError(f"expected each element to be a pair (train_idx, val_idx), got {type(pair).__name__} "
                                 f"at position {j}")
        train, val = (np.asarray(part) for part in pair)
        for part, what in ((train, "train_idx"), (val, "val_idx")):
            if part.ndim != 1 or (part.size and part.dtype.kind not in "iu"):
                raise AssertionError(f"expected {what} of split {j} to be a 1-D array of integer indices, got "
                                     f"shape {part.shape} and dtype {part.dtype}")
        out.append((train, val))
    return out


def check_partition(splits, n, name):
    """Validation folds disjoint and covering range(n); each train set is the complement of its fold."""
    every = np.sort(np.concatenate([val for _, val in splits]))
    if not np.array_equal(every, np.arange(n)):
        missing = np.setdiff1d(np.arange(n), every)
        repeated = np.unique(every[np.flatnonzero(np.diff(every) == 0)]) if every.size > 1 else every[:0]
        raise AssertionError(f"expected the validation folds to cover each index 0..{n - 1} exactly once, got "
                             f"{missing.size} missing index(es) {_short(missing)} and repeated one(s) {_short(repeated)}")
    for j, (train, val) in enumerate(splits):
        expected = np.setdiff1d(np.arange(n), val)
        if not np.array_equal(np.sort(train), expected):
            raise AssertionError(f"expected train_idx of split {j} to be every index except its validation fold "
                                 f"({expected.size} indices), got {train.size} indices "
                                 f"({np.intersect1d(train, val).size} of them in the validation fold)")


PENGUIN_COUNTS = (152, 124, 68)                     # Adelie, Gentoo, Chinstrap in the raw Palmer Penguins file


def penguin_like_labels(seed, names=("Adelie", "Gentoo", "Chinstrap")):
    """344 labels with the class counts of the 344 penguins, in a random order."""
    y = np.repeat(np.array(names), PENGUIN_COUNTS)
    return np.random.default_rng(seed).permutation(y)


# ================================================================== train_test_split (8.13)
def test_train_test_split_docstring_example(ms):
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    X_train, X_test, y_train, y_test = split_parts(
        call(ms.train_test_split, X, y, test_size=0.4, shuffle=False, expected="4 arrays"), 2)
    assert_same(X_test, [[6, 7], [8, 9]], msg="X_test of the docstring example (shuffle=False: the last rows)")
    assert_same(y_train, [0, 1, 0], msg="y_train of the docstring example")
    assert_same(X_train, [[0, 1], [2, 3], [4, 5]], msg="X_train of the docstring example")
    assert_same(y_test, [1, 0], msg="y_test of the docstring example")


SIZE_CASES = [(10, 0.25), (333, 0.25), (344, 0.2), (344, 0.25), (7, 0.5), (100, 0.3), (50, 0.1), (1000, 0.15),
              (9, 0.33), (8, 0.25), (20, 0.5), (40, 0.125), (10, 3), (344, 69), (5, 1), (5, 4)]


@pytest.mark.parametrize("n, test_size", SIZE_CASES, ids=[f"n{n}-{t}" for n, t in SIZE_CASES])
def test_train_test_split_test_size_matches_sklearn(ms, n, test_size):
    data = np.arange(n)
    _, oracle_test = skms.train_test_split(data, test_size=test_size, random_state=0)
    train, test = split_parts(call(ms.train_test_split, data, test_size=test_size,
                                   rng=np.random.default_rng(0), expected="[a_train, a_test]"), 1)
    rule = ("an int test_size is the number of test samples" if isinstance(test_size, int)
            else "n_test = ceil(test_size * n), exactly (an integer product stays as it is)")
    if len(test) != len(oracle_test) or len(train) != n - len(oracle_test):
        raise AssertionError(f"expected {len(oracle_test)} test and {n - len(oracle_test)} train samples for "
                             f"n = {n} and test_size = {test_size} ({rule}), got {len(test)} and {len(train)}")


def test_train_test_split_parts_are_disjoint_and_cover_everything(ms):
    for seed, n in [(0, 10), (1, 101), (2, 344)]:
        data = np.arange(n)
        train, test = split_parts(call(ms.train_test_split, data, test_size=0.3, rng=np.random.default_rng(seed),
                                       expected="[a_train, a_test]"), 1)
        both = np.sort(np.concatenate([train, test]))
        everything = np.arange(n)                      # not `data`: a function that shuffles it in place changes it
        if not np.array_equal(both, everything):
            raise AssertionError(f"expected train and test to share no row and to contain every row once (n = {n}), "
                                 f"got {np.intersect1d(train, test).size} row(s) in both parts and "
                                 f"{np.setdiff1d(everything, both).size} row(s) in neither")


def test_train_test_split_rows_stay_aligned(ms):
    n = 60
    X = np.arange(3 * n).reshape(n, 3)          # row i is [3i, 3i + 1, 3i + 2]
    y = np.arange(n) * 10                       # the label of row i is 10 i
    names = np.array([f"s{i}" for i in range(n)])
    X_train, X_test, y_train, y_test, n_train, n_test = split_parts(
        call(ms.train_test_split, X, y, names, test_size=0.25, rng=np.random.default_rng(3), expected="6 arrays"), 3)
    for Xp, yp, np_, part in ((X_train, y_train, n_train, "train"), (X_test, y_test, n_test, "test")):
        assert_same(yp, Xp[:, 0] // 3 * 10, msg=f"y_{part}[i] must be the label of row X_{part}[i] (rows of X and y "
                                               "must be taken with the same indices)")
        assert_same(np_, np.array([f"s{i}" for i in Xp[:, 0] // 3]), msg=f"third array, {part} part: same rows")


def test_train_test_split_same_seed_gives_the_same_split(ms):
    data = np.arange(100)
    first = split_parts(call(ms.train_test_split, data, rng=np.random.default_rng(7), expected="2 arrays"), 1)
    second = split_parts(call(ms.train_test_split, data, rng=np.random.default_rng(7), expected="2 arrays"), 1)
    other = split_parts(call(ms.train_test_split, data, rng=np.random.default_rng(8), expected="2 arrays"), 1)
    assert_same(second[1], first[1], msg="two calls with np.random.default_rng(7) must give the same test part")
    assert not np.array_equal(np.sort(other[1]), np.sort(first[1])), \
        "expected different test parts for the seeds 7 and 8, got the same rows: use the generator rng to shuffle"


def test_train_test_split_shuffle_false_puts_the_last_rows_in_the_test_part(ms):
    data = np.arange(10) * 2
    train, test = split_parts(call(ms.train_test_split, data, test_size=0.3, shuffle=False, expected="2 arrays"), 1)
    assert_same(test, [14, 16, 18], msg="shuffle=False: the last 3 rows, in their order")
    assert_same(train, [0, 2, 4, 6, 8, 10, 12], msg="shuffle=False: the first 7 rows, in their order")


def test_train_test_split_shuffles_by_default(ms):
    data = np.arange(100)
    train, test = split_parts(call(ms.train_test_split, data, rng=np.random.default_rng(0), expected="2 arrays"), 1)
    assert not np.array_equal(np.sort(test), np.arange(75, 100)), \
        "expected a random test part by default (shuffle=True), got the last 25 rows"


def test_train_test_split_without_rng_is_random(ms):
    data = np.arange(200)
    first = split_parts(call(ms.train_test_split, data, expected="2 arrays"), 1)[1]
    second = split_parts(call(ms.train_test_split, data, expected="2 arrays"), 1)[1]
    assert not np.array_equal(np.sort(first), np.sort(second)), \
        ("expected two different splits with rng=None (np.random.default_rng() without a seed), got the same test "
         "rows twice")


@pytest.mark.parametrize("test_size", [0.2, 0.25, 0.3, 0.1])
def test_train_test_split_stratify_matches_sklearn_counts(ms, test_size):
    y = penguin_like_labels(0)
    n = len(y)
    n_test = math.ceil(test_size * n)
    _, oracle_test = skms.train_test_split(y, test_size=test_size, stratify=y, random_state=0)
    for seed in range(5):
        train, test = split_parts(call(ms.train_test_split, y, test_size=test_size, stratify=y,
                                       rng=np.random.default_rng(seed), expected="2 arrays"), 1)
        if len(test) != n_test:
            raise AssertionError(f"expected {n_test} test samples in total with stratify (n_test = ceil(test_size * n) "
                                 f"for test_size = {test_size}), got {len(test)}")
        for name in ("Adelie", "Gentoo", "Chinstrap"):
            got, want = int(np.sum(test == name)), int(np.sum(oracle_test == name))
            if abs(got - want) > 1:
                raise AssertionError(f"expected about {want} '{name}' in the test part (within 1 of scikit-learn, "
                                     f"{n_test} test samples, class proportions kept), got {got}")
        if not np.array_equal(np.sort(np.concatenate([train, test])), np.sort(y)):
            raise AssertionError("expected train and test to contain the labels of every row exactly once with stratify")


def test_train_test_split_stratify_keeps_each_class_proportion(ms):
    rng = np.random.default_rng(11)
    for counts, test_size in [((90, 10), 0.25), ((5, 5, 5, 3), 0.4), ((152, 124, 68), 0.2), ((7, 2, 31), 0.3),
                              ((50, 50), 0.5)]:
        y = rng.permutation(np.repeat(np.arange(len(counts)), counts))
        X = np.arange(len(y)).reshape(-1, 1)
        n_test = math.ceil(test_size * len(y))
        X_train, X_test, y_train, y_test = split_parts(
            call(ms.train_test_split, X, y, test_size=test_size, stratify=y, rng=np.random.default_rng(5),
                 expected="4 arrays"), 2)
        if X_test.ndim != 2 or y_test.ndim != 1:
            raise AssertionError(f"expected [X_train, X_test, y_train, y_test] (scikit-learn order: both parts of X, "
                                 f"then both parts of y), got arrays of dimensions "
                                 f"{[a.ndim for a in (X_train, X_test, y_train, y_test)]}")
        assert_same(y_test, y[X_test[:, 0]], msg="with stratify, the rows of X and y stay aligned")
        if len(y_test) != n_test:
            raise AssertionError(f"expected {n_test} test samples in total for class counts {counts} and test_size = "
                                 f"{test_size} (n_test = ceil(test_size * n); per class, the floor of the exact share, "
                                 f"then one more for the largest remainders, so that the quotas add up to n_test), "
                                 f"got {len(y_test)}")
        for c, count in enumerate(counts):
            exact = count * n_test / len(y)
            got = int(np.sum(y_test == c))
            if got not in (math.floor(exact), math.ceil(exact)):
                raise AssertionError(f"expected {math.floor(exact)} or {math.ceil(exact)} test samples of class {c} "
                                     f"({count} of {len(y)} samples, {n_test} test samples: {count} * {n_test} / "
                                     f"{len(y)} = {exact:.2f}), got {got}")


def test_train_test_split_stratify_is_random_within_each_class(ms):
    y = penguin_like_labels(1)
    tests = [split_parts(call(ms.train_test_split, np.arange(len(y)), test_size=0.2, stratify=y,
                              rng=np.random.default_rng(seed), expected="2 arrays"), 1)[1] for seed in (0, 1)]
    assert not np.array_equal(np.sort(tests[0]), np.sort(tests[1])), \
        "expected different test rows for two seeds with stratify (draw the rows of each class with rng)"


def test_train_test_split_accepts_lists(ms):
    X = [[0, 1], [2, 3], [4, 5], [6, 7]]
    y = ["a", "b", "a", "b"]
    X_train, X_test, y_train, y_test = split_parts(
        call(ms.train_test_split, X, y, test_size=0.5, shuffle=False, expected="4 arrays"), 2)
    assert_same(X_test, [[4, 5], [6, 7]], msg="lists as input: X_test")
    assert_same(y_train, ["a", "b"], msg="lists as input: y_train")


def test_train_test_split_does_not_modify_its_inputs(ms):
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20) % 3
    X0, y0 = X.copy(), y.copy()
    call(ms.train_test_split, X, y, test_size=0.3, rng=np.random.default_rng(0), expected="4 arrays")
    call(ms.train_test_split, X, y, test_size=0.3, stratify=y, rng=np.random.default_rng(0), expected="4 arrays")
    assert np.array_equal(X, X0) and np.array_equal(y, y0), \
        "expected X and y unchanged after the call (shuffle indices, not the arrays in place)"


INVALID_SPLITS = [
    ((), {}, "no array is given"),
    ((np.zeros((10, 2)), np.zeros(9)), {}, "X has 10 rows and y has 9"),
    ((np.arange(10),), {"test_size": 0.0}, "a float test_size must be in (0, 1): 0.0"),
    ((np.arange(10),), {"test_size": 1.0}, "a float test_size must be in (0, 1): 1.0"),
    ((np.arange(10),), {"test_size": -0.2}, "a negative test_size"),
    ((np.arange(10),), {"test_size": 1.5}, "a float test_size above 1"),
    ((np.arange(10),), {"test_size": 0}, "an int test_size of 0 leaves the test part empty"),
    ((np.arange(10),), {"test_size": 10}, "an int test_size of 10 out of 10 rows leaves the train part empty"),
    ((np.arange(10),), {"test_size": 11}, "an int test_size larger than the number of rows"),
    ((np.arange(10),), {"test_size": 0.95}, "ceil(0.95 * 10) = 10 test rows leave the train part empty"),
    ((np.arange(10),), {"stratify": np.arange(10) % 2, "shuffle": False}, "stratify with shuffle=False"),
    ((np.arange(10),), {"stratify": np.array([0] * 9 + [1])}, "class 1 of stratify has a single member"),
]


@pytest.mark.parametrize("args, kwargs, why", INVALID_SPLITS,
                         ids=["no-array", "lengths-differ", "float-0", "float-1", "negative", "above-1", "int-0",
                              "int-n", "int-above-n", "empty-train", "stratify-no-shuffle", "single-member-class"])
def test_train_test_split_rejects_invalid_inputs(ms, args, kwargs, why):
    assert_raises_value_error(ms.train_test_split, *args, why=why, **kwargs)


# ================================================================== kfold_indices (8.14)
def test_kfold_indices_docstring_example(ms):
    splits = pairs(call(ms.kfold_indices, 5, n_splits=2, expected="2 pairs"), 2, "kfold_indices")
    assert_same(splits[0][0], [3, 4], msg="docstring example, split 0: train_idx")
    assert_same(splits[0][1], [0, 1, 2], msg="docstring example, split 0: val_idx (the first fold has the extra sample)")
    assert_same(splits[1][0], [0, 1, 2], msg="docstring example, split 1: train_idx")
    assert_same(splits[1][1], [3, 4], msg="docstring example, split 1: val_idx")


KFOLD_CASES = [(10, 5), (11, 5), (14, 4), (7, 2), (6, 6), (100, 3), (344, 10), (5, 2), (23, 7)]


@pytest.mark.parametrize("n, k", KFOLD_CASES, ids=[f"n{n}-k{k}" for n, k in KFOLD_CASES])
def test_kfold_indices_matches_sklearn_kfold(ms, n, k):
    splits = pairs(call(ms.kfold_indices, n, n_splits=k, expected=f"{k} pairs"), k, "kfold_indices")
    for j, (oracle_train, oracle_val) in enumerate(skms.KFold(n_splits=k).split(np.zeros((n, 1)))):
        train, val = splits[j]
        assert_same(val, oracle_val, msg=f"n = {n}, k = {k}, split {j}: val_idx (scikit-learn KFold, consecutive "
                                         "folds, the first n % k folds have one more sample)")
        assert_same(train, oracle_train, msg=f"n = {n}, k = {k}, split {j}: train_idx (every other index, in "
                                             "increasing order)")


@pytest.mark.parametrize("n, k, sizes", [(13, 5, [3, 3, 3, 2, 2]), (12, 5, [3, 3, 2, 2, 2]), (10, 5, [2] * 5),
                                         (9, 4, [3, 2, 2, 2])], ids=["13-5", "12-5", "10-5", "9-4"])
def test_kfold_indices_fold_sizes(ms, n, k, sizes):
    splits = pairs(call(ms.kfold_indices, n, n_splits=k, expected=f"{k} pairs"), k, "kfold_indices")
    got = [len(val) for _, val in splits]
    assert got == sizes, (f"expected fold sizes {sizes} for n = {n} and k = {k} (the first n % k folds get one extra "
                          f"sample), got {got}")


def test_kfold_indices_shuffled_folds_partition_the_samples(ms):
    for n, k, seed in [(10, 3, 0), (101, 5, 1), (344, 10, 2)]:
        splits = pairs(call(ms.kfold_indices, n, n_splits=k, shuffle=True, rng=np.random.default_rng(seed),
                            expected=f"{k} pairs"), k, "kfold_indices")
        check_partition(splits, n, "kfold_indices")
        sizes = [len(val) for _, val in splits]
        want = [n // k + 1] * (n % k) + [n // k] * (k - n % k)
        assert sizes == want, f"expected fold sizes {want} with shuffle=True too (n = {n}, k = {k}), got {sizes}"


def test_kfold_indices_shuffle_uses_the_rng(ms):
    first = pairs(call(ms.kfold_indices, 50, n_splits=5, shuffle=True, rng=np.random.default_rng(4),
                       expected="5 pairs"), 5, "kfold_indices")
    second = pairs(call(ms.kfold_indices, 50, n_splits=5, shuffle=True, rng=np.random.default_rng(4),
                        expected="5 pairs"), 5, "kfold_indices")
    for j in range(5):
        assert_same(np.sort(second[j][1]), np.sort(first[j][1]),
                    msg=f"two calls with np.random.default_rng(4): the same fold {j}")
    consecutive = all(np.array_equal(np.sort(val), np.arange(val.min(), val.max() + 1)) for _, val in first)
    assert not consecutive, ("expected shuffled folds with shuffle=True (permute the indices once with rng), got "
                             "consecutive blocks of indices")


def test_kfold_indices_shuffle_without_rng_works(ms):
    splits = pairs(call(ms.kfold_indices, 30, n_splits=3, shuffle=True, expected="3 pairs"), 3, "kfold_indices")
    check_partition(splits, 30, "kfold_indices")


def test_kfold_indices_leave_one_out(ms):
    splits = pairs(call(ms.kfold_indices, 4, n_splits=4, expected="4 pairs"), 4, "kfold_indices")
    for j, (train, val) in enumerate(splits):
        assert_same(val, [j], msg=f"n_splits = n_samples = 4: fold {j} holds sample {j} alone")
        assert_same(train, [i for i in range(4) if i != j], msg=f"n_splits = n_samples = 4: train_idx of split {j}")


@pytest.mark.parametrize("n, k, why", [(5, 1, "n_splits = 1 (at least 2 folds)"), (5, 0, "n_splits = 0"),
                                       (5, 6, "6 folds for 5 samples")], ids=["k1", "k0", "k-above-n"])
def test_kfold_indices_rejects_invalid_n_splits(ms, n, k, why):
    assert_raises_value_error(ms.kfold_indices, n, n_splits=k, why=why)


# ================================================================== stratified_kfold_indices (8.21)
def round_robin(y, k):
    """Oracle: the rule of the docstring with NumPy (stable sort by class, fold i = positions i, i + k, ...)."""
    order = np.argsort(np.asarray(y), kind="stable")
    folds = [np.sort(order[i::k]) for i in range(k)]
    return [(np.setdiff1d(np.arange(len(y)), val), val) for val in folds]


def test_stratified_kfold_indices_docstring_example(ms):
    splits = pairs(call(ms.stratified_kfold_indices, [0, 0, 0, 0, 1, 1], n_splits=2, expected="2 pairs"), 2,
                   "stratified_kfold_indices")
    assert_same(splits[0][1], [0, 2, 4], msg="docstring example, split 0: val_idx")
    assert_same(splits[0][0], [1, 3, 5], msg="docstring example, split 0: train_idx")
    assert_same(splits[1][1], [1, 3, 5], msg="docstring example, split 1: val_idx")
    assert_same(splits[1][0], [0, 2, 4], msg="docstring example, split 1: train_idx")


ROUND_ROBIN_CASES = {
    "penguins-k5": (lambda: penguin_like_labels(2), 5),
    "penguins-k10": (lambda: penguin_like_labels(3), 10),
    "ints-k3": (lambda: np.array([2, 0, 1, 1, 0, 2, 2, 0, 1, 0, 0, 2, 1]), 3),
    "imbalanced-k4": (lambda: np.random.default_rng(4).permutation(np.repeat([0, 1], [37, 6])), 4),
    "strings-k2": (lambda: np.array(["b", "a", "c", "a", "b", "c", "c", "a"]), 2),
}


@pytest.mark.parametrize("case", list(ROUND_ROBIN_CASES), ids=list(ROUND_ROBIN_CASES))
def test_stratified_kfold_indices_follows_the_round_robin_rule(ms, case):
    make, k = ROUND_ROBIN_CASES[case]
    y = make()
    splits = pairs(call(ms.stratified_kfold_indices, y, n_splits=k, expected=f"{k} pairs"), k,
                   "stratified_kfold_indices")
    for j, (train, val) in enumerate(round_robin(y, k)):
        assert_same(splits[j][1], val, msg=f"split {j}: val_idx (indices sorted by class with a stable sort, then "
                                           f"fold {j} takes the positions {j}, {j} + {k}, {j} + 2*{k}... of that order; "
                                           "each fold sorted in increasing order)")
        assert_same(splits[j][0], train, msg=f"split {j}: train_idx (every other index, in increasing order)")


@pytest.mark.parametrize("case", list(ROUND_ROBIN_CASES), ids=list(ROUND_ROBIN_CASES))
def test_stratified_kfold_indices_class_counts_match_sklearn(ms, case):
    make, k = ROUND_ROBIN_CASES[case]
    y = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splits = pairs(call(ms.stratified_kfold_indices, y, n_splits=k, expected=f"{k} pairs"), k,
                       "stratified_kfold_indices")
        oracle = list(skms.StratifiedKFold(n_splits=k).split(np.zeros((len(y), 1)), y))
    for label in np.unique(y):
        got = sorted(int(np.sum(y[val] == label)) for _, val in splits)
        want = sorted(int(np.sum(y[val] == label)) for _, val in oracle)
        assert got == want, (f"class {label!r}: expected the per-fold counts {want} (scikit-learn StratifiedKFold, "
                             f"sorted), got {got}")
    sizes = [len(val) for _, val in splits]
    assert max(sizes) - min(sizes) <= 1, f"expected fold sizes that differ by at most 1, got {sizes}"


def test_stratified_kfold_indices_folds_partition_the_samples_and_are_sorted(ms):
    y = penguin_like_labels(5)
    for shuffle in (False, True):
        splits = pairs(call(ms.stratified_kfold_indices, y, n_splits=5, shuffle=shuffle,
                            rng=np.random.default_rng(0), expected="5 pairs"), 5, "stratified_kfold_indices")
        check_partition(splits, len(y), "stratified_kfold_indices")
        for j, (train, val) in enumerate(splits):
            for part, what in ((train, "train_idx"), (val, "val_idx")):
                assert np.all(np.diff(part) > 0), (f"expected {what} of split {j} sorted in increasing order "
                                                   f"(shuffle={shuffle}; np.sort)")


def test_stratified_kfold_indices_shuffle_keeps_the_class_counts(ms):
    y = penguin_like_labels(6)
    plain = pairs(call(ms.stratified_kfold_indices, y, n_splits=5, expected="5 pairs"), 5, "stratified_kfold_indices")
    shuffled = pairs(call(ms.stratified_kfold_indices, y, n_splits=5, shuffle=True, rng=np.random.default_rng(1),
                          expected="5 pairs"), 5, "stratified_kfold_indices")
    again = pairs(call(ms.stratified_kfold_indices, y, n_splits=5, shuffle=True, rng=np.random.default_rng(1),
                       expected="5 pairs"), 5, "stratified_kfold_indices")
    for j in range(5):
        for label in ("Adelie", "Gentoo", "Chinstrap"):
            want, got = int(np.sum(y[plain[j][1]] == label)), int(np.sum(y[shuffled[j][1]] == label))
            assert got == want, (f"fold {j}, class '{label}': expected {want} samples with shuffle=True as without "
                                 f"(shuffle the indices within each class only, then deal them), got {got}")
        assert_same(again[j][1], shuffled[j][1], msg=f"two calls with np.random.default_rng(1): the same fold {j}")
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(plain, shuffled)), \
        "expected other samples in the folds with shuffle=True (shuffle the indices of each class with rng)"


def test_stratified_kfold_indices_warns_when_a_class_is_too_small(ms):
    y = np.array([0] * 10 + [1] * 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call(ms.stratified_kfold_indices, y, n_splits=3, expected="3 pairs and a UserWarning")
    if not any(issubclass(w.category, UserWarning) for w in caught):
        raise AssertionError("expected a UserWarning (warnings.warn(..., UserWarning)) when a class has fewer members "
                             "than n_splits (here class 1 has 2 members and n_splits = 3), got no warning")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            ms.stratified_kfold_indices(np.array([0] * 10 + [1] * 3), n_splits=3)
        except UserWarning as exc:
            raise AssertionError(f"expected no warning when every class has at least n_splits members, got: {exc}") \
                from None


@pytest.mark.parametrize("y, k, why", [([0, 1, 0, 1], 1, "n_splits = 1"), ([0, 1, 0, 1], 5, "5 folds for 4 samples")],
                         ids=["k1", "k-above-n"])
def test_stratified_kfold_indices_rejects_invalid_n_splits(ms, y, k, why):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_raises_value_error(ms.stratified_kfold_indices, y, n_splits=k, why=why)


def test_stratified_kfold_indices_does_not_modify_y(ms):
    y = penguin_like_labels(7)
    y0 = y.copy()
    call(ms.stratified_kfold_indices, y, n_splits=4, shuffle=True, rng=np.random.default_rng(0), expected="4 pairs")
    assert np.array_equal(y, y0), "expected y unchanged after the call (sort or shuffle copies of the indices)"


# ================================================================== clone (8.22)
class Toy:
    """A small estimator following the convention: __init__ only stores its arguments."""

    def __init__(self, alpha=1.0, degrees=(1, 2), options=None):
        self.alpha = alpha
        self.degrees = degrees
        self.options = options

    def fit(self, X, y):
        self.coef_ = float(np.mean(y)) * self.alpha
        self.n_seen_ = len(X)
        self._cache = {"n": len(X)}
        return self

    def predict(self, X):
        return np.full(len(X), self.coef_)


class NoParam:
    """An estimator without hyperparameters."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self


class Renamed:
    """An estimator that breaks the convention: it stores alpha under another name."""

    def __init__(self, alpha=1.0):
        self.strength = alpha


def test_clone_returns_a_new_object_of_the_same_class(ms):
    toy = Toy(alpha=0.3)
    new = call(ms.clone, toy, expected="a new Toy")
    assert new is not toy, "expected a new object, got the estimator itself (clone(e) is e)"
    assert type(new) is Toy, f"expected an object of the same class (Toy), got {type(new).__name__}"


def test_clone_keeps_the_hyperparameters(ms):
    toy = Toy(alpha=0.3, degrees=[1, 2, 5], options={"scale": True})
    new = call(ms.clone, toy, expected="a new Toy")
    got = {name: getattr(new, name, "<missing>") for name in ("alpha", "degrees", "options")}
    want = {"alpha": 0.3, "degrees": [1, 2, 5], "options": {"scale": True}}
    assert got == want, f"expected the hyperparameters {want}, got {got}"


def test_clone_drops_learnt_and_private_attributes(ms):
    toy = Toy(alpha=2.0).fit(np.zeros((4, 1)), np.array([1.0, 2.0, 3.0, 4.0]))
    new = call(ms.clone, toy, expected="a new, unfitted Toy",
               hint="pass only the attributes that neither start nor end with '_' to the class")
    left = [name for name in ("coef_", "n_seen_", "_cache") if hasattr(new, name)]
    assert not left, f"expected an unfitted estimator (no learnt or private attribute), got {left}"
    assert new.alpha == 2.0, f"expected alpha = 2.0 after cloning a fitted Toy, got {new.alpha!r}"


def test_clone_deep_copies_the_hyperparameters(ms):
    toy = Toy(degrees=[1, 2], options={"weights": [1, 1]})
    new = call(ms.clone, toy, expected="a new Toy")
    if getattr(new, "degrees", None) != [1, 2] or getattr(new, "options", None) != {"weights": [1, 1]}:
        raise AssertionError("expected the clone to have degrees=[1, 2] and options={'weights': [1, 1]}, got "
                             f"degrees={getattr(new, 'degrees', '<missing>')!r}, "
                             f"options={getattr(new, 'options', '<missing>')!r}")
    new.degrees.append(3)
    new.options["weights"].append(5)
    assert toy.degrees == [1, 2] and toy.options == {"weights": [1, 1]}, \
        ("expected the original estimator unchanged when the clone's lists are modified, got "
         f"degrees={toy.degrees}, options={toy.options} (copy.deepcopy each value)")


def test_clone_of_a_sklearn_estimator(ms):
    X, y = make_blobs(n_samples=60, centers=2, random_state=0)
    model = LogisticRegression(C=0.5, max_iter=300).fit(X, y)
    new = call(ms.clone, model, expected="a new LogisticRegression")
    assert type(new) is LogisticRegression, f"expected a LogisticRegression, got {type(new).__name__}"
    assert new.get_params() == sklearn_clone(model).get_params(), \
        f"expected the hyperparameters of sklearn.base.clone, got {new.get_params()}"
    assert not hasattr(new, "coef_"), "expected an unfitted LogisticRegression (no coef_), got a fitted one"


def test_clone_of_an_estimator_without_hyperparameters(ms):
    fitted = NoParam().fit(np.zeros((3, 1)), np.array([1.0, 2.0, 3.0]))
    new = call(ms.clone, fitted, expected="a new NoParam")
    assert type(new) is NoParam and not hasattr(new, "mean_"), \
        f"expected a new, unfitted NoParam, got {type(new).__name__} with attributes {sorted(vars(new))}"


def test_clone_propagates_the_type_error_of_a_badly_written_estimator(ms):
    try:
        ms.clone(Renamed(alpha=0.5))
    except TypeError:
        return
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"expected a TypeError from Renamed(strength=...), got {type(exc).__name__}: {exc}") \
            from None
    raise AssertionError("expected a TypeError: Renamed stores alpha as self.strength, so Renamed(strength=0.5) "
                         "must fail (call type(estimator)(**params), without catching the error)")


def test_clone_does_not_modify_the_estimator(ms):
    toy = Toy(alpha=0.7, degrees=[3]).fit(np.zeros((2, 1)), np.array([1.0, 3.0]))
    before = copy.deepcopy(vars(toy))
    call(ms.clone, toy, expected="a new Toy")
    assert vars(toy) == before, f"expected the estimator unchanged after clone, got {vars(toy)} instead of {before}"


# ================================================================== cross_val_score (8.22)
class Recorder:
    """Records every fit: the object fitted and the row numbers it saw (column 0 of X)."""

    log: list = []

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y):
        self.rows_ = np.asarray(X)[:, 0].astype(int).copy()
        self.mean_ = float(np.mean(y)) + self.shift
        Recorder.log.append(self)
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    def score(self, X, y):
        return float(-np.mean((self.predict(X) - np.asarray(y)) ** 2))


class NoScore:
    """An estimator with fit and predict but no score method."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def regression_data(seed, n=60):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.3 * rng.normal(size=n) + 4.0
    return X, y


def scores_of(ms, *args, **kwargs):
    result = call(ms.cross_val_score, *args, expected="an array of scores", **kwargs)
    if not isinstance(result, np.ndarray):
        raise AssertionError(f"expected a NumPy array of scores, got an object of type {type(result).__name__} "
                             "(np.asarray(scores, dtype=float))")
    return result


def test_cross_val_score_docstring_example(ms):
    X = np.array([[0.0], [10.0], [1.0], [11.0], [2.0], [12.0]])
    y = np.array([0, 1, 0, 1, 0, 1])
    assert_close(scores_of(ms, SklearnNearestCentroid(), X, y, cv=3), [1.0, 1.0, 1.0],
                 msg="docstring example with scikit-learn's NearestCentroid (scoring=None: its score method)")
    accuracy = lambda est, X, y: float(np.mean(est.predict(X) == y))  # noqa: E731
    assert_close(scores_of(ms, SklearnNearestCentroid(), X, y, cv=3, scoring=accuracy), [1.0, 1.0, 1.0],
                 msg="docstring example with a scorer")


@pytest.mark.parametrize("k", [2, 3, 5, 7])
def test_cross_val_score_matches_sklearn_for_a_regression(ms, k):
    X, y = regression_data(k)
    want = skms.cross_val_score(LinearRegression(), X, y, cv=skms.KFold(n_splits=k))
    assert_close(scores_of(ms, LinearRegression(), X, y, cv=k), want, rtol=1e-9, atol=1e-12,
                 msg=f"R² of LinearRegression on {k} folds (scikit-learn cross_val_score with cv=KFold({k}): "
                     "consecutive folds, no shuffle)")


@pytest.mark.parametrize("k", [3, 4, 10])
def test_cross_val_score_matches_sklearn_for_a_classification(ms, k):
    X, y = make_blobs(n_samples=90, centers=3, cluster_std=4.0, random_state=k)
    want = skms.cross_val_score(SklearnNearestCentroid(), X, y, cv=skms.KFold(n_splits=k))
    assert_close(scores_of(ms, SklearnNearestCentroid(), X, y, cv=k), want, atol=1e-12,
                 msg=f"accuracy of NearestCentroid on {k} folds (scikit-learn cross_val_score with cv=KFold({k}))")


def test_cross_val_score_with_a_list_of_splits(ms):
    X, y = make_blobs(n_samples=75, centers=3, cluster_std=5.0, random_state=1)
    splits = list(skms.StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
    want = skms.cross_val_score(SklearnNearestCentroid(), X, y, cv=splits)
    assert_close(scores_of(ms, SklearnNearestCentroid(), X, y, cv=splits), want, atol=1e-12,
                 msg="cv given as a list of (train_idx, val_idx) pairs: one score per pair, in their order")


def test_cross_val_score_with_a_scorer(ms):
    X, y = make_blobs(n_samples=80, centers=3, cluster_std=4.5, random_state=2)
    scorer = lambda est, X, y: balanced_accuracy_score(y, est.predict(X))  # noqa: E731
    want = skms.cross_val_score(SklearnNearestCentroid(), X, y, cv=skms.KFold(n_splits=4), scoring=scorer)
    assert_close(scores_of(ms, SklearnNearestCentroid(), X, y, cv=4, scoring=scorer), want, atol=1e-12,
                 msg="scoring(fitted_estimator, X_val, y_val) instead of estimator.score")


def test_cross_val_score_returns_one_float_per_split(ms):
    X, y = regression_data(9, n=30)
    result = scores_of(ms, LinearRegression(), X, y, cv=6)
    assert result.shape == (6,), f"expected shape (6,), one score per split, got {result.shape}"
    assert result.dtype.kind == "f", f"expected float scores, got dtype {result.dtype}"


def test_cross_val_score_fits_a_fresh_clone_on_each_split(ms):
    n, k = 20, 4
    X = np.column_stack([np.arange(n), np.zeros(n)])        # column 0 = row number
    y = np.arange(n, dtype=float)
    Recorder.log = []
    original = Recorder(shift=1.0)
    scores_of(ms, original, X, y, cv=k)
    fitted = Recorder.log
    assert len(fitted) == k, f"expected {k} calls to fit, one per split, got {len(fitted)}"
    assert len({id(model) for model in fitted}) == k, \
        "expected a new estimator for every split (clone(estimator) inside the loop), got the same object twice"
    assert all(model is not original for model in fitted) and not hasattr(original, "rows_"), \
        "expected the estimator passed in to stay unfitted (fit clones of it, never the estimator itself)"
    assert all(model.shift == 1.0 for model in fitted), "expected clones with the same hyperparameters (shift = 1.0)"
    for j, (train_idx, _) in enumerate(skms.KFold(n_splits=k).split(X)):
        assert_same(fitted[j].rows_, train_idx, msg=f"split {j}: rows used by fit (every fold except fold {j})")


def test_cross_val_score_scores_the_fitted_clone_on_the_validation_rows(ms):
    n, k = 15, 3
    X = np.column_stack([np.arange(n), np.ones(n)])
    y = np.arange(n, dtype=float)
    seen = []

    def scorer(est, X_val, y_val):
        seen.append((est, np.asarray(X_val)[:, 0].astype(int), np.asarray(y_val)))
        return 0.0

    Recorder.log = []
    scores_of(ms, Recorder(), X, y, cv=k, scoring=scorer)
    assert len(seen) == k, f"expected {k} calls to the scorer, one per split, got {len(seen)}"
    for j, (_, val_idx) in enumerate(skms.KFold(n_splits=k).split(X)):
        est, rows, labels = seen[j]
        assert hasattr(est, "rows_"), f"split {j}: expected the scorer to receive the fitted clone, got an unfitted one"
        assert_same(rows, val_idx, msg=f"split {j}: rows given to the scorer (the validation fold {j})")
        assert_same(labels, y[val_idx], msg=f"split {j}: labels given to the scorer (y of the validation fold)")


def test_cross_val_score_accepts_dataframes_and_lists(ms):
    X, y = regression_data(3, n=40)
    frame = pd.DataFrame(X, columns=["a", "b", "c"], index=np.random.default_rng(0).permutation(40) + 100)
    series = pd.Series(y, index=frame.index)
    want = skms.cross_val_score(LinearRegression(), X, y, cv=skms.KFold(n_splits=4))
    got = call(ms.cross_val_score, LinearRegression(), frame, series, cv=4, expected="4 scores",
               hint="X is a DataFrame here: convert X and y with np.asarray before indexing rows")
    assert_close(got, want, atol=1e-12, msg="X as a DataFrame and y as a Series with a shuffled index "
                                            "(np.asarray, then rows by position)")
    got = call(ms.cross_val_score, LinearRegression(), X.tolist(), y.tolist(), cv=4, expected="4 scores")
    assert_close(got, want, atol=1e-12, msg="X and y as lists")


def test_cross_val_score_rejects_invalid_inputs(ms):
    X, y = regression_data(4, n=12)
    assert_raises_value_error(ms.cross_val_score, NoScore(), X, y, cv=3,
                              why="scoring=None and the estimator has no score method")
    assert_raises_value_error(ms.cross_val_score, LinearRegression(), X, y[:-1], cv=3,
                              why="X has 12 rows and y has 11 values")
