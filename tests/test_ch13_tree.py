"""Tests of mylearn.tree (chapter 13): oracle tests and properties.

    pytest tests/test_ch13_tree.py              # your code (mon_travail/mylearn/tree.py)
    pytest tests/test_ch13_tree.py --impl=ref   # the reference

Oracles: hand-computed values, ``scipy.stats.entropy(p, base=2)``, ``np.average``, a brute-force
search written with NumPy, and scikit-learn 1.6+ (``DecisionTreeClassifier`` and
``DecisionTreeRegressor``: ``tree_.feature``, ``tree_.threshold``, ``tree_.impurity``,
``tree_.n_node_samples``, ``tree_.value``, ``predict``, ``predict_proba``, ``get_depth``,
``get_n_leaves``, ``feature_importances_``), plus properties (``sample_weight=2`` equals
duplicating every sample, same ``random_state`` gives the same tree...).
scikit-learn converts X to float32 and visits the features in a random order: the comparisons
use float32-representable continuous data, scikit-learn trees checked to be the same for ten
``random_state`` values (no tie between two features) and without any split of zero gain
(scikit-learn may split with a zero gain, mylearn makes a leaf), at every split a best candidate
that beats the others by more than 1e-9 (no near-tie that rounding could break either way: chosen
offline), and query points farther than 1e-6 from every threshold.
Every test name starts with the name of the function or class it tests, so that each exercise
runs its own group:
``-k "test_gini_impurity_ or test_entropy_impurity_ or test_variance_impurity_ or test_split_gain_"``
(13.20), ``"test_best_split_"`` (13.22), ``"test_decision_tree_classifier_ and not ensemble"``
(13.24), ``"test_decision_tree_classifier_ensemble_"`` (13.25: ``sample_weight``,
``max_features``, ``splitter="random"``, ``random_state``, ``feature_importances_``),
``"test_decision_tree_regressor_ and not ensemble"`` (13.26) and
``"test_decision_tree_regressor_ensemble_"`` (chapter 14). The tests of one group never call the
functions of another group.
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest
import scipy.stats
from sklearn import metrics as skm
from sklearn import tree as sktree


@pytest.fixture
def tr(mylearn_module):
    return mylearn_module("tree")


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
        _fail(msg, f"expected {want[bad]!r} at index {bad}, got {got[bad]!r}",
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
    """A learned attribute (root_, classes_...), with a clear message when fit did not create it."""
    if not hasattr(model, name):
        raise AssertionError(f"expected fit to create the attribute {name}, but the fitted "
                             f"{type(model).__name__} has no {name} (attributes: {sorted(vars(model))})")
    return getattr(model, name)


def _public(model) -> dict:
    return {key: value for key, value in vars(model).items() if not key.startswith("_") and not key.endswith("_")}


def _stored_as_given(got, want) -> bool:
    """A hyperparameter kept exactly as passed: same type and same value."""
    return type(got) is type(want) and got == want


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
                       f"(an attribute like root_ = None makes an unfitted tree look fitted)")
    fit(model, X, y)
    added = sorted(set(_public(model)) - set(want))
    assert not added, (f"expected fit to add only learned attributes (names ending with _, like root_) or private "
                       f"ones (starting with _), got {added}: clone (ch. 8) would take them for hyperparameters")
    changed = {key: vars(model)[key] for key in want if not _stored_as_given(vars(model)[key], want[key])}
    assert not changed, (f"expected fit to leave the hyperparameters unchanged ({want}), got {changed} (write what "
                         "fit computes in a learned attribute)")


def check_returns_self(model, X, y):
    result = call(model.fit, X, y, expected="fit to return the tree itself")
    assert result is model, (f"expected {type(model).__name__}.fit(X, y) to return the tree itself (return self), "
                             f"got {type(result).__name__}: DecisionTreeClassifier().fit(X, y).predict(X) chains")


def sk(cls, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(*args, **kwargs)


def _f32(values):
    """Values exactly representable in float32 (scikit-learn converts X to float32)."""
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def classification_data(seed, n, d, n_classes):
    """Continuous float32-representable features; classes = quantile bins of a noisy linear score."""
    rng = np.random.default_rng(seed)
    X = _f32(rng.normal(size=(n, d)))
    score = X @ rng.normal(size=d) + 0.6 * rng.normal(size=n)
    y = np.digitize(score, np.quantile(score, np.arange(1, n_classes) / n_classes))
    return X, y


def regression_data(seed, n, d):
    rng = np.random.default_rng(seed)
    X = _f32(rng.normal(size=(n, d)))
    y = np.sin(2.0 * X[:, 0]) + X @ rng.normal(size=d) * 0.5 + 0.3 * rng.normal(size=n)
    return X, y


def weights_for(seed, n):
    """Random weights in [3/8, 3], multiples of 1/8: their sums are exact, so a pure node has an impurity of exactly 0
    (with arbitrary floats, 1 - (sum of the weights of a class / total)^2 can come out as 2e-16 instead of 0)."""
    return np.random.default_rng(seed + 1000).integers(3, 25, size=n) / 8.0


def _np_impurity(y, w, criterion):
    """The impurity of a node, computed with NumPy / SciPy (the oracle of the brute-force search)."""
    y, w = np.asarray(y), np.asarray(w, dtype=float)
    if criterion == "squared_error":
        mean = np.average(y, weights=w)
        return float(np.average((y - mean) ** 2, weights=w))
    _, codes = np.unique(y, return_inverse=True)
    p = np.bincount(codes.ravel(), weights=w) / w.sum()
    if criterion == "gini":
        return float(1.0 - np.sum(p ** 2))
    return float(scipy.stats.entropy(p, base=2))


def _np_gain(y, mask, w, criterion):
    y, mask, w = np.asarray(y), np.asarray(mask, dtype=bool), np.asarray(w, dtype=float)
    wl, wr = w[mask].sum(), w[~mask].sum()
    if wl == 0 or wr == 0:
        return 0.0
    return (_np_impurity(y, w, criterion) - wl / (wl + wr) * _np_impurity(y[mask], w[mask], criterion)
            - wr / (wl + wr) * _np_impurity(y[~mask], w[~mask], criterion))


def _brute_force(X, y, criterion, w, features, min_samples_leaf):
    """Every valid (feature, midpoint threshold, gain), feature by feature, threshold by threshold."""
    candidates = []
    for f in features:
        values = np.unique(X[:, f])
        for low, high in zip(values[:-1], values[1:]):
            t = (low + high) / 2
            mask = X[:, f] <= t
            if mask.sum() >= min_samples_leaf and (~mask).sum() >= min_samples_leaf:
                candidates.append((int(f), float(t), _np_gain(y, mask, w, criterion)))
    return candidates


# ------------------------------------------------------------------ trees: oracle and walking the nodes
def _oracle(cls, X, y, sample_weight=None, **params):
    """A scikit-learn tree that the documented rules reproduce exactly (checked, else the test data are wrong).

    The same tree for ten random_state values (scikit-learn breaks a tie between two features at random,
    mylearn takes the lowest index) and no split of zero gain (scikit-learn may split with a zero gain,
    mylearn makes a leaf; with float weights, scikit-learn's rounded impurity of a pure node, about 1e-16,
    can exceed its threshold 2.2e-16). The data of the tests were also chosen offline so that, at every
    split, the best candidate beats all the others by more than 1e-9 (no near-tie that two correct
    implementations could break differently by rounding).
    """
    trees = [sk(cls, random_state=seed, **params).fit(X, y, sample_weight=sample_weight) for seed in range(10)]
    first = trees[0].tree_
    for other in trees[1:]:
        if not (np.array_equal(other.tree_.feature, first.feature)
                and np.array_equal(other.tree_.threshold, first.threshold)):
            raise AssertionError("test setup: these data have a tie between two features (scikit-learn breaks it at "
                                 "random); choose other test data")
    for i in range(first.node_count):
        left, right = first.children_left[i], first.children_right[i]
        if left == -1:
            continue
        w, wl, wr = first.weighted_n_node_samples[[i, left, right]]
        gain = first.impurity[i] - wl / w * first.impurity[left] - wr / w * first.impurity[right]
        if gain <= 1e-9 * first.impurity[0]:      # also a pure node whose rounded impurity is about 1e-16
            raise AssertionError("test setup: scikit-learn split a node with a zero gain; choose other test data")
    return trees[0]


def _queries(seed, d, oracle, m=200):
    """Random float32-representable points farther than 1e-6 from every threshold of the oracle."""
    X = _f32(np.random.default_rng(seed + 500).normal(size=(m, d)) * 1.3)
    t = oracle.tree_
    keep = np.ones(m, dtype=bool)
    for i in np.flatnonzero(t.children_left != -1):
        keep &= np.abs(X[:, t.feature[i]] - t.threshold[i]) > 1e-6
    return X[keep]


def _root(model):
    root = learned(model, "root_")
    for field in ("feature", "threshold", "left", "right", "value", "n_samples", "impurity"):
        if not hasattr(root, field):
            raise AssertionError(f"expected root_ to be a Node (fields feature, threshold, left, right, value, "
                                 f"n_samples, impurity), got a {type(root).__name__} without {field}")
    return root


def _nodes(model):
    """The learner's nodes in pre-order (node, left subtree, right subtree), with their path from the root."""
    out, stack = [], [(_root(model), "root", 0)]
    while stack:
        node, path, depth = stack.pop()
        out.append((node, path, depth))
        if len(out) > 100_000:
            raise AssertionError("expected a finite tree, got more than 100000 nodes (a node is its own child?)")
        if node.left is not None:
            if node.right is None:
                raise AssertionError(f"expected the node at {path} to have two children (or none), got a left child "
                                     "only")
            stack.append((node.right, path + ".right", depth + 1))
            stack.append((node.left, path + ".left", depth + 1))
    return out


def _sk_nodes(oracle):
    t = oracle.tree_
    out, stack = [], [0]
    while stack:
        i = stack.pop()
        out.append(i)
        if t.children_left[i] != -1:
            stack.extend([t.children_right[i], t.children_left[i]])
    return out


def _describe(node):
    return "a leaf" if node.left is None else f"a split on feature {node.feature} at {float(node.threshold)!r}"


def assert_same_tree(model, oracle, label, values=True):
    """The learner's tree has scikit-learn's nodes: same splits, n_samples, impurities and values, node by node."""
    t = oracle.tree_
    regression = t.value.shape[1:] == (1, 1)
    for (node, path, _), i in zip(_nodes(model), _sk_nodes(oracle)):
        if t.children_left[i] == -1:
            want = "a leaf"
        else:
            want = f"a split on feature {t.feature[i]} at {float(t.threshold[i])!r}"
        if (node.left is None) != (t.children_left[i] == -1) or (
                node.left is not None and (node.feature != t.feature[i]
                                           or not np.isclose(node.threshold, t.threshold[i], rtol=0, atol=1e-9))):
            hint = ("stop only when the node is pure, at max_depth, below min_samples_split or without a valid "
                    "split" if node.left is None else
                    "best_split: midpoints between distinct values, the largest gain, ties to the lowest feature "
                    "then the lowest threshold; min_samples_leaf")
            if node.left is not None and t.children_left[i] != -1 and node.feature == t.feature[i]:
                hint = f"threshold off by {abs(node.threshold - t.threshold[i]):.1e}: " + hint
            _fail(label, f"expected {want} at {path} (as scikit-learn), got {_describe(node)} ({hint})")
        if node.n_samples != t.n_node_samples[i]:
            _fail(label, f"expected n_samples = {t.n_node_samples[i]} at {path} (number of training samples that "
                         f"reach the node, not their weight), got {node.n_samples!r}")
        assert_close(node.impurity, t.impurity[i], rtol=1e-9, atol=1e-12,
                     msg=f"{label}: impurity of the node at {path} (as scikit-learn's tree_.impurity)")
        if values:
            want_value = t.value[i, 0, :1] if regression else t.value[i, 0]
            what = ("the weighted mean target, shape (1,)" if regression else
                    "the weighted class proportions, aligned on classes_")
            assert_close(node.value, want_value, rtol=1e-9, atol=1e-12,
                         msg=f"{label}: value of the node at {path} ({what})")


def _structure(model):
    return [(path, node.feature, None if node.threshold is None else float(node.threshold))
            for node, path, _ in _nodes(model)]


def _node_rows(model, X):
    """(node, path, rows of X that reach it) for every node, routing x[feature] <= threshold to the left."""
    out, stack = [], [(_root(model), "root", np.arange(len(X)))]
    while stack:
        node, path, rows = stack.pop()
        out.append((node, path, rows))
        if node.left is not None:
            go_left = X[rows, node.feature] <= node.threshold
            stack.append((node.right, path + ".right", rows[~go_left]))
            stack.append((node.left, path + ".left", rows[go_left]))
    return out


# ================================================================== impurities and split_gain (13.20)
@pytest.mark.parametrize("y, w, want", [
    pytest.param([0, 0, 1, 1], None, 0.5, id="two_balanced"),
    pytest.param(["a", "a", "a", "b"], None, 0.375, id="strings"),
    pytest.param([0, 1], [3.0, 1.0], 0.375, id="weighted"),
])
def test_gini_impurity_docstring_examples(tr, y, w, want):
    got = call(tr.gini_impurity, y, sample_weight=w)
    assert_close(got, want, msg=f"gini_impurity({y}, sample_weight={w})")


@pytest.mark.parametrize("y, want, why", [
    pytest.param([4, 4, 4], 0.0, "a pure node", id="pure"),
    pytest.param([0, 1, 2], 2 / 3, "three balanced classes: 1 - 1/3", id="three_balanced"),
    pytest.param([0, 1, 2, 3] * 3, 0.75, "four balanced classes: 1 - 1/4", id="four_balanced"),
    pytest.param([1, 1, 2], 4 / 9, "proportions 2/3 and 1/3: 1 - 4/9 - 1/9", id="two_thirds"),
    pytest.param([-1.5, 2.0, 2.0, 2.0, 7.0], 0.56, "proportions 0.2, 0.6, 0.2 (float labels)", id="float_labels"),
])
def test_gini_impurity_hand_computed_values(tr, y, want, why):
    assert_close(call(tr.gini_impurity, y), want, msg=f"gini_impurity({y}) ({why})")


@pytest.mark.parametrize("seed, n_classes, weighted", [(0, 2, False), (1, 3, False), (2, 5, True), (3, 3, True)],
                         ids=["2_classes", "3_classes", "5_classes_weighted", "3_classes_weighted"])
def test_gini_impurity_matches_sklearn_root_impurity(tr, seed, n_classes, weighted):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=50)
    w = rng.uniform(0.1, 3.0, size=50) if weighted else None
    X = rng.normal(size=(50, 2))
    want = sk(sktree.DecisionTreeClassifier, max_depth=1).fit(X, y, sample_weight=w).tree_.impurity[0]
    assert_close(call(tr.gini_impurity, y, sample_weight=w), want, rtol=1e-10,
                 msg=f"gini_impurity of {n_classes} random classes{' with random weights' if weighted else ''} "
                     "(scikit-learn's root impurity; p_k = weight of class k / total weight)")


def test_gini_impurity_weights_act_as_repetitions(tr):
    y = np.array([0, 1, 1, 2, 0])
    w = np.array([2, 1, 3, 1, 1])
    repeated = np.repeat(y, w)
    counts = np.bincount(repeated) / len(repeated)
    assert_close(call(tr.gini_impurity, y, sample_weight=w.astype(float)), 1 - np.sum(counts ** 2),
                 msg="gini_impurity with integer weights = gini_impurity of the labels repeated that many times")


def test_gini_impurity_class_of_zero_weight_counts_for_nothing(tr):
    assert_close(call(tr.gini_impurity, [0, 0, 1], sample_weight=[1.0, 1.0, 0.0]), 0.0,
                 msg="gini_impurity([0, 0, 1], sample_weight=[1, 1, 0]) (class 1 has no weight: the node is pure)")
    assert_close(call(tr.gini_impurity, [0, 1, 2], sample_weight=[1.0, 1.0, 0.0]), 0.5,
                 msg="gini_impurity([0, 1, 2], sample_weight=[1, 1, 0]) (two classes of equal weight)")


def test_gini_impurity_returns_a_float(tr):
    assert_python_float(call(tr.gini_impurity, np.array([0, 1, 1])), "gini_impurity")


BAD_WEIGHTS = [
    pytest.param([], None, "y is empty", id="empty"),
    pytest.param([0, 1, 1], [1.0, -1.0, 2.0], "a negative weight", id="negative_weight"),
    pytest.param([0, 1, 1], [0.0, 0.0, 0.0], "the weights sum to 0", id="zero_total"),
    pytest.param([0, 1, 1], [1.0, 2.0], "2 weights for 3 labels", id="wrong_shape"),
]


@pytest.mark.parametrize("y, w, why", BAD_WEIGHTS)
def test_gini_impurity_rejects_bad_inputs(tr, y, w, why):
    assert_raises(ValueError, tr.gini_impurity, np.array(y), sample_weight=None if w is None else np.array(w), why=why)


@pytest.mark.parametrize("y, want", [
    pytest.param([0, 0, 1, 1], 1.0, id="two_balanced"),
    pytest.param([0, 0, 0, 1], 0.8112781244591328, id="three_to_one"),
])
def test_entropy_impurity_docstring_examples(tr, y, want):
    assert_close(call(tr.entropy_impurity, y), want, rtol=1e-12, msg=f"entropy_impurity({y})")


@pytest.mark.parametrize("y, want, why", [
    pytest.param([3, 3, 3], 0.0, "a pure node", id="pure"),
    pytest.param([0, 1, 2, 3], 2.0, "four balanced classes: log2(4) = 2 bits (in BITS: np.log2)", id="four_balanced"),
    pytest.param(["x"] * 7 + ["y"], 0.5435644431995964, "proportions 7/8 and 1/8", id="strings"),
])
def test_entropy_impurity_hand_computed_values(tr, y, want, why):
    assert_close(call(tr.entropy_impurity, y), want, rtol=1e-12, msg=f"entropy_impurity({y}) ({why})")


@pytest.mark.parametrize("seed, n_classes, weighted", [(4, 2, False), (5, 4, False), (6, 3, True), (7, 6, True)],
                         ids=["2_classes", "4_classes", "3_classes_weighted", "6_classes_weighted"])
def test_entropy_impurity_matches_scipy(tr, seed, n_classes, weighted):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=40)
    w = rng.uniform(0.1, 3.0, size=40) if weighted else np.ones(40)
    p = np.bincount(y, weights=w, minlength=n_classes) / w.sum()
    assert_close(call(tr.entropy_impurity, y, sample_weight=w if weighted else None),
                 scipy.stats.entropy(p, base=2), rtol=1e-10,
                 msg=f"entropy_impurity of {n_classes} random classes{' with random weights' if weighted else ''} "
                     "(scipy.stats.entropy(p, base=2): -sum p_k log2 p_k)")


def test_entropy_impurity_matches_sklearn_root_impurity(tr):
    rng = np.random.default_rng(8)
    y = rng.integers(0, 3, size=60)
    w = rng.uniform(0.2, 2.0, size=60)
    X = rng.normal(size=(60, 2))
    want = sk(sktree.DecisionTreeClassifier, criterion="entropy", max_depth=1).fit(X, y, sample_weight=w).tree_.impurity[0]
    assert_close(call(tr.entropy_impurity, y, sample_weight=w), want, rtol=1e-10,
                 msg="entropy_impurity with weights (scikit-learn's root impurity for criterion='entropy', in bits)")


def test_entropy_impurity_class_of_zero_weight_gives_no_nan(tr):
    got = call(tr.entropy_impurity, [0, 0, 1, 2], sample_weight=[1.0, 1.0, 0.0, 2.0])
    assert_close(got, 1.0, msg="entropy_impurity([0, 0, 1, 2], sample_weight=[1, 1, 0, 2]) (class 1 has weight 0: "
                               "0 log 0 = 0, sum only the classes of positive proportion)")


def test_entropy_impurity_returns_a_float(tr):
    assert_python_float(call(tr.entropy_impurity, np.array([0, 1, 1])), "entropy_impurity")


@pytest.mark.parametrize("y, w, why", BAD_WEIGHTS)
def test_entropy_impurity_rejects_bad_inputs(tr, y, w, why):
    assert_raises(ValueError, tr.entropy_impurity, np.array(y), sample_weight=None if w is None else np.array(w),
                  why=why)


def test_variance_impurity_docstring_example(tr):
    assert_close(call(tr.variance_impurity, [1.0, 2.0, 3.0, 4.0]), 1.25, msg="variance_impurity([1, 2, 3, 4])")


def test_variance_impurity_is_the_population_variance(tr):
    assert_close(call(tr.variance_impurity, np.array([0.0, 2.0, 4.0, 6.0])), 5.0,
                 msg="variance_impurity([0, 2, 4, 6]) (divide by n = 4, not n - 1)")


def test_variance_impurity_of_constant_targets_is_zero(tr):
    assert_close(call(tr.variance_impurity, np.full(5, 2.5)), 0.0, msg="variance_impurity of five targets 2.5")


@pytest.mark.parametrize("seed", [9, 10, 11])
def test_variance_impurity_matches_np_average(tr, seed):
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=3.0, scale=2.0, size=30)
    w = rng.uniform(0.0, 2.0, size=30)
    mean = np.average(y, weights=w)
    assert_close(call(tr.variance_impurity, y, sample_weight=w), np.average((y - mean) ** 2, weights=w), rtol=1e-10,
                 msg="variance_impurity with weights: sum w (y - ybar)^2 / sum w, with the WEIGHTED mean ybar")


def test_variance_impurity_matches_sklearn_root_impurity(tr):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 2))
    y = rng.normal(size=40) * 3.0 + 1.0
    w = rng.uniform(0.2, 2.0, size=40)
    want = sk(sktree.DecisionTreeRegressor, max_depth=1).fit(X, y, sample_weight=w).tree_.impurity[0]
    assert_close(call(tr.variance_impurity, y, sample_weight=w), want, rtol=1e-8,
                 msg="variance_impurity with weights (scikit-learn's root impurity of a DecisionTreeRegressor)")


def test_variance_impurity_returns_a_float(tr):
    assert_python_float(call(tr.variance_impurity, np.array([1.0, 2.0])), "variance_impurity")


@pytest.mark.parametrize("y, w, why", [
    pytest.param([], None, "y is empty", id="empty"),
    pytest.param([1.0, 2.0], [1.0, -0.5], "a negative weight", id="negative_weight"),
    pytest.param([1.0, 2.0], [0.0, 0.0], "the weights sum to 0", id="zero_total"),
    pytest.param([1.0, 2.0, 3.0], [1.0, 1.0], "2 weights for 3 targets", id="wrong_shape"),
])
def test_variance_impurity_rejects_bad_inputs(tr, y, w, why):
    assert_raises(ValueError, tr.variance_impurity, np.array(y, dtype=float),
                  sample_weight=None if w is None else np.array(w), why=why)


@pytest.mark.parametrize("criterion, want", [("gini", 0.5), ("entropy", 1.0)])
def test_split_gain_docstring_examples(tr, criterion, want):
    got = call(tr.split_gain, [0, 0, 1, 1], [True, True, False, False], criterion=criterion)
    assert_close(got, want, msg=f"split_gain([0, 0, 1, 1], [True, True, False, False], criterion={criterion!r})")


@pytest.mark.parametrize("y, mask, criterion, w, want, why", [
    pytest.param([0, 0, 1, 1], [True, False, True, False], "gini", None, 0.0,
                 "each child keeps the proportions of the parent", id="useless_split"),
    pytest.param([0, 1, 1, 0], [True, False, False, False], "gini", None, 1 / 6,
                 "0.5 - 1/4 * 0 - 3/4 * 4/9", id="gini_one_vs_three"),
    pytest.param([0, 0, 0, 1], [True, True, False, False], "entropy", None, 0.31127812445913283,
                 "0.8113 - 1/2 * 0 - 1/2 * 1", id="entropy"),
    pytest.param([1.0, 2.0, 3.0, 4.0], [True, True, False, False], "squared_error", None, 1.0,
                 "1.25 - 1/2 * 0.25 - 1/2 * 0.25", id="squared_error"),
    pytest.param([0, 0, 1, 1], [True, False, True, False], "gini", [1.0, 3.0, 1.0, 1.0], 1 / 36,
                 "parent: weights 4 and 2, Gini 4/9; left [0, 1] weights [1, 1]: 0.5 (weight 2/6); right [0, 1] "
                 "weights [3, 1]: 0.375 (weight 4/6)", id="weighted"),
])
def test_split_gain_hand_computed_values(tr, y, mask, criterion, w, want, why):
    got = call(tr.split_gain, np.array(y), np.array(mask), criterion=criterion,
               sample_weight=None if w is None else np.array(w))
    assert_close(got, want, rtol=1e-12, atol=1e-15,
                 msg=f"split_gain({y}, {mask}, {criterion!r}, sample_weight={w}) ({why})")


@pytest.mark.parametrize("criterion", ["gini", "entropy", "squared_error"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_split_gain_matches_sklearn_children_impurities(tr, criterion, weighted):
    rng = np.random.default_rng(13)
    X = _f32(rng.normal(size=(60, 1)))
    w = rng.uniform(0.2, 3.0, size=60) if weighted else None
    if criterion == "squared_error":
        y = X[:, 0] * 2.0 + rng.normal(size=60)
        oracle = sk(sktree.DecisionTreeRegressor, max_depth=1).fit(X, y, sample_weight=w).tree_
    else:
        y = (X[:, 0] + 0.8 * rng.normal(size=60) > 0).astype(int) + (rng.uniform(size=60) < 0.2)
        oracle = sk(sktree.DecisionTreeClassifier, criterion=criterion, max_depth=1).fit(X, y, sample_weight=w).tree_
    mask = X[:, 0] <= oracle.threshold[0]
    n0, nl, nr = oracle.weighted_n_node_samples[:3]
    want = oracle.impurity[0] - nl / n0 * oracle.impurity[1] - nr / n0 * oracle.impurity[2]
    assert_close(call(tr.split_gain, y, mask, criterion=criterion, sample_weight=w), want, rtol=1e-8, atol=1e-12,
                 msg=f"split_gain of scikit-learn's root split ({criterion}{', weighted' if weighted else ''}): "
                     "I(parent) - w_L/w I(left) - w_R/w I(right), with the impurities of tree_.impurity")


@pytest.mark.parametrize("mask", [[True] * 4, [False] * 4], ids=["all_left", "all_right"])
def test_split_gain_with_an_empty_child_is_zero(tr, mask):
    got = call(tr.split_gain, np.array([0, 1, 1, 0]), np.array(mask))
    assert_close(got, 0.0, atol=0.0, msg=f"split_gain([0, 1, 1, 0], {mask}) (one child is empty: the gain is 0.0)")


def test_split_gain_returns_a_float(tr):
    assert_python_float(call(tr.split_gain, np.array([0, 1]), np.array([True, False])), "split_gain")


@pytest.mark.parametrize("kwargs, why", [
    pytest.param({"criterion": "variance"}, "criterion='variance' (only 'gini', 'entropy', 'squared_error')",
                 id="unknown_criterion"),
    pytest.param({"left_mask": np.array([True, False, True])}, "3 mask values for 4 labels", id="mask_shape"),
    pytest.param({"sample_weight": np.ones(3)}, "3 weights for 4 labels", id="weight_shape"),
])
def test_split_gain_rejects_bad_arguments(tr, kwargs, why):
    args = {"y": np.array([0, 0, 1, 1]), "left_mask": np.array([True, True, False, False])}
    args.update(kwargs)
    assert_raises(ValueError, tr.split_gain, why=why, **args)


# ================================================================== best_split (13.22)
def check_split_types(result, what):
    if not (isinstance(result, tuple) and len(result) == 3):
        hint = (" (a split with a positive gain exists here; a child with exactly min_samples_leaf samples is valid)"
                if result is None else "")
        raise AssertionError(f"expected {what} to return a tuple (feature, threshold, gain), got {result!r}{hint}")
    feature, threshold, gain = result
    for value, kind, name in ((feature, int, "feature"), (threshold, float, "threshold"), (gain, float, "gain")):
        if type(value) is not kind:
            raise AssertionError(f"expected {what} to return the {name} as a Python {kind.__name__}, got "
                                 f"{type(value).__name__} {value!r} ({kind.__name__}(...) converts a NumPy number)")


def test_best_split_docstring_example(tr):
    X = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0]])
    result = call(tr.best_split, X, np.array([0, 0, 1, 1]))
    check_split_types(result, "best_split(X, y) of the docstring example")
    assert result == (0, 2.5, 0.5), f"expected best_split(X, y) of the docstring example = (0, 2.5, 0.5), got {result}"


@pytest.mark.parametrize("x, y, want", [
    pytest.param([3.0, 3.0, 4.0, 4.0], [0, 0, 1, 1], 3.5, id="between_3_and_4"),
    pytest.param([1.0, 2.0, 2.0, 5.0], [0, 0, 0, 1], 3.5, id="skips_equal_values"),
    pytest.param([5.0, 1.0, 2.0, 2.0], [1, 0, 0, 0], 3.5, id="unsorted"),
    pytest.param([-1.0, 0.25, 7.0, 7.5], [1, 1, 0, 0], 3.625, id="negative_values"),
])
def test_best_split_threshold_is_the_midpoint_between_distinct_values(tr, x, y, want):
    result = call(tr.best_split, np.array(x).reshape(-1, 1), np.array(y))
    check_split_types(result, f"best_split on x = {x}")
    assert result[:2] == (0, want), (f"expected best_split(x = {x}, y = {y}) to split feature 0 at {want} (the "
                                     f"midpoint between two consecutive DISTINCT sorted values), got {result}")


def _int_data(seed, n, d, criterion, n_values=6):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, n_values, size=(n, d)).astype(float)
    if criterion == "squared_error":
        y = X @ rng.normal(size=d) + rng.normal(size=n)
    else:
        y = (X @ rng.normal(size=d) + rng.normal(size=n) > 0).astype(int) + (rng.uniform(size=n) < 0.15)
    return X, y


@pytest.mark.parametrize("criterion", ["gini", "entropy", "squared_error"])
@pytest.mark.parametrize("seed, n, d, weighted, min_samples_leaf", [
    (20, 30, 3, False, 1), (21, 25, 4, True, 1), (22, 40, 2, False, 6), (23, 35, 3, True, 4), (24, 12, 5, False, 1),
], ids=["n30", "n25_weighted", "n40_leaf6", "n35_weighted_leaf4", "n12_d5"])
def test_best_split_matches_brute_force(tr, criterion, seed, n, d, weighted, min_samples_leaf):
    X, y = _int_data(seed, n, d, criterion)
    w = np.random.default_rng(seed).uniform(0.2, 3.0, size=n) if weighted else np.ones(n)
    candidates = _brute_force(X, y, criterion, w, range(d), min_samples_leaf)
    best = max(gain for _, _, gain in candidates)
    accepted = {(f, t) for f, t, gain in candidates if gain >= best - 1e-9}
    label = (f"best_split({criterion!r}, n={n}, d={d}{', weighted' if weighted else ''}, "
             f"min_samples_leaf={min_samples_leaf}) on integer-valued features")
    result = call(tr.best_split, X, y, criterion=criterion, sample_weight=w if weighted else None,
                  min_samples_leaf=min_samples_leaf)
    if result is None:
        _fail(label, f"expected a split of gain {best:.6g} (brute-force search), got None")
    check_split_types(result, "best_split")
    feature, threshold, gain = result
    if (feature, threshold) not in accepted:
        f, t, g = next(c for c in candidates if (c[0], c[1]) in accepted)
        done = {(c[0], c[1]): c[2] for c in candidates}
        mine = done.get((feature, threshold))
        why = ("this is not a valid candidate (midpoint between distinct values, at least min_samples_leaf samples "
               "in each child)" if mine is None else f"its gain is only {mine:.6g}")
        _fail(label, f"expected the split of largest gain: feature {f} at {t:g} (gain {g:.6g}), got feature "
                     f"{feature} at {threshold:g} ({why})")
    assert_close(gain, best, rtol=1e-9, atol=1e-12, msg=f"{label}: the gain returned with the split")


SKLEARN_ROOT_CASES = [
    pytest.param("gini", False, 30, id="gini"),
    pytest.param("entropy", False, 31, id="entropy"),
    pytest.param("gini", True, 32, id="gini_weighted"),
    pytest.param("squared_error", False, 33, id="squared_error"),
    pytest.param("squared_error", True, 34, id="squared_error_weighted"),
]


@pytest.mark.parametrize("criterion, weighted, seed", SKLEARN_ROOT_CASES)
def test_best_split_matches_sklearn_root(tr, criterion, weighted, seed):
    if criterion == "squared_error":
        X, y = regression_data(seed, 80, 4)
        cls = sktree.DecisionTreeRegressor
        params = {"max_depth": 1}
    else:
        X, y = classification_data(seed, 80, 4, 3)
        cls = sktree.DecisionTreeClassifier
        params = {"max_depth": 1, "criterion": criterion}
    w = weights_for(seed, 80) if weighted else None
    oracle = _oracle(cls, X, y, sample_weight=w, **params).tree_
    label = f"best_split({criterion!r}{', weighted' if weighted else ''}) on 80 continuous samples, 4 features"
    result = call(tr.best_split, X, y, criterion=criterion, sample_weight=w)
    if result is None:
        _fail(label, f"expected feature {oracle.feature[0]} at {oracle.threshold[0]:.8g} (scikit-learn's root), "
                     "got None")
    check_split_types(result, "best_split")
    feature, threshold, gain = result
    if feature != oracle.feature[0] or not np.isclose(threshold, oracle.threshold[0], rtol=0, atol=1e-9):
        _fail(label, f"expected feature {oracle.feature[0]} at {oracle.threshold[0]:.8g} (scikit-learn's root split), "
                     f"got feature {feature} at {threshold:.8g}")
    n0, nl, nr = oracle.weighted_n_node_samples[:3]
    want = oracle.impurity[0] - nl / n0 * oracle.impurity[1] - nr / n0 * oracle.impurity[2]
    assert_close(gain, want, rtol=1e-8, msg=f"{label}: the gain (from scikit-learn's tree_.impurity)")


def test_best_split_ties_go_to_the_lowest_feature_index(tr):
    rng = np.random.default_rng(40)
    good = _f32(rng.normal(size=20))
    y = (good > 0.1).astype(int)
    noise = _f32(rng.normal(size=20))
    X = np.column_stack([noise, good, noise * 2.0, good])           # features 1 and 3: the same perfect split
    result = call(tr.best_split, X, y)
    assert result is not None and result[0] == 1, (
        f"expected feature 1 (features 1 and 3 are identical and give the best gain: ties go to the LOWEST feature "
        f"index), got {result}")
    result = call(tr.best_split, X, y, feature_indices=np.array([3, 1]))
    assert result is not None and result[0] == 1, (
        f"expected feature 1 with feature_indices=[3, 1] (the tie goes to the lowest feature INDEX, whatever the order "
        f"of feature_indices), got {result}")


def test_best_split_ties_go_to_the_lowest_threshold(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 1, 1, 0])               # 1.5 and 3.5 both isolate one 0: gain 1/6 each
    result = call(tr.best_split, X, y)
    assert result is not None and result[:2] == (0, 1.5), (
        f"expected (0, 1.5, 1/6) for x = [1, 2, 3, 4], y = [0, 1, 1, 0] (thresholds 1.5 and 3.5 have the same gain: "
        f"ties go to the LOWEST threshold; keep a candidate only if its gain is strictly larger), got {result}")


def test_best_split_min_samples_leaf_counts_samples_not_weights(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 1, 1, 1])
    w = np.array([5.0, 1.0, 1.0, 1.0])
    result = call(tr.best_split, X, y, sample_weight=w, min_samples_leaf=2)
    assert result is not None and result[:2] == (0, 2.5), (
        f"expected (0, 2.5, ...) with min_samples_leaf=2 and weights [5, 1, 1, 1], got {result} (min_samples_leaf "
        "counts SAMPLES, not weights: 1.5 leaves one sample on the left, of weight 5, which is invalid)")
    assert_close(result[2], _np_gain(y, X[:, 0] <= 2.5, w, "gini"), rtol=1e-12,
                 msg="best_split(min_samples_leaf=2, weights [5, 1, 1, 1]): gain of the split at 2.5")


def test_best_split_respects_min_samples_leaf(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 1, 1, 1, 1, 1])        # the best split (1.5) leaves a single sample on the left
    result = call(tr.best_split, X, y, min_samples_leaf=2)
    assert result is not None and result[:2] == (0, 2.5), (
        f"expected (0, 2.5, ...) with min_samples_leaf=2 (1.5 would leave 1 sample on the left: invalid), got {result}")
    assert_close(result[2], 0.2777777777777778 - 2 / 6 * 0.5, rtol=1e-12,
                 msg="best_split(min_samples_leaf=2): gain of the split at 2.5")


@pytest.mark.parametrize("X, y, kwargs, why", [
    pytest.param([[1.0], [2.0], [3.0]], [1, 1, 1], {}, "the node is pure", id="pure"),
    pytest.param([[2.0, 7.0], [2.0, 7.0], [2.0, 7.0]], [0, 1, 0], {}, "every feature is constant", id="constant"),
    pytest.param([[1.0], [2.0], [3.0], [4.0]], [0, 1, 0, 1], {"min_samples_leaf": 2},
                 "the only valid split (2.5) gives [0, 1] | [0, 1]: gain exactly 0", id="zero_gain"),
    pytest.param([[1.0], [2.0], [3.0]], [0, 1, 0], {"min_samples_leaf": 2},
                 "3 samples cannot give 2 children of 2 samples", id="too_few_samples"),
    pytest.param([[1.0], [2.0], [3.0], [4.0]], [1.5, 1.5, 1.5, 1.5], {"criterion": "squared_error"},
                 "constant targets", id="constant_targets"),
])
def test_best_split_returns_none_without_a_positive_gain(tr, X, y, kwargs, why):
    result = call(tr.best_split, np.array(X), np.array(y), **kwargs)
    assert result is None, f"expected None ({why}: no valid split with a strictly positive gain), got {result}"


def test_best_split_feature_indices_restricts_and_keeps_the_original_indices(tr):
    rng = np.random.default_rng(41)
    X = _f32(rng.normal(size=(30, 4)))
    y = (X[:, 0] > 0).astype(int)                      # feature 0 is perfect
    X[:, 2] = _f32(X[:, 0] + 0.6 * rng.normal(size=30))   # feature 2: second best
    full = call(tr.best_split, X, y)
    assert full is not None and full[0] == 0, f"expected feature 0 (a perfect split) with all the features, got {full}"
    want = max(_brute_force(X, y, "gini", np.ones(30), [1, 2, 3], 1), key=lambda c: c[2])
    result = call(tr.best_split, X, y, feature_indices=np.array([1, 2, 3]))
    assert result is not None and result[:2] == want[:2], (
        f"expected feature {want[0]} at {want[1]:.6g} with feature_indices=[1, 2, 3] (feature 0 is excluded; the "
        f"result gives the index of the feature in X, not in feature_indices), got {result}")


def test_best_split_uses_the_sample_weights(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 1, 0, 0])
    plain = call(tr.best_split, X, y)
    assert plain is not None and plain[:2] == (0, 2.5), f"expected (0, 2.5, ...) without weights, got {plain}"
    result = call(tr.best_split, X, y, sample_weight=np.array([10.0, 1.0, 1.0, 1.0]))
    assert result is not None and result[:2] == (0, 1.5), (
        f"expected (0, 1.5, ...) with sample_weight=[10, 1, 1, 1] (isolating the heavy first sample gives the largest "
        f"WEIGHTED gain), got {result}")
    want = _np_gain(y, X[:, 0] <= 1.5, np.array([10.0, 1.0, 1.0, 1.0]), "gini")
    assert_close(result[2], want, rtol=1e-12, msg="best_split with weights: the weighted gain of the split at 1.5")


def test_best_split_integer_weights_equal_repeated_samples(tr):
    X, y = _int_data(42, 20, 3, "gini", n_values=12)
    w = np.random.default_rng(42).integers(1, 4, size=20)
    weighted = call(tr.best_split, X, y, sample_weight=w.astype(float))
    repeated = call(tr.best_split, np.repeat(X, w, axis=0), np.repeat(y, w))
    assert weighted is not None and repeated is not None and weighted[:2] == repeated[:2], (
        f"expected the same split with integer weights as with the samples repeated that many times, got {weighted} "
        f"with weights and {repeated} with repetitions")
    assert_close(weighted[2], repeated[2], rtol=1e-12, msg="best_split: gain with integer weights = gain with repeats")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_best_split_random_draws_one_uniform_threshold(tr, seed):
    X = np.array([[0.0], [1.0], [2.0], [5.0]])
    y = np.array([0, 0, 1, 1])
    t = np.random.default_rng(seed).uniform(0.0, 5.0)
    want_gain = _np_gain(y, X[:, 0] <= t, np.ones(4), "gini")
    result = call(tr.best_split, X, y, splitter="random", rng=np.random.default_rng(seed))
    assert result is not None and result[0] == 0 and np.isclose(result[1], t, rtol=0, atol=1e-12), (
        f"expected (0, {t:.10g}, ...) with splitter='random', rng=np.random.default_rng({seed}) (a single feature: the "
        f"threshold is rng.uniform(min, max) = rng.uniform(0, 5), not a midpoint), got {result}")
    check_split_types(result, "best_split(splitter='random')")
    assert_close(result[2], want_gain, rtol=1e-12, msg=f"best_split(splitter='random'): gain of the threshold {t:.6g}")


def test_best_split_random_constant_feature_consumes_no_draw(tr):
    X = np.column_stack([np.full(6, 3.0), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    for seed in range(4):
        t = np.random.default_rng(seed).uniform(0.0, 5.0)
        result = call(tr.best_split, X, y, splitter="random", rng=np.random.default_rng(seed))
        assert result is not None and result[0] == 1 and np.isclose(result[1], t, rtol=0, atol=1e-12), (
            f"expected (1, {t:.10g}, ...) with splitter='random', rng=np.random.default_rng({seed}), got {result} "
            "(feature 0 is constant: it is skipped WITHOUT drawing, so the first draw of rng goes to feature 1)")


def test_best_split_random_draws_in_the_order_of_feature_indices(tr):
    X, y = classification_data(45, 60, 4, 2)
    order = [3, 1]
    for seed in range(4):
        rng = np.random.default_rng(seed)
        thresholds = {f: rng.uniform(X[:, f].min(), X[:, f].max()) for f in order}
        gains = {f: _np_gain(y, X[:, f] <= thresholds[f], np.ones(len(y)), "gini") for f in order}
        best = max(order, key=lambda f: (gains[f], -f))
        result = call(tr.best_split, X, y, feature_indices=np.array(order), splitter="random",
                      rng=np.random.default_rng(seed))
        assert result is not None and result[0] == best and np.isclose(result[1], thresholds[best], rtol=0,
                                                                         atol=1e-12), (
            f"expected ({best}, {thresholds[best]:.10g}, ...) with feature_indices=[3, 1], splitter='random', "
            f"rng=np.random.default_rng({seed}), got {result} (draw one threshold per feature in the order of "
            "feature_indices: feature 3 first, then feature 1)")


def test_best_split_random_skips_constant_features(tr):
    rng = np.random.default_rng(43)
    X = np.column_stack([np.full(12, 4.0), _f32(rng.normal(size=12)), np.full(12, -1.0)])
    y = (X[:, 1] > 0).astype(int)
    for seed in range(5):
        result = call(tr.best_split, X, y, splitter="random", rng=np.random.default_rng(seed))
        assert result is not None and result[0] == 1, (
            f"expected feature 1 with splitter='random' (features 0 and 2 are constant: they are skipped), got {result}")


@pytest.mark.parametrize("criterion", ["gini", "squared_error"])
def test_best_split_random_threshold_inside_the_range_and_reproducible(tr, criterion):
    if criterion == "gini":
        X, y = classification_data(44, 50, 5, 2)
    else:
        X, y = regression_data(44, 50, 5)
    for seed in range(8):
        result = call(tr.best_split, X, y, criterion=criterion, splitter="random", rng=np.random.default_rng(seed))
        again = call(tr.best_split, X, y, criterion=criterion, splitter="random", rng=np.random.default_rng(seed))
        assert result == again, (f"expected the same result with the same rng seed ({seed}), got {result} then {again} "
                                 "(draw with the rng argument only)")
        assert result is not None, f"expected a split with splitter='random' (rng seed {seed}), got None"
        feature, threshold, gain = result
        low, high = X[:, feature].min(), X[:, feature].max()
        assert low < threshold < high, (f"expected the random threshold strictly between min = {low:.6g} and max = "
                                        f"{high:.6g} of feature {feature}, got {threshold!r}")
        assert_close(gain, _np_gain(y, X[:, feature] <= threshold, np.ones(len(y)), criterion), rtol=1e-9,
                     msg=f"best_split(splitter='random'): the gain of the returned split (feature {feature})")


def test_best_split_random_keeps_the_best_drawn_threshold(tr):
    X, y = classification_data(45, 40, 2, 2)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        draws = [rng.uniform(X[:, 0].min(), X[:, 0].max()), rng.uniform(X[:, 1].min(), X[:, 1].max())]
        rng = np.random.default_rng(seed)
        reverse = [rng.uniform(X[:, 1].min(), X[:, 1].max()), rng.uniform(X[:, 0].min(), X[:, 0].max())][::-1]
        accepted = []
        for thresholds in (draws, reverse):        # one draw per feature, in the order 0, 1 or 1, 0
            gains = [_np_gain(y, X[:, f] <= thresholds[f], np.ones(40), "gini") for f in (0, 1)]
            best = int(np.argmax(gains))
            accepted.append((best, thresholds[best], gains[best]))
        result = call(tr.best_split, X, y, splitter="random", rng=np.random.default_rng(seed))
        if result is None or not any(result[0] == f and np.isclose(result[1], t, rtol=0, atol=1e-12)
                                     for f, t, _ in accepted):
            f, t, g = accepted[0]
            _fail(f"best_split(splitter='random', rng=np.random.default_rng({seed})) on 2 features",
                  f"expected feature {f} at {t:.8g} (gain {g:.6g}): one threshold drawn per feature, then the drawn "
                  f"threshold of LARGEST gain is kept; got {result}")


@pytest.mark.parametrize("kwargs, why", [
    pytest.param({"criterion": "mse"}, "criterion='mse' is unknown", id="unknown_criterion"),
    pytest.param({"splitter": "worst"}, "splitter='worst' is unknown", id="unknown_splitter"),
    pytest.param({"y": np.array([0, 1, 1])}, "y has 3 labels for 4 samples", id="y_length"),
    pytest.param({"sample_weight": np.ones(5)}, "5 weights for 4 samples", id="weight_length"),
])
def test_best_split_rejects_bad_arguments(tr, kwargs, why):
    args = {"X": np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0]]), "y": np.array([0, 0, 1, 1])}
    args.update(kwargs)
    assert_raises(ValueError, tr.best_split, why=why, **args)


def test_best_split_does_not_modify_its_inputs(tr):
    X, y = classification_data(46, 30, 3, 2)
    w = weights_for(46, 30)
    features = np.array([2, 0])
    saved = (X.copy(), y.copy(), w.copy(), features.copy())
    call(tr.best_split, X, y, sample_weight=w, feature_indices=features, copy_inputs=False)
    call(tr.best_split, X, y, sample_weight=w, feature_indices=features, splitter="random",
         rng=np.random.default_rng(0), copy_inputs=False)
    for name, now, before in zip(("X", "y", "sample_weight", "feature_indices"), (X, y, w, features), saved):
        assert np.array_equal(now, before), f"expected best_split to leave {name} unchanged (sort a copy)"


# ================================================================== DecisionTreeClassifier (13.24)
# (n, d, n_classes, hyperparameters, seed): scikit-learn's tree is checked to be reproducible (see _oracle)
CLASSIFIER_CASES = [
    pytest.param(60, 2, 2, {}, 17, id="n60_d2_full"),
    pytest.param(40, 2, 3, {}, 8, id="n40_d2_3classes_full"),
    pytest.param(60, 2, 2, {"max_depth": 3}, 0, id="max_depth3"),
    pytest.param(120, 3, 3, {"max_depth": 3}, 1, id="n120_max_depth3"),
    pytest.param(200, 4, 2, {"min_samples_split": 12}, 1, id="min_samples_split12"),
    pytest.param(120, 3, 3, {"min_samples_leaf": 5}, 6, id="min_samples_leaf5"),
    pytest.param(40, 2, 3, {"criterion": "entropy"}, 9, id="entropy_full"),
    pytest.param(200, 4, 2, {"criterion": "entropy", "max_depth": 4}, 1, id="entropy_max_depth4"),
]


def test_decision_tree_classifier_docstring_example(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = fit(tr.DecisionTreeClassifier(), X, y)
    root = _root(tree)
    assert (root.feature, root.threshold) == (0, 2.5), (
        f"expected root_.feature, root_.threshold = (0, 2.5), got ({root.feature!r}, {root.threshold!r})")
    assert (call(tree.get_depth), call(tree.get_n_leaves)) == (1, 2), (
        f"expected get_depth(), get_n_leaves() = (1, 2), got ({tree.get_depth()!r}, {tree.get_n_leaves()!r})")
    assert_same_labels(call(tree.predict, np.array([[1.5], [3.5]])), [0, 1], msg="predict([[1.5], [3.5]])")
    assert_close(call(tree.predict_proba, np.array([[1.5], [3.5]])), [[1.0, 0.0], [0.0, 1.0]],
                 msg="predict_proba([[1.5], [3.5]])")


def test_decision_tree_classifier_sample_on_the_threshold_goes_left(tr):
    tree = fit(tr.DecisionTreeClassifier(), np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))
    assert_same_labels(call(tree.predict, np.array([[2.5], [2.5000001]])), [0, 1],
                       msg="predict([[2.5], [2.5000001]]) with the root threshold 2.5 (x <= threshold goes LEFT)")


@pytest.mark.parametrize("n, d, n_classes, params, seed", CLASSIFIER_CASES)
def test_decision_tree_classifier_matches_sklearn(tr, n, d, n_classes, params, seed):
    X, y = classification_data(seed, n, d, n_classes)
    oracle = _oracle(sktree.DecisionTreeClassifier, X, y, **params)
    model = fit(build(tr.DecisionTreeClassifier, **params), X, y)
    label = f"DecisionTreeClassifier({', '.join(f'{k}={v!r}' for k, v in params.items())}) on {n} samples, {d} features"
    assert call(model.get_depth) == oracle.get_depth(), (
        f"{label}: expected get_depth() = {oracle.get_depth()} (as scikit-learn), got {model.get_depth()!r}")
    assert call(model.get_n_leaves) == oracle.get_n_leaves(), (
        f"{label}: expected get_n_leaves() = {oracle.get_n_leaves()} (as scikit-learn), got {model.get_n_leaves()!r}")
    Q = _queries(seed, d, oracle)
    assert_close(call(model.predict_proba, Q), oracle.predict_proba(Q), msg=f"{label}: predict_proba(new points)")
    assert_same_labels(call(model.predict, Q), oracle.predict(Q), msg=f"{label}: predict(new points)")
    assert_close(call(model.predict_proba, X), oracle.predict_proba(X), msg=f"{label}: predict_proba(X_train)")


@pytest.mark.parametrize("n, d, n_classes, params, seed", CLASSIFIER_CASES)
def test_decision_tree_classifier_nodes_match_sklearn(tr, n, d, n_classes, params, seed):
    X, y = classification_data(seed, n, d, n_classes)
    oracle = _oracle(sktree.DecisionTreeClassifier, X, y, **params)
    model = fit(build(tr.DecisionTreeClassifier, **params), X, y)
    assert_same_tree(model, oracle, f"DecisionTreeClassifier({', '.join(f'{k}={v!r}' for k, v in params.items())})")


def test_decision_tree_classifier_full_depth_on_one_feature_matches_sklearn(tr):
    X, y = classification_data(8, 80, 1, 3)
    oracle = _oracle(sktree.DecisionTreeClassifier, X, y)
    model = fit(tr.DecisionTreeClassifier(), X, y)
    assert_same_tree(model, oracle, "DecisionTreeClassifier() grown to purity on 80 samples of one feature")
    Q = _queries(8, 1, oracle)
    assert_same_labels(call(model.predict, Q), oracle.predict(Q), msg="predict(new points), one feature, full depth")


def test_decision_tree_classifier_unlimited_tree_fits_the_training_set(tr):
    X, y = classification_data(51, 80, 3, 3)
    model = fit(tr.DecisionTreeClassifier(), X, y)
    assert_same_labels(call(model.predict, X), y, msg="predict(X_train) of a tree grown until the leaves are pure "
                       "(distinct samples: every training sample is predicted right; a node is a leaf only when pure "
                       "or with FEWER than min_samples_split samples)")
    for node, path, _ in _nodes(model):
        if node.left is None:
            assert_close(node.impurity, 0.0, atol=1e-12, msg=f"impurity of the leaf at {path} (unlimited tree: every "
                         "leaf is pure)")


@pytest.mark.parametrize("max_depth", [1, 2, 4])
def test_decision_tree_classifier_max_depth(tr, max_depth):
    X, y = classification_data(52, 100, 3, 3)
    model = fit(tr.DecisionTreeClassifier(max_depth=max_depth), X, y)
    deepest = max(depth for _, _, depth in _nodes(model))
    assert deepest == max_depth, (f"expected max_depth={max_depth} to give leaves at depth {max_depth} at most (and "
                                  f"some at exactly {max_depth} on these noisy data), got a deepest leaf at {deepest}")
    assert call(model.get_depth) == max_depth, f"expected get_depth() = {max_depth}, got {model.get_depth()!r}"


def test_decision_tree_classifier_min_samples_split(tr):
    X, y = classification_data(53, 100, 3, 2)
    model = fit(tr.DecisionTreeClassifier(min_samples_split=15), X, y)
    for node, path, _ in _nodes(model):
        if node.left is not None:
            assert node.n_samples >= 15, (f"expected only nodes of at least 15 samples to be split "
                                          f"(min_samples_split=15), got a split at {path} with {node.n_samples}")


def test_decision_tree_classifier_node_with_exactly_min_samples_split_samples_is_split(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    model = fit(tr.DecisionTreeClassifier(min_samples_split=4), X, np.array([0, 0, 1, 1]))
    depth = call(model.get_depth)
    assert depth == 1, (f"expected get_depth() = 1 with min_samples_split=4 and 4 samples, got {depth!r} (a node is "
                        "a leaf only with FEWER than min_samples_split samples: 4 samples can be split)")


def test_decision_tree_classifier_min_samples_leaf(tr):
    X, y = classification_data(54, 100, 3, 3)
    model = fit(tr.DecisionTreeClassifier(min_samples_leaf=7), X, y)
    for node, path, _ in _nodes(model):
        if node.left is None:
            assert node.n_samples >= 7, (f"expected every leaf to keep at least 7 samples (min_samples_leaf=7), got "
                                         f"{node.n_samples} at {path}")


def test_decision_tree_classifier_string_labels(tr):
    X, y = classification_data(57, 60, 2, 3)
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    model = fit(tr.DecisionTreeClassifier(max_depth=3), X, names)
    oracle = _oracle(sktree.DecisionTreeClassifier, X, names, max_depth=3)
    assert_same_labels(learned(model, "classes_"), oracle.classes_, msg="classes_ (the sorted distinct labels)")
    Q = _queries(57, 2, oracle)
    assert_same_labels(call(model.predict, Q), oracle.predict(Q), msg="predict with string labels (labels taken "
                       "from classes_, not class indices)")
    assert_close(call(model.predict_proba, Q), oracle.predict_proba(Q), msg="predict_proba with string labels")


def test_decision_tree_classifier_classes_are_sorted_and_values_aligned(tr):
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([9, 9, 2, 5, 5, 5])
    model = fit(tr.DecisionTreeClassifier(max_depth=1), X, y)
    assert_same_labels(learned(model, "classes_"), [2, 5, 9], msg="classes_ (sorted distinct labels)")
    assert learned(model, "n_classes_") == 3, f"expected n_classes_ = 3, got {model.n_classes_!r}"
    assert learned(model, "n_features_in_") == 1, f"expected n_features_in_ = 1, got {model.n_features_in_!r}"
    root = _root(model)
    assert_close(root.value, [1 / 6, 3 / 6, 2 / 6], msg="root_.value (class proportions in the order of classes_ "
                 "[2, 5, 9])")
    assert root.n_samples == 6, f"expected root_.n_samples = 6, got {root.n_samples!r}"
    assert_close(root.impurity, 1 - (1 / 36 + 9 / 36 + 4 / 36), msg="root_.impurity (Gini of the 6 labels)")
    assert (root.feature, root.threshold) == (0, 2.5), (
        f"expected the root split x <= 2.5 (left 9, 9, 2 | right 5, 5, 5), got ({root.feature!r}, {root.threshold!r})")
    assert_close(root.left.value, [1 / 3, 0.0, 2 / 3], msg="root_.left.value (one 2 and two 9s: a zero for the "
                 "absent class 5, in the order of classes_ [2, 5, 9])")
    assert_close(root.right.value, [0.0, 1.0, 0.0], msg="root_.right.value (only 5s)")


def test_decision_tree_classifier_predict_ties_go_to_the_first_class(tr):
    X = np.array([[1.0], [1.0], [1.0], [1.0]])
    y = np.array(["b", "a", "b", "a"])                # identical samples: one leaf with proportions [0.5, 0.5]
    model = fit(tr.DecisionTreeClassifier(), X, y)
    assert_close(call(model.predict_proba, np.array([[1.0]])), [[0.5, 0.5]],
                 msg="predict_proba of a leaf holding two a and two b")
    assert_same_labels(call(model.predict, np.array([[1.0], [0.0]])), ["a", "a"],
                       msg="predict with tied proportions (ties go to the first class of classes_: np.argmax)")


def test_decision_tree_classifier_pure_training_set_gives_a_single_leaf(tr):
    X, _ = classification_data(56, 20, 2, 2)
    model = fit(tr.DecisionTreeClassifier(), X, np.full(20, 3))
    assert (call(model.get_depth), call(model.get_n_leaves)) == (0, 1), (
        f"expected get_depth(), get_n_leaves() = (0, 1) for a pure training set (the root is a leaf), got "
        f"({model.get_depth()!r}, {model.get_n_leaves()!r})")
    assert_same_labels(call(model.predict, X[:3]), [3, 3, 3], msg="predict of a tree whose root is a leaf")


def test_decision_tree_classifier_score_is_the_accuracy(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    model = fit(tr.DecisionTreeClassifier(), X, np.array([0, 0, 1, 1]))
    got = call(model.score, np.array([[1.5], [3.5], [0.0]]), np.array([0, 0, 0]))
    assert_close(got, 2 / 3, msg="score([[1.5], [3.5], [0]], [0, 0, 0]) (predictions 0, 1, 0: 2 right out of 3)")
    assert_python_float(got, "score")


def test_decision_tree_classifier_fit_returns_self(tr):
    check_returns_self(tr.DecisionTreeClassifier(), *classification_data(57, 20, 2, 2))


def test_decision_tree_classifier_init_only_stores_the_hyperparameters(tr):
    params = {"criterion": "entropy", "max_depth": 3, "min_samples_split": 4, "min_samples_leaf": 2,
              "max_features": None, "splitter": "best", "random_state": 7}
    check_init_then_fit(tr.DecisionTreeClassifier(**params), params, *classification_data(58, 30, 2, 2))


def test_decision_tree_classifier_second_fit_forgets_the_first(tr):
    model = fit(tr.DecisionTreeClassifier(), *classification_data(59, 30, 3, 2))
    X2 = np.array([[1.0], [2.0], [3.0], [4.0]])
    fit(model, X2, np.array(["x", "x", "y", "y"]))
    assert learned(model, "n_features_in_") == 1, f"expected n_features_in_ = 1 after the second fit, got " \
                                                  f"{model.n_features_in_!r}"
    assert_same_labels(learned(model, "classes_"), ["x", "y"], msg="classes_ after the second fit")
    assert call(model.get_n_leaves) == 2, f"expected 2 leaves after the second fit, got {model.get_n_leaves()!r}"


def test_decision_tree_classifier_before_fit_raises_runtime_error(tr):
    model = tr.DecisionTreeClassifier()
    X = np.zeros((2, 2))
    assert_raises(RuntimeError, model.predict, X, why="predict called before fit")
    assert_raises(RuntimeError, model.predict_proba, X, why="predict_proba called before fit")
    assert_raises(RuntimeError, model.get_depth, why="get_depth called before fit")
    assert_raises(RuntimeError, model.get_n_leaves, why="get_n_leaves called before fit")


@pytest.mark.parametrize("params, why", [
    pytest.param({"criterion": "squared_error"}, "criterion='squared_error' is for regression trees",
                 id="regression_criterion"),
    pytest.param({"criterion": "misclassification"}, "criterion='misclassification' is unknown",
                 id="unknown_criterion"),
    pytest.param({"max_depth": 0}, "max_depth=0 (must be >= 1 or None)", id="max_depth_0"),
    pytest.param({"max_depth": -2}, "max_depth=-2", id="max_depth_negative"),
    pytest.param({"min_samples_split": 1}, "min_samples_split=1 (must be >= 2)", id="min_samples_split_1"),
    pytest.param({"min_samples_split": 0}, "min_samples_split=0", id="min_samples_split_0"),
    pytest.param({"min_samples_leaf": 0}, "min_samples_leaf=0 (must be >= 1)", id="min_samples_leaf_0"),
    pytest.param({"min_samples_leaf": -1}, "min_samples_leaf=-1", id="min_samples_leaf_negative"),
])
def test_decision_tree_classifier_fit_rejects_bad_hyperparameters(tr, params, why):
    model = build(tr.DecisionTreeClassifier, **params)
    assert_raises(ValueError, model.fit, *classification_data(60, 20, 2, 2), why=why)


@pytest.mark.parametrize("X, y, why", [
    pytest.param(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 1]), "X is 1-D (use X.reshape(-1, 1))", id="1d"),
    pytest.param(np.array([[1.0], [np.nan], [3.0]]), np.array([0, 1, 1]), "X contains NaN (remove the missing values "
                 "first)", id="nan"),
    pytest.param(np.array([[1.0], [2.0], [3.0]]), np.array([0, 1]), "3 samples but 2 labels", id="length"),
])
def test_decision_tree_classifier_fit_rejects_bad_inputs(tr, X, y, why):
    assert_raises(ValueError, tr.DecisionTreeClassifier().fit, X, y, why=why)


def test_decision_tree_classifier_predict_rejects_a_wrong_number_of_features(tr):
    model = fit(tr.DecisionTreeClassifier(), *classification_data(61, 20, 3, 2))
    assert_raises(ValueError, model.predict_proba, np.zeros((2, 2)), why="X has 2 features, fitted with 3")
    assert_raises(ValueError, model.predict, np.zeros((2, 4)), why="X has 4 features, fitted with 3")


def test_decision_tree_classifier_accepts_lists(tr):
    model = fit(tr.DecisionTreeClassifier(), [[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1])
    assert_same_labels(call(model.predict, [[0.0], [5.0]]), [0, 1], msg="fit and predict on lists of lists")


def test_decision_tree_classifier_does_not_modify_its_inputs(tr):
    X, y = classification_data(62, 40, 3, 3)
    X0, y0 = X.copy(), y.copy()
    model = tr.DecisionTreeClassifier(max_depth=4)
    call(model.fit, X, y, copy_inputs=False)
    call(model.predict, X, copy_inputs=False)
    call(model.predict_proba, X, copy_inputs=False)
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "expected fit and predict to leave X and y unchanged"


# ================================================================== DecisionTreeClassifier: options of ch. 14 (13.25)
# (n, d, n_classes, hyperparameters, seed) for the weighted comparisons with scikit-learn
WEIGHTED_CLASSIFIER_CASES = [
    pytest.param(80, 1, 3, {}, 0, id="one_feature_full"),
    pytest.param(60, 2, 2, {"max_depth": 4}, 2, id="n60_max_depth4"),
    pytest.param(120, 3, 3, {"max_depth": 4}, 1, id="n120_max_depth4"),
    pytest.param(100, 3, 2, {"min_samples_leaf": 4}, 0, id="min_samples_leaf4"),
    pytest.param(80, 2, 3, {"criterion": "entropy", "max_depth": 5}, 1, id="entropy_max_depth5"),
]


def test_decision_tree_classifier_ensemble_sample_weight_two_equals_duplicates(tr):
    X, y = classification_data(70, 50, 3, 3)
    weighted = fit(tr.DecisionTreeClassifier(), X, y, sample_weight=np.full(50, 2.0))
    doubled = fit(tr.DecisionTreeClassifier(), np.vstack([X, X]), np.concatenate([y, y]))
    assert _structure(weighted) == _structure(doubled), (
        "expected sample_weight=2 for every sample to give the same splits as duplicating every sample (gains are "
        "computed with weights; n_samples stays unweighted)")
    Q = _f32(np.random.default_rng(70).normal(size=(100, 3)))
    assert_close(call(weighted.predict_proba, Q), call(doubled.predict_proba, Q),
                 msg="predict_proba with sample_weight=2 = predict_proba after duplicating the samples")


def test_decision_tree_classifier_ensemble_integer_weights_equal_repeated_samples(tr):
    X, y = classification_data(71, 60, 3, 3)
    w = np.random.default_rng(71).integers(1, 4, size=60)
    weighted = fit(tr.DecisionTreeClassifier(), X, y, sample_weight=w.astype(float))
    repeated = fit(tr.DecisionTreeClassifier(), np.repeat(X, w, axis=0), np.repeat(y, w))
    assert _structure(weighted) == _structure(repeated), (
        "expected integer weights (1 to 3) to give the same splits as repeating every sample that many times")
    Q = _f32(np.random.default_rng(71).normal(size=(100, 3)))
    assert_close(call(weighted.predict_proba, Q), call(repeated.predict_proba, Q),
                 msg="predict_proba with integer weights = predict_proba with the samples repeated")


@pytest.mark.parametrize("n, d, n_classes, params, seed", WEIGHTED_CLASSIFIER_CASES)
def test_decision_tree_classifier_ensemble_sample_weight_matches_sklearn(tr, n, d, n_classes, params, seed):
    X, y = classification_data(seed + 300, n, d, n_classes)
    w = weights_for(seed + 300, n)
    oracle = _oracle(sktree.DecisionTreeClassifier, X, y, sample_weight=w, **params)
    model = fit(build(tr.DecisionTreeClassifier, **params), X, y, sample_weight=w)
    label = f"DecisionTreeClassifier({', '.join(f'{k}={v!r}' for k, v in params.items())}) with random weights"
    assert_same_tree(model, oracle, label)
    Q = _queries(seed, d, oracle)
    assert_close(call(model.predict_proba, Q), oracle.predict_proba(Q), msg=f"{label}: predict_proba(new points)")


def test_decision_tree_classifier_ensemble_scaling_the_weights_changes_nothing(tr):
    X, y = classification_data(72, 60, 3, 2)
    w = weights_for(72, 60)
    a = fit(tr.DecisionTreeClassifier(), X, y, sample_weight=w)
    b = fit(tr.DecisionTreeClassifier(), X, y, sample_weight=w * 0.25)
    assert _structure(a) == _structure(b), "expected sample_weight and 0.25 * sample_weight to give the same tree " \
                                           "(only proportions of weight matter)"
    assert_close(call(b.predict_proba, X), call(a.predict_proba, X), rtol=1e-9,
                 msg="predict_proba with the weights divided by 4")


def test_decision_tree_classifier_ensemble_sample_weight_none_equals_ones(tr):
    X, y = classification_data(73, 50, 2, 3)
    a = fit(tr.DecisionTreeClassifier(), X, y)
    b = fit(tr.DecisionTreeClassifier(), X, y, sample_weight=np.ones(50))
    assert _structure(a) == _structure(b), "expected sample_weight=None and sample_weight=np.ones(n) to give the " \
                                           "same tree"


def test_decision_tree_classifier_ensemble_zero_weights_count_in_n_samples_only(tr):
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1, 1])
    model = fit(tr.DecisionTreeClassifier(max_depth=1), X, y, sample_weight=np.array([1.0, 3.0, 0.0, 2.0, 2.0]))
    root = _root(model)
    assert_close(root.value, [0.5, 0.5], msg="root_.value with weights [1, 3, 0, 2, 2] (weighted proportions: 4/8 "
                 "and 4/8; the sample of weight 0 counts for nothing)")
    assert root.n_samples == 5, (f"expected root_.n_samples = 5 (the number of samples, weight 0 included, as "
                                 f"scikit-learn's n_node_samples), got {root.n_samples!r}")
    assert_close(root.impurity, 0.5, msg="root_.impurity with weights (weighted Gini)")


@pytest.mark.parametrize("w, why", [
    pytest.param([1.0, -1.0, 1.0, 1.0], "a negative weight", id="negative"),
    pytest.param([1.0, 1.0, 1.0], "3 weights for 4 samples", id="wrong_shape"),
    pytest.param([0.0, 0.0, 0.0, 0.0], "the weights sum to 0", id="zero_total"),
])
def test_decision_tree_classifier_ensemble_rejects_bad_sample_weight(tr, w, why):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert_raises(ValueError, tr.DecisionTreeClassifier().fit, X, np.array([0, 0, 1, 1]), sample_weight=np.array(w),
                  why=why)


IMPORTANCE_CASES = CLASSIFIER_CASES[2:] + [pytest.param(60, 2, 2, {}, 17, id="full_depth")]


@pytest.mark.parametrize("n, d, n_classes, params, seed", IMPORTANCE_CASES)
def test_decision_tree_classifier_ensemble_feature_importances_match_sklearn(tr, n, d, n_classes, params, seed):
    X, y = classification_data(seed, n, d, n_classes)
    oracle = _oracle(sktree.DecisionTreeClassifier, X, y, **params)
    model = fit(build(tr.DecisionTreeClassifier, **params), X, y)
    assert_close(learned(model, "feature_importances_"), oracle.feature_importances_, rtol=1e-7, atol=1e-10,
                 msg=f"feature_importances_ of DecisionTreeClassifier({', '.join(f'{k}={v!r}' for k, v in params.items())}) "
                     "(sum of (w_node / w_root) * gain over the splits on each feature, normalised to sum 1)")


@pytest.mark.parametrize("n, d, n_classes, params, seed", WEIGHTED_CLASSIFIER_CASES[1:])
def test_decision_tree_classifier_ensemble_feature_importances_with_weights(tr, n, d, n_classes, params, seed):
    X, y = classification_data(seed + 300, n, d, n_classes)
    w = weights_for(seed + 300, n)
    oracle = _oracle(sktree.DecisionTreeClassifier, X, y, sample_weight=w, **params)
    model = fit(build(tr.DecisionTreeClassifier, **params), X, y, sample_weight=w)
    assert_close(learned(model, "feature_importances_"), oracle.feature_importances_, rtol=1e-7, atol=1e-10,
                 msg="feature_importances_ with random weights (w_node and w_root are total WEIGHTS)")


def test_decision_tree_classifier_ensemble_feature_importances_of_a_single_leaf_are_zero(tr):
    X, _ = classification_data(74, 10, 3, 2)
    model = fit(tr.DecisionTreeClassifier(), X, np.zeros(10, dtype=int))
    assert_close(learned(model, "feature_importances_"), np.zeros(3), atol=0.0,
                 msg="feature_importances_ when the root is a leaf (no split: all zeros, no division by 0)")


def _ranked_data(seed, n, d, regression=False):
    """Feature j predicts y less and less well as j grows: at the root, feature j beats every feature > j."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = _f32(y[:, None] * np.linspace(3.0, 0.4, d) + rng.normal(size=(n, d)))
    target = y + 0.1 * rng.normal(size=n) if regression else y
    return X, target


def _root_gains_are_ranked(X, y, regression):
    cls = sktree.DecisionTreeRegressor if regression else sktree.DecisionTreeClassifier
    gains = []
    for j in range(X.shape[1]):
        t = sk(cls, max_depth=1).fit(X[:, [j]], y).tree_
        gains.append(t.impurity[0] - (t.weighted_n_node_samples[1] * t.impurity[1]
                                      + t.weighted_n_node_samples[2] * t.impurity[2]) / t.weighted_n_node_samples[0])
    if not np.all(np.diff(gains) < -1e-6):
        raise AssertionError("test setup: the root gains of the features are not strictly decreasing")


# with d = 10 features: sqrt(10) = 3.16 and log2(10) = 3.32 give 3 (int truncates); 0.36 * 10 = 3.6 gives 3
MAX_FEATURES_CASES = [
    pytest.param(1, 1, id="int_1"), pytest.param(3, 3, id="int_3"), pytest.param("sqrt", 3, id="sqrt"),
    pytest.param("log2", 3, id="log2"), pytest.param(0.36, 3, id="float_0.36"), pytest.param(0.75, 7, id="float_0.75"),
    pytest.param(0.05, 1, id="float_tiny"),
]


@pytest.mark.parametrize("max_features, k", MAX_FEATURES_CASES)
def test_decision_tree_classifier_ensemble_max_features_draws_with_rng_choice(tr, max_features, k):
    X, y = _ranked_data(1, 200, 10)
    _root_gains_are_ranked(X, y, regression=False)
    for seed in range(8):
        subset = np.random.default_rng(seed).choice(10, size=k, replace=False)
        model = fit(tr.DecisionTreeClassifier(max_depth=1, max_features=max_features, random_state=seed), X, y)
        root = _root(model)
        assert root.feature == subset.min(), (
            f"max_features={max_features!r}, random_state={seed}: expected the root to split feature "
            f"{subset.min()}, the best of the {k} features drawn by np.random.default_rng({seed}).choice(10, size={k}, "
            f"replace=False) = {subset.tolist()} (feature j beats every feature > j here), got {root.feature!r}")


def test_decision_tree_classifier_ensemble_max_features_draws_depth_first_left_first(tr):
    X, y = classification_data(86, 120, 4, 3)
    for seed in range(4):
        rng = np.random.default_rng(seed)

        def split(rows):
            f = int(rng.choice(4, size=1, replace=False)[0])
            stump = sk(sktree.DecisionTreeClassifier, max_depth=1).fit(X[rows][:, [f]], y[rows]).tree_
            return f, float(stump.threshold[0])

        rows = np.arange(len(y))
        f_root, t_root = split(rows)
        left = rows[X[:, f_root] <= t_root]
        f_left, t_left = split(left)                          # the left child draws before the right child
        f_right, t_right = split(rows[X[:, f_root] > t_root])
        model = fit(tr.DecisionTreeClassifier(max_features=1, max_depth=2, random_state=seed), X, y)
        root = _root(model)
        got = [(root.feature, root.threshold)] + [(child.feature, child.threshold) for child in (root.left, root.right)
                                                  if child is not None]
        want = [(f_root, t_root), (f_left, t_left), (f_right, t_right)]
        assert len(got) == 3 and all(g[0] == w[0] and g[1] is not None and np.isclose(g[1], w[1], rtol=0, atol=1e-9)
                                     for g, w in zip(got, want)), (
            f"expected (feature, threshold) = {want} for the root, then its left and right children "
            f"(max_features=1, max_depth=2, random_state={seed}), got {got} (one draw rng.choice(4, size=1, "
            "replace=False) per node that is tried, the root first, then the left child, then the right child)")


def test_decision_tree_classifier_ensemble_random_splitter_draws_thresholds_after_the_features(tr):
    X, y = classification_data(87, 80, 5, 2)
    for seed in (5, 10, 12, 17):                              # seeds whose two drawn features are not in increasing order
        rng = np.random.default_rng(seed)
        features = rng.choice(5, size=2, replace=False)
        thresholds = [rng.uniform(X[:, f].min(), X[:, f].max()) for f in features]
        gains = [_np_gain(y, X[:, f] <= t, np.ones(len(y)), "gini") for f, t in zip(features, thresholds)]
        k = max(range(2), key=lambda i: (gains[i], -features[i]))
        model = fit(tr.DecisionTreeClassifier(max_features=2, splitter="random", max_depth=1, random_state=seed), X, y)
        root = _root(model)
        assert root.feature == features[k] and np.isclose(root.threshold, thresholds[k], rtol=0, atol=1e-12), (
            f"expected the root ({int(features[k])}, {thresholds[k]:.10g}) with max_features=2, splitter='random', "
            f"random_state={seed}, got ({root.feature}, {root.threshold}) (the generator first draws the features, "
            f"{features.tolist()}, then one threshold per drawn feature, in the drawn order)")


@pytest.mark.parametrize("params", [{"max_features": 1}, {"splitter": "random"}], ids=["max_features_1", "random_splitter"])
def test_decision_tree_classifier_ensemble_second_fit_of_the_same_object_gives_the_same_tree(tr, params):
    X, y = classification_data(88, 80, 5, 3)
    model = tr.DecisionTreeClassifier(random_state=5, **params)
    first = _structure(fit(model, X, y))
    second = _structure(fit(model, X, y))
    assert first == second, (f"expected the same tree when the same object is fitted twice with {params} and "
                             "random_state=5, got two different trees (create the generator "
                             "np.random.default_rng(random_state) in fit, not in __init__)")


def test_decision_tree_classifier_ensemble_max_features_draws_at_every_node(tr):
    X, y = classification_data(85, 120, 5, 3)
    for seed in range(4):
        model = fit(tr.DecisionTreeClassifier(max_features=1, random_state=seed), X, y)
        used = sorted({node.feature for node, _, _ in _nodes(model) if node.left is not None})
        assert len(used) >= 3, (
            f"max_features=1, random_state={seed}: expected the splits of a full tree to use several features (a new "
            f"feature is drawn at EVERY node with the same generator, created once in fit), got splits on {used} only")


def test_decision_tree_classifier_ensemble_max_features_none_ignores_random_state(tr):
    X, y = classification_data(76, 80, 4, 3)
    trees = [fit(tr.DecisionTreeClassifier(random_state=seed), X, y) for seed in (0, 1, 2)]
    assert _structure(trees[0]) == _structure(trees[1]) == _structure(trees[2]), (
        "expected the same tree for random_state 0, 1 and 2 with max_features=None and splitter='best' (nothing is "
        "drawn at random)")
    for max_features in (4, 1.0):
        drawn = fit(tr.DecisionTreeClassifier(max_features=max_features, random_state=3), X, y)
        assert _structure(drawn) == _structure(trees[0]), (
            f"expected max_features={max_features!r} (all 4 features, drawn in a random order) to give the tree of "
            "max_features=None (ties go to the lowest feature index, not to the first feature drawn)")


@pytest.mark.parametrize("params", [{"max_features": 1}, {"max_features": "sqrt"}, {"splitter": "random"},
                                    {"splitter": "random", "max_features": 2}],
                         ids=["max_features_1", "sqrt", "random_splitter", "random_splitter_2_features"])
def test_decision_tree_classifier_ensemble_same_random_state_same_tree(tr, params):
    X, y = classification_data(77, 80, 5, 3)
    a = fit(tr.DecisionTreeClassifier(random_state=5, **params), X, y)
    b = fit(tr.DecisionTreeClassifier(random_state=5, **params), X, y)
    assert _structure(a) == _structure(b), (f"expected the same tree twice with {params} and random_state=5 (create "
                                            "the generator np.random.default_rng(random_state) in fit)")
    Q = _f32(np.random.default_rng(77).normal(size=(50, 5)))
    assert_close(call(a.predict_proba, Q), call(b.predict_proba, Q), msg=f"predict_proba of two trees with {params} "
                 "and random_state=5")


@pytest.mark.parametrize("params", [{"max_features": 1}, {"splitter": "random"}],
                         ids=["max_features_1", "random_splitter"])
def test_decision_tree_classifier_ensemble_different_random_states_different_trees(tr, params):
    X, y = classification_data(78, 80, 5, 3)
    structures = [_structure(fit(tr.DecisionTreeClassifier(random_state=seed, **params), X, y)) for seed in range(5)]
    assert any(s != structures[0] for s in structures[1:]), (
        f"expected at least two different trees among random_state 0 to 4 with {params}, got 5 identical trees (draw "
        "the features / thresholds with the generator of random_state, at every node)")


def test_decision_tree_classifier_ensemble_random_splitter_thresholds_inside_the_node_range(tr):
    X, y = classification_data(79, 80, 4, 3)
    model = fit(tr.DecisionTreeClassifier(splitter="random", random_state=0), X, y)
    midpoints = 0
    for node, path, rows in _node_rows(model, X):
        if node.left is None:
            continue
        values = X[rows, node.feature]
        assert values.min() < node.threshold < values.max(), (
            f"splitter='random': expected the threshold at {path} strictly between min = {values.min():.6g} and max = "
            f"{values.max():.6g} of feature {node.feature} over the node's samples, got {node.threshold!r}")
        sorted_values = np.unique(values)
        midpoints += bool(np.isclose((sorted_values[:-1] + sorted_values[1:]) / 2, node.threshold).any())
    assert midpoints < 3, (f"expected random thresholds (rng.uniform(min, max)), got {midpoints} thresholds that are "
                           "midpoints between two values (the 'best' splitter?)")
    assert_same_labels(call(model.predict, X), y, msg="splitter='random' grown until pure: predict(X_train) = y")


def test_decision_tree_classifier_ensemble_random_splitter_root_uses_the_first_draw(tr):
    X, y = classification_data(80, 40, 1, 2)
    for seed in range(4):
        t = np.random.default_rng(seed).uniform(X[:, 0].min(), X[:, 0].max())
        model = fit(tr.DecisionTreeClassifier(splitter="random", max_depth=1, random_state=seed), X, y)
        root = _root(model)
        assert root.feature == 0 and np.isclose(root.threshold, t, rtol=0, atol=1e-12), (
            f"splitter='random', random_state={seed}, one feature: expected the root threshold "
            f"np.random.default_rng({seed}).uniform(min, max) = {t:.10g} (the first draw of the generator created in "
            f"fit), got {root.threshold!r}")


@pytest.mark.parametrize("max_features, why", [
    pytest.param(0, "max_features=0 (an int must be in 1..d)", id="int_0"),
    pytest.param(5, "max_features=5 > d = 4", id="int_too_big"),
    pytest.param(-1, "max_features=-1", id="int_negative"),
    pytest.param(0.0, "max_features=0.0 (a float must be in (0, 1])", id="float_0"),
    pytest.param(1.5, "max_features=1.5 (a float must be in (0, 1])", id="float_above_1"),
    pytest.param("auto", "max_features='auto' (only 'sqrt' and 'log2')", id="auto"),
])
def test_decision_tree_classifier_ensemble_fit_rejects_bad_max_features(tr, max_features, why):
    model = build(tr.DecisionTreeClassifier, max_features=max_features)
    assert_raises(ValueError, model.fit, *classification_data(81, 20, 4, 2), why=why)


def test_decision_tree_classifier_ensemble_fit_rejects_a_bad_splitter(tr):
    model = build(tr.DecisionTreeClassifier, splitter="worst")
    assert_raises(ValueError, model.fit, *classification_data(82, 20, 2, 2), why="splitter='worst' (only 'best' and "
                  "'random')")


@pytest.mark.parametrize("max_features", [1, 4, 1.0, 0.01, "sqrt", "log2"])
def test_decision_tree_classifier_ensemble_accepts_valid_max_features(tr, max_features):
    X, y = classification_data(83, 40, 4, 2)
    model = fit(tr.DecisionTreeClassifier(max_features=max_features, random_state=0), X, y)
    assert_same_labels(call(model.predict, X), y, msg=f"max_features={max_features!r}, grown until pure: "
                       "predict(X_train) = y")


def test_decision_tree_classifier_ensemble_init_only_stores_the_hyperparameters(tr):
    params = {"criterion": "gini", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,
              "max_features": "sqrt", "splitter": "random", "random_state": 3}
    check_init_then_fit(tr.DecisionTreeClassifier(**params), params, *classification_data(84, 30, 4, 2))


def test_decision_tree_classifier_ensemble_does_not_modify_sample_weight(tr):
    X, y = classification_data(63, 40, 3, 3)
    w = np.random.default_rng(63).uniform(0.5, 3.0, size=len(y))     # does not sum to 1
    w0 = w.copy()
    call(tr.DecisionTreeClassifier(max_depth=4).fit, X, y, sample_weight=w, copy_inputs=False)
    assert np.array_equal(w, w0), ("expected fit to leave sample_weight unchanged (normalise a copy: AdaBoost "
                                   "reuses the same array in ch. 14)")


# ================================================================== DecisionTreeRegressor (13.26)
REGRESSOR_CASES = [
    pytest.param(80, 1, {}, 0, id="one_feature_full"),
    pytest.param(100, 3, {"max_depth": 3}, 1, id="max_depth3"),
    pytest.param(150, 2, {"min_samples_leaf": 6}, 2, id="min_samples_leaf6"),
    pytest.param(120, 4, {"min_samples_split": 20}, 3, id="min_samples_split20"),
    pytest.param(60, 2, {"max_depth": 5, "min_samples_leaf": 2}, 0, id="max_depth5_leaf2"),
]


def test_decision_tree_regressor_docstring_example(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 1.2, 3.0, 3.4])
    model = fit(tr.DecisionTreeRegressor(max_depth=1), X, y)
    assert_close(call(model.predict, np.array([[0.0], [5.0]])), [1.1, 3.2], rtol=1e-12,
                 msg="DecisionTreeRegressor(max_depth=1).predict([[0], [5]]) of the docstring example")
    assert_close(call(model.predict, np.array([[2.5]])), [1.1], rtol=1e-12,
                 msg="predict([[2.5]]) with the root threshold 2.5 (x <= threshold goes LEFT)")


@pytest.mark.parametrize("n, d, params, seed", REGRESSOR_CASES)
def test_decision_tree_regressor_matches_sklearn(tr, n, d, params, seed):
    X, y = regression_data(seed, n, d)
    oracle = _oracle(sktree.DecisionTreeRegressor, X, y, **params)
    model = fit(build(tr.DecisionTreeRegressor, **params), X, y)
    label = f"DecisionTreeRegressor({', '.join(f'{k}={v!r}' for k, v in params.items())}) on {n} samples, {d} features"
    assert call(model.get_depth) == oracle.get_depth(), (
        f"{label}: expected get_depth() = {oracle.get_depth()} (as scikit-learn), got {model.get_depth()!r}")
    assert call(model.get_n_leaves) == oracle.get_n_leaves(), (
        f"{label}: expected get_n_leaves() = {oracle.get_n_leaves()} (as scikit-learn), got {model.get_n_leaves()!r}")
    Q = _queries(seed, d, oracle)
    assert_close(call(model.predict, Q), oracle.predict(Q), rtol=1e-9, atol=1e-12, msg=f"{label}: predict(new points)")
    assert_close(call(model.predict, X), oracle.predict(X), rtol=1e-9, atol=1e-12, msg=f"{label}: predict(X_train)")


@pytest.mark.parametrize("n, d, params, seed", REGRESSOR_CASES)
def test_decision_tree_regressor_nodes_match_sklearn(tr, n, d, params, seed):
    X, y = regression_data(seed, n, d)
    oracle = _oracle(sktree.DecisionTreeRegressor, X, y, **params)
    model = fit(build(tr.DecisionTreeRegressor, **params), X, y)
    assert_same_tree(model, oracle, f"DecisionTreeRegressor({', '.join(f'{k}={v!r}' for k, v in params.items())})")


def test_decision_tree_regressor_unlimited_tree_interpolates_the_training_set(tr):
    X, y = regression_data(90, 50, 3)
    model = fit(tr.DecisionTreeRegressor(), X, y)
    assert_close(call(model.predict, X), y, rtol=1e-12, atol=1e-12,
                 msg="predict(X_train) of a regression tree grown until the leaves are pure (one target per leaf)")
    assert call(model.get_n_leaves) == 50, f"expected 50 leaves (one per distinct sample), got {model.get_n_leaves()!r}"


def test_decision_tree_regressor_leaf_values_are_means_of_shape_1(tr):
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.0, 2.0, 6.0, 7.0, 8.0])
    model = fit(tr.DecisionTreeRegressor(max_depth=1), X, y)
    root = _root(model)
    assert np.shape(root.value) == (1,), f"expected root_.value of shape (1,), got shape {np.shape(root.value)}"
    assert_close(root.value, [4.8], msg="root_.value (the mean target of the 5 samples)")
    assert_close(root.impurity, np.var(y), msg="root_.impurity (the variance of the targets, ddof=0)")
    assert_close(root.left.value, [1.5], msg="root_.left.value (mean of [1, 2])")
    assert_close(root.right.value, [7.0], msg="root_.right.value (mean of [6, 7, 8])")


@pytest.mark.parametrize("max_depth", [1, 3])
def test_decision_tree_regressor_max_depth(tr, max_depth):
    X, y = regression_data(91, 80, 2)
    model = fit(tr.DecisionTreeRegressor(max_depth=max_depth), X, y)
    assert call(model.get_depth) == max_depth, f"expected get_depth() = {max_depth}, got {model.get_depth()!r}"
    assert call(model.get_n_leaves) == 2 ** max_depth, (
        f"expected 2^{max_depth} = {2 ** max_depth} leaves (continuous targets: every node is split until "
        f"max_depth), got {model.get_n_leaves()!r}")


def test_decision_tree_regressor_min_samples_split_and_leaf(tr):
    X, y = regression_data(92, 90, 2)
    model = fit(tr.DecisionTreeRegressor(min_samples_split=12, min_samples_leaf=3), X, y)
    for node, path, _ in _nodes(model):
        if node.left is not None:
            assert node.n_samples >= 12, f"expected splits only at nodes of >= 12 samples, got {node.n_samples} at {path}"
        else:
            assert node.n_samples >= 3, f"expected leaves of >= 3 samples (min_samples_leaf=3), got {node.n_samples} " \
                                        f"at {path}"


def test_decision_tree_regressor_constant_targets_give_a_single_leaf(tr):
    X, _ = regression_data(93, 15, 2)
    model = fit(tr.DecisionTreeRegressor(), X, np.full(15, 2.5))
    assert (call(model.get_depth), call(model.get_n_leaves)) == (0, 1), (
        f"expected get_depth(), get_n_leaves() = (0, 1) for constant targets (the root is pure), got "
        f"({model.get_depth()!r}, {model.get_n_leaves()!r})")
    assert_close(call(model.predict, X[:2]), [2.5, 2.5], msg="predict of a tree whose root is a leaf")


def test_decision_tree_regressor_score_is_r2(tr):
    X, y = regression_data(94, 60, 2)
    model = fit(tr.DecisionTreeRegressor(max_depth=2), X, y)
    X_new, y_new = regression_data(95, 30, 2)
    got = call(model.score, X_new, y_new)
    assert_close(got, skm.r2_score(y_new, call(model.predict, X_new)), rtol=1e-12,
                 msg="score(X, y) = R² of predict(X) against y (sklearn.metrics.r2_score)")
    assert_python_float(got, "score")


def test_decision_tree_regressor_predict_returns_float64(tr):
    model = fit(tr.DecisionTreeRegressor(max_depth=2), *regression_data(96, 20, 2))
    result = call(model.predict, np.zeros((3, 2)))
    assert isinstance(result, np.ndarray) and result.shape == (3,) and result.dtype == np.float64, (
        f"expected predict to return a float64 array of shape (3,), got {type(result).__name__} "
        f"{getattr(result, 'shape', '')} {getattr(result, 'dtype', '')}")


def test_decision_tree_regressor_fit_returns_self(tr):
    check_returns_self(tr.DecisionTreeRegressor(), *regression_data(97, 20, 2))


def test_decision_tree_regressor_init_only_stores_the_hyperparameters(tr):
    params = {"criterion": "squared_error", "max_depth": 4, "min_samples_split": 5, "min_samples_leaf": 2,
              "max_features": None, "splitter": "best", "random_state": None}
    check_init_then_fit(tr.DecisionTreeRegressor(**params), params, *regression_data(98, 30, 2))


def test_decision_tree_regressor_second_fit_forgets_the_first(tr):
    model = fit(tr.DecisionTreeRegressor(), *regression_data(99, 30, 3))
    fit(model, np.array([[1.0], [2.0]]), np.array([5.0, 7.0]))
    assert learned(model, "n_features_in_") == 1, f"expected n_features_in_ = 1 after the second fit, got " \
                                                  f"{model.n_features_in_!r}"
    assert_close(call(model.predict, np.array([[0.0], [3.0]])), [5.0, 7.0], msg="predict after the second fit")


def test_decision_tree_regressor_before_fit_raises_runtime_error(tr):
    model = tr.DecisionTreeRegressor()
    assert_raises(RuntimeError, model.predict, np.zeros((2, 2)), why="predict called before fit")
    assert_raises(RuntimeError, model.get_depth, why="get_depth called before fit")
    assert_raises(RuntimeError, model.get_n_leaves, why="get_n_leaves called before fit")


@pytest.mark.parametrize("params, why", [
    pytest.param({"criterion": "gini"}, "criterion='gini' is for classification", id="gini"),
    pytest.param({"criterion": "absolute_error"}, "criterion='absolute_error' (only 'squared_error')",
                 id="absolute_error"),
    pytest.param({"max_depth": 0}, "max_depth=0", id="max_depth_0"),
    pytest.param({"min_samples_split": 1}, "min_samples_split=1", id="min_samples_split_1"),
    pytest.param({"min_samples_leaf": 0}, "min_samples_leaf=0", id="min_samples_leaf_0"),
])
def test_decision_tree_regressor_fit_rejects_bad_hyperparameters(tr, params, why):
    model = build(tr.DecisionTreeRegressor, **params)
    assert_raises(ValueError, model.fit, *regression_data(100, 20, 2), why=why)


@pytest.mark.parametrize("X, y, why", [
    pytest.param(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.0]), "X is 1-D", id="1d"),
    pytest.param(np.array([[1.0], [np.nan], [3.0]]), np.array([0.0, 1.0, 1.0]), "X contains NaN", id="nan"),
    pytest.param(np.array([[1.0], [2.0], [3.0]]), np.array([0.0, 1.0]), "3 samples but 2 targets", id="length"),
])
def test_decision_tree_regressor_fit_rejects_bad_inputs(tr, X, y, why):
    assert_raises(ValueError, tr.DecisionTreeRegressor().fit, X, y, why=why)


def test_decision_tree_regressor_predict_rejects_a_wrong_number_of_features(tr):
    model = fit(tr.DecisionTreeRegressor(), *regression_data(101, 20, 3))
    assert_raises(ValueError, model.predict, np.zeros((2, 2)), why="X has 2 features, fitted with 3")


def test_decision_tree_regressor_does_not_modify_its_inputs(tr):
    X, y = regression_data(102, 40, 3)
    X0, y0 = X.copy(), y.copy()
    model = tr.DecisionTreeRegressor(max_depth=4)
    call(model.fit, X, y, copy_inputs=False)
    call(model.predict, X, copy_inputs=False)
    assert np.array_equal(X, X0) and np.array_equal(y, y0), "expected fit and predict to leave X and y unchanged"


# ================================================================== DecisionTreeRegressor: options of ch. 14
WEIGHTED_REGRESSOR_CASES = [
    pytest.param(80, 1, {}, 0, id="one_feature_full"),
    pytest.param(100, 3, {"max_depth": 4}, 3, id="max_depth4"),
    pytest.param(120, 2, {"min_samples_leaf": 5}, 2, id="min_samples_leaf5"),
]


def test_decision_tree_regressor_ensemble_sample_weight_two_equals_duplicates(tr):
    X, y = regression_data(110, 40, 3)
    weighted = fit(tr.DecisionTreeRegressor(max_depth=5), X, y, sample_weight=np.full(40, 2.0))
    doubled = fit(tr.DecisionTreeRegressor(max_depth=5), np.vstack([X, X]), np.concatenate([y, y]))
    assert _structure(weighted) == _structure(doubled), (
        "expected sample_weight=2 for every sample to give the same splits as duplicating every sample")
    Q = _f32(np.random.default_rng(110).normal(size=(100, 3)))
    assert_close(call(weighted.predict, Q), call(doubled.predict, Q), rtol=1e-12,
                 msg="predict with sample_weight=2 = predict after duplicating the samples")


@pytest.mark.parametrize("n, d, params, seed", WEIGHTED_REGRESSOR_CASES)
def test_decision_tree_regressor_ensemble_sample_weight_matches_sklearn(tr, n, d, params, seed):
    X, y = regression_data(seed + 300, n, d)
    w = weights_for(seed + 300, n)
    oracle = _oracle(sktree.DecisionTreeRegressor, X, y, sample_weight=w, **params)
    model = fit(build(tr.DecisionTreeRegressor, **params), X, y, sample_weight=w)
    label = f"DecisionTreeRegressor({', '.join(f'{k}={v!r}' for k, v in params.items())}) with random weights"
    assert_same_tree(model, oracle, label)
    Q = _queries(seed, d, oracle)
    assert_close(call(model.predict, Q), oracle.predict(Q), rtol=1e-9, atol=1e-12, msg=f"{label}: predict(new points)")


@pytest.mark.parametrize("n, d, params, seed", REGRESSOR_CASES[1:])
def test_decision_tree_regressor_ensemble_feature_importances_match_sklearn(tr, n, d, params, seed):
    X, y = regression_data(seed, n, d)
    oracle = _oracle(sktree.DecisionTreeRegressor, X, y, **params)
    model = fit(build(tr.DecisionTreeRegressor, **params), X, y)
    assert_close(learned(model, "feature_importances_"), oracle.feature_importances_, rtol=1e-7, atol=1e-10,
                 msg=f"feature_importances_ of DecisionTreeRegressor({', '.join(f'{k}={v!r}' for k, v in params.items())}) "
                     "(normalised weighted variance decrease per feature)")


@pytest.mark.parametrize("max_features, k", MAX_FEATURES_CASES[:3])
def test_decision_tree_regressor_ensemble_max_features_draws_with_rng_choice(tr, max_features, k):
    X, y = _ranked_data(1, 200, 10, regression=True)
    _root_gains_are_ranked(X, y, regression=True)
    for seed in range(6):
        subset = np.random.default_rng(seed).choice(10, size=k, replace=False)
        model = fit(tr.DecisionTreeRegressor(max_depth=1, max_features=max_features, random_state=seed), X, y)
        assert _root(model).feature == subset.min(), (
            f"max_features={max_features!r}, random_state={seed}: expected the root to split feature {subset.min()}, "
            f"the best of the features drawn by np.random.default_rng({seed}).choice(10, size={k}, replace=False) = "
            f"{subset.tolist()}, got {model.root_.feature!r}")


@pytest.mark.parametrize("params", [{"max_features": 1}, {"splitter": "random"}],
                         ids=["max_features_1", "random_splitter"])
def test_decision_tree_regressor_ensemble_random_state_controls_the_tree(tr, params):
    X, y = regression_data(112, 60, 4)
    a = fit(tr.DecisionTreeRegressor(max_depth=4, random_state=1, **params), X, y)
    b = fit(tr.DecisionTreeRegressor(max_depth=4, random_state=1, **params), X, y)
    assert _structure(a) == _structure(b), f"expected the same tree twice with {params} and random_state=1"
    others = [_structure(fit(tr.DecisionTreeRegressor(max_depth=4, random_state=s, **params), X, y)) for s in range(2, 6)]
    assert any(s != _structure(a) for s in others), (
        f"expected at least two different trees among random_state 1 to 5 with {params}, got identical trees")


def test_decision_tree_regressor_ensemble_random_splitter_thresholds_inside_the_node_range(tr):
    X, y = regression_data(113, 60, 3)
    model = fit(tr.DecisionTreeRegressor(splitter="random", random_state=2), X, y)
    for node, path, rows in _node_rows(model, X):
        if node.left is not None:
            values = X[rows, node.feature]
            assert values.min() < node.threshold < values.max(), (
                f"splitter='random': expected the threshold at {path} strictly between {values.min():.6g} and "
                f"{values.max():.6g} (feature {node.feature} over the node's samples), got {node.threshold!r}")
    assert_close(call(model.predict, X), y, rtol=1e-12, atol=1e-12,
                 msg="splitter='random' grown until pure: predict(X_train) = y")


def test_decision_tree_regressor_ensemble_random_splitter_root_uses_the_first_draw(tr):
    X, y = regression_data(114, 30, 1)
    for seed in range(3):
        t = np.random.default_rng(seed).uniform(X[:, 0].min(), X[:, 0].max())
        root = _root(fit(tr.DecisionTreeRegressor(splitter="random", max_depth=1, random_state=seed), X, y))
        assert np.isclose(root.threshold, t, rtol=0, atol=1e-12), (
            f"splitter='random', random_state={seed}, one feature: expected the root threshold "
            f"np.random.default_rng({seed}).uniform(min, max) = {t:.10g}, got {root.threshold!r}")


@pytest.mark.parametrize("max_features, why", [
    pytest.param(0, "max_features=0", id="int_0"),
    pytest.param(4, "max_features=4 > d = 3", id="int_too_big"),
    pytest.param(1.5, "max_features=1.5", id="float_above_1"),
    pytest.param("auto", "max_features='auto'", id="auto"),
])
def test_decision_tree_regressor_ensemble_fit_rejects_bad_max_features(tr, max_features, why):
    model = build(tr.DecisionTreeRegressor, max_features=max_features)
    assert_raises(ValueError, model.fit, *regression_data(115, 20, 3), why=why)


def test_decision_tree_regressor_ensemble_feature_importances_of_a_single_leaf_are_zero(tr):
    X, _ = regression_data(116, 10, 2)
    model = fit(tr.DecisionTreeRegressor(), X, np.full(10, -1.0))
    assert_close(learned(model, "feature_importances_"), np.zeros(2), atol=0.0,
                 msg="feature_importances_ when the root is a leaf (all zeros)")


def test_decision_tree_regressor_ensemble_does_not_modify_sample_weight(tr):
    X, y = regression_data(103, 40, 3)
    w = np.random.default_rng(103).uniform(0.5, 3.0, size=len(y))
    w0 = w.copy()
    call(tr.DecisionTreeRegressor(max_depth=4).fit, X, y, sample_weight=w, copy_inputs=False)
    assert np.array_equal(w, w0), ("expected fit to leave sample_weight unchanged (normalise a copy: gradient "
                                   "boosting and AdaBoost reuse the same array in ch. 14)")
