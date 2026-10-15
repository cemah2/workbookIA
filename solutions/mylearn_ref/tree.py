"""Decision trees (CART) — mylearn, chapter 13 (Classifiers).

Impurity measures (Gini, entropy, variance), the gain of a split, the search for the
best split and the greedy recursive trees for classification and regression. The
trees are built for chapter 14: ``sample_weight`` (AdaBoost), ``max_features`` and
``random_state`` (random forest), ``splitter="random"`` (ExtraTrees) and
``DecisionTreeRegressor`` (gradient boosting) reuse them without any change.

Reference implementation: read it only after trying (``mon_travail/mylearn/tree.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

_CLASSIFICATION_CRITERIA = ("gini", "entropy")
_CRITERIA = ("gini", "entropy", "squared_error")
_SPLITTERS = ("best", "random")

# "Strictly positive gain" in floating point. A gain is a difference of numbers of the order of
# the parent impurity, so a gain that is 0 in exact arithmetic comes out as a rounding residue of
# about 1e-16 times the parent impurity. The reference keeps a split only if its gain exceeds
# _GAIN_RTOL * parent impurity, and treats two candidates whose gains differ by less than that as
# tied (lowest feature index, then lowest threshold). No test depends on gains that small.
_GAIN_RTOL = 1e-12


def _is_int(value) -> bool:
    """An int (Python or NumPy), but not a bool."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_X(X: ArrayLike, n_features: int | None = None) -> np.ndarray:
    """X as a 2-D float64 array (a copy), finite, with at least one sample."""
    try:
        X = np.array(X, dtype=float)               # np.array copies: the caller's data are never modified
    except (TypeError, ValueError) as exc:
        raise ValueError(f"X must hold numbers only ({exc})") from None
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got an array with {X.ndim} dimension(s): "
                         "use X.reshape(-1, 1) for a single feature")
    if np.isnan(X).any():
        raise ValueError("X contains NaN values: remove the missing values first")
    if np.isinf(X).any():
        raise ValueError("X contains infinite values")
    if X.shape[0] == 0:
        raise ValueError("X has no sample")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} feature(s), but the tree was fitted with {n_features}")
    return X


def _check_labels(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Class labels as a non-empty 1-D array without NaN."""
    y = np.array(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {y.shape}")
    if y.size == 0:
        raise ValueError(f"{name} is empty")
    if y.dtype.kind in "fc" and np.isnan(y).any():
        raise ValueError(f"{name} contains NaN values: remove the missing values first")
    return y


def _check_targets(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Real targets as a non-empty 1-D float64 array without NaN."""
    try:
        y = np.array(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold numbers only ({exc})") from None
    return _check_labels(y, name)


def _check_weights(sample_weight: ArrayLike | None, n: int) -> np.ndarray:
    """Non-negative float64 weights of shape (n,) with a positive total; all 1 if None."""
    if sample_weight is None:
        return np.ones(n)
    try:
        w = np.array(sample_weight, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_weight must hold numbers only ({exc})") from None
    if w.shape != (n,):
        raise ValueError(f"sample_weight must have shape ({n},), got {w.shape}")
    if np.isnan(w).any():
        raise ValueError("sample_weight contains NaN values: remove the missing values first")
    if (w < 0).any():
        raise ValueError("sample_weight must be non-negative")
    total = w.sum()
    if not (np.isfinite(total) and total > 0):
        raise ValueError(f"sample_weight must have a positive finite total, got {total}")
    return w


def _class_proportions(y: ArrayLike, sample_weight: ArrayLike | None) -> np.ndarray:
    """Weighted proportions of the classes present in y (classes of weight 0 give 0)."""
    y = _check_labels(y)
    w = _check_weights(sample_weight, len(y))
    _, codes = np.unique(y, return_inverse=True)
    totals = np.bincount(codes.ravel(), weights=w)
    return totals / totals.sum()


def _impurity_function(criterion: str):
    if isinstance(criterion, str):
        if criterion == "gini":
            return gini_impurity
        if criterion == "entropy":
            return entropy_impurity
        if criterion == "squared_error":
            return variance_impurity
    raise ValueError(f"unknown criterion {criterion!r}: expected 'gini', 'entropy' or 'squared_error'")


def _is_pure(y: np.ndarray, w: np.ndarray, criterion: str) -> bool:
    """Impurity exactly 0: one class (or one target value) among the samples of positive weight.

    Tested exactly (not as impurity == 0.0): the variance of seven targets 0.1 is about 1e-34.
    """
    values = y[w > 0]
    if criterion == "squared_error":
        return bool(values.min() == values.max())
    return bool(np.all(values == values[0]))


def _node_stats(y: np.ndarray, w: np.ndarray, criterion: str) -> np.ndarray:
    """Per-sample additive statistics (n, S): their sums over any group give its impurity.

    Classification: the weight of the sample in the column of its class. Regression: (w, w yc, w yc^2) with
    yc = y - mean, centred so that S2/W - (S1/W)^2 suffers no cancellation from a large offset of y.
    """
    if criterion == "squared_error":
        yc = y - np.average(y, weights=w)
        return np.column_stack([w, w * yc, w * yc * yc])
    _, codes = np.unique(y, return_inverse=True)
    codes = codes.ravel()
    stats = np.zeros((len(y), int(codes.max()) + 1))
    stats[np.arange(len(y)), codes] = w
    return stats


def _impurity_from_stats(s: np.ndarray, criterion: str) -> tuple[np.ndarray, np.ndarray]:
    """(impurity, total weight) of groups given their summed statistics s of shape (..., S)."""
    if criterion == "squared_error":
        total = s[..., 0]
        safe = np.where(total > 0, total, 1.0)
        impurity = np.maximum(s[..., 2] / safe - (s[..., 1] / safe) ** 2, 0.0)
    else:
        total = s.sum(axis=-1)
        safe = np.where(total > 0, total, 1.0)
        p = s / safe[..., None]
        if criterion == "gini":
            impurity = 1.0 - np.sum(p * p, axis=-1)
        else:
            positive = p > 0
            impurity = -np.sum(np.where(positive, p * np.log2(np.where(positive, p, 1.0)), 0.0), axis=-1)
    return np.where(total > 0, impurity, 0.0), total


def _gains(left: np.ndarray, right: np.ndarray, parent: float, criterion: str) -> np.ndarray:
    """Gains of many candidate splits at once from the summed statistics of their children."""
    imp_left, w_left = _impurity_from_stats(left, criterion)
    imp_right, w_right = _impurity_from_stats(right, criterion)
    total = w_left + w_right
    gain = parent - (w_left / total) * imp_left - (w_right / total) * imp_right
    return np.where((w_left > 0) & (w_right > 0), gain, 0.0)   # a child without weight: no split


def _best_thresholds(X, stats, features, min_samples_leaf, parent, criterion):
    """splitter="best": every midpoint of every feature, vectorised (sort once, cumulative sums)."""
    n = X.shape[0]
    gains, feats, thresholds = [], [], []
    chunk = max(1, 4_000_000 // max(1, n * stats.shape[1]))         # bounded memory for (n, chunk, S)
    n_left = np.arange(1, n)[:, None]
    allowed = (n_left >= min_samples_leaf) & (n - n_left >= min_samples_leaf)
    for start in range(0, len(features), chunk):
        fs = features[start:start + chunk]
        Xf = X[:, fs]
        order = np.argsort(Xf, axis=0, kind="stable")
        xs = np.take_along_axis(Xf, order, axis=0)
        sorted_stats = stats[order]                                 # (n, m, S)
        left = np.cumsum(sorted_stats, axis=0)[:-1]                 # split after position i: i + 1 samples left
        right = np.cumsum(sorted_stats[::-1], axis=0)[::-1][1:]     # summed from the end: exact zeros stay zeros
        pos, col = np.nonzero((xs[1:] > xs[:-1]) & allowed)         # between two distinct values only
        if pos.size == 0:
            continue
        low, high = xs[pos, col], xs[pos + 1, col]
        threshold = low / 2.0 + high / 2.0                          # the midpoint, as scikit-learn computes it
        threshold = np.where((threshold == high) | ~np.isfinite(threshold), low, threshold)
        gains.append(_gains(left[pos, col], right[pos, col], parent, criterion))
        feats.append(fs[col])
        thresholds.append(threshold)
    if not gains:
        return np.empty(0), np.empty(0, dtype=int), np.empty(0)
    return np.concatenate(gains), np.concatenate(feats), np.concatenate(thresholds)


def _random_thresholds(X, stats, features, min_samples_leaf, parent, criterion, rng):
    """splitter="random": one threshold per non-constant candidate feature, drawn with rng.uniform(min, max)."""
    n = X.shape[0]
    feats, thresholds = [], []
    for f in features:                          # in the given order: one draw per non-constant feature
        low, high = X[:, f].min(), X[:, f].max()
        if low == high:
            continue
        feats.append(int(f))
        thresholds.append(float(rng.uniform(low, high)))
    if not feats:
        return np.empty(0), np.empty(0, dtype=int), np.empty(0)
    feats, thresholds = np.array(feats), np.array(thresholds)
    masks = X[:, feats] <= thresholds                               # (n, m)
    n_left = masks.sum(axis=0)
    valid = (n_left >= min_samples_leaf) & (n - n_left >= min_samples_leaf)
    left = masks.T.astype(float) @ stats
    right = (~masks).T.astype(float) @ stats
    gains = _gains(left, right, parent, criterion)
    return gains[valid], feats[valid], thresholds[valid]


def _check_hyperparameters(tree, criteria: tuple[str, ...]) -> None:
    if not (isinstance(tree.criterion, str) and tree.criterion in criteria):
        raise ValueError(f"criterion must be one of {criteria}, got {tree.criterion!r}")
    if tree.max_depth is not None and not (_is_int(tree.max_depth) and tree.max_depth >= 1):
        raise ValueError(f"max_depth must be None or an int >= 1, got {tree.max_depth!r}")
    if not (_is_int(tree.min_samples_split) and tree.min_samples_split >= 2):
        raise ValueError(f"min_samples_split must be an int >= 2, got {tree.min_samples_split!r}")
    if not (_is_int(tree.min_samples_leaf) and tree.min_samples_leaf >= 1):
        raise ValueError(f"min_samples_leaf must be an int >= 1, got {tree.min_samples_leaf!r}")
    if not (isinstance(tree.splitter, str) and tree.splitter in _SPLITTERS):
        raise ValueError(f"splitter must be 'best' or 'random', got {tree.splitter!r}")


def _n_candidate_features(max_features, d: int) -> int | None:
    """Number k of features drawn at every node (None: all of them, no draw)."""
    if max_features is None:
        return None
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if max_features == "log2":
            return max(1, int(np.log2(d)))
    elif _is_int(max_features):
        if 1 <= max_features <= d:
            return int(max_features)
    elif isinstance(max_features, (float, np.floating)):
        if 0.0 < max_features <= 1.0:
            return max(1, int(max_features * d))
    raise ValueError(f"max_features must be None, an int in 1..{d}, a float in (0, 1], 'sqrt' or 'log2', "
                     f"got {max_features!r}")


def _grow(tree, X: np.ndarray, y: np.ndarray, w: np.ndarray, criterion: str, value_of) -> tuple:
    """Grow the nodes depth-first (left child first) with an explicit stack; return (root, importances)."""
    impurity = _impurity_function(criterion)
    n, d = X.shape
    k = _n_candidate_features(tree.max_features, d)
    rng = np.random.default_rng(tree.random_state)
    importances = np.zeros(d)
    w_root = w.sum()

    def new_node(rows):
        y_rows, w_rows = y[rows], w[rows]
        return Node(feature=None, threshold=None, left=None, right=None, value=value_of(y_rows, w_rows),
                    n_samples=int(len(rows)), impurity=float(impurity(y_rows, w_rows)))

    rows = np.arange(n)
    root = new_node(rows)
    stack = [(root, rows, 0)]
    while stack:
        node, rows, depth = stack.pop()
        if (_is_pure(y[rows], w[rows], criterion)
                or (tree.max_depth is not None and depth >= tree.max_depth)
                or len(rows) < tree.min_samples_split):
            continue                                                # a leaf
        features = None if k is None else rng.choice(d, size=k, replace=False)
        split = best_split(X[rows], y[rows], criterion, w[rows], features, tree.min_samples_leaf,
                           tree.splitter, rng)
        if split is None:
            continue                                                # no split with a positive gain: a leaf
        feature, threshold, gain = split
        go_left = X[rows, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left, node.right = new_node(rows[go_left]), new_node(rows[~go_left])
        importances[feature] += w[rows].sum() / w_root * gain
        stack.append((node.right, rows[~go_left], depth + 1))
        stack.append((node.left, rows[go_left], depth + 1))         # popped first: left subtree first
    total = importances.sum()
    if total > 0:
        importances /= total
    return root, importances


def _leaf_values(root: Node, X: np.ndarray, width: int) -> np.ndarray:
    """(n, width) array: the value of the leaf reached by every row of X (all rows routed together)."""
    out = np.empty((X.shape[0], width))
    stack = [(root, np.arange(X.shape[0]))]
    while stack:
        node, rows = stack.pop()
        if rows.size == 0:
            continue
        if node.left is None:
            out[rows] = node.value
            continue
        go_left = X[rows, node.feature] <= node.threshold
        stack.append((node.right, rows[~go_left]))
        stack.append((node.left, rows[go_left]))
    return out


def _check_fitted(tree) -> Node:
    if not hasattr(tree, "root_"):
        raise RuntimeError(f"this {type(tree).__name__} is not fitted yet: call fit(X, y) first")
    return tree.root_


def _depth(root: Node) -> int:
    deepest, stack = 0, [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.left is None:
            deepest = max(deepest, depth)
        else:
            stack.extend([(node.left, depth + 1), (node.right, depth + 1)])
    return deepest


def _n_leaves(root: Node) -> int:
    count, stack = 0, [root]
    while stack:
        node = stack.pop()
        if node.left is None:
            count += 1
        else:
            stack.extend([node.left, node.right])
    return count


def gini_impurity(y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Compute the Gini impurity ``1 - sum_k p_k^2`` of a set of labels.

    ``p_k`` is the (weighted) proportion of class k: the sum of the weights of the
    samples of class k divided by the total weight. 0 for a pure node, maximal
    (1 - 1/K) when the K classes are equally represented.

    Parameters
    ----------
    y : array-like of shape (n,)
        Class labels (any values).
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.

    Returns
    -------
    float
        Impurity in [0, 1 - 1/K].

    Raises
    ------
    ValueError
        If ``y`` is empty, or the weights are negative, sum to 0 or do not have the
        shape of ``y``.

    Notes
    -----
    Tested on hand-computed values and against the root impurity
    ``tree_.impurity[0]`` of ``sklearn.tree.DecisionTreeClassifier(max_depth=1)``.

    Examples
    --------
    >>> gini_impurity([0, 0, 1, 1])
    0.5
    >>> gini_impurity(["a", "a", "a", "b"])
    0.375
    >>> gini_impurity([0, 1], sample_weight=[3.0, 1.0])
    0.375
    """
    p = _class_proportions(y, sample_weight)
    return float(1.0 - np.sum(p * p))


def entropy_impurity(y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Compute the Shannon entropy in bits ``-sum_k p_k log2 p_k`` of a set of labels.

    ``p_k`` is the (weighted) proportion of class k; by convention 0 log 0 = 0 (only the
    classes present are summed). 0 for a pure node, 1 bit for two balanced classes.

    Parameters
    ----------
    y : array-like of shape (n,)
        Class labels (any values).
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.

    Returns
    -------
    float
        Entropy >= 0, in bits.

    Raises
    ------
    ValueError
        If ``y`` is empty or the weights are invalid (see ``gini_impurity``).

    Notes
    -----
    Tested against ``scipy.stats.entropy(p, base=2)`` and the root impurity of
    ``sklearn.tree.DecisionTreeClassifier(criterion="entropy")`` (also in base 2).

    Examples
    --------
    >>> entropy_impurity([0, 0, 1, 1])
    1.0
    >>> round(entropy_impurity([0, 0, 0, 1]), 4)
    0.8113
    """
    p = _class_proportions(y, sample_weight)
    p = p[p > 0]                                    # 0 log 0 = 0
    return float(-np.sum(p * np.log2(p))) + 0.0     # + 0.0 turns -0.0 into 0.0


def variance_impurity(y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Compute the (weighted) mean squared deviation from the (weighted) mean.

    Impurity of a regression node: ``sum_i w_i (y_i - ybar)^2 / sum_i w_i`` with
    ``ybar = sum_i w_i y_i / sum_i w_i`` (population variance when all weights are 1).

    Parameters
    ----------
    y : array-like of shape (n,)
        Real targets.
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.

    Returns
    -------
    float
        Variance >= 0.

    Raises
    ------
    ValueError
        If ``y`` is empty or the weights are invalid (see ``gini_impurity``).

    Notes
    -----
    Tested against ``np.average`` and the root impurity ``tree_.impurity[0]`` of
    ``sklearn.tree.DecisionTreeRegressor(max_depth=1)``.

    Examples
    --------
    >>> variance_impurity([1.0, 2.0, 3.0, 4.0])
    1.25
    """
    y = _check_targets(y)
    w = _check_weights(sample_weight, len(y))
    mean = np.average(y, weights=w)
    return float(np.average((y - mean) ** 2, weights=w))


def split_gain(
    y: ArrayLike,
    left_mask: ArrayLike,
    criterion: str = "gini",
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute the impurity decrease of a split.

    ``gain = I(parent) - (w_L / w) I(left) - (w_R / w) I(right)`` where I is the
    impurity of the criterion and w_L, w_R, w the total weights of the left child, the
    right child and the parent (the numbers of samples when all weights are 1).

    Parameters
    ----------
    y : array-like of shape (n,)
        Labels (classification) or targets (regression) of the parent node.
    left_mask : array-like of shape (n,)
        Booleans: True sends the sample to the left child, False to the right child.
    criterion : str, default="gini"
        "gini", "entropy" or "squared_error" (variance).
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.

    Returns
    -------
    float
        The gain (>= 0 up to rounding); 0.0 if one child is empty.

    Raises
    ------
    ValueError
        If ``criterion`` is unknown or the shapes of ``y``, ``left_mask`` and
        ``sample_weight`` differ.

    Notes
    -----
    Tested on hand-computed values and for consistency with the impurities of the two
    children in ``tree_.impurity`` of scikit-learn trees.

    Examples
    --------
    >>> split_gain([0, 0, 1, 1], [True, True, False, False])
    0.5
    >>> split_gain([0, 0, 1, 1], [True, True, False, False], criterion="entropy")
    1.0
    """
    impurity = _impurity_function(criterion)
    y = _check_targets(y) if criterion == "squared_error" else _check_labels(y)
    mask = np.asarray(left_mask)
    if mask.shape != y.shape:
        raise ValueError(f"left_mask must have the shape of y {y.shape}, got {mask.shape}")
    mask = mask.astype(bool)
    w = _check_weights(sample_weight, len(y))
    w_left, w_right = w[mask].sum(), w[~mask].sum()
    if w_left == 0 or w_right == 0:                 # an empty child (or one without weight): nothing split
        return 0.0
    total = w_left + w_right
    return float(impurity(y, w) - w_left / total * impurity(y[mask], w[mask])
                 - w_right / total * impurity(y[~mask], w[~mask]))


def best_split(
    X: ArrayLike,
    y: ArrayLike,
    criterion: str = "gini",
    sample_weight: ArrayLike | None = None,
    feature_indices: ArrayLike | None = None,
    min_samples_leaf: int = 1,
    splitter: str = "best",
    rng: np.random.Generator | None = None,
) -> tuple[int, float, float] | None:
    """Find the best (feature, threshold) split of a node.

    Samples with ``X[:, feature] <= threshold`` go left, the others right. Candidate
    thresholds, for every candidate feature:

    - ``splitter="best"``: the midpoints between consecutive distinct sorted values of
      the feature (e.g. 3.5 between 3 and 4), as scikit-learn;
    - ``splitter="random"`` (ExtraTrees): a single threshold drawn uniformly in
      (min, max) of the feature with ``rng.uniform`` (a constant feature is skipped).

    A candidate is valid only if each child gets at least ``min_samples_leaf`` samples.
    The valid candidate with the largest gain (``split_gain``) wins; ties go to the
    lowest feature index, then to the lowest threshold.

    Parameters
    ----------
    X : array-like of shape (n, n_features)
        Samples of the node (no NaN).
    y : array-like of shape (n,)
        Labels or targets of the node.
    criterion : str, default="gini"
        "gini", "entropy" or "squared_error".
    sample_weight : array-like of shape (n,) or None, default=None
        Non-negative weights; None means all 1.
    feature_indices : array-like of int or None, default=None
        Features to consider (the random subset of a forest); None means all.
    min_samples_leaf : int, default=1
        Minimum number of samples in each child.
    splitter : str, default="best"
        "best" or "random".
    rng : np.random.Generator or None, default=None
        Generator used by ``splitter="random"``; ``np.random.default_rng()`` if None.

    Returns
    -------
    tuple of (int, float, float) or None
        ``(feature, threshold, gain)`` as Python int and floats, or None when no valid
        split has a strictly positive gain (the node becomes a leaf; scikit-learn may
        still split with a zero gain).

    Raises
    ------
    ValueError
        If ``criterion`` or ``splitter`` is unknown, or the shapes are inconsistent.

    Notes
    -----
    Tested against a brute-force search and against the root feature and threshold of
    ``sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor(max_depth=1)`` on
    continuous data without ties.

    Examples
    --------
    >>> X = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> best_split(X, y)
    (0, 2.5, 0.5)
    """
    impurity = _impurity_function(criterion)
    if not (isinstance(splitter, str) and splitter in _SPLITTERS):
        raise ValueError(f"unknown splitter {splitter!r}: expected 'best' or 'random'")
    X = _check_X(X)
    n, d = X.shape
    y = _check_targets(y) if criterion == "squared_error" else _check_labels(y)
    if len(y) != n:
        raise ValueError(f"X has {n} samples but y has {len(y)}")
    w = _check_weights(sample_weight, n)
    if not (_is_int(min_samples_leaf) and min_samples_leaf >= 1):
        raise ValueError(f"min_samples_leaf must be an int >= 1, got {min_samples_leaf!r}")
    if feature_indices is None:
        features = np.arange(d)
    else:
        features = np.asarray(feature_indices).ravel()
        if features.size and (features.dtype.kind not in "iu" or features.min() < 0 or features.max() >= d):
            raise ValueError(f"feature_indices must be integers in 0..{d - 1}, got {feature_indices!r}")
        features = features.astype(int)
    if _is_pure(y, w, criterion) or n < 2 * min_samples_leaf or features.size == 0:
        return None
    parent = impurity(y, w)
    tolerance = _GAIN_RTOL * parent
    stats = _node_stats(y, w, criterion)
    parent_from_stats = float(_impurity_from_stats(stats.sum(axis=0), criterion)[0])
    if splitter == "best":
        gains, feats, thresholds = _best_thresholds(X, stats, features, min_samples_leaf, parent_from_stats,
                                                    criterion)
    else:
        gains, feats, thresholds = _random_thresholds(X, stats, features, min_samples_leaf, parent_from_stats,
                                                      criterion, np.random.default_rng(rng))
    if gains.size == 0 or not gains.max() > tolerance:
        return None
    tied = np.flatnonzero(gains >= gains.max() - tolerance)          # the best gain, up to rounding
    first = tied[np.lexsort((thresholds[tied], feats[tied]))[0]]     # lowest feature, then lowest threshold
    feature, threshold = int(feats[first]), float(thresholds[first])
    gain = split_gain(y, X[:, feature] <= threshold, criterion, w)   # the exact gain, as split_gain computes it
    if not gain > tolerance:
        return None
    return feature, threshold, gain


@dataclass(eq=False)
class Node:
    """One node of a decision tree (a leaf when ``left`` is None).

    Provided: already implemented, no exercise. A dataclass: ``Node(...)`` stores the
    fields below, nothing more. An internal node sends a sample to ``left`` when
    ``x[feature] <= threshold``, else to ``right``.

    Parameters
    ----------
    feature : int or None
        Index of the split feature (None for a leaf).
    threshold : float or None
        Split threshold (None for a leaf).
    left : Node or None
        Left child (None for a leaf).
    right : Node or None
        Right child (None for a leaf).
    value : np.ndarray
        Classifier: shape (n_classes,), weighted class proportions of the node, aligned
        on the ``classes_`` of the tree. Regressor: shape (1,), weighted mean target.
    n_samples : int
        Number of training samples that reached the node.
    impurity : float
        Impurity of the node for the criterion of the tree.

    Examples
    --------
    >>> leaf = Node(feature=None, threshold=None, left=None, right=None,
    ...             value=np.array([1.0, 0.0]), n_samples=2, impurity=0.0)
    >>> leaf.left is None
    True
    """

    feature: int | None
    threshold: float | None
    left: Node | None
    right: Node | None
    value: np.ndarray
    n_samples: int
    impurity: float


class DecisionTreeClassifier:
    """Decision tree classifier grown greedily and recursively (CART).

    ``fit`` builds ``root_``: a node becomes a leaf when it is pure, when
    ``max_depth`` is reached, when it has fewer than ``min_samples_split`` samples, or
    when ``best_split`` finds no valid split with a positive gain. Otherwise it is split
    by ``best_split`` and both children are built the same way. A sample is predicted
    by the ``value`` of the leaf it reaches.

    At every node, the candidate features are ``max_features`` features drawn without
    replacement (``rng.choice``) with the generator ``np.random.default_rng(random_state)``
    created in ``fit``; with ``max_features=None`` all features are candidates.

    Parameters
    ----------
    criterion : str, default="gini"
        "gini" or "entropy".
    max_depth : int or None, default=None
        Maximum depth (>= 1); None grows the tree until the leaves are pure.
    min_samples_split : int, default=2
        A node with fewer samples is not split (>= 2).
    min_samples_leaf : int, default=1
        Minimum number of samples in each child (>= 1).
    max_features : int, float, str or None, default=None
        Features examined at each node: None (all d features), an int k in 1..d, a
        float f in (0, 1] (``max(1, int(f * d))``), "sqrt" (``max(1, int(sqrt(d)))``)
        or "log2" (``max(1, int(log2(d)))``).
    splitter : str, default="best"
        "best" or "random" (see ``best_split``).
    random_state : int or None, default=None
        Seed of the generator used for ``max_features`` and ``splitter="random"``.

    Attributes
    ----------
    root_ : Node
        Root of the fitted tree.
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_importances_ : np.ndarray of shape (n_features,)
        For each feature, the sum over the nodes split on it of
        ``(w_node / w_root) * gain``, normalised to sum to 1 (all zeros if the root is
        a leaf).

    Notes
    -----
    Tested against ``sklearn.tree.DecisionTreeClassifier`` with the same
    hyperparameters (``max_features=None``) on continuous data without ties: identical
    ``predict``, ``predict_proba``, ``get_depth`` and ``get_n_leaves``, allclose
    ``feature_importances_``; ``sample_weight=2`` equals duplicating the samples.
    ``max_features`` and ``splitter`` are tested by properties (same tree with the same
    ``random_state``, different trees with different seeds).

    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> tree = DecisionTreeClassifier().fit(X, y)
    >>> tree.root_.feature, tree.root_.threshold
    (0, 2.5)
    >>> tree.get_depth(), tree.get_n_leaves()
    (1, 2)
    >>> tree.predict(np.array([[1.5], [3.5]]))
    array([0, 1])
    >>> tree.predict_proba(np.array([[1.5], [3.5]]))
    array([[1., 0.],
           [0., 1.]])
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        splitter: str = "best",
        random_state: int | None = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self:
        """Grow the tree on (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples (no NaN).
        y : array-like of shape (n_samples,)
            Labels (any sortable values).
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Non-negative weights (AdaBoost); None means all 1.

        Returns
        -------
        Self
            The fitted tree (``self``), with ``root_``, ``classes_``, ``n_classes_``,
            ``n_features_in_`` and ``feature_importances_``.

        Raises
        ------
        ValueError
            If a hyperparameter is invalid, X is not 2-D or contains NaN,
            ``len(X) != len(y)``, or ``sample_weight`` is negative or misshaped.
        """
        _check_hyperparameters(self, _CLASSIFICATION_CRITERIA)
        X = _check_X(X)
        y = _check_labels(y)
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)}")
        w = _check_weights(sample_weight, len(X))
        classes, codes = np.unique(y, return_inverse=True)
        codes = codes.ravel()
        n_classes = len(classes)

        def proportions(codes_node, w_node):
            return np.bincount(codes_node, weights=w_node, minlength=n_classes) / w_node.sum()

        root, importances = _grow(self, X, codes, w, self.criterion, proportions)
        self.classes_ = classes
        self.n_classes_ = int(n_classes)
        self.n_features_in_ = int(X.shape[1])
        self.root_ = root
        self.feature_importances_ = importances
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the class proportions of the leaf reached by every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Weighted class proportions of the leaves; column k is ``classes_[k]``.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        ValueError
            If X has a number of features different from ``n_features_in_``.
        """
        root = _check_fitted(self)
        X = _check_X(X, self.n_features_in_)
        return _leaf_values(root, X, self.n_classes_)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the most probable class of the leaf reached by every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_`` (ties: the first class).

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]       # argmax keeps the first of tied classes

    def get_depth(self) -> int:
        """Return the depth of the tree (0 when the root is a leaf).

        Returns
        -------
        int
            Largest number of edges between the root and a leaf.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        return _depth(_check_fitted(self))

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the tree.

        Returns
        -------
        int
            Number of nodes without children.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        return _n_leaves(_check_fitted(self))

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction of correct predictions, in [0, 1].
        """
        y = _check_labels(y)
        predictions = self.predict(X)
        if len(y) != len(predictions):
            raise ValueError(f"X has {len(predictions)} samples but y has {len(y)}")
        return float(np.mean(predictions == y))


class DecisionTreeRegressor:
    """Decision tree regressor: the same greedy tree with the variance criterion.

    Grown exactly like ``DecisionTreeClassifier`` (same stopping rules, same
    ``max_features`` and ``splitter`` options), with ``variance_impurity`` as impurity;
    every leaf predicts the (weighted) mean target of its training samples. The
    gradient boosting of chapter 14 is made of such trees.

    Parameters
    ----------
    criterion : str, default="squared_error"
        Only "squared_error" (variance) is supported.
    max_depth : int or None, default=None
        Maximum depth (>= 1); None grows the tree until the leaves are pure.
    min_samples_split : int, default=2
        A node with fewer samples is not split (>= 2).
    min_samples_leaf : int, default=1
        Minimum number of samples in each child (>= 1).
    max_features : int, float, str or None, default=None
        Features examined at each node, as in ``DecisionTreeClassifier``.
    splitter : str, default="best"
        "best" or "random".
    random_state : int or None, default=None
        Seed of the generator used for ``max_features`` and ``splitter="random"``.

    Attributes
    ----------
    root_ : Node
        Root of the fitted tree (``value`` of shape (1,) in every node).
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_importances_ : np.ndarray of shape (n_features,)
        Normalised weighted variance decrease per feature (see
        ``DecisionTreeClassifier``).

    Notes
    -----
    Tested against ``sklearn.tree.DecisionTreeRegressor`` with the same hyperparameters
    on continuous data without ties: allclose predictions, same depth and number of
    leaves.

    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y = np.array([1.0, 1.2, 3.0, 3.4])
    >>> reg = DecisionTreeRegressor(max_depth=1).fit(X, y)
    >>> reg.predict(np.array([[0.0], [5.0]])).round(4)
    array([1.1, 3.2])
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        splitter: str = "best",
        random_state: int | None = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self:
        """Grow the regression tree on (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples (no NaN).
        y : array-like of shape (n_samples,)
            Real targets.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Non-negative weights; None means all 1.

        Returns
        -------
        Self
            The fitted tree (``self``), with ``root_``, ``n_features_in_`` and
            ``feature_importances_``.

        Raises
        ------
        ValueError
            If a hyperparameter is invalid (including a criterion other than
            "squared_error"), X is not 2-D or contains NaN, ``len(X) != len(y)``, or
            ``sample_weight`` is negative or misshaped.
        """
        _check_hyperparameters(self, ("squared_error",))
        X = _check_X(X)
        y = _check_targets(y)
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)}")
        w = _check_weights(sample_weight, len(X))

        def mean(y_node, w_node):
            return np.array([np.average(y_node, weights=w_node)])

        root, importances = _grow(self, X, y, w, "squared_error", mean)
        self.n_features_in_ = int(X.shape[1])
        self.root_ = root
        self.feature_importances_ = importances
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the mean target of the leaf reached by every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Float predictions.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        ValueError
            If X has a number of features different from ``n_features_in_``.
        """
        root = _check_fitted(self)
        X = _check_X(X, self.n_features_in_)
        return _leaf_values(root, X, 1)[:, 0]

    def get_depth(self) -> int:
        """Return the depth of the tree (0 when the root is a leaf).

        Returns
        -------
        int
            Largest number of edges between the root and a leaf.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        return _depth(_check_fitted(self))

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the tree.

        Returns
        -------
        int
            Number of nodes without children.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        """
        return _n_leaves(_check_fitted(self))

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            True targets.

        Returns
        -------
        float
            Coefficient of determination R² (as ``linear.r2_score``, chapter 9).
        """
        y = _check_targets(y)
        predictions = self.predict(X)
        if len(y) != len(predictions):
            raise ValueError(f"X has {len(predictions)} samples but y has {len(y)}")
        ss_res = float(np.sum((y - predictions) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:                              # constant target (as scikit-learn's r2_score)
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
