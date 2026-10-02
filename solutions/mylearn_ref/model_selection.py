"""Splits, cloning and cross-validation — mylearn, chapter 8 (Training and Testing).

Honest evaluation tools: hold-out split (``train_test_split``), k-fold indices (plain
and stratified), ``clone`` (a fresh, unfitted copy of an estimator) and
``cross_val_score``. Every later chapter evaluates its models with them.

Reference implementation: read it only after trying (``mon_travail/mylearn/model_selection.py``).
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike


def _n_test(test_size: float | int, n_samples: int) -> int:
    """Number of test samples for ``test_size`` (float fraction or int count), validated."""
    if isinstance(test_size, (bool, np.bool_)):
        raise ValueError(f"test_size must be a float in (0, 1) or an int, got {test_size!r}")
    if isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
    elif isinstance(test_size, (float, np.floating)):
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"a float test_size must be in (0, 1), got {test_size}")
        n_test = math.ceil(test_size * n_samples)
    else:
        raise ValueError(f"test_size must be a float in (0, 1) or an int, got {type(test_size).__name__}")
    if not 0 < n_test < n_samples:
        raise ValueError(f"test_size={test_size!r} gives {n_test} test samples out of {n_samples}: "
                         "both parts must be non-empty")
    return n_test


def _check_n_splits(n_splits: int, n_samples: int) -> None:
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if n_splits > n_samples:
        raise ValueError(f"n_splits={n_splits} is larger than the number of samples ({n_samples})")


def train_test_split(
    *arrays: ArrayLike,
    test_size: float | int = 0.25,
    shuffle: bool = True,
    stratify: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Split several aligned arrays into a train part and a test part.

    All arrays have the same number of rows (e.g. X and y); the same rows go to the
    test part in every array, so X and y stay aligned. With ``shuffle=False`` the last
    rows form the test part. With ``stratify=y`` each class keeps (up to rounding) the
    same proportion in both parts.

    Parameters
    ----------
    *arrays : array-like
        One or more arrays with the same first dimension n.
    test_size : float or int, default=0.25
        Float in (0, 1): ``n_test = ceil(test_size * n)``. Int: number of test samples.
    shuffle : bool, default=True
        Shuffle the rows before splitting.
    stratify : array-like of shape (n,) or None, default=None
        Labels whose class proportions must be preserved (requires ``shuffle=True``).
    rng : np.random.Generator or None, default=None
        Random generator; ``np.random.default_rng()`` if None.

    Returns
    -------
    list of np.ndarray
        ``[a1_train, a1_test, a2_train, a2_test, ...]`` (scikit-learn order).

    Raises
    ------
    ValueError
        If no array is given, the lengths differ, ``test_size`` is out of range or
        leaves an empty part, ``stratify`` is used with ``shuffle=False``, or a class of
        ``stratify`` has a single member.

    Notes
    -----
    Tested by properties (disjoint parts whose union is everything, rows stay aligned,
    same seed gives the same split) and against
    ``sklearn.model_selection.train_test_split`` for the sizes: same test size; with
    ``stratify``, per-class counts within 1 of scikit-learn's.

    Examples
    --------
    >>> X = np.arange(10).reshape(5, 2)
    >>> y = np.array([0, 1, 0, 1, 0])
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
    >>> X_test
    array([[6, 7],
           [8, 9]])
    >>> y_train
    array([0, 1, 0])
    """
    if not arrays:
        raise ValueError("train_test_split needs at least one array")
    arrays = [np.asarray(a) for a in arrays]
    if any(a.ndim == 0 for a in arrays):
        raise ValueError("every array must have at least one dimension (one row per sample)")
    n_samples = len(arrays[0])
    lengths = [len(a) for a in arrays]
    if any(length != n_samples for length in lengths):
        raise ValueError(f"all arrays must have the same number of rows, got {lengths}")
    n_test = _n_test(test_size, n_samples)
    if rng is None:
        rng = np.random.default_rng()

    if stratify is not None:
        if not shuffle:
            raise ValueError("stratify requires shuffle=True")
        labels = np.asarray(stratify)
        if len(labels) != n_samples:
            raise ValueError(f"stratify has {len(labels)} labels for {n_samples} rows")
        _, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if counts.min() < 2:
            raise ValueError("every class of stratify needs at least 2 members")
        # test samples per class: floor of the exact share, then +1 for the largest remainders
        exact = counts * n_test / n_samples
        per_class = np.floor(exact).astype(int)
        missing = n_test - int(per_class.sum())
        per_class[np.argsort(-(exact - per_class), kind="stable")[:missing]] += 1
        test_idx = np.concatenate([rng.permutation(np.flatnonzero(codes == c))[:k]
                                   for c, k in enumerate(per_class)])
        test_idx = rng.permutation(test_idx)
        is_test = np.zeros(n_samples, dtype=bool)
        is_test[test_idx] = True
        train_idx = rng.permutation(np.flatnonzero(~is_test))
    elif shuffle:
        permutation = rng.permutation(n_samples)
        test_idx, train_idx = permutation[:n_test], permutation[n_test:]
    else:
        train_idx = np.arange(n_samples - n_test)
        test_idx = np.arange(n_samples - n_test, n_samples)

    result = []
    for a in arrays:
        result.extend([a[train_idx], a[test_idx]])
    return result


def kfold_indices(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the (train, validation) indices of each of the k folds.

    The indices 0..n_samples-1 (permuted once if ``shuffle``) are cut into
    ``n_splits`` consecutive folds; the first ``n_samples % n_splits`` folds get one
    extra sample. Fold i is the validation set of split i, the other folds its train set.

    Parameters
    ----------
    n_samples : int
        Number of samples n.
    n_splits : int, default=5
        Number of folds k (>= 2).
    shuffle : bool, default=False
        Permute the indices once before cutting the folds.
    rng : np.random.Generator or None, default=None
        Random generator used when ``shuffle=True``; ``np.random.default_rng()`` if None.

    Returns
    -------
    list of tuple of (np.ndarray, np.ndarray)
        ``n_splits`` pairs ``(train_idx, val_idx)`` of int arrays; the validation sets
        are disjoint and together cover ``range(n_samples)``.

    Raises
    ------
    ValueError
        If ``n_splits < 2`` or ``n_splits > n_samples``.

    Notes
    -----
    Tested against ``sklearn.model_selection.KFold(n_splits).split(X)`` (without
    shuffle): identical indices.

    Examples
    --------
    >>> for train_idx, val_idx in kfold_indices(5, n_splits=2):
    ...     print(train_idx, val_idx)
    [3 4] [0 1 2]
    [0 1 2] [3 4]
    """
    _check_n_splits(n_splits, n_samples)
    indices = np.arange(n_samples)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.permutation(n_samples)
    sizes = np.full(n_splits, n_samples // n_splits)
    sizes[: n_samples % n_splits] += 1
    splits = []
    start = 0
    for size in sizes:
        val_idx = np.sort(indices[start:start + size])
        train_idx = np.sort(np.concatenate([indices[:start], indices[start + size:]]))
        splits.append((train_idx, val_idx))
        start += size
    return splits


def stratified_kfold_indices(
    y: ArrayLike,
    n_splits: int = 5,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k-fold indices whose validation folds keep the class proportions of y.

    The sample indices are sorted by class (stable sort; with ``shuffle`` the indices
    are first shuffled within each class), then dealt round-robin over the folds:
    fold i gets the positions i, i + k, i + 2k, ... of this class-sorted order. Both
    the class counts and the fold sizes are therefore balanced.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels.
    n_splits : int, default=5
        Number of folds k (>= 2).
    shuffle : bool, default=False
        Shuffle the indices within each class first.
    rng : np.random.Generator or None, default=None
        Random generator used when ``shuffle=True``; ``np.random.default_rng()`` if None.

    Returns
    -------
    list of tuple of (np.ndarray, np.ndarray)
        ``n_splits`` pairs ``(train_idx, val_idx)`` of int arrays, each sorted in
        increasing order; the validation sets partition ``range(n_samples)``.

    Raises
    ------
    ValueError
        If ``n_splits < 2`` or ``n_splits > n_samples``.

    Warns
    -----
    UserWarning
        If a class has fewer than ``n_splits`` members (some folds will miss it).

    Notes
    -----
    Tested against ``sklearn.model_selection.StratifiedKFold`` (same allocation rule):
    for every class the per-fold counts are identical; fold sizes differ by at most 1;
    the validation folds partition ``range(n)``. The samples themselves may land in
    different folds.

    Examples
    --------
    >>> for train_idx, val_idx in stratified_kfold_indices([0, 0, 0, 0, 1, 1], n_splits=2):
    ...     print(train_idx, val_idx)
    [1 3 5] [0 2 4]
    [0 2 4] [1 3 5]
    """
    y = np.asarray(y)
    n_samples = len(y)
    _check_n_splits(n_splits, n_samples)
    _, codes, counts = np.unique(y, return_inverse=True, return_counts=True)
    if counts.min() < n_splits:
        warnings.warn(f"The least populated class in y has only {counts.min()} members, which is less "
                      f"than n_splits={n_splits}: some validation folds will miss it.", UserWarning,
                      stacklevel=2)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        order = np.concatenate([rng.permutation(np.flatnonzero(codes == c)) for c in range(len(counts))])
    else:
        order = np.argsort(codes, kind="stable")
    everything = np.arange(n_samples)
    splits = []
    for i in range(n_splits):
        val_idx = np.sort(order[i::n_splits])
        splits.append((np.setdiff1d(everything, val_idx), val_idx))
    return splits


def clone(estimator: Any) -> Any:
    """Return a new unfitted estimator with the same hyperparameters.

    Relies on the convention "``__init__`` only stores its arguments, under their own
    names": the hyperparameters are the attributes (``vars(estimator)``) whose name
    neither starts nor ends with ``_``; learnt attributes (``coef_``...) and private
    ones (``_rng``...) are left out. The result is ``type(estimator)(**params)`` with
    deep copies of the parameter values.

    Parameters
    ----------
    estimator : object
        Any estimator following the convention (mylearn and scikit-learn estimators).

    Returns
    -------
    object
        A fresh estimator of the same class, not fitted.

    Raises
    ------
    TypeError
        Propagated from ``__init__`` if it does not accept the stored attributes (the
        convention is not followed).

    Notes
    -----
    Tested against the behaviour of ``sklearn.base.clone``: same hyperparameters, no
    fitted attribute, ``clone(e) is not e``.

    Examples
    --------
    >>> from mylearn.cluster import KMeans
    >>> km = KMeans(n_clusters=3, random_state=0)
    >>> km2 = clone(km)
    >>> km2 is km, km2.n_clusters, km2.random_state
    (False, 3, 0)
    """
    params = {name: copy.deepcopy(value) for name, value in vars(estimator).items()
              if not name.startswith("_") and not name.endswith("_")}
    return type(estimator)(**params)


def cross_val_score(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
    cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 5,
    scoring: Callable[[Any, np.ndarray, np.ndarray], float] | None = None,
) -> np.ndarray:
    """Evaluate an estimator by cross-validation.

    For each ``(train_idx, val_idx)`` split: clone the estimator, fit the clone on the
    train rows and score it on the validation rows. The estimator passed in is never
    fitted.

    Parameters
    ----------
    estimator : object
        Unfitted estimator with ``fit`` and ``predict`` (and ``score`` if ``scoring``
        is None).
    X : array-like of shape (n_samples, n_features)
        Samples.
    y : array-like of shape (n_samples,)
        Targets.
    cv : int or sequence of (train_idx, val_idx), default=5
        Int k: ``kfold_indices(n_samples, k)`` without shuffle. Otherwise an explicit
        list of index pairs, e.g. from ``stratified_kfold_indices``.
    scoring : callable or None, default=None
        A scorer, with the scikit-learn convention
        ``scoring(fitted_estimator, X_val, y_val) -> float`` (higher is better). It
        receives the fitted clone, so it can use ``predict_proba`` (AUC, log loss...).
        To use a metric ``m(y_true, y_pred)``, wrap it:
        ``lambda est, X, y: m(y, est.predict(X))`` (and negate a loss).
        None uses ``estimator.score(X_val, y_val)`` (accuracy for a mylearn classifier,
        R² for a regressor).

    Returns
    -------
    np.ndarray of shape (n_splits,)
        Validation score of every split.

    Raises
    ------
    ValueError
        If ``scoring`` is None and the estimator has no ``score`` method, or if
        ``len(X) != len(y)``.

    Notes
    -----
    Tested against ``sklearn.model_selection.cross_val_score(est, X, y, cv=KFold(k))``
    with deterministic estimators (scikit-learn ``LinearRegression``,
    ``NearestCentroid``), with ``scoring=None`` and with the same callable scorer:
    identical scores.

    Examples
    --------
    >>> from mylearn.cluster import NearestCentroid
    >>> X = np.array([[0.0], [10.0], [1.0], [11.0], [2.0], [12.0]])
    >>> y = np.array([0, 1, 0, 1, 0, 1])
    >>> cross_val_score(NearestCentroid(), X, y, cv=3)
    array([1., 1., 1.])
    >>> accuracy = lambda est, X, y: float(np.mean(est.predict(X) == y))
    >>> cross_val_score(NearestCentroid(), X, y, cv=3, scoring=accuracy)
    array([1., 1., 1.])
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} values")
    if scoring is None and not callable(getattr(estimator, "score", None)):
        raise ValueError(f"{type(estimator).__name__} has no score method: pass a scoring function")
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        splits = kfold_indices(len(X), int(cv))
    else:
        splits = list(cv)
    scores = []
    for train_idx, val_idx in splits:
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])           # fit is not chained: a fit without return self still works
        if scoring is None:
            scores.append(float(model.score(X[val_idx], y[val_idx])))
        else:
            scores.append(float(scoring(model, X[val_idx], y[val_idx])))
    return np.asarray(scores, dtype=float)
