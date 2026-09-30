"""Splits, cloning and cross-validation — mylearn, chapter 8 (Training and Testing).

Honest evaluation tools: hold-out split (``train_test_split``), k-fold indices (plain
and stratified), ``clone`` (a fresh, unfitted copy of an estimator) and
``cross_val_score``. Every later chapter evaluates its models with them.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike


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
    # TODO: validate, compute n_test, choose the test row indices (shuffled,
    # stratified or last rows), then index every array with the same indices.
    raise NotImplementedError("train_test_split() is not implemented yet")


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
    # TODO: fold sizes, then slice the (possibly permuted) indices fold by fold.
    raise NotImplementedError("kfold_indices() is not implemented yet")


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
    # TODO: class-sorted order of the indices, then fold i = positions i, i + k, ...
    # (use warnings.warn for the small-class warning).
    raise NotImplementedError("stratified_kfold_indices() is not implemented yet")


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
    # TODO: collect the hyperparameters from vars(estimator), deep-copy them and call
    # the class of the estimator.
    raise NotImplementedError("clone() is not implemented yet")


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
    # TODO: build the list of splits, then clone / fit / score on each of them.
    raise NotImplementedError("cross_val_score() is not implemented yet")
