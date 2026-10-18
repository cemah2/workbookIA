"""Distances and k nearest neighbours — mylearn, chapter 13 (Classifiers).

The k-nearest-neighbours classifier is "lazy": ``fit`` only stores the training set,
all the work happens at prediction time (distances to every training sample, the k
closest ones vote). This module computes the distance matrices with broadcasting and
implements the brute-force classifier, with uniform or distance-based weights.

Reference implementation: read it only after trying (``mon_travail/mylearn/neighbors.py``).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

_METRICS = ("euclidean", "manhattan")
_WEIGHTS = ("uniform", "distance")


def _check_2d(X: ArrayLike, name: str) -> np.ndarray:
    """X as a 2-D float64 array (a copy), without NaN or infinite values."""
    try:
        X = np.array(X, dtype=float)               # np.array copies: the caller's data are never modified
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold numbers only ({exc})") from None
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got an array with {X.ndim} "
                         "dimension(s): use X.reshape(-1, 1) for a single feature")
    if np.isnan(X).any():
        raise ValueError(f"{name} contains NaN values: remove the missing values first")
    if np.isinf(X).any():
        raise ValueError(f"{name} contains infinite values")
    return X


def _check_no_nan_labels(y: np.ndarray, name: str = "y") -> np.ndarray:
    """y itself, after refusing NaN labels (float y only), like tree and ensemble."""
    if y.dtype.kind in "fc" and np.isnan(y).any():
        raise ValueError(f"{name} contains NaN values: remove the missing values first")
    return y


def _check_metric(metric: str) -> None:
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")


def _distances(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    """Distance matrix of two checked 2-D arrays (one broadcast difference, no Python loop)."""
    diff = A[:, None, :] - B[None, :, :]           # shape (n_a, n_b, n_features)
    if metric == "euclidean":
        return np.sqrt(np.sum(diff**2, axis=-1))
    return np.sum(np.abs(diff), axis=-1)


def pairwise_distances(A: ArrayLike, B: ArrayLike, metric: str = "euclidean") -> np.ndarray:
    """Compute the distance matrix between the rows of A and the rows of B.

    Broadcasting ``A[:, None, :] - B[None, :, :]`` gives all the differences at once
    (no Python loop over the samples). "euclidean": square root of the sum of the
    squared differences; "manhattan": sum of the absolute differences.

    Parameters
    ----------
    A : array-like of shape (n_a, n_features)
        First set of points.
    B : array-like of shape (n_b, n_features)
        Second set of points.
    metric : str, default="euclidean"
        "euclidean" or "manhattan".

    Returns
    -------
    np.ndarray of shape (n_a, n_b)
        ``D[i, j]`` = distance between ``A[i]`` and ``B[j]``, non-negative.

    Raises
    ------
    ValueError
        If A or B is not 2-D, their numbers of features differ, or ``metric`` is
        unknown.

    Notes
    -----
    Tested against ``sklearn.metrics.pairwise_distances`` and
    ``scipy.spatial.distance.cdist`` ("euclidean" and "cityblock").

    Examples
    --------
    >>> A = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> B = np.array([[3.0, 4.0]])
    >>> pairwise_distances(A, B).round(4)
    array([[5.    ],
           [3.6056]])
    >>> pairwise_distances(A, B, metric="manhattan")
    array([[7.],
           [5.]])
    """
    _check_metric(metric)
    A = _check_2d(A, "A")
    B = _check_2d(B, "B")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have the same number of features, got {A.shape[1]} and {B.shape[1]}")
    return _distances(A, B, metric)


class KNeighborsClassifier:
    """k-nearest-neighbours classifier (brute force).

    To classify x: compute its distance to every training sample, keep the k closest
    (equal distances: the lower training index first, i.e. a stable sort) and let them
    vote. ``predict_proba`` gives the (weighted) share of the votes of each class;
    ``predict`` takes the class with the largest share (ties: the first class of
    ``classes_``, i.e. the smallest label, as scikit-learn).

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours k (>= 1).
    weights : str, default="uniform"
        "uniform": every neighbour has one vote. "distance": a neighbour at distance d
        has weight 1/d; if some neighbours are at distance 0, they share all the weight
        (the others get 0), as scikit-learn.
    metric : str, default="euclidean"
        "euclidean" or "manhattan" (see ``pairwise_distances``).

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    n_features_in_ : int
        Number of features seen in ``fit``.
    n_samples_fit_ : int
        Number of stored training samples.

    Notes
    -----
    Tested against ``sklearn.neighbors.KNeighborsClassifier(algorithm="brute")`` with
    the same ``metric`` and ``weights``, on data without distance ties: identical
    ``kneighbors``, ``predict`` and ``predict_proba``.

    Examples
    --------
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    >>> knn.predict(np.array([[1.1]]))
    array([0])
    >>> knn.predict_proba(np.array([[0.9]]))
    array([[0.66666667, 0.33333333]])
    >>> dist, ind = knn.kneighbors(np.array([[1.1]]))
    >>> ind
    array([[1, 2, 0]])
    >>> dist.round(2)
    array([[0.1, 0.9, 1.1]])
    """

    def __init__(
        self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "euclidean"
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Store the training set (lazy learning: no computation here).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        Self
            The fitted classifier (``self``).

        Raises
        ------
        ValueError
            If ``n_neighbors < 1`` or ``n_neighbors > n_samples``, ``weights`` or
            ``metric`` is unknown, X is not 2-D, or ``len(X) != len(y)``.
        """
        X = _check_2d(X, "X")
        y = np.array(y)                                # a copy
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"y must be 1-D with one label per row of X, got shape {y.shape} for {len(X)} rows")
        _check_no_nan_labels(y)
        if self.weights not in _WEIGHTS:
            raise ValueError(f"weights must be one of {_WEIGHTS}, got {self.weights!r}")
        _check_metric(self.metric)
        k = self.n_neighbors
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise ValueError(f"n_neighbors must be an integer, got {k!r}")
        if not 1 <= k <= len(X):
            raise ValueError(f"n_neighbors must be in 1..n_samples = 1..{len(X)}, got {k}")
        self.X_fit_ = X
        self.y_fit_ = y
        self.classes_, self._y_codes = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise RuntimeError("this KNeighborsClassifier is not fitted yet: call fit(X, y) first")

    def kneighbors(
        self,
        X: ArrayLike,
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Find the k nearest training samples of every row of X.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.
        n_neighbors : int or None, default=None
            Number of neighbours; None uses ``self.n_neighbors``.
        return_distance : bool, default=True
            Also return the distances.

        Returns
        -------
        (dist, ind) or ind
            ``ind``: int array of shape (n_queries, k), indices of the neighbours in the
            training set, from the closest to the farthest (stable on ties).
            ``dist``: float array of shape (n_queries, k), the matching distances.

        Raises
        ------
        RuntimeError
            If the classifier is not fitted.
        ValueError
            If k is not in 1..n_samples_fit_ or X has a wrong number of features.
        """
        self._check_fitted()
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= self.n_samples_fit_:
            raise ValueError(f"n_neighbors must be an integer in 1..n_samples_fit_ = 1..{self.n_samples_fit_}, "
                             f"got {k!r}")
        X = _check_2d(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} feature(s), but KNeighborsClassifier was fitted with "
                             f"{self.n_features_in_}")
        D = _distances(X, self.X_fit_, self.metric)
        ind = np.argsort(D, axis=1, kind="stable")[:, :k]      # stable: equal distances keep the lower index first
        if not return_distance:
            return ind
        return np.take_along_axis(D, ind, axis=1), ind

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the (weighted) share of the votes of each class.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        np.ndarray of shape (n_queries, n_classes)
            Rows summing to 1; column k is the share of ``classes_[k]``.

        Raises
        ------
        RuntimeError
            If the classifier is not fitted.
        """
        dist, ind = self.kneighbors(X)
        if self.weights == "uniform":
            w = np.ones_like(dist)
        else:
            zero = dist == 0
            with np.errstate(divide="ignore"):
                w = 1.0 / dist
            # a row with neighbours at distance 0: they share all the weight, the others get 0
            w = np.where(zero.any(axis=1, keepdims=True), zero.astype(float), w)
        codes = self._y_codes[ind]                             # class index of every neighbour
        proba = np.zeros((len(ind), len(self.classes_)))
        rows = np.repeat(np.arange(len(ind)), ind.shape[1])
        np.add.at(proba, (rows, codes.ravel()), w.ravel())
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the largest share of the votes.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        np.ndarray of shape (n_queries,)
            Predicted labels, taken from ``classes_`` (ties: the first class).

        Raises
        ------
        RuntimeError
            If the classifier is not fitted.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]     # argmax: the first maximum on ties

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
        y = _check_no_nan_labels(np.asarray(y))
        pred = self.predict(X)
        if y.shape != pred.shape:
            raise ValueError(f"y must have one label per row of X, got shape {y.shape} for {len(pred)} rows")
        return float(np.mean(pred == y))
