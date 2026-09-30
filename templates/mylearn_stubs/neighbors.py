"""Distances and k nearest neighbours — mylearn, chapter 13 (Classifiers).

The k-nearest-neighbours classifier is "lazy": ``fit`` only stores the training set,
all the work happens at prediction time (distances to every training sample, the k
closest ones vote). This module computes the distance matrices with broadcasting and
implements the brute-force classifier, with uniform or distance-based weights.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike


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
    # TODO: check the shapes and the metric, then one broadcast difference.
    raise NotImplementedError("pairwise_distances() is not implemented yet")


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
        # TODO: validate, then store copies of X and y in attributes whose names end
        # with "_" (clone() takes the other names for hyperparameters).
        raise NotImplementedError("fit() is not implemented yet")

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
        # TODO: distance matrix, then a stable argsort of every row (kind="stable").
        raise NotImplementedError("kneighbors() is not implemented yet")

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
        # TODO: labels of the neighbours, their weights, then a weighted count per class.
        raise NotImplementedError("predict_proba() is not implemented yet")

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
        # TODO: argmax of predict_proba.
        raise NotImplementedError("predict() is not implemented yet")

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
        # TODO: compare the predictions with y and average.
        raise NotImplementedError("score() is not implemented yet")
