"""One-versus-rest and one-versus-one — mylearn, chapter 7 (Classification).

Many classifiers (perceptron, SVM, logistic regression) only separate two classes.
This module turns any such binary classifier into a multi-class one: one-versus-rest
(K models, the highest score wins) and one-versus-one (K(K-1)/2 duels, a vote decides).
You will reuse these wrappers with the perceptron (chapter 10) and the SVM (chapter 13).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import copy
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike


class OneVsRestClassifier:
    """One-versus-rest (OvR) strategy: one binary classifier per class.

    For each class k, a copy of ``estimator`` learns "class k" (coded 1) against all the
    other classes (coded 0). A new sample gets the class whose model gives the highest
    score: ``argmax_k s_k(x)``.

    Parameters
    ----------
    estimator : object
        Unfitted binary classifier with ``fit(X, y)`` and ``decision_function(X)``
        (or ``predict_proba(X)`` if it has no ``decision_function``). It is never
        fitted itself: every model is a ``copy.deepcopy`` of it
        (``mylearn.model_selection.clone`` only arrives in chapter 8).

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    estimators_ : list
        ``estimators_[k]`` is the fitted binary model of class ``classes_[k]``.

    Notes
    -----
    Tested against ``sklearn.multiclass.OneVsRestClassifier`` wrapping the same base
    estimator (e.g. ``LinearSVC(random_state=0)``): identical ``decision_function`` and
    ``predict``.

    Examples
    --------
    >>> from mylearn.cluster import NearestCentroid
    >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> ovr = OneVsRestClassifier(NearestCentroid()).fit(X, y)
    >>> len(ovr.estimators_)
    3
    >>> X_new = np.array([[0.5, 0.5], [5.0, 0.5], [0.5, 5.0]])
    >>> ovr.decision_function(X_new)
    array([[  9.875, -15.125, -16.25 ],
           [-14.875,  27.625, -34.25 ],
           [-10.375, -35.375,  24.25 ]])
    >>> ovr.predict(X_new)
    array([0, 1, 2])
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit one binary model per class (class k coded 1, all the others 0).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels, at least 3 distinct classes.

        Returns
        -------
        Self
            The fitted classifier (``self``), with ``classes_`` and ``estimators_``.

        Raises
        ------
        ValueError
            If ``y`` has fewer than 3 classes (use the binary classifier directly) or if
            the estimator has neither ``decision_function`` nor ``predict_proba``.
        """
        # TODO: for each class, deep-copy the estimator and fit it on 0/1 labels.
        raise NotImplementedError("fit() is not implemented yet")

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return the score of every binary model for every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Column k is ``estimators_[k].decision_function(X)``, or
            ``estimators_[k].predict_proba(X)[:, 1]`` when the model has no
            ``decision_function``.
        """
        # TODO: one column per binary model.
        raise NotImplementedError("decision_function() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the highest score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            ``classes_[argmax of decision_function(X)]`` (ties: first class).
        """
        # TODO: argmax over the columns of decision_function.
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


class OneVsOneClassifier:
    """One-versus-one (OvO) strategy: one binary classifier per pair of classes.

    For every pair (i, j) with i < j, a copy of ``estimator`` is trained only on the
    samples of classes i and j (class j coded 1, class i coded 0): K(K-1)/2 models.
    To predict, every model votes for one of its two classes and the class with the
    most votes wins.

    Parameters
    ----------
    estimator : object
        Unfitted binary classifier with ``fit(X, y)`` and ``predict(X)``; every model is
        a ``copy.deepcopy`` of it.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    pairs_ : list of tuple of (int, int)
        Class indices ``(i, j)``, ``i < j``, in lexicographic order:
        (0, 1), (0, 2), ..., (1, 2), ...
    estimators_ : list
        ``estimators_[m]`` is the fitted model of the pair ``pairs_[m]``.

    Notes
    -----
    Tested against ``sklearn.multiclass.OneVsOneClassifier``: ``votes(X)`` equals
    ``np.round(decision_function(X))`` of scikit-learn, and ``predict`` is identical
    wherever the maximum vote is unique (scikit-learn breaks ties with confidences).

    Examples
    --------
    >>> from mylearn.cluster import NearestCentroid
    >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> ovo = OneVsOneClassifier(NearestCentroid()).fit(X, y)
    >>> ovo.pairs_
    [(0, 1), (0, 2), (1, 2)]
    >>> X_new = np.array([[0.5, 0.5], [5.0, 0.5], [0.5, 5.0]])
    >>> ovo.votes(X_new)
    array([[2, 1, 0],
           [1, 2, 0],
           [1, 0, 2]])
    >>> ovo.predict(X_new)
    array([0, 1, 2])
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit one binary model per pair of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels, at least 3 distinct classes.

        Returns
        -------
        Self
            The fitted classifier (``self``), with ``classes_``, ``pairs_`` and
            ``estimators_``.

        Raises
        ------
        ValueError
            If ``y`` has fewer than 3 classes.
        """
        # TODO: for each pair (i, j), keep the samples of the two classes, code class j
        # as 1 and class i as 0, and fit a deep copy of the estimator.
        raise NotImplementedError("fit() is not implemented yet")

    def votes(self, X: ArrayLike) -> np.ndarray:
        """Count the votes received by every class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Integer vote counts; every row sums to K(K-1)/2.
        """
        # TODO: each model adds one vote to class i or to class j of its pair.
        raise NotImplementedError("votes() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the most votes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            ``classes_[argmax of votes(X)]``; ties go to the smallest class index.
        """
        # TODO: argmax over the columns of votes.
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
