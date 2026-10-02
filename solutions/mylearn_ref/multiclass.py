"""One-versus-rest and one-versus-one — mylearn, chapter 7 (Classification).

Many classifiers (perceptron, SVM, logistic regression) only separate two classes.
This module turns any such binary classifier into a multi-class one: one-versus-rest
(K models, the highest score wins) and one-versus-one (K(K-1)/2 duels, a vote decides).
You will reuse these wrappers with the perceptron (chapter 10) and the SVM (chapter 13).

Reference implementation: read it only after trying (``mon_travail/mylearn/multiclass.py``).
"""

from __future__ import annotations

import copy
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike


def _check_labels(X: ArrayLike, y: ArrayLike, minimum: int, name: str):
    """The labels as a 1-D array and their sorted distinct values, or a ValueError."""
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(f"y must be a 1-D array with one label per sample, got {len(y)} labels for {len(X)} samples")
    classes = np.unique(y)
    if len(classes) < minimum:
        raise ValueError(f"{name} needs at least {minimum} classes, got {len(classes)}: "
                         "use the binary classifier directly")
    return y, classes


def _binary_score(model: Any, X: ArrayLike) -> np.ndarray:
    """The score of the positive class: decision_function(X), else predict_proba(X)[:, 1]."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel()
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


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
        y, classes = _check_labels(X, y, 3, "OneVsRestClassifier")
        if not (hasattr(self.estimator, "decision_function") or hasattr(self.estimator, "predict_proba")):
            raise ValueError("the estimator needs decision_function or predict_proba to score every class")
        self.classes_ = classes
        self.estimators_ = []
        for c in classes:
            model = copy.deepcopy(self.estimator)                # the estimator itself is never fitted
            model.fit(X, (y == c).astype(int))
            self.estimators_.append(model)
        return self

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
        return np.column_stack([_binary_score(model, X) for model in self.estimators_])

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
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

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
        return float(np.mean(self.predict(X) == np.asarray(y)))


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
        y, classes = _check_labels(X, y, 3, "OneVsOneClassifier")
        X_arr = np.asarray(X)
        self.classes_ = classes
        n_classes = len(classes)
        self.pairs_ = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
        self.estimators_ = []
        for i, j in self.pairs_:
            keep = (y == classes[i]) | (y == classes[j])          # only the samples of the two classes
            model = copy.deepcopy(self.estimator)
            model.fit(X_arr[keep], (y[keep] == classes[j]).astype(int))   # class j coded 1, class i coded 0
            self.estimators_.append(model)
        return self

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
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        for (i, j), model in zip(self.pairs_, self.estimators_):
            for_j = np.asarray(model.predict(X)) == 1
            votes[:, j] += for_j
            votes[:, i] += ~for_j
        return votes

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
        return self.classes_[np.argmax(self.votes(X), axis=1)]

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
        return float(np.mean(self.predict(X) == np.asarray(y)))
