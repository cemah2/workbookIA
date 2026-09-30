"""Naive Bayes classifiers — mylearn, chapter 13 (Classifiers).

Bayes' rule (chapter 4) with the "naive" assumption that the features are
conditionally independent given the class: ``log P(c | x) = log P(c) +
sum_j log P(x_j | c) - log P(x)``. Everything is computed with log-probabilities (a
product of hundreds of small probabilities underflows to 0) and normalised with the
stable ``logsumexp``. Gaussian features (measurements) and counts (letters, words).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike


def log_gaussian_pdf(x: ArrayLike, mean: ArrayLike, var: ArrayLike) -> np.ndarray:
    """Compute the log-density of a normal distribution, elementwise.

    ``log N(x; mean, var) = -0.5 log(2 pi var) - (x - mean)^2 / (2 var)``, with
    NumPy broadcasting between ``x``, ``mean`` and ``var``.

    Parameters
    ----------
    x : array-like
        Values.
    mean : array-like
        Means (broadcastable with ``x``).
    var : array-like
        Variances, all > 0 (broadcastable with ``x``).

    Returns
    -------
    np.ndarray
        Log-densities, with the broadcast shape of the three inputs.

    Raises
    ------
    ValueError
        If any variance is <= 0.

    Notes
    -----
    Tested against ``scipy.stats.norm.logpdf(x, loc=mean, scale=np.sqrt(var))``.

    Examples
    --------
    >>> log_gaussian_pdf(np.array([0.0, 1.0]), 0.0, 1.0).round(4)
    array([-0.9189, -1.4189])
    """
    # TODO: check the variances, then the formula (vectorised).
    raise NotImplementedError("log_gaussian_pdf() is not implemented yet")


def logsumexp(a: ArrayLike, axis: int = -1) -> np.ndarray:
    """Compute ``log(sum(exp(a)))`` along an axis without overflow or underflow.

    Subtract the maximum m first: ``log sum exp(a) = m + log sum exp(a - m)``. The
    largest term becomes exp(0) = 1, so nothing overflows, and values around -1000 do
    not all underflow to 0.

    Parameters
    ----------
    a : array-like
        Log-values.
    axis : int, default=-1
        Axis of the reduction.

    Returns
    -------
    np.ndarray
        Array with ``axis`` removed (a 0-d value for a 1-D input).

    Notes
    -----
    Tested against ``scipy.special.logsumexp``, including values around -1000 and
    +1000.

    Examples
    --------
    >>> round(float(logsumexp(np.array([-1000.0, -1000.0]))), 4)
    -999.3069
    >>> logsumexp(np.array([[0.0, 0.0], [1.0, 2.0]]), axis=1).round(4)
    array([0.6931, 2.3133])
    """
    # TODO: maximum along the axis (keepdims=True), then the formula above.
    raise NotImplementedError("logsumexp() is not implemented yet")


class GaussianNB:
    """Gaussian naive Bayes classifier.

    Each feature j of class c follows a normal distribution N(theta_[c, j],
    var_[c, j]) (mean and population variance of the training samples of class c).
    ``predict_joint_log_proba(X)[i, c] = log P(c) + sum_j log N(x_ij; theta_cj, var_cj)``;
    normalising these joint log-probabilities with ``logsumexp`` gives
    ``log P(c | x)``. As scikit-learn, ``epsilon_ = var_smoothing * max_j var(X[:, j])``
    is added to every variance (a feature constant within a class does not give a
    variance of 0).

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Fraction of the largest feature variance added to all the variances.
    priors : array-like of shape (n_classes,) or None, default=None
        Fixed class priors, in the order of ``classes_``; None uses the class
        frequencies of the training set.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    class_count_ : np.ndarray of shape (n_classes,)
        Number of training samples of each class.
    class_prior_ : np.ndarray of shape (n_classes,)
        Prior probability of each class.
    theta_ : np.ndarray of shape (n_classes, n_features)
        Per-class feature means.
    var_ : np.ndarray of shape (n_classes, n_features)
        Per-class feature variances, ``epsilon_`` included.
    epsilon_ : float
        Value added to the variances.

    Notes
    -----
    Tested against ``sklearn.naive_bayes.GaussianNB``: ``theta_``, ``var_``,
    ``class_prior_``, ``epsilon_`` and ``predict_proba`` allclose.

    Examples
    --------
    >>> X = np.array([[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> nb = GaussianNB().fit(X, y)
    >>> nb.theta_.round(4)
    array([[-2.    , -1.3333],
           [ 2.    ,  1.3333]])
    >>> nb.predict(np.array([[-0.8, -1.0]]))
    array([1])
    """

    def __init__(self, var_smoothing: float = 1e-9, priors: ArrayLike | None = None) -> None:
        self.var_smoothing = var_smoothing
        self.priors = priors

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Estimate the class priors and the per-class means and variances.

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
            If ``len(X) != len(y)``, X is not 2-D, or ``priors`` does not have one
            non-negative value per class summing to 1.
        """
        # TODO: one boolean mask per class: counts, means, variances; then epsilon_
        # and the priors.
        raise NotImplementedError("fit() is not implemented yet")

    def predict_joint_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return ``log P(c) + log P(x | c)`` for every sample and class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Joint log-probabilities (not normalised).

        Raises
        ------
        RuntimeError
            If the classifier is not fitted.
        """
        # TODO: log_gaussian_pdf with broadcasting over (samples, classes, features),
        # summed over the features, plus the log-priors.
        raise NotImplementedError("predict_joint_log_proba() is not implemented yet")

    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the posterior log-probabilities ``log P(c | x)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Normalised log-probabilities (``logsumexp`` of every row is 0).
        """
        # TODO: subtract the logsumexp of every row.
        raise NotImplementedError("predict_log_proba() is not implemented yet")

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the posterior probabilities ``P(c | x)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Rows summing to 1; column k is ``classes_[k]``.
        """
        # TODO: exponentiate predict_log_proba.
        raise NotImplementedError("predict_proba() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the largest posterior probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_``.
        """
        # TODO: argmax of the joint log-probabilities.
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


class MultinomialNB:
    """Multinomial naive Bayes classifier for count features (letters, words).

    ``feature_count_[c, j]`` (N_cj) is the total count of feature j in the training
    samples of class c, and N_c the sum of row c. With additive (Laplace) smoothing,
    ``feature_log_prob_[c, j] = log((N_cj + alpha) / (N_c + alpha * n_features))``, so
    that a word never seen in a class does not get probability 0. Then
    ``predict_joint_log_proba(X) = X @ feature_log_prob_.T + class_log_prior_``.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive smoothing (>= 0; 1 is Laplace smoothing).
    fit_prior : bool, default=True
        Learn the class priors from the class frequencies; if False, uniform priors.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted distinct labels seen in ``fit``.
    class_count_ : np.ndarray of shape (n_classes,)
        Number of training samples of each class.
    feature_count_ : np.ndarray of shape (n_classes, n_features)
        Total count of every feature in every class.
    class_log_prior_ : np.ndarray of shape (n_classes,)
        Log prior probability of each class.
    feature_log_prob_ : np.ndarray of shape (n_classes, n_features)
        Smoothed log-probability of every feature given the class.

    Notes
    -----
    Tested against ``sklearn.naive_bayes.MultinomialNB(alpha, force_alpha=True)``:
    ``class_log_prior_``, ``feature_log_prob_`` and ``predict_proba`` allclose.

    Examples
    --------
    >>> X = np.array([[2, 1, 0], [3, 0, 1], [0, 2, 3], [1, 0, 4]])  # word counts
    >>> y = np.array(["en", "en", "fr", "fr"])
    >>> nb = MultinomialNB(alpha=1.0).fit(X, y)
    >>> np.exp(nb.feature_log_prob_).round(4)
    array([[0.6   , 0.2   , 0.2   ],
           [0.1538, 0.2308, 0.6154]])
    >>> nb.predict(np.array([[4, 0, 0], [0, 1, 5]])).tolist()
    ['en', 'fr']
    """

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True) -> None:
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Count the features of every class and compute the smoothed log-probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative counts.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        Self
            The fitted classifier (``self``).

        Raises
        ------
        ValueError
            If X has negative values, ``alpha < 0``, X is not 2-D or
            ``len(X) != len(y)``.
        """
        # TODO: per-class sums of the rows, then the two log-probability attributes.
        raise NotImplementedError("fit() is not implemented yet")

    def predict_joint_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return ``log P(c) + sum_j x_j log P(j | c)`` for every sample and class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative counts.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Joint log-probabilities (not normalised).

        Raises
        ------
        RuntimeError
            If the classifier is not fitted.
        """
        # TODO: one matrix product plus the log-priors.
        raise NotImplementedError("predict_joint_log_proba() is not implemented yet")

    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the posterior log-probabilities ``log P(c | x)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative counts.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Normalised log-probabilities.
        """
        # TODO: subtract the logsumexp of every row.
        raise NotImplementedError("predict_log_proba() is not implemented yet")

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the posterior probabilities ``P(c | x)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative counts.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Rows summing to 1; column k is ``classes_[k]``.
        """
        # TODO: exponentiate predict_log_proba.
        raise NotImplementedError("predict_proba() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the largest posterior probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative counts.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_``.
        """
        # TODO: argmax of the joint log-probabilities.
        raise NotImplementedError("predict() is not implemented yet")

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative counts.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction of correct predictions, in [0, 1].
        """
        # TODO: compare the predictions with y and average.
        raise NotImplementedError("score() is not implemented yet")
