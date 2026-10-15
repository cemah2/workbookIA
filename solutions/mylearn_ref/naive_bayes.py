"""Naive Bayes classifiers — mylearn, chapter 13 (Classifiers).

Bayes' rule (chapter 4) with the "naive" assumption that the features are
conditionally independent given the class: ``log P(c | x) = log P(c) +
sum_j log P(x_j | c) - log P(x)``. Everything is computed with log-probabilities (a
product of hundreds of small probabilities underflows to 0) and normalised with the
stable ``logsumexp``. Gaussian features (measurements) and counts (letters, words).

Reference implementation: read it only after trying (``mon_travail/mylearn/naive_bayes.py``).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike


def _check_2d(X: ArrayLike, name: str = "X") -> np.ndarray:
    """X as a 2-D float64 array (a copy), without NaN."""
    try:
        X = np.array(X, dtype=float)               # np.array copies: the caller's data are never modified
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold numbers only ({exc})") from None
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got an array with {X.ndim} "
                         "dimension(s): use X.reshape(-1, 1) for a single feature")
    if np.isnan(X).any():
        raise ValueError(f"{name} contains NaN values: remove the missing values first")
    return X


def _check_labels(y: ArrayLike, n_samples: int) -> np.ndarray:
    y = np.array(y)                                # a copy
    if y.ndim != 1 or len(y) != n_samples:
        raise ValueError(f"y must be 1-D with one label per row of X, got shape {y.shape} for {n_samples} rows")
    return y


def _check_fitted(estimator) -> None:
    if not hasattr(estimator, "classes_"):
        raise RuntimeError(f"this {type(estimator).__name__} is not fitted yet: call fit(X, y) first")


def _check_n_features(estimator, X: np.ndarray) -> None:
    if X.shape[1] != estimator._n_features:
        raise ValueError(f"X has {X.shape[1]} feature(s), but {type(estimator).__name__} was fitted with "
                         f"{estimator._n_features}")


def _accuracy(estimator, X: ArrayLike, y: ArrayLike) -> float:
    y = np.asarray(y)
    pred = estimator.predict(X)
    if y.shape != pred.shape:
        raise ValueError(f"y must have one label per row of X, got shape {y.shape} for {len(pred)} rows")
    return float(np.mean(pred == y))


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
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    var = np.asarray(var, dtype=float)
    if not np.all(var > 0):                       # also refuses NaN variances
        raise ValueError(f"every variance must be > 0, got minimum {np.min(var)}")
    return -0.5 * np.log(2.0 * np.pi * var) - (x - mean) ** 2 / (2.0 * var)


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
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)          # a slice of -inf only: avoid -inf - (-inf) = NaN
    with np.errstate(divide="ignore"):            # log(0) = -inf for such a slice
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


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
        X = _check_2d(X)
        y = _check_labels(y, len(X))
        if self.var_smoothing < 0:
            raise ValueError(f"var_smoothing must be >= 0, got {self.var_smoothing!r}")
        classes, codes = np.unique(y, return_inverse=True)
        n_classes, n_features = len(classes), X.shape[1]
        theta = np.empty((n_classes, n_features))
        var = np.empty((n_classes, n_features))
        count = np.empty(n_classes)
        for c in range(n_classes):
            Xc = X[codes == c]
            theta[c] = Xc.mean(axis=0)
            var[c] = Xc.var(axis=0)                   # population variance (ddof=0)
            count[c] = len(Xc)
        epsilon = float(self.var_smoothing * np.var(X, axis=0).max())
        if self.priors is None:
            prior = count / len(X)
        else:
            prior = np.array(self.priors, dtype=float)
            if prior.shape != (n_classes,):
                raise ValueError(f"priors must give one value per class ({n_classes}), got shape {prior.shape}")
            if np.isnan(prior).any() or (prior < 0).any():
                raise ValueError(f"priors must be non-negative, got {prior.tolist()}")
            if abs(prior.sum() - 1.0) > 1e-6:
                raise ValueError(f"priors must sum to 1, got {prior.sum()}")
        self.classes_ = classes
        self.class_count_ = count
        self.class_prior_ = prior
        self.theta_ = theta
        self.var_ = var + epsilon
        self.epsilon_ = epsilon
        self._n_features = n_features
        return self

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
        _check_fitted(self)
        X = _check_2d(X)
        _check_n_features(self, X)
        # shapes (n, 1, d) against (1, K, d): one log-density per sample, class and feature
        log_lik = log_gaussian_pdf(X[:, None, :], self.theta_[None, :, :], self.var_[None, :, :]).sum(axis=2)
        with np.errstate(divide="ignore"):            # a prior of 0 gives log 0 = -inf
            return log_lik + np.log(self.class_prior_)

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
        joint = self.predict_joint_log_proba(X)
        return joint - logsumexp(joint, axis=1)[:, None]

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
        return np.exp(self.predict_log_proba(X))

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
        return self.classes_[np.argmax(self.predict_joint_log_proba(X), axis=1)]

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
        return _accuracy(self, X, y)


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
        X = _check_2d(X)
        y = _check_labels(y, len(X))
        if (X < 0).any():
            raise ValueError("X must hold non-negative counts, got a negative value")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha!r}")
        classes, codes = np.unique(y, return_inverse=True)
        Y = (codes[:, None] == np.arange(len(classes))).astype(float)   # one-hot, shape (n, K)
        feature_count = Y.T @ X                       # N_cj: total count of feature j in class c
        class_count = Y.sum(axis=0)
        smoothed = feature_count + self.alpha
        with np.errstate(divide="ignore"):            # alpha = 0 and a count of 0: log 0 = -inf
            self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        if self.fit_prior:
            self.class_log_prior_ = np.log(class_count / len(X))
        else:
            self.class_log_prior_ = np.full(len(classes), -np.log(len(classes)))
        self.classes_ = classes
        self.class_count_ = class_count
        self.feature_count_ = feature_count
        self._n_features = X.shape[1]
        return self

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
        _check_fitted(self)
        X = _check_2d(X)
        _check_n_features(self, X)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

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
        joint = self.predict_joint_log_proba(X)
        return joint - logsumexp(joint, axis=1)[:, None]

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
        return np.exp(self.predict_log_proba(X))

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
        return self.classes_[np.argmax(self.predict_joint_log_proba(X), axis=1)]

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
        return _accuracy(self, X, y)
