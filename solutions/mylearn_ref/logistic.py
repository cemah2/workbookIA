"""Binary logistic regression — mylearn, chapter 13 (Classifiers).

Logistic regression is the bridge between the perceptron (chapter 10) and neural
networks (chapters 16-18): a weighted sum z = x . w + b, squashed by the sigmoid into a
probability, and trained by gradient descent on the log-loss. The log-loss is the one
you wrote in chapter 6 (``info.log_loss``); the sigmoid is provided here (you will
write your own numerically stable version in chapter 17).

Reference implementation: read it only after trying (``mon_travail/mylearn/logistic.py``).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .info import log_loss  # chapter 6: mean log-loss of probabilities (in nats)


def _check_2d(X: ArrayLike, name: str = "X") -> np.ndarray:
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


def sigmoid(z: ArrayLike) -> np.ndarray:
    """Compute the logistic function ``1 / (1 + exp(-z))`` without overflow.

    Provided: already implemented, no exercise (you will derive and code your own
    stable version in chapter 17, exercise 17.13). For z >= 0 it uses
    ``1 / (1 + exp(-z))``, for z < 0 the equivalent ``exp(z) / (1 + exp(z))``: the
    exponential is only ever taken of a non-positive number, so it never overflows.

    Parameters
    ----------
    z : array-like
        Real scores (any shape).

    Returns
    -------
    np.ndarray
        Float array with the shape of ``z``, values in [0, 1] (in (0, 1) up to
        floating-point rounding).

    Notes
    -----
    Tested against ``scipy.special.expit``.

    Examples
    --------
    >>> sigmoid(np.array([-1000.0, 0.0, 2.0]))
    array([0.        , 0.5       , 0.88079708])
    """
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))  # in (0, 1]: no overflow
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


class LogisticRegression:
    """Binary logistic regression trained by full-batch gradient descent.

    Model: ``P(y = classes_[1] | x) = sigmoid(x . w + b)``. With the labels coded 0/1
    (``classes_[0]`` -> 0, ``classes_[1]`` -> 1), ``fit`` minimises the objective
    ``J(w, b) = log_loss(y, p) + (alpha / 2) ||w||^2`` (mean log-loss of chapter 6 plus
    an L2 penalty; the intercept b is not penalised). Its gradient is
    ``dJ/dw = Xᵀ (p - y) / n + alpha * w`` and ``dJ/db = mean(p - y)``.

    w and b start at 0, so ``loss_history_[0] = ln 2``. Each iteration takes one
    gradient step ``w -= learning_rate * dJ/dw`` (and the same for b), then appends the
    new objective to ``loss_history_``; training stops after ``max_iter`` steps or as
    soon as the objective changed by less than ``tol`` during a step.

    Parameters
    ----------
    alpha : float, default=0.0
        L2 penalty strength (>= 0; 0 means no penalty).
    learning_rate : float, default=0.1
        Gradient step size (> 0).
    max_iter : int, default=1000
        Maximum number of gradient steps (>= 1).
    tol : float, default=1e-6
        Stop when ``|J_new - J_old| < tol``.
    fit_intercept : bool, default=True
        Learn the bias b (else b stays 0).

    Attributes
    ----------
    classes_ : np.ndarray of shape (2,)
        Sorted labels; ``classes_[1]`` is the "positive" class.
    coef_ : np.ndarray of shape (n_features,)
        Weights w (scikit-learn's shape is (1, n_features)).
    intercept_ : float
        Bias b.
    n_iter_ : int
        Number of gradient steps done.
    loss_history_ : list of float
        Objective before training and after each step (``n_iter_ + 1`` values).

    Notes
    -----
    Tested against ``sklearn.linear_model.LogisticRegression(C=1 / (alpha * n_samples))``
    (``penalty=None`` when alpha = 0) on standardised, non-separable data: ``coef_``,
    ``intercept_`` and ``predict_proba`` allclose (atol 1e-2); ``loss_history_`` never
    increases for a small learning rate.

    Examples
    --------
    >>> X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    >>> y = np.array([0, 1, 0, 1, 1])
    >>> clf = LogisticRegression().fit(X, y)
    >>> round(clf.loss_history_[0], 4)
    0.6931
    >>> clf.predict(np.array([[-3.0], [3.0]]))
    array([0, 1])
    >>> clf.predict_proba(np.array([[-3.0], [3.0]])).shape
    (2, 2)
    """

    def __init__(
        self,
        alpha: float = 0.0,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit w and b by gradient descent on the penalised log-loss.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples (standardise them first: gradient descent is much faster).
        y : array-like of shape (n_samples,)
            Labels, exactly 2 distinct values (any values).

        Returns
        -------
        Self
            The fitted model (``self``), with ``classes_``, ``coef_``, ``intercept_``,
            ``n_iter_`` and ``loss_history_``.

        Raises
        ------
        ValueError
            If ``y`` does not contain exactly two classes, ``learning_rate <= 0``,
            ``alpha < 0``, ``max_iter < 1``, X is not 2-D or ``len(X) != len(y)``.
        """
        X = _check_2d(X)
        y = np.array(y)                               # a copy
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"y must be 1-D with one label per row of X, got shape {y.shape} for {len(X)} rows")
        _check_no_nan_labels(y)
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if not self.alpha >= 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha!r}")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError(f"max_iter must be an integer >= 1, got {self.max_iter!r}")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"y must contain exactly two classes, got {len(classes)}: {classes.tolist()}")
        t = (y == classes[1]).astype(int)             # classes_[0] -> 0, classes_[1] -> 1
        n, d = X.shape
        alpha, lr = float(self.alpha), float(self.learning_rate)

        def objective(w: np.ndarray, b: float) -> float:
            # mean log-loss of chapter 6 + L2 penalty on w only (the intercept is not penalised)
            return float(log_loss(t, sigmoid(X @ w + b)) + 0.5 * alpha * (w @ w))

        w, b = np.zeros(d), 0.0
        history = [objective(w, b)]                   # ln 2: every probability is 0.5
        n_iter = 0
        for _ in range(self.max_iter):
            residual = sigmoid(X @ w + b) - t         # p - y
            grad_w = X.T @ residual / n + alpha * w
            grad_b = residual.mean()
            w = w - lr * grad_w
            if self.fit_intercept:
                b = b - lr * grad_b
            history.append(objective(w, b))
            n_iter += 1
            if abs(history[-1] - history[-2]) < self.tol:
                break
        self.classes_ = classes
        self.coef_ = w
        self.intercept_ = float(b)
        self.n_iter_ = n_iter
        self.loss_history_ = history
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return the scores ``z = X @ coef_ + intercept_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Scores (log-odds of ``classes_[1]``).

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("this LogisticRegression is not fitted yet: call fit(X, y) first")
        X = _check_2d(X)
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"X has {X.shape[1]} feature(s), but LogisticRegression was fitted with {len(self.coef_)}")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the probabilities of both classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Column 0: ``1 - p``, column 1: ``p = sigmoid(z)``; rows sum to 1.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict ``classes_[1]`` where the score is > 0, ``classes_[0]`` elsewhere.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels, taken from ``classes_``.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        z = self.decision_function(X)
        return self.classes_[(z > 0).astype(int)]      # z = 0 (p = 0.5): classes_[0]

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
