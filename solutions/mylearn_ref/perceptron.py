"""The artificial neuron and the perceptron — mylearn, chapter 10 (Neurons).

An artificial neuron computes a weighted sum z = x . w + b and applies an activation
function. This module holds the perceptron threshold, the bias trick, the forward pass
of one neuron on a whole batch and Rosenblatt's perceptron with its learning rule.
It combines with ``multiclass.py`` (chapter 7) and ``model_selection.py`` (chapter 8).

Reference implementation: read it only after trying (``mon_travail/mylearn/perceptron.py``).
"""

from __future__ import annotations

from typing import Callable, Self

import numpy as np
from numpy.typing import ArrayLike


def sign_step(z: ArrayLike) -> np.ndarray:
    """Apply the perceptron threshold: +1 where z > 0, -1 elsewhere.

    Convention of the book: z = 0 gives -1.

    Parameters
    ----------
    z : array-like
        Weighted sums (any shape).

    Returns
    -------
    np.ndarray
        Float array of +1.0 and -1.0 with the shape of ``z``.

    Notes
    -----
    Tested against ``np.where(z > 0, 1.0, -1.0)``.

    Examples
    --------
    >>> sign_step(np.array([-2.0, 0.0, 3.0]))
    array([-1., -1.,  1.])
    """
    return np.where(np.asarray(z, dtype=float) > 0, 1.0, -1.0)


def add_bias_column(X: ArrayLike) -> np.ndarray:
    """Prepend a column of ones to X (the bias trick).

    With ``X1 = add_bias_column(X)`` and ``w1 = [b, w_1, ..., w_d]``,
    ``X1 @ w1 = X @ w + b``: the bias becomes the weight w0 of a constant input 1.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features + 1)
        Float array whose first column is all ones, followed by the columns of X.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D.

    Notes
    -----
    Tested against ``np.hstack([np.ones((n, 1)), X])``.

    Examples
    --------
    >>> add_bias_column(np.array([[2.0, 3.0], [4.0, 5.0]]))
    array([[1., 2., 3.],
           [1., 4., 5.]])
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got {X.ndim} dimension(s): "
                         "use X.reshape(1, -1) for a single sample")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def neuron_forward(
    X: ArrayLike,
    w: ArrayLike,
    b: float = 0.0,
    activation: Callable[[np.ndarray], np.ndarray] = sign_step,
) -> np.ndarray:
    """Compute the output of one artificial neuron for a batch of samples.

    Output = ``activation(X @ w + b)``: one weighted sum per sample (row of X), all at
    once, then the activation applied elementwise.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or (n_features,)
        Batch of samples, or a single sample.
    w : array-like of shape (n_features,)
        Weights.
    b : float, default=0.0
        Bias.
    activation : callable, default=sign_step
        Vectorised function applied to the weighted sums; the identity
        ``lambda z: z`` returns the weighted sums themselves.

    Returns
    -------
    np.ndarray
        Shape (n_samples,) for a batch, a 0-d array for a single sample.

    Raises
    ------
    ValueError
        If the number of features of ``X`` differs from ``len(w)``.

    Notes
    -----
    Tested against ``torch.nn.functional.linear(X, w[None, :], torch.tensor([b]))``
    followed by the activation, and against NumPy for ``sign_step``.

    Examples
    --------
    >>> X = np.array([[1.0, 2.0], [-1.0, 0.5]])
    >>> w = np.array([1.0, 1.0])
    >>> neuron_forward(X, w, b=-1.0)
    array([ 1., -1.])
    >>> neuron_forward(X, w, b=-1.0, activation=lambda z: z)
    array([ 2. , -1.5])

    A single sample gives a 0-d array:

    >>> neuron_forward(np.array([1.0, 2.0]), w, b=-1.0, activation=lambda z: z)
    array(2.)

    The bias trick: the bias becomes the weight of a constant input 1.

    >>> neuron_forward(add_bias_column(X), np.array([-1.0, 1.0, 1.0]))
    array([ 1., -1.])
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"w must be 1-D (one weight per feature), got shape {w.shape}")
    if X.ndim not in (1, 2):
        raise ValueError(f"X must be (n_samples, n_features) or (n_features,), got shape {X.shape}")
    if X.shape[-1] != len(w):
        raise ValueError(f"X has {X.shape[-1]} feature(s) but w has {len(w)} weight(s)")
    return np.asarray(activation(X @ w + b), dtype=float)


class Perceptron:
    """Rosenblatt's perceptron for two classes, with the classic learning rule.

    Labels are coded -1 (``classes_[0]``) and +1 (``classes_[1]``). The weights and the
    bias start at 0. During an epoch the samples are visited in order (a new random
    order at each epoch if ``shuffle``); a sample is misclassified when
    ``y * (x . w + b) <= 0``, and then ``w += eta0 * y * x`` and ``b += eta0 * y``.
    Training stops after an epoch without mistake (the data are separated) or after
    ``max_iter`` epochs (it never stops early on data that are not linearly separable,
    such as XOR).

    Parameters
    ----------
    eta0 : float, default=1.0
        Learning rate (> 0).
    max_iter : int, default=100
        Maximum number of epochs.
    fit_intercept : bool, default=True
        Learn the bias b (else b stays 0).
    shuffle : bool, default=False
        Visit the samples in a new random order at each epoch.
    random_state : int or None, default=None
        Seed of the generator ``np.random.default_rng(random_state)`` created in
        ``fit`` and used when ``shuffle=True``.

    Attributes
    ----------
    classes_ : np.ndarray of shape (2,)
        Sorted labels; ``classes_[0]`` is coded -1 and ``classes_[1]`` is coded +1.
    coef_ : np.ndarray of shape (n_features,)
        Weights w (scikit-learn's shape is (1, n_features)).
    intercept_ : float
        Bias b.
    n_iter_ : int
        Number of epochs done (``len(errors_)``).
    errors_ : list of int
        Number of mistakes (updates) during each epoch; the last one is 0 if the
        perceptron converged.

    Notes
    -----
    Tested against ``sklearn.linear_model.Perceptron(eta0=eta0, max_iter=max_iter,
    shuffle=False, tol=None, fit_intercept=...)``: identical ``coef_[0]`` and
    ``intercept_[0]``. Properties: converges on AND, OR and NAND, never reaches an
    epoch without mistake on XOR.

    Examples
    --------
    >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    >>> y = np.array([0, 0, 0, 1])  # logical AND
    >>> clf = Perceptron().fit(X, y)
    >>> clf.coef_
    array([3., 2.])
    >>> clf.intercept_
    -4.0
    >>> clf.errors_
    [2, 3, 3, 2, 2, 3, 2, 1, 0]
    >>> clf.predict(X)
    array([0, 0, 0, 1])
    """

    def __init__(
        self,
        eta0: float = 1.0,
        max_iter: int = 100,
        fit_intercept: bool = True,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.eta0 = eta0
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Train the perceptron with the classic learning rule.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Labels, exactly 2 distinct values (any values: 0/1, -1/+1, strings...).

        Returns
        -------
        Self
            The fitted perceptron (``self``), with ``classes_``, ``coef_``,
            ``intercept_``, ``n_iter_`` and ``errors_``.

        Raises
        ------
        ValueError
            If ``y`` does not contain exactly 2 classes, ``eta0 <= 0``,
            ``max_iter < 1`` or ``len(X) != len(y)``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"y must be 1-D with one label per sample: {len(y)} labels for {len(X)} samples")
        if self.eta0 <= 0:
            raise ValueError(f"eta0 must be > 0, got {self.eta0}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"the perceptron separates exactly 2 classes, got {len(classes)}: {classes.tolist()}"
                             " (for more classes, use mylearn.multiclass.OneVsRestClassifier)")
        target = np.where(y == classes[1], 1.0, -1.0)           # classes_[0] -> -1, classes_[1] -> +1
        n_samples, n_features = X.shape
        w, b = np.zeros(n_features), 0.0
        rng = np.random.default_rng(self.random_state)          # created here: every fit starts afresh
        errors = []
        for _ in range(self.max_iter):
            order = rng.permutation(n_samples) if self.shuffle else range(n_samples)
            mistakes = 0
            for i in order:
                if target[i] * (X[i] @ w + b) <= 0:             # on the boundary (z = 0) counts as a mistake
                    w = w + self.eta0 * target[i] * X[i]
                    if self.fit_intercept:
                        b += self.eta0 * target[i]
                    mistakes += 1
            errors.append(mistakes)
            if mistakes == 0:                                   # an epoch without mistake: separated
                break
        self.classes_ = classes
        self.coef_ = w
        self.intercept_ = float(b)
        self.errors_ = errors
        self.n_iter_ = len(errors)
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return the weighted sums ``X @ coef_ + intercept_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Scores; positive means ``classes_[1]``.
        """
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

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
        """
        return np.where(self.decision_function(X) > 0, self.classes_[1], self.classes_[0])

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
