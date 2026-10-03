"""Median house value of California districts with linear models (mini-project of part II): YOUR module.

Every step that learns something from the data happens inside ``HousingModel.fit``. A
cross-validation that clones and refits the model in each fold therefore refits every
step on the training part of the fold only: the validation districts never leak into
the preprocessing. The steps:

1. bounds of each feature, the percentiles ``clip`` of the training rows: the atypical
   districts (up to about 600 people per household on average) no longer drive the
   polynomial;
2. z-scores of the bounded features;
3. polynomial features of degree ``degree`` (``mylearn.linear.polynomial_features``);
4. optionally, ``n_zones`` geographic zones: k-means on the raw latitude and longitude
   (not bounded), standardized with their own training means and standard deviations
   (``mylearn.cluster.KMeans``, a single k-means++ run, like scikit-learn's
   ``n_init="auto"``), one column of 0/1 per zone;
5. z-scores of all the columns, so that the penalty treats them alike;
6. a linear model of ``mylearn.linear``: least squares, Ridge or Lasso.

The helpers measure the root mean squared error in cross-validation, with a clone of the
model per fold (``mylearn.model_selection.clone``): fold by fold, along a hyperparameter
(a validation curve) or along the number of training examples (a learning curve).
No machine learning library: NumPy and the learner's own ``mylearn``.

    model = HousingModel(degree=3, penalty="ridge", alpha=10.0, n_zones=12).fit(X_train, y_train)
    rmse(y_val, model.predict(X_val))

Write every function that raises NotImplementedError (the step of the notebook is given
in the TODO). Keep the signatures and the docstrings: the notebook and the tests rely on
them. You may add private helpers (names starting with _).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mylearn import cluster, linear, model_selection

GEO = (6, 7)                     # columns of the latitude and the longitude in the 8 features
PENALTIES = ("none", "ridge", "lasso")


def make_test_indices(n_samples: int, test_size: float = 0.2, seed: int = 2026) -> np.ndarray:
    """Indices of the test districts, drawn once and for all before any modelling.

    The first ``ceil(test_size * n_samples)`` indices of
    ``np.random.default_rng(seed).permutation(n_samples)``, sorted in increasing order.

    Parameters
    ----------
    n_samples : int
        Number of districts, at least 2.
    test_size : float, default=0.2
        Share of the districts kept for the test, strictly between 0 and 1.
    seed : int, default=2026
        Seed of the generator: the same seed gives the same test set.

    Returns
    -------
    np.ndarray of int
        The sorted test indices, without repetition, between 0 and ``n_samples - 1``.
    """
    raise NotImplementedError  # TODO MP2.1


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, ``sqrt(mylearn.linear.mean_squared_error(y_true, y_pred))``."""
    raise NotImplementedError  # TODO MP2.2


class HousingModel:
    """A linear model of the median house value, with its preprocessing learnt in ``fit``.

    ``__init__`` only stores the hyperparameters, under their own names (the convention of
    ``mylearn.model_selection.clone``); everything learnt ends with ``_`` and is set by ``fit``.

    Parameters
    ----------
    degree : int, default=1
        Degree of the polynomial features of the 8 bounded and standardized features
        (1: the features themselves).
    penalty : {"none", "ridge", "lasso"}, default="none"
        The linear model: ``mylearn.linear.LinearRegression``, ``Ridge(alpha)`` or
        ``Lasso(alpha, max_iter=5000, tol=1e-6)`` (a tight stop: the predictions stay
        within 1e-4 of scikit-learn's).
    alpha : float, default=1.0
        Strength of the penalty, passed unchanged to ``Ridge(alpha)`` or ``Lasso(alpha)``
        (ignored when ``penalty="none"``). The two objectives do not scale alike (Ridge sums
        the squared errors, Lasso averages them over ``2n``): the same alpha is not the same
        strength for both.
    n_zones : int, default=0
        Number of geographic zones (k-means on latitude and longitude); 0: no zone.
    clip : tuple of two floats or None, default=(1.0, 99.0)
        Percentiles of the training rows used as bounds of each feature; None: no bounds.
    random_state : int, default=0
        Seed of the k-means of the zones.

    Attributes
    ----------
    low_, high_ : np.ndarray of shape (n_features,)
        Bounds of each feature (``-inf`` and ``+inf`` when ``clip`` is None).
    mean_, scale_ : np.ndarray of shape (n_features,)
        Means and standard deviations (ddof = 0, like scikit-learn's ``StandardScaler``) of
        the bounded training features; a standard deviation of 0 is replaced by 1.
    geo_mean_, geo_scale_ : np.ndarray of shape (2,)
        Means and standard deviations (ddof = 0) of the raw training latitudes and
        longitudes, not bounded (only when ``n_zones > 0``).
    kmeans_ : mylearn.cluster.KMeans or None
        The k-means of the zones (``n_init=1``: the zones are features, and a slightly
        higher inertia costs nothing measurable), fitted on the standardized training
        coordinates.
    design_mean_, design_scale_ : np.ndarray of shape (n_columns,)
        Means and standard deviations (ddof = 0) of the columns of the design matrix
        (polynomial features, then the zones), standard deviations of 0 replaced by 1.
    regressor_ : mylearn.linear model
        The fitted linear model.
    """

    def __init__(self, degree: int = 1, penalty: str = "none", alpha: float = 1.0, n_zones: int = 0,
                 clip: tuple[float, float] | None = (1.0, 99.0), random_state: int = 0):
        self.degree = degree
        self.penalty = penalty
        self.alpha = alpha
        self.n_zones = n_zones
        self.clip = clip
        self.random_state = random_state

    def fit(self, X, y) -> HousingModel:
        """Learn the bounds, the standardizations, the zones and the linear model on (X, y) only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training districts (the 8 features, latitude and longitude in the columns ``GEO``).
        y : array-like of shape (n_samples,)
            Their median house values.

        Returns
        -------
        HousingModel
            The fitted model itself.

        Raises
        ------
        ValueError
            For an invalid hyperparameter: a degree below 1, an unknown penalty, ``alpha <= 0``
            with Ridge or Lasso, a negative number of zones, bounds that are not two
            percentiles ``0 <= low < high <= 100``; or when X and y differ in length.
        """
        raise NotImplementedError  # TODO MP2.2 (degree=1, penalty="none"), then MP2.3 (degree, Ridge, Lasso) and MP2.4 (zones)

    def transform(self, X) -> np.ndarray:
        """The design matrix of X, built with the statistics learnt by ``fit`` (never refitted here).

        RuntimeError when the model is not fitted yet (as ``zones`` and ``predict``).

        Returns
        -------
        np.ndarray of shape (n_samples, n_columns)
            The standardized polynomial features, then one column per zone.
        """
        raise NotImplementedError  # TODO MP2.2, then MP2.4 (the zones)

    def zones(self, X) -> np.ndarray:
        """The zone (0 to ``n_zones - 1``) of each district; ValueError when the model has no zone."""
        raise NotImplementedError  # TODO MP2.4

    def predict(self, X) -> np.ndarray:
        """Predicted median house value of each district of X."""
        raise NotImplementedError  # TODO MP2.2


def cross_validate(model, X, y, folds: Sequence) -> dict:
    """Train and validation RMSE of the model in each fold.

    For each pair ``(train_idx, val_idx)`` of ``folds``, a fresh clone of the model
    (``mylearn.model_selection.clone``) is fitted on the training rows, then measured on
    them and on the validation rows. The model passed is never fitted.

    Returns
    -------
    dict
        ``{"train_rmse": np.ndarray, "val_rmse": np.ndarray}``, one value per fold.
    """
    raise NotImplementedError  # TODO MP2.2


def out_of_fold_predictions(model, X, y, folds: Sequence) -> np.ndarray:
    """The prediction of each sample by the clone that did not see it (the one of its validation fold).

    Every sample must be in exactly one validation fold (ValueError otherwise).

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    raise NotImplementedError  # TODO MP2.5


def validation_curve(model, param: str, values: Sequence, X, y, folds: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Train and validation RMSE along one hyperparameter.

    For each value, a clone of the model gets ``param = value`` and is measured by
    ``cross_validate``. ValueError when ``param`` is not a hyperparameter of the model.

    Returns
    -------
    tuple of two np.ndarray of shape (len(values), n_folds)
        The training RMSE and the validation RMSE, one row per value, one column per fold.
    """
    raise NotImplementedError  # TODO MP2.3


def learning_curve(model, X, y, sizes: Sequence[int], folds: Sequence, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Train and validation RMSE along the number of training examples.

    In each fold, the training indices are put in a random order,
    ``train_idx[np.random.default_rng(seed).permutation(len(train_idx))]`` (a new generator
    with the same seed in every fold; the fold itself is never modified); for each size, a
    clone of the model learns from the first ``size`` of them and is measured on them and on
    the whole validation fold.
    ValueError when a size is not an integer between 1 and the smallest training part.

    Returns
    -------
    tuple of two np.ndarray of shape (len(sizes), n_folds)
        The training RMSE and the validation RMSE, one row per size, one column per fold.
    """
    raise NotImplementedError  # TODO MP2.5
