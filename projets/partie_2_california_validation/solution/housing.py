"""Median house value of California districts with linear models (mini-project of part II, reference solution).

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
    if int(n_samples) != n_samples or n_samples < 2:
        raise ValueError(f"n_samples must be an integer >= 2, got {n_samples!r}")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size!r}")
    n_test = int(np.ceil(test_size * n_samples))
    return np.sort(np.random.default_rng(seed).permutation(int(n_samples))[:n_test])


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, ``sqrt(mylearn.linear.mean_squared_error(y_true, y_pred))``."""
    return float(np.sqrt(linear.mean_squared_error(y_true, y_pred)))


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

    def _check_params(self) -> None:
        if int(self.degree) != self.degree or self.degree < 1:
            raise ValueError(f"degree must be an integer >= 1, got {self.degree!r}")
        if self.penalty not in PENALTIES:
            raise ValueError(f"penalty must be one of {PENALTIES}, got {self.penalty!r}")
        if self.penalty != "none" and not self.alpha > 0:
            raise ValueError(f"alpha must be > 0 with penalty={self.penalty!r}, got {self.alpha!r}")
        if int(self.n_zones) != self.n_zones or self.n_zones < 0:
            raise ValueError(f"n_zones must be an integer >= 0, got {self.n_zones!r}")
        if self.clip is not None:
            low, high = self.clip
            if not 0 <= low < high <= 100:
                raise ValueError(f"clip must be None or two percentiles 0 <= low < high <= 100, got {self.clip!r}")

    def _features(self, X: np.ndarray) -> np.ndarray:
        """Bounded, standardized features, then their polynomial features."""
        Z = (np.clip(X, self.low_, self.high_) - self.mean_) / self.scale_
        return Z if self.degree == 1 else linear.polynomial_features(Z, degree=int(self.degree))

    def _geo(self, X: np.ndarray) -> np.ndarray:
        return (X[:, list(GEO)] - self.geo_mean_) / self.geo_scale_

    def _columns(self, X: np.ndarray) -> np.ndarray:
        """The design matrix before its last standardization: polynomial features, then one column per zone."""
        P = self._features(X)
        if self.kmeans_ is None:
            return P
        labels = np.asarray(self.kmeans_.predict(self._geo(X)), dtype=int)
        return np.hstack([P, np.eye(int(self.n_zones))[labels]])

    @staticmethod
    def _as_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array (one row per district), got shape {X.shape}")
        return X

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
        self._check_params()
        X = self._as_X(X)
        y = np.asarray(y, dtype=float)
        if len(y) != len(X):
            raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}")
        if self.clip is None:
            self.low_, self.high_ = np.full(X.shape[1], -np.inf), np.full(X.shape[1], np.inf)
        else:
            self.low_, self.high_ = np.percentile(X, list(self.clip), axis=0)
        bounded = np.clip(X, self.low_, self.high_)
        self.mean_, self.scale_ = bounded.mean(axis=0), bounded.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.kmeans_ = None
        if self.n_zones > 0:
            geo = X[:, list(GEO)]
            self.geo_mean_, self.geo_scale_ = geo.mean(axis=0), geo.std(axis=0)
            self.geo_scale_[self.geo_scale_ == 0] = 1.0
            self.kmeans_ = cluster.KMeans(n_clusters=int(self.n_zones), n_init=1, random_state=self.random_state)
            self.kmeans_.fit(self._geo(X))
        C = self._columns(X)
        self.design_mean_, self.design_scale_ = C.mean(axis=0), C.std(axis=0)
        self.design_scale_[self.design_scale_ == 0] = 1.0
        if self.penalty == "none":
            self.regressor_ = linear.LinearRegression()
        elif self.penalty == "ridge":
            self.regressor_ = linear.Ridge(alpha=self.alpha)
        else:
            self.regressor_ = linear.Lasso(alpha=self.alpha, max_iter=5000, tol=1e-6)
        self.regressor_.fit((C - self.design_mean_) / self.design_scale_, y)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "regressor_"):
            raise RuntimeError("call fit before transform, zones or predict")

    def transform(self, X) -> np.ndarray:
        """The design matrix of X, built with the statistics learnt by ``fit`` (never refitted here).

        RuntimeError when the model is not fitted yet (as ``zones`` and ``predict``).

        Returns
        -------
        np.ndarray of shape (n_samples, n_columns)
            The standardized polynomial features, then one column per zone.
        """
        self._check_fitted()
        return (self._columns(self._as_X(X)) - self.design_mean_) / self.design_scale_

    def zones(self, X) -> np.ndarray:
        """The zone (0 to ``n_zones - 1``) of each district; ValueError when the model has no zone."""
        self._check_fitted()
        if self.kmeans_ is None:
            raise ValueError("this model has no geographic zone (n_zones=0)")
        return np.asarray(self.kmeans_.predict(self._geo(self._as_X(X))), dtype=int)

    def predict(self, X) -> np.ndarray:
        """Predicted median house value of each district of X."""
        return np.asarray(self.regressor_.predict(self.transform(X)), dtype=float)


def _check_folds(folds) -> list:
    folds = list(folds)
    if not folds:
        raise ValueError("folds must contain at least one (train_indices, val_indices) pair")
    return folds


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
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    train_scores, val_scores = [], []
    for train_idx, val_idx in _check_folds(folds):
        fitted = model_selection.clone(model).fit(X[train_idx], y[train_idx])
        train_scores.append(rmse(y[train_idx], fitted.predict(X[train_idx])))
        val_scores.append(rmse(y[val_idx], fitted.predict(X[val_idx])))
    return {"train_rmse": np.array(train_scores), "val_rmse": np.array(val_scores)}


def out_of_fold_predictions(model, X, y, folds: Sequence) -> np.ndarray:
    """The prediction of each sample by the clone that did not see it (the one of its validation fold).

    Every sample must be in exactly one validation fold (ValueError otherwise).

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    folds = _check_folds(folds)
    seen = np.concatenate([np.asarray(val_idx) for _, val_idx in folds])
    if len(seen) != len(y) or not np.array_equal(np.sort(seen), np.arange(len(y))):
        raise ValueError("every sample must be in exactly one validation fold")
    predictions = np.empty(len(y))
    for train_idx, val_idx in folds:
        fitted = model_selection.clone(model).fit(X[train_idx], y[train_idx])
        predictions[val_idx] = fitted.predict(X[val_idx])
    return predictions


def validation_curve(model, param: str, values: Sequence, X, y, folds: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Train and validation RMSE along one hyperparameter.

    For each value, a clone of the model gets ``param = value`` and is measured by
    ``cross_validate``. ValueError when ``param`` is not a hyperparameter of the model.

    Returns
    -------
    tuple of two np.ndarray of shape (len(values), n_folds)
        The training RMSE and the validation RMSE, one row per value, one column per fold.
    """
    if param not in vars(model):
        raise ValueError(f"{param!r} is not a hyperparameter of {type(model).__name__}")
    train_rows, val_rows = [], []
    for value in values:
        candidate = model_selection.clone(model)
        setattr(candidate, param, value)
        scores = cross_validate(candidate, X, y, folds)
        train_rows.append(scores["train_rmse"])
        val_rows.append(scores["val_rmse"])
    return np.array(train_rows), np.array(val_rows)


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
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    folds = _check_folds(folds)
    smallest = min(len(train_idx) for train_idx, _ in folds)
    if any(int(size) != size or not 1 <= size <= smallest for size in sizes):
        raise ValueError(f"every size must be an integer between 1 and {smallest} (the smallest training part)")
    train_scores = np.empty((len(sizes), len(folds)))
    val_scores = np.empty((len(sizes), len(folds)))
    for j, (train_idx, val_idx) in enumerate(folds):
        order = np.asarray(train_idx)[np.random.default_rng(seed).permutation(len(train_idx))]
        for i, size in enumerate(sizes):
            rows = order[: int(size)]
            fitted = model_selection.clone(model).fit(X[rows], y[rows])
            train_scores[i, j] = rmse(y[rows], fitted.predict(X[rows]))
            val_scores[i, j] = rmse(y[val_idx], fitted.predict(X[val_idx]))
    return train_scores, val_scores
