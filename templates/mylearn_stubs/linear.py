"""Linear models and regularisation — mylearn, chapter 9 (Overfitting and Underfitting).

Polynomial features, regression metrics (MSE, R²), ordinary least squares, Ridge (L2
penalty) and Lasso (L1 penalty, coordinate descent), the bias-variance decomposition
and the Bayesian posterior of a straight line on a grid. You will reuse them in
chapter 12 (scaling + Ridge), chapter 14 (bagging reduces variance) and mini-project 2.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Self

import numpy as np
from numpy.typing import ArrayLike


def polynomial_features(X: ArrayLike, degree: int = 2, include_bias: bool = False) -> np.ndarray:
    """Build all the monomials of the features up to a total degree.

    For features (a, b) and ``degree=2`` the output columns are a, b, a², ab, b²:
    degree 1 first, then degree 2, ..., each degree in the order of
    ``itertools.combinations_with_replacement(range(n_features), d)`` (scikit-learn's
    order). With ``include_bias=True`` a column of ones comes first.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_features)
        Input features; a 1-D array is treated as a single feature.
    degree : int, default=2
        Maximum total degree (>= 1).
    include_bias : bool, default=False
        Prepend a column of ones (the monomial of degree 0).

    Returns
    -------
    np.ndarray of shape (n_samples, n_output_features)
        Float array of the monomials.

    Raises
    ------
    ValueError
        If ``degree < 1`` or ``X`` has more than 2 dimensions.

    Notes
    -----
    Tested against ``sklearn.preprocessing.PolynomialFeatures(degree,
    include_bias=include_bias).fit_transform(X)``: identical columns in the same order.

    Examples
    --------
    >>> polynomial_features(np.array([[2.0, 3.0], [1.0, -1.0]]), degree=2)
    array([[ 2.,  3.,  4.,  6.,  9.],
           [ 1., -1.,  1., -1.,  1.]])
    >>> polynomial_features(np.array([2.0, 3.0]), degree=3, include_bias=True)
    array([[ 1.,  2.,  4.,  8.],
           [ 1.,  3.,  9., 27.]])
    """
    # TODO: one output column per combination of feature indices (product of the
    # corresponding columns), degree by degree.
    raise NotImplementedError("polynomial_features() is not implemented yet")


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean of the squared residuals (MSE).

    MSE = (1/n) * sum_i (y_i - ŷ_i)^2.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Targets.
    y_pred : array-like of shape (n_samples,)
        Predictions.

    Returns
    -------
    float
        MSE >= 0.

    Raises
    ------
    ValueError
        If the lengths differ or the arrays are empty.

    Notes
    -----
    Tested against ``sklearn.metrics.mean_squared_error``.

    Examples
    --------
    >>> mean_squared_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
    0.375
    """
    # TODO: residuals, squared, averaged.
    raise NotImplementedError("mean_squared_error() is not implemented yet")


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean of the absolute residuals (MAE).

    MAE = (1/n) * sum_i |y_i - ŷ_i|. It is expressed in the unit of the target and,
    unlike the MSE, it does not square large errors, so a few outliers weigh less.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Targets.
    y_pred : array-like of shape (n_samples,)
        Predictions.

    Returns
    -------
    float
        MAE >= 0.

    Raises
    ------
    ValueError
        If the lengths differ or the arrays are empty.

    Notes
    -----
    Tested against ``sklearn.metrics.mean_absolute_error``.

    Examples
    --------
    >>> mean_absolute_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
    0.5
    """
    # TODO: residuals, absolute values, averaged.
    raise NotImplementedError("mean_absolute_error() is not implemented yet")


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the coefficient of determination R² = 1 - SS_res / SS_tot.

    SS_res = sum_i (y_i - ŷ_i)^2 and SS_tot = sum_i (y_i - mean(y))^2. R² = 1 is a
    perfect fit, R² = 0 is as good as always predicting the mean, R² < 0 is worse.
    Special cases (as scikit-learn): 1.0 if SS_res and SS_tot are both 0, 0.0 if only
    SS_tot is 0 (constant target).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Targets.
    y_pred : array-like of shape (n_samples,)
        Predictions.

    Returns
    -------
    float
        R² <= 1 (can be negative).

    Raises
    ------
    ValueError
        If the lengths differ or there are fewer than 2 samples.

    Notes
    -----
    Tested against ``sklearn.metrics.r2_score``.

    Examples
    --------
    >>> round(r2_score([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), 4)
    0.9486
    """
    # TODO: the two sums of squares, then the special cases.
    raise NotImplementedError("r2_score() is not implemented yet")


class LinearRegression:
    """Ordinary least squares linear regression.

    Finds w and b minimising ``sum_i (y_i - x_i . w - b)^2``. With an intercept, the
    data are centred first (X - mean of X, y - mean of y), ``np.linalg.lstsq`` gives w,
    and the intercept is recovered from the means: ``b = mean(y) - mean(X) . w``.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Learn an intercept b; if False, b = 0 and the model goes through the origin.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Weights w.
    intercept_ : float
        Intercept b (0.0 if ``fit_intercept=False``).

    Notes
    -----
    Tested against ``sklearn.linear_model.LinearRegression`` (``coef_``,
    ``intercept_``, ``predict``).

    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> y = np.array([3.0, 5.0, 7.0])
    >>> model = LinearRegression().fit(X, y)
    >>> model.coef_.round(4)
    array([2.])
    >>> round(model.intercept_, 4)
    1.0
    >>> model.predict(np.array([[10.0]])).round(4)
    array([21.])
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the least squares solution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples (must be 2-D).
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        Self
            The fitted model (``self``), with ``coef_`` and ``intercept_``.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D or ``len(X) != len(y)``.
        """
        # TODO: centre (if fit_intercept), solve with np.linalg.lstsq, recover b.
        raise NotImplementedError("fit() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict ``X @ coef_ + intercept_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predictions.
        """
        # TODO: one line.
        raise NotImplementedError("predict() is not implemented yet")

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        float
            ``r2_score(y, predict(X))``.
        """
        # TODO: reuse r2_score.
        raise NotImplementedError("score() is not implemented yet")


class Ridge:
    """Linear least squares with an L2 penalty (ridge regression).

    Minimises ``||y - Xw - b||^2 + alpha * ||w||^2`` (``alpha`` is the book's λ). On
    centred data the solution is ``w = (XᵀX + alpha I)⁻¹ Xᵀy``, computed with
    ``np.linalg.solve`` (never an explicit inverse); the intercept, recovered from the
    means, is not penalised. Larger alpha: smaller weights, less variance, more bias.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularisation strength (>= 0; 0 gives ordinary least squares).
    fit_intercept : bool, default=True
        Learn an unpenalised intercept.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Weights w.
    intercept_ : float
        Intercept b (0.0 if ``fit_intercept=False``).

    Notes
    -----
    Tested against ``sklearn.linear_model.Ridge(alpha)``: same ``coef_`` and
    ``intercept_`` (atol 1e-8).

    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> y = np.array([3.0, 5.0, 7.0])
    >>> model = Ridge(alpha=1.0).fit(X, y)
    >>> model.coef_.round(4)
    array([1.3333])
    >>> round(model.intercept_, 4)
    2.3333
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the closed-form ridge solution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples (must be 2-D).
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        Self
            The fitted model (``self``), with ``coef_`` and ``intercept_``.

        Raises
        ------
        ValueError
            If ``alpha < 0``, ``X`` is not 2-D or ``len(X) != len(y)``.
        """
        # TODO: centre (if fit_intercept), solve (XᵀX + alpha I) w = Xᵀy, recover b.
        raise NotImplementedError("fit() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict ``X @ coef_ + intercept_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predictions.
        """
        # TODO: one line.
        raise NotImplementedError("predict() is not implemented yet")

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        float
            ``r2_score(y, predict(X))``.
        """
        # TODO: reuse r2_score.
        raise NotImplementedError("score() is not implemented yet")


def soft_threshold(z: ArrayLike, gamma: float) -> np.ndarray:
    """Apply the soft-thresholding operator ``sign(z) * max(|z| - gamma, 0)``.

    Values in [-gamma, gamma] become 0, the others move towards 0 by gamma. It is the
    exact solution of the 1-D lasso problem ``argmin_w 0.5 (w - z)^2 + gamma |w|``,
    hence the building block of the coordinate descent of ``Lasso``.

    Parameters
    ----------
    z : array-like or float
        Input values.
    gamma : float
        Threshold (>= 0).

    Returns
    -------
    np.ndarray
        Float array with the shape of ``z`` (0-d for a scalar).

    Raises
    ------
    ValueError
        If ``gamma < 0``.

    Notes
    -----
    Tested by properties: zero on [-gamma, gamma], odd function, slope 1 outside, and
    equal to the minimiser of ``0.5 (w - z)^2 + gamma |w|`` found by a fine grid search.

    Examples
    --------
    >>> soft_threshold(np.array([-3.0, 0.5, 2.0]), 1.0)
    array([-2.,  0.,  1.])
    """
    # TODO: check gamma, then vectorise the formula (no loop, no if on the values).
    raise NotImplementedError("soft_threshold() is not implemented yet")


class Lasso:
    """Linear regression with an L1 penalty (lasso), by cyclic coordinate descent.

    Minimises scikit-learn's objective ``(1 / 2n) ||y - Xw - b||^2 + alpha ||w||_1``.
    Data are centred first (if ``fit_intercept``); the weights start at 0. One pass
    updates each coordinate j in turn, all the others fixed:
    ``w_j = soft_threshold(rho_j, alpha) / z_j`` with ``rho_j = x_jᵀ r_j / n``, where
    ``r_j`` is the residual computed without feature j, and ``z_j = x_jᵀ x_j / n``
    (a column of zeros gets ``w_j = 0``). A large alpha sets many weights exactly to 0.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularisation strength (> 0; use ``LinearRegression`` for alpha = 0).
    fit_intercept : bool, default=True
        Learn an unpenalised intercept (data centred first).
    max_iter : int, default=1000
        Maximum number of full passes over the coordinates.
    tol : float, default=1e-4
        Stop after a pass whose largest absolute change of a coefficient is < tol.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Weights w (many exact zeros when alpha is large).
    intercept_ : float
        Intercept b (0.0 if ``fit_intercept=False``).
    n_iter_ : int
        Number of passes done.

    Notes
    -----
    Tested against ``sklearn.linear_model.Lasso(alpha, tol=1e-12, max_iter=100000)``:
    same ``coef_`` (atol 1e-6); property: all coefficients are 0 when
    ``alpha >= max |Xcᵀ yc| / n`` (Xc, yc: centred data). scikit-learn stops on a
    duality gap instead, so ``n_iter_`` is not compared.

    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> y = np.array([3.0, 5.0, 7.0])
    >>> model = Lasso(alpha=0.5).fit(X, y)
    >>> model.coef_.round(4)
    array([1.25])
    >>> round(model.intercept_, 4)
    2.5
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the lasso by cyclic coordinate descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples (must be 2-D).
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        Self
            The fitted model (``self``), with ``coef_``, ``intercept_`` and ``n_iter_``.

        Raises
        ------
        ValueError
            If ``alpha <= 0``, ``X`` is not 2-D or ``len(X) != len(y)``.
        """
        # TODO: centre, then passes over the coordinates with soft_threshold (keep the
        # residual up to date), stop on tol or max_iter; recover b.
        raise NotImplementedError("fit() is not implemented yet")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict ``X @ coef_ + intercept_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predictions.
        """
        # TODO: one line.
        raise NotImplementedError("predict() is not implemented yet")

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        float
            ``r2_score(y, predict(X))``.
        """
        # TODO: reuse r2_score.
        raise NotImplementedError("score() is not implemented yet")


def bias_variance_decomposition(predictions: ArrayLike, f_true: ArrayLike) -> tuple[float, float]:
    """Measure the squared bias and the variance of a family of models.

    Row m of ``predictions`` holds the predictions of the model trained on the m-th
    dataset, at the same evaluation points x. With ``mean_m`` the average over the
    models and ``mean_x`` the average over the points:
    ``bias2 = mean_x (mean_m f_m(x) - f(x))^2`` and
    ``variance = mean_x var_m f_m(x)`` (population variance, ddof=0), so that
    ``mean_m mean_x (f_m(x) - f(x))^2 = bias2 + variance`` exactly.

    Parameters
    ----------
    predictions : array-like of shape (n_models, n_points)
        Predictions of every model at every evaluation point.
    f_true : array-like of shape (n_points,)
        Noise-free target function at the same points.

    Returns
    -------
    tuple of (float, float)
        ``(bias2, variance)``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or there are fewer than 2 models.

    Notes
    -----
    Tested against the NumPy formulas and the exact identity above.

    Examples
    --------
    >>> bias_variance_decomposition([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
    (1.0, 1.0)
    """
    # TODO: average model, then the two means described above.
    raise NotImplementedError("bias_variance_decomposition() is not implemented yet")


def bayes_line_posterior(
    x: ArrayLike,
    y: ArrayLike,
    slopes: ArrayLike,
    intercepts: ArrayLike,
    noise_std: float = 0.1,
    prior_std: float = 1.0,
) -> np.ndarray:
    """Compute the posterior probability of every line of a (slope, intercept) grid.

    Each grid cell is a line ``y = slope * x + intercept``. Prior: isotropic Gaussian
    centred on (0, 0), ``log prior = -(slope^2 + intercept^2) / (2 prior_std^2)``.
    Likelihood: Gaussian residuals,
    ``log likelihood = -sum_i (y_i - slope * x_i - intercept)^2 / (2 noise_std^2)``.
    The log-posterior (their sum) is computed in log space, the maximum is subtracted
    before ``np.exp`` (no underflow), and the grid is normalised to sum to 1.

    Parameters
    ----------
    x : array-like of shape (n,)
        Abscissas of the observed points; may be empty (then only the prior is left).
    y : array-like of shape (n,)
        Ordinates of the observed points.
    slopes : array-like of shape (n_s,)
        Grid of slopes.
    intercepts : array-like of shape (n_b,)
        Grid of intercepts.
    noise_std : float, default=0.1
        Standard deviation of the residuals in the likelihood (> 0).
    prior_std : float, default=1.0
        Standard deviation of the Gaussian prior (> 0).

    Returns
    -------
    np.ndarray of shape (n_b, n_s)
        Posterior probabilities summing to 1: row = intercept, column = slope (as in
        the book's slope-intercept diagrams).

    Raises
    ------
    ValueError
        If ``len(x) != len(y)`` or a standard deviation is <= 0.

    Notes
    -----
    Tested against the closed-form Bayesian linear regression (Gaussian prior, Bishop
    2006, eq. 3.53-3.54): the posterior mean computed on a fine grid matches the
    analytic mean. Property: updating point by point (each posterior used as the next
    prior) gives the batch posterior.

    Examples
    --------
    >>> post = bayes_line_posterior([2.0], [2.0], slopes=[0.0, 1.0], intercepts=[0.0, 1.0],
    ...                             noise_std=1.0, prior_std=1.0)
    >>> post.round(4)
    array([[0.1015, 0.4551],
           [0.276 , 0.1674]])
    """
    # TODO: grids of slopes and intercepts (np.meshgrid or broadcasting), log prior +
    # log likelihood, subtract the max, exponentiate, normalise.
    raise NotImplementedError("bayes_line_posterior() is not implemented yet")
