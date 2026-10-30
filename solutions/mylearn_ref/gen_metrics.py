"""Metrics for generative models — mylearn, chapter 27 (Generative Adversarial Networks).

Quantitative measures of how close generated samples are to real ones, in
NumPy: the 1-D Wasserstein distance (to follow the training of a 1-D GAN) and
the Fréchet distance between two Gaussians fitted to feature vectors, the
formula behind the FID used to evaluate image generators (DCGAN in this
chapter, diffusion models in bonus chapter B5).

Reference implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _finite_array(x: ArrayLike, name: str) -> np.ndarray:
    """``x`` as a new float64 array; NaN and infinities are refused."""
    try:
        arr = np.array(x, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numbers ({exc})") from None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values: remove them first")
    return arr


def _sample_1d(x: ArrayLike, name: str) -> np.ndarray:
    arr = _finite_array(x, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sample, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def wasserstein_1d(u: ArrayLike, v: ArrayLike) -> float:
    """Earth mover's distance between two 1-D empirical distributions.

    The minimal average distance to move the mass of one sample onto the
    other: the area between their two empirical cumulative distribution
    functions. For samples of the same size it is the mean gap between the
    sorted samples, mean(|sort(u) - sort(v)|).

    Parameters
    ----------
    u : array-like of shape (n,)
        First sample.
    v : array-like of shape (m,)
        Second sample (m may differ from n).

    Returns
    -------
    float
        The distance, >= 0.

    Raises
    ------
    ValueError
        If a sample is empty or not 1-D.

    Notes
    -----
    Tested against ``scipy.stats.wasserstein_distance``.

    Examples
    --------
    >>> wasserstein_1d([0.0, 1.0, 3.0], [5.0, 6.0, 8.0])    # a shift by 5
    5.0
    >>> wasserstein_1d([0.0, 1.0], [0.0, 1.0, 2.0])
    0.5
    """
    u_sorted = np.sort(_sample_1d(u, "u"))
    v_sorted = np.sort(_sample_1d(v, "v"))
    if u_sorted.size == v_sorted.size:
        return float(np.mean(np.abs(u_sorted - v_sorted)))
    # Area between the two empirical CDFs: both are constant between consecutive
    # values of the merged sample, where they equal (number of values <= x) / size.
    merged = np.sort(np.concatenate([u_sorted, v_sorted]))
    widths = np.diff(merged)
    cdf_u = np.searchsorted(u_sorted, merged[:-1], side="right") / u_sorted.size
    cdf_v = np.searchsorted(v_sorted, merged[:-1], side="right") / v_sorted.size
    return float(np.sum(np.abs(cdf_u - cdf_v) * widths))


def frechet_distance(
    mu1: ArrayLike, sigma1: ArrayLike, mu2: ArrayLike, sigma2: ArrayLike
) -> float:
    """Squared Fréchet distance between N(mu1, sigma1) and N(mu2, sigma2), the formula of FID.

    d**2 = ||mu1 - mu2||**2 + Tr(sigma1 + sigma2 - 2 (sigma1 sigma2)**(1/2)).
    The first term compares the means, the second the covariances.

    Parameters
    ----------
    mu1 : array-like of shape (d,)
        Mean of the first Gaussian.
    sigma1 : array-like of shape (d, d)
        Covariance of the first Gaussian (symmetric positive semi-definite).
    mu2 : array-like of shape (d,)
        Mean of the second Gaussian.
    sigma2 : array-like of shape (d, d)
        Covariance of the second Gaussian.

    Returns
    -------
    float
        The squared distance, >= 0 (tiny negative values due to rounding are
        clipped to 0).

    Raises
    ------
    ValueError
        If the shapes are inconsistent.

    Notes
    -----
    No ``scipy.linalg.sqrtm`` in mylearn: the trace of the matrix square root
    is the sum of the square roots of the eigenvalues of sigma1 @ sigma2
    (real and >= 0 in theory; keep the real part and clip at 0).
    Tested against the reference formula with ``scipy.linalg.sqrtm`` (as in
    pytorch-fid) and against the closed form for diagonal covariances.

    Examples
    --------
    >>> frechet_distance(np.zeros(2), np.eye(2), np.array([3.0, 4.0]), np.eye(2))
    25.0
    >>> frechet_distance(np.zeros(2), np.diag([1.0, 4.0]), np.zeros(2), np.diag([4.0, 1.0]))
    2.0
    """
    mu1 = _finite_array(mu1, "mu1")
    mu2 = _finite_array(mu2, "mu2")
    sigma1 = _finite_array(sigma1, "sigma1")
    sigma2 = _finite_array(sigma2, "sigma2")
    if mu1.ndim != 1:
        raise ValueError(f"mu1 must be a 1-D vector (d,), got shape {mu1.shape}")
    d = mu1.size
    if mu2.shape != (d,):
        raise ValueError(f"mu2 must have the shape of mu1 {(d,)}, got {mu2.shape}")
    for name, sigma in (("sigma1", sigma1), ("sigma2", sigma2)):
        if sigma.shape != (d, d):
            raise ValueError(f"{name} must have the shape {(d, d)} (d = len(mu1)), got {sigma.shape}")
    diff = mu1 - mu2
    eigenvalues = np.linalg.eigvals(sigma1 @ sigma2)
    trace_sqrt = np.sum(np.sqrt(np.clip(eigenvalues.real, 0.0, None)))
    d2 = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * trace_sqrt
    return float(max(d2, 0.0))


def frechet_distance_from_features(feats1: ArrayLike, feats2: ArrayLike) -> float:
    """Fit a Gaussian to each set of feature vectors and return their Fréchet distance.

    Each set is summarised by its mean and its sample covariance
    (``np.cov(feats, rowvar=False)``, normalised by n - 1), then
    ``frechet_distance`` compares the two Gaussians. With the features of an
    Inception network this is the FID.

    Parameters
    ----------
    feats1 : array-like of shape (n1, d)
        Features of real samples, one row per sample.
    feats2 : array-like of shape (n2, d)
        Features of generated samples.

    Returns
    -------
    float
        The squared Fréchet distance, >= 0; close to 0 for two large samples
        of the same distribution.

    Raises
    ------
    ValueError
        If the feature dimensions differ, an input is not 2-D, or a set has
        fewer than 2 samples.

    Notes
    -----
    Tested against ``frechet_distance`` applied to ``np.mean`` and
    ``np.cov(rowvar=False)``, and against the FID formula of torchmetrics on
    the same features.

    Examples
    --------
    >>> X = np.random.default_rng(0).normal(size=(500, 3))
    >>> round(frechet_distance_from_features(X, X), 6)
    0.0
    >>> round(frechet_distance_from_features(X, X + 2.0), 6)    # ||(2, 2, 2)||**2
    12.0
    """
    gaussians = []
    for name, feats in (("feats1", feats1), ("feats2", feats2)):
        arr = _finite_array(feats, name)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2-D (n_samples, d), got shape {arr.shape}")
        if arr.shape[0] < 2:
            raise ValueError(f"{name} needs at least 2 samples to estimate a covariance, got {arr.shape[0]}")
        gaussians.append((arr.mean(axis=0), np.cov(arr, rowvar=False).reshape(arr.shape[1], arr.shape[1])))
    (mu1, sigma1), (mu2, sigma2) = gaussians
    if mu1.shape != mu2.shape:
        raise ValueError(f"the feature dimensions differ: {mu1.size} and {mu2.size}")
    return frechet_distance(mu1, sigma1, mu2, sigma2)
