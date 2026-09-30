"""Metrics for generative models — mylearn, chapter 27 (Generative Adversarial Networks).

Quantitative measures of how close generated samples are to real ones, in
NumPy: the 1-D Wasserstein distance (to follow the training of a 1-D GAN) and
the Fréchet distance between two Gaussians fitted to feature vectors, the
formula behind the FID used to evaluate image generators (DCGAN in this
chapter, diffusion models in bonus chapter B5).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


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
    # TODO: for samples of different sizes, integrate |F_u - F_v| (the two empirical
    #   CDFs) between consecutive values of the merged sorted samples.
    raise NotImplementedError("wasserstein_1d() is not implemented yet")


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
    raise NotImplementedError("frechet_distance() is not implemented yet")


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
    raise NotImplementedError("frechet_distance_from_features() is not implemented yet")
