"""VAE building blocks — mylearn, chapter 25 (Autoencoders).

The mathematical core of the variational autoencoder in NumPy: the
reparameterization trick (forward and backward passes), the Gaussian KL
divergence and its gradient, and interpolation in the latent space. With
``mylearn.nn.layers``, ``mylearn.nn.backward`` and ``mylearn.optim``, they are
enough to train a whole VAE in NumPy. ``interpolate_latents`` is reused for
the GANs of chapter 27 and in bonus chapter B5.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def reparameterize(
    mu: ArrayLike, log_var: ArrayLike, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Draw z = mu + exp(0.5 * log_var) * eps with eps ~ N(0, I) (reparameterization trick).

    The randomness is moved into ``eps``, an input that does not depend on the
    network: z becomes a deterministic, differentiable function of ``mu`` and
    ``log_var``, so the gradient can flow back to the encoder.

    Parameters
    ----------
    mu : array-like of shape (n, d)
        Means of the approximate posterior q(z|x), one row per sample.
    log_var : array-like of shape (n, d)
        Log-variances log(sigma**2), same shape as ``mu``.
    rng : numpy.random.Generator or None, default=None
        Generator used to draw eps. None means ``np.random.default_rng()``.

    Returns
    -------
    z : ndarray of shape (n, d)
        The latent codes.
    eps : ndarray of shape (n, d)
        The standard normal noise used, returned so that the backward pass can
        reuse it.

    Raises
    ------
    ValueError
        If ``mu`` and ``log_var`` do not have the same shape.

    Notes
    -----
    Tested against ``mu + torch.exp(0.5 * log_var) * eps`` for the same eps;
    over 10**5 draws, the empirical mean and variance of z match ``mu`` and
    ``exp(log_var)``.

    Examples
    --------
    >>> z, eps = reparameterize(np.zeros((2, 3)), np.zeros((2, 3)),
    ...                         rng=np.random.default_rng(0))
    >>> z.shape
    (2, 3)
    >>> bool(np.array_equal(z, eps))     # mu = 0 and sigma = 1
    True
    >>> reparameterize([[5.0]], [[-100.0]])[0]    # sigma = exp(-50): no noise left
    array([[5.]])
    """
    raise NotImplementedError("reparameterize() is not implemented yet")


def reparameterize_backward(
    dz: ArrayLike, eps: ArrayLike, log_var: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Backward pass of ``reparameterize``: gradients with respect to mu and log_var.

    Through z = mu + exp(0.5 * log_var) * eps, where eps is a constant: no
    gradient flows into the noise.

    Parameters
    ----------
    dz : array-like of shape (n, d)
        Upstream gradient dL/dz.
    eps : array-like of shape (n, d)
        The noise returned by ``reparameterize``.
    log_var : array-like of shape (n, d)
        The log-variances used in the forward pass.

    Returns
    -------
    dmu : ndarray of shape (n, d)
        dL/dmu (equal to dz).
    dlog_var : ndarray of shape (n, d)
        dL/dlog_var = dz * 0.5 * exp(0.5 * log_var) * eps.

    Raises
    ------
    ValueError
        If the three shapes differ.

    Notes
    -----
    Tested against ``torch.autograd`` on ``mu + torch.exp(0.5 * log_var) * eps``
    with eps fixed.

    Examples
    --------
    >>> dmu, dlog_var = reparameterize_backward(np.ones((1, 2)), np.array([[1.0, -1.0]]),
    ...                                         np.zeros((1, 2)))
    >>> dmu
    array([[1., 1.]])
    >>> dlog_var
    array([[ 0.5, -0.5]])
    """
    raise NotImplementedError("reparameterize_backward() is not implemented yet")


def gaussian_kl(mu: ArrayLike, log_var: ArrayLike) -> np.ndarray:
    """KL(N(mu, diag(exp(log_var))) || N(0, I)) for each sample, summed over the latent dimensions.

    KL = 0.5 * sum over j of (mu_j**2 + exp(log_var_j) - 1 - log_var_j),
    in nats. It is 0 exactly when mu = 0 and log_var = 0 (the prior itself).

    Parameters
    ----------
    mu : array-like of shape (n, d)
        Means.
    log_var : array-like of shape (n, d)
        Log-variances.

    Returns
    -------
    ndarray of shape (n,)
        One non-negative KL divergence per sample.

    Raises
    ------
    ValueError
        If the shapes differ.

    Notes
    -----
    Tested against
    ``torch.distributions.kl_divergence(Normal(mu, std), Normal(0, 1)).sum(-1)``.

    Examples
    --------
    >>> gaussian_kl(np.zeros((1, 2)), np.zeros((1, 2)))
    array([0.])
    >>> gaussian_kl([[1.0, 0.0]], [[0.0, 0.0]])
    array([0.5])
    """
    raise NotImplementedError("gaussian_kl() is not implemented yet")


def gaussian_kl_grad(mu: ArrayLike, log_var: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of ``gaussian_kl(mu, log_var).sum()`` with respect to mu and log_var.

    Parameters
    ----------
    mu : array-like of shape (n, d)
        Means.
    log_var : array-like of shape (n, d)
        Log-variances.

    Returns
    -------
    dmu : ndarray of shape (n, d)
        Equal to ``mu``.
    dlog_var : ndarray of shape (n, d)
        Equal to ``0.5 * (exp(log_var) - 1)``.

    Raises
    ------
    ValueError
        If the shapes differ.

    Notes
    -----
    Tested against ``torch.autograd`` on the ``torch.distributions`` KL.

    Examples
    --------
    >>> dmu, dlog_var = gaussian_kl_grad([[1.0, 0.0]], [[0.0, np.log(3.0)]])
    >>> dmu
    array([[1., 0.]])
    >>> dlog_var
    array([[0., 1.]])
    """
    raise NotImplementedError("gaussian_kl_grad() is not implemented yet")


def interpolate_latents(
    z0: ArrayLike, z1: ArrayLike, n_steps: int, method: str = "linear"
) -> np.ndarray:
    """Points from z0 to z1 (both included), on a straight line or along a great circle.

    - ``"linear"``: z(t) = (1 - t) * z0 + t * z1 for t = 0, 1/(n_steps - 1), ..., 1.
    - ``"slerp"`` (spherical interpolation): with omega the angle between z0
      and z1, z(t) = sin((1 - t) * omega) / sin(omega) * z0
      + sin(t * omega) / sin(omega) * z1. Between two unit vectors the points
      stay on the unit sphere with constant angular steps: better suited to
      Gaussian latent spaces, where typical codes lie near a sphere. When
      sin(omega) is (almost) 0 (z0 and z1 parallel or anti-parallel, or a zero
      vector), fall back to linear interpolation.

    Parameters
    ----------
    z0 : array-like of shape (d,)
        Start latent vector.
    z1 : array-like of shape (d,)
        End latent vector.
    n_steps : int
        Number of points, >= 2.
    method : {"linear", "slerp"}, default="linear"
        Interpolation path.

    Returns
    -------
    ndarray of shape (n_steps, d)
        The points in order: first row z0, last row z1.

    Raises
    ------
    ValueError
        For an unknown method, ``n_steps < 2``, or z0 and z1 of different
        shapes.

    Notes
    -----
    Tested against an ``np.linspace``-based interpolation for ``"linear"``; for
    unit vectors, ``"slerp"`` keeps the norm 1 and constant angular steps
    (angles from ``np.arccos`` of the cosine similarity).

    Examples
    --------
    >>> interpolate_latents(np.array([0.0, 0.0]), np.array([1.0, 2.0]), 3)
    array([[0. , 0. ],
           [0.5, 1. ],
           [1. , 2. ]])
    >>> interpolate_latents(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 3, method="slerp")
    array([[1.        , 0.        ],
           [0.70710678, 0.70710678],
           [0.        , 1.        ]])
    """
    # TODO: t goes from 0 to 1 in n_steps points; for slerp, get omega from the
    #   normalised vectors (clip the cosine to [-1, 1] before np.arccos).
    raise NotImplementedError("interpolate_latents() is not implemented yet")
