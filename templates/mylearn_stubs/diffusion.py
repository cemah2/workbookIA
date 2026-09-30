"""DDPM and DDIM mathematics — mylearn, chapter B5 (Diffusion models: a minimal DDPM).

The network-independent part of a diffusion model, in NumPy: noise schedules, the
constants derived from them, closed-form noising, the Gaussian posterior, and one reverse
step (DDPM or DDIM). The notebook plugs these functions into a PyTorch U-Net; their tested
properties (exact round trip with the true noise) catch index and square-root mistakes
before any training. Convention: steps are indexed from 0 to T - 1 (step ``t`` of the
paper is index ``t - 1`` here).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, closed-form formulas and, if installed, diffusers).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike


def linear_beta_schedule(
    n_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> np.ndarray:
    """Linearly spaced betas of Ho et al. (2020).

    Parameters
    ----------
    n_steps : int
        Number of steps ``T``.
    beta_start : float, default=1e-4
        Beta at index 0.
    beta_end : float, default=0.02
        Beta at index ``T - 1``.

    Returns
    -------
    np.ndarray of shape (T,)
        float64 betas from ``beta_start`` to ``beta_end``.

    Raises
    ------
    ValueError
        If ``n_steps < 1`` or if ``0 < beta_start <= beta_end < 1`` does not hold.

    Notes
    -----
    Tested against ``np.linspace``, and ``diffusers.DDPMScheduler(beta_schedule='linear')``
    if diffusers is installed.

    Examples
    --------
    >>> linear_beta_schedule(5)
    array([0.0001  , 0.005075, 0.01005 , 0.015025, 0.02    ])
    """
    # TODO: validate, then one NumPy call.
    raise NotImplementedError("linear_beta_schedule() is not implemented yet")


def cosine_beta_schedule(n_steps: int, s: float = 0.008, max_beta: float = 0.999) -> np.ndarray:
    """Cosine schedule of Nichol & Dhariwal (2021).

    ``f(t) = cos(((t / T) + s) / (1 + s) * pi / 2) ** 2`` and ``alpha_bar(t) = f(t) / f(0)``;
    then ``beta_t = min(1 - alpha_bar(t + 1) / alpha_bar(t), max_beta)`` for
    ``t = 0 .. T - 1``.

    Parameters
    ----------
    n_steps : int
        Number of steps ``T``.
    s : float, default=0.008
        Small offset that keeps ``beta_0`` away from 0.
    max_beta : float, default=0.999
        Clipping value; it is reached near ``t = T``, where ``alpha_bar`` goes to 0.

    Returns
    -------
    np.ndarray of shape (T,)
        float64 betas in (0, max_beta].

    Raises
    ------
    ValueError
        If ``n_steps < 1``, ``s < 0``, or ``max_beta`` is not in (0, 1).

    Notes
    -----
    Tested against the closed form in float64, and
    ``diffusers.DDPMScheduler(beta_schedule='squaredcos_cap_v2')`` if diffusers is
    installed.

    Examples
    --------
    >>> np.round(cosine_beta_schedule(5), 4)
    array([0.1013, 0.2795, 0.4736, 0.7241, 0.999 ])
    """
    # TODO: evaluate f on the T + 1 points t / T, then the ratios of consecutive values.
    raise NotImplementedError("cosine_beta_schedule() is not implemented yet")


def noise_schedule(betas: ArrayLike) -> dict[str, np.ndarray]:
    """Every constant derived from the betas that the forward and reverse processes need.

    With ``alpha_t = 1 - beta_t`` and ``alpha_bar_t = alpha_0 alpha_1 ... alpha_t``:

    * ``alpha_bars_prev[t] = alpha_bar_{t-1}``, equal to 1 at ``t = 0``;
    * ``posterior_variance = beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)``
      (beta tilde of Ho et al.), equal to 0 at ``t = 0``;
    * ``posterior_mean_coef1 = beta_t sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)`` and
      ``posterior_mean_coef2 = (1 - alpha_bar_{t-1}) sqrt(alpha_t) / (1 - alpha_bar_t)``,
      the coefficients of ``x0`` and ``x_t`` in the posterior mean.

    Parameters
    ----------
    betas : array-like of shape (T,)
        Values in (0, 1).

    Returns
    -------
    dict of str to np.ndarray
        Keys ``'betas'``, ``'alphas'``, ``'alpha_bars'``, ``'alpha_bars_prev'``,
        ``'sqrt_alpha_bars'``, ``'sqrt_one_minus_alpha_bars'``, ``'posterior_variance'``,
        ``'posterior_mean_coef1'``, ``'posterior_mean_coef2'``; each value is a float64
        array of shape (T,).

    Raises
    ------
    ValueError
        If ``betas`` is not 1-D or a beta is outside (0, 1).

    Notes
    -----
    Tested against the explicit formulas (``np.cumprod``). Properties: ``alpha_bars`` is
    strictly decreasing in (0, 1), ``posterior_variance[0] == 0`` and
    ``posterior_variance <= betas``.

    Examples
    --------
    >>> sched = noise_schedule([0.1, 0.2, 0.5])
    >>> sched["alpha_bars"]
    array([0.9 , 0.72, 0.36])
    >>> sched["alpha_bars_prev"]
    array([1.  , 0.9 , 0.72])
    >>> np.round(sched["posterior_variance"], 4)
    array([0.    , 0.0714, 0.2187])
    """
    # TODO: np.cumprod, then shift alpha_bars by one step for alpha_bars_prev.
    raise NotImplementedError("noise_schedule() is not implemented yet")


def q_sample(
    x0: ArrayLike,
    t: ArrayLike,
    alpha_bars: ArrayLike,
    noise: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward process in closed form: ``x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps``.

    Each sample of the batch has its own step index ``t[n]``; the coefficients are
    broadcast over the other axes.

    Parameters
    ----------
    x0 : array-like of shape (N, ...)
        Clean data (images scaled to [-1, 1]).
    t : array-like of int, shape (N,)
        Step index of each sample, in ``[0, T)``.
    alpha_bars : array-like of shape (T,)
        Cumulative products ``alpha_bar``.
    noise : array-like with the shape of x0, or None, default=None
        The noise ``eps``; if None, drawn with ``rng.standard_normal(x0.shape)``.
    rng : np.random.Generator or None, default=None
        Random generator; ``np.random.default_rng()`` (not seeded) if None.

    Returns
    -------
    x_t : np.ndarray
        Noisy data, with the shape of ``x0``.
    noise : np.ndarray
        The noise used, with the shape of ``x0`` (the training target of the network).

    Raises
    ------
    ValueError
        If an index of ``t`` is outside ``[0, T)``, or if the shapes do not match.

    Notes
    -----
    Tested against the formula with a given noise (deterministic), and by Monte Carlo:
    mean ``sqrt(alpha_bar_t) x0`` and variance ``1 - alpha_bar_t``.

    Examples
    --------
    >>> alpha_bars = noise_schedule([0.1, 0.2, 0.5])["alpha_bars"]
    >>> x0 = np.array([[1.0, -1.0], [0.5, 0.0]])
    >>> x_t, eps = q_sample(x0, [0, 2], alpha_bars, noise=np.ones((2, 2)))
    >>> np.round(x_t, 4)
    array([[ 1.2649, -0.6325],
           [ 1.1   ,  0.8   ]])
    """
    # TODO: pick alpha_bars[t], reshape to (N, 1, ..., 1), then the formula.
    raise NotImplementedError("q_sample() is not implemented yet")


def q_posterior_mean_variance(
    x0: ArrayLike, x_t: ArrayLike, t: ArrayLike, schedule: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the Gaussian posterior ``q(x_{t-1} | x_t, x0)``.

    ``mean = posterior_mean_coef1[t] x0 + posterior_mean_coef2[t] x_t`` and
    ``variance = posterior_variance[t]`` (Ho et al., eq. 6-7).

    Parameters
    ----------
    x0 : array-like of shape (N, ...)
        Clean data (or its prediction).
    x_t : array-like of shape (N, ...)
        Noisy data at step ``t``.
    t : array-like of int, shape (N,)
        Step index of each sample, in ``[0, T)``.
    schedule : mapping of str to np.ndarray
        Output of ``noise_schedule``.

    Returns
    -------
    mean : np.ndarray
        Posterior mean, with the shape of ``x0``.
    variance : np.ndarray of shape (N, 1, ..., 1)
        Posterior variance of each sample, with as many axes as ``x0`` so that it
        broadcasts over the other axes (shape (N,) when ``x0`` is 1-D).

    Raises
    ------
    ValueError
        If an index of ``t`` is outside ``[0, T)``, or if the shapes do not match.

    Notes
    -----
    Tested against the closed form, and by a 1-D numerical Bayes check (product of the
    two Gaussian densities normalised on a fine grid).

    Examples
    --------
    >>> sched = noise_schedule([0.1, 0.2, 0.5])
    >>> x0 = np.array([[1.0, -1.0], [0.5, 0.0]])
    >>> x_t = np.array([[1.2649, -0.6325], [1.1, 0.8]])
    >>> mean, var = q_posterior_mean_variance(x0, x_t, [0, 2], sched)
    >>> np.round(mean, 4)  # at t = 0 the posterior mean is x0
    array([[ 1.    , -1.    ],
           [ 0.6718,  0.2475]])
    >>> np.round(var, 4)
    array([[0.    ],
           [0.2187]])
    """
    # TODO: pick the coefficients at t, reshape them like in q_sample, combine.
    raise NotImplementedError("q_posterior_mean_variance() is not implemented yet")


def predict_x0_from_eps(
    x_t: ArrayLike, t: ArrayLike, eps: ArrayLike, alpha_bars: ArrayLike
) -> np.ndarray:
    """Invert the forward formula to estimate ``x0`` from ``x_t`` and a noise.

    ``x0_hat = (x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t)``.

    Parameters
    ----------
    x_t : array-like of shape (N, ...)
        Noisy data.
    t : array-like of int, shape (N,)
        Step index of each sample, in ``[0, T)``.
    eps : array-like with the shape of x_t
        Predicted (or true) noise.
    alpha_bars : array-like of shape (T,)
        Cumulative products ``alpha_bar``.

    Returns
    -------
    np.ndarray
        Estimate of ``x0``, with the shape of ``x_t``.

    Raises
    ------
    ValueError
        If an index of ``t`` is outside ``[0, T)``, or if the shapes do not match.

    Notes
    -----
    Tested with the round-trip property
    ``predict_x0_from_eps(q_sample(x0, t, ab, noise)[0], t, noise, ab) == x0``.

    Examples
    --------
    >>> alpha_bars = noise_schedule([0.1, 0.2, 0.5])["alpha_bars"]
    >>> x0 = np.array([[1.0, -1.0], [0.5, 0.0]])
    >>> eps = np.array([[0.3, -2.0], [1.0, 0.5]])
    >>> x_t, _ = q_sample(x0, [1, 2], alpha_bars, noise=eps)
    >>> np.allclose(predict_x0_from_eps(x_t, [1, 2], eps, alpha_bars), x0)
    True
    """
    # TODO: same indexing and reshaping as in q_sample.
    raise NotImplementedError("predict_x0_from_eps() is not implemented yet")


def ddpm_step(
    x_t: ArrayLike,
    t: int,
    eps_pred: ArrayLike,
    schedule: Mapping[str, np.ndarray],
    rng: np.random.Generator | None = None,
    variance: str = "posterior",
    clip_x0: float | None = None,
) -> np.ndarray:
    """One ancestral reverse step ``x_t -> x_{t-1}`` from a noise prediction.

    ``x_{t-1} = mean + sqrt(var) z`` with ``z ~ N(0, I)``, and no noise at all at
    ``t = 0``. Without clipping, ``mean = (x_t - beta_t / sqrt(1 - alpha_bar_t) eps_pred)
    / sqrt(alpha_t)`` (Algorithm 2 of Ho et al.). With ``clip_x0``, ``x0_hat`` is first
    predicted from ``eps_pred``, clipped to ``[-clip_x0, clip_x0]``, and the mean is the
    posterior mean of ``q(x_{t-1} | x_t, x0_hat)`` (both give the same mean when nothing
    is clipped).

    Parameters
    ----------
    x_t : array-like of shape (N, ...)
        Current noisy batch.
    t : int
        Current step index (the same for the whole batch), in ``[0, T)``.
    eps_pred : array-like with the shape of x_t
        Noise predicted by the network.
    schedule : mapping of str to np.ndarray
        Output of ``noise_schedule``.
    rng : np.random.Generator or None, default=None
        Random generator for ``z``; ``np.random.default_rng()`` if None.
    variance : {'posterior', 'beta'}, default='posterior'
        ``var`` = ``posterior_variance[t]`` (beta tilde) or ``betas[t]``.
    clip_x0 : float or None, default=None
        If set, clip ``x0_hat`` to ``[-clip_x0, clip_x0]`` (1.0 for images in [-1, 1]).

    Returns
    -------
    np.ndarray
        ``x_{t-1}``, with the shape of ``x_t``.

    Raises
    ------
    ValueError
        If ``t`` is outside ``[0, T)`` or ``variance`` is unknown.

    Notes
    -----
    Tested: the mean (seen at ``t = 0``, where no noise is added) equals Algorithm 2 of
    Ho et al. when ``clip_x0`` is None; ``diffusers.DDPMScheduler.step`` if diffusers is
    installed.

    Examples
    --------
    >>> sched = noise_schedule([0.1, 0.2, 0.5])
    >>> x_t = np.array([[0.5, -0.5]])
    >>> np.round(ddpm_step(x_t, 0, np.array([[1.0, 1.0]]), sched), 4)  # t = 0: no noise
    array([[ 0.1937, -0.8604]])
    """
    # TODO: compute the mean (with or without clipping), then add noise if t > 0.
    raise NotImplementedError("ddpm_step() is not implemented yet")


def ddim_step(
    x_t: ArrayLike,
    t: int,
    t_prev: int,
    eps_pred: ArrayLike,
    alpha_bars: ArrayLike,
    eta: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One DDIM step from index ``t`` to an earlier index ``t_prev`` (Song et al., 2021).

    With ``ab = alpha_bars[t]`` and ``ab_prev = alpha_bars[t_prev]`` (``ab_prev = 1`` when
    ``t_prev = -1``, i.e. the clean image):
    ``x0_hat = (x_t - sqrt(1 - ab) eps_pred) / sqrt(ab)``,
    ``sigma = eta sqrt((1 - ab_prev) / (1 - ab)) sqrt(1 - ab / ab_prev)`` and
    ``x_prev = sqrt(ab_prev) x0_hat + sqrt(1 - ab_prev - sigma**2) eps_pred + sigma z``.
    Deterministic when ``eta = 0``.

    Parameters
    ----------
    x_t : array-like of shape (N, ...)
        Current noisy batch.
    t : int
        Current step index, in ``[0, T)``.
    t_prev : int
        Target index, ``-1 <= t_prev < t``.
    eps_pred : array-like with the shape of x_t
        Noise predicted by the network.
    alpha_bars : array-like of shape (T,)
        Cumulative products ``alpha_bar``.
    eta : float, default=0.0
        Stochasticity in [0, 1]: 0 = DDIM (deterministic), 1 = DDPM-like.
    rng : np.random.Generator or None, default=None
        Random generator for ``z``, used only if ``eta > 0``;
        ``np.random.default_rng()`` if None.

    Returns
    -------
    np.ndarray
        ``x_{t_prev}``, with the shape of ``x_t``.

    Raises
    ------
    ValueError
        If ``-1 <= t_prev < t < T`` does not hold, or ``eta`` is outside [0, 1].

    Notes
    -----
    Tested with properties: with ``eps_pred`` equal to the true noise and ``eta = 0``,
    ``ddim_step(q_sample(x0, t, ...)[0], t, -1, noise, ...) == x0``; with ``eta = 1`` and
    ``t_prev = t - 1``, ``sigma**2`` equals ``posterior_variance[t]``;
    ``diffusers.DDIMScheduler`` if diffusers is installed.

    Examples
    --------
    >>> alpha_bars = noise_schedule(linear_beta_schedule(1000))["alpha_bars"]
    >>> x0 = np.array([[0.5, -1.0, 0.25]])
    >>> eps = np.array([[1.0, 0.0, -2.0]])
    >>> x_t, _ = q_sample(x0, [700], alpha_bars, noise=eps)
    >>> np.allclose(ddim_step(x_t, 700, -1, eps, alpha_bars), x0)  # straight back to x0
    True
    """
    # TODO: predict x0, then re-noise it at the level of t_prev.
    raise NotImplementedError("ddim_step() is not implemented yet")
