"""DDPM and DDIM mathematics — mylearn, chapter B5 (Diffusion models: a minimal DDPM).

The network-independent part of a diffusion model, in NumPy: noise schedules, the
constants derived from them, closed-form noising, the Gaussian posterior, and one reverse
step (DDPM or DDIM). The notebook plugs these functions into a PyTorch U-Net; their tested
properties (exact round trip with the true noise) catch index and square-root mistakes
before any training. Convention: steps are indexed from 0 to T - 1 (step ``t`` of the
paper is index ``t - 1`` here).

Reference implementation.
"""

from __future__ import annotations

import math
import numbers
import operator
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

SCHEDULE_KEYS = (
    "betas", "alphas", "alpha_bars", "alpha_bars_prev", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars",
    "posterior_variance", "posterior_mean_coef1", "posterior_mean_coef2",
)


def _integer(value, name: str) -> int:
    """An integer argument (Python or NumPy integer; bool refused)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got a bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None


def _real(value, name: str) -> float:
    """A real number (bool refused), as a Python float; NaN is refused."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if math.isnan(value):
        raise ValueError(f"{name} must be a number, got NaN")
    return float(value)


def _rng(rng) -> np.random.Generator:
    """The generator itself, or ``np.random.default_rng(rng)`` for None or a seed (BIBLE §22, Q9)."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _vector(values, name: str) -> np.ndarray:
    """A new 1-D float64 array with at least one value."""
    try:
        arr = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a 1-D numeric array") from None
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {arr.shape}")
    return arr


def _batch(values, name: str) -> np.ndarray:
    """A new float64 array of shape (N, ...)."""
    try:
        arr = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric array") from None
    if arr.ndim == 0:
        raise ValueError(f"{name} must have a batch axis (shape (N, ...)), got a scalar")
    return arr


def _same_shape(arr: np.ndarray, other, name: str, ref_name: str) -> np.ndarray:
    out = _batch(other, name)
    if out.shape != arr.shape:
        raise ValueError(f"{name} must have the shape of {ref_name} {arr.shape}, got {out.shape}")
    return out


def _indices(t, n_samples: int, n_steps: int) -> np.ndarray:
    """Step indices of shape (N,), integers in [0, T) (a scalar is used for every sample)."""
    arr = np.asarray(t)
    if arr.dtype == bool:
        raise TypeError("t must hold integer step indices, got booleans")
    if arr.dtype.kind == "f" and np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)):
        arr = arr.astype(np.int64)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"t must hold integer step indices, got dtype {arr.dtype}")
    if arr.ndim == 0:
        arr = np.full(n_samples, int(arr), dtype=np.int64)
    if arr.shape != (n_samples,):
        raise ValueError(f"t must have shape ({n_samples},), one index per sample, got {arr.shape}")
    if arr.size and (arr.min() < 0 or arr.max() >= n_steps):
        raise ValueError(f"every index of t must be in [0, {n_steps}), got values from {arr.min()} to {arr.max()}")
    return arr.astype(np.int64)


def _per_sample(values: np.ndarray, t: np.ndarray, ndim: int) -> np.ndarray:
    """values[t] reshaped to (N, 1, ..., 1) so that it broadcasts over the other axes."""
    return values[t].reshape((-1,) + (1,) * (ndim - 1))


def _schedule(schedule: Mapping[str, np.ndarray], keys) -> dict[str, np.ndarray]:
    absent = [key for key in keys if key not in schedule]
    if absent:
        raise ValueError(f"schedule is missing the key(s) {', '.join(absent)}: pass the output of noise_schedule")
    out = {key: _vector(schedule[key], f"schedule[{key!r}]") for key in keys}
    sizes = {arr.size for arr in out.values()}
    if len(sizes) != 1:
        raise ValueError("every array of schedule must have the same length T")
    return out


def _step(t, n_steps: int) -> int:
    t = _integer(t, "t")
    if not 0 <= t < n_steps:
        raise ValueError(f"t must be in [0, {n_steps}), got {t}")
    return t


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
    n = _integer(n_steps, "n_steps")
    if n < 1:
        raise ValueError(f"n_steps must be >= 1, got {n}")
    start, end = _real(beta_start, "beta_start"), _real(beta_end, "beta_end")
    if not 0 < start <= end < 1:
        raise ValueError(f"expected 0 < beta_start <= beta_end < 1, got beta_start={start!r}, beta_end={end!r}")
    return np.linspace(start, end, n, dtype=np.float64)


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
    n = _integer(n_steps, "n_steps")
    if n < 1:
        raise ValueError(f"n_steps must be >= 1, got {n}")
    s = _real(s, "s")
    if s < 0:
        raise ValueError(f"s must be >= 0, got {s!r}")
    max_beta = _real(max_beta, "max_beta")
    if not 0 < max_beta < 1:
        raise ValueError(f"max_beta must be in (0, 1), got {max_beta!r}")
    steps = np.arange(n + 1, dtype=np.float64) / n
    f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    return np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], max_beta)


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
    b = _vector(betas, "betas")
    if np.isnan(b).any() or not np.all((b > 0) & (b < 1)):
        raise ValueError("every beta must be in (0, 1)")
    alphas = 1.0 - b
    alpha_bars = np.cumprod(alphas)
    alpha_bars_prev = np.concatenate(([1.0], alpha_bars[:-1]))
    one_minus = 1.0 - alpha_bars
    return {
        "betas": b,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "sqrt_alpha_bars": np.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": np.sqrt(one_minus),
        "posterior_variance": b * (1.0 - alpha_bars_prev) / one_minus,
        "posterior_mean_coef1": b * np.sqrt(alpha_bars_prev) / one_minus,
        "posterior_mean_coef2": (1.0 - alpha_bars_prev) * np.sqrt(alphas) / one_minus,
    }


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
    x0 = _batch(x0, "x0")
    ab = _vector(alpha_bars, "alpha_bars")
    t = _indices(t, x0.shape[0], ab.size)
    if noise is None:
        noise = _rng(rng).standard_normal(x0.shape)
    else:
        noise = _same_shape(x0, noise, "noise", "x0")
    coef = _per_sample(ab, t, x0.ndim)
    return np.sqrt(coef) * x0 + np.sqrt(1.0 - coef) * noise, noise


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
    x0 = _batch(x0, "x0")
    x_t = _same_shape(x0, x_t, "x_t", "x0")
    sched = _schedule(schedule, ("posterior_mean_coef1", "posterior_mean_coef2", "posterior_variance"))
    t = _indices(t, x0.shape[0], sched["posterior_variance"].size)
    mean = (_per_sample(sched["posterior_mean_coef1"], t, x0.ndim) * x0
            + _per_sample(sched["posterior_mean_coef2"], t, x0.ndim) * x_t)
    return mean, _per_sample(sched["posterior_variance"], t, x0.ndim).copy()


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
    x_t = _batch(x_t, "x_t")
    eps = _same_shape(x_t, eps, "eps", "x_t")
    ab = _vector(alpha_bars, "alpha_bars")
    t = _indices(t, x_t.shape[0], ab.size)
    coef = _per_sample(ab, t, x_t.ndim)
    return (x_t - np.sqrt(1.0 - coef) * eps) / np.sqrt(coef)


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
    sched = _schedule(schedule, ("betas", "alphas", "alpha_bars", "posterior_variance",
                                 "posterior_mean_coef1", "posterior_mean_coef2"))
    t = _step(t, sched["betas"].size)
    if variance not in ("posterior", "beta"):
        raise ValueError(f"variance must be 'posterior' or 'beta', got {variance!r}")
    x_t = _batch(x_t, "x_t")
    eps = _same_shape(x_t, eps_pred, "eps_pred", "x_t")
    beta, alpha, ab = sched["betas"][t], sched["alphas"][t], sched["alpha_bars"][t]
    if clip_x0 is None:
        mean = (x_t - beta / np.sqrt(1.0 - ab) * eps) / np.sqrt(alpha)
    else:
        bound = _real(clip_x0, "clip_x0")
        if bound <= 0:
            raise ValueError(f"clip_x0 must be > 0, got {bound!r}")
        x0_hat = np.clip((x_t - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab), -bound, bound)
        mean = sched["posterior_mean_coef1"][t] * x0_hat + sched["posterior_mean_coef2"][t] * x_t
    if t == 0:
        return mean
    var = sched["posterior_variance"][t] if variance == "posterior" else beta
    return mean + np.sqrt(var) * _rng(rng).standard_normal(x_t.shape)


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
    ab_all = _vector(alpha_bars, "alpha_bars")
    t, t_prev = _integer(t, "t"), _integer(t_prev, "t_prev")
    if not -1 <= t_prev < t < ab_all.size:
        raise ValueError(f"expected -1 <= t_prev < t < {ab_all.size}, got t={t}, t_prev={t_prev}")
    eta = _real(eta, "eta")
    if not 0 <= eta <= 1:
        raise ValueError(f"eta must be in [0, 1], got {eta!r}")
    x_t = _batch(x_t, "x_t")
    eps = _same_shape(x_t, eps_pred, "eps_pred", "x_t")
    ab = ab_all[t]
    ab_prev = 1.0 if t_prev == -1 else ab_all[t_prev]
    x0_hat = (x_t - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab)
    sigma = eta * np.sqrt((1.0 - ab_prev) / (1.0 - ab)) * np.sqrt(1.0 - ab / ab_prev)
    x_prev = np.sqrt(ab_prev) * x0_hat + np.sqrt(max(1.0 - ab_prev - sigma**2, 0.0)) * eps
    if eta > 0:
        x_prev = x_prev + sigma * _rng(rng).standard_normal(x_t.shape)
    return x_prev
