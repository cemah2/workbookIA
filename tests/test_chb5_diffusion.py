"""Tests of mylearn.diffusion (bonus chapter B5): oracle tests and properties.

    pytest tests/test_chb5_diffusion.py              # your code (mon_travail/mylearn/diffusion.py)
    pytest tests/test_chb5_diffusion.py --impl=ref   # the reference

Oracles: ``np.linspace`` for the linear schedule; the closed form of the cosine schedule evaluated in float64 with
``math.cos`` and explicit loops; the formulas of ``noise_schedule`` written with explicit loops (checked against
``np.cumprod``); the forward formula with a given noise, sample by sample, and a Monte Carlo check of its mean and
variance; the Gaussian posterior against a numerical Bayes computation in 1-D (the product of the densities of
``q(x_{t-1} | x0)`` and ``q(x_t | x_{t-1})`` normalised on a fine grid); the mean of a DDPM step against
Algorithm 2 of Ho et al. (2020); the DDIM step against the formulas of its docstring and its properties (exact
return to ``x0`` with the true noise and ``eta = 0``, ``sigma**2 = posterior_variance[t]`` with ``eta = 1``).
When diffusers is installed, the same quantities are also compared with ``DDPMScheduler`` and ``DDIMScheduler``
(``prediction_type='epsilon'``, ``thresholding=False``, ``clip_sample`` set like ``clip_x0``,
``variance_type='fixed_small'`` for ``variance='posterior'`` and ``'fixed_large'`` for ``'beta'``,
``set_alpha_to_one=True`` for ``t_prev = -1``; their tensors are switched to float64); without diffusers, those
tests are skipped and the closed forms remain.
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.

How the random part of a reverse step is seen: two calls with the same seed and batches of the same shape draw the
same ``z`` (any way of drawing ``z`` from ``rng`` does that), so ``step(x_t, eps) - step(0, 0)`` is the mean of the
step (exact), and ``step(0, 0)`` alone is the noise ``sqrt(var) z``, whose variance is checked by Monte Carlo.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_linear_beta_schedule_ or test_cosine_beta_schedule_ or test_noise_schedule_"`` (B5.12),
``-k "test_q_sample_"`` (B5.14),
``-k "test_q_posterior_mean_variance_ or test_predict_x0_from_eps_"`` (B5.16),
``-k "test_ddpm_step_ or test_ddim_step_"`` (B5.17).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified;
a NaN does not satisfy a range condition of the docstring (``beta_start``, ``max_beta``, ``eta``, the betas of
``noise_schedule``), so it raises ValueError; with ``n_steps = 1``, ``linear_beta_schedule`` returns
``[beta_start]`` (as ``np.linspace``; ``[beta_end]`` is also accepted); an index ``t`` equal to -1 or to ``T``
raises ValueError (not IndexError, and -1 does not mean the last step); ``t`` must have exactly one index per
sample (a ``t`` of length 1 for 3 samples raises ValueError, even if NumPy could broadcast it), and so must
``noise`` and ``eps`` have exactly the shape of ``x0`` / ``x_t``; ``ddim_step`` does not touch ``rng`` when
``eta = 0``.
The schedules are float64 arrays (docstrings): their values are compared with rtol=1e-9; the other results are
compared with rtol=1e-6, atol=1e-6 (the docstrings fix no tolerance; a computation in float32 passes).
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

from __future__ import annotations

import copy
import math
from unittest import mock

import numpy as np
import pytest

ORACLE_BUG = ("bug in tests/test_chb5_diffusion.py (not in your code): the oracle of this test "
              "disagrees with another one")

SCHEDULE_KEYS = ("betas", "alphas", "alpha_bars", "alpha_bars_prev", "sqrt_alpha_bars",
                 "sqrt_one_minus_alpha_bars", "posterior_variance", "posterior_mean_coef1", "posterior_mean_coef2")


@pytest.fixture
def dif(mylearn_module):
    return mylearn_module("diffusion")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3, max_line_width=10**6)
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


# The docstrings fix no tolerance. Schedules (float64 by contract): rtol=1e-9; everything else: rtol=1e-6,
# atol=1e-6, which lets a computation in float32 pass, not a wrong formula.
SCHED_RTOL, SCHED_ATOL = 1e-9, 1e-15
RTOL, ATOL = 1e-6, 1e-6
# diffusers keeps its schedules in float32 (torch.linspace, torch.tensor(..., dtype=float32))
F32_RTOL, F32_ATOL = 2e-6, 1e-9


def assert_close(result, expected, rtol=RTOL, atol=ATOL, msg=""):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
    if result is None:
        _fail(msg, "expected a number or an array, got None (did you forget the return?)")
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected a number or an array, got an object of type {type(result).__name__}")
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}")
    ok = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=False)
    if not ok.all():
        bad = np.flatnonzero(~ok)
        i = int(bad[0])
        index = np.unravel_index(i, want.shape) if want.ndim > 1 else (i,)
        where = f" at index {tuple(int(k) for k in index)}" if want.ndim else ""
        head = (f"expected {want.ravel()[i] + 0.0:.12g}{where}, got {got.ravel()[i] + 0.0:.12g} "
                f"({bad.size} wrong value(s) out of {want.size}; tolerance rtol={rtol:g}, atol={atol:g})")
        data = f"expected {_short(want, 10)}\n     got {_short(got, 10)}" if want.size <= 16 else ""
        _fail(msg, head, data)


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
    try:
        function(*args, **kwargs)
    except NotImplementedError:          # before `except error`: shown as "⏳"
        raise
    except error:
        return
    except Exception as exc:  # noqa: BLE001 - say which error was raised instead
        raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else "")
                             + f" (it raised {type(exc).__name__}: {exc})") from None
    raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else ""))


def call(function, *args, expected="a result", **kwargs):
    """Call the learner's function; an unexpected exception becomes a failure that says what was expected."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def array_result(result, msg, shape=None, float64=False, inputs="") -> np.ndarray:
    """The learner's result as an ndarray, checked for None, type, shape and (schedules) dtype.

    `inputs` names the shapes of the inputs (e.g. "x0 of shape (5,) and t of shape (5,)") in the shape message.
    """
    if result is None:
        _fail(msg, "expected a NumPy array, got None (did you forget the return?)")
    if not isinstance(result, np.ndarray):
        _fail(msg, f"expected a NumPy array, got {type(result).__name__}")
    if shape is not None and result.shape != tuple(shape):
        _fail(msg, f"expected shape {tuple(shape)}" + (f" for {inputs}" if inputs else "") + f", got shape {result.shape}")
    if float64 and result.dtype != np.float64:
        _fail(msg, f"expected a float64 array (docstring), got dtype {result.dtype}")
    return result


def pair_result(result, msg):
    """The (x_t, noise) pair returned by q_sample / q_posterior_mean_variance (a tuple; a list is accepted)."""
    if result is None:
        _fail(msg, "expected a pair of arrays, got None (did you forget the return?)")
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        what = f"{type(result).__name__}" + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else "")
        _fail(msg, f"expected a pair (tuple of 2 arrays), got {what}")
    return result


def assert_unchanged(before, after, name, function):
    if isinstance(before, dict):
        for key in before:
            assert_unchanged(before[key], after[key], f"{name}[{key!r}]", function)
        return
    if isinstance(before, np.ndarray):
        same = isinstance(after, np.ndarray) and before.shape == after.shape and before.dtype == after.dtype \
            and np.array_equal(before, after, equal_nan=True)
    else:
        same = before == after
    if not same:
        _fail(function, f"expected the input {name} to be left as it was, it was modified "
                        "(work on a copy, e.g. np.array(x, dtype=float), and avoid in-place operations like *=)")


def monte_carlo(samples, mean, var, msg, axis=0):
    """Empirical mean and variance along `axis` within 5 standard errors of `mean` and `var`."""
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[axis]
    m, v = samples.mean(axis=axis), samples.var(axis=axis, ddof=1)
    mean, var = np.broadcast_to(mean, m.shape), np.broadcast_to(var, v.shape)
    tol_m = 5 * np.sqrt(var / n) + 1e-12
    if np.any(np.abs(m - mean) > tol_m):
        i = int(np.argmax(np.abs(m - mean) - tol_m))
        _fail(msg, f"expected an empirical mean of {mean.ravel()[i]:.6g} (± {tol_m.ravel()[i]:.2g}, {n} draws), "
                   f"got {m.ravel()[i]:.6g}")
    tol_v = 5 * var * np.sqrt(2 / (n - 1)) + 1e-12
    if np.any(np.abs(v - var) > tol_v):
        i = int(np.argmax(np.abs(v - var) - tol_v))
        _fail(msg, f"expected an empirical variance of {var.ravel()[i]:.6g} (± {tol_v.ravel()[i]:.2g}, {n} draws), "
                   f"got {v.ravel()[i]:.6g}")


# ------------------------------------------------------------------ oracles
def oracle_linear(n_steps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, n_steps, dtype=np.float64)


def oracle_cosine(n_steps, s=0.008, max_beta=0.999):
    """The closed form of the docstring with math.cos and explicit loops."""
    f = [math.cos(((t / n_steps) + s) / (1 + s) * math.pi / 2) ** 2 for t in range(n_steps + 1)]
    alpha_bar = [v / f[0] for v in f]
    return np.array([min(1 - alpha_bar[t + 1] / alpha_bar[t], max_beta) for t in range(n_steps)])


def oracle_schedule(betas):
    """Every formula of the noise_schedule docstring with explicit loops; checked against np.cumprod."""
    betas = [float(b) for b in betas]
    alpha_bars, prod = [], 1.0
    for b in betas:
        prod *= 1.0 - b
        alpha_bars.append(prod)
    prev = [1.0] + alpha_bars[:-1]
    out = {
        "betas": betas,
        "alphas": [1.0 - b for b in betas],
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": prev,
        "sqrt_alpha_bars": [math.sqrt(a) for a in alpha_bars],
        "sqrt_one_minus_alpha_bars": [math.sqrt(1.0 - a) for a in alpha_bars],
        "posterior_variance": [b * (1.0 - p) / (1.0 - a) for b, p, a in zip(betas, prev, alpha_bars)],
        "posterior_mean_coef1": [b * math.sqrt(p) / (1.0 - a) for b, p, a in zip(betas, prev, alpha_bars)],
        "posterior_mean_coef2": [(1.0 - p) * math.sqrt(1.0 - b) / (1.0 - a)
                                 for b, p, a in zip(betas, prev, alpha_bars)],
    }
    out = {key: np.array(value, dtype=np.float64) for key, value in out.items()}
    assert np.allclose(out["alpha_bars"], np.cumprod(1.0 - np.array(betas)), rtol=1e-12, atol=0), ORACLE_BUG
    return out


def oracle_q_sample(x0, t, alpha_bars, noise):
    """Sample by sample: x_t[n] = sqrt(ab[t[n]]) x0[n] + sqrt(1 - ab[t[n]]) noise[n]."""
    x0, noise = np.asarray(x0, float), np.asarray(noise, float)
    out = np.empty_like(x0)
    for n, tn in enumerate(np.asarray(t).reshape(-1)):
        ab = float(alpha_bars[int(tn)])
        out[n] = math.sqrt(ab) * x0[n] + math.sqrt(1.0 - ab) * noise[n]
    return out


def oracle_predict_x0(x_t, t, eps, alpha_bars):
    x_t, eps = np.asarray(x_t, float), np.asarray(eps, float)
    out = np.empty_like(x_t)
    for n, tn in enumerate(np.asarray(t).reshape(-1)):
        ab = float(alpha_bars[int(tn)])
        out[n] = (x_t[n] - math.sqrt(1.0 - ab) * eps[n]) / math.sqrt(ab)
    return out


def oracle_posterior(x0, x_t, t, sched):
    x0, x_t = np.asarray(x0, float), np.asarray(x_t, float)
    mean = np.empty_like(x0)
    var = np.empty((x0.shape[0],) + (1,) * (x0.ndim - 1))
    for n, tn in enumerate(np.asarray(t).reshape(-1)):
        mean[n] = sched["posterior_mean_coef1"][tn] * x0[n] + sched["posterior_mean_coef2"][tn] * x_t[n]
        var[n] = sched["posterior_variance"][tn]
    return mean, var


def bayes_posterior_1d(x0, x_t, t, betas, n_grid=400_001):
    """Mean and variance of q(x_{t-1} | x_t, x0) by a numerical Bayes rule on a fine grid (t >= 1).

    Prior q(x_{t-1} | x0) = N(sqrt(ab_{t-1}) x0, 1 - ab_{t-1}); likelihood q(x_t | x_{t-1}) = N(sqrt(alpha_t)
    x_{t-1}, beta_t); posterior = product of the two densities, normalised on the grid.
    """
    alphas = 1.0 - np.asarray(betas, float)
    ab_prev = float(np.prod(alphas[:t]))
    m0, v0 = math.sqrt(ab_prev) * x0, 1.0 - ab_prev
    a, b = math.sqrt(alphas[t]), float(betas[t])
    center, width = m0, math.sqrt(v0)
    grid = np.linspace(center - 30 * width, center + 30 * width, n_grid)
    log_p = -(grid - m0) ** 2 / (2 * v0) - (x_t - a * grid) ** 2 / (2 * b)
    w = np.exp(log_p - log_p.max())
    w /= w.sum()
    mean = float(np.sum(w * grid))
    return mean, float(np.sum(w * (grid - mean) ** 2))


def algo2_mean(x_t, t, eps, sched):
    """Algorithm 2 of Ho et al.: (x_t - beta_t / sqrt(1 - ab_t) eps) / sqrt(alpha_t)."""
    return (np.asarray(x_t, float) - sched["betas"][t] / math.sqrt(1.0 - sched["alpha_bars"][t])
            * np.asarray(eps, float)) / math.sqrt(sched["alphas"][t])


def clipped_mean(x_t, t, eps, sched, clip):
    """Posterior mean of q(x_{t-1} | x_t, x0_hat) with x0_hat predicted from eps and clipped."""
    x_t, eps = np.asarray(x_t, float), np.asarray(eps, float)
    ab = sched["alpha_bars"][t]
    x0_hat = np.clip((x_t - math.sqrt(1.0 - ab) * eps) / math.sqrt(ab), -clip, clip)
    return sched["posterior_mean_coef1"][t] * x0_hat + sched["posterior_mean_coef2"][t] * x_t


def ddim_parts(x_t, t, t_prev, eps, alpha_bars, eta):
    """Deterministic part of a DDIM step and sigma, from the formulas of the docstring."""
    x_t, eps = np.asarray(x_t, float), np.asarray(eps, float)
    ab = float(alpha_bars[t])
    ab_prev = 1.0 if t_prev == -1 else float(alpha_bars[t_prev])
    x0_hat = (x_t - math.sqrt(1.0 - ab) * eps) / math.sqrt(ab)
    sigma = eta * math.sqrt((1.0 - ab_prev) / (1.0 - ab)) * math.sqrt(1.0 - ab / ab_prev)
    det = math.sqrt(ab_prev) * x0_hat + math.sqrt(max(1.0 - ab_prev - sigma**2, 0.0)) * eps
    return det, sigma


SMALL = [0.1, 0.2, 0.5]
LINEAR_1000 = oracle_linear(1000)
AB_1000 = np.cumprod(1.0 - LINEAR_1000)      # its alpha_bars


def random_betas(seed, n_steps):
    return np.random.default_rng(seed).uniform(0.01, 0.6, n_steps)


# ------------------------------------------------------------------ diffusers (optional oracle)
@pytest.fixture(scope="module")
def hf():
    """diffusers and torch, or a skip: diffusers is an optional oracle (the closed forms are always tested)."""
    try:
        import torch
        from diffusers import DDIMScheduler, DDPMScheduler
        import diffusers.schedulers.scheduling_ddpm as scheduling_ddpm
    except Exception as exc:  # noqa: BLE001 - ImportError, or a broken installation
        pytest.skip(f"diffusers is not installed (optional oracle; the closed forms are tested anyway): {exc}")
    return {"torch": torch, "DDPM": DDPMScheduler, "DDIM": DDIMScheduler, "ddpm_module": scheduling_ddpm}


def ddpm_float64(hf, betas, variance_type="fixed_small", clip=None):
    """DDPMScheduler with the given betas, its tensors switched to float64."""
    torch = hf["torch"]
    sched = hf["DDPM"](num_train_timesteps=len(betas), trained_betas=np.asarray(betas, float),
                       variance_type=variance_type, prediction_type="epsilon", thresholding=False,
                       clip_sample=clip is not None, clip_sample_range=clip if clip is not None else 1.0)
    b = torch.tensor(np.asarray(betas, float), dtype=torch.float64)
    sched.betas, sched.alphas = b, 1.0 - b
    sched.alphas_cumprod = torch.cumprod(sched.alphas, dim=0)
    sched.one = torch.tensor(1.0, dtype=torch.float64)
    return sched


def ddim_float64(hf, alpha_bars, num_inference_steps):
    """DDIMScheduler (set_alpha_to_one=True: ab = 1 after the last step) with float64 alpha_bars."""
    torch = hf["torch"]
    alpha_bars = np.asarray(alpha_bars, float)
    betas = 1.0 - alpha_bars / np.concatenate(([1.0], alpha_bars[:-1]))
    sched = hf["DDIM"](num_train_timesteps=alpha_bars.size, trained_betas=betas, clip_sample=False,
                       set_alpha_to_one=True, prediction_type="epsilon", thresholding=False)
    sched.alphas_cumprod = torch.tensor(alpha_bars, dtype=torch.float64)
    sched.final_alpha_cumprod = torch.tensor(1.0, dtype=torch.float64)
    sched.set_timesteps(num_inference_steps)
    return sched


# ================================================================== linear_beta_schedule (B5.12)
def linear(dif, *args, **kwargs):
    return call(dif.linear_beta_schedule, *args, expected="an array of betas", **kwargs)


def test_linear_beta_schedule_docstring_example(dif):
    got = array_result(linear(dif, 5), "linear_beta_schedule(5)", (5,), float64=True, inputs="n_steps = 5")
    assert np.allclose(oracle_linear(5), [0.0001, 0.005075, 0.01005, 0.015025, 0.02], rtol=0, atol=1e-12), ORACLE_BUG
    assert_close(got, oracle_linear(5), SCHED_RTOL, SCHED_ATOL, "linear_beta_schedule(5) (as np.linspace)")


@pytest.mark.parametrize("args", [
    pytest.param((1000,), id="defaults-T1000"),
    pytest.param((10, 1e-3, 0.05), id="T10"),
    pytest.param((2, 0.1, 0.2), id="T2-the-two-ends"),
    pytest.param((7, 0.3, 0.3), id="beta-start-equals-beta-end"),
    pytest.param((400, 1e-4, 0.02), id="T400-notebook"),
])
def test_linear_beta_schedule_matches_linspace(dif, args):
    msg = f"linear_beta_schedule{args}"
    got = array_result(linear(dif, *args), msg, (args[0],), float64=True, inputs=f"n_steps = {args[0]}")
    want = oracle_linear(*args)
    assert_close(got[0], want[0], SCHED_RTOL, SCHED_ATOL, msg + " (index 0 is beta_start)")
    assert_close(got[-1], want[-1], SCHED_RTOL, SCHED_ATOL, msg + " (index T - 1 is beta_end)")
    assert_close(got, want, SCHED_RTOL, SCHED_ATOL, msg + " (as np.linspace)")


def test_linear_beta_schedule_one_step(dif):
    got = array_result(linear(dif, 1, 0.01, 0.02), "linear_beta_schedule(1, 0.01, 0.02)", (1,), float64=True,
                       inputs="n_steps = 1")
    if not (np.isclose(got[0], 0.01, rtol=SCHED_RTOL) or np.isclose(got[0], 0.02, rtol=SCHED_RTOL)):
        _fail("linear_beta_schedule(1, 0.01, 0.02)",
              f"expected [0.01] (np.linspace(0.01, 0.02, 1); [0.02] is accepted too), got {_short(got)}"
              + (" (a division by T - 1 = 0?)" if np.isnan(got).any() else ""))


@pytest.mark.parametrize("args,why", [
    pytest.param((0,), "n_steps = 0", id="n-steps-0"),
    pytest.param((-5,), "n_steps = -5", id="n-steps-negative"),
    pytest.param((10, 0.0, 0.02), "beta_start = 0 (0 < beta_start is required)", id="beta-start-0"),
    pytest.param((10, -1e-4, 0.02), "beta_start = -1e-4", id="beta-start-negative"),
    pytest.param((10, 0.03, 0.02), "beta_start = 0.03 > beta_end = 0.02", id="start-above-end"),
    pytest.param((10, 1e-4, 1.0), "beta_end = 1 (beta_end < 1 is required)", id="beta-end-1"),
    pytest.param((10, 1e-4, 1.5), "beta_end = 1.5", id="beta-end-above-1"),
    pytest.param((10, float("nan"), 0.02), "beta_start = nan (NaN is outside (0, 1): comparisons with NaN are False, so "
                                           "0 < beta_start <= beta_end < 1 does not hold)", id="beta-start-nan"),
])
def test_linear_beta_schedule_refuses_invalid_arguments(dif, args, why):
    assert_raises(ValueError, dif.linear_beta_schedule, *args, why=why)


@pytest.mark.parametrize("args", [pytest.param((1000, 1e-4, 0.02), id="T1000"),
                                  pytest.param((10, 1e-3, 0.05), id="T10")])
def test_linear_beta_schedule_matches_diffusers(dif, hf, args):
    sched = hf["DDPM"](num_train_timesteps=args[0], beta_start=args[1], beta_end=args[2], beta_schedule="linear")
    ref = sched.betas.numpy().astype(np.float64)
    assert np.allclose(ref, oracle_linear(*args), rtol=F32_RTOL, atol=F32_ATOL), ORACLE_BUG
    got = array_result(linear(dif, *args), f"linear_beta_schedule{args}", (args[0],), float64=True,
                       inputs=f"n_steps = {args[0]}")
    assert_close(got, ref, F32_RTOL, F32_ATOL, f"linear_beta_schedule{args} against "
                                               "DDPMScheduler(beta_schedule='linear').betas (float32)")


# ================================================================== cosine_beta_schedule (B5.12)
def cosine(dif, *args, **kwargs):
    return call(dif.cosine_beta_schedule, *args, expected="an array of betas", **kwargs)


def test_cosine_beta_schedule_docstring_example(dif):
    assert np.array_equal(np.round(oracle_cosine(5), 4), [0.1013, 0.2795, 0.4736, 0.7241, 0.999]), ORACLE_BUG
    got = array_result(cosine(dif, 5), "cosine_beta_schedule(5)", (5,), float64=True, inputs="n_steps = 5")
    assert_close(got, oracle_cosine(5), SCHED_RTOL, SCHED_ATOL, "cosine_beta_schedule(5) (closed form; the last "
                                                                "beta is clipped to max_beta = 0.999)")


@pytest.mark.parametrize("args", [
    pytest.param((1000,), id="defaults-T1000"),
    pytest.param((50, 0.008, 0.999), id="T50"),
    pytest.param((10, 0.0, 0.999), id="s-0"),
    pytest.param((12, 0.1, 0.999), id="s-0.1"),
    pytest.param((20, 0.008, 0.3), id="max-beta-0.3-clips-several"),
    pytest.param((1,), id="T1"),
    pytest.param((4000,), id="T4000"),
])
def test_cosine_beta_schedule_matches_closed_form(dif, args):
    msg = f"cosine_beta_schedule{args}"
    got = array_result(cosine(dif, *args), msg, (args[0],), float64=True, inputs=f"n_steps = {args[0]}")
    assert_close(got, oracle_cosine(*args), SCHED_RTOL, SCHED_ATOL,
                 msg + " (f evaluated at t / T for t = 0 .. T, beta_t = min(1 - ab(t + 1) / ab(t), max_beta))")


@pytest.mark.parametrize("args", [pytest.param((5,), id="T5"), pytest.param((100,), id="T100"),
                                  pytest.param((30, 0.02, 0.5), id="T30-max-beta-0.5")])
def test_cosine_beta_schedule_properties(dif, args):
    msg = f"cosine_beta_schedule{args}"
    got = array_result(cosine(dif, *args), msg, (args[0],), float64=True, inputs=f"n_steps = {args[0]}")
    max_beta = args[2] if len(args) > 2 else 0.999
    if not (np.all(got > 0) and np.all(got <= max_beta)):
        _fail(msg, f"expected every beta in (0, {max_beta}], got values from {got.min():.6g} to {got.max():.6g}")
    assert_close(got[-1], max_beta, SCHED_RTOL, SCHED_ATOL, msg + " (the last beta reaches max_beta: alpha_bar(T) "
                                                                   "is 0 up to rounding)")
    # the product of the alphas telescopes back to f(t + 1) / f(0) wherever nothing is clipped
    s = args[1] if len(args) > 1 else 0.008
    T = args[0]
    f = np.array([math.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2 for t in range(T + 1)])
    free = np.flatnonzero(np.cumsum(got >= max_beta) == 0)
    assert_close(np.cumprod(1 - got)[free], (f[1:] / f[0])[free], 1e-9, 1e-15,
                 msg + " (the cumulative product of 1 - beta must give back alpha_bar(t + 1) = f(t + 1) / f(0))")


@pytest.mark.parametrize("args,why", [
    pytest.param((0,), "n_steps = 0", id="n-steps-0"),
    pytest.param((-3,), "n_steps = -3", id="n-steps-negative"),
    pytest.param((10, -0.01), "s = -0.01 < 0", id="s-negative"),
    pytest.param((10, 0.008, 0.0), "max_beta = 0, not in (0, 1)", id="max-beta-0"),
    pytest.param((10, 0.008, 1.0), "max_beta = 1, not in (0, 1)", id="max-beta-1"),
    pytest.param((10, 0.008, 1.5), "max_beta = 1.5", id="max-beta-above-1"),
    pytest.param((10, 0.008, -0.2), "max_beta = -0.2", id="max-beta-negative"),
    pytest.param((10, 0.008, float("nan")), "max_beta = nan (NaN is outside (0, 1): comparisons with NaN are False)", id="max-beta-nan"),
])
def test_cosine_beta_schedule_refuses_invalid_arguments(dif, args, why):
    assert_raises(ValueError, dif.cosine_beta_schedule, *args, why=why)


@pytest.mark.parametrize("n_steps", [pytest.param(10, id="T10"), pytest.param(1000, id="T1000")])
def test_cosine_beta_schedule_matches_diffusers(dif, hf, n_steps):
    ref = hf["DDPM"](num_train_timesteps=n_steps, beta_schedule="squaredcos_cap_v2").betas.numpy().astype(float)
    assert np.allclose(ref, oracle_cosine(n_steps), rtol=F32_RTOL, atol=F32_ATOL), ORACLE_BUG
    got = array_result(cosine(dif, n_steps), f"cosine_beta_schedule({n_steps})", (n_steps,), float64=True,
                       inputs=f"n_steps = {n_steps}")
    assert_close(got, ref, F32_RTOL, F32_ATOL, f"cosine_beta_schedule({n_steps}) against "
                                               "DDPMScheduler(beta_schedule='squaredcos_cap_v2').betas (float32)")


# ================================================================== noise_schedule (B5.12)
def schedule_of(dif, betas):
    """The learner's schedule, checked: a dict with the nine keys, float64 arrays of shape (T,)."""
    msg = "noise_schedule"
    result = call(dif.noise_schedule, copy.deepcopy(betas), expected="a dict of arrays")
    if result is None:
        _fail(msg, "expected a dict of arrays, got None (did you forget the return?)")
    if not hasattr(result, "keys"):
        _fail(msg, f"expected a dict of arrays, got {type(result).__name__}")
    missing = [key for key in SCHEDULE_KEYS if key not in result]
    if missing:
        _fail(msg, f"expected the keys {', '.join(SCHEDULE_KEYS)}; missing: {', '.join(missing)}")
    T = len(betas)
    for key in SCHEDULE_KEYS:
        array_result(result[key], f"noise_schedule(...)[{key!r}]", (T,), float64=True, inputs=f"{T} betas")
    return result


SCHEDULE_CASES = [
    pytest.param(SMALL, id="docstring-0.1-0.2-0.5"),
    pytest.param(list(random_betas(1, 8)), id="random-T8"),
    pytest.param(list(LINEAR_1000), id="linear-T1000"),
    pytest.param(list(oracle_cosine(100)), id="cosine-T100"),
    pytest.param([0.3], id="T1"),
]


def test_noise_schedule_docstring_example(dif):
    got = schedule_of(dif, SMALL)
    assert_close(got["alpha_bars"], [0.9, 0.72, 0.36], SCHED_RTOL, SCHED_ATOL, "noise_schedule([0.1, 0.2, 0.5])"
                                                                                "['alpha_bars']")
    assert_close(got["alpha_bars_prev"], [1.0, 0.9, 0.72], SCHED_RTOL, SCHED_ATOL,
                 "noise_schedule([0.1, 0.2, 0.5])['alpha_bars_prev'] (shifted by one step, 1 at t = 0)")
    assert_close(got["posterior_variance"], [0.0, 0.2 * 0.1 / 0.28, 0.5 * 0.28 / 0.64], SCHED_RTOL, SCHED_ATOL,
                 "noise_schedule([0.1, 0.2, 0.5])['posterior_variance']")


@pytest.mark.parametrize("betas", SCHEDULE_CASES)
def test_noise_schedule_matches_formulas(dif, betas):
    got = schedule_of(dif, betas)
    want = oracle_schedule(betas)
    for key in SCHEDULE_KEYS:
        assert_close(got[key], want[key], SCHED_RTOL, SCHED_ATOL,
                     f"noise_schedule of {len(betas)} betas, key {key!r} (formula of the docstring)")


@pytest.mark.parametrize("betas", SCHEDULE_CASES)
def test_noise_schedule_properties(dif, betas):
    got = schedule_of(dif, betas)
    msg = f"noise_schedule of {len(betas)} betas"
    ab = got["alpha_bars"]
    if not (np.all(ab > 0) and np.all(ab < 1)):
        _fail(msg, f"expected alpha_bars in (0, 1), got values from {ab.min():.6g} to {ab.max():.6g}")
    if np.any(np.diff(ab) >= 0):
        i = int(np.flatnonzero(np.diff(ab) >= 0)[0])
        _fail(msg, f"expected alpha_bars strictly decreasing, got alpha_bars[{i}] = {ab[i]:.6g} <= "
                   f"alpha_bars[{i + 1}] = {ab[i + 1]:.6g} (a product of the alphas, not a sum)")
    if got["posterior_variance"][0] != 0:
        _fail(msg, f"expected posterior_variance[0] == 0 exactly (alpha_bars_prev[0] = 1), "
                   f"got {got['posterior_variance'][0]!r}")
    if got["alpha_bars_prev"][0] != 1:
        _fail(msg, f"expected alpha_bars_prev[0] == 1, got {got['alpha_bars_prev'][0]!r}")
    bad = np.flatnonzero(got["posterior_variance"] > np.asarray(betas) * (1 + 1e-12))
    if bad.size:
        i = int(bad[0])
        _fail(msg, f"expected posterior_variance <= betas, got posterior_variance[{i}] = "
                   f"{got['posterior_variance'][i]:.6g} > beta = {betas[i]:.6g}")


@pytest.mark.parametrize("kind", ["tuple", "float32-array", "list"])
def test_noise_schedule_accepts_array_likes(dif, kind):
    base = [0.05, 0.25, 0.125, 0.5]
    betas = {"tuple": tuple(base), "float32-array": np.array(base, dtype=np.float32),
             "list": list(base)}[kind]
    got = schedule_of(dif, betas)
    want = oracle_schedule(np.asarray(betas, dtype=np.float64))
    for key in SCHEDULE_KEYS:
        assert_close(got[key], want[key], SCHED_RTOL, SCHED_ATOL,
                     f"noise_schedule of a {kind}, key {key!r} (float64 results)")


def test_noise_schedule_does_not_modify_betas(dif):
    betas = np.array([0.1, 0.2, 0.5])
    before = betas.copy()
    call(dif.noise_schedule, betas, expected="a dict of arrays")
    assert_unchanged(before, betas, "betas", "noise_schedule")


@pytest.mark.parametrize("betas,why", [
    pytest.param([[0.1, 0.2], [0.3, 0.4]], "betas of shape (2, 2), not 1-D", id="2-d"),
    pytest.param([0.1, 0.0, 0.2], "a beta equal to 0", id="beta-0"),
    pytest.param([0.1, 1.0], "a beta equal to 1", id="beta-1"),
    pytest.param([0.1, -0.2], "a negative beta", id="beta-negative"),
    pytest.param([0.1, 1.5], "a beta above 1", id="beta-above-1"),
    pytest.param([0.1, float("nan")], "a beta equal to nan (NaN is outside (0, 1): comparisons with NaN are False)", id="beta-nan"),
])
def test_noise_schedule_refuses_invalid_betas(dif, betas, why):
    assert_raises(ValueError, dif.noise_schedule, betas, why=why)


def test_noise_schedule_matches_diffusers(dif, hf):
    betas = list(oracle_linear(200))
    sched = hf["DDPM"](num_train_timesteps=200, trained_betas=np.asarray(betas))
    ref_ab = sched.alphas_cumprod.numpy().astype(float)
    ref_var = np.array([float(sched._get_variance(t)) for t in range(1, 200)])
    want = oracle_schedule(betas)
    assert np.allclose(ref_ab, want["alpha_bars"], rtol=1e-5, atol=0), ORACLE_BUG
    assert np.allclose(ref_var, want["posterior_variance"][1:], rtol=1e-4, atol=0), ORACLE_BUG
    got = schedule_of(dif, betas)
    assert_close(got["alpha_bars"], ref_ab, 1e-5, 0, "noise_schedule(...)['alpha_bars'] against "
                                                     "DDPMScheduler.alphas_cumprod (float32)")
    assert_close(got["posterior_variance"][1:], ref_var, 1e-4, 0, "noise_schedule(...)['posterior_variance'] "
                 "against DDPMScheduler._get_variance (float32, variance_type='fixed_small')")


# ================================================================== q_sample (B5.14)
def q_sample(dif, x0, t, alpha_bars, noise=None, rng=None):
    """The learner's q_sample on copies; returns (x_t, noise) as arrays."""
    kwargs = {} if rng is None else {"rng": rng}
    if noise is not None:
        kwargs["noise"] = copy.deepcopy(noise)
    msg = "q_sample"
    result = pair_result(call(dif.q_sample, copy.deepcopy(x0), copy.deepcopy(t), copy.deepcopy(alpha_bars),
                              expected="the pair (x_t, noise)", **kwargs), msg)
    shape = np.shape(x0)
    inputs = f"x0 of shape {shape} and t of shape {np.shape(t)}"
    x_t = array_result(result[0], "q_sample(...)[0] (x_t)", shape, inputs=inputs)
    eps = array_result(result[1], "q_sample(...)[1] (noise)", shape, inputs=inputs)
    return x_t, eps


def test_q_sample_docstring_example(dif):
    ab = oracle_schedule(SMALL)["alpha_bars"]
    x0 = np.array([[1.0, -1.0], [0.5, 0.0]])
    x_t, eps = q_sample(dif, x0, [0, 2], ab, noise=np.ones((2, 2)))
    want = oracle_q_sample(x0, [0, 2], ab, np.ones((2, 2)))
    assert np.allclose(np.round(want, 4), [[1.2649, -0.6325], [1.1, 0.8]], rtol=0, atol=1e-12), ORACLE_BUG
    assert_close(x_t, want, msg="q_sample(x0, [0, 2], alpha_bars of [0.1, 0.2, 0.5], noise=ones) "
                                "(sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps)")
    assert_close(eps, np.ones((2, 2)), msg="q_sample(...)[1]: expected the noise that was given")


Q_CASES = [
    pytest.param((5,), [0, 999, 500, 1, 998], id="1-d-x0"),
    pytest.param((4, 3), [10, 0, 999, 250], id="2-d-x0"),
    pytest.param((3, 1, 4, 4), [999, 0, 400], id="images-NCHW"),
    pytest.param((6, 2), [7, 7, 7, 7, 7, 7], id="same-t-for-all"),
]


@pytest.mark.parametrize("shape,t", Q_CASES)
def test_q_sample_matches_formula_with_given_noise(dif, shape, t):
    rng = np.random.default_rng(sum(shape))
    x0, noise = rng.uniform(-1, 1, shape), rng.standard_normal(shape)
    x_t, eps = q_sample(dif, x0, np.array(t), AB_1000, noise=noise)
    msg = (f"q_sample with x0 of shape {shape}, t = {t}, linear schedule T = 1000 (each sample uses its own "
           "alpha_bar[t[n]], broadcast over its other axes)")
    assert_close(x_t, oracle_q_sample(x0, t, AB_1000, noise), msg=msg)
    assert_close(eps, noise, msg="q_sample(...)[1]: expected the noise that was given")


@pytest.mark.parametrize("kind", ["list", "int32-array", "int64-array"])
def test_q_sample_accepts_t_as_array_like(dif, kind):
    t = [2, 0, 1]
    t_in = {"list": t, "int32-array": np.array(t, dtype=np.int32), "int64-array": np.array(t, dtype=np.int64)}[kind]
    ab = oracle_schedule(SMALL)["alpha_bars"]
    x0 = np.array([[1.0, 2.0], [-0.5, 0.25], [0.0, -1.0]])
    noise = np.array([[0.3, -1.2], [2.0, 0.1], [-0.7, 0.9]])
    x_t, _ = q_sample(dif, x0, t_in, ab, noise=noise)
    assert_close(x_t, oracle_q_sample(x0, t, ab, noise), msg=f"q_sample with t = [2, 0, 1] given as a {kind}")


def test_q_sample_monte_carlo_mean_and_variance(dif):
    ab = oracle_schedule(SMALL)["alpha_bars"]          # 0.9, 0.72, 0.36: sqrt and the plain value differ
    n = 20_000
    x0 = np.tile(np.array([[2.0, -1.0, 0.5]]), (3 * n, 1))
    t = np.repeat([0, 1, 2], n)
    x_t, eps = q_sample(dif, x0, t, ab, rng=np.random.default_rng(7))
    for k in range(3):
        rows = slice(k * n, (k + 1) * n)
        monte_carlo(x_t[rows], math.sqrt(ab[k]) * x0[0], 1 - ab[k], f"q_sample of {n} copies of x0 at t = {k} "
                    f"(alpha_bar = {ab[k]:.2f}): mean sqrt(alpha_bar) x0, variance 1 - alpha_bar")
        monte_carlo(eps[rows], 0.0, 1.0, f"q_sample(...)[1] at t = {k}: the drawn noise must be standard normal")
    assert_close(x_t, oracle_q_sample(x0, t, ab, eps), msg="q_sample without noise: x_t must be computed with the "
                                                           "noise that is returned")


def test_q_sample_draws_noise_with_rng_standard_normal(dif):
    x0 = np.random.default_rng(3).uniform(-1, 1, (4, 2, 3))
    t = np.array([0, 999, 10, 500])
    x_t, eps = q_sample(dif, x0, t, AB_1000, rng=np.random.default_rng(42))
    want = np.random.default_rng(42).standard_normal(x0.shape)
    assert_close(eps, want, msg="q_sample(..., noise=None, rng=np.random.default_rng(42))[1]: expected "
                                "rng.standard_normal(x0.shape) (docstring)")
    assert_close(x_t, oracle_q_sample(x0, t, AB_1000, want), msg="q_sample with the noise drawn from rng")


def test_q_sample_rng_none_draws_fresh_noise(dif):
    x0 = np.zeros((3, 4))
    t = [1, 2, 3]
    _, eps1 = q_sample(dif, x0, t, AB_1000)
    _, eps2 = q_sample(dif, x0, t, AB_1000)
    if np.array_equal(eps1, eps2):
        _fail("q_sample(..., noise=None, rng=None) called twice",
              "expected two different noises (np.random.default_rng(), not seeded), got the same twice")


def test_q_sample_does_not_modify_inputs(dif):
    x0 = np.array([[1.0, -1.0], [0.5, 0.0]])
    t = np.array([0, 2])
    ab = oracle_schedule(SMALL)["alpha_bars"]
    noise = np.array([[0.3, -2.0], [1.0, 0.5]])
    args = [x0.copy(), t.copy(), ab.copy(), noise.copy()]
    call(dif.q_sample, args[0], args[1], args[2], noise=args[3], expected="the pair (x_t, noise)")
    for name, before, after in zip(("x0", "t", "alpha_bars", "noise"), (x0, t, ab, noise), args):
        assert_unchanged(before, after, name, "q_sample")


@pytest.mark.parametrize("t,noise_shape,why", [
    pytest.param([0, 3, 1], None, "t = 3 with T = 3 (indices in [0, T))", id="t-equals-T"),
    pytest.param([0, -1, 1], None, "t = -1 (indices in [0, T); -1 is not the last step)", id="t-negative"),
    pytest.param([0, 1], None, "t has 2 indices for 3 samples", id="t-too-short"),
    pytest.param([1], None, "t has 1 index for 3 samples (one index per sample, even if NumPy could broadcast)",
                 id="t-length-1-for-3-samples"),
    pytest.param([0, 1, 2], (3, 3), "noise of shape (3, 3) for x0 of shape (3, 2)", id="noise-wrong-shape"),
    pytest.param([0, 1, 2], (2,), "noise of shape (2,) for x0 of shape (3, 2) (NumPy would broadcast it)",
                 id="noise-broadcastable"),
])
def test_q_sample_refuses_invalid_inputs(dif, t, noise_shape, why):
    ab = oracle_schedule(SMALL)["alpha_bars"]
    x0 = np.zeros((3, 2))
    kwargs = {} if noise_shape is None else {"noise": np.ones(noise_shape)}
    assert_raises(ValueError, dif.q_sample, x0, t, ab, why=why, **kwargs)


def test_q_sample_matches_diffusers_add_noise(dif, hf):
    torch = hf["torch"]
    sched = ddpm_float64(hf, LINEAR_1000)
    rng = np.random.default_rng(11)
    x0, noise = rng.uniform(-1, 1, (4, 1, 3, 3)), rng.standard_normal((4, 1, 3, 3))
    t = np.array([0, 1, 500, 999])
    ref = sched.add_noise(torch.tensor(x0), torch.tensor(noise), torch.tensor(t)).numpy()
    assert np.allclose(ref, oracle_q_sample(x0, t, AB_1000, noise), rtol=1e-12, atol=1e-12), ORACLE_BUG
    x_t, _ = q_sample(dif, x0, t, AB_1000, noise=noise)
    assert_close(x_t, ref, msg="q_sample against DDPMScheduler.add_noise (float64)")


# ================================================================== q_posterior_mean_variance (B5.16)
def posterior(dif, x0, x_t, t, sched):
    msg = "q_posterior_mean_variance"
    result = pair_result(call(dif.q_posterior_mean_variance, copy.deepcopy(x0), copy.deepcopy(x_t),
                              copy.deepcopy(t), copy.deepcopy(sched), expected="the pair (mean, variance)"), msg)
    x0 = np.asarray(x0)
    inputs = f"x0 of shape {x0.shape} and t of shape {np.shape(t)}"
    mean = array_result(result[0], "q_posterior_mean_variance(...)[0] (mean)", x0.shape, inputs=inputs)
    var_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
    var = array_result(result[1], f"q_posterior_mean_variance(...)[1] (variance of shape (N, 1, ..., 1) = "
                                  f"{var_shape})", var_shape, inputs=inputs)
    return mean, var


def test_q_posterior_mean_variance_docstring_example(dif):
    sched = oracle_schedule(SMALL)
    x0 = np.array([[1.0, -1.0], [0.5, 0.0]])
    x_t = np.array([[1.2649, -0.6325], [1.1, 0.8]])
    mean, var = posterior(dif, x0, x_t, [0, 2], sched)
    want_mean, want_var = oracle_posterior(x0, x_t, [0, 2], sched)
    assert np.allclose(np.round(want_mean, 4), [[1.0, -1.0], [0.6718, 0.2475]], rtol=0, atol=1e-12), ORACLE_BUG
    assert_close(mean, want_mean, msg="q_posterior_mean_variance(x0, x_t, [0, 2], schedule of [0.1, 0.2, 0.5]), "
                                      "mean (coef1[t] x0 + coef2[t] x_t; at t = 0 it is x0)")
    assert_close(var, want_var, msg="q_posterior_mean_variance(...), variance (posterior_variance[t])")


POSTERIOR_CASES = [
    pytest.param((5,), [0, 1, 2, 999, 500], id="1-d-x0"),
    pytest.param((4, 3), [10, 0, 999, 1], id="2-d-x0"),
    pytest.param((3, 1, 4, 4), [999, 0, 400], id="images-NCHW"),
]


@pytest.mark.parametrize("shape,t", POSTERIOR_CASES)
def test_q_posterior_mean_variance_matches_closed_form(dif, shape, t):
    sched = oracle_schedule(LINEAR_1000)
    rng = np.random.default_rng(sum(shape) + 1)
    x0, x_t = rng.uniform(-1, 1, shape), rng.standard_normal(shape)
    mean, var = posterior(dif, x0, x_t, np.array(t), sched)
    want_mean, want_var = oracle_posterior(x0, x_t, t, sched)
    msg = f"q_posterior_mean_variance with x0 of shape {shape}, t = {t}, linear schedule T = 1000"
    assert_close(mean, want_mean, msg=msg + ", mean (coef1[t[n]] x0[n] + coef2[t[n]] x_t[n])")
    assert_close(var, want_var, 1e-6, 1e-12, msg=msg + ", variance (posterior_variance[t[n]])")


@pytest.mark.parametrize("betas,t,x0,x_t", [
    pytest.param(SMALL, 1, 0.8, -0.3, id="small-t1"),
    pytest.param(SMALL, 2, -1.0, 1.7, id="small-t2"),
    pytest.param(list(LINEAR_1000), 1, 0.5, 0.49, id="linear-t1"),
    pytest.param(list(LINEAR_1000), 300, -0.6, 0.2, id="linear-t300"),
    pytest.param(list(LINEAR_1000), 999, 0.9, -2.0, id="linear-t999"),
    pytest.param(list(oracle_cosine(50)), 25, 0.3, 1.1, id="cosine-t25"),
])
def test_q_posterior_mean_variance_matches_numerical_bayes(dif, betas, t, x0, x_t):
    sched = oracle_schedule(betas)
    want_mean, want_var = bayes_posterior_1d(x0, x_t, t, betas)
    o_mean, o_var = oracle_posterior(np.array([x0]), np.array([x_t]), [t], sched)
    assert math.isclose(o_mean[0], want_mean, rel_tol=1e-8, abs_tol=1e-10), ORACLE_BUG
    assert math.isclose(o_var[0], want_var, rel_tol=1e-8, abs_tol=1e-14), ORACLE_BUG
    mean, var = posterior(dif, np.array([x0]), np.array([x_t]), np.array([t]), sched)
    msg = (f"q_posterior_mean_variance at t = {t} (x0 = {x0}, x_t = {x_t}) against Bayes' rule on a grid: "
           "q(x_{t-1} | x0) q(x_t | x_{t-1}), normalised")
    assert_close(mean[0], want_mean, msg=msg + ", mean")
    assert_close(var[0], want_var, 1e-6, 1e-12, msg=msg + ", variance")


def test_q_posterior_mean_variance_does_not_modify_inputs(dif):
    sched = oracle_schedule(SMALL)
    x0 = np.array([[1.0, -1.0], [0.5, 0.0]])
    x_t = np.array([[1.2649, -0.6325], [1.1, 0.8]])
    t = np.array([1, 2])
    args = [x0.copy(), x_t.copy(), t.copy(), copy.deepcopy(sched)]
    call(dif.q_posterior_mean_variance, *args, expected="the pair (mean, variance)")
    for name, before, after in zip(("x0", "x_t", "t", "schedule"), (x0, x_t, t, sched), args):
        assert_unchanged(before, after, name, "q_posterior_mean_variance")


@pytest.mark.parametrize("t,x_t_shape,why", [
    pytest.param([0, 3], (2, 2), "t = 3 with T = 3", id="t-equals-T"),
    pytest.param([-1, 1], (2, 2), "t = -1 (-1 is not the last step)", id="t-negative"),
    pytest.param([0, 1], (2, 3), "x_t of shape (2, 3) for x0 of shape (2, 2)", id="x-t-wrong-shape"),
    pytest.param([0, 1], (2,), "x_t of shape (2,) for x0 of shape (2, 2) (NumPy would broadcast it)",
                 id="x-t-broadcastable"),
    pytest.param([1], (2, 2), "t has 1 index for 2 samples", id="t-length-1-for-2-samples"),
])
def test_q_posterior_mean_variance_refuses_invalid_inputs(dif, t, x_t_shape, why):
    sched = oracle_schedule(SMALL)
    assert_raises(ValueError, dif.q_posterior_mean_variance, np.zeros((2, 2)), np.zeros(x_t_shape), t, sched,
                  why=why)


# ================================================================== predict_x0_from_eps (B5.16)
def predict_x0(dif, x_t, t, eps, alpha_bars):
    result = call(dif.predict_x0_from_eps, copy.deepcopy(x_t), copy.deepcopy(t), copy.deepcopy(eps),
                  copy.deepcopy(alpha_bars), expected="the estimate of x0")
    return array_result(result, "predict_x0_from_eps", np.shape(x_t),
                        inputs=f"x_t of shape {np.shape(x_t)} and t of shape {np.shape(t)}")


def test_predict_x0_from_eps_docstring_example(dif):
    ab = oracle_schedule(SMALL)["alpha_bars"]
    x0 = np.array([[1.0, -1.0], [0.5, 0.0]])
    eps = np.array([[0.3, -2.0], [1.0, 0.5]])
    x_t = oracle_q_sample(x0, [1, 2], ab, eps)
    got = predict_x0(dif, x_t, [1, 2], eps, ab)
    assert_close(got, x0, msg="predict_x0_from_eps(q_sample(x0, [1, 2], ab, noise=eps)[0], [1, 2], eps, ab) "
                              "(the round trip gives x0 back)")


@pytest.mark.parametrize("shape,t", POSTERIOR_CASES)
def test_predict_x0_from_eps_matches_formula(dif, shape, t):
    rng = np.random.default_rng(sum(shape) + 2)
    x_t, eps = rng.standard_normal(shape), rng.standard_normal(shape)
    got = predict_x0(dif, x_t, np.array(t), eps, AB_1000)
    assert_close(got, oracle_predict_x0(x_t, t, eps, AB_1000),
                 msg=f"predict_x0_from_eps with x_t of shape {shape}, t = {t} "
                     "((x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t))")


@pytest.mark.parametrize("schedule_name", ["linear-T1000", "cosine-T1000", "random-T6"])
def test_predict_x0_from_eps_round_trip(dif, schedule_name):
    betas = {"linear-T1000": LINEAR_1000, "cosine-T1000": oracle_cosine(1000),
             "random-T6": random_betas(4, 6)}[schedule_name]
    ab = oracle_schedule(betas)["alpha_bars"]
    T = ab.size
    rng = np.random.default_rng(5)
    t = np.array(sorted({0, 1, T // 2, T - 2, T - 1} & set(range(T))))
    if schedule_name.startswith("cosine"):
        t = t[ab[t] > 1e-6]                    # the last cosine steps: alpha_bar ~ 1e-9, x0 is lost in rounding
    x0, noise = rng.uniform(-1, 1, (t.size, 3, 2)), rng.standard_normal((t.size, 3, 2))
    x_t = oracle_q_sample(x0, t, ab, noise)
    got = predict_x0(dif, x_t, t, noise, ab)
    assert_close(got, x0, msg=f"predict_x0_from_eps(q_sample(x0, t, ab, noise)[0], t, noise, ab), {schedule_name}, "
                              f"t = {t.tolist()} (expected x0 back)")


def test_predict_x0_from_eps_does_not_modify_inputs(dif):
    ab = oracle_schedule(SMALL)["alpha_bars"]
    x_t = np.array([[1.0, -1.0], [0.5, 0.0]])
    eps = np.array([[0.3, -2.0], [1.0, 0.5]])
    t = np.array([1, 2])
    args = [x_t.copy(), t.copy(), eps.copy(), ab.copy()]
    call(dif.predict_x0_from_eps, *args, expected="the estimate of x0")
    for name, before, after in zip(("x_t", "t", "eps", "alpha_bars"), (x_t, t, eps, ab), args):
        assert_unchanged(before, after, name, "predict_x0_from_eps")


@pytest.mark.parametrize("t,eps_shape,why", [
    pytest.param([0, 3], (2, 2), "t = 3 with T = 3", id="t-equals-T"),
    pytest.param([-1, 1], (2, 2), "t = -1 (-1 is not the last step)", id="t-negative"),
    pytest.param([0, 1], (2, 3), "eps of shape (2, 3) for x_t of shape (2, 2)", id="eps-wrong-shape"),
    pytest.param([0, 1], (1, 2), "eps of shape (1, 2) for x_t of shape (2, 2) (NumPy would broadcast it)",
                 id="eps-broadcastable"),
    pytest.param([0, 1, 2], (2, 2), "t has 3 indices for 2 samples", id="t-too-long"),
])
def test_predict_x0_from_eps_refuses_invalid_inputs(dif, t, eps_shape, why):
    ab = oracle_schedule(SMALL)["alpha_bars"]
    assert_raises(ValueError, dif.predict_x0_from_eps, np.zeros((2, 2)), t, np.zeros(eps_shape), ab, why=why)


# ================================================================== ddpm_step (B5.17)
def ddpm(dif, x_t, t, eps, sched, seed=None, rng=None, **kwargs):
    """The learner's ddpm_step on copies (rng = default_rng(seed) when a seed is given)."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    if rng is not None:
        kwargs["rng"] = rng
    result = call(dif.ddpm_step, copy.deepcopy(x_t), t, copy.deepcopy(eps), copy.deepcopy(sched),
                  expected="x_{t-1}", **kwargs)
    return array_result(result, "ddpm_step", np.shape(x_t), inputs=f"x_t of shape {np.shape(x_t)} (t = {t})")


def same_seed_same_noise(first, second, function, t):
    """Two calls with the same seed must give the same result: otherwise the noise does not come from rng."""
    if np.isnan(first).any() or np.isnan(second).any():
        _fail(f"{function} at t = {t}: expected finite values, got nan",
              "nan comes from the square root of a negative number: check alpha_bar_prev (t_prev = -1 means "
              "alpha_bar_prev = 1, the clean image) and 1 - alpha_bar_prev - sigma**2")
    if not np.array_equal(first, second):
        _fail(f"{function} at t = {t}, called twice with np.random.default_rng(seed) and the same seed",
              "expected the same result twice (the noise z must be drawn from rng), got two different results")


def ddpm_mean(dif, x_t, t, eps, sched, seed=0, **kwargs):
    """The mean of the learner's step: step(x_t, eps) - step(0, 0) with the same seed (same z)."""
    noisy = ddpm(dif, x_t, t, eps, sched, seed=seed, **kwargs)
    zero = np.zeros(np.shape(x_t))
    noise = ddpm(dif, zero, t, zero, sched, seed=seed, **kwargs)
    same_seed_same_noise(noise, ddpm(dif, zero, t, zero, sched, seed=seed, **kwargs), "ddpm_step", t)
    return noisy - noise


def test_ddpm_step_docstring_example(dif):
    sched = oracle_schedule(SMALL)
    x_t, eps = np.array([[0.5, -0.5]]), np.array([[1.0, 1.0]])
    want = algo2_mean(x_t, 0, eps, sched)
    assert np.allclose(np.round(want, 4), [[0.1937, -0.8604]], rtol=0, atol=1e-12), ORACLE_BUG
    got = ddpm(dif, x_t, 0, eps, sched)
    assert_close(got, want, msg="ddpm_step([[0.5, -0.5]], 0, [[1.0, 1.0]], schedule of [0.1, 0.2, 0.5]) "
                                "(t = 0: the mean of Algorithm 2, no noise)")


@pytest.mark.parametrize("variance", ["posterior", "beta"])
@pytest.mark.parametrize("clip", [None, 1.0])
def test_ddpm_step_adds_no_noise_at_t0(dif, variance, clip):
    sched = oracle_schedule(LINEAR_1000)
    rng = np.random.default_rng(8)
    x_t, eps = rng.uniform(-0.5, 0.5, (3, 2, 2)), rng.uniform(-0.5, 0.5, (3, 2, 2))
    kwargs = {"variance": variance, "clip_x0": clip}
    want = algo2_mean(x_t, 0, eps, sched) if clip is None else clipped_mean(x_t, 0, eps, sched, clip)
    got1 = ddpm(dif, x_t, 0, eps, sched, seed=1, **kwargs)
    got2 = ddpm(dif, x_t, 0, eps, sched, seed=2, **kwargs)
    msg = f"ddpm_step at t = 0 (variance={variance!r}, clip_x0={clip})"
    assert_close(got1, want, msg=msg + ": expected the mean, without noise")
    assert_close(got2, got1, msg=msg + ": expected the same result with two different seeds (no noise at t = 0)")


DDPM_T = [1, 2, 10, 500, 999]


@pytest.mark.parametrize("t", DDPM_T)
@pytest.mark.parametrize("variance", ["posterior", "beta"])
def test_ddpm_step_mean_matches_algorithm_2(dif, t, variance):
    sched = oracle_schedule(LINEAR_1000)
    rng = np.random.default_rng(t)
    x_t, eps = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
    got = ddpm_mean(dif, x_t, t, eps, sched, variance=variance)
    assert_close(got, algo2_mean(x_t, t, eps, sched),
                 msg=f"mean of ddpm_step at t = {t} (variance={variance!r}), seen as step(x_t, eps) - step(0, 0) "
                     "with the same seed: (x_t - beta_t / sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_t)")


@pytest.mark.parametrize("t", [1, 2, 300, 999])
def test_ddpm_step_mean_with_clipping(dif, t):
    sched = oracle_schedule(LINEAR_1000)
    rng = np.random.default_rng(100 + t)
    ab = sched["alpha_bars"][t]
    x0_hat, eps = rng.uniform(-2, 2, (6, 4)), rng.standard_normal((6, 4))
    x_t = math.sqrt(ab) * x0_hat + math.sqrt(1 - ab) * eps          # about half of x0_hat is clipped
    assert np.any(np.abs(x0_hat) > 1) and np.any(np.abs(x0_hat) < 1), ORACLE_BUG
    got = ddpm_mean(dif, x_t, t, eps, sched, clip_x0=1.0)
    assert_close(got, clipped_mean(x_t, t, eps, sched, 1.0),
                 msg=f"mean of ddpm_step at t = {t} with clip_x0=1.0 (x0_hat predicted, clipped to [-1, 1], then "
                     "posterior_mean_coef1 x0_hat + posterior_mean_coef2 x_t)")
    got_wide = ddpm_mean(dif, x_t, t, eps, sched, clip_x0=1e6)
    assert_close(got_wide, algo2_mean(x_t, t, eps, sched),
                 msg=f"mean of ddpm_step at t = {t} with clip_x0=1e6 (nothing clipped: the mean of Algorithm 2)")


def test_ddpm_step_clip_bound_is_used(dif):
    sched = oracle_schedule(LINEAR_1000)
    rng = np.random.default_rng(9)
    x_t, eps = 2.0 * rng.standard_normal((5, 3)), rng.standard_normal((5, 3))
    got = ddpm_mean(dif, x_t, 50, eps, sched, clip_x0=0.5)
    assert_close(got, clipped_mean(x_t, 50, eps, sched, 0.5),
                 msg="mean of ddpm_step at t = 50 with clip_x0=0.5 (x0_hat clipped to [-0.5, 0.5], not [-1, 1])")


@pytest.mark.parametrize("variance", ["posterior", "beta"])
@pytest.mark.parametrize("t", [1, 2])
def test_ddpm_step_noise_variance(dif, variance, t):
    sched = oracle_schedule(SMALL)       # beta tilde = 0.0714 and 0.2187 against beta = 0.2 and 0.5
    n = 20_000
    zero = np.zeros((n, 3))
    noise = ddpm(dif, zero, t, zero, sched, seed=21, variance=variance)
    var = sched["posterior_variance"][t] if variance == "posterior" else sched["betas"][t]
    what = "posterior_variance[t] (beta tilde)" if variance == "posterior" else "betas[t]"
    monte_carlo(noise, 0.0, var, f"ddpm_step(0, t = {t}, eps = 0, variance={variance!r}) on {n} samples: the added "
                                 f"noise sqrt(var) z must have the variance {what}")
    corr = np.corrcoef(noise[:, 0], noise[:, 1])[0, 1]
    if not abs(corr) < 5 / math.sqrt(n):
        _fail(f"ddpm_step at t = {t}", f"expected independent noise for every value of a sample (z ~ N(0, I) of the "
                                       f"shape of x_t), got a correlation of {corr:.3f} between two columns")


@pytest.mark.parametrize("t", [1, 2])
def test_ddpm_step_variance_choice_scales_the_same_noise(dif, t):
    sched = oracle_schedule(SMALL)
    zero = np.zeros((4, 3))
    post = ddpm(dif, zero, t, zero, sched, seed=5, variance="posterior")
    same_seed_same_noise(post, ddpm(dif, zero, t, zero, sched, seed=5, variance="posterior"), "ddpm_step", t)
    beta = ddpm(dif, zero, t, zero, sched, seed=5, variance="beta")
    ratio = math.sqrt(sched["posterior_variance"][t] / sched["betas"][t])
    assert_close(post, ratio * beta, msg=f"ddpm_step at t = {t}, same seed: the noise with variance='posterior' "
                                         f"must be sqrt(posterior_variance[t] / betas[t]) = {ratio:.6g} times the "
                                         "noise with variance='beta'")


def test_ddpm_step_respects_rng(dif):
    sched = oracle_schedule(LINEAR_1000)
    x_t, eps = np.ones((3, 2)), np.zeros((3, 2))
    a = ddpm(dif, x_t, 500, eps, sched, seed=3)
    b = ddpm(dif, x_t, 500, eps, sched, seed=3)
    c = ddpm(dif, x_t, 500, eps, sched, seed=4)
    assert_close(b, a, msg="ddpm_step at t = 500 twice with np.random.default_rng(3): expected the same result "
                           "(the noise must come from rng)")
    if np.allclose(a, c):
        _fail("ddpm_step at t = 500 with seeds 3 and 4", "expected two different results (the noise must come "
                                                         "from rng), got the same")
    n1, n2 = ddpm(dif, x_t, 500, eps, sched), ddpm(dif, x_t, 500, eps, sched)
    if np.array_equal(n1, n2):
        _fail("ddpm_step(..., rng=None) at t = 500 called twice", "expected two different results "
                                                                  "(np.random.default_rng(), not seeded)")


def test_ddpm_step_does_not_modify_inputs(dif):
    sched = oracle_schedule(SMALL)
    x_t, eps = np.array([[3.0, -0.5]]), np.array([[1.0, 2.0]])
    args = [x_t.copy(), eps.copy(), copy.deepcopy(sched)]
    for clip in (None, 1.0):
        call(dif.ddpm_step, args[0], 2, args[1], args[2], rng=np.random.default_rng(0), clip_x0=clip,
             expected="x_{t-1}")
        for name, before, after in zip(("x_t", "eps_pred", "schedule"), (x_t, eps, sched), args):
            assert_unchanged(before, after, name, f"ddpm_step (clip_x0={clip})")


@pytest.mark.parametrize("t,variance,why", [
    pytest.param(3, "posterior", "t = 3 with T = 3", id="t-equals-T"),
    pytest.param(-1, "posterior", "t = -1 (-1 is not the last step)", id="t-negative"),
    pytest.param(1, "fixed_small", "variance = 'fixed_small' (only 'posterior' or 'beta')", id="variance-fixed-small"),
    pytest.param(1, "Posterior", "variance = 'Posterior' (exact names)", id="variance-capitalised"),
    pytest.param(0, "betas", "variance = 'betas', even at t = 0 where no noise is added", id="variance-betas-at-t0"),
])
def test_ddpm_step_refuses_invalid_arguments(dif, t, variance, why):
    sched = oracle_schedule(SMALL)
    assert_raises(ValueError, dif.ddpm_step, np.zeros((2, 2)), t, np.zeros((2, 2)), sched,
                  rng=np.random.default_rng(0), variance=variance, why=why)


@pytest.mark.parametrize("variance,variance_type", [("posterior", "fixed_small"), ("beta", "fixed_large")])
@pytest.mark.parametrize("clip", [None, 1.0])
def test_ddpm_step_matches_diffusers(dif, hf, variance, variance_type, clip):
    torch = hf["torch"]
    sched = oracle_schedule(LINEAR_1000)
    ref_sched = ddpm_float64(hf, LINEAR_1000, variance_type, clip)
    rng = np.random.default_rng(31)
    x_t, eps = 2.0 * rng.standard_normal((2, 1, 3, 3)), rng.standard_normal((2, 1, 3, 3))

    def zeros(shape, **kw):
        return torch.zeros(shape, dtype=kw.get("dtype"))

    with mock.patch.object(hf["ddpm_module"], "randn_tensor", zeros):
        for t in (0, 1, 250, 999):
            ref = ref_sched.step(torch.tensor(eps), t, torch.tensor(x_t)).prev_sample.numpy()
            want = algo2_mean(x_t, t, eps, sched) if clip is None else clipped_mean(x_t, t, eps, sched, clip)
            assert np.allclose(ref, want, rtol=1e-9, atol=1e-11), ORACLE_BUG
            got = ddpm(dif, x_t, 0, eps, sched, variance=variance, clip_x0=clip) if t == 0 else \
                ddpm_mean(dif, x_t, t, eps, sched, variance=variance, clip_x0=clip)
            assert_close(got, ref, msg=f"{'ddpm_step' if t == 0 else 'mean of ddpm_step'} at t = {t} "
                                       f"(clip_x0={clip}) against DDPMScheduler.step without its noise "
                                       f"(variance_type={variance_type!r}, clip_sample={clip is not None})")
    zero = np.zeros((3, 2))
    for t in (1, 250):
        ref_var = float(ref_sched._get_variance(t))
        want_var = sched["posterior_variance"][t] if variance == "posterior" else sched["betas"][t]
        assert math.isclose(ref_var, want_var, rel_tol=1e-9), ORACLE_BUG
        other = "beta" if variance == "posterior" else "posterior"
        other_var = float(ddpm_float64(hf, LINEAR_1000, "fixed_large" if other == "beta" else "fixed_small")
                          ._get_variance(t))
        got = ddpm(dif, zero, t, zero, sched, seed=6, variance=variance)
        same_seed_same_noise(got, ddpm(dif, zero, t, zero, sched, seed=6, variance=variance), "ddpm_step", t)
        base = ddpm(dif, zero, t, zero, sched, seed=6, variance=other)
        assert_close(got, math.sqrt(ref_var / other_var) * base,
                     msg=f"noise of ddpm_step at t = {t}, same seed: variance={variance!r} against "
                         f"variance={other!r} must scale like DDPMScheduler._get_variance "
                         f"({variance_type!r} against the other)")


# ================================================================== ddim_step (B5.17)
def ddim(dif, x_t, t, t_prev, eps, alpha_bars, seed=None, rng=None, **kwargs):
    if seed is not None:
        rng = np.random.default_rng(seed)
    if rng is not None:
        kwargs["rng"] = rng
    result = call(dif.ddim_step, copy.deepcopy(x_t), t, t_prev, copy.deepcopy(eps), copy.deepcopy(alpha_bars),
                  expected="x_{t_prev}", **kwargs)
    return array_result(result, "ddim_step", np.shape(x_t),
                        inputs=f"x_t of shape {np.shape(x_t)} (t = {t}, t_prev = {t_prev})")


def ddim_det(dif, x_t, t, t_prev, eps, alpha_bars, eta, seed=0):
    """The deterministic part of the learner's step: step(x_t, eps) - step(0, 0) with the same seed."""
    zero = np.zeros(np.shape(x_t))
    noise = ddim(dif, zero, t, t_prev, zero, alpha_bars, seed=seed, eta=eta)
    same_seed_same_noise(noise, ddim(dif, zero, t, t_prev, zero, alpha_bars, seed=seed, eta=eta), "ddim_step", t)
    return ddim(dif, x_t, t, t_prev, eps, alpha_bars, seed=seed, eta=eta) - noise


def test_ddim_step_docstring_example(dif):
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    x0 = np.array([[0.5, -1.0, 0.25]])
    eps = np.array([[1.0, 0.0, -2.0]])
    x_t = oracle_q_sample(x0, [700], ab, eps)
    got = ddim(dif, x_t, 700, -1, eps, ab)
    assert_close(got, x0, msg="ddim_step(x_t, 700, -1, eps, alpha_bars) with the true noise and eta = 0: expected "
                              "x0 (t_prev = -1 means alpha_bar_prev = 1, the clean image)")


DDIM_PAIRS = [(1, 0), (1, -1), (0, -1), (500, 499), (999, 979), (700, 300), (999, -1), (20, 0), (250, 5)]


@pytest.mark.parametrize("t,t_prev", [pytest.param(*p, id=f"{p[0]}-to-{p[1]}") for p in DDIM_PAIRS])
def test_ddim_step_matches_formula_eta0(dif, t, t_prev):
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    rng = np.random.default_rng(t + 7)
    x_t, eps = rng.standard_normal((3, 2, 2)), rng.standard_normal((3, 2, 2))
    want, _ = ddim_parts(x_t, t, t_prev, eps, ab, 0.0)
    got1 = ddim(dif, x_t, t, t_prev, eps, ab, seed=1)
    got2 = ddim(dif, x_t, t, t_prev, eps, ab, seed=2)
    msg = (f"ddim_step from t = {t} to t_prev = {t_prev}, eta = 0 (sqrt(ab_prev) x0_hat + sqrt(1 - ab_prev) eps"
           + (", with ab_prev = 1" if t_prev == -1 else "") + ")")
    assert_close(got1, want, msg=msg)
    assert_close(got2, got1, msg=msg + ": expected the same result with two seeds (deterministic when eta = 0)")


@pytest.mark.parametrize("t,t_prev", [pytest.param(*p, id=f"{p[0]}-to-{p[1]}") for p in DDIM_PAIRS])
def test_ddim_step_true_noise_jumps_to_q_sample_at_t_prev(dif, t, t_prev):
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    rng = np.random.default_rng(t + 13)
    x0, noise = rng.uniform(-1, 1, (4, 3)), rng.standard_normal((4, 3))
    x_t = oracle_q_sample(x0, [t] * 4, ab, noise)
    got = ddim(dif, x_t, t, t_prev, noise, ab)
    want = x0 if t_prev == -1 else oracle_q_sample(x0, [t_prev] * 4, ab, noise)
    assert_close(got, want, msg=f"ddim_step(q_sample(x0, {t}, noise), {t}, {t_prev}, noise, eta=0): with the true "
                                "noise, expected " + ("x0" if t_prev == -1 else
                                                      f"q_sample(x0, {t_prev}, the same noise)"))


@pytest.mark.parametrize("eta", [0.5, 1.0])
@pytest.mark.parametrize("t,t_prev", [pytest.param(*p, id=f"{p[0]}-to-{p[1]}") for p in
                                      [(1, 0), (500, 499), (999, 979), (700, 300), (250, 5)]])
def test_ddim_step_deterministic_part_with_eta(dif, t, t_prev, eta):
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    rng = np.random.default_rng(t + 17)
    x_t, eps = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    want, sigma = ddim_parts(x_t, t, t_prev, eps, ab, eta)
    got = ddim_det(dif, x_t, t, t_prev, eps, ab, eta)
    assert_close(got, want, msg=f"ddim_step from {t} to {t_prev} with eta = {eta}, seen as step(x_t, eps) - "
                                f"step(0, 0) with the same seed: sqrt(ab_prev) x0_hat + sqrt(1 - ab_prev - sigma**2) "
                                f"eps_pred, sigma = {sigma:.6g}")


@pytest.mark.parametrize("t", [1, 2, 300, 999])
def test_ddim_step_eta1_variance_is_posterior_variance(dif, t):
    sched = oracle_schedule(LINEAR_1000)
    _, sigma = ddim_parts(np.zeros(1), t, t - 1, np.zeros(1), sched["alpha_bars"], 1.0)
    assert math.isclose(sigma**2, sched["posterior_variance"][t], rel_tol=1e-9), ORACLE_BUG
    n = 20_000
    zero = np.zeros((n, 2))
    noise = ddim(dif, zero, t, t - 1, zero, sched["alpha_bars"], seed=40, eta=1.0)
    monte_carlo(noise, 0.0, sched["posterior_variance"][t],
                f"ddim_step(0, {t}, {t - 1}, eps=0, eta=1) on {n} samples: the added noise sigma z must have the "
                "variance posterior_variance[t] (beta tilde)")
    corr = np.corrcoef(noise[:, 0], noise[:, 1])[0, 1]
    if not abs(corr) < 5 / math.sqrt(n):
        _fail(f"ddim_step at t = {t}", f"expected independent noise for every value of a sample, got a correlation "
                                       f"of {corr:.3f} between two columns")


@pytest.mark.parametrize("t,t_prev", [(500, 499), (700, 300)])
def test_ddim_step_noise_is_linear_in_eta(dif, t, t_prev):
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    zero = np.zeros((5, 2))
    full = ddim(dif, zero, t, t_prev, zero, ab, seed=9, eta=1.0)
    same_seed_same_noise(full, ddim(dif, zero, t, t_prev, zero, ab, seed=9, eta=1.0), "ddim_step", t)
    half = ddim(dif, zero, t, t_prev, zero, ab, seed=9, eta=0.5)
    assert_close(half, 0.5 * full, msg=f"ddim_step from {t} to {t_prev}, same seed: the noise with eta = 0.5 must "
                                       "be half the noise with eta = 1 (sigma = eta * ...)")


@pytest.mark.parametrize("eta", [0.5, 1.0])
def test_ddim_step_to_clean_image_adds_no_noise(dif, eta):
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    rng = np.random.default_rng(3)
    x_t, eps = rng.standard_normal((3, 2)), rng.standard_normal((3, 2))
    want, sigma = ddim_parts(x_t, 40, -1, eps, ab, eta)
    assert sigma == 0, ORACLE_BUG
    got = ddim(dif, x_t, 40, -1, eps, ab, seed=1, eta=eta)
    assert_close(got, want, msg=f"ddim_step(x_t, 40, -1, eps, eta={eta}): with t_prev = -1, ab_prev = 1 and "
                                "sigma = 0, expected x0_hat exactly (no noise)")


def test_ddim_step_eta0_leaves_rng_untouched(dif):
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    rng = np.random.default_rng(12)
    state = copy.deepcopy(rng.bit_generator.state)
    ddim(dif, np.ones((2, 2)), 500, 400, np.zeros((2, 2)), ab, rng=rng, eta=0.0)
    if rng.bit_generator.state != state:
        _fail("ddim_step(..., eta=0, rng=rng)", "expected rng to be left untouched (it is used only if eta > 0), "
                                                 "but numbers were drawn from it")


def test_ddim_step_respects_rng(dif):
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    x_t, eps = np.ones((3, 2)), np.zeros((3, 2))
    a = ddim(dif, x_t, 500, 400, eps, ab, seed=3, eta=1.0)
    b = ddim(dif, x_t, 500, 400, eps, ab, seed=3, eta=1.0)
    c = ddim(dif, x_t, 500, 400, eps, ab, seed=4, eta=1.0)
    assert_close(b, a, msg="ddim_step(eta=1) twice with np.random.default_rng(3): expected the same result")
    if np.allclose(a, c):
        _fail("ddim_step(eta=1) with seeds 3 and 4", "expected two different results (the noise must come from "
                                                     "rng), got the same")
    n1, n2 = ddim(dif, x_t, 500, 400, eps, ab, eta=1.0), ddim(dif, x_t, 500, 400, eps, ab, eta=1.0)
    if np.array_equal(n1, n2):
        _fail("ddim_step(eta=1, rng=None) called twice", "expected two different results "
                                                         "(np.random.default_rng(), not seeded)")


def test_ddim_step_does_not_modify_inputs(dif):
    ab = oracle_schedule(SMALL)["alpha_bars"]
    x_t, eps = np.array([[3.0, -0.5]]), np.array([[1.0, 2.0]])
    args = [x_t.copy(), eps.copy(), ab.copy()]
    for eta in (0.0, 1.0):
        call(dif.ddim_step, args[0], 2, 0, args[1], args[2], eta=eta, rng=np.random.default_rng(0),
             expected="x_{t_prev}")
        for name, before, after in zip(("x_t", "eps_pred", "alpha_bars"), (x_t, eps, ab), args):
            assert_unchanged(before, after, name, f"ddim_step (eta={eta})")


@pytest.mark.parametrize("t,t_prev,eta,why", [
    pytest.param(2, 2, 0.0, "t_prev = t = 2 (t_prev < t is required)", id="t-prev-equals-t"),
    pytest.param(1, 2, 0.0, "t_prev = 2 > t = 1", id="t-prev-above-t"),
    pytest.param(2, -2, 0.0, "t_prev = -2 (-1 <= t_prev is required)", id="t-prev-minus-2"),
    pytest.param(3, 1, 0.0, "t = 3 with T = 3 (t < T is required)", id="t-equals-T"),
    pytest.param(8, 1, 0.0, "t = 8 with T = 3", id="t-above-T"),
    pytest.param(-1, -2, 0.0, "t = -1, t_prev = -2", id="t-negative"),
    pytest.param(2, 1, -0.1, "eta = -0.1, outside [0, 1]", id="eta-negative"),
    pytest.param(2, 1, 1.5, "eta = 1.5, outside [0, 1]", id="eta-above-1"),
    pytest.param(2, 1, float("nan"), "eta = nan (NaN is outside [0, 1]: comparisons with NaN are False)", id="eta-nan"),
])
def test_ddim_step_refuses_invalid_arguments(dif, t, t_prev, eta, why):
    ab = oracle_schedule(SMALL)["alpha_bars"]
    assert_raises(ValueError, dif.ddim_step, np.zeros((2, 2)), t, t_prev, np.zeros((2, 2)), ab, eta=eta,
                  rng=np.random.default_rng(0), why=why)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("n_inference,t", [pytest.param(50, 700, id="stride20-700"),
                                           pytest.param(50, 19, id="stride20-19-to-clean"),
                                           pytest.param(1000, 500, id="stride1-500"),
                                           pytest.param(10, 999, id="stride100-999"),
                                           pytest.param(1, 999, id="one-step-to-clean")])
def test_ddim_step_matches_diffusers(dif, hf, n_inference, t, eta):
    torch = hf["torch"]
    ab = oracle_schedule(LINEAR_1000)["alpha_bars"]
    ref_sched = ddim_float64(hf, ab, n_inference)
    t_prev = t - 1000 // n_inference
    t_prev = -1 if t_prev < 0 else t_prev
    rng = np.random.default_rng(t + n_inference)
    x_t, eps = rng.standard_normal((2, 1, 3, 3)), rng.standard_normal((2, 1, 3, 3))
    ref = ref_sched.step(torch.tensor(eps), t, torch.tensor(x_t), eta=eta,
                         variance_noise=torch.zeros(x_t.shape, dtype=torch.float64)).prev_sample.numpy()
    want, _ = ddim_parts(x_t, t, t_prev, eps, ab, eta)
    assert np.allclose(ref, want, rtol=1e-12, atol=1e-12), ORACLE_BUG
    got = ddim(dif, x_t, t, t_prev, eps, ab, eta=eta, seed=0) if eta == 0 else \
        ddim_det(dif, x_t, t, t_prev, eps, ab, eta)
    assert_close(got, ref, msg=f"ddim_step from {t} to {t_prev} with eta = {eta} against DDIMScheduler.step "
                               f"({n_inference} inference steps, set_alpha_to_one=True, clip_sample=False"
                               + (", noise removed)" if eta else ")"))
