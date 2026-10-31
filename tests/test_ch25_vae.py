"""Tests of mylearn.vae (chapter 25): oracle tests and properties.

    pytest tests/test_ch25_vae.py              # your code (mon_travail/mylearn/vae.py)
    pytest tests/test_ch25_vae.py --impl=ref   # the reference

Oracles: PyTorch in float64 (``torch.autograd`` on ``mu + torch.exp(0.5 * log_var) * eps`` with eps fixed,
``torch.distributions.kl_divergence(Normal(mu, std), Normal(0, 1))`` and its autograd gradient) and the formulas
of the docstrings, transcribed in this file with NumPy. With PyTorch, the transcribed formulas are checked against
torch at every test (a disagreement is a bug of this file, not yours); without PyTorch, they are the oracle (with a
centred finite difference for the gradients), so that no test is skipped. The interpolation is checked against an
``np.linspace``-based interpolation and, for ``"slerp"``, against the norm and the angles of the points (from
``np.arccos`` of the cosine similarity). Your functions are always called on copies of the test data, and the tests
``..._does_not_modify_...`` check that the inputs are left as they were.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_interpolate_latents_"`` (25.20), ``-k "test_reparameterize_"`` (25.28: ``reparameterize`` and
``reparameterize_backward``), ``-k "test_gaussian_kl_"`` (25.29: ``gaussian_kl`` and ``gaussian_kl_grad``).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified
and may be lists; every array returned is a new float64 array, never a view of an input (``dmu`` and the gradient
with respect to ``mu`` equal an input, but are copies); ``mu``, ``log_var``, ``dz`` and ``eps`` are 2-D arrays
(n, d) of the same shape, and ``z0``, ``z1`` 1-D vectors of the same shape, otherwise ``ValueError``; an input that
contains NaN is refused with ``ValueError`` (the message says to remove the missing values first); ``rng`` may be
None, an integer seed or a ``np.random.Generator``, and the same seed gives the same draws; ``gaussian_kl`` returns
one value per row, shape (n,); ``interpolate_latents`` returns its first row equal to z0 and its last row equal to
z1 exactly, refuses ``n_steps`` < 2 and an unknown method, and falls back to the straight line when z0 and z1 are
parallel, anti-parallel or one of them is zero.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest

try:
    import torch
except ImportError:  # the PyTorch oracles run on Colab; the NumPy formulas take over here
    torch = None

ORACLE_BUG = ("bug in tests/test_ch25_vae.py (not in your code): the NumPy oracle of this test "
              "disagrees with torch")


@pytest.fixture
def vae(mylearn_module):
    return mylearn_module("vae")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3,
                           max_line_width=10**6)
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg=""):
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
                f"({bad.size} wrong value(s) out of {want.size})")
        data = f"expected {_short(want, 10)}\n     got {_short(got, 10)}" if want.size <= 16 else ""
        _fail(msg, head, data)


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
    try:
        function(*args, **kwargs)
    except NotImplementedError:
        raise                  # before `except error`: NotImplementedError is a RuntimeError; shown as "⏳"
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


def pair(result, what, names):
    """Unpack a 2-tuple result, or fail with a message that says what was expected."""
    label = f"the tuple ({', '.join(names)})"
    if result is None:
        _fail("", f"expected {label} from {what}, got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 2:
        _fail("", f"expected {label} from {what}, got {type(result).__name__}"
                  + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else ""))
    return result


def assert_new_float_array(arr, shape, what, inputs=()):
    """A float64 ndarray of the given shape that shares no memory with the inputs."""
    if not isinstance(arr, np.ndarray):
        _fail("", f"expected {what} to be a NumPy array, got {type(arr).__name__}")
    if arr.dtype != np.float64:
        _fail("", f"expected {what} to be float64, got {arr.dtype} (convert the inputs with "
                  "np.asarray(x, dtype=float))")
    if arr.shape != tuple(shape):
        _fail("", f"expected {what} of shape {tuple(shape)}, got {arr.shape}")
    for inp in inputs:
        if isinstance(inp, np.ndarray) and np.shares_memory(arr, inp):
            _fail("", f"expected {what} to be a new array, got a view of an input (changing it would change "
                      "the input: return a copy)")


def assert_unchanged(arrays, saved, what):
    for name, arr in arrays.items():
        if not np.array_equal(arr, saved[name]):
            _fail("", f"expected {name} unchanged by {what}, got {_short(arr)} (compute the result in new arrays)")


def copies(*arrays):
    return [np.array(a, copy=True) for a in arrays]


def _t(a, grad=False):
    return torch.tensor(np.asarray(a, dtype=np.float64), requires_grad=grad)


def _torch_agrees(ours, theirs, atol=1e-12):
    theirs = theirs.detach().numpy()
    assert np.shape(ours) == theirs.shape and np.allclose(ours, theirs, rtol=1e-10, atol=atol), ORACLE_BUG


def gaussians(seed, n, d, spread=1.5):
    """Random means and log-variances of shape (n, d), log-variances in [-spread, spread]."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=1.3, size=(n, d)), rng.uniform(-spread, spread, size=(n, d))


# ------------------------------------------------------------------ NumPy oracles (checked against torch)
def np_reparameterize(mu, log_var, eps):
    """z = mu + exp(0.5 log_var) eps (checked against the same expression in torch)."""
    z = mu + np.exp(0.5 * log_var) * eps
    if torch is not None:
        _torch_agrees(z, _t(mu) + torch.exp(0.5 * _t(log_var)) * _t(eps))
    return z


def np_reparameterize_backward(dz, eps, log_var):
    """(dz, dz * 0.5 * exp(0.5 log_var) * eps), checked against torch.autograd with eps fixed."""
    dmu, dlog_var = dz.copy(), dz * 0.5 * np.exp(0.5 * log_var) * eps
    if torch is not None:
        mu_t = _t(np.zeros_like(dz), grad=True)
        lv_t = _t(log_var, grad=True)
        z = mu_t + torch.exp(0.5 * lv_t) * _t(eps)
        (z * _t(dz)).sum().backward()
        _torch_agrees(dmu, mu_t.grad)
        _torch_agrees(dlog_var, lv_t.grad)
    return dmu, dlog_var


def np_gaussian_kl(mu, log_var):
    """0.5 * sum(mu^2 + exp(log_var) - 1 - log_var) per row (checked against torch.distributions)."""
    kl = 0.5 * np.sum(mu ** 2 + np.expm1(log_var) - log_var, axis=1)
    if torch is not None:
        from torch.distributions import Normal, kl_divergence
        mu_t, lv_t = _t(mu), _t(log_var)
        theirs = kl_divergence(Normal(mu_t, torch.exp(0.5 * lv_t)), Normal(torch.zeros_like(mu_t), 1.0)).sum(-1)
        _torch_agrees(kl, theirs, atol=1e-10)
    return kl


def np_gaussian_kl_grad(mu, log_var):
    """(mu, 0.5 (exp(log_var) - 1)), checked against torch.autograd on the torch.distributions KL."""
    dmu, dlog_var = mu.copy(), 0.5 * np.expm1(log_var)
    if torch is not None:
        from torch.distributions import Normal, kl_divergence
        mu_t, lv_t = _t(mu, grad=True), _t(log_var, grad=True)
        kl_divergence(Normal(mu_t, torch.exp(0.5 * lv_t)), Normal(torch.zeros_like(mu_t), 1.0)).sum().backward()
        _torch_agrees(dmu, mu_t.grad, atol=1e-10)
        _torch_agrees(dlog_var, lv_t.grad, atol=1e-10)
    return dmu, dlog_var


def np_slerp(z0, z1, n_steps):
    """The slerp formula of the docstring, with omega from the normalised vectors (not for parallel vectors)."""
    t = np.linspace(0.0, 1.0, n_steps)[:, None]
    cos = np.clip(np.dot(z0 / np.linalg.norm(z0), z1 / np.linalg.norm(z1)), -1.0, 1.0)
    omega = np.arccos(cos)
    return np.sin((1 - t) * omega) / np.sin(omega) * z0 + np.sin(t * omega) / np.sin(omega) * z1


def np_linear(z0, z1, n_steps):
    """(1 - t) z0 + t z1 on np.linspace(0, 1, n_steps)."""
    t = np.linspace(0.0, 1.0, n_steps)[:, None]
    return (1 - t) * z0 + t * z1


SHAPES = [pytest.param(1, 1, id="n1-d1"), pytest.param(4, 3, id="n4-d3"), pytest.param(7, 20, id="n7-d20"),
          pytest.param(2, 50, id="n2-d50")]


# ================================================================== reparameterize (25.28)
def test_reparameterize_docstring_example(vae):
    result = call(vae.reparameterize, np.zeros((2, 3)), np.zeros((2, 3)), rng=np.random.default_rng(0),
                  expected="the tuple (z, eps)")
    z, eps = pair(result, "reparameterize", ("z", "eps"))
    assert_new_float_array(z, (2, 3), "z")
    assert_new_float_array(eps, (2, 3), "eps")
    if not np.array_equal(z, eps):
        _fail("", "expected z == eps when mu = 0 and log_var = 0 (z = 0 + exp(0) * eps), got "
                  f"z = {_short(z)} and eps = {_short(eps)}")


def test_reparameterize_tiny_variance_leaves_the_mean(vae):
    z, _ = pair(call(vae.reparameterize, [[5.0]], [[-100.0]], expected="the tuple (z, eps)"),
                "reparameterize", ("z", "eps"))
    assert_close(z, [[5.0]], rtol=0, atol=1e-15,
                 msg="sigma = exp(0.5 * log_var) = exp(-50): z must be mu (the noise is multiplied by sigma)")


@pytest.mark.parametrize("n, d", SHAPES)
def test_reparameterize_z_is_mu_plus_sigma_times_eps(vae, n, d):
    mu, log_var = gaussians(10 * n + d, n, d)
    z, eps = pair(call(vae.reparameterize, *copies(mu, log_var), rng=np.random.default_rng(n + d),
                       expected="the tuple (z, eps)"), "reparameterize", ("z", "eps"))
    assert_new_float_array(z, (n, d), "z")
    assert_new_float_array(eps, (n, d), "eps")
    assert_close(z, np_reparameterize(mu, log_var, eps),
                 msg="expected z = mu + exp(0.5 * log_var) * eps with the eps you returned (sigma is "
                     "exp(0.5 * log_var), not exp(log_var) nor log_var)")


def test_reparameterize_draws_have_the_right_mean_and_variance(vae):
    n = 100_000
    mu = np.tile([[2.0, -1.0, 0.5]], (n, 1))
    log_var = np.tile([[0.0, np.log(4.0), np.log(0.25)]], (n, 1))
    z, eps = pair(call(vae.reparameterize, mu, log_var, rng=np.random.default_rng(2025),
                       expected="the tuple (z, eps)"), "reparameterize", ("z", "eps"))
    z, eps = np.asarray(z, dtype=float), np.asarray(eps, dtype=float)
    want_var = np.exp(log_var[0])
    se_mean = np.sqrt(want_var / n)
    if np.any(np.abs(z.mean(axis=0) - mu[0]) > 5 * se_mean):
        _fail("", f"expected the mean of z close to mu = {_short(mu[0])} over {n} draws, got "
                  f"{_short(z.mean(axis=0))} (is eps drawn from N(0, 1)?)")
    if np.any(np.abs(z.var(axis=0) / want_var - 1) > 0.03):
        _fail("", f"expected the variance of z close to exp(log_var) = {_short(want_var)} over {n} draws, got "
                  f"{_short(z.var(axis=0))} (sigma = exp(0.5 * log_var): the variance is sigma**2 = exp(log_var))")
    if abs(eps.mean()) > 0.01 or abs(eps.std() - 1) > 0.01:
        _fail("", f"expected eps drawn from N(0, 1): mean near 0 and std near 1, got mean {eps.mean():.4f} and "
                  f"std {eps.std():.4f}")


def test_reparameterize_same_seed_same_draws(vae):
    mu, log_var = gaussians(3, 5, 4)
    for seed_a, seed_b in ((7, 7), (np.random.default_rng(7), np.random.default_rng(7))):
        z_a, _ = pair(call(vae.reparameterize, mu, log_var, rng=seed_a, expected="the tuple (z, eps)"),
                      "reparameterize", ("z", "eps"))
        z_b, _ = pair(call(vae.reparameterize, mu, log_var, rng=seed_b, expected="the tuple (z, eps)"),
                      "reparameterize", ("z", "eps"))
        if not np.array_equal(z_a, z_b):
            _fail("", f"expected the same z for the same seed ({type(seed_a).__name__}), got two different draws "
                      "(rng may be None, an integer or a Generator: np.random.default_rng(rng) accepts all three)")
    z_c, _ = pair(call(vae.reparameterize, mu, log_var, rng=8, expected="the tuple (z, eps)"),
                  "reparameterize", ("z", "eps"))
    if np.array_equal(z_a, z_c):
        _fail("", "expected different draws for seeds 7 and 8, got the same z (is rng used?)")


def test_reparameterize_without_rng_draws_new_noise(vae):
    mu, log_var = gaussians(4, 3, 5)
    z_a, _ = pair(call(vae.reparameterize, mu, log_var, expected="the tuple (z, eps)"), "reparameterize",
                  ("z", "eps"))
    z_b, _ = pair(call(vae.reparameterize, mu, log_var, expected="the tuple (z, eps)"), "reparameterize",
                  ("z", "eps"))
    if np.array_equal(z_a, z_b):
        _fail("", "expected two calls with rng=None to give different z (np.random.default_rng() is a new "
                  "generator at each call), got the same draws")


def test_reparameterize_accepts_lists(vae):
    z, eps = pair(call(vae.reparameterize, [[0.0, 1.0]], [[0.0, 0.0]], rng=1, expected="the tuple (z, eps)"),
                  "reparameterize", ("z", "eps"))
    assert_new_float_array(z, (1, 2), "z")
    assert_close(z, np.array([[0.0, 1.0]]) + np.asarray(eps, dtype=float), msg="lists are array-likes")


def test_reparameterize_does_not_modify_its_inputs(vae):
    mu, log_var = gaussians(5, 4, 3)
    saved = {"mu": mu.copy(), "log_var": log_var.copy()}
    z, eps = pair(call(vae.reparameterize, mu, log_var, rng=0, expected="the tuple (z, eps)"),
                  "reparameterize", ("z", "eps"))
    assert_unchanged({"mu": mu, "log_var": log_var}, saved, "reparameterize")
    assert_new_float_array(z, (4, 3), "z", inputs=(mu, log_var))


@pytest.mark.parametrize("mu, log_var, why", [
    pytest.param(np.zeros((2, 3)), np.zeros((2, 4)), "mu is (2, 3) and log_var (2, 4)", id="shapes-differ"),
    pytest.param(np.zeros((3, 2)), np.zeros((2, 3)), "mu is (3, 2) and log_var (2, 3)", id="transposed"),
    pytest.param(np.zeros((2, 3)), np.zeros((1, 3)), "mu is (2, 3) and log_var (1, 3): no broadcasting",
                 id="broadcast-rows"),
    pytest.param(np.zeros((2, 3)), np.zeros((2, 1)), "mu is (2, 3) and log_var (2, 1): no broadcasting",
                 id="broadcast-cols"),
    pytest.param(np.zeros(3), np.zeros(3), "mu and log_var are 1-D: they must be (n, d)", id="1d"),
    pytest.param(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), "mu and log_var are 3-D: they must be (n, d)", id="3d"),
    pytest.param(np.array([[0.0, np.nan]]), np.zeros((1, 2)), "mu contains NaN: remove the missing values first",
                 id="nan-mu"),
    pytest.param(np.zeros((1, 2)), np.array([[np.nan, 0.0]]), "log_var contains NaN", id="nan-log_var"),
])
def test_reparameterize_refuses_invalid_inputs(vae, mu, log_var, why):
    assert_raises(ValueError, vae.reparameterize, mu, log_var, rng=0, why=why)


# ================================================================== reparameterize_backward (25.28)
def test_reparameterize_backward_docstring_example(vae):
    dmu, dlog_var = pair(call(vae.reparameterize_backward, np.ones((1, 2)), np.array([[1.0, -1.0]]),
                              np.zeros((1, 2)), expected="the tuple (dmu, dlog_var)"),
                         "reparameterize_backward", ("dmu", "dlog_var"))
    assert_close(dmu, [[1.0, 1.0]], msg="dz/dmu = 1, so dmu = dz")
    assert_close(dlog_var, [[0.5, -0.5]], msg="dz/dlog_var = 0.5 * exp(0.5 * log_var) * eps")


@pytest.mark.parametrize("n, d", SHAPES)
def test_reparameterize_backward_matches_autograd(vae, n, d):
    rng = np.random.default_rng(100 * n + d)
    dz, eps = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    log_var = rng.uniform(-2.0, 2.0, size=(n, d))
    want_mu, want_lv = np_reparameterize_backward(dz, eps, log_var)
    dmu, dlog_var = pair(call(vae.reparameterize_backward, *copies(dz, eps, log_var),
                              expected="the tuple (dmu, dlog_var)"), "reparameterize_backward", ("dmu", "dlog_var"))
    assert_new_float_array(dmu, (n, d), "dmu")
    assert_new_float_array(dlog_var, (n, d), "dlog_var")
    assert_close(dmu, want_mu, msg="expected dmu = dz (the upstream gradient goes through z = mu + ... unchanged)")
    assert_close(dlog_var, want_lv,
                 msg="expected dlog_var = dz * 0.5 * exp(0.5 * log_var) * eps (the chain rule through "
                     "sigma = exp(0.5 * log_var))")


def test_reparameterize_backward_matches_a_finite_difference(vae):
    rng = np.random.default_rng(31)
    mu, log_var, eps, w = (rng.normal(size=(3, 4)) for _ in range(4))
    loss = lambda m, lv: float(np.sum(w * (m + np.exp(0.5 * lv) * eps)))   # noqa: E731, dL/dz = w
    h = 1e-6
    num_lv = np.zeros_like(log_var)
    for idx in np.ndindex(log_var.shape):
        plus, minus = log_var.copy(), log_var.copy()
        plus[idx] += h
        minus[idx] -= h
        num_lv[idx] = (loss(mu, plus) - loss(mu, minus)) / (2 * h)
    _, dlog_var = pair(call(vae.reparameterize_backward, w.copy(), eps.copy(), log_var.copy(),
                            expected="the tuple (dmu, dlog_var)"), "reparameterize_backward", ("dmu", "dlog_var"))
    assert_close(dlog_var, num_lv, rtol=1e-6, atol=1e-8,
                 msg="expected dlog_var close to a centred finite difference of L = sum(w * z)")


def test_reparameterize_backward_no_gradient_when_eps_is_zero(vae):
    _, dlog_var = pair(call(vae.reparameterize_backward, np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2)),
                            expected="the tuple (dmu, dlog_var)"), "reparameterize_backward", ("dmu", "dlog_var"))
    assert_close(dlog_var, np.zeros((2, 2)), msg="with eps = 0, z = mu does not depend on log_var: dlog_var = 0")


def test_reparameterize_backward_returns_new_arrays(vae):
    dz, eps, log_var = np.ones((2, 3)), np.full((2, 3), 0.5), np.zeros((2, 3))
    dmu, dlog_var = pair(call(vae.reparameterize_backward, dz, eps, log_var, expected="the tuple (dmu, dlog_var)"),
                         "reparameterize_backward", ("dmu", "dlog_var"))
    assert_new_float_array(dmu, (2, 3), "dmu", inputs=(dz, eps, log_var))
    assert_new_float_array(dlog_var, (2, 3), "dlog_var", inputs=(dz, eps, log_var))


def test_reparameterize_backward_accepts_lists(vae):
    dmu, dlog_var = pair(call(vae.reparameterize_backward, [[2, 2]], [[1, -1]], [[0, 0]],
                              expected="the tuple (dmu, dlog_var)"), "reparameterize_backward", ("dmu", "dlog_var"))
    assert_new_float_array(dmu, (1, 2), "dmu")
    assert_close(dlog_var, [[1.0, -1.0]], msg="lists (of ints) are array-likes")


def test_reparameterize_backward_does_not_modify_its_inputs(vae):
    rng = np.random.default_rng(9)
    dz, eps, log_var = rng.normal(size=(3, 2)), rng.normal(size=(3, 2)), rng.normal(size=(3, 2))
    saved = {"dz": dz.copy(), "eps": eps.copy(), "log_var": log_var.copy()}
    call(vae.reparameterize_backward, dz, eps, log_var, expected="the tuple (dmu, dlog_var)")
    assert_unchanged({"dz": dz, "eps": eps, "log_var": log_var}, saved, "reparameterize_backward")


@pytest.mark.parametrize("dz, eps, log_var, why", [
    pytest.param(np.ones((2, 3)), np.ones((2, 2)), np.ones((2, 3)), "eps is (2, 2), dz and log_var (2, 3)",
                 id="eps-shape"),
    pytest.param(np.ones((2, 3)), np.ones((2, 3)), np.ones((3, 2)), "log_var is (3, 2), dz and eps (2, 3)",
                 id="log_var-shape"),
    pytest.param(np.ones((1, 3)), np.ones((2, 3)), np.ones((2, 3)), "dz has 1 row, eps and log_var 2", id="dz-rows"),
    pytest.param(np.ones(3), np.ones(3), np.ones(3), "1-D inputs: they must be (n, d)", id="1d"),
    pytest.param(np.array([[np.nan]]), np.ones((1, 1)), np.ones((1, 1)), "dz contains NaN", id="nan-dz"),
    pytest.param(np.ones((1, 1)), np.array([[np.nan]]), np.ones((1, 1)), "eps contains NaN", id="nan-eps"),
    pytest.param(np.ones((1, 1)), np.ones((1, 1)), np.array([[np.nan]]), "log_var contains NaN", id="nan-log_var"),
])
def test_reparameterize_backward_refuses_invalid_inputs(vae, dz, eps, log_var, why):
    assert_raises(ValueError, vae.reparameterize_backward, dz, eps, log_var, why=why)


# ================================================================== gaussian_kl (25.29)
def test_gaussian_kl_docstring_examples(vae):
    assert_close(call(vae.gaussian_kl, np.zeros((1, 2)), np.zeros((1, 2)), expected="an array of shape (1,)"),
                 [0.0], msg="the prior itself: mu = 0 and log_var = 0 give KL = 0")
    assert_close(call(vae.gaussian_kl, [[1.0, 0.0]], [[0.0, 0.0]], expected="an array of shape (1,)"),
                 [0.5], msg="mu = (1, 0), sigma = 1: KL = 0.5 * (1 + 1 - 1 - 0) on the first dimension")


@pytest.mark.parametrize("n, d", SHAPES)
def test_gaussian_kl_matches_torch_distributions(vae, n, d):
    mu, log_var = gaussians(7 * n + d, n, d, spread=2.5)
    want = np_gaussian_kl(mu, log_var)
    kl = call(vae.gaussian_kl, *copies(mu, log_var), expected=f"an array of shape ({n},)")
    assert_new_float_array(kl, (n,), "the KL (one value per row, summed over the d latent dimensions)")
    assert_close(kl, want, rtol=1e-9, atol=1e-11,
                 msg="expected KL = 0.5 * sum over j of (mu_j**2 + exp(log_var_j) - 1 - log_var_j), per row")


def test_gaussian_kl_sums_the_dimensions(vae):
    mu = np.array([[1.0, 2.0, 0.0]])
    log_var = np.array([[0.0, 0.0, np.log(2.0)]])
    parts = [float(np_gaussian_kl(mu[:, [j]], log_var[:, [j]])[0]) for j in range(3)]
    assert_close(call(vae.gaussian_kl, mu, log_var, expected="an array of shape (1,)"), [sum(parts)],
                 msg="the KL of independent dimensions is the sum of the KL of each dimension (sum over axis 1)")


def test_gaussian_kl_is_non_negative_and_zero_only_at_the_prior(vae):
    mu, log_var = gaussians(77, 200, 6, spread=4.0)
    kl = np.asarray(call(vae.gaussian_kl, mu, log_var, expected="an array of shape (200,)"), dtype=float)
    if np.any(kl < 0):
        _fail("", f"expected KL >= 0 for every row, got {kl.min():.6g} (check the sign of each term)")
    kl0 = np.asarray(call(vae.gaussian_kl, np.zeros((3, 4)), np.zeros((3, 4)), expected="an array of shape (3,)"))
    assert_close(kl0, np.zeros(3), atol=0, msg="mu = 0 and log_var = 0 is the prior: KL = 0 exactly")


def test_gaussian_kl_extreme_log_variances(vae):
    mu = np.zeros((1, 2))
    log_var = np.array([[30.0, -30.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        kl = call(vae.gaussian_kl, mu, log_var, expected="an array of shape (1,)")
    assert_close(kl, np_gaussian_kl(mu, log_var), rtol=1e-12,
                 msg="log_var = 30 and -30 are allowed (sigma**2 = e**30 and e**-30)")


def test_gaussian_kl_accepts_lists(vae):
    assert_close(call(vae.gaussian_kl, [[0, 0], [1, 1]], [[0, 0], [0, 0]], expected="an array of shape (2,)"),
                 [0.0, 1.0], msg="lists (of ints) are array-likes")


def test_gaussian_kl_does_not_modify_its_inputs(vae):
    mu, log_var = gaussians(8, 3, 3)
    saved = {"mu": mu.copy(), "log_var": log_var.copy()}
    call(vae.gaussian_kl, mu, log_var, expected="an array of shape (3,)")
    assert_unchanged({"mu": mu, "log_var": log_var}, saved, "gaussian_kl")


@pytest.mark.parametrize("mu, log_var, why", [
    pytest.param(np.zeros((2, 3)), np.zeros((3, 2)), "mu is (2, 3) and log_var (3, 2)", id="shapes-differ"),
    pytest.param(np.zeros((2, 3)), np.zeros((1, 3)), "mu is (2, 3) and log_var (1, 3): no broadcasting",
                 id="broadcast-rows"),
    pytest.param(np.zeros((2, 3)), np.zeros((2, 1)), "mu is (2, 3) and log_var (2, 1): no broadcasting",
                 id="broadcast-cols"),
    pytest.param(np.zeros(3), np.zeros(3), "1-D inputs: they must be (n, d)", id="1d"),
    pytest.param(np.array([[np.nan, 0.0]]), np.zeros((1, 2)), "mu contains NaN", id="nan-mu"),
    pytest.param(np.zeros((1, 2)), np.array([[0.0, np.nan]]), "log_var contains NaN", id="nan-log_var"),
])
def test_gaussian_kl_refuses_invalid_inputs(vae, mu, log_var, why):
    assert_raises(ValueError, vae.gaussian_kl, mu, log_var, why=why)


# ================================================================== gaussian_kl_grad (25.29)
def test_gaussian_kl_grad_docstring_example(vae):
    dmu, dlog_var = pair(call(vae.gaussian_kl_grad, [[1.0, 0.0]], [[0.0, np.log(3.0)]],
                              expected="the tuple (dmu, dlog_var)"), "gaussian_kl_grad", ("dmu", "dlog_var"))
    assert_close(dmu, [[1.0, 0.0]], msg="dKL/dmu = mu")
    assert_close(dlog_var, [[0.0, 1.0]], msg="dKL/dlog_var = 0.5 * (exp(log_var) - 1)")


@pytest.mark.parametrize("n, d", SHAPES)
def test_gaussian_kl_grad_matches_autograd(vae, n, d):
    mu, log_var = gaussians(13 * n + d, n, d, spread=2.0)
    want_mu, want_lv = np_gaussian_kl_grad(mu, log_var)
    dmu, dlog_var = pair(call(vae.gaussian_kl_grad, *copies(mu, log_var), expected="the tuple (dmu, dlog_var)"),
                         "gaussian_kl_grad", ("dmu", "dlog_var"))
    assert_new_float_array(dmu, (n, d), "dmu")
    assert_new_float_array(dlog_var, (n, d), "dlog_var")
    assert_close(dmu, want_mu, msg="expected dmu = mu (the derivative of 0.5 * mu**2)")
    assert_close(dlog_var, want_lv, rtol=1e-9, atol=1e-11,
                 msg="expected dlog_var = 0.5 * (exp(log_var) - 1) (the derivative of 0.5 * (exp(lv) - 1 - lv))")


def test_gaussian_kl_grad_matches_a_finite_difference(vae):
    mu, log_var = gaussians(21, 2, 3)
    h = 1e-6
    total = lambda m, lv: float(np.sum(np_gaussian_kl(m, lv)))   # noqa: E731
    num_mu, num_lv = np.zeros_like(mu), np.zeros_like(log_var)
    for idx in np.ndindex(mu.shape):
        for arr, num, other_first in ((mu, num_mu, True), (log_var, num_lv, False)):
            plus, minus = arr.copy(), arr.copy()
            plus[idx] += h
            minus[idx] -= h
            if other_first:
                num[idx] = (total(plus, log_var) - total(minus, log_var)) / (2 * h)
            else:
                num[idx] = (total(mu, plus) - total(mu, minus)) / (2 * h)
    dmu, dlog_var = pair(call(vae.gaussian_kl_grad, mu.copy(), log_var.copy(), expected="the tuple (dmu, dlog_var)"),
                         "gaussian_kl_grad", ("dmu", "dlog_var"))
    assert_close(dmu, num_mu, rtol=1e-6, atol=1e-8, msg="expected dmu close to a finite difference of sum(KL)")
    assert_close(dlog_var, num_lv, rtol=1e-6, atol=1e-8,
                 msg="expected dlog_var close to a finite difference of sum(KL)")


def test_gaussian_kl_grad_is_zero_at_the_prior(vae):
    dmu, dlog_var = pair(call(vae.gaussian_kl_grad, np.zeros((2, 3)), np.zeros((2, 3)),
                              expected="the tuple (dmu, dlog_var)"), "gaussian_kl_grad", ("dmu", "dlog_var"))
    assert_close(dmu, np.zeros((2, 3)), atol=0, msg="the KL is minimal at the prior: dmu = 0")
    assert_close(dlog_var, np.zeros((2, 3)), atol=0, msg="the KL is minimal at the prior: dlog_var = 0")


def test_gaussian_kl_grad_returns_new_arrays(vae):
    mu, log_var = gaussians(22, 2, 2)
    dmu, dlog_var = pair(call(vae.gaussian_kl_grad, mu, log_var, expected="the tuple (dmu, dlog_var)"),
                         "gaussian_kl_grad", ("dmu", "dlog_var"))
    assert_new_float_array(dmu, (2, 2), "dmu (equal to mu, but a copy)", inputs=(mu, log_var))
    assert_new_float_array(dlog_var, (2, 2), "dlog_var", inputs=(mu, log_var))


def test_gaussian_kl_grad_does_not_modify_its_inputs(vae):
    mu, log_var = gaussians(23, 3, 2)
    saved = {"mu": mu.copy(), "log_var": log_var.copy()}
    call(vae.gaussian_kl_grad, mu, log_var, expected="the tuple (dmu, dlog_var)")
    assert_unchanged({"mu": mu, "log_var": log_var}, saved, "gaussian_kl_grad")


@pytest.mark.parametrize("mu, log_var, why", [
    pytest.param(np.zeros((2, 3)), np.zeros((2, 2)), "mu is (2, 3) and log_var (2, 2)", id="shapes-differ"),
    pytest.param(np.zeros(2), np.zeros(2), "1-D inputs: they must be (n, d)", id="1d"),
    pytest.param(np.array([[np.nan]]), np.zeros((1, 1)), "mu contains NaN", id="nan-mu"),
    pytest.param(np.zeros((1, 1)), np.array([[np.nan]]), "log_var contains NaN", id="nan-log_var"),
])
def test_gaussian_kl_grad_refuses_invalid_inputs(vae, mu, log_var, why):
    assert_raises(ValueError, vae.gaussian_kl_grad, mu, log_var, why=why)


# ================================================================== interpolate_latents (25.20)
def test_interpolate_latents_docstring_examples(vae):
    got = call(vae.interpolate_latents, np.array([0.0, 0.0]), np.array([1.0, 2.0]), 3, expected="a (3, 2) array")
    assert_close(got, [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]], msg="linear: (1 - t) z0 + t z1 for t = 0, 0.5, 1")
    got = call(vae.interpolate_latents, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 3, method="slerp",
               expected="a (3, 2) array")
    half = np.sqrt(0.5)
    assert_close(got, [[1.0, 0.0], [half, half], [0.0, 1.0]],
                 msg="slerp between two orthogonal unit vectors: the middle point is at 45 degrees, on the circle")


@pytest.mark.parametrize("d, n_steps", [pytest.param(1, 2, id="d1-2steps"), pytest.param(4, 6, id="d4-6steps"),
                                        pytest.param(20, 11, id="d20-11steps")])
def test_interpolate_latents_linear_matches_linspace(vae, d, n_steps):
    rng = np.random.default_rng(d + n_steps)
    z0, z1 = rng.normal(size=d), rng.normal(size=d)
    got = call(vae.interpolate_latents, *copies(z0, z1), n_steps, expected=f"a ({n_steps}, {d}) array")
    assert_new_float_array(got, (n_steps, d), "the interpolated points (one row per point)")
    assert_close(got, np_linear(z0, z1, n_steps), rtol=1e-12, atol=1e-14,
                 msg=f"expected (1 - t) z0 + t z1 for {n_steps} values of t evenly spaced from 0 to 1, both included")


def test_interpolate_latents_default_method_is_linear(vae):
    z0, z1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    got = call(vae.interpolate_latents, z0, z1, 3, expected="a (3, 2) array")
    assert_close(got, np_linear(z0, z1, 3), msg="method='linear' is the default")


@pytest.mark.parametrize("method", ["linear", "slerp"])
def test_interpolate_latents_endpoints_are_exact(vae, method):
    rng = np.random.default_rng(5)
    z0, z1 = rng.normal(size=7), 3.0 * rng.normal(size=7)
    got = np.asarray(call(vae.interpolate_latents, *copies(z0, z1), 5, method=method, expected="a (5, 7) array"))
    if got.shape != (5, 7) or not (np.array_equal(got[0], z0) and np.array_equal(got[-1], z1)):
        _fail("", f"expected the first row equal to z0 and the last row equal to z1 exactly ({method}), got "
                  f"first {_short(got[0]) if got.ndim == 2 else _short(got)}")


def test_interpolate_latents_slerp_stays_on_the_unit_sphere(vae):
    rng = np.random.default_rng(11)
    z0, z1 = rng.normal(size=10), rng.normal(size=10)
    z0, z1 = z0 / np.linalg.norm(z0), z1 / np.linalg.norm(z1)
    got = np.asarray(call(vae.interpolate_latents, *copies(z0, z1), 9, method="slerp", expected="a (9, 10) array"),
                     dtype=float)
    norms = np.linalg.norm(got, axis=1)
    assert_close(norms, np.ones(9), rtol=0, atol=1e-12,
                 msg="between two unit vectors, every slerp point has norm 1 (a straight line would cut inside)")
    cos = np.clip(np.sum(got[:-1] * got[1:], axis=1), -1, 1)
    steps = np.arccos(cos)
    total = np.arccos(np.clip(z0 @ z1, -1, 1))
    assert_close(steps, np.full(8, total / 8), rtol=1e-9, atol=1e-12,
                 msg="slerp moves by the same angle between consecutive points (omega / (n_steps - 1))")


@pytest.mark.parametrize("angle", [0.05, 1e-3], ids=["0.05-rad", "1e-3-rad"])
def test_interpolate_latents_slerp_at_a_small_angle(vae, angle):
    # small, but sin(omega) is far from 0: still a real slerp (at 1e-3 rad, the straight line is 1.5e-7 away)
    z0 = np.array([1.0, 0.0, 0.0])
    z1 = 1.5 * np.array([np.cos(angle), np.sin(angle), 0.0])
    got = call(vae.interpolate_latents, *copies(z0, z1), 6, method="slerp", expected="a (6, 3) array")
    assert_close(got, np_slerp(z0, z1, 6), rtol=1e-9, atol=1e-12,
                 msg=f"at an angle of {angle} rad, sin(omega) ≈ {angle} is not « almost 0 »: the slerp formula applies "
                     "(the fallback is only for sin(omega) of the order of rounding errors)")


def test_interpolate_latents_slerp_formula_for_vectors_of_any_norm(vae):
    rng = np.random.default_rng(12)
    z0, z1 = 2.5 * rng.normal(size=6), 0.4 * rng.normal(size=6)
    got = call(vae.interpolate_latents, *copies(z0, z1), 7, method="slerp", expected="a (7, 6) array")
    assert_close(got, np_slerp(z0, z1, 7), rtol=1e-10, atol=1e-12,
                 msg="expected sin((1 - t) omega) / sin(omega) * z0 + sin(t omega) / sin(omega) * z1, with omega "
                     "the angle between z0 and z1 (from the normalised vectors)")


@pytest.mark.parametrize("z0, z1, case", [
    pytest.param(np.array([1.0, 2.0, -1.0]), np.array([3.0, 6.0, -3.0]), "parallel", id="parallel"),
    pytest.param(np.array([0.0, -2.0, 0.0]), np.array([0.0, 3.0, 0.0]), "anti-parallel", id="anti-parallel"),
    pytest.param(np.zeros(3), np.array([1.0, 0.5, 2.0]), "z0 is the zero vector", id="zero"),
])
def test_interpolate_latents_slerp_falls_back_to_linear(vae, z0, z1, case):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            got = vae.interpolate_latents(*copies(z0, z1), 5, method="slerp")
        except RuntimeWarning as warning:
            _fail("", f"expected no RuntimeWarning ({case}), got « {warning} » (test the norms and sin(omega) "
                      "before dividing by them)")
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            _fail("", f"expected an array of finite values ({case}), but interpolate_latents raised "
                      f"{type(exc).__name__}: {exc}")
    assert_close(got, np_linear(z0, z1, 5), rtol=1e-6, atol=1e-9,
                 msg=f"{case}: sin(omega) is (almost) 0, so slerp falls back to the straight line")


def test_interpolate_latents_accepts_lists(vae):
    got = call(vae.interpolate_latents, [0, 0], [2, 4], 3, expected="a (3, 2) array")
    assert_new_float_array(got, (3, 2), "the interpolated points")
    assert_close(got, [[0, 0], [1, 2], [2, 4]], msg="lists (of ints) are array-likes")


def test_interpolate_latents_does_not_modify_its_inputs(vae):
    z0, z1 = np.array([1.0, 0.0, 2.0]), np.array([0.0, 1.0, -1.0])
    saved = {"z0": z0.copy(), "z1": z1.copy()}
    for method in ("linear", "slerp"):
        got = call(vae.interpolate_latents, z0, z1, 4, method=method, expected="a (4, 3) array")
        assert_unchanged({"z0": z0, "z1": z1}, saved, f"interpolate_latents({method!r})")
        assert_new_float_array(got, (4, 3), "the interpolated points", inputs=(z0, z1))


@pytest.mark.parametrize("args, kwargs, why", [
    pytest.param((np.zeros(2), np.ones(2), 1), {}, "n_steps = 1: at least the two endpoints", id="one-step"),
    pytest.param((np.zeros(2), np.ones(2), 0), {}, "n_steps = 0", id="zero-steps"),
    pytest.param((np.zeros(2), np.ones(2), 3), {"method": "cubic"}, "unknown method 'cubic'", id="method"),
    pytest.param((np.zeros(2), np.ones(3), 3), {}, "z0 has 2 values and z1 3", id="shapes-differ"),
    pytest.param((np.zeros((1, 2)), np.ones((1, 2)), 3), {}, "z0 and z1 must be single vectors of shape (d,)",
                 id="2d"),
    pytest.param((np.array([0.0, np.nan]), np.ones(2), 3), {}, "z0 contains NaN", id="nan"),
])
def test_interpolate_latents_refuses_invalid_inputs(vae, args, kwargs, why):
    assert_raises(ValueError, vae.interpolate_latents, *args, why=why, **kwargs)
