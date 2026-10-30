"""Tests of mylearn.gen_metrics (chapter 27): oracle tests and properties.

    pytest tests/test_ch27_gen_metrics.py              # your code (mon_travail/mylearn/gen_metrics.py)
    pytest tests/test_ch27_gen_metrics.py --impl=ref   # the reference

Oracles: ``scipy.stats.wasserstein_distance`` for the 1-D Wasserstein distance (samples of equal and of different
sizes, with ties, shifted, scaled); for the Fréchet distance, the formula of pytorch-fid transcribed in this file
(``scipy.linalg.sqrtm`` of sigma1 @ sigma2, real part) on random symmetric positive definite matrices, a symmetric
formula ``Tr((sigma1^(1/2) sigma2 sigma1^(1/2))^(1/2))`` computed with ``np.linalg.eigh`` (stable on singular
covariances), the FID formula of torchmetrics transcribed in PyTorch (``torch.linalg.eigvals``; the function of
torchmetrics itself when it is installed), and closed forms: diagonal or commuting covariances,
sum((sqrt(a) - sqrt(b))**2) + ||mu1 - mu2||**2; 1-D Gaussians, (m1 - m2)**2 + (s1 - s2)**2.
``frechet_distance_from_features`` is compared with these oracles applied to ``np.mean(axis=0)`` and
``np.cov(rowvar=False)`` (the tests of one function never use another function of yours as an oracle). Your
functions are called on copies of the test data, and the tests ``..._does_not_modify_...`` check that the inputs
are left as they were.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_wasserstein_1d_"`` (27.11), ``-k "test_frechet_distance_"`` (27.22: ``frechet_distance`` and
``frechet_distance_from_features``; ``-k "test_frechet_distance_from_features_"`` for the second one alone).

Contract precisions written in the statements of these exercises and checked here:
- the inputs may be lists (of integers too) and are never modified (sort a copy, ``np.sort``, not ``.sort()``);
- the three functions return a Python float (``float(...)``, not ``np.float64``), >= 0;
- invalid inputs raise ``ValueError``: NaN or an infinite value in any input; for ``wasserstein_1d`` an empty
  sample or a sample that is not 1-D (a number, a 2-D array); for ``frechet_distance`` a mean that is not 1-D
  (d,), a covariance whose shape is not (d, d) with the d of ``mu1``, means of different lengths; for
  ``frechet_distance_from_features`` an input that is not 2-D (n, d), feature dimensions that differ, a set of
  fewer than 2 samples;
- ``frechet_distance`` takes the trace of the square root from the eigenvalues of sigma1 @ sigma2 (real part,
  clipped at 0 before the square root) and clips a tiny negative result to 0: identical Gaussians give 0.0
  exactly or a tiny positive number, never a negative one, and no NumPy warning is emitted on singular
  covariances (more features than samples, duplicated features, rank-1 covariances);
- ``frechet_distance_from_features`` uses the covariance normalised by n - 1 and works with a single feature
  (n, 1) (``np.cov`` then returns a 0-d array: reshape it to (1, 1)).
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest
from scipy import linalg, stats

try:
    import torch
except ImportError:  # the PyTorch oracle runs on Colab; the NumPy oracles remain
    torch = None

try:  # the FID formula of torchmetrics itself, when it is installed (else its transcription below)
    from torchmetrics.image.fid import _compute_fid as _torchmetrics_compute_fid
except Exception:  # noqa: BLE001 - absent, or another version without this private function
    _torchmetrics_compute_fid = None


@pytest.fixture
def gm(mylearn_module):
    return mylearn_module("gen_metrics")


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
        _fail(msg, "expected a number, got None (did you forget the return?)")
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected a number, got an object of type {type(result).__name__}")
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected a single number (shape {want.shape}), got shape {got.shape}")
    ok = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=False)
    if not ok.all():
        i = int(np.flatnonzero(~ok)[0])
        _fail(msg, f"expected {want.ravel()[i] + 0.0:.12g}, got {got.ravel()[i] + 0.0:.12g}")


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


def call(function, *args, expected="a float", **kwargs):
    """Call the learner's function; an unexpected exception becomes a failure that says what was expected."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def call_quietly(function, *args, expected="a float", **kwargs):
    """`call`, with every warning (a NumPy RuntimeWarning, a ComplexWarning…) turned into a failure."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call(function, *args, expected=expected, **kwargs)
    if caught:
        w = caught[0]
        _fail("", f"expected no warning, got {w.category.__name__}: {w.message} (clip the eigenvalues at 0 "
                  "before np.sqrt and keep their real part)")
    return result


def assert_python_float(value, what):
    if value is None:
        _fail("", f"expected {what} to be a Python float, got None (did you forget the return?)")
    if type(value) is not float:
        _fail("", f"expected {what} to be a Python float, got {type(value).__name__} {value!r} "
                  "(convert it with float(...): a NumPy scalar does not print like the docstring)")


def assert_unchanged(arrays, saved, what):
    for name, arr in arrays.items():
        if not np.array_equal(arr, saved[name]):
            _fail("", f"expected {name} unchanged by {what}, got {_short(arr)} (work on a copy: np.sort(x), "
                      "not x.sort())")


# ------------------------------------------------------------------ oracles written in this file
def random_spd(rng, d, cond=None):
    """A random symmetric positive definite matrix (d, d); with `cond`, eigenvalues from 1 to cond."""
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig = rng.uniform(0.2, 3.0, size=d) if cond is None else np.geomspace(1.0, cond, d)
    s = (q * eig) @ q.T
    return (s + s.T) / 2


def fd_pytorch_fid(mu1, sigma1, mu2, sigma2):
    """The formula of pytorch-fid (calculate_frechet_distance): scipy.linalg.sqrtm of sigma1 @ sigma2, real part."""
    diff = np.asarray(mu1, float) - np.asarray(mu2, float)
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(len(diff)) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = np.real(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _psd_sqrt(s):
    w, v = np.linalg.eigh((s + s.T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def fd_symmetric(mu1, sigma1, mu2, sigma2):
    """Tr((sigma1 sigma2)^(1/2)) = Tr((sigma1^(1/2) sigma2 sigma1^(1/2))^(1/2)): only symmetric matrices, eigh."""
    diff = np.asarray(mu1, float) - np.asarray(mu2, float)
    r = _psd_sqrt(sigma1)
    m = r @ sigma2 @ r
    tr = np.sum(np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)))
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr)


def fd_diagonal(mu1, a, mu2, b):
    """Closed form for diagonal (or commuting) covariances of eigenvalues a and b."""
    diff = np.asarray(mu1, float) - np.asarray(mu2, float)
    return float(diff @ diff + np.sum((np.sqrt(a) - np.sqrt(b)) ** 2))


def fid_torchmetrics(feats1, feats2):
    """FID of torchmetrics on features: its mean and covariance (n - 1), then a + b - 2 * sum(sqrt(eigvals))."""
    f1 = torch.tensor(np.asarray(feats1, float), dtype=torch.float64)
    f2 = torch.tensor(np.asarray(feats2, float), dtype=torch.float64)
    stats_ = []
    for f in (f1, f2):
        n = f.shape[0]
        mu = f.mean(dim=0)
        cov = (f.T @ f - n * torch.outer(mu, mu)) / (n - 1)
        stats_.append((mu, cov))
    (mu1, s1), (mu2, s2) = stats_
    if _torchmetrics_compute_fid is not None:
        return float(_torchmetrics_compute_fid(mu1, s1, mu2, s2))
    a = (mu1 - mu2).square().sum(dim=-1)
    b = s1.trace() + s2.trace()
    c = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum(dim=-1)
    return float(a + b - 2 * c)


def gaussian_stats(feats):
    feats = np.asarray(feats, float)
    return feats.mean(axis=0), np.cov(feats, rowvar=False).reshape(feats.shape[1], feats.shape[1])


def copies(*arrays):
    return [np.array(a, dtype=float, copy=True) for a in arrays]


# ================================================================== wasserstein_1d (27.11)
def test_wasserstein_1d_docstring_examples(gm):
    got = call(gm.wasserstein_1d, [0.0, 1.0, 3.0], [5.0, 6.0, 8.0], expected="5.0")
    assert_python_float(got, "wasserstein_1d([0, 1, 3], [5, 6, 8])")
    assert_close(got, 5.0, msg="a shift by 5: every unit of mass moves by 5")
    got = call(gm.wasserstein_1d, [0.0, 1.0], [0.0, 1.0, 2.0], expected="0.5")
    assert_python_float(got, "wasserstein_1d([0, 1], [0, 1, 2])")
    assert_close(got, 0.5, msg="|F_u - F_v| = 1/6 on [0, 1] and 1/3 on [1, 2]")


def _sample(gen, kind, size):
    if kind == "normal":
        return gen.normal(size=size)
    if kind == "uniform":
        return gen.uniform(-3, 5, size=size)
    if kind == "ties":                                   # many equal values
        return gen.integers(-3, 4, size=size).astype(float)
    if kind == "skewed":
        return gen.exponential(2.0, size=size)
    raise ValueError(kind)


@pytest.mark.parametrize("seed, n, m, kind", [
    pytest.param(0, 1, 1, "normal", id="1-vs-1"),
    pytest.param(1, 10, 10, "normal", id="equal-10"),
    pytest.param(2, 500, 500, "uniform", id="equal-500"),
    pytest.param(3, 30, 30, "ties", id="equal-ties"),
    pytest.param(4, 1, 7, "normal", id="1-vs-7"),
    pytest.param(5, 7, 3, "normal", id="7-vs-3"),
    pytest.param(6, 50, 80, "uniform", id="50-vs-80"),
    pytest.param(7, 25, 40, "ties", id="25-vs-40-ties"),
    pytest.param(8, 300, 120, "skewed", id="300-vs-120"),
    pytest.param(9, 2, 1000, "normal", id="2-vs-1000"),
    pytest.param(10, 13, 17, "ties", id="13-vs-17-ties"),
])
def test_wasserstein_1d_matches_scipy(gm, seed, n, m, kind):
    gen = np.random.default_rng(seed)
    u, v = _sample(gen, kind, n), _sample(gen, kind, m) + gen.normal()
    got = call(gm.wasserstein_1d, *copies(u, v), expected="a float")
    assert_python_float(got, "wasserstein_1d")
    assert_close(got, stats.wasserstein_distance(u, v), rtol=1e-9, atol=1e-12,
                 msg=f"scipy.stats.wasserstein_distance, samples of sizes {n} and {m} ({kind})")


@pytest.mark.parametrize("seed", range(8))
def test_wasserstein_1d_matches_scipy_on_random_sizes(gm, seed):
    gen = np.random.default_rng(100 + seed)
    for _ in range(5):
        n, m = (int(k) for k in gen.integers(1, 60, size=2))
        u = gen.normal(gen.normal(), gen.uniform(0.5, 3), size=n).round(int(gen.integers(0, 3)))   # ties
        v = gen.normal(gen.normal(), gen.uniform(0.5, 3), size=m).round(int(gen.integers(0, 3)))
        got = call(gm.wasserstein_1d, *copies(u, v), expected="a float")
        assert_close(got, stats.wasserstein_distance(u, v), rtol=1e-9, atol=1e-12,
                     msg=f"scipy.stats.wasserstein_distance, sizes {n} and {m}")


def test_wasserstein_1d_unsorted_inputs(gm):
    gen = np.random.default_rng(3)
    u, v = gen.normal(size=40), gen.normal(1.0, 2.0, size=40)
    got = call(gm.wasserstein_1d, *copies(u, v), expected="a float")
    assert_close(got, np.mean(np.abs(np.sort(u) - np.sort(v))), msg="mean(|sort(u) - sort(v)|): sort BOTH samples")
    got = call(gm.wasserstein_1d, *copies(u, v[:25]), expected="a float")
    assert_close(got, stats.wasserstein_distance(u, v[:25]), msg="unsorted samples of sizes 40 and 25")


@pytest.mark.parametrize("shift", [0.0, 2.5, -7.0], ids=["shift0", "shift2.5", "shift-7"])
def test_wasserstein_1d_of_a_shift_is_the_shift(gm, shift):
    u = np.random.default_rng(11).normal(size=50)
    got = call(gm.wasserstein_1d, u.copy(), u + shift, expected=f"{abs(shift)}")
    assert_close(got, abs(shift), rtol=1e-12, atol=1e-12, msg=f"v = u + {shift}: all the mass moves by |{shift}|")


def test_wasserstein_1d_properties(gm):
    gen = np.random.default_rng(21)
    u, v, w = gen.normal(size=31), gen.uniform(-1, 3, size=17), gen.exponential(size=23)
    d_uv = call(gm.wasserstein_1d, *copies(u, v))
    assert_close(call(gm.wasserstein_1d, *copies(v, u)), d_uv, msg="symmetry: W(v, u) = W(u, v)")
    assert_close(call(gm.wasserstein_1d, u + 4.0, v + 4.0), d_uv, msg="translation: W(u + 4, v + 4) = W(u, v)")
    assert_close(call(gm.wasserstein_1d, -3.0 * u, -3.0 * v), 3.0 * d_uv,
                 msg="scale: W(-3 u, -3 v) = 3 W(u, v)")
    assert_close(call(gm.wasserstein_1d, u.copy(), u.copy()), 0.0, msg="W(u, u) = 0")
    assert_close(call(gm.wasserstein_1d, np.repeat(u, 3), u.copy()), 0.0,
                 msg="W(u repeated 3 times, u) = 0: the same empirical distribution")
    d_uw, d_wv = call(gm.wasserstein_1d, *copies(u, w)), call(gm.wasserstein_1d, *copies(w, v))
    if d_uv > d_uw + d_wv + 1e-12:
        _fail("", f"expected the triangle inequality W(u, v) <= W(u, w) + W(w, v), got {d_uv:.6g} > "
                  f"{d_uw:.6g} + {d_wv:.6g}")
    if d_uv < 0:
        _fail("", f"expected a distance >= 0, got {d_uv}")


def test_wasserstein_1d_accepts_lists_tuples_and_integers(gm):
    got = call(gm.wasserstein_1d, [3, 1, 2], (5, 4, 6), expected="3.0")
    assert_python_float(got, "wasserstein_1d([3, 1, 2], (5, 4, 6))")
    assert_close(got, 3.0, msg="integer lists: a shift by 3")
    assert_close(call(gm.wasserstein_1d, [0, 0, 1], [1], expected="2/3"), 2 / 3,
                 msg="two thirds of the mass move from 0 to 1")


def test_wasserstein_1d_does_not_modify_its_inputs(gm):
    u, v = np.array([3.0, -1.0, 2.0, 0.5]), np.array([1.0, 7.0, -2.0])
    call(gm.wasserstein_1d, u, v)
    call(gm.wasserstein_1d, u, v[:2].copy())
    assert_unchanged({"u": u, "v": v}, {"u": np.array([3.0, -1.0, 2.0, 0.5]), "v": np.array([1.0, 7.0, -2.0])},
                     "wasserstein_1d")


@pytest.mark.parametrize("u, v, why", [
    pytest.param([], [1.0, 2.0], "u is empty", id="u-empty"),
    pytest.param([1.0], np.array([]), "v is empty", id="v-empty"),
    pytest.param([[1.0, 2.0]], [1.0, 2.0], "u of shape (1, 2): a sample is 1-D", id="u-2d"),
    pytest.param([1.0, 2.0], np.ones((3, 1)), "v of shape (3, 1): a sample is 1-D", id="v-2d"),
    pytest.param(1.0, [1.0, 2.0], "u is a single number: a sample is a 1-D array", id="u-0d"),
    pytest.param([1.0, np.nan], [0.0], "u contains NaN", id="u-nan"),
    pytest.param([0.0, 1.0], [2.0, np.inf], "v contains inf", id="v-inf"),
    pytest.param([-np.inf, 0.0], [0.0, 1.0], "u contains -inf", id="u-minus-inf"),
])
def test_wasserstein_1d_refuses_invalid_inputs(gm, u, v, why):
    assert_raises(ValueError, gm.wasserstein_1d, u, v, why=why)


# ================================================================== frechet_distance (27.22)
def test_frechet_distance_docstring_examples(gm):
    got = call(gm.frechet_distance, np.zeros(2), np.eye(2), np.array([3.0, 4.0]), np.eye(2), expected="25.0")
    assert_python_float(got, "frechet_distance(...)")
    assert_close(got, 25.0, msg="same covariance: only ||mu1 - mu2||**2 = 3**2 + 4**2")
    got = call(gm.frechet_distance, np.zeros(2), np.diag([1.0, 4.0]), np.zeros(2), np.diag([4.0, 1.0]),
               expected="2.0")
    assert_python_float(got, "frechet_distance(...)")
    assert_close(got, 2.0, msg="Tr = 5 + 5 - 2 * (2 + 2): the eigenvalues of sigma1 @ sigma2 are 4 and 4")


@pytest.mark.parametrize("seed, d", [
    pytest.param(0, 1, id="d1"), pytest.param(1, 2, id="d2"), pytest.param(2, 5, id="d5"),
    pytest.param(3, 10, id="d10"), pytest.param(4, 32, id="d32"), pytest.param(5, 64, id="d64"),
])
def test_frechet_distance_matches_pytorch_fid(gm, seed, d):
    gen = np.random.default_rng(seed)
    mu1, mu2 = gen.normal(size=d), gen.normal(0.5, 1.0, size=d)
    s1, s2 = random_spd(gen, d), random_spd(gen, d)
    got = call(gm.frechet_distance, *copies(mu1, s1, mu2, s2))
    assert_python_float(got, "frechet_distance")
    want = fd_pytorch_fid(mu1, s1, mu2, s2)
    assert_close(got, want, rtol=1e-8, atol=1e-9,
                 msg=f"||mu1 - mu2||**2 + Tr(s1 + s2 - 2 sqrtm(s1 @ s2)) (pytorch-fid), d = {d}")
    assert_close(got, fd_symmetric(mu1, s1, mu2, s2), rtol=1e-8, atol=1e-9, msg="the symmetric formula")


@pytest.mark.parametrize("seed, d, cond", [
    pytest.param(10, 6, 1e3, id="cond1e3"), pytest.param(11, 12, 1e6, id="cond1e6"),
    pytest.param(12, 20, 1e2, id="d20-cond1e2"),
])
def test_frechet_distance_ill_conditioned(gm, seed, d, cond):
    gen = np.random.default_rng(seed)
    mu1, mu2 = gen.normal(size=d), gen.normal(size=d)
    s1, s2 = random_spd(gen, d, cond), random_spd(gen, d, cond)
    got = call_quietly(gm.frechet_distance, *copies(mu1, s1, mu2, s2))
    assert_close(got, fd_symmetric(mu1, s1, mu2, s2), rtol=1e-7, atol=1e-7 * cond,
                 msg=f"eigenvalues of the covariances from 1 to {cond:g}")


@pytest.mark.parametrize("seed, d", [pytest.param(0, 1, id="d1"), pytest.param(1, 4, id="d4"),
                                     pytest.param(2, 50, id="d50")])
def test_frechet_distance_diagonal_closed_form(gm, seed, d):
    gen = np.random.default_rng(20 + seed)
    a, b = gen.uniform(0.1, 5, size=d), gen.uniform(0.1, 5, size=d)
    mu1, mu2 = gen.normal(size=d), gen.normal(size=d)
    got = call(gm.frechet_distance, mu1.copy(), np.diag(a), mu2.copy(), np.diag(b))
    assert_close(got, fd_diagonal(mu1, a, mu2, b), rtol=1e-10, atol=1e-12,
                 msg="diagonal covariances: ||mu1 - mu2||**2 + sum((sqrt(a) - sqrt(b))**2)")


def test_frechet_distance_commuting_covariances_closed_form(gm):
    """Same eigenvectors, other eigenvalues: the closed form holds, but the diagonals of the matrices are useless."""
    gen = np.random.default_rng(31)
    d = 6
    q, _ = np.linalg.qr(gen.normal(size=(d, d)))
    a, b = gen.uniform(0.1, 4, size=d), gen.uniform(0.1, 4, size=d)
    s1, s2 = (q * a) @ q.T, (q * b) @ q.T
    mu = gen.normal(size=d)
    got = call(gm.frechet_distance, mu.copy(), s1, mu.copy(), s2)
    assert_close(got, fd_diagonal(mu, a, mu, b), rtol=1e-9, atol=1e-10,
                 msg="covariances Q diag(a) Q^T and Q diag(b) Q^T: sum((sqrt(a) - sqrt(b))**2) (use the "
                     "eigenvalues of sigma1 @ sigma2, not its diagonal)")


def test_frechet_distance_one_dimension(gm):
    got = call(gm.frechet_distance, [1.0], [[4.0]], [-2.0], [[9.0]], expected="10.0")
    assert_close(got, 10.0, msg="1-D Gaussians: (m1 - m2)**2 + (s1 - s2)**2 = 3**2 + (2 - 3)**2")


@pytest.mark.parametrize("seed", range(12))
def test_frechet_distance_identical_gaussians_give_zero(gm, seed):
    gen = np.random.default_rng(40 + seed)
    d = int(gen.integers(2, 40))
    mu, s = gen.normal(size=d) * 10, random_spd(gen, d) * gen.uniform(0.5, 50)
    got = call_quietly(gm.frechet_distance, *copies(mu, s, mu, s), expected="0.0")
    assert_python_float(got, "frechet_distance of two identical Gaussians")
    if got < 0:
        _fail("", f"expected a result >= 0 for two identical Gaussians, got {got!r} (rounding gives a tiny "
                  "negative number: clip it to 0)")
    assert_close(got, 0.0, atol=1e-8 * np.trace(s), msg=f"two identical Gaussians, d = {d}")


def test_frechet_distance_properties(gm):
    gen = np.random.default_rng(51)
    d = 7
    mu1, mu2, t = gen.normal(size=d), gen.normal(size=d), gen.normal(size=d) * 5
    s1, s2 = random_spd(gen, d), random_spd(gen, d)
    base = call(gm.frechet_distance, *copies(mu1, s1, mu2, s2))
    if base < 0:
        _fail("", f"expected a squared distance >= 0, got {base}")
    assert_close(call(gm.frechet_distance, *copies(mu2, s2, mu1, s1)), base, rtol=1e-9, atol=1e-10,
                 msg="symmetry: d(N2, N1) = d(N1, N2)")
    assert_close(call(gm.frechet_distance, *copies(mu1 + t, s1, mu2 + t, s2)), base, rtol=1e-9, atol=1e-10,
                 msg="translation of both means: the distance does not change")
    assert_close(call(gm.frechet_distance, *copies(3 * mu1, 9 * s1, 3 * mu2, 9 * s2)), 9 * base, rtol=1e-9,
                 atol=1e-9, msg="scaling the data by 3 (means x 3, covariances x 9) multiplies the distance by 9")
    same_cov = call(gm.frechet_distance, *copies(mu1, s1, mu2, s1))
    assert_close(same_cov, np.sum((mu1 - mu2) ** 2), rtol=1e-9, atol=1e-10,
                 msg="same covariance: only ||mu1 - mu2||**2 (squared, not its square root)")


def test_frechet_distance_singular_covariances(gm):
    """Rank-deficient covariances (rank 1 and rank 3 out of 8): no warning, >= 0, the symmetric oracle."""
    gen = np.random.default_rng(61)
    d = 8
    x = gen.normal(size=d)
    rank1 = np.outer(x, x)
    a = gen.normal(size=(d, 3))
    rank3 = a @ a.T
    mu1, mu2 = gen.normal(size=d), gen.normal(size=d)
    for name, s1, s2 in (("rank 1 and rank 3", rank1, rank3), ("rank 3 and itself", rank3, rank3),
                         ("rank 3 and the identity", rank3, np.eye(d))):
        got = call_quietly(gm.frechet_distance, *copies(mu1, s1, mu2, s2))
        assert_python_float(got, "frechet_distance")
        assert_close(got, fd_symmetric(mu1, s1, mu2, s2), rtol=1e-6, atol=1e-6, msg=f"covariances of {name}")
        same = call_quietly(gm.frechet_distance, *copies(mu1, s1, mu1, s1), expected="0.0")
        if not 0.0 <= same <= 1e-6:
            _fail("", f"expected 0.0 (or a tiny number >= 0) for identical Gaussians of {name}, got {same!r}")


def test_frechet_distance_accepts_lists(gm):
    got = call(gm.frechet_distance, [0, 0], [[1, 0], [0, 1]], [3, 4], [[1, 0], [0, 1]], expected="25.0")
    assert_python_float(got, "frechet_distance of lists")
    assert_close(got, 25.0, msg="lists of integers")


def test_frechet_distance_does_not_modify_its_inputs(gm):
    gen = np.random.default_rng(71)
    arrays = {"mu1": gen.normal(size=4), "sigma1": random_spd(gen, 4), "mu2": gen.normal(size=4),
              "sigma2": random_spd(gen, 4)}
    saved = {k: v.copy() for k, v in arrays.items()}
    call(gm.frechet_distance, arrays["mu1"], arrays["sigma1"], arrays["mu2"], arrays["sigma2"])
    assert_unchanged(arrays, saved, "frechet_distance")


def _fd_args(**changes):
    args = {"mu1": np.zeros(3), "sigma1": np.eye(3), "mu2": np.ones(3), "sigma2": 2 * np.eye(3)}
    args.update(changes)
    return args


@pytest.mark.parametrize("args, why", [
    pytest.param(_fd_args(mu1=np.zeros((1, 3))), "mu1 of shape (1, 3): a mean is 1-D (d,)", id="mu1-2d"),
    pytest.param(_fd_args(mu1=0.0), "mu1 is a single number: a mean is 1-D (d,)", id="mu1-0d"),
    pytest.param(_fd_args(mu2=np.ones(2)), "mu2 has 2 values, mu1 has 3", id="mu2-length"),
    pytest.param(_fd_args(mu2=np.ones(1)), "mu2 has 1 value, mu1 has 3 (no broadcasting)", id="mu2-length-1"),
    pytest.param(_fd_args(mu1=np.zeros(1), mu2=np.ones(1)),
                 "means of length 1 with covariances (3, 3) (no broadcasting)", id="means-length-1"),
    pytest.param(_fd_args(sigma1=np.ones((3, 2)), sigma2=np.ones((2, 3))),
                 "covariances of shapes (3, 2) and (2, 3): sigma1 @ sigma2 exists but they are not square",
                 id="sigmas-not-square"),
    pytest.param(_fd_args(mu1=np.zeros(2), mu2=np.ones(2)),
                 "means of length 2 with covariances (3, 3)", id="means-vs-covariances"),
    pytest.param(_fd_args(sigma1=np.ones(3)), "sigma1 of shape (3,): a covariance is (d, d)", id="sigma1-1d"),
    pytest.param(_fd_args(sigma1=np.ones((3, 2))), "sigma1 of shape (3, 2) is not square", id="sigma1-not-square"),
    pytest.param(_fd_args(sigma2=np.eye(2)), "sigma2 of shape (2, 2) for d = 3", id="sigma2-size"),
    pytest.param(_fd_args(sigma2=np.eye(4)), "sigma2 of shape (4, 4) for d = 3", id="sigma2-too-large"),
    pytest.param(_fd_args(mu1=np.array([0.0, np.nan, 0.0])), "mu1 contains NaN", id="mu1-nan"),
    pytest.param(_fd_args(mu2=np.array([0.0, np.inf, 0.0])), "mu2 contains inf", id="mu2-inf"),
    pytest.param(_fd_args(sigma1=np.diag([1.0, np.nan, 1.0])), "sigma1 contains NaN", id="sigma1-nan"),
    pytest.param(_fd_args(sigma2=np.diag([1.0, 1.0, np.inf])), "sigma2 contains inf", id="sigma2-inf"),
])
def test_frechet_distance_refuses_invalid_inputs(gm, args, why):
    assert_raises(ValueError, gm.frechet_distance, **args, why=why)


# ================================================================== frechet_distance_from_features (27.22)
def test_frechet_distance_from_features_docstring_examples(gm):
    X = np.random.default_rng(0).normal(size=(500, 3))
    got = call(gm.frechet_distance_from_features, X.copy(), X.copy(), expected="0.0")
    assert_python_float(got, "frechet_distance_from_features(X, X)")
    if not 0.0 <= got < 1e-9:
        _fail("", f"expected 0.0 (or a tiny number >= 0) for the same features, got {got!r}")
    got = call(gm.frechet_distance_from_features, X.copy(), X + 2.0, expected="12.0")
    assert_python_float(got, "frechet_distance_from_features(X, X + 2)")
    assert_close(got, 12.0, rtol=1e-9, atol=1e-9, msg="X + 2: same covariance, means apart by (2, 2, 2)")


@pytest.mark.parametrize("seed, n1, n2, d", [
    pytest.param(0, 5, 8, 3, id="5-vs-8-d3"), pytest.param(1, 2, 2, 1, id="2-vs-2-d1"),
    pytest.param(2, 50, 30, 4, id="50-vs-30-d4"), pytest.param(3, 200, 300, 16, id="200-vs-300-d16"),
    pytest.param(4, 1000, 400, 64, id="1000-vs-400-d64"),
])
def test_frechet_distance_from_features_matches_mean_and_cov(gm, seed, n1, n2, d):
    gen = np.random.default_rng(seed)
    mix = gen.normal(size=(d, d))
    f1 = gen.normal(size=(n1, d)) @ mix
    f2 = gen.normal(0.3, 1.0, size=(n2, d)) @ (mix + 0.3 * gen.normal(size=(d, d)))
    got = call(gm.frechet_distance_from_features, *copies(f1, f2))
    assert_python_float(got, "frechet_distance_from_features")
    mu1, s1 = gaussian_stats(f1)
    mu2, s2 = gaussian_stats(f2)
    want = fd_symmetric(mu1, s1, mu2, s2)
    if d <= 16:   # sqrtm is slow and less accurate on large matrices
        assert_close(fd_pytorch_fid(mu1, s1, mu2, s2), want, rtol=1e-7, atol=1e-8,
                     msg="BUG OF THIS FILE (not yours): pytorch-fid and the symmetric formula disagree")
    assert_close(got, want, rtol=1e-7, atol=1e-8,
                 msg=f"Fréchet distance of np.mean(axis=0) and np.cov(rowvar=False) (normalised by n - 1), "
                     f"{n1} and {n2} samples of dimension {d}")
    if torch is not None:
        assert_close(got, fid_torchmetrics(f1, f2), rtol=1e-7, atol=1e-8, msg="the FID formula of torchmetrics")


def test_frechet_distance_from_features_one_feature(gm):
    gen = np.random.default_rng(81)
    a, b = gen.normal(1.0, 2.0, size=(40, 1)), gen.normal(-1.0, 0.5, size=(25, 1))
    want = (a.mean() - b.mean()) ** 2 + (a.std(ddof=1) - b.std(ddof=1)) ** 2
    got = call(gm.frechet_distance_from_features, *copies(a, b),
               expected="a float (np.cov of one feature is 0-d: reshape it to (1, 1))")
    assert_close(got, want, rtol=1e-9, atol=1e-12,
                 msg="one feature: (m1 - m2)**2 + (s1 - s2)**2 with standard deviations normalised by n - 1")


def test_frechet_distance_from_features_uses_n_minus_1(gm):
    f1 = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    f2 = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0], [0.0, 1.0]])
    want = fd_symmetric(*gaussian_stats(f1), *gaussian_stats(f2))
    got = call(gm.frechet_distance_from_features, *copies(f1, f2))
    assert_close(got, want, rtol=1e-9, atol=1e-12,
                 msg="3 and 4 samples: the covariance is normalised by n - 1 (np.cov(feats, rowvar=False)), "
                     "not by n")


def test_frechet_distance_from_features_same_distribution_is_small(gm):
    gen = np.random.default_rng(91)
    mix = gen.normal(size=(4, 4))
    f1, f2 = gen.normal(size=(4000, 4)) @ mix, gen.normal(size=(4000, 4)) @ mix
    got = call(gm.frechet_distance_from_features, f1, f2)
    scale = np.trace(mix @ mix.T)
    if not 0.0 <= got < 0.02 * scale:
        _fail("", f"expected a small distance (< {0.02 * scale:.3g}) for two large samples of the same "
                  f"distribution, got {got:.6g}")
    shifted = call(gm.frechet_distance_from_features, f1, f2 + np.array([1.0, 0.0, -1.0, 2.0]))
    if not shifted > got + 5.0:
        _fail("", f"expected a distance close to ||(1, 0, -1, 2)||**2 = 6 more after a shift, got {shifted:.6g} "
                  f"(before: {got:.6g})")


@pytest.mark.parametrize("n1, n2, d", [pytest.param(10, 12, 30, id="d30-more-than-n"),
                                       pytest.param(5, 60, 20, id="d20-n5"),
                                       pytest.param(100, 80, 6, id="duplicated-features")])
def test_frechet_distance_from_features_rank_deficient(gm, n1, n2, d):
    """More features than samples (or duplicated features): singular covariances, as for an FID on few images."""
    gen = np.random.default_rng(n1 + n2 + d)
    f1, f2 = gen.normal(size=(n1, d)), gen.normal(0.2, 1.5, size=(n2, d))
    if n1 > d:                                            # duplicated features: column 0 copied into column 1
        f1[:, 1], f2[:, 1] = f1[:, 0], f2[:, 0]
    got = call_quietly(gm.frechet_distance_from_features, *copies(f1, f2))
    assert_python_float(got, "frechet_distance_from_features")
    assert_close(got, fd_symmetric(*gaussian_stats(f1), *gaussian_stats(f2)), rtol=1e-6, atol=1e-6,
                 msg=f"singular covariances ({n1} and {n2} samples of dimension {d})")
    same = call_quietly(gm.frechet_distance_from_features, *copies(f1, f1), expected="0.0")
    assert_python_float(same, "frechet_distance_from_features(X, X)")
    if not 0.0 <= same <= 1e-6:
        _fail("", f"expected 0.0 (or a tiny number >= 0) for the same features, got {same!r} (clip the "
                  "eigenvalues at 0 and a negative result to 0)")


def test_frechet_distance_from_features_accepts_lists(gm):
    f1 = [[0, 0], [2, 0], [0, 2]]
    f2 = [[1, 1], [3, 1], [1, 3]]
    got = call(gm.frechet_distance_from_features, f1, f2, expected="2.0")
    assert_python_float(got, "frechet_distance_from_features of lists")
    assert_close(got, 2.0, rtol=1e-9, atol=1e-12, msg="lists of integers; f2 = f1 + (1, 1)")


def test_frechet_distance_from_features_does_not_modify_its_inputs(gm):
    gen = np.random.default_rng(101)
    arrays = {"feats1": gen.normal(size=(20, 3)), "feats2": gen.normal(size=(15, 3))}
    saved = {k: v.copy() for k, v in arrays.items()}
    call(gm.frechet_distance_from_features, arrays["feats1"], arrays["feats2"])
    assert_unchanged(arrays, saved, "frechet_distance_from_features")


@pytest.mark.parametrize("f1, f2, why", [
    pytest.param(np.ones(5), np.ones((5, 1)), "feats1 of shape (5,): features are 2-D (n, d)", id="feats1-1d"),
    pytest.param(np.ones((4, 2)), np.ones((4, 2, 1)), "feats2 of shape (4, 2, 1) is 3-D", id="feats2-3d"),
    pytest.param(np.ones((4, 2)), np.ones((4, 3)), "2 features against 3", id="dimensions-differ"),
    pytest.param(np.ones((1, 2)), np.ones((4, 2)), "feats1 has a single sample: no covariance", id="n1-is-1"),
    pytest.param(np.ones((4, 2)), np.ones((0, 2)), "feats2 has no sample", id="n2-is-0"),
    pytest.param(np.array([[0.0, 1.0], [np.nan, 2.0], [1.0, 1.0]]), np.ones((3, 2)), "feats1 contains NaN",
                 id="feats1-nan"),
    pytest.param(np.ones((3, 2)), np.array([[0.0, 1.0], [np.inf, 2.0], [1.0, 1.0]]), "feats2 contains inf",
                 id="feats2-inf"),
])
def test_frechet_distance_from_features_refuses_invalid_inputs(gm, f1, f2, why):
    assert_raises(ValueError, gm.frechet_distance_from_features, f1, f2, why=why)
