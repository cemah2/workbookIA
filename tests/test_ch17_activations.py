"""Tests of mylearn.nn.activations (chapter 17): oracle tests and properties.

    pytest tests/test_ch17_activations.py              # your code (mon_travail/mylearn/nn/activations.py)
    pytest tests/test_ch17_activations.py --impl=ref   # the reference

Oracles: PyTorch (``torch.relu``, ``torch.sigmoid``, ``torch.nn.functional.*`` for the values,
``torch.autograd.grad`` for the derivatives, ``torch.autograd.functional.jacobian`` for the Jacobian of
the softmax), SciPy (``scipy.special.expit``, ``erf``, ``softmax``, ``log_softmax``, ``logsumexp``) and
NumPy (``np.heaviside``, ``np.tanh``, ``np.logaddexp``, ``np.expm1``, central finite differences of the
SciPy and NumPy formulas). Without PyTorch, its oracle tests are skipped and the SciPy and NumPy
oracles take over.

The tests are written the way exercise 17.22 asks you to write yours: one parametrized test per
property, run on the whole family of elementwise activations (ids such as ``[relu]``,
``[softplus-beta2]``, ``[gelu-approx_derivative]``), plus the tests of each function. Every test id
contains the name of the function it tests, so that each exercise runs its own group:
``-k "identity or step or relu"`` (17.12), ``-k "sigmoid or tanh or softplus"`` (17.13),
``-k "(elu or silu or gelu or get_activation) and not relu"`` (17.14), ``-k "softmax and not derivative"``
(17.15), ``-k softmax_derivative`` (17.16). The tests of one function never call another function of
your module.

Contract precisions written in the statements of 17.12 to 17.16 and checked here: every function
returns a NEW float64 array (never a view of its input, which it leaves unchanged), of the shape of
``z`` (a 0-d array for a single number); no function emits a RuntimeWarning (overflow, invalid value,
division by zero) and every value stays finite for inputs between -1000 and 1000; at a kink the
derivative is the one PyTorch uses (relu'(0) = 0, leaky'(0) = negative_slope, elu'(0) = alpha); a
step at the threshold takes ``high``; softplus is linear only strictly above the threshold.
The first line of every failure message says what was expected (it is the line that ``pytest -rf``
shows).
"""

import math
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from scipy import special

try:
    import torch
except ImportError:  # the oracle tests run on Colab
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")


@pytest.fixture
def act(mylearn_module):
    return mylearn_module("nn.activations")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for the first line of a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    if arr.dtype.kind in "biuf":
        text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3,
                               max_line_width=10**6)
    else:
        text = repr(arr.tolist())
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


def as_array(result, what="an array"):
    """The result as a float array, or a failure that says what was expected."""
    if result is None:
        _fail("", f"expected {what}, got None (did you forget the return?)")
    try:
        return np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail("", f"expected {what} as a NumPy array, got an object of type {type(result).__name__}")


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg="", z=None):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
    got = as_array(result)
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}")
    ok = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True)
    if not ok.all():
        bad = np.flatnonzero(~ok)
        i = int(bad[0])
        where = ""
        if z is not None:
            zz = np.broadcast_to(np.asarray(z, dtype=float), want.shape).ravel()
            where = f" at z = {zz[i] + 0.0:.10g}"
        head = (f"expected {want.ravel()[i] + 0.0:.12g}{where}, got {got.ravel()[i] + 0.0:.12g} "
                f"({bad.size} wrong value(s) out of {want.size})")
        data = ""
        if want.size <= 12:
            data = f"expected {_short(want, 10)}\n     got {_short(got, 10)}"
        _fail(msg, head, data)


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
    try:
        function(*args, **kwargs)
    except error:
        return
    except NotImplementedError:
        raise                  # shown as "⏳ pas encore implémenté" by conftest.py
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


@contextmanager
def no_runtime_warning(what):
    """Turn every NumPy floating-point warning (overflow, invalid value, division by zero) into a failure."""
    with warnings.catch_warnings(record=True) as caught, np.errstate(over="warn", invalid="warn",
                                                                     divide="warn", under="ignore"):
        warnings.simplefilter("always")
        yield
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    if runtime:
        text = str(runtime[0].message)
        if "overflow" in text:
            hint = ("compute exp only of numbers that cannot overflow: a non-positive number, or z below the "
                    "threshold; np.where computes both branches")
        elif "log" in text or "divide by zero" in text:
            hint = "never take the log of a probability that can round to 0: z - logsumexp(z) instead"
        else:
            hint = "an inf - inf, 0 * inf or 0 / 0 somewhere: subtract the maximum before exp"
        _fail("", f"expected no RuntimeWarning for {what}, got: {text} ({hint})")


def _signature(name, kwargs):
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    return f"{name}(z{', ' + args if args else ''})"


# ------------------------------------------------------------------ trusted formulas (NumPy and SciPy)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _gelu(z, approximate="none"):
    if approximate == "tanh":
        return 0.5 * z * (1.0 + np.tanh(SQRT_2_OVER_PI * (z + 0.044715 * z**3)))
    return 0.5 * z * (1.0 + special.erf(z / math.sqrt(2.0)))


def _gelu_derivative(z, approximate="none"):
    if approximate == "tanh":
        u = SQRT_2_OVER_PI * (z + 0.044715 * z**3)
        t = np.tanh(u)
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t**2) * SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * z**2)
    return 0.5 * (1.0 + special.erf(z / math.sqrt(2.0))) + z * np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


ORACLE = {
    "identity": lambda z: z.copy(),
    "step": lambda z, threshold=0.0, low=0.0, high=1.0: low + (high - low) * np.heaviside(z - threshold, 1.0),
    "relu": lambda z: np.maximum(z, 0.0),
    "leaky_relu": lambda z, negative_slope=0.01: np.where(z > 0, z, negative_slope * z),
    "sigmoid": special.expit,
    "tanh": np.tanh,
    "softplus": lambda z, beta=1.0, threshold=20.0: np.where(beta * z > threshold, z,
                                                             np.logaddexp(0.0, beta * z) / beta),
    "elu": lambda z, alpha=1.0: np.where(z > 0, z, alpha * np.expm1(np.minimum(z, 0.0))),
    "silu": lambda z: z * special.expit(z),
    "gelu": _gelu,
}

ORACLE_DERIVATIVE = {
    "identity": lambda z: np.ones_like(z),
    "step": lambda z, threshold=0.0, low=0.0, high=1.0: np.zeros_like(z),
    "relu": lambda z: (z > 0).astype(float),
    "leaky_relu": lambda z, negative_slope=0.01: np.where(z > 0, 1.0, negative_slope),
    "sigmoid": lambda z: special.expit(z) * special.expit(-z),
    "tanh": lambda z: 1.0 - np.tanh(z) ** 2,
    "softplus": lambda z, beta=1.0, threshold=20.0: np.where(beta * z > threshold, 1.0, special.expit(beta * z)),
    "elu": lambda z, alpha=1.0: np.where(z > 0, 1.0, alpha * np.exp(np.minimum(z, 0.0))),
    "silu": lambda z: special.expit(z) * (1.0 + z * special.expit(-z)),
    "gelu": _gelu_derivative,
}

# where each function has a kink or a jump (no finite difference there)
KINKS = {"step": lambda kw: [kw.get("threshold", 0.0)], "relu": lambda kw: [0.0],
         "leaky_relu": lambda kw: [0.0], "elu": lambda kw: [0.0],
         "softplus": lambda kw: [kw.get("threshold", 20.0) / kw.get("beta", 1.0)]}

# (function name, keyword arguments, test id); no id contains the name of another function
CASES = [
    ("identity", {}, "identity"),
    ("step", {}, "step"),
    ("step", {"threshold": 0.5, "low": -1.0, "high": 2.0}, "step-shifted"),
    ("relu", {}, "relu"),
    ("leaky_relu", {}, "leaky_relu"),
    ("leaky_relu", {"negative_slope": 0.2}, "leaky_relu-slope0.2"),
    ("sigmoid", {}, "sigmoid"),
    ("tanh", {}, "tanh"),
    ("softplus", {}, "softplus"),
    ("softplus", {"beta": 2.0, "threshold": 15.0}, "softplus-beta2"),
    ("softplus", {"beta": 0.5}, "softplus-beta0.5"),
    ("elu", {}, "elu"),
    ("elu", {"alpha": 1.5}, "elu-alpha1.5"),
    ("silu", {}, "silu"),
    ("gelu", {"approximate": "none"}, "gelu-exact"),
    ("gelu", {"approximate": "tanh"}, "gelu-approx"),
]
VALUE_CASES = [pytest.param(n, kw, id=i) for n, kw, i in CASES]
DERIVATIVE_CASES = [pytest.param(n, kw, id=f"{i}_derivative") for n, kw, i in CASES]
BOTH_CASES = ([pytest.param(n, kw, False, id=i) for n, kw, i in CASES]
              + [pytest.param(n, kw, True, id=f"{i}_derivative") for n, kw, i in CASES])

GRID = np.arange(-80, 81) / 10.0                      # -8.0 ... 8.0 with an exact 0.0 (k / 10)
RNG_Z = np.random.default_rng(1712).normal(scale=4.0, size=(6, 9))
EXTREME = np.array([-1000.0, -710.0, -100.0, -40.0, -20.0, -1e-12, 0.0, 1e-12, 20.0, 40.0, 100.0, 710.0, 1000.0])


def _function(act, name, derivative):
    return getattr(act, f"{name}_derivative" if derivative else name)


def _points_away_from_kinks(name, kwargs, z):
    kinks = KINKS.get(name, lambda kw: [])(kwargs)
    keep = np.ones(z.shape, dtype=bool)
    for k in kinks:
        keep &= np.abs(z - k) > 1e-3
    return z[keep]


# ===================================================================== the whole family (17.12 to 17.14)
@pytest.mark.parametrize("name, kwargs", VALUE_CASES)
def test_elementwise_values_match_numpy_and_scipy(act, name, kwargs):
    f = getattr(act, name)
    for z, what in ((GRID, "z from -8 to 8"), (RNG_Z, "a (6, 9) array of random z"),
                    (EXTREME, "very large |z|")):
        expected = ORACLE[name](z.copy(), **kwargs)
        assert_close(call(f, z.copy(), **kwargs), expected, z=z,
                     msg=f"{_signature(name, kwargs)} on {what}, as the NumPy/SciPy formula")


@pytest.mark.parametrize("name, kwargs", DERIVATIVE_CASES)
def test_elementwise_derivative_matches_numpy_and_scipy(act, name, kwargs):
    df = getattr(act, f"{name}_derivative")
    for z, what in ((GRID, "z from -8 to 8 (kinks included)"), (RNG_Z, "a (6, 9) array of random z"),
                    (EXTREME, "very large |z|")):
        expected = ORACLE_DERIVATIVE[name](z.copy(), **kwargs)
        assert_close(call(df, z.copy(), **kwargs), expected, z=z,
                     msg=f"{_signature(name + '_derivative', kwargs)} on {what}: the derivative with respect to "
                         f"the pre-activation z, evaluated at z")


@pytest.mark.parametrize("name, kwargs", DERIVATIVE_CASES)
def test_elementwise_derivative_matches_central_finite_differences(act, name, kwargs):
    df = getattr(act, f"{name}_derivative")
    z = _points_away_from_kinks(name, kwargs, np.concatenate([GRID, RNG_Z.ravel()]))
    h = 1e-5
    f = ORACLE[name]
    numeric = (f(z + h, **kwargs) - f(z - h, **kwargs)) / (2 * h)
    assert_close(call(df, z.copy(), **kwargs), numeric, rtol=1e-6, atol=1e-7, z=z,
                 msg=f"{_signature(name + '_derivative', kwargs)} away from the kinks, as the central difference "
                     f"(f(z + h) - f(z - h)) / 2h of the trusted {name}, h = 1e-5")


@pytest.mark.parametrize("name, kwargs, derivative", BOTH_CASES)
def test_elementwise_returns_a_new_float64_array_of_the_same_shape(act, name, kwargs, derivative):
    f = _function(act, name, derivative)
    label = _signature(name + ("_derivative" if derivative else ""), kwargs)
    for z, what in (([[-2, 0, 3]], "a list of Python integers [[-2, 0, 3]]"),
                    (np.linspace(-3, 3, 24, dtype=np.float32).reshape(2, 3, 4), "a float32 array of shape (2, 3, 4)"),
                    (np.float64(1.5), "a single number np.float64(1.5)")):
        got = call(f, z, **kwargs, expected="an array")
        if not isinstance(got, (np.ndarray, np.floating)):   # 0-d arithmetic gives an np.float64: accepted
            _fail("", f"expected a NumPy array from {label} for {what}, got {type(got).__name__} "
                      f"(start with z = np.asarray(z, dtype=float))")
        if got.dtype != np.float64:
            _fail("", f"expected dtype float64 from {label} for {what}, got {got.dtype} "
                      f"(start with z = np.asarray(z, dtype=float))")
        shape = np.shape(z)
        if got.shape != shape:
            _fail("", f"expected shape {shape} from {label} for {what}, got {got.shape} (an elementwise function "
                      f"keeps the shape of z)")


@pytest.mark.parametrize("name, kwargs, derivative", BOTH_CASES)
def test_elementwise_does_not_modify_or_share_its_input(act, name, kwargs, derivative):
    f = _function(act, name, derivative)
    label = _signature(name + ("_derivative" if derivative else ""), kwargs)
    z = np.random.default_rng(17120).normal(scale=3.0, size=(4, 5))
    before = z.copy()
    got = call(f, z, **kwargs)
    if not np.array_equal(z, before):
        _fail("", f"expected z unchanged by {label} (compute in a new array: no z[...] = ..., no +=, no out=z)")
    if isinstance(got, np.ndarray) and np.shares_memory(got, z):
        _fail("", f"expected a NEW array from {label}, got a view of z (modifying the result would modify z: "
                  f"np.asarray does not copy a float64 array, np.array(z, dtype=float) does)")


@pytest.mark.parametrize("name, kwargs, derivative", BOTH_CASES)
def test_elementwise_large_inputs_stay_finite_without_warnings(act, name, kwargs, derivative):
    f = _function(act, name, derivative)
    label = _signature(name + ("_derivative" if derivative else ""), kwargs)
    with no_runtime_warning(f"{label} on z = ±1000, ±710, ±100, ±40, ±20"):
        got = call(f, EXTREME.copy(), **kwargs)
    got = as_array(got)
    if not np.all(np.isfinite(got)):
        i = int(np.flatnonzero(~np.isfinite(got))[0])
        _fail("", f"expected finite values from {label} for |z| <= 1000, got {got[i]} at z = {EXTREME[i]:g}")


_TORCH = {
    "identity": lambda t: t.clone(),
    "step": lambda t, threshold=0.0, low=0.0, high=1.0: low + (high - low) * torch.heaviside(
        t - threshold, torch.tensor(1.0, dtype=t.dtype)),
    "relu": lambda t: torch.relu(t),
    "leaky_relu": lambda t, negative_slope=0.01: torch.nn.functional.leaky_relu(t, negative_slope),
    "sigmoid": lambda t: torch.sigmoid(t),
    "tanh": lambda t: torch.tanh(t),
    "softplus": lambda t, beta=1.0, threshold=20.0: torch.nn.functional.softplus(t, beta, threshold),
    "elu": lambda t, alpha=1.0: torch.nn.functional.elu(t, alpha),
    "silu": lambda t: torch.nn.functional.silu(t),
    "gelu": lambda t, approximate="none": torch.nn.functional.gelu(t, approximate=approximate),
}


@needs_torch
@pytest.mark.parametrize("name, kwargs", VALUE_CASES)
def test_elementwise_values_match_torch(act, name, kwargs):
    z = np.concatenate([GRID, RNG_Z.ravel(), EXTREME])
    expected = _TORCH[name](torch.from_numpy(z.copy()), **kwargs).numpy()
    assert_close(call(getattr(act, name), z.copy(), **kwargs), expected, z=z,
                 msg=f"{_signature(name, kwargs)} as torch in float64")


@needs_torch
@pytest.mark.parametrize("name, kwargs", [c for c in DERIVATIVE_CASES if c.values[0] != "step"])
def test_elementwise_derivative_matches_torch_autograd(act, name, kwargs):
    z = np.concatenate([GRID, RNG_Z.ravel(), EXTREME])
    t = torch.from_numpy(z.copy()).requires_grad_(True)
    (expected,) = torch.autograd.grad(_TORCH[name](t, **kwargs).sum(), t)
    assert_close(call(getattr(act, f"{name}_derivative"), z.copy(), **kwargs), expected.numpy(), z=z,
                 msg=f"{_signature(name + '_derivative', kwargs)} as torch.autograd.grad (kinks included: "
                     f"PyTorch's convention)")


# ===================================================================== identity, step, relu, leaky_relu (17.12)
def test_identity_docstring_examples(act):
    assert_close(call(act.identity, [-2, 0, 2]), [-2.0, 0.0, 2.0], msg="identity([-2, 0, 2])")
    assert_close(call(act.identity_derivative, [-2.0, 0.0, 2.0]), [1.0, 1.0, 1.0],
                 msg="identity_derivative([-2, 0, 2])")


def test_identity_returns_a_copy_that_can_be_modified(act):
    z = np.array([1.0, 2.0, 3.0])
    got = call(act.identity, z)
    if got is z:
        _fail("", "expected a new array from identity, got the same object z (return np.array(z, dtype=float))")
    out = as_array(got)
    out[0] = 99.0
    if z[0] != 1.0:
        _fail("", "expected z unchanged after modifying identity(z), got z[0] = 99 (identity must copy)")


def test_step_docstring_examples(act):
    assert_close(call(act.step, [-2.0, 0.0, 2.0]), [0.0, 1.0, 1.0],
                 msg="step([-2, 0, 2]): the unit step takes the high value AT the threshold")
    assert_close(call(act.step, [-2.0, 0.0, 2.0], low=-1.0), [-1.0, 1.0, 1.0],
                 msg="step([-2, 0, 2], low=-1): the sign function of the perceptron, sign(0) = +1")
    assert_close(call(act.step_derivative, [-2.0, 0.0, 2.0]), [0.0, 0.0, 0.0], msg="step_derivative([-2, 0, 2])")


@pytest.mark.parametrize("threshold, low, high", [(0.5, 0.0, 1.0), (-1.25, -1.0, 1.0), (2.0, 3.0, -4.0)],
                         ids=["shifted_unit_step", "shifted_sign", "decreasing"])
def test_step_value_at_threshold_and_on_each_side(act, threshold, low, high):
    z = np.array([threshold - 1.0, threshold - 1e-9, threshold, threshold + 1e-9, threshold + 1.0])
    assert_close(call(act.step, z, threshold=threshold, low=low, high=high), [low, low, high, high, high], z=z,
                 msg=f"step(z, threshold={threshold}, low={low}, high={high}): low strictly below the threshold, "
                     f"high at and above it")


def test_relu_docstring_examples_and_kink(act):
    assert_close(call(act.relu, [-2.0, 0.0, 2.0]), [0.0, 0.0, 2.0], msg="relu([-2, 0, 2])")
    assert_close(call(act.relu_derivative, [-2.0, 0.0, 2.0]), [0.0, 0.0, 1.0],
                 msg="relu_derivative([-2, 0, 2]): PyTorch's convention at the kink is relu'(0) = 0")


def test_relu_derivative_is_zero_for_every_non_positive_z(act):
    z = np.array([-1e300, -5.0, -1e-300, -0.0, 0.0, 1e-300, 7.0])
    assert_close(call(act.relu_derivative, z), [0, 0, 0, 0, 0, 1, 1], z=z,
                 msg="relu_derivative: 1 only where z > 0 strictly (0 at z = 0 and at -0.0)")


def test_leaky_relu_docstring_examples_and_kink(act):
    assert_close(call(act.leaky_relu, [-2.0, 0.0, 2.0], negative_slope=0.1), [-0.2, 0.0, 2.0],
                 msg="leaky_relu([-2, 0, 2], negative_slope=0.1)")
    assert_close(call(act.leaky_relu_derivative, [-2.0, 0.0, 2.0], negative_slope=0.1), [0.1, 0.1, 1.0],
                 msg="leaky_relu_derivative([-2, 0, 2], negative_slope=0.1): at the kink, PyTorch takes the slope")


def test_leaky_relu_default_slope_is_one_hundredth(act):
    assert_close(call(act.leaky_relu, [-3.0, 4.0]), [-0.03, 4.0], msg="leaky_relu([-3, 4]) with the default slope 0.01")
    assert_close(call(act.leaky_relu_derivative, [-3.0, 4.0]), [0.01, 1.0],
                 msg="leaky_relu_derivative([-3, 4]) with the default slope 0.01")


def test_leaky_relu_with_slope_one_is_the_identity(act):
    z = np.array([-4.0, -0.5, 0.0, 3.0])
    assert_close(call(act.leaky_relu, z, negative_slope=1.0), z, z=z,
                 msg="leaky_relu(z, negative_slope=1): a slope of 1 on both sides gives the identity (no kink)")


# ===================================================================== sigmoid, tanh, softplus (17.13)
def test_sigmoid_docstring_examples(act):
    assert_close(call(act.sigmoid, [-2.0, 0.0, 2.0]), [0.11920292202211755, 0.5, 0.8807970779778823],
                 msg="sigmoid([-2, 0, 2])")
    with no_runtime_warning("sigmoid([-1000, 0, 1000])"):
        got = call(act.sigmoid, [-1000.0, 0.0, 1000.0])
    assert_close(got, [0.0, 0.5, 1.0], msg="sigmoid([-1000, 0, 1000])")
    assert_close(call(act.sigmoid_derivative, [-2.0, 0.0, 2.0]),
                 [0.10499358540350652, 0.25, 0.10499358540350652], msg="sigmoid_derivative([-2, 0, 2])")


def test_sigmoid_keeps_tiny_probabilities(act):
    z = np.array([-30.0, -100.0, -700.0])
    assert_close(call(act.sigmoid, z), special.expit(z), rtol=1e-6, atol=0.0, z=z,
                 msg="sigmoid of very negative z, with a RELATIVE tolerance: exp(z) / (1 + exp(z)) keeps them; "
                     "1 - 1 / (1 + exp(z)) rounds them to 0")


def test_sigmoid_is_symmetric(act):
    z = np.concatenate([GRID, RNG_Z.ravel()])
    total = as_array(call(act.sigmoid, z)) + as_array(call(act.sigmoid, -z))
    assert_close(total, np.ones_like(z), z=z, msg="sigmoid(z) + sigmoid(-z) = 1 (property)")


def test_sigmoid_derivative_peaks_at_one_quarter(act):
    got = as_array(call(act.sigmoid_derivative, GRID))
    if not np.isclose(got.max(), 0.25, atol=1e-15) or GRID[int(np.argmax(got))] != 0.0:
        _fail("", f"expected the largest value of sigmoid_derivative to be 0.25 at z = 0, got {got.max():.6g} at "
                  f"z = {GRID[int(np.argmax(got))]:g}")
    assert_close(got, got[::-1], msg="sigmoid_derivative(z) = sigmoid_derivative(-z) (an even function)", z=GRID)


def test_tanh_docstring_examples_and_limits(act):
    assert_close(call(act.tanh, [-2.0, 0.0, 2.0]), [-0.9640275800758169, 0.0, 0.9640275800758169],
                 msg="tanh([-2, 0, 2])")
    assert_close(call(act.tanh_derivative, [-2.0, 0.0, 2.0]), [0.07065082485316443, 1.0, 0.07065082485316443],
                 msg="tanh_derivative([-2, 0, 2])")
    with no_runtime_warning("tanh([-1000, 1000])"):
        got = call(act.tanh, [-1000.0, 1000.0])
    assert_close(got, [-1.0, 1.0], msg="tanh([-1000, 1000]): (e^z - e^-z) / (e^z + e^-z) gives inf / inf here")


def test_tanh_is_a_rescaled_sigmoid(act):
    z = np.concatenate([GRID, RNG_Z.ravel()])
    assert_close(call(act.tanh, z), 2.0 * special.expit(2.0 * z) - 1.0, z=z,
                 msg="tanh(z) = 2 sigmoid(2z) - 1 (property, sigmoid from scipy.special.expit)")
    got = as_array(call(act.tanh, z))
    assert_close(got, -as_array(call(act.tanh, -z)), z=z, msg="tanh(-z) = -tanh(z) (an odd function)")


def test_softplus_docstring_examples(act):
    assert_close(call(act.softplus, [-2.0, 0.0, 2.0]), [0.12692801104297263, 0.6931471805599453, 2.1269280110429727],
                 msg="softplus([-2, 0, 2])")
    with no_runtime_warning("softplus([1000.0])"):
        got = call(act.softplus, [1000.0])
    assert_close(got, [1000.0], msg="softplus([1000]): linear above the threshold, without computing exp(1000)")
    assert_close(call(act.softplus_derivative, [-2.0, 0.0, 2.0]), [0.11920292202211755, 0.5, 0.8807970779778823],
                 msg="softplus_derivative([-2, 0, 2]) = sigmoid(z)")


@pytest.mark.parametrize("beta, threshold", [(1.0, 20.0), (2.0, 15.0), (0.5, 4.0)],
                         ids=["default", "beta2", "beta0.5"])
def test_softplus_is_linear_only_strictly_above_the_threshold(act, beta, threshold):
    at = threshold / beta                                  # beta * z == threshold exactly
    z = np.array([at, at + 1.0, at + 50.0])
    expected = np.array([np.logaddexp(0.0, threshold) / beta, at + 1.0, at + 50.0])
    assert_close(call(act.softplus, z, beta=beta, threshold=threshold), expected, rtol=1e-13, atol=0.0, z=z,
                 msg=f"softplus(z, beta={beta}, threshold={threshold}): the smooth formula while beta*z <= threshold "
                     f"(at beta*z = threshold too), z itself only above")
    expected_d = np.array([special.expit(threshold), 1.0, 1.0])
    got_d = as_array(call(act.softplus_derivative, z, beta=beta, threshold=threshold))
    assert_close(got_d, expected_d, rtol=1e-13, atol=0.0, z=z,
                 msg=f"softplus_derivative(z, beta={beta}, threshold={threshold}): sigmoid(beta*z) while "
                     f"beta*z <= threshold, exactly 1 above")


def test_softplus_minus_its_mirror_is_z(act):
    z = np.concatenate([GRID, RNG_Z.ravel()])
    diff = as_array(call(act.softplus, z)) - as_array(call(act.softplus, -z))
    assert_close(diff, z, atol=1e-9, z=z, msg="softplus(z) - softplus(-z) = z (property)")


# ===================================================================== elu, silu, gelu, get_activation (17.14)
def test_elu_docstring_examples_and_kink(act):
    assert_close(call(act.elu, [-2.0, 0.0, 2.0]), [-0.8646647167633873, 0.0, 2.0], msg="elu([-2, 0, 2])")
    assert_close(call(act.elu_derivative, [-2.0, 0.0, 2.0]), [0.1353352832366127, 1.0, 1.0],
                 msg="elu_derivative([-2, 0, 2]); with alpha = 1 both sides give 1 at z = 0")
    assert_close(call(act.elu_derivative, [0.0], alpha=1.5), [1.5],
                 msg="elu_derivative([0], alpha=1.5): PyTorch's convention at z = 0 is alpha (the left-hand side)")


def test_elu_tends_to_minus_alpha(act):
    for alpha in (1.0, 0.3):
        assert_close(call(act.elu, [-1000.0, -50.0], alpha=alpha), [-alpha, -alpha], rtol=1e-12,
                     msg=f"elu([-1000, -50], alpha={alpha}) tends to -alpha")


def test_elu_keeps_precision_near_zero(act):
    z = np.array([-1e-10, -1e-8])
    assert_close(call(act.elu, z), np.expm1(z), rtol=1e-7, atol=0.0, z=z,
                 msg="elu of a tiny negative z, with a RELATIVE tolerance (np.expm1(z) is more precise than "
                     "np.exp(z) - 1)")


def test_silu_docstring_examples(act):
    assert_close(call(act.silu, [-2.0, 0.0, 2.0]), [-0.2384058440442351, 0.0, 1.7615941559557646],
                 msg="silu([-2, 0, 2])")
    assert_close(call(act.silu_derivative, [-2.0, 0.0, 2.0]), [-0.09078424878489121, 0.5, 1.0907842487848914],
                 msg="silu_derivative([-2, 0, 2])")


def test_silu_minus_its_mirror_is_z(act):
    z = np.concatenate([GRID, RNG_Z.ravel()])
    diff = as_array(call(act.silu, z)) - as_array(call(act.silu, -z))
    assert_close(diff, z, atol=1e-9, z=z, msg="silu(z) - silu(-z) = z (property)")


def test_gelu_docstring_examples(act):
    assert_close(call(act.gelu, [-2.0, 0.0, 2.0]), [-0.04550026389635842, 0.0, 1.9544997361036416],
                 msg='gelu([-2, 0, 2]) (approximate="none", with erf)')
    assert_close(call(act.gelu, [-2.0, 0.0, 2.0], approximate="tanh"),
                 [-0.04540230591222698, 0.0, 1.954597694087773],
                 msg='gelu([-2, 0, 2], approximate="tanh")')
    assert_close(call(act.gelu_derivative, [-2.0, 0.0, 2.0]), [-0.08523180107819767, 0.5, 1.0852318010781977],
                 msg='gelu_derivative([-2, 0, 2]) (approximate="none")')
    assert_close(call(act.gelu_derivative, [-2.0, 0.0, 2.0], approximate="tanh"),
                 [-0.08609925660415786, 0.5, 1.0860992566041579],
                 msg='gelu_derivative([-2, 0, 2], approximate="tanh")')


def test_gelu_exact_minus_its_mirror_is_z(act):
    z = np.concatenate([GRID, RNG_Z.ravel()])
    diff = as_array(call(act.gelu, z)) - as_array(call(act.gelu, -z))
    assert_close(diff, z, atol=1e-9, z=z, msg="gelu(z) - gelu(-z) = z, since Phi(-z) = 1 - Phi(z) (property)")


def test_gelu_approximation_is_close_but_not_equal(act):
    exact = as_array(call(act.gelu, GRID))
    approx = as_array(call(act.gelu, GRID, approximate="tanh"))
    gap = np.max(np.abs(exact - approx))
    if not 1e-5 < gap < 1e-3:
        _fail("", f'expected the two versions of GELU to differ by between 1e-5 and 1e-3 on [-8, 8] (the tanh '
                  f'formula is an approximation), got a largest gap of {gap:.3g} (is "approximate" ignored?)')


@pytest.mark.parametrize("approximate, why", [
    pytest.param("exact", 'the exact version is called "none", as in PyTorch', id="word_exact"),
    pytest.param("sigmoid", "an approximation PyTorch does not offer", id="unknown"),
    pytest.param("", "an empty string", id="empty"),
    pytest.param(None, 'None is not the string "none"', id="none_object"),
])
@pytest.mark.parametrize("derivative", [False, True], ids=["gelu", "gelu_derivative"])
def test_gelu_rejects_an_unknown_approximation(act, approximate, why, derivative):
    f = act.gelu_derivative if derivative else act.gelu
    assert_raises(ValueError, f, np.array([0.5, 1.0]), approximate=approximate, why=why)


ALL_NAMES = ["identity", "step", "relu", "leaky_relu", "elu", "softplus", "sigmoid", "tanh", "silu", "gelu"]


def _pair(act, name):
    got = call(act.get_activation, name, expected=f"the pair of functions for {name!r}")
    try:
        f, df = got
    except (TypeError, ValueError):
        _fail("", f"expected a pair (function, derivative) for get_activation({name!r}), got {got!r}")
    return f, df


def test_get_activation_returns_the_module_functions(act):
    for name in ALL_NAMES:
        f, df = _pair(act, name)
        want_f, want_df = getattr(act, name), getattr(act, f"{name}_derivative")
        if f is not want_f or df is not want_df:
            _fail("", f"expected get_activation({name!r}) to be ({name}, {name}_derivative), the functions of the "
                      f"module themselves, got ({getattr(f, '__name__', f)!r}, {getattr(df, '__name__', df)!r})")


def test_get_activation_ignores_case_and_knows_swish(act):
    for name, target in (("ReLU", "relu"), ("GELU", "gelu"), ("Tanh", "tanh"), ("LEAKY_RELU", "leaky_relu"),
                         ("swish", "silu"), ("Swish", "silu"), ("SiLU", "silu")):
        f, df = _pair(act, name)
        if f is not getattr(act, target) or df is not getattr(act, f"{target}_derivative"):
            _fail("", f"expected get_activation({name!r}) to be ({target}, {target}_derivative) (the name is "
                      f"case-insensitive, swish is an alias of silu), got ({getattr(f, '__name__', f)!r}, "
                      f"{getattr(df, '__name__', df)!r})")


@pytest.mark.parametrize("name, why", [
    pytest.param("relu6", "an activation the module does not have", id="unknown_variant"),
    pytest.param("softmax", "the softmax is not an elementwise activation (no pair f, f')", id="not_elementwise"),
    pytest.param("", "an empty name", id="empty"),
    pytest.param("sigmoïde", "the French spelling is not a valid name", id="accented"),
])
def test_get_activation_rejects_unknown_names(act, name, why):
    assert_raises(ValueError, act.get_activation, name, why=why)


def test_get_activation_error_lists_the_valid_names(act):
    try:
        act.get_activation("rectifier")
    except NotImplementedError:
        raise
    except ValueError as exc:
        text = str(exc).lower()
        missing = [n for n in ALL_NAMES if n not in text]
        if missing:
            _fail("", f"expected the ValueError message to list the valid names, got {str(exc)!r} "
                      f"(missing: {', '.join(missing)})")
        return
    except Exception as exc:  # noqa: BLE001
        _fail("", f"expected a ValueError for get_activation('rectifier'), got {type(exc).__name__}: {exc}")
    _fail("", "expected a ValueError for get_activation('rectifier'), got no error")


# ===================================================================== softmax and log_softmax (17.15)
def test_softmax_docstring_examples(act):
    assert_close(call(act.softmax, [1.0, 2.0, 3.0]), [0.09003057317038046, 0.24472847105479764, 0.6652409557748219],
                 msg="softmax([1, 2, 3])")
    with no_runtime_warning("softmax([1001, 1002, 1003])"):
        got = call(act.softmax, [1001.0, 1002.0, 1003.0])
    assert_close(got, [0.09003057317038046, 0.24472847105479764, 0.6652409557748219],
                 msg="softmax([1001, 1002, 1003]): subtract the max first, or exp(1001) overflows")


@pytest.mark.parametrize("shape, axis", [((5,), -1), ((4, 6), -1), ((4, 6), 1), ((4, 6), 0), ((2, 3, 4), 1),
                                         ((2, 3, 4), 0)],
                         ids=["1d", "2d_last_axis", "2d_axis1", "2d_axis0", "3d_axis1", "3d_axis0"])
def test_softmax_matches_scipy(act, shape, axis):
    z = np.random.default_rng(1715).normal(scale=3.0, size=shape)
    assert_close(call(act.softmax, z.copy(), axis=axis), special.softmax(z, axis=axis),
                 msg=f"softmax of an array of shape {shape} along axis={axis}, as scipy.special.softmax")


@pytest.mark.parametrize("shape, axis", [((5,), -1), ((4, 6), -1), ((4, 6), 0), ((2, 3, 4), 1)],
                         ids=["1d", "2d_last_axis", "2d_axis0", "3d_axis1"])
def test_log_softmax_matches_scipy(act, shape, axis):
    z = np.random.default_rng(17151).normal(scale=3.0, size=shape)
    assert_close(call(act.log_softmax, z.copy(), axis=axis), special.log_softmax(z, axis=axis),
                 msg=f"log_softmax of an array of shape {shape} along axis={axis}, as scipy.special.log_softmax")


@needs_torch
@pytest.mark.parametrize("axis", [-1, 0], ids=["last_axis", "axis0"])
def test_softmax_and_log_softmax_match_torch(act, axis):
    z = np.random.default_rng(17152).normal(scale=5.0, size=(7, 4))
    t = torch.from_numpy(z.copy())
    assert_close(call(act.softmax, z.copy(), axis=axis), torch.softmax(t, dim=axis).numpy(),
                 msg=f"softmax along axis={axis}, as torch.softmax(z, dim={axis})")
    assert_close(call(act.log_softmax, z.copy(), axis=axis), torch.log_softmax(t, dim=axis).numpy(),
                 msg=f"log_softmax along axis={axis}, as torch.log_softmax(z, dim={axis})")


def test_softmax_sums_to_one_along_the_axis_only(act):
    z = np.random.default_rng(17153).normal(size=(3, 5))
    p = as_array(call(act.softmax, z.copy(), axis=0))
    if p.shape == (3, 5) and not np.allclose(p.sum(axis=0), 1.0, atol=1e-12):
        _fail("", f"expected each COLUMN to sum to 1 with axis=0, got column sums {_short(p.sum(axis=0))} "
                  f"(use axis=axis and keepdims=True in max and sum)")
    p = as_array(call(act.softmax, z.copy()))
    if p.shape == (3, 5) and (not np.allclose(p.sum(axis=1), 1.0, atol=1e-12) or p.min() < 0):
        _fail("", f"expected each ROW to sum to 1 with the default axis=-1, got row sums {_short(p.sum(axis=1))}")


def test_softmax_ignores_a_shift_of_each_row(act):
    z = np.random.default_rng(17154).normal(size=(4, 3))
    shift = np.array([[1000.0], [-1000.0], [0.5], [-3.0]])
    with no_runtime_warning("softmax of rows shifted by ±1000"):
        shifted = call(act.softmax, z + shift)
    assert_close(shifted, special.softmax(z, axis=-1),
                 msg="softmax(z + c) = softmax(z) for a constant c per row (property), here c = ±1000")


def test_softmax_keeps_the_order_of_the_scores(act):
    z = np.array([0.3, -1.2, 2.5, 0.31, 1.0])
    p = as_array(call(act.softmax, z))
    if p.shape == z.shape and list(np.argsort(p)) != list(np.argsort(z)):
        _fail("", f"expected the probabilities in the same order as the scores {list(np.argsort(z))} "
                  f"(np.argsort), got {list(np.argsort(p))}")


def test_softmax_of_very_negative_logits_is_not_nan(act):
    with no_runtime_warning("softmax([-1000, -1001, -1003])"):
        got = call(act.softmax, [-1000.0, -1001.0, -1003.0])
    assert_close(got, special.softmax([-1000.0, -1001.0, -1003.0]),
                 msg="softmax([-1000, -1001, -1003]): without subtracting the max, every exp underflows to 0 "
                     "and 0 / 0 gives nan")


def test_softmax_of_a_single_class_is_one(act):
    assert_close(call(act.softmax, [[3.0], [-7.0]]), [[1.0], [1.0]], msg="softmax of one class per row")
    assert_close(call(act.log_softmax, [[3.0], [-7.0]]), [[0.0], [0.0]], msg="log_softmax of one class per row")


def test_log_softmax_docstring_examples(act):
    assert_close(call(act.log_softmax, [1.0, 2.0, 3.0]), [-2.4076059644443806, -1.4076059644443806,
                                                          -0.40760596444438063], msg="log_softmax([1, 2, 3])")
    with no_runtime_warning("log_softmax([0, 10000])"):
        got = call(act.log_softmax, [0.0, 10000.0])
    assert_close(got, [-10000.0, 0.0],
                 msg="log_softmax([0, 10000]): z - logsumexp(z), never log(softmax(z)) (the softmax of 0 is 0 "
                     "here, and log(0) = -inf)")


def test_log_softmax_is_consistent_with_the_softmax(act):
    z = np.random.default_rng(17155).normal(scale=4.0, size=(5, 6))
    logp = as_array(call(act.log_softmax, z.copy()))
    if logp.shape == z.shape:
        if logp.max() > 1e-15:
            _fail("", f"expected log-probabilities <= 0, got a largest value of {logp.max():.3g}")
        assert_close(np.exp(logp), special.softmax(z, axis=-1),
                     msg="exp(log_softmax(z)) = softmax(z) (property, softmax from scipy)")
        assert_close(special.logsumexp(logp, axis=-1), np.zeros(5), atol=1e-12,
                     msg="logsumexp(log_softmax(z)) = 0 on each row (the probabilities sum to 1)")


def test_log_softmax_of_large_logits_without_warnings(act):
    z = np.array([[0.0, 5000.0, -5000.0], [1e4, 1e4, 0.0]])
    with no_runtime_warning("log_softmax of logits of order 1e4"):
        got = call(act.log_softmax, z)
    assert_close(got, special.log_softmax(z, axis=-1), msg="log_softmax of logits of order 1e4, as scipy")


@pytest.mark.parametrize("z, why", [pytest.param(3.0, "a Python float", id="float"),
                                    pytest.param(np.float64(-1.0), "a NumPy scalar", id="numpy_scalar"),
                                    pytest.param(np.array(2.0), "a 0-d array", id="zero_d_array")])
@pytest.mark.parametrize("which", ["softmax", "log_softmax"])
def test_softmax_and_log_softmax_reject_a_single_number(act, z, why, which):
    assert_raises(ValueError, getattr(act, which), z, why=f"{why} has no axis of classes")


@pytest.mark.parametrize("which", ["softmax", "log_softmax"])
def test_softmax_and_log_softmax_return_a_new_float64_array(act, which):
    f = getattr(act, which)
    got = call(f, [[1, 2, 3]])
    if not isinstance(got, np.ndarray) or got.dtype != np.float64 or got.shape != (1, 3):
        _fail("", f"expected a float64 array of shape (1, 3) from {which}([[1, 2, 3]]), got "
                  f"{type(got).__name__} of dtype {getattr(got, 'dtype', '?')} and shape {getattr(got, 'shape', '?')}")
    z = np.random.default_rng(17156).normal(size=(3, 4))
    before = z.copy()
    out = call(f, z)
    if not np.array_equal(z, before):
        _fail("", f"expected z unchanged by {which} (no z -= z.max(...): compute z - m in a new array)")
    if isinstance(out, np.ndarray) and np.shares_memory(out, z):
        _fail("", f"expected a new array from {which}, got a view of z")


# ===================================================================== softmax_derivative (17.16)
def _numeric_jacobian(z, h=1e-6):
    K = z.shape[-1]
    J = np.empty(z.shape[:-1] + (K, K))
    for j in range(K):
        e = np.zeros(K)
        e[j] = h
        J[..., :, j] = (special.softmax(z + e, axis=-1) - special.softmax(z - e, axis=-1)) / (2 * h)
    return J


def test_softmax_derivative_docstring_examples(act):
    assert_close(call(act.softmax_derivative, [0.0, 0.0]), [[0.25, -0.25], [-0.25, 0.25]],
                 msg="softmax_derivative([0, 0])")
    got = as_array(call(act.softmax_derivative, np.zeros((4, 3))))
    if got.shape != (4, 3, 3):
        _fail("", f"expected shape (4, 3, 3) for 4 samples of 3 logits (one K x K matrix per sample), got {got.shape}")


@pytest.mark.parametrize("shape", [(3,), (5,), (6, 4)], ids=["K3", "K5", "batch6_K4"])
def test_softmax_derivative_matches_central_finite_differences(act, shape):
    z = np.random.default_rng(1716).normal(scale=2.0, size=shape)
    assert_close(call(act.softmax_derivative, z.copy()), _numeric_jacobian(z), rtol=1e-6, atol=1e-8,
                 msg=f"Jacobian for logits of shape {shape}: entry [i, j] is dp_i/dz_j, as the central differences "
                     f"of scipy.special.softmax (column j: z_j moved by ±h)")


@needs_torch
@pytest.mark.parametrize("shape", [(4,), (3, 5)], ids=["one_sample", "batch"])
def test_softmax_derivative_matches_torch_jacobian(act, shape):
    z = np.random.default_rng(17161).normal(scale=2.0, size=shape)
    rows = z.reshape(-1, shape[-1])
    expected = np.stack([torch.autograd.functional.jacobian(lambda t: torch.softmax(t, -1),
                                                            torch.from_numpy(r.copy())).numpy() for r in rows])
    expected = expected.reshape(shape + (shape[-1],))
    assert_close(call(act.softmax_derivative, z.copy()), expected,
                 msg=f"Jacobian for logits of shape {shape}, as torch.autograd.functional.jacobian of the softmax")


def test_softmax_derivative_is_symmetric_with_rows_summing_to_zero(act):
    z = np.random.default_rng(17162).normal(scale=3.0, size=(5, 6))
    J = as_array(call(act.softmax_derivative, z.copy()))
    if J.shape != (5, 6, 6):
        _fail("", f"expected shape (5, 6, 6), got {J.shape}")
    assert_close(J, np.swapaxes(J, -1, -2), msg="J[i, j] = J[j, i]: each Jacobian is symmetric (property)")
    assert_close(J.sum(axis=-1), np.zeros((5, 6)), atol=1e-12,
                 msg="each row of the Jacobian sums to 0: the probabilities always sum to 1 (property)")
    p = special.softmax(z, axis=-1)
    assert_close(np.diagonal(J, axis1=-2, axis2=-1), p * (1 - p),
                 msg="the diagonal J[i, i] = p_i (1 - p_i), as for a sigmoid")


def test_softmax_derivative_of_large_logits_without_warnings(act):
    z = np.array([1000.0, 0.0, -1000.0])
    with no_runtime_warning("softmax_derivative([1000, 0, -1000])"):
        got = call(act.softmax_derivative, z)
    assert_close(got, np.zeros((3, 3)), atol=1e-12,
                 msg="softmax_derivative([1000, 0, -1000]): p = (1, 0, 0), so every entry p_i (delta_ij - p_j) is 0")


@pytest.mark.parametrize("z, why", [pytest.param(np.array(1.5), "a single number", id="zero_d"),
                                    pytest.param(np.zeros((2, 3, 4)), "a 3-D array", id="three_d")])
def test_softmax_derivative_rejects_other_dimensions(act, z, why):
    assert_raises(ValueError, act.softmax_derivative, z, why=f"{why}: logits must have shape (K,) or (n, K)")


def test_softmax_derivative_does_not_modify_its_input(act):
    z = np.random.default_rng(17163).normal(size=(2, 3))
    before = z.copy()
    call(act.softmax_derivative, z)
    if not np.array_equal(z, before):
        _fail("", "expected z unchanged by softmax_derivative (compute z - m in a new array)")
