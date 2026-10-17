"""Tests of mylearn.nn.backward (chapter 18): oracle tests and properties.

    pytest tests/test_ch18_backward.py              # your code (mon_travail/mylearn/nn/backward.py)
    pytest tests/test_ch18_backward.py --impl=ref   # the reference

Oracles: PyTorch (``torch.nn.functional.mse_loss``, ``binary_cross_entropy_with_logits``,
``cross_entropy`` and ``linear``, ``torch.nn.Sequential`` with copied weights, and autograd for every
gradient), SciPy (``scipy.special.expit``, ``softmax``, ``log_softmax``), NumPy (``np.logaddexp``, a forward
pass written in this file, and central finite differences of NumPy formulas). Without PyTorch, its oracle
tests are skipped and the SciPy and NumPy oracles take over.

Every test name starts with the name of the function it tests, so that each exercise runs its own
group: ``-k test_mse_loss_`` (18.12), ``-k "test_binary_cross_entropy or test_softmax_cross_entropy"``
(18.14), ``-k test_dense_backward_`` (18.15), ``-k test_gradient_check_`` (18.16),
``-k test_mlp_backward_`` (18.18). The tests of one function never call another function of your module
(the caches of ``mlp_backward`` are built here by a NumPy forward pass).

Contract precisions written in the statements of 18.12 to 18.18 and checked here: every loss returns a
Python float (or a NumPy float) and a NEW float64 gradient of the shape of its first argument; no input
is modified; no broadcasting between the prediction and the target (shapes (3,) and (3, 1) are refused);
the losses stay finite, without RuntimeWarning, for logits of +/-1000 (binary) and 1e4 (softmax);
``softmax_cross_entropy`` refuses a label equal to K or to -1 (NumPy would accept -1 silently);
``gradient_check`` uses the centred difference with the given ``eps``, perturbs ``n_checks`` distinct
entries when asked, and restores ``param`` bit for bit.
The first line of every failure message says what was expected (it is the line that ``pytest -rf``
shows).
"""

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
def bw(mylearn_module):
    return mylearn_module("nn.backward")


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


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg=""):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
    got = as_array(result)
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}")
    ok = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True)
    if not ok.all():
        bad = np.flatnonzero(~ok)
        i = int(bad[0])
        index = np.unravel_index(i, want.shape) if want.ndim > 1 else (i,)
        where = f" at index {tuple(int(k) for k in index)}" if want.ndim else ""
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
        _fail("", f"expected no RuntimeWarning for {what}, got: {runtime[0].message} "
                  "(an exponential or a log of 0 somewhere: compute from the logits, as the docstring says)")


def loss_pair(result, name, grad_shape):
    """Unpack (loss, grad): a scalar float and a float64 array of the expected shape."""
    if result is None:
        _fail("", f"expected (loss, grad) from {name}, got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 2:
        _fail("", f"expected a tuple (loss, grad) from {name}, got {type(result).__name__}"
                  + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else ""))
    loss, grad = result
    if not isinstance(loss, (float, np.floating)) or np.ndim(loss) != 0:
        _fail("", f"expected the loss as a float from {name}, got {type(loss).__name__} "
                  "(convert with float(...))")
    if not isinstance(grad, np.ndarray):
        _fail("", f"expected the gradient as a NumPy array from {name}, got {type(grad).__name__}")
    if grad.dtype != np.float64:
        _fail("", f"expected a float64 gradient from {name}, got dtype {grad.dtype}")
    if grad.shape != tuple(grad_shape):
        _fail("", f"expected a gradient of shape {tuple(grad_shape)} from {name}, got {grad.shape} "
                  "(the gradient has the shape of the first argument)")
    return float(loss), grad


def torch_tensor(values, grad=False):
    return torch.tensor(np.asarray(values, dtype=np.float64), dtype=torch.float64, requires_grad=grad)


RNG = np.random.default_rng(1800)


# ================================================================== mse_loss (18.12)
MSE_SHAPES = [(1,), (5,), (4, 1), (3, 2), (2, 3, 2)]


@pytest.mark.parametrize("shape", MSE_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_mse_loss_matches_numpy(bw, shape):
    rng = np.random.default_rng(1801 + len(shape))
    y_pred, y_true = rng.normal(size=shape), rng.normal(size=shape)
    loss, grad = loss_pair(call(bw.mse_loss, y_pred.copy(), y_true.copy(), expected="(loss, grad)"),
                           "mse_loss", shape)
    assert_close(loss, np.mean((y_pred - y_true) ** 2), msg="mse_loss value (mean over ALL elements)")
    assert_close(grad, 2.0 * (y_pred - y_true) / y_pred.size,
                 msg="mse_loss gradient: 2 (y_pred - y_true) / y_pred.size")


@needs_torch
@pytest.mark.parametrize("shape", MSE_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_mse_loss_matches_torch_autograd(bw, shape):
    rng = np.random.default_rng(1811 + len(shape))
    y_pred, y_true = rng.normal(size=shape), rng.normal(size=shape)
    p = torch_tensor(y_pred, grad=True)
    expected = torch.nn.functional.mse_loss(p, torch_tensor(y_true))
    expected.backward()
    loss, grad = loss_pair(call(bw.mse_loss, y_pred.copy(), y_true.copy(), expected="(loss, grad)"),
                           "mse_loss", shape)
    assert_close(loss, expected.item(), msg="mse_loss value against torch.nn.functional.mse_loss")
    assert_close(grad, p.grad.numpy(), msg="mse_loss gradient against torch autograd")


def test_mse_loss_docstring_example(bw):
    loss, grad = loss_pair(call(bw.mse_loss, [1.0, 2.0], [0.0, 0.0], expected="(loss, grad)"), "mse_loss", (2,))
    assert_close(loss, 2.5, msg="mse_loss([1, 2], [0, 0])")
    assert_close(grad, [1.0, 2.0], msg="gradient of mse_loss([1, 2], [0, 0])")


def test_mse_loss_accepts_lists_of_integers(bw):
    loss, grad = loss_pair(call(bw.mse_loss, [3, 1, 2], [1, 1, 1], expected="(loss, grad)"), "mse_loss", (3,))
    assert_close(loss, 5 / 3, msg="mse_loss on lists of integers")
    assert_close(grad, [4 / 3, 0.0, 2 / 3], msg="gradient on lists of integers (a float64 array)")


@pytest.mark.parametrize("pred_shape, true_shape", [
    pytest.param((3,), (3, 1), id="3_vs_3x1-would-broadcast-to-3x3"),
    pytest.param((3, 1), (3,), id="3x1_vs_3"),
    pytest.param((2,), (3,), id="2_vs_3"),
])
def test_mse_loss_refuses_different_shapes(bw, pred_shape, true_shape):
    assert_raises(ValueError, bw.mse_loss, np.zeros(pred_shape), np.ones(true_shape),
                  why=f"y_pred of shape {pred_shape} and y_true of shape {true_shape} (no broadcasting)")


def test_mse_loss_does_not_modify_its_inputs(bw):
    y_pred, y_true = RNG.normal(size=(4, 2)), RNG.normal(size=(4, 2))
    before = (y_pred.copy(), y_true.copy())
    call(bw.mse_loss, y_pred, y_true)
    if not (np.array_equal(y_pred, before[0]) and np.array_equal(y_true, before[1])):
        _fail("", "expected y_pred and y_true unchanged after mse_loss, but they were modified in place")


def test_mse_loss_gradient_is_a_new_array(bw):
    y_pred, y_true = np.array([1.0, 2.0, 3.0]), np.zeros(3)
    _, grad = loss_pair(call(bw.mse_loss, y_pred, y_true), "mse_loss", (3,))
    if np.shares_memory(grad, y_pred) or np.shares_memory(grad, y_true):
        _fail("", "expected a NEW gradient array, got a view of an input (it would change with the input)")


def test_mse_loss_gradient_matches_finite_differences(bw):
    y_pred, y_true = RNG.normal(size=6), RNG.normal(size=6)
    _, grad = loss_pair(call(bw.mse_loss, y_pred.copy(), y_true.copy()), "mse_loss", (6,))
    eps = 1e-6
    numeric = np.array([(np.mean((y_pred + eps * e - y_true) ** 2) - np.mean((y_pred - eps * e - y_true) ** 2))
                        / (2 * eps) for e in np.eye(6)])
    assert_close(grad, numeric, rtol=1e-6, atol=1e-9, msg="mse_loss gradient against central differences")


# ================================================================== binary cross-entropy (18.14)
def bce_oracle(z, y):
    z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean(np.logaddexp(0.0, z) - y * z)), (special.expit(z) - y) / z.size


BCE_CASES = [
    pytest.param((8,), "binary", id="8-labels01"),
    pytest.param((6, 1), "binary", id="6x1-labels01"),
    pytest.param((7,), "soft", id="7-soft-targets"),
    pytest.param((1,), "binary", id="1-single-sample"),
]


def _bce_data(shape, kind, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=shape)
    y = rng.integers(0, 2, size=shape).astype(float) if kind == "binary" else rng.uniform(0, 1, size=shape)
    return z, y


@pytest.mark.parametrize("shape, kind", BCE_CASES)
def test_binary_cross_entropy_with_logits_matches_numpy(bw, shape, kind):
    z, y = _bce_data(shape, kind, 1821)
    loss, grad = loss_pair(call(bw.binary_cross_entropy_with_logits, z.copy(), y.copy(), expected="(loss, grad)"),
                           "binary_cross_entropy_with_logits", shape)
    want_loss, want_grad = bce_oracle(z, y)
    assert_close(loss, want_loss, msg="BCE value: mean of log(1 + e^z) - y z (natural log)")
    assert_close(grad, want_grad, msg="BCE gradient: (sigmoid(z) - y) / n")


@needs_torch
@pytest.mark.parametrize("shape, kind", BCE_CASES)
def test_binary_cross_entropy_with_logits_matches_torch_autograd(bw, shape, kind):
    z, y = _bce_data(shape, kind, 1822)
    zt = torch_tensor(z, grad=True)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(zt, torch_tensor(y))
    expected.backward()
    loss, grad = loss_pair(call(bw.binary_cross_entropy_with_logits, z.copy(), y.copy(), expected="(loss, grad)"),
                           "binary_cross_entropy_with_logits", shape)
    assert_close(loss, expected.item(), msg="BCE value against torch binary_cross_entropy_with_logits")
    assert_close(grad, zt.grad.numpy(), msg="BCE gradient against torch autograd")


def test_binary_cross_entropy_with_logits_docstring_examples(bw):
    loss, grad = loss_pair(call(bw.binary_cross_entropy_with_logits, [0.0, 0.0], [1.0, 0.0]),
                           "binary_cross_entropy_with_logits", (2,))
    assert_close(loss, np.log(2.0), msg="BCE of logits [0, 0] with targets [1, 0]")
    assert_close(grad, [-0.25, 0.25], msg="BCE gradient of logits [0, 0] with targets [1, 0]")
    loss, _ = loss_pair(call(bw.binary_cross_entropy_with_logits, [1000.0, -1000.0], [0.0, 0.0]),
                        "binary_cross_entropy_with_logits", (2,))
    assert_close(loss, 500.0, msg="BCE of logits [1000, -1000] with targets [0, 0] (the first costs 1000 nats)")


@pytest.mark.parametrize("y, expected_loss, expected_grad", [
    pytest.param([0.0, 0.0], 500.0, [0.5, 0.0], id="z+-1000-targets00"),
    pytest.param([1.0, 1.0], 500.0, [0.0, -0.5], id="z+-1000-targets11"),
    pytest.param([1.0, 0.0], 0.0, [0.0, 0.0], id="z+-1000-perfect"),
    pytest.param([0.0, 1.0], 1000.0, [0.5, -0.5], id="z+-1000-confidently-wrong"),
])
def test_binary_cross_entropy_with_logits_is_finite_for_huge_logits(bw, y, expected_loss, expected_grad):
    z = np.array([1000.0, -1000.0])
    with no_runtime_warning("logits of +1000 and -1000"):
        loss, grad = loss_pair(call(bw.binary_cross_entropy_with_logits, z, np.array(y)),
                               "binary_cross_entropy_with_logits", (2,))
    assert_close(loss, expected_loss, msg=f"BCE of logits [1000, -1000] with targets {y} (never log(0))")
    assert_close(grad, expected_grad, msg=f"BCE gradient for logits [1000, -1000] and targets {y}")


@pytest.mark.parametrize("bad", [
    pytest.param(1.5, id="target-1.5"),
    pytest.param(-0.1, id="target--0.1"),
    pytest.param(2.0, id="target-2-as-a-class-number"),
])
def test_binary_cross_entropy_with_logits_refuses_targets_outside_0_1(bw, bad):
    assert_raises(ValueError, bw.binary_cross_entropy_with_logits, np.zeros(3), np.array([0.0, 1.0, bad]),
                  why=f"a target equal to {bad} (targets lie in [0, 1])")


@pytest.mark.parametrize("z_shape, y_shape", [
    pytest.param((3,), (3, 1), id="3_vs_3x1-would-broadcast"),
    pytest.param((4,), (3,), id="4_vs_3"),
])
def test_binary_cross_entropy_with_logits_refuses_different_shapes(bw, z_shape, y_shape):
    assert_raises(ValueError, bw.binary_cross_entropy_with_logits, np.zeros(z_shape), np.zeros(y_shape),
                  why=f"logits of shape {z_shape} and targets of shape {y_shape} (no broadcasting)")


def test_binary_cross_entropy_with_logits_does_not_modify_its_inputs(bw):
    z, y = _bce_data((5,), "binary", 1823)
    before = (z.copy(), y.copy())
    call(bw.binary_cross_entropy_with_logits, z, y)
    if not (np.array_equal(z, before[0]) and np.array_equal(y, before[1])):
        _fail("", "expected logits and targets unchanged, but they were modified in place")


def test_binary_cross_entropy_with_logits_gradient_matches_finite_differences(bw):
    z, y = _bce_data((5,), "soft", 1824)
    _, grad = loss_pair(call(bw.binary_cross_entropy_with_logits, z.copy(), y.copy()),
                        "binary_cross_entropy_with_logits", (5,))
    eps = 1e-6
    numeric = np.array([(bce_oracle(z + eps * e, y)[0] - bce_oracle(z - eps * e, y)[0]) / (2 * eps)
                        for e in np.eye(5)])
    assert_close(grad, numeric, rtol=1e-6, atol=1e-9, msg="BCE gradient against central differences")


# ================================================================== softmax cross-entropy (18.14)
def ce_oracle(z, Y):
    z, Y = np.asarray(z, dtype=float), np.asarray(Y, dtype=float)
    n = z.shape[0]
    return -float(np.sum(Y * special.log_softmax(z, axis=1))) / n, (special.softmax(z, axis=1) - Y) / n


def one_hot(labels, k):
    out = np.zeros((len(labels), k))
    out[np.arange(len(labels)), labels] = 1.0
    return out


CE_CASES = [pytest.param(6, 3, id="6x3"), pytest.param(1, 4, id="1x4"), pytest.param(5, 2, id="5x2"),
            pytest.param(9, 10, id="9x10")]


@pytest.mark.parametrize("n, k", CE_CASES)
def test_softmax_cross_entropy_with_labels_matches_scipy(bw, n, k):
    rng = np.random.default_rng(1831 + n)
    z, labels = rng.normal(scale=2.0, size=(n, k)), rng.integers(0, k, size=n)
    loss, grad = loss_pair(call(bw.softmax_cross_entropy, z.copy(), labels.copy(), expected="(loss, grad)"),
                           "softmax_cross_entropy", (n, k))
    want_loss, want_grad = ce_oracle(z, one_hot(labels, k))
    assert_close(loss, want_loss, msg="cross-entropy with integer labels: -mean(log softmax(z)[i, y_i])")
    assert_close(grad, want_grad, msg="cross-entropy gradient: (softmax(z) - one_hot(y)) / n")


@pytest.mark.parametrize("n, k", CE_CASES)
def test_softmax_cross_entropy_with_probabilities_matches_scipy(bw, n, k):
    rng = np.random.default_rng(1841 + n)
    z = rng.normal(scale=2.0, size=(n, k))
    Y = special.softmax(rng.normal(size=(n, k)), axis=1)
    loss, grad = loss_pair(call(bw.softmax_cross_entropy, z.copy(), Y.copy(), expected="(loss, grad)"),
                           "softmax_cross_entropy", (n, k))
    want_loss, want_grad = ce_oracle(z, Y)
    assert_close(loss, want_loss, msg="cross-entropy with target probabilities: -mean(sum Y log softmax(z))")
    assert_close(grad, want_grad, msg="cross-entropy gradient with probabilities: (softmax(z) - Y) / n")


@needs_torch
@pytest.mark.parametrize("target", ["labels", "probabilities"])
@pytest.mark.parametrize("n, k", CE_CASES)
def test_softmax_cross_entropy_matches_torch_autograd(bw, n, k, target):
    rng = np.random.default_rng(1851 + n + k)
    z = rng.normal(scale=2.0, size=(n, k))
    y = rng.integers(0, k, size=n) if target == "labels" else special.softmax(rng.normal(size=(n, k)), axis=1)
    zt = torch_tensor(z, grad=True)
    yt = torch.tensor(y) if target == "labels" else torch_tensor(y)
    expected = torch.nn.functional.cross_entropy(zt, yt)
    expected.backward()
    loss, grad = loss_pair(call(bw.softmax_cross_entropy, z.copy(), y.copy(), expected="(loss, grad)"),
                           "softmax_cross_entropy", (n, k))
    assert_close(loss, expected.item(), msg=f"cross-entropy ({target}) against torch.nn.functional.cross_entropy")
    assert_close(grad, zt.grad.numpy(), msg=f"cross-entropy gradient ({target}) against torch autograd")


def test_softmax_cross_entropy_docstring_examples(bw):
    loss, grad = loss_pair(call(bw.softmax_cross_entropy, [[0.0, 0.0]], [0]), "softmax_cross_entropy", (1, 2))
    assert_close(loss, np.log(2.0), msg="cross-entropy of logits [[0, 0]] with label 0")
    assert_close(grad, [[-0.5, 0.5]], msg="gradient for logits [[0, 0]] and label 0")
    loss, grad = loss_pair(call(bw.softmax_cross_entropy, [[0.0, 0.0]], [[0.25, 0.75]]),
                           "softmax_cross_entropy", (1, 2))
    assert_close(loss, np.log(2.0), msg="cross-entropy of logits [[0, 0]] with target probabilities [0.25, 0.75]")
    assert_close(grad, [[0.25, -0.25]], msg="gradient for logits [[0, 0]] and probabilities [0.25, 0.75]")


def test_softmax_cross_entropy_one_hot_rows_equal_integer_labels(bw):
    z, labels = RNG.normal(size=(5, 4)), np.array([3, 0, 1, 1, 2])
    a = loss_pair(call(bw.softmax_cross_entropy, z.copy(), labels), "softmax_cross_entropy", (5, 4))
    b = loss_pair(call(bw.softmax_cross_entropy, z.copy(), one_hot(labels, 4)), "softmax_cross_entropy", (5, 4))
    assert_close(b[0], a[0], msg="the same loss with one-hot rows as with integer labels")
    assert_close(b[1], a[1], msg="the same gradient with one-hot rows as with integer labels")


def test_softmax_cross_entropy_is_finite_for_logits_of_1e4(bw):
    z = np.array([[1e4, 0.0, -1e4], [0.0, 1e4, 0.0]])
    with no_runtime_warning("logits of order 1e4"):
        loss, grad = loss_pair(call(bw.softmax_cross_entropy, z, np.array([1, 1])), "softmax_cross_entropy", (2, 3))
    # first row: label 1 has logit 0 while the best is 1e4 -> 1e4 nats; second row is right -> 0
    assert_close(loss, 5000.0, msg="cross-entropy of [[1e4, 0, -1e4], [0, 1e4, 0]] with labels [1, 1] "
                                   "(use log_softmax: never log(0))")
    assert_close(grad, [[0.5, -0.5, 0.0], [0.0, 0.0, 0.0]], msg="gradient for logits of order 1e4")


def test_softmax_cross_entropy_is_unchanged_by_shifting_the_logits(bw):
    z, labels = RNG.normal(size=(4, 3)), np.array([0, 2, 1, 2])
    a = loss_pair(call(bw.softmax_cross_entropy, z.copy(), labels), "softmax_cross_entropy", (4, 3))
    b = loss_pair(call(bw.softmax_cross_entropy, z + 50.0, labels), "softmax_cross_entropy", (4, 3))
    assert_close(b[0], a[0], rtol=1e-9, atol=1e-9, msg="loss after adding 50 to every logit (softmax ignores a shift)")
    assert_close(b[1], a[1], rtol=1e-7, atol=1e-12, msg="gradient after adding 50 to every logit")


@pytest.mark.parametrize("labels, why", [
    pytest.param([0, 1, 3], "a label equal to K = 3 (labels lie in [0, K))", id="label-equal-to-K"),
    pytest.param([0, -1, 2], "a label equal to -1 (NumPy would silently take the last column)", id="label--1"),
])
def test_softmax_cross_entropy_refuses_out_of_range_labels(bw, labels, why):
    assert_raises(ValueError, bw.softmax_cross_entropy, np.zeros((3, 3)), np.array(labels), why=why)


@pytest.mark.parametrize("y, why", [
    pytest.param(np.array([0, 1]), "2 labels for 3 rows of logits", id="too-few-labels"),
    pytest.param(np.full((3, 4), 0.25), "target probabilities of shape (3, 4) for logits of shape (3, 3)",
                 id="probabilities-wrong-K"),
    pytest.param(np.full((2, 3), 1 / 3), "target probabilities of shape (2, 3) for logits of shape (3, 3)",
                 id="probabilities-wrong-n"),
    pytest.param(np.full((1, 3), 1 / 3), "target probabilities of shape (1, 3) for logits of shape (3, 3) "
                 "(they would broadcast)", id="probabilities-one-row-would-broadcast"),
])
def test_softmax_cross_entropy_refuses_inconsistent_shapes(bw, y, why):
    assert_raises(ValueError, bw.softmax_cross_entropy, np.zeros((3, 3)), y, why=why)


def test_softmax_cross_entropy_does_not_modify_its_inputs(bw):
    z, Y = RNG.normal(size=(3, 4)), special.softmax(RNG.normal(size=(3, 4)), axis=1)
    labels = np.array([1, 3, 0])
    before = (z.copy(), Y.copy(), labels.copy())
    call(bw.softmax_cross_entropy, z, Y)
    call(bw.softmax_cross_entropy, z, labels)
    if not (np.array_equal(z, before[0]) and np.array_equal(Y, before[1]) and np.array_equal(labels, before[2])):
        _fail("", "expected logits, probabilities and labels unchanged, but one was modified in place")


def test_softmax_cross_entropy_gradient_rows_sum_to_zero(bw):
    z, labels = RNG.normal(size=(6, 5)), RNG.integers(0, 5, size=6)
    _, grad = loss_pair(call(bw.softmax_cross_entropy, z, labels), "softmax_cross_entropy", (6, 5))
    assert_close(grad.sum(axis=1), np.zeros(6), atol=1e-12,
                 msg="each row of the gradient sums to 0 (probabilities and one-hot rows both sum to 1)")


# ================================================================== dense_backward (18.15)
def dense_oracle(dout, x, W):
    """dx, dW, db by central differences of L = sum(dout * (x W + b)): exact for a linear map, up to rounding."""
    eps = 1e-3

    def loss(xx, ww, bb):
        return float(np.sum(dout * (xx @ ww + bb)))

    b = np.zeros(W.shape[1])
    out = []
    for array_index, array in enumerate((x, W, b)):
        grad = np.zeros_like(array)
        for idx in np.ndindex(array.shape):
            args = [x.copy(), W.copy(), b.copy()]
            args[array_index][idx] += eps
            plus = loss(*args)
            args[array_index][idx] -= 2 * eps
            minus = loss(*args)
            grad[idx] = (plus - minus) / (2 * eps)
        out.append(grad)
    return tuple(out)


DENSE_SHAPES = [pytest.param(4, 3, 2, id="n4-in3-out2"), pytest.param(1, 2, 2, id="n1-in2-out2"),
                pytest.param(5, 1, 3, id="n5-in1-out3"), pytest.param(3, 4, 1, id="n3-in4-out1")]


def _dense_data(n, n_in, n_out, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_out)), rng.normal(size=(n, n_in)), rng.normal(size=(n_in, n_out))


def dense_triple(result, n, n_in, n_out):
    if result is None:
        _fail("", "expected (dx, dW, db) from dense_backward, got None (did you forget the return?)")
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        _fail("", f"expected a tuple (dx, dW, db) from dense_backward, got {type(result).__name__}")
    names = ("dx", "dW", "db")
    shapes = ((n, n_in), (n_in, n_out), (n_out,))
    for name, value, shape in zip(names, result, shapes, strict=True):
        arr = as_array(value, f"{name} as an array")
        if arr.shape != shape:
            _fail("", f"expected {name} of shape {shape} (the shape of {'x' if name == 'dx' else name[1:]}), "
                      f"got {arr.shape}")
    return tuple(np.asarray(v, dtype=float) for v in result)


@pytest.mark.parametrize("n, n_in, n_out", DENSE_SHAPES)
def test_dense_backward_matches_finite_differences(bw, n, n_in, n_out):
    dout, x, W = _dense_data(n, n_in, n_out, 1861 + n)
    dx, dW, db = dense_triple(call(bw.dense_backward, dout.copy(), x.copy(), W.copy(),
                                   expected="(dx, dW, db)"), n, n_in, n_out)
    want = dense_oracle(dout, x, W)
    assert_close(dx, want[0], rtol=1e-7, atol=1e-9, msg="dx = dL/dx (one row per sample, like x)")
    assert_close(dW, want[1], rtol=1e-7, atol=1e-9, msg="dW = dL/dW, summed over the samples")
    assert_close(db, want[2], rtol=1e-7, atol=1e-9, msg="db = dL/db, summed over the samples")


@needs_torch
@pytest.mark.parametrize("n, n_in, n_out", DENSE_SHAPES)
def test_dense_backward_matches_torch_autograd(bw, n, n_in, n_out):
    dout, x, W = _dense_data(n, n_in, n_out, 1871 + n)
    b = np.random.default_rng(5).normal(size=n_out)
    xt, Wt, bt = torch_tensor(x, True), torch_tensor(W, True), torch_tensor(b, True)
    z = torch.nn.functional.linear(xt, Wt.T, bt)
    z.backward(torch_tensor(dout))
    dx, dW, db = dense_triple(call(bw.dense_backward, dout.copy(), x.copy(), W.copy(),
                                   expected="(dx, dW, db)"), n, n_in, n_out)
    assert_close(dx, xt.grad.numpy(), rtol=1e-10, atol=1e-10, msg="dx against torch autograd")
    assert_close(dW, Wt.grad.numpy(), rtol=1e-10, atol=1e-10, msg="dW against torch autograd")
    assert_close(db, bt.grad.numpy(), rtol=1e-10, atol=1e-10, msg="db against torch autograd")


def test_dense_backward_docstring_example(bw):
    dx, dW, db = dense_triple(call(bw.dense_backward, np.array([[1.0, -1.0]]), np.array([[1.0, 2.0]]),
                                   np.array([[1.0, 2.0], [3.0, 4.0]])), 1, 2, 2)
    assert_close(dx, [[-1.0, -1.0]], msg="dx of the docstring example")
    assert_close(dW, [[1.0, -1.0], [2.0, -2.0]], msg="dW of the docstring example")
    assert_close(db, [1.0, -1.0], msg="db of the docstring example")


@pytest.mark.parametrize("shapes, why", [
    pytest.param(((4, 3), (4, 2), (2, 2)), "dout of shape (4, 3) for W of shape (2, 2) (n_out differs)",
                 id="dout-wrong-n_out"),
    pytest.param(((4, 2), (5, 3), (3, 2)), "x with 5 rows but dout with 4 rows", id="x-wrong-n"),
    pytest.param(((4, 2), (4, 3), (2, 2)), "x with 3 columns but W with 2 rows (n_in differs)", id="W-wrong-n_in"),
    pytest.param(((4, 2), (4, 3), (3, 3)), "W with 3 columns but dout with 2 (n_out differs)", id="W-wrong-n_out"),
])
def test_dense_backward_refuses_inconsistent_shapes(bw, shapes, why):
    dout, x, W = (np.ones(s) for s in shapes)
    assert_raises(ValueError, bw.dense_backward, dout, x, W, why=why)


def test_dense_backward_does_not_modify_its_inputs(bw):
    dout, x, W = _dense_data(3, 4, 2, 1881)
    before = (dout.copy(), x.copy(), W.copy())
    call(bw.dense_backward, dout, x, W)
    if not all(np.array_equal(a, b) for a, b in zip((dout, x, W), before, strict=True)):
        _fail("", "expected dout, x and W unchanged after dense_backward, but one was modified in place")


def test_dense_backward_is_linear_in_dout(bw):
    dout, x, W = _dense_data(3, 2, 2, 1882)
    one = dense_triple(call(bw.dense_backward, dout.copy(), x.copy(), W.copy()), 3, 2, 2)
    three = dense_triple(call(bw.dense_backward, 3.0 * dout, x.copy(), W.copy()), 3, 2, 2)
    for name, a, b in zip(("dx", "dW", "db"), one, three, strict=True):
        assert_close(b, 3.0 * a, msg=f"{name} for 3 * dout: three times {name} for dout")


# ================================================================== gradient_check (18.16)
def test_gradient_check_accepts_an_exact_gradient_of_a_quadratic(bw):
    w = np.array([1.0, -2.0, 3.0, 0.5])
    result = call(bw.gradient_check, lambda: float(np.sum(w ** 2)), w, 2 * w, expected="a float")
    if not isinstance(result, (float, np.floating)):
        _fail("", f"expected a float from gradient_check, got {type(result).__name__}")
    if not result < 1e-7:
        _fail("", f"expected a relative error below 1e-7 for the exact gradient 2w of sum(w**2), got {result:.3g}")


def test_gradient_check_accepts_the_exact_gradients_of_a_dense_layer(bw):
    rng = np.random.default_rng(1891)
    x, W, b, target = rng.normal(size=(5, 3)), rng.normal(size=(3, 2)), rng.normal(size=2), rng.normal(size=(5, 2))

    def loss():
        return float(np.mean((x @ W + b - target) ** 2))

    delta = 2.0 * (x @ W + b - target) / target.size
    for param, grad, name in ((W, x.T @ delta, "W"), (b, delta.sum(axis=0), "b")):
        result = call(bw.gradient_check, loss, param, grad, expected="a float")
        if not float(result) < 1e-7:
            _fail("", f"expected a relative error below 1e-7 for the exact gradient of {name}, got {float(result):.3g}")


def test_gradient_check_gives_about_1_for_a_sign_error(bw):
    w = np.array([1.0, -2.0, 3.0])
    result = float(call(bw.gradient_check, lambda: float(np.sum(w ** 2)), w, -2 * w, expected="a float"))
    assert_close(result, 1.0, rtol=0, atol=1e-6, msg="relative error |a - n| / (|a| + |n|) for a wrong sign")


def test_gradient_check_measures_the_relative_error_of_the_docstring(bw):
    w = np.array([2.0])
    # numerical derivative of w**2 at 2 is 4; an analytic value of 5 gives |5 - 4| / (5 + 4) = 1/9
    result = float(call(bw.gradient_check, lambda: float(w[0] ** 2), w, np.array([5.0]), expected="a float"))
    assert_close(result, 1 / 9, rtol=0, atol=1e-6, msg="relative error for an analytic 5 against a numerical 4 "
                                                       "(|a - n| / max(|a| + |n|, 1e-12))")


def test_gradient_check_returns_the_maximum_over_the_entries(bw):
    w = np.array([1.0, 1.0, 1.0])
    grad = np.array([2.0, 3.0, 2.0])     # right, wrong (error 1/5), right
    result = float(call(bw.gradient_check, lambda: float(np.sum(w ** 2)), w, grad, expected="a float"))
    assert_close(result, 0.2, rtol=0, atol=1e-6, msg="maximum relative error over the entries (one entry is off)")


def test_gradient_check_returns_0_when_both_gradients_are_0(bw):
    w = np.array([1.0, 2.0])
    with no_runtime_warning("a constant loss and a zero gradient (0 / 0)"):
        result = float(call(bw.gradient_check, lambda: 3.0, w, np.zeros(2), expected="a float"))
    assert_close(result, 0.0, atol=1e-12, msg="relative error when the analytic and numerical gradients are both 0 "
                                              "(the denominator has a floor of 1e-12)")


def test_gradient_check_uses_a_centred_difference_with_eps(bw):
    w = np.array([1.0])
    eps = 1e-3
    # centred: ((1 + h)^3 - (1 - h)^3) / (2h) = 3 + h^2, so the relative error is h^2 / (6 + h^2)
    result = float(call(bw.gradient_check, lambda: float(w[0] ** 3), w, np.array([3.0]), eps=eps, expected="a float"))
    assert_close(result, eps ** 2 / (6 + eps ** 2), rtol=1e-3, atol=0,
                 msg="relative error for w**3 at 1 with eps=1e-3 (centred difference: f(w+eps) - f(w-eps), "
                     "divided by 2 eps)")


def test_gradient_check_restores_param_bit_for_bit(bw):
    # for these values, old - eps + eps (or old + eps - 2 eps + eps) differs from old in the last bits
    w = np.array([0.1, 1 / 3, 4.13259793472436e-08, 0.7, -9.217253762584194e-07])
    before = w.tobytes()
    call(bw.gradient_check, lambda: float(np.sum(np.sin(w))), w, np.cos(w))
    if w.tobytes() != before:
        _fail("", f"expected param identical bit for bit after gradient_check, got {_short(w, 17)} "
                  "(save the old value and put it back, never old + eps - eps)")


def test_gradient_check_perturbs_param_in_place(bw):
    w = np.array([1.0, 2.0])
    seen = []

    def loss():
        seen.append(w.copy())
        return float(np.sum(w ** 2))

    call(bw.gradient_check, loss, w, 2 * w)
    if not any(not np.array_equal(s, [1.0, 2.0]) for s in seen):
        _fail("", "expected f to see a perturbed param (the docstring: param is modified IN PLACE, +eps then -eps), "
                  "but every call saw the original values")


def _perturbed_entries(bw, size, n_checks, seed):
    w = np.linspace(1.0, 2.0, size)
    original = w.copy()
    touched = set()

    def loss():
        changed = np.flatnonzero(w != original)
        if len(changed) > 1:
            _fail("", f"expected one entry perturbed at a time, got {len(changed)} entries changed together")
        touched.update(int(i) for i in changed)
        return float(np.sum(w ** 2))

    call(bw.gradient_check, loss, w, 2 * w, n_checks=n_checks, rng=np.random.default_rng(seed))
    return touched


def test_gradient_check_with_n_checks_perturbs_that_many_distinct_entries(bw):
    touched = _perturbed_entries(bw, 6, 5, 1892)
    if len(touched) != 5:
        _fail("", f"expected 5 distinct entries checked with n_checks=5 among 6, got {len(touched)} "
                  "(draw the entries without replacement)")


def test_gradient_check_without_n_checks_checks_every_entry(bw):
    touched = _perturbed_entries(bw, 7, None, 1893)
    if touched != set(range(7)):
        _fail("", f"expected the 7 entries checked with n_checks=None, got {sorted(touched)}")


def test_gradient_check_with_the_same_rng_seed_checks_the_same_entries(bw):
    a = _perturbed_entries(bw, 30, 4, 1894)
    b = _perturbed_entries(bw, 30, 4, 1894)
    if a != b:
        _fail("", f"expected the same entries with two generators of the same seed, got {sorted(a)} and {sorted(b)}")


def test_gradient_check_works_on_a_2d_param(bw):
    rng = np.random.default_rng(1895)
    W = rng.normal(size=(3, 4))
    result = float(call(bw.gradient_check, lambda: float(np.sum(W ** 3)), W, 3 * W ** 2, expected="a float"))
    if not result < 1e-7:
        _fail("", f"expected a relative error below 1e-7 for the exact gradient of a (3, 4) param, got {result:.3g}")


@pytest.mark.parametrize("param, grad, why", [
    pytest.param(np.array([1.0, 2.0, 3.0]), np.zeros((3, 1)), "a gradient of shape (3, 1) for a param of shape (3,)",
                 id="shapes-differ"),
    pytest.param(np.array([1, 2, 3]), np.zeros(3), "an integer param (w + 1e-6 would be truncated)",
                 id="integer-param"),
    pytest.param([1.0, 2.0], np.zeros(2), "a list as param (it cannot be modified in place)", id="list-param"),
])
def test_gradient_check_refuses_invalid_inputs(bw, param, grad, why):
    assert_raises(ValueError, bw.gradient_check, lambda: 0.0, param, grad, why=why)


# ================================================================== mlp_backward (18.18)
def numpy_forward(x, params, f, f_out):
    """Forward pass with its cache (a_prev, z), written here so that no function of the learner is used."""
    a, cache = x, []
    for index, (W, b) in enumerate(params):
        z = a @ W + b
        cache.append((a, z))
        g = f_out if index == len(params) - 1 else f
        a = z if g is None else g(z)
    return a, cache


ACTIVATIONS = {
    "identity": (None, None, None),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2, "Tanh"),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(float), "ReLU"),
    "sigmoid": (special.expit, lambda z: special.expit(z) * (1 - special.expit(z)), "Sigmoid"),
}
NETWORKS = [pytest.param([2, 2, 4, 3, 2], id="2-2-4-3-2"), pytest.param([3, 5, 1], id="3-5-1"),
            pytest.param([4, 3], id="4-3-single-layer")]


def _network(sizes, seed, n=6):
    rng = np.random.default_rng(seed)
    params = [(rng.normal(size=(a, b)) / np.sqrt(a), rng.normal(scale=0.5, size=b))
              for a, b in zip(sizes[:-1], sizes[1:], strict=True)]
    return rng.normal(size=(n, sizes[0])), params, rng.normal(size=(n, sizes[-1]))


def grads_list(result, params, return_input_grad=False):
    """Validate the result of mlp_backward: a list of (dW, db) with the shapes of params (and dx)."""
    if result is None:
        _fail("", "expected the gradients from mlp_backward, got None (did you forget the return?)")
    dx = None
    if return_input_grad:
        if not isinstance(result, tuple) or len(result) != 2:
            _fail("", "expected a tuple (grads, dx) from mlp_backward(..., return_input_grad=True)")
        result, dx = result
    if not isinstance(result, (list, tuple)) or len(result) != len(params):
        _fail("", f"expected a list of {len(params)} (dW, db) tuples, one per layer, got "
                  f"{type(result).__name__}" + (f" of length {len(result)}" if hasattr(result, '__len__') else ""))
    out = []
    for index, (pair, (W, b)) in enumerate(zip(result, params, strict=True)):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            _fail("", f"expected a (dW, db) pair for layer {index}, got {type(pair).__name__}")
        dW, db = (as_array(v, "a gradient") for v in pair)
        if dW.shape != W.shape or db.shape != np.shape(b):
            _fail("", f"expected dW of shape {W.shape} and db of shape {np.shape(b)} for layer {index}, "
                      f"got {dW.shape} and {db.shape} (the shapes of params)")
        out.append((dW, db))
    return (out, as_array(dx, "dx")) if return_input_grad else out


def torch_mlp_grads(x, params, dout, act_name, out_name=None):
    layers = []
    for index, (W, b) in enumerate(params):
        lin = torch.nn.Linear(*W.shape).double()
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W.T))
            lin.bias.copy_(torch.tensor(b))
        layers.append(lin)
        name = out_name if index == len(params) - 1 else ACTIVATIONS[act_name][2]
        if name:
            layers.append(getattr(torch.nn, name)())
    model = torch.nn.Sequential(*layers)
    xt = torch_tensor(x, grad=True)
    model(xt).backward(torch_tensor(dout))
    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    return [(m.weight.grad.numpy().T, m.bias.grad.numpy()) for m in linears], xt.grad.numpy()


def finite_difference_grads(x, params, dout, f, f_out):
    eps = 1e-6

    def loss(ps, xx):
        return float(np.sum(dout * numpy_forward(xx, ps, f, f_out)[0]))

    grads = []
    for index, (W, b) in enumerate(params):
        pair = []
        for which, array in enumerate((W, b)):
            g = np.zeros_like(array)
            for idx in np.ndindex(array.shape):
                for sign in (1, -1):
                    ps = [(Wk.copy(), bk.copy()) for Wk, bk in params]
                    ps[index][which][idx] += sign * eps
                    g[idx] += sign * loss(ps, x) / (2 * eps)
            pair.append(g)
        grads.append(tuple(pair))
    dx = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        for sign in (1, -1):
            xx = x.copy()
            xx[idx] += sign * eps
            dx[idx] += sign * loss(params, xx) / (2 * eps)
    return grads, dx


def _compare(got, want, act, rtol, atol):
    label = act if "," in act else f"{act} hidden activation"
    for index, ((dW, db), (eW, eb)) in enumerate(zip(got, want, strict=True)):
        assert_close(dW, eW, rtol=rtol, atol=atol, msg=f"dW of layer {index} ({label})")
        assert_close(db, eb, rtol=rtol, atol=atol, msg=f"db of layer {index} ({label})")


@needs_torch
@pytest.mark.parametrize("act", list(ACTIVATIONS))
@pytest.mark.parametrize("sizes", NETWORKS)
def test_mlp_backward_matches_torch_autograd(bw, sizes, act):
    x, params, dout = _network(sizes, 1901 + len(sizes))
    f, df, _ = ACTIVATIONS[act]
    _, cache = numpy_forward(x, params, f, None)
    got, dx = grads_list(call(bw.mlp_backward, dout.copy(), cache, params, activation_derivative=df,
                              return_input_grad=True, expected="(grads, dx)"), params, return_input_grad=True)
    want, want_dx = torch_mlp_grads(x, params, dout, act)
    _compare(got, want, act, rtol=1e-9, atol=1e-9)
    assert_close(dx, want_dx, rtol=1e-9, atol=1e-9, msg=f"dx = dL/dx ({act} hidden activation)")


@pytest.mark.parametrize("act", list(ACTIVATIONS))
@pytest.mark.parametrize("sizes", NETWORKS)
def test_mlp_backward_matches_finite_differences(bw, sizes, act):
    x, params, dout = _network(sizes, 1911 + len(sizes), n=3)
    f, df, _ = ACTIVATIONS[act]
    _, cache = numpy_forward(x, params, f, None)
    got, dx = grads_list(call(bw.mlp_backward, dout.copy(), cache, params, activation_derivative=df,
                              return_input_grad=True, expected="(grads, dx)"), params, return_input_grad=True)
    want, want_dx = finite_difference_grads(x, params, dout, f, None)
    _compare(got, want, act, rtol=1e-5, atol=1e-7)
    assert_close(dx, want_dx, rtol=1e-5, atol=1e-7, msg=f"dx = dL/dx ({act} hidden activation)")


@needs_torch
def test_mlp_backward_applies_the_output_activation_derivative_at_the_last_z(bw):
    x, params, dout = _network([3, 4, 2], 1921)
    f, df, _ = ACTIVATIONS["tanh"]
    _, cache = numpy_forward(x, params, f, special.expit)
    got = grads_list(call(bw.mlp_backward, dout.copy(), cache, params, activation_derivative=df,
                          output_activation_derivative=ACTIVATIONS["sigmoid"][1]), params)
    want, _ = torch_mlp_grads(x, params, dout, "tanh", out_name="Sigmoid")
    _compare(got, want, "tanh hidden, sigmoid output", rtol=1e-9, atol=1e-9)


def test_mlp_backward_output_activation_matches_finite_differences(bw):
    x, params, dout = _network([3, 4, 2], 1922, n=3)
    f, df, _ = ACTIVATIONS["tanh"]
    _, cache = numpy_forward(x, params, f, special.expit)
    got = grads_list(call(bw.mlp_backward, dout.copy(), cache, params, activation_derivative=df,
                          output_activation_derivative=ACTIVATIONS["sigmoid"][1]), params)
    want, _ = finite_difference_grads(x, params, dout, f, special.expit)
    _compare(got, want, "tanh, sigmoid output", rtol=1e-5, atol=1e-7)


def test_mlp_backward_without_return_input_grad_returns_only_the_list(bw):
    x, params, dout = _network([2, 3, 1], 1923)
    _, cache = numpy_forward(x, params, np.tanh, None)
    result = call(bw.mlp_backward, dout, cache, params, activation_derivative=ACTIVATIONS["tanh"][1])
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], np.ndarray):
        _fail("", "expected only the list of (dW, db) without return_input_grad, got a (grads, dx) tuple")
    grads_list(result, params)


def test_mlp_backward_docstring_example(bw):
    x = np.array([[1.0, 1.0]])
    params = [(np.array([[1.0, -1.0], [0.5, -2.0]]), np.array([0.0, 1.0])),
              (np.array([[1.0], [1.0]]), np.array([-1.0]))]
    cache = [(x, np.array([[1.5, -2.0]])), (np.array([[1.5, 0.0]]), np.array([[0.5]]))]
    got, dx = grads_list(call(bw.mlp_backward, np.array([[1.0]]), cache, params,
                              activation_derivative=lambda z: (z > 0).astype(float), return_input_grad=True),
                         params, return_input_grad=True)
    assert_close(got[0][0], [[1.0, 0.0], [1.0, 0.0]], msg="dW of the hidden layer in the docstring example "
                                                          "(the ReLU derivative at z = -2 blocks the second neuron)")
    assert_close(got[0][1], [1.0, 0.0], msg="db of the hidden layer in the docstring example")
    assert_close(got[1][0], [[1.5], [0.0]], msg="dW of the output layer in the docstring example")
    assert_close(got[1][1], [1.0], msg="db of the output layer in the docstring example")
    assert_close(dx, [[1.0, 0.5]], msg="dx of the docstring example")


def test_mlp_backward_evaluates_the_derivative_at_the_cached_z(bw):
    x, params, dout = _network([2, 3, 2], 1924)
    _, cache = numpy_forward(x, params, np.tanh, None)
    seen = []

    def derivative(z):
        seen.append(np.array(z, copy=True))
        return 1.0 - np.tanh(z) ** 2

    call(bw.mlp_backward, dout, cache, params, activation_derivative=derivative)
    if len(seen) != 1 or not np.allclose(seen[0], cache[0][1]):
        _fail("", "expected the hidden derivative called once, on the cached pre-activation z of the hidden layer "
                  f"(cache[0][1]), got {len(seen)} call(s)"
                  + ("" if not seen else " on other values (the activation a instead of z?)"))


def test_mlp_backward_refuses_a_cache_of_another_length(bw):
    x, params, dout = _network([2, 3, 2], 1925)
    _, cache = numpy_forward(x, params, np.tanh, None)
    assert_raises(ValueError, bw.mlp_backward, dout, cache[:1], params,
                  why="a cache of 1 layer for params of 2 layers")


def test_mlp_backward_refuses_dout_of_the_wrong_shape(bw):
    x, params, dout = _network([2, 3, 2], 1926)
    _, cache = numpy_forward(x, params, np.tanh, None)
    assert_raises(ValueError, bw.mlp_backward, np.ones((6, 3)), cache, params,
                  why="dout of shape (6, 3) for a network output of shape (6, 2)")


@pytest.mark.parametrize("sigmoid_output", [False, True], ids=["linear-output", "sigmoid-output"])
def test_mlp_backward_does_not_modify_its_inputs(bw, sigmoid_output):
    x, params, dout = _network([2, 3, 4, 2], 1927)
    _, cache = numpy_forward(x, params, np.tanh, special.expit if sigmoid_output else None)
    copies = (dout.copy(), [(a.copy(), z.copy()) for a, z in cache], [(W.copy(), b.copy()) for W, b in params])
    call(bw.mlp_backward, dout, cache, params, activation_derivative=ACTIVATIONS["tanh"][1], return_input_grad=True,
         output_activation_derivative=ACTIVATIONS["sigmoid"][1] if sigmoid_output else None)
    same = (np.array_equal(dout, copies[0])
            and all(np.array_equal(a, c[0]) and np.array_equal(z, c[1]) for (a, z), c in zip(cache, copies[1], strict=True))
            and all(np.array_equal(W, c[0]) and np.array_equal(b, c[1]) for (W, b), c in zip(params, copies[2], strict=True)))
    if not same:
        _fail("", "expected dout, the cache and params unchanged after mlp_backward, but one was modified in place "
                  "(compute delta = grad * f'(z) in a new array, never with *=)")


def test_mlp_backward_with_identity_equals_the_collapsed_linear_layer(bw):
    x, params, dout = _network([3, 4, 2], 1928)
    _, cache = numpy_forward(x, params, None, None)
    got, dx = grads_list(call(bw.mlp_backward, dout, cache, params, return_input_grad=True), params,
                         return_input_grad=True)
    (W1, _), (W2, _) = params
    assert_close(dx, dout @ (W1 @ W2).T, msg="dx of a network without activation: dout (W1 W2)^T")
    assert_close(got[1][1], dout.sum(axis=0), msg="db of the output layer: dout summed over the samples")
