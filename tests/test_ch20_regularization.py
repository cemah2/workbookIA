"""Tests of mylearn.nn.regularization (chapter 20): oracle tests and properties.

    pytest tests/test_ch20_regularization.py              # your code (mon_travail/mylearn/nn/regularization.py)
    pytest tests/test_ch20_regularization.py --impl=ref   # the reference

Oracles: PyTorch (``torch.nn.functional.dropout`` for the properties of dropout, autograd through
``x * mask``; ``torch.nn.BatchNorm1d`` and ``torch.nn.BatchNorm2d`` in float64, outputs, running statistics
and gradients, in training and in evaluation mode; ``torch.nn.functional.layer_norm``; autograd of
``0.5 * lam * sum(w**2)`` and one step of ``torch.optim.SGD(weight_decay=lam)``) and NumPy: the formulas of
the sheet, transcribed in this file. With PyTorch, the transcribed formulas are checked against torch at
every test (a disagreement is a bug of this file, not yours); without PyTorch, they are the oracle, so that
no test is skipped. ``EarlyStopping`` is tested on scripted sequences of metrics.

Every test name starts with the name of what it tests, so that each exercise runs its own group:
``-k "test_dropout_"`` (20.16), ``-k "test_layer_norm_ or test_l2_penalty_ or test_early_stopping_"``
(20.17), ``-k "test_batchnorm_"`` (20.22).

Contract precisions written in the statements of 20.16, 20.17 and 20.22 and checked here: the inputs are
never modified (except the running statistics of the batch normalization, updated IN PLACE in training
mode, and the parameters given to ``EarlyStopping.restore``); the arrays returned are new float64 arrays
(lists are accepted as inputs); ``dropout_forward`` draws a new mask at every call, with the generator
it receives (``rng.random(x.shape) >= p`` is one way, not the only one: the tests only check properties);
``batchnorm_forward`` accepts ``(N, D)`` and ``(N, C, H, W)``, refuses a single value per feature in
training mode but not in evaluation mode; ``l2_penalty`` returns a Python float; ``EarlyStopping.step``
returns a Python bool, the first finite value is always an improvement and a nan never is.
The first line of every failure message says what was expected (it is the line that ``pytest -rf``
shows).
"""

import copy
import math

import numpy as np
import pytest

try:
    import torch
except ImportError:  # the PyTorch oracles run on Colab; the NumPy formulas take over here
    torch = None

# a failed comparison between this file's NumPy oracle and PyTorch is a bug of the tests, not of the learner
ORACLE_BUG = ("bug in tests/test_ch20_regularization.py (not in your code): the NumPy oracle of this test "
              "disagrees with torch")


@pytest.fixture
def reg(mylearn_module):
    return mylearn_module("nn.regularization")


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


def assert_close(result, expected, rtol=1e-10, atol=1e-12, msg=""):
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
        data = f"expected {_short(want, 10)}\n     got {_short(got, 10)}" if want.size <= 12 else ""
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


def pair(result, what, names=("out", "mask")):
    """Unpack a 2-tuple result, or fail with a message that says what was expected."""
    if result is None:
        _fail("", f"expected the tuple ({', '.join(names)}) from {what}, got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != len(names):
        _fail("", f"expected the tuple ({', '.join(names)}) from {what}, got {type(result).__name__}"
                  + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else ""))
    return result


def assert_new_float_array(arr, like, what):
    if not isinstance(arr, np.ndarray):
        _fail("", f"expected {what} to be a NumPy array, got {type(arr).__name__}")
    if arr.dtype != np.float64:
        _fail("", f"expected {what} to be float64, got {arr.dtype}")
    if arr.shape != np.shape(like):
        _fail("", f"expected {what} of shape {np.shape(like)}, got {arr.shape}")


# ================================================================== dropout (20.16)
def test_dropout_forward_docstring_example(reg):
    out, mask = pair(call(reg.dropout_forward, np.ones((2, 4)), p=0.5, rng=np.random.default_rng(0)),
                     "dropout_forward")
    if not set(np.unique(out).tolist()) <= {0.0, 2.0}:
        _fail("", f"expected only 0 and 2 = 1/(1 - 0.5) in out, got the values {sorted(set(np.unique(out).tolist()))}")
    assert_close(out, np.ones((2, 4)) * mask, msg="expected out == x * mask")
    out, mask = pair(call(reg.dropout_forward, np.ones(3), training=False), "dropout_forward(training=False)")
    assert_close(out, np.ones(3), msg="evaluation mode: expected the identity")
    if mask is not None:
        _fail("", f"evaluation mode: expected mask None, got {_short(mask)}")


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.8], ids=lambda p: f"p={p}")
def test_dropout_forward_mask_values_and_scaling(reg, p):
    x = np.random.default_rng(1).normal(size=(50, 40))
    out, mask = pair(call(reg.dropout_forward, x.copy(), p=p, rng=np.random.default_rng(2)), "dropout_forward")
    assert_new_float_array(mask, x, "mask")
    assert_new_float_array(out, x, "out")
    values = set(np.round(np.unique(mask), 12).tolist())
    allowed = {0.0, round(1 / (1 - p), 12)}
    if not values <= allowed:
        _fail("", f"expected the mask to hold only 0 and 1/(1 - p) = {1 / (1 - p):.6g}, got the values "
                  f"{sorted(values)[:5]} (inverted dropout: the survivors are scaled up)")
    assert_close(out, x * mask, msg="expected out == x * mask")


@pytest.mark.parametrize("p", [0.2, 0.5, 0.9], ids=lambda p: f"p={p}")
def test_dropout_forward_drops_a_fraction_p_and_keeps_the_mean(reg, p):
    x = np.ones(1_000_000)
    out, _ = pair(call(reg.dropout_forward, x, p=p, rng=np.random.default_rng(3)), "dropout_forward")
    zeros = float(np.mean(np.asarray(out) == 0))
    if abs(zeros - p) > 0.01:
        _fail("", f"expected a fraction of zeros close to p = {p} (within 1 %), got {zeros:.4f} on 1e6 values")
    mean = float(np.mean(out))
    if abs(mean - 1.0) > 0.02:
        _fail("", f"expected the mean of out close to the mean of x, 1.0 (inverted dropout), got {mean:.4f}")
    if torch is not None:                              # the same properties as PyTorch's dropout
        ref = torch.nn.functional.dropout(torch.ones(1_000_000, dtype=torch.float64), p=p, training=True)
        assert abs(float((ref == 0).double().mean()) - p) < 0.01, ORACLE_BUG
        assert np.allclose(np.unique(ref.numpy()), [0.0, 1 / (1 - p)]), ORACLE_BUG


def test_dropout_forward_draws_a_new_mask_at_every_call(reg):
    rng = np.random.default_rng(4)
    _, first = pair(call(reg.dropout_forward, np.ones((20, 30)), p=0.5, rng=rng), "dropout_forward")
    _, second = pair(call(reg.dropout_forward, np.ones((20, 30)), p=0.5, rng=rng), "dropout_forward")
    if np.array_equal(first, second):
        _fail("", "expected two calls with the same generator to draw two different masks (a new mask at "
                  "EVERY call, i.e. every mini-batch), got twice the same mask")


def test_dropout_forward_drops_each_value_independently(reg):
    _, mask = pair(call(reg.dropout_forward, np.ones((1000, 50)), p=0.5, rng=np.random.default_rng(5)),
                   "dropout_forward")
    mask = np.asarray(mask)
    if np.all(mask == mask[:1]) or np.all(mask == mask[:, :1]):
        _fail("", "expected each value to be dropped independently (a mask of the shape of x, drawn value by "
                  "value), got the same mask for every example (or for every feature)")


def test_dropout_forward_same_seed_same_mask(reg):
    _, a = pair(call(reg.dropout_forward, np.ones((10, 10)), p=0.3, rng=np.random.default_rng(7)), "dropout_forward")
    _, b = pair(call(reg.dropout_forward, np.ones((10, 10)), p=0.3, rng=np.random.default_rng(7)), "dropout_forward")
    if not np.array_equal(a, b):
        _fail("", "expected the same mask from two generators with the same seed (draw the mask with the "
                  "rng you receive, not with np.random)")


def test_dropout_forward_rng_none_works(reg):
    out, mask = pair(call(reg.dropout_forward, np.ones(1000), p=0.5), "dropout_forward(rng=None)")
    assert_new_float_array(mask, np.ones(1000), "mask")
    assert_close(out, mask, msg="expected out == x * mask with x = ones")


def test_dropout_forward_without_rng_draws_a_new_mask(reg):
    np.random.seed(2016)
    expected_next = np.random.random()
    np.random.seed(2016)
    _, first = pair(call(reg.dropout_forward, np.ones(1000), p=0.5), "dropout_forward(rng=None)")
    _, second = pair(call(reg.dropout_forward, np.ones(1000), p=0.5), "dropout_forward(rng=None)")
    if np.array_equal(first, second):
        _fail("", "expected two different masks from two calls without rng (rng=None means a fresh "
                  "np.random.default_rng(), never a fixed seed)")
    if np.random.random() != expected_next:
        _fail("", "expected the global NumPy generator (np.random.seed, np.random.random...) untouched: use "
                  "np.random.default_rng() when rng is None")


@pytest.mark.parametrize("training, p", [(False, 0.5), (True, 0.0)], ids=["eval-mode", "p=0"])
def test_dropout_forward_identity_returns_a_copy_and_no_mask(reg, training, p):
    x = np.array([[1.0, -2.0], [3.0, 4.0]])
    out, mask = pair(call(reg.dropout_forward, x, p=p, training=training), "dropout_forward")
    if mask is not None:
        _fail("", f"expected mask None (training={training}, p={p}), got {_short(mask)}")
    assert_close(out, x, msg="expected the identity")
    if out is x or np.shares_memory(out, x):
        _fail("", "expected out to be a copy of x, got x itself (or a view): changing out would change x")


def test_dropout_forward_p_one_gives_zeros(reg):
    out, mask = pair(call(reg.dropout_forward, np.full((3, 2), 5.0), p=1.0, rng=np.random.default_rng(0)),
                     "dropout_forward(p=1)")
    assert_close(out, np.zeros((3, 2)), msg="p = 1: expected out all zeros (no division by 1 - p = 0)")
    assert_close(mask, np.zeros((3, 2)), msg="p = 1: expected mask all zeros")


def test_dropout_forward_does_not_modify_x_and_accepts_lists(reg):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    saved = x.copy()
    call(reg.dropout_forward, x, p=0.5, rng=np.random.default_rng(0))
    if not np.array_equal(x, saved):
        _fail("", f"expected x unchanged, got {_short(x)} (compute out in a new array)")
    out, _ = pair(call(reg.dropout_forward, [[1, 2], [3, 4]], p=0.5, rng=np.random.default_rng(0)),
                  "dropout_forward on a list of ints")
    assert_new_float_array(out, np.zeros((2, 2)), "out (from a list of ints)")


def test_dropout_forward_keeps_any_shape(reg):
    for shape in [(7,), (3, 4), (2, 3, 4, 5)]:
        out, mask = pair(call(reg.dropout_forward, np.ones(shape), p=0.4, rng=np.random.default_rng(0)),
                         f"dropout_forward on shape {shape}")
        assert_new_float_array(out, np.ones(shape), f"out for shape {shape}")
        assert_new_float_array(mask, np.ones(shape), f"mask for shape {shape}")


@pytest.mark.parametrize("p, why", [
    pytest.param(-0.1, "p < 0", id="negative"),
    pytest.param(1.5, "p > 1", id="above-one"),
    pytest.param(float("nan"), "p is nan", id="nan"),
])
def test_dropout_forward_refuses_invalid_p(reg, p, why):
    assert_raises(ValueError, reg.dropout_forward, np.ones(3), p=p, why=why)


def test_dropout_backward_docstring_example(reg):
    got = call(reg.dropout_backward, np.array([1.0, 1.0, 1.0]), np.array([0.0, 2.0, 2.0]))
    assert_close(got, [0.0, 2.0, 2.0], msg="dropout_backward([1, 1, 1], [0, 2, 2])")


def test_dropout_backward_matches_autograd(reg):
    rng = np.random.default_rng(5)
    x, dout = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    mask = (rng.random((6, 5)) >= 0.3) / 0.7
    want = dout * mask                                  # d(sum(dout * x * mask))/dx
    if torch is not None:
        tx = torch.tensor(x, requires_grad=True)
        (tx * torch.tensor(mask) * torch.tensor(dout)).sum().backward()
        assert np.allclose(tx.grad.numpy(), want), ORACLE_BUG
    saved = (dout.copy(), mask.copy())
    got = call(reg.dropout_backward, dout, mask)
    assert_new_float_array(got, dout, "dx")
    assert_close(got, want, msg="expected dx = dout * mask (the same mask as the forward pass)")
    if not (np.array_equal(dout, saved[0]) and np.array_equal(mask, saved[1])):
        _fail("", "expected dout and mask unchanged (compute dx in a new array)")


def test_dropout_backward_mask_none_is_the_identity(reg):
    dout = np.array([[1.0, -2.0], [0.5, 3.0]])
    got = call(reg.dropout_backward, dout, None)
    assert_close(got, dout, msg="mask None (evaluation mode, p = 0): expected dx = dout")
    if isinstance(got, np.ndarray) and np.shares_memory(got, dout):
        _fail("", "mask None: expected dx to be a new array, a copy of dout, got dout itself (or a view)")


@pytest.mark.parametrize("mask_shape", [(3, 2), (1, 3), (3,)], ids=["transposed", "broadcastable-row", "1d"])
def test_dropout_backward_refuses_a_mask_of_another_shape(reg, mask_shape):
    assert_raises(ValueError, reg.dropout_backward, np.ones((2, 3)), np.ones(mask_shape),
                  why=f"mask of shape {mask_shape} for dout of shape (2, 3) (no broadcasting)")


def test_dropout_forward_then_backward_chain(reg):
    rng = np.random.default_rng(6)
    x = rng.normal(size=(4, 8))
    out, mask = pair(call(reg.dropout_forward, x, p=0.25, rng=rng), "dropout_forward")
    dx = call(reg.dropout_backward, np.ones_like(x), mask)
    assert_close(dx, mask, msg="the gradient of sum(out) with respect to x is the mask")


# ================================================================== batch normalization (20.22)
def _bn_oracle(x, gamma, beta, running_mean, running_var, training, momentum, eps):
    """The formulas of the sheet (biased variance to normalize, unbiased for the running variance)."""
    axes = (0,) if x.ndim == 2 else (0, 2, 3)
    shape = [1] * x.ndim
    shape[1] = x.shape[1]
    m = x.size // x.shape[1]
    rm, rv = running_mean.copy(), running_var.copy()
    if training:
        mean, var = x.mean(axis=axes), x.var(axis=axes)
        rm = (1 - momentum) * rm + momentum * mean
        rv = (1 - momentum) * rv + momentum * var * m / (m - 1)
    else:
        mean, var = running_mean, running_var
    out = gamma.reshape(shape) * (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps) + beta.reshape(shape)
    return out, rm, rv


def _bn_backward_oracle(x, gamma, dout, training, running_mean, running_var, eps):
    """dx, dgamma, dbeta by the closed form (training) or the fixed affine map (evaluation)."""
    axes = (0,) if x.ndim == 2 else (0, 2, 3)
    shape = [1] * x.ndim
    shape[1] = x.shape[1]
    m = x.size // x.shape[1]
    if training:
        mean, var = x.mean(axis=axes), x.var(axis=axes)
    else:
        mean, var = running_mean, running_var
    inv = 1 / np.sqrt(var + eps)
    x_hat = (x - mean.reshape(shape)) * inv.reshape(shape)
    dbeta, dgamma = dout.sum(axis=axes), (dout * x_hat).sum(axis=axes)
    scale = (gamma * inv).reshape(shape)
    if not training:
        return dout * scale, dgamma, dbeta
    return scale / m * (m * dout - dbeta.reshape(shape) - x_hat * dgamma.reshape(shape)), dgamma, dbeta


def _torch_bn(x, gamma, beta, running_mean, running_var, training, momentum, eps, dout=None):
    """The same computation with torch.nn.BatchNorm1d/2d in float64 (PyTorch needed)."""
    layer_cls = torch.nn.BatchNorm1d if x.ndim == 2 else torch.nn.BatchNorm2d
    layer = layer_cls(x.shape[1], momentum=momentum, eps=eps).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(gamma))
        layer.bias.copy_(torch.tensor(beta))
        layer.running_mean.copy_(torch.tensor(running_mean))
        layer.running_var.copy_(torch.tensor(running_var))
    layer.train(training)
    tx = torch.tensor(x, requires_grad=True)
    out = layer(tx)
    grads = None
    if dout is not None:
        (out * torch.tensor(dout)).sum().backward()
        grads = (tx.grad.numpy().copy(), layer.weight.grad.numpy().copy(), layer.bias.grad.numpy().copy())
    return (out.detach().numpy(), layer.running_mean.numpy().copy(), layer.running_var.numpy().copy(), grads)


def _bn_case(shape, seed):
    rng = np.random.default_rng(seed)
    c = shape[1]
    x = rng.normal(2.0, 3.0, size=shape)
    return x, rng.normal(1.0, 0.3, c), rng.normal(0.0, 0.5, c), rng.normal(0.0, 0.2, c), rng.uniform(0.5, 2.0, c)


SHAPES = [(8, 3), (5, 4), (4, 3, 5, 6), (2, 2, 3, 3)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_batchnorm_forward_training_matches_torch(reg, shape):
    x, gamma, beta, rm, rv = _bn_case(shape, 11)
    want, want_rm, want_rv = _bn_oracle(x, gamma, beta, rm, rv, True, 0.1, 1e-5)
    if torch is not None:
        t_out, t_rm, t_rv, _ = _torch_bn(x, gamma, beta, rm, rv, True, 0.1, 1e-5)
        assert np.allclose(t_out, want) and np.allclose(t_rm, want_rm) and np.allclose(t_rv, want_rv), ORACLE_BUG
    running_mean, running_var = rm.copy(), rv.copy()
    out, _cache = pair(call(reg.batchnorm_forward, x.copy(), gamma.copy(), beta.copy(), running_mean, running_var),
                       "batchnorm_forward", ("out", "cache"))
    assert_new_float_array(out, x, "out")
    axes = "axis 0" if len(shape) == 2 else "axes (0, 2, 3)"
    assert_close(out, want, msg=f"training output for x of shape {shape} (batch mean and BIASED variance over "
                                f"{axes})")
    assert_close(running_mean, want_rm, msg="running_mean after one training step, updated IN PLACE: "
                                            "(1 - momentum) * running_mean + momentum * batch_mean")
    assert_close(running_var, want_rv, msg="running_var after one training step, updated IN PLACE with the "
                                           "UNBIASED batch variance")


@pytest.mark.parametrize("shape", [(6, 4), (3, 2, 4, 4)], ids=lambda s: "x".join(map(str, s)))
def test_batchnorm_running_stats_after_three_steps_match_torch(reg, shape):
    rng = np.random.default_rng(12)
    c = shape[1]
    gamma, beta = rng.normal(1, 0.2, c), rng.normal(0, 0.2, c)
    running_mean, running_var = np.zeros(c), np.ones(c)
    rm, rv = running_mean.copy(), running_var.copy()
    for step in range(3):
        x = rng.normal(step, 1 + step, size=shape)
        _, rm, rv = _bn_oracle(x, gamma, beta, rm, rv, True, 0.3, 1e-5)
        pair(call(reg.batchnorm_forward, x, gamma, beta, running_mean, running_var, momentum=0.3),
             "batchnorm_forward", ("out", "cache"))
    assert_close(running_mean, rm, msg="running_mean after 3 training steps (momentum 0.3)")
    assert_close(running_var, rv, msg="running_var after 3 training steps (momentum 0.3, unbiased variance)")


@pytest.mark.parametrize("shape", [(5, 3), (3, 2, 4, 4)], ids=lambda s: "x".join(map(str, s)))
def test_batchnorm_forward_eval_uses_running_stats_and_leaves_them(reg, shape):
    x, gamma, beta, rm, rv = _bn_case(shape, 13)
    want, _, _ = _bn_oracle(x, gamma, beta, rm, rv, False, 0.1, 1e-5)
    if torch is not None:
        t_out, t_rm, t_rv, _ = _torch_bn(x, gamma, beta, rm, rv, False, 0.1, 1e-5)
        assert np.allclose(t_out, want) and np.allclose(t_rm, rm) and np.allclose(t_rv, rv), ORACLE_BUG
    running_mean, running_var = rm.copy(), rv.copy()
    out, _ = pair(call(reg.batchnorm_forward, x, gamma, beta, running_mean, running_var, training=False),
                  "batchnorm_forward(training=False)", ("out", "cache"))
    assert_close(out, want, msg="evaluation output: normalized with running_mean and running_var")
    if not (np.array_equal(running_mean, rm) and np.array_equal(running_var, rv)):
        _fail("", "evaluation mode: expected running_mean and running_var unchanged")


def test_batchnorm_forward_eval_accepts_a_single_example(reg):
    x = np.array([[1.0, 2.0, 3.0]])
    out, _ = pair(call(reg.batchnorm_forward, x, np.ones(3), np.zeros(3), np.zeros(3), np.ones(3), training=False),
                  "batchnorm_forward(training=False) on one example", ("out", "cache"))
    assert_close(out, x / np.sqrt(1 + 1e-5), msg="evaluation mode on a single example")


def test_batchnorm_forward_docstring_example(reg):
    x = np.array([[1.0, 2.0], [3.0, 6.0]])
    running_mean, running_var = np.zeros(2), np.ones(2)
    out, _ = pair(call(reg.batchnorm_forward, x, np.ones(2), np.zeros(2), running_mean, running_var),
                  "batchnorm_forward", ("out", "cache"))
    assert_close(out, [[-1 / math.sqrt(1 + 1e-5), -2 / math.sqrt(4 + 1e-5)],
                       [1 / math.sqrt(1 + 1e-5), 2 / math.sqrt(4 + 1e-5)]], msg="the docstring example")
    assert_close(running_mean, [0.2, 0.4], msg="the docstring example: running_mean")
    assert_close(running_var, [1.1, 1.7], msg="the docstring example: running_var (unbiased variances 2 and 8)")


@pytest.mark.parametrize("momentum, eps", [(0.5, 1e-3), (0.01, 0.1)], ids=["m0.5-eps1e-3", "m0.01-eps0.1"])
def test_batchnorm_forward_momentum_and_eps(reg, momentum, eps):
    x, gamma, beta, rm, rv = _bn_case((6, 3), 14)
    want, want_rm, want_rv = _bn_oracle(x, gamma, beta, rm, rv, True, momentum, eps)
    running_mean, running_var = rm.copy(), rv.copy()
    out, _ = pair(call(reg.batchnorm_forward, x, gamma, beta, running_mean, running_var, momentum=momentum, eps=eps),
                  "batchnorm_forward", ("out", "cache"))
    assert_close(out, want, msg=f"output with eps={eps} (added to the variance, inside the square root)")
    assert_close(running_mean, want_rm, msg=f"running_mean with momentum={momentum} (weight of the NEW batch statistic)")
    assert_close(running_var, want_rv, msg=f"running_var with momentum={momentum}")


def test_batchnorm_forward_does_not_modify_its_inputs(reg):
    x, gamma, beta, rm, rv = _bn_case((5, 2), 15)
    saved = (x.copy(), gamma.copy(), beta.copy())
    call(reg.batchnorm_forward, x, gamma, beta, rm, rv)
    if not (np.array_equal(x, saved[0]) and np.array_equal(gamma, saved[1]) and np.array_equal(beta, saved[2])):
        _fail("", "expected x, gamma and beta unchanged (only running_mean and running_var change, in place)")


def test_batchnorm_forward_accepts_a_list(reg):
    out, _ = pair(call(reg.batchnorm_forward, [[1, 2], [3, 6]], np.ones(2), np.zeros(2), np.zeros(2), np.ones(2)),
                  "batchnorm_forward on a list of ints", ("out", "cache"))
    assert_new_float_array(out, np.zeros((2, 2)), "out (from a list of ints)")


@pytest.mark.parametrize("args, why", [
    pytest.param(((2, 3, 4), 3, True), "a 3-D input", id="3d-input"),
    pytest.param(((4,), 4, True), "a 1-D input", id="1d-input"),
    pytest.param(((4, 3), 2, True), "gamma of shape (2,) for 3 features", id="gamma-shape"),
    pytest.param(((1, 3), 3, True), "one value per feature in training mode", id="one-example-training"),
    pytest.param(((1, 2, 1, 1), 2, True), "one value per channel in training mode", id="one-pixel-training"),
])
def test_batchnorm_forward_refuses_invalid_inputs(reg, args, why):
    shape, c, training = args
    assert_raises(ValueError, reg.batchnorm_forward, np.ones(shape), np.ones(c), np.zeros(c), np.zeros(c),
                  np.ones(c), training=training, why=why)


@pytest.mark.parametrize("which", ["gamma", "beta"])
def test_batchnorm_forward_refuses_a_broadcastable_parameter(reg, which):
    args = {"gamma": np.ones(3), "beta": np.zeros(3)}
    args[which] = np.ones(1)
    assert_raises(ValueError, reg.batchnorm_forward, np.ones((4, 3)), args["gamma"], args["beta"], np.zeros(3),
                  np.ones(3), why=f"{which} of shape (1,) for 3 features (it would broadcast: check the shapes)")


def test_batchnorm_forward_refuses_running_stats_of_another_shape(reg):
    assert_raises(ValueError, reg.batchnorm_forward, np.ones((4, 3)), np.ones(3), np.zeros(3), np.zeros(2),
                  np.ones(3), why="running_mean of shape (2,) for 3 features")


def test_batchnorm_forward_eval_refuses_broadcastable_running_stats(reg):
    assert_raises(ValueError, reg.batchnorm_forward, np.ones((4, 3)), np.ones(3), np.zeros(3), np.zeros(3),
                  np.ones(1), training=False,
                  why="evaluation mode, running_var of shape (1,) for 3 features (it would broadcast: check the "
                      "shapes)")


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_batchnorm_backward_training_matches_autograd(reg, shape):
    x, gamma, beta, rm, rv = _bn_case(shape, 21)
    dout = np.random.default_rng(22).normal(size=shape)
    want = _bn_backward_oracle(x, gamma, dout, True, rm, rv, 1e-5)
    if torch is not None:
        t_grads = _torch_bn(x, gamma, beta, rm, rv, True, 0.1, 1e-5, dout=dout)[3]
        assert all(np.allclose(a, b, atol=1e-9) for a, b in zip(t_grads, want)), ORACLE_BUG
    _, cache = pair(call(reg.batchnorm_forward, x, gamma, beta, rm.copy(), rv.copy()), "batchnorm_forward",
                    ("out", "cache"))
    got = call(reg.batchnorm_backward, dout.copy(), cache)
    dx, dgamma, dbeta = pair(got, "batchnorm_backward", ("dx", "dgamma", "dbeta"))
    assert_close(dgamma, want[1], rtol=1e-9, atol=1e-9, msg="dgamma = sum(dout * x_hat) over the normalized axes")
    assert_close(dbeta, want[2], rtol=1e-9, atol=1e-9, msg="dbeta = sum(dout) over the normalized axes")
    assert_close(dx, want[0], rtol=1e-8, atol=1e-9,
                 msg=f"training dx for shape {shape} (the gradient also flows through the batch mean and variance)")
    sums = np.asarray(dx).sum(axis=(0,) if len(shape) == 2 else (0, 2, 3))
    if not np.allclose(sums, 0, atol=1e-9):
        _fail("", f"expected the training dx to sum to 0 over the normalized axes, got {_short(sums)}")


@pytest.mark.parametrize("shape", [(5, 3), (2, 3, 4, 4)], ids=lambda s: "x".join(map(str, s)))
def test_batchnorm_backward_eval_matches_autograd(reg, shape):
    x, gamma, beta, rm, rv = _bn_case(shape, 23)
    dout = np.random.default_rng(24).normal(size=shape)
    want = _bn_backward_oracle(x, gamma, dout, False, rm, rv, 1e-5)
    if torch is not None:
        t_grads = _torch_bn(x, gamma, beta, rm, rv, False, 0.1, 1e-5, dout=dout)[3]
        assert all(np.allclose(a, b, atol=1e-9) for a, b in zip(t_grads, want)), ORACLE_BUG
    _, cache = pair(call(reg.batchnorm_forward, x, gamma, beta, rm.copy(), rv.copy(), training=False),
                    "batchnorm_forward(training=False)", ("out", "cache"))
    dx, dgamma, dbeta = pair(call(reg.batchnorm_backward, dout, cache), "batchnorm_backward", ("dx", "dgamma", "dbeta"))
    assert_close(dx, want[0], rtol=1e-9, atol=1e-10,
                 msg="evaluation dx = dout * gamma / sqrt(running_var + eps) (the statistics are constants)")
    assert_close(dgamma, want[1], rtol=1e-9, atol=1e-10, msg="evaluation dgamma")
    assert_close(dbeta, want[2], rtol=1e-9, atol=1e-10, msg="evaluation dbeta")


def test_batchnorm_backward_docstring_example(reg):
    x = np.array([[1.0, 2.0], [3.0, 6.0]])
    _, cache = pair(call(reg.batchnorm_forward, x, np.ones(2), np.zeros(2), np.zeros(2), np.ones(2)),
                    "batchnorm_forward", ("out", "cache"))
    dx, dgamma, dbeta = pair(call(reg.batchnorm_backward, np.eye(2), cache), "batchnorm_backward",
                             ("dx", "dgamma", "dbeta"))
    assert_close(dgamma, [-1 / math.sqrt(1 + 1e-5), 2 / math.sqrt(4 + 1e-5)], msg="the docstring example: dgamma")
    assert_close(dbeta, [1.0, 1.0], msg="the docstring example: dbeta")


def test_batchnorm_backward_against_finite_differences(reg):
    """An independent check of the training gradient: central differences of sum(dout * out) in x."""
    x, gamma, beta, rm, rv = _bn_case((4, 3), 25)
    dout = np.random.default_rng(26).normal(size=(4, 3))

    def loss(xx):
        out, _, _ = _bn_oracle(xx, gamma, beta, rm, rv, True, 0.1, 1e-5)
        return float(np.sum(dout * out))

    numeric = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[idx] += 1e-6
        minus[idx] -= 1e-6
        numeric[idx] = (loss(plus) - loss(minus)) / 2e-6
    _, cache = pair(call(reg.batchnorm_forward, x, gamma, beta, rm.copy(), rv.copy()), "batchnorm_forward",
                    ("out", "cache"))
    dx, _, _ = pair(call(reg.batchnorm_backward, dout, cache), "batchnorm_backward", ("dx", "dgamma", "dbeta"))
    assert_close(dx, numeric, rtol=1e-5, atol=1e-6, msg="training dx against central finite differences")


def test_batchnorm_backward_does_not_modify_dout(reg):
    x, gamma, beta, rm, rv = _bn_case((5, 2), 27)
    dout = np.random.default_rng(28).normal(size=(5, 2))
    saved = dout.copy()
    _, cache = pair(call(reg.batchnorm_forward, x, gamma, beta, rm, rv), "batchnorm_forward", ("out", "cache"))
    call(reg.batchnorm_backward, dout, cache)
    if not np.array_equal(dout, saved):
        _fail("", "expected dout unchanged (compute dx in a new array)")


# ================================================================== layer norm (20.17)
def _ln_oracle(x, gamma, beta, eps):
    mean, var = x.mean(axis=-1, keepdims=True), x.var(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    if torch is not None:
        ref = torch.nn.functional.layer_norm(torch.tensor(x), (x.shape[-1],),
                                             None if gamma is None else torch.tensor(gamma),
                                             None if beta is None else torch.tensor(beta), eps)
        assert np.allclose(ref.numpy(), out, atol=1e-10), ORACLE_BUG
    return out


def test_layer_norm_docstring_examples(reg):
    assert_close(call(reg.layer_norm, [[1.0, 2.0, 3.0]]), _ln_oracle(np.array([[1.0, 2.0, 3.0]]), None, None, 1e-5),
                 msg="layer_norm([[1, 2, 3]])")
    assert_close(call(reg.layer_norm, [[1.0, 2.0, 3.0]], gamma=np.full(3, 2.0), beta=np.ones(3)),
                 _ln_oracle(np.array([[1.0, 2.0, 3.0]]), np.full(3, 2.0), np.ones(3), 1e-5),
                 msg="layer_norm([[1, 2, 3]], gamma=2, beta=1)")


@pytest.mark.parametrize("shape", [(5,), (4, 6), (2, 3, 7)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("affine", [False, True], ids=["no-affine", "gamma-beta"])
def test_layer_norm_matches_torch(reg, shape, affine):
    rng = np.random.default_rng(31)
    x = rng.normal(3.0, 2.0, size=shape)
    gamma = rng.normal(1.0, 0.3, shape[-1]) if affine else None
    beta = rng.normal(0.0, 0.5, shape[-1]) if affine else None
    got = call(reg.layer_norm, x.copy(), gamma, beta)
    assert_new_float_array(got, x, "the output")
    assert_close(got, _ln_oracle(x, gamma, beta, 1e-5), msg=f"layer_norm on shape {shape} (statistics over the LAST "
                                                            "axis of each sample, biased variance)")


def test_layer_norm_eps(reg):
    x = np.array([[0.0, 0.1, 0.2], [5.0, 5.0, 5.0]])
    assert_close(call(reg.layer_norm, x, eps=0.01), _ln_oracle(x, None, None, 0.01),
                 msg="eps=0.01, added to the variance inside the square root (a constant row gives zeros)")


def test_layer_norm_each_sample_on_its_own(reg):
    x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    got = np.asarray(call(reg.layer_norm, x))
    if not np.allclose(got.mean(axis=-1), 0, atol=1e-12) or not np.allclose(got.std(axis=-1), 1, atol=1e-4):
        _fail("", f"expected each row to have mean 0 and standard deviation 1, got means {_short(got.mean(axis=-1))} "
                  f"and standard deviations {_short(got.std(axis=-1))} (normalize along the LAST axis, not the batch)")


def test_layer_norm_does_not_modify_x(reg):
    x = np.array([[1.0, 4.0], [2.0, 8.0]])
    saved = x.copy()
    call(reg.layer_norm, x, np.ones(2), np.zeros(2))
    if not np.array_equal(x, saved):
        _fail("", "expected x unchanged (compute the output in a new array)")


@pytest.mark.parametrize("gamma, beta, why", [
    pytest.param(np.ones(2), None, "gamma of shape (2,) for D = 3", id="gamma-shape"),
    pytest.param(np.ones(1), None, "gamma of shape (1,) for D = 3 (it would broadcast: check the shapes)",
                 id="gamma-broadcastable"),
    pytest.param(None, np.ones((1, 3)), "beta of shape (1, 3) instead of (3,)", id="beta-2d"),
])
def test_layer_norm_refuses_parameters_of_another_shape(reg, gamma, beta, why):
    assert_raises(ValueError, reg.layer_norm, np.ones((4, 3)), gamma, beta, why=why)


# ================================================================== L2 penalty (20.17)
def test_l2_penalty_docstring_example(reg):
    got = pair(call(reg.l2_penalty, [np.array([1.0, 2.0]), np.array([[3.0]])], lam=0.5), "l2_penalty",
               ("penalty", "grads"))
    penalty, grads = got
    if not isinstance(penalty, float):
        _fail("", f"expected the penalty as a Python float, got {type(penalty).__name__} (convert with float(...))")
    assert_close(penalty, 3.5, msg="(lam / 2) * (1 + 4 + 9) with lam = 0.5")
    if not isinstance(grads, list) or len(grads) != 2:
        _fail("", f"expected a list of 2 gradients, got {type(grads).__name__}")
    assert_close(grads[0], [0.5, 1.0], msg="grads[0] = lam * W")
    assert_close(grads[1], [[1.5]], msg="grads[1] = lam * W")


def test_l2_penalty_matches_autograd(reg):
    rng = np.random.default_rng(41)
    weights = [rng.normal(size=(4, 3)), rng.normal(size=(3, 2)), rng.normal(size=5)]
    lam = 0.03
    want_penalty = 0.5 * lam * sum(float(np.sum(w * w)) for w in weights)
    want_grads = [lam * w for w in weights]
    if torch is not None:
        tw = [torch.tensor(w, requires_grad=True) for w in weights]
        total = 0.5 * lam * sum((t ** 2).sum() for t in tw)
        total.backward()
        assert abs(float(total.detach()) - want_penalty) < 1e-12, ORACLE_BUG
        assert all(np.allclose(t.grad.numpy(), g) for t, g in zip(tw, want_grads)), ORACLE_BUG
    saved = [w.copy() for w in weights]
    penalty, grads = pair(call(reg.l2_penalty, weights, lam), "l2_penalty", ("penalty", "grads"))
    assert_close(penalty, want_penalty, msg="penalty = (lam / 2) * sum of the squared weights (the 1/2 matters)")
    for i, (g, want) in enumerate(zip(grads, want_grads)):
        assert_new_float_array(g, want, f"grads[{i}]")
        assert_close(g, want, msg=f"grads[{i}] = lam * weights[{i}]")
    if not all(np.array_equal(w, s) for w, s in zip(weights, saved)):
        _fail("", "expected the weights unchanged (the gradients are new arrays)")
    if any(g is w or np.shares_memory(g, w) for g, w in zip(grads, weights)):
        _fail("", "expected new gradient arrays, got a weight array itself (or a view)")


def test_l2_penalty_gradients_reproduce_sgd_weight_decay(reg):
    rng = np.random.default_rng(42)
    w, g = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    lam, lr = 0.1, 0.05
    want = w - lr * (g + lam * w)                       # torch.optim.SGD(lr, weight_decay=lam), one step
    if torch is not None:
        tw = torch.tensor(w.copy(), requires_grad=True)
        opt = torch.optim.SGD([tw], lr=lr, weight_decay=lam)
        tw.grad = torch.tensor(g.copy())
        opt.step()
        assert np.allclose(tw.detach().numpy(), want), ORACLE_BUG
    _, grads = pair(call(reg.l2_penalty, [w], lam), "l2_penalty", ("penalty", "grads"))
    assert_close(w - lr * (g + grads[0]), want, msg="expected the data gradient plus your L2 gradient to give the "
                                                    "step of SGD(weight_decay=lam)")


def test_l2_penalty_zero_lambda(reg):
    penalty, grads = pair(call(reg.l2_penalty, [np.array([3.0, -4.0])], 0.0), "l2_penalty(lam=0)", ("penalty", "grads"))
    assert_close(penalty, 0.0, msg="lam = 0: expected a penalty of 0")
    assert_close(grads[0], [0.0, 0.0], msg="lam = 0: expected zero gradients")


def test_l2_penalty_refuses_a_negative_lambda(reg):
    assert_raises(ValueError, reg.l2_penalty, [np.ones(2)], -0.1, why="lam < 0")


# ================================================================== EarlyStopping (20.17)
def _stopper(reg, **kwargs):
    return call(reg.EarlyStopping, expected="an EarlyStopping object", **kwargs)


def _run(stopper, values, params=None):
    out = []
    for v in values:
        result = call(stopper.step, v, params, expected="a bool from step()")
        if not isinstance(result, bool):
            kind = f"{type(result).__module__}.{type(result).__name__}".replace("builtins.", "")
            _fail("", f"expected step() to return a Python bool, got {kind} ({result!r}): convert with bool(...)")
        out.append(result)
    return out


def test_early_stopping_docstring_example(reg):
    stopper = _stopper(reg, patience=2)
    got = _run(stopper, [1.0, 0.8, 0.9, 0.85])
    if got != [False, False, False, True]:
        _fail("", f"expected [False, False, False, True] for patience=2 and the values 1.0, 0.8, 0.9, 0.85, got {got}")
    if (stopper.best_, stopper.best_epoch_, stopper.counter_) != (0.8, 1, 2):
        _fail("", f"expected (best_, best_epoch_, counter_) = (0.8, 1, 2), got "
                  f"{(stopper.best_, stopper.best_epoch_, stopper.counter_)}")


@pytest.mark.parametrize("patience", [1, 3, 10], ids=lambda p: f"patience={p}")
def test_early_stopping_stops_patience_checks_after_the_best(reg, patience):
    values = [5.0, 4.0, 3.0, 2.0] + [2.5] * 20                 # best at check 3 (0-based)
    stopper = _stopper(reg, patience=patience)
    got = _run(stopper, values[:4 + patience])
    expected = [False] * (3 + patience) + [True]
    if got != expected:
        _fail("", f"expected the first True at check {3 + patience} (best at check 3, patience {patience}), got {got}")
    if stopper.best_epoch_ != 3 or stopper.counter_ != patience or not stopper.stopped_:
        _fail("", f"expected best_epoch_ = 3, counter_ = {patience}, stopped_ = True, got {stopper.best_epoch_}, "
                  f"{stopper.counter_}, {stopper.stopped_}")


def test_early_stopping_counter_restarts_after_an_improvement(reg):
    stopper = _stopper(reg, patience=3)
    got = _run(stopper, [1.0, 1.2, 1.1, 0.9, 1.0, 1.0, 1.0])
    if got != [False, False, False, False, False, False, True]:
        _fail("", f"expected the counter to go back to 0 at the improvement 0.9 (check 3), so the first True at check 6, got {got}")
    if stopper.best_epoch_ != 3:
        _fail("", f"expected best_epoch_ = 3, got {stopper.best_epoch_}")


def test_early_stopping_attributes_before_and_during(reg):
    stopper = _stopper(reg, patience=2)
    if not (stopper.best_ == math.inf and stopper.best_epoch_ == -1 and stopper.counter_ == 0
            and stopper.n_checks_ == 0 and stopper.stopped_ is False):
        _fail("", "expected best_ = inf, best_epoch_ = -1, counter_ = 0, n_checks_ = 0, stopped_ = False before the first check")
    _run(stopper, [2.0])
    if not (stopper.best_ == 2.0 and stopper.best_epoch_ == 0 and stopper.improved_ is True and stopper.n_checks_ == 1):
        _fail("", f"expected the first check to be an improvement: best_ = 2.0, best_epoch_ = 0, improved_ = True, "
                  f"n_checks_ = 1; got {stopper.best_}, {stopper.best_epoch_}, {stopper.improved_}, {stopper.n_checks_}")
    _run(stopper, [2.5])
    if stopper.improved_ is not False or stopper.counter_ != 1 or stopper.n_checks_ != 2:
        _fail("", f"expected improved_ = False, counter_ = 1, n_checks_ = 2 after a worse value, got "
                  f"{stopper.improved_}, {stopper.counter_}, {stopper.n_checks_}")


def test_early_stopping_an_equal_value_is_not_an_improvement(reg):
    stopper = _stopper(reg, patience=1)
    got = _run(stopper, [1.0, 1.0])
    if got != [False, True]:
        _fail("", f"mode 'min', min_delta 0: expected an equal value not to be an improvement ([False, True]), "
                  f"got {got} (the test is value < best_ - min_delta, strictly)")


def test_early_stopping_max_mode(reg):
    stopper = _stopper(reg, patience=2, mode="max")
    if stopper.best_ != -math.inf:
        _fail("", f"mode 'max': expected best_ = -inf before the first check, got {stopper.best_}")
    got = _run(stopper, [0.80, 0.85, 0.84, 0.86, 0.86, 0.85])
    if got != [False, False, False, False, False, True]:
        _fail("", f"mode 'max' (an accuracy): expected [False, False, False, False, False, True], got {got} "
                  "(in mode 'max', a HIGHER value is better)")
    if stopper.best_ != 0.86 or stopper.best_epoch_ != 3:
        _fail("", f"mode 'max': expected best_ = 0.86 at check 3, got {stopper.best_} at check {stopper.best_epoch_}")


@pytest.mark.parametrize("mode, values, expected", [
    pytest.param("min", [1.0, 0.95], [False, True], id="min-too-small"),
    pytest.param("min", [1.0, 0.85, 0.80], [False, False, True], id="min-enough"),
    pytest.param("max", [0.5, 0.55], [False, True], id="max-too-small"),
    pytest.param("max", [0.5, 0.65, 0.70], [False, False, True], id="max-enough"),
])
def test_early_stopping_min_delta(reg, mode, values, expected):
    stopper = _stopper(reg, patience=1, min_delta=0.1, mode=mode)
    got = _run(stopper, values)
    if got != expected:
        _fail("", f"mode {mode!r}, min_delta 0.1, patience 1: expected {expected} for {values}, got {got} (an "
                  "improvement must beat best_ by MORE than min_delta: "
                  f"{'value < best_ - min_delta' if mode == 'min' else 'value > best_ + min_delta'})")


def test_early_stopping_nan_is_never_an_improvement(reg):
    stopper = _stopper(reg, patience=2)
    got = _run(stopper, [1.0, float("nan"), float("nan")])
    if got != [False, False, True] or stopper.best_ != 1.0:
        _fail("", f"expected a nan never to be an improvement ([False, False, True], best_ = 1.0), got {got}, best_ = {stopper.best_}")
    stopper = _stopper(reg, patience=1)
    got = _run(stopper, [float("nan"), 2.0])
    if got != [True, False] or stopper.best_ != 2.0:
        # a nan first: no improvement (the counter reaches 1 = patience); the next finite value is one
        _fail("", f"expected a first nan not to count as an improvement (step returns True with patience 1), "
                  f"then 2.0 to be one: got {got}, best_ = {stopper.best_}")


def test_early_stopping_keeps_deep_copies_of_the_best_params(reg):
    w, b = np.array([1.0, 2.0]), np.array([0.5])
    stopper = _stopper(reg, patience=2)
    _run(stopper, [0.5], [w, b])
    w[0], b[0] = 9.0, 9.0                                       # training goes on: the arrays change in place
    _run(stopper, [0.7], [w, b])
    if stopper.best_params_ is None or len(stopper.best_params_) != 2:
        _fail("", "expected best_params_ to hold copies of the 2 arrays of the best check")
    assert_close(stopper.best_params_[0], [1.0, 2.0], msg="best_params_[0] (a COPY taken at the best check, "
                                                          "not the array itself)")
    assert_close(stopper.best_params_[1], [0.5], msg="best_params_[1]")


def test_early_stopping_restore_writes_in_place(reg):
    w = np.array([[1.0, -1.0], [2.0, 0.0]])
    keep = w
    stopper = _stopper(reg, patience=1)
    _run(stopper, [0.3], [w])
    w += 5.0
    _run(stopper, [0.4], [w])
    call(stopper.restore, [w], expected="restore() to run")
    if keep is not w:
        _fail("", "the test array was replaced")
    assert_close(w, [[1.0, -1.0], [2.0, 0.0]], msg="expected restore to copy the best values back INTO the given "
                                                   "arrays (p[...] = saved)")


def test_early_stopping_restore_best_false_saves_nothing(reg):
    stopper = _stopper(reg, patience=1, restore_best=False)
    _run(stopper, [0.3], [np.ones(2)])
    if stopper.best_params_ is not None:
        _fail("", "restore_best=False: expected best_params_ to stay None")
    assert_raises(ValueError, stopper.restore, [np.ones(2)], why="restore_best=False: nothing was saved")


def test_early_stopping_restore_without_params_refuses(reg):
    stopper = _stopper(reg, patience=1)
    _run(stopper, [0.3])
    assert_raises(ValueError, stopper.restore, [np.ones(2)], why="step() never received params")


@pytest.mark.parametrize("saved, target", [((2,), (3,)), ((1,), (3,)), ((2,), (2, 2))],
                         ids=["2-into-3", "1-into-3-broadcastable", "2-into-2x2"])
def test_early_stopping_restore_refuses_another_shape(reg, saved, target):
    stopper = _stopper(reg, patience=1)
    _run(stopper, [0.3], [np.ones(saved)])
    assert_raises(ValueError, stopper.restore, [np.zeros(target)],
                  why=f"an array of shape {target} for a saved {saved} (no broadcasting)")


def test_early_stopping_saves_only_at_improvements(reg):
    w = np.array([1.0])
    stopper = _stopper(reg, patience=5)
    for value, new in [(1.0, 1.0), (0.5, 2.0), (0.6, 3.0), (0.4, 4.0), (0.45, 5.0)]:
        w[0] = new
        _run(stopper, [value], [w])
    assert_close(stopper.best_params_[0], [4.0], msg="expected the parameters of the best check (0.4, w = 4)")


@pytest.mark.parametrize("kwargs, why", [
    pytest.param({"patience": 0}, "patience < 1", id="patience-0"),
    pytest.param({"min_delta": -0.1}, "min_delta < 0", id="negative-delta"),
    pytest.param({"mode": "maximum"}, "unknown mode", id="bad-mode"),
])
def test_early_stopping_refuses_invalid_hyperparameters(reg, kwargs, why):
    _run(_stopper(reg, patience=1), [1.0])                      # a valid stopper works (__init__ is provided)
    assert_raises(ValueError, reg.EarlyStopping, why=why, **kwargs)


def test_early_stopping_two_instances_are_independent(reg):
    a, b = _stopper(reg, patience=2), _stopper(reg, patience=2)
    _run(a, [1.0, 2.0])
    if b.n_checks_ != 0 or b.counter_ != 0:
        _fail("", "expected a second EarlyStopping to be untouched by the first (no class attributes)")
    copy.deepcopy(a)                                            # a stopper can be copied (saved with a checkpoint)
