"""Tests of mylearn.conv (chapter 21): oracle tests and properties.

    pytest tests/test_ch21_conv.py              # your code (mon_travail/mylearn/conv.py)
    pytest tests/test_ch21_conv.py --impl=ref   # the reference

Oracles: PyTorch in float64 (output shapes of ``torch.nn.functional.conv2d`` and ``max_pool2d``;
``torch.nn.functional.pad``, ``unfold``, ``conv2d``, ``max_pool2d``, ``avg_pool2d``, ``conv_transpose2d``;
``torch.autograd.grad`` through ``conv2d`` for the backward pass) and NumPy: ``numpy.pad`` and the formulas
of the docstrings, transcribed in this file with explicit loops (over the output positions, the channels
and the kernel offsets). With PyTorch, the transcribed formulas are checked against torch at every test
(a disagreement is a bug of this file, not yours); without PyTorch, they are the oracle, so that no test
is skipped. Your functions are always called on copies of the test data, and the tests
``..._does_not_modify_...`` check that the inputs are left as they were.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_conv_output_size_ or test_pad2d_"`` (21.12), ``-k "test_im2col_"`` (21.14),
``-k "test_conv2d_ and not test_conv2d_backward_"`` (21.15), ``-k "test_max_pool2d_ or test_avg_pool2d_"``
(21.18), ``-k "test_conv_transpose2d_"`` (21.30), ``-k "test_conv2d_backward_"`` (21.31).

Contract precisions written in the statements of these exercises and checked here: the inputs are never
modified and may be lists (of ints); every array returned is a new float64 array, never a view of an input
(even ``pad2d`` with a padding of 0, ``im2col`` with a 1 x 1 kernel or ``max_pool2d`` with a 1 x 1 window);
``conv_output_size`` returns a Python ``int`` (not a ``numpy.int64``, a float or a bool) and refuses a size
of 0 even when the padding would give a window; every argument that takes "an int or a tuple of 2 ints"
accepts both, the first number for the rows (H) and the second for the columns (W); a kernel size, stride
or dilation < 1, a negative padding and a window that does not fit raise ``ValueError`` (as in
``conv_output_size``) in ``im2col``, ``conv2d``, ``max_pool2d`` and ``avg_pool2d`` (those that have the
argument), and so does, in ``conv2d`` and ``conv_transpose2d``, a 3-D weight, ``groups`` < 1 or a ``bias``
of another shape than (C_out,), even a broadcastable one; ``padding="same"`` with an even kernel span puts the extra row and column of zeros
at the bottom and on the right (as PyTorch), and refuses a stride > 1 along any axis; ``max_pool2d`` and
``avg_pool2d`` refuse a padding above ``kernel_size // 2`` along any axis; ``conv2d_backward`` gives a zero
gradient to the input cells that no window sees (a stride that does not divide the input) and refuses a
``dout`` of another shape than the forward output, even a broadcastable one; ``conv_transpose2d`` refuses a
padding outside [0, k - 1] along any axis, a stride < 1 and an empty output. With ties in max pooling, only
the values are compared (any window position may be the maximum).
The first line of every failure message says what was expected (it is the line that ``pytest -rf``
shows).
"""

import warnings

import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # the PyTorch oracles run on Colab; the NumPy formulas take over here
    torch = None
    F = None

# a failed comparison between this file's NumPy oracle and PyTorch is a bug of the tests, not of the learner
ORACLE_BUG = ("bug in tests/test_ch21_conv.py (not in your code): the NumPy oracle of this test "
              "disagrees with torch")


@pytest.fixture
def conv(mylearn_module):
    return mylearn_module("conv")


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


def assert_close(result, expected, rtol=1e-9, atol=1e-10, msg=""):
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


def triple(result, what="conv2d_backward"):
    """Unpack the (dx, dW, db) result, or fail with a message that says what was expected."""
    if result is None:
        _fail("", f"expected the tuple (dx, dW, db) from {what}, got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 3:
        _fail("", f"expected the tuple (dx, dW, db) from {what}, got {type(result).__name__}"
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
            _fail("", f"expected {name} unchanged by {what}, got {_short(arr)} (compute the result in a new array)")


def _tensor(a):
    return torch.tensor(np.asarray(a, dtype=np.float64))


def _torch_agrees(ours, theirs, atol=1e-9):
    assert np.shape(ours) == tuple(theirs.shape) and np.allclose(ours, theirs.detach().numpy(), rtol=1e-9,
                                                                 atol=atol), ORACLE_BUG


def _pairs(value):
    return tuple(value) if isinstance(value, tuple) else (value, value)


# ------------------------------------------------------------------ NumPy oracles (checked against torch)
def out_len(size, k, s=1, p=0, d=1):
    """The formula of conv_output_size (checked against the output shape of F.conv2d)."""
    result = (size + 2 * p - d * (k - 1) - 1) // s + 1
    if torch is not None:
        shape = F.conv2d(torch.zeros(1, 1, size, 1, dtype=torch.float64), torch.zeros(1, 1, k, 1, dtype=torch.float64),
                         stride=(s, 1), padding=(p, 0), dilation=(d, 1)).shape
        assert shape[2] == result, ORACLE_BUG
        if p <= k // 2 and d == 1:
            assert F.max_pool2d(torch.zeros(1, 1, size, 1), (k, 1), (s, 1), (p, 0)).shape[2] == result, ORACLE_BUG
    return result


def np_pad(x, padding, value=0.0):
    """numpy.pad on the two spatial axes (checked against F.pad)."""
    ph, pw = _pairs(padding)
    result = np.pad(np.asarray(x, dtype=float), ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=value)
    if torch is not None:
        _torch_agrees(result, F.pad(_tensor(x), (pw, pw, ph, ph), value=value))
    return result


def np_unfold(x, kernel, stride=1, padding=0, dilation=1):
    """Column l = i * W_out + j, row (c * k_h + u) * k_w + v (checked against F.unfold)."""
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = map(_pairs, (kernel, stride, padding, dilation))
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    ho, wo = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1, (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    cols = np.zeros((n, c * kh * kw, ho * wo))
    for ch in range(c):
        for u in range(kh):
            for v in range(kw):
                for i in range(ho):
                    for j in range(wo):
                        cols[:, (ch * kh + u) * kw + v, i * wo + j] = xp[:, ch, i * sh + u * dh, j * sw + v * dw]
    if torch is not None:
        _torch_agrees(cols, F.unfold(_tensor(x), (kh, kw), dilation=(dh, dw), padding=(ph, pw), stride=(sh, sw)))
    return cols


def np_conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    """out[n, o, i, j] = b[o] + sum over the channels c of the group of o, u, v of
    w[o, c, u, v] * x_padded[n, c, i * s_h + u * d_h, j * s_w + v * d_w] (checked against F.conv2d)."""
    (sh, sw), (dh, dw) = _pairs(stride), _pairs(dilation)
    n, c_in, h, wd = x.shape
    c_out, c_group, kh, kw = w.shape
    if padding == "valid":
        pads = (0, 0, 0, 0)
    elif padding == "same":                     # the extra row / column at the bottom / on the right
        th, tw = dh * (kh - 1), dw * (kw - 1)
        pads = (th // 2, th - th // 2, tw // 2, tw - tw // 2)
    else:
        ph, pw = _pairs(padding)
        pads = (ph, ph, pw, pw)
    xp = np.pad(x, ((0, 0), (0, 0), pads[:2], pads[2:]))
    ho = (xp.shape[2] - dh * (kh - 1) - 1) // sh + 1
    wo = (xp.shape[3] - dw * (kw - 1) - 1) // sw + 1
    per_group = c_out // groups
    out = np.zeros((n, c_out, ho, wo))
    for o in range(c_out):
        g = o // per_group
        for i in range(ho):
            for j in range(wo):
                window = xp[:, g * c_group:(g + 1) * c_group,
                            i * sh:i * sh + dh * (kh - 1) + 1:dh, j * sw:j * sw + dw * (kw - 1) + 1:dw]
                out[:, o, i, j] = (window * w[o]).sum(axis=(1, 2, 3)) + (0.0 if b is None else b[o])
    if torch is not None:
        with warnings.catch_warnings():          # "same" with an even kernel: a harmless UserWarning
            warnings.simplefilter("ignore")
            ref = F.conv2d(_tensor(x), _tensor(w), None if b is None else _tensor(b), stride=(sh, sw),
                           padding=padding if isinstance(padding, str) else _pairs(padding), dilation=(dh, dw),
                           groups=groups)
        _torch_agrees(out, ref)
    return out


def np_pool(x, kernel, stride=None, padding=0, kind="max"):
    """Max (padding -inf) or mean (padding 0, divided by k_h * k_w) of each window (checked against torch)."""
    kh, kw = _pairs(kernel)
    sh, sw = (kh, kw) if stride is None else _pairs(stride)
    ph, pw = _pairs(padding)
    fill = -np.inf if kind == "max" else 0.0
    xp = np.pad(np.asarray(x, dtype=float), ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=fill)
    ho, wo = (xp.shape[2] - kh) // sh + 1, (xp.shape[3] - kw) // sw + 1
    out = np.zeros(x.shape[:2] + (ho, wo))
    for i in range(ho):
        for j in range(wo):
            window = xp[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            out[:, :, i, j] = window.max(axis=(2, 3)) if kind == "max" else window.sum(axis=(2, 3)) / (kh * kw)
    if torch is not None:
        function = F.max_pool2d if kind == "max" else F.avg_pool2d
        _torch_agrees(out, function(_tensor(x), (kh, kw), (sh, sw), (ph, pw)))
    return out


def np_conv2d_backward(dout, x, w, stride=1, padding=0):
    """Each output gradient sent back to its window (checked against torch.autograd.grad through F.conv2d)."""
    (sh, sw), (ph, pw) = _pairs(stride), _pairs(padding)
    n, c_in, h, wd = x.shape
    kh, kw = w.shape[2:]
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    dxp, dw = np.zeros_like(xp), np.zeros_like(w)
    for i in range(dout.shape[2]):
        for j in range(dout.shape[3]):
            rows, cols = slice(i * sh, i * sh + kh), slice(j * sw, j * sw + kw)
            g = dout[:, :, i, j]                                        # (N, C_out)
            dw += np.einsum("no,nauv->oauv", g, xp[:, :, rows, cols])   # weight sharing: a sum over positions
            dxp[:, :, rows, cols] += np.einsum("no,oauv->nauv", g, w)
    dx, db = dxp[:, :, ph:ph + h, pw:pw + wd], dout.sum(axis=(0, 2, 3))
    if torch is not None:
        tx, tw = _tensor(x).requires_grad_(), _tensor(w).requires_grad_()
        tb = torch.zeros(w.shape[0], dtype=torch.float64, requires_grad=True)
        out = F.conv2d(tx, tw, tb, stride=(sh, sw), padding=(ph, pw))
        for ours, theirs in zip((dx, dw, db), torch.autograd.grad(out, (tx, tw, tb), _tensor(dout))):
            _torch_agrees(ours, theirs)
    return dx, dw, db


def np_conv_transpose2d(x, w, b=None, stride=1, padding=0):
    """Each input cell spreads x[n, c, i, j] * w[c] over its window of the output, then the padding is cut
    (checked against F.conv_transpose2d)."""
    (sh, sw), (ph, pw) = _pairs(stride), _pairs(padding)
    n, c_in, h, wd = x.shape
    c_out, kh, kw = w.shape[1:]
    full = np.zeros((n, c_out, (h - 1) * sh + kh, (wd - 1) * sw + kw))
    for i in range(h):
        for j in range(wd):
            full[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw] += np.einsum("nc,couv->nouv", x[:, :, i, j], w)
    out = full[:, :, ph:full.shape[2] - ph, pw:full.shape[3] - pw]
    if b is not None:
        out = out + np.asarray(b).reshape(1, -1, 1, 1)
    if torch is not None:
        _torch_agrees(out, F.conv_transpose2d(_tensor(x), _tensor(w), None if b is None else _tensor(b),
                                              stride=(sh, sw), padding=(ph, pw)))
    return out


def _rand(seed, *shape):
    return np.random.default_rng(seed).normal(size=shape)


# ================================================================== conv_output_size (21.12)
@pytest.mark.parametrize("args, want", [((28, 3), 26), ((28, 3, 1, 1), 28), ((28, 2, 2), 14),
                                        ((300, 3, 3, 2), 101)], ids=["28-k3", "28-k3-p1", "28-k2-s2", "300-k3-s3-p2"])
def test_conv_output_size_docstring_examples(conv, args, want):
    names = ("size", "kernel_size", "stride", "padding")
    got = call(conv.conv_output_size, *args)
    if got != want:
        _fail("", f"expected {want} for conv_output_size({', '.join(f'{n}={a}' for n, a in zip(names, args))}), got {got!r}")


@pytest.mark.parametrize("size, k, s, p, d", [
    pytest.param(7, 3, 1, 0, 1, id="k3"),
    pytest.param(7, 3, 2, 0, 1, id="k3-s2"),
    pytest.param(8, 3, 2, 0, 1, id="k3-s2-rounded-down"),
    pytest.param(10, 2, 3, 0, 1, id="stride-above-kernel"),
    pytest.param(9, 4, 1, 2, 1, id="even-kernel-p2"),
    pytest.param(11, 3, 1, 0, 2, id="dilation-2"),
    pytest.param(12, 3, 2, 1, 3, id="dilation-3-s2-p1"),
    pytest.param(5, 5, 1, 0, 1, id="fits-exactly"),
    pytest.param(5, 3, 1, 0, 2, id="dilated-fits-exactly"),
    pytest.param(1, 3, 1, 1, 1, id="size-1-padded"),
    pytest.param(6, 1, 1, 0, 1, id="k1"),
    pytest.param(6, 1, 4, 0, 1, id="k1-s4"),
])
def test_conv_output_size_matches_torch_shapes(conv, size, k, s, p, d):
    want = out_len(size, k, s, p, d)
    got = call(conv.conv_output_size, size, k, stride=s, padding=p, dilation=d)
    if got != want:
        _fail("", f"expected {want} for size={size}, kernel_size={k}, stride={s}, padding={p}, dilation={d}, "
                  f"got {got!r} (floor((size + 2p - d(k - 1) - 1) / s) + 1)")


def test_conv_output_size_returns_a_python_int(conv):
    for args in [(28, 3), (300, 3, 3, 2), (8, 3, 2), (5, 3, 1, 0, 2)]:
        got = call(conv.conv_output_size, *args)
        if not isinstance(got, int) or isinstance(got, bool):
            kind = f"{type(got).__module__}.{type(got).__name__}".replace("builtins.", "")
            _fail("", f"expected a Python int for conv_output_size{args}, got {kind} ({got!r}): use // or "
                      "convert with int(...)")


@pytest.mark.parametrize("args, why", [
    pytest.param((0, 1, 1, 1, 1), "size = 0 (the input length must be >= 1, even with padding)", id="size-0"),
    pytest.param((5, 0), "kernel_size = 0", id="kernel-0"),
    pytest.param((5, 2, 0), "stride = 0", id="stride-0"),
    pytest.param((5, 2, -1), "stride < 0", id="stride-negative"),
    pytest.param((5, 2, 1, -1), "padding < 0", id="padding-negative"),
    pytest.param((5, 2, 1, 0, 0), "dilation = 0", id="dilation-0"),
    pytest.param((2, 3), "a window of 3 in an input of 2 (the result would be 0)", id="window-too-big"),
    pytest.param((3, 6, 1, 1), "a window of 6 in an input of 3 padded to 5", id="window-too-big-padded"),
    pytest.param((4, 3, 1, 0, 2), "a dilated window spanning 2 * (3 - 1) + 1 = 5 cells in an input of 4",
                 id="dilated-window-too-big"),
])
def test_conv_output_size_refuses_invalid_arguments(conv, args, why):
    assert_raises(ValueError, conv.conv_output_size, *args, why=why)


# ================================================================== pad2d (21.12)
def test_pad2d_docstring_examples(conv):
    got = call(conv.pad2d, np.ones((1, 1, 2, 2)), 1)
    assert_close(got, np_pad(np.ones((1, 1, 2, 2)), 1), msg="pad2d(np.ones((1, 1, 2, 2)), 1)")
    assert_close(call(conv.pad2d, np.ones((1, 1, 1, 1)), (0, 1)), [[[[0.0, 1.0, 0.0]]]],
                 msg="pad2d(np.ones((1, 1, 1, 1)), (0, 1)): p_h = 0 rows, p_w = 1 column on each side")


@pytest.mark.parametrize("shape, padding, value", [
    pytest.param((2, 3, 4, 5), 2, 0.0, id="p2"),
    pytest.param((1, 2, 3, 4), (1, 3), 0.0, id="rows-1-cols-3"),
    pytest.param((2, 1, 4, 2), (3, 0), 0.0, id="rows-only"),
    pytest.param((1, 1, 2, 3), (0, 2), 0.0, id="cols-only"),
    pytest.param((2, 2, 3, 3), (2, 1), -1.5, id="value"),
    pytest.param((1, 2, 3, 3), 1, -np.inf, id="minus-inf"),
])
def test_pad2d_matches_numpy_and_torch(conv, shape, padding, value):
    x = _rand(1, *shape)
    want = np_pad(x, padding, value)
    got = call(conv.pad2d, x.copy(), padding, value=value)
    assert_close(got, want, msg=f"pad2d(x of shape {shape}, padding={padding}, value={value}) (p_h rows at the top "
                                "AND at the bottom, p_w columns on the left AND on the right)")


def test_pad2d_zero_padding_returns_a_new_float_array(conv):
    x = _rand(2, 2, 3, 4, 5)
    got = call(conv.pad2d, x, 0)
    assert_new_float_array(got, x.shape, "pad2d(x, 0)", inputs=(x,))
    assert_close(got, x, msg="pad2d(x, 0) is a copy of x")
    got = call(conv.pad2d, np.arange(4).reshape(1, 1, 2, 2), 1, value=0.5)
    assert_new_float_array(got, (1, 1, 4, 4), "pad2d of an int array")
    assert_close(got, np_pad(np.arange(4.0).reshape(1, 1, 2, 2), 1, 0.5),
                 msg="pad2d of an int array with value 0.5 (a float array: the 0.5 is kept)")


def test_pad2d_does_not_modify_x_and_accepts_lists(conv):
    x = _rand(3, 1, 2, 3, 3)
    saved = x.copy()
    call(conv.pad2d, x, (1, 2), value=7.0)
    assert_unchanged({"x": x}, {"x": saved}, "pad2d")
    got = call(conv.pad2d, [[[[1, 2], [3, 4]]]], 1)
    assert_new_float_array(got, (1, 1, 4, 4), "pad2d of a list of ints")
    assert_close(got, np_pad(np.array([[[[1.0, 2.0], [3.0, 4.0]]]]), 1), msg="pad2d of a list of ints")


@pytest.mark.parametrize("shape, padding, why", [
    pytest.param((3, 4, 4), 1, "a 3-D x (one image without the batch axis)", id="3d"),
    pytest.param((4, 4), 1, "a 2-D x", id="2d"),
    pytest.param((1, 1, 1, 4, 4), 1, "a 5-D x", id="5d"),
    pytest.param((1, 1, 4, 4), -1, "a negative padding", id="negative"),
    pytest.param((1, 1, 4, 4), (1, -1), "a negative p_w", id="negative-pw"),
    pytest.param((1, 1, 4, 4), (-2, 0), "a negative p_h", id="negative-ph"),
])
def test_pad2d_refuses_invalid_inputs(conv, shape, padding, why):
    assert_raises(ValueError, conv.pad2d, np.ones(shape), padding, why=why)


# ================================================================== im2col (21.14)
def test_im2col_docstring_example(conv):
    x = np.arange(9.0).reshape(1, 1, 3, 3)
    assert_close(call(conv.im2col, x.copy(), 2), [[[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]]],
                 msg="im2col(arange(9).reshape(1, 1, 3, 3), 2): column l = the patch at position l")


IM2COL_CASES = [
    pytest.param((1, 1, 4, 4), 3, 1, 0, 1, id="k3"),
    pytest.param((2, 3, 5, 6), 3, 1, 1, 1, id="batch-channels-p1"),
    pytest.param((1, 2, 5, 7), (2, 3), 1, 0, 1, id="rectangular-kernel"),
    pytest.param((2, 2, 7, 6), 3, (2, 1), 0, 1, id="stride-2-1"),
    pytest.param((1, 2, 5, 5), (3, 2), (1, 2), (2, 1), 1, id="padding-2-1"),
    pytest.param((1, 1, 7, 8), 3, 1, 0, (2, 1), id="dilation-2-1"),
    pytest.param((2, 1, 9, 9), (2, 3), 3, (1, 0), (1, 2), id="everything"),
    pytest.param((1, 3, 6, 6), 2, 4, 0, 1, id="stride-above-kernel"),
]


@pytest.mark.parametrize("shape, k, s, p, d", IM2COL_CASES)
def test_im2col_matches_unfold(conv, shape, k, s, p, d):
    x = _rand(4, *shape)
    want = np_unfold(x, k, s, p, d)
    got = call(conv.im2col, x.copy(), k, stride=s, padding=p, dilation=d)
    assert_close(got, want, msg=f"im2col(x of shape {shape}, kernel_size={k}, stride={s}, padding={p}, dilation={d}) "
                                "(row (c * k_h + u) * k_w + v, column i * W_out + j)")


@pytest.mark.parametrize("shape, k, s, p, why", [
    pytest.param((2, 3, 4, 5), 1, 1, 0, "a 1 x 1 kernel: one column per pixel, one row per channel", id="k1"),
    pytest.param((1, 2, 4, 5), (4, 5), 1, 0, "a window that fits exactly: a single column", id="fits-exactly"),
    pytest.param((1, 1, 2, 2), 4, 1, 1, "a window that fits exactly in the padded input", id="fits-padded"),
    pytest.param((1, 1, 7, 7), 2, 3, 0, "stride 3 > kernel 2: some pixels are skipped", id="stride-above-kernel"),
])
def test_im2col_edge_cases(conv, shape, k, s, p, why):
    x = _rand(5, *shape)
    assert_close(call(conv.im2col, x.copy(), k, stride=s, padding=p), np_unfold(x, k, s, p), msg=f"im2col, {why}")


def test_im2col_returns_a_new_float_array(conv):
    x = _rand(6, 2, 3, 4, 4)
    got = call(conv.im2col, x, 1)
    assert_new_float_array(got, (2, 3, 16), "im2col(x, 1)", inputs=(x,))
    got = call(conv.im2col, [[[[1, 2, 3], [4, 5, 6]]]], (2, 2))
    assert_new_float_array(got, (1, 4, 2), "im2col of a list of ints")
    assert_close(got, [[[1, 2], [2, 3], [4, 5], [5, 6]]], msg="im2col of a list of ints")


def test_im2col_does_not_modify_x(conv):
    x = _rand(7, 1, 2, 5, 5)
    saved = x.copy()
    call(conv.im2col, x, 3, stride=2, padding=1, dilation=1)
    assert_unchanged({"x": x}, {"x": saved}, "im2col")


@pytest.mark.parametrize("shape, kwargs, why", [
    pytest.param((2, 4, 4), {"kernel_size": 2}, "a 3-D x", id="3d"),
    pytest.param((1, 1, 3, 3), {"kernel_size": 4}, "a window of 4 in an input of 3", id="window-too-big"),
    pytest.param((1, 1, 5, 2), {"kernel_size": (2, 3)}, "a window 3 wide in an input 2 wide", id="window-too-wide"),
    pytest.param((1, 1, 4, 4), {"kernel_size": 2, "dilation": 4}, "a dilated window spanning 4 * (2 - 1) + 1 = 5 cells in 4",
                 id="dilated-too-big"),
    pytest.param((1, 1, 4, 4), {"kernel_size": 0}, "kernel_size = 0", id="kernel-0"),
    pytest.param((1, 1, 4, 4), {"kernel_size": 2, "stride": 0}, "stride = 0", id="stride-0"),
    pytest.param((1, 1, 4, 4), {"kernel_size": 2, "stride": (1, 0)}, "stride s_w = 0", id="stride-w-0"),
    pytest.param((1, 1, 4, 4), {"kernel_size": 2, "padding": -1}, "a negative padding", id="padding-negative"),
    pytest.param((1, 1, 4, 4), {"kernel_size": 2, "dilation": 0}, "dilation = 0", id="dilation-0"),
])
def test_im2col_refuses_invalid_inputs(conv, shape, kwargs, why):
    assert_raises(ValueError, conv.im2col, np.ones(shape), why=why, **kwargs)


# ================================================================== conv2d (21.15)
def _conv_data(seed, n, c_in, c_out, h, w, k, groups=1, bias=True):
    rng = np.random.default_rng(seed)
    kh, kw = _pairs(k)
    x = rng.normal(size=(n, c_in, h, w))
    weight = rng.normal(size=(c_out, c_in // groups, kh, kw))
    return x, weight, (rng.normal(size=c_out) if bias else None)


def test_conv2d_docstring_examples(conv):
    x = np.arange(9.0).reshape(1, 1, 3, 3)
    assert_close(call(conv.conv2d, x.copy(), np.ones((1, 1, 2, 2))), [[[[8.0, 12.0], [20.0, 24.0]]]],
                 msg="conv2d(arange(9).reshape(1, 1, 3, 3), ones((1, 1, 2, 2)))")
    got = call(conv.conv2d, np.zeros((2, 3, 28, 28)), np.zeros((8, 3, 3, 3)), padding="same")
    assert_new_float_array(got, (2, 8, 28, 28), 'conv2d(..., padding="same")')


CONV_CASES = [
    #            n, c_in, c_out, h, w, k, stride, padding, dilation, groups, bias
    pytest.param((2, 3, 4, 7, 6, 3, 1, 0, 1, 1, True), id="basic-3x3"),
    pytest.param((1, 2, 3, 6, 7, (2, 3), 1, 0, 1, 1, True), id="rectangular-kernel"),
    pytest.param((2, 2, 2, 9, 8, 3, (2, 1), 0, 1, 1, True), id="stride-2-1"),
    pytest.param((1, 3, 2, 7, 7, 3, 2, 1, 1, 1, False), id="stride-2-padding-1-no-bias"),
    pytest.param((2, 1, 3, 5, 6, (3, 2), 1, (2, 0), 1, 1, True), id="padding-2-0"),
    pytest.param((1, 2, 2, 9, 9, 3, 1, 0, 2, 1, True), id="dilation-2"),
    pytest.param((1, 2, 3, 10, 9, (2, 3), (2, 1), (1, 2), (3, 2), 1, True), id="everything"),
    pytest.param((2, 4, 6, 6, 6, 3, 1, 1, 1, 2, True), id="groups-2"),
    pytest.param((1, 6, 3, 5, 5, 2, 1, 0, 1, 3, True), id="groups-3"),
    pytest.param((2, 3, 3, 6, 5, 3, 1, 1, 1, 3, True), id="depthwise"),
    pytest.param((1, 3, 6, 6, 6, 3, 2, 1, 1, 3, False), id="depthwise-multiplier-2"),
    pytest.param((2, 4, 3, 5, 6, 1, 1, 0, 1, 1, True), id="k1-channel-mixing"),
    pytest.param((1, 2, 2, 4, 5, (4, 5), 1, 0, 1, 1, True), id="fits-exactly-output-1x1"),
    pytest.param((1, 1, 2, 10, 9, 2, 3, 0, 1, 1, True), id="stride-above-kernel"),
]


@pytest.mark.parametrize("spec", CONV_CASES)
def test_conv2d_matches_torch(conv, spec):
    n, c_in, c_out, h, w, k, s, p, d, groups, bias = spec
    x, weight, b = _conv_data(8, n, c_in, c_out, h, w, k, groups, bias)
    want = np_conv2d(x, weight, b, s, p, d, groups)
    got = call(conv.conv2d, x.copy(), weight.copy(), None if b is None else b.copy(), stride=s, padding=p,
               dilation=d, groups=groups)
    assert_close(got, want, msg=f"conv2d(x {x.shape}, weight {weight.shape}, bias={'yes' if bias else 'None'}, "
                                f"stride={s}, padding={p}, dilation={d}, groups={groups})")


@pytest.mark.parametrize("k, d", [
    pytest.param(3, 1, id="k3"),
    pytest.param(4, 1, id="k4-even"),
    pytest.param(2, 1, id="k2-even"),
    pytest.param((2, 3), 1, id="k2x3"),
    pytest.param(2, 3, id="k2-dilation-3"),
    pytest.param(3, 2, id="k3-dilation-2"),
    pytest.param((4, 1), (1, 2), id="k4x1-dilation-1-2"),
])
def test_conv2d_same_padding_matches_torch(conv, k, d):
    x, weight, b = _conv_data(9, 2, 2, 3, 7, 6, k)
    want = np_conv2d(x, weight, b, 1, "same", d)
    got = call(conv.conv2d, x.copy(), weight.copy(), b.copy(), padding="same", dilation=d)
    assert_close(got, want, msg=f'conv2d(padding="same", kernel {_pairs(k)}, dilation={d}) (total padding '
                                "d * (k - 1) per axis; when it is odd, the extra row and column go at the bottom "
                                "and on the right)")


def test_conv2d_same_padding_accepts_an_explicit_stride_of_1(conv):
    x, weight, b = _conv_data(10, 1, 2, 2, 5, 5, 3)
    want = np_conv2d(x, weight, b, 1, "same")
    assert_close(call(conv.conv2d, x.copy(), weight.copy(), b.copy(), stride=(1, 1), padding="same"), want,
                 msg='conv2d(stride=(1, 1), padding="same")')


def test_conv2d_valid_padding_is_no_padding(conv):
    x, weight, b = _conv_data(11, 2, 3, 2, 6, 7, (3, 2))
    want = np_conv2d(x, weight, b, (2, 1), "valid")
    assert_close(call(conv.conv2d, x.copy(), weight.copy(), b.copy(), stride=(2, 1), padding="valid"), want,
                 msg='conv2d(padding="valid") (no padding)')


def test_conv2d_is_a_cross_correlation(conv):
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    top_left = np.array([[[[1.0, 0.0], [0.0, 0.0]]]])
    assert_close(call(conv.conv2d, x.copy(), top_left), x[:, :, :3, :3],
                 msg="a kernel with a single 1 at (u, v) = (0, 0) picks x[i, j] (the kernel is NOT flipped)")


def test_conv2d_returns_float64_and_accepts_lists(conv):
    got = call(conv.conv2d, [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], [[[[1, 0], [0, -1]]]], [2])
    assert_new_float_array(got, (1, 1, 2, 2), "conv2d of lists of ints")
    assert_close(got, [[[[-2.0, -2.0], [-2.0, -2.0]]]], msg="conv2d of lists of ints (x[i, j] - x[i+1, j+1] + 2)")


def test_conv2d_does_not_modify_its_inputs(conv):
    x, weight, b = _conv_data(12, 2, 4, 4, 5, 5, 3, groups=2)
    arrays = {"x": x, "weight": weight, "bias": b}
    saved = {name: a.copy() for name, a in arrays.items()}
    call(conv.conv2d, x, weight, b, stride=1, padding="same", groups=2)
    call(conv.conv2d, x, weight, b, stride=2, padding=1, groups=2)
    assert_unchanged(arrays, saved, "conv2d")


@pytest.mark.parametrize("x_shape, w_shape, b_shape, kwargs, why", [
    pytest.param((3, 5, 5), (2, 3, 3, 3), None, {}, "a 3-D x", id="x-3d"),
    pytest.param((1, 3, 5, 5), (3, 3, 3), None, {}, "a 3-D weight", id="weight-3d"),
    pytest.param((1, 3, 5, 5), (2, 4, 3, 3), None, {}, "weight for 4 input channels, x with 3", id="c-in-mismatch"),
    pytest.param((1, 4, 5, 5), (2, 4, 3, 3), None, {"groups": 2},
                 "groups=2: the weight must have C_in // groups = 2 input channels, not 4", id="groups-weight-shape"),
    pytest.param((1, 3, 5, 5), (4, 1, 3, 3), None, {"groups": 2}, "groups=2 does not divide C_in = 3",
                 id="groups-c-in"),
    pytest.param((1, 4, 5, 5), (3, 2, 3, 3), None, {"groups": 2}, "groups=2 does not divide C_out = 3",
                 id="groups-c-out"),
    pytest.param((1, 2, 5, 5), (2, 2, 3, 3), None, {"groups": 0}, "groups = 0", id="groups-0"),
    pytest.param((1, 2, 5, 5), (3, 2, 3, 3), (4,), {}, "bias of shape (4,) for 3 output channels",
                 id="bias-shape"),
    pytest.param((1, 2, 5, 5), (3, 2, 3, 3), (1,), {},
                 "bias of shape (1,) for 3 output channels (it would broadcast: check the shape)",
                 id="bias-broadcastable"),
    pytest.param((1, 2, 5, 5), (3, 2, 3, 3), (3, 1), {}, "bias of shape (3, 1) instead of (3,)", id="bias-2d"),
    pytest.param((1, 1, 5, 5), (1, 1, 3, 3), None, {"padding": "full"}, 'an unknown padding string "full"',
                 id="padding-full"),
    pytest.param((1, 1, 5, 5), (1, 1, 3, 3), None, {"padding": "zeros"}, 'an unknown padding string "zeros"',
                 id="padding-zeros"),
    pytest.param((1, 1, 6, 6), (1, 1, 3, 3), None, {"padding": "same", "stride": 2}, '"same" with stride 2',
                 id="same-stride-2"),
    pytest.param((1, 1, 6, 6), (1, 1, 3, 3), None, {"padding": "same", "stride": (1, 2)}, '"same" with stride (1, 2)',
                 id="same-stride-1-2"),
    pytest.param((1, 1, 3, 3), (1, 1, 4, 4), None, {}, "a 4 x 4 kernel on a 3 x 3 image", id="window-too-big"),
    pytest.param((1, 1, 5, 5), (1, 1, 3, 3), None, {"dilation": 3}, "a dilated kernel spanning 7 cells on 5",
                 id="dilated-too-big"),
    pytest.param((1, 1, 5, 5), (1, 1, 3, 3), None, {"padding": -1}, "a negative padding", id="padding-negative"),
    pytest.param((1, 1, 5, 5), (1, 1, 3, 3), None, {"padding": (0, -1)}, "a negative p_w", id="padding-w-negative"),
    pytest.param((1, 1, 5, 5), (1, 1, 3, 3), None, {"stride": 0}, "stride = 0", id="stride-0"),
    pytest.param((1, 1, 5, 5), (1, 1, 3, 3), None, {"dilation": (1, 0)}, "dilation d_w = 0", id="dilation-0"),
])
def test_conv2d_refuses_invalid_inputs(conv, x_shape, w_shape, b_shape, kwargs, why):
    bias = None if b_shape is None else np.ones(b_shape)
    assert_raises(ValueError, conv.conv2d, np.ones(x_shape), np.ones(w_shape), bias, why=why, **kwargs)


# ================================================================== max_pool2d and avg_pool2d (21.18)
POOL_CASES = [
    pytest.param((1, 1, 4, 4), 2, None, 0, id="k2"),
    pytest.param((2, 3, 6, 5), 2, None, 0, id="k2-odd-width"),
    pytest.param((2, 2, 5, 5), 3, 1, 0, id="k3-s1-overlap"),
    pytest.param((1, 2, 6, 7), (2, 3), (1, 2), 0, id="k2x3-s1x2"),
    pytest.param((2, 1, 5, 5), 3, 2, 1, id="k3-s2-p1"),
    pytest.param((1, 2, 4, 6), (2, 3), None, (1, 0), id="k2x3-p1x0"),
    pytest.param((1, 1, 5, 4), (4, 2), 1, (2, 1), id="padding-at-the-limit"),
    pytest.param((1, 2, 7, 7), 2, 3, 0, id="stride-above-kernel"),
    pytest.param((1, 1, 3, 4), (3, 4), None, 0, id="fits-exactly-output-1x1"),
    pytest.param((2, 2, 3, 3), 1, None, 0, id="k1"),
]


def test_max_pool2d_docstring_example(conv):
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    assert_close(call(conv.max_pool2d, x.copy(), 2), [[[[5.0, 7.0], [13.0, 15.0]]]],
                 msg="max_pool2d(arange(16).reshape(1, 1, 4, 4), 2)")


@pytest.mark.parametrize("shape, k, s, p", POOL_CASES)
def test_max_pool2d_matches_torch(conv, shape, k, s, p):
    x = _rand(13, *shape)
    want = np_pool(x, k, s, p, "max")
    got = call(conv.max_pool2d, x.copy(), k, stride=s, padding=p)
    assert_close(got, want, msg=f"max_pool2d(x of shape {shape}, kernel_size={k}, stride={s}, padding={p})"
                                + (" (stride None means stride = kernel_size)" if s is None else ""))


def test_max_pool2d_padding_acts_as_minus_infinity(conv):
    x = -1.0 - np.arange(18.0).reshape(1, 2, 3, 3)               # all values < 0
    want = np_pool(x, 3, 2, 1, "max")
    got = call(conv.max_pool2d, x.copy(), 3, stride=2, padding=1)
    assert_close(got, want, msg="max_pool2d of negative values with padding=1 (a padded cell is never the maximum: "
                                "pad with -inf, not 0)")


def test_max_pool2d_ties_only_values_count(conv):
    x = np.array([[[[1, 1, 0, 0], [1, 1, 0, 0], [2, 2, 2, 2], [2, 2, 2, 2]],
                   [[-3, -3, 5, 5], [-3, -3, 5, 5], [7, 7, -1, -1], [7, 7, -1, -1]]]], dtype=float)
    want = np_pool(x, 2, None, 0, "max")
    assert_close(call(conv.max_pool2d, x.copy(), 2), want, msg="max_pool2d with ties (several maxima per window)")
    assert_close(call(conv.max_pool2d, x.copy(), 3, stride=1, padding=1), np_pool(x, 3, 1, 1, "max"),
                 msg="max_pool2d with ties, overlapping windows and padding")


def test_max_pool2d_returns_a_new_float_array(conv):
    x = _rand(14, 2, 3, 4, 4)
    got = call(conv.max_pool2d, x, 1)
    assert_new_float_array(got, x.shape, "max_pool2d(x, 1)", inputs=(x,))
    got = call(conv.max_pool2d, [[[[1, 2], [3, 4]]]], 2)
    assert_new_float_array(got, (1, 1, 1, 1), "max_pool2d of a list of ints")
    assert_close(got, [[[[4.0]]]], msg="max_pool2d of a list of ints")


def test_max_pool2d_does_not_modify_x(conv):
    x = _rand(15, 1, 2, 5, 5)
    saved = x.copy()
    call(conv.max_pool2d, x, 3, stride=2, padding=1)
    assert_unchanged({"x": x}, {"x": saved}, "max_pool2d")


POOL_INVALID = [
    pytest.param((2, 4, 4), {"kernel_size": 2}, "a 3-D x", id="3d"),
    pytest.param((1, 1, 6, 6), {"kernel_size": 2, "padding": 2}, "padding 2 > kernel_size // 2 = 1",
                 id="padding-above-half"),
    pytest.param((1, 1, 6, 6), {"kernel_size": 3, "padding": 2}, "padding 2 > 3 // 2 = 1", id="padding-above-half-k3"),
    pytest.param((1, 1, 6, 6), {"kernel_size": (4, 2), "padding": (2, 2)}, "p_w = 2 > k_w // 2 = 1",
                 id="padding-above-half-w"),
    pytest.param((1, 1, 6, 6), {"kernel_size": 3, "padding": -1}, "a negative padding", id="padding-negative"),
    pytest.param((1, 1, 3, 3), {"kernel_size": 4}, "a window of 4 in an input of 3", id="window-too-big"),
    pytest.param((1, 1, 5, 2), {"kernel_size": (2, 3)}, "a window 3 wide in an input 2 wide", id="window-too-wide"),
    pytest.param((1, 1, 4, 4), {"kernel_size": 0}, "kernel_size = 0", id="kernel-0"),
    pytest.param((1, 1, 4, 4), {"kernel_size": 2, "stride": 0}, "stride = 0", id="stride-0"),
]


@pytest.mark.parametrize("shape, kwargs, why", POOL_INVALID)
def test_max_pool2d_refuses_invalid_inputs(conv, shape, kwargs, why):
    assert_raises(ValueError, conv.max_pool2d, np.ones(shape), why=why, **kwargs)


def test_avg_pool2d_docstring_example(conv):
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    assert_close(call(conv.avg_pool2d, x.copy(), 2), [[[[2.5, 4.5], [10.5, 12.5]]]],
                 msg="avg_pool2d(arange(16).reshape(1, 1, 4, 4), 2)")


@pytest.mark.parametrize("shape, k, s, p", POOL_CASES)
def test_avg_pool2d_matches_torch(conv, shape, k, s, p):
    x = _rand(16, *shape)
    want = np_pool(x, k, s, p, "avg")
    got = call(conv.avg_pool2d, x.copy(), k, stride=s, padding=p)
    assert_close(got, want, msg=f"avg_pool2d(x of shape {shape}, kernel_size={k}, stride={s}, padding={p})"
                                + (" (stride None means stride = kernel_size)" if s is None else ""))


def test_avg_pool2d_counts_the_padding(conv):
    want = np_pool(np.ones((1, 1, 3, 3)), 3, 1, 1, "avg")       # corners 4/9, edges 6/9, centre 1
    got = call(conv.avg_pool2d, np.ones((1, 1, 3, 3)), 3, stride=1, padding=1)
    assert_close(got, want, msg="avg_pool2d of ones with padding=1 (count_include_pad=True: every window is "
                                "divided by k_h * k_w, padded zeros included)")


def test_avg_pool2d_returns_a_new_float_array(conv):
    x = _rand(17, 2, 3, 4, 4)
    got = call(conv.avg_pool2d, x, 1)
    assert_new_float_array(got, x.shape, "avg_pool2d(x, 1)", inputs=(x,))
    got = call(conv.avg_pool2d, [[[[1, 2], [3, 4]]]], 2)
    assert_new_float_array(got, (1, 1, 1, 1), "avg_pool2d of a list of ints")
    assert_close(got, [[[[2.5]]]], msg="avg_pool2d of a list of ints")


def test_avg_pool2d_does_not_modify_x(conv):
    x = _rand(18, 1, 2, 5, 5)
    saved = x.copy()
    call(conv.avg_pool2d, x, 3, stride=2, padding=1)
    assert_unchanged({"x": x}, {"x": saved}, "avg_pool2d")


@pytest.mark.parametrize("shape, kwargs, why", POOL_INVALID)
def test_avg_pool2d_refuses_invalid_inputs(conv, shape, kwargs, why):
    assert_raises(ValueError, conv.avg_pool2d, np.ones(shape), why=why, **kwargs)


# ================================================================== conv_transpose2d (21.30)
def test_conv_transpose2d_docstring_examples(conv):
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    want = [[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]]
    assert_close(call(conv.conv_transpose2d, x.copy(), np.ones((1, 1, 2, 2)), stride=2), want,
                 msg="conv_transpose2d([[1, 2], [3, 4]], ones((1, 1, 2, 2)), stride=2)")
    assert_close(call(conv.conv_transpose2d, np.ones((1, 1, 2, 2)), np.ones((1, 1, 2, 2))),
                 [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], msg="conv_transpose2d(ones((1, 1, 2, 2)), ones((1, 1, 2, 2)))")


TRANSPOSE_CASES = [
    #            n, c_in, c_out, h, w, k, stride, padding, bias
    pytest.param((1, 1, 1, 3, 3, 3, 1, 0, False), id="k3-s1"),
    pytest.param((2, 2, 3, 3, 4, 3, 2, 0, True), id="k3-s2"),
    pytest.param((1, 3, 2, 4, 3, 4, 2, 1, True), id="k4-s2-p1"),
    pytest.param((2, 2, 2, 3, 3, (3, 2), (2, 1), (1, 0), True), id="rectangular-s2x1-p1x0"),
    pytest.param((1, 2, 4, 2, 3, 3, 3, 1, False), id="k3-s3-p1"),
    pytest.param((1, 1, 2, 3, 3, 3, 1, 2, True), id="padding-k-minus-1"),
    pytest.param((2, 3, 1, 3, 2, 1, 2, 0, True), id="k1-s2"),
    pytest.param((1, 2, 3, 1, 1, (2, 3), 1, 0, True), id="one-pixel"),
]


@pytest.mark.parametrize("spec", TRANSPOSE_CASES)
def test_conv_transpose2d_matches_torch(conv, spec):
    n, c_in, c_out, h, w, k, s, p, bias = spec
    rng = np.random.default_rng(19)
    kh, kw = _pairs(k)
    x, weight = rng.normal(size=(n, c_in, h, w)), rng.normal(size=(c_in, c_out, kh, kw))
    b = rng.normal(size=c_out) if bias else None
    want = np_conv_transpose2d(x, weight, b, s, p)
    got = call(conv.conv_transpose2d, x.copy(), weight.copy(), None if b is None else b.copy(), stride=s, padding=p)
    assert_close(got, want, msg=f"conv_transpose2d(x {x.shape}, weight {weight.shape} (C_in, C_out, k_h, k_w), "
                                f"bias={'yes' if bias else 'None'}, stride={s}, padding={p})")


@pytest.mark.parametrize("h, w, k, s, p", [
    pytest.param(4, 4, 2, 2, 0, id="double"), pytest.param(5, 3, 3, 2, 1, id="k3-s2-p1"),
    pytest.param(3, 6, (4, 3), (3, 1), (1, 2), id="rectangular"), pytest.param(7, 7, 5, 1, 2, id="k5-p2"),
])
def test_conv_transpose2d_output_shape(conv, h, w, k, s, p):
    (kh, kw), (sh, sw), (ph, pw) = _pairs(k), _pairs(s), _pairs(p)
    shape = (2, 5, (h - 1) * sh - 2 * ph + kh, (w - 1) * sw - 2 * pw + kw)
    got = call(conv.conv_transpose2d, np.ones((2, 3, h, w)), np.ones((3, 5, kh, kw)), stride=s, padding=p)
    assert_new_float_array(got, shape, f"conv_transpose2d(H={h}, W={w}, k={k}, stride={s}, padding={p}): "
                                       "the output, with H_out = (H - 1) * s - 2p + k,")


@pytest.mark.parametrize("h, w, k, s, p", [
    pytest.param(4, 4, 3, 1, 0, id="s1"), pytest.param(5, 5, 3, 2, 0, id="s2"),
    pytest.param(5, 7, 3, 2, 1, id="s2-p1"), pytest.param(6, 5, (2, 3), (2, 1), (0, 1), id="rectangular"),
])
def test_conv_transpose2d_is_the_input_gradient_of_conv2d(conv, h, w, k, s, p):
    kh, kw = _pairs(k)
    rng = np.random.default_rng(20)
    x, weight = rng.normal(size=(2, 3, h, w)), rng.normal(size=(4, 3, kh, kw))     # conv2d: 3 -> 4 channels
    ho = out_len(h, kh, _pairs(s)[0], _pairs(p)[0])
    wo = out_len(w, kw, _pairs(s)[1], _pairs(p)[1])
    dout = rng.normal(size=(2, 4, ho, wo))
    dx, _, _ = np_conv2d_backward(dout, x, weight, s, p)
    got = call(conv.conv_transpose2d, dout.copy(), weight.copy(), stride=s, padding=p)
    assert_close(got, dx, msg=f"conv_transpose2d(dout, W) with the weight W of conv2d (C_out, C_in, k_h, k_w) read "
                              f"as (C_in, C_out, k_h, k_w) of the transposed convolution (stride={s}, padding={p}): "
                              "expected dL/dx of conv2d")


def test_conv_transpose2d_returns_float64_and_accepts_lists(conv):
    got = call(conv.conv_transpose2d, [[[[1, 2]]]], [[[[1, -1]]]], [3], stride=2)
    assert_new_float_array(got, (1, 1, 1, 4), "conv_transpose2d of lists of ints")
    assert_close(got, [[[[4.0, 2.0, 5.0, 1.0]]]], msg="conv_transpose2d([[1, 2]], [[1, -1]], bias 3, stride=2)")


def test_conv_transpose2d_does_not_modify_its_inputs(conv):
    rng = np.random.default_rng(21)
    arrays = {"x": rng.normal(size=(2, 2, 3, 3)), "weight": rng.normal(size=(2, 3, 3, 3)),
              "bias": rng.normal(size=3)}
    saved = {name: a.copy() for name, a in arrays.items()}
    call(conv.conv_transpose2d, arrays["x"], arrays["weight"], arrays["bias"], stride=2, padding=1)
    assert_unchanged(arrays, saved, "conv_transpose2d")


@pytest.mark.parametrize("x_shape, w_shape, b_shape, kwargs, why", [
    pytest.param((2, 3, 3), (2, 1, 2, 2), None, {}, "a 3-D x", id="x-3d"),
    pytest.param((1, 2, 3, 3), (2, 2, 2), None, {}, "a 3-D weight", id="weight-3d"),
    pytest.param((1, 2, 3, 3), (3, 2, 2, 2), None, {},
                 "weight with 3 input channels (axis 0) for x with 2", id="c-in-mismatch"),
    pytest.param((1, 2, 3, 3), (2, 3, 2, 2), (2,), {}, "bias of shape (2,) for 3 output channels", id="bias-shape"),
    pytest.param((1, 2, 3, 3), (2, 3, 2, 2), (1,), {},
                 "bias of shape (1,) for 3 output channels (it would broadcast: check the shape)",
                 id="bias-broadcastable"),
    pytest.param((1, 1, 3, 3), (1, 1, 3, 3), None, {"padding": -1}, "padding -1 < 0", id="padding-negative"),
    pytest.param((1, 1, 6, 6), (1, 1, 3, 3), None, {"padding": 3}, "padding 3 > k - 1 = 2 (the output would be 2 x 2)",
                 id="padding-k"),
    pytest.param((1, 1, 4, 4), (1, 1, 3, 1), None, {"padding": (1, 1)}, "p_w = 1 > k_w - 1 = 0",
                 id="padding-w-above"),
    pytest.param((1, 1, 3, 3), (1, 1, 2, 2), None, {"stride": 0}, "stride = 0", id="stride-0"),
    pytest.param((1, 1, 1, 1), (1, 1, 3, 3), None, {"padding": 2},
                 "an empty output: (1 - 1) * 1 - 2 * 2 + 3 = -1", id="empty-output"),
])
def test_conv_transpose2d_refuses_invalid_inputs(conv, x_shape, w_shape, b_shape, kwargs, why):
    bias = None if b_shape is None else np.ones(b_shape)
    assert_raises(ValueError, conv.conv_transpose2d, np.ones(x_shape), np.ones(w_shape), bias, why=why, **kwargs)


# ================================================================== conv2d_backward (21.31)
def test_conv2d_backward_docstring_example(conv):
    x = np.arange(9.0).reshape(1, 1, 3, 3)
    dx, dw, db = triple(call(conv.conv2d_backward, np.ones((1, 1, 2, 2)), x.copy(), np.ones((1, 1, 2, 2))))
    assert_close(dx, [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], msg="the docstring example: dx (how many windows see "
                                                                "each pixel)")
    assert_close(dw, [[[[8, 12], [20, 24]]]], msg="the docstring example: dW")
    assert_close(db, [4.0], msg="the docstring example: db")


BACKWARD_CASES = [
    #            n, c_in, c_out, h, w, k, stride, padding
    pytest.param((1, 1, 1, 4, 4, 2, 1, 0), id="one-channel"),
    pytest.param((2, 3, 4, 6, 5, 3, 1, 0), id="batch-channels"),
    pytest.param((2, 2, 3, 7, 7, 3, 2, 0), id="stride-2"),
    pytest.param((1, 2, 2, 6, 7, 3, 2, 0), id="stride-2-unseen-cells"),
    pytest.param((2, 2, 2, 5, 5, 3, 1, 1), id="padding-1"),
    pytest.param((1, 3, 2, 6, 5, (3, 2), (2, 1), (1, 0)), id="rectangular-s2x1-p1x0"),
    pytest.param((2, 1, 3, 5, 6, (2, 3), 3, (0, 2)), id="stride-above-kernel-p0x2"),
    pytest.param((3, 4, 2, 4, 3, 1, 1, 0), id="k1"),
    pytest.param((2, 2, 3, 4, 5, (4, 5), 1, 0), id="fits-exactly-output-1x1"),
    pytest.param((1, 2, 2, 4, 4, 3, 2, 1), id="s2-p1"),
]


@pytest.mark.parametrize("spec", BACKWARD_CASES)
def test_conv2d_backward_matches_autograd(conv, spec):
    n, c_in, c_out, h, w, k, s, p = spec
    x, weight, _ = _conv_data(22, n, c_in, c_out, h, w, k, bias=False)
    ho, wo = out_len(h, weight.shape[2], _pairs(s)[0], _pairs(p)[0]), out_len(w, weight.shape[3], _pairs(s)[1],
                                                                              _pairs(p)[1])
    dout = _rand(23, n, c_out, ho, wo)
    want_dx, want_dw, want_db = np_conv2d_backward(dout, x, weight, s, p)
    dx, dw, db = triple(call(conv.conv2d_backward, dout.copy(), x.copy(), weight.copy(), stride=s, padding=p))
    case = f"x {x.shape}, weight {weight.shape}, stride={s}, padding={p}"
    assert_close(db, want_db, msg=f"db for {case} (sum of dout over the images and the positions)")
    assert_close(dw, want_dw, msg=f"dW for {case} (sum over the images and the positions of dout times the window)")
    assert_close(dx, want_dx, msg=f"dx for {case} (each dout value sent back to the cells of its window, padding "
                                  "removed)")


def test_conv2d_backward_shared_weights_sum_over_positions_and_images(conv):
    x = _rand(24, 3, 1, 4, 5)                         # 3 images
    dout = np.ones((3, 1, 3, 4))                      # dL/dout = 1 everywhere: L = sum of the outputs
    want_dw = np.array([[[[x[:, 0, u:u + 3, v:v + 4].sum() for v in range(2)] for u in range(2)]]])
    np_conv2d_backward(dout, x, np.ones((1, 1, 2, 2)))                  # checks the oracle against torch
    dx, dw, db = triple(call(conv.conv2d_backward, dout, x.copy(), np.ones((1, 1, 2, 2))))
    assert_close(dw, want_dw, msg="dW[0, 0, u, v] = sum over the 3 images AND the 12 positions of the input cell "
                                  "seen by the weight (u, v) (weight sharing)")
    assert_close(db, [36.0], msg="db = sum of dout over the 3 images and the 12 positions")


def test_conv2d_backward_cells_seen_by_no_window_get_zero(conv):
    x, weight = _rand(25, 1, 1, 5, 5), _rand(26, 1, 1, 2, 2)
    dout = _rand(27, 1, 1, 2, 2)                      # stride 2: row 4 and column 4 are never seen
    want_dx = np_conv2d_backward(dout, x, weight, 2)[0]
    dx, _, _ = triple(call(conv.conv2d_backward, dout, x.copy(), weight.copy(), stride=2))
    assert_close(dx, want_dx, msg="dx with stride 2 on a 5 x 5 image and a 2 x 2 kernel (the last row and column, "
                                  "seen by no window, get a zero gradient)")


def test_conv2d_backward_returns_three_new_float_arrays_and_accepts_lists(conv):
    x, weight = _rand(28, 2, 3, 5, 4), _rand(29, 2, 3, 2, 2)
    dout = _rand(30, 2, 2, 4, 3)
    dx, dw, db = triple(call(conv.conv2d_backward, dout, x, weight))
    assert_new_float_array(dx, x.shape, "dx", inputs=(x, dout, weight))
    assert_new_float_array(dw, weight.shape, "dW", inputs=(x, dout, weight))
    assert_new_float_array(db, (2,), "db", inputs=(x, dout, weight))
    dx, dw, db = triple(call(conv.conv2d_backward, [[[[1, 1]]]], [[[[1, 2, 3]]]], [[[[2, -1]]]]))
    assert_new_float_array(dx, (1, 1, 1, 3), "dx from lists of ints")
    assert_close(dx, [[[[2.0, 1.0, -1.0]]]], msg="dx from lists of ints")
    assert_close(dw, [[[[3.0, 5.0]]]], msg="dW from lists of ints")
    assert_close(db, [2.0], msg="db from lists of ints")


def test_conv2d_backward_does_not_modify_its_inputs(conv):
    arrays = {"dout": _rand(31, 2, 2, 3, 3), "x": _rand(32, 2, 3, 5, 5), "weight": _rand(33, 2, 3, 3, 3)}
    saved = {name: a.copy() for name, a in arrays.items()}
    call(conv.conv2d_backward, arrays["dout"], arrays["x"], arrays["weight"], stride=2, padding=1)
    assert_unchanged(arrays, saved, "conv2d_backward")


@pytest.mark.parametrize("dout_shape, why", [
    pytest.param((2, 2, 4, 4), "dout for 2 images, x has 3", id="batch"),
    pytest.param((3, 3, 4, 4), "dout with 3 channels, the weight has C_out = 2", id="channels"),
    pytest.param((3, 2, 6, 6), "dout of the shape of x (the forward output is 4 x 4)", id="shape-of-x"),
    pytest.param((3, 2, 4, 3), "dout 4 x 3 for a 4 x 4 output", id="width"),
    pytest.param((1, 2, 4, 4), "dout for 1 image, x has 3 (it would broadcast: check the shape)",
                 id="broadcastable-batch"),
    pytest.param((3, 2, 2, 8), "dout 2 x 8 for a 4 x 4 output (the same number of values)", id="same-size"),
    pytest.param((2, 4, 4), "a 3-D dout", id="3d"),
])
def test_conv2d_backward_refuses_dout_of_the_wrong_shape(conv, dout_shape, why):
    assert_raises(ValueError, conv.conv2d_backward, np.ones(dout_shape), np.ones((3, 1, 6, 6)), np.ones((2, 1, 3, 3)),
                  why=why)
