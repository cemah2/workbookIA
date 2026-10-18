"""Tests of mylearn.rnn (chapter 22): oracle tests and properties.

    pytest tests/test_ch22_rnn.py              # your code (mon_travail/mylearn/rnn.py)
    pytest tests/test_ch22_rnn.py --impl=ref   # the reference

Oracles: PyTorch in float64 (``torch.nn.RNNCell``, ``RNN``, ``LSTMCell``, ``LSTM``, ``GRUCell``, whose weights
are copied into your functions with ``.T``) and the equations of the docstrings, transcribed in this file with
NumPy (one gate at a time). With PyTorch, the transcribed equations are checked against torch at every test (a
disagreement is a bug of this file, not yours); without PyTorch, they are the oracle, so that no test is skipped.
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_rnn_cell_forward_ or test_rnn_forward_"`` (22.15), ``-k "test_lstm_cell_forward_ or test_lstm_forward_"``
(22.16), ``-k "test_gru_cell_forward_"`` (22.17).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified
and may be lists (of ints); every array returned is a new float64 array, never a view of an input (even when the
new state equals the previous one); the weights have the mylearn layout (D, kH) and (H, kH), so the PyTorch layout
(kH, D) is refused with ``ValueError`` as soon as the shapes differ; a bias of another shape than (kH,), even a
broadcastable one ((1,) or (kH, 1)), is refused; ``x_t`` and the states must be 2-D with the same number of rows,
``x`` 3-D with at least one time step; an unknown ``nonlinearity`` is refused (``"tanh"`` and ``"relu"`` only);
pre-activations of +-1000 give no ``RuntimeWarning`` (a stable sigmoid); ``rnn_forward`` and ``lstm_forward``
return ``h_T`` (and ``c_T``) as arrays of shape (N, H), the ``(h_T, c_T)`` pair as a tuple.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest

try:
    import torch
except ImportError:  # the PyTorch oracles run on Colab; the NumPy equations take over here
    torch = None

ORACLE_BUG = ("bug in tests/test_ch22_rnn.py (not in your code): the NumPy oracle of this test "
              "disagrees with torch")


@pytest.fixture
def rnn(mylearn_module):
    return mylearn_module("rnn")


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


def pair(result, what, names=("h_t", "c_t")):
    """Unpack a 2-tuple result, or fail with a message that says what was expected."""
    label = f"the tuple ({', '.join(names)})"
    if result is None:
        _fail("", f"expected {label} from {what}, got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 2:
        _fail("", f"expected {label} from {what}, got {type(result).__name__}"
                  + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else ""))
    return result


def lstm_result(result, what="lstm_forward"):
    """Unpack (outputs, (h_T, c_T)), or fail with a message that says what was expected."""
    outputs, last = pair(result, what, names=("outputs", "(h_T, c_T)"))
    h_t, c_t = pair(last, what + " (its second element)", names=("h_T", "c_T"))
    return outputs, h_t, c_t


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
    return [None if a is None else np.array(a, copy=True) for a in arrays]


def _t(a):
    return torch.tensor(np.asarray(a, dtype=np.float64))


def _torch_agrees(ours, theirs, atol=1e-10):
    theirs = theirs.detach().numpy()
    assert np.shape(ours) == theirs.shape and np.allclose(ours, theirs, rtol=1e-9, atol=atol), ORACLE_BUG


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))         # the logistic function, without overflow


def weights(rng, d, h, gates, bias=True, scale=0.6):
    """Random weights in the mylearn layout (D, gH), (H, gH) and biases (gH,) or None."""
    w_ih = rng.normal(scale=scale, size=(d, gates * h))
    w_hh = rng.normal(scale=scale, size=(h, gates * h))
    if not bias:
        return w_ih, w_hh, None, None
    return w_ih, w_hh, rng.normal(scale=0.5, size=gates * h), rng.normal(scale=0.5, size=gates * h)


def _bias(b, size):
    return np.zeros(size) if b is None else np.asarray(b, dtype=float)


def _copy_into(module, w_ih, w_hh, b_ih, b_hh, suffix=""):
    """Copy mylearn-layout weights into a torch module (weight_ih = W_ih.T)."""
    with torch.no_grad():
        getattr(module, "weight_ih" + suffix).copy_(_t(w_ih.T))
        getattr(module, "weight_hh" + suffix).copy_(_t(w_hh.T))
        if b_ih is not None:
            getattr(module, "bias_ih" + suffix).copy_(_t(b_ih))
            getattr(module, "bias_hh" + suffix).copy_(_t(b_hh))


# ------------------------------------------------------------------ NumPy oracles (checked against torch)
def np_rnn_cell(x, h, w_ih, w_hh, b_ih=None, b_hh=None, nonlinearity="tanh"):
    """h_t = f(x W_ih + b_ih + h W_hh + b_hh) (checked against torch.nn.RNNCell)."""
    hid = w_hh.shape[0]
    z = x @ w_ih + _bias(b_ih, hid) + h @ w_hh + _bias(b_hh, hid)
    result = np.tanh(z) if nonlinearity == "tanh" else np.maximum(z, 0.0)
    if torch is not None:
        cell = torch.nn.RNNCell(w_ih.shape[0], hid, bias=b_ih is not None, nonlinearity=nonlinearity).double()
        _copy_into(cell, w_ih, w_hh, b_ih, b_hh)
        _torch_agrees(result, cell(_t(x), _t(h)))
    return result


def np_rnn(x, w_ih, w_hh, b_ih=None, b_hh=None, h0=None, nonlinearity="tanh"):
    """The cell applied at every time step (checked against torch.nn.RNN(batch_first=True))."""
    n, t, _ = x.shape
    hid = w_hh.shape[0]
    h = np.zeros((n, hid)) if h0 is None else np.asarray(h0, dtype=float)
    states = []
    for step in range(t):
        z = x[:, step] @ w_ih + _bias(b_ih, hid) + h @ w_hh + _bias(b_hh, hid)
        h = np.tanh(z) if nonlinearity == "tanh" else np.maximum(z, 0.0)
        states.append(h)
    outputs = np.stack(states, axis=1)
    if torch is not None:
        layer = torch.nn.RNN(x.shape[2], hid, batch_first=True, bias=b_ih is not None,
                             nonlinearity=nonlinearity).double()
        _copy_into(layer, w_ih, w_hh, b_ih, b_hh, suffix="_l0")
        out, h_n = layer(_t(x), None if h0 is None else _t(h0)[None])
        _torch_agrees(outputs, out)
        _torch_agrees(outputs[:, -1], h_n[0])
    return outputs, outputs[:, -1]


def np_lstm_cell(x, h, c, w_ih, w_hh, b_ih=None, b_hh=None):
    """Gates i, f, g, o in this order (checked against torch.nn.LSTMCell)."""
    hid = w_hh.shape[0]
    a = x @ w_ih + _bias(b_ih, 4 * hid) + h @ w_hh + _bias(b_hh, 4 * hid)
    i, f = _sigmoid(a[:, :hid]), _sigmoid(a[:, hid:2 * hid])
    g, o = np.tanh(a[:, 2 * hid:3 * hid]), _sigmoid(a[:, 3 * hid:])
    c_t = f * c + i * g
    h_t = o * np.tanh(c_t)
    if torch is not None:
        cell = torch.nn.LSTMCell(w_ih.shape[0], hid, bias=b_ih is not None).double()
        _copy_into(cell, w_ih, w_hh, b_ih, b_hh)
        th, tc = cell(_t(x), (_t(h), _t(c)))
        _torch_agrees(h_t, th)
        _torch_agrees(c_t, tc)
    return h_t, c_t


def np_lstm(x, w_ih, w_hh, b_ih=None, b_hh=None, h0=None, c0=None):
    """The LSTM cell at every time step (checked against torch.nn.LSTM(batch_first=True))."""
    n, t, _ = x.shape
    hid = w_hh.shape[0]
    h = np.zeros((n, hid)) if h0 is None else np.asarray(h0, dtype=float)
    c = np.zeros((n, hid)) if c0 is None else np.asarray(c0, dtype=float)
    states = []
    for step in range(t):
        a = x[:, step] @ w_ih + _bias(b_ih, 4 * hid) + h @ w_hh + _bias(b_hh, 4 * hid)
        i, f = _sigmoid(a[:, :hid]), _sigmoid(a[:, hid:2 * hid])
        g, o = np.tanh(a[:, 2 * hid:3 * hid]), _sigmoid(a[:, 3 * hid:])
        c = f * c + i * g
        h = o * np.tanh(c)
        states.append(h)
    outputs = np.stack(states, axis=1)
    if torch is not None:
        layer = torch.nn.LSTM(x.shape[2], hid, batch_first=True, bias=b_ih is not None).double()
        _copy_into(layer, w_ih, w_hh, b_ih, b_hh, suffix="_l0")
        init = None
        if h0 is not None or c0 is not None:
            init = (_t(np.zeros((n, hid)) if h0 is None else h0)[None], _t(np.zeros((n, hid)) if c0 is None else c0)[None])
        out, (h_n, c_n) = layer(_t(x), init)
        _torch_agrees(outputs, out)
        _torch_agrees(c, c_n[0])
    return outputs, outputs[:, -1], c


def np_gru_cell(x, h, w_ih, w_hh, b_ih=None, b_hh=None):
    """Gates r, z, n in this order, r multiplying h W_hn + b_hn (checked against torch.nn.GRUCell)."""
    hid = w_hh.shape[0]
    gi = x @ w_ih + _bias(b_ih, 3 * hid)
    gh = h @ w_hh + _bias(b_hh, 3 * hid)
    r = _sigmoid(gi[:, :hid] + gh[:, :hid])
    z = _sigmoid(gi[:, hid:2 * hid] + gh[:, hid:2 * hid])
    n = np.tanh(gi[:, 2 * hid:] + r * gh[:, 2 * hid:])
    result = (1.0 - z) * n + z * h
    if torch is not None:
        cell = torch.nn.GRUCell(w_ih.shape[0], hid, bias=b_ih is not None).double()
        _copy_into(cell, w_ih, w_hh, b_ih, b_hh)
        _torch_agrees(result, cell(_t(x), _t(h)))
    return result


SHAPES = [pytest.param(1, 1, 1, id="N1-D1-H1"), pytest.param(3, 4, 5, id="N3-D4-H5"),
          pytest.param(2, 7, 3, id="N2-D7-H3"), pytest.param(5, 2, 2, id="N5-D2-H2")]


# ================================================================== rnn_cell_forward (22.15)
def test_rnn_cell_forward_docstring_example(rnn):
    out = call(rnn.rnn_cell_forward, [[1.0, 0.0]], [[0.0, 0.5]], np.eye(2), np.eye(2))
    assert_close(out, [[np.tanh(1.0), np.tanh(0.5)]], msg="rnn_cell_forward([[1, 0]], [[0, 0.5]], I, I)")


@pytest.mark.parametrize("n,d,h", SHAPES)
@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_rnn_cell_forward_matches_torch_rnncell(rnn, n, d, h, nonlinearity):
    rng = np.random.default_rng(10 * n + d + h)
    x, hp = rng.normal(size=(n, d)), rng.normal(size=(n, h))
    w = weights(rng, d, h, 1)
    want = np_rnn_cell(x, hp, *w, nonlinearity=nonlinearity)
    out = call(rnn.rnn_cell_forward, *copies(x, hp, *w), nonlinearity=nonlinearity)
    assert_close(out, want, msg=f"h_t = {nonlinearity}(x_t @ W_ih + b_ih + h_prev @ W_hh + b_hh), as torch.nn.RNNCell")


def test_rnn_cell_forward_without_biases_matches_rnncell_bias_false(rnn):
    rng = np.random.default_rng(1)
    x, hp = rng.normal(size=(3, 4)), rng.normal(size=(3, 2))
    w_ih, w_hh, _, _ = weights(rng, 4, 2, 1)
    want = np_rnn_cell(x, hp, w_ih, w_hh)
    out = call(rnn.rnn_cell_forward, *copies(x, hp, w_ih, w_hh))
    assert_close(out, want, msg="b_ih = b_hh = None means no bias (torch.nn.RNNCell(bias=False))")


def test_rnn_cell_forward_one_bias_only(rnn):
    rng = np.random.default_rng(2)
    x, hp = rng.normal(size=(2, 3)), rng.normal(size=(2, 4))
    w_ih, w_hh, b_ih, _ = weights(rng, 3, 4, 1)
    want = np.tanh(x @ w_ih + b_ih + hp @ w_hh)
    out = call(rnn.rnn_cell_forward, *copies(x, hp, w_ih, w_hh, b_ih), b_hh=None)
    assert_close(out, want, msg="b_ih given and b_hh None: only b_ih is added")
    want2 = np.tanh(x @ w_ih + hp @ w_hh + b_ih)
    out2 = call(rnn.rnn_cell_forward, *copies(x, hp, w_ih, w_hh), b_ih=None, b_hh=b_ih.copy())
    assert_close(out2, want2, msg="b_ih None and b_hh given: only b_hh is added")


def test_rnn_cell_forward_relu_keeps_positive_part(rnn):
    x, hp = np.array([[1.0, -2.0]]), np.array([[0.5, 0.0]])
    out = call(rnn.rnn_cell_forward, x, hp, np.eye(2), np.eye(2), nonlinearity="relu")
    assert_close(out, [[1.5, 0.0]], msg='nonlinearity="relu": max(z, 0), not tanh')


def test_rnn_cell_forward_accepts_lists_of_ints(rnn):
    out = call(rnn.rnn_cell_forward, [[1, 0]], [[0, 1]], [[1, 0], [0, 1]], [[0, 1], [1, 0]], [0, 0], [1, 1])
    assert_new_float_array(out, (1, 2), "rnn_cell_forward(lists of ints)")
    assert_close(out, np.tanh([[3.0, 1.0]]), msg="rnn_cell_forward on lists of ints")


def test_rnn_cell_forward_returns_a_new_float_array(rnn):
    hp = np.zeros((2, 3))
    x = np.zeros((2, 3))
    out = call(rnn.rnn_cell_forward, x, hp, np.zeros((3, 3)), np.eye(3), nonlinearity="relu")
    assert_new_float_array(out, (2, 3), "the new state", inputs=(x, hp))


def test_rnn_cell_forward_does_not_modify_its_inputs(rnn):
    rng = np.random.default_rng(3)
    arrays = dict(zip(["x_t", "h_prev", "W_ih", "W_hh", "b_ih", "b_hh"],
                      [rng.normal(size=(2, 3)), rng.normal(size=(2, 2)), *weights(rng, 3, 2, 1)]))
    saved = {k: v.copy() for k, v in arrays.items()}
    call(rnn.rnn_cell_forward, *arrays.values())
    assert_unchanged(arrays, saved, "rnn_cell_forward")


def test_rnn_cell_forward_no_overflow_warning(rnn):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = call(rnn.rnn_cell_forward, [[1000.0, -1000.0]], [[0.0, 0.0]], np.eye(2), np.eye(2))
    assert_close(out, [[1.0, -1.0]], msg="tanh of +-1000, without a RuntimeWarning")


@pytest.mark.parametrize("args,why", [
    pytest.param(dict(x_t=np.ones(3), h_prev=np.ones((1, 2))), "x_t is 1-D: it must be (N, D)", id="x_t-1d"),
    pytest.param(dict(x_t=np.ones((2, 3)), h_prev=np.ones((3, 2))), "x_t has 2 rows and h_prev 3", id="rows-differ"),
    pytest.param(dict(x_t=np.ones((1, 3)), h_prev=np.ones(2)), "h_prev is 1-D: it must be (N, H)", id="h_prev-1d"),
    pytest.param(dict(W_ih=np.ones((2, 3))), "W_ih in the PyTorch layout (H, D) = (2, 3) instead of (D, H) = (3, 2)",
                 id="W_ih-torch-layout"),
    pytest.param(dict(W_hh=np.ones((2, 3))), "W_hh of shape (2, 3) instead of (H, H) = (2, 2)", id="W_hh-shape"),
    pytest.param(dict(b_ih=np.ones(1)), "b_ih of shape (1,): it must be (H,) = (2,), even if (1,) would broadcast",
                 id="b_ih-broadcastable"),
    pytest.param(dict(b_hh=np.ones((2, 1))), "b_hh of shape (2, 1) instead of (2,)", id="b_hh-2d"),
    pytest.param(dict(nonlinearity="sigmoid"), 'nonlinearity="sigmoid" (only "tanh" and "relu")', id="unknown-nonlinearity"),
])
def test_rnn_cell_forward_refuses_invalid_inputs(rnn, args, why):
    base = dict(x_t=np.ones((1, 3)), h_prev=np.ones((1, 2)), W_ih=np.ones((3, 2)), W_hh=np.ones((2, 2)),
                b_ih=np.ones(2), b_hh=np.ones(2), nonlinearity="tanh")
    base.update(args)
    assert_raises(ValueError, rnn.rnn_cell_forward, why=why, **base)


# ================================================================== rnn_forward (22.15)
def test_rnn_forward_docstring_example(rnn):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 3))
    result = call(rnn.rnn_forward, x, rng.normal(size=(3, 4)), rng.normal(size=(4, 4)), expected="(outputs, h_T)")
    outputs, h_t = pair(result, "rnn_forward", names=("outputs", "h_T"))
    assert_new_float_array(outputs, (2, 5, 4), "outputs", inputs=(x,))
    assert_new_float_array(h_t, (2, 4), "h_T", inputs=(x,))
    assert_close(h_t, outputs[:, -1], msg="h_T must equal outputs[:, -1]")


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
@pytest.mark.parametrize("with_h0", [False, True], ids=["h0-None", "h0-given"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_rnn_forward_matches_torch_rnn(rnn, nonlinearity, with_h0, bias):
    rng = np.random.default_rng(7 + with_h0 + 2 * bias)
    n, t, d, h = 3, 6, 4, 5
    x = rng.normal(size=(n, t, d))
    w = weights(rng, d, h, 1, bias=bias, scale=0.4)
    h0 = rng.normal(size=(n, h)) if with_h0 else None
    want, want_h = np_rnn(x, *w, h0=h0, nonlinearity=nonlinearity)
    result = call(rnn.rnn_forward, *copies(x, *w), h0=None if h0 is None else h0.copy(), nonlinearity=nonlinearity,
                  expected="(outputs, h_T)")
    outputs, h_t = pair(result, "rnn_forward", names=("outputs", "h_T"))
    assert_close(outputs, want, msg="outputs of rnn_forward against torch.nn.RNN(batch_first=True)")
    assert_close(h_t, want_h, msg="h_T of rnn_forward against h_n[0] of torch.nn.RNN")


def test_rnn_forward_uses_the_same_weights_at_every_step(rnn):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 4, 3))
    w = weights(rng, 3, 2, 1)
    outputs, _ = pair(call(rnn.rnn_forward, *copies(x, *w), expected="(outputs, h_T)"), "rnn_forward",
                      names=("outputs", "h_T"))
    previous = np.zeros((2, 2))
    for step in range(4):
        want = np_rnn_cell(x[:, step], previous, *w)
        assert_close(outputs[:, step], want, msg=f"outputs[:, {step}] = cell(x[:, {step}], outputs[:, {step - 1}]) "
                                                 "with the same weights (h0 = zeros before the first step)")
        previous = want


def test_rnn_forward_one_time_step(rnn):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(4, 1, 2))
    w = weights(rng, 2, 3, 1)
    outputs, h_t = pair(call(rnn.rnn_forward, *copies(x, *w), expected="(outputs, h_T)"), "rnn_forward",
                        names=("outputs", "h_T"))
    want = np_rnn_cell(x[:, 0], np.zeros((4, 3)), *w)
    assert_close(outputs, want[:, None], msg="T = 1: outputs of shape (N, 1, H)")
    assert_close(h_t, want, msg="T = 1: h_T is the only state")


def test_rnn_forward_accepts_lists(rnn):
    x = [[[1], [0], [2]]]
    result = call(rnn.rnn_forward, x, [[0.5]], [[0.5]], [0], [0], expected="(outputs, h_T)")
    outputs, h_t = pair(result, "rnn_forward", names=("outputs", "h_T"))
    want, _ = np_rnn(np.array(x, dtype=float), np.array([[0.5]]), np.array([[0.5]]), np.zeros(1), np.zeros(1))
    assert_new_float_array(outputs, (1, 3, 1), "outputs (lists as inputs)")
    assert_close(outputs, want, msg="rnn_forward on lists")


def test_rnn_forward_returns_new_arrays(rnn):
    x = np.zeros((2, 1, 2))
    h0 = np.ones((2, 2))
    outputs, h_t = pair(call(rnn.rnn_forward, x, np.zeros((2, 2)), np.eye(2), h0=h0, nonlinearity="relu",
                             expected="(outputs, h_T)"), "rnn_forward", names=("outputs", "h_T"))
    assert_new_float_array(outputs, (2, 1, 2), "outputs", inputs=(x, h0))
    assert_new_float_array(h_t, (2, 2), "h_T", inputs=(x, h0))


def test_rnn_forward_does_not_modify_its_inputs(rnn):
    rng = np.random.default_rng(13)
    arrays = dict(zip(["x", "W_ih", "W_hh", "b_ih", "b_hh"], [rng.normal(size=(2, 3, 2)), *weights(rng, 2, 3, 1)]))
    arrays["h0"] = rng.normal(size=(2, 3))
    saved = {k: v.copy() for k, v in arrays.items()}
    call(rnn.rnn_forward, **arrays)
    assert_unchanged(arrays, saved, "rnn_forward")


@pytest.mark.parametrize("args,why", [
    pytest.param(dict(x=np.ones((2, 3))), "x is 2-D: it must be (N, T, D)", id="x-2d"),
    pytest.param(dict(x=np.ones((2, 0, 3))), "x has no time step (T = 0)", id="T-zero"),
    pytest.param(dict(h0=np.ones((3, 2))), "h0 has 3 rows for N = 2 sequences", id="h0-rows"),
    pytest.param(dict(h0=np.ones((2, 5))), "h0 has 5 columns for H = 2", id="h0-columns"),
    pytest.param(dict(W_ih=np.ones((2, 3))), "W_ih in the PyTorch layout (H, D) instead of (D, H)", id="W_ih-torch-layout"),
    pytest.param(dict(nonlinearity="gelu"), 'nonlinearity="gelu"', id="unknown-nonlinearity"),
])
def test_rnn_forward_refuses_invalid_inputs(rnn, args, why):
    base = dict(x=np.ones((2, 4, 3)), W_ih=np.ones((3, 2)), W_hh=np.ones((2, 2)), b_ih=None, b_hh=None, h0=None,
                nonlinearity="tanh")
    base.update(args)
    assert_raises(ValueError, rnn.rnn_forward, why=why, **base)


# ================================================================== lstm_cell_forward (22.16)
def test_lstm_cell_forward_docstring_example(rnn):
    result = call(rnn.lstm_cell_forward, np.zeros((1, 3)), np.zeros((1, 2)), np.ones((1, 2)),
                  np.zeros((3, 8)), np.zeros((2, 8)), expected="(h_t, c_t)")
    h_t, c_t = pair(result, "lstm_cell_forward")
    assert_close(c_t, [[0.5, 0.5]], msg="c_t with all weights at 0 (f = 0.5, g = 0)")
    assert_close(h_t, [[0.5 * np.tanh(0.5)] * 2], msg="h_t with all weights at 0 (o = 0.5)")


@pytest.mark.parametrize("n,d,h", SHAPES)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_lstm_cell_forward_matches_torch_lstmcell(rnn, n, d, h, bias):
    rng = np.random.default_rng(100 + 10 * n + d + h + bias)
    x, hp, cp = rng.normal(size=(n, d)), rng.normal(size=(n, h)), rng.normal(size=(n, h))
    w = weights(rng, d, h, 4, bias=bias)
    want_h, want_c = np_lstm_cell(x, hp, cp, *w)
    h_t, c_t = pair(call(rnn.lstm_cell_forward, *copies(x, hp, cp, *w), expected="(h_t, c_t)"), "lstm_cell_forward")
    assert_close(c_t, want_c, msg="c_t = f * c_prev + i * g against torch.nn.LSTMCell (gates i, f, g, o)")
    assert_close(h_t, want_h, msg="h_t = o * tanh(c_t) against torch.nn.LSTMCell")


def test_lstm_cell_forward_returns_h_then_c(rnn):
    """With a closed output gate, h_t is 0 and c_t is not: the order of the tuple shows."""
    hid = 2
    b = np.zeros(4 * hid)
    b[3 * hid:] = -50.0                                    # o = sigmoid(-50) ~ 0
    h_t, c_t = pair(call(rnn.lstm_cell_forward, np.zeros((1, 1)), np.zeros((1, hid)), np.ones((1, hid)),
                         np.zeros((1, 4 * hid)), np.zeros((hid, 4 * hid)), b, np.zeros(4 * hid),
                         expected="(h_t, c_t)"), "lstm_cell_forward")
    assert_close(h_t, np.zeros((1, hid)), atol=1e-12, msg="return (h_t, c_t) in this order: here the output gate is "
                                                         "closed (o ~ 0), so h_t ~ 0")
    assert_close(c_t, np.full((1, hid), 0.5), msg="c_t = f * c_prev + i * g = 0.5 * 1 + 0.5 * 0")


@pytest.mark.parametrize("block,want_c", [
    pytest.param(0, 0.5, id="block0-is-i"), pytest.param(1, 1.0, id="block1-is-f"),
    pytest.param(2, 0.5 + 0.5 * np.tanh(50.0), id="block2-is-g"), pytest.param(3, 0.5, id="block3-is-o"),
])
def test_lstm_cell_forward_gate_order_i_f_g_o(rnn, block, want_c):
    """A large bias on one block of H columns at a time shows which gate the block is."""
    hid = 1
    b = np.zeros(4)
    b[block] = 50.0
    h_t, c_t = pair(call(rnn.lstm_cell_forward, np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)),
                         np.zeros((1, 4)), np.zeros((1, 4)), b, np.zeros(4), expected="(h_t, c_t)"), "lstm_cell_forward")
    want_h, want_cc = np_lstm_cell(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 4)),
                                   np.zeros((1, 4)), b, np.zeros(4))
    assert abs(want_cc[0, 0] - want_c) < 1e-9, ORACLE_BUG
    names = "ifgo"
    assert_close(c_t, want_cc, msg=f"bias 50 on columns block {block} (gate {names[block]} in the order i, f, g, o)")
    assert_close(h_t, want_h, msg=f"bias 50 on columns block {block} (gate {names[block]} in the order i, f, g, o)")


def test_lstm_cell_forward_no_overflow_warning(rnn):
    b = np.array([1000.0, -1000.0, 1000.0, -1000.0, -1000.0, 1000.0, -1000.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        h_t, c_t = pair(call(rnn.lstm_cell_forward, np.zeros((1, 1)), np.zeros((1, 2)), np.ones((1, 2)),
                             np.zeros((1, 8)), np.zeros((2, 8)), b, np.zeros(8), expected="(h_t, c_t)"),
                        "lstm_cell_forward")
    want_h, want_c = np_lstm_cell(np.zeros((1, 1)), np.zeros((1, 2)), np.ones((1, 2)), np.zeros((1, 8)),
                                  np.zeros((2, 8)), b, np.zeros(8))
    assert_close(c_t, want_c, msg="pre-activations of +-1000: sigmoid without overflow (no RuntimeWarning)")
    assert_close(h_t, want_h, msg="pre-activations of +-1000")


def test_lstm_cell_forward_accepts_lists_of_ints(rnn):
    w_ih = [[1, 0, 0, 1]]
    w_hh = [[0, 1, 1, 0]]
    h_t, c_t = pair(call(rnn.lstm_cell_forward, [[1]], [[1]], [[2]], w_ih, w_hh, expected="(h_t, c_t)"),
                    "lstm_cell_forward")
    want_h, want_c = np_lstm_cell(np.ones((1, 1)), np.ones((1, 1)), np.full((1, 1), 2.0), np.array(w_ih, float),
                                  np.array(w_hh, float))
    assert_new_float_array(h_t, (1, 1), "h_t (lists as inputs)")
    assert_new_float_array(c_t, (1, 1), "c_t (lists as inputs)")
    assert_close(c_t, want_c, msg="lstm_cell_forward on lists of ints")
    assert_close(h_t, want_h, msg="lstm_cell_forward on lists of ints")


def test_lstm_cell_forward_returns_new_arrays(rnn):
    """With f = 1 and i = 0, c_t equals c_prev: it must still be a new array."""
    b = np.array([-60.0, 60.0, 0.0, 0.0])
    x, hp, cp = np.zeros((1, 1)), np.zeros((1, 1)), np.full((1, 1), 0.3)
    h_t, c_t = pair(call(rnn.lstm_cell_forward, x, hp, cp, np.zeros((1, 4)), np.zeros((1, 4)), b, np.zeros(4),
                         expected="(h_t, c_t)"), "lstm_cell_forward")
    assert_new_float_array(c_t, (1, 1), "c_t", inputs=(x, hp, cp))
    assert_new_float_array(h_t, (1, 1), "h_t", inputs=(x, hp, cp))


def test_lstm_cell_forward_does_not_modify_its_inputs(rnn):
    rng = np.random.default_rng(14)
    arrays = dict(zip(["x_t", "h_prev", "c_prev", "W_ih", "W_hh", "b_ih", "b_hh"],
                      [rng.normal(size=(2, 3)), rng.normal(size=(2, 2)), rng.normal(size=(2, 2)),
                       *weights(rng, 3, 2, 4)]))
    saved = {k: v.copy() for k, v in arrays.items()}
    call(rnn.lstm_cell_forward, *arrays.values())
    assert_unchanged(arrays, saved, "lstm_cell_forward")


@pytest.mark.parametrize("args,why", [
    pytest.param(dict(W_ih=np.ones((3, 6))), "W_ih with 3H = 6 columns: an LSTM needs 4H = 8", id="W_ih-3H"),
    pytest.param(dict(W_ih=np.ones((8, 3))), "W_ih in the PyTorch layout (4H, D) = (8, 3)", id="W_ih-torch-layout"),
    pytest.param(dict(W_hh=np.ones((2, 6))), "W_hh with 6 columns instead of 4H = 8", id="W_hh-shape"),
    pytest.param(dict(c_prev=np.ones((1, 3))), "c_prev of shape (1, 3) for h_prev of shape (1, 2)", id="c_prev-shape"),
    pytest.param(dict(x_t=np.ones((2, 3))), "x_t has 2 rows and h_prev 1", id="rows-differ"),
    pytest.param(dict(b_ih=np.ones(2)), "b_ih of shape (2,) instead of (4H,) = (8,)", id="b_ih-shape"),
    pytest.param(dict(b_hh=np.ones((8, 1))), "b_hh of shape (8, 1) instead of (8,)", id="b_hh-2d"),
])
def test_lstm_cell_forward_refuses_invalid_inputs(rnn, args, why):
    base = dict(x_t=np.ones((1, 3)), h_prev=np.ones((1, 2)), c_prev=np.ones((1, 2)), W_ih=np.ones((3, 8)),
                W_hh=np.ones((2, 8)), b_ih=np.ones(8), b_hh=np.ones(8))
    base.update(args)
    assert_raises(ValueError, rnn.lstm_cell_forward, why=why, **base)


# ================================================================== lstm_forward (22.16)
def test_lstm_forward_docstring_example(rnn):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 3))
    result = call(rnn.lstm_forward, x, rng.normal(size=(3, 16)), rng.normal(size=(4, 16)),
                  expected="(outputs, (h_T, c_T))")
    outputs, h_t, c_t = lstm_result(result)
    assert_new_float_array(outputs, (2, 5, 4), "outputs", inputs=(x,))
    assert_new_float_array(h_t, (2, 4), "h_T", inputs=(x,))
    assert_new_float_array(c_t, (2, 4), "c_T", inputs=(x,))
    assert_close(h_t, outputs[:, -1], msg="h_T must equal outputs[:, -1]")


@pytest.mark.parametrize("init", ["none", "h0", "h0-c0", "c0"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_lstm_forward_matches_torch_lstm(rnn, init, bias):
    rng = np.random.default_rng(200 + len(init) + bias)
    n, t, d, h = 3, 7, 4, 3
    x = rng.normal(size=(n, t, d))
    w = weights(rng, d, h, 4, bias=bias, scale=0.5)
    h0 = rng.normal(size=(n, h)) if "h0" in init else None
    c0 = rng.normal(size=(n, h)) if "c0" in init else None
    want, want_h, want_c = np_lstm(x, *w, h0=h0, c0=c0)
    result = call(rnn.lstm_forward, *copies(x, *w), h0=None if h0 is None else h0.copy(),
                  c0=None if c0 is None else c0.copy(), expected="(outputs, (h_T, c_T))")
    outputs, h_t, c_t = lstm_result(result)
    assert_close(outputs, want, msg="outputs of lstm_forward against torch.nn.LSTM(batch_first=True)")
    assert_close(h_t, want_h, msg="h_T of lstm_forward against h_n[0] of torch.nn.LSTM")
    assert_close(c_t, want_c, msg="c_T of lstm_forward against c_n[0] of torch.nn.LSTM")


def test_lstm_forward_returns_a_tuple_of_states(rnn):
    rng = np.random.default_rng(15)
    result = call(rnn.lstm_forward, rng.normal(size=(1, 2, 2)), rng.normal(size=(2, 4)), rng.normal(size=(1, 4)),
                  expected="(outputs, (h_T, c_T))")
    outputs, last = pair(result, "lstm_forward", names=("outputs", "(h_T, c_T)"))
    if not isinstance(last, tuple):
        _fail("", f"expected the second element of the result to be the tuple (h_T, c_T), as nn.LSTM, "
                  f"got {type(last).__name__}")


def test_lstm_forward_carries_the_cell_memory(rnn):
    """f = 1 and i = 0 at every step: the memory c0 must cross the whole sequence unchanged."""
    b = np.array([-60.0, 60.0, 0.0, 60.0])
    c0 = np.array([[0.7]])
    outputs, h_t, c_t = lstm_result(call(rnn.lstm_forward, np.zeros((1, 5, 1)), np.zeros((1, 4)), np.zeros((1, 4)),
                                         b, np.zeros(4), c0=c0.copy(), expected="(outputs, (h_T, c_T))"))
    assert_close(c_t, c0, msg="with f = 1 and i = 0, c_T = c0 (pass c_t from one step to the next)")
    assert_close(outputs, np.full((1, 5, 1), np.tanh(0.7)), atol=1e-9, msg="with o = 1, every h_t = tanh(c0)")


def test_lstm_forward_one_time_step(rnn):
    rng = np.random.default_rng(16)
    x = rng.normal(size=(3, 1, 2))
    w = weights(rng, 2, 2, 4)
    outputs, h_t, c_t = lstm_result(call(rnn.lstm_forward, *copies(x, *w), expected="(outputs, (h_T, c_T))"))
    want_h, want_c = np_lstm_cell(x[:, 0], np.zeros((3, 2)), np.zeros((3, 2)), *w)
    assert_close(outputs, want_h[:, None], msg="T = 1: outputs of shape (N, 1, H)")
    assert_close(c_t, want_c, msg="T = 1: c_T after one step")


def test_lstm_forward_does_not_modify_its_inputs(rnn):
    rng = np.random.default_rng(17)
    arrays = dict(zip(["x", "W_ih", "W_hh", "b_ih", "b_hh"], [rng.normal(size=(2, 3, 2)), *weights(rng, 2, 3, 4)]))
    arrays["h0"] = rng.normal(size=(2, 3))
    arrays["c0"] = rng.normal(size=(2, 3))
    saved = {k: v.copy() for k, v in arrays.items()}
    call(rnn.lstm_forward, **arrays)
    assert_unchanged(arrays, saved, "lstm_forward")


def test_lstm_forward_returns_new_arrays(rnn):
    b = np.array([-60.0, 60.0, 0.0, 0.0])
    x, h0, c0 = np.zeros((1, 1, 1)), np.zeros((1, 1)), np.full((1, 1), 0.4)
    outputs, h_t, c_t = lstm_result(call(rnn.lstm_forward, x, np.zeros((1, 4)), np.zeros((1, 4)), b, np.zeros(4),
                                         h0=h0, c0=c0, expected="(outputs, (h_T, c_T))"))
    for arr, name, shape in [(outputs, "outputs", (1, 1, 1)), (h_t, "h_T", (1, 1)), (c_t, "c_T", (1, 1))]:
        assert_new_float_array(arr, shape, name, inputs=(x, h0, c0))


@pytest.mark.parametrize("args,why", [
    pytest.param(dict(x=np.ones((2, 3))), "x is 2-D: it must be (N, T, D)", id="x-2d"),
    pytest.param(dict(x=np.ones((2, 0, 3))), "x has no time step (T = 0)", id="T-zero"),
    pytest.param(dict(h0=np.ones((3, 2))), "h0 has 3 rows for N = 2 sequences", id="h0-rows"),
    pytest.param(dict(c0=np.ones((2, 3))), "c0 of shape (2, 3) for H = 2", id="c0-shape"),
    pytest.param(dict(W_ih=np.ones((3, 6))), "W_ih with 3H columns: an LSTM needs 4H", id="W_ih-3H"),
])
def test_lstm_forward_refuses_invalid_inputs(rnn, args, why):
    base = dict(x=np.ones((2, 4, 3)), W_ih=np.ones((3, 8)), W_hh=np.ones((2, 8)), b_ih=None, b_hh=None, h0=None,
                c0=None)
    base.update(args)
    assert_raises(ValueError, rnn.lstm_forward, why=why, **base)


# ================================================================== gru_cell_forward (22.17)
def test_gru_cell_forward_docstring_example(rnn):
    out = call(rnn.gru_cell_forward, np.zeros((1, 3)), [[1.0, -1.0]], np.zeros((3, 6)), np.zeros((2, 6)))
    assert_close(out, [[0.5, -0.5]], msg="all weights at 0: r = z = 0.5 and n = 0, so h_t = 0.5 * h_prev")


@pytest.mark.parametrize("n,d,h", SHAPES)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_gru_cell_forward_matches_torch_grucell(rnn, n, d, h, bias):
    rng = np.random.default_rng(300 + 10 * n + d + h + bias)
    x, hp = rng.normal(size=(n, d)), rng.normal(size=(n, h))
    w = weights(rng, d, h, 3, bias=bias)
    want = np_gru_cell(x, hp, *w)
    out = call(rnn.gru_cell_forward, *copies(x, hp, *w))
    assert_close(out, want, msg="h_t of the GRU against torch.nn.GRUCell (gates r, z, n)")


def test_gru_cell_forward_reset_gate_multiplies_the_hidden_bias(rnn):
    """b_hn inside r * (h_prev @ W_hn + b_hn): adding b_hn outside the product gives another value."""
    hid = 1
    b_ih = np.array([-2.0, 0.0, 0.0])                     # r = sigmoid(-2) ~ 0.12
    b_hh = np.array([0.0, 0.0, 3.0])                      # b_hn = 3
    hp = np.array([[0.5]])
    want = np_gru_cell(np.zeros((1, 1)), hp, np.zeros((1, 3)), np.zeros((1, 3)), b_ih, b_hh)
    out = call(rnn.gru_cell_forward, np.zeros((1, 1)), hp.copy(), np.zeros((1, 3)), np.zeros((1, 3)), b_ih.copy(),
               b_hh.copy())
    assert_close(out, want, msg="n = tanh(x W_in + b_in + r * (h_prev W_hn + b_hn)): the reset gate also multiplies "
                                "b_hn (keep the two bias vectors separate)")


def test_gru_cell_forward_update_gate_keeps_the_previous_state(rnn):
    """z ~ 1 (large bias on the z block): h_t = (1 - z) * n + z * h_prev ~ h_prev, not n."""
    b_ih = np.array([0.0, 60.0, 5.0])
    hp = np.array([[0.25, -0.75]]).reshape(1, 2)
    b = np.repeat(b_ih, 2)
    out = call(rnn.gru_cell_forward, np.zeros((1, 1)), hp.copy(), np.zeros((1, 6)), np.zeros((2, 6)), b, None)
    assert_close(out, hp, atol=1e-9, msg="z ~ 1 must keep h_prev: h_t = (1 - z) * n + z * h_prev (PyTorch convention, "
                                         "not z * n + (1 - z) * h_prev)")


def test_gru_cell_forward_no_overflow_warning(rnn):
    b = np.array([1000.0, -1000.0, 1000.0, -1000.0, 1000.0, -1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = call(rnn.gru_cell_forward, np.zeros((1, 1)), np.ones((1, 2)), np.zeros((1, 6)), np.zeros((2, 6)), b,
                   np.zeros(6))
    want = np_gru_cell(np.zeros((1, 1)), np.ones((1, 2)), np.zeros((1, 6)), np.zeros((2, 6)), b, np.zeros(6))
    assert_close(out, want, msg="pre-activations of +-1000: sigmoid without overflow (no RuntimeWarning)")


def test_gru_cell_forward_accepts_lists_of_ints(rnn):
    w_ih = [[1, 0, 1]]
    w_hh = [[0, 1, 1]]
    out = call(rnn.gru_cell_forward, [[2]], [[1]], w_ih, w_hh, [0, 0, 0], [1, 0, 1])
    want = np_gru_cell(np.full((1, 1), 2.0), np.ones((1, 1)), np.array(w_ih, float), np.array(w_hh, float),
                       np.zeros(3), np.array([1.0, 0.0, 1.0]))
    assert_new_float_array(out, (1, 1), "h_t (lists as inputs)")
    assert_close(out, want, msg="gru_cell_forward on lists of ints")


def test_gru_cell_forward_returns_a_new_array(rnn):
    x, hp = np.zeros((1, 1)), np.full((1, 1), 0.6)
    out = call(rnn.gru_cell_forward, x, hp, np.zeros((1, 3)), np.zeros((1, 3)), np.array([0.0, 60.0, 0.0]), None)
    assert_new_float_array(out, (1, 1), "h_t", inputs=(x, hp))


def test_gru_cell_forward_does_not_modify_its_inputs(rnn):
    rng = np.random.default_rng(18)
    arrays = dict(zip(["x_t", "h_prev", "W_ih", "W_hh", "b_ih", "b_hh"],
                      [rng.normal(size=(2, 3)), rng.normal(size=(2, 2)), *weights(rng, 3, 2, 3)]))
    saved = {k: v.copy() for k, v in arrays.items()}
    call(rnn.gru_cell_forward, *arrays.values())
    assert_unchanged(arrays, saved, "gru_cell_forward")


@pytest.mark.parametrize("args,why", [
    pytest.param(dict(W_ih=np.ones((3, 8))), "W_ih with 4H = 8 columns: a GRU needs 3H = 6", id="W_ih-4H"),
    pytest.param(dict(W_ih=np.ones((6, 3))), "W_ih in the PyTorch layout (3H, D) = (6, 3)", id="W_ih-torch-layout"),
    pytest.param(dict(W_hh=np.ones((3, 6))), "W_hh of shape (3, 6) instead of (H, 3H) = (2, 6)", id="W_hh-shape"),
    pytest.param(dict(h_prev=np.ones((2, 2))), "x_t has 1 row and h_prev 2", id="rows-differ"),
    pytest.param(dict(b_hh=np.ones(3)), "b_hh of shape (3,) instead of (3H,) = (6,)", id="b_hh-shape"),
    pytest.param(dict(b_ih=np.ones((1,))), "b_ih of shape (1,), even if it would broadcast", id="b_ih-broadcastable"),
    pytest.param(dict(x_t=np.ones(3)), "x_t is 1-D: it must be (N, D)", id="x_t-1d"),
])
def test_gru_cell_forward_refuses_invalid_inputs(rnn, args, why):
    base = dict(x_t=np.ones((1, 3)), h_prev=np.ones((1, 2)), W_ih=np.ones((3, 6)), W_hh=np.ones((2, 6)),
                b_ih=np.ones(6), b_hh=np.ones(6))
    base.update(args)
    assert_raises(ValueError, rnn.gru_cell_forward, why=why, **base)
