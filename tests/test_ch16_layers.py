"""Tests of mylearn.nn.layers (chapter 16): oracle tests and properties.

    pytest tests/test_ch16_layers.py              # your code (mon_travail/mylearn/nn/layers.py)
    pytest tests/test_ch16_layers.py --impl=ref   # the reference

Oracles: PyTorch (``nn.Linear`` and ``nn.Sequential`` for the parameter counts and the forward pass,
``torch.nn.functional.linear``, the ``torch.nn.init`` draws for the spread of each initialisation),
scikit-learn (``MLPClassifier.predict`` from its ``coefs_`` and ``intercepts_``), NumPy, and the
standard library (``graphlib.TopologicalSorter``, plus a small greedy order written in this file for
the alphabetical rule of ``topological_order``). Without PyTorch, its oracle tests are skipped and
NumPy formulas take over.
Every test name starts with the name of the function it tests, so that each exercise runs its own
group: ``-k test_count_parameters_`` (16.12), ``test_dense_forward_`` (16.13),
``test_init_weights_`` (16.14), ``test_topological_order_`` (16.16),
``"test_init_mlp_ or test_mlp_forward_"`` (16.17). The tests of one function never call another
function of your module (the tests of ``mlp_forward`` build their parameters here).
The first line of every failure message says what was expected (it is the line that ``pytest -rf``
shows). Random tests use fixed seeds and margins that a correct implementation fails with a
probability far below one in a million.
"""

import graphlib
import itertools

import numpy as np
import pytest

try:
    import torch
except ImportError:  # the oracle tests run on Colab
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")


@pytest.fixture
def layers(mylearn_module):
    return mylearn_module("nn.layers")


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


def as_array(result, what):
    """The result as a float array, or a failure that says what was expected."""
    if result is None:
        _fail("", f"expected {what}, got None (did you forget the return?)")
    try:
        return np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail("", f"expected {what} as a NumPy array, got an object of type {type(result).__name__}")


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg="", data=""):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
    got = as_array(result, "an array")
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}", data)
    if not np.allclose(got, want, rtol=rtol, atol=atol, equal_nan=True):
        bad = np.flatnonzero(~np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True))
        where = f" (first difference at flat index {int(bad[0])}, largest gap {np.max(np.abs(got - want)):.3g})"
        have, need = _short(result), _short(expected)
        if have == need:
            have, need = _short(result, 17), _short(expected, 17)
        _fail(msg, f"expected {need}, got {have}{where}", data)


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
    if name == "<lambda>":
        name = "the call"
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


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    shifted = np.exp(z - z.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


# ===================================================================== count_parameters (16.12)
@pytest.mark.parametrize("sizes, expected", [([2, 4, 4, 1], 37), ([784, 256, 128, 10], 235146), ([3, 1], 4),
                                             ([1, 1, 1, 1, 1], 8)], ids=["2-4-4-1", "mnist", "one_layer", "chain"])
def test_count_parameters_docstring_and_small_cases(layers, sizes, expected):
    got = call(layers.count_parameters, sizes, expected=f"{expected} parameters")
    if got != expected:
        _fail("", f"expected {expected} parameters for {sizes} (weights n_(l-1)*n_l plus one bias per neuron), "
                  f"got {got!r}")


def test_count_parameters_returns_a_python_int(layers):
    got = call(layers.count_parameters, [5, 3, 2])
    if type(got) is not int:
        _fail("", f"expected a Python int (convert with int(...)): the docstring prints 37, not np.int64(37); "
                  f"got {got!r} of type {type(got).__name__}")


def test_count_parameters_without_bias(layers):
    got = call(layers.count_parameters, [2, 4, 4, 1], bias=False)
    if got != 28:
        _fail("", f"expected 28 weights for [2, 4, 4, 1] with bias=False (2*4 + 4*4 + 4*1, no bias), got {got!r}")


@needs_torch
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_count_parameters_matches_torch(layers, bias):
    rng = np.random.default_rng(1612)
    for _ in range(20):
        sizes = [int(s) for s in rng.integers(1, 60, size=rng.integers(2, 7))]
        model = torch.nn.Sequential(*[torch.nn.Linear(a, b, bias=bias) for a, b in zip(sizes[:-1], sizes[1:])])
        expected = sum(p.numel() for p in model.parameters())
        got = call(layers.count_parameters, sizes, bias=bias)
        if got != expected:
            _fail("", f"expected {expected} parameters for {sizes} (bias={bias}), as torch counts for the same "
                      f"stack of nn.Linear layers, got {got!r}")


def test_count_parameters_accepts_a_tuple_and_numpy_integers(layers):
    got = call(layers.count_parameters, (np.int64(10), np.int64(5), np.int64(2)),
               expected="a count for a tuple of NumPy integers")
    if got != 67:
        _fail("", f"expected 67 parameters for (10, 5, 2) given as a tuple of np.int64 (10*5 + 5 + 5*2 + 2), "
                  f"got {got!r}")


@pytest.mark.parametrize("sizes, why", [
    pytest.param([], "no size at all", id="empty"),
    pytest.param([784], "a single size: inputs but no layer", id="one_size"),
    pytest.param([3, 0, 2], "a layer of width 0", id="zero_width"),
    pytest.param([0, 4], "0 input features", id="zero_inputs"),
    pytest.param([3, -2], "a negative width", id="negative"),
    pytest.param([3, True], "a bool is not a size (True would count as 1)", id="bool"),
    pytest.param([3, 2.5], "a size that is not an integer", id="non_integer"),
])
def test_count_parameters_rejects_invalid_sizes(layers, sizes, why):
    assert_raises(ValueError, layers.count_parameters, sizes, why=why)


# ===================================================================== dense_forward (16.13)
def test_dense_forward_docstring_examples(layers):
    W = np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 2.0]])
    assert_close(call(layers.dense_forward, [[1.0, 2.0]], W, np.array([0.0, 0.0, 1.0])), [[2.0, 2.0, 4.0]],
                 msg="z = x @ W + b for x = [[1, 2]] (the docstring example)")
    assert_close(call(layers.dense_forward, [1.0, 2.0], W), [2.0, 2.0, 3.0],
                 msg="a 1-D x gives a 1-D z of shape (n_out,), without bias (the docstring example)")


@needs_torch
@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
def test_dense_forward_matches_torch_linear(layers, with_bias):
    rng = np.random.default_rng(1613)
    x, W = rng.normal(size=(32, 7)), rng.normal(size=(7, 5))
    b = rng.normal(size=5) if with_bias else None
    expected = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(W.T),
                                          None if b is None else torch.from_numpy(b)).numpy()
    assert_close(call(layers.dense_forward, x.copy(), W.copy(), None if b is None else b.copy()), expected,
                 atol=1e-12, msg="z of 32 samples through a 7 -> 5 layer, as torch.nn.functional.linear(x, W.T, b) "
                                 "(W has shape (n_in, n_out): torch stores its transpose)")


def test_dense_forward_matches_numpy_on_a_batch(layers):
    rng = np.random.default_rng(16131)
    x, W, b = rng.normal(size=(100, 4)), rng.normal(size=(4, 3)), rng.normal(size=3)
    expected = np.einsum("ni,ij->nj", x, W) + b
    assert_close(call(layers.dense_forward, x.copy(), W.copy(), b.copy()), expected,
                 msg="z for 100 samples: every row gets the same bias vector (broadcast over the rows)")


def test_dense_forward_rows_are_independent(layers):
    rng = np.random.default_rng(16132)
    x, W, b = rng.normal(size=(6, 4)), rng.normal(size=(4, 2)), rng.normal(size=2)
    together = as_array(call(layers.dense_forward, x.copy(), W.copy(), b.copy()), "z")
    alone = np.vstack([as_array(call(layers.dense_forward, x[i:i + 1].copy(), W.copy(), b.copy()), "z")
                       for i in range(6)])
    assert_close(together, alone, msg="each row of z depends only on its own sample (the same z computed one "
                                      "sample at a time)")


def test_dense_forward_returns_float64(layers):
    got = call(layers.dense_forward, [[1, 2]], np.array([[1, 0], [0, 1]]), np.array([3, 4]))
    if not isinstance(got, np.ndarray) or got.dtype != np.float64:
        _fail("", f"expected a float64 NumPy array for integer inputs (convert x with np.asarray(x, dtype=float)), "
                  f"got {type(got).__name__} of dtype {getattr(got, 'dtype', '?')}")
    assert_close(got, [[4.0, 6.0]], msg="z for integer inputs")


def test_dense_forward_does_not_modify_its_inputs(layers):
    rng = np.random.default_rng(16133)
    x, W, b = rng.normal(size=(5, 3)), rng.normal(size=(3, 2)), rng.normal(size=2)
    x0, W0, b0 = x.copy(), W.copy(), b.copy()
    call(layers.dense_forward, x, W, b)
    for name, before, after in (("x", x0, x), ("W", W0, W), ("b", b0, b)):
        if not np.array_equal(before, after):
            _fail("", f"expected {name} unchanged by dense_forward (compute z = x @ W + b in a new array, never "
                      f"with +=)")


@pytest.mark.parametrize("x_shape, W_shape, b_shape, why", [
    pytest.param((4, 3), (2, 5), (5,), "x has 3 features, W expects 2 (x.shape[-1] != W.shape[0])", id="x_vs_W"),
    pytest.param((3,), (4, 2), None, "a 1-D x of 3 values for a W with n_in = 4", id="one_sample_vs_W"),
    pytest.param((4, 3), (3, 5), (4,), "b has 4 values for 5 neurons", id="b_length"),
    pytest.param((4, 3), (3, 5), (1, 5), "b of shape (1, 5) instead of (5,)", id="b_2d"),
    pytest.param((4, 3), (5, 3), (3,), "W transposed: shape (n_out, n_in) like torch", id="W_transposed"),
    pytest.param((2, 4, 3), (3, 5), None, "a 3-D x: neither one sample nor a batch of rows", id="x_3d"),
    pytest.param((4, 3), (3,), None, "a 1-D W: weights must have shape (n_in, n_out)", id="W_1d"),
])
def test_dense_forward_rejects_inconsistent_shapes(layers, x_shape, W_shape, b_shape, why):
    x, W = np.ones(x_shape), np.ones(W_shape)
    b = None if b_shape is None else np.ones(b_shape)
    assert_raises(ValueError, layers.dense_forward, x, W, b, why=why)


# ===================================================================== init_weights (16.14)
# method -> (kind, spread as a function of n_in and n_out); the targets of torch.nn.init are checked below
SPREAD = {
    "lecun_uniform": ("uniform", lambda i, o: np.sqrt(3 / i)),
    "lecun_normal": ("normal", lambda i, o: np.sqrt(1 / i)),
    "glorot_uniform": ("uniform", lambda i, o: np.sqrt(6 / (i + o))),
    "xavier_uniform": ("uniform", lambda i, o: np.sqrt(6 / (i + o))),
    "glorot_normal": ("normal", lambda i, o: np.sqrt(2 / (i + o))),
    "xavier_normal": ("normal", lambda i, o: np.sqrt(2 / (i + o))),
    "he_uniform": ("uniform", lambda i, o: np.sqrt(6 / i)),
    "he_normal": ("normal", lambda i, o: np.sqrt(2 / i)),
}
RANDOM_METHODS = ["uniform", "normal", *SPREAD]


def target_std(method, n_in, n_out, scale=0.05):
    """Standard deviation that a draw of `method` must have (a uniform law U(-a, a) has std a / sqrt(3))."""
    if method == "uniform":
        return scale / np.sqrt(3)
    if method == "normal":
        return scale
    kind, spread = SPREAD[method]
    return spread(n_in, n_out) / (np.sqrt(3) if kind == "uniform" else 1.0)


def draw(layers, n_in, n_out, method, seed, **kwargs):
    got = call(layers.init_weights, n_in, n_out, method, rng=np.random.default_rng(seed), **kwargs,
               expected=f"a ({n_in}, {n_out}) array for method={method!r}")
    if not isinstance(got, np.ndarray):
        _fail("", f"expected a NumPy array for method={method!r}, got {type(got).__name__}")
    if got.shape != (n_in, n_out):
        _fail("", f"expected shape (n_in, n_out) = ({n_in}, {n_out}) for method={method!r}, got {got.shape} "
                  f"(W has one row per input and one column per neuron)")
    if got.dtype != np.float64:
        _fail("", f"expected dtype float64 for method={method!r}, got {got.dtype}")
    return got


def test_init_weights_docstring_examples(layers):
    assert_close(call(layers.init_weights, 2, 3, "constant", scale=0.5), np.full((2, 3), 0.5),
                 msg="init_weights(2, 3, 'constant', scale=0.5): every weight equals scale")
    W = draw(layers, 100, 50, "glorot_uniform", 0)
    if not np.abs(W).max() <= np.sqrt(6 / 150):
        _fail("", f"expected every glorot_uniform weight within the bound sqrt(6 / (n_in + n_out)) = "
                  f"{np.sqrt(6 / 150):.6f}, got a largest |w| of {np.abs(W).max():.6f}")
    W1, W2 = draw(layers, 3, 2, "he_normal", 42), draw(layers, 3, 2, "he_normal", 42)
    if not np.array_equal(W1, W2):
        _fail("", "expected identical arrays for two generators of the same seed (draw only with the rng you receive)")


def test_init_weights_zeros_and_constant(layers):
    assert_close(call(layers.init_weights, 4, 3, "zeros"), np.zeros((4, 3)), msg="init_weights(4, 3, 'zeros')")
    assert_close(call(layers.init_weights, 2, 5, "constant", scale=-1.25), np.full((2, 5), -1.25),
                 msg="init_weights(2, 5, 'constant', scale=-1.25): every weight equals scale")
    assert_close(call(layers.init_weights, 3, 3, "constant"), np.full((3, 3), 0.05),
                 msg="the default scale of 'constant' is 0.05")


@pytest.mark.parametrize("method", RANDOM_METHODS)
def test_init_weights_spread_of_each_method(layers, method):
    n_in, n_out = 1000, 1000
    W = draw(layers, n_in, n_out, method, 16140)
    want = target_std(method, n_in, n_out)
    got = float(W.std())
    if abs(got / want - 1) > 0.02:
        rule = {"uniform": "U(-scale, scale)", "normal": "N(0, scale**2)"}.get(method, method)
        _fail("", f"expected a standard deviation of {want:.5f} for {rule} with n_in = n_out = 1000 (within 2 %), "
                  f"got {got:.5f}; remember that U(-a, a) has std a / sqrt(3), and that a normal law takes its "
                  f"std, not its variance")
    if abs(float(W.mean())) > 5 * want / np.sqrt(W.size):
        _fail("", f"expected weights centred on 0 for {method!r}, got a mean of {float(W.mean()):.3g}")


@pytest.mark.parametrize("method", [m for m in RANDOM_METHODS if m.endswith("uniform")])
def test_init_weights_uniform_draws_fill_their_interval(layers, method):
    n_in, n_out = 400, 600
    W = draw(layers, n_in, n_out, method, 16141)
    bound = 0.05 if method == "uniform" else SPREAD[method][1](n_in, n_out)
    if np.abs(W).max() > bound:
        _fail("", f"expected every {method} weight inside [-a, a] with a = {bound:.6f}, got {np.abs(W).max():.6f}")
    if np.abs(W).max() < 0.999 * bound:
        _fail("", f"expected {method} weights spread over the whole interval [-a, a], a = {bound:.6f}; the largest "
                  f"|w| is only {np.abs(W).max():.6f}")
    inside_half = float(np.mean(np.abs(W) < bound / 2))
    if abs(inside_half - 0.5) > 0.01:
        _fail("", f"expected half of the {method} weights in [-a/2, a/2] (a uniform law), got {inside_half:.3f}: "
                  f"is it a normal draw?")


@pytest.mark.parametrize("method", [m for m in RANDOM_METHODS if m.endswith("normal")])
def test_init_weights_normal_draws_are_bell_shaped_and_not_truncated(layers, method):
    n_in, n_out = 1000, 1000
    W = draw(layers, n_in, n_out, method, 16142)
    s = target_std(method, n_in, n_out)
    within = float(np.mean(np.abs(W) < s))
    if abs(within - 0.6827) > 0.01:
        _fail("", f"expected about 68.3 % of the {method} weights within one standard deviation ({s:.5f}) of 0, got "
                  f"{within:.1%}: is it a uniform draw?")
    if np.abs(W).max() < 4 * s:
        _fail("", f"expected some {method} weights beyond 4 standard deviations among 10**6 draws (normal draws are NOT "
                  f"truncated, PyTorch convention), got a largest |w| of {np.abs(W).max() / s:.2f} std")


@pytest.mark.parametrize("method", ["lecun_normal", "he_normal", "he_uniform", "lecun_uniform"])
def test_init_weights_uses_the_fan_in(layers, method):
    wide_in = draw(layers, 2000, 20, method, 16143)
    wide_out = draw(layers, 20, 2000, method, 16144)
    for (n_in, n_out), W in (((2000, 20), wide_in), ((20, 2000), wide_out)):
        want = target_std(method, n_in, n_out)
        if abs(float(W.std()) / want - 1) > 0.03:
            _fail("", f"expected a standard deviation of {want:.4f} for {method} with n_in = {n_in}, n_out = {n_out} "
                      f"(it depends on the fan-in n_in only), got {float(W.std()):.4f}: are n_in and n_out swapped?")


@pytest.mark.parametrize("method", ["glorot_normal", "glorot_uniform"])
def test_init_weights_glorot_uses_both_fans(layers, method):
    W = draw(layers, 1500, 500, method, 16145)
    want = target_std(method, 1500, 500)
    if abs(float(W.std()) / want - 1) > 0.03:
        _fail("", f"expected a standard deviation of {want:.5f} for {method} with n_in = 1500 and n_out = 500 "
                  f"(Glorot averages the two fans: 2 / (n_in + n_out)), got {float(W.std()):.5f}")


@needs_torch
@pytest.mark.parametrize("method, torch_init", [
    ("glorot_uniform", lambda t: torch.nn.init.xavier_uniform_(t)),
    ("glorot_normal", lambda t: torch.nn.init.xavier_normal_(t)),
    ("he_uniform", lambda t: torch.nn.init.kaiming_uniform_(t, mode="fan_in", nonlinearity="relu")),
    ("he_normal", lambda t: torch.nn.init.kaiming_normal_(t, mode="fan_in", nonlinearity="relu")),
    ("lecun_uniform", lambda t: torch.nn.init.kaiming_uniform_(t, mode="fan_in", nonlinearity="linear")),
    ("lecun_normal", lambda t: torch.nn.init.kaiming_normal_(t, mode="fan_in", nonlinearity="linear")),
], ids=lambda value: value if isinstance(value, str) else "torch")
def test_init_weights_matches_the_spread_of_torch_init(layers, method, torch_init):
    n_in, n_out = 800, 300
    torch.manual_seed(1614)
    reference = torch_init(torch.empty(n_out, n_in, dtype=torch.float64)).numpy().T   # torch stores (out, in)
    W = draw(layers, n_in, n_out, method, 16146)
    if abs(float(W.std()) / float(reference.std()) - 1) > 0.02:
        _fail("", f"expected the standard deviation of torch.nn.init for {method} with n_in = {n_in}, n_out = {n_out}: "
                  f"{float(reference.std()):.5f} (within 2 %), got {float(W.std()):.5f}")
    if method.endswith("uniform") and np.abs(W).max() > np.abs(reference).max() * 1.001:
        _fail("", f"expected {method} weights within the bound of torch ({np.abs(reference).max():.5f}), got "
                  f"{np.abs(W).max():.5f}")


def test_init_weights_aliases_xavier_and_glorot(layers):
    for glorot, xavier in (("glorot_uniform", "xavier_uniform"), ("glorot_normal", "xavier_normal")):
        if not np.array_equal(draw(layers, 30, 20, glorot, 7), draw(layers, 30, 20, xavier, 7)):
            _fail("", f"expected {xavier!r} to be the same method as {glorot!r} (same seed, same array)")


@pytest.mark.parametrize("method", RANDOM_METHODS)
def test_init_weights_same_seed_same_draw_other_seed_other_draw(layers, method):
    first, again = draw(layers, 20, 30, method, 11), draw(layers, 20, 30, method, 11)
    other = draw(layers, 20, 30, method, 12)
    if not np.array_equal(first, again):
        _fail("", f"expected the same array for two generators of seed 11 (method={method!r}): draw only with the rng "
                  f"you receive")
    if np.array_equal(first, other):
        _fail("", f"expected another array for seed 12 than for seed 11 (method={method!r}): is the rng ignored?")


def test_init_weights_shared_generator_moves_on(layers):
    rng = np.random.default_rng(16147)
    first = call(layers.init_weights, 10, 10, "he_normal", rng=rng)
    second = call(layers.init_weights, 10, 10, "he_normal", rng=rng)
    if np.array_equal(first, second):
        _fail("", "expected two different arrays from two calls with the SAME generator (it moves on after each "
                  "draw; never recreate it from a fixed seed inside the function)")


def test_init_weights_without_rng_is_random_and_leaves_the_global_state(layers):
    np.random.seed(1614)
    expected_next = np.random.random()
    np.random.seed(1614)
    first = call(layers.init_weights, 8, 8, "he_normal")
    second = call(layers.init_weights, 8, 8, "he_normal")
    if np.random.random() != expected_next:
        _fail("", "expected the global NumPy generator (np.random.seed, np.random.normal...) untouched: use "
                  "np.random.default_rng() when rng is None")
    if np.array_equal(first, second):
        _fail("", "expected two different arrays from two calls without rng (rng=None means a fresh, non-reproducible "
                  "generator)")


def test_init_weights_default_method_is_he_normal(layers):
    W = call(layers.init_weights, 500, 400, rng=np.random.default_rng(16148))
    want = np.sqrt(2 / 500)
    if abs(float(np.asarray(W).std()) / want - 1) > 0.02:
        _fail("", f"expected the default method he_normal (std sqrt(2 / n_in) = {want:.5f}), got a std of "
                  f"{float(np.asarray(W).std()):.5f}")


def test_init_weights_scale_is_used_by_uniform_and_normal(layers):
    W_u = draw(layers, 300, 300, "uniform", 3, scale=0.3)
    W_n = draw(layers, 300, 300, "normal", 4, scale=0.3)
    if not 0.299 < np.abs(W_u).max() <= 0.3:
        _fail("", f"expected uniform weights in [-0.3, 0.3] filling the interval for scale=0.3, got a largest |w| of "
                  f"{np.abs(W_u).max():.4f}")
    if abs(float(W_n.std()) / 0.3 - 1) > 0.02:
        _fail("", f"expected a std of 0.3 for normal with scale=0.3 (scale is the standard deviation), got "
                  f"{float(W_n.std()):.4f}")


@pytest.mark.parametrize("args, why", [
    pytest.param((3, 4, "kaiming"), "an unknown method name", id="unknown_method"),
    pytest.param((3, 4, "He_normal"), "names are case-sensitive", id="case"),
    pytest.param((0, 4, "he_normal"), "n_in = 0", id="zero_fan_in"),
    pytest.param((3, 0, "glorot_uniform"), "n_out = 0", id="zero_fan_out"),
    pytest.param((-2, 4, "zeros"), "a negative n_in", id="negative"),
    pytest.param((True, 4, "he_normal"), "a bool is not a size", id="bool"),
    pytest.param((3, 2.5, "lecun_normal"), "a size that is not an integer", id="non_integer"),
])
def test_init_weights_rejects_invalid_arguments(layers, args, why):
    assert_raises(ValueError, layers.init_weights, *args, rng=np.random.default_rng(0), why=why)


# ===================================================================== init_mlp (16.17)
def get_params(layers, sizes, method="he_normal", seed=0):
    params = call(layers.init_mlp, sizes, method, rng=np.random.default_rng(seed),
                  expected=f"a list of (W, b) for {sizes}")
    if params is None:
        _fail("", "expected a list of (W, b) tuples, got None: did you forget the return?")
    params = list(params)
    if len(params) != len(sizes) - 1:
        _fail("", f"expected {len(sizes) - 1} layers (one (W, b) per pair of consecutive sizes) for {sizes}, got "
                  f"{len(params)}")
    return params


def test_init_mlp_docstring_example(layers):
    params = get_params(layers, [2, 4, 4, 1])
    shapes = [np.shape(W) for W, b in params]
    if shapes != [(2, 4), (4, 4), (4, 1)]:
        _fail("", f"expected the weight shapes [(2, 4), (4, 4), (4, 1)] (W of layer l: (n_(l-1), n_l)), got {shapes}")
    total = sum(np.size(W) + np.size(b) for W, b in params)
    if total != 37:
        _fail("", f"expected 37 parameters in total for [2, 4, 4, 1], got {total}")


def test_init_mlp_biases_are_zero_vectors(layers):
    for index, (W, b) in enumerate(get_params(layers, [5, 7, 3, 2])):
        b = np.asarray(b)
        if b.shape != (np.shape(W)[1],) or np.any(b != 0) or b.dtype != np.float64:
            _fail("", f"expected the bias of layer {index} to be np.zeros({np.shape(W)[1]}) (float64, shape (n_l,)), "
                      f"got {_short(b)} of shape {b.shape}")


def test_init_mlp_each_layer_uses_its_own_fan_in(layers):
    sizes = [1000, 200, 800, 50]
    for index, ((W, _), (n_in, n_out)) in enumerate(zip(get_params(layers, sizes, "he_normal", 3),
                                                        zip(sizes[:-1], sizes[1:]))):
        want = np.sqrt(2 / n_in)
        if abs(float(np.std(W)) / want - 1) > 0.05:
            _fail("", f"expected layer {index} ({n_in} -> {n_out}) drawn with he_normal, std sqrt(2 / {n_in}) = "
                      f"{want:.4f}, got {float(np.std(W)):.4f}")


def test_init_mlp_uses_the_requested_method(layers):
    params = get_params(layers, [300, 300, 300], "glorot_uniform", 5)
    bound = np.sqrt(6 / 600)
    for index, (W, _) in enumerate(params):
        if not 0.99 * bound < np.abs(W).max() <= bound:
            _fail("", f"expected layer {index} drawn with glorot_uniform, bound {bound:.5f}, got a largest |w| of "
                      f"{np.abs(W).max():.5f}")


def test_init_mlp_layers_are_different_draws(layers):
    (W1, _), (W2, _), (W3, _) = get_params(layers, [20, 20, 20, 20], "he_normal", 9)
    if np.array_equal(W1, W2) or np.array_equal(W2, W3):
        _fail("", "expected different weights in layers of the same shape: share ONE generator, drawn layer after "
                  "layer (never a new generator with the same seed for each layer)")


def test_init_mlp_same_seed_same_parameters(layers):
    first, again = get_params(layers, [6, 5, 4], seed=21), get_params(layers, [6, 5, 4], seed=21)
    other = get_params(layers, [6, 5, 4], seed=22)
    if not all(np.array_equal(W1, W2) for (W1, _), (W2, _) in zip(first, again)):
        _fail("", "expected the same parameters for two generators of the same seed")
    if all(np.array_equal(W1, W2) for (W1, _), (W2, _) in zip(first, other)):
        _fail("", "expected other parameters for another seed: is the rng ignored?")


def test_init_mlp_draws_the_layers_in_order(layers):
    short = get_params(layers, [6, 5], seed=31)
    longer = get_params(layers, [6, 5, 4, 3], seed=31)
    if not np.array_equal(short[0][0], longer[0][0]):
        _fail("", "expected the first layer of [6, 5, 4, 3] to equal the only layer of [6, 5] with the same seed: the "
                  "generator is drawn in layer order, first layer first")


@pytest.mark.parametrize("sizes, method, why", [
    pytest.param([3], "he_normal", "a single size", id="one_size"),
    pytest.param([3, 0, 2], "he_normal", "a layer of width 0", id="zero_width"),
    pytest.param([3, 4], "kaiming", "an unknown method", id="unknown_method"),
])
def test_init_mlp_rejects_invalid_arguments(layers, sizes, method, why):
    assert_raises(ValueError, layers.init_mlp, sizes, method, rng=np.random.default_rng(0), why=why)


# ===================================================================== mlp_forward (16.17)
def make_params(sizes, seed):
    """Parameters built here (never with your init_mlp): standard normal weights and biases, scaled by 1/sqrt(n_in)."""
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=(a, b)) / np.sqrt(a), rng.normal(size=b)) for a, b in zip(sizes[:-1], sizes[1:])]


def numpy_forward(x, params, activation=None, output_activation=None):
    a = x
    for index, (W, b) in enumerate(params):
        z = a @ W + b
        function = output_activation if index == len(params) - 1 else activation
        a = z if function is None else function(z)
    return a


def test_mlp_forward_docstring_example(layers):
    params = [(np.array([[1.0, -1.0], [0.5, -2.0]]), np.array([0.0, 1.0])), (np.array([[1.0], [1.0]]), np.array([-1.0]))]
    x = np.array([[1.0, 1.0]])
    assert_close(call(layers.mlp_forward, x, params, activation=relu), [[0.5]],
                 msg="the docstring example: hidden z = [1.5, -2], relu = [1.5, 0], output 1.5 - 1")
    result = call(layers.mlp_forward, x, params, activation=relu, return_cache=True)
    if not isinstance(result, tuple) or len(result) != 2:
        _fail("", f"expected a tuple (out, cache) with return_cache=True, got {type(result).__name__}")
    out, cache = result
    if len(cache) != 2:
        _fail("", f"expected one cache entry per layer (2), got {len(cache)}")
    assert_close(cache[0][1], [[1.5, -2.0]], msg="cache[0][1]: the pre-activation z of the hidden layer, BEFORE relu")
    assert_close(cache[1][0], [[1.5, 0.0]], msg="cache[1][0]: the input of the output layer, relu of cache[0][1]")


@needs_torch
@pytest.mark.parametrize("name", ["tanh", "relu"])
def test_mlp_forward_matches_torch_sequential(layers, name):
    sizes = [6, 16, 12, 3]
    params = make_params(sizes, 1617)
    x = np.random.default_rng(16171).normal(size=(40, 6))
    act_np, act_torch = {"tanh": (np.tanh, torch.nn.Tanh), "relu": (relu, torch.nn.ReLU)}[name]
    modules = []
    for index, (W, b) in enumerate(params):
        linear = torch.nn.Linear(W.shape[0], W.shape[1], dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(W.T))     # torch stores W.T, shape (out, in)
            linear.bias.copy_(torch.from_numpy(b))
        modules.append(linear)
        if index < len(params) - 1:
            modules.append(act_torch())
    with torch.no_grad():
        expected = torch.nn.Sequential(*modules)(torch.from_numpy(x)).numpy()
    got = call(layers.mlp_forward, x.copy(), [(W.copy(), b.copy()) for W, b in params], activation=act_np)
    assert_close(got, expected, atol=1e-10, msg=f"the output of a 6-16-12-3 network with {name} hidden layers and a "
                                               f"linear output, as torch.nn.Sequential with the same weights")


def test_mlp_forward_matches_numpy_with_an_output_activation(layers):
    sizes = [4, 8, 5]
    params = make_params(sizes, 16172)
    x = np.random.default_rng(16173).normal(size=(25, 4))
    expected = numpy_forward(x, params, np.tanh, softmax)
    got = call(layers.mlp_forward, x.copy(), params, activation=np.tanh, output_activation=softmax)
    assert_close(got, expected, msg="tanh on the hidden layer, softmax on the output layer only")


def test_mlp_forward_without_activation_is_one_affine_map(layers):
    params = make_params([3, 5, 4, 2], 16174)
    x = np.random.default_rng(16175).normal(size=(10, 3))
    W_all = params[0][0] @ params[1][0] @ params[2][0]
    b_all = (params[0][1] @ params[1][0] + params[1][1]) @ params[2][0] + params[2][1]
    assert_close(call(layers.mlp_forward, x, params), x @ W_all + b_all,
                 msg="with activation=None (identity), three layers compose into one affine map x @ W + b")


def test_mlp_forward_activation_is_applied_to_hidden_layers_only(layers):
    params = make_params([3, 4, 2], 16176)
    x = np.random.default_rng(16177).normal(size=(7, 3))
    kill = lambda z: np.zeros_like(z)   # noqa: E731 - every hidden output becomes 0
    assert_close(call(layers.mlp_forward, x, params, activation=kill), np.tile(params[1][1], (7, 1)),
                 msg="with a hidden activation that returns zeros, the output must be the last bias (the activation "
                     "is applied to the hidden layers, not to the output)")
    single = make_params([3, 2], 16178)
    assert_close(call(layers.mlp_forward, x, single, activation=kill), x @ single[0][0] + single[0][1],
                 msg="a network of ONE layer has no hidden layer: `activation` must not be applied to its output")


def test_mlp_forward_cache_holds_inputs_and_pre_activations(layers):
    params = make_params([4, 6, 5, 3], 16179)
    x = np.random.default_rng(16180).normal(size=(9, 4))
    result = call(layers.mlp_forward, x, params, activation=np.tanh, return_cache=True)
    if not isinstance(result, tuple) or len(result) != 2:
        _fail("", f"expected a tuple (out, cache) with return_cache=True, got {type(result).__name__}")
    out, cache = result
    if len(cache) != 3:
        _fail("", f"expected one (a_prev, z) per layer (3), got {len(cache)}")
    a = x
    for index, ((W, b), entry) in enumerate(zip(params, cache)):
        if len(entry) != 2:
            _fail("", f"expected cache[{index}] to be a pair (a_prev, z), got {len(entry)} items")
        a_prev, z = entry
        assert_close(a_prev, a, msg=f"cache[{index}][0]: the input of layer {index}" + (" (x itself)" if index == 0 else
                                                                                      " (tanh of the previous z)"))
        assert_close(z, a @ W + b, msg=f"cache[{index}][1]: z = a_prev @ W + b of layer {index}, before tanh")
        a = np.tanh(z) if index < 2 else z
    assert_close(out, cache[-1][1], msg="without output_activation, the output equals the last pre-activation z")


def test_mlp_forward_without_cache_returns_only_the_output(layers):
    params = make_params([3, 4, 2], 16181)
    got = call(layers.mlp_forward, np.ones((2, 3)), params, activation=relu)
    if isinstance(got, tuple):
        _fail("", "expected only the output array when return_cache is False (the default), got a tuple")
    assert_close(got, numpy_forward(np.ones((2, 3)), params, relu), msg="the output for return_cache=False")


def test_mlp_forward_does_not_modify_its_inputs(layers):
    params = make_params([3, 5, 2], 16182)
    copies = [(W.copy(), b.copy()) for W, b in params]
    x = np.random.default_rng(16183).normal(size=(4, 3))
    x0 = x.copy()
    call(layers.mlp_forward, x, params, activation=relu, return_cache=True)
    if not np.array_equal(x, x0):
        _fail("", "expected x unchanged by mlp_forward")
    if not all(np.array_equal(W, W0) and np.array_equal(b, b0) for (W, b), (W0, b0) in zip(params, copies)):
        _fail("", "expected the parameters unchanged by mlp_forward (never update W or b in place)")


def test_mlp_forward_rebuilds_the_predictions_of_sklearn(layers):
    from sklearn.datasets import make_blobs
    from sklearn.neural_network import MLPClassifier

    X, y = make_blobs(n_samples=150, centers=3, n_features=4, cluster_std=2.5, random_state=1618)
    clf = MLPClassifier(hidden_layer_sizes=(8, 6), activation="relu", max_iter=300, random_state=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # a ConvergenceWarning does not matter here
        clf.fit(X, y)
    params = [(W.copy(), b.copy()) for W, b in zip(clf.coefs_, clf.intercepts_)]
    logits = as_array(call(layers.mlp_forward, X.copy(), params, activation=relu), "the logits")
    predicted = clf.classes_[np.argmax(logits, axis=1)]
    if not np.array_equal(predicted, clf.predict(X)):
        _fail("", f"expected the argmax of your logits to give clf.predict on all 150 samples (coefs_[l] has shape "
                  f"(n_in, n_out), like your W), got {int(np.sum(predicted != clf.predict(X)))} differences")


@pytest.mark.parametrize("case, why", [
    pytest.param("empty", "an empty list of parameters", id="empty_params"),
    pytest.param("chain", "layer 1 expects 5 inputs but layer 0 has 4 neurons", id="broken_chain"),
    pytest.param("input", "x has 2 features, the first layer expects 3", id="bad_input"),
])
def test_mlp_forward_rejects_invalid_networks(layers, case, why):
    params = {"empty": [], "chain": [(np.ones((3, 4)), np.zeros(4)), (np.ones((5, 2)), np.zeros(2))],
              "input": make_params([3, 4, 2], 1)}[case]
    x = np.ones((2, 2)) if case == "input" else np.ones((2, 3))
    assert_raises(ValueError, layers.mlp_forward, x, params, why=why)


# ===================================================================== topological_order (16.16)
def greedy_order(graph):
    """The order of the docstring, written the slow way: repeatedly place the smallest node whose parents are placed."""
    nodes = set(graph) | {child for kids in graph.values() for child in kids}
    parents = {node: {p for p, kids in graph.items() if node in kids} for node in nodes}
    order, placed = [], set()
    while len(order) < len(nodes):
        ready = sorted(node for node in nodes - placed if parents[node] <= placed)
        if not ready:
            return None   # a cycle
        order.append(ready[0])
        placed.add(ready[0])
    return order


def has_cycle(graph):
    sorter = graphlib.TopologicalSorter({node: [] for node in graph})
    for parent, kids in graph.items():
        for child in kids:
            sorter.add(child, parent)
    try:
        sorter.prepare()
    except graphlib.CycleError:
        return True
    return False


def get_order(layers, graph):
    copy = {key: list(value) for key, value in graph.items()}
    got = call(layers.topological_order, copy, expected=f"an order of the nodes of {graph}")
    if copy != {key: list(value) for key, value in graph.items()}:
        _fail("", "expected the graph unchanged by topological_order (count the parents in a separate dict)")
    if not isinstance(got, list) or not all(isinstance(node, str) for node in got):
        _fail("", f"expected a list of node names (str), got {got!r}")
    return got


def random_dag(rng, n_nodes, p_edge):
    names = [f"n{i:02d}" for i in range(n_nodes)]
    order = list(rng.permutation(names))          # the hidden order: edges only go forward in it
    graph = {}
    for i, j in itertools.combinations(range(n_nodes), 2):
        if rng.random() < p_edge:
            graph.setdefault(order[i], []).append(order[j])
    for name in names:
        if rng.random() < 0.5:
            graph.setdefault(name, [])
    return graph


def test_topological_order_docstring_examples(layers):
    got = get_order(layers, {"x": ["h1", "h2"], "h1": ["y"], "h2": ["y"]})
    if got != ["x", "h1", "h2", "y"]:
        _fail("", f"expected ['x', 'h1', 'h2', 'y'] (x first, then the ready nodes in alphabetical order), got {got}")
    got = get_order(layers, {"b": ["c"], "a": ["c"]})
    if got != ["a", "b", "c"]:
        _fail("", f"expected ['a', 'b', 'c'] (a and b are ready together: the alphabetically smallest first), got {got}")


def test_topological_order_includes_nodes_that_are_only_children(layers):
    got = get_order(layers, {"in": ["mid"], "mid": ["out1", "out2"]})
    if got != ["in", "mid", "out1", "out2"]:
        _fail("", f"expected ['in', 'mid', 'out1', 'out2'] (out1 and out2 only appear as children), got {got}")


def test_topological_order_keeps_isolated_nodes(layers):
    got = get_order(layers, {"b": [], "a": ["c"], "z": []})
    if got != ["a", "b", "c", "z"]:
        _fail("", f"expected ['a', 'b', 'c', 'z'] (b and z have no edge but are nodes of the graph), got {got}")


def test_topological_order_smallest_ready_node_not_breadth_first(layers):
    # c becomes ready only once b is placed: a level-by-level order (batches of graphlib) would give a, b, z, c
    graph = {"a": ["b", "z"], "b": ["c"]}
    got = get_order(layers, graph)
    if got != ["a", "b", "c", "z"]:
        _fail("", f"expected ['a', 'b', 'c', 'z']: after a, the ready nodes are b and z (b first); placing b makes c ready, "
                  f"and c comes before z. Got {got} (a level-by-level order, a, b, z, c, is not the rule of the docstring)")


def test_topological_order_matches_the_greedy_rule_on_random_dags(layers):
    rng = np.random.default_rng(1616)
    for trial in range(40):
        graph = random_dag(rng, int(rng.integers(2, 16)), float(rng.uniform(0.1, 0.5)))
        expected = greedy_order(graph)
        got = get_order(layers, graph)
        if got != expected:
            _fail("", f"expected {expected} (at each step, the alphabetically smallest node whose parents are all placed), "
                      f"got {got}", data=f"graph = {graph}")


def test_topological_order_respects_every_edge_of_the_book_figure(layers):
    # a graph like figure 16.1 (a): skip connections, nodes that feed several layers
    graph = {"i1": ["a"], "i2": ["b"], "i3": ["b"], "i4": ["c"], "a": ["d"], "b": ["e", "c"], "c": ["f"],
             "e": ["d", "f"], "d": ["o1", "f"], "f": ["o2"]}
    got = get_order(layers, graph)
    position = {node: i for i, node in enumerate(got)}
    nodes = set(graph) | {child for kids in graph.values() for child in kids}
    if set(got) != nodes or len(got) != len(nodes):
        _fail("", f"expected each of the {len(nodes)} nodes exactly once, got {got}")
    for parent, kids in graph.items():
        for child in kids:
            if position[parent] > position[child]:
                _fail("", f"expected {parent!r} before its child {child!r}, got {got}")


@pytest.mark.parametrize("graph, why", [
    pytest.param({"a": ["a"]}, "a node that feeds itself", id="self_loop"),
    pytest.param({"a": ["b"], "b": ["a"]}, "a two-node loop", id="two_nodes"),
    pytest.param({"x": ["h"], "h": ["y"], "y": ["h"]}, "a loop after the input", id="loop_after_input"),
    pytest.param({"s": ["t"], "p": ["q"], "q": ["r"], "r": ["p"]}, "a loop with no entry, beside a valid part",
                 id="loop_without_entry"),
])
def test_topological_order_rejects_cycles(layers, graph, why):
    assert has_cycle(graph), "the test graph must have a cycle"
    assert_raises(ValueError, layers.topological_order, graph, why=f"{why} (a feedback loop)")


def test_topological_order_detects_exactly_the_cycles_of_graphlib(layers):
    rng = np.random.default_rng(16161)
    for trial in range(40):
        graph = random_dag(rng, int(rng.integers(3, 10)), 0.3)
        if trial % 2:                                   # add one edge that may close a loop
            nodes = sorted(set(graph) | {child for kids in graph.values() for child in kids})
            a, b = rng.choice(nodes, size=2, replace=True)
            graph.setdefault(str(a), []).append(str(b))
        if has_cycle(graph):
            assert_raises(ValueError, layers.topological_order, graph,
                          why=f"graphlib finds a cycle in {graph}")
        else:
            got = get_order(layers, graph)
            if got != greedy_order(graph):
                _fail("", f"expected {greedy_order(graph)}, got {got}", data=f"graph = {graph}")
