"""Tests of mylearn.perceptron (chapter 10): oracle tests and properties.

    pytest tests/test_ch10_perceptron.py              # your code (mon_travail/mylearn/perceptron.py)
    pytest tests/test_ch10_perceptron.py --impl=ref   # the reference

Oracles: NumPy (``np.where`` for the threshold, ``np.hstack`` for the bias column),
PyTorch (``torch.nn.functional.linear`` in double precision for the weighted sums of a
neuron) and scikit-learn (``sklearn.linear_model.Perceptron(shuffle=False, tol=None)``,
whose weights are exactly those of the classic learning rule starting from zero).
Every test name starts with the name of what it tests, so that each exercise runs its
own group: ``-k "test_sign_step_ or test_add_bias_column_"`` (10.12),
``"test_neuron_forward_"`` (10.14), ``"test_perceptron_"`` (10.21).
The tests of one exercise never call the functions of another one: the tests of the
perceptron do not use your ``sign_step`` (your class may use it, or not).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import Perceptron as SklearnPerceptron


@pytest.fixture
def pcp(mylearn_module):
    return mylearn_module("perceptron")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for the first line of a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    if arr.dtype.kind in "biuf":
        text = np.array2string(arr, precision=digits, separator=", ", threshold=12,
                               edgeitems=2 if arr.ndim > 1 else 3, max_line_width=10**6)
    else:
        text = repr(arr.tolist())
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = "", details: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data and details."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    if details:
        lines.append(details)
    raise AssertionError("\n".join(lines)) from None


def _first_difference(got, want, rtol, atol) -> str:
    """' (first difference at index i: expected a, got b)' for two arrays of the same shape, '' otherwise."""
    if got.ndim == 0 or got.shape != want.shape:
        return ""
    bad = np.flatnonzero(~np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True))
    if bad.size == 0:
        return ""
    i = int(bad[0])
    index = np.unravel_index(i, got.shape)
    where = int(index[0]) if got.ndim == 1 else tuple(int(k) for k in index)
    a, b = want.ravel()[i], got.ravel()[i]
    digits = 6 if f"{a:.6g}" != f"{b:.6g}" else 17
    return f" (first difference at index {where}: expected {a:.{digits}g}, got {b:.{digits}g})"


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg="", data=""):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected {_short(expected)}, got an object of type {type(result).__name__}", data)
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}: expected {_short(expected)}, "
                   f"got {_short(result)}", data)
    if np.isnan(got).any() and not np.isnan(want).any():
        _fail(msg, f"expected {_short(expected)}, got NaN values: {_short(result)}", data)
    try:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    except AssertionError as exc:
        have, need = _short(result), _short(expected)
        if have == need:
            have, need = _short(result, 17), _short(expected, 17)
        _fail(msg, f"expected {need}, got {have}{_first_difference(got, want, rtol, atol)}", data, str(exc))


def assert_raises_value_error(function, *args, why="", **kwargs):
    """The call must raise ValueError (or a subclass); `why` names the case in the failure message."""
    name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
    if name == "<lambda>":
        name = "the call"
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    except NotImplementedError:
        raise                  # shown as "⏳ pas encore implémenté" by conftest.py
    except Exception as exc:  # noqa: BLE001 - say which error was raised instead
        raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else "")
                             + f" (it raised {type(exc).__name__}: {exc})") from None
    raise AssertionError(f"{name} must raise ValueError" + (f" here: {why}" if why else ""))


def _copied(value):
    return value.copy() if isinstance(value, np.ndarray) else value


def call(function, *args, expected="a result", copy_inputs=True, **kwargs):
    """Call the learner's function; an unexpected exception becomes a failure that says what was expected.

    The function receives copies of the array arguments, so that an array changed in place cannot spoil an oracle
    computed afterwards: only the tests named ..._does_not_modify_... (copy_inputs=False) pass the originals.
    """
    if copy_inputs:
        args = tuple(_copied(value) for value in args)
        kwargs = {key: _copied(value) for key, value in kwargs.items()}
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def fit(model, X, y):
    """model.fit(X, y) without chaining: a fit that forgets `return self` is reported by its own test."""
    call(model.fit, X, y, expected="fit to succeed")
    return model


def learned(model, name):
    """A learned attribute (coef_, intercept_, errors_...), with a clear message when fit did not create it."""
    if not hasattr(model, name):
        raise AssertionError(f"expected fit to create the attribute {name}, but the fitted "
                             f"{type(model).__name__} has no {name} (attributes: {sorted(vars(model))})")
    return getattr(model, name)


def sklearn_perceptron(X, y, eta0=1.0, max_iter=100, fit_intercept=True):
    """The oracle: scikit-learn's perceptron in order (shuffle=False), for exactly max_iter epochs (tol=None)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SklearnPerceptron(eta0=eta0, max_iter=max_iter, shuffle=False, tol=None,
                                 fit_intercept=fit_intercept).fit(X, y)


GATES_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
GATES = {"AND": np.array([0, 0, 0, 1]), "OR": np.array([0, 1, 1, 1]),
         "NAND": np.array([1, 1, 1, 0]), "XOR": np.array([0, 1, 1, 0])}


def separable_data(seed, n=40, p=2, margin=0.3, through_origin=False):
    """Points on both sides of a random hyperplane, none closer than `margin` to it (known separator w, b)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=p)
    w /= np.linalg.norm(w)
    b = 0.0 if through_origin else float(rng.uniform(-0.5, 0.5))
    X = []
    while len(X) < n:
        x = rng.uniform(-2, 2, size=p)
        if abs(x @ w + b) >= margin:
            X.append(x)
    X = np.array(X)
    y = np.where(X @ w + b > 0, 1, 0)
    if len(np.unique(y)) < 2:               # both classes are needed
        X[0], y[0] = -w * (margin + 0.5) - b * w, 0
        X[1], y[1] = w * (margin + 0.5) - b * w, 1
    return X, y, w, b


# ================================================================== sign_step (10.12)
SIGN_CASES = [((7,), 0), ((3, 4), 1), ((2, 3, 2), 2), ((1,), 3)]


@pytest.mark.parametrize("shape, seed", SIGN_CASES, ids=["1d", "2d", "3d", "one-value"])
def test_sign_step_matches_numpy(pcp, shape, seed):
    z = np.random.default_rng(seed).normal(size=shape) * 3
    z.ravel()[0] = 0.0                                       # one weighted sum exactly on the threshold
    got = call(pcp.sign_step, z, expected=f"an array of shape {shape}")
    assert_close(got, np.where(z > 0, 1.0, -1.0), atol=0,
                 msg=f"+1 where z > 0, -1 elsewhere (z = 0 gives -1), same shape {shape} as z")


def test_sign_step_docstring_example(pcp):
    assert_close(call(pcp.sign_step, np.array([-2.0, 0.0, 3.0])), [-1.0, -1.0, 1.0], atol=0,
                 msg="sign_step(np.array([-2.0, 0.0, 3.0]))")


def test_sign_step_zero_gives_minus_one(pcp):
    got = call(pcp.sign_step, np.array([0.0, -0.0, 1e-12, -1e-12]))
    assert_close(got, [-1.0, -1.0, 1.0, -1.0], atol=0,
                 msg="z = 0 (and -0.0) gives -1, the book's convention: compare with z > 0, strictly "
                     "(np.sign would give 0 there)")


def test_sign_step_returns_floats_of_the_same_shape(pcp):
    z = np.array([[3, -1, 0], [0, 5, -2]])                   # integers
    got = call(pcp.sign_step, z)
    if not isinstance(got, np.ndarray) or got.dtype.kind != "f" or got.shape != z.shape:
        raise AssertionError(f"expected a float NumPy array of shape {z.shape}, got "
                             f"{type(got).__name__} of dtype {getattr(got, 'dtype', None)} and shape "
                             f"{np.shape(got)} (use 1.0 and -1.0, not 1 and -1)")
    assert_close(got, [[1, -1, -1], [-1, 1, -1]], atol=0, msg="integer weighted sums")


def test_sign_step_accepts_lists_and_scalars(pcp):
    assert_close(call(pcp.sign_step, [2.5, -0.5, 0.0]), [1.0, -1.0, -1.0], atol=0,
                 msg="a list must be accepted (convert it with np.asarray)")
    got = call(pcp.sign_step, 4.0)
    if np.shape(got) != ():
        raise AssertionError(f"expected a 0-d result for a single number (shape ()), got shape {np.shape(got)}")
    assert_close(got, 1.0, atol=0, msg="sign_step(4.0)")


def test_sign_step_does_not_modify_its_input(pcp):
    z = np.array([-1.5, 0.0, 2.0])
    call(pcp.sign_step, z, copy_inputs=False)
    assert np.array_equal(z, [-1.5, 0.0, 2.0]), "expected sign_step to leave z unchanged, but it modified it"


# ================================================================== add_bias_column (10.12)
BIAS_SHAPES = [(1, 1), (4, 2), (6, 5), (3, 1)]


@pytest.mark.parametrize("shape", BIAS_SHAPES, ids=[f"{n}x{d}" for n, d in BIAS_SHAPES])
def test_add_bias_column_matches_numpy(pcp, shape):
    X = np.random.default_rng(sum(shape)).normal(size=shape)
    got = call(pcp.add_bias_column, X, expected=f"an array of shape ({shape[0]}, {shape[1] + 1})")
    assert_close(got, np.hstack([np.ones((shape[0], 1)), X]), atol=0,
                 msg=f"a column of ones FIRST, then the {shape[1]} column(s) of X")


def test_add_bias_column_docstring_example(pcp):
    got = call(pcp.add_bias_column, np.array([[2.0, 3.0], [4.0, 5.0]]))
    assert_close(got, [[1, 2, 3], [1, 4, 5]], atol=0, msg="add_bias_column(np.array([[2.0, 3.0], [4.0, 5.0]]))")


def test_add_bias_column_gives_the_bias_trick(pcp):
    rng = np.random.default_rng(10)
    X, w, b = rng.normal(size=(5, 3)), rng.normal(size=3), -0.75
    X1 = np.asarray(call(pcp.add_bias_column, X), dtype=float)
    if X1.shape != (5, 4):
        raise AssertionError(f"expected shape (5, 4), got shape {X1.shape}")
    assert_close(X1 @ np.concatenate([[b], w]), X @ w + b,
                 msg="add_bias_column(X) @ [b, w1, w2, w3] must equal X @ w + b (the bias trick)")


def test_add_bias_column_returns_floats_and_keeps_X(pcp):
    X = np.array([[1, 2], [3, 4]])
    X0 = X.copy()
    got = call(pcp.add_bias_column, X, copy_inputs=False)
    if not isinstance(got, np.ndarray) or got.dtype.kind != "f":
        raise AssertionError(f"expected a float NumPy array, got {type(got).__name__} of dtype "
                             f"{getattr(got, 'dtype', None)} (convert with np.asarray(X, dtype=float))")
    assert np.array_equal(X, X0), "expected add_bias_column to leave X unchanged, but it modified it"
    assert_close(got, [[1, 1, 2], [1, 3, 4]], atol=0, msg="integer X")


@pytest.mark.parametrize("X, why", [
    (np.array([2.0, 3.0]), "a 1-D array (one sample must be passed as X.reshape(1, -1))"),
    (np.ones((2, 2, 2)), "X with 3 dimensions"),
    (np.float64(3.0), "a single number"),
], ids=["X-1d", "X-3d", "scalar"])
def test_add_bias_column_rejects_invalid_inputs(pcp, X, why):
    assert_raises_value_error(pcp.add_bias_column, X, why=why)


# ================================================================== neuron_forward (10.14)
def torch_weighted_sums(X, w, b):
    """The oracle: torch.nn.functional.linear in double precision, one weighted sum per row of X."""
    torch = pytest.importorskip("torch")
    X_t = torch.as_tensor(np.atleast_2d(X), dtype=torch.float64)
    w_t = torch.as_tensor(np.asarray(w, dtype=float), dtype=torch.float64)
    out = torch.nn.functional.linear(X_t, w_t[None, :], torch.tensor([float(b)], dtype=torch.float64))
    return out[:, 0].numpy()


FORWARD_CASES = [(0, 5, 3, 0.5), (1, 12, 1, -2.0), (2, 1, 4, 0.0), (3, 30, 6, 1.25)]


@pytest.mark.parametrize("seed, n, d, b", FORWARD_CASES, ids=[f"n{n}-d{d}" for _, n, d, _ in FORWARD_CASES])
def test_neuron_forward_weighted_sums_match_torch(pcp, seed, n, d, b):
    rng = np.random.default_rng(seed)
    X, w = rng.normal(size=(n, d)), rng.normal(size=d)
    got = call(pcp.neuron_forward, X, w, b=b, activation=lambda z: z, expected=f"an array of shape ({n},)")
    assert_close(got, torch_weighted_sums(X, w, b), rtol=1e-12,
                 msg=f"identity activation: the weighted sums X @ w + b, one per row of X (n = {n}, d = {d})")


@pytest.mark.parametrize("seed, n, d, b", FORWARD_CASES, ids=[f"n{n}-d{d}" for _, n, d, _ in FORWARD_CASES])
def test_neuron_forward_default_activation_is_the_threshold(pcp, seed, n, d, b):
    rng = np.random.default_rng(seed + 100)
    X, w = rng.normal(size=(n, d)), rng.normal(size=d)
    got = call(pcp.neuron_forward, X, w, b=b, expected=f"an array of shape ({n},)")
    assert_close(got, np.where(X @ w + b > 0, 1.0, -1.0), atol=0,
                 msg="default activation (sign_step): +1 where X @ w + b > 0, -1 elsewhere")


def test_neuron_forward_applies_any_activation(pcp):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(14)
    X, w, b = rng.normal(size=(8, 3)), rng.normal(size=3), 0.3
    z = torch_weighted_sums(X, w, b)
    for name, activation, oracle in [("np.tanh", np.tanh, torch.tanh),
                                     ("a ReLU, lambda z: np.maximum(z, 0)", lambda v: np.maximum(v, 0.0),
                                      torch.relu)]:
        got = call(pcp.neuron_forward, X, w, b=b, activation=activation)
        assert_close(got, oracle(torch.as_tensor(z)).numpy(), rtol=1e-12,
                     msg=f"activation={name}: activation(X @ w + b)")


def test_neuron_forward_docstring_examples(pcp):
    X, w = np.array([[1.0, 2.0], [-1.0, 0.5]]), np.array([1.0, 1.0])
    assert_close(call(pcp.neuron_forward, X, w, b=-1.0), [1.0, -1.0], atol=0,
                 msg="neuron_forward(X, w, b=-1.0), first docstring example")
    assert_close(call(pcp.neuron_forward, X, w, b=-1.0, activation=lambda z: z), [2.0, -1.5],
                 msg="neuron_forward(X, w, b=-1.0, activation=lambda z: z), second docstring example")


def test_neuron_forward_bias_defaults_to_zero(pcp):
    X, w = np.array([[1.0, -1.0], [2.0, 0.5]]), np.array([0.5, 2.0])
    assert_close(call(pcp.neuron_forward, X, w, activation=lambda z: z), X @ w,
                 msg="without b, the bias is 0: the weighted sums are X @ w")


def test_neuron_forward_single_sample_gives_a_0d_array(pcp):
    w = np.array([1.0, 1.0])
    got = call(pcp.neuron_forward, np.array([1.0, 2.0]), w, b=-1.0, activation=lambda z: z,
               expected="a 0-d array")
    if not isinstance(got, np.ndarray) or got.ndim != 0:
        raise AssertionError(f"expected a 0-d NumPy array for a single sample, as the docstring says, got "
                             f"{type(got).__name__} of shape {np.shape(got)} (np.asarray(...) turns a NumPy "
                             "scalar such as np.float64(2.0) into a 0-d array)")
    assert_close(got, 2.0, msg="one sample [1, 2], w = [1, 1], b = -1: z = 1 + 2 - 1")
    got = call(pcp.neuron_forward, np.array([1.0, 2.0]), w, b=-1.0)
    if np.shape(got) != ():
        raise AssertionError(f"expected a 0-d result for a single sample with sign_step, got shape {np.shape(got)}")
    assert_close(got, 1.0, atol=0, msg="one sample with the default activation")


def test_neuron_forward_calls_the_activation_once_on_the_whole_batch(pcp):
    calls = []

    def spy(z):
        calls.append(np.shape(z))
        return np.asarray(z, dtype=float) * 2.0

    X = np.arange(12.0).reshape(6, 2)
    got = call(pcp.neuron_forward, X, np.array([0.5, -1.0]), b=1.0, activation=spy)
    if calls != [(6,)]:
        raise AssertionError(f"expected ONE call of the activation, on the 6 weighted sums at once (shape (6,)), "
                             f"got the calls {calls} (compute X @ w + b for the whole batch, then apply the "
                             "activation once: no loop over the samples)")
    assert_close(got, 2.0 * (X @ np.array([0.5, -1.0]) + 1.0), msg="activation(X @ w + b)")


def test_neuron_forward_accepts_lists(pcp):
    got = call(pcp.neuron_forward, [[1, 2], [3, -4]], [1, 1], b=0.5, activation=lambda z: z)
    assert_close(got, [3.5, -0.5], msg="lists for X and w must be accepted (convert them with np.asarray)")


def test_neuron_forward_does_not_modify_its_inputs(pcp):
    X, w = np.array([[1.0, 2.0], [-1.0, 0.5]]), np.array([1.0, -1.0])
    keep = (X.copy(), w.copy())
    call(pcp.neuron_forward, X, w, b=2.0, copy_inputs=False)
    assert np.array_equal(X, keep[0]) and np.array_equal(w, keep[1]), \
        "expected neuron_forward to leave X and w unchanged, but it modified them"


@pytest.mark.parametrize("X, w, why", [
    (np.ones((3, 2)), np.ones(3), "X has 2 features but w has 3 weights"),
    (np.ones((3, 4)), np.ones(3), "X has 4 features but w has 3 weights"),
    (np.ones(3), np.ones(2), "one sample with 3 features but w has 2 weights"),
], ids=["X-too-narrow", "X-too-wide", "one-sample-wrong-length"])
def test_neuron_forward_rejects_invalid_inputs(pcp, X, w, why):
    assert_raises_value_error(pcp.neuron_forward, X, w, why=why)


# ================================================================== Perceptron (10.21)
def random_binary(seed, n, p, labels=(0, 1)):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) + rng.normal(size=p)
    y = np.array(labels)[(X @ rng.normal(size=p) + 0.3 * rng.normal(size=n) > 0).astype(int)]
    if len(np.unique(y)) < 2:
        y[0], y[1] = labels
    return X, y


SKLEARN_CASES = [  # seed, n, p, eta0, max_iter, fit_intercept
    (0, 20, 2, 1.0, 100, True), (1, 50, 3, 0.1, 30, True), (2, 35, 5, 2.5, 10, True),
    (3, 60, 4, 1.0, 7, False), (4, 12, 1, 0.5, 50, True), (5, 80, 2, 0.01, 25, True),
]


@pytest.mark.parametrize("seed, n, p, eta0, max_iter, fit_intercept", SKLEARN_CASES,
                         ids=[f"n{n}-p{p}-eta{e}-epochs{m}" + ("" if f else "-no-intercept")
                              for _, n, p, e, m, f in SKLEARN_CASES])
def test_perceptron_matches_sklearn(pcp, seed, n, p, eta0, max_iter, fit_intercept):
    X, y = random_binary(seed, n, p)
    model = fit(pcp.Perceptron(eta0=eta0, max_iter=max_iter, fit_intercept=fit_intercept), X, y)
    oracle = sklearn_perceptron(X, y, eta0, max_iter, fit_intercept)
    data = (f"(eta0={eta0}, max_iter={max_iter}, fit_intercept={fit_intercept}; weights and bias start at 0, "
            "samples in order, update when y·(x·w + b) <= 0)")
    assert_close(learned(model, "coef_"), oracle.coef_[0], msg="coef_ = scikit-learn's coef_[0]", data=data)
    assert_close(learned(model, "intercept_"), oracle.intercept_[0], msg="intercept_ = scikit-learn's intercept_[0]",
                 data=data)


def test_perceptron_docstring_example(pcp):
    model = fit(pcp.Perceptron(), GATES_X, GATES["AND"])
    assert_close(learned(model, "coef_"), [3.0, 2.0], atol=0, msg="AND gate: coef_ (docstring example)")
    assert_close(learned(model, "intercept_"), -4.0, atol=0, msg="AND gate: intercept_ (docstring example)")
    if repr(model.intercept_) != "-4.0":
        raise AssertionError(f"expected clf.intercept_ to display -4.0, as in the docstring (a Python float: "
                             f"float(b)), got {model.intercept_!r} (a NumPy scalar displays differently, and the "
                             "doctest of the docstring would fail)")
    errors = learned(model, "errors_")
    if list(errors) != [2, 3, 3, 2, 2, 3, 2, 1, 0]:
        raise AssertionError(f"expected errors_ == [2, 3, 3, 2, 2, 3, 2, 1, 0] on the AND gate (docstring "
                             f"example: the number of updates in each epoch, then stop), got {errors}")
    assert_close(call(model.predict, GATES_X), [0, 0, 0, 1], atol=0, msg="AND gate: predict(X)")


@pytest.mark.parametrize("gate", ["AND", "OR", "NAND"])
def test_perceptron_learns_the_separable_gates(pcp, gate):
    y = GATES[gate]
    model = fit(pcp.Perceptron(), GATES_X, y)
    errors = list(learned(model, "errors_"))
    if not errors or errors[-1] != 0:
        raise AssertionError(f"expected the last epoch to have 0 mistakes on {gate} (linearly separable: the "
                             f"perceptron converges), got errors_ = {errors}")
    assert_close(call(model.predict, GATES_X), y, atol=0, msg=f"{gate}: the four predictions")
    assert_close(learned(model, "coef_"), sklearn_perceptron(GATES_X, y).coef_[0], atol=0,
                 msg=f"{gate}: the same weights as scikit-learn")


def test_perceptron_never_converges_on_xor(pcp):
    model = fit(pcp.Perceptron(max_iter=40), GATES_X, GATES["XOR"])
    errors = list(learned(model, "errors_"))
    if len(errors) != 40 or 0 in errors:
        raise AssertionError(f"expected 40 epochs, all with at least one mistake, on XOR (not linearly "
                             f"separable: training must run until max_iter), got errors_ = {errors}")
    if learned(model, "n_iter_") != 40:
        raise AssertionError(f"expected n_iter_ == 40 (max_iter reached on XOR), got {model.n_iter_}")
    score = call(model.score, GATES_X, GATES["XOR"])
    if not score < 1.0:
        raise AssertionError(f"expected an accuracy below 1 on XOR (no straight line separates it), got {score}")


def test_perceptron_stops_after_the_first_epoch_without_mistake(pcp):
    for seed in range(4):
        X, y, _, _ = separable_data(seed, n=30, margin=0.2)
        model = fit(pcp.Perceptron(max_iter=1000), X, y)
        errors = list(learned(model, "errors_"))
        if not errors or errors[-1] != 0 or 0 in errors[:-1]:
            raise AssertionError(f"expected errors_ to end with its FIRST 0 (stop after an epoch without "
                                 f"mistake) on separable data (seed {seed}), got {errors}")
        if learned(model, "n_iter_") != len(errors):
            raise AssertionError(f"expected n_iter_ == len(errors_) = {len(errors)}, got {model.n_iter_}")


def test_perceptron_counts_a_sum_of_zero_as_a_mistake(pcp):
    X, y = np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1])
    model = fit(pcp.Perceptron(max_iter=5), X, y)
    errors = list(learned(model, "errors_"))
    if not errors or errors[0] == 0:
        raise AssertionError(f"expected at least one update in the first epoch, got errors_ = {errors}: with "
                             "w = 0 and b = 0, y·(x·w + b) = 0 for the first sample, which counts as a mistake "
                             "(test y·z <= 0, not < 0; otherwise the weights never leave 0)")
    assert_close(learned(model, "coef_"), sklearn_perceptron(X, y, max_iter=5).coef_[0], atol=0,
                 msg="the same weights as scikit-learn")


def test_perceptron_corrects_a_sum_of_zero_whatever_the_class(pcp):
    X, y = np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1])
    errors = list(learned(fit(pcp.Perceptron(max_iter=5), X, y), "errors_"))
    if errors != [2, 0]:
        raise AssertionError(
            f"expected errors_ == [2, 0], got {errors}: in the first epoch, the first sample (class 0, target −1) "
            "has z = 0, a mistake since y·z <= 0, then the second one is misclassified; the second epoch has no "
            "mistake. Test y·z <= 0 with y in {−1, +1}, not « the prediction differs from y »: with z = 0 the "
            "prediction is the first class, so a sample of the first class with z = 0 would never be corrected")


def test_perceptron_learning_rate_eta0(pcp):
    X, y = random_binary(21, 40, 3)
    base = fit(pcp.Perceptron(eta0=1.0, max_iter=15), X, y)
    for eta0 in (0.1, 3.0):
        other = fit(pcp.Perceptron(eta0=eta0, max_iter=15), X, y)
        assert_close(learned(other, "coef_"), eta0 * np.asarray(learned(base, "coef_")), rtol=1e-9, atol=1e-12,
                     msg=f"eta0={eta0}: every update is multiplied by eta0 and the weights start at 0, so the "
                         f"weights are eta0 times those of eta0=1")
        if list(learned(other, "errors_")) != list(learned(base, "errors_")):
            raise AssertionError(f"expected the same errors_ for eta0={eta0} as for eta0=1 (same signs of x·w + b, "
                                 f"so the same mistakes), got {other.errors_} instead of {base.errors_}")


def test_perceptron_fit_intercept_false_keeps_the_bias_at_zero(pcp):
    X, y, _, _ = separable_data(7, n=30, through_origin=True)
    model = fit(pcp.Perceptron(fit_intercept=False), X, y)
    intercept = learned(model, "intercept_")
    if intercept != 0.0:
        raise AssertionError(f"expected intercept_ == 0.0 with fit_intercept=False, got {intercept}")
    assert_close(learned(model, "coef_"), sklearn_perceptron(X, y, fit_intercept=False).coef_[0],
                 msg="fit_intercept=False: the same weights as scikit-learn")


def test_perceptron_attributes_have_the_documented_types(pcp):
    X, y = random_binary(8, 25, 3)
    model = fit(pcp.Perceptron(max_iter=12), X, y)
    coef = learned(model, "coef_")
    if not isinstance(coef, np.ndarray) or coef.shape != (3,):
        raise AssertionError(f"expected coef_ to be an array of shape (3,) (one weight per feature; "
                             f"scikit-learn's shape (1, 3) is not the one asked here), got "
                             f"{type(coef).__name__} of shape {np.shape(coef)}")
    intercept = learned(model, "intercept_")
    if isinstance(intercept, (bool, np.bool_)) or not isinstance(intercept, float):
        raise AssertionError(f"expected intercept_ to be a float, got {type(intercept).__name__}"
                             + (f" of shape {np.shape(intercept)}" if isinstance(intercept, np.ndarray) else "")
                             + " (use float(b))")
    errors = learned(model, "errors_")
    if not isinstance(errors, list) or not all(type(e) is int for e in errors):
        raise AssertionError(f"expected errors_ to be a list of Python int, got {type(errors).__name__} of "
                             f"{sorted({type(e).__name__ for e in errors})} (count with a Python int, "
                             "or use int(...))")
    n_iter = learned(model, "n_iter_")
    if isinstance(n_iter, bool) or not isinstance(n_iter, (int, np.integer)) or n_iter != len(errors):
        raise AssertionError(f"expected n_iter_ to be the int len(errors_) = {len(errors)}, got {n_iter!r}")
    classes = learned(model, "classes_")
    if not np.array_equal(np.asarray(classes), [0, 1]):
        raise AssertionError(f"expected classes_ == [0, 1] (the sorted labels), got {classes}")


@pytest.mark.parametrize("labels", [("cat", "dog"), (-1, 1), (2, 7)], ids=["strings", "minus-one-plus-one", "2-and-7"])
def test_perceptron_accepts_any_two_labels(pcp, labels):
    X, y01 = random_binary(9, 40, 2)
    y = np.array(labels)[y01]
    model = fit(pcp.Perceptron(max_iter=20), X, y)
    classes = np.asarray(learned(model, "classes_"))
    if classes.tolist() != sorted(labels):
        raise AssertionError(f"expected classes_ == {sorted(labels)} (the sorted labels), got {classes.tolist()}")
    oracle = sklearn_perceptron(X, y, max_iter=20)
    assert_close(learned(model, "coef_"), oracle.coef_[0],
                 msg=f"labels {labels}: classes_[0] = {sorted(labels)[0]!r} is coded -1, "
                     f"classes_[1] = {sorted(labels)[1]!r} is coded +1")
    predictions = np.asarray(call(model.predict, X))
    if predictions.tolist() != oracle.predict(X).tolist():
        raise AssertionError(f"expected predict to return the labels {sorted(labels)} like scikit-learn, got "
                             f"{predictions[:6].tolist()}... instead of {oracle.predict(X)[:6].tolist()}...")


def test_perceptron_decision_function_and_predict(pcp):
    X, y = random_binary(11, 30, 3)
    model = fit(pcp.Perceptron(max_iter=10), X, y)
    X_new = np.random.default_rng(12).normal(size=(9, 3))
    coef, intercept = np.asarray(learned(model, "coef_"), dtype=float), float(learned(model, "intercept_"))
    scores = call(model.decision_function, X_new, expected="an array of shape (9,)")
    assert_close(scores, X_new @ coef + intercept, msg="decision_function(X) = X @ coef_ + intercept_")
    oracle = sklearn_perceptron(X, y, max_iter=10)
    assert_close(scores, oracle.decision_function(X_new), msg="the same scores as scikit-learn")
    assert_close(call(model.predict, X_new), np.where(X_new @ coef + intercept > 0, 1, 0), atol=0,
                 msg="predict: classes_[1] where the score is > 0, classes_[0] elsewhere")


def test_perceptron_predicts_the_first_class_on_the_boundary(pcp):
    X, y = np.array([[1.0, 0.0], [0.0, 1.0]]), np.array(["no", "yes"])
    model = fit(pcp.Perceptron(max_iter=5), X, y)
    assert_close(learned(model, "coef_"), [-1.0, 1.0], atol=0, msg="coef_ after training on [1, 0] -> 'no', "
                                                                 "[0, 1] -> 'yes' (as scikit-learn)")
    assert_close(learned(model, "intercept_"), 0.0, atol=0, msg="intercept_ after this training (as scikit-learn)")
    got = call(model.predict, np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 0.5]]))
    if np.asarray(got).tolist() != ["no", "no", "yes"]:
        raise AssertionError(f"expected ['no', 'no', 'yes']: the scores are 0, 0 and 0.5, and the positive class "
                             f"classes_[1] needs a score > 0, strictly; got {np.asarray(got).tolist()}")


def test_perceptron_score_is_the_accuracy(pcp):
    X, y = random_binary(13, 50, 2)
    model = fit(pcp.Perceptron(max_iter=3), X, y)
    oracle = sklearn_perceptron(X, y, max_iter=3)
    score = call(model.score, X, y, expected="a float")
    if isinstance(score, (bool, np.bool_)) or not isinstance(score, float):
        raise AssertionError(f"expected score to return a float, got {type(score).__name__} (use float(...))")
    assert_close(score, oracle.score(X, y), msg="score(X, y) = fraction of correct predictions")


def test_perceptron_fit_returns_self(pcp):
    model = pcp.Perceptron(max_iter=5)
    result = call(model.fit, GATES_X, GATES["OR"], expected="fit to return self")
    if result is not model:
        raise AssertionError(f"expected fit to return self (the fitted perceptron, so that "
                             f"Perceptron().fit(X, y).predict(X) works), got {type(result).__name__}")


def test_perceptron_refit_starts_from_zero(pcp):
    model = fit(pcp.Perceptron(max_iter=30), GATES_X, GATES["NAND"])
    fit(model, GATES_X, GATES["AND"])
    assert_close(learned(model, "coef_"), [3.0, 2.0], atol=0,
                 msg="a second fit (AND after NAND) must start again from w = 0, b = 0: the AND weights of the "
                     "docstring")
    errors = list(learned(model, "errors_"))
    if errors != [2, 3, 3, 2, 2, 3, 2, 1, 0]:
        raise AssertionError(f"expected errors_ == [2, 3, 3, 2, 2, 3, 2, 1, 0] after the second fit (a new list, "
                             f"not appended to the first one), got {errors}")


def test_perceptron_does_not_modify_its_inputs(pcp):
    X, y = random_binary(15, 20, 2)
    keep = (X.copy(), y.copy())
    model = pcp.Perceptron(max_iter=10, shuffle=True, random_state=0)
    call(model.fit, X, y, expected="fit to succeed", copy_inputs=False)
    assert np.array_equal(X, keep[0]) and np.array_equal(y, keep[1]), \
        "expected fit to leave X and y unchanged (shuffle the ORDER of visit, never the arrays), but it modified them"


def test_perceptron_shuffle_is_reproducible_and_changes_the_order(pcp):
    X, y, _, _ = separable_data(16, n=40, margin=0.05)
    first = fit(pcp.Perceptron(shuffle=True, random_state=3, max_iter=500), X, y)
    again = fit(pcp.Perceptron(shuffle=True, random_state=3, max_iter=500), X, y)
    assert_close(learned(again, "coef_"), learned(first, "coef_"), atol=0,
                 msg="the same random_state must give the same weights (create the generator in fit)")
    refit = fit(first, X, y)
    assert_close(learned(refit, "coef_"), learned(again, "coef_"), atol=0,
                 msg="fitting the same perceptron twice must give the same weights: create "
                     "np.random.default_rng(random_state) in fit, not in __init__")
    if learned(first, "errors_")[-1] != 0:
        raise AssertionError(f"expected the shuffled perceptron to converge on separable data, got errors_ ending "
                             f"with {first.errors_[-5:]}")
    in_order = np.asarray(learned(fit(pcp.Perceptron(max_iter=500), X, y), "coef_"), dtype=float)
    shuffled = [np.asarray(learned(fit(pcp.Perceptron(shuffle=True, random_state=seed, max_iter=500), X, y), "coef_"),
                           dtype=float) for seed in range(5)]
    if all(coef.shape == in_order.shape and np.allclose(coef, in_order) for coef in shuffled):
        raise AssertionError("expected shuffle=True to change the order of visit (and so the weights) for at "
                             "least one of the seeds 0 to 4, but every shuffled fit gave the weights of shuffle=False")


def test_perceptron_shuffle_draws_a_new_order_at_each_epoch(pcp):
    runs = {seed: list(learned(fit(pcp.Perceptron(shuffle=True, random_state=seed, max_iter=30), GATES_X,
                                   GATES["XOR"]), "errors_")) for seed in range(5)}
    if all(len(set(errors[1:])) <= 1 for errors in runs.values()):
        _fail("", "expected shuffle=True to draw a NEW order of visit at each epoch: on XOR, any fixed order (even a "
                  "random one drawn once, or the same seed reused at every epoch) gives the same number of mistakes "
                  "at every epoch after the first, and this happened for every seed from 0 to 4. Call "
                  "rng.permutation(n) inside the loop over the epochs, with ONE generator created before it",
              f"errors_ for seed 0: {runs[0][:12]}")


def test_perceptron_respects_the_novikoff_bound(pcp):
    for seed in range(5):
        X, y, w_star, _ = separable_data(seed + 30, n=50, margin=0.25, through_origin=True)
        model = fit(pcp.Perceptron(fit_intercept=False, max_iter=10_000), X, y)
        errors = list(learned(model, "errors_"))
        signs = np.where(y == 1, 1.0, -1.0)
        gamma = np.min(signs * (X @ w_star))              # margin of a separator of norm 1
        bound = (np.max(np.linalg.norm(X, axis=1)) / gamma) ** 2
        if errors[-1] != 0 or sum(errors) > bound:
            raise AssertionError(f"expected convergence with at most (R/γ)² = {bound:.1f} updates in total "
                                 f"(Novikoff's theorem, seed {seed}), got {sum(errors)} updates, errors_ = "
                                 f"{errors[:8]}{'...' if len(errors) > 8 else ''}")


def test_perceptron_max_iter_limits_the_epochs(pcp):
    model = fit(pcp.Perceptron(max_iter=7), GATES_X, GATES["XOR"])
    if len(learned(model, "errors_")) != 7 or learned(model, "n_iter_") != 7:
        raise AssertionError(f"expected 7 epochs (max_iter=7) on XOR, got n_iter_ = {model.n_iter_} and "
                             f"errors_ = {model.errors_}")


@pytest.mark.parametrize("params, X, y, why", [
    ({}, GATES_X, np.array([1, 1, 1, 1]), "a single class in y"),
    ({}, GATES_X, np.array([0, 1, 2, 1]), "three classes in y (use OneVsRestClassifier)"),
    ({"eta0": 0.0}, GATES_X, GATES["AND"], "eta0 = 0"),
    ({"eta0": -1.0}, GATES_X, GATES["AND"], "eta0 < 0"),
    ({"max_iter": 0}, GATES_X, GATES["AND"], "max_iter = 0"),
    ({}, GATES_X, np.array([0, 1, 1]), "4 samples but 3 labels"),
], ids=["one-class", "three-classes", "eta0-zero", "eta0-negative", "max-iter-zero", "lengths-differ"])
def test_perceptron_rejects_invalid_inputs(pcp, params, X, y, why):
    assert_raises_value_error(pcp.Perceptron(**params).fit, X, y, why=why)
