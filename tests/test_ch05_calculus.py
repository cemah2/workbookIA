"""Tests of mylearn.calculus (chapter 5): oracle tests and properties.

    pytest tests/test_ch05_calculus.py              # your code (mon_travail/mylearn/calculus.py)
    pytest tests/test_ch05_calculus.py --impl=ref   # the reference

Oracles: analytic derivatives written here, PyTorch (``torch.autograd.grad`` and
``torch.optim.SGD`` in double precision), ``wb.synth.rosenbrock_grad``,
``scipy.signal.argrelextrema`` and the eigenvalues of Hessian matrices
(``np.linalg.eigvalsh``). Every test name starts with the name of the function it
tests (``-k "test_numerical_gradient_"``); the tests of one exercise never call the
functions of another one (``classify_critical_point`` may reuse ``numerical_gradient``
inside the module, as its docstring says).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import numpy as np
import pytest
from scipy import signal

try:
    import torch
except ImportError:      # PyTorch is optional on a computer (Colab has it): its oracle tests are skipped
    torch = None

from wb import synth


needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")


@pytest.fixture
def calc(mylearn_module):
    return mylearn_module("calculus")


# ------------------------------------------------------------------ helpers
def listed(values) -> list:
    """Plain Python values for the messages (NumPy 2 would print np.float64(0.5))."""
    return np.asarray(values).tolist()


def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for the first line of a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    if arr.dtype.kind in "biuf":
        text = np.array2string(arr, precision=digits, separator=", ", threshold=12, max_line_width=10**6)
    else:
        text = repr(arr.tolist())
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str, details: str):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data and details."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    if details:
        lines.append(details)
    raise AssertionError("\n".join(lines)) from None


def _shape_head(got, expected):
    """When two arrays have different shapes, say so first: truncated views of them can look identical."""
    want = np.asarray(expected)
    if got.ndim != want.ndim or got.shape != want.shape:
        return (f"expected shape {want.shape}, got shape {got.shape}: "
                f"expected {_short(expected)}, got {_short(got)}")
    return None


def _first_difference(got, want, rtol, atol) -> str:
    """' (first difference at index i: expected a, got b)' for two arrays of the same shape, '' otherwise."""
    if got.ndim == 0 or got.shape != want.shape:
        return ""
    bad = np.flatnonzero(~np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=False))
    if bad.size == 0:
        return ""
    i = int(bad[0])
    index = np.unravel_index(i, got.shape)
    where = index[0] if got.ndim == 1 else tuple(int(k) for k in index)
    return f" (first difference at index {where}: expected {want.ravel()[i]:.6g}, got {got.ravel()[i]:.6g})"


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg="", data=""):
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected {_short(expected)}, got an object of type {type(result).__name__}", data, "")
    head = _shape_head(got, expected)
    if head:
        _fail(msg, head, data, "")
    want = np.asarray(expected, dtype=float)
    if np.isnan(got).any() and not np.isnan(want).any():
        _fail(msg, f"expected {_short(expected)}, got NaN values: {_short(result)}", data, "")
    try:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    except AssertionError as exc:
        have, need = _short(result), _short(expected)
        if have == need:   # they differ beyond the 6th digit: show every digit
            have, need = _short(result, 17), _short(expected, 17)
        _fail(msg, f"expected {need}, got {have}{_first_difference(got, want, rtol, atol)}", data, str(exc))


def assert_equal_ints(result, expected, msg="", data=""):
    """Integer arrays (indices): same shape, same values, an integer dtype."""
    got = np.asarray(result)
    head = _shape_head(got, np.asarray(expected))
    if head:
        _fail(msg, head, data, "")
    if got.size and got.dtype.kind not in "iu":
        _fail(msg, f"expected integer indices {_short(expected)}, got values of dtype {got.dtype}: {_short(got)}",
              data, "")
    if not np.array_equal(got, expected):
        _fail(msg, f"expected {_short(expected)}, got {_short(got)}", data, "")


def assert_python_float(value, name):
    assert isinstance(value, float), f"{name} must return a Python float for a scalar x, got {type(value).__name__}"


def assert_float_array(value, name, shape):
    assert isinstance(value, np.ndarray), f"{name} must return a NumPy array, got {type(value).__name__}"
    assert value.shape == shape, f"{name} must return an array of shape {shape} (the shape of x), got {value.shape}"
    assert value.dtype.kind == "f", f"{name} must return a float array, got dtype {value.dtype}"


def assert_raises_value_error(function, *args, why="", **kwargs):
    """The call must raise ValueError (or a subclass); `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
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


def torch_derivative(f_torch, x: float) -> float:
    """df/dx at x with torch.autograd.grad, in double precision (oracle)."""
    t = torch.tensor(float(x), dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(f_torch(t), t)
    return float(grad)


def torch_gradient(f_torch, x) -> np.ndarray:
    """The gradient of a scalar function of an array, with torch.autograd.grad in double precision (oracle)."""
    t = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(f_torch(t), t)
    return grad.numpy()


# Functions with their analytic derivatives (written by hand: the oracle of the 1-D tests)
ANALYTIC = {
    "sin": (np.sin, np.cos, lambda x: -np.sin(x)),
    "exp": (np.exp, np.exp, np.exp),
    "square": (lambda x: x ** 2, lambda x: 2 * x, lambda x: 2 + 0 * x),
    "cubic": (lambda x: x ** 3 - 2 * x, lambda x: 3 * x ** 2 - 2, lambda x: 6 * x),
    "inverse": (lambda x: 1 / x, lambda x: -1 / x ** 2, lambda x: 2 / x ** 3),
}
POINTS = [-1.7, -0.4, 0.5, 1.0, 2.3]


# ------------------------------------------------------------------ numerical_derivative (5.11)
@pytest.mark.parametrize("name", list(ANALYTIC))
def test_numerical_derivative_matches_the_analytic_derivative(calc, name):
    f, df, _ = ANALYTIC[name]
    for x in POINTS:
        assert_close(calc.numerical_derivative(f, x), df(x), rtol=1e-7, atol=1e-8,
                     msg=f"the central difference of {name} must approach its derivative (h = 1e-5)",
                     data=f"x={x}")


@needs_torch
def test_numerical_derivative_matches_torch_autograd(calc):
    def f_numpy(x):
        return x * np.tanh(x) + np.exp(-x ** 2)

    def f_torch(t):
        return t * torch.tanh(t) + torch.exp(-t ** 2)

    for x in np.random.default_rng(5).uniform(-3, 3, 12):
        assert_close(calc.numerical_derivative(f_numpy, float(x)), torch_derivative(f_torch, x), rtol=1e-7,
                     atol=1e-8, msg="the central difference must agree with torch.autograd", data=f"x={x!r}")


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_numerical_derivative_follows_the_formula_of_each_method(calc, method):
    f = ANALYTIC["cubic"][0]
    formulas = {"central": lambda x, h: (f(x + h) - f(x - h)) / (2 * h),
                "forward": lambda x, h: (f(x + h) - f(x)) / h,
                "backward": lambda x, h: (f(x) - f(x - h)) / h}
    for x, h in [(1.0, 0.1), (-0.5, 0.25), (2.0, 1e-3)]:
        assert_close(calc.numerical_derivative(f, x, h=h, method=method), formulas[method](x, h), rtol=1e-12,
                     atol=1e-12, msg=f"method={method!r} must apply its own quotient (see the docstring)",
                     data=f"f(x) = x**3 - 2x, x={x}, h={h}")


def test_numerical_derivative_default_method_is_central(calc):
    # for x**2 the central quotient is exact for every h, the forward one is 2x + h
    assert_close(calc.numerical_derivative(lambda x: x ** 2, 3.0, h=0.1), 6.0, rtol=1e-12,
                 msg="without the method argument, use the central difference (f(x + h) - f(x - h)) / (2h)")


def test_numerical_derivative_scalar_x_gives_a_python_float(calc):
    for x in [2.0, 2, np.float64(2.0)]:
        result = calc.numerical_derivative(lambda v: v ** 2, x)
        assert_python_float(result, "numerical_derivative")
        assert_close(result, 4.0, rtol=1e-8, msg=f"the slope of x**2 at x = {x!r} (a {type(x).__name__})")


@pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 2, 2)], ids=["1-D", "2-D", "3-D"])
def test_numerical_derivative_array_keeps_the_shape_of_x(calc, shape):
    x = np.linspace(-2, 2, int(np.prod(shape))).reshape(shape)
    result = calc.numerical_derivative(np.sin, x)
    assert_float_array(result, "numerical_derivative", shape)
    assert_close(result, np.cos(x), rtol=1e-7, atol=1e-8, msg="one slope per point of the array, element by element")


def test_numerical_derivative_accepts_a_list(calc):
    result = calc.numerical_derivative(np.exp, [0.0, 1.0])
    assert_float_array(result, "numerical_derivative", (2,))
    assert_close(result, np.exp([0.0, 1.0]), rtol=1e-7, msg="a list of points works like an array (np.asarray)")


def test_numerical_derivative_does_not_modify_the_callers_array(calc):
    x = np.array([0.1, 0.7, 1.3])
    saved = x.copy()
    calc.numerical_derivative(np.sin, x)
    assert np.array_equal(x, saved), f"the caller's x must not change: before {listed(saved)}, after {listed(x)}"


def test_numerical_derivative_central_error_shrinks_like_h_squared(calc):
    x = 0.5
    errors = {method: [abs(calc.numerical_derivative(np.exp, x, h=h, method=method) - np.exp(x))
                       for h in (1e-2, 1e-3)] for method in ("central", "forward")}
    central_ratio = errors["central"][0] / errors["central"][1]
    forward_ratio = errors["forward"][0] / errors["forward"][1]
    assert 70 < central_ratio < 130, (f"dividing h by 10 should divide the error of the central difference by about "
                                      f"100, got a factor {central_ratio:.3g}")
    assert 7 < forward_ratio < 13, (f"dividing h by 10 should divide the error of the forward difference by about 10, "
                                    f"got a factor {forward_ratio:.3g}")


@pytest.mark.parametrize("h", [0.0, -1e-5, -1.0], ids=["zero", "negative-small", "negative"])
def test_numerical_derivative_rejects_a_step_that_is_not_positive(calc, h):
    assert_raises_value_error(calc.numerical_derivative, np.sin, 1.0, h=h, why=f"h = {h} (the step must be > 0)")


@pytest.mark.parametrize("method", ["middle", "symmetric", ""], ids=["middle", "symmetric", "empty"])
def test_numerical_derivative_rejects_an_unknown_method(calc, method):
    assert_raises_value_error(calc.numerical_derivative, np.sin, 1.0, method=method,
                              why=f"method={method!r} is not 'central', 'forward' or 'backward'")


# ------------------------------------------------------------------ second_derivative (5.11)
@pytest.mark.parametrize("name", list(ANALYTIC))
def test_second_derivative_matches_the_analytic_second_derivative(calc, name):
    f, _, d2f = ANALYTIC[name]
    for x in POINTS:
        assert_close(calc.second_derivative(f, x), d2f(x), rtol=1e-5, atol=1e-5,
                     msg=f"the second difference of {name} must approach its second derivative (h = 1e-4)",
                     data=f"x={x}")


def test_second_derivative_follows_its_formula(calc):
    f = np.sin
    for x, h in [(1.0, 0.1), (-0.3, 0.5)]:
        expected = (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2
        assert_close(calc.second_derivative(f, x, h=h), expected, rtol=1e-12, atol=1e-12,
                     msg="apply (f(x + h) - 2 f(x) + f(x - h)) / h**2 with the given h", data=f"x={x}, h={h}")


def test_second_derivative_sign_tells_valley_from_hilltop(calc):
    valley = calc.second_derivative(lambda x: (x - 1) ** 2, 1.0)
    hilltop = calc.second_derivative(lambda x: 3 - (x + 2) ** 2, -2.0)
    assert valley > 0, f"at the bottom of a valley the second derivative is positive (expected 2), got {valley!r}"
    assert hilltop < 0, f"on a hilltop the second derivative is negative (expected -2), got {hilltop!r}"


def test_second_derivative_scalar_and_array(calc):
    result = calc.second_derivative(lambda x: x ** 3, 2)
    assert_python_float(result, "second_derivative")
    assert_close(result, 12.0, rtol=1e-6, msg="the second derivative of x**3 at x = 2 (an int)")
    x = np.array([[0.5, -1.0], [2.0, 3.0]])
    result = calc.second_derivative(lambda v: v ** 3, x)
    assert_float_array(result, "second_derivative", (2, 2))
    assert_close(result, 6 * x, rtol=1e-6, atol=1e-6, msg="one second derivative per point, element by element")


@pytest.mark.parametrize("h", [0.0, -1e-4], ids=["zero", "negative"])
def test_second_derivative_rejects_a_step_that_is_not_positive(calc, h):
    assert_raises_value_error(calc.second_derivative, np.sin, 1.0, h=h, why=f"h = {h} (the step must be > 0)")


# ------------------------------------------------------------------ numerical_gradient (5.15)
def rosenbrock_vec(v):
    return synth.rosenbrock(v[0], v[1])


@pytest.mark.parametrize("point", [(-1.5, 2.0), (0.0, 0.0), (1.0, 1.0), (0.3, -0.7), (2.0, 4.1)],
                         ids=lambda p: f"({p[0]}, {p[1]})")
def test_numerical_gradient_matches_rosenbrock_grad(calc, point):
    expected = np.array(synth.rosenbrock_grad(*point))
    result = calc.numerical_gradient(rosenbrock_vec, np.array(point))
    assert_float_array(result, "numerical_gradient", (2,))
    assert_close(result, expected, rtol=1e-6, atol=1e-5,
                 msg="the numerical gradient of Rosenbrock must match wb.synth.rosenbrock_grad",
                 data=f"point={point}")


@pytest.mark.parametrize("shape", [(3,), (3, 4), (2, 2, 3)], ids=["1-D", "2-D", "3-D"])
@needs_torch
def test_numerical_gradient_matches_torch_autograd(calc, shape):
    rng = np.random.default_rng(len(shape))
    a = rng.normal(size=shape)
    x = rng.normal(size=shape)

    def f_numpy(w):
        return float(np.sum(np.sin(w) * a) + np.sqrt(np.sum(w ** 2) + 1))

    def f_torch(t):
        return torch.sum(torch.sin(t) * torch.tensor(a)) + torch.sqrt(torch.sum(t ** 2) + 1)

    result = calc.numerical_gradient(f_numpy, x)
    assert_float_array(result, "numerical_gradient", shape)
    assert_close(result, torch_gradient(f_torch, x), rtol=1e-6, atol=1e-8,
                 msg=f"the numerical gradient of a function of an array of shape {shape} must match torch.autograd "
                     "(one central difference per entry, whatever the shape)")


def test_numerical_gradient_follows_the_central_formula(calc):
    def f(v):
        return v[0] ** 3 + v[0] * v[1] ** 2

    x, h = np.array([1.0, 2.0]), 0.1
    expected = [(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(2)]
    assert_close(calc.numerical_gradient(f, x, h=h), expected, rtol=1e-12, atol=1e-12,
                 msg="entry i is (f(x + h e_i) - f(x - h e_i)) / (2h), with the given h", data="x=[1.0, 2.0], h=0.1")


def test_numerical_gradient_never_modifies_the_callers_x(calc):
    x = np.array([0.1, 1e-3, 123.456, -7.77])
    saved = x.copy()
    calc.numerical_gradient(lambda v: float(np.sum(v ** 3)), x)
    assert np.array_equal(x, saved), (f"the caller's x must stay exactly the same (work on a float copy): "
                                      f"before {listed(saved)}, after {listed(x)}")


def test_numerical_gradient_does_not_touch_the_callers_x_even_during_the_calls(calc):
    x = np.array([0.5, -1.25, 2.0])
    saved = x.copy()
    changed = []

    def f(w):
        if not np.array_equal(x, saved):     # f looks at the CALLER's array while the gradient is computed
            changed.append(listed(x))
        return float(np.sum(w ** 2))

    calc.numerical_gradient(f, x)
    assert not changed, (f"the caller's x was modified while f was called (seen: {changed[0]}): nudge the "
                         "coordinates of a float COPY (np.array(x, dtype=float)), never of the caller's array")


def test_numerical_gradient_accepts_an_integer_point(calc):
    x = np.array([-1, 1])                   # integers: x + h must not be rounded back to an integer
    result = calc.numerical_gradient(rosenbrock_vec, x)
    assert_close(result, [-4.0, 0.0], rtol=1e-6, atol=1e-5,
                 msg="for an integer point, work on a FLOAT copy (np.array(x, dtype=float)): the gradient of "
                     "Rosenbrock at (-1, 1)")
    assert np.array_equal(x, [-1, 1]), f"the caller's integer x must not change, got {listed(x)}"


def test_numerical_gradient_accepts_a_list(calc):
    result = calc.numerical_gradient(rosenbrock_vec, [0.5, 1.0])
    assert_float_array(result, "numerical_gradient", (2,))
    assert_close(result, synth.rosenbrock_grad(0.5, 1.0), rtol=1e-6, atol=1e-5,
                 msg="a list works like an array: the gradient of Rosenbrock at (0.5, 1.0)")


def test_numerical_gradient_calls_f_with_float_arrays_shaped_like_x(calc):
    seen = []

    def f(w):
        seen.append((type(w).__name__, np.shape(w), np.asarray(w).dtype.kind))
        return float(np.sum(np.asarray(w) ** 2))

    calc.numerical_gradient(f, [[1, 2], [3, 4]])
    bad = [s for s in seen if s != ("ndarray", (2, 2), "f")]
    assert seen and not bad, ("f must receive float NumPy arrays shaped like x (here (2, 2)), got "
                              f"{bad[0] if bad else 'no call'}")


@pytest.mark.parametrize("h", [0.0, -1e-5], ids=["zero", "negative"])
def test_numerical_gradient_rejects_a_step_that_is_not_positive(calc, h):
    assert_raises_value_error(calc.numerical_gradient, rosenbrock_vec, [0.0, 0.0], h=h,
                              why=f"h = {h} (the step must be > 0)")


# ------------------------------------------------------------------ gradient_descent (5.18)
def sgd_oracle(grad, x0, lr, n_steps, maximize=False):
    """The path of torch.optim.SGD (plain SGD, double precision) fed with the same gradients (oracle).

    Without PyTorch, the same update x <- x -/+ lr * grad(x), written with NumPy.
    """
    if torch is None:
        x = np.asarray(x0, dtype=float)
        path = [x.copy()]
        for _ in range(n_steps):
            x = x + (1 if maximize else -1) * lr * np.asarray(grad(x.copy()), dtype=float)
            path.append(x.copy())
        return np.array(path)
    p = torch.tensor(np.asarray(x0, dtype=float), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([p], lr=lr, maximize=maximize)
    path = [p.detach().numpy().copy()]
    for _ in range(n_steps):
        optimizer.zero_grad()
        p.grad = torch.tensor(np.asarray(grad(p.detach().numpy().copy()), dtype=float), dtype=torch.float64)
        optimizer.step()
        path.append(p.detach().numpy().copy())
    return np.array(path)


def bowl_grad(v):
    """Gradient of (x - 2)**2 + 3 (y + 1)**2."""
    return np.array([2 * (v[0] - 2), 6 * (v[1] + 1)])


def rosenbrock_grad_vec(v):
    return np.array(synth.rosenbrock_grad(v[0], v[1]))


@pytest.mark.parametrize("grad, x0, lr, n_steps, maximize", [
    pytest.param(bowl_grad, [0.0, 0.0], 0.1, 25, False, id="bowl"),
    pytest.param(rosenbrock_grad_vec, [-1.5, 2.0], 0.001, 40, False, id="rosenbrock"),
    pytest.param(lambda v: -bowl_grad(v), [3.0, 1.0], 0.05, 30, True, id="ascent-on-a-hill"),
    pytest.param(lambda w: 2 * w - 1, [[1.0, -2.0], [0.5, 3.0]], 0.2, 15, False, id="2-D-point"),
])
def test_gradient_descent_matches_torch_sgd(calc, grad, x0, lr, n_steps, maximize):
    x_final, path = calc.gradient_descent(grad, x0, lr=lr, n_steps=n_steps, maximize=maximize)
    expected = sgd_oracle(grad, x0, lr, n_steps, maximize)
    assert_close(path, expected, rtol=1e-10, atol=1e-12,
                 msg=f"the path must match torch.optim.SGD(lr={lr}, maximize={maximize}) step by step "
                     "(x - lr * grad(x) to descend, x + lr * grad(x) to climb)", data=f"x0={x0}")
    assert_close(x_final, expected[-1], rtol=1e-10, atol=1e-12, msg="x_final is the last point of the path")


def test_gradient_descent_docstring_example(calc):
    x_final, path = calc.gradient_descent(lambda x: 2 * x, [1.0], lr=0.25, n_steps=3)
    assert_close(path, [[1.0], [0.5], [0.25], [0.125]], rtol=1e-12,
                 msg="three steps x <- x - 0.25 * 2x from 1.0: the path includes the start (shape (4, 1))")
    assert_close(x_final, [0.125], rtol=1e-12, msg="x_final has the shape of x0 (here (1,))")


def test_gradient_descent_path_starts_at_x0_and_has_one_row_per_step(calc):
    x0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    start = x0.copy()                   # what x0 was before the call (a function that changes x0 must not hide it)
    x_final, path = calc.gradient_descent(lambda w: w, x0, lr=0.5, n_steps=4)
    assert isinstance(path, np.ndarray) and path.shape == (5, 2, 2), (
        f"path must have shape (n_steps + 1, *x0.shape) = (5, 2, 2), got "
        f"{getattr(path, 'shape', type(path).__name__)}")
    assert_close(path[0], start, rtol=0, atol=0, msg="path[0] must be the starting point x0 (a copy of it)")
    assert_close(path[-1], x_final, rtol=0, atol=0, msg="path[-1] must be x_final")
    assert path.dtype.kind == "f" and x_final.dtype.kind == "f", "x_final and path must be float arrays"


def test_gradient_descent_tol_stops_before_moving(calc):
    _, path = calc.gradient_descent(lambda x: 2 * x, [1.0], lr=0.25, n_steps=100, tol=0.3)
    assert path.shape == (4, 1), (f"with tol=0.3 the loop stops when |grad| < 0.3 (at x = 0.125, after 3 steps): "
                                  f"expected a path of shape (4, 1), got {path.shape}")
    x_final, path = calc.gradient_descent(lambda x: 2 * x, [0.1], lr=0.25, n_steps=100, tol=0.3)
    assert path.shape == (1, 1), (f"when the gradient at x0 is already below tol, no step is taken: expected a path of "
                                  f"shape (1, 1), got {path.shape}")
    assert_close(x_final, [0.1], rtol=0, atol=0, msg="no step taken: x_final is x0")


def test_gradient_descent_tol_uses_the_euclidean_norm_of_all_entries(calc):
    # every entry is 0.6 (< 1), but the norm of the gradient is 1.2 (> 1): no early stop
    _, path = calc.gradient_descent(lambda w: np.full((2, 2), 0.6), np.zeros((2, 2)), lr=0.1, n_steps=5, tol=1.0)
    assert path.shape == (6, 2, 2), ("tol is compared with the Euclidean norm of the WHOLE gradient (np.linalg.norm), "
                                     f"here 1.2 > 1: expected 5 steps (a path of shape (6, 2, 2)), got {path.shape}")


def test_gradient_descent_tol_is_a_strict_inequality(calc):
    # the norm of [3, 4] is exactly 5: with tol=5 the loop must not stop (it stops only if norm < tol)
    _, path = calc.gradient_descent(lambda v: np.array([3.0, 4.0]), [0.0, 0.0], lr=0.01, n_steps=3, tol=5.0)
    assert path.shape == (4, 2), ("the loop stops only when the norm is strictly below tol (here norm = tol = 5): "
                                  f"expected 3 steps (a path of shape (4, 2)), got {path.shape}")


def test_gradient_descent_zero_steps_returns_the_start(calc):
    x_final, path = calc.gradient_descent(bowl_grad, [0.0, 0.0], lr=0.1, n_steps=0)
    assert path.shape == (1, 2), f"n_steps=0: the path only holds x0, expected shape (1, 2), got {path.shape}"
    assert_close(x_final, [0.0, 0.0], rtol=0, atol=0, msg="n_steps=0: x_final is x0")


def test_gradient_descent_evaluates_the_gradient_at_the_current_point(calc):
    calls = []

    def grad(v):
        calls.append(np.array(v, dtype=float))
        return bowl_grad(v)

    _, path = calc.gradient_descent(grad, [0.0, 0.0], lr=0.1, n_steps=6)
    assert len(calls) >= 6, f"the gradient must be evaluated once per step (6 steps), it was called {len(calls)} times"
    assert_close(np.array(calls[:6]), path[:6], rtol=0, atol=1e-15,
                 msg="step t must use the gradient at the point reached after t steps (path[t])")


def test_gradient_descent_does_not_modify_x0_and_accepts_integers(calc):
    x0 = np.array([4, -3])
    x_final, path = calc.gradient_descent(bowl_grad, x0, lr=0.1, n_steps=3)
    assert np.array_equal(x0, [4, -3]), f"the caller's x0 must not change, got {listed(x0)}"
    assert x_final.dtype.kind == "f", f"x_final must be a float array even for an integer x0, got dtype {x_final.dtype}"
    assert_close(path, sgd_oracle(bowl_grad, [4.0, -3.0], 0.1, 3), rtol=1e-10, atol=1e-12,
                 msg="an integer x0 is converted to a float copy before the first step")


def test_gradient_descent_does_not_modify_a_float_x0(calc):
    x0 = np.array([0.5, -0.25])
    gradient_descent_result = calc.gradient_descent(bowl_grad, x0, lr=0.1, n_steps=5)
    assert np.array_equal(x0, [0.5, -0.25]), (f"the caller's x0 must not change (work on a float copy), got "
                                              f"{listed(x0)}")
    assert_close(gradient_descent_result[1], sgd_oracle(bowl_grad, [0.5, -0.25], 0.1, 5), rtol=1e-10, atol=1e-12,
                 msg="5 steps on the bowl from (0.5, -0.25)")


def test_gradient_descent_maximize_climbs_to_the_top(calc):
    x_final, _ = calc.gradient_descent(lambda x: -2 * (x - 2), [0.0], lr=0.25, n_steps=3, maximize=True)
    assert_close(x_final, [1.75], rtol=1e-12, msg="maximize=True climbs: x <- x + lr * grad(x) (docstring example)")


def test_gradient_descent_reaches_the_bottom_of_a_bowl(calc):
    x_final, _ = calc.gradient_descent(bowl_grad, [0.0, 0.0], lr=0.1, n_steps=200)
    assert_close(x_final, [2.0, -1.0], rtol=0, atol=1e-8, msg="200 steps on the bowl (x - 2)**2 + 3 (y + 1)**2")


@pytest.mark.parametrize("kwargs, why", [
    pytest.param({"lr": 0.0}, "lr = 0 (the learning rate must be > 0)", id="lr-zero"),
    pytest.param({"lr": -0.1}, "lr < 0 (the learning rate must be > 0; use maximize=True to climb)", id="lr-negative"),
    pytest.param({"n_steps": -1}, "n_steps < 0", id="n-steps-negative"),
    pytest.param({"tol": -1e-3}, "tol < 0", id="tol-negative"),
])
def test_gradient_descent_rejects_invalid_arguments(calc, kwargs, why):
    assert_raises_value_error(calc.gradient_descent, bowl_grad, [0.0, 0.0], **kwargs, why=why)


# ------------------------------------------------------------------ find_local_extrema (5.14)
def extrema_oracle(y, order):
    y = np.asarray(y)
    return (signal.argrelextrema(y, np.less, order=order)[0],
            signal.argrelextrema(y, np.greater, order=order)[0])


def test_find_local_extrema_docstring_example(calc):
    y = [0, 2, 1, 3, 1, 0, 1]
    minima, maxima = calc.find_local_extrema(y)
    assert_equal_ints(minima, [2, 5], msg="order=1: the local minima of [0, 2, 1, 3, 1, 0, 1]")
    assert_equal_ints(maxima, [1, 3], msg="order=1: the local maxima of [0, 2, 1, 3, 1, 0, 1]")
    minima, maxima = calc.find_local_extrema(y, order=2)
    assert_equal_ints(minima, [5], msg="order=2: a minimum must beat its 2 neighbours on each side")
    assert_equal_ints(maxima, [3], msg="order=2: a maximum must beat its 2 neighbours on each side")


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_find_local_extrema_matches_scipy_argrelextrema(calc, order):
    rng = np.random.default_rng(order)
    for trial in range(25):
        n = int(rng.integers(0, 40))
        y = rng.integers(0, 6, n) if trial % 2 == 0 else rng.normal(size=n)   # integers: many ties
        minima, maxima = calc.find_local_extrema(y, order=order)
        expected_min, expected_max = extrema_oracle(y, order)
        assert_equal_ints(minima, expected_min, msg=f"local minima (order={order}), as scipy.signal.argrelextrema",
                          data=f"y={listed(y)}")
        assert_equal_ints(maxima, expected_max, msg=f"local maxima (order={order}), as scipy.signal.argrelextrema",
                          data=f"y={listed(y)}")


def test_find_local_extrema_on_a_noisy_wave(calc):
    t = np.linspace(0, 6 * np.pi, 400)
    y = np.sin(t) + 0.05 * np.random.default_rng(3).normal(size=t.size)
    for order in (1, 10, 40):
        minima, maxima = calc.find_local_extrema(y, order=order)
        expected_min, expected_max = extrema_oracle(y, order)
        assert_equal_ints(minima, expected_min, msg=f"local minima of a noisy sine wave (order={order})")
        assert_equal_ints(maxima, expected_max, msg=f"local maxima of a noisy sine wave (order={order})")


EDGE_RULES = ("the first and last samples are never extrema; equal neighbours (a plateau) prevent a strict "
              "extremum")


@pytest.mark.parametrize("y, expected_min, expected_max, why", [
    pytest.param([0, 1, 1, 0], [], [], "a flat top (two equal samples) is not a strict maximum", id="flat-top"),
    pytest.param([3, 1, 1, 3], [], [], "a flat bottom is not a strict minimum", id="flat-bottom"),
    pytest.param([5, 1, 2, 3, 0], [1], [3], "the first (5) and last (0) samples are never extrema", id="edges"),
    pytest.param([2, 2, 2, 2], [], [], "a constant curve has no strict extremum", id="constant"),
], )
def test_find_local_extrema_strict_comparisons_and_edges(calc, y, expected_min, expected_max, why):
    minima, maxima = calc.find_local_extrema(y)
    assert_equal_ints(minima, expected_min, msg=f"minima of {y} ({EDGE_RULES})", data=f"case: {why}")
    assert_equal_ints(maxima, expected_max, msg=f"maxima of {y} ({EDGE_RULES})", data=f"case: {why}")


@pytest.mark.parametrize("y", [[], [4.0], [1.0, 2.0]], ids=["empty", "one-sample", "two-samples"])
def test_find_local_extrema_short_curves_have_no_extremum(calc, y):
    minima, maxima = calc.find_local_extrema(y)
    assert_equal_ints(minima, np.array([], dtype=int), msg=f"no minimum in a curve of {len(y)} sample(s)")
    assert_equal_ints(maxima, np.array([], dtype=int), msg=f"no maximum in a curve of {len(y)} sample(s)")


def test_find_local_extrema_neighbours_beyond_the_ends_are_ignored(calc):
    # sample 1 is lower than everything within 3 samples (only sample 0 lies on its left)
    minima, maxima = calc.find_local_extrema([4, 0, 5, 6, 7, 8, 9], order=3)
    assert_equal_ints(minima, [1], msg="order=3: neighbours beyond the start of the array are ignored, "
                                       "so sample 1 is still a minimum", data="y=[4, 0, 5, 6, 7, 8, 9]")
    assert_equal_ints(maxima, np.array([], dtype=int), msg="no maximum here (the last sample never counts)")


@pytest.mark.parametrize("y, order, why", [
    pytest.param([[0, 1, 0], [1, 0, 1]], 1, "y is 2-D (a curve is a 1-D array)", id="2-D"),
    pytest.param([0, 1, 0, 1], 0, "order = 0 (compare with at least one neighbour)", id="order-zero"),
    pytest.param([0, 1, 0, 1], -2, "order < 0", id="order-negative"),
])
def test_find_local_extrema_rejects_invalid_inputs(calc, y, order, why):
    assert_raises_value_error(calc.find_local_extrema, y, order=order, why=why)


# ------------------------------------------------------------------ classify_critical_point (5.24)
KNOWN_CASES = [
    pytest.param(lambda v: v[0] ** 2 + v[1] ** 2, [0.0, 0.0], "minimum", id="bowl"),
    pytest.param(lambda v: -v[0] ** 2 - 2 * v[1] ** 2, [0.0, 0.0], "maximum", id="hilltop"),
    pytest.param(lambda v: v[0] ** 2 - v[1] ** 2, [0.0, 0.0], "saddle", id="saddle-x2-minus-y2"),
    pytest.param(lambda v: v[0] * v[1], [0.0, 0.0], "saddle", id="saddle-xy-needs-the-diagonals"),
    pytest.param(lambda v: 5.0, [1.0, 2.0], "flat", id="constant"),
    pytest.param(lambda v: v[0] ** 3 - 3 * v[0] * v[1] ** 2, [0.0, 0.0], "flat", id="monkey-saddle"),
    pytest.param(lambda v: (v[0] - 1) ** 2 - (v[1] + 2) ** 2, [1.0, -2.0], "saddle", id="saddle-away-from-0"),
    pytest.param(rosenbrock_vec, [1.0, 1.0], "minimum", id="rosenbrock-minimum"),
    pytest.param(lambda v: v[0] ** 3 - 3 * v[0] + v[1] ** 2, [1.0, 0.0], "minimum", id="cubic-valley-minimum"),
    pytest.param(lambda v: v[0] ** 3 - 3 * v[0] + v[1] ** 2, [-1.0, 0.0], "saddle", id="cubic-valley-saddle"),
    pytest.param(lambda v: v[0] ** 2, [0.0, 0.0], "flat", id="valley-floor-is-inconclusive"),
    pytest.param(lambda v: -(v[1] - 3) ** 2, [2.0, 3.0], "flat", id="ridge-is-inconclusive"),
    pytest.param(lambda v: -v[0] ** 2, [0.0], "maximum", id="one-variable"),
    pytest.param(lambda v: v[0] ** 2 + v[1] ** 2 - v[2] ** 2, [0.0, 0.0, 0.0], "saddle", id="three-variables"),
]


@pytest.mark.parametrize("f, x, expected", KNOWN_CASES)
def test_classify_critical_point_known_cases(calc, f, x, expected):
    result = calc.classify_critical_point(f, x)
    assert result == expected, f"expected {expected!r}, got {result!r} (at x={x})"


def _directional_class(a, tol):
    """The class given by the documented rule for the quadratic form v -> v A v (second difference 2 d A d)."""
    n = len(a)
    eye = np.eye(n)
    dirs = [eye[i] for i in range(n)] + [eye[i] + s * eye[j] for i in range(n) for j in range(i + 1, n)
                                          for s in (1, -1)]
    curv = np.array([2 * d @ a @ d for d in dirs])
    if (curv > tol).any() and (curv < -tol).any():
        return "saddle", curv
    if (curv > tol).all():
        return "minimum", curv
    if (curv < -tol).all():
        return "maximum", curv
    return "flat", curv


def test_classify_critical_point_matches_the_signs_of_the_hessian_eigenvalues(calc):
    rng = np.random.default_rng(24)
    checked = 0
    while checked < 30:
        n = int(rng.integers(2, 5))
        q = np.linalg.qr(rng.normal(size=(n, n)))[0]
        eigenvalues = rng.choice([-1, 1], n) * rng.uniform(0.5, 3.0, n)
        a = q @ np.diag(eigenvalues) @ q.T / 2        # Hessian of v -> v A v is 2A, eigenvalues "eigenvalues"
        signs = np.sign(np.linalg.eigvalsh(2 * a))
        truth = "minimum" if (signs > 0).all() else "maximum" if (signs < 0).all() else "saddle"
        rule, curv = _directional_class(a, 1e-6)
        if rule != truth or np.abs(curv).min() < 1e-3:
            continue                                # the axes and diagonals cannot see this one: skip it
        center = rng.normal(size=n)
        result = calc.classify_critical_point(lambda v: (v - center) @ a @ (v - center), center)
        assert result == truth, (f"expected {truth!r} (signs of the eigenvalues of the Hessian), got {result!r} "
                                 f"for the quadratic form (v - c) A (v - c), A = {np.round(a, 3).tolist()}, "
                                 f"c = {np.round(center, 3).tolist()}")
        checked += 1


def test_classify_critical_point_uses_tol(calc):
    def tiny_bowl(v):
        return 1e-8 * (v[0] ** 2 + v[1] ** 2)       # second differences of 2e-8, below the default tol

    assert calc.classify_critical_point(tiny_bowl, [0.0, 0.0]) == "flat", (
        "second differences with |D| <= tol (here 2e-8 <= 1e-6) count as zero: expected 'flat'")
    assert calc.classify_critical_point(tiny_bowl, [0.0, 0.0], tol=1e-9) == "minimum", (
        "with tol=1e-9 the same second differences (2e-8) are positive: expected 'minimum'")


def test_classify_critical_point_uses_grad_tol(calc):
    def bowl(v):
        return v[0] ** 2 + v[1] ** 2

    assert_raises_value_error(calc.classify_critical_point, bowl, [1e-3, 0.0],
                              why="at (0.001, 0) the gradient has norm 0.002 > grad_tol = 1e-4: not a critical point")
    result = calc.classify_critical_point(bowl, [1e-3, 0.0], grad_tol=1e-2)
    assert result == "minimum", f"with grad_tol=1e-2 the point (0.001, 0) counts as critical: expected 'minimum', " \
                                f"got {result!r}"


@pytest.mark.parametrize("f, x, why", [
    pytest.param(rosenbrock_vec, [0.0, 0.0], "the gradient of Rosenbrock at (0, 0) is (-2, 0): not a critical point",
                 id="not-critical"),
    pytest.param(lambda v: float(np.sum(v ** 2)), [[0.0], [0.0]], "x has shape (2, 1): it must be 1-D", id="x-2-D"),
])
def test_classify_critical_point_rejects_invalid_points(calc, f, x, why):
    assert_raises_value_error(calc.classify_critical_point, f, x, why=why)


def test_classify_critical_point_does_not_modify_x(calc):
    x = np.array([1.0, 1.0])
    calc.classify_critical_point(rosenbrock_vec, x)
    assert np.array_equal(x, [1.0, 1.0]), f"the caller's x must not change, got {listed(x)}"


def test_classify_critical_point_returns_a_string(calc):
    result = calc.classify_critical_point(lambda v: v[0] ** 2 + v[1] ** 2, np.array([0.0, 0.0]))
    assert isinstance(result, str), f"expected a str ('minimum'), got {type(result).__name__}"
