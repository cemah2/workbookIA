"""Tests of mylearn.autograd_mini (chapter 18): oracle tests and properties.

    pytest tests/test_ch18_autograd_mini.py              # your code (mon_travail/mylearn/autograd_mini.py)
    pytest tests/test_ch18_autograd_mini.py --impl=ref   # the reference

Oracles: PyTorch scalar tensors in float64 with ``requires_grad=True`` (``.backward()`` on the same
expressions, ``torch.nn.Sequential`` loaded with ``to_params()``), the ``math`` module and hand-written
derivatives, central finite differences of expressions evaluated here with plain Python floats, and a
NumPy forward pass written in this file. Without PyTorch, its oracle tests are skipped and the other
oracles take over.

Every test name starts with ``test_value_`` (exercise 18.26, ``-k test_value_``) or ``test_mini_mlp_``
(exercise 18.28, ``-k test_mini_mlp_``). The tests of ``Value`` never use ``MiniMLP``; the tests of
``MiniMLP`` use your ``Value`` (MiniMLP is made of Values).

Contract precisions written in the statement of 18.26 and checked here: gradients ACCUMULATE (a Value
used several times receives the sum of its contributions, ``a * a`` included); plain numbers work on
either side of ``+``, ``-``, ``*`` and ``/``; ``Value ** Value``, or a string, None or a list as an operand,
raise a TypeError; ``log`` of a non-positive value raises a ValueError; ``relu`` has a derivative of 0 at
0; ``sigmoid`` gives 1.0 and 0.0, without OverflowError, at +1000 and -1000; ``backward`` sets the grad of
its Value to 1.0 and works on graphs of a few hundred levels. The first line of every failure message
says what was expected (it is the line that ``pytest -rf`` shows).
"""

import math
import random

import numpy as np
import pytest

try:
    import torch
except ImportError:  # the oracle tests run on Colab
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")


@pytest.fixture
def ag(mylearn_module):
    return mylearn_module("autograd_mini")


# ------------------------------------------------------------------ helpers
def _fail(msg: str, head: str, data: str = ""):
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


def assert_number(got, expected, what, rtol=1e-12, atol=1e-12):
    if got is None:
        _fail("", f"expected {what} = {expected:.12g}, got None")
    try:
        value = float(got)
    except (TypeError, ValueError):
        _fail("", f"expected {what} as a float ({expected:.12g}), got {type(got).__name__}")
    if not math.isclose(value, expected, rel_tol=rtol, abs_tol=atol):
        _fail("", f"expected {what} = {expected:.12g}, got {value:.12g}")


def call(function, *args, expected="a result", **kwargs):
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the operation")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def as_value(ag, result, what):
    if result is None:
        _fail("", f"expected a Value from {what}, got None (did you forget the return?)")
    if not isinstance(result, ag.Value):
        _fail("", f"expected a Value from {what}, got {type(result).__name__}")
    return result


def backward(ag, out, what):
    out = as_value(ag, out, what)
    call(out.backward, expected=f"backward() to run on {what}")
    return out


def assert_raises(error, function, why):
    try:
        function()
    except NotImplementedError:
        raise                  # before `except error`: NotImplementedError is a RuntimeError; shown as "⏳"
    except error:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"expected {error.__name__} for {why}, got {type(exc).__name__}: {exc}") from None
    raise AssertionError(f"expected {error.__name__} for {why}, but nothing was raised")


# ================================================================== Value: elementary operations (18.26)
def test_value_repr_follows_the_docstring(ag):
    text = call(repr, ag.Value(2.0), expected="a repr")
    if text != "Value(data=2.0, grad=0.0)":
        _fail("", f"expected repr(Value(2.0)) == 'Value(data=2.0, grad=0.0)', got {text!r}")


def test_value_docstring_example(ag):
    a, b = ag.Value(2.0), ag.Value(-3.0)
    c = backward(ag, a * b + a, "a * b + a")
    assert_number(c.data, -4.0, "(a * b + a).data for a = 2, b = -3")
    assert_number(a.grad, -2.0, "a.grad for c = a * b + a (a is used twice: b + 1)")
    assert_number(b.grad, 2.0, "b.grad for c = a * b + a")
    assert_number(call(lambda: (1 - a / 4).data), 0.5, "(1 - a / 4).data for a = 2")


def test_value_backward_docstring_example(ag):
    x = ag.Value(3.0)
    backward(ag, x * x + 2 * x, "x * x + 2 * x")
    assert_number(x.grad, 8.0, "d(x*x + 2x)/dx at x = 3 (2x + 2: x * x counts twice)")


# (name, expression on Values, expression on floats, d/da, d/db) with a = 1.5, b = -0.8
A, B = 1.5, -0.8
BINARY = [
    ("add", lambda a, b: a + b, 1.0, 1.0),
    ("sub", lambda a, b: a - b, 1.0, -1.0),
    ("mul", lambda a, b: a * b, B, A),
    ("truediv", lambda a, b: a / b, 1 / B, -A / B ** 2),
]


@pytest.mark.parametrize("name, op, da, db", BINARY, ids=[case[0] for case in BINARY])
def test_value_binary_operation_data_and_grads(ag, name, op, da, db):
    a, b = ag.Value(A), ag.Value(B)
    out = backward(ag, call(op, a, b, expected=f"a Value from a {name} b"), f"a {name} b")
    assert_number(out.data, op(A, B), f"data of a {name} b for a = {A}, b = {B}")
    assert_number(a.grad, da, f"a.grad after (a {name} b).backward()")
    assert_number(b.grad, db, f"b.grad after (a {name} b).backward()")


# expressions with a plain number on one side: (id, Value expression, float expression, derivative)
WITH_NUMBERS = [
    ("number-plus-value", lambda a: 3 + a, lambda a: 3 + a, 1.0),
    ("value-plus-number", lambda a: a + 3, lambda a: a + 3, 1.0),
    ("number-minus-value", lambda a: 3 - a, lambda a: 3 - a, -1.0),
    ("value-minus-number", lambda a: a - 3.5, lambda a: a - 3.5, 1.0),
    ("number-times-value", lambda a: 2 * a, lambda a: 2 * a, 2.0),
    ("value-times-number", lambda a: a * -4.0, lambda a: a * -4.0, -4.0),
    ("number-over-value", lambda a: 2 / a, lambda a: 2 / a, -2 / A ** 2),
    ("value-over-number", lambda a: a / 4, lambda a: a / 4, 0.25),
    ("negation", lambda a: -a, lambda a: -a, -1.0),
]


@pytest.mark.parametrize("name, value_expr, float_expr, derivative", WITH_NUMBERS, ids=[c[0] for c in WITH_NUMBERS])
def test_value_operations_with_plain_numbers(ag, name, value_expr, float_expr, derivative):
    a = ag.Value(A)
    out = backward(ag, call(value_expr, a, expected=f"a Value for {name}"), name)
    assert_number(out.data, float_expr(A), f"data of {name} for a = {A}")
    assert_number(a.grad, derivative, f"a.grad after backward() of {name}")


UNARY = [
    ("pow-2", lambda v: v ** 2, lambda x: x ** 2, lambda x: 2 * x, 1.5),
    ("pow-3", lambda v: v ** 3, lambda x: x ** 3, lambda x: 3 * x ** 2, -1.2),
    ("pow-minus-1", lambda v: v ** -1, lambda x: 1 / x, lambda x: -1 / x ** 2, 0.8),
    ("pow-0.5", lambda v: v ** 0.5, math.sqrt, lambda x: 0.5 / math.sqrt(x), 2.25),
    ("exp", lambda v: v.exp(), math.exp, math.exp, 0.7),
    ("log", lambda v: v.log(), math.log, lambda x: 1 / x, 2.5),
    ("tanh", lambda v: v.tanh(), math.tanh, lambda x: 1 - math.tanh(x) ** 2, 0.6),
    ("relu-positive", lambda v: v.relu(), lambda x: max(x, 0.0), lambda x: 1.0, 1.3),
    ("relu-negative", lambda v: v.relu(), lambda x: max(x, 0.0), lambda x: 0.0, -1.3),
    ("relu-at-zero", lambda v: v.relu(), lambda x: 0.0, lambda x: 0.0, 0.0),
    ("sigmoid", lambda v: v.sigmoid(), lambda x: 1 / (1 + math.exp(-x)),
     lambda x: math.exp(-x) / (1 + math.exp(-x)) ** 2, -0.9),
]


@pytest.mark.parametrize("name, value_op, data_f, grad_f, x", UNARY, ids=[c[0] for c in UNARY])
def test_value_unary_operation_data_and_grad(ag, name, value_op, data_f, grad_f, x):
    a = ag.Value(x)
    out = backward(ag, call(value_op, a, expected=f"a Value from {name}"), name)
    assert_number(out.data, data_f(x), f"data of {name} at {x}")
    assert_number(a.grad, grad_f(x), f"grad of {name} at {x}" + (" (relu'(0) = 0, as in PyTorch)" if x == 0 else ""))


@needs_torch
@pytest.mark.parametrize("name, value_op, data_f, grad_f, x", UNARY, ids=[c[0] for c in UNARY])
def test_value_unary_operation_matches_torch(ag, name, value_op, data_f, grad_f, x):
    torch_ops = {"pow-2": lambda t: t ** 2, "pow-3": lambda t: t ** 3, "pow-minus-1": lambda t: t ** -1,
                 "pow-0.5": lambda t: t ** 0.5, "exp": torch.exp, "log": torch.log, "tanh": torch.tanh,
                 "relu-positive": torch.relu, "relu-negative": torch.relu, "relu-at-zero": torch.relu,
                 "sigmoid": torch.sigmoid}
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    expected = torch_ops[name](t)
    expected.backward()
    a = ag.Value(x)
    out = backward(ag, call(value_op, a, expected=f"a Value from {name}"), name)
    assert_number(out.data, expected.item(), f"data of {name} at {x} against torch")
    assert_number(a.grad, t.grad.item(), f"grad of {name} at {x} against torch")


@pytest.mark.parametrize("x, expected", [pytest.param(1000.0, 1.0, id="plus-1000"),
                                         pytest.param(-1000.0, 0.0, id="minus-1000")])
def test_value_sigmoid_does_not_overflow(ag, x, expected):
    a = ag.Value(x)
    out = backward(ag, call(a.sigmoid, expected="sigmoid() without OverflowError (take exp of a non-positive number)"),
                   f"sigmoid({x})")
    assert_number(out.data, expected, f"sigmoid({x})")
    assert_number(a.grad, 0.0, f"grad of sigmoid at {x}")


def test_value_operations_do_not_change_their_operands(ag):
    a, b = ag.Value(A), ag.Value(B)
    for build in (lambda: a + b, lambda: a * b, lambda: a - b, lambda: a / b, lambda: a ** 2, lambda: a.tanh(),
                  lambda: b.relu(), lambda: a.exp(), lambda: a.log(), lambda: b.sigmoid(), lambda: -a):
        call(build, expected="a Value")
    if a.data != A or b.data != B or a.grad != 0.0 or b.grad != 0.0:
        _fail("", f"expected the operands unchanged before backward (data {A} and {B}, grad 0), got "
                  f"a = {a!r}, b = {b!r}: each operation must return a NEW Value")


def test_value_operation_records_its_parents(ag):
    a, b = ag.Value(2.0), ag.Value(3.0)
    out = as_value(ag, call(lambda: a * b), "a * b")
    if set(map(id, out._prev)) != {id(a), id(b)}:
        _fail("", "expected (a * b)._prev to contain a and b themselves (the parents of the new Value)")
    square = as_value(ag, call(lambda: a * a), "a * a")
    if len(square._prev) != 2 or not all(p is a for p in square._prev):
        _fail("", f"expected (a * a)._prev == (a, a), got {len(square._prev)} parent(s): a Value used twice "
                  "appears twice")


@pytest.mark.parametrize("build, why", [
    pytest.param(lambda V: V(2.0) ** V(3.0), "Value ** Value (the exponent must be an int or a float)",
                 id="value-pow-value"),
    pytest.param(lambda V: V(2.0) + "1", "Value + '1' (a string)", id="add-string"),
    pytest.param(lambda V: V(2.0) * None, "Value * None", id="mul-None"),
    pytest.param(lambda V: V(2.0) - [1.0], "Value - [1.0] (a list)", id="sub-list"),
    pytest.param(lambda V: V(2.0) / "x", "Value / 'x' (a string)", id="div-string"),
])
def test_value_refuses_unsupported_operands(ag, build, why):
    assert_raises(TypeError, lambda: build(ag.Value), why)


@pytest.mark.parametrize("x", [pytest.param(0.0, id="zero"), pytest.param(-2.0, id="negative")])
def test_value_log_refuses_non_positive_values(ag, x):
    assert_raises(ValueError, lambda: ag.Value(x).log(), f"Value({x}).log()")


# ================================================================== Value: graphs and accumulation (18.26)
def test_value_gradients_accumulate_for_a_square(ag):
    a = ag.Value(-1.5)
    backward(ag, a * a, "a * a")
    assert_number(a.grad, -3.0, "a.grad for a * a at -1.5 (2a: both factors are a, the gradients ADD UP)")


def test_value_gradients_accumulate_for_a_sum_of_the_same_value(ag):
    a = ag.Value(0.4)
    backward(ag, a + a + a, "a + a + a")
    assert_number(a.grad, 3.0, "a.grad for a + a + a (each use contributes 1)")


def test_value_gradients_accumulate_through_a_diamond(ag):
    # x feeds two branches that join again: f = (x * 3).tanh() * (x + 1)
    x = ag.Value(0.3)
    left, right = x * 3, x + 1
    backward(ag, left.tanh() * right, "(3x).tanh() * (x + 1)")
    expected = 3 * (1 - math.tanh(0.9) ** 2) * 1.3 + math.tanh(0.9)
    assert_number(x.grad, expected, "x.grad through two paths (product rule: the two contributions add up)")


def test_value_backward_sets_the_output_grad_to_1(ag):
    a = ag.Value(2.0)
    out = backward(ag, a * 5, "a * 5")
    assert_number(out.grad, 1.0, "the grad of the Value backward() was called on (d out / d out = 1)")
    leaf = ag.Value(7.0)
    backward(ag, leaf, "a leaf")
    assert_number(leaf.grad, 1.0, "the grad of a leaf after leaf.backward()")


def test_value_backward_accumulates_across_calls(ag):
    # backward() sets out.grad = 1 and ADDS to the other grads: two calls on x * x give twice d(x*x)/dx
    x = ag.Value(3.0)
    y = x * x
    backward(ag, y, "x * x")
    backward(ag, y, "x * x, a second time without zero_grad")
    assert_number(y.grad, 1.0, "y.grad after a second y.backward() (backward sets it to 1, it does not add 1)")
    assert_number(x.grad, 12.0, "x.grad after two y.backward() without zero_grad: 6 + 6 (grads accumulate, "
                                "backward never resets them)")
    z = 4 * x
    backward(ag, z, "4 * x, after the two backward() of x * x")
    assert_number(x.grad, 16.0, "x.grad after z = 4 x adds its contribution to the previous 12")


def test_value_backward_leaves_unrelated_values_alone(ag):
    a, b, other = ag.Value(1.0), ag.Value(2.0), ag.Value(3.0)
    _ = other * a          # built, but not part of the graph of out
    backward(ag, a * b, "a * b")
    assert_number(other.grad, 0.0, "the grad of a Value that out does not depend on")


def test_value_backward_visits_each_node_once(ag):
    # a chain where every node is used twice: without a visited set, the work (and the grads) blow up
    x = ag.Value(1.0001)
    y = x
    for _ in range(12):
        y = y * 0.5 + y * 0.5
    backward(ag, y, "a chain of 12 levels where each level uses the previous one twice")
    assert_number(x.grad, 1.0, "x.grad for 12 levels of y = 0.5 y + 0.5 y (each node must be visited once)")


def test_value_backward_on_a_long_chain(ag):
    x = ag.Value(0.5)
    y = x
    for _ in range(300):
        y = y + x * 0.01
    out = backward(ag, y, "a chain of 300 additions")
    assert_number(out.data, 0.5 * (1 + 300 * 0.01), "data of the chain")
    assert_number(x.grad, 1 + 300 * 0.01, "x.grad of the chain of 300 additions", rtol=1e-10)


# random expression graphs: built in parallel with Values and with plain floats
UNARY_RANDOM = ["tanh", "sigmoid", "exp_small", "relu", "square", "neg"]
BINARY_RANDOM = ["+", "-", "*", "/"]


def random_expression(seed, n_leaves=4, n_ops=14):
    """A list of operations (op, i, j) on a growing list of nodes; leaves are numbers away from kinks."""
    rnd = random.Random(seed)
    leaves = [rnd.choice([-1, 1]) * rnd.uniform(0.3, 1.5) for _ in range(n_leaves)]
    ops = []
    for k in range(n_ops):
        n_nodes = n_leaves + k
        if rnd.random() < 0.55:
            ops.append((rnd.choice(BINARY_RANDOM), rnd.randrange(n_nodes), rnd.randrange(n_nodes)))
        else:
            ops.append((rnd.choice(UNARY_RANDOM), rnd.randrange(n_nodes), None))
    return leaves, ops


def evaluate(leaves, ops, kind, values=None):
    """Evaluate the expression with Values ('value'), floats ('float') or torch tensors ('torch')."""
    nodes = list(values) if values is not None else list(leaves)
    for op, i, j in ops:
        x = nodes[i]
        y = nodes[j] if j is not None else None
        if op == "+":
            r = x + y
        elif op == "-":
            r = x - y
        elif op == "*":
            r = x * y
        elif op == "/":
            # divide by (1 + y^2) to stay away from 0
            r = x / (1 + y * y)
        elif op == "tanh":
            r = math.tanh(x) if kind == "float" else x.tanh()
        elif op == "sigmoid":
            r = 1 / (1 + math.exp(-x)) if kind == "float" else x.sigmoid()
        elif op == "exp_small":
            r = math.exp(0.1 * x) if kind == "float" else (x * 0.1).exp()
        elif op == "relu":
            # relu of (x + 3): away from the kink for the magnitudes used here
            r = max(x + 3, 0.0) if kind == "float" else (x + 3).relu()
        elif op == "square":
            r = x ** 2
        else:
            r = -x
        nodes.append(r)
    # the output mixes every node, so that every leaf matters
    out = nodes[-1]
    for node in nodes[:-1]:
        out = out + node * 0.1
    return out


@pytest.mark.parametrize("seed", range(8))
def test_value_random_graphs_match_finite_differences(ag, seed):
    leaves, ops = random_expression(1930 + seed)
    values = [ag.Value(v) for v in leaves]
    out = backward(ag, call(evaluate, leaves, ops, "value", values, expected="a Value expression"),
                   f"random expression {seed}")
    assert_number(out.data, evaluate(leaves, ops, "float"), f"data of random expression {seed}", rtol=1e-10)
    eps = 1e-6
    for k, v in enumerate(values):
        plus, minus = list(leaves), list(leaves)
        plus[k] += eps
        minus[k] -= eps
        numeric = (evaluate(plus, ops, "float") - evaluate(minus, ops, "float")) / (2 * eps)
        assert_number(v.grad, numeric, f"grad of leaf {k} in random expression {seed} (central differences)",
                      rtol=1e-6, atol=1e-7)


@needs_torch
@pytest.mark.parametrize("seed", range(8))
def test_value_random_graphs_match_torch(ag, seed):
    leaves, ops = random_expression(1940 + seed)
    tensors = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in leaves]
    expected = evaluate(leaves, ops, "torch", tensors)
    expected.backward()
    values = [ag.Value(v) for v in leaves]
    out = backward(ag, call(evaluate, leaves, ops, "value", values, expected="a Value expression"),
                   f"random expression {seed}")
    assert_number(out.data, expected.item(), f"data of random expression {seed} against torch", rtol=1e-10)
    for k, (v, t) in enumerate(zip(values, tensors, strict=True)):
        assert_number(v.grad, t.grad.item(), f"grad of leaf {k} in random expression {seed} against torch",
                      rtol=1e-9, atol=1e-12)


@needs_torch
def test_value_neuron_matches_torch(ag):
    xs, ws, b = [0.5, -1.2, 2.0], [0.3, -0.7, 0.1], 0.25
    tx = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in xs]
    tw = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in ws]
    tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    tz = tb + sum(x * w for x, w in zip(tx, tw, strict=True))
    tloss = (torch.tanh(tz) - 1.0) ** 2
    tloss.backward()
    vx, vw, vb = [ag.Value(v) for v in xs], [ag.Value(v) for v in ws], ag.Value(b)
    z = vb
    for x, w in zip(vx, vw, strict=True):
        z = z + x * w
    loss = backward(ag, call(lambda: (z.tanh() - 1.0) ** 2), "(tanh(b + sum x w) - 1) ** 2")
    assert_number(loss.data, tloss.item(), "loss of one tanh neuron against torch")
    for name, ours, theirs in [("w", vw, tw), ("x", vx, tx)]:
        for k, (v, t) in enumerate(zip(ours, theirs, strict=True)):
            assert_number(v.grad, t.grad.item(), f"grad of {name}{k} for one tanh neuron against torch")
    assert_number(vb.grad, tb.grad.item(), "grad of the bias for one tanh neuron against torch")


# ================================================================== MiniMLP (18.28)
def numpy_mlp(params, x, activation):
    a = np.asarray(x, dtype=float)
    f = {"relu": lambda z: np.maximum(z, 0.0), "tanh": np.tanh, "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
         "linear": lambda z: z}[activation]
    for index, (W, b) in enumerate(params):
        a = a @ W + b
        if index < len(params) - 1:
            a = f(a)
    return a


def model_params(model):
    params = model.to_params()
    if not isinstance(params, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in params):
        _fail("", "expected to_params() to return a list of (W, b) tuples, one per layer")
    for layer, (W, b) in enumerate(params):
        shape = (len(model.weights_[layer]), len(model.weights_[layer][0]))
        if not (isinstance(W, np.ndarray) and isinstance(b, np.ndarray)) or W.dtype != np.float64 or b.dtype != np.float64:
            _fail("", f"expected float64 NumPy arrays (W, b) for layer {layer} of to_params() (they serve here as the "
                      "reference: fix to_params() first)")
        if W.shape != shape or b.shape != (shape[1],):
            _fail("", f"expected W of shape {shape} and b of shape {(shape[1],)} for layer {layer} of to_params(), "
                      f"got {W.shape} and {b.shape} (fix to_params() first)")
    return params


def outputs_as_floats(ag, out, n_out):
    if n_out == 1:
        return [as_value(ag, out, "a model with one output").data]
    if not isinstance(out, list) or len(out) != n_out:
        _fail("", f"expected a list of {n_out} Values from a model with {n_out} outputs, got {type(out).__name__}")
    return [as_value(ag, o, "each output").data for o in out]


ARCHITECTURES = [
    pytest.param(2, [3, 1], "tanh", id="2-3-1-tanh"),
    pytest.param(3, [4, 4, 2], "relu", id="3-4-4-2-relu"),
    pytest.param(2, [2, 3], "sigmoid", id="2-2-3-sigmoid"),
    pytest.param(4, [3, 1], "linear", id="4-3-1-linear"),
    pytest.param(3, [2], "relu", id="3-2-single-layer"),
]


def test_mini_mlp_docstring_example(ag):
    model = ag.MiniMLP(2, [3, 1], activation="tanh", random_state=0)
    shapes = [W.shape for W, b in model_params(model)]
    if shapes != [(2, 3), (3, 1)]:
        _fail("", f"expected the shapes [(2, 3), (3, 1)] from to_params(), got {shapes}")
    if len(call(model.parameters)) != 13:
        _fail("", f"expected 13 parameters for MiniMLP(2, [3, 1]), got {len(model.parameters())}")
    out = call(model, [1.0, -1.0], expected="a Value")
    backward(ag, out, "the output of MiniMLP(2, [3, 1])")
    call(model.zero_grad)
    if not all(p.grad == 0.0 for p in model.parameters()):
        _fail("", "expected every grad equal to 0.0 after zero_grad()")


@pytest.mark.parametrize("n_in, sizes, activation", ARCHITECTURES)
def test_mini_mlp_outputs_match_a_numpy_forward_pass(ag, n_in, sizes, activation):
    model = ag.MiniMLP(n_in, sizes, activation=activation, random_state=1950)
    params = model_params(model)
    rng = np.random.default_rng(1951)
    for x in rng.normal(size=(4, n_in)):
        got = outputs_as_floats(ag, call(model, list(x), expected="the output of the model"), sizes[-1])
        want = numpy_mlp(params, x, activation)
        for k, (g, w) in enumerate(zip(got, want, strict=True)):
            assert_number(g, w, f"output {k} of MiniMLP({n_in}, {sizes}, {activation!r}) against a NumPy forward "
                                "pass on to_params() (hidden layers activated, last layer linear)", rtol=1e-12)


@needs_torch
@pytest.mark.parametrize("n_in, sizes, activation", ARCHITECTURES)
def test_mini_mlp_outputs_and_gradients_match_torch(ag, n_in, sizes, activation):
    model = ag.MiniMLP(n_in, sizes, activation=activation, random_state=1952)
    params = model_params(model)
    layers = []
    for index, (W, b) in enumerate(params):
        lin = torch.nn.Linear(*W.shape).double()
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W.T))
            lin.bias.copy_(torch.tensor(b))
        layers.append(lin)
        if index < len(params) - 1 and activation != "linear":
            layers.append({"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid}[activation]())
    net = torch.nn.Sequential(*layers)
    x = np.random.default_rng(1953).normal(size=n_in)
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    tout = net(xt)
    tloss = (tout ** 2).sum()
    tloss.backward()

    xv = [ag.Value(v) for v in x]
    out = call(model, xv, expected="the output of the model on Values")
    outs = [out] if sizes[-1] == 1 else out
    loss = outs[0] ** 2
    for o in outs[1:]:
        loss = loss + o ** 2
    backward(ag, loss, "the sum of the squared outputs")
    assert_number(loss.data, tloss.item(), "loss (sum of squared outputs) against torch", rtol=1e-12)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    for layer, lin in enumerate(linears):
        tW, tb = lin.weight.grad.numpy().T, lin.bias.grad.numpy()
        for i in range(tW.shape[0]):
            for j in range(tW.shape[1]):
                assert_number(model.weights_[layer][i][j].grad, tW[i, j],
                              f"grad of weights_[{layer}][{i}][{j}] against torch", rtol=1e-10, atol=1e-12)
        for j in range(len(tb)):
            assert_number(model.biases_[layer][j].grad, tb[j], f"grad of biases_[{layer}][{j}] against torch",
                          rtol=1e-10, atol=1e-12)
    for k, v in enumerate(xv):
        assert_number(v.grad, xt.grad[k].item(), f"grad of input {k} against torch", rtol=1e-10, atol=1e-12)


def test_mini_mlp_gradients_match_finite_differences(ag):
    model = ag.MiniMLP(3, [4, 2], activation="tanh", random_state=1954)
    params = model_params(model)
    x = np.array([0.4, -1.1, 0.8])
    outs = call(model, list(x), expected="the output of the model")
    loss = outs[0] * 0.5 - outs[1] ** 2
    backward(ag, loss, "0.5 out0 - out1**2")
    eps = 1e-6

    def f(ps):
        o = numpy_mlp(ps, x, "tanh")
        return 0.5 * o[0] - o[1] ** 2

    for layer, (W, b) in enumerate(params):
        for i, j in np.ndindex(W.shape):
            plus = [(Wk.copy(), bk.copy()) for Wk, bk in params]
            minus = [(Wk.copy(), bk.copy()) for Wk, bk in params]
            plus[layer][0][i, j] += eps
            minus[layer][0][i, j] -= eps
            numeric = (f(plus) - f(minus)) / (2 * eps)
            assert_number(model.weights_[layer][i][j].grad, numeric,
                          f"grad of weights_[{layer}][{i}][{j}] (central differences)", rtol=1e-6, atol=1e-8)


def test_mini_mlp_returns_a_value_or_a_list(ag):
    single = call(ag.MiniMLP(2, [3, 1], random_state=1), [0.5, 0.5])
    if not isinstance(single, ag.Value):
        _fail("", f"expected a single Value from a model with one output, got {type(single).__name__}")
    several = call(ag.MiniMLP(2, [3, 4], random_state=1), [0.5, 0.5])
    if not (isinstance(several, list) and len(several) == 4 and all(isinstance(v, ag.Value) for v in several)):
        _fail("", f"expected a list of 4 Values from a model with 4 outputs, got {type(several).__name__}")


def test_mini_mlp_parameters_are_the_values_themselves_in_order(ag):
    model = ag.MiniMLP(2, [3, 2], random_state=1955)
    params = call(model.parameters)
    expected = []
    for W, b in zip(model.weights_, model.biases_, strict=True):
        for row in W:
            expected.extend(row)
        expected.extend(b)
    if not isinstance(params, list) or len(params) != len(expected):
        _fail("", f"expected a list of {len(expected)} Values (2*3 + 3 + 3*2 + 2), got "
                  f"{len(params) if hasattr(params, '__len__') else type(params).__name__}")
    for k, (p, e) in enumerate(zip(params, expected, strict=True)):
        if p is not e:
            _fail("", f"expected parameter {k} to be the Value object itself, in the order: layer by layer, weights "
                      "row by row (weights_[l][0][0], weights_[l][0][1], ...), then the biases")


@pytest.mark.parametrize("n_in, sizes", [(2, [3, 1]), (3, [4, 4, 2]), (1, [1])])
def test_mini_mlp_parameters_count(ag, n_in, sizes):
    widths = [n_in, *sizes]
    expected = sum(a * b + b for a, b in zip(widths[:-1], widths[1:], strict=True))
    got = len(call(ag.MiniMLP(n_in, sizes, random_state=0).parameters))
    if got != expected:
        _fail("", f"expected {expected} parameters for MiniMLP({n_in}, {sizes}), got {got}")


def test_mini_mlp_zero_grad_resets_every_parameter(ag):
    model = ag.MiniMLP(2, [3, 2], activation="tanh", random_state=1956)
    outs = call(model, [0.3, -0.4])
    backward(ag, outs[0] + outs[1], "the sum of the outputs")
    call(model.zero_grad)
    nonzero = [p for p in model.parameters() if p.grad != 0.0]
    if nonzero:
        _fail("", f"expected every grad equal to 0.0 after zero_grad(), got {len(nonzero)} nonzero")


def test_mini_mlp_to_params_returns_float64_copies(ag):
    model = ag.MiniMLP(2, [3, 1], random_state=1957)
    params = model_params(model)
    for layer, (W, b) in enumerate(params):
        if not (isinstance(W, np.ndarray) and isinstance(b, np.ndarray)) or W.dtype != np.float64:
            _fail("", f"expected float64 NumPy arrays for layer {layer} of to_params()")
        if b.shape != (W.shape[1],):
            _fail("", f"expected b of shape {(W.shape[1],)} for layer {layer}, got {b.shape}")
    first = model.weights_[0][0][0].data
    params[0][0][0, 0] += 10.0
    if model.weights_[0][0][0].data != first:
        _fail("", "expected to_params() to return copies: changing the array changed the Value")
    model.weights_[0][0][0].data = 0.123
    if model_params(model)[0][0][0, 0] != 0.123:
        _fail("", "expected to_params() to read the CURRENT data of the Values")


@pytest.mark.parametrize("length", [2, 4], ids=["too-short", "too-long"])
def test_mini_mlp_refuses_an_input_of_the_wrong_length(ag, length):
    model = ag.MiniMLP(3, [2, 1], random_state=0)
    x = [1.0, 2.0, 3.0, 4.0][:length]
    assert_raises(ValueError, lambda: model(x), f"an input of length {length} for n_in = 3 (zip would silently truncate)")


def test_mini_mlp_gradient_steps_reduce_the_loss(ag):
    model = ag.MiniMLP(2, [4, 1], activation="tanh", random_state=1958)
    data = [([0.5, -0.2], 1.0), ([-0.7, 0.3], -1.0), ([0.1, 0.9], 0.5)]

    def loss_value():
        total = ag.Value(0.0)
        for x, y in data:
            total = total + (call(model, x) - y) ** 2
        return total

    first = loss_value()
    losses = [first.data]
    for _ in range(20):
        model.zero_grad()
        loss = loss_value()
        backward(ag, loss, "the training loss")
        for p in model.parameters():
            p.data -= 0.05 * p.grad
        losses.append(loss_value().data)
    if not losses[-1] < 0.8 * losses[0]:
        _fail("", f"expected 20 gradient steps (lr 0.05) to cut the loss by at least 20 %, got {losses[0]:.4g} -> "
                  f"{losses[-1]:.4g} (are the gradients right and the parameters the objects themselves?)")
