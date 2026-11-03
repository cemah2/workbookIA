"""Tests of mylearn.explain (bonus chapter B6): oracle tests and properties.

    pytest tests/test_chb6_explain.py              # your code (mon_travail/mylearn/explain.py)
    pytest tests/test_chb6_explain.py --impl=ref   # the reference

Oracles: Shapley values from a transcription of the docstring formula (``itertools.product`` over the coalitions,
``math.factorial`` weights, ``f`` averaged over the background rows), checked against
``shap.explainers.Exact(f, shap.maskers.Independent(background, max_samples=len(background)))`` (shap 0.54.1: same
values to about 1e-15, ``base_values`` = mean of ``f`` over the background) and against
``shap.TreeExplainer(feature_perturbation='interventional')`` on a shallow tree, with the properties of
efficiency, symmetry, null player, additivity and ``phi_i = w_i (x_i - mean_i)`` for a linear model;
integrated gradients against the closed forms of the four rules for linear, quadratic and cubic models (the rules
weight ``a`` and ``a**2`` differently: ``mean(a)`` and ``mean(a**2)`` of each rule are written in closed form),
a transcription of the rules with explicit loops, the completeness property (error decreasing with ``n_steps``),
and ``captum.attr.IntegratedGradients`` when captum is installed (captum 0.9: 'left', 'right' and 'midpoint' are
its 'riemann_left', 'riemann_right' and 'riemann_middle' with the same ``n_steps``; its 'riemann_trapezoid' with
``n_steps + 1`` points weighs them by ``1 / (n_steps + 1)``, so it equals the 'trapezoid' rule of the docstring
multiplied by ``n_steps / (n_steps + 1)``: the test applies that factor); Grad-CAM against an explicit
``torch.einsum`` computation and a loop over the channels, with the properties of the docstring; per-group rates
against ``sklearn.metrics.confusion_matrix`` computed on each subgroup and hand-computed values, and against
``fairlearn.metrics`` when fairlearn is installed (on data where every rate is defined: fairlearn returns a number
where the docstring raises).
Without shap, torch, captum or fairlearn, the tests that need them are skipped and the other oracles remain.
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.
``group_rates`` may use your ``confusion_matrix`` (chapter 3): if a test says that function is not written yet,
write it first (mylearn/metrics.py).

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_shapley_values_"`` (B6.14),
``-k "test_integrated_gradients_ or test_grad_cam_"`` (B6.18),
``-k "test_group_rates_ or test_demographic_parity_difference_ or test_equalized_odds_difference_"`` (B6.23).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified;
``shapley_values`` calls ``f`` only on 2-D arrays of shape ``(n, M)``, uses every background row (no subsampling),
accepts ``M == max_features`` and returns the base value as one number; ``integrated_gradients`` calls
``grad_fn`` only on batches of shape ``(n, *x.shape)`` (even with ``n_steps = 1``), refuses a baseline that NumPy
could broadcast but that does not have the shape of ``x`` (shape ``(1,)`` for an ``x`` of shape ``(3,)``) and the
captum names of the rules ('riemann_trapezoid'), and returns a float array even for an integer ``x``;
``grad_cam`` normalises each map by its own largest absolute value with no epsilon added (the invariance to a
positive factor is tested down to a factor of 1e-3), and a map that is zero after the ReLU stays zero (no NaN);
``group_rates`` keeps the group values as keys (``rates[10]`` for an integer group 10, NumPy scalars accepted),
gives ``'n'`` as an integer (Python or NumPy) and the rates as floats (``float`` or ``np.float64``), accepts a
single group, and refuses NaN in the labels or the predictions (NaN is not 0 or 1); the two gaps return one
number.
The attributions are compared with rtol=1e-6, atol=1e-6 (the docstrings fix no tolerance; a computation in float32
passes, a wrong rule does not); rates and gaps, ratios of counts, with rtol=1e-9.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

from __future__ import annotations

import copy
import itertools
import math
import numbers
import traceback
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

try:
    import torch
except ImportError:  # the torch.einsum oracle runs on Colab; the loop oracle remains
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")

ORACLE_BUG = ("bug in tests/test_chb6_explain.py (not in your code): the oracle of this test "
              "disagrees with another one")


@pytest.fixture
def exp(mylearn_module):
    return mylearn_module("explain")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3, max_line_width=10**6)
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


# The docstrings fix no tolerance: rtol=1e-6, atol=1e-6 lets a computation in float32 pass, not a wrong rule.
RTOL, ATOL = 1e-6, 1e-6
# Rates and gaps are ratios of counts: any correct computation is exact up to rounding.
RATE_RTOL, RATE_ATOL = 1e-9, 1e-12


def assert_close(result, expected, rtol=RTOL, atol=ATOL, msg=""):
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
                f"({bad.size} wrong value(s) out of {want.size}; tolerance rtol={rtol:g}, atol={atol:g})")
        data = f"expected {_short(want, 10)}\n     got {_short(got, 10)}" if want.size <= 16 else ""
        _fail(msg, head, data)


# Functions of earlier chapters that explain.py imports: where to find them when they are not written yet.
_EARLIER = {"confusion_matrix": ("metrics.py", "3")}


def _pending(exc: NotImplementedError, function) -> NotImplementedError:
    """The same "⏳" error, with the file and the chapter when it comes from confusion_matrix."""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        where = _EARLIER.get(frame.name)
        if where and Path(frame.filename).as_posix().endswith(where[0]):
            caller = getattr(function, "__name__", "your function")
            return NotImplementedError(f"{exc} (mylearn/{where[0]}, chapter {where[1]}, called by {caller})")
    return exc


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
    try:
        function(*args, **kwargs)
    except NotImplementedError as exc:   # before `except error`: a RuntimeError; shown as "⏳"
        new = _pending(exc, function)
        if new is exc:
            raise
        raise new from None
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
    except NotImplementedError as exc:
        new = _pending(exc, function)
        if new is exc:
            raise
        raise new from None
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def array_result(result, msg, shape, inputs="") -> np.ndarray:
    """The learner's result as a float ndarray of the given shape (None, type and shape checked first)."""
    if result is None:
        _fail(msg, "expected a NumPy array, got None (did you forget the return?)")
    if not isinstance(result, np.ndarray):
        _fail(msg, f"expected a NumPy array, got {type(result).__name__}")
    if result.shape != tuple(shape):
        _fail(msg, f"expected shape {tuple(shape)}" + (f" for {inputs}" if inputs else "") + f", got shape {result.shape}")
    if result.dtype.kind != "f":
        _fail(msg, f"expected an array of floats, got dtype {result.dtype}")
    return result


def number_result(result, msg) -> float:
    """One real number (Python or NumPy float, or a 0-d array), not a bool or an array of several values."""
    if result is None:
        _fail(msg, "expected one number, got None (did you forget the return?)")
    if isinstance(result, np.ndarray) and result.ndim == 0:
        result = result[()]
    if isinstance(result, (bool, np.bool_)) or not isinstance(result, numbers.Real):
        _fail(msg, f"expected one number (a float), got {type(result).__name__}"
                   + (f" of shape {result.shape}" if isinstance(result, np.ndarray) else ""))
    return float(result)


def assert_unchanged(before, after, name, function):
    if isinstance(before, np.ndarray):
        same = isinstance(after, np.ndarray) and before.shape == after.shape and before.dtype == after.dtype \
            and np.array_equal(before, after, equal_nan=before.dtype.kind in "fc")
    else:
        same = before == after
    if not same:
        _fail(function, f"expected the input {name} to be left as it was, it was modified "
                        "(work on a copy, e.g. np.array(x, dtype=float), and avoid in-place operations like *=)")


# ================================================================== shapley_values (B6.14)
def vectorised(func, M):
    """A model output f: (n, M) -> (n,), that says so if it is called on anything else."""
    def f(X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != M:
            raise AssertionError(f"f must be called on a 2-D array of shape (n, {M}) (one row per evaluation: f is "
                                 f"vectorised), it was called on an array of shape {X.shape}")
        return np.asarray(func(np.asarray(X, dtype=float)), dtype=float)
    return f


def shapley_oracle(f, x, background):
    """The docstring formula with itertools and math.factorial: v(S) = mean over the rows of f(row with x on S)."""
    x = np.asarray(x, dtype=float)
    bg = np.asarray(background, dtype=float)
    M = x.size
    v = {}
    for S in itertools.product((False, True), repeat=M):
        rows = bg.copy()
        for j in range(M):
            if S[j]:
                rows[:, j] = x[j]
        v[S] = float(np.mean(f(rows)))
    phi = np.zeros(M)
    for i in range(M):
        for S, value in v.items():
            if S[i]:
                continue
            k = sum(S)
            with_i = tuple(True if j == i else S[j] for j in range(M))
            phi[i] += math.factorial(k) * math.factorial(M - k - 1) / math.factorial(M) * (v[with_i] - value)
    return phi, v[(False,) * M]


def shapley_of(exp, f, x, background, M=None, **kwargs):
    """The learner's (phi, base_value), checked: a pair, phi a float array of shape (M,), base one number."""
    M = np.asarray(x).shape[0] if M is None else M
    msg = "shapley_values"
    result = call(exp.shapley_values, f, copy.deepcopy(x), copy.deepcopy(background),
                  expected="the pair (phi, base_value)", **kwargs)
    if result is None:
        _fail(msg, "expected the pair (phi, base_value), got None (did you forget the return?)")
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        what = type(result).__name__ + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else "")
        _fail(msg, f"expected the pair (phi, base_value) (a tuple of 2), got {what}")
    phi = array_result(result[0], "shapley_values: phi", (M,), inputs=f"an x of {M} features")
    base = number_result(result[1], "shapley_values: base_value")
    return phi, base


def _nonlinear(M, seed):
    """A smooth nonlinear model with interactions (f(mean of the rows) differs from the mean of f)."""
    W = np.random.default_rng(seed).normal(size=(M, M))

    def func(X):
        out = np.tanh(X @ W).sum(axis=1) + X[:, 0] ** 3 / 4
        if M >= 2:
            out = out + X[:, 0] * X[:, -1] + np.sin(X[:, 1])
        return out
    return func


def _case(M, B, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=M), rng.normal(size=(B, M))


SHAPLEY_CASES = [
    pytest.param(1, 4, id="M1-B4"),
    pytest.param(2, 1, id="M2-one-background-row"),
    pytest.param(3, 5, id="M3-B5"),
    pytest.param(4, 9, id="M4-B9"),
    pytest.param(6, 7, id="M6-B7"),
    pytest.param(3, 120, id="M3-B120-more-than-100-rows"),
]


def test_shapley_values_docstring_examples(exp):
    w = np.array([2.0, -1.0, 0.5])
    background = np.array([[0., 0., 0.], [2., 2., 2.]])
    phi, base = shapley_of(exp, vectorised(lambda X: X @ w, 3), [3., 1., 1.], background)
    assert_close(phi, [4., 0., 0.], msg="shapley_values(lambda X: X @ [2, -1, 0.5], [3, 1, 1], background) (docstring)")
    assert_close(base, 1.5, msg="shapley_values: base_value of the docstring example (mean of f over the background)")
    phi, base = shapley_of(exp, vectorised(lambda X: X[:, 0] * X[:, 1], 2), [1., 1.], [[0., 0.]])
    assert_close(phi, [0.5, 0.5], msg="shapley_values(lambda X: X[:, 0] * X[:, 1], [1, 1], [[0, 0]]) (docstring: "
                                      "the interaction is shared equally)")
    assert_close(base, 0.0, msg="shapley_values: base_value of the second docstring example")


@pytest.mark.parametrize("M,B", SHAPLEY_CASES)
def test_shapley_values_match_the_formula(exp, M, B):
    x, bg = _case(M, B, seed=10 * M + B)
    f = vectorised(_nonlinear(M, seed=M), M)
    want_phi, want_base = shapley_oracle(f, x, bg)
    phi, base = shapley_of(exp, f, x, bg)
    assert_close(base, want_base, msg=f"shapley_values (M={M}, B={B}): base_value = v(empty set), the mean of f over "
                                      "the background rows (f first, then the mean)")
    assert_close(phi, want_phi, msg=f"shapley_values (M={M}, B={B}, nonlinear f): phi_i = sum over S without i of "
                                    "|S|! (M - |S| - 1)! / M! (v(S + {i}) - v(S)), v(S) = mean of f over the rows")


@pytest.fixture(scope="module")
def shap_lib():
    """shap, or a skip: shap is an oracle (the transcription of the formula is tested anyway)."""
    try:
        import shap
    except Exception as exc:  # noqa: BLE001 - ImportError, or a broken installation
        pytest.skip(f"shap is not installed (the transcription of the formula is tested anyway): {exc}")
    return shap


def shap_exact(shap, f, x, bg):
    explainer = shap.explainers.Exact(f, shap.maskers.Independent(bg, max_samples=len(bg)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = explainer(np.asarray(x, dtype=float)[None, :])
    return np.asarray(out.values[0], dtype=float), float(out.base_values[0])


def _logistic_model(M, seed):
    """lambda X: model.predict_proba(X)[:, 1] for a LogisticRegression fitted on random data."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, M))
    y = (X[:, 0] - X[:, 1] + X[:, 0] * X[:, -1] + 0.5 * rng.normal(size=200) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return lambda Z: model.predict_proba(Z)[:, 1]


@pytest.mark.parametrize("M,B,model", [
    pytest.param(3, 6, "nonlinear", id="M3-B6-nonlinear"),
    pytest.param(5, 11, "nonlinear", id="M5-B11-nonlinear"),
    pytest.param(4, 20, "logistic", id="M4-B20-predict-proba"),
    pytest.param(3, 150, "nonlinear", id="M3-B150-all-rows"),
    pytest.param(8, 50, "logistic", id="M8-B50-notebook-size"),
])
def test_shapley_values_match_shap_exact(exp, shap_lib, M, B, model):
    x, bg = _case(M, B, seed=100 + M + B)
    func = _nonlinear(M, seed=7) if model == "nonlinear" else _logistic_model(M, seed=3)
    f = vectorised(func, M)
    ref_phi, ref_base = shap_exact(shap_lib, func, x, bg)
    if M <= 5:
        o_phi, o_base = shapley_oracle(func, x, bg)
        assert np.allclose(o_phi, ref_phi, rtol=1e-9, atol=1e-12) and math.isclose(o_base, ref_base), ORACLE_BUG
    phi, base = shapley_of(exp, f, x, bg)
    assert_close(base, ref_base, msg=f"shapley_values (M={M}, B={B}) against shap.explainers.Exact: base_values "
                                     "(mean of f over all the background rows)")
    assert_close(phi, ref_phi, msg=f"shapley_values (M={M}, B={B}) against shap.explainers.Exact with "
                                   "shap.maskers.Independent(background, max_samples=len(background))")


def test_shapley_values_match_tree_explainer_interventional(exp, shap_lib):
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.1 * rng.normal(size=300)
    tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    bg, xs = X[:30], X[100:103]
    explainer = shap_lib.TreeExplainer(tree, data=bg, feature_perturbation="interventional")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.asarray(explainer.shap_values(xs), dtype=float)
    base_ref = float(np.ravel(explainer.expected_value)[0])
    for i, x in enumerate(xs):
        phi, base = shapley_of(exp, vectorised(tree.predict, 5), x, bg)
        assert_close(base, base_ref, msg=f"shapley_values on a depth-4 tree, row {i}: base_value against "
                                         "TreeExplainer(feature_perturbation='interventional').expected_value")
        assert_close(phi, ref[i], msg=f"shapley_values on a depth-4 tree, row {i}, against "
                                      "TreeExplainer(feature_perturbation='interventional')")


@pytest.mark.parametrize("M,B", [pytest.param(3, 4, id="M3-B4"), pytest.param(5, 8, id="M5-B8")])
def test_shapley_values_base_value_is_the_mean_of_f_over_the_background(exp, M, B):
    x, bg = _case(M, B, seed=5)
    func = _nonlinear(M, seed=2)
    want = float(np.mean(func(bg)))
    assert abs(want - float(func(bg.mean(axis=0, keepdims=True))[0])) > 1e-3, ORACLE_BUG
    _, base = shapley_of(exp, vectorised(func, M), x, bg)
    assert_close(base, want, msg="shapley_values: base_value = mean of f over the background rows (not f of the mean "
                                 "row)")


@pytest.mark.parametrize("M,B", [pytest.param(2, 3, id="M2"), pytest.param(4, 6, id="M4"),
                                 pytest.param(7, 5, id="M7")])
def test_shapley_values_efficiency(exp, M, B):
    x, bg = _case(M, B, seed=20 + M)
    func = _nonlinear(M, seed=M + 1)
    phi, base = shapley_of(exp, vectorised(func, M), x, bg)
    assert_close(phi.sum() + base, float(func(x[None, :])[0]),
                 msg=f"shapley_values (M={M}): efficiency, phi.sum() + base_value == f(x)")


def test_shapley_values_symmetry(exp):
    rng = np.random.default_rng(4)
    bg = rng.normal(size=(6, 3))
    bg[:, 1] = bg[:, 0]                       # features 0 and 1 are exchangeable in the background...
    x = np.array([0.7, 0.7, -1.2])            # ... and in x
    f = vectorised(lambda X: X[:, 0] * X[:, 1] + np.exp(X[:, 0] + X[:, 1]) + X[:, 2] ** 2 * (X[:, 0] + X[:, 1]), 3)
    phi, _ = shapley_of(exp, f, x, bg)
    assert_close(phi[1], phi[0], msg="shapley_values: symmetry, two features that play the same role (same values in "
                                     "x and in every background row, f symmetric in them) get the same phi")


def test_shapley_values_null_player(exp):
    x, bg = _case(4, 6, seed=8)
    f = vectorised(lambda X: np.tanh(X[:, 0] * X[:, 1]) + X[:, 3] ** 2, 4)     # feature 2 is ignored
    phi, _ = shapley_of(exp, f, x, bg)
    assert_close(phi[2], 0.0, msg="shapley_values: null player, a feature that f ignores gets phi = 0")


def test_shapley_values_additivity(exp):
    x, bg = _case(4, 5, seed=9)
    f1, f2 = _nonlinear(4, seed=1), _nonlinear(4, seed=2)
    p1, b1 = shapley_of(exp, vectorised(f1, 4), x, bg)
    p2, b2 = shapley_of(exp, vectorised(f2, 4), x, bg)
    p, b = shapley_of(exp, vectorised(lambda X: f1(X) + 3.0 * f2(X), 4), x, bg)
    assert_close(p, p1 + 3.0 * p2, msg="shapley_values: additivity, phi(f1 + 3 f2) == phi(f1) + 3 phi(f2)")
    assert_close(b, b1 + 3.0 * b2, msg="shapley_values: base_value(f1 + 3 f2) == base_value(f1) + 3 base_value(f2)")


@pytest.mark.parametrize("M,B", [pytest.param(3, 2, id="M3-B2"), pytest.param(5, 7, id="M5-B7"),
                                 pytest.param(9, 3, id="M9-B3")])
def test_shapley_values_linear_model(exp, M, B):
    rng = np.random.default_rng(M * B)
    w, b0 = rng.normal(size=M), 0.25
    x, bg = rng.normal(size=M), rng.normal(size=(B, M))
    phi, base = shapley_of(exp, vectorised(lambda X: X @ w + b0, M), x, bg)
    assert_close(phi, w * (x - bg.mean(axis=0)), msg=f"shapley_values (linear model, M={M}): phi_i = w_i (x_i - mean_i)")
    assert_close(base, float(bg.mean(axis=0) @ w + b0), msg="shapley_values (linear model): base_value = w . mean + b")


def test_shapley_values_interaction_of_three_features(exp):
    f = vectorised(lambda X: X[:, 0] * X[:, 1] * X[:, 2], 3)
    phi, base = shapley_of(exp, f, [1.0, 2.0, 3.0], [[0.0, 0.0, 0.0]])
    assert_close(phi, [2.0, 2.0, 2.0], msg="shapley_values(x0 x1 x2, x = [1, 2, 3], background = [[0, 0, 0]]): the "
                                           "product 6 appears only with the full coalition, so each feature gets 6 / 3")
    assert_close(base, 0.0, msg="shapley_values: base_value of x0 x1 x2 on a zero background")
    # an asymmetric case: f = x0 + x0 x1 on a zero background, x = (1, 4): phi = (1 + 2, 2)
    phi, _ = shapley_of(exp, vectorised(lambda X: X[:, 0] + X[:, 0] * X[:, 1], 2), [1.0, 4.0], [[0.0, 0.0]])
    assert_close(phi, [3.0, 2.0], msg="shapley_values(x0 + x0 x1, x = [1, 4], background = [[0, 0]]): the main effect "
                                      "1 goes to x0, the interaction 4 is shared equally")


def test_shapley_values_accept_max_features_features(exp):
    rng = np.random.default_rng(12)
    w, x, bg = rng.normal(size=12), rng.normal(size=12), rng.normal(size=(2, 12))
    phi, _ = shapley_of(exp, vectorised(lambda X: X @ w, 12), x, bg)
    assert_close(phi, w * (x - bg.mean(axis=0)), msg="shapley_values with M = 12 = max_features (default): allowed "
                                                     "(only M > max_features raises)")
    phi, _ = shapley_of(exp, vectorised(lambda X: X @ w[:4], 4), x[:4], bg[:, :4], max_features=4)
    assert_close(phi, w[:4] * (x[:4] - bg[:, :4].mean(axis=0)), msg="shapley_values with M = 4 and max_features=4")


def test_shapley_values_do_not_modify_their_inputs(exp):
    x, bg = _case(3, 4, seed=3)
    x_before, bg_before = x.copy(), bg.copy()
    call(exp.shapley_values, vectorised(_nonlinear(3, seed=0), 3), x, bg, expected="the pair (phi, base_value)")
    assert_unchanged(x_before, x, "x", "shapley_values")
    assert_unchanged(bg_before, bg, "background", "shapley_values")


@pytest.mark.parametrize("x,background,kwargs,why", [
    pytest.param(np.zeros(13), np.zeros((2, 13)), {}, "M = 13 > max_features = 12 (default)", id="13-features-default"),
    pytest.param(np.zeros(4), np.zeros((3, 4)), {"max_features": 3}, "M = 4 > max_features = 3",
                 id="4-features-max-3"),
    pytest.param(np.zeros((1, 3)), np.zeros((2, 3)), {}, "x of shape (1, 3) is not 1-D", id="x-2d"),
    pytest.param(np.zeros(3), np.zeros(3), {}, "background of shape (3,) is not 2-D", id="background-1d"),
    pytest.param(np.zeros(3), np.zeros((2, 4)), {}, "background has 4 columns, x has 3 features",
                 id="background-wrong-columns"),
    pytest.param(np.zeros(3), np.zeros((2, 2, 3)), {}, "background of shape (2, 2, 3) is not 2-D", id="background-3d"),
])
def test_shapley_values_refuse_invalid_arguments(exp, x, background, kwargs, why):
    assert_raises(ValueError, exp.shapley_values, lambda X: np.asarray(X, float).sum(axis=-1), x, background,
                  why=why, **kwargs)


# ================================================================== integrated_gradients (B6.18)
METHODS = ("left", "right", "midpoint", "trapezoid")


def batched(grad, shape):
    """grad_fn: (n, *shape) -> (n, *shape), that says so if it is called on anything else."""
    shape = tuple(shape)

    def grad_fn(P):
        P = np.asarray(P)
        if P.ndim != len(shape) + 1 or P.shape[1:] != shape:
            raise AssertionError(f"grad_fn must be called on a batch of shape (n, *x.shape) = (n, "
                                 f"{', '.join(map(str, shape))}{',' if len(shape) == 1 else ''}), it was called "
                                 f"on an array of shape {P.shape}")
        return np.asarray(grad(np.asarray(P, dtype=float)), dtype=float)
    return grad_fn


def rule(method, n):
    """Points and weights of the docstring, with explicit loops."""
    if method == "left":
        return [k / n for k in range(n)], [1 / n] * n
    if method == "right":
        return [k / n for k in range(1, n + 1)], [1 / n] * n
    if method == "midpoint":
        return [(k + 0.5) / n for k in range(n)], [1 / n] * n
    weights = [1 / n] * (n + 1)
    weights[0] /= 2
    weights[-1] /= 2
    return [k / n for k in range(n + 1)], weights


def moments(method, n):
    """Closed forms of sum(w a) and sum(w a**2) for each rule (sum(w) = 1 for all four)."""
    if method == "left":
        return (n - 1) / (2 * n), (n - 1) * (2 * n - 1) / (6 * n**2)
    if method == "right":
        return (n + 1) / (2 * n), (n + 1) * (2 * n + 1) / (6 * n**2)
    if method == "midpoint":
        return 0.5, 1 / 3 - 1 / (12 * n**2)
    return 0.5, 1 / 3 + 1 / (6 * n**2)


def ig_oracle(grad, x, baseline, n, method):
    """The docstring formula point by point: (x - x') * sum_k w_k grad(x' + a_k (x - x'))."""
    x, baseline = np.asarray(x, dtype=float), np.asarray(baseline, dtype=float)
    total = np.zeros_like(x)
    for a, w in zip(*rule(method, n)):
        total = total + w * grad((baseline + a * (x - baseline))[None])[0]
    return (x - baseline) * total


def poly_closed_form(c, power, x, baseline, n, method):
    """Attributions of f = sum c_i x_i**power (power 1, 2 or 3) under a rule, from its moments."""
    c, x, b = (np.asarray(v, dtype=float) for v in (c, x, baseline))
    d = x - b
    m1, m2 = moments(method, n)
    if power == 1:
        return c * d
    if power == 2:
        return 2 * c * d * (b + d * m1)
    return 3 * c * d * (b**2 + 2 * b * d * m1 + d**2 * m2)


def poly_grad(c, power):
    c = np.asarray(c, dtype=float)
    return lambda P: power * c * P ** (power - 1)


def ig_of(exp, grad_fn, x, *args, shape=None, **kwargs):
    """The learner's attributions, checked: a float array with the shape of x."""
    shape = np.shape(x) if shape is None else shape
    result = call(exp.integrated_gradients, grad_fn, copy.deepcopy(x), *copy.deepcopy(args),
                  expected="the attributions", **copy.deepcopy(kwargs))
    return array_result(result, "integrated_gradients", shape, inputs=f"an x of shape {tuple(shape)}")


def test_integrated_gradients_docstring_examples(exp):
    grad_fn = batched(lambda X: np.stack([2 * X[:, 0], np.full(len(X), 3.0)], axis=1), (2,))
    got = ig_of(exp, grad_fn, [1.0, 1.0], n_steps=4, method="left")
    assert_close(got, [0.75, 3.0], msg="integrated_gradients(grad of x0**2 + 3 x1, [1, 1], n_steps=4, method='left') "
                                       "(docstring)")
    got = ig_of(exp, grad_fn, [1.0, 1.0], n_steps=4, method="trapezoid")
    assert_close(got, [1.0, 3.0], msg="integrated_gradients(..., n_steps=4, method='trapezoid') (docstring)")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 3, 50])
def test_integrated_gradients_linear_model_is_exact(exp, method, n):
    rng = np.random.default_rng(n)
    c, x, b = rng.normal(size=4), rng.normal(size=4), rng.normal(size=4)
    got = ig_of(exp, batched(lambda P: np.broadcast_to(c, P.shape), (4,)), x, b, n_steps=n, method=method)
    assert_close(got, c * (x - b), msg=f"integrated_gradients (linear model, method={method!r}, n_steps={n}): every "
                                       "rule gives (x - x') * w exactly when the gradient is constant")


@pytest.mark.parametrize("power", [2, 3])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 2, 7])
def test_integrated_gradients_polynomial_closed_forms(exp, power, method, n):
    rng = np.random.default_rng(10 * n + power)
    c, x, b = rng.normal(size=3), rng.normal(size=3) * 2, rng.normal(size=3)
    want = poly_closed_form(c, power, x, b, n, method)
    assert np.allclose(ig_oracle(poly_grad(c, power), x, b, n, method), want, rtol=1e-12, atol=1e-12), ORACLE_BUG
    got = ig_of(exp, batched(poly_grad(c, power), (3,)), x, b, n_steps=n, method=method)
    a_mean, a2_mean = moments(method, n)
    assert_close(got, want, msg=f"integrated_gradients (f = sum c x**{power}, method={method!r}, n_steps={n}): the "
                                f"points of the rule have mean(a) = {a_mean:.6g}"
                                + (f" and mean(a**2) = {a2_mean:.6g}" if power == 3 else ""))


def test_integrated_gradients_defaults_are_trapezoid_with_50_steps(exp):
    c, x = np.array([1.0, -2.0]), np.array([1.5, 0.8])
    want = poly_closed_form(c, 3, x, np.zeros(2), 50, "trapezoid")
    for other in (poly_closed_form(c, 3, x, np.zeros(2), 50, "midpoint"),
                  poly_closed_form(c, 3, x, np.zeros(2), 49, "trapezoid"),
                  poly_closed_form(c, 3, x, np.zeros(2), 51, "trapezoid")):
        assert not np.allclose(want, other, rtol=RTOL, atol=ATOL), ORACLE_BUG
    got = ig_of(exp, batched(poly_grad(c, 3), (2,)), x)
    assert_close(got, want, msg="integrated_gradients(grad_fn, x) with the defaults: baseline zeros, n_steps=50, "
                                "method='trapezoid' (f = sum c x**3)")


def test_integrated_gradients_baseline_none_means_zeros(exp):
    c, x = np.array([0.5, 1.0, -1.0]), np.array([2.0, -1.0, 0.5])
    grad_fn = batched(poly_grad(c, 2), (3,))
    got = ig_of(exp, grad_fn, x, None, n_steps=5, method="right")
    assert_close(got, poly_closed_form(c, 2, x, np.zeros(3), 5, "right"),
                 msg="integrated_gradients(..., baseline=None): the baseline is zeros")


@pytest.mark.parametrize("method", METHODS)
def test_integrated_gradients_one_step(exp, method):
    c, x, b = np.array([1.0, 2.0]), np.array([2.0, -1.0]), np.array([0.5, 1.0])
    grad = poly_grad(c, 3)
    point = {"left": b, "right": x, "midpoint": (b + x) / 2}.get(method)
    want = (x - b) * (grad(point[None])[0] if point is not None else (grad(b[None])[0] + grad(x[None])[0]) / 2)
    got = ig_of(exp, batched(grad, (2,)), x, b, n_steps=1, method=method)
    where = {"left": "at the baseline", "right": "at x", "midpoint": "halfway",
             "trapezoid": "the mean of the gradients at the baseline and at x"}[method]
    assert_close(got, want, msg=f"integrated_gradients(..., n_steps=1, method={method!r}): one gradient {where}")


@pytest.mark.parametrize("shape", [pytest.param((2, 3), id="x-2d"), pytest.param((1, 2, 2), id="x-3d-image")])
@pytest.mark.parametrize("method", ["left", "trapezoid"])
def test_integrated_gradients_inputs_of_any_shape(exp, shape, method):
    rng = np.random.default_rng(len(shape))
    c, x, b = rng.normal(size=shape), rng.normal(size=shape), rng.normal(size=shape)
    want = poly_closed_form(c, 3, x, b, 6, method)
    got = ig_of(exp, batched(poly_grad(c, 3), shape), x, b, n_steps=6, method=method)
    assert_close(got, want, msg=f"integrated_gradients with an x of shape {shape} (grad_fn takes (n, *x.shape)), "
                                f"method={method!r}")


def _smooth_model(seed):
    """f(x) = log(1 + exp(w.x + 0.5 x0 x1)) + 0.1 (v.x)**3, and its gradient."""
    rng = np.random.default_rng(seed)
    w, v = rng.normal(size=3), rng.normal(size=3)

    def f(X):
        X = np.atleast_2d(X)
        return np.logaddexp(0.0, X @ w + 0.5 * X[:, 0] * X[:, 1]) + 0.1 * (X @ v) ** 3

    def grad(X):
        z = X @ w + 0.5 * X[:, 0] * X[:, 1]
        s = 1.0 / (1.0 + np.exp(-z))
        dz = np.broadcast_to(w, X.shape).copy()
        dz[:, 0] += 0.5 * X[:, 1]
        dz[:, 1] += 0.5 * X[:, 0]
        return s[:, None] * dz + 0.3 * (X @ v)[:, None] ** 2 * v
    return f, grad


@pytest.mark.parametrize("method", METHODS)
def test_integrated_gradients_completeness_improves_with_n_steps(exp, method):
    f, grad = _smooth_model(seed=3)
    x, b = np.array([1.2, -0.7, 2.0]), np.array([-0.3, 0.4, 0.1])
    target = float(f(x)[0] - f(b)[0])
    steps = [2, 8, 32, 128]
    oracle_err = [abs(ig_oracle(grad, x, b, n, method).sum() - target) for n in steps]
    assert all(e1 > e2 for e1, e2 in zip(oracle_err, oracle_err[1:])) and oracle_err[-1] < 0.05, ORACLE_BUG
    errors = []
    for n in steps:
        got = ig_of(exp, batched(grad, (3,)), x, b, n_steps=n, method=method)
        assert_close(got, ig_oracle(grad, x, b, n, method),
                     msg=f"integrated_gradients (smooth nonlinear model, method={method!r}, n_steps={n})")
        errors.append(abs(float(got.sum()) - target))
    if not all(e1 > e2 for e1, e2 in zip(errors, errors[1:])):
        _fail(f"integrated_gradients (method={method!r})", "expected |sum of the attributions - (f(x) - f(x'))| to "
              f"decrease as n_steps grows ({steps}), got {[f'{e:.3g}' for e in errors]}")


def test_integrated_gradients_accept_integer_lists(exp):
    c = np.array([1.0, -1.0])
    got = ig_of(exp, batched(poly_grad(c, 2), (2,)), [2, 3], [0, 1], n_steps=4, method="midpoint")
    assert_close(got, poly_closed_form(c, 2, [2.0, 3.0], [0.0, 1.0], 4, "midpoint"),
                 msg="integrated_gradients with x = [2, 3] and baseline = [0, 1] (lists of ints): float attributions")


def test_integrated_gradients_grad_fn_may_be_called_on_batches(exp):
    """The learner may call grad_fn once or several times: the points it receives cover the rule exactly."""
    seen = []
    c, x, b = np.array([1.0, 2.0, -1.0]), np.array([1.0, 2.0, 3.0]), np.zeros(3)
    inner = batched(poly_grad(c, 2), (3,))

    def grad_fn(P):
        seen.append(np.array(P, dtype=float, copy=True))
        return inner(P)
    ig_of(exp, grad_fn, x, b, n_steps=5, method="right")
    points = np.concatenate(seen) if seen else np.zeros((0, 3))
    alphas = np.sort(points[:, 0] / x[0])
    if alphas.shape != (5,) or not np.allclose(alphas, np.arange(1, 6) / 5, atol=1e-12):
        _fail("integrated_gradients(..., n_steps=5, method='right')", "expected grad_fn to be evaluated at the 5 points "
              "x' + a (x - x') for a = 1/5 .. 5/5 (once each), got a = " + _short(alphas))


def test_integrated_gradients_do_not_modify_their_inputs(exp):
    x, b = np.array([1.0, 2.0]), np.array([0.5, -0.5])
    x_before, b_before = x.copy(), b.copy()
    call(exp.integrated_gradients, batched(poly_grad([1.0, 1.0], 2), (2,)), x, b, n_steps=3,
         expected="the attributions")
    assert_unchanged(x_before, x, "x", "integrated_gradients")
    assert_unchanged(b_before, b, "baseline", "integrated_gradients")


@pytest.mark.parametrize("x,kwargs,why", [
    pytest.param([1.0, 2.0], {"n_steps": 0}, "n_steps = 0 < 1", id="n-steps-0"),
    pytest.param([1.0, 2.0], {"n_steps": -3}, "n_steps = -3 < 1", id="n-steps-negative"),
    pytest.param([1.0, 2.0], {"method": "simpson"}, "method = 'simpson' is unknown", id="method-simpson"),
    pytest.param([1.0, 2.0], {"method": "riemann_trapezoid"}, "method = 'riemann_trapezoid' is captum's name, not one "
                 "of 'left', 'right', 'midpoint', 'trapezoid'", id="method-captum-name"),
    pytest.param([1.0, 2.0], {"baseline": [0.0, 0.0, 0.0]}, "baseline of shape (3,) for an x of shape (2,)",
                 id="baseline-longer"),
    pytest.param([1.0, 2.0, 3.0], {"baseline": [0.0]}, "baseline of shape (1,) for an x of shape (3,) (NumPy could "
                 "broadcast it, but it does not have the shape of x)", id="baseline-broadcastable"),
    pytest.param([1.0, 2.0, 3.0], {"baseline": [[0.0, 0.0, 0.0]]}, "baseline of shape (1, 3) for an x of shape (3,)",
                 id="baseline-extra-axis"),
])
def test_integrated_gradients_refuse_invalid_arguments(exp, x, kwargs, why):
    assert_raises(ValueError, exp.integrated_gradients, lambda P: np.ones_like(np.asarray(P, float)), x, why=why,
                  **kwargs)


@pytest.fixture(scope="module")
def captum_ig():
    """captum's IntegratedGradients, or a skip: captum is an optional oracle (absent from the frozen environment)."""
    try:
        from captum.attr import IntegratedGradients
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"captum is not installed (optional oracle; the closed forms are tested anyway): {exc}")
    if torch is None:
        pytest.skip("captum needs PyTorch")
    return IntegratedGradients


CAPTUM_NAMES = {"left": "riemann_left", "right": "riemann_right", "midpoint": "riemann_middle"}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [2, 5, 20])
def test_integrated_gradients_match_captum(exp, captum_ig, method, n):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)).double()

    def forward(X):
        return net(X)[:, 0]

    def grad(P):
        t = torch.tensor(P, dtype=torch.float64, requires_grad=True)
        forward(t).sum().backward()
        return t.grad.numpy()
    x, b = np.array([0.5, -1.0, 2.0]), np.array([0.1, 0.2, -0.3])
    ig = captum_ig(forward)
    X, B = torch.tensor(x[None]), torch.tensor(b[None])
    if method == "trapezoid":   # captum: n + 1 points weighted 1 / (n + 1), halved at both ends
        ref = ig.attribute(X, B, n_steps=n + 1, method="riemann_trapezoid").detach().numpy()[0] * (n + 1) / n
        how = (f"captum 'riemann_trapezoid' with n_steps={n + 1}, times {n + 1}/{n} (its weights sum to "
               f"{n}/{n + 1})")
    else:
        ref = ig.attribute(X, B, n_steps=n, method=CAPTUM_NAMES[method]).detach().numpy()[0]
        how = f"captum {CAPTUM_NAMES[method]!r} with n_steps={n}"
    assert np.allclose(ref, ig_oracle(grad, x, b, n, method), rtol=1e-6, atol=1e-7), ORACLE_BUG
    got = ig_of(exp, batched(grad, (3,)), x, b, n_steps=n, method=method)
    assert_close(got, ref, msg=f"integrated_gradients(..., n_steps={n}, method={method!r}) against {how}")


# ================================================================== grad_cam (B6.18)
def cam_oracle(A, G, relu=True, normalize=True):
    """Loop over the images and the channels: w_c = mean of G[n, c]; map = sum_c w_c A[n, c]."""
    A, G = np.asarray(A, dtype=float), np.asarray(G, dtype=float)
    N, C, H, W = A.shape
    out = np.zeros((N, H, W))
    for n in range(N):
        for c in range(C):
            out[n] += G[n, c].sum() / (H * W) * A[n, c]
        if relu:
            out[n] = np.maximum(out[n], 0.0)
        if normalize:
            peak = np.abs(out[n]).max()
            if peak > 0:
                out[n] /= peak
    return out


def cam_of(exp, A, G, **kwargs):
    shape = np.shape(A)
    result = call(exp.grad_cam, copy.deepcopy(A), copy.deepcopy(G), expected="the Grad-CAM maps", **kwargs)
    N, H, W = shape[0], shape[2], shape[3]
    return array_result(result, "grad_cam", (N, H, W), inputs=f"activations of shape {tuple(shape)} (N, C, H, W)")


def _maps(seed, shape=(3, 5, 4, 6)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) + 0.3, rng.normal(size=shape) * 0.1 + 0.02


OPTIONS = [pytest.param(dict(relu=r, normalize=z), id=f"relu-{r}-normalize-{z}")
           for r in (True, False) for z in (True, False)]


def test_grad_cam_docstring_examples(exp):
    A = np.array([[[[1., 2.], [0., 1.]], [[0., 1.], [3., 0.]]]])
    G = np.array([[[[1., 1.], [1., 1.]], [[-1., -1.], [-1., -1.]]]])
    got = cam_of(exp, A, G, relu=False, normalize=False)
    assert_close(got, [[[1., 1.], [-3., 1.]]], msg="grad_cam(A, G, relu=False, normalize=False) (docstring: channel 0 "
                                                   "minus channel 1)")
    got = cam_of(exp, A, G)
    assert_close(got, [[[1., 1.], [0., 1.]]], msg="grad_cam(A, G) (docstring)")


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("shape", [pytest.param((3, 5, 4, 6), id="N3-C5-H4-W6"),
                                   pytest.param((1, 1, 1, 1), id="one-pixel"),
                                   pytest.param((2, 8, 3, 2), id="N2-C8-H3-W2")])
def test_grad_cam_matches_the_loop_over_channels(exp, options, shape):
    A, G = _maps(seed=sum(shape), shape=shape)
    got = cam_of(exp, A, G, **options)
    assert_close(got, cam_oracle(A, G, **options), msg=f"grad_cam(..., {options}) for maps of shape {shape}: w_c = "
                                                       "spatial mean of the gradients of channel c (over H and W)")


@needs_torch
@pytest.mark.parametrize("options", OPTIONS)
def test_grad_cam_matches_torch_einsum(exp, options):
    A, G = _maps(seed=42, shape=(4, 6, 5, 7))
    tA, tG = torch.tensor(A), torch.tensor(G)
    ref = torch.einsum("nc,nchw->nhw", tG.mean(dim=(2, 3)), tA)
    if options["relu"]:
        ref = torch.relu(ref)
    if options["normalize"]:
        peak = ref.abs().amax(dim=(1, 2), keepdim=True)
        ref = torch.where(peak > 0, ref / torch.where(peak > 0, peak, torch.ones_like(peak)), ref)
    ref = ref.numpy()
    assert np.allclose(ref, cam_oracle(A, G, **options), rtol=1e-12, atol=1e-12), ORACLE_BUG
    got = cam_of(exp, A, G, **options)
    assert_close(got, ref, msg=f"grad_cam(..., {options}) against torch.einsum('nc,nchw->nhw', G.mean((2, 3)), A)")


def test_grad_cam_relu_comes_after_the_weighted_sum(exp):
    # channel 0 pushes the pixel (0, 0) up by 3, channel 1 pulls it down by 2: the sum is 1 > 0
    A = np.array([[[[3.0, 0.0]], [[2.0, 1.0]]]])          # N=1, C=2, H=1, W=2
    G = np.array([[[[1.0, 1.0]], [[-1.0, -1.0]]]])        # weights +1 and -1
    got = cam_of(exp, A, G, relu=True, normalize=False)
    assert_close(got, [[[1.0, 0.0]]], msg="grad_cam(..., relu=True, normalize=False): ReLU(sum_c w_c A_c), applied "
                                          "once to the sum (not to each channel or each weight)")


def test_grad_cam_normalises_each_image_separately(exp):
    A, G = _maps(seed=1, shape=(3, 4, 3, 5))
    G[1] *= 1000.0                                          # image 1 is 1000 times stronger
    got = cam_of(exp, A, G)
    want = cam_oracle(A, G)
    assert_close(got, want, msg="grad_cam(...): each map is divided by its own largest absolute value (image 1 is "
                                "1000 times stronger than the others)")
    peaks = got.reshape(3, -1).max(axis=1)
    assert_close(peaks, np.where(want.reshape(3, -1).max(axis=1) > 0, 1.0, 0.0),
                 msg="grad_cam(...): the largest value of each normalised map is 1")


def test_grad_cam_normalize_without_relu_uses_the_largest_absolute_value(exp):
    A = np.array([[[[1.0, 0.0], [0.0, -4.0]]]])           # N=1, C=1
    G = np.ones_like(A) * 0.5
    got = cam_of(exp, A, G, relu=False, normalize=True)
    assert_close(got, [[[0.25, 0.0], [0.0, -1.0]]], msg="grad_cam(..., relu=False, normalize=True): divided by the "
                                                         "largest absolute value (4 x 0.5 = 2), so the map is in [-1, 1]")


@pytest.mark.parametrize("relu", [True, False])
def test_grad_cam_zero_gradients_give_a_zero_map(exp, relu):
    A, G = _maps(seed=2, shape=(2, 3, 4, 4))
    G[:] = 0.0
    got = cam_of(exp, A, G, relu=relu, normalize=True)
    if np.isnan(got).any():
        _fail(f"grad_cam(..., relu={relu}, normalize=True)", "expected an all-zero map when the gradients are zero, got "
              "NaN (a division by a largest value of 0: an all-zero map stays zero)")
    assert_close(got, np.zeros((2, 4, 4)), msg=f"grad_cam(..., relu={relu}): zero gradients give a zero map")


def test_grad_cam_map_that_relu_zeroes_stays_zero(exp):
    A, G = _maps(seed=3, shape=(2, 3, 4, 5))
    A[0] = np.abs(A[0])
    G[0] = -np.abs(G[0]) - 0.1                              # image 0: only negative evidence
    got = cam_of(exp, A, G)
    if np.isnan(got).any():
        _fail("grad_cam(...)", "expected a zero map for image 0 (all its evidence is negative, the ReLU removes it), "
              "got NaN (an all-zero map stays zero: do not divide by a largest value of 0)")
    assert_close(got, cam_oracle(A, G), msg="grad_cam(...): image 0 has only negative evidence (zero map after the "
                                            "ReLU), image 1 is normalised as usual")


@pytest.mark.parametrize("relu", [True, False])
@pytest.mark.parametrize("factor", [1e-3, 0.5, 4.0, 1e3])
def test_grad_cam_invariant_to_a_positive_factor_on_the_gradients(exp, relu, factor):
    A, G = _maps(seed=4, shape=(2, 4, 3, 3))
    base = cam_of(exp, A, G, relu=relu, normalize=True)
    scaled = cam_of(exp, A, G * factor, relu=relu, normalize=True)
    assert_close(scaled, base, msg=f"grad_cam(A, {factor:g} * G, relu={relu}, normalize=True) must equal "
                                   "grad_cam(A, G): the normalised map does not change when the gradients are "
                                   "multiplied by a positive number (an epsilon added to the largest value breaks "
                                   "it; test for an all-zero map instead)")


def test_grad_cam_accepts_lists(exp):
    A, G = _maps(seed=5, shape=(1, 2, 2, 3))
    got = cam_of(exp, A.tolist(), G.tolist())
    assert_close(got, cam_oracle(A, G), msg="grad_cam with nested lists (array-like)")


def test_grad_cam_does_not_modify_its_inputs(exp):
    A, G = _maps(seed=6, shape=(2, 3, 2, 2))
    A_before, G_before = A.copy(), G.copy()
    for options in ({}, {"relu": False}, {"normalize": False}):
        call(exp.grad_cam, A, G, expected="the Grad-CAM maps", **options)
    assert_unchanged(A_before, A, "activations", "grad_cam")
    assert_unchanged(G_before, G, "gradients", "grad_cam")


@pytest.mark.parametrize("A_shape,G_shape,why", [
    pytest.param((3, 4, 4), (3, 4, 4), "3-D arrays (C, H, W): the N axis is missing", id="3d-both"),
    pytest.param((1, 2, 4, 4), (1, 3, 4, 4), "2 channels of activations and 3 of gradients", id="channels-differ"),
    pytest.param((2, 2, 4, 4), (1, 2, 4, 4), "2 images of activations and 1 of gradients", id="batch-differs"),
    pytest.param((1, 2, 4, 4), (1, 2, 4), "gradients of shape (1, 2, 4) are not 4-D", id="gradients-3d"),
    pytest.param((1, 1, 2, 3, 3), (1, 1, 2, 3, 3), "5-D arrays", id="5d-both"),
])
def test_grad_cam_refuses_invalid_shapes(exp, A_shape, G_shape, why):
    assert_raises(ValueError, exp.grad_cam, np.ones(A_shape), np.ones(G_shape), why=why)


# ================================================================== group_rates (B6.23)
RATE_KEYS = ("accuracy", "base_rate", "fpr", "n", "precision", "selection_rate", "tpr")


def _ratio(num, den):
    return num / den if den else float("nan")


def rates_oracle(y_true, y_pred, groups):
    """sklearn.metrics.confusion_matrix on each subgroup (groups in np.unique order)."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        tn, fp, fn, tp = (int(v) for v in sk_confusion_matrix(y_true[m], y_pred[m], labels=[0, 1]).ravel())
        n = tn + fp + fn + tp
        out[g.item() if isinstance(g, np.generic) else g] = {
            "n": n, "base_rate": (tp + fn) / n, "selection_rate": (tp + fp) / n, "tpr": _ratio(tp, tp + fn),
            "fpr": _ratio(fp, fp + tn), "precision": _ratio(tp, tp + fp), "accuracy": (tp + tn) / n}
    return out


WHY_NAN = {"tpr": "no positive example (TP + FN = 0)", "fpr": "no negative example (FP + TN = 0)",
           "precision": "no positive prediction (TP + FP = 0)"}


def rates_of(exp, y_true, y_pred, groups):
    """The learner's dict, checked: one entry per group with the seven keys, 'n' an int, the rates floats."""
    msg = "group_rates"
    result = call(exp.group_rates, copy.deepcopy(y_true), copy.deepcopy(y_pred), copy.deepcopy(groups),
                  expected="a dict {group: {rate: value}}")
    if result is None:
        _fail(msg, "expected a dict {group: {rate: value}}, got None (did you forget the return?)")
    if not isinstance(result, dict):
        _fail(msg, f"expected a dict {{group: {{rate: value}}}}, got {type(result).__name__}")
    for key, entry in result.items():
        if not hasattr(entry, "keys"):
            _fail(msg, f"expected a dict of rates for group {key!r}, got {type(entry).__name__}")
        if sorted(entry) != sorted(RATE_KEYS):
            missing = [k for k in RATE_KEYS if k not in entry]
            extra = [k for k in entry if k not in RATE_KEYS]
            _fail(msg, f"expected the keys {', '.join(RATE_KEYS)} for group {key!r}"
                       + (f"; missing: {', '.join(missing)}" if missing else "")
                       + (f"; unexpected: {', '.join(map(repr, extra))}" if extra else ""))
        n = entry["n"]
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral):
            _fail(msg, f"expected 'n' of group {key!r} to be an int, got {type(n).__name__} ({n!r})")
        for k in RATE_KEYS:
            if k != "n" and not isinstance(entry[k], (float, np.floating)):
                _fail(msg, f"expected the rate {k!r} of group {key!r} to be a float, got {type(entry[k]).__name__} "
                           f"({entry[k]!r})")
    return result


def assert_rates(got, want, msg):
    """Same groups in the same (sorted) order, same 'n', same rates (nan where the denominator is 0)."""
    want_keys = list(want)
    for key in want_keys:
        if key not in got:
            _fail(msg, f"expected the group {key!r} as a key (the group value itself, as np.unique gives it), got the "
                       f"keys {list(got)!r}")
    if len(got) != len(want_keys):
        _fail(msg, f"expected {len(want_keys)} groups {want_keys!r}, got {len(got)}: {list(got)!r}")
    order = list(got)
    if not all(a == b for a, b in zip(order, want_keys)):
        _fail(msg, f"expected the groups in sorted order (as np.unique) {want_keys!r}, got {order!r}")
    for key in want_keys:
        g, w = got[key], want[key]
        if g["n"] != w["n"]:
            _fail(msg, f"expected n = {w['n']} for group {key!r}, got {g['n']}")
        for k in RATE_KEYS:
            if k == "n":
                continue
            value, expected = float(g[k]), w[k]
            if math.isnan(expected):
                if not math.isnan(value):
                    _fail(msg, f"expected {k} = nan for group {key!r} ({WHY_NAN[k]}: the denominator is 0), "
                               f"got {value:.6g}")
            elif math.isnan(value) or not math.isclose(value, expected, rel_tol=RATE_RTOL, abs_tol=RATE_ATOL):
                _fail(msg, f"expected {k} = {expected:.6g} for group {key!r}, got {value:.6g}")


def make(*blocks):
    """Labels, predictions and groups from blocks (group, TP, FN, FP, TN), in a shuffled order."""
    y_true, y_pred, groups = [], [], []
    for g, tp, fn, fp, tn in blocks:
        for t, p, count in ((1, 1, tp), (1, 0, fn), (0, 1, fp), (0, 0, tn)):
            y_true += [t] * count
            y_pred += [p] * count
            groups += [g] * count
    order = np.random.default_rng(len(y_true)).permutation(len(y_true))
    return ([y_true[i] for i in order], [y_pred[i] for i in order], [groups[i] for i in order])


def test_group_rates_docstring_example(exp):
    y_true = [1, 0, 1, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
    got = rates_of(exp, y_true, y_pred, groups)
    want = rates_oracle(y_true, y_pred, groups)
    assert want["a"]["tpr"] == 0.5 and want["b"]["fpr"] == 0.0, ORACLE_BUG
    assert_rates(got, want, "group_rates (docstring example)")


def test_group_rates_hand_computed(exp):
    # group 'a': TP 2, FN 1, FP 3, TN 2 (n 8); group 'b': TP 0, FN 1, FP 0, TN 2 (n 3, no positive prediction)
    y_true, y_pred, groups = make(("b", 0, 1, 0, 2), ("a", 2, 1, 3, 2))
    got = rates_of(exp, y_true, y_pred, groups)
    want = {"a": {"n": 8, "base_rate": 3 / 8, "selection_rate": 5 / 8, "tpr": 2 / 3, "fpr": 3 / 5,
                  "precision": 2 / 5, "accuracy": 4 / 8},
            "b": {"n": 3, "base_rate": 1 / 3, "selection_rate": 0.0, "tpr": 0.0, "fpr": 0.0,
                  "precision": float("nan"), "accuracy": 2 / 3}}
    oracle = rates_oracle(y_true, y_pred, groups)
    assert all(oracle[g][k] == want[g][k] or (math.isnan(oracle[g][k]) and math.isnan(want[g][k]))
               for g in want for k in want[g]), ORACLE_BUG
    assert_rates(got, want, "group_rates (group 'a': TP 2, FN 1, FP 3, TN 2; group 'b': TP 0, FN 1, FP 0, TN 2)")


@pytest.mark.parametrize("seed,n,labels", [
    pytest.param(0, 60, ["f", "m"], id="two-str-groups"),
    pytest.param(1, 200, ["18-25", "26-40", "41-65", "65+"], id="four-age-brackets"),
    pytest.param(2, 90, [3, 1, 2], id="int-groups"),
    pytest.param(3, 25, ["x", "y", "z", "w", "v"], id="five-small-groups"),
])
def test_group_rates_match_sklearn_confusion_matrix(exp, seed, n, labels):
    rng = np.random.default_rng(seed)
    groups = list(rng.choice(np.array(labels, dtype=object), size=n))
    y_true = rng.integers(0, 2, size=n)
    y_pred = np.where(rng.random(n) < 0.7, y_true, 1 - y_true)
    got = rates_of(exp, y_true, y_pred, groups)
    assert_rates(got, rates_oracle(y_true, y_pred, groups),
                 f"group_rates against sklearn.metrics.confusion_matrix on each subgroup ({n} examples)")


def test_group_rates_groups_in_sorted_order(exp):
    groups = [10, 2, 33, 2, 10, 33, 2]                     # first seen: 10, 2, 33; sorted: 2, 10, 33
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 1, 0, 1, 0, 0, 0]
    got = rates_of(exp, y_true, y_pred, groups)
    assert_rates(got, rates_oracle(y_true, y_pred, groups), "group_rates with the integer groups 10, 2, 33 (sorted as "
                                                           "numbers: 2, 10, 33)")
    groups = ["b", "c", "a", "b", "a", "c"]
    got = rates_of(exp, [1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0], groups)
    assert_rates(got, rates_oracle([1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0], groups),
                 "group_rates with the groups 'b', 'c', 'a' (in order of appearance)")


def test_group_rates_undefined_rates_are_nan(exp):
    # 'p': only positives (fpr undefined); 'q': only negatives, never predicted positive (tpr, precision undefined)
    y_true, y_pred, groups = make(("p", 2, 1, 0, 0), ("q", 0, 0, 0, 4))
    got = rates_of(exp, y_true, y_pred, groups)
    assert_rates(got, rates_oracle(y_true, y_pred, groups),
                 "group_rates (group 'p' has no negative example, group 'q' no positive example and no positive "
                 "prediction)")


def test_group_rates_single_group(exp):
    y_true, y_pred, groups = make(("all", 3, 1, 2, 4))
    got = rates_of(exp, y_true, y_pred, groups)
    assert_rates(got, rates_oracle(y_true, y_pred, groups), "group_rates with a single group (allowed)")


def test_group_rates_accept_numpy_arrays(exp):
    rng = np.random.default_rng(7)
    groups = rng.choice(np.array(["u", "v"]), size=30)
    y_true, y_pred = rng.integers(0, 2, 30), rng.integers(0, 2, 30)
    got = rates_of(exp, y_true, y_pred, groups)
    assert_rates(got, rates_oracle(y_true, y_pred, groups), "group_rates with NumPy arrays")


def test_group_rates_does_not_modify_its_inputs(exp):
    y_true, y_pred = np.array([1, 0, 1, 0, 1]), np.array([1, 1, 0, 0, 1])
    groups = np.array(["b", "a", "b", "a", "b"])
    before = [y_true.copy(), y_pred.copy(), groups.copy()]
    call(exp.group_rates, y_true, y_pred, groups, expected="a dict {group: {rate: value}}")
    for name, old, new in zip(("y_true", "y_pred", "groups"), before, (y_true, y_pred, groups)):
        assert_unchanged(old, new, name, "group_rates")


@pytest.mark.parametrize("y_true,y_pred,groups,why", [
    pytest.param([1, 0, 1], [1, 0], ["a", "a", "b"], "y_pred has 2 values, y_true 3", id="y-pred-shorter"),
    pytest.param([1, 0, 1], [1, 0, 1], ["a", "b"], "groups has 2 values, the labels 3", id="groups-shorter"),
    pytest.param([1, 2, 0], [1, 0, 0], ["a", "a", "b"], "the label 2 is not 0 or 1", id="label-2"),
    pytest.param([1, 0, 0], [1, -1, 0], ["a", "a", "b"], "the prediction -1 is not 0 or 1", id="prediction-minus-1"),
    pytest.param([1, 0.5, 0], [1, 0, 0], ["a", "a", "b"], "the label 0.5 is not 0 or 1", id="label-0.5"),
    pytest.param([1, 0, 0], [1, float("nan"), 0], ["a", "a", "b"], "the prediction nan is not 0 or 1 (remove the "
                 "missing values first)", id="prediction-nan"),
    pytest.param([float("nan"), 0, 1], [1, 0, 0], ["a", "a", "b"], "the label nan is not 0 or 1", id="label-nan"),
])
def test_group_rates_refuses_invalid_arguments(exp, y_true, y_pred, groups, why):
    assert_raises(ValueError, exp.group_rates, y_true, y_pred, groups, why=why)


@pytest.fixture(scope="module")
def fairlearn_metrics():
    """fairlearn.metrics, or a skip: fairlearn is an optional oracle (absent from the frozen environment)."""
    try:
        import fairlearn.metrics as fm
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"fairlearn is not installed (optional oracle; sklearn and hand-computed values are tested "
                    f"anyway): {exc}")
    return fm


def _fair_data(seed, n=300, labels=("a", "b", "c")):
    """Random labels and predictions where every group has positives, negatives and positive predictions."""
    rng = np.random.default_rng(seed)
    groups = rng.choice(np.array(labels), size=n)
    bias = {g: 0.2 * i for i, g in enumerate(labels)}
    y_true = (rng.random(n) < 0.4).astype(int)
    p = np.array([bias[g] for g in groups]) + np.where(y_true == 1, 0.6, 0.1)
    y_pred = (rng.random(n) < p).astype(int)
    return y_true, y_pred, groups


def test_group_rates_match_fairlearn_metric_frame(exp, fairlearn_metrics):
    from sklearn.metrics import accuracy_score, precision_score

    fm = fairlearn_metrics
    y_true, y_pred, groups = _fair_data(0)
    frame = fm.MetricFrame(
        metrics={"n": fm.count, "base_rate": lambda t, p: float(np.mean(t)), "selection_rate": fm.selection_rate,
                 "tpr": fm.true_positive_rate, "fpr": fm.false_positive_rate, "precision": precision_score,
                 "accuracy": accuracy_score},
        y_true=y_true, y_pred=y_pred, sensitive_features=groups)
    table = frame.by_group
    want = {g: {k: (int(table.loc[g, k]) if k == "n" else float(table.loc[g, k])) for k in RATE_KEYS}
            for g in table.index}
    oracle = rates_oracle(y_true, y_pred, groups)
    assert all(math.isclose(oracle[g][k], want[g][k], rel_tol=1e-12) for g in want for k in RATE_KEYS), ORACLE_BUG
    assert_rates(rates_of(exp, y_true, y_pred, groups), want, "group_rates against fairlearn.metrics.MetricFrame")


# ================================================================== demographic_parity_difference (B6.23)
def gap_of(function, *args, what):
    result = call(function, *copy.deepcopy(args), expected=f"the {what} (one number)")
    return number_result(result, function.__name__)


def assert_gap(got, want, msg):
    if not math.isclose(got, want, rel_tol=RATE_RTOL, abs_tol=RATE_ATOL):
        _fail(msg, f"expected {want:.6g}, got {got:.6g}")


def test_demographic_parity_difference_docstring_examples(exp):
    groups = ["a"] * 4 + ["b"] * 4
    got = gap_of(exp.demographic_parity_difference, [1, 0, 0, 1, 1, 1, 0, 0], groups, what="demographic parity gap")
    assert_gap(got, 0.0, "demographic_parity_difference([1, 0, 0, 1, 1, 1, 0, 0], ...) (docstring)")
    got = gap_of(exp.demographic_parity_difference, [1, 1, 1, 0, 1, 0, 0, 0], groups, what="demographic parity gap")
    assert_gap(got, 0.5, "demographic_parity_difference([1, 1, 1, 0, 1, 0, 0, 0], ...) (docstring)")


def test_demographic_parity_difference_hand_computed(exp):
    # selection rates: 'b' 2/4, 'c' 2/2, 'a' 1/4 (in order of appearance): largest - smallest = 1 - 1/4
    groups = ["b"] * 4 + ["c"] * 2 + ["a"] * 4
    y_pred = [1, 1, 0, 0] + [1, 1] + [1, 0, 0, 0]
    got = gap_of(exp.demographic_parity_difference, y_pred, groups, what="demographic parity gap")
    assert_gap(got, 0.75, "demographic_parity_difference (selection rates 'a' 1/4, 'b' 1/2, 'c' 1, groups of sizes 4, "
                          "4 and 2): largest minus smallest selection rate")


def test_demographic_parity_difference_against_group_selection_rates(exp):
    for seed in range(4):
        y_true, y_pred, groups = _fair_data(seed, n=150, labels=("g1", "g2", "g3", "g4"))
        rates = [r["selection_rate"] for r in rates_oracle(y_true, y_pred, groups).values()]
        got = gap_of(exp.demographic_parity_difference, y_pred, groups, what="demographic parity gap")
        assert_gap(got, max(rates) - min(rates), f"demographic_parity_difference (4 groups, seed {seed}): max - min of "
                                                 "the selection rates P(y_hat = 1 | group)")


def test_demographic_parity_difference_does_not_modify_its_inputs(exp):
    y_pred, groups = np.array([1, 0, 1, 1]), np.array([1, 1, 2, 2])
    before = [y_pred.copy(), groups.copy()]
    call(exp.demographic_parity_difference, y_pred, groups, expected="the demographic parity gap")
    assert_unchanged(before[0], y_pred, "y_pred", "demographic_parity_difference")
    assert_unchanged(before[1], groups, "groups", "demographic_parity_difference")


@pytest.mark.parametrize("y_pred,groups,why", [
    pytest.param([1, 0, 1], ["a", "b"], "y_pred has 3 values, groups 2", id="lengths-differ"),
    pytest.param([1, 2, 0, 0], ["a", "a", "b", "b"], "the prediction 2 is not 0 or 1", id="prediction-2"),
    pytest.param([1, float("nan"), 0, 0], ["a", "a", "b", "b"], "the prediction nan is not 0 or 1", id="prediction-nan"),
    pytest.param([1, 0, 1, 0], ["a", "a", "a", "a"], "a single group (fewer than two)", id="one-group"),
])
def test_demographic_parity_difference_refuses_invalid_arguments(exp, y_pred, groups, why):
    assert_raises(ValueError, exp.demographic_parity_difference, y_pred, groups, why=why)


def test_demographic_parity_difference_matches_fairlearn(exp, fairlearn_metrics):
    for seed in range(3):
        y_true, y_pred, groups = _fair_data(seed)
        ref = float(fairlearn_metrics.demographic_parity_difference(y_true, y_pred, sensitive_features=groups))
        got = gap_of(exp.demographic_parity_difference, y_pred, groups, what="demographic parity gap")
        assert_gap(got, ref, f"demographic_parity_difference against fairlearn (seed {seed})")


# ================================================================== equalized_odds_difference (B6.23)
def test_equalized_odds_difference_docstring_example(exp):
    got = gap_of(exp.equalized_odds_difference, [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 1, 0, 0],
                 ["a"] * 4 + ["b"] * 4, what="equalized odds gap")
    assert_gap(got, 0.5, "equalized_odds_difference (docstring example)")


@pytest.mark.parametrize("blocks,want,why", [
    pytest.param([("a", 4, 0, 1, 1), ("b", 1, 3, 1, 3)], 0.75,
                 "TPR 1 and 1/4 (gap 3/4), FPR 1/2 and 1/4 (gap 1/4): the TPR gap is the larger", id="tpr-gap-larger"),
    pytest.param([("a", 1, 1, 0, 3), ("b", 2, 2, 2, 1)], 2 / 3,
                 "TPR 1/2 and 1/2 (gap 0), FPR 0 and 2/3 (gap 2/3): the FPR gap is the larger", id="fpr-gap-larger"),
    pytest.param([("m", 3, 1, 1, 4), ("k", 1, 1, 1, 1), ("z", 9, 1, 0, 5)], 0.5,
                 "three groups: TPR 3/4, 1/2, 9/10 for 'm', 'k', 'z' (gap 0.4), FPR 1/5, 1/2, 0 (gap 0.5)",
                 id="three-groups"),
])
def test_equalized_odds_difference_hand_computed(exp, blocks, want, why):
    y_true, y_pred, groups = make(*blocks)
    rates = rates_oracle(y_true, y_pred, groups).values()
    tpr, fpr = [r["tpr"] for r in rates], [r["fpr"] for r in rates]
    assert math.isclose(max(max(tpr) - min(tpr), max(fpr) - min(fpr)), want), ORACLE_BUG
    got = gap_of(exp.equalized_odds_difference, y_true, y_pred, groups, what="equalized odds gap")
    assert_gap(got, want, f"equalized_odds_difference ({why}): the larger of the two gaps")


def test_equalized_odds_difference_against_group_rates(exp):
    for seed in range(4):
        y_true, y_pred, groups = _fair_data(seed + 10, n=200, labels=("g1", "g2", "g3", "g4"))
        rates = rates_oracle(y_true, y_pred, groups).values()
        tpr, fpr = [r["tpr"] for r in rates], [r["fpr"] for r in rates]
        got = gap_of(exp.equalized_odds_difference, y_true, y_pred, groups, what="equalized odds gap")
        assert_gap(got, max(max(tpr) - min(tpr), max(fpr) - min(fpr)),
                   f"equalized_odds_difference (4 groups, seed {seed}): max(TPR gap, FPR gap), each gap = largest "
                   "minus smallest over the groups")


def test_equalized_odds_difference_does_not_modify_its_inputs(exp):
    y_true, y_pred, groups = (np.array(v) for v in make(("a", 2, 1, 1, 2), ("b", 1, 1, 2, 2)))
    before = [y_true.copy(), y_pred.copy(), groups.copy()]
    call(exp.equalized_odds_difference, y_true, y_pred, groups, expected="the equalized odds gap")
    for name, old, new in zip(("y_true", "y_pred", "groups"), before, (y_true, y_pred, groups)):
        assert_unchanged(old, new, name, "equalized_odds_difference")


@pytest.mark.parametrize("blocks,change,why", [
    pytest.param([("a", 1, 1, 1, 1), ("b", 1, 1, 1, 1)], "short", "y_pred has one value less than y_true",
                 id="lengths-differ"),
    pytest.param([("a", 1, 1, 1, 1), ("b", 1, 1, 1, 1)], "label-2", "a label is 2", id="label-2"),
    pytest.param([("a", 1, 1, 1, 1), ("b", 1, 1, 1, 1)], "pred-nan", "a prediction is nan", id="prediction-nan"),
    pytest.param([("a", 2, 1, 1, 2)], None, "a single group (fewer than two)", id="one-group"),
    pytest.param([("a", 0, 0, 1, 2), ("b", 1, 1, 1, 1)], None, "group 'a' has no positive example (its TPR is "
                 "undefined)", id="group-without-positive"),
    pytest.param([("a", 1, 1, 1, 1), ("b", 2, 1, 0, 0)], None, "group 'b' has no negative example (its FPR is "
                 "undefined)", id="group-without-negative"),
])
def test_equalized_odds_difference_refuses_invalid_arguments(exp, blocks, change, why):
    y_true, y_pred, groups = (list(v) for v in make(*blocks))
    if change == "short":
        y_pred = y_pred[:-1]
    elif change == "label-2":
        y_true[0] = 2
    elif change == "pred-nan":
        y_pred = [float(v) for v in y_pred]
        y_pred[1] = float("nan")
    assert_raises(ValueError, exp.equalized_odds_difference, y_true, y_pred, groups, why=why)


def test_equalized_odds_difference_matches_fairlearn(exp, fairlearn_metrics):
    for seed in range(3):
        y_true, y_pred, groups = _fair_data(seed)
        ref = float(fairlearn_metrics.equalized_odds_difference(y_true, y_pred, sensitive_features=groups,
                                                                agg="worst_case"))
        got = gap_of(exp.equalized_odds_difference, y_true, y_pred, groups, what="equalized odds gap")
        assert_gap(got, ref, f"equalized_odds_difference against fairlearn (agg='worst_case', seed {seed})")
