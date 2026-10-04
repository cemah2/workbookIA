"""Tests of mylearn.info (chapter 6): oracle tests and properties.

    pytest tests/test_ch06_info.py              # your code (mon_travail/mylearn/info.py)
    pytest tests/test_ch06_info.py --impl=ref   # the reference

Oracles: NumPy logarithms, ``scipy.stats.entropy``, ``scipy.special.rel_entr``,
``scipy.spatial.distance.jensenshannon``, ``scipy.stats.gmean``, ``collections.Counter``,
an independent Huffman construction written here with ``heapq`` (expected length only),
``sklearn.metrics.log_loss`` and PyTorch (``torch.nn.functional.cross_entropy`` and
``nll_loss``). Every test name starts with the name of the function it tests
(``-k "test_entropy_"``); the tests of one exercise never call the functions of
another one (``js_divergence`` may reuse ``kl_divergence``, and ``char_distribution``
``token_distribution``, inside the module, as their docstrings say).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import heapq
import math
import warnings
from collections import Counter

import numpy as np
import pytest
from scipy import special, stats
from scipy.spatial import distance
from sklearn import metrics

try:
    import torch
except ImportError:      # PyTorch is optional on a computer (Colab has it): its oracle tests are skipped
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")


@pytest.fixture
def info(mylearn_module):
    return mylearn_module("info")


# ------------------------------------------------------------------ helpers
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


def _fail(msg: str, head: str, data: str = "", details: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data and details."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    if details:
        lines.append(details)
    raise AssertionError("\n".join(lines)) from None


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
        if have == need:   # they differ beyond the 6th digit: show every digit
            have, need = _short(result, 17), _short(expected, 17)
        _fail(msg, f"expected {need}, got {have}", data, str(exc))


def assert_python_float(value, name):
    assert isinstance(value, float), f"{name} must return a Python float, got {type(value).__name__}"


def call_without_numpy_warning(function, *args, **kwargs):
    """Call the function and fail clearly if NumPy warns (log of 0, 0 x inf...): test q == 0 BEFORE the log."""
    name = getattr(function, "__name__", "the function")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(divide="raise", invalid="raise"):
            try:
                return function(*args, **kwargs)
            except (RuntimeWarning, FloatingPointError) as exc:
                raise AssertionError(f"expected {name} to return inf without any NumPy warning (test q == 0 before "
                                     f"taking the log), got the warning: {exc}") from None


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


def random_distribution(rng, n, zeros=0):
    """A random distribution on n outcomes (Dirichlet), with `zeros` outcomes of probability exactly 0."""
    p = rng.dirichlet(np.ones(n))
    if zeros:
        p[rng.choice(n, size=zeros, replace=False)] = 0.0
        p = p / p.sum()
    return p


def reference_huffman_length(probs) -> float:
    """Expected length of an optimal prefix code (oracle): the sum of the weights of every merge."""
    heap = list(map(float, probs))
    if len(heap) == 1:
        return 1.0
    heapq.heapify(heap)
    total = 0.0
    while len(heap) > 1:
        merged = heapq.heappop(heap) + heapq.heappop(heap)
        total += merged
        heapq.heappush(heap, merged)
    return total


def reference_huffman_code(symbols, probs) -> dict:
    """A Huffman code built here, independently of yours (used only to test encode and decode)."""
    heap = [(float(w), i, [s]) for i, (s, w) in enumerate(zip(symbols, probs))]
    heapq.heapify(heap)
    code = {s: "" for s in symbols}
    order = len(heap)
    while len(heap) > 1:
        w0, _, g0 = heapq.heappop(heap)
        w1, _, g1 = heapq.heappop(heap)
        for s in g0:
            code[s] = "0" + code[s]
        for s in g1:
            code[s] = "1" + code[s]
        heapq.heappush(heap, (w0 + w1, order, g0 + g1))
        order += 1
    return code


BASES = [2.0, np.e, 10.0]
BASE_IDS = ["bits", "nats", "base-10"]
BAD_BASES = [(1.0, "base 1 (log 1 = 0)"), (0.0, "base 0"), (-2.0, "a negative base")]
BAD_BASE_IDS = ["one", "zero", "negative"]
BAD_DISTRIBUTIONS = [
    ([0.5, 0.6], "the probabilities sum to 1.1"),
    ([0.3, 0.3, 0.3], "the probabilities sum to 0.9"),
    ([0.5, 0.50001], "the probabilities sum to 1.00001 (the tolerance is 1e-6)"),
    ([1.2, -0.2], "a negative probability"),
    ([[0.5, 0.5]], "a 2-D array (a distribution is 1-D)"),
    ([0.5, float("nan")], "a NaN probability"),
]
BAD_DISTRIBUTION_IDS = ["sum-1.1", "sum-0.9", "sum-1.00001", "negative", "2-D", "nan"]


# ------------------------------------------------------------------ self_information (6.12)
@pytest.mark.parametrize("p", [1.0, 0.5, 0.25, 0.125, 1 / 6, 0.9, 1e-6])
def test_self_information_is_minus_log2_p_in_bits(info, p):
    assert_close(info.self_information(p), -np.log2(p), msg=f"the surprise of an event of probability {p!r}, "
                                                              f"in bits, is -log2(p)")


@pytest.mark.parametrize("base", BASES, ids=BASE_IDS)
def test_self_information_in_other_bases(info, base):
    p = np.array([0.5, 0.2, 0.01])
    assert_close(info.self_information(p, base=base), -np.log(p) / np.log(base),
                 msg=f"with base={base:.6g}, divide the natural log by log(base)", data=f"p={p.tolist()}")


def test_self_information_of_a_certain_event_is_zero(info):
    value = info.self_information(1.0)
    assert value == 0, f"a certain event (p = 1) carries no information: expected 0.0, got {value!r}"


def test_self_information_scalar_gives_a_python_float(info):
    for p in [0.5, np.float64(0.5), 1]:
        result = info.self_information(p)
        assert_python_float(result, "self_information")
        assert_close(result, -np.log2(float(p)), msg=f"the surprise of p = {p!r}")


@pytest.mark.parametrize("shape", [(3,), (2, 3)], ids=["1-D", "2-D"])
def test_self_information_array_keeps_the_shape(info, shape):
    p = np.linspace(0.1, 1.0, int(np.prod(shape))).reshape(shape)
    result = info.self_information(p)
    assert isinstance(result, np.ndarray), f"for an array p, expected a NumPy array, got {type(result).__name__}"
    assert_close(result, -np.log2(p), msg="one surprise per probability, with the shape of p")


def test_self_information_accepts_a_list(info):
    assert_close(info.self_information([0.5, 0.25]), [1.0, 2.0], msg="a list works like an array (np.asarray)")


def test_self_information_adds_up_for_independent_events(info):
    # property 4 of the book: the information of two unrelated events is the sum of their information
    for p, q in [(0.5, 0.25), (0.1, 0.3), (1 / 26, 1 / 26)]:
        assert_close(info.self_information(p * q), info.self_information(p) + info.self_information(q),
                     rtol=1e-12, msg="I(p q) must equal I(p) + I(q) (independent events)", data=f"p={p}, q={q}")


def test_self_information_rarer_events_are_more_surprising(info):
    values = info.self_information(np.array([0.9, 0.5, 0.1, 0.01]))
    assert np.all(np.diff(values) > 0), (f"the rarer the event, the larger its surprise: expected increasing values "
                                         f"for p = [0.9, 0.5, 0.1, 0.01], got {_short(values)}")


@pytest.mark.parametrize("p, why", [(0.0, "p = 0 (an impossible event)"), (-0.5, "a negative probability"),
                                    (1.5, "a probability above 1"), (float("nan"), "p is NaN"),
                                    ([0.5, 0.0], "an array that contains a 0")],
                         ids=["zero", "negative", "above-one", "nan", "array-with-zero"])
def test_self_information_rejects_probabilities_outside_0_1(info, p, why):
    assert_raises_value_error(info.self_information, p, why=why)


@pytest.mark.parametrize("base, why", BAD_BASES, ids=BAD_BASE_IDS)
def test_self_information_rejects_an_invalid_base(info, base, why):
    assert_raises_value_error(info.self_information, 0.5, base=base, why=why)


# ------------------------------------------------------------------ entropy (6.12)
@pytest.mark.parametrize("base", BASES, ids=BASE_IDS)
def test_entropy_matches_scipy(info, base):
    rng = np.random.default_rng(61)
    for n in [2, 3, 5, 26, 100]:
        p = random_distribution(rng, n)
        assert_close(info.entropy(p, base=base), stats.entropy(p, base=base), rtol=1e-10,
                     msg=f"entropy (base {base:.6g}) must agree with scipy.stats.entropy", data=f"n={n}")


def test_entropy_ignores_impossible_outcomes(info):
    # convention 0 log 0 = 0: an outcome of probability 0 adds nothing (and must not give NaN)
    rng = np.random.default_rng(62)
    for _ in range(5):
        p = random_distribution(rng, 8, zeros=3)
        assert_close(info.entropy(p), stats.entropy(p, base=2), rtol=1e-10,
                     msg="outcomes with p_i = 0 contribute 0 (0 log 0 = 0), not NaN", data=f"p={_short(p)}")


def test_entropy_of_a_certain_outcome_is_zero(info):
    for p in ([1.0], [0.0, 1.0, 0.0]):
        assert_close(info.entropy(p), 0.0, atol=1e-15, msg="one outcome is certain: no surprise on average",
                     data=f"p={p}")


@pytest.mark.parametrize("n", [2, 4, 6, 26])
def test_entropy_of_the_uniform_distribution_is_log2_n(info, n):
    assert_close(info.entropy(np.full(n, 1 / n)), np.log2(n), rtol=1e-12,
                 msg=f"the uniform distribution on {n} outcomes has the largest entropy, log2({n})")


def test_entropy_is_at_most_log2_n(info):
    rng = np.random.default_rng(63)
    for n in [2, 5, 30]:
        for _ in range(10):
            p = random_distribution(rng, n)
            value = info.entropy(p)
            assert -1e-12 <= value <= np.log2(n) + 1e-12, (f"expected an entropy between 0 and log2({n}) = "
                                                            f"{np.log2(n):.6g}, got {value!r}")


def test_entropy_returns_a_python_float_and_accepts_a_list(info):
    value = info.entropy([0.5, 0.5])
    assert_python_float(value, "entropy")
    assert_close(value, 1.0, msg="a fair coin carries 1 bit")


def test_entropy_accepts_a_sum_within_the_tolerance(info):
    # np.sum([0.6, 0.3, 0.1]) is 0.9999999999999999 (Python's sum() may say 1.0); 1 - 4e-7 is still within 1e-6
    for p in ([0.6, 0.3, 0.1], [0.5, 0.5 - 4e-7]):
        assert_close(info.entropy(p), stats.entropy(p, base=2), rtol=1e-5,
                     msg="a sum of 1 within the tolerance 1e-6 is a valid distribution (do not compare with == 1)",
                     data=f"p={p}, np.sum(p)={np.sum(p)!r}")


def test_entropy_does_not_modify_its_argument(info):
    p = np.array([0.25, 0.0, 0.75])
    saved = p.copy()
    info.entropy(p)
    assert np.array_equal(p, saved), f"the caller's array must not change: before {saved.tolist()}, after {p.tolist()}"


@pytest.mark.parametrize("p, why", BAD_DISTRIBUTIONS, ids=BAD_DISTRIBUTION_IDS)
def test_entropy_rejects_an_invalid_distribution(info, p, why):
    assert_raises_value_error(info.entropy, p, why=why)


@pytest.mark.parametrize("base, why", BAD_BASES, ids=BAD_BASE_IDS)
def test_entropy_rejects_an_invalid_base(info, base, why):
    assert_raises_value_error(info.entropy, [0.5, 0.5], base=base, why=why)


# ------------------------------------------------------------------ cross_entropy (6.16)
@pytest.mark.parametrize("base", BASES, ids=BASE_IDS)
def test_cross_entropy_matches_scipy(info, base):
    rng = np.random.default_rng(64)
    for n in [2, 4, 26]:
        p, q = random_distribution(rng, n), random_distribution(rng, n)
        expected = stats.entropy(p, base=base) + stats.entropy(p, q, base=base)
        assert_close(info.cross_entropy(p, q, base=base), expected, rtol=1e-10,
                     msg=f"H(p, q) = -sum p_i log q_i (base {base:.6g}) must agree with "
                         f"scipy: entropy(p) + entropy(p, q)", data=f"n={n}")


@needs_torch
def test_cross_entropy_matches_torch_for_a_one_hot_p(info):
    # PyTorch's loss for one sample: -ln q[target], q given here as probabilities (logits = log q)
    rng = np.random.default_rng(65)
    for target in range(4):
        q = random_distribution(rng, 4)
        p = np.eye(4)[target]
        expected = float(torch.nn.functional.cross_entropy(torch.log(torch.tensor(q)).unsqueeze(0),
                                                           torch.tensor([target])))
        assert_close(info.cross_entropy(p, q, base=np.e), expected, rtol=1e-10,
                     msg="with a one-hot p, the cross-entropy in nats is PyTorch's loss -ln q[target]",
                     data=f"target={target}, q={_short(q)}")


def test_cross_entropy_of_p_with_itself_is_the_entropy(info):
    rng = np.random.default_rng(66)
    p = random_distribution(rng, 6)
    assert_close(info.cross_entropy(p, p), stats.entropy(p, base=2), rtol=1e-10,
                 msg="coding p with its own code costs H(p, p) = H(p)")


def test_cross_entropy_is_never_below_the_entropy(info):
    rng = np.random.default_rng(67)
    for _ in range(20):
        p, q = random_distribution(rng, 5), random_distribution(rng, 5)
        value = info.cross_entropy(p, q)
        if value < stats.entropy(p, base=2) - 1e-12:
            _fail("a wrong code always costs more", f"expected H(p, q) >= H(p) = {stats.entropy(p, base=2):.6g}, "
                                                    f"got {value:.6g}", f"p={_short(p)}, q={_short(q)}")


def test_cross_entropy_is_infinite_when_q_rules_out_an_observed_outcome(info):
    value = call_without_numpy_warning(info.cross_entropy, [0.5, 0.5, 0.0], [1.0, 0.0, 0.0])
    assert value == float("inf"), (f"q gives probability 0 to an outcome that p produces: expected inf, got {value!r}"
                                   " (and no NaN, no ZeroDivisionError)")


def test_cross_entropy_ignores_outcomes_that_p_never_produces(info):
    # q_i = 0 is harmless where p_i = 0: 0 x log 0 counts as 0
    value = info.cross_entropy([0.5, 0.5, 0.0], [0.25, 0.75, 0.0])
    assert_close(value, 0.5 * 2 + 0.5 * np.log2(4 / 3), rtol=1e-12,
                 msg="outcomes with p_i = 0 contribute nothing, even when q_i = 0")


def test_cross_entropy_returns_a_python_float(info):
    assert_python_float(info.cross_entropy([0.5, 0.5], [0.25, 0.75]), "cross_entropy")


@pytest.mark.parametrize("p, q, why", [([0.5, 0.5], [0.2, 0.3, 0.5], "p and q have different lengths"),
                                       ([0.5, 0.6], [0.5, 0.5], "p sums to 1.1"),
                                       ([0.5, 0.5], [1.5, -0.5], "q has a negative probability"),
                                       ([0.5, 0.5], [0.5, 0.50001], "q sums to 1.00001 (the tolerance is 1e-6)")],
                         ids=["shapes", "p-sum", "q-negative", "q-sum-1.00001"])
def test_cross_entropy_rejects_invalid_inputs(info, p, q, why):
    assert_raises_value_error(info.cross_entropy, p, q, why=why)


def test_cross_entropy_rejects_an_invalid_base(info):
    assert_raises_value_error(info.cross_entropy, [0.5, 0.5], [0.5, 0.5], base=1.0, why="base 1")


# ------------------------------------------------------------------ kl_divergence (6.16)
@pytest.mark.parametrize("base", BASES, ids=BASE_IDS)
def test_kl_divergence_matches_scipy(info, base):
    rng = np.random.default_rng(68)
    for n in [2, 3, 26]:
        p, q = random_distribution(rng, n), random_distribution(rng, n)
        assert_close(info.kl_divergence(p, q, base=base), stats.entropy(p, q, base=base), rtol=1e-9, atol=1e-12,
                     msg=f"KL(p || q) = sum p_i log(p_i / q_i) (base {base:.6g}) must agree with "
                         f"scipy.stats.entropy(p, q)", data=f"n={n}")


def test_kl_divergence_matches_rel_entr(info):
    rng = np.random.default_rng(69)
    p, q = random_distribution(rng, 10, zeros=2), random_distribution(rng, 10)
    assert_close(info.kl_divergence(p, q, base=np.e), special.rel_entr(p, q).sum(), rtol=1e-10,
                 msg="in nats, KL(p || q) is the sum of scipy.special.rel_entr(p, q) (terms with p_i = 0 count 0)")


def test_kl_divergence_is_the_extra_cost_of_the_wrong_code(info):
    rng = np.random.default_rng(70)
    p, q = random_distribution(rng, 5), random_distribution(rng, 5)
    extra = (stats.entropy(p, base=2) + stats.entropy(p, q, base=2)) - stats.entropy(p, base=2)
    assert_close(info.kl_divergence(p, q), extra, rtol=1e-9, msg="KL(p || q) = H(p, q) - H(p)")


def test_kl_divergence_of_p_with_itself_is_zero(info):
    rng = np.random.default_rng(71)
    for zeros in (0, 2):
        p = random_distribution(rng, 7, zeros=zeros)
        assert_close(info.kl_divergence(p, p), 0.0, atol=1e-12, msg="KL(p || p) = 0: the right code costs nothing extra")


def test_kl_divergence_is_never_negative(info):
    rng = np.random.default_rng(72)
    for _ in range(30):
        p, q = random_distribution(rng, 4), random_distribution(rng, 4)
        value = info.kl_divergence(p, q)
        assert value >= -1e-12, f"expected KL(p || q) >= 0, got {value!r} for p={_short(p)}, q={_short(q)}"


def test_kl_divergence_is_never_negative_even_when_q_is_almost_p(info):
    # the exact KL is about 1e-18 here, but rounding gives -1e-17 for many pairs: the docstring asks for 0 then
    rng = np.random.default_rng(73)
    for _ in range(40):
        p = rng.dirichlet(np.ones(6))
        q = p * (1 + 1e-9 * rng.standard_normal(6))
        q = q / q.sum()
        value = info.kl_divergence(p, q)
        assert value >= 0.0, (f"expected KL(p || q) >= 0 even when q is almost p (return 0 when rounding makes the "
                              f"sum negative), got {value!r} for p={_short(p, 17)}, q={_short(q, 17)}")


def test_kl_divergence_is_not_symmetric(info):
    p, q = np.array([0.8, 0.2]), np.array([0.5, 0.5])
    forward, backward = info.kl_divergence(p, q), info.kl_divergence(q, p)
    assert_close([forward, backward], [stats.entropy(p, q, base=2), stats.entropy(q, p, base=2)], rtol=1e-10,
                 msg="KL(p || q) and KL(q || p) differ: the first argument weights the terms")


def test_kl_divergence_is_infinite_when_q_rules_out_an_observed_outcome(info):
    value = call_without_numpy_warning(info.kl_divergence, [0.5, 0.5], [1.0, 0.0])
    assert value == float("inf"), f"q gives probability 0 to an outcome that p produces: expected inf, got {value!r}"


def test_kl_divergence_returns_a_python_float(info):
    assert_python_float(info.kl_divergence([0.5, 0.5], [0.25, 0.75]), "kl_divergence")


@pytest.mark.parametrize("p, q, why", [([0.5, 0.5], [1.0], "p and q have different lengths"),
                                       ([0.3, 0.3], [0.5, 0.5], "p sums to 0.6"),
                                       ([0.5, 0.5], [float("nan"), 1.0], "q contains NaN")],
                         ids=["shapes", "p-sum", "q-nan"])
def test_kl_divergence_rejects_invalid_inputs(info, p, q, why):
    assert_raises_value_error(info.kl_divergence, p, q, why=why)


# ------------------------------------------------------------------ js_divergence (6.16)
@pytest.mark.parametrize("base", [2.0, np.e], ids=["bits", "nats"])
def test_js_divergence_matches_scipy(info, base):
    rng = np.random.default_rng(73)
    for n in [2, 5, 26]:
        p, q = random_distribution(rng, n, zeros=n // 3), random_distribution(rng, n)
        expected = distance.jensenshannon(p, q, base=base) ** 2   # SciPy returns the square root
        assert_close(info.js_divergence(p, q, base=base), expected, rtol=1e-8, atol=1e-12,
                     msg=f"JS(p, q) = (KL(p || m) + KL(q || m)) / 2 with m = (p + q) / 2 (base {base:.6g})",
                     data=f"n={n}")


def test_js_divergence_is_symmetric(info):
    rng = np.random.default_rng(74)
    p, q = random_distribution(rng, 6), random_distribution(rng, 6)
    assert_close(info.js_divergence(p, q), info.js_divergence(q, p), rtol=1e-12,
                 msg="JS(p, q) must equal JS(q, p)")


def test_js_divergence_is_one_bit_for_disjoint_distributions(info):
    assert_close(info.js_divergence([0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.3, 0.7]), 1.0, rtol=1e-12,
                 msg="no outcome in common: the maximum, 1 bit")


def test_js_divergence_is_zero_for_equal_distributions_and_at_most_one_bit(info):
    rng = np.random.default_rng(75)
    p = random_distribution(rng, 5)
    assert_close(info.js_divergence(p, p), 0.0, atol=1e-12, msg="JS(p, p) = 0")
    for _ in range(10):
        p, q = random_distribution(rng, 5, zeros=2), random_distribution(rng, 5, zeros=2)
        value = info.js_divergence(p, q)
        assert -1e-12 <= value <= 1 + 1e-12, f"expected 0 <= JS <= 1 bit, got {value!r}"


def test_js_divergence_stays_finite_where_kl_is_infinite(info):
    value = info.js_divergence([0.5, 0.5], [1.0, 0.0])
    assert math.isfinite(value), f"the mixture m is never 0 where p or q is not: expected a finite value, got {value!r}"
    assert_close(value, distance.jensenshannon([0.5, 0.5], [1.0, 0.0], base=2) ** 2, rtol=1e-10,
                 msg="JS([0.5, 0.5], [1, 0]) in bits")


def test_js_divergence_returns_a_python_float(info):
    assert_python_float(info.js_divergence([0.5, 0.5], [0.25, 0.75]), "js_divergence")


def test_js_divergence_rejects_invalid_inputs(info):
    assert_raises_value_error(info.js_divergence, [0.5, 0.5], [0.2, 0.3, 0.5], why="p and q have different lengths")
    assert_raises_value_error(info.js_divergence, [0.5, 0.7], [0.5, 0.5], why="p sums to 1.2")


# ------------------------------------------------------------------ token_distribution (6.13)
WORDS = "the cat and the dog and the bird saw the cat".split()


def test_token_distribution_matches_counter(info):
    counts = Counter(WORDS)
    vocab, probs = info.token_distribution(WORDS)
    expected_vocab = sorted(counts)
    assert list(vocab) == expected_vocab, f"without a vocabulary, expected the sorted distinct tokens {expected_vocab}, " \
                                          f"got {list(vocab)}"
    assert_close(probs, [counts[w] / len(WORDS) for w in expected_vocab], rtol=1e-12,
                 msg="the probability of a token is its count divided by the number of tokens")


def test_token_distribution_returns_a_list_and_a_float_array(info):
    vocab, probs = info.token_distribution(["b", "a", "b"])
    assert isinstance(vocab, list), f"the vocabulary must be returned as a list, got {type(vocab).__name__}"
    assert isinstance(probs, np.ndarray) and probs.dtype.kind == "f", \
        f"the probabilities must be a float NumPy array, got {type(probs).__name__}"


def test_token_distribution_returns_a_new_list_for_any_vocabulary(info):
    vocab, _ = info.token_distribution(["a", "b", "a"], vocabulary=("a", "b"))
    assert type(vocab) is list and vocab == ["a", "b"], \
        f"a tuple vocabulary must come back as a list: expected ['a', 'b'], got {vocab!r}"
    letters, _ = info.char_distribution("Hello", alphabet="helo")
    assert type(letters) is list and letters == ["h", "e", "l", "o"], \
        f"a string alphabet must come back as a list of characters: expected ['h', 'e', 'l', 'o'], got {letters!r}"


def test_token_distribution_follows_the_given_vocabulary(info):
    vocab, probs = info.token_distribution(WORDS, vocabulary=["dog", "the", "zebra"])
    assert list(vocab) == ["dog", "the", "zebra"], f"the given order must be kept: expected ['dog', 'the', 'zebra'], " \
                                                   f"got {list(vocab)}"
    assert_close(probs, [1 / 5, 4 / 5, 0.0], rtol=1e-12,
                 msg="tokens outside the vocabulary are ignored (5 tokens kept: 1 dog, 4 the, 0 zebra)")


def test_token_distribution_applies_laplace_smoothing(info):
    alpha = 0.5
    vocab = ["the", "cat", "zebra", "and"]
    kept = Counter(w for w in WORDS if w in vocab)
    total = sum(kept.values()) + alpha * len(vocab)
    _, probs = info.token_distribution(WORDS, vocabulary=vocab, smoothing=alpha)
    assert_close(probs, [(kept[w] + alpha) / total for w in vocab], rtol=1e-12,
                 msg="smoothing adds alpha to every count of the vocabulary: (count + alpha) / (total + alpha * V)",
                 data=f"alpha={alpha}, vocabulary={vocab}")


def test_token_distribution_large_smoothing_is_almost_uniform(info):
    _, probs = info.token_distribution(WORDS, smoothing=1e9)
    n = len(set(WORDS))
    assert_close(probs, np.full(n, 1 / n), rtol=1e-6, msg="when the smoothing dominates the counts, the distribution "
                                                           "tends to the uniform one")


def test_token_distribution_sums_to_one(info):
    rng = np.random.default_rng(76)
    tokens = rng.integers(0, 50, size=500).tolist()
    _, probs = info.token_distribution(tokens, smoothing=0.3)
    assert_close(probs.sum(), 1.0, rtol=1e-12, msg="the probabilities must sum to 1")


def test_token_distribution_accepts_a_generator(info):
    vocab, probs = info.token_distribution(word for word in WORDS)
    assert_close(probs, [Counter(WORDS)[w] / len(WORDS) for w in sorted(set(WORDS))], rtol=1e-12,
                 msg="tokens can be any iterable, even a generator that can be read only once")
    assert list(vocab) == sorted(set(WORDS)), f"expected the vocabulary {sorted(set(WORDS))}, got {list(vocab)}"


def test_token_distribution_counts_non_string_tokens(info):
    vocab, probs = info.token_distribution([3, 1, 3, 3], vocabulary=[1, 2, 3])
    assert_close(probs, [0.25, 0.0, 0.75], rtol=1e-12, msg="any hashable token works (here, integers)")


def test_token_distribution_does_not_modify_the_vocabulary(info):
    vocabulary = ["the", "cat"]
    info.token_distribution(WORDS, vocabulary=vocabulary, smoothing=1)
    assert vocabulary == ["the", "cat"], f"the caller's vocabulary must not change, got {vocabulary}"


@pytest.mark.parametrize("kwargs, why", [({"smoothing": -1}, "smoothing = -1 (a negative pseudo-count)"),
                                         ({"vocabulary": ["the", "cat", "the"]}, "the vocabulary repeats 'the'"),
                                         ({"vocabulary": ["zebra", "lion"]},
                                          "no token of the vocabulary occurs and smoothing == 0")],
                         ids=["negative-smoothing", "repeated-token", "nothing-to-count"])
def test_token_distribution_rejects_invalid_inputs(info, kwargs, why):
    assert_raises_value_error(info.token_distribution, WORDS, why=why, **kwargs)


def test_token_distribution_rejects_an_empty_sequence(info):
    assert_raises_value_error(info.token_distribution, [], why="no token at all (total count 0)")


# ------------------------------------------------------------------ char_distribution (6.13)
TEXT = "Elementary, my dear Watson!\nThe game is afoot."


def test_char_distribution_matches_counter(info):
    counts = Counter(TEXT.lower())
    alphabet, probs = info.char_distribution(TEXT)
    expected = sorted(counts)
    assert list(alphabet) == expected, (f"without an alphabet, expected every distinct character of the lower-cased "
                                        f"text, sorted (spaces, punctuation and '\\n' included): {expected}, got "
                                        f"{list(alphabet)}")
    assert_close(probs, [counts[c] / len(TEXT) for c in expected], rtol=1e-12,
                 msg="the probability of a character is its count divided by the length of the text")


def test_char_distribution_with_an_alphabet(info):
    alphabet = "aeiou"
    letters = Counter(c for c in TEXT.lower() if c in alphabet)
    total = sum(letters.values())
    got_alphabet, probs = info.char_distribution(TEXT, alphabet=alphabet)
    assert list(got_alphabet) == list(alphabet), f"expected the alphabet in its given order {list(alphabet)}, " \
                                                 f"got {list(got_alphabet)}"
    assert_close(probs, [letters[c] / total for c in alphabet], rtol=1e-12,
                 msg="only the characters of the alphabet count; the others are ignored")


def test_char_distribution_keeps_the_case_when_asked(info):
    counts = Counter("AaAb")
    alphabet, probs = info.char_distribution("AaAb", lowercase=False)
    assert list(alphabet) == sorted(counts), f"with lowercase=False, 'A' and 'a' are different characters: " \
                                             f"expected {sorted(counts)}, got {list(alphabet)}"
    assert_close(probs, [counts[c] / 4 for c in sorted(counts)], rtol=1e-12, msg="counts of 'A', 'a', 'b'")


def test_char_distribution_uses_the_alphabet_as_given(info):
    # the text is lower-cased, the alphabet is not: an upper-case letter of the alphabet never occurs
    _, probs = info.char_distribution("Abba", alphabet=["A", "b"], smoothing=1)
    assert_close(probs, [1 / 4, 3 / 4], rtol=1e-12,
                 msg="'A' never occurs in the lower-cased text (count 0 + 1), 'b' occurs twice (2 + 1)")


def test_char_distribution_applies_smoothing(info):
    alphabet = "xyz"
    _, probs = info.char_distribution("xxy", alphabet=alphabet, smoothing=1)
    assert_close(probs, [3 / 6, 2 / 6, 1 / 6], rtol=1e-12, msg="(count + 1) / (3 + 3): z gets 1/6 although it never "
                                                               "occurs")


def test_char_distribution_accepts_a_list_alphabet(info):
    alphabet, probs = info.char_distribution("aab", alphabet=["b", "a"])
    assert list(alphabet) == ["b", "a"], f"expected ['b', 'a'], got {list(alphabet)}"
    assert_close(probs, [1 / 3, 2 / 3], rtol=1e-12, msg="an alphabet given as a list of characters")


def test_char_distribution_rejects_invalid_inputs(info):
    assert_raises_value_error(info.char_distribution, "abc", smoothing=-0.5, why="a negative smoothing")
    assert_raises_value_error(info.char_distribution, "abc", alphabet="xyz",
                              why="no character of the alphabet occurs and smoothing == 0")


# ------------------------------------------------------------------ huffman_code (6.23)
def _check_prefix_free(code):
    words = sorted(code.values())
    for a, b in zip(words, words[1:]):
        if b.startswith(a):
            _fail("a Huffman code is prefix-free", f"expected no codeword to begin another, but {a!r} begins {b!r}")


def _expected_length(code, symbols, probs):
    return float(sum(p * len(code[s]) for s, p in zip(symbols, probs)))


def test_huffman_code_gives_one_binary_codeword_per_symbol(info):
    symbols = list("abcdef")
    probs = [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
    code = info.huffman_code(symbols, probs)
    assert isinstance(code, dict), f"expected a dict {{symbol: codeword}}, got {type(code).__name__}"
    assert sorted(code) == symbols, f"expected one codeword per symbol {symbols}, got keys {sorted(code)}"
    bad = {s: w for s, w in code.items() if not isinstance(w, str) or not w or set(w) - {"0", "1"}}
    assert not bad, f"each codeword must be a non-empty string of '0' and '1', got {bad}"


def test_huffman_code_is_prefix_free(info):
    rng = np.random.default_rng(77)
    for n in [2, 3, 7, 26, 60]:
        symbols = list(range(n))
        _check_prefix_free(info.huffman_code(symbols, random_distribution(rng, n)))


def test_huffman_code_has_an_optimal_expected_length(info):
    rng = np.random.default_rng(78)
    for n in [2, 3, 5, 10, 26, 100]:
        probs = random_distribution(rng, n)
        symbols = [f"s{i}" for i in range(n)]
        code = info.huffman_code(symbols, probs)
        assert_close(_expected_length(code, symbols, probs), reference_huffman_length(probs), rtol=1e-9,
                     msg=f"the expected length sum p_i len(c_i) must be the optimal one (merge the two LEAST "
                         f"probable groups at each step)", data=f"n={n}")


def test_huffman_code_satisfies_kraft_with_equality(info):
    rng = np.random.default_rng(79)
    for n in [2, 4, 9, 26]:
        code = info.huffman_code(list(range(n)), random_distribution(rng, n))
        kraft = sum(2.0 ** -len(w) for w in code.values())
        assert_close(kraft, 1.0, rtol=1e-12, msg="a Huffman code is complete: sum of 2**(-len(codeword)) = 1",
                     data=f"n={n}, lengths={sorted(len(w) for w in code.values())}")


def test_huffman_code_is_within_one_bit_of_the_entropy(info):
    rng = np.random.default_rng(80)
    for n in [2, 6, 26]:
        probs = random_distribution(rng, n)
        symbols = list(range(n))
        length = _expected_length(info.huffman_code(symbols, probs), symbols, probs)
        entropy = stats.entropy(probs, base=2)
        assert entropy - 1e-12 <= length < entropy + 1, (f"expected H <= expected length < H + 1, with H = "
                                                          f"{entropy:.6g} bits, got {length:.6g}")


def test_huffman_code_matches_minus_log2_p_for_powers_of_two(info):
    symbols = ["a", "b", "c", "d", "e"]
    probs = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 16]
    code = info.huffman_code(symbols, probs)
    lengths = [len(code[s]) for s in symbols]
    assert lengths == [1, 2, 3, 4, 4], (f"for probabilities that are powers of 1/2, each length is -log2(p): expected "
                                        f"[1, 2, 3, 4, 4], got {lengths}")


def test_huffman_code_of_a_lone_symbol(info):
    code = info.huffman_code(["x"], [1.0])
    assert code == {"x": "0"}, f"a lone symbol gets '0': expected {{'x': '0'}}, got {code}"


def test_huffman_code_with_zero_probabilities(info):
    symbols = list("abcd")
    code = info.huffman_code(symbols, [0.5, 0.5, 0.0, 0.0])
    assert sorted(code) == symbols, f"symbols of probability 0 still get a codeword: expected keys {symbols}, got " \
                                    f"{sorted(code)}"
    _check_prefix_free(code)


def test_huffman_code_with_non_string_symbols(info):
    symbols = [("t", "h"), ("h", "e"), 7]
    try:
        code = info.huffman_code(symbols, [0.5, 0.3, 0.2])
    except TypeError as exc:   # symbols of different types cannot be compared or sorted
        raise AssertionError(f"expected a code for tuple and integer symbols, got TypeError ({exc}): symbols of "
                             f"different types cannot be compared, so never sort them, and in the priority queue use "
                             f"(probability, number, group) so that two groups are never compared") from None
    assert set(code) == set(symbols), f"any hashable symbol works (tuples, numbers): expected keys {symbols}, got " \
                                      f"{list(code)}"


@pytest.mark.parametrize("symbols, probs, why", [(["a", "b"], [0.5, 0.25, 0.25], "symbols and probs differ in length"),
                                                 ([], [], "no symbol at all"),
                                                 (["a", "a"], [0.5, 0.5], "the symbol 'a' is repeated"),
                                                 (["a", "b"], [0.7, 0.7], "probs sums to 1.4"),
                                                 (["a", "b"], [1.5, -0.5], "a negative probability"),
                                                 (["a", "b"], [[0.5, 0.5]], "probs is 2-D (a row of probabilities)")],
                         ids=["lengths", "empty", "repeated", "sum", "negative", "2-D"])
def test_huffman_code_rejects_invalid_inputs(info, symbols, probs, why):
    assert_raises_value_error(info.huffman_code, symbols, probs, why=why)


# ------------------------------------------------------------------ huffman_encode (6.23)
CODE = {"a": "0", "b": "10", "c": "110", "d": "111"}


def test_huffman_encode_concatenates_the_codewords(info):
    for message in ["abcd", "aaab", "dcba", "a"]:
        bits = info.huffman_encode(message, CODE)
        expected = "".join(CODE[s] for s in message)
        assert bits == expected, f"expected the codewords one after the other {expected!r} for {message!r}, " \
                                 f"got {bits!r}"


def test_huffman_encode_accepts_a_list_of_symbols(info):
    code = {"the": "0", "cat": "10", "sat": "11"}
    assert info.huffman_encode(["the", "cat", "sat", "the"], code) == "010110", \
        "a list of words is a sequence of symbols too: expected '010110'"


def test_huffman_encode_length_is_the_sum_of_the_codeword_lengths(info):
    rng = np.random.default_rng(81)
    message = "".join(rng.choice(list("abcd"), size=300))
    bits = info.huffman_encode(message, CODE)
    expected = sum(len(CODE[s]) for s in message)
    assert isinstance(bits, str), f"expected a str of '0' and '1', got {type(bits).__name__}"
    assert len(bits) == expected, f"expected {expected} bits (the sum of the lengths of the codewords), got {len(bits)}"


def test_huffman_encode_of_an_empty_message_is_empty(info):
    assert info.huffman_encode("", CODE) == "", "an empty message gives an empty bit string"


def test_huffman_encode_rejects_a_symbol_without_codeword(info):
    assert_raises_value_error(info.huffman_encode, "abz", CODE, why="'z' has no codeword (ValueError, not KeyError)")


# ------------------------------------------------------------------ huffman_decode (6.23)
def test_huffman_decode_reads_the_bits_back(info):
    decoded = info.huffman_decode("0101101110", CODE)
    assert decoded == ["a", "b", "c", "d", "a"], f"expected ['a', 'b', 'c', 'd', 'a'], got {decoded!r}"


def test_huffman_decode_round_trip_with_an_independent_code(info):
    rng = np.random.default_rng(82)
    symbols = list("etaoinshrdlu")
    probs = random_distribution(rng, len(symbols))
    code = reference_huffman_code(symbols, probs)
    message = list(rng.choice(symbols, size=400, p=probs))
    bits = "".join(code[s] for s in message)
    decoded = info.huffman_decode(bits, code)
    assert decoded == message, (f"decoding the bits of a 400-symbol message must give it back: first symbols "
                                f"expected {message[:8]}, got {list(decoded)[:8]}")


def test_huffman_decode_returns_a_list(info):
    decoded = info.huffman_decode("10", CODE)
    assert isinstance(decoded, list), f"expected a list of symbols, got {type(decoded).__name__}"
    assert info.huffman_decode("", CODE) == [], "no bits: no symbol (an empty list)"


@pytest.mark.parametrize("bits, code, why", [("0120", CODE, "the bits contain '2'"),
                                             ("01 10", CODE, "the bits contain a space"),
                                             ("00", {"a": "0", "b": "01"}, "the code is not prefix-free ('0' "
                                                                           "begins '01'), even if these bits can be read"),
                                             ("011", CODE, "the bits end in the middle of a codeword ('11' is left)"),
                                             ("11", {"a": "0", "b": "10"}, "'11' matches no codeword")],
                         ids=["digit-2", "space", "not-prefix-free", "truncated", "no-codeword"])
def test_huffman_decode_rejects_invalid_inputs(info, bits, code, why):
    assert_raises_value_error(info.huffman_decode, bits, code, why=why)


# ------------------------------------------------------------------ perplexity (6.22)
@needs_torch
def test_perplexity_matches_torch(info):
    torch.manual_seed(0)
    logits = torch.randn(200, 30, dtype=torch.float64)
    targets = torch.randint(0, 30, (200,))
    probs = torch.softmax(logits, dim=1)[torch.arange(200), targets].numpy()
    expected = float(torch.exp(torch.nn.functional.cross_entropy(logits, targets)))
    assert_close(info.perplexity(probs), expected, rtol=1e-10,
                 msg="perplexity = exp(mean of -ln p): PyTorch's exp(cross_entropy(logits, targets))")


def test_perplexity_is_one_over_the_geometric_mean(info):
    rng = np.random.default_rng(83)
    probs = rng.uniform(0.01, 1.0, size=50)
    assert_close(info.perplexity(probs), 1 / stats.gmean(probs), rtol=1e-10,
                 msg="the perplexity is 1 / (geometric mean of the probabilities)")


@pytest.mark.parametrize("vocabulary_size", [2, 26, 50257])
def test_perplexity_of_a_uniform_model_is_the_vocabulary_size(info, vocabulary_size):
    assert_close(info.perplexity(np.full(10, 1 / vocabulary_size)), vocabulary_size, rtol=1e-10,
                 msg=f"a model that hesitates between {vocabulary_size} tokens has perplexity {vocabulary_size}")


def test_perplexity_of_a_perfect_model_is_one(info):
    assert_close(info.perplexity([1.0, 1.0, 1.0]), 1.0, rtol=1e-12, msg="probability 1 for every observed token")


def test_perplexity_returns_a_python_float(info):
    assert_python_float(info.perplexity([0.5, 0.25]), "perplexity")


@pytest.mark.parametrize("probs, why", [([], "no token at all"), ([0.5, 0.0], "a probability of 0"),
                                        ([0.5, 1.5], "a probability above 1"), ([-0.2], "a negative probability"),
                                        ([float("nan")], "NaN")],
                         ids=["empty", "zero", "above-one", "negative", "nan"])
def test_perplexity_rejects_invalid_probabilities(info, probs, why):
    assert_raises_value_error(info.perplexity, probs, why=why)


# ------------------------------------------------------------------ log_loss (6.22)
def test_log_loss_matches_sklearn_binary(info):
    rng = np.random.default_rng(84)
    y = rng.integers(0, 2, size=60)
    p = rng.uniform(0.01, 0.99, size=60)
    assert_close(info.log_loss(y, p), metrics.log_loss(y, p, labels=[0, 1]), rtol=1e-10,
                 msg="binary log loss (y_prob = probability of class 1) must agree with sklearn.metrics.log_loss")


def test_log_loss_matches_sklearn_multiclass(info):
    rng = np.random.default_rng(85)
    for k in [3, 5]:
        y = rng.integers(0, k, size=40)
        p = rng.dirichlet(np.ones(k), size=40)
        assert_close(info.log_loss(y, p), metrics.log_loss(y, p, labels=list(range(k))), rtol=1e-10,
                     msg=f"multiclass log loss ({k} classes, one row of probabilities per sample) must agree with "
                         f"sklearn.metrics.log_loss", data=f"k={k}")


@needs_torch
def test_log_loss_matches_torch_nll_loss(info):
    rng = np.random.default_rng(86)
    y = rng.integers(0, 4, size=30)
    p = rng.dirichlet(np.ones(4), size=30)
    expected = float(torch.nn.functional.nll_loss(torch.log(torch.tensor(p)), torch.tensor(y)))
    assert_close(info.log_loss(y, p), expected, rtol=1e-10, msg="the mean of -ln p(true class): torch's nll_loss of "
                                                                "the log-probabilities")


def test_log_loss_in_bits(info):
    rng = np.random.default_rng(87)
    y = rng.integers(0, 3, size=20)
    p = rng.dirichlet(np.ones(3), size=20)
    assert_close(info.log_loss(y, p, base=2), metrics.log_loss(y, p, labels=[0, 1, 2]) / np.log(2), rtol=1e-10,
                 msg="with base=2, the loss in nats is divided by ln 2")


def test_log_loss_clips_confident_mistakes(info):
    y, p = [0, 1], [1.0, 0.0]    # two certain predictions, both wrong
    value = info.log_loss(y, p)
    assert math.isfinite(value), f"a confident mistake must give a large but finite loss, got {value!r}"
    assert_close(value, metrics.log_loss(y, p, labels=[0, 1]), rtol=1e-10,
                 msg="clip the probabilities to [eps, 1 - eps] with eps = np.finfo(float).eps, as scikit-learn")


def test_log_loss_of_perfect_predictions_is_almost_zero(info):
    value = info.log_loss([0, 1, 2], np.eye(3))
    assert 0 <= value < 1e-12, f"certain and right predictions: expected a loss of about 0, got {value!r}"


def test_log_loss_returns_a_python_float(info):
    assert_python_float(info.log_loss([0, 1], [0.2, 0.7]), "log_loss")


def test_log_loss_does_not_modify_its_arguments(info):
    y = np.array([0, 1, 1])
    p = np.array([0.0, 1.0, 0.5])
    saved_y, saved_p = y.copy(), p.copy()
    info.log_loss(y, p)
    assert np.array_equal(y, saved_y) and np.array_equal(p, saved_p), \
        f"the caller's arrays must not change (clip a copy): y_prob became {p.tolist()}"


@pytest.mark.parametrize("y, p, why", [([0, 1, 1], [0.2, 0.7], "y_true and y_prob have different lengths"),
                                       ([0, 2], [0.2, 0.7], "the binary label 2 is out of range"),
                                       ([0, 3], [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]], "label 3 with 3 classes (0 to 2)"),
                                       ([0, 1], [0.2, 1.3], "a probability above 1"),
                                       ([0, 1], [[0.5, 0.4], [0.3, 0.7]], "a row sums to 0.9"),
                                       ([0, 1], [-0.1, 0.5], "a negative probability"),
                                       ([0, 1], [0.5, float("nan")], "a NaN probability"),
                                       ([0, 1.5], [0.2, 0.7], "the label 1.5 is not a whole number"),
                                       ([0, 1], [[0.5, 0.5001], [0.2, 0.8]], "a row sums to 1.0001 (tolerance 1e-6)")],
                         ids=["lengths", "binary-label-2", "label-out-of-range", "above-one", "row-sum", "negative",
                              "nan", "label-1.5", "row-sum-1.0001"])
def test_log_loss_rejects_invalid_inputs(info, y, p, why):
    assert_raises_value_error(info.log_loss, y, p, why=why)


def test_log_loss_rejects_an_invalid_base(info):
    assert_raises_value_error(info.log_loss, [0, 1], [0.2, 0.7], base=1.0, why="base 1")
