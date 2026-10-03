"""Tests of mylearn.bandit (chapter 11): oracle tests and properties.

    pytest tests/test_ch11_bandit.py              # your code (mon_travail/mylearn/bandit.py)
    pytest tests/test_ch11_bandit.py --impl=ref   # the reference

A bandit is random: most tests are statistical, with fixed seeds and generous margins (a correct
implementation fails them with a probability below one in a million). Oracles: NumPy (the
formulas of the incremental update and of UCB, the means of many pulls), SciPy (``chisquare`` for
uniform choices, ``kstest`` against ``scipy.stats.norm`` for the Gaussian rewards, Monte Carlo
draws of ``scipy.stats.beta`` for Thompson sampling) and small environments and policies written
in this file for ``run_bandit``.
Every test name starts with the name of what it tests, so that each exercise runs its own group:
``-k "test_bernoulli_bandit_ or test_gaussian_bandit_"`` (11.19),
``"test_argmax_random_tie_ or test_epsilon_greedy_action_ or test_incremental_update_"`` (11.20),
``"test_run_bandit_"`` (11.21), ``"test_ucb_action_ or test_thompson_action_"`` (11.23).
The tests of one exercise never call the functions of another one: the tests of ``run_bandit``
use environments and policies written here (your ``run_bandit`` may use your
``incremental_update``, as its docstring says).
The first line of every failure message says what was expected (it is the line that
``pytest -rf`` shows).
"""

import math

import numpy as np
import pytest
from scipy import stats


@pytest.fixture
def bdt(mylearn_module):
    return mylearn_module("bandit")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for the first line of a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    if arr.dtype.kind in "biuf":
        text = np.array2string(arr, precision=digits, separator=", ", threshold=12,
                               edgeitems=3, max_line_width=10**6)
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
    if not np.allclose(got, want, rtol=rtol, atol=atol, equal_nan=True):
        bad = np.flatnonzero(~np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True))
        where = f" (first difference at index {int(bad[0])})" if got.ndim and bad.size else ""
        have, need = _short(result), _short(expected)
        if have == need:
            have, need = _short(result, 17), _short(expected, 17)
        _fail(msg, f"expected {need}, got {have}{where}", data)


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
    if name == "<lambda>":
        name = "the call"
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
        name = getattr(function, "__qualname__", getattr(function, "__name__", "the function"))
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def python_int(value, what):
    """The value must be a Python int (the docstring examples print 1, not np.int64(1))."""
    if type(value) is not int:
        raise AssertionError(f"expected {what} to be a Python int (convert with int(...)): the docstring "
                             f"examples print 1, not np.int64(1); got {value!r} of type {type(value).__name__}")
    return value


def python_float(value, what):
    """The value must be a Python float (the docstring examples print 1.0, not np.float64(1.0))."""
    if type(value) is not float:
        raise AssertionError(f"expected {what} to be a Python float (convert with float(...)): the docstring "
                             f"examples print 1.0, not np.float64(1.0); got {value!r} of type {type(value).__name__}")
    return value


def chisquare_p(counts, expected_probs):
    """p-value of the chi-square goodness-of-fit test of observed counts against expected probabilities."""
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(expected_probs, dtype=float)
    return float(stats.chisquare(counts, counts.sum() * probs / probs.sum()).pvalue)


def frequencies(draw, n, size):
    """How many times each of the `size` values 0..size-1 is returned by `draw()` in n calls."""
    counts = np.zeros(size, dtype=int)
    for _ in range(n):
        value = draw()
        if not 0 <= value < size:
            raise AssertionError(f"expected an index in 0..{size - 1}, got {value!r}")
        counts[value] += 1
    return counts


# ------------------------------------------------------------------ 11.19 BernoulliBandit and GaussianBandit
def test_bernoulli_bandit_docstring_example(bdt):
    bandit = bdt.BernoulliBandit([0.2, 0.5, 0.9], random_state=0)
    assert (bandit.n_arms, bandit.best_arm, bandit.best_mean) == (3, 2, 0.9)   # given by __init__
    reward = call(bdt.BernoulliBandit([0.0, 1.0]).pull, 1, expected="pull(1) to return 1.0")
    if reward != 1.0:
        raise AssertionError(f"expected pull(1) == 1.0 for an arm that always pays (probability 1), got {reward!r}")
    python_float(reward, "the reward")


@pytest.mark.parametrize("probs", [[0.1, 0.5, 0.93], [0.0, 1.0, 0.3, 0.7]], ids=["three-arms", "with-0-and-1"])
def test_bernoulli_bandit_law_of_large_numbers(bdt, probs):
    n = 40_000
    bandit = bdt.BernoulliBandit(probs, random_state=11)
    for arm, p in enumerate(probs):
        rewards = np.array([call(bandit.pull, arm, expected="a reward") for _ in range(n)], dtype=float)
        if not np.isin(rewards, [0.0, 1.0]).all():
            raise AssertionError(f"expected rewards 0.0 or 1.0 only, got {_short(np.unique(rewards))} for arm {arm}")
        se = math.sqrt(p * (1 - p) / n)
        mean = float(rewards.mean())
        if abs(mean - p) > 5 * se + 1e-12:
            raise AssertionError(f"expected the mean of {n} pulls of arm {arm} to be close to probs[{arm}] = {p} "
                                 f"(within 5 standard errors, ±{5 * se:.4f}), got {mean:.4f}: arm a pays 1 with "
                                 f"probability probs[a]")


def test_bernoulli_bandit_pull_returns_python_floats(bdt):
    bandit = bdt.BernoulliBandit([0.5, 0.5], random_state=3)
    for arm in (0, 1, 0, 1):
        python_float(call(bandit.pull, arm, expected="a reward"), "the reward")


def test_bernoulli_bandit_same_seed_same_rewards(bdt):
    probs, arms = [0.3, 0.6, 0.45], [0, 1, 2, 2, 1, 0] * 30
    one = bdt.BernoulliBandit(probs, random_state=21)
    two = bdt.BernoulliBandit(probs, random_state=21)
    seq_one = [one.pull(a) for a in arms]
    np.random.seed(0)
    np.random.random(5)                                   # the global NumPy generator must play no role
    seq_two = [two.pull(a) for a in arms]
    if seq_one != seq_two:
        raise AssertionError("expected the same rewards from two bandits with the same random_state: draw with "
                             "the internal generator self._rng (created by __init__), never with np.random.* "
                             "or a new generator at each pull")
    other = bdt.BernoulliBandit(probs, random_state=22)
    if [other.pull(a) for a in arms] == seq_one:
        raise AssertionError("expected another random_state to give other rewards (180 pulls identical): the "
                             "rewards must come from the internal generator self._rng")


@pytest.mark.parametrize("arm, why", [(3, "arm == n_arms"), (-1, "negative arm"), (10, "arm far too large")],
                         ids=["n-arms", "negative", "too-large"])
def test_bernoulli_bandit_rejects_bad_arms(bdt, arm, why):
    bandit = bdt.BernoulliBandit([0.2, 0.5, 0.9], random_state=0)
    assert_raises(IndexError, bandit.pull, arm, why=why)


@pytest.mark.parametrize("means, std", [([0.0, 1.5, -0.5], 1.0), ([2.0, -3.0], 2.5)], ids=["std-1", "std-2.5"])
def test_gaussian_bandit_law_of_large_numbers(bdt, means, std):
    n = 30_000
    bandit = bdt.GaussianBandit(means, std=std, random_state=5)
    for arm, mu in enumerate(means):
        rewards = np.array([call(bandit.pull, arm, expected="a reward") for _ in range(n)], dtype=float)
        mean, sd = float(rewards.mean()), float(rewards.std(ddof=1))
        se_mean, se_sd = std / math.sqrt(n), std / math.sqrt(2 * n)
        if abs(mean - mu) > 5 * se_mean:
            raise AssertionError(f"expected the mean of {n} pulls of arm {arm} to be close to means[{arm}] = {mu} "
                                 f"(within ±{5 * se_mean:.4f}), got {mean:.4f}")
        if abs(sd - std) > 5 * se_sd:
            raise AssertionError(f"expected the standard deviation of {n} pulls of arm {arm} to be close to std = "
                                 f"{std} (within ±{5 * se_sd:.4f}), got {sd:.4f}: rng.normal takes the standard "
                                 f"deviation (not the variance) as second argument")
        p_value = stats.kstest(rewards, "norm", args=(mu, std)).pvalue
        if p_value < 1e-6:
            raise AssertionError(f"expected rewards drawn from N(means[a], std²) (Kolmogorov-Smirnov test against "
                                 f"scipy.stats.norm), got a p-value of {p_value:.2g} for arm {arm}")


def test_gaussian_bandit_pull_returns_python_floats(bdt):
    bandit = bdt.GaussianBandit([0.0, 1.5, -0.5], random_state=0)
    for arm in (0, 1, 2):
        python_float(call(bandit.pull, arm, expected="a reward"), "the reward")


def test_gaussian_bandit_same_seed_same_rewards(bdt):
    arms = [0, 1, 2, 1] * 40
    one = bdt.GaussianBandit([0.0, 1.5, -0.5], random_state=8)
    two = bdt.GaussianBandit([0.0, 1.5, -0.5], random_state=8)
    seq_one = [one.pull(a) for a in arms]
    np.random.seed(1)
    np.random.random(3)                                   # the global NumPy generator must play no role
    seq_two = [two.pull(a) for a in arms]
    if seq_one != seq_two:
        raise AssertionError("expected the same rewards from two bandits with the same random_state: draw with "
                             "the internal generator self._rng, never with np.random.* or a new generator")
    other = bdt.GaussianBandit([0.0, 1.5, -0.5], random_state=9)
    if [other.pull(a) for a in arms] == seq_one:
        raise AssertionError("expected another random_state to give other rewards")


@pytest.mark.parametrize("arm, why", [(2, "arm == n_arms"), (-1, "negative arm")], ids=["n-arms", "negative"])
def test_gaussian_bandit_rejects_bad_arms(bdt, arm, why):
    bandit = bdt.GaussianBandit([0.0, 1.5], random_state=0)
    assert_raises(IndexError, bandit.pull, arm, why=why)


# ------------------------------------------------------------------ 11.20 argmax_random_tie
def test_argmax_random_tie_docstring_example(bdt):
    result = call(bdt.argmax_random_tie, [1.0, 3.0, 2.0], expected="the index 1")
    if result != 1:
        raise AssertionError(f"expected 1 (the index of the unique maximum 3.0), got {result!r}")
    python_int(result, "the index")


def test_argmax_random_tie_always_returns_a_maximiser(bdt):
    rng = np.random.default_rng(31)
    for case in range(200):
        values = rng.integers(0, 4, size=rng.integers(1, 9)).astype(float)     # many ties
        result = call(bdt.argmax_random_tie, values, rng=np.random.default_rng(case))
        if not (isinstance(result, (int, np.integer)) and 0 <= result < values.size and values[result] == values.max()):
            raise AssertionError(f"expected the index of a maximum of {values.tolist()} (maximum "
                                 f"{values.max():g}), got {result!r}")


def test_argmax_random_tie_chooses_uniformly_among_ties(bdt):
    values = [1.0, 3.0, 3.0, 0.0, 3.0]
    rng = np.random.default_rng(32)
    counts = frequencies(lambda: bdt.argmax_random_tie(values, rng=rng), 30_000, len(values))
    if counts[0] or counts[3]:
        raise AssertionError(f"expected only the maxima 1, 2 and 4 of {values}, got the counts {counts.tolist()}")
    p_value = chisquare_p(counts[[1, 2, 4]], [1, 1, 1])
    if p_value < 1e-6:
        raise AssertionError(f"expected each of the 3 tied maxima to be chosen about 1/3 of the time, got the counts "
                             f"{counts[[1, 2, 4]].tolist()} out of 30000 (chi-square p-value {p_value:.2g}): "
                             f"np.argmax always returns the first maximum")


def test_argmax_random_tie_infinite_and_integer_values(bdt):
    rng = np.random.default_rng(33)
    counts = frequencies(lambda: bdt.argmax_random_tie([np.inf, 0.0, np.inf], rng=rng), 6_000, 3)
    if counts[1] or min(counts[0], counts[2]) < 2_500:
        raise AssertionError(f"expected the two +inf to be chosen half of the time each, got the counts "
                             f"{counts.tolist()}")
    result = call(bdt.argmax_random_tie, [2, 7, 1], rng=rng)
    if result != 1:
        raise AssertionError(f"expected 1 for the integer values [2, 7, 1], got {result!r}")


def test_argmax_random_tie_uses_the_given_generator(bdt):
    values = [5.0, 5.0, 5.0, 5.0]
    first = [bdt.argmax_random_tie(values, rng=g) for g in [np.random.default_rng(40)] for _ in range(50)]
    g1, g2 = np.random.default_rng(41), np.random.default_rng(41)
    seq1 = [bdt.argmax_random_tie(values, rng=g1) for _ in range(50)]
    np.random.seed(2)
    seq2 = [bdt.argmax_random_tie(values, rng=g2) for _ in range(50)]
    if seq1 != seq2:
        raise AssertionError("expected the same choices from two generators with the same seed: break the ties "
                             "with the rng argument (rng.choice or rng.integers), not with np.random.*")
    if len(set(first)) == 1:
        raise AssertionError(f"expected the 4 tied values to be chosen at random, got index {first[0]} 50 times")
    result = call(bdt.argmax_random_tie, values, expected="an index with rng=None")
    if result not in (0, 1, 2, 3):
        raise AssertionError(f"expected an index in 0..3 with rng=None (np.random.default_rng()), got {result!r}")


def test_argmax_random_tie_does_not_modify_its_input(bdt):
    values = np.array([0.5, 2.0, 2.0, -1.0])
    keep = values.copy()
    call(bdt.argmax_random_tie, values, rng=np.random.default_rng(0))
    if not np.array_equal(values, keep):
        raise AssertionError(f"expected values to stay {keep.tolist()}, got {values.tolist()}: work on a copy")


@pytest.mark.parametrize("values, why", [([], "empty values"), ([1.0, np.nan, 0.5], "NaN in values")],
                         ids=["empty", "nan"])
def test_argmax_random_tie_rejects_invalid_inputs(bdt, values, why):
    assert_raises(ValueError, bdt.argmax_random_tie, values, why=why)


# ------------------------------------------------------------------ 11.20 epsilon_greedy_action
def test_epsilon_greedy_action_docstring_example(bdt):
    result = call(bdt.epsilon_greedy_action, [0.2, 0.9, 0.5], epsilon=0.0, expected="the action 1")
    if result != 1:
        raise AssertionError(f"expected 1 with epsilon=0 (the greedy action, value 0.9), got {result!r}")
    python_int(result, "the action")


@pytest.mark.parametrize("epsilon, q", [(0.1, [0.2, 0.9, 0.5, 0.1]), (0.5, [1.0, -1.0, 0.0, 2.0, 0.5])],
                         ids=["eps-0.1-K4", "eps-0.5-K5"])
def test_epsilon_greedy_action_best_action_frequency(bdt, epsilon, q):
    n, k = 30_000, len(q)
    best = int(np.argmax(q))
    rng = np.random.default_rng(50)
    counts = frequencies(lambda: bdt.epsilon_greedy_action(q, epsilon, rng=rng), n, k)
    p_best = 1 - epsilon + epsilon / k
    se = math.sqrt(p_best * (1 - p_best) / n)
    freq = counts[best] / n
    if abs(freq - p_best) > 5 * se:
        raise AssertionError(f"expected the best action to be chosen with frequency 1 - epsilon + epsilon/K = "
                             f"{p_best:.4f} (±{5 * se:.4f}), got {freq:.4f}: explore with probability epsilon, "
                             f"among ALL the actions (the best one included)")
    others = np.delete(counts, best)
    p_value = chisquare_p(others, np.ones(k - 1))
    if p_value < 1e-6:
        raise AssertionError(f"expected the other actions to be equally likely (epsilon/K each), got the counts "
                             f"{others.tolist()} (chi-square p-value {p_value:.2g})")


def test_epsilon_greedy_action_epsilon_zero_is_greedy_with_random_ties(bdt):
    rng = np.random.default_rng(51)
    counts = frequencies(lambda: bdt.epsilon_greedy_action([0.3, 0.8, 0.8, 0.1], 0.0, rng=rng), 20_000, 4)
    if counts[0] or counts[3]:
        raise AssertionError(f"expected only the two maxima 1 and 2 with epsilon=0, got the counts {counts.tolist()}")
    p_value = chisquare_p(counts[[1, 2]], [1, 1])
    if p_value < 1e-6:
        raise AssertionError(f"expected the two tied maxima half of the time each with epsilon=0, got the counts "
                             f"{counts[[1, 2]].tolist()}: exploit with argmax_random_tie, not np.argmax")


def test_epsilon_greedy_action_epsilon_one_is_uniform(bdt):
    rng = np.random.default_rng(52)
    counts = frequencies(lambda: bdt.epsilon_greedy_action([0.0, 5.0, 1.0], 1.0, rng=rng), 30_000, 3)
    p_value = chisquare_p(counts, [1, 1, 1])
    if p_value < 1e-6:
        raise AssertionError(f"expected uniform choices with epsilon=1 (each action 1/3 of the time, the best one "
                             f"included), got the counts {counts.tolist()} (chi-square p-value {p_value:.2g})")


def test_epsilon_greedy_action_uses_the_given_generator(bdt):
    q = [0.2, 0.9, 0.5, 0.9]
    g1, g2 = np.random.default_rng(53), np.random.default_rng(53)
    seq1 = [bdt.epsilon_greedy_action(q, 0.3, rng=g1) for _ in range(100)]
    np.random.seed(3)
    seq2 = [bdt.epsilon_greedy_action(q, 0.3, rng=g2) for _ in range(100)]
    if seq1 != seq2:
        raise AssertionError("expected the same actions from two generators with the same seed: draw with the rng "
                             "argument, never with np.random.* or a new generator")
    python_int(seq1[0], "the action")
    result = call(bdt.epsilon_greedy_action, q, 0.3, expected="an action with rng=None")
    if result not in (0, 1, 2, 3):
        raise AssertionError(f"expected an action in 0..3 with rng=None, got {result!r}")


@pytest.mark.parametrize("q, epsilon, why", [
    ([0.1, 0.2], -0.1, "epsilon < 0"), ([0.1, 0.2], 1.5, "epsilon > 1"), ([0.1, 0.2], float("nan"), "epsilon NaN"),
    ([], 0.1, "empty q_values"),
], ids=["negative", "above-one", "nan", "empty"])
def test_epsilon_greedy_action_rejects_invalid_inputs(bdt, q, epsilon, why):
    assert_raises(ValueError, bdt.epsilon_greedy_action, q, epsilon, rng=np.random.default_rng(0), why=why)


# ------------------------------------------------------------------ 11.20 incremental_update
def test_incremental_update_docstring_example(bdt):
    result = call(bdt.incremental_update, 2.0, 4.0, step_size=0.5, expected="3.0")
    assert_close(result, 3.0, msg="incremental_update(2.0, 4.0, step_size=0.5)")
    if not isinstance(result, float):
        raise AssertionError(f"expected a float for a float estimate, got {result!r} of type {type(result).__name__}")


def test_incremental_update_matches_the_formula(bdt):
    rng = np.random.default_rng(60)
    for _ in range(50):
        estimate, target, step = float(rng.normal()), float(rng.normal() * 3), float(rng.uniform(0.01, 1.0))
        result = call(bdt.incremental_update, estimate, target, step)
        assert_close(result, estimate + step * (target - estimate),
                     msg=f"estimate={estimate:.4g}, target={target:.4g}, step_size={step:.4g}: estimate + "
                         f"step_size * (target - estimate)")


def test_incremental_update_with_one_over_n_is_the_running_mean(bdt):
    rewards = np.random.default_rng(61).normal(1.0, 2.0, size=60)
    estimate = 0.0
    for n, reward in enumerate(rewards, start=1):
        estimate = call(bdt.incremental_update, estimate, float(reward), 1.0 / n)
        assert_close(estimate, rewards[:n].mean(), rtol=1e-10, atol=1e-12,
                     msg=f"after {n} rewards with step_size = 1/n, the estimate must be np.mean of the rewards")


def test_incremental_update_constant_step_weights_recent_rewards(bdt):
    rewards = np.random.default_rng(62).uniform(0, 1, size=40)
    alpha, q0 = 0.2, 5.0
    estimate = q0
    for reward in rewards:
        estimate = call(bdt.incremental_update, estimate, float(reward), alpha)
    n = rewards.size
    weights = alpha * (1 - alpha) ** (n - 1 - np.arange(n))
    expected = (1 - alpha) ** n * q0 + np.sum(weights * rewards)
    assert_close(estimate, expected, rtol=1e-10,
                 msg="a constant step gives (1-α)^n Q0 + Σ α(1-α)^(n-i) R_i, an exponential recency-weighted average")


def test_incremental_update_step_one_gives_the_target(bdt):
    assert_close(call(bdt.incremental_update, 7.5, -2.0, 1.0), -2.0, msg="with step_size = 1 the estimate becomes "
                                                                      "the target")


def test_incremental_update_works_on_arrays(bdt):
    estimate = np.array([0.0, 1.0, 2.0, -4.0])
    target = np.array([1.0, 1.0, 0.0, 4.0])
    keep = estimate.copy()
    result = call(bdt.incremental_update, estimate, target, 0.25, expected="an array")
    assert_close(result, keep + 0.25 * (target - keep), msg="element by element on arrays")
    if not np.array_equal(estimate, keep):
        raise AssertionError(f"expected the estimate array to stay {keep.tolist()} (return a new array), got "
                             f"{estimate.tolist()}: += changes the caller's array in place")


@pytest.mark.parametrize("step, why", [(0.0, "step_size = 0"), (-0.1, "negative step_size"),
                                       (1.5, "step_size > 1")], ids=["zero", "negative", "above-one"])
def test_incremental_update_rejects_invalid_step_sizes(bdt, step, why):
    assert_raises(ValueError, bdt.incremental_update, 1.0, 2.0, step, why=why)


# ------------------------------------------------------------------ 11.21 run_bandit
class ScriptedBandit:
    """A bandit whose rewards are written in advance: arm a pays rewards[a][k] at its k-th pull (then repeats)."""

    def __init__(self, means, rewards):
        self.means = np.asarray(means, dtype=float)
        self.n_arms = len(means)
        self.best_arm = int(np.argmax(self.means))
        self.best_mean = float(self.means.max())
        self.rewards = [list(r) for r in rewards]
        self.pulls = []                                   # the arms pulled, in order

    def pull(self, arm):
        k = sum(1 for a in self.pulls if a == arm)
        self.pulls.append(arm)
        series = self.rewards[arm]
        return float(series[k % len(series)])


class NoisyBandit:
    """A Gaussian bandit with its own seeded generator (written here, not the learner's GaussianBandit)."""

    def __init__(self, means, seed):
        self.means = np.asarray(means, dtype=float)
        self.n_arms = len(means)
        self.best_arm = int(np.argmax(self.means))
        self.best_mean = float(self.means.max())
        self._rng = np.random.default_rng(seed)

    def pull(self, arm):
        return float(self._rng.normal(self.means[arm], 1.0))


def cycle_policy(q, n, t, g):
    """Arms 0, 1, 2, 0, 1, 2... (t starts at 1)."""
    return (t - 1) % len(q)


def history_of(bdt, bandit, policy, **kwargs):
    history = call(bdt.run_bandit, bandit, policy, expected="a history dictionary", **kwargs)
    if not isinstance(history, dict):
        raise AssertionError(f"expected run_bandit to return a dict, got {type(history).__name__}")
    keys = {"actions", "rewards", "optimal", "regret", "q_values", "counts"}
    if set(history) != keys:
        raise AssertionError(f"expected the keys {sorted(keys)}, got {sorted(history)}")
    return history


def test_run_bandit_docstring_example(bdt):
    bandit = ScriptedBandit([0.25, 0.75], [[1, 0, 0, 1], [1]])
    history = history_of(bdt, bandit, lambda q, n, t, g: 0, n_steps=4, rng=np.random.default_rng(0))
    if np.asarray(history["counts"]).tolist() != [4, 0]:
        raise AssertionError(f"expected counts [4, 0] (arm 0 pulled 4 times), got {_short(history['counts'])}")
    assert_close(history["regret"], [0.5, 1.0, 1.5, 2.0],
                 msg="regret = cumulative sum of best_mean - means[a_t], with means [0.25, 0.75] and arm 0 always")


def test_run_bandit_shapes_and_types(bdt):
    bandit = ScriptedBandit([0.2, 0.5, 0.8], [[0, 1], [1, 0, 1], [1]])
    history = history_of(bdt, bandit, cycle_policy, n_steps=9, rng=np.random.default_rng(0))
    expected = {"actions": ((9,), "i"), "rewards": ((9,), "f"), "optimal": ((9,), "b"), "regret": ((9,), "f"),
                "q_values": ((3,), "f"), "counts": ((3,), "i")}
    for key, (shape, kind) in expected.items():
        value = np.asarray(history[key])
        if not isinstance(history[key], np.ndarray) or value.shape != shape or value.dtype.kind != kind:
            names = {"i": "int", "f": "float", "b": "bool"}
            raise AssertionError(f"expected history[{key!r}] to be a NumPy array of shape {shape} and dtype "
                                 f"{names[kind]}, got {type(history[key]).__name__} of shape {value.shape} and "
                                 f"dtype {value.dtype}")


def test_run_bandit_records_actions_rewards_and_counts(bdt):
    bandit = ScriptedBandit([0.2, 0.5, 0.8], [[0, 1], [1, 0, 1], [1, 1, 0]])
    history = history_of(bdt, bandit, cycle_policy, n_steps=8, rng=np.random.default_rng(0))
    if np.asarray(history["actions"]).tolist() != [0, 1, 2, 0, 1, 2, 0, 1]:
        raise AssertionError(f"expected the actions [0, 1, 2, 0, 1, 2, 0, 1] of the policy, got "
                             f"{_short(history['actions'])}")
    if bandit.pulls != [0, 1, 2, 0, 1, 2, 0, 1]:
        raise AssertionError(f"expected bandit.pull to be called once per step with the chosen arm, got the pulls "
                             f"{bandit.pulls}")
    assert_close(history["rewards"], [0, 1, 1, 1, 0, 1, 0, 1], msg="history['rewards']: the rewards returned by pull")
    if np.asarray(history["counts"]).tolist() != [3, 3, 2]:
        raise AssertionError(f"expected counts [3, 3, 2], got {_short(history['counts'])}")


def test_run_bandit_sample_averages(bdt):
    bandit = NoisyBandit([0.0, 1.0, -0.5, 2.0], seed=70)
    rng = np.random.default_rng(71)
    history = history_of(bdt, bandit, lambda q, n, t, g: int(g.integers(4)) if t < 400 else 2, n_steps=500, rng=rng)
    actions, rewards = np.asarray(history["actions"]), np.asarray(history["rewards"], dtype=float)
    if np.asarray(history["counts"]).sum() != 500:
        raise AssertionError(f"expected counts.sum() == n_steps = 500, got {np.asarray(history['counts']).sum()}")
    assert_close(history["counts"], np.bincount(actions, minlength=4), msg="counts = number of pulls of each arm")
    means = [rewards[actions == a].mean() for a in range(4)]
    assert_close(history["q_values"], means, rtol=1e-9,
                 msg="with step_size=None and initial_value=0, q_values[a] is the mean of the rewards of arm a "
                     "(step 1/counts[a])")


def test_run_bandit_unplayed_arms_keep_the_initial_value(bdt):
    bandit = ScriptedBandit([0.1, 0.9, 0.4], [[1, 0, 0], [1], [0]])
    history = history_of(bdt, bandit, lambda q, n, t, g: 0, n_steps=3, initial_value=2.5,
                         rng=np.random.default_rng(0))
    assert_close(history["q_values"], [1 / 3, 2.5, 2.5],
                 msg="initial_value=2.5: the arm played 3 times has the mean of its rewards (the first update, step "
                     "1/1, replaces the initial value), the others keep 2.5")


def test_run_bandit_constant_step_size(bdt):
    rewards = [1, 0, 0, 1, 1, 0, 1]
    bandit = ScriptedBandit([0.6, 0.3], [rewards, [0]])
    history = history_of(bdt, bandit, lambda q, n, t, g: 0, n_steps=7, step_size=0.3, initial_value=4.0,
                         rng=np.random.default_rng(0))
    estimate = 4.0
    for reward in rewards:                                # the oracle: the exponential average, written here
        estimate += 0.3 * (reward - estimate)
    assert_close(history["q_values"], [estimate, 4.0],
                 msg="step_size=0.3: q <- q + 0.3 (r - q) at every pull of the arm, starting from initial_value=4")


def test_run_bandit_passes_the_state_before_each_step(bdt):
    seen = []
    rng = np.random.default_rng(72)

    def policy(q, n, t, g):
        seen.append((np.array(q, dtype=float), np.array(n), t, g))
        return (t - 1) % 2

    bandit = ScriptedBandit([0.5, 0.7], [[1, 0], [0, 1]])
    history_of(bdt, bandit, policy, n_steps=4, initial_value=0.5, rng=rng)
    if [t for _, _, t, _ in seen] != [1, 2, 3, 4]:
        raise AssertionError(f"expected the policy to be called with t = 1, 2, 3, 4, got {[t for _, _, t, _ in seen]}")
    if any(g is not rng for _, _, _, g in seen):
        raise AssertionError("expected the policy to receive the generator passed as rng (the same object)")
    want_q = [[0.5, 0.5], [1.0, 0.5], [1.0, 0.0], [0.5, 0.0]]
    want_n = [[0, 0], [1, 0], [1, 1], [2, 1]]
    for (q, n, t, _), wq, wn in zip(seen, want_q, want_n):
        if n.tolist() != wn:
            raise AssertionError(f"expected counts {wn} at step t = {t} (the counts before this step), got "
                                 f"{n.tolist()}")
        assert_close(q, wq, msg=f"q_values seen by the policy at step t = {t} (before this step)")


def test_run_bandit_regret_and_optimal_actions(bdt):
    bandit = ScriptedBandit([0.3, 0.9, 0.9, 0.1], [[0], [1], [1], [0]])
    actions = [0, 1, 3, 2, 2, 0]
    history = history_of(bdt, bandit, lambda q, n, t, g: actions[t - 1], n_steps=6, rng=np.random.default_rng(0))
    if np.asarray(history["optimal"]).tolist() != [False, True, False, True, True, False]:
        raise AssertionError(f"expected optimal == [False, True, False, True, True, False] (means[a] == best_mean, "
                             f"two arms tie for the best), got {_short(history['optimal'])}")
    gaps = 0.9 - np.array([0.3, 0.9, 0.1, 0.9, 0.9, 0.3])
    assert_close(history["regret"], np.cumsum(gaps),
                 msg="regret: cumulative sum of best_mean - means[a_t] (the expected loss, not the rewards)")


def test_run_bandit_same_seeds_same_history(bdt):
    def policy(q, n, t, g):
        return int(g.integers(3)) if g.random() < 0.3 else int(np.argmax(q))

    one = history_of(bdt, NoisyBandit([0.1, 0.5, 0.3], seed=73), policy, n_steps=200, rng=np.random.default_rng(74))
    two = history_of(bdt, NoisyBandit([0.1, 0.5, 0.3], seed=73), policy, n_steps=200, rng=np.random.default_rng(74))
    if not np.array_equal(one["actions"], two["actions"]) or not np.allclose(one["rewards"], two["rewards"]):
        raise AssertionError("expected the same history with the same seeds (pass the rng argument to the policy, "
                             "do not create a new generator inside run_bandit)")
    other = history_of(bdt, NoisyBandit([0.1, 0.5, 0.3], seed=73), policy, n_steps=200,
                       rng=np.random.default_rng(75))
    if np.array_equal(one["actions"], other["actions"]):
        raise AssertionError("expected another rng to change the actions of a random policy (200 identical "
                             "actions): pass rng to select_action")


def test_run_bandit_without_rng(bdt):
    history = history_of(bdt, ScriptedBandit([0.4, 0.6], [[1], [0]]), lambda q, n, t, g: int(g.integers(2)),
                         n_steps=20)
    if np.asarray(history["counts"]).sum() != 20:
        raise AssertionError("expected 20 pulls with rng=None (np.random.default_rng())")


@pytest.mark.parametrize("arm, why", [(2, "the policy returns n_arms"), (-1, "the policy returns -1")],
                         ids=["n-arms", "negative"])
def test_run_bandit_rejects_invalid_arms(bdt, arm, why):
    bandit = ScriptedBandit([0.4, 0.6], [[1], [0]])
    assert_raises(ValueError, bdt.run_bandit, bandit, lambda q, n, t, g: arm, n_steps=3,
                  rng=np.random.default_rng(0), why=why)


def test_run_bandit_rejects_zero_steps(bdt):
    assert_raises(ValueError, bdt.run_bandit, ScriptedBandit([0.4, 0.6], [[1], [0]]), cycle_policy, n_steps=0,
                  rng=np.random.default_rng(0), why="n_steps = 0")


# ------------------------------------------------------------------ 11.23 ucb_action
def ucb_oracle(q, n, t, c):
    """NumPy: an untried arm first (smallest index), else the first maximum of q + c sqrt(ln t / n)."""
    q, n = np.asarray(q, dtype=float), np.asarray(n, dtype=float)
    untried = np.flatnonzero(n == 0)
    if untried.size:
        return int(untried[0])
    return int(np.argmax(q + c * np.sqrt(np.log(t) / n)))


def test_ucb_action_docstring_examples(bdt):
    cases = [(([0.5, 0.0, 0.0], [3, 0, 0], 4), {}, 1), (([0.5, 0.4], [10, 1], 11), {}, 1),
             (([0.5, 0.4], [10, 1], 11), {"c": 0.0}, 0)]
    for (q, n, t), kwargs, want in cases:
        result = call(bdt.ucb_action, q, n, t, expected=f"the arm {want}", **kwargs)
        if result != want:
            raise AssertionError(f"expected ucb_action({q}, counts={n}, t={t}{', c=0.0' if kwargs else ''}) == {want}, "
                                 f"got {result!r}")
        python_int(result, "the arm")


def test_ucb_action_plays_an_untried_arm_first(bdt):
    for q, n in [([0.9, 0.1, 0.5, 0.2], [5, 0, 2, 0]), ([0.0, 0.0, 0.0], [0, 0, 0]), ([0.3, 0.8], [4, 0])]:
        want = ucb_oracle(q, n, 10, 2.0)
        result = call(bdt.ucb_action, q, n, 10)
        if result != want:
            raise AssertionError(f"expected the untried arm {want} (smallest index with counts 0) for counts {n}, "
                                 f"got {result!r}")


WRONG_UCB_BOUNDS = {                                      # classic slips, used only to choose telling cases
    "ln(t + 1)": lambda q, n, t, c: q + c * np.sqrt(np.log(t + 1) / n),
    "log2 t": lambda q, n, t, c: q + c * np.sqrt(np.log2(t) / n),
    "log10 t": lambda q, n, t, c: q + c * np.sqrt(np.log10(t) / n),
    "c inside the root": lambda q, n, t, c: q + np.sqrt(c * np.log(t) / n),
    "no square root": lambda q, n, t, c: q + c * np.log(t) / n,
    "root of ln t only": lambda q, n, t, c: q + c * np.sqrt(np.log(t)) / n,
    "counts + 1": lambda q, n, t, c: q + c * np.sqrt(np.log(t) / (n + 1)),
}


def ucb_cases(seed, per_slip=8):
    """Random cases (every arm tried), with at least `per_slip` cases where each classic slip changes the arm."""
    rng = np.random.default_rng(seed)
    cases, found = [], dict.fromkeys(WRONG_UCB_BOUNDS, 0)
    while min(found.values()) < per_slip:
        k = int(rng.integers(2, 6))
        q = np.round(rng.uniform(0, 1, size=k), 3)
        n = rng.integers(1, 30, size=k)
        t = int(n.sum()) + int(rng.integers(0, 3))
        c = float(rng.choice([0.5, 1.0, math.sqrt(2), 2.0, 3.0]))
        want = ucb_oracle(q, n, t, c)
        telling = [name for name, bound in WRONG_UCB_BOUNDS.items()
                   if int(np.argmax(bound(q, n, t, c))) != want and found[name] < per_slip]
        for name in telling:
            found[name] += 1
        if telling:
            cases.append((q, n, t, c, want))
    return cases


@pytest.mark.parametrize("seed", [81, 82], ids=["cases-1", "cases-2"])
def test_ucb_action_matches_the_formula(bdt, seed):
    for q, n, t, c, want in ucb_cases(seed):
        result = call(bdt.ucb_action, q, n, t, c=c)
        if result != want:
            bounds = q + c * np.sqrt(np.log(t) / n)
            _fail("", f"expected the arm {want} = argmax of q + c*sqrt(ln t / counts) (natural log, c={c:.3g}, "
                      f"t={t}), got {result!r}", f"q={_short(q)}, counts={n.tolist()}, bounds={_short(bounds)}")


def test_ucb_action_c_zero_is_greedy_and_ties_give_the_first(bdt):
    result = call(bdt.ucb_action, [0.2, 0.7, 0.7, 0.1], [3, 2, 9, 1], 20, c=0.0)
    if result != 1:
        raise AssertionError(f"expected 1 with c=0 (greedy; the first of the tied maxima 1 and 2), got {result!r}")
    result = call(bdt.ucb_action, [0.5, 0.5], [4, 4], 9)
    if result != 0:
        raise AssertionError(f"expected 0 for two identical arms (first maximum on ties, deterministic), got {result!r}")
    result = call(bdt.ucb_action, [0.1, 0.6], [5, 5], 1)
    if result != 1:
        raise AssertionError(f"expected 1 at t=1 with every arm tried (ln 1 = 0, no bonus), got {result!r}")


def test_ucb_action_does_not_modify_its_inputs(bdt):
    q, n = np.array([0.5, 0.2, 0.9]), np.array([3, 1, 7])
    keep_q, keep_n = q.copy(), n.copy()
    call(bdt.ucb_action, q, n, 12)
    if not (np.array_equal(q, keep_q) and np.array_equal(n, keep_n)):
        raise AssertionError("expected q_values and counts to be left unchanged")


@pytest.mark.parametrize("q, n, t, c, why", [
    ([0.1, 0.2], [1, 1, 1], 3, 2.0, "shapes differ"), ([0.1, 0.2], [1, 1], 0, 2.0, "t = 0"),
    ([0.1, 0.2], [1, 1], 3, -1.0, "c < 0"),
], ids=["shapes", "t-zero", "negative-c"])
def test_ucb_action_rejects_invalid_inputs(bdt, q, n, t, c, why):
    assert_raises(ValueError, bdt.ucb_action, q, n, t, c=c, why=why)


# ------------------------------------------------------------------ 11.23 thompson_action
def thompson_probs(successes, failures, n=400_000, seed=90):
    """Monte Carlo with scipy.stats.beta: P(arm a has the largest draw of Beta(1 + s_a, 1 + f_a))."""
    random_state = np.random.default_rng(seed)
    draws = np.column_stack([stats.beta.rvs(1 + s, 1 + f, size=n, random_state=random_state)
                             for s, f in zip(successes, failures)])
    return np.bincount(draws.argmax(axis=1), minlength=len(successes)) / n


def test_thompson_action_docstring_example(bdt):
    result = call(bdt.thompson_action, [100, 0], [0, 100], rng=np.random.default_rng(0), expected="the arm 0")
    if result != 0:
        raise AssertionError(f"expected 0 (arm 0: 100 successes, arm 1: 100 failures), got {result!r}")
    python_int(result, "the arm")


@pytest.mark.parametrize("successes, failures", [([3, 5, 1], [4, 3, 1]), ([0, 2, 7, 1], [0, 5, 6, 0])],
                         ids=["three-arms", "four-arms"])
def test_thompson_action_choice_frequencies(bdt, successes, failures):
    probs = thompson_probs(successes, failures)
    rng = np.random.default_rng(91)
    counts = frequencies(lambda: bdt.thompson_action(successes, failures, rng=rng), 20_000, len(successes))
    p_value = chisquare_p(counts, probs)
    if p_value < 1e-6:
        raise AssertionError(f"expected each arm chosen with the probability that its draw from Beta(1 + s, 1 + f) "
                             f"is the largest, {_short(probs, 3)}, got the frequencies {_short(counts / 20_000, 3)} "
                             f"(chi-square p-value {p_value:.2g})")


def test_thompson_action_uniform_prior(bdt):
    rng = np.random.default_rng(92)
    counts = frequencies(lambda: bdt.thompson_action([0, 0, 0, 0], [0, 0, 0, 0], rng=rng), 20_000, 4)
    p_value = chisquare_p(counts, [1, 1, 1, 1])
    if p_value < 1e-6:
        raise AssertionError(f"expected each untried arm a quarter of the time (uniform prior Beta(1, 1)), got the "
                             f"counts {counts.tolist()} (chi-square p-value {p_value:.2g})")


def test_thompson_action_uses_the_given_generator(bdt):
    s, f = [2, 4, 3], [3, 2, 3]
    g1, g2 = np.random.default_rng(93), np.random.default_rng(93)
    seq1 = [bdt.thompson_action(s, f, rng=g1) for _ in range(100)]
    np.random.seed(4)
    seq2 = [bdt.thompson_action(s, f, rng=g2) for _ in range(100)]
    if seq1 != seq2:
        raise AssertionError("expected the same arms from two generators with the same seed: draw with rng.beta, "
                             "never with np.random.* or a new generator")
    result = call(bdt.thompson_action, s, f, expected="an arm with rng=None")
    if result not in (0, 1, 2):
        raise AssertionError(f"expected an arm in 0..2 with rng=None, got {result!r}")


@pytest.mark.parametrize("s, f, why", [([1, 2], [1, 2, 3], "shapes differ"), ([1, -1], [0, 2], "negative count"),
                                        ([1.0, 2.0], [-0.5, 0.0], "negative count, even above -1 (Beta(0.5, ...) exists)")],
                         ids=["shapes", "negative", "negative-fraction"])
def test_thompson_action_rejects_invalid_inputs(bdt, s, f, why):
    assert_raises(ValueError, bdt.thompson_action, s, f, rng=np.random.default_rng(0), why=why)
