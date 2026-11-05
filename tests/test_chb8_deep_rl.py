"""Tests of mylearn.deep_rl (bonus chapter B8): oracle tests and properties.

    pytest tests/test_chb8_deep_rl.py              # your code (mon_travail/mylearn/deep_rl.py)
    pytest tests/test_chb8_deep_rl.py --impl=ref   # the reference

Oracles: for ``compute_gae``, a reverse loop over time written in this file in the style of CleanRL's ``ppo.py``
(one scalar ``lastgaelam`` per environment; the flags of the docstring belong to step t, and a truncated step keeps
its bootstrap), and closed forms computed here with explicit sums of powers of gamma: ``lam = 0`` gives the TD
errors, ``lam = 1`` with ``V = 0`` and no truncation gives the discounted returns of each episode, and ``lam = 1``
with ``V != 0`` the Monte Carlo advantage (return bootstrapped at the end of the rollout, minus ``V``); hand-made
rollouts where one step is terminated, truncated or both. For ``explained_variance``,
``sklearn.metrics.explained_variance_score(y_true, y_pred)`` when ``Var(y_true) > 0`` and the formula of the
docstring (population variances) otherwise. For ``clipped_surrogate``, ``torch.autograd`` on the expression of the
docstring written with ``torch.min`` and ``torch.clamp`` in float64, a NumPy transcription of the docstring, and the
four cases (sign of the advantage, ratio below or above the interval) plus the ratios inside it and on its exact
bounds (``clip_eps`` = 0.25 or 0.5: the bounds 0.75, 1.25, 0.5 and 1.5 are exact in binary; there the unclipped
term equals the clipped one, so the docstring gives ``-A/n``, as does autograd). For ``bradley_terry_loss`` and
``dpo_loss``, ``-torch.nn.functional.logsigmoid(...).mean()`` and its autograd gradients in float64, a NumPy
transcription with ``np.logaddexp``, equation 7 of Rafailov et al. (2023), and properties: finite values for
differences of +-1000 and more, invariance to a common shift, the effect of swapping chosen and rejected
(``loss(b, a) - loss(a, b) = mean(a - b)``), margins proportional to beta. ``rl.returns_to_go`` and the other
functions of your library are never used as oracles; trl is not used (it is not installed in the frozen
environment).
Without PyTorch, the tests that need it are skipped and the NumPy oracles remain.
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_compute_gae_ or test_explained_variance_"`` (B8.18),
``-k "test_clipped_surrogate_"`` (B8.19),
``-k "test_bradley_terry_loss_ or test_dpo_loss_"`` (B8.25).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified;
"the shapes differ" includes shapes that NumPy could broadcast (``(3,)`` and ``(3, 1)``): they raise ValueError;
the bounds of the intervals are allowed (``gamma`` and ``lam`` equal to 0 or 1, a ratio of 0); the flags may be
lists of Python bools or NumPy bool arrays, the numbers lists or float32 / float64 arrays; ``explained_variance``
returns nan only when ``Var(y_true)`` is exactly 0 (tested with constants exact in binary: 3.0, 0.0, -2.5, and a
single value).
The results are compared with rtol=1e-6, atol=1e-6 (the docstrings fix no tolerance: a computation in float32
passes, a wrong formula does not), rtol=1e-4, atol=1e-4 for float32 inputs.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

from __future__ import annotations

import copy
import math
import numbers

import numpy as np
import pytest
from sklearn.metrics import explained_variance_score

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # the PyTorch oracle runs on Colab; the NumPy oracles remain
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")

ORACLE_BUG = ("bug in tests/test_chb8_deep_rl.py (not in your code): the oracle of this test "
              "disagrees with another one")


@pytest.fixture
def drl(mylearn_module):
    return mylearn_module("deep_rl")


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


# The docstrings fix no tolerance: rtol=1e-6, atol=1e-6 lets a computation in float32 pass, not a wrong formula.
RTOL, ATOL = 1e-6, 1e-6
# Inputs given in float32: a computation kept in float32 over a rollout loses a few more digits.
RTOL32, ATOL32 = 1e-4, 1e-4


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
    """Call the learner's function on deep copies of the arguments; an unexpected exception becomes a failure
    that says what was expected."""
    args = copy.deepcopy(args)
    kwargs = copy.deepcopy(kwargs)
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
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


def tuple_result(result, msg, names) -> tuple:
    """The learner's tuple of len(names) values (None and length checked first)."""
    k = len(names)
    what = "(" + ", ".join(names) + ")"
    if result is None:
        _fail(msg, f"expected the tuple {what}, got None (did you forget the return?)")
    if not isinstance(result, (tuple, list)) or len(result) != k:
        got = type(result).__name__ + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else "")
        _fail(msg, f"expected the tuple {what} (a tuple of {k}), got {got}")
    return tuple(result)


def assert_unchanged(before, after, name, function):
    same = (isinstance(after, np.ndarray) and before.shape == after.shape and before.dtype == after.dtype
            and np.array_equal(before, after))
    if not same:
        _fail(function, f"expected the input {name} to be left as it was, it was modified "
                        "(work on a copy, e.g. np.array(x, dtype=float), and avoid in-place operations like -=)")


# ================================================================== compute_gae (B8.18)
def gae_oracle(rewards, values, next_values, terminated, truncated, gamma, lam):
    """The docstring recursion as a reverse loop, in the style of CleanRL's ppo.py (one `lastgaelam` per env)."""
    r = np.asarray(rewards, dtype=float)
    shape = r.shape
    T = shape[0]
    r = r.reshape(T, -1)
    v = np.asarray(values, dtype=float).reshape(T, -1)
    nv = np.asarray(next_values, dtype=float).reshape(T, -1)
    term = np.asarray(terminated, dtype=bool).reshape(T, -1)
    trunc = np.asarray(truncated, dtype=bool).reshape(T, -1)
    adv = np.zeros_like(r)
    for e in range(r.shape[1]):
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nextnonterminal = 0.0 if term[t, e] else 1.0                  # no bootstrap after a true end
            nextnotdone = 0.0 if (term[t, e] or trunc[t, e]) else 1.0     # the recursion stops at both
            delta = r[t, e] + gamma * nextnonterminal * nv[t, e] - v[t, e]
            lastgaelam = delta + gamma * lam * nextnotdone * lastgaelam
            adv[t, e] = lastgaelam
    return adv.reshape(shape), (adv + v).reshape(shape)


def rollout(seed, shape, p_term=0.15, p_trunc=0.1):
    """Random rewards, values, next values and flags (both flags may be True on the same step)."""
    rng = np.random.default_rng(seed)
    rewards = rng.normal(1.0, 1.0, size=shape)
    values = rng.normal(2.0, 1.5, size=shape)
    next_values = rng.normal(2.0, 1.5, size=shape)
    terminated = rng.random(shape) < p_term
    truncated = rng.random(shape) < p_trunc
    return rewards, values, next_values, terminated, truncated


def gae_of(drl, rewards, values, next_values, terminated, truncated, msg="compute_gae", **kwargs):
    """The learner's (advantages, returns), checked: a pair of float arrays with the shape of rewards."""
    shape = np.shape(rewards)
    result = call(drl.compute_gae, rewards, values, next_values, terminated, truncated,
                  expected="the pair (advantages, returns)", **kwargs)
    adv, ret = tuple_result(result, msg, ("advantages", "returns"))
    inputs = f"rewards of shape {shape}"
    return (array_result(adv, f"{msg}: advantages", shape, inputs),
            array_result(ret, f"{msg}: returns", shape, inputs))


def test_compute_gae_docstring_examples(drl):
    r, v = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
    adv, ret = gae_of(drl, r, v, v, [False, False, True], [False, False, False], gamma=0.5, lam=1.0)
    assert_close(adv, [1.75, 1.5, 1.0], msg="compute_gae: advantages of the first docstring example")
    assert_close(ret, [1.75, 1.5, 1.0], msg="compute_gae: returns of the first docstring example (V = 0)")
    adv, ret = gae_of(drl, r, [0.5, 0.5, 0.5], [0.5, 0.5, 2.0], [False, False, False], [False, True, False],
                      gamma=0.9, lam=0.5)
    assert_close(adv, [1.3775, 0.95, 2.3], msg="compute_gae: advantages of the second docstring example "
                                               "(truncation at step 1)")


GAE_SETTINGS = [
    pytest.param(0.99, 0.95, id="gamma0.99-lam0.95"),
    pytest.param(0.9, 0.5, id="gamma0.9-lam0.5"),
    pytest.param(0.5, 0.8, id="gamma0.5-lam0.8"),
    pytest.param(1.0, 1.0, id="gamma1-lam1"),
    pytest.param(0.0, 0.7, id="gamma0"),
]


@pytest.mark.parametrize("gamma,lam", GAE_SETTINGS)
@pytest.mark.parametrize("seed,T", [pytest.param(0, 12, id="T12"), pytest.param(1, 40, id="T40"),
                                    pytest.param(2, 1, id="T1")])
def test_compute_gae_matches_the_reverse_loop(drl, seed, T, gamma, lam):
    data = rollout(seed, (T,))
    want_adv, want_ret = gae_oracle(*data, gamma, lam)
    adv, ret = gae_of(drl, *data, gamma=gamma, lam=lam)
    flags = f"terminated at {np.flatnonzero(data[3]).tolist()}, truncated at {np.flatnonzero(data[4]).tolist()}"
    assert_close(adv, want_adv, msg=f"compute_gae: advantages, rollout of {T} steps (gamma={gamma}, lam={lam}, "
                                    f"{flags})")
    assert_close(ret, want_ret, msg=f"compute_gae: returns (advantages + values), rollout of {T} steps")


@pytest.mark.parametrize("shape", [pytest.param((7, 3), id="T7-3envs"), pytest.param((4, 4), id="T4-4envs-square"),
                                   pytest.param((1, 5), id="T1-5envs"), pytest.param((30, 2), id="T30-2envs")])
def test_compute_gae_several_environments(drl, shape):
    data = rollout(10 + shape[0], shape, p_term=0.2, p_trunc=0.15)
    want_adv, want_ret = gae_oracle(*data, 0.9, 0.8)
    adv, ret = gae_of(drl, *data, gamma=0.9, lam=0.8)
    assert_close(adv, want_adv, msg=f"compute_gae: advantages of shape {shape} (time on the first axis, one "
                                    "independent recursion per environment, i.e. per column)")
    assert_close(ret, want_ret, msg=f"compute_gae: returns of shape {shape}")


def test_compute_gae_environments_do_not_mix(drl):
    # env 0 ends at step 1, env 1 never ends: env 1 must not see the end of env 0
    rewards = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    zeros = np.zeros((3, 2))
    terminated = np.array([[False, False], [True, False], [False, False]])
    truncated = np.zeros((3, 2), dtype=bool)
    adv, _ = gae_of(drl, rewards, zeros, zeros, terminated, truncated, gamma=0.5, lam=1.0)
    assert_close(adv[:, 1], [1.75, 1.5, 1.0], msg="compute_gae: advantages of env 1 (column 1, never ends), "
                                                 "independent of the end of env 0")
    assert_close(adv[:, 0], [1.5, 1.0, 1.0], msg="compute_gae: advantages of env 0 (column 0, ends at step 1)")


def td_errors(rewards, values, next_values, terminated, gamma):
    """delta_t = r_t + gamma (1 - terminated_t) V(s_{t+1}) - V(s_t), all at once."""
    return rewards + gamma * np.where(terminated, 0.0, next_values) - values


@pytest.mark.parametrize("shape", [pytest.param((15,), id="T15"), pytest.param((9, 3), id="T9-3envs")])
@pytest.mark.parametrize("gamma", [0.99, 0.6])
def test_compute_gae_lam_zero_gives_the_td_errors(drl, shape, gamma):
    data = rollout(20, shape, p_term=0.2, p_trunc=0.2)
    want = td_errors(data[0], data[1], data[2], data[3], gamma)
    adv, _ = gae_of(drl, *data, gamma=gamma, lam=0.0)
    assert_close(adv, want, msg=f"compute_gae with lam=0: expected the TD errors r_t + gamma (1 - terminated_t) "
                                f"next_values_t - values_t (gamma={gamma}; truncated steps keep the bootstrap)")


def discounted_returns(rewards, terminated, gamma):
    """Closed form, 1-D: G_t = sum_{k=t}^{end} gamma^(k-t) r_k, with `end` the first terminated step >= t
    (or the last step of the rollout)."""
    rewards = np.asarray(rewards, dtype=float)
    T = rewards.size
    G = np.zeros(T)
    for t in range(T):
        end = next((k for k in range(t, T) if terminated[k]), T - 1)
        G[t] = sum(gamma ** (k - t) * rewards[k] for k in range(t, end + 1))
    return G


@pytest.mark.parametrize("gamma", [1.0, 0.9, 0.5])
def test_compute_gae_lam_one_value_zero_gives_the_discounted_returns(drl, gamma):
    rng = np.random.default_rng(30)
    rewards = rng.integers(-2, 4, size=14).astype(float)
    terminated = np.zeros(14, dtype=bool)
    terminated[[3, 4, 9]] = True                              # episodes 0-3, 4, 5-9, 10-13 (cut by the rollout)
    zeros = np.zeros(14)
    want = discounted_returns(rewards, terminated, gamma)
    adv, ret = gae_of(drl, rewards, zeros, zeros, terminated, np.zeros(14, dtype=bool), gamma=gamma, lam=1.0)
    msg = (f"compute_gae with lam=1, V=0 and no truncation (gamma={gamma}): expected the discounted return of "
           "each step within its episode (episodes end at the terminated steps 3, 4 and 9)")
    assert_close(adv, want, msg=msg)
    assert_close(ret, want, msg=msg.replace("compute_gae", "compute_gae: returns"))


def test_compute_gae_lam_one_value_zero_several_environments(drl):
    rng = np.random.default_rng(31)
    rewards = rng.normal(size=(10, 3))
    terminated = rng.random((10, 3)) < 0.25
    zeros = np.zeros((10, 3))
    want = np.column_stack([discounted_returns(rewards[:, e], terminated[:, e], 0.8) for e in range(3)])
    adv, _ = gae_of(drl, rewards, zeros, zeros, terminated, np.zeros((10, 3), dtype=bool), gamma=0.8, lam=1.0)
    assert_close(adv, want, msg="compute_gae with lam=1, V=0, no truncation, 3 environments: expected the "
                                "discounted returns of each column")


@pytest.mark.parametrize("last_terminated", [True, False], ids=["ends-terminated", "cut-by-the-rollout"])
def test_compute_gae_lam_one_is_the_monte_carlo_advantage(drl, last_terminated):
    rng = np.random.default_rng(32)
    T, gamma = 8, 0.7
    rewards, values = rng.normal(size=T), rng.normal(size=T)
    next_values = np.append(values[1:], rng.normal())     # one episode: V(s_{t+1}) is values[t + 1]
    terminated = np.zeros(T, dtype=bool)
    terminated[T - 1] = last_terminated
    G = discounted_returns(rewards, terminated, gamma)
    if not last_terminated:                    # the last TD error bootstraps on next_values[T - 1]
        G = G + gamma ** (T - np.arange(T)) * next_values[T - 1]
    want = G - values
    adv, ret = gae_of(drl, rewards, values, next_values, terminated, np.zeros(T, dtype=bool), gamma=gamma, lam=1.0)
    end = "terminated at the last step" if last_terminated else \
        "not ended: the return bootstraps on next_values[T-1] = gamma^(T-t) V(s_T)"
    assert_close(adv, want, msg=f"compute_gae with lam=1 (gamma={gamma}): expected the Monte Carlo advantage "
                                f"G_t - V(s_t) ({end})")
    assert_close(ret, G, msg="compute_gae with lam=1: expected the returns G_t (advantages + values)")


# A hand-made rollout of 5 steps where step 2 ends an episode; the next steps belong to a new episode.
FLAG_CASE = dict(rewards=[1.0, 2.0, 3.0, 4.0, 5.0], values=[0.5, 1.0, 1.5, 2.0, 2.5],
                 next_values=[1.0, 1.5, 8.0, 2.5, 3.0], gamma=0.5, lam=0.5)


def flag_case(term_at=None, trunc_at=None):
    c = FLAG_CASE
    term, trunc = np.zeros(5, dtype=bool), np.zeros(5, dtype=bool)
    if term_at is not None:
        term[term_at] = True
    if trunc_at is not None:
        trunc[trunc_at] = True
    return (np.array(c["rewards"]), np.array(c["values"]), np.array(c["next_values"]), term, trunc)


def test_compute_gae_truncated_step_bootstraps_on_next_values(drl):
    data = flag_case(trunc_at=2)
    adv, _ = gae_of(drl, *data, gamma=0.5, lam=0.5)
    # A_2 = delta_2 = 3 + 0.5 * 8 - 1.5 (bootstrap on the real final observation, recursion stopped)
    assert_close(adv[2], 5.5, msg="compute_gae: advantage of the truncated step 2: expected delta_2 = r_2 + gamma "
                                  "next_values[2] - values[2] = 5.5 (a truncated step still bootstraps, and the "
                                  "recursion stops: nothing from step 3)")


def test_compute_gae_truncation_stops_the_recursion(drl):
    data = flag_case(trunc_at=2)
    want, _ = gae_oracle(*data, 0.5, 0.5)
    adv, _ = gae_of(drl, *data, gamma=0.5, lam=0.5)
    assert_close(adv, want, msg="compute_gae: advantages with step 2 truncated (steps 0-2 one episode, steps 3-4 "
                                "the next one: A_2 must not include A_3)")


def test_compute_gae_terminated_step_does_not_bootstrap(drl):
    data = flag_case(term_at=2)
    adv, _ = gae_of(drl, *data, gamma=0.5, lam=0.5)
    assert_close(adv[2], 1.5, msg="compute_gae: advantage of the terminated step 2: expected r_2 - values[2] = 1.5 "
                                  "(true end: no bootstrap on next_values[2], recursion stopped)")
    want, _ = gae_oracle(*data, 0.5, 0.5)
    assert_close(adv, want, msg="compute_gae: advantages with step 2 terminated")


def test_compute_gae_terminated_and_truncated_differ_only_by_the_bootstrap(drl):
    adv_term, _ = gae_of(drl, *flag_case(term_at=2), gamma=0.5, lam=0.5)
    adv_trunc, _ = gae_of(drl, *flag_case(trunc_at=2), gamma=0.5, lam=0.5)
    assert_close(adv_trunc[3:], adv_term[3:], msg="compute_gae: the steps after the end (3 and 4, a new episode) "
                                                  "must be the same whether step 2 is terminated or truncated")
    # only the bootstrap gamma * next_values[2] = 4 differs at step 2, and it reaches step 1 through gamma lam
    assert_close(adv_trunc[:3] - adv_term[:3], [4 * 0.25 ** 2, 4 * 0.25, 4.0],
                 msg="compute_gae: truncated minus terminated advantages at steps 0-2: expected the bootstrap "
                     "gamma * next_values[2] = 4 at step 2, discounted by (gamma lam)^k before")


def test_compute_gae_terminated_and_truncated_together(drl):
    data = flag_case(term_at=2, trunc_at=2)
    adv, _ = gae_of(drl, *data, gamma=0.5, lam=0.5)
    want, _ = gae_oracle(*data, 0.5, 0.5)
    assert_close(adv, want, msg="compute_gae: step 2 both terminated and truncated (gymnasium may return both): "
                                "expected no bootstrap and the recursion stopped, as for terminated")


def test_compute_gae_last_step_bootstraps_when_not_ended(drl):
    # the rollout stops in the middle of an episode: A_T = 0, but delta_{T-1} uses next_values[T-1]
    data = flag_case()
    adv, _ = gae_of(drl, *data, gamma=0.5, lam=0.5)
    assert_close(adv[4], 5.0 + 0.5 * 3.0 - 2.5, msg="compute_gae: advantage of the last step of a rollout that "
                                                    "stops mid-episode: expected r + gamma next_values[-1] - "
                                                    "values[-1] = 4 (the bootstrap is kept, A_T = 0)")
    want, _ = gae_oracle(*data, 0.5, 0.5)
    assert_close(adv, want, msg="compute_gae: advantages of a rollout without any end")


def test_compute_gae_uses_next_values_not_the_next_values_entry(drl):
    # next_values[t] is V(s_{t+1}); it differs from values[t + 1] after an end, and is given for every step
    rewards = np.zeros(4)
    values = np.array([1.0, 2.0, 3.0, 4.0])
    next_values = np.array([10.0, 20.0, 30.0, 40.0])
    flags = np.zeros(4, dtype=bool)
    adv, _ = gae_of(drl, rewards, values, next_values, flags, flags, gamma=0.5, lam=0.0)
    assert_close(adv, [4.0, 8.0, 12.0, 16.0], msg="compute_gae with lam=0: expected gamma next_values[t] - "
                                                  "values[t] (the bootstrap comes from next_values, not from "
                                                  "values[t + 1])")


def test_compute_gae_returns_are_advantages_plus_values(drl):
    data = rollout(40, (11,))
    adv, ret = gae_of(drl, *data, gamma=0.95, lam=0.9)
    assert_close(ret, adv + data[1], msg="compute_gae: expected returns = advantages + values (the targets of the "
                                         "critic)")


def test_compute_gae_defaults_are_099_and_095(drl):
    data = rollout(41, (25,), p_term=0.05, p_trunc=0.05)
    want, _ = gae_oracle(*data, 0.99, 0.95)
    adv, _ = gae_of(drl, *data)
    assert_close(adv, want, msg="compute_gae without gamma and lam: expected the defaults gamma=0.99, lam=0.95")


def test_compute_gae_gamma_and_lam_play_different_roles(drl):
    # gamma discounts the bootstrap of delta, lam does not: swapping them changes the advantages
    data = flag_case(trunc_at=2)
    want, _ = gae_oracle(*data, 0.9, 0.3)
    adv, _ = gae_of(drl, *data, gamma=0.9, lam=0.3)
    assert_close(adv, want, msg="compute_gae with gamma=0.9, lam=0.3: expected delta_t with gamma (not lam), and "
                                "the product gamma lam in the recursion")


@pytest.mark.parametrize("gamma,lam", [pytest.param(0.0, 0.0, id="both-0"), pytest.param(1.0, 0.0, id="gamma1-lam0"),
                                       pytest.param(0.0, 1.0, id="gamma0-lam1"), pytest.param(1.0, 1.0, id="both-1")])
def test_compute_gae_accepts_the_bounds_of_gamma_and_lam(drl, gamma, lam):
    data = rollout(42, (6,))
    want, _ = gae_oracle(*data, gamma, lam)
    adv, _ = gae_of(drl, *data, gamma=gamma, lam=lam)
    assert_close(adv, want, msg=f"compute_gae with gamma={gamma}, lam={lam} (the bounds of [0, 1] are allowed)")


def test_compute_gae_accepts_float32_arrays(drl):
    # what a PyTorch rollout gives (tensor.numpy()), with NumPy bool flags
    data = rollout(43, (64, 4), p_term=0.05, p_trunc=0.03)
    data32 = tuple(a.astype(np.float32) for a in data[:3]) + data[3:]
    want_adv, want_ret = gae_oracle(*data32, 0.99, 0.95)
    adv, ret = gae_of(drl, *data32, gamma=0.99, lam=0.95)
    assert_close(adv, want_adv, rtol=RTOL32, atol=ATOL32, msg="compute_gae on float32 arrays of shape (64, 4)")
    assert_close(ret, want_ret, rtol=RTOL32, atol=ATOL32, msg="compute_gae: returns on float32 arrays")


def test_compute_gae_does_not_modify_its_inputs(drl):
    data = rollout(44, (6, 2))
    before = [a.copy() for a in data]
    names = ("rewards", "values", "next_values", "terminated", "truncated")
    try:
        drl.compute_gae(*data, gamma=0.9, lam=0.8)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail("compute_gae", f"expected the pair (advantages, returns), but it raised {type(exc).__name__}: {exc}")
    for name, old, new in zip(names, before, data):
        assert_unchanged(old, new, name, "compute_gae")


def _gae_args(**change):
    base = dict(rewards=np.ones(3), values=np.zeros(3), next_values=np.zeros(3),
                terminated=np.zeros(3, dtype=bool), truncated=np.zeros(3, dtype=bool))
    kwargs = {}
    for key, value in change.items():
        (base if key in base else kwargs)[key] = value
    return tuple(base.values()), kwargs


@pytest.mark.parametrize("change,why", [
    pytest.param({"values": np.zeros(4)}, "values has 4 steps, rewards 3", id="values-longer"),
    pytest.param({"next_values": np.zeros(2)}, "next_values has 2 steps, rewards 3", id="next-values-shorter"),
    pytest.param({"terminated": np.zeros(4, dtype=bool)}, "terminated has 4 steps, rewards 3", id="terminated-longer"),
    pytest.param({"truncated": np.zeros(2, dtype=bool)}, "truncated has 2 steps, rewards 3", id="truncated-shorter"),
    pytest.param({"values": np.zeros((3, 1))}, "values of shape (3, 1), rewards (3,) (NumPy would broadcast them "
                 "to (3, 3))", id="values-3x1-broadcast"),
    pytest.param({"terminated": np.zeros((3, 2), dtype=bool)}, "terminated of shape (3, 2), rewards (3,)",
                 id="terminated-2d"),
    pytest.param({"truncated": np.zeros((3, 1), dtype=bool)}, "truncated of shape (3, 1), rewards (3,) (NumPy would "
                 "broadcast them)", id="truncated-column"),
    pytest.param({"gamma": -0.1}, "gamma = -0.1 is outside [0, 1]", id="gamma-negative"),
    pytest.param({"gamma": 1.01}, "gamma = 1.01 is outside [0, 1]", id="gamma-above-1"),
    pytest.param({"lam": -0.5}, "lam = -0.5 is outside [0, 1]", id="lam-negative"),
    pytest.param({"lam": 1.5}, "lam = 1.5 is outside [0, 1]", id="lam-above-1"),
])
def test_compute_gae_refuses_invalid_arguments(drl, change, why):
    args, kwargs = _gae_args(**change)
    assert_raises(ValueError, drl.compute_gae, *args, why=why, **kwargs)


# ================================================================== explained_variance (B8.18)
def ev_formula(y_pred, y_true):
    """The docstring: 1 - Var(y_true - y_pred) / Var(y_true), population variances; nan if Var(y_true) == 0."""
    y_pred, y_true = np.asarray(y_pred, float), np.asarray(y_true, float)
    var_y = np.var(y_true)
    return float("nan") if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y


def ev_of(drl, y_pred, y_true, msg):
    return number_result(call(drl.explained_variance, y_pred, y_true, expected="one number"), msg)


def test_explained_variance_docstring_examples(drl):
    got = ev_of(drl, [1.0, 2.0, 3.0], [1.0, 2.0, 4.0], "explained_variance([1, 2, 3], [1, 2, 4]) (docstring)")
    assert_close(got, 1 - (2 / 9) / (14 / 9), msg="explained_variance([1, 2, 3], [1, 2, 4]) (docstring: 0.8571)")
    got = ev_of(drl, [1.0, 2.0], [3.0, 3.0], "explained_variance([1, 2], [3, 3]) (docstring)")
    if not math.isnan(got):
        _fail("explained_variance([1, 2], [3, 3]) (docstring)", f"expected nan (Var(y_true) == 0), got {got!r}")


@pytest.mark.parametrize("seed,n", [pytest.param(0, 2, id="n2"), pytest.param(1, 10, id="n10"),
                                    pytest.param(2, 128, id="n128"), pytest.param(3, 2048, id="n2048")])
def test_explained_variance_matches_sklearn(drl, seed, n):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(5.0, 3.0, size=n)
    y_pred = 0.7 * y_true + rng.normal(0.0, 1.5, size=n) + 2.0
    want = explained_variance_score(y_true, y_pred)
    if not np.isclose(want, ev_formula(y_pred, y_true), rtol=1e-12, atol=1e-12):
        raise AssertionError(ORACLE_BUG)
    got = ev_of(drl, y_pred, y_true, "explained_variance")
    assert_close(got, want, msg=f"explained_variance(y_pred, y_true), n = {n}: expected "
                                "sklearn.metrics.explained_variance_score(y_true, y_pred)")


def test_explained_variance_prediction_comes_first(drl):
    y_pred = np.array([0.0, 0.0, 1.0, 1.0])            # Var(y_pred) = 0.25
    y_true = np.array([0.0, 2.0, 4.0, 6.0])            # Var(y_true) = 5
    want = explained_variance_score(y_true, y_pred)    # 1 - 3.25 / 5 = 0.35 (the other order gives -12)
    got = ev_of(drl, y_pred, y_true, "explained_variance")
    assert_close(got, want, msg="explained_variance(y_pred, y_true): expected 0.35, the variance of the residuals "
                                "divided by Var(y_true), the second argument (the prediction comes first)")


def test_explained_variance_constant_prediction_is_not_nan(drl):
    # Var(y_pred) = 0 does not matter: only Var(y_true) == 0 gives nan
    got = ev_of(drl, [2.0, 2.0, 2.0, 2.0], [1.0, 3.0, 2.0, 6.0], "explained_variance")
    assert_close(got, 0.0, msg="explained_variance of a constant prediction for varying returns: expected 0 "
                               "(no better than a constant; nan only when Var(y_true) == 0)")


@pytest.mark.parametrize("y_pred,y_true,want,why", [
    pytest.param([1.0, 2.0, 4.0], [1.0, 2.0, 4.0], 1.0, "a perfect critic gives 1", id="perfect"),
    pytest.param([7.0, 7.0, 7.0], [1.0, 2.0, 4.0], 0.0, "any constant prediction gives 0", id="constant"),
    pytest.param([4.0, 2.0, 1.0], [1.0, 2.0, 4.0], 1 - 6 / (14 / 9), "the reversed prediction is worse than a "
                 "constant: 1 - Var([-3, 0, 3]) / Var([1, 2, 4]) < 0", id="worse-than-a-constant"),
    pytest.param([11.0, 12.0, 14.0], [1.0, 2.0, 4.0], 1.0, "a constant offset is not penalised (variance of the "
                 "residuals)", id="offset"),
])
def test_explained_variance_reference_values(drl, y_pred, y_true, want, why):
    got = ev_of(drl, y_pred, y_true, "explained_variance")
    assert_close(got, want, msg=f"explained_variance({y_pred}, {y_true}): {why}")


@pytest.mark.parametrize("y_pred,y_true", [
    pytest.param([1.0, 2.0, 0.0, 5.0], [3.0, 3.0, 3.0, 3.0], id="constant-3"),
    pytest.param([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], id="all-zero"),
    pytest.param([1.0, -1.0], [-2.5, -2.5], id="constant-minus-2.5"),
    pytest.param([4.0], [1.0], id="single-value"),
])
def test_explained_variance_nan_when_returns_are_constant(drl, y_pred, y_true):
    got = ev_of(drl, y_pred, y_true, "explained_variance")
    if not math.isnan(got):
        _fail(f"explained_variance({y_pred}, {y_true})", f"expected nan (Var(y_true) == 0, as in CleanRL), got {got!r}")


def test_explained_variance_is_at_most_one(drl):
    rng = np.random.default_rng(5)
    for trial in range(20):
        y_true, y_pred = rng.normal(size=30), rng.normal(size=30) * rng.uniform(0.1, 5)
        got = ev_of(drl, y_pred, y_true, "explained_variance")
        if not got <= 1.0 + 1e-12:
            _fail("explained_variance", f"expected a value <= 1, got {got!r} (trial {trial})")


def test_explained_variance_accepts_float32_arrays(drl):
    rng = np.random.default_rng(6)
    y_true = rng.normal(size=256).astype(np.float32)
    y_pred = (y_true + rng.normal(scale=0.5, size=256)).astype(np.float32)
    got = ev_of(drl, y_pred, y_true, "explained_variance")
    assert_close(got, explained_variance_score(y_true.astype(float), y_pred.astype(float)), rtol=RTOL32,
                 atol=ATOL32, msg="explained_variance on float32 arrays")


def test_explained_variance_does_not_modify_its_inputs(drl):
    y_pred, y_true = np.array([1.0, 2.0, 3.0, 5.0]), np.array([1.5, 2.0, 2.5, 6.0])
    before = (y_pred.copy(), y_true.copy())
    try:
        drl.explained_variance(y_pred, y_true)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail("explained_variance", f"expected one number, but it raised {type(exc).__name__}: {exc}")
    assert_unchanged(before[0], y_pred, "y_pred", "explained_variance")
    assert_unchanged(before[1], y_true, "y_true", "explained_variance")


@pytest.mark.parametrize("y_pred,y_true,why", [
    pytest.param(np.zeros(3), np.arange(4.0), "y_pred has 3 values, y_true 4", id="lengths-differ"),
    pytest.param(np.zeros((3, 1)), np.arange(3.0), "y_pred of shape (3, 1), y_true (3,) (NumPy would broadcast "
                 "them to (3, 3))", id="pred-3x1-broadcast"),
    pytest.param(np.arange(3.0), np.zeros((1, 3)), "y_true of shape (1, 3), y_pred (3,)", id="true-1x3"),
])
def test_explained_variance_refuses_invalid_arguments(drl, y_pred, y_true, why):
    assert_raises(ValueError, drl.explained_variance, y_pred, y_true, why=why)


# ================================================================== clipped_surrogate (B8.19)
def surrogate_numpy(ratio, advantages, eps):
    """The docstring transcribed: loss = -mean(min(r A, clip(r) A)), gradient -A/n where r A <= clip(r) A."""
    r, A = np.asarray(ratio, float), np.asarray(advantages, float)
    unclipped, clipped = r * A, np.clip(r, 1 - eps, 1 + eps) * A
    n = r.size
    grad = np.array([-A[i] / n if unclipped[i] <= clipped[i] else 0.0 for i in range(n)])
    return -float(np.mean(np.minimum(unclipped, clipped))), grad


def surrogate_torch(ratio, advantages, eps):
    """The same expression with torch.min and torch.clamp, gradient by autograd (float64)."""
    r = torch.tensor(np.asarray(ratio, float), dtype=torch.float64, requires_grad=True)
    A = torch.tensor(np.asarray(advantages, float), dtype=torch.float64)
    loss = -torch.min(r * A, torch.clamp(r, 1 - eps, 1 + eps) * A).mean()
    loss.backward()
    return float(loss.detach()), r.grad.numpy().copy()


def surrogate_of(drl, ratio, advantages, msg="clipped_surrogate", **kwargs):
    n = np.shape(ratio)[0]
    result = call(drl.clipped_surrogate, ratio, advantages, expected="the pair (loss, grad)", **kwargs)
    loss, grad = tuple_result(result, msg, ("loss", "grad"))
    return number_result(loss, f"{msg}: loss"), array_result(grad, f"{msg}: grad", (n,), f"{n} ratios")


def test_clipped_surrogate_docstring_example(drl):
    loss, grad = surrogate_of(drl, [0.5, 1.0, 1.5], [1.0, 1.0, 1.0])
    assert_close(loss, -0.9, msg="clipped_surrogate([0.5, 1.0, 1.5], [1, 1, 1]): loss of the docstring "
                                 "(-(0.5 + 1.0 + 1.2) / 3)")
    assert_close(grad, [-1 / 3, -1 / 3, 0.0], msg="clipped_surrogate([0.5, 1.0, 1.5], [1, 1, 1]): gradient of the "
                                                  "docstring (the third ratio is clipped)")


def _surrogate_case(seed, n):
    rng = np.random.default_rng(seed)
    ratio = rng.uniform(0.0, 2.5, size=n)
    advantages = rng.normal(0.0, 2.0, size=n)
    return ratio, advantages


SURROGATE_CASES = [pytest.param(0, 1, 0.2, id="n1"), pytest.param(1, 8, 0.2, id="n8"),
                   pytest.param(2, 64, 0.1, id="n64-eps0.1"), pytest.param(3, 500, 0.3, id="n500-eps0.3"),
                   pytest.param(4, 40, 1.5, id="eps1.5")]


@needs_torch
@pytest.mark.parametrize("seed,n,eps", SURROGATE_CASES)
def test_clipped_surrogate_matches_torch_autograd(drl, seed, n, eps):
    ratio, advantages = _surrogate_case(seed, n)
    want_loss, want_grad = surrogate_torch(ratio, advantages, eps)
    np_loss, np_grad = surrogate_numpy(ratio, advantages, eps)
    if not (np.isclose(np_loss, want_loss, rtol=1e-12) and np.allclose(np_grad, want_grad, rtol=1e-12, atol=1e-15)):
        raise AssertionError(ORACLE_BUG)
    loss, grad = surrogate_of(drl, ratio, advantages, clip_eps=eps)
    assert_close(loss, want_loss, msg=f"clipped_surrogate, {n} ratios, clip_eps={eps}: loss, expected "
                                      "-torch.min(r A, torch.clamp(r, 1 - eps, 1 + eps) A).mean()")
    assert_close(grad, want_grad, msg=f"clipped_surrogate, {n} ratios, clip_eps={eps}: gradient, expected the "
                                      "torch autograd gradient with respect to the ratios")


@pytest.mark.parametrize("seed,n,eps", SURROGATE_CASES)
def test_clipped_surrogate_matches_the_formula(drl, seed, n, eps):
    ratio, advantages = _surrogate_case(seed, n)
    want_loss, want_grad = surrogate_numpy(ratio, advantages, eps)
    loss, grad = surrogate_of(drl, ratio, advantages, clip_eps=eps)
    assert_close(loss, want_loss, msg=f"clipped_surrogate, {n} ratios, clip_eps={eps}: loss, expected "
                                      "-mean(min(r A, clip(r, 1 - eps, 1 + eps) A))")
    assert_close(grad, want_grad, msg=f"clipped_surrogate, {n} ratios, clip_eps={eps}: gradient, expected -A/n "
                                      "where r A <= clip(r) A, else 0")


@pytest.mark.parametrize("r,A,want_grad,why", [
    pytest.param(1.1, 2.0, -2.0, "A > 0, ratio inside the interval: the unclipped term is the minimum",
                 id="A-positive-inside"),
    pytest.param(1.5, 2.0, 0.0, "A > 0, ratio above 1 + eps: the clipped term (1.2 A) is strictly smaller, "
                 "PPO stops pushing", id="A-positive-above"),
    pytest.param(0.5, 2.0, -2.0, "A > 0, ratio below 1 - eps: r A is the minimum, the gradient still pushes the "
                 "ratio up", id="A-positive-below"),
    pytest.param(0.9, -2.0, 2.0, "A < 0, ratio inside the interval: the unclipped term is the minimum",
                 id="A-negative-inside"),
    pytest.param(1.5, -2.0, 2.0, "A < 0, ratio above 1 + eps: r A is the minimum, the gradient still pushes the "
                 "ratio down", id="A-negative-above"),
    pytest.param(0.5, -2.0, 0.0, "A < 0, ratio below 1 - eps: the clipped term (0.8 A) is strictly smaller, PPO "
                 "stops pushing", id="A-negative-below"),
])
def test_clipped_surrogate_four_cases(drl, r, A, want_grad, why):
    want_loss = -min(r * A, min(max(r, 0.8), 1.2) * A)
    loss, grad = surrogate_of(drl, [r], [A])
    assert_close(grad, [want_grad], msg=f"clipped_surrogate([{r}], [{A}]): gradient ({why})")
    assert_close(loss, want_loss, msg=f"clipped_surrogate([{r}], [{A}]): loss -min(r A, clip(r) A)")


@pytest.mark.parametrize("eps,r", [pytest.param(0.25, 0.75, id="eps0.25-lower-bound-0.75"),
                                   pytest.param(0.25, 1.25, id="eps0.25-upper-bound-1.25"),
                                   pytest.param(0.5, 0.5, id="eps0.5-lower-bound-0.5"),
                                   pytest.param(0.5, 1.5, id="eps0.5-upper-bound-1.5")])
@pytest.mark.parametrize("A", [1.5, -1.5], ids=["A-positive", "A-negative"])
def test_clipped_surrogate_ratio_on_a_bound(drl, eps, r, A):
    # on a bound clip(r) = r exactly: r A <= clip(r) A holds (equality), so the docstring gives -A/n
    ratio, advantages = [r, 1.0], [A, 1.0]
    if torch is not None:
        _, tgrad = surrogate_torch(ratio, advantages, eps)
        if not np.allclose(tgrad, [-A / 2, -0.5], rtol=1e-12):
            raise AssertionError(ORACLE_BUG)
    loss, grad = surrogate_of(drl, ratio, advantages, clip_eps=eps)
    assert_close(grad, [-A / 2, -0.5], msg=f"clipped_surrogate([{r}, 1.0], [{A}, 1.0], clip_eps={eps}): gradient, "
                                           f"the ratio {r} is on the bound: r A = clip(r) A, the unclipped term is "
                                           "the minimum (<=), so -A/n")
    assert_close(loss, -(r * A + 1.0) / 2, msg=f"clipped_surrogate([{r}, 1.0], [{A}, 1.0], clip_eps={eps}): loss")


def test_clipped_surrogate_gradient_is_divided_by_n(drl):
    loss, grad = surrogate_of(drl, [1.0, 1.05, 0.95, 1.1], [1.0, -2.0, 3.0, 4.0])
    assert_close(grad, [-0.25, 0.5, -0.75, -1.0], msg="clipped_surrogate, 4 ratios inside the interval: expected "
                                                      "the gradient -A/n of the mean, n = 4")
    assert_close(loss, -(1.0 - 2.1 + 2.85 + 4.4) / 4, msg="clipped_surrogate, 4 ratios inside the interval: loss "
                                                          "-mean(r A)")


def test_clipped_surrogate_clips_the_ratio_not_the_advantage(drl):
    loss, grad = surrogate_of(drl, [1.5, 0.5], [5.0, -5.0])
    assert_close(loss, -(1.2 * 5.0 + 0.8 * -5.0) / 2, msg="clipped_surrogate([1.5, 0.5], [5, -5]): expected the "
                                                          "loss -mean(min(r A, clip(r) A)) with the ratio clipped "
                                                          "to [0.8, 1.2] (the advantages are not clipped)")
    assert_close(grad, [0.0, 0.0], msg="clipped_surrogate([1.5, 0.5], [5, -5]): both ratios are clipped on the "
                                       "side where the clipped term is strictly smaller: gradient 0")


def test_clipped_surrogate_default_clip_eps_is_02(drl):
    loss, grad = surrogate_of(drl, [1.19, 1.21, 0.81, 0.79], [1.0, 1.0, -1.0, -1.0])
    assert_close(grad, [-0.25, 0.0, 0.25, 0.0], msg="clipped_surrogate without clip_eps: expected the default "
                                                    "eps = 0.2 (1.21 and 0.79 are clipped, 1.19 and 0.81 are not)")
    assert_close(loss, -(1.19 + 1.2 - 0.81 - 0.8) / 4, msg="clipped_surrogate without clip_eps: loss with eps = 0.2")


def test_clipped_surrogate_zero_advantage_gives_zero(drl):
    loss, grad = surrogate_of(drl, [0.3, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert_close(loss, 0.0, msg="clipped_surrogate with zero advantages: expected the loss 0")
    assert_close(grad, [0.0, 0.0, 0.0], msg="clipped_surrogate with zero advantages: expected a zero gradient")


def test_clipped_surrogate_accepts_a_zero_ratio(drl):
    loss, grad = surrogate_of(drl, [0.0, 1.0], [2.0, -1.0])
    assert_close(loss, -(0.0 - 1.0) / 2, msg="clipped_surrogate([0, 1], [2, -1]): a ratio of 0 is allowed (>= 0); "
                                             "expected the loss -mean(min(r A, clip(r) A))")
    assert_close(grad, [-1.0, 0.5], msg="clipped_surrogate([0, 1], [2, -1]): gradient (r = 0 with A > 0: the "
                                        "unclipped term is the minimum)")


def test_clipped_surrogate_accepts_float32_arrays(drl):
    ratio, advantages = _surrogate_case(7, 128)
    ratio, advantages = ratio.astype(np.float32), advantages.astype(np.float32)
    want_loss, want_grad = surrogate_numpy(ratio, advantages, 0.2)
    loss, grad = surrogate_of(drl, ratio, advantages)
    assert_close(loss, want_loss, rtol=RTOL32, atol=ATOL32, msg="clipped_surrogate on float32 arrays: loss")
    assert_close(grad, want_grad, rtol=RTOL32, atol=ATOL32, msg="clipped_surrogate on float32 arrays: gradient")


def test_clipped_surrogate_does_not_modify_its_inputs(drl):
    ratio, advantages = np.array([0.5, 1.0, 1.5, 2.0]), np.array([1.0, -1.0, 2.0, -2.0])
    before = (ratio.copy(), advantages.copy())
    try:
        drl.clipped_surrogate(ratio, advantages, 0.2)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail("clipped_surrogate", f"expected the pair (loss, grad), but it raised {type(exc).__name__}: {exc}")
    assert_unchanged(before[0], ratio, "ratio", "clipped_surrogate")
    assert_unchanged(before[1], advantages, "advantages", "clipped_surrogate")


@pytest.mark.parametrize("ratio,advantages,kwargs,why", [
    pytest.param([1.0, 1.1], [1.0, 1.0], {"clip_eps": 0.0}, "clip_eps = 0 (must be > 0)", id="eps-zero"),
    pytest.param([1.0, 1.1], [1.0, 1.0], {"clip_eps": -0.2}, "clip_eps = -0.2 (must be > 0)", id="eps-negative"),
    pytest.param([1.0, -0.5], [1.0, 1.0], {}, "the ratio -0.5 is negative", id="ratio-negative"),
    pytest.param([1.0, -0.001, 0.5], [1.0, 1.0, 1.0], {}, "the ratio -0.001 is negative", id="ratio-slightly-negative"),
    pytest.param([1.0, 1.1, 0.9], [1.0, 1.0], {}, "3 ratios, 2 advantages", id="lengths-differ"),
    pytest.param(np.ones((3, 1)), np.ones(3), {}, "ratio of shape (3, 1), advantages (3,) (NumPy would broadcast "
                 "them to (3, 3))", id="ratio-3x1-broadcast"),
])
def test_clipped_surrogate_refuses_invalid_arguments(drl, ratio, advantages, kwargs, why):
    assert_raises(ValueError, drl.clipped_surrogate, ratio, advantages, why=why, **kwargs)


# ================================================================== bradley_terry_loss (B8.25)
def bt_numpy(r_chosen, r_rejected):
    """mean(logaddexp(0, -d)) and its gradients -sigmoid(-d)/n, +sigmoid(-d)/n, d = r_chosen - r_rejected."""
    d = np.asarray(r_chosen, float) - np.asarray(r_rejected, float)
    n = d.size
    s = np.exp(-np.logaddexp(0.0, d))                      # sigmoid(-d), stably
    return float(np.mean(np.logaddexp(0.0, -d))), -s / n, s / n


def bt_torch(r_chosen, r_rejected):
    """-F.logsigmoid(r_chosen - r_rejected).mean() and its autograd gradients (float64)."""
    rc = torch.tensor(np.asarray(r_chosen, float), dtype=torch.float64, requires_grad=True)
    rr = torch.tensor(np.asarray(r_rejected, float), dtype=torch.float64, requires_grad=True)
    loss = -F.logsigmoid(rc - rr).mean()
    loss.backward()
    return float(loss.detach()), rc.grad.numpy().copy(), rr.grad.numpy().copy()


def bt_of(drl, r_chosen, r_rejected, msg="bradley_terry_loss"):
    n = np.shape(r_chosen)[0]
    result = call(drl.bradley_terry_loss, r_chosen, r_rejected, expected="the tuple (loss, grad_chosen, "
                                                                         "grad_rejected)")
    loss, gc, gr = tuple_result(result, msg, ("loss", "grad_chosen", "grad_rejected"))
    return (number_result(loss, f"{msg}: loss"), array_result(gc, f"{msg}: grad_chosen", (n,), f"{n} pairs"),
            array_result(gr, f"{msg}: grad_rejected", (n,), f"{n} pairs"))


def test_bradley_terry_loss_docstring_examples(drl):
    loss, gc, gr = bt_of(drl, [0.0], [0.0])
    assert_close(loss, math.log(2), msg="bradley_terry_loss([0], [0]): loss of the docstring (log 2)")
    assert_close(gc, [-0.5], msg="bradley_terry_loss([0], [0]): grad_chosen of the docstring")
    assert_close(gr, [0.5], msg="bradley_terry_loss([0], [0]): grad_rejected of the docstring")
    loss, _, _ = bt_of(drl, [-1000.0], [1000.0])
    assert_close(loss, 2000.0, msg="bradley_terry_loss([-1000], [1000]): loss of the docstring (finite: computed "
                                   "with np.logaddexp, not log(sigmoid))")


def _bt_case(seed, n, scale=2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=n), rng.normal(0.0, scale, size=n)


BT_CASES = [pytest.param(0, 1, 2.0, id="n1"), pytest.param(1, 6, 2.0, id="n6"), pytest.param(2, 200, 5.0, id="n200"),
            pytest.param(3, 10, 300.0, id="n10-large-rewards")]


@needs_torch
@pytest.mark.parametrize("seed,n,scale", BT_CASES)
def test_bradley_terry_loss_matches_torch_logsigmoid(drl, seed, n, scale):
    rc, rr = _bt_case(seed, n, scale)
    want = bt_torch(rc, rr)
    numpy_version = bt_numpy(rc, rr)
    if not all(np.allclose(a, b, rtol=1e-12, atol=1e-15) for a, b in zip(want, numpy_version)):
        raise AssertionError(ORACLE_BUG)
    loss, gc, gr = bt_of(drl, rc, rr)
    assert_close(loss, want[0], msg=f"bradley_terry_loss, {n} pairs: loss, expected "
                                    "-F.logsigmoid(r_chosen - r_rejected).mean()")
    assert_close(gc, want[1], msg=f"bradley_terry_loss, {n} pairs: grad_chosen, expected the autograd gradient "
                                  "of the mean loss with respect to r_chosen")
    assert_close(gr, want[2], msg=f"bradley_terry_loss, {n} pairs: grad_rejected, expected the autograd gradient "
                                  "with respect to r_rejected")


@pytest.mark.parametrize("seed,n,scale", BT_CASES)
def test_bradley_terry_loss_matches_the_formula(drl, seed, n, scale):
    rc, rr = _bt_case(seed, n, scale)
    want = bt_numpy(rc, rr)
    loss, gc, gr = bt_of(drl, rc, rr)
    assert_close(loss, want[0], msg=f"bradley_terry_loss, {n} pairs: loss, expected mean(-log sigmoid(r_chosen - "
                                    "r_rejected))")
    assert_close(gc, want[1], msg=f"bradley_terry_loss, {n} pairs: grad_chosen, expected -sigmoid(r_rejected - "
                                  "r_chosen) / n")
    assert_close(gr, want[2], msg=f"bradley_terry_loss, {n} pairs: grad_rejected, expected -grad_chosen")


@pytest.mark.parametrize("rc,rr", [
    pytest.param([1000.0, -1000.0, 0.5], [-1000.0, 1000.0, 0.0], id="plus-minus-1000"),
    pytest.param([-5000.0, 40.0], [5000.0, -800.0], id="plus-minus-5000"),
    pytest.param([800.0, 800.0], [0.0, 1600.0], id="difference-800"),
])
def test_bradley_terry_loss_is_stable_for_large_differences(drl, rc, rr):
    with np.errstate(all="ignore"):
        loss, gc, gr = bt_of(drl, rc, rr)
    want = bt_numpy(rc, rr)
    if not (np.isfinite(loss) and np.isfinite(gc).all() and np.isfinite(gr).all()):
        _fail(f"bradley_terry_loss({rc}, {rr})", f"expected finite values (loss {want[0]:.6g}), got loss {loss!r}, "
                                                 f"grad_chosen {_short(gc)} (compute -log sigmoid(d) as "
                                                 "np.logaddexp(0, -d), and sigmoid without overflow)")
    assert_close(loss, want[0], msg=f"bradley_terry_loss({rc}, {rr}): loss for large differences")
    assert_close(gc, want[1], msg=f"bradley_terry_loss({rc}, {rr}): grad_chosen for large differences")
    assert_close(gr, want[2], msg=f"bradley_terry_loss({rc}, {rr}): grad_rejected for large differences")


def test_bradley_terry_loss_gradients_are_opposite(drl):
    rc, rr = _bt_case(10, 25)
    _, gc, gr = bt_of(drl, rc, rr)
    if not (gc <= 0).all():
        _fail("bradley_terry_loss", f"expected grad_chosen <= 0 (raising the reward of the chosen item lowers the "
                                    f"loss), got {_short(gc)}")
    assert_close(gr, -gc, msg="bradley_terry_loss: expected grad_rejected = -grad_chosen")


def test_bradley_terry_loss_gradient_is_divided_by_n(drl):
    _, gc, gr = bt_of(drl, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    assert_close(gc, [-0.125] * 4, msg="bradley_terry_loss with 4 ties: expected grad_chosen -sigmoid(0) / n = "
                                       "-0.5 / 4 (gradient of the mean)")
    assert_close(gr, [0.125] * 4, msg="bradley_terry_loss with 4 ties: expected grad_rejected 0.5 / 4")


def test_bradley_terry_loss_chosen_comes_first(drl):
    rc, rr = np.array([3.0, 1.0, 2.5]), np.array([1.0, 0.5, -1.0])
    loss_ab, _, _ = bt_of(drl, rc, rr)
    loss_ba, _, _ = bt_of(drl, rr, rc)
    assert_close(loss_ab, bt_numpy(rc, rr)[0], msg="bradley_terry_loss(r_chosen, r_rejected): expected a small "
                                                   "loss when the chosen rewards are the larger ones")
    assert_close(loss_ba - loss_ab, np.mean(rc - rr), msg="bradley_terry_loss: swapping chosen and rejected must "
                                                          "raise the loss by mean(r_chosen - r_rejected)")


def test_bradley_terry_loss_invariant_to_a_common_shift(drl):
    rc, rr = _bt_case(11, 7)
    loss, gc, gr = bt_of(drl, rc, rr)
    loss2, gc2, gr2 = bt_of(drl, rc + 100.0, rr + 100.0)
    assert_close(loss2, loss, msg="bradley_terry_loss: adding 100 to every reward must not change the loss (only "
                                  "the differences count)")
    assert_close(gc2, gc, msg="bradley_terry_loss: adding 100 to every reward must not change grad_chosen")


def test_bradley_terry_loss_accepts_float32_arrays(drl):
    rc, rr = _bt_case(12, 50)
    rc, rr = rc.astype(np.float32), rr.astype(np.float32)
    want = bt_numpy(rc, rr)
    loss, gc, gr = bt_of(drl, rc, rr)
    assert_close(loss, want[0], rtol=RTOL32, atol=ATOL32, msg="bradley_terry_loss on float32 arrays: loss")
    assert_close(gc, want[1], rtol=RTOL32, atol=ATOL32, msg="bradley_terry_loss on float32 arrays: grad_chosen")


def test_bradley_terry_loss_does_not_modify_its_inputs(drl):
    rc, rr = np.array([1.0, 2.0, -1.0]), np.array([0.5, 3.0, -2.0])
    before = (rc.copy(), rr.copy())
    try:
        drl.bradley_terry_loss(rc, rr)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail("bradley_terry_loss", f"expected a tuple, but it raised {type(exc).__name__}: {exc}")
    assert_unchanged(before[0], rc, "r_chosen", "bradley_terry_loss")
    assert_unchanged(before[1], rr, "r_rejected", "bradley_terry_loss")


@pytest.mark.parametrize("rc,rr,why", [
    pytest.param(np.zeros(3), np.zeros(2), "r_chosen has 3 values, r_rejected 2", id="lengths-differ"),
    pytest.param(np.zeros((3, 1)), np.zeros(3), "r_chosen of shape (3, 1), r_rejected (3,) (NumPy would "
                 "broadcast them to (3, 3))", id="chosen-3x1-broadcast"),
    pytest.param(np.zeros(1), np.zeros(4), "r_chosen has 1 value, r_rejected 4 (NumPy would broadcast)",
                 id="chosen-1-broadcast"),
])
def test_bradley_terry_loss_refuses_invalid_arguments(drl, rc, rr, why):
    assert_raises(ValueError, drl.bradley_terry_loss, rc, rr, why=why)


# ================================================================== dpo_loss (B8.25)
def dpo_numpy(pc, pr, rc, rr, beta):
    """Equation 7 of Rafailov et al. (2023): margins and mean(-log sigmoid(margins)) with np.logaddexp."""
    pc, pr, rc, rr = (np.asarray(a, float) for a in (pc, pr, rc, rr))
    margins = beta * ((pc - rc) - (pr - rr))
    return float(np.mean(np.logaddexp(0.0, -margins))), margins


def dpo_torch(pc, pr, rc, rr, beta):
    """Bradley-Terry on the implicit rewards beta log(pi / pi_ref), with torch.nn.functional.logsigmoid."""
    t = {k: torch.tensor(np.asarray(v, float), dtype=torch.float64) for k, v in
         dict(pc=pc, pr=pr, rc=rc, rr=rr).items()}
    reward_chosen = beta * (t["pc"] - t["rc"])
    reward_rejected = beta * (t["pr"] - t["rr"])
    return float(-F.logsigmoid(reward_chosen - reward_rejected).mean())


def dpo_of(drl, pc, pr, rc, rr, msg="dpo_loss", **kwargs):
    n = np.shape(pc)[0]
    result = call(drl.dpo_loss, pc, pr, rc, rr, expected="the pair (loss, margins)", **kwargs)
    loss, margins = tuple_result(result, msg, ("loss", "margins"))
    return number_result(loss, f"{msg}: loss"), array_result(margins, f"{msg}: margins", (n,), f"{n} pairs")


def _dpo_case(seed, n, scale=3.0):
    rng = np.random.default_rng(seed)
    ref_c, ref_r = rng.normal(-40.0, 10.0, size=n), rng.normal(-40.0, 10.0, size=n)
    return (ref_c + rng.normal(0.0, scale, size=n), ref_r + rng.normal(0.0, scale, size=n), ref_c, ref_r)


def test_dpo_loss_docstring_example(drl):
    loss, margins = dpo_of(drl, [-10.0, -12.0], [-11.0, -9.0], [-10.5, -11.0], [-10.5, -10.0], beta=0.1)
    assert_close(margins, [0.1, -0.2], msg="dpo_loss: margins of the docstring example")
    assert_close(loss, (math.log1p(math.exp(-0.1)) + math.log1p(math.exp(0.2))) / 2,
                 msg="dpo_loss: loss of the docstring example (0.7213)")


DPO_CASES = [pytest.param(0, 1, 0.1, id="n1-beta0.1"), pytest.param(1, 8, 0.1, id="n8-beta0.1"),
             pytest.param(2, 64, 0.5, id="n64-beta0.5"), pytest.param(3, 16, 2.0, id="n16-beta2")]


@pytest.mark.parametrize("seed,n,beta", DPO_CASES)
def test_dpo_loss_matches_equation_7(drl, seed, n, beta):
    data = _dpo_case(seed, n)
    want_loss, want_margins = dpo_numpy(*data, beta)
    loss, margins = dpo_of(drl, *data, beta=beta)
    assert_close(margins, want_margins, msg=f"dpo_loss, {n} pairs, beta={beta}: margins, expected beta ((log pi_c - "
                                            "log ref_c) - (log pi_r - log ref_r))")
    assert_close(loss, want_loss, msg=f"dpo_loss, {n} pairs, beta={beta}: loss, expected mean(-log sigmoid(margins))")


@needs_torch
@pytest.mark.parametrize("seed,n,beta", DPO_CASES)
def test_dpo_loss_is_bradley_terry_on_the_implicit_rewards(drl, seed, n, beta):
    data = _dpo_case(seed, n)
    want = dpo_torch(*data, beta)
    if not np.isclose(want, dpo_numpy(*data, beta)[0], rtol=1e-12):
        raise AssertionError(ORACLE_BUG)
    loss, _ = dpo_of(drl, *data, beta=beta)
    assert_close(loss, want, msg=f"dpo_loss, {n} pairs, beta={beta}: expected -F.logsigmoid(r_c - r_r).mean() with "
                                 "the implicit rewards r = beta (log pi - log ref)")


def test_dpo_loss_default_beta_is_01(drl):
    data = _dpo_case(4, 5)
    want_loss, want_margins = dpo_numpy(*data, 0.1)
    loss, margins = dpo_of(drl, *data)
    assert_close(margins, want_margins, msg="dpo_loss without beta: expected the default beta = 0.1 in the margins")
    assert_close(loss, want_loss, msg="dpo_loss without beta: loss with beta = 0.1")


def test_dpo_loss_margins_are_proportional_to_beta(drl):
    data = _dpo_case(5, 6)
    _, m1 = dpo_of(drl, *data, beta=0.1)
    _, m2 = dpo_of(drl, *data, beta=0.7)
    assert_close(m2, 7.0 * m1, msg="dpo_loss: the margins with beta = 0.7 must be 7 times those with beta = 0.1 "
                                   "(beta multiplies the whole difference)")


def test_dpo_loss_policy_equal_to_reference(drl):
    rng = np.random.default_rng(6)
    ref_c, ref_r = rng.normal(-30.0, 5.0, size=4), rng.normal(-30.0, 5.0, size=4)
    loss, margins = dpo_of(drl, ref_c, ref_r, ref_c, ref_r, beta=0.5)
    assert_close(margins, np.zeros(4), msg="dpo_loss with the policy equal to the reference: expected zero margins")
    assert_close(loss, math.log(2), msg="dpo_loss with the policy equal to the reference: expected log 2")


def test_dpo_loss_positive_margin_when_the_policy_prefers_chosen(drl):
    # the policy raises the chosen answer by 2 nats and lowers the rejected one by 1 nat, relative to the reference
    loss, margins = dpo_of(drl, [-8.0], [-11.0], [-10.0], [-10.0], beta=0.5)
    assert_close(margins, [1.5], msg="dpo_loss: chosen +2 nats and rejected -1 nat relative to the reference: "
                                     "expected the margin beta (2 - (-1)) = 1.5 (log pi - log ref, chosen minus "
                                     "rejected)")
    assert_close(loss, math.log1p(math.exp(-1.5)), msg="dpo_loss: loss -log sigmoid(1.5)")


def test_dpo_loss_only_ratios_to_the_reference_count(drl):
    data = _dpo_case(7, 5)
    _, margins = dpo_of(drl, *data, beta=0.3)
    pc, pr, rc, rr = data
    _, shifted = dpo_of(drl, pc + 50.0, pr, rc + 50.0, rr, beta=0.3)
    assert_close(shifted, margins, msg="dpo_loss: adding 50 to log pi_c and to log ref_c must not change the margins "
                                       "(only log pi - log ref counts)")


@pytest.mark.parametrize("scale", [pytest.param(1e4, id="logps-1e4"), pytest.param(1e6, id="logps-1e6")])
def test_dpo_loss_is_stable_for_large_margins(drl, scale):
    pc = np.array([-scale, 0.0, -1.0])
    pr = np.array([0.0, -scale, -1.0])
    zeros = np.zeros(3)
    with np.errstate(all="ignore"):
        loss, margins = dpo_of(drl, pc, pr, zeros, zeros, beta=1.0)
    want_loss, want_margins = dpo_numpy(pc, pr, zeros, zeros, 1.0)
    if not np.isfinite(loss):
        _fail("dpo_loss", f"expected a finite loss {want_loss:.6g} for margins of +-{scale:g}, got {loss!r} (compute "
                          "-log sigmoid(m) as np.logaddexp(0, -m))")
    assert_close(margins, want_margins, msg=f"dpo_loss: margins of +-{scale:g}")
    assert_close(loss, want_loss, msg=f"dpo_loss: loss for margins of +-{scale:g} (mean of {scale:g}, 0 and log 2)")


def test_dpo_loss_accepts_lists_and_float32(drl):
    data = _dpo_case(8, 12)
    want_loss, want_margins = dpo_numpy(*[a.astype(np.float32) for a in data], 0.1)
    loss, margins = dpo_of(drl, *[a.astype(np.float32) for a in data])
    assert_close(margins, want_margins, rtol=RTOL32, atol=ATOL32, msg="dpo_loss on float32 arrays: margins")
    assert_close(loss, want_loss, rtol=RTOL32, atol=ATOL32, msg="dpo_loss on float32 arrays: loss")
    loss, margins = dpo_of(drl, *[a.tolist() for a in data])
    assert_close(margins, dpo_numpy(*data, 0.1)[1], msg="dpo_loss on lists: margins")


def test_dpo_loss_does_not_modify_its_inputs(drl):
    data = _dpo_case(9, 4)
    before = [a.copy() for a in data]
    try:
        drl.dpo_loss(*data, beta=0.1)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail("dpo_loss", f"expected the pair (loss, margins), but it raised {type(exc).__name__}: {exc}")
    names = ("policy_chosen_logps", "policy_rejected_logps", "ref_chosen_logps", "ref_rejected_logps")
    for name, old, new in zip(names, before, data):
        assert_unchanged(old, new, name, "dpo_loss")


def _dpo_args(position=None, value=None):
    args = [np.full(3, -5.0), np.full(3, -6.0), np.full(3, -5.5), np.full(3, -5.5)]
    if position is not None:
        args[position] = value
    return args


@pytest.mark.parametrize("position,value,kwargs,why", [
    pytest.param(None, None, {"beta": 0.0}, "beta = 0 (must be > 0)", id="beta-zero"),
    pytest.param(None, None, {"beta": -0.1}, "beta = -0.1 (must be > 0)", id="beta-negative"),
    pytest.param(0, np.zeros(4), {}, "policy_chosen_logps has 4 values, the others 3", id="policy-chosen-longer"),
    pytest.param(1, np.zeros(2), {}, "policy_rejected_logps has 2 values, the others 3", id="policy-rejected-shorter"),
    pytest.param(2, np.zeros(4), {}, "ref_chosen_logps has 4 values, the others 3", id="ref-chosen-longer"),
    pytest.param(3, np.zeros(1), {}, "ref_rejected_logps has 1 value, the others 3 (NumPy would broadcast)",
                 id="ref-rejected-1-broadcast"),
    pytest.param(0, np.zeros((3, 1)), {}, "policy_chosen_logps of shape (3, 1), the others (3,) (NumPy would "
                 "broadcast them to (3, 3))", id="policy-chosen-3x1-broadcast"),
])
def test_dpo_loss_refuses_invalid_arguments(drl, position, value, kwargs, why):
    assert_raises(ValueError, drl.dpo_loss, *_dpo_args(position, value), why=why, **kwargs)
