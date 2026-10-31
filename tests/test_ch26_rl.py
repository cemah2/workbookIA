"""Tests of mylearn.rl (chapter 26): oracle tests and properties.

    pytest tests/test_ch26_rl.py              # your code (mon_travail/mylearn/rl.py)
    pytest tests/test_ch26_rl.py --impl=ref   # the reference

Oracles: ``numpy.polynomial.polynomial.polyval`` and ``scipy.signal.lfilter`` for the returns, ``np.argmax``
for the greedy policy, ``torch.softmax`` (``scipy.special.softmax`` without PyTorch) with
``scipy.stats.chisquare`` and ``Generator.choice`` for the Boltzmann draws, the TD formulas of the docstrings
transcribed in this file with NumPy for the updates (also replayed over whole training runs), gymnasium
(``FrozenLake-v1``, ``CliffWalking-v1``) and small environments written in this file for the training loop and
the evaluation, a value iteration and an exact policy evaluation (``np.linalg.solve`` of the linear Bellman
system) written in this file, and closed-form values on chain MDPs. Random tests use fixed seeds and generous
margins (a correct implementation fails them with a probability below one in a million). The tests of one
function never use another function of yours as an oracle. Your functions are called on copies of the test
data, and the tests ``..._does_not_modify_...`` check that the inputs are left as they were.

Every test name starts with the name of what it tests, so that each exercise runs its own group:
``-k "test_discounted_return_ or test_returns_to_go_"`` (26.14),
``-k "test_greedy_policy_ or test_softmax_action_"`` (26.19),
``-k "test_q_learning_update_ or test_sarsa_update_"`` (26.20),
``-k "test_train_tabular_ or test_evaluate_policy_"`` (26.21), ``-k "test_value_iteration_"`` (26.26),
``-k "test_replay_buffer_"`` (26.31).

Contract precisions written in the statements of these exercises and checked here:
- invalid inputs raise ``ValueError`` (never ``IndexError`` nor ``TypeError``): gamma outside [0, 1] (outside
  [0, 1) for ``value_iteration``), alpha outside (0, 1], epsilon outside [0, 1], temperature <= 0, a count
  (``n_episodes``, ``max_steps``, ``max_iter``, ``batch_size``) below 1 or given as a bool, rewards that are
  not 1-D, a Q that is not 2-D, NaN in the rewards, in Q or in a reward, empty ``q_values``, a state or an
  action outside 0 .. n - 1 (a negative index is refused, not wrapped) or given as a bool;
- the returns are new float64 arrays; ``discounted_return`` and the TD errors are Python floats (``float(...)``,
  not ``np.float64``), ``softmax_action`` a Python int; an empty episode gives 0.0 and an array of shape (0,);
- ``greedy_policy`` sends ties to the lowest index; ``softmax_action`` subtracts the maximum before ``exp`` and
  draws exactly one number per call, with ``rng.choice(len(q_values), p=probs)``;
- ``q_learning_update`` and ``sarsa_update`` take a float NumPy array Q (else ``ValueError``), modify it in place
  (only ``Q[state, action]``) and return the TD error computed before the update; ``terminated=True`` gives the
  target ``reward`` even if ``Q[next_state]`` is not zero (the mask and ``next_action`` are then ignored);
  ``next_mask`` may be boolean or 0/1 integers (gymnasium gives int8 masks) and an all-False mask with
  ``terminated=False`` is refused;
- ``train_tabular`` draws the seed of every ``env.reset`` with ``int(rng.integers(2**31))`` (a Python int, the
  first draw of the episode), calls ``epsilon(episode_index)`` once per episode, never steps an episode that has
  ended, treats ``max_steps`` and ``truncated`` as truncations (the last update bootstraps), plays only legal
  actions when ``info`` has an ``"action_mask"`` (bool or 0/1), breaks ties at random (``epsilon_greedy_action``),
  and returns a float64 Q and the history ``returns`` (float64), ``lengths`` (integers), ``epsilons`` (float64);
  ``Q_init`` must have the shape (n_states, n_actions);
- ``evaluate_policy`` draws its seeds like ``train_tabular``, accepts a list or an integer array as a table and
  refuses a float table; it returns Python floats;
- ``value_iteration`` starts from V = 0, its sweeps are synchronous (with ``max_iter = k`` it returns exactly the
  k-th iterate), it stops when the largest change is strictly below ``theta`` (> 0), its Q is one more backup of
  the returned V; P is refused when a state has no actions, when the probabilities of an action do not sum to 1,
  or when a next state is outside 0 .. n_states - 1;
- ``ReplayBuffer.push`` refuses an observation whose shape is not ``obs_shape`` (no broadcasting), and ``sample``
  returns arrays that share no memory with the buffer.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

import copy
import warnings

import numpy as np
import pytest
from numpy.polynomial import polynomial as npoly
from scipy import signal, stats

gym = pytest.importorskip("gymnasium")

try:
    import torch
except ImportError:  # the PyTorch oracle runs on Colab; scipy.special.softmax takes over here
    torch = None


@pytest.fixture
def rl(mylearn_module):
    return mylearn_module("rl")


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


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg=""):
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


def pair(result, what, names):
    """Unpack a 2-tuple result, or fail with a message that says what was expected."""
    label = f"the tuple ({', '.join(names)})"
    if result is None:
        _fail("", f"expected {label} from {what}, got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 2:
        _fail("", f"expected {label} from {what}, got {type(result).__name__}"
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


def assert_python_float(value, what):
    if value is None:
        _fail("", f"expected {what} to be a Python float, got None (did you forget the return?)")
    if type(value) is not float:
        why = ("" if type(value).__module__.startswith("torch")
               else ": a NumPy scalar does not print like the docstring")
        _fail("", f"expected {what} to be a Python float, got {type(value).__name__} {value!r} "
                  f"(convert it with float(...){why})")


def assert_unchanged(arrays, saved, what):
    for name, arr in arrays.items():
        if not np.array_equal(arr, saved[name]):
            _fail("", f"expected {name} unchanged by {what}, got {_short(arr)} (compute the result in new arrays)")


def chisquare_p(counts, probs):
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(stats.chisquare(counts, counts.sum() * probs / probs.sum()).pvalue)


def softmax_probs(q, temperature):
    """Boltzmann probabilities from torch.softmax (scipy.special.softmax without PyTorch)."""
    q = np.asarray(q, dtype=np.float64) / temperature
    if torch is not None:
        return torch.softmax(torch.tensor(q, dtype=torch.float64), dim=0).numpy()
    from scipy.special import softmax

    return softmax(q)


def np_q_target(Q, reward, next_state, terminated, gamma, mask=None, next_action=None):
    """The TD target of the docstrings (Q-learning when next_action is None, SARSA otherwise)."""
    if terminated:
        return reward
    if next_action is not None:
        return reward + gamma * Q[next_state, next_action]
    row = Q[next_state] if mask is None else Q[next_state][np.asarray(mask, dtype=bool)]
    return reward + gamma * row.max()


# ------------------------------------------------------------------ MDP oracles written in this file
def np_backup(P, V, gamma):
    """Q[s, a] = sum of p * (r + gamma * V[s'] * (1 - done)), with plain loops over P."""
    S, A = len(P), len(P[0])
    Q = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            Q[s, a] = sum(p * (r + gamma * V[int(s2)] * (1.0 - float(d))) for p, s2, r, d in P[s][a])
    return Q


def np_value_iteration(P, gamma, sweeps):
    """`sweeps` synchronous Bellman optimality sweeps from V = 0; returns (V, Q = backup(V))."""
    V = np.zeros(len(P))
    for _ in range(sweeps):
        V = np_backup(P, V, gamma).max(axis=1)
    return V, np_backup(P, V, gamma)


def np_policy_value(P, gamma, policy):
    """Exact V and Q of a deterministic policy: np.linalg.solve of (I - gamma P_pi) V = R_pi."""
    S = len(P)
    T, R = np.zeros((S, S)), np.zeros(S)
    for s in range(S):
        for p, s2, r, d in P[s][int(policy[s])]:
            R[s] += p * r
            if not d:
                T[s, int(s2)] += p * gamma
    V = np.linalg.solve(np.eye(S) - T, R)
    return V, np_backup(P, V, gamma)


def chain_mdp(n, p=1.0, reward=1.0, towards=+1):
    """States 0 .. n-1; the goal (n-1 if towards=+1, else 0) is absorbing and terminal. Action 0 stays (reward 0);
    action 1 moves one step towards the goal with probability p (else stays); entering the goal pays `reward`
    and ends the episode."""
    goal = n - 1 if towards > 0 else 0
    P = {}
    for s in range(n):
        if s == goal:
            P[s] = {0: [(1.0, s, 0.0, True)], 1: [(1.0, s, 0.0, True)]}
            continue
        s2 = s + towards
        move = (p, s2, reward if s2 == goal else 0.0, s2 == goal)
        P[s] = {0: [(1.0, s, 0.0, False)], 1: [move] if p == 1.0 else [move, (1.0 - p, s, 0.0, False)]}
    return P, goal


def chain_closed_form(n, p, reward, gamma, towards=+1):
    """V(k) = c**(k-1) * v1 at distance k >= 1 of the goal, v1 = p r / (1 - (1-p) gamma),
    c = p gamma / (1 - (1-p) gamma); Q(s, stay) = gamma V(s), Q(s, move) = V(s); 0 at the goal."""
    goal = n - 1 if towards > 0 else 0
    v1 = p * reward / (1.0 - (1.0 - p) * gamma)
    c = p * gamma / (1.0 - (1.0 - p) * gamma)
    V = np.array([0.0 if s == goal else c ** (abs(goal - s) - 1) * v1 for s in range(n)])
    Q = np.column_stack([gamma * V, V])
    Q[goal] = 0.0
    return V, Q


# ------------------------------------------------------------------ environments written in this file
class Discrete:
    def __init__(self, n):
        self.n = n


class EpisodeOver(Exception):
    """step() called after the end of an episode, or before reset()."""


class IllegalAction(Exception):
    """An action outside info['action_mask'] was played."""


class LoopEnv:
    """One state 0, ``n_actions`` actions, ``reward`` per step, never terminated; truncated after ``limit``
    steps (None: never), terminated after ``terminate_after`` steps (None: never). Records seeds and actions."""

    def __init__(self, n_actions=1, reward=1.0, limit=None, terminate_after=None):
        self.observation_space, self.action_space = Discrete(1), Discrete(n_actions)
        self.reward, self.limit, self.terminate_after = reward, limit, terminate_after
        self.seeds, self.actions, self.t, self.over = [], [], 0, True

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.actions.append([])
        self.t, self.over = 0, False
        return 0, {}

    def step(self, action):
        if self.over:
            raise EpisodeOver("step() after the end of the episode (call reset() first)")
        self.t += 1
        self.actions[-1].append(int(action))
        terminated = self.terminate_after is not None and self.t >= self.terminate_after
        truncated = self.limit is not None and self.t >= self.limit
        self.over = terminated or truncated
        return 0, self.reward, terminated, truncated, {}


class RandomMDPEnv:
    """States 0..4 and the terminal state 5, 3 actions; the dynamics come from a generator seeded by reset().
    Every episode terminates (state 5, or terminated=True after 25 steps). Records every transition."""

    def __init__(self):
        self.observation_space, self.action_space = Discrete(6), Discrete(3)
        self.transitions, self.seeds, self.over = [], [], True

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.gen = np.random.default_rng(seed)
        self.state, self.t, self.over = 0, 0, False
        return 0, {"step": 0}

    def step(self, action):
        if self.over:
            raise EpisodeOver("step() after the end of the episode (call reset() first)")
        a = int(action)
        s = self.state
        s2 = int((s + a + self.gen.integers(0, 3)) % 6)
        r = float(self.gen.normal(s - a, 1.0))
        self.t += 1
        terminated = s2 == 5 or self.t >= 25
        self.transitions.append((s, a, r, s2, terminated))
        self.state, self.over = s2, terminated
        return s2, r, terminated, False, {"step": self.t}


class MaskEnv:
    """Three states, three actions; state 2 is terminal. State 0: legal actions 0 (-> state 1, reward 0) and
    1 (-> state 2, reward 0); state 1: legal action 2 only (-> state 2, reward 1). The mask of the terminal
    state is all False. Masks of dtype ``mask_dtype`` (bool, or int8 like gymnasium); an illegal action raises."""

    MASKS = {0: [1, 1, 0], 1: [0, 0, 1], 2: [0, 0, 0]}

    def __init__(self, mask_dtype=bool):
        self.observation_space, self.action_space = Discrete(3), Discrete(3)
        self.dtype, self.played, self.state = mask_dtype, [], None

    def _info(self, s):
        return {"action_mask": np.array(self.MASKS[s], dtype=self.dtype)}

    def reset(self, seed=None):
        self.state = 0
        return 0, self._info(0)

    def step(self, action):
        s, a = self.state, int(action)
        if not self.MASKS[s][a]:
            raise IllegalAction(f"action {a} played in state {s}, where the legal actions are "
                                f"{np.flatnonzero(self.MASKS[s]).tolist()} (info['action_mask'])")
        self.played.append((s, a))
        if s == 0 and a == 0:
            self.state, r = 1, 0.0
        else:
            self.state, r = 2, 1.0 if s == 1 else 0.0
        return self.state, r, self.state == 2, False, self._info(self.state)


class Recorder:
    """Wraps a gymnasium environment: records the seeds, the rewards and the number of steps of every episode,
    and refuses a step after the end of an episode."""

    def __init__(self, env):
        self.env = env
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.seeds, self.returns, self.lengths, self.over = [], [], [], True

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.returns.append(0.0)
        self.lengths.append(0)
        self.over = False
        return self.env.reset(seed=seed)

    def step(self, action):
        if self.over:
            raise EpisodeOver("step() after the end of the episode (call reset() first)")
        out = self.env.step(action)
        self.returns[-1] += float(out[1])
        self.lengths[-1] += 1
        self.over = bool(out[2] or out[3])
        return out


def frozen_lake(slippery=False, map_name="4x4"):
    return gym.make("FrozenLake-v1", is_slippery=slippery, map_name=map_name)


def greedy_path(Q, P, start, max_len=60):
    """States visited by the greedy policy np.argmax(Q[s]) in a deterministic model P, until termination."""
    s, path = start, [start]
    for _ in range(max_len):
        _, s2, _, done = P[s][int(np.argmax(Q[s]))][0]
        s = int(s2)
        path.append(s)
        if done:
            break
    return path


def train(rl, env, n, **kwargs):
    result = call(rl.train_tabular, env, n, expected="the tuple (Q, history)", **kwargs)
    return pair(result, "train_tabular", ("Q", "history"))


# ================================================================== discounted_return (26.14)
def test_discounted_return_docstring_examples(rl):
    got = call(rl.discounted_return, [1.0, 1.0, 1.0], gamma=0.5, expected="1.75")
    assert_python_float(got, "discounted_return([1, 1, 1], 0.5)")
    assert_close(got, 1.75, msg="1 + 0.5 * 1 + 0.25 * 1")
    empty = call(rl.discounted_return, [], gamma=0.9, expected="0.0 for an empty episode")
    assert_python_float(empty, "discounted_return([], 0.9)")
    assert_close(empty, 0.0, msg="an empty episode")


@pytest.mark.parametrize("seed, T, gamma", [
    pytest.param(0, 1, 0.9, id="T1"), pytest.param(1, 7, 0.9, id="T7"), pytest.param(2, 50, 0.99, id="T50"),
    pytest.param(3, 20, 0.0, id="gamma0"), pytest.param(4, 20, 1.0, id="gamma1"),
    pytest.param(5, 30, 0.37, id="gamma0.37"),
])
def test_discounted_return_matches_polyval(rl, seed, T, gamma):
    rewards = np.random.default_rng(seed).normal(size=T)
    got = call(rl.discounted_return, rewards.copy(), gamma, expected="a float")
    assert_python_float(got, "discounted_return")
    assert_close(got, npoly.polyval(gamma, rewards), rtol=1e-10,
                 msg=f"sum of gamma**k * r_k (k from 0), gamma = {gamma}, T = {T}")


def test_discounted_return_gamma_zero_is_the_first_reward(rl):
    assert_close(call(rl.discounted_return, [3.0, 5.0, 7.0], 0.0, expected="3.0"), 3.0,
                 msg="with gamma = 0 only r_0 counts (gamma**0 = 1)")


def test_discounted_return_accepts_lists_tuples_and_integers(rl):
    assert_close(call(rl.discounted_return, (1, 2, 3), 0.5, expected="2.75"), 2.75, msg="the tuple (1, 2, 3)")


def test_discounted_return_does_not_modify_its_input(rl):
    rewards = np.array([1.0, -2.0, 3.0])
    call(rl.discounted_return, rewards, 0.9, expected="a float")
    assert_unchanged({"rewards": rewards}, {"rewards": np.array([1.0, -2.0, 3.0])}, "discounted_return")


@pytest.mark.parametrize("rewards, gamma, why", [
    pytest.param([1.0, 2.0], -0.1, "gamma = -0.1 is outside [0, 1]", id="gamma-negative"),
    pytest.param([1.0, 2.0], 1.1, "gamma = 1.1 is outside [0, 1]", id="gamma-above-1"),
    pytest.param([[1.0, 2.0]], 0.5, "rewards of shape (1, 2): one episode is a 1-D array", id="2d"),
    pytest.param(1.0, 0.5, "rewards is a single number: one episode is a 1-D array", id="0d"),
    pytest.param([1.0, np.nan], 0.5, "rewards contain NaN", id="nan"),
])
def test_discounted_return_refuses_invalid_inputs(rl, rewards, gamma, why):
    assert_raises(ValueError, rl.discounted_return, rewards, gamma, why=why)


# ================================================================== returns_to_go (26.14)
def test_returns_to_go_docstring_examples(rl):
    got = call(rl.returns_to_go, [1.0, 0.0, 2.0], gamma=0.5, expected="array([1.5, 1. , 2. ])")
    assert_new_float_array(got, (3,), "returns_to_go([1, 0, 2], 0.5)")
    assert_close(got, [1.5, 1.0, 2.0], msg="G_2 = 2, G_1 = 0 + 0.5 * 2, G_0 = 1 + 0.5 * 1")
    assert_close(call(rl.returns_to_go, [1.0, 1.0, 1.0], expected="array([3., 2., 1.])"), [3.0, 2.0, 1.0],
                 msg="default gamma = 1: the reversed cumulative sum")


@pytest.mark.parametrize("seed, T, gamma", [
    pytest.param(0, 1, 0.9, id="T1"), pytest.param(1, 8, 0.9, id="T8"), pytest.param(2, 60, 0.99, id="T60"),
    pytest.param(3, 15, 0.0, id="gamma0"), pytest.param(4, 15, 0.5, id="gamma0.5"),
])
def test_returns_to_go_matches_lfilter(rl, seed, T, gamma):
    rewards = np.random.default_rng(seed).normal(size=T)
    got = call(rl.returns_to_go, rewards.copy(), gamma, expected=f"an array of shape ({T},)")
    assert_new_float_array(got, (T,), "the returns")
    want = signal.lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1]
    assert_close(got, want, rtol=1e-10, atol=1e-12, msg=f"G_t = r_t + gamma * G_(t+1), gamma = {gamma}")
    assert_close(got[0], npoly.polyval(gamma, rewards), rtol=1e-10, msg="G_0 = discounted return of the episode")


def test_returns_to_go_gamma_one_is_the_reversed_cumsum(rl):
    rewards = np.random.default_rng(7).integers(-3, 4, size=12).astype(float)
    got = call(rl.returns_to_go, rewards.copy(), 1.0, expected="an array of shape (12,)")
    assert_close(got, np.cumsum(rewards[::-1])[::-1], msg="gamma = 1: G_t = r_t + ... + r_(T-1)")


def test_returns_to_go_empty_episode(rl):
    got = call(rl.returns_to_go, [], 0.9, expected="an empty array of shape (0,)")
    assert_new_float_array(got, (0,), "returns_to_go([])")


def test_returns_to_go_returns_a_new_array(rl):
    rewards = np.array([1.0, 2.0, 3.0])
    got = call(rl.returns_to_go, rewards, 0.0, expected="an array of shape (3,)")
    assert_unchanged({"rewards": rewards}, {"rewards": np.array([1.0, 2.0, 3.0])}, "returns_to_go")
    assert_new_float_array(got, (3,), "the returns (here equal to the rewards, gamma = 0)", inputs=(rewards,))


def test_returns_to_go_accepts_integer_lists(rl):
    got = call(rl.returns_to_go, [1, 2, 3], 0.5, expected="an array of shape (3,)")
    assert_new_float_array(got, (3,), "returns_to_go([1, 2, 3])")
    assert_close(got, [2.75, 3.5, 3.0], msg="integer rewards")


@pytest.mark.parametrize("rewards, gamma, why", [
    pytest.param([1.0, 2.0], -0.5, "gamma = -0.5 is outside [0, 1]", id="gamma-negative"),
    pytest.param([1.0, 2.0], 2.0, "gamma = 2 is outside [0, 1]", id="gamma-above-1"),
    pytest.param(np.ones((2, 2)), 0.5, "rewards of shape (2, 2): one episode is a 1-D array", id="2d"),
    pytest.param([np.nan], 0.5, "rewards contain NaN", id="nan"),
])
def test_returns_to_go_refuses_invalid_inputs(rl, rewards, gamma, why):
    assert_raises(ValueError, rl.returns_to_go, rewards, gamma, why=why)


# ================================================================== greedy_policy (26.19)
def test_greedy_policy_docstring_example(rl):
    got = call(rl.greedy_policy, [[0.0, 1.0], [2.0, 2.0]], expected="array([1, 0])")
    if not isinstance(got, np.ndarray) or not np.issubdtype(got.dtype, np.integer):
        _fail("", f"expected an integer NumPy array, got {type(got).__name__} {_short(got)}")
    if got.tolist() != [1, 0]:
        _fail("", f"expected [1, 0] (a tie goes to the lowest index), got {got.tolist()}")


@pytest.mark.parametrize("seed, shape", [pytest.param(0, (16, 4), id="16x4"), pytest.param(1, (48, 4), id="48x4"),
                                         pytest.param(2, (5, 9), id="5x9"), pytest.param(3, (1, 1), id="1x1")])
def test_greedy_policy_matches_argmax_with_ties(rl, seed, shape):
    Q = np.random.default_rng(seed).integers(0, 3, size=shape).astype(float)   # many ties
    for _ in range(3):                                                        # ties never go at random
        got = call(rl.greedy_policy, Q.copy(), expected=f"an array of shape ({shape[0]},)")
        if not isinstance(got, np.ndarray):
            _fail("", f"expected a NumPy array of shape ({shape[0]},), got {type(got).__name__} "
                      "(convert with np.asarray(...))")
        if got.shape != (shape[0],):
            _fail("", f"expected an array of shape ({shape[0]},), got shape {got.shape}")
        want = np.argmax(Q, axis=1)
        if not np.array_equal(got, want):
            i = int(np.flatnonzero(got != want)[0])
            _fail("", f"expected action {want[i]} in state {i} (np.argmax: the lowest index among the ties), got "
                      f"{got[i]}", f"Q[{i}] = {_short(Q[i])}")


def test_greedy_policy_does_not_modify_its_input(rl):
    Q = np.array([[1.0, 2.0], [3.0, 0.0]])
    call(rl.greedy_policy, Q, expected="an array of shape (2,)")
    assert_unchanged({"Q": Q}, {"Q": np.array([[1.0, 2.0], [3.0, 0.0]])}, "greedy_policy")


@pytest.mark.parametrize("Q, why", [
    pytest.param(np.zeros(4), "Q of shape (4,): a table (n_states, n_actions) is 2-D", id="1d"),
    pytest.param(np.zeros((2, 2, 2)), "Q of shape (2, 2, 2) is 3-D", id="3d"),
    pytest.param(np.array([[0.0, np.nan]]), "Q contains NaN", id="nan"),
    pytest.param(np.zeros((3, 0)), "Q has no action (shape (3, 0))", id="no-action"),
])
def test_greedy_policy_refuses_invalid_inputs(rl, Q, why):
    assert_raises(ValueError, rl.greedy_policy, Q, why=why)


# ================================================================== softmax_action (26.19)
def test_softmax_action_docstring_example(rl):
    got = call(rl.softmax_action, [0.0, 10.0, 0.0], temperature=0.01, rng=np.random.default_rng(0),
               expected="1")
    if type(got) is not int or got != 1:
        _fail("", f"expected the Python int 1 (T = 0.01: almost greedy), got {type(got).__name__} {got!r}")


@pytest.mark.parametrize("q, temperature", [
    pytest.param([0.0, 1.0, 2.0], 1.0, id="T1"), pytest.param([3.0, -1.0, 0.5, 0.5], 0.7, id="T0.7"),
    pytest.param([1e4, 1e4 + 2.0, 1e4 - 1.0], 2.0, id="large-q"), pytest.param([0.0, 1000.0], 1.0, id="certain"),
])
def test_softmax_action_draws_like_rng_choice(rl, q, temperature):
    probs = softmax_probs(q, temperature)
    mine, ref = np.random.default_rng(5), np.random.default_rng(5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # an overflow in exp would be a RuntimeWarning
        got = [call(rl.softmax_action, list(q), temperature, rng=mine, expected="an action") for _ in range(300)]
    want = [int(ref.choice(len(q), p=probs)) for _ in range(300)]
    if got != want:
        i = next(k for k, (a, b) in enumerate(zip(got, want)) if a != b)
        _fail("", f"expected the draw {i} to be {want[i]} like rng.choice({len(q)}, p=softmax(q / T)), got {got[i]}",
              f"probabilities {_short(probs)} (subtract max(q) before exp, then draw once with rng.choice)")
    if mine.bit_generator.state != ref.bit_generator.state:
        _fail("", "expected exactly one draw from rng per call (rng.choice(n_actions, p=probs)), got a generator "
                  "that has moved differently")


@pytest.mark.parametrize("q, temperature, n", [
    pytest.param([0.5, -0.3, 1.2, 0.0], 0.8, 100_000, id="T0.8"),
    pytest.param([1e4, 1e4 + 1.0, 1e4 - 1.0, 1e4], 1.0, 40_000, id="large-q"),
])
def test_softmax_action_frequencies_follow_torch_softmax(rl, q, temperature, n):
    probs = softmax_probs(q, temperature)
    rng = np.random.default_rng(12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        draws = [call(rl.softmax_action, q, temperature, rng=rng, expected="an action") for _ in range(n)]
    counts = np.bincount(draws, minlength=len(q))
    if counts.size != len(q):
        _fail("", f"expected actions in 0 .. {len(q) - 1}, got {sorted(set(draws))}")
    p_value = chisquare_p(counts, probs)
    if p_value < 1e-6:
        _fail("", f"expected frequencies close to softmax(q / T) = {_short(probs, 4)}, got "
                  f"{_short(counts / n, 4)} (chi-square p-value {p_value:.1e})")


def test_softmax_action_temperature_extremes(rl):
    rng = np.random.default_rng(3)
    cold = {call(rl.softmax_action, [0.0, 1.0, 0.5], 1e-3, rng=rng, expected="an action") for _ in range(200)}
    if cold != {1}:
        _fail("", f"expected only action 1 with T = 0.001 (greedy), got the actions {sorted(cold)}")
    hot = [call(rl.softmax_action, [0.0, 1.0, 0.5], 1e6, rng=rng, expected="an action") for _ in range(30_000)]
    p_value = chisquare_p(np.bincount(hot, minlength=3), [1, 1, 1])
    if p_value < 1e-6:
        _fail("", f"expected almost uniform choices with T = 1e6, got the frequencies "
                  f"{_short(np.bincount(hot, minlength=3) / 30_000, 3)}")


def test_softmax_action_without_rng(rl):
    got = {call(rl.softmax_action, [0.0, 0.0], expected="an action") for _ in range(100)}
    if not got <= {0, 1} or len(got) != 2:
        _fail("", f"expected both actions 0 and 1 over 100 draws with equal values and rng=None, got {sorted(got)}")


def test_softmax_action_does_not_modify_its_input(rl):
    q = np.array([1.0, 3.0, 2.0])
    call(rl.softmax_action, q, 0.5, rng=np.random.default_rng(0), expected="an action")
    assert_unchanged({"q_values": q}, {"q_values": np.array([1.0, 3.0, 2.0])}, "softmax_action")


@pytest.mark.parametrize("q, temperature, why", [
    pytest.param([1.0, 2.0], 0.0, "temperature = 0 (it must be > 0)", id="T0"),
    pytest.param([1.0, 2.0], -1.0, "temperature = -1 (it must be > 0)", id="T-negative"),
    pytest.param([1.0, 2.0], np.nan, "temperature is NaN", id="T-nan"),
    pytest.param([], 1.0, "q_values is empty: no action to choose", id="empty"),
    pytest.param([[1.0, 2.0]], 1.0, "q_values of shape (1, 2): the values of one state are 1-D", id="2d"),
    pytest.param([1.0, np.nan], 1.0, "q_values contain NaN", id="nan"),
])
def test_softmax_action_refuses_invalid_inputs(rl, q, temperature, why):
    assert_raises(ValueError, rl.softmax_action, q, temperature, rng=np.random.default_rng(0), why=why)


# ================================================================== q_learning_update (26.20)
def test_q_learning_update_docstring_example(rl):
    Q = np.zeros((2, 2))
    Q[1] = [1.0, 3.0]
    delta = call(rl.q_learning_update, Q, 0, 1, reward=1.0, next_state=1, terminated=False, alpha=0.5, gamma=0.9,
                 expected="the TD error 3.7")
    assert_python_float(delta, "the TD error")
    assert_close(delta, 3.7, msg="delta = 1 + 0.9 * max(Q[1]) - Q[0, 1], computed before the update")
    assert_close(Q, [[0.0, 1.85], [1.0, 3.0]], msg="Q after the update, in place (only Q[0, 1] changes)")


def test_q_learning_update_hand_computed(rl):
    Q = np.array([[0.5, 0.0], [2.0, -1.0]])
    delta = call(rl.q_learning_update, Q, 0, 0, -1.0, 1, False, 0.1, 0.5, expected="the TD error -0.5")
    assert_close(delta, -0.5, msg="target = -1 + 0.5 * 2 = 0, delta = 0 - 0.5")
    assert_close(Q, [[0.45, 0.0], [2.0, -1.0]], msg="Q[0, 0] = 0.5 + 0.1 * (-0.5)")


@pytest.mark.parametrize("seed", range(6))
def test_q_learning_update_matches_the_formula(rl, seed):
    gen = np.random.default_rng(seed)
    Q = gen.normal(size=(6, 4))
    s, a, s2 = (int(x) for x in gen.integers(0, [6, 4, 6]))
    reward, alpha, gamma = float(gen.normal()), float(gen.uniform(0.05, 1.0)), float(gen.uniform(0, 1))
    mask = None if seed % 2 == 0 else gen.random(4) < 0.6
    if mask is not None:
        mask[gen.integers(4)] = True
    want_Q = Q.copy()
    target = np_q_target(Q, reward, s2, False, gamma, mask=mask)
    want_delta = target - Q[s, a]
    want_Q[s, a] += alpha * want_delta
    work = Q.copy()
    delta = call(rl.q_learning_update, work, s, a, reward, s2, False, alpha, gamma,
                 next_mask=None if mask is None else mask.copy(), expected="the TD error")
    assert_python_float(delta, "the TD error")
    assert_close(delta, want_delta, msg=f"delta = r + gamma * max(Q[{s2}]{' legal' if mask is not None else ''})"
                                        f" - Q[{s}, {a}]")
    assert_close(work, want_Q, msg=f"Q after the update (only Q[{s}, {a}] changes, by alpha * delta)")


def test_q_learning_update_no_bootstrap_on_a_terminal_state(rl):
    Q = np.array([[1.0, 0.0], [5.0, 7.0]])           # Q[next_state] is not zero
    delta = call(rl.q_learning_update, Q, 0, 0, 2.0, 1, True, 0.5, 0.9, expected="the TD error 1.0")
    assert_close(delta, 1.0, msg="terminated: target = reward = 2 (no gamma * max Q[1]), delta = 2 - 1")
    assert_close(Q[0, 0], 1.5, msg="Q[0, 0] = 1 + 0.5 * 1")


def test_q_learning_update_terminal_ignores_an_all_false_mask(rl):
    Q = np.array([[1.0, 0.0], [5.0, 7.0]])
    delta = call(rl.q_learning_update, Q, 0, 1, 1.0, 1, True, 1.0, 0.9, next_mask=np.zeros(2, dtype=bool),
                 expected="the TD error 1.0 (the mask of a terminal state is ignored)")
    assert_close(delta, 1.0, msg="terminated with an all-False mask (a game over): target = reward")


@pytest.mark.parametrize("dtype", [bool, np.int8], ids=["bool-mask", "int8-mask"])
def test_q_learning_update_max_over_the_legal_actions(rl, dtype):
    Q = np.zeros((2, 3))
    Q[1] = [5.0, 1.0, 3.0]
    mask = np.array([0, 1, 1], dtype=dtype)
    delta = call(rl.q_learning_update, Q, 0, 2, 0.0, 1, False, 1.0, 1.0, next_mask=mask,
                 expected="the TD error 3.0")
    assert_close(delta, 3.0, msg=f"next_mask {mask.tolist()} ({np.dtype(dtype).name}): the max is over Q[1, 1] "
                                 "and Q[1, 2] only (convert a 0/1 mask with .astype(bool))")
    assert_unchanged({"next_mask": mask}, {"next_mask": np.array([0, 1, 1], dtype=dtype)}, "q_learning_update")


def test_q_learning_update_self_transition(rl):
    Q = np.array([[2.0, 1.0]])
    delta = call(rl.q_learning_update, Q, 0, 0, 0.0, 0, False, 0.5, 0.5, expected="the TD error -1.0")
    assert_close(delta, -1.0, msg="s' = s: target = 0 + 0.5 * max(2, 1) = 1 with the values before the update")
    assert_close(Q, [[1.5, 1.0]], msg="Q[0, 0] = 2 + 0.5 * (-1)")


def test_q_learning_update_accepts_numpy_integers(rl):
    Q = np.zeros((3, 2))
    Q[2] = [1.0, 0.0]
    delta = call(rl.q_learning_update, Q, np.int64(1), np.int64(0), np.float64(1.0), np.int64(2), np.False_,
                 0.5, 1.0, expected="the TD error 2.0 (states from gymnasium may be NumPy integers)")
    assert_close(delta, 2.0, msg="NumPy integers as indices")


def _update_args(**changes):
    args = dict(Q=np.zeros((3, 2)), state=0, action=1, reward=1.0, next_state=2, terminated=False, alpha=0.5,
                gamma=0.9)
    args.update(changes)
    return args


INVALID_UPDATES = [
    pytest.param(_update_args(alpha=0.0), "alpha = 0 is outside (0, 1]", id="alpha0"),
    pytest.param(_update_args(alpha=1.5), "alpha = 1.5 is outside (0, 1]", id="alpha-above-1"),
    pytest.param(_update_args(gamma=-0.1), "gamma = -0.1 is outside [0, 1]", id="gamma-negative"),
    pytest.param(_update_args(gamma=1.01), "gamma = 1.01 is outside [0, 1]", id="gamma-above-1"),
    pytest.param(_update_args(state=3), "state 3 is outside 0 .. 2", id="state-too-large"),
    pytest.param(_update_args(state=-1), "state -1: a negative index is refused, not wrapped", id="state-negative"),
    pytest.param(_update_args(action=2), "action 2 is outside 0 .. 1", id="action-too-large"),
    pytest.param(_update_args(next_state=5), "next_state 5 is outside 0 .. 2", id="next_state-too-large"),
    pytest.param(_update_args(state=True), "state=True: a bool is not a state", id="state-bool"),
    pytest.param(_update_args(action=0.5), "action 0.5 is not an integer", id="action-float"),
    pytest.param(_update_args(Q=np.zeros(3)), "Q of shape (3,) is not a table", id="Q-1d"),
    pytest.param(_update_args(Q=np.array([[0.0, np.nan], [0, 0], [0, 0]])), "Q contains NaN", id="Q-nan"),
    pytest.param(_update_args(Q=np.zeros((3, 2), dtype=int)), "Q of integers: the update would be truncated",
                 id="Q-int"),
    pytest.param(_update_args(reward=np.nan), "the reward is NaN", id="reward-nan"),
]


@pytest.mark.parametrize("args, why", INVALID_UPDATES + [
    pytest.param(_update_args(next_mask=[True, False, True]), "next_mask has 3 values for 2 actions",
                 id="mask-length"),
    pytest.param(_update_args(next_mask=[False, False]), "no legal action in a non-terminal next state",
                 id="mask-all-false"),
])
def test_q_learning_update_refuses_invalid_inputs(rl, args, why):
    assert_raises(ValueError, rl.q_learning_update, **copy.deepcopy(args), why=why)


# ================================================================== sarsa_update (26.20)
def test_sarsa_update_docstring_example(rl):
    Q = np.zeros((2, 2))
    Q[1] = [1.0, 3.0]
    delta = call(rl.sarsa_update, Q, 0, 1, reward=1.0, next_state=1, next_action=0, terminated=False, alpha=0.5,
                 gamma=0.9, expected="the TD error 1.9")
    assert_python_float(delta, "the TD error")
    assert_close(delta, 1.9, msg="delta = 1 + 0.9 * Q[1, next_action=0] - Q[0, 1] (the action chosen, not the max)")
    assert_close(Q, [[0.0, 0.95], [1.0, 3.0]], msg="Q after the update, in place (only Q[0, 1] changes)")


@pytest.mark.parametrize("seed", range(5))
def test_sarsa_update_matches_the_formula(rl, seed):
    gen = np.random.default_rng(100 + seed)
    Q = gen.normal(size=(5, 3))
    s, a, s2, a2 = (int(x) for x in gen.integers(0, [5, 3, 5, 3]))
    reward, alpha, gamma = float(gen.normal()), float(gen.uniform(0.05, 1.0)), float(gen.uniform(0, 1))
    want_delta = np_q_target(Q, reward, s2, False, gamma, next_action=a2) - Q[s, a]
    want_Q = Q.copy()
    want_Q[s, a] += alpha * want_delta
    work = Q.copy()
    delta = call(rl.sarsa_update, work, s, a, reward, s2, a2, False, alpha, gamma, expected="the TD error")
    assert_python_float(delta, "the TD error")
    assert_close(delta, want_delta, msg=f"delta = r + gamma * Q[{s2}, {a2}] - Q[{s}, {a}]")
    assert_close(work, want_Q, msg=f"Q after the update (only Q[{s}, {a}] changes, by alpha * delta)")


def test_sarsa_update_equals_q_learning_for_a_greedy_next_action(rl):
    Q = np.array([[0.2, -0.4, 0.1], [1.0, 4.0, 2.0]])
    want = 0.5 + 0.8 * 4.0 - Q[0, 2]
    delta = call(rl.sarsa_update, Q, 0, 2, 0.5, 1, 1, False, 0.25, 0.8, expected=f"the TD error {want:.4g}")
    assert_close(delta, want, msg="next_action = argmax Q[1]: the Q-learning target r + gamma * max Q[1]")


@pytest.mark.parametrize("next_action", [0, None], ids=["next-action-0", "next-action-None"])
def test_sarsa_update_no_bootstrap_on_a_terminal_state(rl, next_action):
    Q = np.array([[1.0, 0.0], [5.0, 7.0]])
    delta = call(rl.sarsa_update, Q, 0, 0, 2.0, 1, next_action, True, 0.5, 0.9, expected="the TD error 1.0")
    assert_close(delta, 1.0, msg="terminated: target = reward = 2, next_action ignored (it may be None)")
    assert_close(Q[0, 0], 1.5, msg="Q[0, 0] = 1 + 0.5 * 1")


@pytest.mark.parametrize("args, why", INVALID_UPDATES + [
    pytest.param(_update_args(next_action=2), "next_action 2 is outside 0 .. 1", id="next_action-too-large"),
    pytest.param(_update_args(next_action=-1), "next_action -1 is refused, not wrapped", id="next_action-negative"),
])
def test_sarsa_update_refuses_invalid_inputs(rl, args, why):
    args = copy.deepcopy(args)
    args.setdefault("next_action", 0)
    assert_raises(ValueError, rl.sarsa_update, **args, why=why)


# ================================================================== train_tabular (26.21)
def test_train_tabular_docstring_example(rl):
    Q, history = train(rl, frozen_lake(), 300, rng=np.random.default_rng(0))
    assert_new_float_array(Q, (16, 4), "Q")
    if not isinstance(history, dict) or not {"returns", "lengths", "epsilons"} <= set(history):
        _fail("", f"expected a dict with the keys 'returns', 'lengths', 'epsilons', got {_short(history)}")
    for key in ("returns", "epsilons"):
        assert_new_float_array(history[key], (300,), f"history[{key!r}]")
    lengths = history["lengths"]
    if not isinstance(lengths, np.ndarray) or not np.issubdtype(lengths.dtype, np.integer) or lengths.shape != (300,):
        _fail("", f"expected history['lengths'] to be an integer array of shape (300,), got "
                  f"{type(lengths).__name__} {getattr(lengths, 'dtype', '')} {np.shape(lengths)}")
    assert_close(history["epsilons"], np.full(300, 0.1), msg="history['epsilons'] with the default epsilon = 0.1")


def test_train_tabular_history_matches_the_environment(rl):
    env = Recorder(gym.make("CliffWalking-v1"))
    _, history = train(rl, env, 40, alpha=0.5, epsilon=0.2, max_steps=50, rng=np.random.default_rng(1))
    if len(env.lengths) != 40:
        _fail("", f"expected 40 calls to env.reset (one per episode), got {len(env.lengths)}")
    assert_close(history["lengths"], env.lengths, msg="history['lengths'] = number of env.step calls of each episode "
                                                       "(at most max_steps = 50)")
    assert_close(history["returns"], env.returns, msg="history['returns'] = sum of the undiscounted rewards")
    if max(env.lengths) != 50:
        _fail("", f"expected some episodes cut at max_steps = 50, got the lengths {env.lengths}")


def test_train_tabular_reset_seeds_come_from_rng(rl):
    env = Recorder(gym.make("CliffWalking-v1"))
    train(rl, env, 20, max_steps=30, rng=np.random.default_rng(7))
    first = int(np.random.default_rng(7).integers(2**31))
    bad = [seed for seed in env.seeds if type(seed) is not int]
    if bad:
        _fail("", f"expected env.reset(seed=int(rng.integers(2**31))) with a Python int (gymnasium refuses NumPy "
                  f"integers and None makes the run impossible to reproduce), got the seed {bad[0]!r}")
    if env.seeds[0] != first:
        _fail("", f"expected the first seed to be int(rng.integers(2**31)) = {first} (the first draw of rng), "
                  f"got {env.seeds[0]}")
    if len(set(env.seeds)) < 19:
        _fail("", f"expected a new seed for every episode, got {env.seeds}")


def test_train_tabular_same_rng_same_history(rl):
    runs = []
    for _ in range(2):
        env = RandomMDPEnv()
        runs.append(train(rl, env, 60, alpha=0.3, gamma=0.9, epsilon=0.3, rng=np.random.default_rng(42)))
    (Q1, h1), (Q2, h2) = runs
    assert_close(Q2, Q1, rtol=0, atol=0, msg="the same rng must give the same Q (no global random state)")
    for key in ("returns", "lengths", "epsilons"):
        assert_close(h2[key], h1[key], rtol=0, atol=0, msg=f"the same rng must give the same history[{key!r}]")
    Q3, _ = train(rl, RandomMDPEnv(), 60, alpha=0.3, gamma=0.9, epsilon=0.3, rng=np.random.default_rng(43))
    if np.array_equal(Q3, Q1):
        _fail("", "expected another rng to give another Q, got the same table")


def test_train_tabular_without_rng(rl):
    Q, history = train(rl, LoopEnv(n_actions=2), 3, max_steps=4)
    assert_close(history["lengths"], [4, 4, 4], msg="rng=None: history['lengths']")


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_train_tabular_replays_the_td_updates(rl, algorithm):
    """Your Q equals the updates of the docstrings replayed on the transitions your agent played."""
    env = RandomMDPEnv()
    Q_init = np.random.default_rng(9).normal(size=(6, 3))
    Q_init[5] = 0.0
    alpha, gamma = 0.3, 0.8
    Q, _ = train(rl, env, 50, algorithm=algorithm, alpha=alpha, gamma=gamma, epsilon=0.3, Q_init=Q_init.copy(),
                 max_steps=1000, rng=np.random.default_rng(4))
    want = Q_init.copy()
    trans = env.transitions
    for k, (s, a, r, s2, term) in enumerate(trans):
        a2 = None if (term or algorithm == "q_learning") else trans[k + 1][1]   # the action played next (SARSA)
        want[s, a] += alpha * (np_q_target(want, r, s2, term, gamma, next_action=a2) - want[s, a])
    rule = ("r + gamma * max Q[s'] (r alone when terminated)" if algorithm == "q_learning"
            else "r + gamma * Q[s', a'] with a' the action played next (r alone when terminated)")
    assert_close(Q, want, rtol=1e-9, atol=1e-10,
                 msg=f"Q after {len(trans)} {algorithm} updates replayed from Q_init with the target {rule}")


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
@pytest.mark.parametrize("q0, alpha", [pytest.param(0.0, 1.0, id="alpha1"), pytest.param(0.0, 0.5, id="alpha0.5"),
                                       pytest.param(1.0, 1.0, id="Q_init1")])
def test_train_tabular_max_steps_is_a_truncation(rl, algorithm, q0, alpha):
    env = LoopEnv(reward=1.0)                        # never terminates
    Q, history = train(rl, env, 3, algorithm=algorithm, alpha=alpha, gamma=0.5, epsilon=0.0, max_steps=5,
                       Q_init=np.array([[q0]]), rng=np.random.default_rng(0))
    assert_close(history["lengths"], [5, 5, 5], msg="history['lengths'] with max_steps = 5")
    assert_close(history["returns"], [5.0, 5.0, 5.0], msg="history['returns']: 5 rewards of 1")
    q = q0
    for _ in range(15):
        q += alpha * (1.0 + 0.5 * q - q)            # every update bootstraps: max_steps is not a termination
    assert_close(Q, [[q]], msg="Q after 15 updates r + gamma * Q[s'] (the last one of each episode bootstraps too)")


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_train_tabular_env_truncation_keeps_the_bootstrap(rl, algorithm):
    env = LoopEnv(reward=1.0, limit=3)               # truncated=True after 3 steps
    Q, history = train(rl, env, 2, algorithm=algorithm, alpha=1.0, gamma=0.5, epsilon=0.0, max_steps=200,
                       rng=np.random.default_rng(0))
    assert_close(history["lengths"], [3, 3], msg="the episodes end when the environment truncates them")
    assert_close(Q, [[2.0 - 2.0 ** -5]], msg="Q after 6 updates Q = 1 + 0.5 * Q (truncated is not terminated)")


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_train_tabular_termination_ends_the_episode(rl, algorithm):
    env = LoopEnv(reward=1.0, terminate_after=4)
    Q, history = train(rl, env, 2, algorithm=algorithm, alpha=1.0, gamma=0.5, epsilon=0.0, max_steps=200,
                       rng=np.random.default_rng(0))
    assert_close(history["lengths"], [4, 4], msg="the episodes end when the environment terminates them")
    assert_close(Q, [[1.0]], msg="the last update of each episode targets r = 1 alone (terminated), so Q = 1")


def test_train_tabular_breaks_ties_at_random(rl):
    env = LoopEnv(n_actions=3, reward=0.0)           # Q stays 0: three tied actions
    train(rl, env, 1, epsilon=0.0, max_steps=300, rng=np.random.default_rng(0))
    counts = np.bincount(env.actions[0], minlength=3)
    if (counts == 0).any():
        _fail("", f"expected the three tied actions to be played (epsilon_greedy_action breaks ties at random), "
                  f"got the counts {counts.tolist()} (np.argmax always picks the first one)")


def test_train_tabular_epsilon_schedule(rl):
    env = LoopEnv(n_actions=2, reward=0.0)
    schedule = [0.0] * 5 + [1.0] * 5
    calls = []

    def epsilon(i):
        calls.append(i)
        return schedule[i]

    Q_init = np.array([[10.0, 0.0]])                  # greedy = action 0
    _, history = train(rl, env, 10, alpha=0.01, gamma=0.9, epsilon=epsilon, max_steps=40, Q_init=Q_init,
                       rng=np.random.default_rng(2))
    assert_close(history["epsilons"], schedule, msg="history['epsilons'] = epsilon(episode_index), from index 0")
    if sorted(set(calls)) != list(range(10)):
        _fail("", f"expected epsilon to be called with the episode indices 0 .. 9, got {sorted(set(calls))}")
    explored = [1 in actions for actions in env.actions]
    if any(explored[:5]) or not all(explored[5:]):
        _fail("", f"expected action 1 only in the episodes where epsilon = 1 (5 .. 9), got it in the episodes "
                  f"{[i for i, e in enumerate(explored) if e]}")


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
@pytest.mark.parametrize("dtype", [bool, np.int8], ids=["bool-mask", "int8-mask"])
def test_train_tabular_action_masks(rl, algorithm, dtype):
    env = MaskEnv(dtype)
    Q_init = np.zeros((3, 3))
    Q_init[0, 2] = Q_init[1, 0] = Q_init[1, 1] = 50.0          # illegal actions look very good
    try:
        Q, _ = rl.train_tabular(env, 300, algorithm=algorithm, alpha=0.5, gamma=0.9, epsilon=0.5, Q_init=Q_init,
                                rng=np.random.default_rng(3))
    except NotImplementedError:
        raise
    except IllegalAction as exc:
        _fail("", f"expected only legal actions (epsilon-greedy on Q[s, legal], then re-indexed), got {exc}")
    except Exception as exc:  # noqa: BLE001
        _fail("", f"expected the tuple (Q, history), but train_tabular raised {type(exc).__name__}: {exc}")
    played = set(env.played)
    if {(0, 0), (0, 1), (1, 2)} - played:
        _fail("", f"expected every legal action to be explored, got the (state, action) pairs {sorted(played)}")
    assert_close(Q[[0, 1, 1], [2, 0, 1]], [50.0, 50.0, 50.0],
                 msg="Q of the illegal actions (never played, so never updated)")
    assert_close(Q[1, 2], 1.0, atol=1e-6, msg="Q[1, 2] (reward 1, then terminal)")
    assert_close(Q[0, 0], 0.9, atol=1e-6, msg="Q[0, 0] = 0.9 * Q[1, 2]: the target uses the legal actions of state "
                                                "1 only (with next_mask)")


def test_train_tabular_frozen_lake_converges_to_q_star(rl):
    env = frozen_lake()
    _, Q_star = np_value_iteration(env.unwrapped.P, 0.9, 200)
    Q, _ = train(rl, env, 2500, alpha=0.5, gamma=0.9, epsilon=1.0, rng=np.random.default_rng(0))
    assert_close(Q, Q_star, rtol=0, atol=1e-3,
                 msg="Q-learning (off-policy, epsilon = 1) on deterministic FrozenLake must reach Q*")
    path = greedy_path(Q, env.unwrapped.P, 0)
    if path[-1] != 15 or len(path) != 7:
        _fail("", f"expected the greedy policy to reach the goal 15 in 6 moves, got the path {path}")


def test_train_tabular_frozen_lake_with_a_decaying_epsilon(rl):
    env = frozen_lake()
    Q, history = train(rl, env, 400, alpha=0.5, gamma=0.95, epsilon=lambda i: max(0.05, 1.0 - i / 200),
                       rng=np.random.default_rng(1))
    path = greedy_path(Q, env.unwrapped.P, 0)
    if path[-1] != 15:
        _fail("", f"expected the greedy policy to reach the goal 15 after 400 episodes, got the path {path}")
    if history["returns"][-50:].mean() < 0.8:
        _fail("", f"expected a success rate of at least 0.8 over the last 50 episodes (epsilon = 0.05), got "
                  f"{history['returns'][-50:].mean():.2f}")


CLIFF_EDGE = set(range(25, 35))                      # row 2, just above the cliff (cells 37 .. 46)


def test_train_tabular_cliff_walking_q_learning_follows_the_cliff(rl):
    env = gym.make("CliffWalking-v1")
    Q, _ = train(rl, env, 1000, algorithm="q_learning", alpha=0.1, gamma=1.0, epsilon=0.1,
                 rng=np.random.default_rng(10))
    path = greedy_path(Q, env.unwrapped.P, 36)
    if path != [36, 24, *range(25, 36), 47]:
        _fail("", f"expected the greedy path of Q-learning to be the shortest one along the cliff "
                  f"(36, 24, 25 .. 35, 47), got {path}")


def test_train_tabular_cliff_walking_sarsa_stays_away_from_the_cliff(rl):
    env = gym.make("CliffWalking-v1")
    Q, _ = train(rl, env, 1000, algorithm="sarsa", alpha=0.1, gamma=1.0, epsilon=0.1,
                 rng=np.random.default_rng(10))
    path = greedy_path(Q, env.unwrapped.P, 36)
    edge = [s for s in path if s in CLIFF_EDGE]
    if path[-1] != 47 or len(path) > 26 or len(edge) > 2:
        _fail("", f"expected the greedy path of SARSA to reach the goal 47 while staying away from the cliff (at "
                  f"most 2 cells of row 2), got {path}")


def test_train_tabular_does_not_modify_Q_init(rl):
    Q_init = np.full((16, 4), 0.5)
    Q, _ = train(rl, frozen_lake(), 20, Q_init=Q_init, rng=np.random.default_rng(0))
    assert_unchanged({"Q_init": Q_init}, {"Q_init": np.full((16, 4), 0.5)}, "train_tabular (copy it)")
    assert_new_float_array(Q, (16, 4), "Q", inputs=(Q_init,))


class _BoxEnv:
    observation_space = object()                      # no n: a continuous space
    action_space = Discrete(2)

    def reset(self, seed=None):
        return np.zeros(4), {}

    def step(self, action):
        return np.zeros(4), 0.0, True, False, {}


@pytest.mark.parametrize("kwargs, why", [
    pytest.param({"algorithm": "td"}, "unknown algorithm 'td'", id="algorithm"),
    pytest.param({"n_episodes": 0}, "n_episodes = 0 (it must be >= 1)", id="n_episodes0"),
    pytest.param({"n_episodes": True}, "n_episodes=True: a bool is not a number of episodes", id="n_episodes-bool"),
    pytest.param({"max_steps": 0}, "max_steps = 0 (it must be >= 1)", id="max_steps0"),
    pytest.param({"alpha": 0.0}, "alpha = 0 is outside (0, 1]", id="alpha0"),
    pytest.param({"alpha": 1.5}, "alpha = 1.5 is outside (0, 1]", id="alpha-above-1"),
    pytest.param({"gamma": 1.5}, "gamma = 1.5 is outside [0, 1]", id="gamma-above-1"),
    pytest.param({"epsilon": 1.5}, "epsilon = 1.5 is outside [0, 1]", id="epsilon-above-1"),
    pytest.param({"epsilon": -0.1}, "epsilon = -0.1 is outside [0, 1]", id="epsilon-negative"),
    pytest.param({"Q_init": np.zeros((2, 2))}, "Q_init of shape (2, 2) for 1 state and 2 actions", id="Q_init-shape"),
    pytest.param({"Q_init": np.array([[np.nan, 0.0]])}, "Q_init contains NaN", id="Q_init-nan"),
    pytest.param({"env": _BoxEnv()}, "the observation space has no n: not a discrete environment", id="not-discrete"),
])
def test_train_tabular_refuses_invalid_inputs(rl, kwargs, why):
    kwargs = dict(kwargs)
    env = kwargs.pop("env", None) or LoopEnv(n_actions=2, limit=3)
    n = kwargs.pop("n_episodes", 2)
    assert_raises(ValueError, rl.train_tabular, env, n, **kwargs, rng=np.random.default_rng(0), why=why)


# ================================================================== evaluate_policy (26.21)
def _docstring_policy():
    policy = np.zeros(16, dtype=int)
    policy[[0, 1, 14]] = 2
    policy[[2, 6, 10]] = 1
    return policy


def test_evaluate_policy_docstring_example(rl):
    got = call(rl.evaluate_policy, frozen_lake(), _docstring_policy(), n_episodes=5, expected="a dict")
    want = {"mean_return": 1.0, "std_return": 0.0, "mean_length": 6.0}
    if not isinstance(got, dict) or set(got) != set(want):
        _fail("", f"expected a dict with the keys {sorted(want)}, got {got!r}")
    for key, value in want.items():
        assert_python_float(got[key], f"result[{key!r}]")
        assert_close(got[key], value, msg=f"result[{key!r}] (the path 2, 2, 1, 1, 1, 2 reaches the goal in 6 moves)")


def _monte_carlo(env, policy, n_episodes, max_steps, seed):
    """Direct Monte Carlo loop with gymnasium, seeds int(rng.integers(2**31)) drawn from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    returns, lengths = [], []
    for _ in range(n_episodes):
        state, info = env.reset(seed=int(rng.integers(2**31)))
        total, steps = 0.0, 0
        while steps < max_steps:
            state, reward, terminated, truncated, info = env.step(int(policy[state]))
            total, steps = total + reward, steps + 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
    return {"mean_return": np.mean(returns), "std_return": np.std(returns), "mean_length": np.mean(lengths)}


@pytest.mark.parametrize("max_steps", [200, 15])
def test_evaluate_policy_matches_a_monte_carlo_loop(rl, max_steps):
    env = frozen_lake(slippery=True)
    _, Q = np_value_iteration(env.unwrapped.P, 0.99, 300)
    policy = np.argmax(Q, axis=1)
    want = _monte_carlo(env, policy, 150, max_steps, seed=11)
    got = call(rl.evaluate_policy, env, policy.copy(), 150, max_steps=max_steps, rng=np.random.default_rng(11),
               expected="a dict")
    for key in want:
        assert_close(got.get(key) if isinstance(got, dict) else None, want[key], rtol=1e-12,
                     msg=f"result[{key!r}] on slippery FrozenLake (seeds int(rng.integers(2**31)), std with ddof=0, "
                         f"max_steps = {max_steps})")


def test_evaluate_policy_callable_policy(rl):
    env = frozen_lake(slippery=True)
    table = np.random.default_rng(2).integers(0, 4, size=16)
    seen = []

    def policy(state, info):
        seen.append(isinstance(info, dict))
        return int(table[state])

    want = _monte_carlo(env, table, 60, 200, seed=3)
    got = call(rl.evaluate_policy, env, policy, 60, rng=np.random.default_rng(3), expected="a dict")
    if not seen or not all(seen):
        _fail("", "expected the policy to be called as policy(state, info) with the info dict of the environment")
    for key in want:
        assert_close(got[key], want[key], rtol=1e-12, msg=f"result[{key!r}] with a callable policy")


def test_evaluate_policy_list_policy(rl):
    got = call(rl.evaluate_policy, frozen_lake(), _docstring_policy().tolist(), 2, rng=np.random.default_rng(0),
               expected="a dict")
    assert_close(got["mean_length"], 6.0, msg="a list of actions as the policy")


@pytest.mark.parametrize("max_steps, limit, length", [pytest.param(7, None, 7.0, id="max_steps"),
                                                      pytest.param(200, 4, 4.0, id="env-truncation")])
def test_evaluate_policy_stops_at_max_steps_or_truncation(rl, max_steps, limit, length):
    env = LoopEnv(reward=2.0, limit=limit)
    got = call(rl.evaluate_policy, env, np.zeros(1, dtype=int), 3, max_steps=max_steps,
               rng=np.random.default_rng(0), expected="a dict")
    assert_close([got["mean_length"], got["mean_return"], got["std_return"]], [length, 2 * length, 0.0],
                 msg=f"[mean_length, mean_return, std_return] (max_steps = {max_steps}, environment limit = {limit})")


def test_evaluate_policy_stops_at_termination(rl):
    policy = np.ones(16, dtype=int)                   # always down: 0 -> 4 -> 8 -> 12, a hole
    got = call(rl.evaluate_policy, Recorder(frozen_lake()), policy, 4, rng=np.random.default_rng(0),
               expected="a dict (and no step after the end of an episode)")
    assert_close([got["mean_length"], got["mean_return"]], [3.0, 0.0],
                 msg="[mean_length, mean_return] when the agent falls into the hole 12 after 3 moves")


def test_evaluate_policy_does_not_modify_the_policy(rl):
    policy = _docstring_policy()
    call(rl.evaluate_policy, frozen_lake(), policy, 2, rng=np.random.default_rng(0), expected="a dict")
    assert_unchanged({"policy": policy}, {"policy": _docstring_policy()}, "evaluate_policy")


@pytest.mark.parametrize("args, kwargs, why", [
    pytest.param((0,), {}, "n_episodes = 0 (it must be >= 1)", id="n_episodes0"),
    pytest.param((True,), {}, "n_episodes=True: a bool is not a number of episodes", id="n_episodes-bool"),
    pytest.param((3,), {"max_steps": 0}, "max_steps = 0 (it must be >= 1)", id="max_steps0"),
    pytest.param((3,), {"policy": np.zeros((16, 4), dtype=int)}, "a policy table of shape (16, 4) is not 1-D",
                 id="policy-2d"),
    pytest.param((3,), {"policy": np.full(16, 2.0)}, "a policy table of floats: actions are integers",
                 id="policy-float"),
])
def test_evaluate_policy_refuses_invalid_inputs(rl, args, kwargs, why):
    kwargs = dict(kwargs)
    policy = kwargs.pop("policy", _docstring_policy())
    assert_raises(ValueError, rl.evaluate_policy, frozen_lake(), policy, *args, **kwargs,
                  rng=np.random.default_rng(0), why=why)


# ================================================================== value_iteration (26.26)
def _vi(rl, P, gamma, **kwargs):
    result = call(rl.value_iteration, P, gamma, expected="the tuple (V, Q)", **kwargs)
    return pair(result, "value_iteration", ("V", "Q"))


def test_value_iteration_docstring_example(rl):
    P = {0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, True)]},
         1: {0: [(1.0, 1, 0.0, True)], 1: [(1.0, 1, 0.0, True)]}}
    V, Q = _vi(rl, P, 0.9)
    assert_new_float_array(V, (2,), "V")
    assert_new_float_array(Q, (2, 2), "Q")
    assert_close(V, [1.0, 0.0], msg="V: leaving with reward 1 is worth 1, the terminal state 0")
    assert_close(Q, [[0.9, 1.0], [0.0, 0.0]], msg="Q: staying is worth 0.9 * V[0]")


@pytest.mark.parametrize("n, p, reward, gamma, towards", [
    pytest.param(5, 1.0, 1.0, 0.9, +1, id="deterministic"), pytest.param(6, 0.7, 2.0, 0.95, +1, id="p0.7"),
    pytest.param(4, 0.3, 1.0, 0.5, -1, id="p0.3-leftwards"), pytest.param(8, 1.0, 5.0, 0.0, -1, id="gamma0"),
])
def test_value_iteration_chain_closed_form(rl, n, p, reward, gamma, towards):
    P, _ = chain_mdp(n, p, reward, towards)
    V, Q = _vi(rl, P, gamma)
    want_V, want_Q = chain_closed_form(n, p, reward, gamma, towards)
    assert_close(V, want_V, rtol=1e-8, atol=1e-9, msg=f"V of the chain (move succeeds with probability {p})")
    assert_close(Q, want_Q, rtol=1e-8, atol=1e-9, msg="Q of the chain (stay: gamma V(s), move: V(s))")


@pytest.mark.parametrize("name, gamma", [
    pytest.param("4x4", 0.99, id="frozen-lake-4x4"), pytest.param("8x8", 0.95, id="frozen-lake-8x8"),
    pytest.param("cliff", 0.9, id="cliff-walking"),
])
def test_value_iteration_matches_an_exact_policy_evaluation(rl, name, gamma):
    env = gym.make("CliffWalking-v1") if name == "cliff" else frozen_lake(slippery=True, map_name=name)
    P = env.unwrapped.P
    V, Q = _vi(rl, P, gamma)
    S, A = len(P), len(P[0])
    assert_new_float_array(V, (S,), "V")
    assert_new_float_array(Q, (S, A), "Q")
    policy = np.argmax(Q, axis=1)
    V_pi, Q_pi = np_policy_value(P, gamma, policy)
    assert_close(V, V_pi, rtol=0, atol=1e-6,
                 msg=f"V = value of greedy(Q) solved exactly with np.linalg.solve ({name}, gamma = {gamma})")
    assert_close(Q, Q_pi, rtol=0, atol=1e-6, msg="Q = one backup of V: sum of p * (r + gamma * V[s'] * (1 - done))")
    gain = Q_pi.max(axis=1) - V_pi
    if gain.max() > 1e-8:
        s = int(np.argmax(gain))
        _fail("", f"expected an optimal greedy policy (no action better than it), got state {s} improvable by "
                  f"{gain[s]:.3g}")


@pytest.mark.parametrize("towards", [+1, -1], ids=["rightwards", "leftwards"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_value_iteration_max_iter_gives_synchronous_sweeps(rl, towards, k):
    P, _ = chain_mdp(7, 0.8, 1.0, towards)
    V, Q = _vi(rl, P, 0.9, max_iter=k)
    want_V, want_Q = np_value_iteration(P, 0.9, k)
    assert_close(V, want_V, rtol=1e-12, atol=1e-14,
                 msg=f"V after max_iter = {k} synchronous sweeps from V = 0 (each new V from the previous one only)")
    assert_close(Q, want_Q, rtol=1e-12, atol=1e-14, msg="Q computed from the returned V (one more backup)")


def test_value_iteration_stops_strictly_below_theta(rl):
    P = {0: {0: [(1.0, 0, 1.0, False)]}}             # V_k = 2 - 2**(1-k): changes 1, 0.5, 0.25, 0.125
    V, Q = _vi(rl, P, 0.5, theta=0.25)
    assert_close(V, [1.875], rtol=0, atol=0, msg="theta = 0.25: a change of 0.25 is not < theta, the change 0.125 "
                                                   "stops after the 4th sweep")
    assert_close(Q, [[1.9375]], rtol=0, atol=0, msg="Q = 1 + 0.5 * V")


def test_value_iteration_does_not_modify_P(rl):
    P = frozen_lake(slippery=True).unwrapped.P
    saved = copy.deepcopy(P)
    _vi(rl, P, 0.9)
    if P != saved:
        _fail("", "expected P unchanged by value_iteration (build arrays from it instead)")


def _bad_P(kind):
    good = {0: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 0, 0.0, False)]},
            1: {0: [(1.0, 1, 0.0, True)], 1: [(1.0, 1, 0.0, True)]}}
    if kind == "empty":
        return {}
    if kind == "actions-differ":
        good[1] = {0: [(1.0, 1, 0.0, True)]}
    elif kind == "no-actions":
        good[1] = {}
    elif kind == "probs-sum":
        good[0][0] = [(0.5, 1, 0.0, False)]
    elif kind == "next-state":
        good[0][1] = [(1.0, 2, 0.0, False)]
    elif kind == "state-keys":
        good = {0: good[0], 2: good[1]}
    return good


@pytest.mark.parametrize("P, gamma, kwargs, why", [
    pytest.param(_bad_P(""), 1.0, {}, "gamma = 1: value iteration needs gamma < 1", id="gamma1"),
    pytest.param(_bad_P(""), -0.1, {}, "gamma = -0.1 is outside [0, 1)", id="gamma-negative"),
    pytest.param(_bad_P(""), 1.5, {}, "gamma = 1.5 is outside [0, 1)", id="gamma-above-1"),
    pytest.param(_bad_P(""), 0.9, {"max_iter": 0}, "max_iter = 0 (it must be >= 1)", id="max_iter0"),
    pytest.param(_bad_P(""), 0.9, {"theta": 0.0}, "theta = 0 (it must be > 0)", id="theta0"),
    pytest.param(_bad_P("empty"), 0.9, {}, "P is empty", id="P-empty"),
    pytest.param(_bad_P("actions-differ"), 0.9, {}, "state 0 has 2 actions and state 1 has 1", id="actions-differ"),
    pytest.param(_bad_P("no-actions"), 0.9, {}, "state 1 has no action", id="no-actions"),
    pytest.param(_bad_P("probs-sum"), 0.9, {}, "the probabilities of P[0][0] sum to 0.5", id="probs-sum"),
    pytest.param(_bad_P("next-state"), 0.9, {}, "P[0][1] leads to state 2, outside 0 .. 1", id="next-state"),
    pytest.param(_bad_P("state-keys"), 0.9, {}, "the states are 0 and 2, not 0 .. 1", id="state-keys"),
])
def test_value_iteration_refuses_invalid_inputs(rl, P, gamma, kwargs, why):
    assert_raises(ValueError, rl.value_iteration, P, gamma, **kwargs, why=why)


# ================================================================== ReplayBuffer (26.31)
def _push_range(buffer, values, obs=None):
    """Push the transitions (s, s % 2, s / 10, s + 1, s % 3 == 0) for s in values."""
    for s in values:
        state = s if obs is None else np.full(obs, s, dtype=float)
        nxt = s + 1 if obs is None else np.full(obs, s + 1, dtype=float)
        call(buffer.push, state, s % 2, s / 10, nxt, s % 3 == 0, expected="no error from push")


def _check_rows(batch, obs=None):
    """Row i of every array belongs to the same pushed transition."""
    s = batch["states"] if obs is None else batch["states"][:, 0]
    want = {"actions": s % 2, "rewards": s / 10, "terminated": s % 3 == 0,
            "next_states": (s + 1) if obs is None else (s + 1)[:, None] * np.ones(obs)}
    for key, value in want.items():
        if not np.allclose(batch[key], value):
            _fail("", f"expected batch[{key!r}] to match batch['states'] row by row (the same transition), got "
                      f"states {_short(s)} and {key} {_short(batch[key])}")


def _sample(buffer, batch_size):
    batch = call(buffer.sample, batch_size, expected="a dict of arrays")
    keys = {"states", "actions", "rewards", "next_states", "terminated"}
    if not isinstance(batch, dict) or set(batch) != keys:
        _fail("", f"expected a dict with the keys {sorted(keys)}, got {batch!r}")
    return batch


def test_replay_buffer_docstring_example(rl):
    buffer = rl.ReplayBuffer(capacity=2, random_state=0)
    call(buffer.push, 0, 1, 0.5, 1, False, expected="no error from push")
    call(buffer.push, 1, 0, 1.0, 2, True, expected="no error from push")
    call(buffer.push, 2, 1, 0.0, 3, False, expected="no error from push")
    if call(len, buffer, expected="len(buffer)") != 2:
        _fail("", f"expected len(buffer) == 2 (the capacity), got {len(buffer)}")
    batch = _sample(buffer, 2)
    if sorted(batch["states"].tolist()) != [1, 2]:
        _fail("", f"expected the states [1, 2] (the oldest transition 0 was overwritten), got "
                  f"{sorted(batch['states'].tolist())}")
    if batch["terminated"].dtype != np.bool_:
        _fail("", f"expected batch['terminated'] of dtype bool, got {batch['terminated'].dtype}")


def test_replay_buffer_len_grows_then_stays_at_capacity(rl):
    buffer = rl.ReplayBuffer(capacity=4, random_state=0)
    sizes = []
    for s in range(7):
        sizes.append(call(len, buffer, expected="len(buffer)"))
        _push_range(buffer, [s])
    sizes.append(len(buffer))
    if sizes != [0, 1, 2, 3, 4, 4, 4, 4]:
        _fail("", f"expected len = 0, 1, 2, 3, 4, 4, 4, 4 while pushing 7 transitions into 4 places, got {sizes}")


@pytest.mark.parametrize("capacity, n_pushed", [(3, 7), (5, 5), (5, 3), (4, 12), (1, 3)])
def test_replay_buffer_fifo_eviction(rl, capacity, n_pushed):
    buffer = rl.ReplayBuffer(capacity=capacity, random_state=1)
    _push_range(buffer, range(n_pushed))
    size = min(capacity, n_pushed)
    batch = _sample(buffer, size)
    want = list(range(n_pushed - size, n_pushed))
    if sorted(batch["states"].tolist()) != want:
        _fail("", f"expected the {size} most recent transitions {want} (FIFO: the oldest is overwritten), got "
                  f"{sorted(batch['states'].tolist())}")
    _check_rows(batch)


def test_replay_buffer_shapes_and_dtypes_for_cartpole(rl):
    buffer = rl.ReplayBuffer(capacity=10, obs_shape=(4,), obs_dtype="float32", random_state=0)
    _push_range(buffer, range(13), obs=(4,))
    batch = _sample(buffer, 6)
    want = {"states": ((6, 4), np.float32), "next_states": ((6, 4), np.float32), "actions": ((6,), np.int64),
            "rewards": ((6,), np.float64), "terminated": ((6,), np.bool_)}
    for key, (shape, dtype) in want.items():
        arr = batch[key]
        if not isinstance(arr, np.ndarray) or arr.shape != shape or arr.dtype != dtype:
            _fail("", f"expected batch[{key!r}] of shape {shape} and dtype {np.dtype(dtype).name}, got "
                      f"{type(arr).__name__} {np.shape(arr)} {getattr(arr, 'dtype', '')}")
    if not set(batch["states"][:, 0].tolist()) <= set(range(3, 13)):
        _fail("", f"expected states among the 10 most recent ones (3 .. 12), got {batch['states'][:, 0].tolist()}")
    _check_rows(batch, obs=(4,))


def test_replay_buffer_samples_without_replacement(rl):
    buffer = rl.ReplayBuffer(capacity=10, random_state=3)
    _push_range(buffer, range(10))
    for _ in range(200):
        states = _sample(buffer, 6)["states"].tolist()
        if len(set(states)) != 6:
            _fail("", f"expected 6 distinct transitions (sampling without replacement), got the states {states}")


def test_replay_buffer_samples_uniformly(rl):
    buffer = rl.ReplayBuffer(capacity=8, random_state=4)
    _push_range(buffer, range(13))                  # stored: 5 .. 12, after wrapping around
    singles = np.array([_sample(buffer, 1)["states"][0] for _ in range(16_000)])
    counts = np.bincount(singles - 5, minlength=8)
    if counts.size != 8 or (singles < 5).any():
        _fail("", f"expected only the stored states 5 .. 12, got {sorted(set(singles.tolist()))}")
    p_value = chisquare_p(counts, np.ones(8))
    if p_value < 1e-6:
        _fail("", f"expected each of the 8 stored transitions with frequency 1/8, got the counts {counts.tolist()} "
                  f"(chi-square p-value {p_value:.1e})")
    triples = np.concatenate([_sample(buffer, 3)["states"] for _ in range(6000)])
    counts = np.bincount(triples - 5, minlength=8)
    p_value = chisquare_p(counts, np.ones(8))
    if p_value < 1e-6:
        _fail("", f"expected each stored transition in a batch of 3 with probability 3/8, got the counts "
                  f"{counts.tolist()} (chi-square p-value {p_value:.1e})")


def test_replay_buffer_reproducible_with_random_state(rl):
    draws = []
    for seed in (5, 5, 6):
        buffer = rl.ReplayBuffer(capacity=20, random_state=seed)
        _push_range(buffer, range(20))
        draws.append([_sample(buffer, 5)["states"].tolist() for _ in range(5)])
    if draws[0] != draws[1]:
        _fail("", "expected the same random_state to give the same batches (use self._rng), got different ones")
    if draws[0] == draws[2]:
        _fail("", "expected another random_state to give other batches, got the same ones")


def test_replay_buffer_returns_copies(rl):
    buffer = rl.ReplayBuffer(capacity=5, obs_shape=(2,), obs_dtype="float32", random_state=0)
    _push_range(buffer, range(5), obs=(2,))
    batch = _sample(buffer, 5)
    stored = {"states": buffer._states, "actions": buffer._actions, "rewards": buffer._rewards,
              "next_states": buffer._next_states, "terminated": buffer._terminated}
    for key, arr in stored.items():
        if np.shares_memory(batch[key], arr):
            _fail("", f"expected batch[{key!r}] to be a copy, got a view of the storage of the buffer")


def test_replay_buffer_push_refuses_a_wrong_shape(rl):
    buffer = rl.ReplayBuffer(capacity=3, obs_shape=(4,), obs_dtype="float32")
    assert_raises(ValueError, buffer.push, 0.0, 1, 0.0, np.zeros(4), False,
                  why="a number as the state of a buffer with obs_shape=(4,) (no broadcasting)")
    assert_raises(ValueError, buffer.push, np.zeros(4), 1, 0.0, np.zeros(3), False,
                  why="next_state of shape (3,) for obs_shape=(4,)")


@pytest.mark.parametrize("n_pushed, batch_size, why", [
    pytest.param(3, 0, "batch_size = 0 (it must be >= 1)", id="zero"),
    pytest.param(3, 4, "batch_size = 4 > len(buffer) = 3", id="too-large"),
    pytest.param(0, 1, "the buffer is empty", id="empty"),
    pytest.param(3, True, "batch_size=True: a bool is not a size", id="bool"),
])
def test_replay_buffer_sample_refuses_invalid_sizes(rl, n_pushed, batch_size, why):
    buffer = rl.ReplayBuffer(capacity=5, random_state=0)
    _push_range(buffer, range(n_pushed))
    assert_raises(ValueError, buffer.sample, batch_size, why=why)

