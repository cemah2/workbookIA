"""Tabular reinforcement learning — mylearn, chapter 26 (Reinforcement Learning).

Returns, policies, Q-learning and SARSA updates, a generic training loop for
discrete environments with the gymnasium API (action masks included), policy
evaluation, value iteration (the exact reference when the model is known) and
a replay buffer (reused by the DQN of bonus chapter B8). Exploration reuses
chapter 11: ``epsilon_greedy_action`` and ``argmax_random_tie`` come from
``bandit.py``, and a Q update is an ``incremental_update`` towards the TD
target.

Reference implementation.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .bandit import (  # chapter 11, to reuse below
    argmax_random_tie,  # noqa: F401 - used through epsilon_greedy_action
    epsilon_greedy_action,
    incremental_update,
)


def _real(value: Any, name: str) -> float:
    """A real number (not a bool) as a Python float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    return float(value)


def _check_gamma(gamma: Any, *, below_one: bool = False) -> float:
    """gamma in [0, 1] (or in [0, 1) when ``below_one``)."""
    g = _real(gamma, "gamma")
    ok = 0.0 <= g < 1.0 if below_one else 0.0 <= g <= 1.0
    if not ok:
        interval = "[0, 1)" if below_one else "[0, 1]"
        raise ValueError(f"gamma must be in {interval}, got {gamma}")
    return g


def _check_alpha(alpha: Any) -> float:
    a = _real(alpha, "alpha")
    if not 0.0 < a <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return a


def _check_epsilon(epsilon: Any) -> float:
    e = _real(epsilon, "epsilon")
    if not 0.0 <= e <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    return e


def _count(value: Any, name: str, minimum: int = 1) -> int:
    """An integer (not a bool) >= minimum."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _index(value: Any, n: int, name: str) -> int:
    """An integer index in 0 .. n - 1 (a negative index is refused, not wrapped)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer index, got {value!r}")
    if not 0 <= value < n:
        raise ValueError(f"{name} must be in 0 .. {n - 1}, got {value}")
    return int(value)


def _rewards(rewards: ArrayLike) -> np.ndarray:
    arr = np.array(rewards, dtype=np.float64, copy=True)
    if arr.ndim != 1:
        raise ValueError(f"rewards must be 1-D (one episode), got shape {arr.shape}")
    if np.isnan(arr).any():
        raise ValueError("rewards contain NaN values")
    return arr


def _q_table(Q: Any, check_nan: bool = True) -> np.ndarray:
    """The Q-table updated in place: a 2-D float ndarray, without NaN (checked by the caller if not check_nan)."""
    if not isinstance(Q, np.ndarray) or not np.issubdtype(Q.dtype, np.floating):
        raise ValueError("Q must be a NumPy array of floats (it is modified in place), got "
                         + (f"dtype {Q.dtype}" if isinstance(Q, np.ndarray) else type(Q).__name__))
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2-D (n_states, n_actions), got shape {Q.shape}")
    if check_nan and np.isnan(Q).any():
        raise ValueError("Q contains NaN values")
    return Q


def _mask_of(info: Any) -> np.ndarray | None:
    """The legal actions of ``info["action_mask"]``, or None."""
    if isinstance(info, Mapping) and info.get("action_mask") is not None:
        return np.asarray(info["action_mask"]).astype(bool)
    return None


def _explore(Q: np.ndarray, state: int, info: Any, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy on Q[state], restricted to the legal actions when info has an action mask."""
    mask = _mask_of(info)
    if mask is None:
        return epsilon_greedy_action(Q[state], epsilon, rng)
    if mask.shape != (Q.shape[1],):
        raise ValueError(f"info['action_mask'] must have shape ({Q.shape[1]},), got {mask.shape}")
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise ValueError(f"no legal action in state {state} (info['action_mask'] is all False)")
    return int(legal[epsilon_greedy_action(Q[state, legal], epsilon, rng)])


def _discrete_sizes(env: Any) -> tuple[int, int]:
    try:
        return int(env.observation_space.n), int(env.action_space.n)
    except AttributeError:
        raise ValueError("the observation and action spaces must be discrete (with an attribute n)") from None


def _parse_model(P: Any):
    """Flat arrays of the outcomes of P: (state-action index, prob, next state, reward, done, n_states, n_actions)."""
    if not isinstance(P, Mapping) or len(P) == 0:
        raise ValueError("P must be a non-empty mapping P[s][a] -> list of (prob, next_state, reward, terminated)")
    n_states = len(P)
    if set(P.keys()) != set(range(n_states)):
        raise ValueError(f"the states of P must be 0 .. {n_states - 1}")
    n_actions = None
    sa, prob, nxt, rew, done = [], [], [], [], []
    for s in range(n_states):
        actions = P[s]
        if not isinstance(actions, Mapping) or len(actions) == 0:
            raise ValueError(f"P[{s}] must be a non-empty mapping action -> outcomes")
        if n_actions is None:
            n_actions = len(actions)
        elif len(actions) != n_actions:
            raise ValueError(f"every state must have the same actions: P[0] has {n_actions}, "
                             f"P[{s}] has {len(actions)}")
        if set(actions.keys()) != set(range(n_actions)):
            raise ValueError(f"the actions of P[{s}] must be 0 .. {n_actions - 1}")
        for a in range(n_actions):
            outcomes = actions[a]
            if len(outcomes) == 0:
                raise ValueError(f"P[{s}][{a}] has no outcome")
            total = 0.0
            for outcome in outcomes:
                if len(outcome) != 4:
                    raise ValueError(f"an outcome of P[{s}][{a}] must be (prob, next_state, reward, terminated), "
                                     f"got {outcome!r}")
                p, s2, r, d = outcome
                p, r = _real(p, "a probability"), _real(r, "a reward")
                if not 0.0 <= p <= 1.0 or np.isnan(r):
                    raise ValueError(f"invalid outcome {outcome!r} in P[{s}][{a}]")
                sa.append(s * n_actions + a)
                prob.append(p)
                nxt.append(_index(s2, n_states, f"a next state of P[{s}][{a}]"))
                rew.append(r)
                done.append(float(bool(d)))
                total += p
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"the probabilities of P[{s}][{a}] sum to {total}, not 1")
    return (np.array(sa), np.array(prob), np.array(nxt), np.array(rew), np.array(done), n_states, n_actions)


def discounted_return(rewards: ArrayLike, gamma: float) -> float:
    """Discounted return of one episode: G_0 = sum over k of gamma**k * r_k.

    This is the book's discounted future reward (DFR) of the first move.

    Parameters
    ----------
    rewards : array-like of shape (T,)
        Rewards r_0, ..., r_(T-1) of one episode, in order.
    gamma : float
        Discount factor, in [0, 1].

    Returns
    -------
    float
        G_0 (0.0 for an empty episode).

    Raises
    ------
    ValueError
        If ``gamma`` is outside [0, 1] or ``rewards`` is not 1-D.

    Notes
    -----
    Tested against ``numpy.polynomial.polynomial.polyval(gamma, rewards)``.

    Examples
    --------
    >>> discounted_return([1.0, 1.0, 1.0], gamma=0.5)
    1.75
    >>> discounted_return([], gamma=0.9)
    0.0
    """
    rewards = _rewards(rewards)
    gamma = _check_gamma(gamma)
    return float(returns_to_go(rewards, gamma)[0]) if rewards.size else 0.0


def returns_to_go(rewards: ArrayLike, gamma: float = 1.0) -> np.ndarray:
    """Return G_t = r_t + gamma * G_(t+1) of every step of an episode (G_T = 0).

    With gamma = 1 it is the book's total future reward (TFR) of each move.

    Parameters
    ----------
    rewards : array-like of shape (T,)
        Rewards of one episode, in order.
    gamma : float, default=1.0
        Discount factor, in [0, 1].

    Returns
    -------
    ndarray of shape (T,)
        The returns, as float64; the first one equals
        ``discounted_return(rewards, gamma)``.

    Raises
    ------
    ValueError
        If ``gamma`` is outside [0, 1] or ``rewards`` is not 1-D.

    Notes
    -----
    Tested against ``scipy.signal.lfilter([1], [1, -gamma], rewards[::-1])[::-1]``
    (for gamma = 1: the reversed cumulative sum).

    Examples
    --------
    >>> returns_to_go([1.0, 0.0, 2.0], gamma=0.5)
    array([1.5, 1. , 2. ])
    >>> returns_to_go([1.0, 1.0, 1.0])
    array([3., 2., 1.])
    """
    rewards = _rewards(rewards)
    gamma = _check_gamma(gamma)
    G = np.zeros_like(rewards)
    future = 0.0
    for t in range(rewards.size - 1, -1, -1):         # backwards: G_t = r_t + gamma * G_(t+1)
        future = rewards[t] + gamma * future
        G[t] = future
    return G


def greedy_policy(Q: ArrayLike) -> np.ndarray:
    """Deterministic greedy policy: the best action of every state.

    Ties go to the lowest action index (like ``np.argmax``), so the policy is
    reproducible.

    Parameters
    ----------
    Q : array-like of shape (n_states, n_actions)
        Action values.

    Returns
    -------
    ndarray of shape (n_states,)
        Integer actions.

    Raises
    ------
    ValueError
        If ``Q`` is not 2-D.

    Notes
    -----
    Tested against ``np.argmax(Q, axis=1)``.

    Examples
    --------
    >>> greedy_policy([[0.0, 1.0], [2.0, 2.0]])
    array([1, 0])
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2-D (n_states, n_actions), got shape {Q.shape}")
    if Q.shape[1] == 0:
        raise ValueError("Q has no action")
    if np.isnan(Q).any():
        raise ValueError("Q contains NaN values")
    return np.argmax(Q, axis=1)                      # first maximum on ties


def softmax_action(
    q_values: ArrayLike, temperature: float = 1.0, rng: np.random.Generator | None = None
) -> int:
    """Boltzmann exploration: draw an action with probability proportional to exp(q / T).

    Computed in a numerically stable way (subtract the maximum before ``exp``).

    Parameters
    ----------
    q_values : array-like of shape (n_actions,)
        Values of the actions of one state.
    temperature : float, default=1.0
        T > 0: a large T gives almost uniform choices, a small T almost
        greedy ones.
    rng : numpy.random.Generator or None, default=None
        Generator used for the draw. None means ``np.random.default_rng()``.

    Returns
    -------
    int
        Index of the chosen action.

    Raises
    ------
    ValueError
        If ``temperature <= 0`` or ``q_values`` is empty.

    Notes
    -----
    Tested on the empirical frequencies of 10**5 draws against
    ``torch.softmax(q / T)`` (``scipy.stats.chisquare``), and with large q
    values (no overflow).

    Examples
    --------
    >>> softmax_action([0.0, 10.0, 0.0], temperature=0.01, rng=np.random.default_rng(0))
    1
    """
    q = np.asarray(q_values, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError(f"q_values must be a non-empty 1-D array, got shape {q.shape}")
    if np.isnan(q).any():
        raise ValueError("q_values contain NaN values")
    T = _real(temperature, "temperature")
    if not T > 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    rng = np.random.default_rng() if rng is None else rng
    probs = np.exp((q - q.max()) / T)                 # max subtracted: exp never overflows
    probs /= probs.sum()
    return int(rng.choice(q.size, p=probs))           # one draw: inverse CDF of one rng.random()


def q_learning_update(
    Q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    terminated: bool,
    alpha: float,
    gamma: float,
    next_mask: ArrayLike | None = None,
) -> float:
    """Q-learning update of Q[state, action], in place; returns the TD error.

    target = reward if ``terminated``, else
    reward + gamma * max over the (legal) actions a' of Q[next_state, a'];
    delta = target - Q[state, action]; Q[state, action] += alpha * delta.
    No bootstrap from a terminal state: its future is worth 0.

    Parameters
    ----------
    Q : ndarray of shape (n_states, n_actions)
        Action-value table, modified in place.
    state : int
        State s.
    action : int
        Action a taken in s.
    reward : float
        Reward r received.
    next_state : int
        State s' reached.
    terminated : bool
        True if s' is terminal (the target is then r). A truncation (time
        limit) is NOT a termination: pass False and keep the bootstrap.
    alpha : float
        Learning rate, in (0, 1].
    gamma : float
        Discount factor, in [0, 1].
    next_mask : array-like of bool, shape (n_actions,), or None, default=None
        Legal actions in s' (for example ``info["action_mask"]`` in a game);
        the max is taken over them only. None means all actions.

    Returns
    -------
    float
        The TD error delta, computed before the update.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1] or ``gamma`` is not in [0, 1].

    Notes
    -----
    Tested on hand-computed values (the example of book figure 26.30), and by
    convergence to the Q* of ``value_iteration`` on deterministic FrozenLake.

    Examples
    --------
    >>> Q = np.zeros((2, 2))
    >>> Q[1] = [1.0, 3.0]
    >>> q_learning_update(Q, 0, 1, reward=1.0, next_state=1, terminated=False,
    ...                   alpha=0.5, gamma=0.9)     # target = 1 + 0.9 * 3
    3.7
    >>> Q[0]
    array([0.  , 1.85])
    """
    Q = _q_table(Q, check_nan=False)
    n_states, n_actions = Q.shape
    state = _index(state, n_states, "state")
    action = _index(action, n_actions, "action")
    next_state = _index(next_state, n_states, "next_state")
    if np.isnan(Q[state]).any() or np.isnan(Q[next_state]).any():   # the rows the update reads (checking the whole
        raise ValueError("Q contains NaN values")                     # table at each step would slow a long loop)
    reward = _real(reward, "reward")
    if np.isnan(reward):
        raise ValueError("reward is NaN")
    alpha, gamma = _check_alpha(alpha), _check_gamma(gamma)
    legal = np.ones(n_actions, dtype=bool)
    if next_mask is not None:
        legal = np.asarray(next_mask).astype(bool)
        if legal.shape != (n_actions,):
            raise ValueError(f"next_mask must have shape ({n_actions},), got {legal.shape}")
    if terminated:
        target = reward                               # a terminal state's future is worth 0
    else:
        if not legal.any():
            raise ValueError("next_mask has no legal action in a non-terminal next state")
        target = reward + gamma * Q[next_state, legal].max()
    delta = float(target - Q[state, action])
    Q[state, action] = incremental_update(Q[state, action], target, alpha)
    return delta


def sarsa_update(
    Q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    next_action: int,
    terminated: bool,
    alpha: float,
    gamma: float,
) -> float:
    """SARSA update of Q[state, action], in place; returns the TD error.

    Like ``q_learning_update``, but the target uses the action actually chosen
    in the next state: target = reward if ``terminated``, else
    reward + gamma * Q[next_state, next_action].

    Parameters
    ----------
    Q : ndarray of shape (n_states, n_actions)
        Action-value table, modified in place.
    state : int
        State s.
    action : int
        Action a taken in s.
    reward : float
        Reward r received.
    next_state : int
        State s' reached.
    next_action : int
        Action a' chosen in s' by the behaviour policy (ignored if
        ``terminated``).
    terminated : bool
        True if s' is terminal (the target is then r).
    alpha : float
        Learning rate, in (0, 1].
    gamma : float
        Discount factor, in [0, 1].

    Returns
    -------
    float
        The TD error delta, computed before the update.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1] or ``gamma`` is not in [0, 1].

    Notes
    -----
    Tested on hand-computed values; identical to ``q_learning_update`` when
    ``next_action`` is greedy.

    Examples
    --------
    >>> Q = np.zeros((2, 2))
    >>> Q[1] = [1.0, 3.0]
    >>> sarsa_update(Q, 0, 1, reward=1.0, next_state=1, next_action=0, terminated=False,
    ...              alpha=0.5, gamma=0.9)     # target = 1 + 0.9 * 1
    1.9
    >>> Q[0]
    array([0.  , 0.95])
    """
    Q = _q_table(Q, check_nan=False)
    n_states, n_actions = Q.shape
    state = _index(state, n_states, "state")
    action = _index(action, n_actions, "action")
    next_state = _index(next_state, n_states, "next_state")
    if np.isnan(Q[state]).any() or np.isnan(Q[next_state]).any():   # the rows the update reads (checking the whole
        raise ValueError("Q contains NaN values")                     # table at each step would slow a long loop)
    reward = _real(reward, "reward")
    if np.isnan(reward):
        raise ValueError("reward is NaN")
    alpha, gamma = _check_alpha(alpha), _check_gamma(gamma)
    if terminated:
        target = reward                               # next_action is ignored
    else:
        next_action = _index(next_action, n_actions, "next_action")
        target = reward + gamma * Q[next_state, next_action]
    delta = float(target - Q[state, action])
    Q[state, action] = incremental_update(Q[state, action], target, alpha)
    return delta


def train_tabular(
    env: Any,
    n_episodes: int,
    *,
    algorithm: str = "q_learning",
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float | Callable[[int], float] = 0.1,
    max_steps: int = 200,
    Q_init: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Train a tabular agent (Q-learning or SARSA) with epsilon-greedy exploration.

    For each episode: reset the environment (with a seed drawn from ``rng``),
    then until the episode ends (terminated, truncated, or ``max_steps``
    steps): choose an action with ``epsilon_greedy_action`` (chapter 11), step
    the environment, update Q with ``q_learning_update`` or ``sarsa_update``.
    With SARSA, the next action is chosen before the update and then played.

    Action masks: when ``info`` contains ``"action_mask"`` (boolean array of
    the legal actions), epsilon-greedy is applied to the sub-vector of Q of
    the legal actions (then re-indexed), and the Q-learning max is taken over
    the legal actions of the next state.

    Parameters
    ----------
    env : object with the gymnasium API
        ``reset(seed=None) -> (state, info)``,
        ``step(action) -> (state, reward, terminated, truncated, info)``,
        and discrete spaces ``observation_space.n`` and ``action_space.n``
        (FrozenLake, CliffWalking, ``wb.envs.TicTacToe``...).
    n_episodes : int
        Number of training episodes, >= 1.
    algorithm : {"q_learning", "sarsa"}, default="q_learning"
        Update rule.
    alpha : float, default=0.1
        Learning rate, in (0, 1].
    gamma : float, default=0.99
        Discount factor, in [0, 1].
    epsilon : float or callable, default=0.1
        Exploration rate: a constant, or a schedule ``episode_index -> epsilon``
        (episode_index starts at 0).
    max_steps : int, default=200
        Cut-off per episode. Reaching it counts as a truncation: the last
        update keeps its bootstrap.
    Q_init : ndarray of shape (n_states, n_actions) or None, default=None
        Initial table (copied, never modified). None means zeros.
    rng : numpy.random.Generator or None, default=None
        Controls the exploration and the seeds passed to ``env.reset``: the
        same generator state gives the same history. None means
        ``np.random.default_rng()``.

    Returns
    -------
    Q : ndarray of shape (n_states, n_actions)
        The learned table.
    history : dict of str to ndarray
        ``"returns"``: sum of the (undiscounted) rewards of each episode,
        ``"lengths"``: number of steps of each episode, ``"epsilons"``:
        epsilon used in each episode; each of shape (n_episodes,).

    Raises
    ------
    ValueError
        For an unknown algorithm, or if the observation or action space is not
        discrete (no ``n`` attribute).

    Notes
    -----
    Tested on ``FrozenLake-v1`` (``is_slippery=False``): the greedy policy
    reaches the goal and Q is close to the Q* of ``value_iteration``; on
    ``CliffWalking-v1``: the greedy path of Q-learning follows the cliff,
    that of SARSA stays away from it; the same rng gives the same history.

    Examples
    --------
    >>> import gymnasium as gym
    >>> env = gym.make("FrozenLake-v1", is_slippery=False)
    >>> Q, history = train_tabular(env, 300, rng=np.random.default_rng(0))
    >>> Q.shape, history["returns"].shape
    ((16, 4), (300,))
    """
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError(f"algorithm must be 'q_learning' or 'sarsa', got {algorithm!r}")
    n_episodes = _count(n_episodes, "n_episodes")
    max_steps = _count(max_steps, "max_steps")
    alpha, gamma = _check_alpha(alpha), _check_gamma(gamma)
    if not callable(epsilon):
        _check_epsilon(epsilon)
    n_states, n_actions = _discrete_sizes(env)
    if Q_init is None:
        Q = np.zeros((n_states, n_actions))
    else:
        Q = np.array(Q_init, dtype=np.float64, copy=True)
        if Q.shape != (n_states, n_actions):
            raise ValueError(f"Q_init must have shape ({n_states}, {n_actions}), got {Q.shape}")
        if np.isnan(Q).any():
            raise ValueError("Q_init contains NaN values")
    rng = np.random.default_rng() if rng is None else rng
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    epsilons = np.zeros(n_episodes)
    for episode in range(n_episodes):
        eps = _check_epsilon(epsilon(episode) if callable(epsilon) else epsilon)
        state, info = env.reset(seed=int(rng.integers(2**31)))
        action = _explore(Q, state, info, eps, rng)
        total, steps = 0.0, 0
        while True:
            next_state, reward, terminated, truncated, next_info = env.step(action)
            steps += 1
            total += float(reward)
            next_action = None
            if algorithm == "q_learning":
                mask = None if terminated else _mask_of(next_info)
                q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma, next_mask=mask)
            else:                                     # SARSA: choose a' first, update, then play a'
                if not terminated:
                    next_action = _explore(Q, next_state, next_info, eps, rng)
                sarsa_update(Q, state, action, reward, next_state, next_action, terminated, alpha, gamma)
            if terminated or truncated or steps >= max_steps:
                break                                 # max_steps is a truncation: the update above bootstrapped
            if next_action is None:
                next_action = _explore(Q, next_state, next_info, eps, rng)
            state, info, action = next_state, next_info, next_action
        returns[episode], lengths[episode], epsilons[episode] = total, steps, eps
    return Q, {"returns": returns, "lengths": lengths, "epsilons": epsilons}


def evaluate_policy(
    env: Any,
    policy: ArrayLike | Callable[[Any, dict[str, Any]], int],
    n_episodes: int = 100,
    *,
    max_steps: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Run a fixed policy for several episodes and summarise the results.

    Parameters
    ----------
    env : object with the gymnasium API
        ``reset(seed=None)`` and ``step(action)`` as in ``train_tabular``.
    policy : array-like of shape (n_states,) or callable
        A table of actions (for a discrete state space: the action is
        ``policy[state]``), or a function ``(state, info) -> action`` (any
        state space, for example CartPole in bonus chapter B8).
    n_episodes : int, default=100
        Number of episodes, >= 1.
    max_steps : int, default=200
        Cut-off per episode.
    rng : numpy.random.Generator or None, default=None
        Draws the seeds passed to ``env.reset``. None means
        ``np.random.default_rng()``.

    Returns
    -------
    dict of str to float
        ``"mean_return"`` and ``"std_return"`` (population standard deviation,
        ddof=0) of the undiscounted episode returns, and ``"mean_length"``.

    Raises
    ------
    ValueError
        If ``n_episodes < 1``.

    Notes
    -----
    Tested against a direct Monte Carlo loop with gymnasium on FrozenLake
    (same seeds), and on exact values for deterministic FrozenLake.

    Examples
    --------
    On the 4 x 4 FrozenLake without slipping, the path right, right, down,
    down, down, right (actions 2, 2, 1, 1, 1, 2) reaches the goal:

    >>> import gymnasium as gym
    >>> env = gym.make("FrozenLake-v1", is_slippery=False)
    >>> policy = np.zeros(16, dtype=int)
    >>> policy[[0, 1, 14]] = 2
    >>> policy[[2, 6, 10]] = 1
    >>> evaluate_policy(env, policy, n_episodes=5)
    {'mean_return': 1.0, 'std_return': 0.0, 'mean_length': 6.0}
    """
    n_episodes = _count(n_episodes, "n_episodes")
    max_steps = _count(max_steps, "max_steps")
    if callable(policy):
        act = policy
    else:
        table = np.array(policy, copy=True)
        if table.ndim != 1 or table.size == 0:
            raise ValueError(f"policy must be a non-empty 1-D table of actions, got shape {table.shape}")
        if not np.issubdtype(table.dtype, np.integer):
            raise ValueError(f"policy must hold integer actions, got dtype {table.dtype}")

        def act(state, info):
            return int(table[state])
    rng = np.random.default_rng() if rng is None else rng
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes)
    for episode in range(n_episodes):
        state, info = env.reset(seed=int(rng.integers(2**31)))
        total, steps = 0.0, 0
        while steps < max_steps:
            state, reward, terminated, truncated, info = env.step(act(state, info))
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        returns[episode], lengths[episode] = total, steps
    return {"mean_return": float(returns.mean()), "std_return": float(returns.std()),
            "mean_length": float(lengths.mean())}


def value_iteration(
    P: Mapping[int, Mapping[int, Sequence[tuple[float, int, float, bool]]]],
    gamma: float = 0.99,
    *,
    theta: float = 1e-10,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal state values and action values of a known MDP (Bellman optimality backups).

    Repeat synchronous sweeps (each new V is computed from the previous one):
    Q[s, a] = sum over the outcomes (p, s', r, done) of
    p * (r + gamma * V[s'] * (1 - done)); V[s] = max over a of Q[s, a];
    stop when max |V_new - V| < theta, or after ``max_iter`` sweeps.

    Parameters
    ----------
    P : mapping
        Tabular model in the gymnasium format (``env.unwrapped.P``):
        ``P[s][a]`` is a list of outcomes ``(prob, next_state, reward, terminated)``.
        States are 0 .. n_states - 1 and every state has the same actions
        0 .. n_actions - 1.
    gamma : float, default=0.99
        Discount factor, in [0, 1).
    theta : float, default=1e-10
        Stopping threshold on the largest change of V in a sweep.
    max_iter : int, default=10_000
        Maximum number of sweeps (safety cap).

    Returns
    -------
    V : ndarray of shape (n_states,)
        Optimal state values.
    Q : ndarray of shape (n_states, n_actions)
        Optimal action values, computed from the final V
        (``greedy_policy(Q)`` is an optimal policy).

    Raises
    ------
    ValueError
        If ``gamma`` is not in [0, 1), or if ``P`` is malformed (empty, or
        states with different numbers of actions).

    Notes
    -----
    Tested by evaluating the returned greedy policy exactly (solving the
    linear Bellman system with ``np.linalg.solve``), and against closed-form
    values on a small chain MDP.

    Examples
    --------
    State 0: action 0 stays (reward 0), action 1 ends the episode with
    reward 1. State 1 is terminal.

    >>> P = {0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, True)]},
    ...      1: {0: [(1.0, 1, 0.0, True)], 1: [(1.0, 1, 0.0, True)]}}
    >>> V, Q = value_iteration(P, gamma=0.9)
    >>> V
    array([1., 0.])
    >>> Q
    array([[0.9, 1. ],
           [0. , 0. ]])
    """
    gamma = _check_gamma(gamma, below_one=True)
    theta = _real(theta, "theta")
    if not theta > 0:
        raise ValueError(f"theta must be > 0, got {theta}")
    max_iter = _count(max_iter, "max_iter")
    sa, prob, nxt, rew, done, n_states, n_actions = _parse_model(P)

    def backup(V: np.ndarray) -> np.ndarray:
        """Q[s, a] = sum of p * (r + gamma * V[s'] * (1 - done)) over the outcomes of (s, a)."""
        contrib = prob * (rew + gamma * V[nxt] * (1.0 - done))
        return np.bincount(sa, weights=contrib, minlength=n_states * n_actions).reshape(n_states, n_actions)

    V = np.zeros(n_states)
    for _ in range(max_iter):
        V_new = backup(V).max(axis=1)                 # synchronous sweep: from the previous V only
        change = np.max(np.abs(V_new - V))
        V = V_new
        if change < theta:
            break
    return V, backup(V)


class ReplayBuffer:
    """Fixed-size FIFO memory of transitions, sampled uniformly (experience replay).

    Once full, each new transition overwrites the oldest one. Sampling random
    mini-batches breaks the correlation between consecutive transitions. Also
    used by the DQN of bonus chapter B8 (``obs_shape=(4,)``,
    ``obs_dtype="float32"`` for CartPole).

    ``__init__`` is already written: it checks ``capacity`` and prepares the
    storage, i.e. one array per field with ``capacity`` rows
    (``self._states`` and ``self._next_states`` of shape
    ``(capacity, *obs_shape)`` and dtype ``obs_dtype``, ``self._actions``
    int64, ``self._rewards`` float64, ``self._terminated`` bool), the next
    write position ``self._pos``, the number of stored transitions
    ``self._size`` and the generator ``self._rng``.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions, >= 1.
    obs_shape : tuple of int, default=()
        Shape of one observation: ``()`` for a discrete state (an int),
        ``(4,)`` for CartPole.
    obs_dtype : str, default="int64"
        NumPy dtype of the stored observations.
    random_state : int or None, default=None
        Seed of the sampling generator.

    Raises
    ------
    ValueError
        If ``capacity < 1``.

    Notes
    -----
    Tested on properties: FIFO eviction beyond the capacity, shapes and dtypes,
    uniform sampling (chi-square test), reproducible with ``random_state``.

    Examples
    --------
    >>> buffer = ReplayBuffer(capacity=2, random_state=0)
    >>> buffer.push(0, 1, 0.5, 1, False)
    >>> buffer.push(1, 0, 1.0, 2, True)
    >>> buffer.push(2, 1, 0.0, 3, False)    # overwrites the oldest transition
    >>> len(buffer)
    2
    >>> batch = buffer.sample(2)
    >>> sorted(batch["states"].tolist())
    [1, 2]
    >>> batch["terminated"].dtype
    dtype('bool')
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...] = (),
        obs_dtype: str = "int64",
        random_state: int | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.obs_shape = tuple(obs_shape)
        self.obs_dtype = obs_dtype
        self.random_state = random_state
        self._states = np.zeros((capacity, *self.obs_shape), dtype=obs_dtype)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._next_states = np.zeros((capacity, *self.obs_shape), dtype=obs_dtype)
        self._terminated = np.zeros(capacity, dtype=bool)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(random_state)

    def push(
        self, state: Any, action: int, reward: float, next_state: Any, terminated: bool
    ) -> None:
        """Store one transition, overwriting the oldest one when the buffer is full.

        Parameters
        ----------
        state : int or array-like of shape obs_shape
            Observation before the action.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : int or array-like of shape obs_shape
            Observation after the action.
        terminated : bool
            Whether ``next_state`` is terminal (not merely truncated).
        """
        state, next_state = np.asarray(state), np.asarray(next_state)
        for value, name in ((state, "state"), (next_state, "next_state")):
            if value.shape != self.obs_shape:
                raise ValueError(f"{name} must have shape {self.obs_shape}, got {value.shape}")
        i = self._pos
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._terminated[i] = bool(terminated)
        self._pos = (self._pos + 1) % self.capacity       # the oldest transition is overwritten next
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Draw ``batch_size`` distinct stored transitions uniformly (without replacement).

        Parameters
        ----------
        batch_size : int
            Number of transitions, 1 <= batch_size <= len(self).

        Returns
        -------
        dict of str to ndarray
            Keys ``"states"`` and ``"next_states"`` (shape
            ``(batch_size, *obs_shape)``, dtype ``obs_dtype``), ``"actions"``
            (int64), ``"rewards"`` (float64) and ``"terminated"`` (bool), each
            with leading dimension ``batch_size``; row i of every array
            belongs to the same transition. The arrays are copies.

        Raises
        ------
        ValueError
            If ``batch_size < 1`` or ``batch_size > len(self)``.
        """
        batch_size = _count(batch_size, "batch_size")
        if batch_size > self._size:
            raise ValueError(f"batch_size must be <= len(self) = {self._size}, got {batch_size}")
        idx = self._rng.choice(self._size, size=batch_size, replace=False)
        return {"states": self._states[idx], "actions": self._actions[idx], "rewards": self._rewards[idx],
                "next_states": self._next_states[idx], "terminated": self._terminated[idx]}

    def __len__(self) -> int:
        """Return the number of transitions currently stored (at most ``capacity``)."""
        return self._size
