"""Tabular reinforcement learning — mylearn, chapter 26 (Reinforcement Learning).

Returns, policies, Q-learning and SARSA updates, a generic training loop for
discrete environments with the gymnasium API (action masks included), policy
evaluation, value iteration (the exact reference when the model is known) and
a replay buffer (reused by the DQN of bonus chapter B8). Exploration reuses
chapter 11: ``epsilon_greedy_action`` and ``argmax_random_tie`` come from
``bandit.py``, and a Q update is an ``incremental_update`` towards the TD
target.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .bandit import (  # chapter 11, to reuse below
    argmax_random_tie,
    epsilon_greedy_action,
    incremental_update,
)


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
    raise NotImplementedError("discounted_return() is not implemented yet")


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
    raise NotImplementedError("returns_to_go() is not implemented yet")


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
    raise NotImplementedError("greedy_policy() is not implemented yet")


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
    raise NotImplementedError("softmax_action() is not implemented yet")


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
    raise NotImplementedError("q_learning_update() is not implemented yet")


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
    raise NotImplementedError("sarsa_update() is not implemented yet")


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
    # TODO: one loop over the episodes, one over the steps; keep the info (action
    #   mask) of the current and of the next state.
    raise NotImplementedError("train_tabular() is not implemented yet")


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
    # TODO: same episode loop as train_tabular, without exploration nor updates.
    raise NotImplementedError("evaluate_policy() is not implemented yet")


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
    # TODO: sweep until the largest change of V is below theta; compute Q from
    #   the final V.
    raise NotImplementedError("value_iteration() is not implemented yet")


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
        # TODO: write at self._pos, then advance it modulo capacity and grow self._size.
        raise NotImplementedError("push() is not implemented yet")

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
        # TODO: draw distinct indices among the stored transitions with self._rng.
        raise NotImplementedError("sample() is not implemented yet")

    def __len__(self) -> int:
        """Return the number of transitions currently stored (at most ``capacity``)."""
        raise NotImplementedError("__len__() is not implemented yet")
