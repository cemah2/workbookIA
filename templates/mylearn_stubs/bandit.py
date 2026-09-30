"""Multi-armed bandits — mylearn, chapter 11 (Learning and Reasoning).

Learning by trial and error: environments with K arms (Bernoulli and Gaussian rewards),
action-selection rules (greedy with random tie-breaking, epsilon-greedy, UCB, Thompson
sampling), the incremental update of an estimate and the interaction loop that ties
them together. ``rl.py`` (chapter 26) reuses ``epsilon_greedy_action``,
``argmax_random_tie`` and ``incremental_update`` for tabular reinforcement learning.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike


class BernoulliBandit:
    """K-armed bandit whose arm a pays 1 with probability ``probs[a]``, else 0.

    An environment, not an estimator: ``__init__`` (already written) validates the
    probabilities, computes the attributes below and creates the internal random
    generator. You write ``pull``.

    Parameters
    ----------
    probs : array-like of shape (n_arms,)
        Success probability of every arm, each in [0, 1].
    random_state : int or None, default=None
        Seed of the internal generator ``np.random.default_rng(random_state)``: the same
        seed gives the same sequence of rewards.

    Attributes
    ----------
    means : np.ndarray of shape (n_arms,)
        Expected reward of every arm (= ``probs`` as a float array).
    n_arms : int
        Number of arms K.
    best_arm : int
        Index of the best arm (the first one if several are tied).
    best_mean : float
        Expected reward of the best arm.

    Raises
    ------
    ValueError
        If ``probs`` is empty or not 1-D, or a probability is outside [0, 1].

    Notes
    -----
    Tested by the law of large numbers (the mean of 10^5 pulls of arm a is within 4
    standard errors of ``probs[a]``) and by reproducibility with ``random_state``.

    Examples
    --------
    >>> bandit = BernoulliBandit([0.2, 0.5, 0.9], random_state=0)
    >>> bandit.n_arms, bandit.best_arm, bandit.best_mean
    (3, 2, 0.9)
    >>> BernoulliBandit([0.0, 1.0]).pull(1)
    1.0
    """

    def __init__(self, probs: ArrayLike, random_state: int | None = None) -> None:
        means = np.asarray(probs, dtype=float)
        if means.ndim != 1 or means.size == 0:
            raise ValueError("probs must be a non-empty 1-D array of probabilities")
        if not np.all((means >= 0.0) & (means <= 1.0)):
            raise ValueError("every probability of probs must be in [0, 1]")
        self.means = means
        self.n_arms = int(means.size)
        self.best_arm = int(np.argmax(means))
        self.best_mean = float(means[self.best_arm])
        self._rng = np.random.default_rng(random_state)

    def pull(self, arm: int) -> float:
        """Pull one arm and return its reward.

        Parameters
        ----------
        arm : int
            Index of the arm, in 0..n_arms-1.

        Returns
        -------
        float
            1.0 with probability ``means[arm]``, else 0.0.

        Raises
        ------
        IndexError
            If ``arm`` is not in 0..n_arms-1 (negative indices are refused too).
        """
        # TODO: check the index, then one draw with the internal generator self._rng.
        raise NotImplementedError("pull() is not implemented yet")


class GaussianBandit:
    """K-armed bandit whose arm a pays a reward drawn from N(means[a], std²).

    The classic "10-armed testbed" of Sutton & Barto uses 10 arms whose means are drawn
    from N(0, 1) and ``std = 1``. Same interface as ``BernoulliBandit``; ``__init__``
    is already written, you write ``pull``.

    Parameters
    ----------
    means : array-like of shape (n_arms,)
        Expected reward of every arm.
    std : float, default=1.0
        Standard deviation of the rewards (> 0), the same for every arm.
    random_state : int or None, default=None
        Seed of the internal generator ``np.random.default_rng(random_state)``.

    Attributes
    ----------
    means : np.ndarray of shape (n_arms,)
        Expected reward of every arm (float array).
    std : float
        Standard deviation of the rewards.
    n_arms : int
        Number of arms K.
    best_arm : int
        Index of the best arm (the first one if several are tied).
    best_mean : float
        Expected reward of the best arm.

    Raises
    ------
    ValueError
        If ``means`` is empty, not 1-D or not finite, or if ``std <= 0``.

    Notes
    -----
    Tested with the empirical mean and standard deviation of 10^5 pulls against
    ``(means[a], std)`` and by reproducibility with ``random_state``.

    Examples
    --------
    >>> bandit = GaussianBandit([0.0, 1.5, -0.5], std=1.0, random_state=0)
    >>> bandit.n_arms, bandit.best_arm, bandit.best_mean
    (3, 1, 1.5)
    """

    def __init__(
        self, means: ArrayLike, std: float = 1.0, random_state: int | None = None
    ) -> None:
        arm_means = np.asarray(means, dtype=float)
        if arm_means.ndim != 1 or arm_means.size == 0:
            raise ValueError("means must be a non-empty 1-D array")
        if not np.all(np.isfinite(arm_means)):
            raise ValueError("every value of means must be finite")
        if not std > 0:
            raise ValueError("std must be > 0")
        self.means = arm_means
        self.std = float(std)
        self.n_arms = int(arm_means.size)
        self.best_arm = int(np.argmax(arm_means))
        self.best_mean = float(arm_means[self.best_arm])
        self._rng = np.random.default_rng(random_state)

    def pull(self, arm: int) -> float:
        """Pull one arm and return its reward.

        Parameters
        ----------
        arm : int
            Index of the arm, in 0..n_arms-1.

        Returns
        -------
        float
            A reward drawn from N(means[arm], std²).

        Raises
        ------
        IndexError
            If ``arm`` is not in 0..n_arms-1 (negative indices are refused too).
        """
        # TODO: check the index, then one draw with the internal generator self._rng.
        raise NotImplementedError("pull() is not implemented yet")


def argmax_random_tie(values: ArrayLike, rng: np.random.Generator | None = None) -> int:
    """Return the index of a maximum of values, chosen uniformly among ties.

    ``np.argmax`` always returns the first maximum: an agent that uses it always
    prefers the first arms while all its estimates are still equal (e.g. all 0).

    Parameters
    ----------
    values : array-like of shape (n,)
        Values to compare (e.g. estimated values of the actions).
    rng : np.random.Generator or None, default=None
        Random generator; ``np.random.default_rng()`` if None.

    Returns
    -------
    int
        Index of a maximum; each of the m tied maxima has probability 1/m.

    Raises
    ------
    ValueError
        If ``values`` is empty or contains NaN.

    Notes
    -----
    Tested by properties: the result is always a maximiser; with m ties each index is
    chosen with frequency close to 1/m (chi-square test).

    Examples
    --------
    >>> argmax_random_tie([1.0, 3.0, 2.0])
    1
    """
    # TODO: indices of all the maxima, then one uniform choice among them.
    raise NotImplementedError("argmax_random_tie() is not implemented yet")


def epsilon_greedy_action(
    q_values: ArrayLike, epsilon: float, rng: np.random.Generator | None = None
) -> int:
    """Choose an action with the epsilon-greedy rule.

    With probability ``epsilon``: explore, a uniformly random action among all of them
    (the best one included). Otherwise: exploit, ``argmax_random_tie(q_values)``.
    The best action is therefore chosen with probability about 1 - epsilon + epsilon/K.
    In chapter 26, ``q_values`` is the row ``Q[state]`` of a Q-table.

    Parameters
    ----------
    q_values : array-like of shape (n_actions,)
        Estimated values of the actions.
    epsilon : float
        Exploration rate in [0, 1].
    rng : np.random.Generator or None, default=None
        Random generator; ``np.random.default_rng()`` if None.

    Returns
    -------
    int
        The chosen action.

    Raises
    ------
    ValueError
        If ``epsilon`` is outside [0, 1] or ``q_values`` is empty.

    Notes
    -----
    Tested by properties: the frequency of the unique best action is close to
    1 - epsilon + epsilon/K; with epsilon = 0 the action is always a maximiser.

    Examples
    --------
    >>> epsilon_greedy_action([0.2, 0.9, 0.5], epsilon=0.0)
    1
    """
    # TODO: one uniform draw decides between exploring and exploiting.
    raise NotImplementedError("epsilon_greedy_action() is not implemented yet")


def incremental_update(
    estimate: float | np.ndarray, target: float | np.ndarray, step_size: float
) -> float | np.ndarray:
    """Move an estimate towards a target: ``estimate + step_size * (target - estimate)``.

    With ``step_size = 1/n`` for the n-th observation, the estimate is exactly the
    running mean of the observations; a constant step size gives more weight to recent
    observations and tracks a target that changes over time. Chapter 26 uses the same
    form for temporal-difference updates.

    Parameters
    ----------
    estimate : float or np.ndarray
        Current estimate.
    target : float or np.ndarray
        New observation (a reward here, a TD target in chapter 26).
    step_size : float
        Step size in (0, 1].

    Returns
    -------
    float or np.ndarray
        Updated estimate, with the type and shape of ``estimate``.

    Raises
    ------
    ValueError
        If ``step_size`` is outside (0, 1].

    Notes
    -----
    Tested by properties: with ``step_size = 1/n`` over a sequence, the result equals
    ``np.mean`` of the sequence (float tolerance).

    Examples
    --------
    >>> incremental_update(2.0, 4.0, step_size=0.5)
    3.0
    """
    # TODO: check step_size, then one line.
    raise NotImplementedError("incremental_update() is not implemented yet")


def ucb_action(q_values: ArrayLike, counts: ArrayLike, t: int, c: float = 2.0) -> int:
    """Choose an arm with the upper confidence bound (UCB1-style) rule.

    An arm never tried is played first (the one with the smallest index). Otherwise
    play ``argmax_a q_values[a] + c * sqrt(ln t / counts[a])`` (first maximum on ties):
    the bonus is large for arms tried rarely, which forces exploration where the
    estimate is uncertain.

    Parameters
    ----------
    q_values : array-like of shape (n_arms,)
        Estimated values of the arms.
    counts : array-like of shape (n_arms,)
        Number of pulls of every arm so far.
    t : int
        Current step number (>= 1).
    c : float, default=2.0
        Exploration strength (>= 0); c = 0 gives the greedy choice.

    Returns
    -------
    int
        The chosen arm (deterministic).

    Raises
    ------
    ValueError
        If ``q_values`` and ``counts`` have different shapes, ``t < 1`` or ``c < 0``.

    Notes
    -----
    Tested against the NumPy formula on hand-made cases; c = 0 with all arms tried is
    the greedy choice.

    Examples
    --------
    >>> ucb_action([0.5, 0.0, 0.0], counts=[3, 0, 0], t=4)
    1
    >>> ucb_action([0.5, 0.4], counts=[10, 1], t=11)
    1
    >>> ucb_action([0.5, 0.4], counts=[10, 1], t=11, c=0.0)
    0
    """
    # TODO: untried arm first, otherwise argmax of the upper confidence bounds.
    raise NotImplementedError("ucb_action() is not implemented yet")


def thompson_action(
    successes: ArrayLike, failures: ArrayLike, rng: np.random.Generator | None = None
) -> int:
    """Choose a Bernoulli arm by Thompson sampling.

    With a uniform prior on the success probability θ_a of each arm, the posterior after
    s successes and f failures is Beta(1 + s, 1 + f) (the coin of chapter 4). Draw one
    θ_a from each posterior (``rng.beta``) and play the arm with the largest draw: an
    arm is played with the probability that it is the best one.

    Parameters
    ----------
    successes : array-like of shape (n_arms,)
        Number of rewards 1 of every arm.
    failures : array-like of shape (n_arms,)
        Number of rewards 0 of every arm.
    rng : np.random.Generator or None, default=None
        Random generator; ``np.random.default_rng()`` if None.

    Returns
    -------
    int
        The chosen arm.

    Raises
    ------
    ValueError
        If the shapes differ or a count is negative.

    Notes
    -----
    Tested by comparing the choice frequencies with P(arm a has the largest θ),
    estimated by Monte Carlo with ``scipy.stats.beta``.

    Examples
    --------
    >>> thompson_action([100, 0], [0, 100], rng=np.random.default_rng(0))
    0
    """
    # TODO: one Beta draw per arm, then the argmax.
    raise NotImplementedError("thompson_action() is not implemented yet")


def run_bandit(
    bandit: Any,
    select_action: Callable[[np.ndarray, np.ndarray, int, np.random.Generator], int],
    n_steps: int = 1000,
    step_size: float | None = None,
    initial_value: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Let a policy play a bandit for n_steps and record the history.

    ``q_values`` starts at ``initial_value`` for every arm and ``counts`` at 0. At each
    step t = 1..n_steps: ``a = select_action(q_values, counts, t, rng)``,
    ``r = bandit.pull(a)``, ``counts[a] += 1``, then
    ``q_values[a] = incremental_update(q_values[a], r, step)`` with
    ``step = 1 / counts[a]`` (sample average) if ``step_size`` is None, else
    ``step = step_size``.

    Parameters
    ----------
    bandit : object
        Any environment with ``n_arms``, ``means``, ``best_mean`` and ``pull(arm)``,
        e.g. ``BernoulliBandit`` or ``GaussianBandit``.
    select_action : callable
        Policy ``(q_values, counts, t, rng) -> arm``, for example
        ``lambda q, n, t, g: epsilon_greedy_action(q, 0.1, g)`` or
        ``lambda q, n, t, g: ucb_action(q, n, t)``. It must not modify its arguments.
    n_steps : int, default=1000
        Number of pulls (>= 1).
    step_size : float or None, default=None
        None: sample average (step 1/n); otherwise a constant step size in (0, 1].
    initial_value : float, default=0.0
        Initial estimate of every arm (a large value gives optimistic initialisation).
    rng : np.random.Generator or None, default=None
        Generator passed to the policy; ``np.random.default_rng()`` if None.

    Returns
    -------
    dict of str to np.ndarray
        ``"actions"``: (n_steps,) int, the arm played at each step;
        ``"rewards"``: (n_steps,) float, the rewards received;
        ``"optimal"``: (n_steps,) bool, True where ``means[a_t] == best_mean``;
        ``"regret"``: (n_steps,) float, cumulative pseudo-regret
        ``sum_{s <= t} (best_mean - means[a_s])``;
        ``"q_values"``: (n_arms,) float, final estimates;
        ``"counts"``: (n_arms,) int, final numbers of pulls.

    Raises
    ------
    ValueError
        If ``n_steps < 1`` or the policy returns an arm outside 0..n_arms-1.

    Notes
    -----
    Tested by properties: ``counts.sum() == n_steps``; with ``step_size=None`` and
    ``initial_value=0``, ``q_values[a]`` is the mean of the rewards of arm a; the regret
    never decreases; the same seeds give the same history.

    Examples
    --------
    >>> bandit = BernoulliBandit([0.25, 0.75], random_state=0)
    >>> always_first = lambda q, n, t, g: 0
    >>> history = run_bandit(bandit, always_first, n_steps=4)
    >>> history["counts"]
    array([4, 0])
    >>> history["regret"]
    array([0.5, 1. , 1.5, 2. ])
    """
    # TODO: initialise the arrays, loop over the steps, then build the dictionary
    # (np.cumsum gives the cumulative regret).
    raise NotImplementedError("run_bandit() is not implemented yet")
