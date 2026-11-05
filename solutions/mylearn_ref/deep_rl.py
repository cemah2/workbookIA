"""Deep RL computations — mylearn, chapter B8 (Modern RL: DQN, policy gradient, PPO and RLHF).

The parts of modern deep RL that can be tested without a network or an environment, in
NumPy: generalized advantage estimation (with the terminated / truncated distinction),
the clipped PPO objective and its gradient, the explained variance of a critic, and the
preference losses of RLHF (Bradley-Terry reward model, DPO). DQN, REINFORCE and PPO
themselves are written in PyTorch in the notebook, with ``rl.py`` from chapter 26.

Reference implementation.
"""

from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike


def _real(value, name: str) -> float:
    """A real number (not a bool) as a Python float; NaN refused."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if np.isnan(value):
        raise ValueError(f"{name} must be a real number, got nan")
    return value


def _unit_interval(value, name: str) -> float:
    """A real number in [0, 1]."""
    value = _real(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _positive(value, name: str) -> float:
    """A real number > 0."""
    value = _real(value, name)
    if not value > 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _floats(values, name: str) -> np.ndarray:
    """A new float64 array (the caller's data is never modified); NaN refused (BIBLE §22, Q9)."""
    try:
        arr = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric array") from None
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values: remove the missing values first")
    return arr


def _flags(values, name: str) -> np.ndarray:
    """A new bool array: booleans, or numbers that are all 0 or 1 (NaN and other values refused)."""
    arr = np.asarray(values)
    if arr.dtype == np.bool_:
        return arr.copy()
    if arr.dtype.kind in "iuf" and np.isin(arr, (0, 1)).all():
        return arr.astype(np.bool_)
    raise ValueError(f"{name} must hold booleans (True / False), got values of dtype {arr.dtype}")


def _same_shape(arrays: dict) -> tuple[int, ...]:
    """The common shape of the named arrays (ValueError naming the first one that differs)."""
    names = list(arrays)
    shape = arrays[names[0]].shape
    for name in names[1:]:
        if arrays[name].shape != shape:
            raise ValueError(f"{name} must have the shape of {names[0]} {shape}, got {arrays[name].shape}")
    return shape


def _vector(arrays: dict) -> int:
    """The common length n >= 1 of 1-D arrays of the same shape."""
    shape = _same_shape(arrays)
    first = next(iter(arrays))
    if len(shape) != 1:
        raise ValueError(f"{first} must be 1-D (one value per example), got shape {shape}")
    if shape[0] == 0:
        raise ValueError(f"{first} is empty")
    return shape[0]


def _sigmoid_of_minus(d: np.ndarray) -> np.ndarray:
    """sigmoid(-d) = 1 / (1 + exp(d)), stably: exp(-logaddexp(0, d)) never overflows."""
    return np.exp(-np.logaddexp(0.0, d))


def compute_gae(
    rewards: ArrayLike,
    values: ArrayLike,
    next_values: ArrayLike,
    terminated: ArrayLike,
    truncated: ArrayLike,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation (GAE) over a rollout.

    ``delta_t = r_t + gamma (1 - terminated_t) V(s_{t+1}) - V(s_t)`` and, going backwards
    from the last step, ``A_t = delta_t + gamma lam (1 - done_t) A_{t+1}`` with
    ``done_t = terminated_t or truncated_t`` and ``A_T = 0`` after the last step.
    A truncated episode (time limit) still bootstraps on ``V(s_{t+1})``, but the
    recursion stops there, because the next step belongs to a new episode.

    Parameters
    ----------
    rewards : array-like of shape (T,) or (T, n_envs)
        Reward ``r_t`` of each step (time on the first axis).
    values : array-like, same shape
        Critic values ``V(s_t)``.
    next_values : array-like, same shape
        Critic values ``V(s_{t+1})`` of the next observations; for a truncated step, the
        value of the real final observation (not of the reset one).
    terminated : array-like of bool, same shape
        True end of the episode (goal reached, fall...): no bootstrap.
    truncated : array-like of bool, same shape
        Episode cut by a time limit: bootstrap, but the recursion stops.
    gamma : float, default=0.99
        Discount factor in [0, 1].
    lam : float, default=0.95
        GAE lambda in [0, 1]: 0 gives the one-step TD errors, 1 the Monte Carlo
        advantages.

    Returns
    -------
    advantages : np.ndarray
        ``A_t``, with the shape of ``rewards``.
    returns : np.ndarray
        ``advantages + values`` (the targets of the critic), same shape.

    Raises
    ------
    ValueError
        If the shapes differ, or ``gamma`` or ``lam`` is outside [0, 1].

    Notes
    -----
    Tested against a reverse-loop reference (in the style of CleanRL's ``ppo.py``), and
    with properties: ``lam = 0`` gives the TD errors ``delta_t``; ``lam = 1``, ``V = 0``
    and no truncation give ``rl.returns_to_go`` within each episode.

    Examples
    --------
    >>> r = [1.0, 1.0, 1.0]
    >>> v = [0.0, 0.0, 0.0]
    >>> compute_gae(r, v, v, [False, False, True], [False, False, False], gamma=0.5, lam=1.0)
    (array([1.75, 1.5 , 1.  ]), array([1.75, 1.5 , 1.  ]))

    Truncation at step 1: the TD error bootstraps, the recursion stops.

    >>> adv, ret = compute_gae(r, [0.5, 0.5, 0.5], [0.5, 0.5, 2.0], [False, False, False],
    ...                        [False, True, False], gamma=0.9, lam=0.5)
    >>> adv
    array([1.3775, 0.95  , 2.3   ])
    """
    rewards = _floats(rewards, "rewards")
    arrays = {"rewards": rewards, "values": _floats(values, "values"),
              "next_values": _floats(next_values, "next_values"),
              "terminated": _flags(terminated, "terminated"), "truncated": _flags(truncated, "truncated")}
    shape = _same_shape(arrays)
    if len(shape) not in (1, 2):
        raise ValueError(f"rewards must have shape (T,) or (T, n_envs), got {shape}")
    gamma = _unit_interval(gamma, "gamma")
    lam = _unit_interval(lam, "lam")
    values, next_values = arrays["values"], arrays["next_values"]
    terminated, truncated = arrays["terminated"], arrays["truncated"]
    # TD errors of all steps at once; no bootstrap after a true end (np.where: 0 * inf would give NaN)
    deltas = rewards + gamma * np.where(terminated, 0.0, next_values) - values
    done = terminated | truncated
    advantages = np.zeros_like(rewards)
    future = np.zeros(shape[1:])                       # A_T = 0 after the last step (one per environment)
    for t in range(shape[0] - 1, -1, -1):              # backwards: A_t = delta_t + gamma lam (1 - done_t) A_(t+1)
        future = deltas[t] + gamma * lam * np.where(done[t], 0.0, future)
        advantages[t] = future
    return advantages, advantages + values


def explained_variance(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Explained variance ``1 - Var(y_true - y_pred) / Var(y_true)`` of a critic.

    1 = perfect critic, 0 = no better than a constant, negative = worse than a constant.
    Careful: the prediction comes first, as in CleanRL (scikit-learn uses the opposite
    order).

    Parameters
    ----------
    y_pred : array-like of shape (n,)
        Values predicted by the critic.
    y_true : array-like of shape (n,)
        Empirical returns.

    Returns
    -------
    float
        Explained variance, <= 1; ``nan`` when ``Var(y_true) == 0`` (as in CleanRL).

    Raises
    ------
    ValueError
        If the shapes differ.

    Notes
    -----
    Tested against ``sklearn.metrics.explained_variance_score(y_true, y_pred)`` when
    ``Var(y_true) > 0``.

    Examples
    --------
    >>> round(explained_variance([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]), 4)
    0.8571
    >>> explained_variance([1.0, 2.0], [3.0, 3.0])
    nan
    """
    y_pred = _floats(y_pred, "y_pred")
    y_true = _floats(y_true, "y_true")
    _vector({"y_pred": y_pred, "y_true": y_true})
    var_y = np.var(y_true)                             # ddof=0, as CleanRL and scikit-learn
    if var_y == 0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def clipped_surrogate(
    ratio: ArrayLike, advantages: ArrayLike, clip_eps: float = 0.2
) -> tuple[float, np.ndarray]:
    """PPO clipped loss and its gradient with respect to the probability ratios.

    ``loss = -mean(min(r A, clip(r, 1 - eps, 1 + eps) A))``. Its gradient with respect to
    ``r_i`` is ``-A_i / n`` where the unclipped term is the minimum
    (``r_i A_i <= clip(r_i) A_i``) and 0 where the clipped term is strictly smaller:
    there the ratio has already moved far enough, so PPO stops pushing it.

    Parameters
    ----------
    ratio : array-like of shape (n,)
        Probability ratios ``pi_new(a|s) / pi_old(a|s)``, >= 0.
    advantages : array-like of shape (n,)
        Advantage estimates ``A``.
    clip_eps : float, default=0.2
        Clipping range ``eps`` > 0.

    Returns
    -------
    loss : float
        The loss to minimise.
    grad : np.ndarray of shape (n,)
        Gradient of the loss with respect to each ratio.

    Raises
    ------
    ValueError
        If ``clip_eps <= 0``, a ratio is negative, or the shapes differ.

    Notes
    -----
    Tested against torch autograd on the same expression (``torch.min``,
    ``torch.clamp``).

    Examples
    --------
    >>> loss, grad = clipped_surrogate([0.5, 1.0, 1.5], [1.0, 1.0, 1.0])
    >>> round(loss, 4)  # -(0.5 + 1.0 + 1.2) / 3
    -0.9
    >>> grad  # the third ratio is clipped: no gradient
    array([-0.33333333, -0.33333333,  0.        ])
    """
    ratio = _floats(ratio, "ratio")
    advantages = _floats(advantages, "advantages")
    n = _vector({"ratio": ratio, "advantages": advantages})
    eps = _positive(clip_eps, "clip_eps")
    if (ratio < 0).any():
        raise ValueError(f"ratio must be >= 0 (a ratio of probabilities), got {ratio[ratio < 0][0]}")
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages
    loss = -float(np.mean(np.minimum(unclipped, clipped)))
    # the unclipped term is the minimum (ties included): gradient -A/n; the clipped term strictly smaller: 0
    grad = np.where(unclipped <= clipped, -advantages / n, 0.0)
    return loss, grad


def bradley_terry_loss(
    r_chosen: ArrayLike, r_rejected: ArrayLike
) -> tuple[float, np.ndarray, np.ndarray]:
    """Reward-model loss ``mean(-log sigmoid(r_chosen - r_rejected))`` and its gradients.

    Bradley-Terry model: the probability that the chosen item is preferred is
    ``sigmoid(r_chosen - r_rejected)``. Computed stably with ``np.logaddexp``
    (``-log sigmoid(d) = logaddexp(0, -d)``), so it stays finite for differences of
    +-1000.

    Parameters
    ----------
    r_chosen : array-like of shape (n,)
        Rewards of the preferred items.
    r_rejected : array-like of shape (n,)
        Rewards of the other items of each pair.

    Returns
    -------
    loss : float
        Mean loss over the ``n`` pairs.
    grad_chosen : np.ndarray of shape (n,)
        Gradient of the loss with respect to ``r_chosen`` (<= 0).
    grad_rejected : np.ndarray of shape (n,)
        Gradient of the loss with respect to ``r_rejected`` (= ``-grad_chosen``).

    Raises
    ------
    ValueError
        If the shapes differ.

    Notes
    -----
    Tested against torch: ``-F.logsigmoid(r_chosen - r_rejected).mean()`` and autograd.

    Examples
    --------
    >>> bradley_terry_loss([0.0], [0.0])  # no preference yet: loss = log 2
    (0.6931471805599453, array([-0.5]), array([0.5]))
    >>> bradley_terry_loss([-1000.0], [1000.0])[0]
    2000.0
    """
    r_chosen = _floats(r_chosen, "r_chosen")
    r_rejected = _floats(r_rejected, "r_rejected")
    n = _vector({"r_chosen": r_chosen, "r_rejected": r_rejected})
    d = r_chosen - r_rejected
    loss = float(np.mean(np.logaddexp(0.0, -d)))      # -log sigmoid(d), finite for d = -1000
    grad_chosen = -_sigmoid_of_minus(d) / n            # d/dd [-log sigmoid(d)] = -sigmoid(-d)
    return loss, grad_chosen, -grad_chosen


def dpo_loss(
    policy_chosen_logps: ArrayLike,
    policy_rejected_logps: ArrayLike,
    ref_chosen_logps: ArrayLike,
    ref_rejected_logps: ArrayLike,
    beta: float = 0.1,
) -> tuple[float, np.ndarray]:
    """Direct Preference Optimization (DPO) loss of Rafailov et al. (2023), equation 7.

    ``margins = beta ((log pi_c - log ref_c) - (log pi_r - log ref_r))`` and
    ``loss = mean(-log sigmoid(margins))``: the Bradley-Terry loss applied to the implicit
    rewards ``beta log(pi / pi_ref)``.

    Parameters
    ----------
    policy_chosen_logps : array-like of shape (n,)
        Log-probabilities of the chosen answers (whole sequences) under the trained policy.
    policy_rejected_logps : array-like of shape (n,)
        Log-probabilities of the rejected answers under the trained policy.
    ref_chosen_logps : array-like of shape (n,)
        Log-probabilities of the chosen answers under the frozen reference model.
    ref_rejected_logps : array-like of shape (n,)
        Log-probabilities of the rejected answers under the reference model.
    beta : float, default=0.1
        Strength of the implicit KL constraint (> 0).

    Returns
    -------
    loss : float
        Mean DPO loss.
    margins : np.ndarray of shape (n,)
        The margins above (positive = the policy already prefers the chosen answer more
        than the reference does).

    Raises
    ------
    ValueError
        If ``beta <= 0`` or the shapes differ.

    Notes
    -----
    Tested against equation 7 of Rafailov et al. (2023), and with the property that it
    equals ``bradley_terry_loss`` applied to the implicit rewards; ``trl.DPOTrainer`` if
    trl is installed.

    Examples
    --------
    >>> loss, margins = dpo_loss([-10.0, -12.0], [-11.0, -9.0], [-10.5, -11.0],
    ...                          [-10.5, -10.0], beta=0.1)
    >>> margins
    array([ 0.1, -0.2])
    >>> round(loss, 4)
    0.7213
    """
    arrays = {name: _floats(value, name) for name, value in (
        ("policy_chosen_logps", policy_chosen_logps), ("policy_rejected_logps", policy_rejected_logps),
        ("ref_chosen_logps", ref_chosen_logps), ("ref_rejected_logps", ref_rejected_logps))}
    _vector(arrays)
    beta = _positive(beta, "beta")
    chosen = arrays["policy_chosen_logps"] - arrays["ref_chosen_logps"]       # log(pi / pi_ref), chosen
    rejected = arrays["policy_rejected_logps"] - arrays["ref_rejected_logps"]
    margins = beta * (chosen - rejected)
    loss = float(np.mean(np.logaddexp(0.0, -margins)))
    return loss, margins
