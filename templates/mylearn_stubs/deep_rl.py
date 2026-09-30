"""Deep RL computations — mylearn, chapter B8 (Modern RL: DQN, policy gradient, PPO and RLHF).

The parts of modern deep RL that can be tested without a network or an environment, in
NumPy: generalized advantage estimation (with the terminated / truncated distinction),
the clipped PPO objective and its gradient, the explained variance of a critic, and the
preference losses of RLHF (Bradley-Terry reward model, DPO). DQN, REINFORCE and PPO
themselves are written in PyTorch in the notebook, with ``rl.py`` from chapter 26.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


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
    # TODO: TD errors for all steps at once, then one backward loop over time.
    raise NotImplementedError("compute_gae() is not implemented yet")


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
    # TODO: two variances (np.var), with the special case Var(y_true) == 0.
    raise NotImplementedError("explained_variance() is not implemented yet")


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
    # TODO: both terms, their minimum, then the gradient with np.where.
    raise NotImplementedError("clipped_surrogate() is not implemented yet")


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
    # TODO: the loss with np.logaddexp, the gradients with a stable sigmoid.
    raise NotImplementedError("bradley_terry_loss() is not implemented yet")


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
    # TODO: the margins, then the same stable formula as in bradley_terry_loss.
    raise NotImplementedError("dpo_loss() is not implemented yet")
