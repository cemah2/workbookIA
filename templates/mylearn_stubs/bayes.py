"""Bayes' rule on a finite set of hypotheses — mylearn, chapter 4 (Bayes' rule).

Bayes' rule for a finite list of hypotheses (evidence and posterior), the
posterior-prior loop that updates beliefs one observation at a time, the bias of a
coin estimated on a grid of candidate values with log-probabilities (no underflow
after thousands of flips), and the credible interval of a discrete posterior. Reused
in chapter 9 (Bayesian fit of a line) and chapter 13 (Naive Bayes).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def evidence(prior: ArrayLike, likelihood: ArrayLike) -> float:
    """Compute the evidence: the probability of the observation over all hypotheses.

    ``P(obs) = Σ_i P(obs | H_i) P(H_i)``: the sum of all the ways the observation
    can happen, one per hypothesis. It is the denominator of Bayes' rule.

    Parameters
    ----------
    prior : array-like of shape (n_hypotheses,)
        Prior probabilities ``P(H_i)``: non-negative, summing to 1 (tolerance 1e-8).
    likelihood : array-like of shape (n_hypotheses,)
        Likelihoods ``P(obs | H_i)``, each in [0, 1]; they need not sum to 1.

    Returns
    -------
    float
        The evidence, in [0, 1].

    Raises
    ------
    ValueError
        If the shapes differ, if ``prior`` is not a probability distribution, or if
        a likelihood is outside [0, 1].

    Notes
    -----
    Tested against an exact computation with ``fractions.Fraction`` on small cases
    and against ``np.dot(prior, likelihood)``.

    Examples
    --------
    A fair coin or a two-headed coin, equally likely; we observe heads:

    >>> evidence([0.5, 0.5], [0.5, 1.0])
    0.75
    """
    # TODO: validate both inputs, then add up likelihood times prior.
    raise NotImplementedError("evidence() is not implemented yet")


def bayes_posterior(prior: ArrayLike, likelihood: ArrayLike) -> np.ndarray:
    """Apply Bayes' rule to a finite set of hypotheses.

    ``P(H_i | obs) = P(obs | H_i) P(H_i) / P(obs)``, where ``P(obs)`` is the
    ``evidence``. A hypothesis with prior 0 keeps posterior 0; a likelihood that is
    the same for every hypothesis leaves the prior unchanged.

    Parameters
    ----------
    prior : array-like of shape (n_hypotheses,)
        Prior probabilities ``P(H_i)``: non-negative, summing to 1 (tolerance 1e-8).
    likelihood : array-like of shape (n_hypotheses,)
        Likelihoods ``P(obs | H_i)``, each in [0, 1].

    Returns
    -------
    np.ndarray of shape (n_hypotheses,)
        Posterior probabilities, summing to 1.

    Raises
    ------
    ValueError
        Same cases as ``evidence``, and if the evidence is 0 (the observation is
        impossible under every hypothesis).

    Notes
    -----
    Tested against the book's values 3/7 (heads) and 3/5 (tails) computed with
    ``fractions.Fraction``, plus properties: sums to 1, a zero prior stays zero, a
    uniform likelihood leaves the prior unchanged.

    Examples
    --------
    A fair coin or a coin that gives heads 2 times out of 3, equally likely:

    >>> bayes_posterior([0.5, 0.5], [0.5, 2 / 3])  # heads observed
    array([0.42857143, 0.57142857])
    >>> bayes_posterior([0.5, 0.5], [0.5, 1 / 3])  # tails observed
    array([0.6, 0.4])
    """
    # TODO: reuse evidence, refuse an evidence of 0, then normalize the products.
    raise NotImplementedError("bayes_posterior() is not implemented yet")


def update_discrete(
    prior: ArrayLike,
    likelihoods: ArrayLike,
    observations: ArrayLike,
    return_history: bool = False,
) -> np.ndarray:
    """Update the beliefs one observation at a time (the posterior-prior loop).

    Bayes' rule is applied once per observation, and each posterior becomes the
    prior of the next step. The distribution is normalized at every step, so it
    never underflows, even after thousands of observations.

    Parameters
    ----------
    prior : array-like of shape (n_hypotheses,)
        Prior probabilities: non-negative, summing to 1 (tolerance 1e-8).
    likelihoods : array-like of shape (n_hypotheses, n_outcomes)
        ``likelihoods[i, o] = P(outcome o | H_i)``, each in [0, 1].
    observations : array-like of shape (n_observations,)
        Observed outcome indices in ``[0, n_outcomes)``, e.g. 1 = heads, 0 = tails.
    return_history : bool, default=False
        If True, also keep every intermediate posterior.

    Returns
    -------
    np.ndarray
        The final posterior, of shape (n_hypotheses,); or, if ``return_history``,
        an array of shape (n_observations + 1, n_hypotheses) whose row 0 is the
        prior and row k the posterior after the first k observations.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, if an observation is not an integer index
        in ``[0, n_outcomes)``, if an observation has evidence 0, or in the cases
        of ``bayes_posterior``.

    Notes
    -----
    Tested against the batch formula in log space (log prior plus the sum of the
    log-likelihoods, normalized with ``scipy.special.logsumexp``), plus order
    invariance: shuffled observations give the same final posterior.

    Examples
    --------
    >>> likelihoods = [[0.5, 0.5], [1 / 3, 2 / 3]]  # columns: tails, heads
    >>> update_discrete([0.5, 0.5], likelihoods, [1, 1, 0])
    array([0.45762712, 0.54237288])
    >>> update_discrete([0.5, 0.5], likelihoods, [1, 1, 0], return_history=True)
    array([[0.5       , 0.5       ],
           [0.42857143, 0.57142857],
           [0.36      , 0.64      ],
           [0.45762712, 0.54237288]])
    """
    # TODO: validate the shapes and the observations, then loop: the column of the
    # observed outcome is the likelihood, the posterior becomes the next prior.
    raise NotImplementedError("update_discrete() is not implemented yet")


def coin_bias_posterior(
    flips: ArrayLike, grid: ArrayLike, prior: ArrayLike | None = None
) -> np.ndarray:
    """Compute the posterior over candidate coin biases after the observed flips.

    Each grid value θ is a hypothesis "the coin gives heads with probability θ".
    After h heads and t tails, the posterior is proportional to
    ``prior(θ) · θ^h · (1 - θ)^t``. The computation is done with logarithms, so that
    thousands of flips do not underflow to 0, and the result is normalized at the
    end. A grid value of 0 (or 1) is ruled out as soon as one head (or one tail) is
    observed; with no head (no tail), the factor ``θ^0 = 1`` applies as usual.

    Parameters
    ----------
    flips : array-like of shape (n_flips,)
        Observed flips: 0 (tails) or 1 (heads).
    grid : array-like of shape (n,)
        Candidate biases, each in [0, 1].
    prior : array-like of shape (n,) or None, default=None
        Prior probabilities on the grid (non-negative, summing to 1, tolerance
        1e-8); uniform if None.

    Returns
    -------
    np.ndarray of shape (n,)
        Posterior probabilities on the grid, summing to 1.

    Raises
    ------
    ValueError
        If ``flips`` contains values other than 0 and 1, if a grid value is outside
        [0, 1], if ``prior`` is invalid (wrong shape or not a distribution), or if
        every hypothesis gets probability 0.

    Notes
    -----
    Tested against ``scipy.stats.beta(heads + 1, tails + 1).pdf(grid)`` normalized
    on the grid (uniform prior), and against ``update_discrete`` on short sequences.

    Examples
    --------
    >>> grid = np.linspace(0, 1, 5)
    >>> grid
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> coin_bias_posterior([1, 1, 0], grid)
    array([0.  , 0.15, 0.4 , 0.45, 0.  ])
    """
    # TODO: count heads and tails, build the log-posterior (mind log 0), then go
    # back to probabilities without underflow and normalize.
    raise NotImplementedError("coin_bias_posterior() is not implemented yet")


def credible_interval(
    grid: ArrayLike, posterior: ArrayLike, mass: float = 0.95
) -> tuple[float, float]:
    """Compute the equal-tailed credible interval of a discrete posterior.

    The interval leaves ``(1 - mass) / 2`` of the probability outside on each side:
    the parameter lies inside with probability (about) ``mass``, given the data.

    Parameters
    ----------
    grid : array-like of shape (n,)
        Parameter values, strictly increasing.
    posterior : array-like of shape (n,)
        Posterior probabilities on the grid: non-negative, summing to 1 (tolerance
        1e-8).
    mass : float, default=0.95
        Probability inside the interval, strictly between 0 and 1.

    Returns
    -------
    tuple of (float, float)
        ``(low, high)`` as Python floats: ``low`` is the first grid value whose
        cumulative probability reaches ``(1 - mass) / 2``, ``high`` the first grid
        value whose cumulative probability reaches ``(1 + mass) / 2``.

    Raises
    ------
    ValueError
        If ``grid`` is not strictly increasing, if the shapes differ, if
        ``posterior`` is not a distribution or if ``mass`` is not in (0, 1).

    Notes
    -----
    Tested against ``scipy.stats.beta(h + 1, t + 1).ppf([(1 - mass) / 2,
    (1 + mass) / 2])`` within one grid step (uniform prior, fine grid).

    Examples
    --------
    >>> grid = np.linspace(0, 1, 5)
    >>> posterior = [0.0, 0.15, 0.4, 0.45, 0.0]
    >>> credible_interval(grid, posterior)
    (0.25, 0.75)
    >>> credible_interval(grid, posterior, mass=0.5)
    (0.5, 0.75)
    """
    # TODO: validate the inputs, accumulate the probabilities, then find the first
    # grid value that reaches each of the two levels.
    raise NotImplementedError("credible_interval() is not implemented yet")
