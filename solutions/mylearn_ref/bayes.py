"""Bayes' rule on a finite set of hypotheses — mylearn, chapter 4 (Bayes' rule).

Bayes' rule for a finite list of hypotheses (evidence and posterior), the
posterior-prior loop that updates beliefs one observation at a time, the bias of a
coin estimated on a grid of candidate values with log-probabilities (no underflow
after thousands of flips), and the credible interval of a discrete posterior. Reused
in chapter 9 (Bayesian fit of a line) and chapter 13 (Naive Bayes).

Reference implementation: read it only after trying (``mon_travail/mylearn/bayes.py``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

_TOL = 1e-8   # a distribution must sum to 1 within this tolerance


def _distribution(p: ArrayLike, name: str) -> np.ndarray:
    """A non-empty 1-D float array of probabilities: non-negative, summing to 1 (within _TOL)."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {arr.shape}")
    if np.isnan(arr).any() or (arr < 0).any():
        raise ValueError(f"{name} must contain non-negative probabilities")
    if abs(arr.sum() - 1.0) > _TOL:
        raise ValueError(f"{name} must sum to 1, got {arr.sum()!r}")
    return arr


def _probabilities(p: ArrayLike, name: str) -> np.ndarray:
    """A float array whose values are all in [0, 1] (they need not sum to 1)."""
    arr = np.asarray(p, dtype=float)
    if np.isnan(arr).any() or (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"every value of {name} must be a probability in [0, 1]")
    return arr


def _outcome_indices(observations: ArrayLike, n_outcomes: int) -> np.ndarray:
    """A 1-D integer array of outcome indices, each in [0, n_outcomes)."""
    obs = np.asarray(observations)
    if obs.ndim != 1:
        raise ValueError(f"observations must be a 1-D sequence of outcome indices, got shape {obs.shape}")
    if obs.size == 0:
        return obs.astype(int)
    if obs.dtype.kind == "b":
        obs = obs.astype(int)
    elif obs.dtype.kind == "f":
        if not np.isfinite(obs).all() or (obs != np.round(obs)).any():
            raise ValueError("observations must be integer indices (1.5 is not an outcome)")
        obs = obs.astype(int)
    elif obs.dtype.kind not in "iu":
        raise ValueError(f"observations must be integer indices, got dtype {obs.dtype}")
    if (obs < 0).any() or (obs >= n_outcomes).any():
        raise ValueError(f"every observation must be an index in [0, {n_outcomes}), "
                         f"got values from {obs.min()} to {obs.max()}")
    return obs


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
    prior = _distribution(prior, "prior")
    likelihood = _probabilities(likelihood, "likelihood")
    if likelihood.shape != prior.shape:
        raise ValueError(f"prior and likelihood must have the same shape, got {prior.shape} and {likelihood.shape}")
    return float(np.dot(prior, likelihood))   # all the ways the observation can happen


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
    total = evidence(prior, likelihood)          # also checks both inputs
    if total == 0:
        raise ValueError("the evidence is 0: the observation is impossible under every hypothesis")
    return np.asarray(prior, dtype=float) * np.asarray(likelihood, dtype=float) / total


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
    posterior = _distribution(prior, "prior")
    table = _probabilities(likelihoods, "likelihoods")
    if table.ndim != 2 or table.shape[0] != posterior.size:
        raise ValueError(f"likelihoods must have shape (n_hypotheses, n_outcomes) with n_hypotheses = "
                         f"{posterior.size}, got shape {table.shape}")
    obs = _outcome_indices(observations, table.shape[1])
    history = [posterior]
    for outcome in obs:
        # the column of the observed outcome holds P(outcome | H_i); the posterior becomes the next prior
        posterior = bayes_posterior(posterior, table[:, outcome])
        history.append(posterior)
    return np.array(history) if return_history else posterior


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
    flips = np.asarray(flips)
    if flips.ndim != 1:
        raise ValueError(f"flips must be a 1-D sequence of 0 and 1, got shape {flips.shape}")
    if flips.size and not np.isin(flips, [0, 1]).all():
        raise ValueError("flips must contain only 0 (tails) and 1 (heads)")
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f"grid must be a non-empty 1-D array, got shape {grid.shape}")
    if np.isnan(grid).any() or (grid < 0).any() or (grid > 1).any():
        raise ValueError("every grid value must be a bias in [0, 1]")
    if prior is None:
        prior = np.full(grid.size, 1.0 / grid.size)
    else:
        prior = _distribution(prior, "prior")
        if prior.shape != grid.shape:
            raise ValueError(f"prior and grid must have the same shape, got {prior.shape} and {grid.shape}")
    heads = int(np.sum(flips == 1))
    tails = int(flips.size - heads)
    with np.errstate(divide="ignore"):            # log(0) = -inf is wanted here
        log_post = np.log(prior)                   # -inf where the prior is 0
        if heads:                                  # theta ** 0 = 1, even for theta = 0
            log_post = log_post + heads * np.log(grid)
        if tails:
            log_post = log_post + tails * np.log1p(-grid)
    if np.isneginf(log_post).all():
        raise ValueError("every hypothesis gets probability 0 after these flips")
    weights = np.exp(log_post - log_post.max())   # the largest becomes exp(0) = 1: no underflow
    return weights / weights.sum()


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
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f"grid must be a non-empty 1-D array, got shape {grid.shape}")
    if np.isnan(grid).any() or not (np.diff(grid) > 0).all():
        raise ValueError("grid must be strictly increasing")
    posterior = _distribution(posterior, "posterior")
    if posterior.shape != grid.shape:
        raise ValueError(f"grid and posterior must have the same shape, got {grid.shape} and {posterior.shape}")
    if not 0 < mass < 1:
        raise ValueError(f"mass must be strictly between 0 and 1, got {mass!r}")
    cdf = np.cumsum(posterior)                     # non-decreasing: searchsorted finds the first index
    last = grid.size - 1                           # that reaches each level (side="left": cdf >= level)
    low = min(int(np.searchsorted(cdf, (1 - mass) / 2, side="left")), last)
    high = min(int(np.searchsorted(cdf, (1 + mass) / 2, side="left")), last)
    return float(grid[low]), float(grid[high])
