"""Statistics and resampling — mylearn, chapter 2 (Randomness and basic statistics).

Descriptive statistics (central tendency, spread, percentiles, histogram, covariance
and correlation), reproducible random sampling and the bootstrap, written with NumPy.
Reused by preprocessing (chapter 12: mean and std), by ensembles (chapter 14: sample)
and in the checkpoints. Every function refuses NaN: clean the data first (e.g. with
``dropna``). ``ddof=0`` is the default everywhere, including where ``np.cov`` and
pandas use 1.

Reference implementation: read it only after trying (``mon_travail/mylearn/stats.py``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


def _numbers(x: ArrayLike, name: str = "x") -> np.ndarray:
    """Convert to a float array; refuse an empty input and NaN."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN: remove the missing values first (e.g. with dropna)")
    return arr


def _count(arr: np.ndarray, axis: int | None) -> int:
    """Number of values reduced: all of them, or the length of the axis."""
    return arr.size if axis is None else arr.shape[axis]


def _pair(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Two 1-D float arrays of the same length (no NaN)."""
    xa, ya = _numbers(x, "x"), _numbers(y, "y")
    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError(f"x and y must be 1-D, got shapes {xa.shape} and {ya.shape}")
    if len(xa) != len(ya):
        raise ValueError(f"x and y have different lengths: {len(xa)} and {len(ya)}")
    return xa, ya


def _matrix(X: ArrayLike) -> np.ndarray:
    """A 2-D float array (no NaN)."""
    A = _numbers(X, "X")
    if A.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {A.shape}")
    return A


def mean(x: ArrayLike, axis: int | None = None) -> float | np.ndarray:
    """Compute the arithmetic mean: the sum of the values divided by their count.

    Parameters
    ----------
    x : array-like of any shape
        Numbers (no NaN).
    axis : int or None, default=None
        None to average all the values, otherwise the axis to reduce (NumPy
        convention: ``axis=0`` gives one mean per column of a 2-D array).

    Returns
    -------
    float or np.ndarray
        A Python float when ``axis`` is None, otherwise a float array with the
        shape of ``x`` without that axis.

    Raises
    ------
    ValueError
        If ``x`` (or the reduced axis) is empty, or if ``x`` contains NaN (the
        message suggests removing missing values first, e.g. with ``dropna``).

    Notes
    -----
    Tested against ``np.mean``.

    Examples
    --------
    >>> mean([1, 2, 3, 4])
    2.5
    >>> mean([[1, 2], [3, 4]], axis=0)
    array([2., 3.])
    """
    arr = _numbers(x)
    n = _count(arr, axis)
    if n == 0:
        raise ValueError("cannot average an empty axis")
    total = arr.sum(axis=axis)
    return float(total / n) if axis is None else total / n


def median(x: ArrayLike, axis: int | None = None) -> float | np.ndarray:
    """Compute the median: the middle value of the sorted data.

    When the count is even, the median is the mean of the two middle values.

    Parameters
    ----------
    x : array-like of any shape
        Numbers (no NaN).
    axis : int or None, default=None
        None to use all the values, otherwise the axis to reduce.

    Returns
    -------
    float or np.ndarray
        A Python float when ``axis`` is None, otherwise a float array with the
        shape of ``x`` without that axis.

    Raises
    ------
    ValueError
        If ``x`` (or the reduced axis) is empty, or if ``x`` contains NaN.

    Notes
    -----
    Tested against ``np.median``.

    Examples
    --------
    >>> median([3, 1, 4, 1, 5, 9])
    3.5
    >>> median([[1, 5, 2], [8, 3, 4]], axis=1)
    array([2., 4.])
    """
    arr = _numbers(x)
    if axis is None:
        values = np.sort(arr, axis=None)
    else:
        values = np.moveaxis(np.sort(arr, axis=axis), axis, 0)   # the reduced axis comes first
    n = values.shape[0]
    middle = n // 2
    if n % 2 == 1:
        result = values[middle]
    else:
        result = (values[middle - 1] + values[middle]) / 2
    return float(result) if axis is None else np.array(result, dtype=float)


def mode(x: ArrayLike) -> np.ndarray:
    """Return all the values that occur most often, sorted in ascending order.

    Ties are all returned; if every value occurs the same number of times, every
    value is returned (the book would say there is "no mode").

    Parameters
    ----------
    x : array-like of shape (n,)
        Hashable values: numbers or strings.

    Returns
    -------
    np.ndarray of shape (n_modes,)
        The most frequent values, sorted in ascending order.

    Raises
    ------
    ValueError
        If ``x`` is empty or not 1-D.

    Notes
    -----
    Tested against ``sorted(statistics.multimode(x))`` and
    ``pandas.Series(x).mode()``.

    Examples
    --------
    >>> mode([1, 2, 2, 3, 3])
    array([2, 3])
    >>> mode(["b", "a", "b"])
    array(['b'], dtype='<U1')
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("x is empty")
    values, counts = np.unique(arr, return_counts=True)   # sorted values and their counts
    return values[counts == counts.max()]


def variance(x: ArrayLike, ddof: int = 0, axis: int | None = None) -> float | np.ndarray:
    """Compute the variance: squared deviations from the mean, summed, over n - ddof.

    ``ddof=0`` gives the population variance (divide by n); ``ddof=1`` gives the
    sample estimate (divide by n - 1).

    Parameters
    ----------
    x : array-like of any shape
        Numbers (no NaN).
    ddof : int, default=0
        Delta degrees of freedom: the divisor is ``n - ddof`` (0 or 1 in practice).
    axis : int or None, default=None
        None to use all the values, otherwise the axis to reduce.

    Returns
    -------
    float or np.ndarray
        A Python float when ``axis`` is None, otherwise a float array with the
        shape of ``x`` without that axis.

    Raises
    ------
    ValueError
        If ``x`` is empty or contains NaN, or if ``n - ddof <= 0`` (``n`` = number
        of values reduced).

    Notes
    -----
    Tested against ``np.var(x, ddof=ddof, axis=axis)``.

    Examples
    --------
    >>> variance([1, 2, 3, 4])
    1.25
    >>> variance([1, 2, 3, 4], ddof=1)
    1.6666666666666667
    """
    arr = _numbers(x)
    n = _count(arr, axis)
    if n - ddof <= 0:
        raise ValueError(f"n - ddof must be positive (n = {n}, ddof = {ddof})")
    deviations = arr - arr.sum(axis=axis, keepdims=True) / n
    result = (deviations ** 2).sum(axis=axis) / (n - ddof)
    return float(result) if axis is None else result


def std(x: ArrayLike, ddof: int = 0, axis: int | None = None) -> float | np.ndarray:
    """Compute the standard deviation: the square root of the variance.

    Parameters
    ----------
    x : array-like of any shape
        Numbers (no NaN).
    ddof : int, default=0
        As in ``variance``.
    axis : int or None, default=None
        None to use all the values, otherwise the axis to reduce.

    Returns
    -------
    float or np.ndarray
        A Python float when ``axis`` is None, otherwise a float array with the
        shape of ``x`` without that axis.

    Raises
    ------
    ValueError
        Same cases as ``variance``.

    Notes
    -----
    Tested against ``np.std(x, ddof=ddof, axis=axis)``.

    Examples
    --------
    >>> std([2, 4, 4, 4, 5, 5, 7, 9])
    2.0
    """
    result = np.sqrt(variance(x, ddof=ddof, axis=axis))
    return float(result) if axis is None else result


def percentile(
    x: ArrayLike, q: float | ArrayLike, axis: int | None = None
) -> float | np.ndarray:
    """Compute the q-th percentile(s) with linear interpolation.

    The sorted values are placed at positions 0, 1, ..., n - 1; the q-th percentile
    sits at position ``q / 100 * (n - 1)``, interpolated linearly between the two
    neighbouring values (NumPy's default ``'linear'`` method).

    Parameters
    ----------
    x : array-like of any shape
        Numbers (no NaN).
    q : float or array-like of float
        Percentile(s), between 0 and 100.
    axis : int or None, default=None
        None to use all the values, otherwise the axis to reduce.

    Returns
    -------
    float or np.ndarray
        A Python float for a scalar ``q`` and ``axis=None``; otherwise a float
        array laid out like NumPy's: shape ``q.shape + (shape of x without the
        axis)``.

    Raises
    ------
    ValueError
        If a ``q`` is outside [0, 100], or if ``x`` is empty or contains NaN.

    Notes
    -----
    Tested against ``np.percentile(x, q, axis=axis, method='linear')``.

    Examples
    --------
    >>> percentile([1, 2, 3, 4], 50)
    2.5
    >>> percentile([1, 2, 3, 4], [25, 75])
    array([1.75, 3.25])
    """
    arr = _numbers(x)
    q_arr = np.asarray(q, dtype=float)
    if np.isnan(q_arr).any() or np.any(q_arr < 0) or np.any(q_arr > 100):
        raise ValueError("percentiles must be between 0 and 100")
    if axis is None:
        values = np.sort(arr, axis=None)
    else:
        values = np.moveaxis(np.sort(arr, axis=axis), axis, 0)
    n = values.shape[0]
    position = q_arr / 100 * (n - 1)                      # fractional position in the sorted values
    below = np.floor(position).astype(int)
    above = np.minimum(below + 1, n - 1)
    fraction = (position - below).reshape(q_arr.shape + (1,) * (values.ndim - 1))
    low, high = values[below], values[above]
    result = low + fraction * (high - low)
    return float(result) if result.ndim == 0 else result


def zscore(x: ArrayLike, ddof: int = 0, axis: int | None = None) -> np.ndarray:
    """Standardize the data: (x - mean) / std.

    The result has mean 0 and standard deviation 1, globally (``axis=None``) or
    along the given axis (``axis=0``: per column, i.e. per feature).

    Parameters
    ----------
    x : array-like of any shape
        Numbers (no NaN).
    ddof : int, default=0
        Passed to ``std``.
    axis : int or None, default=None
        None to standardize with the statistics of all the values; ``0`` to use
        the statistics of each column.

    Returns
    -------
    np.ndarray
        Float array with the shape of ``x``.

    Raises
    ------
    ValueError
        If a standard deviation is 0 (constant data, or a constant column when
        ``axis=0``), or if ``x`` is empty or contains NaN.

    Notes
    -----
    Tested against ``scipy.stats.zscore(x, axis=axis, ddof=ddof)`` (always called
    with the same ``axis``: SciPy's default is 0, ours is None).

    Examples
    --------
    >>> zscore([2, 4, 4, 4, 5, 5, 7, 9])
    array([-1.5, -0.5, -0.5, -0.5,  0. ,  0. ,  1. ,  2. ])
    """
    arr = _numbers(x)
    center = mean(arr, axis=axis)
    spread = std(arr, ddof=ddof, axis=axis)
    if np.any(np.asarray(spread) == 0) or np.any(np.ptp(arr, axis=axis) == 0):
        raise ValueError("standard deviation is 0 (constant data): the z-score is undefined")
    if axis is not None:                                  # put the reduced axis back for broadcasting
        center, spread = np.expand_dims(center, axis), np.expand_dims(spread, axis)
    return (arr - center) / spread


def histogram(
    x: ArrayLike,
    bins: int = 10,
    bin_range: tuple[float, float] | None = None,
    density: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Count the values that fall in equal-width bins.

    The interval ``[low, high]`` is cut into ``bins`` bins of equal width. Each bin
    contains its left edge but not its right edge, except the last bin, which also
    contains ``high``. Values outside ``[low, high]`` are ignored.

    Parameters
    ----------
    x : array-like of shape (n,)
        Numbers (no NaN).
    bins : int, default=10
        Number of bins, ``>= 1``.
    bin_range : tuple of (float, float) or None, default=None
        ``(low, high)``, the outer edges. If None, ``(min(x), max(x))``. (Named
        ``bin_range`` so as not to shadow the built-in ``range``.)
    density : bool, default=False
        If True, divide each count by ``(number of counted values * bin width)``,
        so that the bars have a total area of 1.

    Returns
    -------
    counts : np.ndarray of shape (bins,)
        Number of values in each bin (int), or the density (float) if ``density``.
    edges : np.ndarray of shape (bins + 1,)
        Float edges of the bins, from ``low`` to ``high``.

    Raises
    ------
    ValueError
        If ``bins < 1``, if ``low >= high`` (this includes ``bin_range=None`` with
        constant data: pass an explicit ``bin_range`` then), or if ``x`` is empty or
        contains NaN.

    Notes
    -----
    Tested against ``np.histogram(x, bins=bins, range=bin_range, density=density)``.

    Examples
    --------
    >>> counts, edges = histogram([0, 1, 2, 3, 4], bins=4)
    >>> counts
    array([1, 1, 1, 2])
    >>> edges
    array([0., 1., 2., 3., 4.])
    >>> histogram([1, 2, 2, 3, 3, 3, 9], bins=2, bin_range=(0, 4))
    (array([1, 5]), array([0., 2., 4.]))
    >>> histogram([1, 2, 2, 3, 3, 3], bins=2, bin_range=(0, 4), density=True)[0]
    array([0.08333333, 0.41666667])
    """
    arr = _numbers(x).ravel()
    if int(bins) != bins or bins < 1:
        raise ValueError(f"bins must be an integer >= 1, got {bins}")
    bins = int(bins)
    low, high = (float(arr.min()), float(arr.max())) if bin_range is None else map(float, bin_range)
    if not low < high:
        raise ValueError(f"the range must satisfy low < high, got ({low}, {high})"
                         + (" (constant data: pass bin_range)" if bin_range is None else ""))
    edges = np.linspace(low, high, bins + 1)
    inside = arr[(arr >= low) & (arr <= high)]            # values outside the range are ignored
    index = np.searchsorted(edges, inside, side="right") - 1
    index[inside == high] = bins - 1                      # the last bin keeps its right edge
    counts = np.bincount(index, minlength=bins)
    if density:
        return counts / (counts.sum() * np.diff(edges)), edges
    return counts, edges


def covariance(x: ArrayLike, y: ArrayLike, ddof: int = 0) -> float:
    """Compute the covariance of two variables.

    Average product of the deviations of ``x`` and ``y`` from their means (the sum
    of the products is divided by ``n - ddof``). Positive when ``x`` and ``y`` tend
    to move together, negative when one goes up as the other goes down.
    ``covariance(x, x) == variance(x)`` for the same ``ddof``.

    Parameters
    ----------
    x : array-like of shape (n,)
        Numbers (no NaN).
    y : array-like of shape (n,)
        Numbers (no NaN), same length as ``x``.
    ddof : int, default=0
        Delta degrees of freedom, 0 (default, like ``variance``) or 1.

    Returns
    -------
    float
        The covariance.

    Raises
    ------
    ValueError
        If ``x`` or ``y`` is not 1-D, if their lengths differ, if ``n - ddof <= 0``
        or if they contain NaN.

    Notes
    -----
    Tested against ``np.cov(x, y, ddof=ddof)[0, 1]``.

    Examples
    --------
    >>> covariance([1, 2, 3, 4], [2, 4, 6, 8])
    2.5
    >>> round(covariance([1, 2, 3, 4], [2, 4, 6, 8], ddof=1), 4)
    3.3333
    """
    xa, ya = _pair(x, y)
    n = len(xa)
    if n - ddof <= 0:
        raise ValueError(f"n - ddof must be positive (n = {n}, ddof = {ddof})")
    dx = xa - xa.sum() / n
    dy = ya - ya.sum() / n
    return float((dx * dy).sum() / (n - ddof))


def correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Compute the Pearson correlation coefficient of two variables.

    The covariance divided by both standard deviations: unit-free, between -1
    (perfect decreasing line) and 1 (perfect increasing line); 0 means no linear
    relation.

    Parameters
    ----------
    x : array-like of shape (n,)
        Numbers (no NaN).
    y : array-like of shape (n,)
        Numbers (no NaN), same length as ``x``.

    Returns
    -------
    float
        The correlation, in [-1, 1].

    Raises
    ------
    ValueError
        If the lengths differ, if ``n < 2``, if there is NaN, or if ``x`` or ``y``
        is constant (the correlation is then undefined).

    Notes
    -----
    Tested against ``np.corrcoef(x, y)[0, 1]`` and
    ``scipy.stats.pearsonr(x, y).statistic``.

    Examples
    --------
    >>> round(correlation([1, 2, 3, 4], [1, 3, 2, 4]), 4)
    0.8
    >>> round(correlation([1, 2, 3], [3, 2, 1]), 4)
    -1.0
    """
    xa, ya = _pair(x, y)
    if len(xa) < 2:
        raise ValueError("the correlation needs at least 2 points")
    if np.ptp(xa) == 0 or np.ptp(ya) == 0:
        raise ValueError("x or y is constant: the correlation is undefined")
    r = covariance(xa, ya) / (std(xa) * std(ya))
    return float(np.clip(r, -1.0, 1.0))


def covariance_matrix(X: ArrayLike, ddof: int = 0) -> np.ndarray:
    """Compute the covariances of every pair of columns of X.

    Entry ``(j, k)`` is ``covariance(X[:, j], X[:, k], ddof)``. Rows are samples and
    columns are features (unlike ``np.cov``, whose default is the other way round).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Numbers (no NaN).
    ddof : int, default=0
        As in ``covariance``.

    Returns
    -------
    np.ndarray of shape (n_features, n_features)
        Symmetric matrix whose diagonal holds the variances of the columns.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, if ``n_samples - ddof <= 0`` or if ``X`` contains NaN.

    Notes
    -----
    Tested against ``np.cov(X, rowvar=False, ddof=ddof)``.

    Examples
    --------
    >>> covariance_matrix([[1, 2], [2, 4], [3, 6], [4, 8]])
    array([[1.25, 2.5 ],
           [2.5 , 5.  ]])
    """
    A = _matrix(X)
    n = A.shape[0]
    if n - ddof <= 0:
        raise ValueError(f"n_samples - ddof must be positive (n_samples = {n}, ddof = {ddof})")
    centered = A - A.sum(axis=0) / n
    C = centered.T @ centered / (n - ddof)
    return (C + C.T) / 2                                  # exactly symmetric


def correlation_matrix(X: ArrayLike) -> np.ndarray:
    """Compute the Pearson correlations of every pair of columns of X.

    Entry ``(j, k)`` is ``correlation(X[:, j], X[:, k])``; the diagonal holds ones.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Numbers (no NaN).

    Returns
    -------
    np.ndarray of shape (n_features, n_features)
        Symmetric matrix with values in [-1, 1].

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, has fewer than 2 rows, contains NaN or has a constant
        column.

    Notes
    -----
    Tested against ``np.corrcoef(X, rowvar=False)`` and
    ``pandas.DataFrame(X).corr()``.

    Examples
    --------
    >>> np.round(correlation_matrix([[1, 1], [2, 3], [3, 2], [4, 4]]), 4)
    array([[1. , 0.8],
           [0.8, 1. ]])
    """
    A = _matrix(X)
    if A.shape[0] < 2:
        raise ValueError("the correlation needs at least 2 rows")
    if np.any(np.ptp(A, axis=0) == 0):
        raise ValueError("a column is constant: its correlation is undefined")
    C = covariance_matrix(A)
    spread = np.sqrt(np.diag(C))
    R = np.clip(C / np.outer(spread, spread), -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    return R


def sample(
    population: ArrayLike,
    size: int,
    replace: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw ``size`` elements of a population at random, with or without replacement.

    With replacement, a drawn element is put back and can come out again (copies
    are possible); without replacement, each element comes out at most once.
    Elements are taken along the first axis (rows of a 2-D array).

    Parameters
    ----------
    population : array-like of shape (n, ...)
        The population; its elements are ``population[0]``, ..., ``population[n-1]``.
    size : int
        Number of draws, ``>= 0``.
    replace : bool, default=True
        True: with replacement; False: without replacement.
    rng : np.random.Generator or None, default=None
        Random generator. If None, ``np.random.default_rng()`` (not reproducible).

    Returns
    -------
    np.ndarray of shape (size, ...)
        The drawn elements (shape ``(size,) + population.shape[1:]``).

    Raises
    ------
    ValueError
        If ``size < 0``, if ``population`` is empty, or if ``size > n`` without
        replacement.

    Notes
    -----
    Documented algorithm, tested for exact equality under the same seed:
    ``population[rng.integers(0, n, size)]`` with replacement and
    ``population[rng.permutation(n)[:size]]`` without. Also tested: no repeated
    index without replacement, and a share of distinct elements close to
    ``1 - (1 - 1/n)**n`` with replacement and ``size = n``.

    Examples
    --------
    >>> sample(["a", "b", "c", "d"], 6, rng=np.random.default_rng(0))
    array(['d', 'c', 'c', 'b', 'b', 'a'], dtype='<U1')
    >>> sample(["a", "b", "c", "d"], 3, replace=False, rng=np.random.default_rng(0))
    array(['c', 'a', 'b'], dtype='<U1')
    """
    pop = np.asarray(population)
    if pop.ndim == 0 or pop.shape[0] == 0:
        raise ValueError("the population is empty")
    if int(size) != size or size < 0:
        raise ValueError(f"size must be an integer >= 0, got {size}")
    n = pop.shape[0]
    if not replace and size > n:
        raise ValueError(f"cannot draw {size} elements without replacement from {n}")
    if rng is None:
        rng = np.random.default_rng()
    if replace:
        index = rng.integers(0, n, int(size))
    else:
        index = rng.permutation(n)[:int(size)]
    return pop[index]


def sample_categorical(
    p: ArrayLike, size: int | None = None, rng: np.random.Generator | None = None
) -> int | np.ndarray:
    """Draw category indices from a discrete distribution (the carnival wheel).

    Category ``k`` has probability ``p[k]``. Each draw takes a uniform number ``u``
    in [0, 1) and returns the category whose slice of the cumulative sum of ``p``
    contains ``u``: the cumulative distribution is inverted.

    Parameters
    ----------
    p : array-like of shape (n_categories,)
        Probabilities: non-negative, summing to 1.
    size : int or None, default=None
        Number of draws; None for a single draw.
    rng : np.random.Generator or None, default=None
        Random generator. If None, ``np.random.default_rng()`` (not reproducible).

    Returns
    -------
    int or np.ndarray
        A Python int if ``size`` is None, otherwise an int array of shape (size,).

    Raises
    ------
    ValueError
        If ``p`` is not 1-D, has a negative entry or does not sum to 1
        (tolerance 1e-8).

    Notes
    -----
    Documented algorithm, tested for exact equality under the same seed:
    ``u = rng.random(size)``, then ``np.searchsorted(np.cumsum(p), u, side='right')``,
    clipped to ``len(p) - 1`` (protection against rounding errors in the cumulative
    sum). Also tested with ``scipy.stats.chisquare`` on 100 000 draws.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> sample_categorical([0.2, 0.5, 0.3], size=8, rng=rng)
    array([1, 1, 0, 0, 2, 2, 1, 2])
    >>> sample_categorical([0.0, 1.0], rng=rng)
    1
    """
    probs = np.asarray(p, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError(f"p must be a non-empty 1-D array, got shape {probs.shape}")
    if np.isnan(probs).any() or np.any(probs < 0):
        raise ValueError("probabilities must be non-negative numbers")
    if abs(probs.sum() - 1) > 1e-8:
        raise ValueError(f"probabilities must sum to 1, got {probs.sum()}")
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(size)
    k = np.searchsorted(np.cumsum(probs), u, side="right")
    k = np.minimum(k, len(probs) - 1)                     # rounding errors of the cumulative sum
    return int(k) if size is None else k.astype(int)


def bootstrap_distribution(
    x: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 1000,
    sample_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute a statistic on many bootstrap resamples of the data.

    Each resample draws ``sample_size`` elements of ``x`` with replacement (along
    the first axis, so the rows of a 2-D array stay paired, e.g. ``(n, 2)`` for a
    correlation). The spread of the returned values shows how much the statistic
    would vary from one dataset to another.

    Parameters
    ----------
    x : array-like of shape (n, ...)
        The data.
    statistic : callable, default=np.mean
        Function of one resample (an array of shape ``(sample_size, ...)``)
        returning a float.
    n_boot : int, default=1000
        Number of resamples, ``>= 1``.
    sample_size : int or None, default=None
        Size of each resample; ``len(x)`` if None (modern practice: the book uses
        smaller bootstraps).
    rng : np.random.Generator or None, default=None
        Random generator. If None, ``np.random.default_rng()`` (not reproducible).

    Returns
    -------
    np.ndarray of shape (n_boot,)
        Float values of the statistic, one per resample.

    Raises
    ------
    ValueError
        If ``x`` is empty, if ``n_boot < 1`` or if ``sample_size < 1``.

    Notes
    -----
    Documented algorithm, tested for exact equality under the same seed: for each
    resample in turn, ``idx = rng.integers(0, n, size=sample_size)``, then
    ``statistic(x[idx])``. Also compared with ``scipy.stats.bootstrap`` (Monte Carlo
    tolerance).

    Examples
    --------
    >>> bootstrap_distribution([1, 2, 3, 4], n_boot=3, rng=np.random.default_rng(0))
    array([3.  , 1.25, 3.  ])
    """
    data = np.asarray(x)
    if data.ndim == 0 or data.shape[0] == 0:
        raise ValueError("x is empty")
    if np.issubdtype(data.dtype, np.number) and np.isnan(data).any():
        raise ValueError("x contains NaN: remove the missing values first (e.g. with dropna)")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    n = data.shape[0]
    size = n if sample_size is None else sample_size
    if size < 1:
        raise ValueError(f"sample_size must be >= 1, got {size}")
    if rng is None:
        rng = np.random.default_rng()
    values = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        index = rng.integers(0, n, size=size)             # one resample, with replacement
        values[b] = statistic(data[index])
    return values


def bootstrap_ci(
    x: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence: float = 0.95,
    n_boot: int = 1000,
    sample_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Compute a percentile bootstrap confidence interval.

    The interval keeps the central share ``confidence`` of the bootstrap
    distribution: it cuts ``(1 - confidence) / 2`` of the values on each side.

    Parameters
    ----------
    x : array-like of shape (n, ...)
        As in ``bootstrap_distribution``.
    statistic : callable, default=np.mean
        As in ``bootstrap_distribution``.
    confidence : float, default=0.95
        Confidence level, strictly between 0 and 1 (the book's example uses 0.80).
    n_boot : int, default=1000
        As in ``bootstrap_distribution``.
    sample_size : int or None, default=None
        As in ``bootstrap_distribution``.
    rng : np.random.Generator or None, default=None
        As in ``bootstrap_distribution``.

    Returns
    -------
    tuple of (float, float)
        ``(low, high)`` as Python floats: the percentiles
        ``50 * (1 - confidence)`` and ``50 * (1 + confidence)`` of the bootstrap
        distribution.

    Raises
    ------
    ValueError
        If ``confidence`` is not in (0, 1), plus the errors of
        ``bootstrap_distribution``.

    Notes
    -----
    Tested for exact equality with ``bootstrap_distribution`` (same arguments, same
    seed) followed by ``np.percentile`` (default linear method); also compared with
    ``scipy.stats.bootstrap(..., method='percentile')`` within Monte Carlo tolerance.

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> bootstrap_ci(data, confidence=0.8, rng=np.random.default_rng(0))
    (3.5, 5.5)
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    values = bootstrap_distribution(x, statistic, n_boot=n_boot, sample_size=sample_size, rng=rng)
    low, high = percentile(values, [50 * (1 - confidence), 50 * (1 + confidence)])
    return float(low), float(high)
