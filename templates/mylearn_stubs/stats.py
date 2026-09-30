"""Statistics and resampling — mylearn, chapter 2 (Randomness and basic statistics).

Descriptive statistics (central tendency, spread, percentiles, histogram, covariance
and correlation), reproducible random sampling and the bootstrap, written with NumPy.
Reused by preprocessing (chapter 12: mean and std), by ensembles (chapter 14: sample)
and in the checkpoints. Every function refuses NaN: clean the data first (e.g. with
``dropna``). ``ddof=0`` is the default everywhere, including where ``np.cov`` and
pandas use 1.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


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
    # TODO: convert and check the input, then divide the sum by the count
    # (along the axis if one is given).
    raise NotImplementedError("mean() is not implemented yet")


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
    # TODO: sort the values (along the axis), then pick the middle one or average
    # the two middle ones.
    raise NotImplementedError("median() is not implemented yet")


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
    # TODO: count each value, find the largest count, keep the values that reach it.
    raise NotImplementedError("mode() is not implemented yet")


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
    # TODO: compute the deviations from the mean, square them, sum them and divide.
    raise NotImplementedError("variance() is not implemented yet")


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
    # TODO: reuse variance.
    raise NotImplementedError("std() is not implemented yet")


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
    # TODO: sort the values, compute the fractional position of each q, then
    # interpolate between the two neighbouring sorted values.
    raise NotImplementedError("percentile() is not implemented yet")


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
    # TODO: reuse mean and std, refuse a zero standard deviation, then standardize
    # (put the reduced axis back, e.g. with np.expand_dims, so that broadcasting works).
    raise NotImplementedError("zscore() is not implemented yet")


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
    # TODO: compute the edges, then find the bin of each value inside the range
    # (mind the last bin, which keeps its right edge) and count.
    raise NotImplementedError("histogram() is not implemented yet")


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
    # TODO: compute both deviation vectors, multiply them, sum and divide.
    raise NotImplementedError("covariance() is not implemented yet")


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
    # TODO: reuse covariance and std (with the same ddof), refuse constant data.
    raise NotImplementedError("correlation() is not implemented yet")


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
    # TODO: check X, center every column, then combine the centered columns pairwise.
    raise NotImplementedError("covariance_matrix() is not implemented yet")


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
    # TODO: reuse covariance_matrix, then divide each entry by the product of the
    # two standard deviations.
    raise NotImplementedError("correlation_matrix() is not implemented yet")


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
    # TODO: check the arguments, create the generator if needed, then apply the
    # documented algorithm.
    raise NotImplementedError("sample() is not implemented yet")


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
    # TODO: check p, draw the uniform numbers, then find the slice of each one.
    raise NotImplementedError("sample_categorical() is not implemented yet")


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
    # TODO: check the arguments, then loop over the resamples following the
    # documented algorithm.
    raise NotImplementedError("bootstrap_distribution() is not implemented yet")


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
    # TODO: check the confidence, reuse bootstrap_distribution, then cut its tails.
    raise NotImplementedError("bootstrap_ci() is not implemented yet")
