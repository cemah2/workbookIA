"""Small generic helpers — mylearn, chapter 0A (Python, notebooks and tools).

Four helpers written while learning Python and NumPy, then reused all along the
workbook: counting values (chapters 3, 6 and 13), finding the index of the largest
score to turn scores into predictions (chapters 13 to 18), one-hot encoding class
labels (chapters 12 and 18) and cutting a dataset into mini-batches (every NumPy
training loop, chapters 18 to 20).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike


def count_values(
    values: Iterable[Hashable], normalize: bool = False
) -> dict[Hashable, int | float]:
    """Count how many times each distinct value occurs.

    Works like ``collections.Counter`` or pandas ``value_counts``: the result maps
    each distinct value to its number of occurrences, or to its proportion
    (count / total) when ``normalize`` is True.

    Parameters
    ----------
    values : iterable of hashable
        Items to count: list, tuple, str (one item per character), 1-D NumPy array
        or pandas Series.
    normalize : bool, default=False
        If True, return proportions that sum to 1 instead of counts.

    Returns
    -------
    dict
        ``{value: count}`` with int counts, or ``{value: proportion}`` with float
        proportions if ``normalize`` is True. Keys appear in order of first
        appearance in ``values`` (they are the items produced by iterating over
        ``values``, e.g. NumPy scalars for a NumPy array).

    Raises
    ------
    ValueError
        If ``values`` is empty, or if it contains NaN (``NaN != NaN``, so missing
        values cannot be counted reliably: drop them first, e.g. with ``dropna``).

    Notes
    -----
    Tested against ``dict(collections.Counter(values))`` (same pairs and same key
    order) and ``pandas.Series(values).value_counts(normalize=...)`` (same pairs).
    Counts sum to the number of items; proportions sum to 1.

    Examples
    --------
    >>> count_values(["b", "a", "b", "c", "b"])
    {'b': 3, 'a': 1, 'c': 1}
    >>> count_values("abca", normalize=True)
    {'a': 0.5, 'b': 0.25, 'c': 0.25}
    """
    # TODO: walk through the values once, reject NaN, update a dictionary of counts;
    # then handle the empty case and the normalize option.
    raise NotImplementedError("count_values() is not implemented yet")


def argmax(values: ArrayLike, axis: int | None = None) -> int | np.ndarray:
    """Return the index of the first largest value, computed with explicit loops.

    When several entries share the maximum, the first one wins (the NumPy rule).
    Write it with ``for`` loops: ``np.argmax`` and ``np.max`` are not allowed in
    the body (they are the oracle of the tests).

    Parameters
    ----------
    values : array-like of shape (n,) or (n_rows, n_cols)
        Numbers (no NaN).
    axis : {None, 0, 1, -1}, default=None
        ``None``: index in the whole array, flattened in row-major (C) order.
        ``0``: one index per column. ``1`` or ``-1``: one index per row.
        For a 1-D input, ``0`` and ``-1`` behave like ``None``.

    Returns
    -------
    int or np.ndarray
        A Python int when the result is a single index; otherwise a 1-D int64 array
        of shape (n_cols,) for ``axis=0`` or (n_rows,) for ``axis=1`` / ``-1``.

    Raises
    ------
    ValueError
        If ``values`` is empty, contains NaN or is not 1-D or 2-D, or if ``axis``
        is not None, 0, 1 or -1 (``axis=1`` is also refused for a 1-D input).

    Notes
    -----
    Tested against ``np.argmax`` (same tie rule) on random integer arrays with
    many ties.

    Examples
    --------
    >>> argmax([3, 7, 7, 1])
    1
    >>> argmax([[1, 9, 9], [8, 2, 8]])
    1
    >>> argmax([[1, 9, 9], [8, 2, 8]], axis=0)
    array([1, 0, 0])
    >>> argmax([[1, 9, 9], [8, 2, 8]], axis=1)
    array([1, 0])
    """
    # TODO: check the input and the axis, then scan the values with loops, keeping
    # the best index found so far (a strictly larger value replaces it).
    raise NotImplementedError("argmax() is not implemented yet")


def one_hot(
    y: ArrayLike, n_classes: int | None = None, dtype: DTypeLike = np.float64
) -> np.ndarray:
    """One-hot encode integer class labels.

    Row ``i`` of the result has a 1 in column ``y[i]`` and 0 everywhere else, so
    every row sums to 1. This is the target format of the losses of chapter 18.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Integer labels in ``[0, n_classes)``. Integer-valued floats such as
        ``2.0`` are accepted.
    n_classes : int or None, default=None
        Number of columns. If None, ``max(y) + 1``.
    dtype : data-type, default=np.float64
        Data type of the result.

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes)
        The one-hot matrix, with the requested ``dtype``.

    Raises
    ------
    ValueError
        If ``y`` is not 1-D, is empty while ``n_classes`` is None, or contains a
        non-integer label, a negative label or a label ``>= n_classes``.

    Notes
    -----
    Tested against ``np.eye(n_classes)[y]``,
    ``torch.nn.functional.one_hot(torch.as_tensor(y), n_classes)`` and
    ``sklearn.preprocessing.OneHotEncoder(categories=[list(range(n_classes))],
    sparse_output=False)``.

    Examples
    --------
    >>> one_hot([0, 2, 1])
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    >>> one_hot([1, 0], n_classes=3, dtype=int)
    array([[0, 1, 0],
           [1, 0, 0]])
    """
    # TODO: validate the labels and choose the number of columns, then start from
    # a matrix of zeros and put one 1 per row.
    raise NotImplementedError("one_hot() is not implemented yet")


def iterate_minibatches(
    n_samples: int,
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
    drop_last: bool = False,
) -> list[np.ndarray]:
    """Split the indices ``0 .. n_samples - 1`` into consecutive mini-batches.

    The indices are first put in an order (shuffled or not), then cut into
    consecutive slices of ``batch_size`` indices; the last slice may be smaller.
    These index arrays drive every NumPy training loop of chapters 18 to 20::

        for idx in iterate_minibatches(len(X), 32, rng=rng):
            X_batch, y_batch = X[idx], y[idx]

    Parameters
    ----------
    n_samples : int
        Number of examples, ``>= 1``.
    batch_size : int
        Size of every mini-batch, ``>= 1`` (the last one may be smaller).
    shuffle : bool, default=True
        If True, the order is ``rng.permutation(n_samples)`` (exactly this call,
        so a new order at every call, i.e. at every epoch). If False, the order is
        ``0, 1, ..., n_samples - 1``.
    rng : np.random.Generator or None, default=None
        Random generator used when ``shuffle`` is True. If None,
        ``np.random.default_rng()`` (not reproducible).
    drop_last : bool, default=False
        If True, drop the last batch when it is incomplete, like
        ``torch.utils.data.DataLoader(..., drop_last=True)``.

    Returns
    -------
    list of np.ndarray
        1-D int64 index arrays ``order[k * batch_size:(k + 1) * batch_size]``:
        ``ceil(n_samples / batch_size)`` of them, or ``floor(n_samples / batch_size)``
        with ``drop_last``. Without ``drop_last`` they form a partition of
        ``range(n_samples)``.

    Raises
    ------
    ValueError
        If ``n_samples < 1`` or ``batch_size < 1``.

    Notes
    -----
    Tested against ``list(torch.utils.data.BatchSampler(order, batch_size,
    drop_last))`` with ``order = np.random.default_rng(seed).permutation(n_samples)``
    (or ``range(n_samples)`` if ``shuffle`` is False), plus properties: partition,
    batch sizes, same seed gives the same batches, two calls with one generator
    give two different orders.

    Examples
    --------
    >>> iterate_minibatches(5, 2, shuffle=False)
    [array([0, 1]), array([2, 3]), array([4])]
    >>> iterate_minibatches(5, 2, shuffle=False, drop_last=True)
    [array([0, 1]), array([2, 3])]
    >>> iterate_minibatches(5, 2, rng=np.random.default_rng(0))
    [array([2, 4]), array([3, 0]), array([1])]
    """
    # TODO: check the arguments, build the order of the indices, then cut it into
    # consecutive slices (and drop the last one if asked and incomplete).
    raise NotImplementedError("iterate_minibatches() is not implemented yet")
