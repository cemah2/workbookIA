"""Example module: shows how a mylearn module and its tests work.

1. Implement ``mean`` below (replace the ``raise NotImplementedError`` line).
2. Run ``pytest tests/test_example_mylearn.py`` from the repository root.
3. All tests green? Your first function is done.
"""


def mean(values):
    """Return the arithmetic mean of a non-empty sequence of numbers.

    Parameters
    ----------
    values : sequence of float
        Numbers to average (list, tuple or 1-D NumPy array).

    Returns
    -------
    float
        The sum of the values divided by their count.

    Raises
    ------
    ValueError
        If ``values`` is empty.

    Examples
    --------
    >>> mean([1, 2, 3, 4])
    2.5
    """
    # TODO: raise ValueError if values is empty, otherwise return sum / count.
    raise NotImplementedError("mean() is not implemented yet")
