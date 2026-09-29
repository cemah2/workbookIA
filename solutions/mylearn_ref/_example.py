"""Example module (reference implementation)."""


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
    values = list(values)
    if not values:
        raise ValueError("mean() of an empty sequence")
    return float(sum(values)) / len(values)
