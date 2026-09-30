"""Linear algebra in pure Python — mylearn, chapter 0B (From high-school maths to ML).

Vectors and matrices written with plain Python lists, to see every multiplication
and addition hidden behind NumPy's ``@``. A vector is a list of numbers; a matrix
is a list of rows, all of the same length. Do not use NumPy in this module (only
the standard library, e.g. ``math``): NumPy is the oracle of the tests. Inputs are
never modified: functions that return a vector or a matrix build new lists of floats.

Reference implementation: read it only after trying (``mon_travail/mylearn/linalg_basics.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _check_same_length(u: Sequence[float], v: Sequence[float]) -> None:
    """Raise ValueError if two vectors do not have the same length."""
    if len(u) != len(v):
        raise ValueError(f"vectors of different lengths: {len(u)} and {len(v)}")


def vector_add(u: Sequence[float], v: Sequence[float]) -> list[float]:
    """Add two vectors component by component: u + v.

    Parameters
    ----------
    u : sequence of float, length n
        First vector.
    v : sequence of float, length n
        Second vector.

    Returns
    -------
    list of float, length n
        New list ``[u[0] + v[0], ..., u[n-1] + v[n-1]]``.

    Raises
    ------
    ValueError
        If ``u`` and ``v`` have different lengths.

    Notes
    -----
    Tested against ``np.add``; the inputs must be left unchanged.

    Examples
    --------
    >>> vector_add([1.0, 2.0], [3.0, 0.5])
    [4.0, 2.5]
    """
    _check_same_length(u, v)
    return [float(a + b) for a, b in zip(u, v)]


def vector_subtract(u: Sequence[float], v: Sequence[float]) -> list[float]:
    """Subtract two vectors component by component: u - v.

    Parameters
    ----------
    u : sequence of float, length n
        First vector.
    v : sequence of float, length n
        Vector to subtract.

    Returns
    -------
    list of float, length n
        New list ``[u[0] - v[0], ..., u[n-1] - v[n-1]]``.

    Raises
    ------
    ValueError
        If ``u`` and ``v`` have different lengths.

    Notes
    -----
    Tested against ``np.subtract``.

    Examples
    --------
    >>> vector_subtract([1.0, 2.0], [3.0, 0.5])
    [-2.0, 1.5]
    """
    _check_same_length(u, v)
    return [float(a - b) for a, b in zip(u, v)]


def scalar_multiply(c: float, v: Sequence[float]) -> list[float]:
    """Multiply every component of a vector by the number c.

    Parameters
    ----------
    c : float
        The number (scalar).
    v : sequence of float, length n
        The vector.

    Returns
    -------
    list of float, length n
        New list ``[c * v[0], ..., c * v[n-1]]``.

    Notes
    -----
    Tested against ``c * np.asarray(v)``.

    Examples
    --------
    >>> scalar_multiply(2.0, [1.0, -0.5, 3.0])
    [2.0, -1.0, 6.0]
    """
    return [float(c * x) for x in v]


def hadamard(u: Sequence[float], v: Sequence[float]) -> list[float]:
    """Multiply two vectors element by element (Hadamard product u ⊙ v).

    This is NOT the dot product: the result is a vector, not a number. The sum of
    its components is the dot product: ``sum(hadamard(u, v)) == dot(u, v)``.

    Parameters
    ----------
    u : sequence of float, length n
        First vector.
    v : sequence of float, length n
        Second vector.

    Returns
    -------
    list of float, length n
        New list ``[u[0] * v[0], ..., u[n-1] * v[n-1]]``.

    Raises
    ------
    ValueError
        If ``u`` and ``v`` have different lengths.

    Notes
    -----
    Tested against ``np.multiply``, plus the property above.

    Examples
    --------
    >>> hadamard([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    [4.0, 10.0, 18.0]
    """
    _check_same_length(u, v)
    return [float(a * b) for a, b in zip(u, v)]


def dot(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute the dot product of two vectors: Σ u_i v_i.

    Parameters
    ----------
    u : sequence of float, length n
        First vector.
    v : sequence of float, length n
        Second vector.

    Returns
    -------
    float
        The sum of the products ``u[i] * v[i]``; 0.0 for two empty vectors (like
        ``np.dot``).

    Raises
    ------
    ValueError
        If ``u`` and ``v`` have different lengths.

    Notes
    -----
    Tested against ``np.dot``.

    Examples
    --------
    >>> dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    32.0
    """
    _check_same_length(u, v)
    total = 0.0
    for a, b in zip(u, v):
        total += a * b
    return float(total)


def norm(v: Sequence[float], p: float = 2) -> float:
    """Compute the p-norm (length) of a vector.

    ``‖v‖_p = (Σ |v_i|^p)^(1/p)``: the Euclidean length for ``p = 2``, the sum of
    the absolute values for ``p = 1``, and the largest absolute value for
    ``p = math.inf``.

    Parameters
    ----------
    v : sequence of float, length n
        The vector.
    p : float, default=2
        Order of the norm: 1, 2, any real number ``>= 1``, or ``math.inf``.

    Returns
    -------
    float
        The norm, ``>= 0``; 0.0 for an empty vector (whatever ``p``).

    Raises
    ------
    ValueError
        If ``p < 1``.

    Notes
    -----
    Tested against ``np.linalg.norm(v, ord=p)`` for ``p`` in (1, 2, 3, inf).

    Examples
    --------
    >>> norm([3.0, 4.0])
    5.0
    >>> norm([3.0, -4.0], p=1)
    7.0
    >>> norm([3.0, -4.0], p=math.inf)
    4.0
    """
    if p < 1:
        raise ValueError(f"the order p of a norm must be >= 1, got {p}")
    if len(v) == 0:
        return 0.0
    if p == math.inf:
        return float(max(abs(x) for x in v))
    if p == 1:
        return float(sum(abs(x) for x in v))
    if p == 2:
        return math.sqrt(sum(x * x for x in v))
    return float(sum(abs(x) ** p for x in v) ** (1 / p))


def distance(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute the Euclidean distance ‖u − v‖ between two points.

    Parameters
    ----------
    u : sequence of float, length n
        First point.
    v : sequence of float, length n
        Second point.

    Returns
    -------
    float
        The distance, ``>= 0``.

    Raises
    ------
    ValueError
        If ``u`` and ``v`` have different lengths.

    Notes
    -----
    Tested against ``math.dist`` and ``np.linalg.norm(np.subtract(u, v))``.

    Examples
    --------
    >>> distance([0.0, 0.0], [3.0, 4.0])
    5.0
    """
    return norm(vector_subtract(u, v))


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute the cosine of the angle between two vectors: dot(u, v) / (‖u‖ ‖v‖).

    1 means same direction, 0 perpendicular, -1 opposite directions. The value
    does not change if ``u`` or ``v`` is multiplied by a positive number.

    Parameters
    ----------
    u : sequence of float, length n
        First vector.
    v : sequence of float, length n
        Second vector.

    Returns
    -------
    float
        The cosine similarity, in [-1, 1].

    Raises
    ------
    ValueError
        If ``u`` and ``v`` have different lengths, or if ``u`` or ``v`` has
        norm 0 (the angle is then undefined).

    Notes
    -----
    Tested against ``sklearn.metrics.pairwise.cosine_similarity`` and
    ``1 - scipy.spatial.distance.cosine(u, v)``.

    Examples
    --------
    >>> cosine_similarity([3.0, 4.0], [4.0, 3.0])
    0.96
    >>> cosine_similarity([1.0, 0.0], [0.0, 2.0])
    0.0
    >>> cosine_similarity([3.0, 4.0], [-6.0, -8.0])
    -1.0
    """
    _check_same_length(u, v)
    norm_u, norm_v = norm(u), norm(v)
    if norm_u == 0 or norm_v == 0:
        raise ValueError("the cosine similarity is undefined for a zero vector")
    value = dot(u, v) / (norm_u * norm_v)
    return max(-1.0, min(1.0, value))  # rounding errors can give 1.0000000000000002


def shape(A: Sequence[Sequence[float]]) -> tuple[int, int]:
    """Return the shape (n_rows, n_cols) of a matrix stored as a list of rows.

    Parameters
    ----------
    A : sequence of sequences of float
        The matrix: a list of rows, all of the same length.

    Returns
    -------
    tuple of (int, int)
        ``(n_rows, n_cols)``.

    Raises
    ------
    ValueError
        If ``A`` is empty, has an empty row, or has rows of different lengths
        (a "ragged" matrix).

    Notes
    -----
    Tested against ``np.asarray(A).shape``. The other matrix functions of this
    module use it to validate their inputs and to write their error messages.

    Examples
    --------
    >>> shape([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    (2, 3)
    """
    if len(A) == 0:
        raise ValueError("a matrix needs at least one row")
    n_cols = len(A[0])
    if n_cols == 0:
        raise ValueError("a matrix needs at least one column")
    for i, row in enumerate(A):
        if len(row) != n_cols:
            raise ValueError(f"ragged matrix: row 0 has {n_cols} entries, row {i} has {len(row)}")
    return (len(A), n_cols)


def transpose(A: Sequence[Sequence[float]]) -> list[list[float]]:
    """Transpose a matrix: row i of the result is column i of A.

    Parameters
    ----------
    A : sequence of sequences of float, shape (m, n)
        The matrix.

    Returns
    -------
    list of list of float, shape (n, m)
        New matrix ``T`` with ``T[j][i] == A[i][j]``.

    Raises
    ------
    ValueError
        If ``A`` is not a valid matrix (see ``shape``).

    Notes
    -----
    Tested against ``np.transpose(A).tolist()``, plus the property
    ``transpose(transpose(A)) == A``.

    Examples
    --------
    >>> transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    """
    n_rows, n_cols = shape(A)
    return [[float(A[i][j]) for i in range(n_rows)] for j in range(n_cols)]


def identity(n: int) -> list[list[float]]:
    """Build the identity matrix of size n: 1.0 on the diagonal, 0.0 elsewhere.

    Parameters
    ----------
    n : int
        Size of the matrix, ``>= 1``.

    Returns
    -------
    list of list of float, shape (n, n)
        New identity matrix.

    Raises
    ------
    ValueError
        If ``n < 1``.

    Notes
    -----
    Tested against ``np.eye(n).tolist()``.

    Examples
    --------
    >>> identity(2)
    [[1.0, 0.0], [0.0, 1.0]]
    """
    if n < 1:
        raise ValueError(f"the size of an identity matrix must be >= 1, got {n}")
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matvec(A: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    """Multiply a matrix by a vector: A v, one dot product per row of A.

    Parameters
    ----------
    A : sequence of sequences of float, shape (m, n)
        The matrix.
    v : sequence of float, length n
        The vector.

    Returns
    -------
    list of float, length m
        New vector whose entry ``i`` is ``dot(A[i], v)``.

    Raises
    ------
    ValueError
        If ``A`` is not a valid matrix (see ``shape``), or if ``len(v)`` differs
        from the number of columns of ``A`` (the message shows both shapes, e.g.
        ``cannot multiply (2, 3) by (2,)``).

    Notes
    -----
    Tested against ``np.asarray(A) @ np.asarray(v)``, plus the property
    ``matvec(identity(n), v) == v``.

    Examples
    --------
    >>> matvec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])
    [3.0, 7.0]
    """
    n_rows, n_cols = shape(A)
    if len(v) != n_cols:
        raise ValueError(f"cannot multiply {(n_rows, n_cols)} by ({len(v)},)")
    return [dot(row, v) for row in A]


def matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> list[list[float]]:
    """Multiply two matrices: entry (i, j) of A B is dot(row i of A, column j of B).

    Parameters
    ----------
    A : sequence of sequences of float, shape (m, n)
        Left matrix.
    B : sequence of sequences of float, shape (n, p)
        Right matrix: its number of rows must equal the number of columns of A.

    Returns
    -------
    list of list of float, shape (m, p)
        New matrix, the product ``A B``.

    Raises
    ------
    ValueError
        If ``A`` or ``B`` is not a valid matrix (see ``shape``), or if the number
        of columns of ``A`` differs from the number of rows of ``B``; the message
        shows both shapes, e.g. ``cannot multiply (2, 3) by (2, 2)``.

    Notes
    -----
    Tested against ``np.matmul`` on random integer and float matrices, plus the
    properties ``matmul(A, identity(n)) == A`` and
    ``transpose(matmul(A, B)) == matmul(transpose(B), transpose(A))``.

    Examples
    --------
    >>> matmul([[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [1.0, 0.0]])
    [[2.0, 1.0], [4.0, 3.0]]
    >>> matmul([[1.0, 2.0, 3.0]], [[1.0], [0.0], [2.0]])
    [[7.0]]
    """
    shape_a, shape_b = shape(A), shape(B)
    if shape_a[1] != shape_b[0]:
        raise ValueError(f"cannot multiply {shape_a} by {shape_b}")
    columns_b = transpose(B)
    return [[dot(row, column) for column in columns_b] for row in A]
