"""Linear algebra in pure Python — mylearn, chapter 0B (From high-school maths to ML).

Vectors and matrices written with plain Python lists, to see every multiplication
and addition hidden behind NumPy's ``@``. A vector is a list of numbers; a matrix
is a list of rows, all of the same length. Do not use NumPy in this module (only
the standard library, e.g. ``math``): NumPy is the oracle of the tests. Inputs are
never modified: functions that return a vector or a matrix build new lists of floats.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


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
    # TODO: check the lengths, then build a new list, one sum per position.
    raise NotImplementedError("vector_add() is not implemented yet")


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
    # TODO: check the lengths, then build a new list, one difference per position.
    raise NotImplementedError("vector_subtract() is not implemented yet")


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
    # TODO: build a new list with one product per component.
    raise NotImplementedError("scalar_multiply() is not implemented yet")


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
    # TODO: check the lengths, then build a new list, one product per position.
    raise NotImplementedError("hadamard() is not implemented yet")


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
    # TODO: check the lengths, then accumulate the products in a running total.
    raise NotImplementedError("dot() is not implemented yet")


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
    # TODO: check p, treat p = inf apart, otherwise apply the formula above.
    raise NotImplementedError("norm() is not implemented yet")


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
    # TODO: reuse two functions of this module.
    raise NotImplementedError("distance() is not implemented yet")


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
    # TODO: reuse dot and norm; refuse a zero vector before dividing.
    raise NotImplementedError("cosine_similarity() is not implemented yet")


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
    # TODO: check that there is at least one row and that every row has the same,
    # non-zero length.
    raise NotImplementedError("shape() is not implemented yet")


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
    # TODO: get the shape, then build the new rows one column of A at a time.
    raise NotImplementedError("transpose() is not implemented yet")


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
    # TODO: build n new rows; entry (i, j) depends only on whether i == j.
    raise NotImplementedError("identity() is not implemented yet")


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
    # TODO: check the shapes, then compute one dot product per row.
    raise NotImplementedError("matvec() is not implemented yet")


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
    # TODO: check both shapes, then fill the m x p result with dot products
    # (transposing B first gives easy access to its columns).
    raise NotImplementedError("matmul() is not implemented yet")
