"""Count-based word embeddings — mylearn, chapter B2 (Tokenization and embeddings).

Distributional embeddings "the old way", in NumPy: a word-context co-occurrence matrix,
its positive pointwise mutual information (PPMI, linked to the mutual information of
chapter 6), cosine similarity and nearest-neighbour search (analogies).
``cosine_similarity_matrix`` and ``most_similar`` are reused in chapter B4 (dense search).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


def cooccurrence_matrix(
    ids: ArrayLike,
    vocab_size: int,
    window: int = 2,
    weighting: str = "uniform",
    symmetric: bool = True,
) -> np.ndarray:
    """Count word-context co-occurrences inside a sliding window over a sequence of token ids.

    For every position ``i`` and every distance ``d = 1 .. window``, the word ``ids[i]``
    and its right context ``ids[i + d]`` add a weight to ``C[ids[i], ids[i + d]]``; with
    ``symmetric=True``, the left context ``ids[i - d]`` also adds a weight to
    ``C[ids[i], ids[i - d]]``. The weight is 1 ('uniform') or ``1 / d`` ('harmonic', as in
    GloVe).

    Parameters
    ----------
    ids : array-like of shape (n,)
        Sequence of token ids, each in ``[0, vocab_size)``.
    vocab_size : int
        Number of rows and columns of the matrix.
    window : int, default=2
        Number of context positions on each side.
    weighting : {'uniform', 'harmonic'}, default='uniform'
        Weight of a pair at distance ``d``: 1 or ``1 / d``.
    symmetric : bool, default=True
        Count the left and right contexts (True) or the right contexts only (False).

    Returns
    -------
    np.ndarray of shape (vocab_size, vocab_size)
        Float matrix ``C``, ``C[w, c]`` = weighted count of context ``c`` around word ``w``.

    Raises
    ------
    ValueError
        If ``ids`` is not 1-D, ``window < 1``, ``weighting`` is unknown, or an id is
        outside ``[0, vocab_size)``.

    Notes
    -----
    Tested against a double-loop reference. Property: 'uniform' + ``symmetric=True`` gives
    ``C == C.T`` and ``C.sum() == 2 * sum(n - d for d in 1..window)`` (for ``n > window``).

    Examples
    --------
    >>> cooccurrence_matrix([0, 1, 2, 1], vocab_size=3, window=1)
    array([[0., 1., 0.],
           [1., 0., 2.],
           [0., 2., 0.]])
    >>> cooccurrence_matrix([0, 1, 2, 1], vocab_size=3, window=1, symmetric=False)
    array([[0., 1., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    >>> cooccurrence_matrix([0, 1, 2, 1], vocab_size=3, window=2, weighting="harmonic")
    array([[0. , 1. , 0.5],
           [1. , 1. , 2. ],
           [0.5, 2. , 0. ]])
    """
    # TODO: validate, then add the weight of each (word, context) pair within the window.
    raise NotImplementedError("cooccurrence_matrix() is not implemented yet")


def ppmi(C: ArrayLike, alpha: float = 1.0) -> np.ndarray:
    """Positive pointwise mutual information of a co-occurrence matrix.

    ``PPMI(w, c) = max(0, log(p(w, c) / (p(w) p_alpha(c))))`` with the natural log,
    ``p(w, c) = C[w, c] / C.sum()``, ``p(w)`` the row sums divided by ``C.sum()``, and the
    smoothed context distribution ``p_alpha(c) = count(c)**alpha / sum(count(c')**alpha)``
    where ``count(c)`` is the column sum. Cells where ``C == 0`` are set to 0.

    Parameters
    ----------
    C : array-like of shape (n_words, n_contexts)
        Non-negative co-occurrence counts (rows = words, columns = contexts).
    alpha : float, default=1.0
        Context smoothing exponent: 1.0 gives the plain PPMI, 0.75 is the value of
        Levy, Goldberg & Dagan (2015).

    Returns
    -------
    np.ndarray of shape (n_words, n_contexts)
        Float array, >= 0, equal to 0 where ``C == 0``.

    Raises
    ------
    ValueError
        If ``C`` is not 2-D, has negative entries or sums to 0.

    Notes
    -----
    Tested against the explicit float64 formula with ``np.log`` on random matrices.
    Property: an "independent" matrix ``C = np.outer(u, v)`` gives a PPMI of 0 everywhere
    (up to rounding).

    Examples
    --------
    >>> C = np.array([[0., 2., 1.], [2., 0., 3.], [1., 3., 0.]])
    >>> np.round(ppmi(C), 4)
    array([[0.    , 0.47  , 0.    ],
           [0.47  , 0.    , 0.5878],
           [0.    , 0.5878, 0.    ]])
    >>> np.round(ppmi(C, alpha=0.75), 4)
    array([[0.    , 0.5218, 0.    ],
           [0.3941, 0.    , 0.5838],
           [0.    , 0.6396, 0.    ]])
    """
    # TODO: turn the counts into probabilities, compute the log ratio, keep the positive
    # part, and put 0 where C == 0 (avoid warnings from log(0)).
    raise NotImplementedError("ppmi() is not implemented yet")


def cosine_similarity_matrix(
    A: ArrayLike, B: ArrayLike | None = None, eps: float = 1e-12
) -> np.ndarray:
    """Pairwise cosine similarities between the rows of ``A`` and the rows of ``B``.

    ``S[i, j] = A[i] . B[j] / ((||A[i]|| + eps) (||B[j]|| + eps))``: a row of zeros has a
    similarity of 0 with every row.

    Parameters
    ----------
    A : array-like of shape (n, d)
        First set of vectors.
    B : array-like of shape (m, d) or None, default=None
        Second set of vectors; ``B = A`` if None.
    eps : float, default=1e-12
        Added to the norms to avoid a division by zero.

    Returns
    -------
    np.ndarray of shape (n, m)
        Similarities in [-1, 1].

    Raises
    ------
    ValueError
        If ``A`` or ``B`` is not 2-D, or if they have different numbers of columns.

    Notes
    -----
    Tested against ``sklearn.metrics.pairwise.cosine_similarity``.

    Examples
    --------
    >>> A = np.array([[1., 0.], [1., 1.], [0., 0.]])
    >>> np.round(cosine_similarity_matrix(A), 4)
    array([[1.    , 0.7071, 0.    ],
           [0.7071, 1.    , 0.    ],
           [0.    , 0.    , 0.    ]])
    >>> np.round(cosine_similarity_matrix(A, [[0., 2.], [-3., 0.]]), 4)
    array([[ 0.    , -1.    ],
           [ 0.7071, -0.7071],
           [ 0.    ,  0.    ]])
    """
    # TODO: normalise the rows of A and B, then one matrix product.
    raise NotImplementedError("cosine_similarity_matrix() is not implemented yet")


def most_similar(
    E: ArrayLike, query: ArrayLike, k: int = 10, exclude: Sequence[int] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Find the ``k`` rows of ``E`` closest to a query vector by cosine similarity.

    For an analogy "a is to b what c is to ?", use ``query = E[b] - E[a] + E[c]`` and
    ``exclude=(a, b, c)``. Similarities are computed as in ``cosine_similarity_matrix``
    (default ``eps``).

    Parameters
    ----------
    E : array-like of shape (vocab_size, d)
        Embedding matrix, one row per token.
    query : array-like of shape (d,)
        Query vector.
    k : int, default=10
        Number of neighbours. If fewer than ``k`` rows remain after the exclusions, all
        of them are returned.
    exclude : sequence of int, default=()
        Row indices that must not be returned.

    Returns
    -------
    indices : np.ndarray of shape (min(k, n_remaining),)
        Integer row indices, by decreasing similarity; ties go to the smaller index.
    similarities : np.ndarray of shape (min(k, n_remaining),)
        The matching cosine similarities.

    Raises
    ------
    ValueError
        If ``k < 1``, or if ``query`` is not a vector of length ``E.shape[1]``.

    Notes
    -----
    Tested against ``torch.topk`` on ``torch.nn.functional.cosine_similarity`` and
    ``sklearn.neighbors.NearestNeighbors(metric='cosine')``.

    Examples
    --------
    >>> E = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [2., 0.1]])
    >>> idx, sims = most_similar(E, [1., 0.], k=3)
    >>> idx
    array([0, 4, 2])
    >>> np.round(sims, 4)
    array([1.    , 0.9988, 0.7071])
    >>> most_similar(E, [1., 0.], k=3, exclude=(0,))[0]
    array([4, 2, 1])
    """
    # TODO: similarities of the query with every row, drop the excluded rows, sort.
    raise NotImplementedError("most_similar() is not implemented yet")
