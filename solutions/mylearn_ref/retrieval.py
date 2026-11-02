"""Retrieval — mylearn, chapter B4 (LLMs in practice: Hugging Face, prompting, RAG and LoRA).

The search side of retrieval-augmented generation, in NumPy and the standard library:
chunking with overlap, BM25 lexical ranking, exhaustive dense search by cosine
similarity, fusion of rankings (reciprocal rank fusion), and the recall@k and MRR
metrics used to evaluate a retriever. Reusable in the final project.

Reference implementation.
"""

from __future__ import annotations

import math
import numbers
import operator
from collections import Counter
from fractions import Fraction
from typing import Collection, Self, Sequence

import numpy as np
from numpy.typing import ArrayLike

# Your own function from chapter B2.
from .embeddings import cosine_similarity_matrix


def _count(value, name: str, minimum: int) -> int:
    """An integer argument (bool refused) that must be >= minimum."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got a bool")
    try:
        n = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if n < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {n}")
    return n


def _real(value, name: str) -> float:
    """A finite real number (bool refused), as a Python float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _ids(values, name: str) -> list[int]:
    """A list of integer ids (NumPy integers accepted)."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integer ids, got a string")
    try:
        return [operator.index(i) for i in values]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of integer ids") from None


def chunk_text(
    text: str, chunk_size: int = 200, overlap: int = 50, unit: str = "word"
) -> list[str]:
    """Split a long text into overlapping chunks of ``chunk_size`` words (or characters).

    The chunks start every ``chunk_size - overlap`` units: 0, step, 2 step, ... The last
    chunk is the first one that reaches the end of the text, so it may be shorter.

    Parameters
    ----------
    text : str
        Input text.
    chunk_size : int, default=200
        Length of a chunk, in units.
    overlap : int, default=50
        Number of units shared by two consecutive chunks.
    unit : {'word', 'char'}, default='word'
        'word': the text is split on whitespace (``str.split()``) and each chunk is
        re-joined with single spaces; 'char': chunks of characters.

    Returns
    -------
    list of str
        The chunks, in order; an empty list if the text has no unit.

    Raises
    ------
    ValueError
        If ``chunk_size < 1``, ``overlap < 0``, ``overlap >= chunk_size``, or ``unit`` is
        unknown.

    Notes
    -----
    Tested with properties: every unit appears in at least one chunk, consecutive chunks
    share exactly ``overlap`` units, and removing the overlaps then joining gives back the
    (whitespace-normalised) text.

    Examples
    --------
    >>> chunk_text("a b c d e f g h i j", chunk_size=4, overlap=2)
    ['a b c d', 'c d e f', 'e f g h', 'g h i j']
    >>> chunk_text("one two  three\\nfour five", chunk_size=3, overlap=1)
    ['one two three', 'three four five']
    >>> chunk_text("abcdefg", chunk_size=3, overlap=1, unit="char")
    ['abc', 'cde', 'efg']
    """
    size = _count(chunk_size, "chunk_size", 1)
    over = _count(overlap, "overlap", 0)
    if over >= size:
        raise ValueError(f"overlap must be < chunk_size, got overlap={over} and chunk_size={size}")
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if unit == "word":
        units, sep = text.split(), " "
    elif unit == "char":
        units, sep = text, ""
    else:
        raise ValueError(f"unit must be 'word' or 'char', not {unit!r}")
    step = size - over
    chunks: list[str] = []
    start = 0
    while start < len(units):
        chunks.append(sep.join(units[start:start + size]))
        if start + size >= len(units):          # this chunk reaches the end: it is the last one
            break
        start += step
    return chunks


class BM25:
    """Okapi BM25 lexical ranking.

    For a query ``q`` and a document ``d``::

        score(d, q) = sum over the terms t of q of
                      idf(t) * f(t, d) (k1 + 1) / (f(t, d) + k1 (1 - b + b |d| / avgdl))
        idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))

    where ``f(t, d)`` is the number of occurrences of ``t`` in ``d``, ``|d|`` the length of
    ``d``, ``avgdl`` the mean document length, ``N`` the number of documents and ``n_t``
    the number of documents that contain ``t``. This idf is always positive.

    Parameters
    ----------
    k1 : float, default=1.5
        Term-frequency saturation (>= 0; 0 ignores the frequency).
    b : float, default=0.75
        Length normalisation, in [0, 1] (0 = none).

    Attributes
    ----------
    idf_ : dict of str to float
        idf of every term of the corpus.
    term_freqs_ : list of dict of str to int
        For each document, the number of occurrences of each of its terms.
    doc_len_ : np.ndarray of shape (n_docs,)
        Length of each document (number of tokens).
    avgdl_ : float
        Mean document length.
    n_docs_ : int
        Number of documents ``N``.

    Notes
    -----
    Tested against scores computed by hand on a 3-document corpus, and with properties:
    the score increases and saturates with the term frequency, ``b = 0`` removes the
    length normalisation, unknown query terms add 0. ``rank_bm25.BM25Okapi`` differs only
    by its idf (it can be negative there, then replaced by a floor).

    Examples
    --------
    >>> corpus = [["the", "cat", "sat"], ["the", "dog", "sat", "down"],
    ...           ["a", "cat", "and", "a", "cat"]]
    >>> bm25 = BM25().fit(corpus)
    >>> bm25.n_docs_, bm25.avgdl_
    (3, 4.0)
    >>> round(bm25.idf_["cat"], 4), round(bm25.idf_["dog"], 4)
    (0.47, 0.9808)
    >>> np.round(bm25.get_scores(["cat"]), 4)
    array([0.5296, 0.    , 0.6215])
    >>> np.round(bm25.get_scores(["cat", "sat"]), 4)
    array([1.0592, 0.47  , 0.6215])
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def fit(self, corpus: Sequence[Sequence[str]]) -> Self:
        """Compute the statistics of a tokenized corpus.

        Parameters
        ----------
        corpus : sequence of sequence of str
            The documents, each given as a list of tokens (e.g. ``chunk.lower().split()``).

        Returns
        -------
        BM25
            Returns self, with ``idf_``, ``term_freqs_``, ``doc_len_``, ``avgdl_`` and
            ``n_docs_``.

        Raises
        ------
        ValueError
            If ``k1 < 0``, ``b`` is outside [0, 1], or the corpus is empty (no document,
            or only empty documents).
        """
        k1 = _real(self.k1, "k1")
        b = _real(self.b, "b")
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        if isinstance(corpus, (str, bytes)):
            raise TypeError("corpus must be a sequence of tokenized documents, got a string")
        docs = []
        for doc in corpus:
            if isinstance(doc, (str, bytes)):
                raise TypeError("each document must be a list of tokens, got a string: tokenize it first "
                                "(e.g. chunk.lower().split())")
            docs.append(list(doc))
        if not docs or not any(docs):
            raise ValueError("the corpus is empty: it needs at least one non-empty document")
        term_freqs = [dict(Counter(doc)) for doc in docs]
        doc_freq: Counter = Counter()
        for tf in term_freqs:
            doc_freq.update(tf.keys())
        n = len(docs)
        self.n_docs_ = n
        self.term_freqs_ = term_freqs
        self.doc_len_ = np.array([len(doc) for doc in docs], dtype=np.int64)
        self.avgdl_ = float(self.doc_len_.mean())
        self.idf_ = {t: math.log1p((n - n_t + 0.5) / (n_t + 0.5)) for t, n_t in doc_freq.items()}
        return self

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        """BM25 score of every document for a tokenized query.

        Each occurrence of a query term adds its contribution (a term repeated in the
        query counts several times, as in ``rank_bm25``); terms absent from the corpus
        add 0.

        Parameters
        ----------
        query : sequence of str
            Query tokens, tokenized like the corpus.

        Returns
        -------
        np.ndarray of shape (n_docs,)
            One float score per document (higher = more relevant).

        Raises
        ------
        ValueError
            If ``fit`` has not been called.
        """
        if not hasattr(self, "idf_"):
            raise ValueError("this BM25 is not fitted yet: call fit(corpus) first")
        if isinstance(query, (str, bytes)):
            raise TypeError("query must be a list of tokens, got a string: tokenize it like the corpus")
        k1, b = float(self.k1), float(self.b)
        norm = k1 * (1.0 - b + b * self.doc_len_ / self.avgdl_)
        scores = np.zeros(self.n_docs_, dtype=np.float64)
        for term in query:
            idf = self.idf_.get(term)
            if idf is None:                     # a term absent from the corpus adds 0
                continue
            f = np.array([tf.get(term, 0) for tf in self.term_freqs_], dtype=np.float64)
            has = f > 0                         # f = 0 adds 0 (also when k1 = 0 or |d| = 0 with b = 1)
            scores[has] += idf * f[has] * (k1 + 1.0) / (f[has] + norm[has])
        return scores


def cosine_top_k(
    query_emb: ArrayLike, doc_embs: ArrayLike, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine similarities of the ``k`` documents closest to each query.

    Dense retrieval by exhaustive search (what a vector database approximates).

    Parameters
    ----------
    query_emb : array-like of shape (d,) or (q, d)
        One query embedding, or one per row.
    doc_embs : array-like of shape (n, d)
        Document embeddings.
    k : int, default=5
        Number of results, ``1 <= k <= n``.

    Returns
    -------
    indices : np.ndarray of shape (q, k), or (k,) for a single query
        Integer document indices, by decreasing similarity; ties go to the smaller index.
    similarities : np.ndarray of shape (q, k), or (k,) for a single query
        The matching cosine similarities.

    Raises
    ------
    ValueError
        If ``k < 1``, ``k > n``, or the dimensions differ.

    Notes
    -----
    Tested against ``sklearn.neighbors.NearestNeighbors(metric='cosine',
    algorithm='brute').kneighbors`` (which returns distances ``1 - similarity``).
    Reuses your ``cosine_similarity_matrix`` (chapter B2).

    Examples
    --------
    >>> docs = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
    >>> idx, sims = cosine_top_k([1., 0.2], docs, k=2)
    >>> idx, np.round(sims, 4)
    (array([0, 2]), array([0.9806, 0.8321]))
    >>> cosine_top_k([[1., 0.2], [0., -1.]], docs, k=2)[0]  # tie at 0: smaller index first
    array([[0, 2],
           [0, 3]])
    """
    top = _count(k, "k", 1)
    q = np.array(query_emb, dtype=np.float64, copy=True)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    if q.ndim != 2:
        raise ValueError(f"query_emb must have shape (d,) or (q, d), got shape {q.shape}")
    docs = np.array(doc_embs, dtype=np.float64, copy=True)
    if docs.ndim != 2:
        raise ValueError(f"doc_embs must have shape (n, d), got shape {docs.shape}")
    if q.shape[1] != docs.shape[1]:
        raise ValueError(f"the embeddings must have the same dimension, got {q.shape[1]} (queries) "
                         f"and {docs.shape[1]} (documents)")
    if top > docs.shape[0]:
        raise ValueError(f"k must be <= the number of documents {docs.shape[0]}, got {top}")
    sims = cosine_similarity_matrix(q, docs)
    order = np.argsort(-sims, axis=1, kind="stable")[:, :top]   # stable: ties keep the smaller index first
    values = np.take_along_axis(sims, order, axis=1)
    if single:
        return order[0], values[0]
    return order, values


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> np.ndarray:
    """Fuse several rankings of document ids with reciprocal rank fusion (RRF).

    ``score(d) = sum over the rankings r that contain d of 1 / (k + rank_r(d))``, ranks
    starting at 1 (Cormack et al., 2009). A document absent from a ranking gets nothing
    from it.

    Parameters
    ----------
    rankings : sequence of sequence of int
        Lists of document ids, best first (e.g. the BM25 and the dense results). An id
        may appear at most once in each list.
    k : int, default=60
        Smoothing constant (60 in Cormack et al.).

    Returns
    -------
    np.ndarray of int
        Every id found in the rankings, sorted by decreasing fused score (ties: smaller
        id first).

    Raises
    ------
    ValueError
        If ``rankings`` is empty, ``k < 0``, or an id appears twice in the same ranking.

    Notes
    -----
    Tested against a hand-computed example, and with properties: identical rankings are
    preserved, and a document present in every list outranks a document present in a
    single list at the same rank.

    Examples
    --------
    >>> reciprocal_rank_fusion([[3, 1, 2], [1, 2, 4]])
    array([1, 2, 3, 4])
    >>> reciprocal_rank_fusion([[3, 1, 2], [1, 2, 4]], k=0)
    array([1, 3, 2, 4])
    """
    if isinstance(rankings, (str, bytes)):
        raise TypeError("rankings must be a sequence of rankings (lists of ids)")
    lists = [_ids(r, "each ranking") for r in rankings]
    if not lists:
        raise ValueError("rankings is empty: give at least one ranking")
    smoothing = _real(k, "k")
    if smoothing < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    c = Fraction(smoothing)                     # exact sums: mathematically equal scores are equal ties
    scores: dict[int, Fraction] = {}
    for ids in lists:
        if len(set(ids)) != len(ids):
            twice = next(i for i in ids if ids.count(i) > 1)
            raise ValueError(f"the id {twice} appears twice in the same ranking")
        for rank, doc in enumerate(ids, start=1):
            scores[doc] = scores.get(doc, Fraction(0)) + 1 / (c + rank)
    order = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return np.array(order, dtype=np.int64)


def _queries(retrieved, relevant) -> tuple[list, list]:
    """The two per-query sequences as lists, checked: at least one query, the same lengths."""
    ret, rel = list(retrieved), list(relevant)
    if not ret:
        raise ValueError("there is no query: retrieved is empty")
    if len(ret) != len(rel):
        raise ValueError(f"retrieved and relevant must have one entry per query, got {len(ret)} and {len(rel)}")
    return ret, rel


def recall_at_k(
    retrieved: Sequence[Sequence[int]], relevant: Sequence[Collection[int]], k: int
) -> float:
    """Mean over the queries of ``|top-k retrieved ∩ relevant| / |relevant|``.

    Parameters
    ----------
    retrieved : sequence of sequence of int
        For each query, the ranked list of retrieved ids (best first).
    relevant : sequence of collection of int
        For each query, the set of relevant ids.
    k : int
        Cut-off: only the first ``k`` retrieved ids count.

    Returns
    -------
    float
        Recall@k in [0, 1].

    Raises
    ------
    ValueError
        If there is no query, the two sequences have different lengths, ``k < 1``, or a
        relevant set is empty.

    Notes
    -----
    Tested against hand-computed values; with a single relevant document per query, it
    equals ``sklearn.metrics.top_k_accuracy_score`` on the matching score matrix.

    Examples
    --------
    >>> retrieved = [[3, 1, 2], [0, 4, 5, 6]]
    >>> relevant = [{1, 7}, {6}]
    >>> recall_at_k(retrieved, relevant, k=2)
    0.25
    >>> recall_at_k(retrieved, relevant, k=4)
    0.75
    """
    ret, rel = _queries(retrieved, relevant)
    cut = _count(k, "k", 1)
    total = 0.0
    for q, (ids, good) in enumerate(zip(ret, rel)):
        good = set(good)
        if not good:
            raise ValueError(f"the relevant set of query {q} is empty: recall is not defined")
        top = set(list(ids)[:cut])
        total += len(top & good) / len(good)
    return total / len(ret)


def mean_reciprocal_rank(
    retrieved: Sequence[Sequence[int]], relevant: Sequence[Collection[int]]
) -> float:
    """Mean over the queries of ``1 / rank`` of the first relevant document retrieved.

    Ranks start at 1; a query with no relevant document retrieved scores 0.

    Parameters
    ----------
    retrieved : sequence of sequence of int
        For each query, the ranked list of retrieved ids (best first).
    relevant : sequence of collection of int
        For each query, the set of relevant ids.

    Returns
    -------
    float
        MRR in [0, 1].

    Raises
    ------
    ValueError
        If there is no query or the two sequences have different lengths.

    Notes
    -----
    Tested against hand-computed values.

    Examples
    --------
    >>> mean_reciprocal_rank([[3, 1, 2], [0, 4, 5, 6]], [{1, 7}, {6}])
    0.375
    >>> mean_reciprocal_rank([[3, 1, 2], [0, 4]], [{9}, {0}])
    0.5
    """
    ret, rel = _queries(retrieved, relevant)
    total = 0.0
    for ids, good in zip(ret, rel):
        good = set(good)
        for rank, doc in enumerate(ids, start=1):
            if doc in good:
                total += 1.0 / rank
                break
    return total / len(ret)
