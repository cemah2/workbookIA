"""Sequence preparation and sampling — mylearn, chapter 22 (Recurrent Neural Networks).

Tools to turn raw sequences into training data for recurrent networks:
sliding windows over a time series, input/target pairs shifted by one step,
a character- or word-level vocabulary with encoding and decoding, and the
sampling of the next token with a temperature. Reused in chapter 24 (text
generation word by word) and in the bonus chapters.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def make_windows(
    series: ArrayLike, window: int, horizon: int = 1, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a series into overlapping windows (inputs) and the value that follows each one.

    Sample ``i`` starts at time ``start = i * stride``: its input is
    ``series[start:start + window]`` and its target is the value ``horizon``
    steps after the last step of the window, ``series[start + window - 1 + horizon]``.

    Parameters
    ----------
    series : array-like of shape (T,) or (T, D)
        Values in time order (D variables per time step).
    window : int
        Number of time steps w per sample, >= 1.
    horizon : int, default=1
        h >= 1: how far after the window the target is (1: the next value).
    stride : int, default=1
        Shift s >= 1 between the starts of two consecutive windows.

    Returns
    -------
    X : ndarray of shape (N, w, D)
        Batch-first inputs, with D = 1 for a 1-D series and
        N = (T - w - h) // s + 1. A new array (not a view of ``series``), with
        the dtype of ``series``.
    y : ndarray of shape (N,) for a 1-D series, (N, D) otherwise
        Targets, also a new array with the dtype of ``series``.

    Raises
    ------
    ValueError
        If ``window``, ``horizon`` or ``stride`` is < 1, or if the series is too
        short for a single sample (N < 1).

    Notes
    -----
    Tested against ``numpy.lib.stride_tricks.sliding_window_view`` and an
    explicit loop on small cases.

    Examples
    --------
    >>> X, y = make_windows(np.arange(6.0), window=3)
    >>> X.shape
    (3, 3, 1)
    >>> X[:, :, 0]
    array([[0., 1., 2.],
           [1., 2., 3.],
           [2., 3., 4.]])
    >>> y
    array([3., 4., 5.])
    >>> make_windows(np.arange(6.0), window=3, horizon=2)[1]
    array([4., 5.])
    """
    raise NotImplementedError("make_windows() is not implemented yet")


def make_sequence_pairs(
    seq: ArrayLike, seq_len: int, stride: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs and next-step targets for many-to-many training.

    Chunk ``i`` starts at ``start = i * stride``: X[i] = seq[start:start + L]
    and Y[i] = seq[start + 1:start + L + 1] (the same chunk shifted by one
    step), so the model learns to predict every next token.

    Parameters
    ----------
    seq : array-like of shape (T,) or (T, D)
        Token ids or values, in order.
    seq_len : int
        Chunk length L, >= 1.
    stride : int or None, default=None
        Shift between two chunks, >= 1. None means L (non-overlapping chunks).

    Returns
    -------
    X : ndarray of shape (N, L) or (N, L, D)
        Inputs, with N = (T - 1 - L) // stride + 1 and the dtype of ``seq``.
    Y : ndarray of the shape of X
        Targets, such that ``Y[:, :-1] == X[:, 1:]``, with the dtype of ``seq``.

    Raises
    ------
    ValueError
        If ``seq_len`` or ``stride`` is < 1, or if the sequence is too short
        (N < 1).

    Notes
    -----
    Tested against an explicit slicing loop and the shift property.

    Examples
    --------
    >>> make_sequence_pairs(np.arange(7), seq_len=3)
    (array([[0, 1, 2],
           [3, 4, 5]]), array([[1, 2, 3],
           [4, 5, 6]]))
    """
    raise NotImplementedError("make_sequence_pairs() is not implemented yet")


def build_vocab(
    tokens: str | Iterable[str],
    max_size: int | None = None,
    min_freq: int = 1,
    unk_token: str | None = None,
) -> tuple[dict[str, int], list[str]]:
    """Map each distinct token (character of a string, or word of a list) to an integer id.

    Ids follow decreasing frequency, ties being broken in Python string order
    (``sorted``); ``unk_token``, when given, comes first with id 0 and stands
    for every out-of-vocabulary token.

    Parameters
    ----------
    tokens : str or iterable of str
        A string (character-level vocabulary) or an iterable of words
        (word-level vocabulary).
    max_size : int or None, default=None
        Keep only the most frequent tokens so that the vocabulary, unk_token
        included, has at most ``max_size`` entries. None keeps them all.
    min_freq : int, default=1
        Drop the tokens seen fewer than ``min_freq`` times.
    unk_token : str or None, default=None
        Name of the unknown token (for example ``"<unk>"``), or None for no
        unknown token.

    Returns
    -------
    stoi : dict of str to int
        Token -> id ("string to index").
    itos : list of str
        Id -> token ("index to string"): ``itos[stoi[t]] == t``.

    Raises
    ------
    ValueError
        If ``tokens`` is empty, ``max_size < 1`` or ``min_freq < 1``.

    Notes
    -----
    Tested against ``collections.Counter.most_common`` and round-trip
    properties.

    Examples
    --------
    >>> stoi, itos = build_vocab("hello")
    >>> itos
    ['l', 'e', 'h', 'o']
    >>> stoi["l"]
    0
    >>> build_vocab(["the", "cat", "the", "dog"], max_size=2, unk_token="<unk>")[1]
    ['<unk>', 'the']
    """
    # TODO: count with collections.Counter, sort by (-count, token), then apply
    #   min_freq and max_size (unk_token first).
    raise NotImplementedError("build_vocab() is not implemented yet")


def encode(
    tokens: str | Sequence[str], stoi: Mapping[str, int], unk_token: str | None = None
) -> np.ndarray:
    """Turn a string (characters) or a list of words into an array of ids.

    Parameters
    ----------
    tokens : str or sequence of str
        Text (one token per character) or list of words.
    stoi : mapping of str to int
        Token -> id mapping, as returned by ``build_vocab``.
    unk_token : str or None, default=None
        Token used in place of the tokens missing from ``stoi`` (it must be a
        key of ``stoi``). None means that an unknown token is an error.

    Returns
    -------
    ndarray of shape (L,)
        The ids, as int64.

    Raises
    ------
    ValueError
        On an unknown token when ``unk_token`` is None.

    Notes
    -----
    Tested against an explicit dictionary lookup, and on the round trip
    ``decode(encode(t), itos) == t``.

    Examples
    --------
    >>> stoi, itos = build_vocab("hello")
    >>> encode("hole", stoi)
    array([2, 3, 0, 1])
    """
    raise NotImplementedError("encode() is not implemented yet")


def decode(ids: ArrayLike, itos: Sequence[str], sep: str = "") -> str:
    """Turn ids back into text: ``sep=""`` for characters, ``sep=" "`` for words.

    Parameters
    ----------
    ids : array-like of shape (L,)
        Integer ids.
    itos : sequence of str
        Id -> token list, as returned by ``build_vocab``.
    sep : str, default=""
        Separator inserted between two tokens.

    Returns
    -------
    str
        The decoded text.

    Raises
    ------
    ValueError
        If an id is negative or >= ``len(itos)``.

    Notes
    -----
    Tested on the round trip with ``encode``.

    Examples
    --------
    >>> stoi, itos = build_vocab("hello")
    >>> decode([2, 3, 0, 1], itos)
    'hole'
    >>> decode([1, 0], ["cat", "the"], sep=" ")
    'the cat'
    """
    raise NotImplementedError("decode() is not implemented yet")


def temperature_softmax(logits: ArrayLike, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax(logits / T) along the last axis.

    T < 1 sharpens the distribution (towards the argmax), T > 1 flattens it
    (towards uniform); T = 1 is the plain softmax.

    Parameters
    ----------
    logits : array-like of shape (V,) or (N, V)
        Raw scores over a vocabulary of size V.
    temperature : float, default=1.0
        T > 0.

    Returns
    -------
    ndarray of the shape of logits
        Probabilities summing to 1 along the last axis.

    Raises
    ------
    ValueError
        If ``temperature <= 0``.

    Notes
    -----
    Tested against ``torch.softmax(logits / T, dim=-1)``.

    Examples
    --------
    >>> temperature_softmax([1.0, 2.0, 3.0])
    array([0.09003057, 0.24472847, 0.66524096])
    >>> temperature_softmax([1.0, 2.0, 3.0], temperature=0.5)
    array([0.01587624, 0.11731043, 0.86681333])
    """
    raise NotImplementedError("temperature_softmax() is not implemented yet")


def sample_from_logits(
    logits: ArrayLike,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> int | np.ndarray:
    """Draw token ids from softmax(logits / T), optionally among the k best only.

    - ``temperature == 0``: greedy decoding, the argmax (first one on ties),
      no randomness;
    - ``top_k``: logits strictly smaller than the k-th largest one are removed
      before sampling (ties with it are kept), as in nanoGPT.

    Parameters
    ----------
    logits : array-like of shape (V,) or (N, V)
        Raw scores of one distribution or of N distributions.
    temperature : float, default=1.0
        T >= 0.
    top_k : int or None, default=None
        None, or 1 <= k <= V.
    rng : numpy.random.Generator or None, default=None
        Generator for reproducible draws. None means ``np.random.default_rng()``.

    Returns
    -------
    int or ndarray of shape (N,)
        A Python int for (V,) logits, an int64 array for (N, V) logits (one id
        per row).

    Raises
    ------
    ValueError
        If ``temperature < 0`` or ``top_k`` is out of range.

    Notes
    -----
    Tested on the empirical frequencies of 20 000 draws against
    ``torch.softmax`` probabilities, the argmax when T = 0, and the support
    restricted by ``top_k``.

    Examples
    --------
    >>> sample_from_logits([1.0, 5.0, 2.0], temperature=0)
    1
    >>> sample_from_logits([1.0, 5.0, 2.0], top_k=1, rng=np.random.default_rng(0))
    1
    >>> sample_from_logits(np.zeros((4, 3)), rng=np.random.default_rng(0)).shape
    (4,)
    """
    # TODO: handle temperature 0 first; apply top_k, then draw from
    #   temperature_softmax with rng.
    raise NotImplementedError("sample_from_logits() is not implemented yet")
