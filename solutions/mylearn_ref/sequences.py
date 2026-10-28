"""Sequence preparation and sampling — mylearn, chapter 22 (Recurrent Neural Networks).

Tools to turn raw sequences into training data for recurrent networks:
sliding windows over a time series, input/target pairs shifted by one step,
a character- or word-level vocabulary with encoding and decoding, and the
sampling of the next token with a temperature. Reused in chapter 24 (text
generation word by word) and in the bonus chapters.

Reference implementation.
"""

from __future__ import annotations

import numbers
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def _positive_int(value, name: str) -> int:
    """An int >= 1 (a bool is not an int), as a Python int."""
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _series(values, name: str) -> np.ndarray:
    """A NumPy view of a 1-D or 2-D sequence (its own dtype); the callers copy what they return."""
    arr = np.asarray(values)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape (T,) or (T, D), got {arr.ndim}-D of shape {arr.shape}")
    return arr


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
    w = _positive_int(window, "window")
    h = _positive_int(horizon, "horizon")
    s = _positive_int(stride, "stride")
    arr = _series(series, "series")
    t = arr.shape[0]
    n = (t - w - h) // s + 1 if t - w - h >= 0 else 0
    if n < 1:
        raise ValueError(f"a series of {t} steps is too short for one window of {w} steps and a horizon of {h}")
    starts = np.arange(n) * s
    two_d = arr.reshape(t, -1)                                  # (T, D), D = 1 for a 1-D series
    X = two_d[starts[:, None] + np.arange(w)[None, :]]          # fancy indexing: a new array (N, w, D)
    y = arr[starts + w - 1 + h].copy()
    return X, y


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
    length = _positive_int(seq_len, "seq_len")
    s = length if stride is None else _positive_int(stride, "stride")
    arr = _series(seq, "seq")
    t = arr.shape[0]
    n = (t - 1 - length) // s + 1 if t - 1 - length >= 0 else 0
    if n < 1:
        raise ValueError(f"a sequence of {t} steps is too short for one chunk of {length} steps and its target")
    idx = np.arange(n)[:, None] * s + np.arange(length)[None, :]   # (N, L)
    return arr[idx], arr[idx + 1]


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
    if max_size is not None:
        max_size = _positive_int(max_size, "max_size")
    min_freq = _positive_int(min_freq, "min_freq")
    counts = Counter(tokens)                                   # a string counts its characters
    if not counts:
        raise ValueError("tokens is empty: nothing to build a vocabulary from")
    if "" in counts:
        raise ValueError("tokens contains an empty token ''")
    ranked = sorted((tok for tok, cnt in counts.items() if cnt >= min_freq and tok != unk_token),
                    key=lambda tok: (-counts[tok], tok))
    itos = ([unk_token] if unk_token is not None else []) + ranked
    if max_size is not None:
        itos = itos[:max_size]
    return {tok: i for i, tok in enumerate(itos)}, itos


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
    if unk_token is not None and unk_token not in stoi:
        raise ValueError(f"unk_token {unk_token!r} is not a key of stoi")
    ids = []
    for tok in tokens:
        if tok in stoi:
            ids.append(stoi[tok])
        elif unk_token is not None:
            ids.append(stoi[unk_token])
        else:
            raise ValueError(f"token {tok!r} is not in the vocabulary (pass unk_token to map it)")
    return np.array(ids, dtype=np.int64)


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
    arr = np.asarray(ids)
    if arr.ndim != 1:
        raise ValueError(f"ids must be 1-D, got shape {arr.shape}")
    if arr.size and (arr.dtype == bool or not np.issubdtype(arr.dtype, np.integer)):
        raise ValueError(f"ids must be integers, got dtype {arr.dtype}")
    if arr.size and ((arr < 0).any() or (arr >= len(itos)).any()):
        raise ValueError(f"ids must be in [0, {len(itos)}), got min {arr.min()} and max {arr.max()}")
    return sep.join(itos[int(i)] for i in arr)


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
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    z = np.array(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)                      # no overflow: the largest exponent is 0
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


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
    if not temperature >= 0:
        raise ValueError(f"temperature must be >= 0, got {temperature!r}")
    z = np.array(logits, dtype=np.float64)
    if z.ndim not in (1, 2) or z.shape[-1] < 1:
        raise ValueError(f"logits must have shape (V,) or (N, V), got {z.shape}")
    v = z.shape[-1]
    if top_k is not None:
        k = _positive_int(top_k, "top_k")
        if k > v:
            raise ValueError(f"top_k must be <= V = {v}, got {k}")
        kth = np.sort(z, axis=-1)[..., v - k]                    # the k-th largest logit of each row
        z = np.where(z < kth[..., None], -np.inf, z)            # ties with it are kept
    if temperature == 0:
        ids = np.argmax(z, axis=-1)                             # greedy decoding: the first maximum
    else:
        gen = np.random.default_rng(rng)
        p = temperature_softmax(z, temperature)
        cum = np.cumsum(p, axis=-1)
        u = gen.random(z.shape[:-1] + (1,))
        ids = np.minimum((u * cum[..., -1:] >= cum).sum(axis=-1), v - 1)   # inverse of the cumulative sum
    return int(ids) if z.ndim == 1 else ids.astype(np.int64)
