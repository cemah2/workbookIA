"""Entropy, divergences and coding — mylearn, chapter 6 (Information theory).

Shannon's self-information and entropy, cross-entropy, Kullback-Leibler and
Jensen-Shannon divergences, smoothed distributions of tokens and characters, Huffman
codes, perplexity and the log loss. Reused by decision trees (entropy criterion,
chapter 13), the losses of chapter 18, text models (chapters 22, B2 and B3) and the
variational autoencoder (chapter 25). Logarithms are in base 2 (bits) by default;
pass ``base=np.e`` for nats, the unit of PyTorch's losses. The one exception is
``log_loss``, which follows scikit-learn and is in nats by default (``base=2`` for bits).

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, SciPy, scikit-learn or PyTorch).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def self_information(p: float | ArrayLike, base: float = 2.0) -> float | np.ndarray:
    """Compute the self-information (surprise) of an event of probability p.

    ``I(p) = -log_base(p)``: 0 for a certain event, larger for rarer events. In
    bits (base 2), an event of probability 1/2 carries 1 bit.

    Parameters
    ----------
    p : float or array-like of any shape
        Probability or probabilities, in (0, 1].
    base : float, default=2.0
        Base of the logarithm: 2 for bits, ``np.e`` for nats.

    Returns
    -------
    float or np.ndarray
        A Python float for a scalar ``p``, otherwise a float array with the shape
        of ``p``.

    Raises
    ------
    ValueError
        If a probability is ``<= 0`` or ``> 1``, or if ``base <= 0`` or
        ``base == 1``.

    Notes
    -----
    Tested against ``-np.log2(p)`` and ``-np.log(p) / np.log(base)``.

    Examples
    --------
    >>> self_information(0.5)
    1.0
    >>> self_information([0.5, 0.25])
    array([1., 2.])
    """
    # TODO: check p and base, then change the base of a natural logarithm.
    raise NotImplementedError("self_information() is not implemented yet")


def entropy(p: ArrayLike, base: float = 2.0) -> float:
    """Compute the Shannon entropy of a distribution: its average surprise.

    ``H(p) = -Σ p_i log_base(p_i)``, with the convention ``0 log 0 = 0`` (an
    impossible outcome contributes nothing). The entropy is 0 for a certain outcome
    and maximal, ``log_base(n)``, for the uniform distribution on n outcomes.

    Parameters
    ----------
    p : array-like of shape (n,)
        Probabilities: non-negative, summing to 1 (tolerance 1e-6).
    base : float, default=2.0
        Base of the logarithm: 2 for bits, ``np.e`` for nats.

    Returns
    -------
    float
        The entropy, between 0 and ``log_base(n)``.

    Raises
    ------
    ValueError
        If ``p`` is not 1-D, has a negative entry or does not sum to 1, or if
        ``base <= 0`` or ``base == 1``.

    Notes
    -----
    Tested against ``scipy.stats.entropy(p, base=base)``.

    Examples
    --------
    >>> entropy([0.5, 0.5])
    1.0
    >>> entropy([0.5, 0.25, 0.25])
    1.5
    >>> entropy([0.25, 0.25, 0.25, 0.25])
    2.0
    """
    # TODO: validate p, then average the surprise over the outcomes with p_i > 0.
    raise NotImplementedError("entropy() is not implemented yet")


def cross_entropy(p: ArrayLike, q: ArrayLike, base: float = 2.0) -> float:
    """Compute the cross-entropy of q relative to p.

    ``H(p, q) = -Σ p_i log_base(q_i)``: the average cost of coding data drawn from
    ``p`` with a code built for ``q``. Outcomes with ``p_i = 0`` contribute nothing;
    the result is infinite if ``q_i = 0`` for an outcome with ``p_i > 0``. Always
    ``H(p, q) >= H(p)``, with equality when ``q == p``.

    Parameters
    ----------
    p : array-like of shape (n,)
        True distribution: non-negative, summing to 1 (tolerance 1e-6).
    q : array-like of shape (n,)
        Model (code) distribution, same constraints.
    base : float, default=2.0
        Base of the logarithm: 2 for bits, ``np.e`` for nats.

    Returns
    -------
    float
        The cross-entropy, ``>= entropy(p)``, possibly ``inf``.

    Raises
    ------
    ValueError
        If the shapes differ, if ``p`` or ``q`` is not a distribution, or if
        ``base <= 0`` or ``base == 1``.

    Notes
    -----
    Tested against ``scipy.stats.entropy(p, base=base) + scipy.stats.entropy(p, q,
    base=base)``, and against ``torch.nn.functional.cross_entropy`` for a one-hot
    ``p`` (in nats).

    Examples
    --------
    >>> round(cross_entropy([0.5, 0.5], [0.25, 0.75]), 4)
    1.2075
    >>> cross_entropy([0.5, 0.5], [1.0, 0.0])
    inf
    """
    # TODO: validate both distributions, then sum over the outcomes with p_i > 0
    # (an infinite surprise gives an infinite result).
    raise NotImplementedError("cross_entropy() is not implemented yet")


def kl_divergence(p: ArrayLike, q: ArrayLike, base: float = 2.0) -> float:
    """Compute the Kullback-Leibler divergence KL(p ‖ q).

    ``KL(p ‖ q) = Σ p_i log_base(p_i / q_i) = H(p, q) - H(p)``: the extra cost of
    using the code of ``q`` for data drawn from ``p``. It is ``>= 0``, equal to 0
    only when ``p == q``, and not symmetric: ``KL(p ‖ q) != KL(q ‖ p)`` in general.
    Same conventions as ``cross_entropy`` (``inf`` if ``q_i = 0`` where
    ``p_i > 0``).

    Parameters
    ----------
    p : array-like of shape (n,)
        Distribution of the data: non-negative, summing to 1 (tolerance 1e-6).
    q : array-like of shape (n,)
        Distribution of the model, same constraints.
    base : float, default=2.0
        Base of the logarithm: 2 for bits, ``np.e`` for nats.

    Returns
    -------
    float
        The divergence, ``>= 0``, possibly ``inf``.

    Raises
    ------
    ValueError
        If the shapes differ, if ``p`` or ``q`` is not a distribution, or if
        ``base <= 0`` or ``base == 1``.

    Notes
    -----
    Tested against ``scipy.stats.entropy(p, q, base=base)`` and
    ``scipy.special.rel_entr(p, q).sum() / np.log(base)``.

    Examples
    --------
    >>> round(kl_divergence([0.5, 0.5], [0.25, 0.75]), 4)
    0.2075
    >>> round(kl_divergence([0.25, 0.75], [0.5, 0.5]), 4)
    0.1887
    """
    # TODO: validate both distributions, then sum over the outcomes with p_i > 0.
    raise NotImplementedError("kl_divergence() is not implemented yet")


def js_divergence(p: ArrayLike, q: ArrayLike, base: float = 2.0) -> float:
    """Compute the Jensen-Shannon divergence, a symmetric and always finite divergence.

    With the mixture ``m = (p + q) / 2``:
    ``JS(p, q) = (KL(p ‖ m) + KL(q ‖ m)) / 2``. It is symmetric, 0 only when
    ``p == q``, and at most ``log_base(2)`` (1 bit), reached when ``p`` and ``q``
    have no outcome in common.

    Parameters
    ----------
    p : array-like of shape (n,)
        First distribution: non-negative, summing to 1 (tolerance 1e-6).
    q : array-like of shape (n,)
        Second distribution, same constraints.
    base : float, default=2.0
        Base of the logarithm: 2 for bits, ``np.e`` for nats.

    Returns
    -------
    float
        The divergence, between 0 and ``log_base(2)`` (between 0 and 1 in bits).

    Raises
    ------
    ValueError
        If the shapes differ, if ``p`` or ``q`` is not a distribution, or if
        ``base <= 0`` or ``base == 1``.

    Notes
    -----
    Tested against ``scipy.spatial.distance.jensenshannon(p, q, base=base) ** 2``
    (SciPy returns the square root of the divergence).

    Examples
    --------
    >>> js_divergence([1.0, 0.0], [0.0, 1.0])
    1.0
    >>> round(js_divergence([0.5, 0.5], [0.25, 0.75]), 4)
    0.0488
    """
    # TODO: build the mixture, then reuse kl_divergence twice.
    raise NotImplementedError("js_divergence() is not implemented yet")


def token_distribution(
    tokens: Iterable[Hashable],
    vocabulary: Sequence[Hashable] | None = None,
    smoothing: float = 0.0,
) -> tuple[list[Hashable], np.ndarray]:
    """Compute the empirical probability of each token of a vocabulary.

    The probability of a token is its count divided by the total count. Adding
    ``smoothing`` to every count of the vocabulary (Laplace smoothing) avoids zero
    probabilities for tokens that never occur; when ``smoothing`` dominates the
    counts, the distribution tends to the uniform one.

    Parameters
    ----------
    tokens : iterable of hashable
        The tokens (words, characters...).
    vocabulary : sequence of hashable or None, default=None
        Tokens to keep, in this order. If None, the sorted distinct tokens. Tokens
        outside the vocabulary are ignored.
    smoothing : float, default=0.0
        Pseudo-count added to every vocabulary entry, ``>= 0``.

    Returns
    -------
    vocabulary : list
        The vocabulary, as a list (in the given order, or sorted).
    probs : np.ndarray of shape (len(vocabulary),)
        Float probabilities aligned with the vocabulary, summing to 1.

    Raises
    ------
    ValueError
        If ``smoothing < 0``, if ``vocabulary`` contains a repeated token, or if the
        total count is 0 (no token of the vocabulary occurs and ``smoothing == 0``).

    Notes
    -----
    Tested against ``collections.Counter``, plus properties: probabilities sum to 1,
    the order of ``vocabulary`` is respected, a large ``smoothing`` gives a nearly
    uniform distribution.

    Examples
    --------
    >>> token_distribution(["a", "b", "a", "c"])
    (['a', 'b', 'c'], array([0.5 , 0.25, 0.25]))
    >>> token_distribution(["a", "b", "a", "c"], vocabulary=["a", "z"], smoothing=1)
    (['a', 'z'], array([0.75, 0.25]))
    """
    # TODO: choose the vocabulary, count its tokens, add the smoothing, normalize.
    raise NotImplementedError("token_distribution() is not implemented yet")


def char_distribution(
    text: str,
    alphabet: str | Sequence[str] | None = None,
    lowercase: bool = True,
    smoothing: float = 0.0,
) -> tuple[list[str], np.ndarray]:
    """Compute the letter frequencies of a text as a probability distribution.

    The character version of ``token_distribution`` (the book's letter-frequency
    chart): ``char_distribution(text)`` equals
    ``token_distribution(list(text.lower()))``.

    Parameters
    ----------
    text : str
        The text.
    alphabet : str or sequence of str or None, default=None
        Characters to count, in this order, e.g. ``'abcdefghijklmnopqrstuvwxyz'``;
        other characters are ignored. It is used as given (never lower-cased). If
        None, every distinct character of the text (lower-cased first if
        ``lowercase``), sorted, including spaces, punctuation and line breaks.
    lowercase : bool, default=True
        If True, convert the text to lower case first.
    smoothing : float, default=0.0
        As in ``token_distribution``.

    Returns
    -------
    alphabet : list of str
        The characters, in the given order or sorted.
    probs : np.ndarray of shape (len(alphabet),)
        Float probabilities aligned with the characters, summing to 1.

    Raises
    ------
    ValueError
        If ``smoothing < 0``, or if no character of the alphabet occurs in the text
        and ``smoothing == 0``.

    Notes
    -----
    Tested against ``collections.Counter`` on the same characters and against
    ``token_distribution(list(text))``.

    Examples
    --------
    >>> char_distribution("Abba")
    (['a', 'b'], array([0.5, 0.5]))
    >>> char_distribution("Hello", alphabet="helo")
    (['h', 'e', 'l', 'o'], array([0.2, 0.2, 0.4, 0.2]))
    """
    # TODO: prepare the text, then reuse token_distribution.
    raise NotImplementedError("char_distribution() is not implemented yet")


def huffman_code(symbols: Sequence[Hashable], probs: ArrayLike) -> dict[Hashable, str]:
    """Build an optimal prefix-free binary code (Huffman code).

    The code is built by repeatedly merging the two least probable groups of
    symbols until one group is left; the symbols of the two merged groups get one
    more bit, 0 on one side and 1 on the other. Frequent symbols end up with short
    codewords. No codeword is the beginning of another (prefix-free), so a bit
    string can be decoded without separators. Ties can be broken in any order: the
    codewords are then not unique, but the expected length is always the same,
    and optimal.

    Parameters
    ----------
    symbols : sequence of hashable
        The distinct symbols.
    probs : array-like of shape (n_symbols,)
        Their probabilities: non-negative, summing to 1 (tolerance 1e-6).

    Returns
    -------
    dict
        ``{symbol: codeword}``, each codeword a string of ``'0'`` and ``'1'``. A
        lone symbol gets ``'0'``.

    Raises
    ------
    ValueError
        If the lengths differ, if ``symbols`` is empty or has a repeated symbol, or
        if ``probs`` is not a distribution.

    Notes
    -----
    Tested with properties: prefix-free, Kraft sum ``Σ 2**(-len(codeword)) = 1``,
    expected length equal to that of an independent ``heapq`` reference, and
    ``entropy <= expected length < entropy + 1`` (in bits).

    Examples
    --------
    >>> code = huffman_code(["a", "b", "c"], [0.5, 0.25, 0.25])
    >>> sorted((symbol, len(word)) for symbol, word in code.items())
    [('a', 1), ('b', 2), ('c', 2)]
    >>> huffman_code(["x"], [1.0])
    {'x': '0'}
    """
    # TODO: validate the inputs, then merge groups with a priority queue (heapq),
    # prepending a bit to the codewords of each merged group.
    raise NotImplementedError("huffman_code() is not implemented yet")


def huffman_encode(symbols: Iterable[Hashable], code: Mapping[Hashable, str]) -> str:
    """Encode a sequence of symbols: concatenate their codewords into one bit string.

    Parameters
    ----------
    symbols : iterable of hashable
        The symbols to send, in order (a str is a sequence of characters).
    code : mapping of hashable to str
        A prefix-free code, e.g. from ``huffman_code``.

    Returns
    -------
    str
        A string of ``'0'`` and ``'1'`` whose length is the sum of the lengths of
        the codewords used.

    Raises
    ------
    ValueError
        If a symbol has no codeword in ``code``.

    Notes
    -----
    Tested with the round trip through ``huffman_decode`` and the length property.

    Examples
    --------
    >>> huffman_encode("abca", {"a": "0", "b": "10", "c": "11"})
    '010110'
    """
    # TODO: look up each codeword and join them.
    raise NotImplementedError("huffman_encode() is not implemented yet")


def huffman_decode(bits: str, code: Mapping[Hashable, str]) -> list[Hashable]:
    """Decode a bit string back into symbols with a prefix-free code.

    Read the bits from left to right; as soon as the bits read so far form a
    codeword, output its symbol and start again. This works because the code is
    prefix-free.

    Parameters
    ----------
    bits : str
        String of ``'0'`` and ``'1'``.
    code : mapping of hashable to str
        The prefix-free code used for encoding.

    Returns
    -------
    list
        The decoded symbols, in order.

    Raises
    ------
    ValueError
        If ``bits`` contains other characters than ``'0'`` and ``'1'``, if the code
        is not prefix-free, if a sequence of bits matches no codeword, or if
        ``bits`` ends in the middle of a codeword.

    Notes
    -----
    Tested with the round trip
    ``huffman_decode(huffman_encode(s, c), c) == list(s)``.

    Examples
    --------
    >>> huffman_decode("010110", {"a": "0", "b": "10", "c": "11"})
    ['a', 'b', 'c', 'a']
    """
    # TODO: check the bits and the code, invert the code (codeword -> symbol), then
    # read the bits one by one.
    raise NotImplementedError("huffman_decode() is not implemented yet")


def perplexity(token_probs: ArrayLike) -> float:
    """Compute the perplexity of a model on a sequence of observed tokens.

    The exponential of the mean negative log-probability that the model gave to
    each token actually observed: ``exp(-mean(log p_i))``. It reads as an
    "effective number of choices": a model that is uniform over V tokens has
    perplexity V. It does not depend on the base of the logarithm, so there is no
    ``base`` parameter.

    Parameters
    ----------
    token_probs : array-like of shape (n_tokens,)
        Probabilities that the model assigned to the observed tokens, each in
        (0, 1].

    Returns
    -------
    float
        The perplexity, ``>= 1`` (lower is better).

    Raises
    ------
    ValueError
        If ``token_probs`` is empty or a value is outside (0, 1].

    Notes
    -----
    Tested against ``torch.exp(torch.nn.functional.cross_entropy(logits, targets))``
    with logits equal to the log of the same probabilities; a uniform model over V
    tokens gives V.

    Examples
    --------
    >>> round(perplexity([0.25, 0.25, 0.25]), 6)
    4.0
    >>> round(perplexity([0.5, 0.25, 0.125]), 6)
    4.0
    """
    # TODO: check the probabilities, then apply the formula.
    raise NotImplementedError("perplexity() is not implemented yet")


def log_loss(y_true: ArrayLike, y_prob: ArrayLike, base: float = np.e) -> float:
    """Compute the log loss: the mean cross-entropy between true classes and predictions.

    ``-mean(log p_i)``, where ``p_i`` is the probability that the model gave to the
    true class of sample ``i``. Unlike the other functions of this module, the default
    unit is the nat (natural logarithm), as in scikit-learn and PyTorch; ``base=2``
    gives bits, and then, when no probability needs clipping, ``log_loss(y, P, base=2)``
    is the mean over the samples of ``cross_entropy(one_hot(y_i), P_i)``. Probabilities
    are first clipped to ``[eps, 1 - eps]`` with ``eps = np.finfo(float).eps``, so that a
    confident mistake gives a large but finite loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Integer labels in ``0 .. k-1`` (0 or 1 for a binary problem).
    y_prob : array-like of shape (n_samples,) or (n_samples, k)
        Binary problem: the probability of class 1 for each sample (the probability
        of class 0 is ``1 - y_prob``). Multiclass problem: one row of k class
        probabilities per sample, each row summing to 1.
    base : float, default=np.e
        Base of the logarithm: ``np.e`` for nats, 2 for bits.

    Returns
    -------
    float
        The log loss, ``>= 0``, in nats by default (lower is better).

    Raises
    ------
    ValueError
        If the lengths differ, if a label is out of range, if a probability is
        outside [0, 1], or if a row of ``y_prob`` does not sum to 1 (tolerance 1e-6).

    Notes
    -----
    Tested against ``sklearn.metrics.log_loss(y_true, y_prob, labels=range(k))``
    and ``torch.nn.functional.nll_loss(torch.log(p), y)`` (divided by ``ln 2`` for
    ``base=2``).

    Examples
    --------
    >>> round(log_loss([0, 1, 1], [0.1, 0.9, 0.8]), 4)
    0.1446
    >>> round(log_loss([0, 2], [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]), 4)
    0.4338
    >>> round(log_loss([0, 1, 1], [0.1, 0.9, 0.8], base=2), 4)
    0.2086
    """
    # TODO: validate the inputs, pick the probability of the true class of each
    # sample, clip it, then average the negative logs (in the requested base).
    raise NotImplementedError("log_loss() is not implemented yet")
