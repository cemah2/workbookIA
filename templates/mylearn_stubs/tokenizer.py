"""Byte-pair encoding tokenizer — mylearn, chapter B2 (Tokenization and embeddings).

A BPE tokenizer written from scratch with the standard library: pair counting, merging,
training (``BPETokenizer.fit``), encoding by merge rank, decoding (lossless at byte level
with the GPT-2 pre-tokenizer) and JSON saving. Ties are broken like Hugging Face
``tokenizers``, so that your merges can be compared exactly with a real tokenizer. The
tokenizer is reused in chapters B3 (a mini-GPT on BPE tokens) and B4.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: Hugging Face tokenizers and plain Python counts).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Self, Sequence

# Pre-tokenization patterns (used by the provided function ``pre_tokenize``).
# Same pattern as tokenizers.pre_tokenizers.Whitespace: words, or runs of punctuation.
WHITESPACE_PATTERN = r"\w+|[^\w\s]+"
# GPT-2 pattern, approximated with the standard ``re`` module: GPT-2 uses \p{L} (letters)
# and \p{N} (numbers), which ``re`` does not support; here [^\W\d_] stands for letters and
# \d for digits. Every character belongs to one piece, so joining the pieces gives the text.
GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"


def pre_tokenize(
    text: str, pre_tokenizer: str | None = "whitespace", special_tokens: Sequence[str] = ()
) -> list[str]:
    """Split a text into pre-tokens; BPE merges never cross a pre-token boundary.

    Provided: already implemented, no exercise.

    The text is first cut around the special tokens, which are kept whole as their own
    pieces (the longest special token wins when several match at the same place). The
    other parts are split with ``WHITESPACE_PATTERN`` ('whitespace'), ``GPT2_PATTERN``
    ('gpt2') or kept whole (None).

    Parameters
    ----------
    text : str
        Text to split.
    pre_tokenizer : {'whitespace', 'gpt2'} or None, default='whitespace'
        'whitespace' drops the spaces (like ``tokenizers.pre_tokenizers.Whitespace``);
        'gpt2' keeps them, attached to the start of the next word, so that
        ``"".join(pieces) == text``; None keeps each part whole.
    special_tokens : sequence of str, default=()
        Strings that must never be split (e.g. ``'<|endoftext|>'``).

    Returns
    -------
    list of str
        The pieces, in order. Special tokens appear as pieces of their own.

    Raises
    ------
    ValueError
        If ``pre_tokenizer`` is not 'whitespace', 'gpt2' or None.

    Examples
    --------
    >>> pre_tokenize("Hello, world!")
    ['Hello', ',', 'world', '!']
    >>> pre_tokenize("Hello, world!", "gpt2")
    ['Hello', ',', ' world', '!']
    >>> pre_tokenize("Hi you<|endoftext|>Bye", "gpt2", special_tokens=["<|endoftext|>"])
    ['Hi', ' you', '<|endoftext|>', 'Bye']
    """
    if pre_tokenizer == "whitespace":
        pattern = WHITESPACE_PATTERN
    elif pre_tokenizer == "gpt2":
        pattern = GPT2_PATTERN
    elif pre_tokenizer is None:
        pattern = None
    else:
        raise ValueError(
            f"pre_tokenizer must be 'whitespace', 'gpt2' or None, not {pre_tokenizer!r}"
        )
    specials = sorted({s for s in special_tokens if s}, key=lambda s: (-len(s), s))
    if specials:
        parts = re.split("(" + "|".join(re.escape(s) for s in specials) + ")", text)
    else:
        parts = [text]
    pieces: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # re.split puts the captured separators at odd positions
            pieces.append(part)
        elif pattern is None:
            if part:
                pieces.append(part)
        else:
            pieces.extend(re.findall(pattern, part))
    return pieces


def get_pair_counts(words: Mapping[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """Count adjacent id pairs over a corpus given as {word as a tuple of ids: frequency}.

    Each word contributes ``freq`` to every pair of adjacent ids it contains (overlapping
    pairs included: the word ``(1, 1, 1)`` holds the pair ``(1, 1)`` twice). Pairs never
    cross word boundaries.

    Parameters
    ----------
    words : mapping of tuple of int to int
        Pre-tokenized corpus: each word, written as a tuple of token ids, is mapped to its
        number of occurrences.

    Returns
    -------
    dict of (int, int) to int
        ``{(left_id, right_id): weighted count}``, only for pairs that occur. The counts
        sum to ``sum(freq * (len(word) - 1))`` over the words.

    Raises
    ------
    ValueError
        If a frequency is negative.

    Notes
    -----
    Tested against a brute-force count over the expanded corpus (``collections.Counter``).

    Examples
    --------
    >>> get_pair_counts({(1, 2, 3): 2, (1, 2): 1})
    {(1, 2): 3, (2, 3): 2}
    """
    # TODO: loop over the words and their adjacent pairs, adding the word frequency.
    raise NotImplementedError("get_pair_counts() is not implemented yet")


def merge_pair(ids: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace each non-overlapping occurrence of ``pair`` (left to right) by ``new_id``.

    Parameters
    ----------
    ids : sequence of int
        Token ids of one word or text.
    pair : tuple of (int, int)
        ``(left_id, right_id)``, the two adjacent ids to merge.
    new_id : int
        Id of the merged token.

    Returns
    -------
    list of int
        A new list of ids; the input is not modified.

    Raises
    ------
    ValueError
        If ``pair`` does not contain exactly two ids.

    Notes
    -----
    Tested on hand-made cases, and with the property that the bytes of the tokens,
    joined, do not change.

    Examples
    --------
    >>> merge_pair([3, 1, 2, 1, 2], (1, 2), 7)
    [3, 7, 7]
    >>> merge_pair([1, 1, 1], (1, 1), 5)
    [5, 1]
    """
    # TODO: walk through ids with an index; when the pair starts at the index, emit new_id
    # and jump over both ids.
    raise NotImplementedError("merge_pair() is not implemented yet")


class BPETokenizer:
    """Byte-pair encoding (BPE) tokenizer trained from scratch.

    Ids ``0 .. n_base_ - 1`` are the base alphabet: the 256 byte values (``level='byte'``,
    id ``i`` is the byte ``i``) or the characters seen during ``fit`` sorted by code point
    (``level='char'``). Then comes one id per merge, in learning order (merge ``i`` creates
    id ``n_base_ + i``), then the special tokens.

    Parameters
    ----------
    vocab_size : int, default=1000
        Target total size: base alphabet + merges + special tokens (as in
        ``tokenizers.trainers.BpeTrainer``). Training may stop before reaching it.
    level : {'byte', 'char'}, default='byte'
        'byte' works on UTF-8 bytes (any text can be encoded, no unknown token);
        'char' works on the Unicode characters seen during ``fit``.
    pre_tokenizer : {'whitespace', 'gpt2'} or None, default='whitespace'
        How ``pre_tokenize`` splits the text before BPE. 'whitespace' drops the spaces
        (``decode`` is then not lossless); 'gpt2' keeps them (lossless at byte level);
        None applies BPE to each whole text.
    min_frequency : int, default=2
        Training stops as soon as the most frequent pair occurs fewer times than this.
    special_tokens : sequence of str, default=()
        Strings that ``encode`` never splits (e.g. ``'<|endoftext|>'``). They get the last
        ids, in the given order, and take no part in the merges.
    unk_token : str or None, default=None
        Char level only: token used for each character unseen during ``fit``. If it is not
        already in ``special_tokens``, it is added after them.

    Attributes
    ----------
    merges_ : list of tuple of (int, int)
        The learned merges ``(left_id, right_id)``, in learning order (= rank).
    vocab_ : dict of int to bytes
        The bytes of every token: ``vocab_[n_base_ + i] = vocab_[a] + vocab_[b]`` where
        ``(a, b) = merges_[i]``; a character or a special token is stored as its UTF-8 bytes.
    n_base_ : int
        Size of the base alphabet (256 at byte level).
    special_ids_ : dict of str to int
        Id of each special token (``unk_token`` included).

    Notes
    -----
    ``encode``, ``decode`` and ``tokenize`` must rely only on the hyperparameters and on
    the four learned attributes above, so that ``load`` can rebuild a working tokenizer.

    Tests: at char level with ``pre_tokenizer='whitespace'`` and no special token,
    ``merges_`` and the ids are identical to those of ``tokenizers`` (``models.BPE``
    trained by ``trainers.BpeTrainer`` with the same ``vocab_size`` and ``min_frequency``,
    ``pre_tokenizers.Whitespace``). At byte level with ``pre_tokenizer='gpt2'``,
    ``decode(encode(s)) == s`` for random Unicode strings, and the number of tokens stays
    within 2 % of a ``tokenizers`` ByteLevel BPE of the same size.

    Two differences with ``tokenizers``: it gives the first ids to the special tokens (here
    they come last), and when a merge produces a string that is already a token (rare), it
    reuses that id (here every merge creates a new id).

    Examples
    --------
    >>> tok = BPETokenizer(vocab_size=16, level="char").fit("low lower lowest newer newest")
    >>> tok.n_base_, len(tok.merges_)
    (8, 6)
    >>> tok.merges_[:3]
    [(7, 0), (1, 3), (0, 8)]
    >>> tok.vocab_[8], tok.vocab_[9], tok.vocab_[10]
    (b'we', b'lo', b'ewe')
    >>> tok.tokenize("lowest newer")
    ['lowe', 'st', 'newe', 'r']
    >>> tok.encode("lowest")
    [13, 12]
    >>> tok.decode([13, 12])
    'lowest'

    At byte level with the GPT-2 pre-tokenizer, decoding gives back the exact text:

    >>> byte_tok = BPETokenizer(vocab_size=260, pre_tokenizer="gpt2",
    ...                         special_tokens=["<|endoftext|>"])
    >>> byte_tok = byte_tok.fit("low lower lowest newer newest")
    >>> byte_tok.merges_
    [(119, 101), (108, 111), (32, 110)]
    >>> byte_tok.special_ids_
    {'<|endoftext|>': 259}
    >>> ids = byte_tok.encode("lowest<|endoftext|> été")
    >>> ids[:5]
    [257, 256, 115, 116, 259]
    >>> byte_tok.decode(ids)
    'lowest<|endoftext|> été'
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        level: str = "byte",
        pre_tokenizer: str | None = "whitespace",
        min_frequency: int = 2,
        special_tokens: Sequence[str] = (),
        unk_token: str | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.level = level
        self.pre_tokenizer = pre_tokenizer
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.unk_token = unk_token

    def fit(self, texts: str | Iterable[str]) -> Self:
        """Learn the merges from one text or an iterable of texts.

        The texts are cut into pre-tokens by ``pre_tokenize`` (special tokens are left
        out), and identical pre-tokens are counted once with their frequency. Each step
        merges the most frequent adjacent pair; ties go to the smallest pair
        ``(left_id, right_id)``. Training stops when ``n_base_ + number of merges +
        number of special tokens`` reaches ``vocab_size``, when no pair is left, or when
        the best pair occurs fewer than ``min_frequency`` times.

        Parameters
        ----------
        texts : str or iterable of str
            Training corpus. A single ``str`` is one text (not an iterable of characters).

        Returns
        -------
        BPETokenizer
            Returns self, with ``merges_``, ``vocab_``, ``n_base_`` and ``special_ids_``.

        Raises
        ------
        ValueError
            If ``level`` or ``pre_tokenizer`` is unknown, if ``unk_token`` is set at byte
            level, or if ``vocab_size < n_base_ + number of special tokens``.

        Examples
        --------
        >>> tok = BPETokenizer(vocab_size=16, level="char").fit(["low lower", "lowest"])
        >>> [tok.vocab_[i] for i in range(tok.n_base_, len(tok.vocab_))]
        [b'lo', b'low', b'lowe']
        """
        # TODO: build the word frequencies and the base alphabet, then repeat:
        # count the pairs, pick the best one, merge it in every word.
        raise NotImplementedError("fit() is not implemented yet")

    def encode(self, text: str) -> list[int]:
        """Convert a text into token ids.

        Special tokens (the keys of ``special_ids_``) become their id. Inside each other
        pre-token (never across two), start from the base ids (UTF-8 bytes, or
        characters), then repeatedly merge the adjacent pair of lowest rank (the earliest
        learned merge) until no adjacent pair is in ``merges_``. At char level, each
        unseen character becomes the id of ``unk_token`` (one id per character).

        Parameters
        ----------
        text : str
            Text to encode.

        Returns
        -------
        list of int
            Token ids.

        Raises
        ------
        ValueError
            If the tokenizer is not fitted, or at char level if the text contains a
            character unseen during ``fit`` and ``unk_token`` is None.

        Examples
        --------
        >>> tok = BPETokenizer(vocab_size=16, level="char", unk_token="<unk>")
        >>> tok = tok.fit("low lower lowest newer newest")
        >>> tok.special_ids_
        {'<unk>': 14}
        >>> tok.encode("lowly")
        [9, 7, 1, 14]
        """
        # TODO: pre-tokenize, then apply the merges by rank inside each pre-token.
        raise NotImplementedError("encode() is not implemented yet")

    def decode(self, ids: Sequence[int]) -> str:
        """Convert token ids back into a text.

        The bytes of the tokens are joined, then decoded as UTF-8 with
        ``errors='replace'``: an incomplete multi-byte character becomes the replacement
        character '�' (U+FFFD) instead of raising an error.

        Parameters
        ----------
        ids : sequence of int
            Token ids.

        Returns
        -------
        str
            The text. With ``pre_tokenizer='gpt2'`` at byte level,
            ``decode(encode(s)) == s``; with 'whitespace' the spaces are lost.

        Raises
        ------
        ValueError
            If the tokenizer is not fitted or an id is not in ``vocab_``.

        Examples
        --------
        >>> tok = BPETokenizer(vocab_size=258, pre_tokenizer="gpt2").fit("été été")
        >>> tok.decode(tok.encode("été"))
        'été'
        >>> tok.decode([195])  # first byte of 'é' alone
        '�'
        """
        # TODO: join the bytes of the tokens, then decode them.
        raise NotImplementedError("decode() is not implemented yet")

    def tokenize(self, text: str) -> list[str]:
        """Encode a text and return the tokens as readable strings.

        Each token's bytes are decoded as UTF-8 with ``errors='replace'``, so a token
        holding part of a multi-byte character shows as '�'.

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        -------
        list of str
            One string per token id of ``encode(text)``.

        Raises
        ------
        ValueError
            As ``encode``.

        Examples
        --------
        >>> tok = BPETokenizer(vocab_size=16, level="char").fit("low lower lowest newer newest")
        >>> tok.tokenize("newest lower")
        ['newe', 'st', 'lowe', 'r']
        """
        # TODO: reuse encode, then vocab_.
        raise NotImplementedError("tokenize() is not implemented yet")

    def save(self, path: str | Path) -> None:
        """Save the fitted tokenizer to a JSON file.

        Provided: already implemented, no exercise.

        The file holds the hyperparameters, the base alphabet (char level), ``merges_``
        and ``special_ids_``; ``vocab_`` is rebuilt from them by ``load``.

        Parameters
        ----------
        path : str or Path
            Destination file (overwritten if it exists).

        Raises
        ------
        ValueError
            If the tokenizer is not fitted.
        """
        if not hasattr(self, "merges_"):
            raise ValueError("this BPETokenizer is not fitted yet: call fit() first")
        alphabet = None
        if self.level == "char":
            alphabet = [self.vocab_[i].decode("utf-8") for i in range(self.n_base_)]
        state = {
            "format": "mylearn.BPETokenizer",
            "version": 1,
            "params": {
                "vocab_size": self.vocab_size,
                "level": self.level,
                "pre_tokenizer": self.pre_tokenizer,
                "min_frequency": self.min_frequency,
                "special_tokens": list(self.special_tokens),
                "unk_token": self.unk_token,
            },
            "alphabet": alphabet,
            "merges": [[int(a), int(b)] for a, b in self.merges_],
            "special_ids": {token: int(i) for token, i in self.special_ids_.items()},
        }
        text = json.dumps(state, ensure_ascii=False, indent=1)
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a tokenizer saved by ``save``.

        Provided: already implemented, no exercise.

        Parameters
        ----------
        path : str or Path
            JSON file written by ``save``.

        Returns
        -------
        BPETokenizer
            A fitted tokenizer: ``encode`` and ``decode`` give the same results as the
            saved one.

        Examples
        --------
        >>> import tempfile, os
        >>> tok = BPETokenizer(vocab_size=16, level="char").fit("low lower lowest")
        >>> path = os.path.join(tempfile.mkdtemp(), "bpe.json")
        >>> tok.save(path)
        >>> BPETokenizer.load(path).merges_ == tok.merges_
        True
        """
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(**state["params"])
        if state["alphabet"] is None:  # byte level: id i is the byte i
            vocab = {i: bytes([i]) for i in range(256)}
        else:
            vocab = {i: char.encode("utf-8") for i, char in enumerate(state["alphabet"])}
        tok.n_base_ = len(vocab)
        tok.merges_ = [(int(a), int(b)) for a, b in state["merges"]]
        for new_id, (a, b) in enumerate(tok.merges_, start=tok.n_base_):
            vocab[new_id] = vocab[a] + vocab[b]
        tok.special_ids_ = {token: int(i) for token, i in state["special_ids"].items()}
        for token, i in tok.special_ids_.items():
            vocab[i] = token.encode("utf-8")
        tok.vocab_ = vocab
        return tok
