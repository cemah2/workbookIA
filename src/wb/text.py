"""Texts and real tokenizers for bonus chapter B2 (tokenization and embeddings).

Quick reference
---------------
>>> H = load_text("holmes")                  # The Adventures of Sherlock Holmes (Gutenberg header removed)
>>> part = excerpt("verne")                  # the frozen excerpt [100000:150000] of Verne (50 000 characters)
>>> train, val = train_val("verne")          # 90 % / 10 %, cut at the end of a line, never shuffled
>>> ws = words("holmes")                     # re.findall(r"[a-z]+", text.lower()): 105 863 words
>>> stoi, itos = top_vocab(ws, 4000)         # the 4 000 most frequent words, id 0 = the most frequent
>>> ids = word_ids(ws, stoi)                 # int64, -1 for a word outside the vocabulary
>>> tok = load_tokenizer("gpt2")             # transformers tokenizer at a fixed revision (files only, a few MB)
>>> tok = try_tokenizer("qwen3", "B2.14")    # None, with a ⏳ line, when it cannot be loaded
>>> n = count_tokens(tok, part)              # number of tokens, special tokens left out
>>> merges = byte_bpe_merges(train, 8000)    # the merges of mylearn's byte-level BPE ('gpt2'), learned in Rust

The tokenizers (``TOKENIZERS``) are read at the commits frozen on 2026-11-01 (``data/cards/pretrained.md``): only
their tokenizer files are downloaded (``AutoTokenizer``), never the weights of the models.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import re
from collections import Counter
from typing import Iterable

import numpy as np

from wb import datasets

# ---------------------------------------------------------------------------
# Texts, excerpts and splits (frozen: the exercises of chapter B2 and B3 rely on them)
# ---------------------------------------------------------------------------
TEXTS = ("holmes", "verne")
EXCERPT_START, EXCERPT_STOP = 100_000, 150_000
TRAIN_SHARE = 0.9
# The words of the embeddings: lower case, runs of letters (the map of the fiche for Holmes; accents kept for Verne)
WORD_PATTERNS = {"holmes": r"[a-z]+", "verne": r"[^\W\d_]+"}

_TEXTS: dict = {}


def load_text(name: str) -> str:
    """``"holmes"`` (``wb.datasets.load_holmes()``, English) or ``"verne"`` (``load_verne()``, French), read once."""
    if name not in TEXTS:
        raise ValueError(f"unknown text {name!r}; choose among {TEXTS}")
    if name not in _TEXTS:
        _TEXTS[name] = datasets.load_holmes() if name == "holmes" else datasets.load_verne()
    return _TEXTS[name]


def excerpt(name: str, start: int = EXCERPT_START, stop: int = EXCERPT_STOP) -> str:
    """The characters ``[start:stop]`` of a text (default: the 50 000 characters of the frozen excerpt)."""
    return load_text(name)[start:stop]


def split_point(text: str, share: float = TRAIN_SHARE) -> int:
    """The index where the validation part starts: the beginning of the line that holds the character at
    ``share * len(text)`` (so that no word is cut in two). Raises ValueError if share is not in (0, 1)."""
    if not 0 < share < 1:
        raise ValueError(f"share must be in (0, 1), got {share!r}")
    cut = int(share * len(text))
    newline = text.rfind("\n", 0, cut)
    return newline + 1 if newline > 0 else cut


def train_val(name: str, share: float = TRAIN_SHARE) -> tuple[str, str]:
    """``(train, val)``: the first ``share`` of the text and the rest, cut at a line boundary, never shuffled
    (a tokenizer trained on ``train`` never sees ``val``). ``train + val`` is the whole text."""
    text = load_text(name)
    cut = split_point(text, share)
    return text[:cut], text[cut:]


def words(name_or_text: str, lang: str | None = None) -> list[str]:
    """The words of a text for the embeddings, in lower case: ``"holmes"`` keeps the runs of ``a`` to ``z``
    (``[a-z]+``, as the map of the fiche), ``"verne"`` the runs of letters, accents included (``[^\\W\\d_]+``).

    ``name_or_text`` is ``"holmes"``, ``"verne"`` or a text; for a text, ``lang`` (``"holmes"`` or ``"verne"``)
    chooses the pattern."""
    if name_or_text in TEXTS and lang is None:
        lang, text = name_or_text, load_text(name_or_text)
    else:
        text = name_or_text
    if lang not in WORD_PATTERNS:
        raise ValueError(f"lang must be one of {sorted(WORD_PATTERNS)} for a text, got {lang!r}")
    return re.findall(WORD_PATTERNS[lang], text.lower())


def top_vocab(tokens: Iterable[str], size: int) -> tuple[dict[str, int], list[str]]:
    """``(stoi, itos)`` of the ``size`` most frequent tokens: id 0 is the most frequent; ties in the order of
    ``sorted`` (as ``build_vocab`` of chapter 22). Fewer tokens than ``size``: all of them."""
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 1:
        raise ValueError(f"size must be an int >= 1, got {size!r}")
    counts = Counter(tokens)
    itos = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]]
    return {t: i for i, t in enumerate(itos)}, itos


def word_ids(tokens: Iterable[str], stoi: dict[str, int]) -> np.ndarray:
    """The ids of the tokens (``int64``), ``-1`` for a token outside ``stoi`` (to be skipped, never merged)."""
    return np.array([stoi.get(t, -1) for t in tokens], dtype=np.int64)


# ---------------------------------------------------------------------------
# Real tokenizers (Hugging Face Hub, files of the tokenizer only)
# ---------------------------------------------------------------------------
TOKENIZERS = {
    "gpt2": {"repo": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
             "license": "MIT", "kind": "BPE octet (GPT-2, 2019)"},
    "bert": {"repo": "google-bert/bert-base-uncased", "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
             "license": "Apache 2.0", "kind": "WordPiece, minuscules sans accents (BERT, 2018)"},
    "e5": {"repo": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
           "license": "MIT", "kind": "SentencePiece Unigram multilingue (celui de XLM-R)"},
    "camembert": {"repo": "almanach/camembert-base", "revision": "a75967561c78f2aa81cc41045378d3b4ee25af9e",
                  "license": "MIT", "kind": "SentencePiece, appris sur du français (CamemBERT, 2020)"},
    "qwen3": {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
              "license": "Apache 2.0", "kind": "BPE octet multilingue, un chiffre par token (Qwen3, 2025)"},
    "smollm2": {"repo": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "license": "Apache 2.0", "kind": "BPE octet, surtout anglais, un chiffre par token (SmolLM2, 2024)"},
}


class TokenizerUnavailable(RuntimeError):
    """A tokenizer cannot be loaded here (no network and no cache, a missing library)."""


@contextlib.contextmanager
def quiet_transformers():
    """No info log, warning or progress bar of transformers and huggingface_hub (restored afterwards): in
    particular not « Token indices sequence length is longer than the specified maximum », which is about the
    model (a count of tokens never goes through it)."""
    restore = []
    try:
        logs = importlib.import_module("transformers.utils.logging")
        level = logs.get_verbosity()
        logs.set_verbosity_error()
        restore.append(lambda: logs.set_verbosity(level))
    except Exception:  # noqa: BLE001 - transformers absent or another version
        pass
    try:
        with datasets.quiet_huggingface():
            yield
    finally:
        for undo in restore:
            with contextlib.suppress(Exception):
                undo()


_LOADED: dict = {}


def load_tokenizer(name: str, *, reuse: bool = True):
    """The ``transformers`` tokenizer ``name`` (a key of ``TOKENIZERS``) at its frozen revision, loaded once per
    session (``reuse=False`` loads it again). Raises :class:`TokenizerUnavailable` with the reason when its
    files cannot be read or downloaded."""
    if name not in TOKENIZERS:
        raise ValueError(f"unknown tokenizer {name!r}; choose among {sorted(TOKENIZERS)}")
    if reuse and name in _LOADED:
        return _LOADED[name]
    spec = TOKENIZERS[name]
    try:
        with quiet_transformers():
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    except Exception as exc:  # noqa: BLE001 - no network, a proxy, a missing package: the caller falls back
        raise TokenizerUnavailable(f"tokenizer {name} ({spec['repo']}) impossible à charger : "
                                   f"{type(exc).__name__} : {str(exc).strip()[:300]}") from exc
    _LOADED[name] = tok
    return tok


def try_tokenizer(name: str, ex_id: str | None = None):
    """``load_tokenizer(name)``, or None with a ⏳ line when it cannot be loaded (the exercises that need it then
    say so instead of crashing; to be validated on Colab)."""
    try:
        return load_tokenizer(name)
    except TokenizerUnavailable as exc:
        where = f"Ex {ex_id} : " if ex_id else ""
        print(f"⏳ {where}{exc}. Sans ce tokenizer, cette partie est « à valider sur Colab » ; le reste du notebook "
              "continue.")
        return None


def count_tokens(tok, text: str) -> int:
    """The number of tokens of ``text`` for a ``transformers`` tokenizer, special tokens left out
    (``add_special_tokens=False``: no ``[CLS]``, ``[SEP]``, ``<s>``, ``</s>``)."""
    with quiet_transformers():
        return len(tok(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# The merges of a byte-level BPE learned by the library tokenizers (fast), exactly those of mylearn's BPETokenizer
# ---------------------------------------------------------------------------
# The pattern of the 'gpt2' pre-tokenizer of mylearn.tokenizer (the stub of chapter B2, never changed)
GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
_CJK = 0x4E00          # byte b <-> chr(0x4E00 + b): ideographs, all in \w, in the order of the bytes


class MergeMismatch(RuntimeError):
    """tokenizers learned a merge that gives back an existing token (it then reuses its id, BPETokenizer does not)."""


def byte_bpe_merges(texts: str | Iterable[str], vocab_size: int, min_frequency: int = 2) -> list[tuple[int, int]]:
    """The merges that ``BPETokenizer(vocab_size, level="byte", pre_tokenizer="gpt2", min_frequency)`` of mylearn
    (without special token) learns on ``texts``, as ``(left_id, right_id)`` pairs in learning order, computed by
    ``tokenizers.trainers.BpeTrainer`` (Rust, 50 to 100 times faster than Python).

    The exact oracle of ``tests/test_chb2_tokenizer.py``: each byte ``b`` of a GPT-2 pre-token becomes the character
    ``chr(0x4E00 + b)``, the 256 characters are the initial alphabet (ids 0 to 255, in the order of the bytes), and
    the ties of ``BpeTrainer`` go to the smallest pair of ids, as in ``BPETokenizer.fit``. Raises
    :class:`MergeMismatch` in the rare case where ``tokenizers`` reuses an id for a merge."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    if isinstance(texts, str):
        texts = [texts]
    pattern = re.compile(GPT2_PATTERN)
    corpus = [" ".join("".join(chr(_CJK + b) for b in piece.encode("utf-8")) for piece in pattern.findall(text))
              for text in texts]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, show_progress=False,
                                  initial_alphabet=[chr(_CJK + b) for b in range(256)])
    tok.train_from_iterator(corpus, trainer)
    state = json.loads(tok.to_str())["model"]
    vocab = state["vocab"]
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in state["merges"]]
    if any(vocab.get(chr(_CJK + b)) != b for b in range(256)) or len(vocab) != 256 + len(merges):
        raise MergeMismatch("tokenizers reused a token for a merge: train BPETokenizer.fit instead")
    return [(vocab[a], vocab[b]) for a, b in merges]
