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
>>> E, stoi, itos = reference_ppmi_svd(words("holmes"))   # PPMI + SVD word vectors of the fiche (check of B2.24)
>>> sm = try_sentence_model("minilm", "B2.28")  # all-MiniLM-L6-v2 at a fixed revision, frozen; None with ⏳
>>> S = sm.embed(["a text", "another"])     # mean pooling with the mask, norm 1: float32 (2, 384)
>>> passages, story = holmes_passages()     # 1 055 passages of at most 120 words, and their story (0 to 11)
>>> P = passage_embeddings(sm, passages, "holmes-120")   # computed once, then read from the cache

The tokenizers (``TOKENIZERS``) are read at the commits frozen on 2026-11-01 (``data/cards/pretrained.md``): only
their tokenizer files are downloaded (``AutoTokenizer``), never the weights of the models. The two sentence-embedding
models (``SENTENCE_MODELS``: all-MiniLM-L6-v2, ≈ 90 MB, and multilingual-e5-small, ≈ 470 MB, an option) are read at
frozen commits too.
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


# ---------------------------------------------------------------------------
# Word vectors by counting (📦 B2.24): the reference of the check cell, computed with scipy.sparse
# ---------------------------------------------------------------------------
def reference_ppmi_svd(tokens: list[str], n_words: int = 4000, window: int = 2, dim: int = 100,
                       alpha: float = 0.75) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """``(E, stoi, itos)``: the PPMI + SVD word vectors of the fiche (§202.8), the reference of 📦 B2.24.

    The vocabulary is ``top_vocab(tokens, n_words)``; a word outside it keeps its place in the text but its pairs are
    not counted (skipped, never merged: two words far apart never become neighbours). Counts in a symmetric window
    with the uniform weighting, PPMI with the natural log and the context smoothing ``alpha``, then
    ``TruncatedSVD(dim, algorithm="arpack", random_state=0)``; the rows of ``U_d S_d^(1/2)``, normalised to norm 1,
    are the vectors (``float64``, shape ``(len(itos), dim)``)."""
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD

    stoi, itos = top_vocab(tokens, n_words)
    v = len(itos)
    ids = word_ids(tokens, stoi)
    rows, cols = [], []
    for d in range(1, window + 1):
        a, b = ids[:-d], ids[d:]
        ok = (a >= 0) & (b >= 0)
        rows += [a[ok], b[ok]]
        cols += [b[ok], a[ok]]
    r, c = np.concatenate(rows), np.concatenate(cols)
    counts = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(v, v)).tocsr().tocoo()   # duplicates summed
    total = counts.sum()
    p_w = np.asarray(counts.tocsr().sum(axis=1)).ravel() / total
    c_alpha = np.asarray(counts.tocsr().sum(axis=0)).ravel() ** alpha
    p_c = c_alpha / c_alpha.sum()
    pmi = np.log((counts.data / total) / (p_w[counts.row] * p_c[counts.col]))
    keep = pmi > 0
    M = sp.csr_matrix((pmi[keep], (counts.row[keep], counts.col[keep])), shape=(v, v))
    svd = TruncatedSVD(dim, algorithm="arpack", random_state=0)
    E = svd.fit_transform(M) / np.sqrt(svd.singular_values_)
    return E / np.linalg.norm(E, axis=1, keepdims=True), stoi, itos


# ---------------------------------------------------------------------------
# Sentence embeddings (📦 B2.28, 🔬 B2.29): frozen models, mean pooling, the passages of Holmes
# ---------------------------------------------------------------------------
SENTENCE_MODELS = {
    "minilm": {"repo": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
               "license": "Apache 2.0", "dim": 384, "max_length": 256, "query": "", "passage": "",
               "kind": "anglais, 6 couches, 22,7 M de paramètres (≈ 90 Mo)"},
    "e5": {"repo": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
           "license": "MIT", "dim": 384, "max_length": 512, "query": "query: ", "passage": "passage: ",
           "kind": "une centaine de langues, 12 couches, 117,7 M de paramètres (≈ 470 Mo)"},
}
POOLING = "mean pooling with the attention mask, then norm 1"


class SentenceModelUnavailable(RuntimeError):
    """The weights of a sentence-embedding model cannot be loaded here (no network and no cache)."""


class SentenceModel:
    """A frozen sentence-embedding model of ``SENTENCE_MODELS``: ``tokenizer`` and ``model`` (``AutoModel`` in
    ``eval()``, no gradient), with its revision, its ``max_length`` and its prefixes (``query``, ``passage``)."""

    def __init__(self, name: str, tokenizer, model):
        spec = SENTENCE_MODELS[name]
        self.name, self.tokenizer, self.model = name, tokenizer, model
        self.repo, self.revision, self.dim = spec["repo"], spec["revision"], spec["dim"]
        self.max_length, self.prefixes = spec["max_length"], {"query": spec["query"], "passage": spec["passage"]}

    def __repr__(self) -> str:
        return f"SentenceModel({self.name!r}: {self.repo} @ {self.revision[:7]}, d = {self.dim}, max_length = {self.max_length})"

    def embed(self, texts, kind: str | None = None, batch_size: int = 32) -> np.ndarray:
        """float32 ``(N, dim)``: the reference sentence embeddings (fiche §202.10), mean pooling with the attention
        mask then norm 1, texts truncated at ``max_length`` tokens; ``kind`` (``"query"`` or ``"passage"``) adds the
        prefix of the model (e5). Texts are encoded by batches of similar lengths (less padding), in their order."""
        import torch
        import torch.nn.functional as F

        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if kind not in (None, "query", "passage"):
            raise ValueError(f"kind must be None, 'query' or 'passage', got {kind!r}")
        prefix = self.prefixes[kind] if kind else ""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        with quiet_transformers(), torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                idx = order[start:start + batch_size]
                batch = self.tokenizer([prefix + texts[i] for i in idx], padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt")
                h = self.model(**batch).last_hidden_state
                m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                out[idx] = F.normalize((h * m).sum(dim=1) / m.sum(dim=1), dim=1).float().numpy()
        return out

    def word_vector(self, sentence: str, word: str) -> np.ndarray:
        """The contextual vector of ``word`` in ``sentence`` (🔬 B2.29): the output of the last block for the tokens of
        its first occurrence (a whole word, found with the offsets of the tokenizer), averaged: float32 ``(dim,)``.
        Raises ValueError when ``word`` is not in ``sentence``."""
        import torch

        found = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", sentence)
        if found is None:
            raise ValueError(f"{word!r} is not a word of {sentence!r}")
        batch = self.tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True,
                               max_length=self.max_length)
        offsets = batch.pop("offset_mapping")[0].tolist()
        keep = [k for k, (a, b) in enumerate(offsets) if b > a and a < found.end() and b > found.start()]
        with quiet_transformers(), torch.inference_mode():
            h = self.model(**batch).last_hidden_state[0]
        return h[keep].mean(dim=0).float().numpy()

    def static_vector(self, word: str) -> np.ndarray:
        """The row of the input table (``model.embeddings.word_embeddings``) of ``word``, averaged over its tokens
        when it has several: the same vector in every sentence, float32 ``(dim,)``."""
        ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
        table = self.model.get_input_embeddings().weight
        return table[ids].mean(dim=0).detach().float().numpy()


_SENTENCE_LOADED: dict = {}


def load_sentence_model(name: str = "minilm", *, reuse: bool = True) -> SentenceModel:
    """The sentence-embedding model ``name`` (a key of ``SENTENCE_MODELS``) at its frozen revision, frozen in
    ``eval()``, loaded once per session. Raises :class:`SentenceModelUnavailable` with the reason when its files
    cannot be read or downloaded."""
    if name not in SENTENCE_MODELS:
        raise ValueError(f"unknown sentence model {name!r}; choose among {sorted(SENTENCE_MODELS)}")
    if reuse and name in _SENTENCE_LOADED:
        return _SENTENCE_LOADED[name]
    spec = SENTENCE_MODELS[name]
    try:
        with quiet_transformers():
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
            model = AutoModel.from_pretrained(spec["repo"], revision=spec["revision"]).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    except Exception as exc:  # noqa: BLE001 - no network, a proxy, a missing package: the caller falls back
        raise SentenceModelUnavailable(f"modèle {name} ({spec['repo']}) impossible à charger : "
                                       f"{type(exc).__name__} : {str(exc).strip()[:300]}") from exc
    _SENTENCE_LOADED[name] = SentenceModel(name, tok, model)
    return _SENTENCE_LOADED[name]


def try_sentence_model(name: str = "minilm", ex_id: str | None = None) -> SentenceModel | None:
    """``load_sentence_model(name)``, or None with a ⏳ line when it cannot be loaded (to be validated on Colab)."""
    try:
        return load_sentence_model(name)
    except SentenceModelUnavailable as exc:
        where = f"Ex {ex_id} : " if ex_id else ""
        print(f"⏳ {where}{exc}. Sans ce modèle, cette partie est « à valider sur Colab » ; le reste du notebook "
              "continue.")
        return None


_STORY_TITLE = re.compile(r"^([IVX]+)\. ([A-Z][A-Z \-'’]+)$", flags=re.M)


def holmes_stories() -> list[tuple[str, str]]:
    """The twelve stories of Holmes, ``[(short title, text)]`` in the order of the book (« A Scandal in Bohemia »
    … « The Copper Beeches »: « The Adventure of the » is dropped from the titles)."""
    text = load_text("holmes")
    heads = list(_STORY_TITLE.finditer(text))
    stories = []
    for k, head in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        title = head.group(2).title().replace("’S", "’s").replace("The Adventure Of The ", "The ")
        title = re.sub(r"\b(In|Of|With|The)\b", lambda m: m.group(0).lower(), title)
        stories.append((title[0].upper() + title[1:], text[head.end():end]))
    return stories


def holmes_passages(max_words: int = 120) -> tuple[list[str], np.ndarray]:
    """``(passages, story)``: Holmes cut into passages of at most ``max_words`` words (frozen for 📦 B2.28), and the
    index of the story of each passage (``holmes_stories``). Consecutive paragraphs are packed together, a longer
    paragraph is cut between sentences; line breaks become spaces. With 120 words, 1 055 passages, all under the 256
    tokens that all-MiniLM-L6-v2 reads."""
    passages, story = [], []
    for s, (_, body) in enumerate(holmes_stories()):
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body)]
        paragraphs = [p for p in paragraphs if p and not re.fullmatch(r"[IVX]+\.", p)]
        current: list[str] = []
        for paragraph in paragraphs:
            pieces = [paragraph] if len(paragraph.split()) <= max_words else re.split(r"(?<=[.!?”])\s+", paragraph)
            for piece in pieces:
                if current and len(" ".join(current).split()) + len(piece.split()) > max_words:
                    passages.append(" ".join(current))
                    story.append(s)
                    current = []
                current.append(piece)
        if current:
            passages.append(" ".join(current))
            story.append(s)
    return passages, np.array(story, dtype=np.int64)


def passage_embeddings(sm: SentenceModel, passages: list[str], subset: str, kind: str | None = "passage",
                       folder=None, verbose: bool = True) -> np.ndarray:
    """``sm.embed(passages, kind)``, computed once and then read from the embeddings cache of ``wb.transfer``
    (``data/downloads/embeddings``, never versioned), under a key made of the model, its revision, ``max_length``,
    the pooling, the prefix and a fingerprint of the passages (≈ 20 s for the 1 055 passages of Holmes on 2 cores)."""
    import hashlib

    from wb import transfer

    digest = hashlib.sha256("\x00".join(passages).encode("utf-8")).hexdigest()[:16]
    key = {"version": 1, "model": sm.repo, "revision": sm.revision, "max_length": sm.max_length, "pooling": POOLING,
           "prefix": sm.prefixes[kind] if kind else "", "subset": subset, "n": len(passages), "passages": digest}
    arrays = transfer.cached_arrays(key, lambda: {"embeddings": sm.embed(passages, kind)}, folder=folder,
                                    verbose=verbose)
    if arrays["embeddings"].shape != (len(passages), sm.dim):     # never hand back a partial matrix
        raise RuntimeError(f"cached embeddings have the shape {arrays['embeddings'].shape}, not ({len(passages)}, {sm.dim})")
    return arrays["embeddings"]
