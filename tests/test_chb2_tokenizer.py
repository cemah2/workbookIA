"""Tests of mylearn.tokenizer (bonus chapter B2): oracle tests and properties.

    pytest tests/test_chb2_tokenizer.py              # your code (mon_travail/mylearn/tokenizer.py)
    pytest tests/test_chb2_tokenizer.py --impl=ref   # the reference

Oracles: a brute-force count over the expanded corpus with ``collections.Counter`` (pair counts); ``str.replace``,
which replaces the non-overlapping occurrences from left to right (merges); Hugging Face ``tokenizers`` 0.23
(``models.BPE`` trained by ``trainers.BpeTrainer``, ``pre_tokenizers.Whitespace``): at char level, the same
merges, the same ids and the same tokens, with the same ``vocab_size`` and ``min_frequency``. At byte level,
``tokenizers`` gives its own ids to the bytes (``pre_tokenizers.ByteLevel`` maps them to other characters), so the
exact oracle relabels them: each byte ``b`` becomes the character ``chr(0x4E00 + b)`` (a CJK ideograph, so that
the order of the bytes is the order of the characters) and the 256 characters form the initial alphabet; the
pre-tokens come from the GPT-2 pattern of the docstring, copied in this file. Then, as the docstring says: the
round trip ``decode(encode(s)) == s`` on random Unicode strings, and a number of tokens within 2 % of a
``tokenizers`` ByteLevel BPE of the same size. The small cases of ``tokenizers`` are also written in this file
(checked against the library when it is installed); the large ones need it (``pip install tokenizers==0.23.3``).
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_get_pair_counts_ or test_merge_pair_"`` (B2.15),
``-k "test_pre_tokenize_ or test_bpe_fit_"`` (B2.16),
``-k "test_bpe_encode_ or test_bpe_decode_ or test_bpe_tokenize_ or test_bpe_save_load_"`` (B2.17).

Contract precisions written in the statements of these exercises and checked here: ``get_pair_counts`` accepts
any mapping (read only) and returns a dict of ``(left, right)`` tuples; ``merge_pair`` returns a new ``list`` and
accepts a tuple; ``fit`` returns self, takes a single ``str`` as one text or any iterable of str (a generator
too), and leaves the list of texts unchanged; with special tokens, the merges are those of the texts with the
special tokens taken out; ``unk_token`` gets one id even when it is also in ``special_tokens``; ``encode`` returns
a list of int (``[]`` for an empty text) and encodes a literal ``unk_token`` in the text as its id; ``decode``
raises ``ValueError`` (not ``KeyError``) for an unknown id, -1 included; ``encode`` and ``decode`` work on a
tokenizer rebuilt by ``load`` (they use only the hyperparameters and the four learned attributes).
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

from __future__ import annotations

import functools
import json
import numbers
import random
import re
import types
from collections import Counter

import pytest

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError:  # the small cases below are written in this file; the large ones need the library
    Tokenizer = None

needs_tokenizers = pytest.mark.skipif(
    Tokenizer is None, reason="oracle Hugging Face tokenizers absent : pip install tokenizers==0.23.3")

ORACLE_BUG = ("bug in tests/test_chb2_tokenizer.py (not in your code): the oracle of this test "
              "disagrees with another one")

# The patterns of the docstring, copied here so that the oracles do not depend on the module under test.
WHITESPACE = r"\w+|[^\w\s]+"
GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
CJK = 0x4E00          # byte b <-> chr(CJK + b): ideographs, all in \w, in the order of the bytes

DOC_CORPUS = "low lower lowest newer newest"


@pytest.fixture
def tk(mylearn_module):
    return mylearn_module("tokenizer")


# ------------------------------------------------------------------ helpers
def _fail(msg: str, head: str, data: str = ""):
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


def assert_raises(error, function, *args, why="", **kwargs):
    name = getattr(function, "__name__", "the function")
    try:
        function(*args, **kwargs)
    except NotImplementedError:
        raise
    except error:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else "")
                             + f" (it raised {type(exc).__name__}: {exc})") from None
    raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else ""))


def call(function, *args, expected="a result", **kwargs):
    try:
        return function(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def _first_diff(got, want) -> int:
    return next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))


def _at(seq, i):
    return seq[i] if i < len(seq) else "nothing"


def assert_id_list(result, want, msg):
    """``result`` is a list of int equal to ``want``."""
    if result is None:
        _fail(msg, "expected a list of ids, got None (did you forget the return?)")
    if not isinstance(result, list):
        _fail(msg, f"expected a list of ids, got {type(result).__name__}")
    bad = [x for x in result if not isinstance(x, numbers.Integral) or isinstance(x, bool)]
    if bad:
        _fail(msg, f"expected a list of int, got an element {bad[0]!r} of type {type(bad[0]).__name__}")
    if result != want:
        i = _first_diff(result, want)
        _fail(msg, f"expected id {_at(want, i)!r} at position {i}, got {_at(result, i)!r} "
                   f"({len(want)} ids expected, {len(result)} given)",
              f"expected {want[:24]}\n     got {result[:24]}" if len(want) <= 24 else "")


def make(tk, **params):
    return call(tk.BPETokenizer, expected="a BPETokenizer", **params)


def fit(tk, texts, **params):
    """A fitted tokenizer of the learner (the object itself: a fit without ``return self`` has its own test)."""
    tok = make(tk, **params)
    call(tok.fit, texts, expected="fit to learn the merges")
    for name in ("merges_", "vocab_", "n_base_", "special_ids_"):
        if not hasattr(tok, name):
            _fail("", f"expected fit to set the attribute {name}, it is missing")
    return tok


_FITTED: dict = {}


def fit_cached(tk, texts: tuple, **params):
    """``fit`` on a large corpus, once per test session (the tests only read the result)."""
    key = (id(tk.BPETokenizer), texts, tuple(sorted(params.items())))
    if key not in _FITTED:
        _FITTED[key] = fit(tk, list(texts), **params)
    return _FITTED[key]


def merges_as_text(tok):
    """The merges of the learner as pairs of strings (through its vocab_), for the messages."""
    out = []
    for pair in tok.merges_:
        try:
            a, b = pair
            out.append((tok.vocab_[a].decode("utf-8", "replace"), tok.vocab_[b].decode("utf-8", "replace")))
        except Exception:  # noqa: BLE001
            out.append(pair)
    return out


def assert_merges(tok, want_ids, want_text, msg):
    """``merges_`` equal to ``want_ids`` (a list of id pairs); the message shows the strings of ``want_text``."""
    got = tok.merges_
    if not isinstance(got, list):
        _fail(msg, f"expected merges_ to be a list of (left_id, right_id) tuples, got {type(got).__name__}")
    got_pairs = [tuple(p) if isinstance(p, (tuple, list)) else p for p in got]
    if got_pairs != [tuple(p) for p in want_ids]:
        i = _first_diff(got_pairs, want_ids)
        mine = merges_as_text(tok)
        _fail(msg, f"expected merge {i} = {_at(want_ids, i)} {_at(want_text, i)!r}, got {_at(got_pairs, i)} "
                   f"{_at(mine, i)!r} ({len(want_ids)} merges expected, {len(got_pairs)} learned)",
              f"expected the first merges {want_text[:max(i + 1, 4)]}\n     got {mine[:max(i + 1, 4)]}")
    bad = [p for p in got if not isinstance(p, tuple)]
    if bad:
        _fail(msg, f"expected each merge to be a tuple (left_id, right_id), got {type(bad[0]).__name__}")


def whitespace_words(text):
    return re.findall(WHITESPACE, text)


def gpt2_pieces(text, specials=()):
    """The pre-tokens of the GPT-2 pattern, with the special tokens taken out (they never take part in merges)."""
    parts = re.split("(" + "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")",
                     text) if specials else [text]
    return [p for i, part in enumerate(parts) if i % 2 == 0 for p in re.findall(GPT2, part)]


def to_cjk(piece: str) -> str:
    return "".join(chr(CJK + b) for b in piece.encode("utf-8"))


# ------------------------------------------------------------------ oracles (Hugging Face tokenizers)
def hf_state(tok):
    state = json.loads(tok.to_str())["model"]
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in state["merges"]]
    return state["vocab"], merges


def hf_char(texts, vocab_size, min_frequency):
    """Char-level BPE of tokenizers: (tokenizer, alphabet, merges as id pairs, merges as strings)."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, show_progress=False)
    tok.train_from_iterator(list(texts), trainer)
    vocab, merges = hf_state(tok)
    alphabet = sorted({c for t in texts for w, _ in tok.pre_tokenizer.pre_tokenize_str(t) for c in w})
    assert len(vocab) == len(alphabet) + len(merges), (
        "this corpus makes tokenizers reuse an existing token for a merge (a case where it differs from the "
        "docstring): choose another corpus")
    return tok, alphabet, [(vocab[a], vocab[b]) for a, b in merges], merges


def hf_bytes(pieces_per_text, vocab_size, min_frequency):
    """Byte-level BPE of tokenizers on relabeled bytes: byte b -> chr(CJK + b), the 256 of them as alphabet."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, show_progress=False,
                                  initial_alphabet=[chr(CJK + b) for b in range(256)])
    tok.train_from_iterator([" ".join(to_cjk(p) for p in pieces) for pieces in pieces_per_text], trainer)
    vocab, merges = hf_state(tok)
    assert all(vocab[chr(CJK + b)] == b for b in range(256)), ORACLE_BUG
    assert len(vocab) == 256 + len(merges), "tokenizers reused a token for a merge: choose another corpus"
    text = [(bytes(ord(c) - CJK for c in a), bytes(ord(c) - CJK for c in b)) for a, b in merges]
    return tok, [(vocab[a], vocab[b]) for a, b in merges], text


@functools.lru_cache(maxsize=None)
def corpus(name: str) -> str:
    """Excerpts of the two texts of the chapter (versioned copies in data/text)."""
    from wb.datasets import load_holmes, load_verne

    if name == "holmes":
        return load_holmes()[:120_000]
    return load_verne()[:120_000]


# Small char-level cases, written here (checked against tokenizers when it is installed):
# (text, vocab_size, min_frequency, sorted alphabet, merges as strings)
SMALL_CASES = [
    pytest.param(DOC_CORPUS, 16, 2, "elnorstw",
                 [("w", "e"), ("l", "o"), ("e", "we"), ("n", "ewe"), ("s", "t"), ("lo", "we")],
                 id="docstring-min-frequency-stops"),
    pytest.param(DOC_CORPUS, 12, 2, "elnorstw", [("w", "e"), ("l", "o"), ("e", "we"), ("n", "ewe")],
                 id="docstring-vocab-size-stops"),
    pytest.param(DOC_CORPUS, 100, 1, "elnorstw",
                 [("w", "e"), ("l", "o"), ("e", "we"), ("n", "ewe"), ("s", "t"), ("lo", "we"), ("lo", "w"),
                  ("newe", "r"), ("newe", "st"), ("lowe", "r"), ("lowe", "st")],
                 id="docstring-no-pair-left"),
    pytest.param("ab ba ab ba cd dc cd dc", 40, 1, "abcd", [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")],
                 id="four-ties"),
    pytest.param("zy yz xw wx zy yz", 30, 1, "wxyz", [("y", "z"), ("z", "y"), ("w", "x"), ("x", "w")],
                 id="ties-seen-in-reverse-order"),
    pytest.param("aaa aaaa aaaaa b bb", 40, 1, "ab",
                 [("a", "a"), ("aa", "a"), ("b", "b"), ("aa", "aa"), ("aa", "aaa")], id="overlapping-pairs"),
    pytest.param("bc bc bc bc ab ab ab", 50, 2, "abc", [("b", "c"), ("a", "b")], id="two-merges"),
    pytest.param("élève éléphant été Été ça, ça! Où ? œuvre", 40, 1, "!,?OaehlnprtuvÉçèéùœ",
                 [("t", "é"), ("ç", "a"), ("é", "l"), ("O", "ù"), ("a", "n"), ("h", "an"), ("p", "han"),
                  ("r", "e"), ("u", "v"), ("v", "e"), ("É", "té"), ("è", "ve"), ("é", "té"), ("é", "phan"),
                  ("œ", "uv"), ("él", "ève"), ("él", "éphan"), ("œuv", "re"), ("éléphan", "t")],
                 id="french-accents-and-punctuation"),
]


def ids_of_text_merges(alphabet: str, merges_text):
    """Id pairs of string merges: the alphabet first, then one id per merge."""
    stoi = {c: i for i, c in enumerate(alphabet)}
    out = []
    for a, b in merges_text:
        out.append((stoi[a], stoi[b]))
        stoi[a + b] = len(stoi)
    return out


def check_base(tok, alphabet, msg):
    if tok.n_base_ != len(alphabet):
        _fail(msg, f"expected n_base_ = {len(alphabet)} (the distinct characters of the pre-tokens), "
                   f"got {tok.n_base_!r}")
    got = [tok.vocab_.get(i) for i in range(len(alphabet))]
    want = [c.encode("utf-8") for c in alphabet]
    if got != want:
        i = _first_diff(got, want)
        _fail(msg, f"expected vocab_[{i}] = {want[i]!r} (characters sorted by code point), got {got[i]!r}",
              f"expected {''.join(alphabet)!r}")


# ================================================================== pre_tokenize (provided)
def test_pre_tokenize_docstring_examples(tk):
    for args, kwargs, want in [
        (("Hello, world!",), {}, ["Hello", ",", "world", "!"]),
        (("Hello, world!", "gpt2"), {}, ["Hello", ",", " world", "!"]),
        (("Hi you<|endoftext|>Bye", "gpt2"), {"special_tokens": ["<|endoftext|>"]},
         ["Hi", " you", "<|endoftext|>", "Bye"]),
        (("a b<s>c",), {"pre_tokenizer": None, "special_tokens": ["<s>"]}, ["a b", "<s>", "c"]),
    ]:
        got = call(tk.pre_tokenize, *args, expected="the pieces", **kwargs)
        if got != want:
            _fail("", f"expected pre_tokenize{args} {kwargs or ''} = {want!r}, got {got!r} "
                      "(pre_tokenize is provided: restore it from templates/mylearn_stubs/tokenizer.py)")


@needs_tokenizers
@pytest.mark.parametrize("name", ["holmes", "verne"])
def test_pre_tokenize_whitespace_matches_tokenizers(tk, name):
    text = corpus(name)[:20_000]
    want = [w for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(text)]
    got = call(tk.pre_tokenize, text, "whitespace", expected="the pieces")
    if got != want:
        i = _first_diff(got, want)
        _fail("", f"expected piece {i} = {_at(want, i)!r} (like pre_tokenizers.Whitespace), got {_at(got, i)!r}")


def test_pre_tokenize_gpt2_pieces_join_into_the_text(tk):
    rng = random.Random(0)
    for _ in range(50):
        text = random_text(rng, 40)
        got = call(tk.pre_tokenize, text, "gpt2", ["<|endoftext|>"], expected="the pieces")
        if "".join(got) != text:
            _fail("", f"expected the gpt2 pieces of {text!r} to join into the text, got {got!r}")


def test_pre_tokenize_refuses_an_unknown_pre_tokenizer(tk):
    assert_raises(ValueError, tk.pre_tokenize, "a b", "bert", why="pre_tokenizer='bert' is unknown")


# ================================================================== get_pair_counts (B2.15)
def pairs_oracle(words):
    """Brute force: expand the corpus (each word repeated freq times), count every adjacent pair."""
    expanded = [w for w, f in words.items() for _ in range(f)]
    return Counter(pair for w in expanded for pair in zip(w, w[1:]))


def check_pair_counts(result, want, msg):
    if result is None:
        _fail(msg, "expected a dict {(left_id, right_id): count}, got None (did you forget the return?)")
    if not isinstance(result, dict):
        _fail(msg, f"expected a dict {{(left_id, right_id): count}}, got {type(result).__name__}")
    bad_keys = [k for k in result if not (isinstance(k, tuple) and len(k) == 2)]
    if bad_keys:
        _fail(msg, f"expected keys that are (left_id, right_id) tuples, got the key {bad_keys[0]!r}")
    want = dict(want)
    for pair in sorted(set(want) | set(result)):
        if result.get(pair) != want.get(pair):
            _fail(msg, f"expected count {want.get(pair, 'absent (the pair does not occur)')} for the pair {pair}, "
                       f"got {result.get(pair, 'absent')}",
                  f"expected {dict(sorted(want.items()))}" if len(want) <= 10 else "")


def test_get_pair_counts_docstring_example(tk):
    got = call(tk.get_pair_counts, {(1, 2, 3): 2, (1, 2): 1}, expected="the pair counts")
    check_pair_counts(got, {(1, 2): 3, (2, 3): 2}, "get_pair_counts({(1, 2, 3): 2, (1, 2): 1})")


@pytest.mark.parametrize("words,why", [
    pytest.param({(1, 1, 1): 1}, "the word (1, 1, 1) holds the pair (1, 1) twice", id="overlap-three"),
    pytest.param({(1, 1, 1, 1): 3}, "(1, 1) three times in the word, the word 3 times", id="overlap-four"),
    pytest.param({(1, 2): 1, (3, 4): 1}, "no pair (2, 3) across two words", id="no-pair-across-words"),
    pytest.param({(5,): 7, (1, 2): 2}, "a word of one id holds no pair", id="one-id-word"),
    pytest.param({(2, 1): 4, (1, 2): 1}, "(2, 1) and (1, 2) are two different pairs", id="order-matters"),
])
def test_get_pair_counts_hand_cases(tk, words, why):
    got = call(tk.get_pair_counts, dict(words), expected="the pair counts")
    check_pair_counts(got, pairs_oracle(words), f"get_pair_counts({words}): {why}")


@pytest.mark.parametrize("seed", range(6))
def test_get_pair_counts_matches_a_brute_force_counter(tk, seed):
    rng = random.Random(seed)
    words = {}
    for _ in range(rng.randint(5, 30)):
        words[tuple(rng.randint(0, 3 + seed) for _ in range(rng.randint(1, 8)))] = rng.randint(1, 9)
    got = call(tk.get_pair_counts, dict(words), expected="the pair counts")
    check_pair_counts(got, pairs_oracle(words), f"random corpus {seed} (brute-force count over the expanded corpus)")
    total = sum(f * (len(w) - 1) for w, f in words.items())
    if sum(got.values()) != total:
        _fail("", f"expected the counts to sum to sum(freq * (len(word) - 1)) = {total}, got {sum(got.values())}")


def test_get_pair_counts_of_a_real_corpus(tk):
    freqs = Counter(whitespace_words(corpus("verne")[:30_000]))
    alphabet = {c: i for i, c in enumerate(sorted({c for w in freqs for c in w}))}
    words = {tuple(alphabet[c] for c in w): f for w, f in freqs.items()}
    got = call(tk.get_pair_counts, dict(words), expected="the pair counts")
    check_pair_counts(got, pairs_oracle(words), "the words of an excerpt of Verne, as tuples of character ids")


def test_get_pair_counts_accepts_any_mapping(tk):
    words = {(1, 2, 1, 2): 2, (2, 1): 5}
    got = call(tk.get_pair_counts, types.MappingProxyType(dict(words)),
               expected="the pair counts of a read-only mapping (do not modify the corpus)")
    check_pair_counts(got, pairs_oracle(words), "a read-only mapping (types.MappingProxyType)")
    got2 = call(tk.get_pair_counts, Counter(words), expected="the pair counts of a Counter")
    check_pair_counts(got2, pairs_oracle(words), "a collections.Counter")


def test_get_pair_counts_does_not_modify_its_input(tk):
    words = {(1, 2, 3): 2, (3, 1, 2): 4}
    saved = dict(words)
    call(tk.get_pair_counts, words)
    if words != saved:
        _fail("", f"expected the corpus unchanged by get_pair_counts, got {words!r} instead of {saved!r}")


def test_get_pair_counts_refuses_a_negative_frequency(tk):
    assert_raises(ValueError, tk.get_pair_counts, {(1, 2): 3, (2, 3): -1}, why="the word (2, 3) has frequency -1")


# ================================================================== merge_pair (B2.15)
def merge_oracle(ids, pair, new_id):
    """str.replace replaces the non-overlapping occurrences, left to right: one character per id."""
    offset = 0x100
    text = "".join(chr(offset + i) for i in ids)
    merged = text.replace(chr(offset + pair[0]) + chr(offset + pair[1]), chr(offset + new_id))
    return [ord(c) - offset for c in merged]


def check_merge(result, want, msg):
    if result is None:
        _fail(msg, "expected a list of ids, got None (did you forget the return?)")
    if not isinstance(result, list):
        _fail(msg, f"expected a new list, got {type(result).__name__}")
    if list(result) != want:
        _fail(msg, f"expected {want}, got {list(result)}")


@pytest.mark.parametrize("ids,pair,new_id,want", [
    pytest.param([3, 1, 2, 1, 2], (1, 2), 7, [3, 7, 7], id="docstring-two-occurrences"),
    pytest.param([1, 1, 1], (1, 1), 5, [5, 1], id="docstring-overlap"),
    pytest.param([1, 1, 1, 1], (1, 1), 5, [5, 5], id="four-identical"),
    pytest.param([2, 1, 1, 1, 2], (1, 1), 9, [2, 9, 1, 2], id="overlap-in-the-middle"),
    pytest.param([1, 2, 1, 2, 1], (2, 1), 8, [1, 8, 8], id="pair-reversed"),
    pytest.param([4, 4, 5], (4, 5), 6, [4, 6], id="pair-at-the-end"),
    pytest.param([4, 5], (4, 5), 6, [6], id="whole-word"),
    pytest.param([1, 2, 3], (3, 1), 9, [1, 2, 3], id="no-occurrence"),
    pytest.param([], (1, 2), 3, [], id="empty"),
    pytest.param([7], (7, 7), 8, [7], id="one-id"),
])
def test_merge_pair_hand_cases(tk, ids, pair, new_id, want):
    assert merge_oracle(ids, pair, new_id) == want, ORACLE_BUG
    got = call(tk.merge_pair, list(ids), pair, new_id, expected="the merged ids")
    check_merge(got, want, f"merge_pair({ids}, {pair}, {new_id}) (non-overlapping, left to right)")


@pytest.mark.parametrize("seed", range(8))
def test_merge_pair_matches_str_replace(tk, seed):
    rng = random.Random(seed)
    ids = [rng.randint(0, 2) for _ in range(rng.randint(5, 40))]
    pair = (rng.randint(0, 2), rng.randint(0, 2))
    want = merge_oracle(ids, pair, 3)
    got = call(tk.merge_pair, list(ids), pair, 3, expected="the merged ids")
    check_merge(got, want, f"merge_pair({ids}, {pair}, 3) (as str.replace: non-overlapping, left to right)")


@pytest.mark.parametrize("seed", range(4))
def test_merge_pair_keeps_the_joined_bytes(tk, seed):
    rng = random.Random(100 + seed)
    vocab = {i: bytes([97 + i]) for i in range(3)}
    ids = [rng.randint(0, 2) for _ in range(30)]
    for step in range(4):           # four merges in a row, each on the result of the previous one
        pairs = list(zip(ids, ids[1:]))
        if not pairs:
            break
        pair = rng.choice(pairs)
        new_id = len(vocab)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        before = b"".join(vocab[i] for i in ids)
        got = call(tk.merge_pair, list(ids), pair, new_id, expected="the merged ids")
        check_merge(got, merge_oracle(ids, pair, new_id), f"merge {step} of seed {seed}")
        if b"".join(vocab[i] for i in got) != before:
            _fail("", f"expected the joined bytes {before!r} unchanged by merge_pair")
        ids = got


def test_merge_pair_accepts_a_tuple_and_returns_a_new_list(tk):
    got = call(tk.merge_pair, (1, 2, 3), (2, 3), 4, expected="the merged ids of a tuple")
    check_merge(got, [1, 4], "merge_pair((1, 2, 3), (2, 3), 4)")
    ids = [1, 3, 1]
    out = call(tk.merge_pair, ids, (2, 3), 4, expected="the ids unchanged (no occurrence)")
    if out is ids:
        _fail("", "expected a new list, got the input list itself (changing one would change the other)")


def test_merge_pair_does_not_modify_its_input(tk):
    ids = [1, 2, 1, 2, 2, 1, 2]
    saved = list(ids)
    call(tk.merge_pair, ids, (1, 2), 9)
    if ids != saved:
        _fail("", f"expected the input list unchanged by merge_pair, got {ids} instead of {saved}")


@pytest.mark.parametrize("pair,why", [
    pytest.param((1,), "a pair of one id", id="one-id"),
    pytest.param((1, 2, 3), "a pair of three ids", id="three-ids"),
])
def test_merge_pair_refuses_a_pair_that_is_not_two_ids(tk, pair, why):
    assert_raises(ValueError, tk.merge_pair, [1, 2, 3], pair, 9, why=why)


# ================================================================== BPETokenizer.fit (B2.16)
def test_bpe_fit_docstring_examples(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    if (tok.n_base_, len(tok.merges_)) != (8, 6):
        _fail("", f"expected (n_base_, len(merges_)) = (8, 6) on the docstring corpus, got "
                  f"({tok.n_base_!r}, {len(tok.merges_)})")
    assert_merges(tok, [(7, 0), (1, 3), (0, 8), (2, 10), (5, 6), (9, 8)],
                  [("w", "e"), ("l", "o"), ("e", "we"), ("n", "ewe"), ("s", "t"), ("lo", "we")], "docstring")
    got = (tok.vocab_.get(8), tok.vocab_.get(9), tok.vocab_.get(10))
    if got != (b"we", b"lo", b"ewe"):
        _fail("", f"expected vocab_[8], vocab_[9], vocab_[10] = (b'we', b'lo', b'ewe'), got {got!r}")
    tok2 = fit(tk, ["low lower", "lowest"], vocab_size=16, level="char")
    new = [tok2.vocab_.get(i) for i in range(tok2.n_base_, len(tok2.vocab_))]
    if new != [b"lo", b"low", b"lowe"]:
        _fail("", f"expected the new tokens [b'lo', b'low', b'lowe'] for fit(['low lower', 'lowest']), got {new!r}")


def test_bpe_fit_docstring_byte_level_example(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=260, pre_tokenizer="gpt2", special_tokens=["<|endoftext|>"])
    assert_merges(tok, [(119, 101), (108, 111), (32, 110)], [(b"w", b"e"), (b"l", b"o"), (b" ", b"n")],
                  "byte level, gpt2, vocab_size=260 with one special token")
    if tok.special_ids_ != {"<|endoftext|>": 259}:
        _fail("", f"expected special_ids_ = {{'<|endoftext|>': 259}} (after the 256 bytes and the 3 merges), "
                  f"got {tok.special_ids_!r}")


def test_bpe_fit_returns_self(tk):
    tok = make(tk, vocab_size=12, level="char")
    result = call(tok.fit, DOC_CORPUS, expected="fit to return self")
    if result is None:
        _fail("", "expected fit to return self, got None (did you forget the return self?)")
    if result is not tok:
        _fail("", f"expected fit to return self (the tokenizer itself), got {type(result).__name__}")


@pytest.mark.parametrize("text,vocab_size,min_frequency,alphabet,want_text", SMALL_CASES)
def test_bpe_fit_small_cases_match_tokenizers(tk, text, vocab_size, min_frequency, alphabet, want_text):
    want = ids_of_text_merges(alphabet, want_text)
    if Tokenizer is not None:
        _, hf_alphabet, hf_ids, hf_text = hf_char([text], vocab_size, min_frequency)
        assert ("".join(hf_alphabet), hf_ids, hf_text) == (alphabet, want, want_text), ORACLE_BUG
    tok = fit(tk, text, vocab_size=vocab_size, level="char", min_frequency=min_frequency)
    msg = f"BpeTrainer(vocab_size={vocab_size}, min_frequency={min_frequency}) on {text!r}"
    check_base(tok, alphabet, msg)
    assert_merges(tok, want, want_text, msg + " (most frequent pair, ties to the smallest (left_id, right_id))")


@needs_tokenizers
@pytest.mark.parametrize("seed", range(10))
def test_bpe_fit_random_corpora_match_tokenizers(tk, seed):
    rng = random.Random(seed)
    alphabet = ["ab", "abc", "aé", "xyz.,", "ab!", "abcd"][seed % 6]
    words = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 7))) for _ in range(rng.randint(5, 25))]
    text = " ".join(words)
    vocab_size = rng.randint(len(set(text) - {" "}) + 2, 40)
    min_frequency = [0, 1, 2, 3][seed % 4]
    _, hf_alpha, hf_ids, hf_text = hf_char([text], vocab_size, min_frequency)
    tok = fit(tk, text, vocab_size=vocab_size, level="char", min_frequency=min_frequency)
    msg = f"BpeTrainer(vocab_size={vocab_size}, min_frequency={min_frequency}) on {text!r}"
    check_base(tok, hf_alpha, msg)
    assert_merges(tok, hf_ids, hf_text, msg)


@needs_tokenizers
@pytest.mark.parametrize("name,size,vocab_size,min_frequency,lines", [
    pytest.param("verne", 20_000, 400, 2, False, id="verne-400"),
    pytest.param("holmes", 40_000, 500, 3, True, id="holmes-lines-500"),
])
def test_bpe_fit_real_texts_match_tokenizers(tk, name, size, vocab_size, min_frequency, lines):
    text = corpus(name)[:size]
    texts = tuple(text.split("\n")) if lines else (text,)
    _, hf_alpha, hf_ids, hf_text = hf_char(texts, vocab_size, min_frequency)
    tok = fit_cached(tk, texts, vocab_size=vocab_size, level="char", min_frequency=min_frequency)
    msg = f"BpeTrainer(vocab_size={vocab_size}, min_frequency={min_frequency}) on {size} characters of {name}"
    check_base(tok, hf_alpha, msg)
    assert_merges(tok, hf_ids, hf_text, msg)


@needs_tokenizers
@pytest.mark.parametrize("name,texts,vocab_size,min_frequency", [
    pytest.param("docstring", (DOC_CORPUS,), 270, 1, id="docstring-corpus"),
    pytest.param("accents", ("été été étés ça ça çà où où", "Noël, naïf ; cœur !"), 280, 1, id="multi-byte"),
    pytest.param("verne", None, 500, 2, id="verne-500"),
    pytest.param("holmes", None, 450, 2, id="holmes-450"),
])
def test_bpe_fit_byte_level_matches_relabeled_tokenizers(tk, name, texts, vocab_size, min_frequency):
    texts = texts or (corpus(name)[:20_000],)
    _, want, want_text = hf_bytes([gpt2_pieces(t) for t in texts], vocab_size, min_frequency)
    tok = fit_cached(tk, texts, vocab_size=vocab_size, pre_tokenizer="gpt2", min_frequency=min_frequency)
    if tok.n_base_ != 256:
        _fail("", f"expected n_base_ = 256 at byte level, got {tok.n_base_!r}")
    assert_merges(tok, want, want_text, f"byte level, gpt2, vocab_size={vocab_size} on {name} (ties to the smallest "
                                        "(left_id, right_id), ids = byte values)")


def test_bpe_fit_byte_level_base_alphabet(tk):
    tok = fit(tk, "été", vocab_size=256, pre_tokenizer="gpt2")
    bad = [i for i in range(256) if tok.vocab_.get(i) != bytes([i])]
    if bad:
        _fail("", f"expected vocab_[{bad[0]}] = {bytes([bad[0]])!r} at byte level (id i is the byte i), "
                  f"got {tok.vocab_.get(bad[0])!r}")
    if tok.n_base_ != 256 or tok.merges_ != [] or len(tok.vocab_) != 256:
        _fail("", f"expected 256 bytes and no merge with vocab_size=256, got n_base_={tok.n_base_!r}, "
                  f"{len(tok.merges_)} merges, {len(tok.vocab_)} tokens")


def test_bpe_fit_char_alphabet_sorted_by_code_point(tk):
    text = "zèbre Zoé abc Été !"
    tok = fit(tk, text, vocab_size=17, level="char")
    check_base(tok, sorted(set("".join(whitespace_words(text)))), f"fit({text!r}) at char level")


def test_bpe_fit_vocab_holds_the_bytes_of_every_token(tk):
    tok = fit(tk, corpus("verne")[:5_000], vocab_size=300, level="char", special_tokens=["<|endoftext|>", "<pad>"],
              unk_token="<unk>")
    n_merges = len(tok.merges_)
    want_len = tok.n_base_ + n_merges + 3
    if len(tok.vocab_) != want_len:
        _fail("", f"expected len(vocab_) = n_base_ + merges + special tokens = {want_len}, got {len(tok.vocab_)}")
    for i, (a, b) in enumerate(tok.merges_):
        if tok.vocab_.get(tok.n_base_ + i) != tok.vocab_[a] + tok.vocab_[b]:
            _fail("", f"expected vocab_[{tok.n_base_ + i}] = vocab_[{a}] + vocab_[{b}] = "
                      f"{tok.vocab_[a] + tok.vocab_[b]!r} (merge {i}), got {tok.vocab_.get(tok.n_base_ + i)!r}")
    for token in ("<|endoftext|>", "<pad>", "<unk>"):
        i = tok.special_ids_.get(token)
        if i is None or tok.vocab_.get(i) != token.encode("utf-8"):
            _fail("", f"expected vocab_[special_ids_[{token!r}]] = {token.encode()!r}, got "
                      f"{None if i is None else tok.vocab_.get(i)!r}")


def test_bpe_fit_stops_at_vocab_size_with_the_special_tokens(tk):
    text = corpus("holmes")[:3_000]
    for specials in ([], ["<a>"], ["<a>", "<b>", "<c>"]):
        tok = fit(tk, text, vocab_size=300, special_tokens=specials, pre_tokenizer="gpt2", min_frequency=1)
        if len(tok.merges_) != 300 - 256 - len(specials):
            _fail("", f"expected {300 - 256 - len(specials)} merges with vocab_size=300, the 256 bytes and "
                      f"{len(specials)} special token(s), got {len(tok.merges_)}")
    tok = fit(tk, text, vocab_size=258, special_tokens=["<a>", "<b>"], pre_tokenizer="gpt2")
    if tok.merges_ != []:
        _fail("", f"expected no merge when vocab_size = 256 + 2 special tokens, got {len(tok.merges_)}")


def test_bpe_fit_stops_when_no_pair_is_left(tk):
    tok = fit(tk, "a b c a b", vocab_size=100, level="char", min_frequency=0)
    if tok.merges_ != []:
        _fail("", f"expected no merge for words of one character, got {tok.merges_!r}")
    tok = fit(tk, "ab ab", vocab_size=100, level="char", min_frequency=0)
    if tok.merges_ != [(0, 1)]:
        _fail("", f"expected the single merge [(0, 1)] for 'ab ab', then no pair left, got {tok.merges_!r}")


def test_bpe_fit_special_tokens_get_the_last_ids_and_no_merge(tk):
    specials = ["<|endoftext|>", "<sep>"]
    text = "low<|endoftext|>lower <sep> lowest<sep>newer<|endoftext|> newest"
    plain = re.sub("|".join(re.escape(s) for s in specials), " ", text)
    want_alpha = sorted(set("".join(whitespace_words(plain))))
    want_text = SMALL_CASES[0].values[4]                     # the docstring corpus: the same words
    tok = fit(tk, text, vocab_size=18, level="char", special_tokens=specials)
    check_base(tok, want_alpha, "special tokens are taken out of the training texts")
    assert_merges(tok, ids_of_text_merges("".join(want_alpha), want_text), want_text,
                  "with the special tokens taken out, the words are those of the docstring corpus")
    n = tok.n_base_ + len(tok.merges_)
    if tok.special_ids_ != {"<|endoftext|>": n, "<sep>": n + 1}:
        _fail("", f"expected special_ids_ = {{'<|endoftext|>': {n}, '<sep>': {n + 1}}} (after the merges, in the "
                  f"given order), got {tok.special_ids_!r}")


def test_bpe_fit_unk_token_comes_after_the_special_tokens(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char", unk_token="<unk>")
    if tok.special_ids_ != {"<unk>": 14}:
        _fail("", f"expected special_ids_ = {{'<unk>': 14}} (docstring of encode), got {tok.special_ids_!r}")
    tok = fit(tk, DOC_CORPUS, vocab_size=30, level="char", special_tokens=["<s>"], unk_token="<unk>",
              min_frequency=2)
    n = tok.n_base_ + len(tok.merges_)
    if tok.special_ids_ != {"<s>": n, "<unk>": n + 1}:
        _fail("", f"expected special_ids_ = {{'<s>': {n}, '<unk>': {n + 1}}} (unk_token added after the special "
                  f"tokens), got {tok.special_ids_!r}")
    tok = fit(tk, DOC_CORPUS, vocab_size=30, level="char", special_tokens=["<unk>", "<s>"], unk_token="<unk>")
    n = tok.n_base_ + len(tok.merges_)
    if tok.special_ids_ != {"<unk>": n, "<s>": n + 1}:
        _fail("", f"expected special_ids_ = {{'<unk>': {n}, '<s>': {n + 1}}} (unk_token already in special_tokens: "
                  f"one id only), got {tok.special_ids_!r}")


def test_bpe_fit_one_str_is_one_text(tk):
    want = fit(tk, [DOC_CORPUS], vocab_size=16, level="char").merges_
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    if tok.merges_ != want:
        _fail("", f"expected fit('low lower ...') to learn the merges of fit(['low lower ...']), {want!r}, got "
                  f"{tok.merges_!r} (a single str is one text, not an iterable of characters)")
    gen = fit(tk, (t for t in [DOC_CORPUS]), vocab_size=16, level="char")
    if gen.merges_ != want:
        _fail("", f"expected the same merges from a generator of texts, got {gen.merges_!r}")


def test_bpe_fit_several_texts_count_their_words_together(tk):
    texts = DOC_CORPUS.split()
    tok = fit(tk, texts, vocab_size=16, level="char")
    assert_merges(tok, [(7, 0), (1, 3), (0, 8), (2, 10), (5, 6), (9, 8)],
                  SMALL_CASES[0].values[4], "the five words given as five texts")


def test_bpe_fit_does_not_modify_its_input(tk):
    texts = ["low lower", "lowest", "newer newest"]
    saved = list(texts)
    fit(tk, texts, vocab_size=16, level="char")
    if texts != saved:
        _fail("", f"expected the list of texts unchanged by fit, got {texts!r}")


@pytest.mark.parametrize("params,why", [
    pytest.param(dict(level="word"), "level='word' is unknown", id="level"),
    pytest.param(dict(pre_tokenizer="bert"), "pre_tokenizer='bert' is unknown", id="pre-tokenizer"),
    pytest.param(dict(unk_token="<unk>"), "unk_token at byte level (every text can be encoded)",
                 id="unk-at-byte-level"),
    pytest.param(dict(vocab_size=256, special_tokens=["<s>"]), "vocab_size=256 < 256 bytes + 1 special token",
                 id="vocab-too-small-bytes"),
    pytest.param(dict(vocab_size=7, level="char"), "vocab_size=7 < the 8 characters of the corpus",
                 id="vocab-too-small-chars"),
    pytest.param(dict(vocab_size=9, level="char", special_tokens=["<a>", "<b>"]),
                 "vocab_size=9 < 8 characters + 2 special tokens", id="vocab-too-small-specials"),
])
def test_bpe_fit_refuses_invalid_parameters(tk, params, why):
    tok = make(tk, **params)
    assert_raises(ValueError, tok.fit, DOC_CORPUS, why=why)


# ================================================================== BPETokenizer.encode (B2.17)
def test_bpe_encode_docstring_examples(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    assert_id_list(call(tok.encode, "lowest", expected="the ids"), [13, 12], "encode('lowest')")
    unk = fit(tk, DOC_CORPUS, vocab_size=16, level="char", unk_token="<unk>")
    assert_id_list(call(unk.encode, "lowly", expected="the ids"), [9, 7, 1, 14],
                   "encode('lowly') with unk_token='<unk>' (the unseen 'y' becomes the id of '<unk>')")


def test_bpe_encode_docstring_byte_level_example(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=260, pre_tokenizer="gpt2", special_tokens=["<|endoftext|>"])
    ids = call(tok.encode, "lowest<|endoftext|> été", expected="the ids")
    assert_id_list(ids, [257, 256, 115, 116, 259, 32, 195, 169, 116, 195, 169],
                   "encode('lowest<|endoftext|> été') (the special token is one id)")


@pytest.mark.parametrize("text,vocab_size,min_frequency,alphabet,want_text", SMALL_CASES)
def test_bpe_encode_small_cases_match_tokenizers(tk, text, vocab_size, min_frequency, alphabet, want_text):
    rng = random.Random(len(text))
    probe = " ".join("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 9))) for _ in range(12))
    want = encode_by_rank(probe, alphabet, want_text)
    if Tokenizer is not None:
        hf, *_ = hf_char([text], vocab_size, min_frequency)
        assert hf.encode(probe).ids == want, ORACLE_BUG
    tok = fit(tk, text, vocab_size=vocab_size, level="char", min_frequency=min_frequency)
    assert_id_list(call(tok.encode, probe, expected="the ids"), want,
                   f"encode({probe!r}) after fit({text!r}) (like tokenizers: lowest rank first, in each word)")


def encode_by_rank(text, alphabet, merges_text):
    """The tokens of tokenizers, rebuilt from the merges written in this file (small cases only)."""
    stoi = {c: i for i, c in enumerate(alphabet)}
    rank = {}
    for r, (a, b) in enumerate(merges_text):
        rank[(a, b)] = r
        stoi[a + b] = len(stoi)
    out = []
    for word in whitespace_words(text):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(rank.get((x, y), 10**9), i) for i, (x, y) in enumerate(zip(parts, parts[1:]))]
            r, i = min(pairs)
            if r == 10**9:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        out.extend(stoi[p] for p in parts)
    return out


@needs_tokenizers
@pytest.mark.parametrize("name,vocab_size", [pytest.param("verne", 400, id="verne"),
                                             pytest.param("holmes", 500, id="holmes")])
def test_bpe_encode_char_level_matches_tokenizers(tk, name, vocab_size):
    text = corpus(name)
    train = (text[:20_000],)
    hf, alphabet, _, _ = hf_char(train, vocab_size, 2)
    seen = set(alphabet)
    probe = "".join(c if c in seen or c.isspace() else " " for c in text[60_000:64_000])
    tok = fit_cached(tk, train, vocab_size=vocab_size, level="char", min_frequency=2)
    assert_id_list(call(tok.encode, probe, expected="the ids"), hf.encode(probe).ids,
                   f"encode of another excerpt of {name} (same ids as tokenizers)")


@needs_tokenizers
@pytest.mark.parametrize("name,vocab_size", [pytest.param("verne", 500, id="verne"),
                                             pytest.param("holmes", 450, id="holmes")])
def test_bpe_encode_byte_level_matches_relabeled_tokenizers(tk, name, vocab_size):
    text = corpus(name)
    train = (text[:20_000],)
    hf, _, _ = hf_bytes([gpt2_pieces(train[0])], vocab_size, 2)
    probe = text[60_000:64_000] + " naïve, cœur, 東京 🙂"
    want = hf.encode(" ".join(to_cjk(p) for p in gpt2_pieces(probe))).ids
    tok = fit_cached(tk, train, vocab_size=vocab_size, pre_tokenizer="gpt2", min_frequency=2)
    assert_id_list(call(tok.encode, probe, expected="the ids"), want,
                   f"byte-level encode of another excerpt of {name} (lowest rank first, inside each pre-token)")


def test_bpe_encode_applies_the_merges_by_rank_not_in_text_order(tk):
    tok = fit(tk, "bc bc bc bc ab ab ab", vocab_size=50, level="char")     # merges: (b, c) then (a, b)
    assert_id_list(call(tok.encode, "abc", expected="the ids"), [0, 3],
                   "encode('abc') after the merges (b, c) then (a, b): 'a' + 'bc' (the merge of lowest rank "
                   "first, wherever it is in the word), not 'ab' + 'c'")
    assert_id_list(call(tok.encode, "abcab", expected="the ids"), [0, 3, 4], "encode('abcab'): 'a' + 'bc' + 'ab'")


def test_bpe_encode_never_merges_across_pre_tokens(tk):
    tok = fit(tk, "ab ab ab", vocab_size=10, level="char")                # one merge: (a, b) -> 2
    assert_id_list(call(tok.encode, "a b ab", expected="the ids"), [0, 1, 2],
                   "encode('a b ab'): 'a' and 'b' are two words, never merged together")
    gpt = fit(tk, "sa sa sa", vocab_size=257, pre_tokenizer="gpt2")      # one merge: (115, 97) 'sa' -> 256
    if gpt.merges_ != [(115, 97)]:
        _fail("", f"expected the single merge [(115, 97)] ('s' + 'a') for 'sa sa sa' with vocab_size=257, "
                  f"got {gpt.merges_!r}")
    assert_id_list(call(gpt.encode, "'sa", expected="the ids"), [39, 115, 97],
                   "encode(\"'sa\") with gpt2: the pre-tokens are \"'s\" and 'a', so 's' and 'a' are never merged")
    assert_id_list(call(gpt.encode, "sa'sa", expected="the ids"), [256, 39, 115, 97],
                   "encode(\"sa'sa\") with gpt2: pre-tokens 'sa', \"'s\", 'a'")


def test_bpe_encode_unknown_characters(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char", unk_token="<unk>")
    assert_id_list(call(tok.encode, "lozzw", expected="the ids"), [9, 14, 14, 7],
                   "encode('lozzw'): one '<unk>' id per unseen character")
    assert_id_list(call(tok.encode, "<unk> low", expected="the ids"), [14, 9, 7],
                   "encode('<unk> low'): '<unk>' written in the text is a special token (key of special_ids_)")
    no_unk = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    assert_raises(ValueError, no_unk.encode, "lowly", why="'y' was not seen during fit and unk_token is None")


def test_bpe_encode_special_tokens_are_never_split(tk):
    specials = ["<|endoftext|>", "<sep>"]
    tok = fit(tk, "low lower lowest", vocab_size=300, pre_tokenizer="gpt2", special_tokens=specials, min_frequency=1)
    eot, sep = tok.special_ids_["<|endoftext|>"], tok.special_ids_["<sep>"]
    ids = call(tok.encode, "<sep>low<|endoftext|><|endoftext|> lower<sep>", expected="the ids")
    if not isinstance(ids, list) or ids[:1] != [sep] or ids.count(eot) != 2 or ids[-1:] != [sep]:
        _fail("", f"expected the ids of '<sep>low<|endoftext|><|endoftext|> lower<sep>' to start and end with "
                  f"<sep> = {sep} and to hold <|endoftext|> = {eot} twice, got {ids!r}")
    want = [sep] + call(tok.encode, "low") + [eot, eot] + call(tok.encode, " lower") + [sep]
    assert_id_list(ids, want, "a special token is one id, and the text around it is encoded on its own")
    char = fit(tk, "low lower", vocab_size=20, level="char", special_tokens=["<s>"])
    s = char.special_ids_["<s>"]
    got = call(char.encode, "low<s>low", expected="the ids (the characters '<', 's', '>' are unseen, but '<s>' is "
                                                  "a special token)")
    if not isinstance(got, list) or got.count(s) != 1 or len(got) != 2 * len(call(char.encode, "low")) + 1:
        _fail("", f"expected encode('low<s>low') = encode('low') + [{s}] + encode('low'), got {got!r}")


def test_bpe_encode_returns_a_list_and_an_empty_list_for_an_empty_text(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    assert_id_list(call(tok.encode, "", expected="[]"), [], "encode('')")
    byte = fit(tk, DOC_CORPUS, vocab_size=260, pre_tokenizer="gpt2")
    assert_id_list(call(byte.encode, "", expected="[]"), [], "encode('') at byte level")


def test_bpe_encode_refuses_an_unfitted_tokenizer(tk):
    assert_raises(ValueError, make(tk).encode, "abc", why="the tokenizer is not fitted")


# ================================================================== BPETokenizer.decode (B2.17)
def check_text(result, want, msg):
    if result is None:
        _fail(msg, "expected a str, got None (did you forget the return?)")
    if not isinstance(result, str):
        _fail(msg, f"expected a str, got {type(result).__name__}")
    if result != want:
        i = _first_diff(result, want)
        _fail(msg, f"expected {want[i:i + 20]!r} at character {i}, got {result[i:i + 20]!r}",
              f"expected {want[:80]!r}\n     got {result[:80]!r}")


def test_bpe_decode_docstring_examples(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    check_text(call(tok.decode, [13, 12], expected="'lowest'"), "lowest", "decode([13, 12])")
    byte = fit(tk, "été été", vocab_size=258, pre_tokenizer="gpt2")
    check_text(call(byte.decode, call(byte.encode, "été"), expected="'été'"), "été", "decode(encode('été'))")
    check_text(call(byte.decode, [195], expected="'�'"), "�",
               "decode([195]), the first byte of 'é' alone: errors='replace' gives '�' (U+FFFD)")


def test_bpe_decode_joins_the_bytes_before_decoding(tk):
    byte = fit(tk, "abc", vocab_size=256, pre_tokenizer="gpt2")         # no merge: id i is the byte i
    check_text(call(byte.decode, [0xC3, 0xA9], expected="'é'"), "é",
               "decode([0xC3, 0xA9]): the two bytes of 'é' in two tokens (join the bytes, then decode once)")
    check_text(call(byte.decode, [104, 0xC3, 0xA9, 0xC3], expected="'hé�'"), "hé�",
               "decode([104, 0xC3, 0xA9, 0xC3]): an incomplete character at the end becomes '�'")
    check_text(call(byte.decode, [0xE6, 0x9D, 0xB1, 0xF0, 0x9F, 0x99, 0x82], expected="'東🙂'"), "東🙂",
               "decode of the 3 bytes of '東' and the 4 bytes of '🙂'")


def random_text(rng, n):
    pool = (list("abcdefghijklmnopqrstuvwxyz ABCXYZ0123456789") + list(" \t\n\r  .,;:!?'\"-_()[]{}@#$%&*/\\") +
            list("éèàçùœæÉÀÇ") + ["́", " ", " ", "‍", "﻿", "ß", "ﬁ", "Ω", "ж", "Я",
                                 "ي", "ע", "東", "京", "ü", "ñ", "🙂", "👍🏽", "🇫🇷", "\x00", "\x7f", "'s", "'ll", "  "])
    out = []
    for _ in range(n):
        if rng.random() < 0.15:
            cp = rng.randint(0x20, 0x2FFFF)
            if 0xD800 <= cp <= 0xDFFF:
                cp = 0x41
            out.append(chr(cp))
        else:
            out.append(rng.choice(pool))
    return "".join(out)


def test_bpe_decode_round_trip_at_byte_level(tk):
    train = (corpus("verne")[:8_000] + " The 'quick' fox's 3 dogs—naïve 東京 🙂 ",)
    tok = fit_cached(tk, train, vocab_size=400, pre_tokenizer="gpt2", special_tokens=("<|endoftext|>",))
    rng = random.Random(2026)
    texts = [random_text(rng, rng.randint(0, 60)) for _ in range(300)]
    texts += ["", " ", "   leading and trailing   ", "a\n\n\nb\t\t c  ", "<|endoftext|>", "x<|endoftext|> y",
              corpus("verne")[50_000:52_000], corpus("holmes")[50_000:52_000], "l'été d'Éloïse ; « Où ? »"]
    for text in texts:
        ids = call(tok.encode, text, expected="the ids")
        back = call(tok.decode, ids, expected="the text")
        check_text(back, text, f"decode(encode(s)) at byte level with gpt2 for s = {text[:40]!r}")


def test_bpe_decode_whitespace_drops_the_spaces(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    check_text(call(tok.decode, call(tok.encode, "low lower newest"), expected="the text"), "lowlowernewest",
               "decode(encode('low lower newest')) with pre_tokenizer='whitespace' (the spaces are lost)")


def test_bpe_decode_special_tokens_and_char_level(tk):
    tok = fit(tk, "élève été", vocab_size=20, level="char", special_tokens=["<|endoftext|>"], unk_token="<unk>")
    ids = [tok.special_ids_["<|endoftext|>"], 0, tok.special_ids_["<unk>"]]
    want = "<|endoftext|>e<unk>"                                     # id 0: 'e', the smallest code point
    check_text(call(tok.decode, ids, expected="the text"), want, f"decode({ids}): special tokens give back their text")


@pytest.mark.parametrize("which,why", [
    pytest.param("len", "id len(vocab_) is not a token", id="one-past-the-end"),
    pytest.param("neg", "id -1 is not a token (no Python negative index)", id="minus-one"),
    pytest.param("big", "id 10**6 is not a token", id="huge"),
])
def test_bpe_decode_refuses_an_unknown_id(tk, which, why):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    bad = {"len": len(tok.vocab_), "neg": -1, "big": 10**6}[which]
    assert_raises(ValueError, tok.decode, [0, bad], why=why)


def test_bpe_decode_refuses_an_unfitted_tokenizer(tk):
    assert_raises(ValueError, make(tk).decode, [1, 2], why="the tokenizer is not fitted")


@needs_tokenizers
def test_bpe_encode_number_of_tokens_close_to_a_tokenizers_byte_level_bpe(tk):
    train, probe, vocab_size = (corpus("holmes")[:30_000],), corpus("holmes")[60_000:80_000], 456
    hf = Tokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    hf.train_from_iterator(list(train), trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    want = len(hf.encode(probe).ids)
    assert hf.decode(hf.encode(probe).ids) == probe, ORACLE_BUG
    tok = fit_cached(tk, train, vocab_size=vocab_size, pre_tokenizer="gpt2", min_frequency=2)
    ids = call(tok.encode, probe, expected="the ids")
    check_text(call(tok.decode, ids, expected="the text"), probe, "decode(encode(s)) on an excerpt of Holmes")
    if abs(len(ids) - want) > 0.02 * want:
        _fail("", f"expected about {want} tokens (within 2 % of a tokenizers ByteLevel BPE of the same size) for "
                  f"20 000 characters of Holmes, got {len(ids)}")


# ================================================================== BPETokenizer.tokenize (B2.17)
def check_tokens(result, want, msg):
    if result is None:
        _fail(msg, "expected a list of str, got None (did you forget the return?)")
    if not isinstance(result, list) or not all(isinstance(t, str) for t in result):
        _fail(msg, f"expected a list of str, got {type(result).__name__}"
                   + (f" holding {type(result[0]).__name__}" if isinstance(result, list) and result else ""))
    if result != want:
        i = _first_diff(result, want)
        _fail(msg, f"expected token {i} = {_at(want, i)!r}, got {_at(result, i)!r}",
              f"expected {want[:20]}\n     got {result[:20]}")


def test_bpe_tokenize_docstring_examples(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=16, level="char")
    check_tokens(call(tok.tokenize, "lowest newer", expected="the tokens"), ["lowe", "st", "newe", "r"],
                 "tokenize('lowest newer')")
    check_tokens(call(tok.tokenize, "newest lower", expected="the tokens"), ["newe", "st", "lowe", "r"],
                 "tokenize('newest lower')")


@needs_tokenizers
def test_bpe_tokenize_matches_tokenizers(tk):
    train = (corpus("verne")[:20_000],)
    hf, alphabet, _, _ = hf_char(train, 400, 2)
    seen = set(alphabet)
    probe = "".join(c if c in seen or c.isspace() else " " for c in corpus("verne")[70_000:72_000])
    tok = fit_cached(tk, train, vocab_size=400, level="char", min_frequency=2)
    check_tokens(call(tok.tokenize, probe, expected="the tokens"), hf.encode(probe).tokens,
                 "tokenize of an excerpt of Verne (the tokens of tokenizers)")


def test_bpe_tokenize_partial_characters_show_as_replacement(tk):
    byte = fit(tk, "abc abc", vocab_size=256, pre_tokenizer="gpt2")     # no merge
    check_tokens(call(byte.tokenize, "é!", expected="the tokens"), ["�", "�", "!"],
                 "tokenize('é!') without merge: each byte of 'é' alone is '�' (errors='replace')")
    merged = fit(tk, "été été été", vocab_size=258, pre_tokenizer="gpt2")   # first merge: the two bytes of 'é'
    check_tokens(call(merged.tokenize, "é<|x|>", expected="the tokens"), ["é", "<", "|", "x", "|", ">"],
                 "tokenize('é<|x|>') once the two bytes of 'é' are merged")


def test_bpe_tokenize_special_tokens(tk):
    tok = fit(tk, DOC_CORPUS, vocab_size=17, level="char", special_tokens=["<|endoftext|>"])
    check_tokens(call(tok.tokenize, "lowest<|endoftext|>newer", expected="the tokens"),
                 ["lowe", "st", "<|endoftext|>", "newe", "r"], "tokenize('lowest<|endoftext|>newer')")


# ================================================================== save / load (provided)
@pytest.mark.parametrize("params", [
    pytest.param(dict(vocab_size=40, level="char", unk_token="<unk>", special_tokens=["<s>"]), id="char"),
    pytest.param(dict(vocab_size=300, pre_tokenizer="gpt2", special_tokens=["<|endoftext|>"]), id="byte-gpt2"),
])
def test_bpe_save_load_gives_the_same_encoding(tk, tmp_path, params):
    text = "l'été d'Éloïse, l'été de Noé ; les étés d'Éloïse"
    tok = fit(tk, text, **params)
    path = tmp_path / "bpe.json"
    call(tok.save, path, expected="save to write the file (save is provided: restore it if you changed it)")
    loaded = call(tk.BPETokenizer.load, path, expected="the loaded tokenizer")
    for name in ("merges_", "special_ids_", "n_base_", "vocab_"):
        if getattr(loaded, name) != getattr(tok, name):
            _fail("", f"expected the loaded tokenizer to have the same {name} (merges_ must hold (int, int) tuples, "
                      "vocab_ bytes)")
    probe = "l'été<s> d'Éloïse<|endoftext|> zzz" if params.get("unk_token") else "l'été<|endoftext|> d'Éloïse 🙂"
    want = call(tok.encode, probe, expected="the ids")
    got = call(loaded.encode, probe, expected="the ids from the loaded tokenizer (encode must use only the "
                                              "hyperparameters and merges_, vocab_, n_base_, special_ids_)")
    assert_id_list(got, want, "encode of the loaded tokenizer")
    check_text(call(loaded.decode, got, expected="the text from the loaded tokenizer"), call(tok.decode, want),
               "decode of the loaded tokenizer")
