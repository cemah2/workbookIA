"""Tests of mylearn.retrieval (bonus chapter B4): oracle tests and properties.

    pytest tests/test_chb4_retrieval.py              # your code (mon_travail/mylearn/retrieval.py)
    pytest tests/test_chb4_retrieval.py --impl=ref   # the reference

Oracles: chunks rebuilt from their closed-form start positions (``j * (chunk_size - overlap)``) and the properties
of the docstring (coverage, exact overlap, reconstruction); BM25 scores computed by hand on 3-document corpora and
a transcription of the docstring formula with explicit loops and ``math.log``, checked at every test against
``rank_bm25.BM25Okapi`` when that library is installed (its idf replaced by the idf of the docstring, the only
difference: ``rank_bm25`` uses ``ln((N - n_t + 0.5) / (n_t + 0.5))`` with a floor); dense search against
``sklearn.neighbors.NearestNeighbors(metric='cosine', algorithm='brute')`` and a stable sort of the similarities;
reciprocal rank fusion against hand-computed examples and an exact computation with ``fractions.Fraction``;
recall@k against hand-computed values and ``sklearn.metrics.top_k_accuracy_score`` (one relevant document per
query), MRR against hand-computed values and ``sklearn.metrics.label_ranking_average_precision_score`` (which
equals the MRR with one relevant document per query).
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.
``cosine_top_k`` may use your ``cosine_similarity_matrix`` (chapter B2): if a test says that function is not
written yet, write it first (mylearn/embeddings.py).

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_chunk_text_ or test_bm25_"`` (B4.17),
``-k "test_cosine_top_k_ or test_reciprocal_rank_fusion_ or test_recall_at_k_ or test_mean_reciprocal_rank_"``
(B4.18).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified;
a text of exactly ``chunk_size`` units gives one chunk; the 'char' unit keeps every character (spaces and newlines
too); a document that does not contain a query term gets 0 from it, also with ``k1 = 0`` or an empty document with
``b = 1`` (never NaN); the corpus may hold empty documents (they count in ``N`` and in ``avgdl``, and score 0);
``fit`` called again replaces every statistic; ``cosine_top_k`` gives a similarity of 0 to a row of zeros (as
``cosine_similarity_matrix``) and returns NumPy arrays (integer indices, float similarities);
``reciprocal_rank_fusion`` accepts rankings of different lengths, empty ones and NumPy arrays (the rows returned by
``cosine_top_k``); ``recall_at_k`` and ``mean_reciprocal_rank`` accept a 2-D NumPy array of retrieved ids (the
indices returned by ``cosine_top_k``), relevant ids given as sets, lists or tuples, retrieved lists shorter than
``k``, and return one number; ``mean_reciprocal_rank`` gives 0 to a query whose relevant set is empty.
The values are compared with rtol=1e-6, atol=1e-6 (the docstrings fix no tolerance; a computation in float32
passes); indices, ids and chunks are compared exactly.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

from __future__ import annotations

import copy
import math
import numbers
import sys
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import label_ranking_average_precision_score, top_k_accuracy_score
from sklearn.neighbors import NearestNeighbors

try:  # optional oracle (absent from Colab and from the frozen environment): the transcription takes over
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

ORACLE_BUG = ("bug in tests/test_chb4_retrieval.py (not in your code): the oracle of this test "
              "disagrees with another one")


@pytest.fixture
def ret(mylearn_module):
    return mylearn_module("retrieval")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3, max_line_width=10**6)
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


# The docstrings fix no tolerance: rtol=1e-6, atol=1e-6 lets a computation in float32 pass, not a wrong formula.
RTOL, ATOL = 1e-6, 1e-6


def assert_close(result, expected, rtol=RTOL, atol=ATOL, msg=""):
    """Numbers or arrays equal up to rounding; the message starts with 'expected …, got …'."""
    if result is None:
        _fail(msg, "expected a number or an array, got None (did you forget the return?)")
    try:
        got = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        _fail(msg, f"expected a number or an array, got an object of type {type(result).__name__}")
    want = np.asarray(expected, dtype=float)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}")
    ok = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=False)
    if not ok.all():
        bad = np.flatnonzero(~ok)
        i = int(bad[0])
        index = np.unravel_index(i, want.shape) if want.ndim > 1 else (i,)
        where = f" at index {tuple(int(k) for k in index)}" if want.ndim else ""
        head = (f"expected {want.ravel()[i] + 0.0:.12g}{where}, got {got.ravel()[i] + 0.0:.12g} "
                f"({bad.size} wrong value(s) out of {want.size}; tolerance rtol={rtol:g}, atol={atol:g})")
        data = f"expected {_short(want, 10)}\n     got {_short(got, 10)}" if want.size <= 16 else ""
        _fail(msg, head, data)


# Functions of earlier chapters that retrieval.py imports: where to find them when they are not written yet.
_EARLIER = {"cosine_similarity_matrix": ("embeddings.py", "B2")}


def _pending(exc: NotImplementedError, function) -> NotImplementedError:
    """The same "⏳" error, with the file and the chapter when it comes from cosine_similarity_matrix."""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        where = _EARLIER.get(frame.name)
        if where and Path(frame.filename).as_posix().endswith(where[0]):
            caller = getattr(function, "__name__", "your function")
            return NotImplementedError(f"{exc} (mylearn/{where[0]}, chapter {where[1]}, called by {caller})")
    return exc


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
    try:
        function(*args, **kwargs)
    except NotImplementedError as exc:   # before `except error`: a RuntimeError; shown as "⏳"
        new = _pending(exc, function)
        if new is exc:
            raise
        raise new from None
    except error:
        return
    except Exception as exc:  # noqa: BLE001 - say which error was raised instead
        raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else "")
                             + f" (it raised {type(exc).__name__}: {exc})") from None
    raise AssertionError(f"{name} must raise {error.__name__}" + (f" here: {why}" if why else ""))


def call(function, *args, expected="a result", **kwargs):
    """Call the learner's function; an unexpected exception becomes a failure that says what was expected."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError as exc:
        new = _pending(exc, function)
        if new is exc:
            raise
        raise new from None
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def assert_number(result, want, msg):
    """One number (a Python or NumPy float, not an array) equal to `want` up to rounding."""
    if result is None:
        _fail(msg, f"expected the number {want:.6g}, got None (did you forget the return?)")
    if isinstance(result, (bool, np.bool_)) or not isinstance(result, numbers.Real):
        _fail(msg, f"expected one number ({want:.6g}), got {type(result).__name__}"
                   + (f" of shape {result.shape}" if isinstance(result, np.ndarray) else ""))
    assert_close(float(result), want, msg=msg)


def assert_list_equal(got, want, msg, what="item"):
    """Two lists equal; the message gives the first position that differs."""
    if got != want:
        i = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        if i < len(want):
            found = repr(got[i]) if i < len(got) else "nothing"
            head = f"expected {what} {want[i]!r} at position {i}, got {found} ({len(want)} expected, {len(got)} given)"
        else:
            head = f"expected {len(want)} {what}(s), got {len(got)} (one too many at position {i}: {got[i]!r})"
        _fail(msg, head, f"expected {want[:12]}\n     got {got[:12]}")


# ================================================================== chunk_text (B4.17)
class _Runaway(BaseException):
    """Raised inside the learner's code when it runs far more steps than the task needs (an endless loop);
    a BaseException, so that no `except Exception` of the learner or of the helpers catches it."""


STEP_BUDGET = 300_000      # lines of Python executed: the longest valid test needs a few thousand


def bounded(function, *args, **kwargs):
    """Call `function` and stop it with _Runaway after STEP_BUDGET lines (an endless loop would fill the memory)."""
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        count += 1
        if count > STEP_BUDGET:
            raise _Runaway
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        return function(*args, **kwargs)
    finally:
        sys.settrace(previous)


def chunk_call(ret, *args, **kwargs):
    """chunk_text through `call`, stopped if it never ends."""
    try:
        return bounded(call, ret.chunk_text, *args, expected="the list of chunks", **kwargs)
    except _Runaway:
        _fail("chunk_text", f"expected chunk_text to stop, it ran more than {STEP_BUDGET} steps (does the start of the "
                            "window move by chunk_size - overlap, and does the loop stop after the chunk that reaches "
                            "the end?)")


def chunks_of(ret, text, *args, **kwargs):
    """The learner's chunks, checked to be a list of str."""
    result = chunk_call(ret, text, *args, **kwargs)
    if result is None:
        _fail("chunk_text", "expected a list of str, got None (did you forget the return?)")
    if not isinstance(result, list):
        _fail("chunk_text", f"expected a list of str, got {type(result).__name__}")
    bad = next((c for c in result if not isinstance(c, str)), None)
    if bad is not None:
        _fail("chunk_text", f"expected every chunk to be a str, got a {type(bad).__name__}: {bad!r}")
    return result


def chunk_oracle(text, chunk_size, overlap, unit):
    """Closed form: chunk j starts at j * step; m = 1 + ceil((n - size) / step) chunks when n > size."""
    units = text.split() if unit == "word" else list(text)
    sep = " " if unit == "word" else ""
    n, step = len(units), chunk_size - overlap
    if n == 0:
        return []
    m = 1 if n <= chunk_size else 1 + math.ceil((n - chunk_size) / step)
    return [sep.join(units[j * step: j * step + chunk_size]) for j in range(m)]


def test_chunk_text_docstring_examples(ret):
    got = chunks_of(ret, "a b c d e f g h i j", chunk_size=4, overlap=2)
    assert_list_equal(got, ["a b c d", "c d e f", "e f g h", "g h i j"],
                      'chunk_text("a b c d e f g h i j", chunk_size=4, overlap=2)', "chunk")
    got = chunks_of(ret, "one two  three\nfour five", chunk_size=3, overlap=1)
    assert_list_equal(got, ["one two three", "three four five"],
                      'chunk_text("one two  three\\nfour five", chunk_size=3, overlap=1) (words re-joined with '
                      "single spaces)", "chunk")
    got = chunks_of(ret, "abcdefg", chunk_size=3, overlap=1, unit="char")
    assert_list_equal(got, ["abc", "cde", "efg"], 'chunk_text("abcdefg", 3, 1, unit="char")', "chunk")


@pytest.mark.parametrize("text,size,overlap,unit,want", [
    pytest.param("a b c d e", 4, 2, "word", ["a b c d", "c d e"], id="last-chunk-shorter"),
    pytest.param("a b c", 5, 1, "word", ["a b c"], id="text-shorter-than-a-chunk"),
    pytest.param("a b c d", 4, 1, "word", ["a b c d"], id="exactly-one-chunk-no-extra"),
    pytest.param("a b c d e f g", 3, 0, "word", ["a b c", "d e f", "g"], id="no-overlap-last-single-word"),
    pytest.param("a b c d e f", 3, 0, "word", ["a b c", "d e f"], id="no-overlap-exact-end"),
    pytest.param("a b c d", 3, 2, "word", ["a b c", "b c d"], id="overlap-size-minus-one"),
    pytest.param("x y z", 1, 0, "word", ["x", "y", "z"], id="chunk-size-1"),
    pytest.param("abcdefgh", 3, 1, "char", ["abc", "cde", "efg", "gh"], id="char-last-shorter"),
    pytest.param("ab cd\nef", 3, 1, "char", ["ab ", " cd", "d\ne", "ef"], id="char-keeps-spaces-and-newlines"),
    pytest.param("abc", 3, 2, "char", ["abc"], id="char-exactly-one-chunk"),
])
def test_chunk_text_hand_examples(ret, text, size, overlap, unit, want):
    assert chunk_oracle(text, size, overlap, unit) == want, ORACLE_BUG
    got = chunks_of(ret, text, chunk_size=size, overlap=overlap, unit=unit)
    assert_list_equal(got, want, f"chunk_text({text!r}, chunk_size={size}, overlap={overlap}, unit={unit!r}) "
                                 "(starts 0, step, 2 step, ...; the last chunk is the first that reaches the end)",
                      "chunk")


def test_chunk_text_default_arguments(ret):
    words = [f"w{i}" for i in range(450)]
    got = chunks_of(ret, " ".join(words))
    want = [" ".join(words[0:200]), " ".join(words[150:350]), " ".join(words[300:450])]
    assert_list_equal([c.split()[0] + "..." + c.split()[-1] + f" ({len(c.split())} words)" for c in got],
                      [c.split()[0] + "..." + c.split()[-1] + f" ({len(c.split())} words)" for c in want],
                      "chunk_text of 450 words with the defaults (chunk_size=200, overlap=50, unit='word': "
                      "starts 0, 150, 300)", "chunk")


def _random_text(rng, n_words, unit):
    seps = [" ", " ", " ", "  ", "\n", "\t", " \n "]
    words = [f"{'abcdefgh'[int(rng.integers(8))]}{i}" for i in range(n_words)]
    text = "".join(w + seps[int(rng.integers(len(seps)))] for w in words)
    return (" \n" + text) if unit == "word" else text


@pytest.mark.parametrize("n,size,overlap,unit", [
    pytest.param(1, 3, 1, "word", id="word-1-unit"),
    pytest.param(37, 8, 3, "word", id="word-37-8-3"),
    pytest.param(100, 10, 0, "word", id="word-100-10-0-exact"),
    pytest.param(101, 10, 9, "word", id="word-101-10-9"),
    pytest.param(64, 7, 6, "word", id="word-64-7-6"),
    pytest.param(250, 50, 20, "word", id="word-250-50-20"),
    pytest.param(30, 16, 5, "char", id="char-30-words-16-5"),
    pytest.param(12, 9, 0, "char", id="char-12-words-9-0"),
    pytest.param(25, 5, 4, "char", id="char-25-words-5-4"),
])
def test_chunk_text_properties(ret, n, size, overlap, unit):
    rng = np.random.default_rng(n * 31 + size)
    text = _random_text(rng, n, unit)
    units = text.split() if unit == "word" else list(text)
    got = chunks_of(ret, text, chunk_size=size, overlap=overlap, unit=unit)
    msg = f"chunk_text of {len(units)} {unit}s, chunk_size={size}, overlap={overlap}"
    parts = [c.split(" ") if unit == "word" else list(c) for c in got]
    want = chunk_oracle(text, size, overlap, unit)
    if len(got) != len(want):
        _fail(msg, f"expected {len(want)} chunks (starts every {size - overlap} units, stop after the first chunk "
                   f"that reaches the end), got {len(got)}")
    for j, p in enumerate(parts[:-1]):
        if len(p) != size:
            _fail(msg, f"expected every chunk but the last to have {size} {unit}s, chunk {j} has {len(p)}")
    covered = set()
    for j, p in enumerate(parts):
        start = j * (size - overlap)
        if p != units[start:start + size]:
            _fail(msg, f"expected chunk {j} to be the {unit}s {start} to {min(start + size, len(units)) - 1} of the "
                       f"text, got {got[j]!r}")
        covered.update(range(start, start + len(p)))
    if covered != set(range(len(units))):
        _fail(msg, f"expected every {unit} in at least one chunk, {len(units) - len(covered)} missing")
    for j in range(len(parts) - 1):
        if overlap and parts[j][-overlap:] != parts[j + 1][:overlap]:
            _fail(msg, f"expected chunks {j} and {j + 1} to share exactly {overlap} {unit}s")
    rebuilt = parts[0] + [u for p in parts[1:] for u in p[overlap:]] if parts else []
    if rebuilt != units:
        _fail(msg, "expected the chunks, overlaps removed, to give back the text"
                   + (" (whitespace-normalised)" if unit == "word" else " exactly"))
    if unit == "char" and "".join([got[0]] + [c[overlap:] for c in got[1:]]) != text:
        _fail(msg, "expected the char chunks, overlaps removed, to give back the text exactly")


def test_chunk_text_normalises_whitespace_in_word_mode(ret):
    got = chunks_of(ret, "  one\ttwo\r\n\nthree   four \n five  ", chunk_size=2, overlap=0)
    assert_list_equal(got, ["one two", "three four", "five"],
                      "chunk_text with tabs, newlines and repeated spaces, chunk_size=2, overlap=0 (str.split(), "
                      "then single spaces)", "chunk")


@pytest.mark.parametrize("text,unit,want", [
    pytest.param("", "word", [], id="empty-word"),
    pytest.param("", "char", [], id="empty-char"),
    pytest.param("  \n\t ", "word", [], id="only-whitespace-word"),
    pytest.param("  \n", "char", ["  \n"], id="only-whitespace-char-is-3-chars"),
])
def test_chunk_text_text_with_no_unit(ret, text, unit, want):
    got = chunks_of(ret, text, chunk_size=4, overlap=1, unit=unit)
    assert_list_equal(got, want, f"chunk_text({text!r}, chunk_size=4, overlap=1, unit={unit!r})", "chunk")


@pytest.mark.parametrize("kwargs,why", [
    pytest.param({"chunk_size": 0, "overlap": 0}, "chunk_size = 0", id="chunk-size-0"),
    pytest.param({"chunk_size": -3, "overlap": 0}, "chunk_size = -3", id="chunk-size-negative"),
    pytest.param({"chunk_size": 4, "overlap": -1}, "overlap = -1", id="overlap-negative"),
    pytest.param({"chunk_size": 4, "overlap": 4}, "overlap = chunk_size = 4 (the window would not move)",
                 id="overlap-equals-chunk-size"),
    pytest.param({"chunk_size": 1, "overlap": 1}, "overlap = chunk_size = 1", id="overlap-equals-chunk-size-1"),
    pytest.param({"chunk_size": 3, "overlap": 5}, "overlap = 5 > chunk_size = 3", id="overlap-larger"),
    pytest.param({"chunk_size": 3, "overlap": 1, "unit": "token"}, "unit = 'token'", id="unit-token"),
    pytest.param({"chunk_size": 3, "overlap": 1, "unit": "words"}, "unit = 'words'", id="unit-words"),
])
def test_chunk_text_refuses_invalid_arguments(ret, kwargs, why):
    try:
        bounded(assert_raises, ValueError, ret.chunk_text, "a b c d e f g", why=why, **kwargs)
    except _Runaway:
        _fail("chunk_text", f"expected chunk_text to raise ValueError here: {why}; it never stopped (with "
                            "overlap >= chunk_size the window does not move: check the arguments first)")


# ================================================================== BM25 (B4.17)
DOC_CORPUS = [["the", "cat", "sat"], ["the", "dog", "sat", "down"], ["a", "cat", "and", "a", "cat"]]


def bm25_oracle(corpus, query, k1=1.5, b=0.75):
    """The docstring formula with explicit loops and math.log; checked against rank_bm25 when it is installed."""
    corpus = [list(d) for d in corpus]
    N = len(corpus)
    avgdl = sum(len(d) for d in corpus) / N
    vocab = {t for d in corpus for t in d}
    idf = {t: math.log(1 + (N - sum(t in d for d in corpus) + 0.5) / (sum(t in d for d in corpus) + 0.5))
           for t in vocab}
    scores = []
    for d in corpus:
        s = 0.0
        for t in query:
            f = d.count(t)
            if f:
                s += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    scores = np.array(scores)
    if BM25Okapi is not None and k1 > 0 and all(corpus):
        okapi = BM25Okapi(corpus, k1=k1, b=b)
        okapi.idf = idf                                   # the only difference with the docstring: its idf
        assert np.allclose(okapi.get_scores(list(query)), scores, rtol=1e-12, atol=1e-12), ORACLE_BUG
        assert okapi.doc_freqs == [{t: d.count(t) for t in set(d)} for d in corpus], ORACLE_BUG
        assert math.isclose(okapi.avgdl, avgdl), ORACLE_BUG
    return scores, idf, avgdl


def fitted_bm25(ret, corpus, **params):
    """BM25(**params).fit(copy of corpus), checked to return the object itself."""
    model = call(ret.BM25, expected="a BM25 object", **params)
    out = call(model.fit, copy.deepcopy(corpus), expected="the fitted BM25 (self)")
    if out is None:
        _fail("BM25.fit", "expected fit to return self, got None (add `return self` at the end of fit)")
    if out is not model:
        _fail("BM25.fit", f"expected fit to return self (the same object), got {type(out).__name__}")
    return model


def scores_of(model, query, n_docs, msg):
    result = call(model.get_scores, list(query), expected="an array of scores")
    if result is None:
        _fail(msg, "expected an array of scores, got None (did you forget the return?)")
    if not isinstance(result, np.ndarray):
        _fail(msg, f"expected a NumPy array of scores, got {type(result).__name__}")
    if result.shape != (n_docs,):
        _fail(msg, f"expected scores of shape ({n_docs},) (one per document), got shape {result.shape}")
    if not np.issubdtype(result.dtype, np.floating):
        _fail(msg, f"expected float scores, got dtype {result.dtype}")
    return result


def test_bm25_docstring_examples(ret):
    bm25 = fitted_bm25(ret, DOC_CORPUS)
    if bm25.n_docs_ != 3:
        _fail("BM25().fit(corpus of the docstring)", f"expected n_docs_ = 3, got {bm25.n_docs_!r}")
    assert_number(bm25.avgdl_, 4.0, "avgdl_ of the docstring corpus (lengths 3, 4, 5)")
    assert_close(bm25.idf_["cat"], 0.47, rtol=0, atol=5e-5, msg='idf_["cat"] of the docstring')
    assert_close(bm25.idf_["dog"], 0.9808, rtol=0, atol=5e-5, msg='idf_["dog"] of the docstring')
    assert_close(scores_of(bm25, ["cat"], 3, "get_scores"), [0.5296, 0.0, 0.6215], rtol=0, atol=5e-5,
                 msg='get_scores(["cat"]) of the docstring')
    assert_close(scores_of(bm25, ["cat", "sat"], 3, "get_scores"), [1.0592, 0.47, 0.6215], rtol=0, atol=5e-5,
                 msg='get_scores(["cat", "sat"]) of the docstring')


def test_bm25_fit_returns_self(ret):
    fitted_bm25(ret, DOC_CORPUS)


def test_bm25_fit_learns_the_statistics_by_hand(ret):
    bm25 = fitted_bm25(ret, DOC_CORPUS)
    msg = "BM25().fit(corpus of the docstring)"
    tf = bm25.term_freqs_
    if not isinstance(tf, list) or len(tf) != 3:
        _fail(msg, f"expected term_freqs_ to be a list of 3 dicts (one per document), got {type(tf).__name__}"
                   + (f" of length {len(tf)}" if hasattr(tf, "__len__") else ""))
    want_tf = [{"the": 1, "cat": 1, "sat": 1}, {"the": 1, "dog": 1, "sat": 1, "down": 1}, {"a": 2, "cat": 2, "and": 1}]
    for j, (got, want) in enumerate(zip(tf, want_tf)):
        if not isinstance(got, dict):
            _fail(msg, f"expected term_freqs_[{j}] to be a dict, got {type(got).__name__}")
        if {t: c for t, c in got.items() if c} != want:
            _fail(msg, f"expected term_freqs_[{j}] = {want}, got {dict(got)}")
    if not isinstance(bm25.doc_len_, np.ndarray):
        _fail(msg, f"expected doc_len_ to be a NumPy array, got {type(bm25.doc_len_).__name__}")
    assert_close(bm25.doc_len_, [3, 4, 5], msg=msg + ": doc_len_ (number of tokens of each document)")
    if not isinstance(bm25.n_docs_, numbers.Integral) or bm25.n_docs_ != 3:
        _fail(msg, f"expected n_docs_ = 3 (an int), got {bm25.n_docs_!r}")
    idf = bm25.idf_
    if not isinstance(idf, dict):
        _fail(msg, f"expected idf_ to be a dict, got {type(idf).__name__}")
    vocab = {"the", "cat", "sat", "dog", "down", "a", "and"}
    if set(idf) != vocab:
        _fail(msg, f"expected idf_ to have one entry per term of the corpus {sorted(vocab)}, got {sorted(idf)}")
    # N = 3: a term in 2 documents has idf ln(1 + 1.5 / 2.5) = ln(1.6); a term in 1 document ln(1 + 2.5 / 1.5)
    want = {t: math.log(1.6) for t in ("the", "cat", "sat")} | {t: math.log(8 / 3) for t in ("dog", "down", "a", "and")}
    for t in sorted(vocab):
        assert_close(idf[t], want[t], msg=f'{msg}: idf_["{t}"] (n_t = {2 if want[t] < 0.6 else 1} of N = 3, '
                                          "ln(1 + (N - n_t + 0.5) / (n_t + 0.5)))")


def test_bm25_idf_is_positive_even_for_a_term_in_every_document(ret):
    corpus = [["x", "a"], ["x", "b", "b"], ["x", "x", "c"], ["x"]]
    bm25 = fitted_bm25(ret, corpus)
    # N = 4, n_x = 4: ln(1 + 0.5 / 4.5) = ln(10 / 9) > 0 (the idf of rank_bm25 would be negative)
    assert_close(bm25.idf_["x"], math.log(10 / 9), msg='idf_["x"] for a term in all 4 documents (always positive: '
                                                       "ln(1 + (N - n_t + 0.5) / (n_t + 0.5)))")
    scores = scores_of(bm25, ["x"], 4, "get_scores")
    if (scores <= 0).any():
        _fail('get_scores(["x"])', f"expected a positive score for every document (each contains x), "
                                   f"got {_short(scores)}")


def test_bm25_scores_by_hand_on_a_3_document_corpus(ret):
    corpus = [["a", "b"], ["a", "a", "c"], ["b", "c", "c", "d"]]
    bm25 = fitted_bm25(ret, corpus, k1=1.2, b=0.5)
    # avgdl = 3; K(d) = k1 (1 - b + b |d| / avgdl) = 1.0, 1.2, 1.4 for |d| = 2, 3, 4
    # idf(a) = idf(b) = idf(c) = ln(1.6) (2 documents of 3), idf(d) = ln(8 / 3) (1 document)
    L, D = math.log(1.6), math.log(8 / 3)
    msg = "BM25(k1=1.2, b=0.5) on [[a, b], [a, a, c], [b, c, c, d]]"
    assert_close(scores_of(bm25, ["a"], 3, msg), [L * 1 * 2.2 / (1 + 1.0), L * 2 * 2.2 / (2 + 1.2), 0.0],
                 msg=msg + ', get_scores(["a"]) (f = 1, 2, 0)')
    assert_close(scores_of(bm25, ["c", "d"], 3, msg),
                 [0.0, L * 1 * 2.2 / (1 + 1.2), L * 2 * 2.2 / (2 + 1.4) + D * 1 * 2.2 / (1 + 1.4)],
                 msg=msg + ', get_scores(["c", "d"])')
    assert_close(scores_of(bm25, ["b"], 3, msg), [L * 2.2 / 2.0, 0.0, L * 2.2 / 2.4],
                 msg=msg + ', get_scores(["b"]) (same f = 1, the longer document scores less)')
    want, _, _ = bm25_oracle(corpus, ["a", "b", "c", "d"], k1=1.2, b=0.5)
    assert_close(scores_of(bm25, ["a", "b", "c", "d"], 3, msg), want, msg=msg + ", get_scores([a, b, c, d])")


BM25_CASES = [
    # id, n_docs, vocab size, max length, k1, b, with empty documents
    ("defaults", 12, 15, 10, 1.5, 0.75, False),
    ("k1-1.2-b-1", 20, 30, 15, 1.2, 1.0, False),
    ("k1-2-b-0", 8, 10, 12, 2.0, 0.0, False),
    ("k1-0.5-b-0.3", 30, 40, 20, 0.5, 0.3, False),
    ("k1-0-ignores-frequency", 10, 12, 8, 0.0, 0.75, False),
    ("small-vocab-many-repeats", 6, 4, 25, 1.5, 0.75, False),
    ("empty-documents", 9, 10, 8, 1.5, 0.75, True),
    ("empty-documents-b-1", 9, 10, 8, 1.2, 1.0, True),
]


def _zipf_corpus(rng, n_docs, vocab, max_len, empty):
    p = 1.0 / np.arange(1, vocab + 1)
    p /= p.sum()
    words = [f"t{i}" for i in range(vocab)]
    corpus = [[words[int(i)] for i in rng.choice(vocab, size=int(rng.integers(1, max_len + 1)), p=p)]
              for _ in range(n_docs)]
    if empty:
        corpus[1] = []
        corpus[-1] = []
    return corpus, words


@pytest.mark.parametrize("case", BM25_CASES, ids=[c[0] for c in BM25_CASES])
def test_bm25_matches_the_formula_of_the_docstring(ret, case):
    name, n_docs, vocab, max_len, k1, b, empty = case
    rng = np.random.default_rng(len(name) * 7 + n_docs)
    corpus, words = _zipf_corpus(rng, n_docs, vocab, max_len, empty)
    bm25 = fitted_bm25(ret, corpus, k1=k1, b=b)
    _, idf, avgdl = bm25_oracle(corpus, [], k1, b)
    msg = f"BM25(k1={k1}, b={b}), case {name} ({n_docs} documents)"
    assert_number(bm25.avgdl_, avgdl, msg + ": avgdl_ (mean length over all the documents"
                  + (", empty ones included)" if empty else ")"))
    if set(bm25.idf_) != set(idf):
        _fail(msg, f"expected idf_ for the {len(idf)} terms of the corpus, got {len(bm25.idf_)} terms")
    for t in sorted(idf):
        assert_close(bm25.idf_[t], idf[t], msg=msg + f': idf_["{t}"]')
    queries = [[words[0]], [words[1], words[2]], [words[0], words[0], words[3]], words[:6],
               [words[-1], "unknown-term"], [str(w) for w in rng.choice(words, size=5)]]
    for q in queries:
        want, _, _ = bm25_oracle(corpus, q, k1, b)
        got = scores_of(bm25, q, n_docs, msg)
        if np.isnan(got).any():
            _fail(msg + f", get_scores({q})", f"expected no NaN, got NaN for document(s) "
                  f"{np.flatnonzero(np.isnan(got)).tolist()} (a document without the term gets 0 from it, even "
                  "when the denominator is 0)")
        assert_close(got, want, msg=msg + f", get_scores({q})")


def test_bm25_score_increases_and_saturates_with_term_frequency(ret):
    L = 10
    corpus = [["t"] * f + [f"x{f}"] * (L - f) for f in range(1, L + 1)] + [["y"] * L] * 5
    for k1, b in ((1.5, 0.75), (1.2, 0.0), (3.0, 1.0)):
        bm25 = fitted_bm25(ret, corpus, k1=k1, b=b)
        s = scores_of(bm25, ["t"], len(corpus), "get_scores")[:L]
        msg = f"BM25(k1={k1}, b={b}), get_scores(['t']) for documents of length {L} with f = 1 .. {L}"
        gains = np.diff(s)
        if (gains <= 0).any():
            _fail(msg, f"expected the score to increase with f, got {_short(s)}")
        if (np.diff(gains) >= 0).any():
            _fail(msg, f"expected smaller and smaller gains (saturation), got the gains {_short(gains)}")
        bound = bm25.idf_["t"] * (k1 + 1)
        if (s >= bound).any():
            _fail(msg, f"expected every score below idf * (k1 + 1) = {bound:.6g}, got {_short(s)}")


def test_bm25_b_zero_removes_length_normalisation(ret):
    corpus = [["t", "u"], ["t"] + ["u"] * 9, ["v", "w", "w"], ["t", "t", "v"]]
    flat = fitted_bm25(ret, corpus, k1=1.5, b=0.0)
    s = scores_of(flat, ["t"], 4, "get_scores")
    idf = flat.idf_["t"]
    assert_close(s, [idf * 2.5 / 2.5, idf * 2.5 / 2.5, 0.0, idf * 2 * 2.5 / 3.5],
                 msg="BM25(b=0), get_scores(['t']): idf f (k1 + 1) / (f + k1), whatever the length")
    norm = fitted_bm25(ret, corpus, k1=1.5, b=0.75)
    s2 = scores_of(norm, ["t"], 4, "get_scores")
    if not s2[0] > s2[1]:
        _fail("BM25(b=0.75), get_scores(['t'])", f"expected the 2-token document to score more than the 10-token one "
                                                 f"(same f = 1), got {s2[0]:.6g} and {s2[1]:.6g}")


def test_bm25_k1_zero_ignores_the_frequency(ret):
    corpus = [["t", "u"], ["t", "t", "t", "u"], ["u", "v"], []]
    bm25 = fitted_bm25(ret, corpus, k1=0.0, b=0.75)
    s = scores_of(bm25, ["t", "v", "t"], 4, "get_scores")
    it, iv = bm25.idf_["t"], bm25.idf_["v"]
    if np.isnan(s).any():
        _fail("BM25(k1=0), get_scores(['t', 'v', 't'])", "expected no NaN, got NaN (with k1 = 0, a document "
              "without the term gets 0 from it: 0 / 0 must not be computed)")
    assert_close(s, [2 * it, 2 * it, iv, 0.0], msg="BM25(k1=0), get_scores(['t', 'v', 't']) (each occurrence of a "
                                                  "query term adds idf if the document contains it, whatever f)")


def test_bm25_unknown_query_terms_add_zero(ret):
    bm25 = fitted_bm25(ret, DOC_CORPUS)
    base = scores_of(bm25, ["cat", "sat"], 3, "get_scores")
    with_unknown = scores_of(bm25, ["zebra", "cat", "CAT", "sat", "unicorn"], 3, "get_scores")
    assert_close(with_unknown, base, msg='get_scores(["zebra", "cat", "CAT", "sat", "unicorn"]) (unknown terms, '
                                         'and "CAT" is not "cat", add 0)')
    assert_close(scores_of(bm25, ["zebra"], 3, "get_scores"), [0.0, 0.0, 0.0],
                 msg='get_scores(["zebra"]) (only unknown terms: every score is 0)')
    assert_close(scores_of(bm25, [], 3, "get_scores"), [0.0, 0.0, 0.0], msg="get_scores([]) (empty query)")


def test_bm25_repeated_query_terms_count_several_times(ret):
    bm25 = fitted_bm25(ret, DOC_CORPUS)
    once = scores_of(bm25, ["cat"], 3, "get_scores")
    assert_close(scores_of(bm25, ["cat", "cat"], 3, "get_scores"), 2 * once,
                 msg='get_scores(["cat", "cat"]) (twice the contribution of "cat", as in rank_bm25)')
    assert_close(scores_of(bm25, ["dog", "cat", "dog", "dog"], 3, "get_scores"),
                 3 * scores_of(bm25, ["dog"], 3, "get_scores") + once, msg='get_scores(["dog", "cat", "dog", "dog"])')


def test_bm25_empty_documents_count_in_n_and_avgdl(ret):
    corpus = [["a", "b"], [], ["b", "c", "c", "c"]]
    bm25 = fitted_bm25(ret, corpus)
    msg = "BM25().fit([[a, b], [], [b, c, c, c]])"
    if bm25.n_docs_ != 3:
        _fail(msg, f"expected n_docs_ = 3 (the empty document counts), got {bm25.n_docs_!r}")
    assert_number(bm25.avgdl_, 2.0, msg + ": avgdl_ = (2 + 0 + 4) / 3")
    assert_close(bm25.idf_["a"], math.log(1 + 2.5 / 1.5), msg=msg + ': idf_["a"] (N = 3)')
    s = scores_of(bm25, ["a", "b", "c"], 3, msg)
    want, _, _ = bm25_oracle(corpus, ["a", "b", "c"])
    assert_close(s, want, msg=msg + ", get_scores([a, b, c]) (the empty document scores 0)")
    full = fitted_bm25(ret, corpus, b=1.0)
    s1 = scores_of(full, ["a", "b", "c"], 3, msg)
    if np.isnan(s1).any():
        _fail("BM25(b=1).fit([[a, b], [], [b, c, c, c]]).get_scores([a, b, c])",
              "expected no NaN, got NaN for the empty document (it contains no term: it gets 0, even though "
              "k1 (1 - b + b |d| / avgdl) = 0 there)")
    want1, _, _ = bm25_oracle(corpus, ["a", "b", "c"], b=1.0)
    assert_close(s1, want1, msg="BM25(b=1) with an empty document, get_scores([a, b, c])")


def test_bm25_accepts_tuples(ret):
    corpus = tuple(tuple(d) for d in DOC_CORPUS)
    bm25 = fitted_bm25(ret, corpus)
    want, _, _ = bm25_oracle(DOC_CORPUS, ["cat", "sat"])
    got = call(bm25.get_scores, ("cat", "sat"), expected="an array of scores")
    assert_close(got, want, msg="BM25 fitted on a tuple of tuples, get_scores(('cat', 'sat'))")


def test_bm25_does_not_modify_the_corpus_or_the_query(ret):
    corpus = [list(d) for d in DOC_CORPUS]
    saved = copy.deepcopy(corpus)
    bm25 = call(ret.BM25, expected="a BM25 object")
    call(bm25.fit, corpus, expected="the fitted BM25")
    if corpus != saved:
        _fail("", "expected the corpus unchanged by fit (count the terms without changing the documents)")
    query = ["cat", "zebra", "cat"]
    call(bm25.get_scores, query, expected="an array of scores")
    if query != ["cat", "zebra", "cat"]:
        _fail("", "expected the query unchanged by get_scores")
    call(bm25.get_scores, ["dog"], expected="an array of scores")
    if corpus != saved:
        _fail("", "expected the corpus unchanged by get_scores")


def test_bm25_fit_again_replaces_the_statistics(ret):
    other = [["x", "y"], ["y", "z", "z"]]
    bm25 = fitted_bm25(ret, DOC_CORPUS)
    out = call(bm25.fit, copy.deepcopy(other), expected="the fitted BM25 (self)")
    if out is not bm25:
        _fail("BM25.fit", "expected fit to return self")
    msg = "BM25 fitted on the docstring corpus, then on [[x, y], [y, z, z]]"
    if bm25.n_docs_ != 2:
        _fail(msg, f"expected n_docs_ = 2 (the statistics of the new corpus only), got {bm25.n_docs_!r}")
    if set(bm25.idf_) != {"x", "y", "z"}:
        _fail(msg, f"expected idf_ for x, y and z only, got {sorted(bm25.idf_)}")
    assert_number(bm25.avgdl_, 2.5, msg + ": avgdl_")
    want, _, _ = bm25_oracle(other, ["z", "cat", "y"])
    assert_close(scores_of(bm25, ["z", "cat", "y"], 2, msg), want, msg=msg + ", get_scores([z, cat, y])")


def _new_fit(ret, corpus, **params):
    """BM25(**params).fit(corpus): a ValueError may come from either (fit must raise at the latest)."""
    return ret.BM25(**params).fit(corpus)


_new_fit.__name__ = "BM25(...).fit"


@pytest.mark.parametrize("corpus,params,why", [
    pytest.param(DOC_CORPUS, {"k1": -0.5}, "k1 = -0.5 < 0", id="k1-negative"),
    pytest.param(DOC_CORPUS, {"b": -0.1}, "b = -0.1 outside [0, 1]", id="b-negative"),
    pytest.param(DOC_CORPUS, {"b": 1.5}, "b = 1.5 outside [0, 1]", id="b-above-1"),
    pytest.param([], {}, "an empty corpus (no document)", id="no-document"),
    pytest.param([[], []], {}, "a corpus of empty documents only", id="only-empty-documents"),
])
def test_bm25_fit_refuses_invalid_arguments(ret, corpus, params, why):
    assert_raises(ValueError, _new_fit, ret, corpus, why=why, **params)


def test_bm25_get_scores_before_fit_raises(ret):
    model = call(ret.BM25, expected="a BM25 object")
    assert_raises(ValueError, model.get_scores, ["cat"], why="get_scores called before fit")


# ================================================================== cosine_top_k (B4.18)
DOC_DOCS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]


def top_k_of(ret, query, docs, k, shape):
    """The learner's (indices, similarities) on copies, checked: NumPy arrays of the expected shape and kinds."""
    args = [np.array(a, copy=True) if isinstance(a, np.ndarray) else copy.deepcopy(a) for a in (query, docs)]
    result = call(ret.cosine_top_k, *args, k=k, expected="the tuple (indices, similarities)")
    msg = f"cosine_top_k(query of shape {np.shape(query)}, docs of shape {np.shape(docs)}, k={k})"
    if result is None:
        _fail(msg, "expected the tuple (indices, similarities), got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 2:
        _fail(msg, f"expected the tuple (indices, similarities), got {type(result).__name__}"
                   + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else ""))
    idx, sims = result
    for value, what in ((idx, "indices"), (sims, "similarities")):
        if not isinstance(value, np.ndarray):
            _fail(msg, f"expected {what} as a NumPy array, got {type(value).__name__}")
        if value.shape != shape:
            _fail(msg, f"expected {what} of shape {shape}, got shape {value.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        _fail(msg, f"expected integer indices, got dtype {idx.dtype}")
    if not np.issubdtype(sims.dtype, np.floating):
        _fail(msg, f"expected float similarities, got dtype {sims.dtype}")
    return idx, sims, msg


def assert_indices(idx, want, msg):
    got, want = np.asarray(idx), np.asarray(want)
    if not np.array_equal(got, want):
        pos = tuple(int(i) for i in np.argwhere(got != want)[0])
        _fail(msg, f"expected index {int(want[pos])} at position {pos}, got {int(got[pos])}",
              f"expected {_short(want)}\n     got {_short(got)}" if want.size <= 40 else "")


def topk_oracle(Q, D, k):
    """Cosine of the docstring of cosine_similarity_matrix (eps = 1e-12), then a stable sort of each row."""
    Q, D = np.atleast_2d(np.asarray(Q, dtype=float)), np.asarray(D, dtype=float)
    S = np.array([[q @ d / ((math.sqrt(q @ q) + 1e-12) * (math.sqrt(d @ d) + 1e-12)) for d in D] for q in Q])
    idx = np.array([sorted(range(D.shape[0]), key=lambda j: (-row[j], j))[:k] for row in S], dtype=int)
    return idx, np.take_along_axis(S, idx, axis=1)


def test_cosine_top_k_docstring_examples(ret):
    idx, sims, msg = top_k_of(ret, [1.0, 0.2], DOC_DOCS, 2, (2,))
    assert_indices(idx, [0, 2], msg + " of the docstring")
    assert_close(sims, [0.9806, 0.8321], rtol=0, atol=5e-5, msg=msg + ": the similarities of the docstring")
    idx2, _, msg2 = top_k_of(ret, [[1.0, 0.2], [0.0, -1.0]], DOC_DOCS, 2, (2, 2))
    assert_indices(idx2, [[0, 2], [0, 3]], msg2 + " of the docstring (second query: tie at 0 between the documents "
                                                  "0 and 3, smaller index first)")


@pytest.mark.parametrize("n,d,q,k", [
    pytest.param(50, 8, 1, 5, id="50x8-one-query-2d-k5"),
    pytest.param(200, 16, 7, 10, id="200x16-7-queries-k10"),
    pytest.param(30, 4, 3, 30, id="30x4-k-equals-n"),
    pytest.param(12, 3, 5, 1, id="12x3-k1"),
    pytest.param(40, 32, 4, 6, id="40x32-varied-norms"),
])
def test_cosine_top_k_matches_nearest_neighbors(ret, n, d, q, k):
    rng = np.random.default_rng(n + d + q + k)
    D = rng.normal(size=(n, d)) * rng.uniform(0.1, 10.0, size=(n, 1))
    Q = rng.normal(size=(q, d)) * rng.uniform(0.1, 10.0, size=(q, 1))
    want_idx, want_sims = topk_oracle(Q, D, k)
    dist, pos = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(D).kneighbors(Q)
    assert np.array_equal(pos, want_idx), ORACLE_BUG
    assert np.allclose(1 - dist, want_sims, rtol=1e-9, atol=1e-9), ORACLE_BUG
    idx, sims, msg = top_k_of(ret, Q, D, k, (q, k))
    assert_indices(idx, want_idx, msg + " (by decreasing cosine similarity, NearestNeighbors(metric='cosine'))")
    assert_close(sims, want_sims, msg=msg + ": similarities (1 - the distances of NearestNeighbors)")


def test_cosine_top_k_single_query_gives_1d_arrays(ret):
    rng = np.random.default_rng(3)
    D, query = rng.normal(size=(20, 5)), rng.normal(size=5)
    want_idx, want_sims = topk_oracle(query, D, 4)
    idx, sims, msg = top_k_of(ret, query, D, 4, (4,))
    assert_indices(idx, want_idx[0], msg + " (a query of shape (d,) gives arrays of shape (k,))")
    assert_close(sims, want_sims[0], msg=msg + ": similarities")
    idx2, sims2, msg2 = top_k_of(ret, query[None, :], D, 4, (1, 4))
    assert_indices(idx2, want_idx, msg2 + " (a query of shape (1, d) gives arrays of shape (1, k))")
    assert_close(sims2, want_sims, msg=msg2 + ": similarities")


def test_cosine_top_k_uses_the_cosine_not_the_dot_product(ret):
    D = np.array([[10.0, 10.0], [1.0, 0.1], [0.5, -0.4]])
    idx, sims, msg = top_k_of(ret, [3.0, 0.0], D, 3, (3,))
    assert_indices(idx, [1, 2, 0], msg + " (the long row [10, 10] has the largest dot product, not the largest "
                                         "cosine: divide by both norms)")
    assert_close(sims, topk_oracle([3.0, 0.0], D, 3)[1][0], msg=msg + ": cosine similarities (in [-1, 1])")


def test_cosine_top_k_ties_go_to_the_smaller_index(ret):
    E = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    idx, _, msg = top_k_of(ret, [1.0, 0.0], E, 4, (4,))
    assert_indices(idx, [1, 3, 4, 0], msg + ": three identical rows 1, 3, 4 (ties: smaller index first), then row 0")
    rows = np.repeat(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 40, axis=0)
    big = rows[np.random.default_rng(0).permutation(120)]
    want, _ = topk_oracle(np.array([[1.0, 0.2], [-1.0, 0.0]]), big, 60)
    idx2, _, msg2 = top_k_of(ret, np.array([[1.0, 0.2], [-1.0, 0.0]]), big, 60, (2, 60))
    assert_indices(idx2, want, msg2 + ": 120 rows, three distinct ones repeated 40 times (each tie in increasing "
                                      "index order: a stable sort of each row)")
    idx3, sims3, msg3 = top_k_of(ret, [0.0, 0.0], E, 3, (3,))
    assert_indices(idx3, [0, 1, 2], msg3 + ": a zero query (every similarity is 0, so the 3 smallest indices)")
    assert_close(sims3, [0.0, 0.0, 0.0], msg=msg3 + ": a zero query has similarity 0 with every row")
    Z = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [2.0, 1.0]])
    idx4, sims4, msg4 = top_k_of(ret, [1.0, 0.0], Z, 4, (4,))
    if np.isnan(sims4).any():
        _fail(msg4, "expected a similarity of 0 for a row of zeros (as cosine_similarity_matrix), got NaN")
    assert_indices(idx4, [3, 0, 2, 1], msg4 + ": two rows of zeros (similarity 0, smaller index first)")


def test_cosine_top_k_accepts_lists(ret):
    idx, sims, msg = top_k_of(ret, [[1, 0], [0, 2]], [[1, 0], [0, 1], [1, 1], [-1, 0]], 3, (2, 3))
    assert_indices(idx, [[0, 2, 1], [1, 2, 0]], msg + " (lists of ints)")
    assert_close(sims, topk_oracle([[1, 0], [0, 2]], DOC_DOCS, 3)[1], msg=msg + ": similarities (lists of ints)")


def test_cosine_top_k_does_not_modify_its_inputs(ret):
    rng = np.random.default_rng(8)
    D, Q = rng.normal(size=(10, 4)), rng.normal(size=(3, 4))
    sd, sq = D.copy(), Q.copy()
    call(ret.cosine_top_k, Q, D, k=3, expected="the tuple (indices, similarities)")
    if not (np.array_equal(D, sd) and np.array_equal(Q, sq)):
        _fail("", "expected query_emb and doc_embs unchanged by cosine_top_k (normalise copies)")


@pytest.mark.parametrize("query,k,why", [
    pytest.param([1.0, 0.0], 0, "k = 0", id="k-0"),
    pytest.param([1.0, 0.0], -2, "k = -2", id="k-negative"),
    pytest.param([1.0, 0.0], 5, "k = 5 > n = 4 documents", id="k-larger-than-n"),
    pytest.param([[1.0, 0.0], [0.0, 1.0]], 5, "k = 5 > n = 4 documents (two queries)", id="k-larger-than-n-2d"),
    pytest.param([1.0, 0.0, 0.0], 2, "a query of dimension 3 for documents of dimension 2", id="query-dimension"),
    pytest.param([[1.0, 0.0, 0.0]], 2, "queries of dimension 3 for documents of dimension 2", id="queries-dimension"),
])
def test_cosine_top_k_refuses_invalid_arguments(ret, query, k, why):
    assert_raises(ValueError, ret.cosine_top_k, query, np.array(DOC_DOCS), k=k, why=why)


# ================================================================== reciprocal_rank_fusion (B4.18)
def rrf_oracle(rankings, k=60):
    """Exact fused scores (fractions), sorted by decreasing score then increasing id."""
    scores: dict[int, Fraction] = {}
    for r in rankings:
        for rank, doc in enumerate(r, start=1):
            scores[int(doc)] = scores.get(int(doc), Fraction(0)) + Fraction(1) / (Fraction(k) + rank)
    return sorted(scores, key=lambda d: (-scores[d], d)), scores


def fused_of(ret, rankings, msg, **kwargs):
    """The learner's fused ranking on a copy, checked: a 1-D NumPy array of integer ids; returned as a list."""
    arg = [np.array(r, copy=True) if isinstance(r, np.ndarray) else copy.deepcopy(r) for r in rankings]
    result = call(ret.reciprocal_rank_fusion, arg, expected="an array of ids", **kwargs)
    if result is None:
        _fail(msg, "expected an array of ids, got None (did you forget the return?)")
    if not isinstance(result, np.ndarray):
        _fail(msg, f"expected a NumPy array of ids, got {type(result).__name__}")
    if result.ndim != 1:
        _fail(msg, f"expected a 1-D array of ids, got shape {result.shape}")
    if result.size and not np.issubdtype(result.dtype, np.integer):
        _fail(msg, f"expected integer ids, got dtype {result.dtype}")
    return [int(i) for i in result]


def test_reciprocal_rank_fusion_docstring_examples(ret):
    got = fused_of(ret, [[3, 1, 2], [1, 2, 4]], "reciprocal_rank_fusion")
    assert_list_equal(got, [1, 2, 3, 4], "reciprocal_rank_fusion([[3, 1, 2], [1, 2, 4]]) of the docstring", "id")
    got = fused_of(ret, [[3, 1, 2], [1, 2, 4]], "reciprocal_rank_fusion", k=0)
    assert_list_equal(got, [1, 3, 2, 4], "reciprocal_rank_fusion([[3, 1, 2], [1, 2, 4]], k=0) of the docstring "
                                         "(ranks start at 1: score 1 / rank)", "id")


def test_reciprocal_rank_fusion_hand_computed_examples(ret):
    # k = 1, ranks from 1: id 3 -> 1/2; id 1 -> 1/3 + 1/2 = 5/6; id 2 -> 1/4 + 1/3 = 7/12; id 4 -> 1/4
    got = fused_of(ret, [[3, 1, 2], [1, 2, 4]], "reciprocal_rank_fusion", k=1)
    assert_list_equal(got, [1, 2, 3, 4], "reciprocal_rank_fusion([[3, 1, 2], [1, 2, 4]], k=1): scores 5/6, 7/12, "
                                         "1/2, 1/4 (ranks start at 1)", "id")
    # three rankings, k = 1: 20 -> 1/3 + 1/2 + 1/3 = 7/6; 10 -> 1/2 + 1/4 = 3/4; 30 -> 1/4 + 1/2 = 3/4 (tie);
    # 40 -> 1/3; 50 -> 1/5
    rankings = [[10, 20, 30], [20, 40], [30, 20, 10, 50]]
    got = fused_of(ret, rankings, "reciprocal_rank_fusion", k=1)
    assert_list_equal(got, [20, 10, 30, 40, 50], f"reciprocal_rank_fusion({rankings}, k=1): scores 7/6, 3/4, 3/4, "
                                                 "1/3, 1/5 (tie between 10 and 30: smaller id first)", "id")
    # default k = 60: id 7 at ranks 2 and 14, id 8 at ranks 6 and 9: 1/62 + 1/74 < 1/66 + 1/69
    first = [100, 7, 101, 102, 103, 8, 104, 105, 106, 107, 108, 109, 110, 111]
    second = [200, 201, 202, 203, 204, 205, 206, 207, 8, 208, 209, 210, 211, 7]
    want, _ = rrf_oracle([first, second])
    assert want[:2] == [8, 7], ORACLE_BUG
    got = fused_of(ret, [first, second], "reciprocal_rank_fusion")
    assert_list_equal(got, want, "reciprocal_rank_fusion of two rankings of 14 ids with k=60: id 8 (ranks 6 and 9) "
                                 "before id 7 (ranks 2 and 14), 1/66 + 1/69 > 1/62 + 1/74 (ranks start at 1)", "id")


def _rrf_cases():
    """Random rankings; with 3 lists or more, no two ids with equal exact scores (a float sum may break such ties)."""
    cases, seed = [], 0
    for m, n_ids, length, k in [(2, 30, 10, 60), (2, 12, 12, 0), (2, 50, 20, 5), (3, 25, 10, 60), (4, 40, 15, 60),
                                (3, 20, 8, 1), (5, 15, 5, 60)]:
        while True:
            rng = np.random.default_rng(seed)
            seed += 1
            rankings = [[int(i) for i in rng.choice(n_ids, size=int(rng.integers(1, length + 1)), replace=False)]
                        for _ in range(m)]
            _, scores = rrf_oracle(rankings, k)
            if m == 2 or len(set(scores.values())) == len(scores):
                break
        cases.append(pytest.param(rankings, k, id=f"{m}-rankings-k{k}"))
    return cases


@pytest.mark.parametrize("rankings,k", _rrf_cases())
def test_reciprocal_rank_fusion_matches_the_formula(ret, rankings, k):
    want, _ = rrf_oracle(rankings, k)
    got = fused_of(ret, rankings, "reciprocal_rank_fusion", k=k)
    msg = f"reciprocal_rank_fusion of {len(rankings)} rankings with k={k}"
    if sorted(got) != sorted(want):
        missing = sorted(set(want) - set(got))
        _fail(msg, f"expected every id found in the rankings exactly once ({len(want)} ids), got {len(got)} ids"
                   + (f", missing {missing[:5]}" if missing else ""))
    assert_list_equal(got, want, msg + " (decreasing sum of 1 / (k + rank), ranks from 1; ties: smaller id first)",
                      "id")


@pytest.mark.parametrize("ranking,m", [
    pytest.param([5, 3, 9, 0, 7], 1, id="one-ranking"),
    pytest.param([5, 3, 9, 0, 7], 2, id="two-copies"),
    pytest.param([12, 4, 8, 1, 30, 2, 6], 3, id="three-copies"),
])
def test_reciprocal_rank_fusion_identical_rankings_are_preserved(ret, ranking, m):
    for k in (60, 0):
        got = fused_of(ret, [list(ranking) for _ in range(m)], "reciprocal_rank_fusion", k=k)
        assert_list_equal(got, list(ranking), f"reciprocal_rank_fusion of {m} copies of {ranking} with k={k} "
                                              "(the ranking is preserved)", "id")


@pytest.mark.parametrize("m,rank,k", [
    pytest.param(2, 1, 60, id="2-lists-rank-1"), pytest.param(3, 4, 60, id="3-lists-rank-4"),
    pytest.param(2, 2, 0, id="2-lists-rank-2-k0"), pytest.param(4, 5, 1, id="4-lists-rank-5-k1"),
])
def test_reciprocal_rank_fusion_document_in_every_list_outranks_one_in_a_single_list(ret, m, rank, k):
    # id 99 at rank `rank` in m lists; id 98 at the same rank in one more list, where 99 is absent
    lists = [[100 * j + i for i in range(1, 7)] for j in range(1, m + 2)]
    for j in range(m):
        lists[j][rank - 1] = 99
    lists[m][rank - 1] = 98
    got = fused_of(ret, lists, "reciprocal_rank_fusion", k=k)
    msg = f"reciprocal_rank_fusion of {m + 1} lists, k={k}"
    if 99 not in got or 98 not in got:
        _fail(msg, f"expected the ids 99 and 98 in the result, got {got}")
    if got.index(99) > got.index(98):
        _fail(msg, f"expected id 99 (rank {rank} in {m} lists) before id 98 (rank {rank} in a single list), "
                   f"got {got.index(99)} and {got.index(98)} as their positions")
    got = fused_of(ret, [[7, 2, 3], [8, 2, 4], [9, 2, 5]], "reciprocal_rank_fusion", k=k)
    if got[0] != 2:
        _fail(f"reciprocal_rank_fusion([[7, 2, 3], [8, 2, 4], [9, 2, 5]], k={k})",
              f"expected id 2 first (rank 2 in every list; 7, 8 and 9 are first in a single list), got {got[0]}")


def test_reciprocal_rank_fusion_ties_go_to_the_smaller_id(ret):
    for rankings, want in (([[5, 2], [2, 5]], [2, 5]), ([[7], [3]], [3, 7]), ([[9, 4, 1], [1, 4, 9]], [1, 9, 4]),
                           ([[8, 6], [6, 8], [3]], [6, 8, 3])):
        for k in (60, 0):
            exact, _ = rrf_oracle(rankings, k)
            assert exact == want or k == 0, ORACLE_BUG
            got = fused_of(ret, rankings, "reciprocal_rank_fusion", k=k)
            assert_list_equal(got, exact, f"reciprocal_rank_fusion({rankings}, k={k}) (equal fused scores: smaller "
                                          "id first)", "id")


def test_reciprocal_rank_fusion_rankings_of_different_lengths_and_empty_ones(ret):
    rankings = [[4, 8, 15, 16, 23, 42], [], [42, 4]]
    want, _ = rrf_oracle(rankings)
    got = fused_of(ret, rankings, "reciprocal_rank_fusion")
    assert_list_equal(got, want, f"reciprocal_rank_fusion({rankings}) (an empty ranking adds nothing)", "id")
    got = fused_of(ret, [[]], "reciprocal_rank_fusion")
    if got:
        _fail("reciprocal_rank_fusion([[]])", f"expected an empty array (no id found), got {got}")


def test_reciprocal_rank_fusion_accepts_numpy_arrays(ret):
    rankings = [np.array([3, 1, 2]), np.array([1, 2, 4], dtype=np.int32), (4, 3)]
    want, _ = rrf_oracle(rankings)
    got = fused_of(ret, rankings, "reciprocal_rank_fusion")
    assert_list_equal(got, want, "reciprocal_rank_fusion of NumPy arrays (int64, int32) and a tuple (the rows "
                                 "returned by cosine_top_k are arrays)", "id")


def test_reciprocal_rank_fusion_does_not_modify_its_inputs(ret):
    a, b = [3, 1, 2], np.array([1, 2, 4])
    rankings = [a, b]
    call(ret.reciprocal_rank_fusion, rankings, expected="an array of ids")
    if a != [3, 1, 2] or not np.array_equal(b, [1, 2, 4]) or len(rankings) != 2:
        _fail("", "expected the rankings unchanged by reciprocal_rank_fusion")


@pytest.mark.parametrize("rankings,k,why", [
    pytest.param([], 60, "rankings is empty", id="no-ranking"),
    pytest.param([[1, 2]], -1, "k = -1 < 0", id="k-negative"),
    pytest.param([[1, 2, 1]], 60, "the id 1 appears twice in the same ranking", id="duplicate-id"),
    pytest.param([[1, 2], [3, 4, 3]], 60, "the id 3 appears twice in the second ranking", id="duplicate-id-second"),
])
def test_reciprocal_rank_fusion_refuses_invalid_arguments(ret, rankings, k, why):
    assert_raises(ValueError, ret.reciprocal_rank_fusion, rankings, k=k, why=why)


# ================================================================== recall_at_k and MRR (B4.18)
def recall_oracle(retrieved, relevant, k):
    return sum(len(set(list(r)[:k]) & set(g)) / len(set(g)) for r, g in zip(retrieved, relevant)) / len(retrieved)


def mrr_oracle(retrieved, relevant):
    total = 0.0
    for r, g in zip(retrieved, relevant):
        ranks = [i for i, doc in enumerate(r, start=1) if doc in set(g)]
        total += 1.0 / ranks[0] if ranks else 0.0
    return total / len(retrieved)


def metric_of(function, retrieved, relevant, *args, **kwargs):
    """The learner's metric on copies of the inputs."""
    r = np.array(retrieved, copy=True) if isinstance(retrieved, np.ndarray) else copy.deepcopy(retrieved)
    return call(function, r, copy.deepcopy(relevant), *args, expected="a number", **kwargs)


def _permutations(rng, q, n):
    return [[int(i) for i in rng.permutation(n)] for _ in range(q)]


def _score_matrix(retrieved, n):
    """Scores whose ranking of each row is the retrieved permutation (best first: the highest score)."""
    S = np.zeros((len(retrieved), n))
    for i, r in enumerate(retrieved):
        S[i, r] = np.arange(n, 0, -1, dtype=float)
    return S


def test_recall_at_k_docstring_examples(ret):
    retrieved, relevant = [[3, 1, 2], [0, 4, 5, 6]], [{1, 7}, {6}]
    assert_number(metric_of(ret.recall_at_k, retrieved, relevant, k=2), 0.25,
                  "recall_at_k([[3, 1, 2], [0, 4, 5, 6]], [{1, 7}, {6}], k=2) of the docstring")
    assert_number(metric_of(ret.recall_at_k, retrieved, relevant, k=4), 0.75,
                  "recall_at_k(..., k=4) of the docstring")


def test_recall_at_k_hand_computed_values(ret):
    retrieved = [[5, 2, 9, 1], [4, 8], [7, 3, 6]]
    relevant = [{2, 1, 0}, {8}, {6, 3}]
    # k = 2: 1/3 (2 of {2, 1, 0}), 1 (8), 1/2 (3 of {6, 3}) -> 11/18; k = 3: 1/3, 1, 1 -> 7/9; k = 4: 2/3, 1, 1 -> 8/9
    for k, want, detail in ((2, 11 / 18, "(1/3 + 1 + 1/2) / 3"), (3, 7 / 9, "(1/3 + 1 + 1) / 3"),
                            (4, 8 / 9, "(2/3 + 1 + 1) / 3"), (10, 8 / 9, "k larger than every retrieved list")):
        assert_number(metric_of(ret.recall_at_k, retrieved, relevant, k=k), want,
                      f"recall_at_k({retrieved}, {relevant}, k={k}) = {detail} (divide by the number of relevant "
                      "ids of each query)")
    assert_number(metric_of(ret.recall_at_k, [[1, 2, 3]], [{1, 2, 3, 4, 5}], k=3), 0.6,
                  "recall_at_k([[1, 2, 3]], [{1, 2, 3, 4, 5}], k=3) = 3/5 (relevant ids never retrieved count)")
    assert_number(metric_of(ret.recall_at_k, [[9, 8], [1]], [{0}, {1}], k=1), 0.5,
                  "recall_at_k([[9, 8], [1]], [{0}, {1}], k=1) = (0 + 1) / 2")


@pytest.mark.parametrize("q,n,k", [
    pytest.param(20, 10, 1, id="k1"), pytest.param(30, 12, 3, id="k3"),
    pytest.param(15, 6, 5, id="k5-of-6"), pytest.param(50, 25, 10, id="k10"),
])
def test_recall_at_k_equals_top_k_accuracy_with_one_relevant_document(ret, q, n, k):
    rng = np.random.default_rng(q + n + k)
    retrieved = _permutations(rng, q, n)
    targets = [int(t) for t in rng.integers(0, n, size=q)]
    want = top_k_accuracy_score(targets, _score_matrix(retrieved, n), k=k, labels=np.arange(n))
    assert math.isclose(want, recall_oracle(retrieved, [{t} for t in targets], k)), ORACLE_BUG
    assert_number(metric_of(ret.recall_at_k, retrieved, [{t} for t in targets], k=k), want,
                  f"recall_at_k of {q} queries with one relevant document each, k={k} "
                  "(sklearn.metrics.top_k_accuracy_score)")


def test_recall_at_k_matches_the_formula(ret):
    rng = np.random.default_rng(11)
    retrieved = [[int(i) for i in rng.choice(30, size=int(rng.integers(1, 15)), replace=False)] for _ in range(25)]
    relevant = [{int(i) for i in rng.choice(30, size=int(rng.integers(1, 6)), replace=False)} for _ in range(25)]
    previous = -1.0
    for k in (1, 2, 5, 10, 20):
        want = recall_oracle(retrieved, relevant, k)
        got = metric_of(ret.recall_at_k, retrieved, relevant, k=k)
        assert_number(got, want, f"recall_at_k of 25 random queries, k={k}")
        if float(got) < previous:
            _fail("recall_at_k", f"expected a recall that never decreases with k, got {float(got):.6g} after "
                                 f"{previous:.6g}")
        previous = float(got)


def test_recall_at_k_accepts_arrays_lists_and_tuples(ret):
    idx = np.array([[3, 1, 2], [0, 4, 5]])
    assert_number(metric_of(ret.recall_at_k, idx, [[1, 7], (5,)], k=2), 0.25,
                  "recall_at_k of a 2-D array of retrieved ids (as returned by cosine_top_k), relevant ids as a list "
                  "and a tuple")
    assert_number(metric_of(ret.recall_at_k, [np.array([3, 1]), np.array([5])], [frozenset({1, 7}), {5}], k=3),
                  0.75, "recall_at_k of a list of arrays, retrieved lists shorter than k, a frozenset")


def test_recall_at_k_does_not_modify_its_inputs(ret):
    retrieved, relevant = [[3, 1, 2], [0, 4, 5, 6]], [{1, 7}, [6]]
    saved = copy.deepcopy((retrieved, relevant))
    call(ret.recall_at_k, retrieved, relevant, 2, expected="a number")
    if (retrieved, relevant) != saved:
        _fail("", "expected retrieved and relevant unchanged by recall_at_k")


@pytest.mark.parametrize("retrieved,relevant,k,why", [
    pytest.param([], [], 3, "no query", id="no-query"),
    pytest.param([[1, 2]], [{1}, {2}], 1, "1 retrieved list for 2 relevant sets", id="different-lengths"),
    pytest.param([[1, 2], [3]], [{1}], 1, "2 retrieved lists for 1 relevant set", id="different-lengths-2"),
    pytest.param([[1, 2]], [{1}], 0, "k = 0", id="k-0"),
    pytest.param([[1, 2], [3]], [{1}, set()], 2, "an empty relevant set (recall is not defined)", id="empty-relevant"),
])
def test_recall_at_k_refuses_invalid_arguments(ret, retrieved, relevant, k, why):
    assert_raises(ValueError, ret.recall_at_k, retrieved, relevant, k, why=why)


def test_mean_reciprocal_rank_docstring_examples(ret):
    assert_number(metric_of(ret.mean_reciprocal_rank, [[3, 1, 2], [0, 4, 5, 6]], [{1, 7}, {6}]), 0.375,
                  "mean_reciprocal_rank([[3, 1, 2], [0, 4, 5, 6]], [{1, 7}, {6}]) of the docstring")
    assert_number(metric_of(ret.mean_reciprocal_rank, [[3, 1, 2], [0, 4]], [{9}, {0}]), 0.5,
                  "mean_reciprocal_rank([[3, 1, 2], [0, 4]], [{9}, {0}]) of the docstring")


def test_mean_reciprocal_rank_hand_computed_values(ret):
    retrieved = [[4, 2, 7], [1, 9], [3, 5, 8, 6], [0]]
    relevant = [{7, 2}, {9}, {6}, {5}]
    # first relevant: 2 at rank 2, 9 at rank 2, 6 at rank 4, none -> (1/2 + 1/2 + 1/4 + 0) / 4 = 0.3125
    assert_number(metric_of(ret.mean_reciprocal_rank, retrieved, relevant), 0.3125,
                  f"mean_reciprocal_rank({retrieved}, {relevant}) = (1/2 + 1/2 + 1/4 + 0) / 4 (rank of the first "
                  "relevant id, from 1; 0 when none is retrieved)")
    assert_number(metric_of(ret.mean_reciprocal_rank, [[5, 6, 7]], [{5, 6}]), 1.0,
                  "mean_reciprocal_rank([[5, 6, 7]], [{5, 6}]) = 1 (first relevant id at rank 1)")
    assert_number(metric_of(ret.mean_reciprocal_rank, [[1, 2, 3], [4, 5, 6]], [{3}, {7}]), 1 / 6,
                  "mean_reciprocal_rank([[1, 2, 3], [4, 5, 6]], [{3}, {7}]) = (1/3 + 0) / 2")


def test_mean_reciprocal_rank_empty_lists_and_sets_score_zero(ret):
    assert_number(metric_of(ret.mean_reciprocal_rank, [[], [1, 2]], [{1}, {2}]), 0.25,
                  "mean_reciprocal_rank([[], [1, 2]], [{1}, {2}]) = (0 + 1/2) / 2 (nothing retrieved: 0)")
    assert_number(metric_of(ret.mean_reciprocal_rank, [[1, 2], [3]], [set(), {3}]), 0.5,
                  "mean_reciprocal_rank([[1, 2], [3]], [set(), {3}]) = (0 + 1) / 2 (an empty relevant set: no "
                  "relevant id is retrieved, the query scores 0)")


@pytest.mark.parametrize("q,n", [pytest.param(20, 8, id="20x8"), pytest.param(40, 30, id="40x30"),
                                 pytest.param(5, 3, id="5x3")])
def test_mean_reciprocal_rank_equals_lrap_with_one_relevant_document(ret, q, n):
    rng = np.random.default_rng(q * n)
    retrieved = _permutations(rng, q, n)
    targets = [int(t) for t in rng.integers(0, n, size=q)]
    y_true = np.zeros((q, n), dtype=int)
    y_true[np.arange(q), targets] = 1
    want = label_ranking_average_precision_score(y_true, _score_matrix(retrieved, n))
    assert math.isclose(want, mrr_oracle(retrieved, [{t} for t in targets])), ORACLE_BUG
    assert_number(metric_of(ret.mean_reciprocal_rank, retrieved, [{t} for t in targets]), want,
                  f"mean_reciprocal_rank of {q} queries with one relevant document each "
                  "(sklearn.metrics.label_ranking_average_precision_score)")


def test_mean_reciprocal_rank_matches_the_formula(ret):
    rng = np.random.default_rng(12)
    retrieved = [[int(i) for i in rng.choice(30, size=int(rng.integers(0, 12)), replace=False)] for _ in range(30)]
    relevant = [{int(i) for i in rng.choice(30, size=int(rng.integers(1, 6)), replace=False)} for _ in range(30)]
    assert_number(metric_of(ret.mean_reciprocal_rank, retrieved, relevant), mrr_oracle(retrieved, relevant),
                  "mean_reciprocal_rank of 30 random queries (several relevant ids: only the first one retrieved "
                  "counts)")


def test_mean_reciprocal_rank_accepts_arrays_lists_and_tuples(ret):
    idx = np.array([[3, 1, 2], [0, 4, 5]])
    assert_number(metric_of(ret.mean_reciprocal_rank, idx, [[1, 7], (5,)]), (1 / 2 + 1 / 3) / 2,
                  "mean_reciprocal_rank of a 2-D array of retrieved ids (as returned by cosine_top_k), relevant ids "
                  "as a list and a tuple")


def test_mean_reciprocal_rank_does_not_modify_its_inputs(ret):
    retrieved, relevant = [[3, 1, 2], [0, 4, 5, 6]], [{1, 7}, [6]]
    saved = copy.deepcopy((retrieved, relevant))
    call(ret.mean_reciprocal_rank, retrieved, relevant, expected="a number")
    if (retrieved, relevant) != saved:
        _fail("", "expected retrieved and relevant unchanged by mean_reciprocal_rank")


@pytest.mark.parametrize("retrieved,relevant,why", [
    pytest.param([], [], "no query", id="no-query"),
    pytest.param([[1, 2]], [{1}, {2}], "1 retrieved list for 2 relevant sets", id="different-lengths"),
    pytest.param([[1], [2], [3]], [{1}, {2}], "3 retrieved lists for 2 relevant sets", id="different-lengths-2"),
])
def test_mean_reciprocal_rank_refuses_invalid_arguments(ret, retrieved, relevant, why):
    assert_raises(ValueError, ret.mean_reciprocal_rank, retrieved, relevant, why=why)
