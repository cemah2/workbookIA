"""Tests of mylearn.sequences (chapter 22): oracle tests and properties.

    pytest tests/test_ch22_sequences.py              # your code (mon_travail/mylearn/sequences.py)
    pytest tests/test_ch22_sequences.py --impl=ref   # the reference

Oracles: ``numpy.lib.stride_tricks.sliding_window_view`` and explicit slicing loops (windows and pairs),
``collections.Counter.most_common`` with the tie rule of the docstring (vocabulary), an explicit dictionary
lookup (encoding), ``scipy.special.softmax`` and, when PyTorch is installed, ``torch.softmax`` (temperature),
and the frequencies of 20 000 draws against these probabilities (sampling; the way the draw is done is not part
of the contract, so no test compares the drawn ids themselves to another generator).
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_build_vocab_ or test_encode_ or test_decode_"`` (22.12),
``-k "test_make_windows_ or test_make_sequence_pairs_"`` (22.13),
``-k "test_temperature_softmax_ or test_sample_from_logits_"`` (22.20).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified
and may be lists; ``make_windows`` and ``make_sequence_pairs`` return new arrays (never views of the series) with
the dtype of the series (int64 ids stay int64), accept a series of shape (T,) or (T, D) only, and refuse a window,
horizon, seq_len or stride that is not an int >= 1 (a bool is not an int) and a series too short for one sample;
``build_vocab`` takes the characters of a string or the words of any iterable of str (a generator too), ranks by
decreasing count then in ``sorted`` order, puts ``unk_token`` first (id 0) even when it appears in the text,
counts it in ``max_size`` and refuses empty tokens and a ``max_size`` or ``min_freq`` that is not an int >= 1;
``encode`` returns int64 ids (shape (0,) for an empty text) and raises ``ValueError`` (not ``KeyError``) on an
unknown token without ``unk_token``, or when ``unk_token`` is not in ``stoi``; ``decode`` refuses ids < 0 or
>= len(itos) (Python would accept -1 silently) and ids that are not 1-D; ``temperature_softmax`` keeps the shape
and works along the last axis, without overflow for logits of +-1000, and refuses T <= 0; ``sample_from_logits``
returns a Python int for (V,) logits and an int64 array of shape (N,) for (N, V) logits, takes the first argmax
when T = 0 (no draw), keeps the ties with the k-th largest logit for ``top_k``, refuses T < 0 and a ``top_k``
that is not an int in [1, V], and gives the same ids for two generators created with the same seed.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

from collections import Counter

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import softmax

try:
    import torch
except ImportError:
    torch = None

ORACLE_BUG = ("bug in tests/test_ch22_sequences.py (not in your code): the oracle of this test "
              "disagrees with another one")


@pytest.fixture
def seq(mylearn_module):
    return mylearn_module("sequences")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3,
                           max_line_width=10**6)
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = ""):
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


def assert_close(result, expected, rtol=1e-9, atol=1e-12, msg=""):
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
                f"({bad.size} wrong value(s) out of {want.size})")
        data = f"expected {_short(want, 10)}\n     got {_short(got, 10)}" if want.size <= 16 else ""
        _fail(msg, head, data)


def assert_array(result, expected, msg, dtype=None):
    """Same shape, same values (exactly) and, if given, the same dtype."""
    if result is None:
        _fail(msg, "expected an array, got None (did you forget the return?)")
    if not isinstance(result, np.ndarray):
        _fail(msg, f"expected a NumPy array, got {type(result).__name__}")
    want = np.asarray(expected)
    if result.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {result.shape}")
    if dtype is not None and result.dtype != dtype:
        _fail(msg, f"expected dtype {np.dtype(dtype)}, got {result.dtype}")
    if not np.array_equal(result, want):
        bad = np.flatnonzero(result.ravel() != want.ravel())
        i = int(bad[0])
        _fail(msg, f"expected {want.ravel()[i]!r} at flat index {i}, got {result.ravel()[i]!r} "
                   f"({bad.size} wrong value(s) out of {want.size})",
              f"expected {_short(want)}\n     got {_short(result)}" if want.size <= 24 else "")


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


def pair(result, what, names):
    label = f"the tuple ({', '.join(names)})"
    if result is None:
        _fail("", f"expected {label} from {what}, got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 2:
        _fail("", f"expected {label} from {what}, got {type(result).__name__}"
                  + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else ""))
    return result


def assert_new_array(arr, inputs, what):
    for inp in inputs:
        if isinstance(inp, np.ndarray) and np.shares_memory(arr, inp):
            _fail("", f"expected {what} to be a new array, got a view of the series (changing it would change the "
                      "series: return a copy)")


# ------------------------------------------------------------------ oracles
def windows_oracle(series, w, h=1, s=1):
    """Explicit loop over the starts (checked against sliding_window_view)."""
    arr = np.asarray(series)
    t = arr.shape[0]
    n = (t - w - h) // s + 1
    two_d = arr.reshape(t, -1)
    X = np.stack([two_d[i * s:i * s + w] for i in range(n)])
    y = np.stack([arr[i * s + w - 1 + h] for i in range(n)])
    view = sliding_window_view(two_d, w, axis=0)[::s][:n]     # (N, D, w)
    assert np.array_equal(np.moveaxis(view, -1, 1), X), ORACLE_BUG
    return X, y


def pairs_oracle(seq_arr, length, stride=None):
    s = length if stride is None else stride
    arr = np.asarray(seq_arr)
    n = (arr.shape[0] - 1 - length) // s + 1
    X = np.stack([arr[i * s:i * s + length] for i in range(n)])
    Y = np.stack([arr[i * s + 1:i * s + length + 1] for i in range(n)])
    return X, Y


def vocab_oracle(tokens, max_size=None, min_freq=1, unk=None):
    """Counter.most_common, then the tie rule of the docstring (sorted order of the tokens)."""
    counts = Counter(tokens)
    ranked = [tok for tok, cnt in counts.most_common() if cnt >= min_freq and tok != unk]
    ranked.sort(key=lambda tok: (-counts[tok], tok))           # most_common keeps first-seen order on ties
    itos = ([unk] if unk is not None else []) + ranked
    if max_size is not None:
        itos = itos[:max_size]
    return {tok: i for i, tok in enumerate(itos)}, itos


def softmax_oracle(logits, temperature=1.0):
    z = np.asarray(logits, dtype=float) / temperature
    p = softmax(z, axis=-1)
    if torch is not None:
        tp = torch.softmax(torch.tensor(z), dim=-1).numpy()
        assert np.allclose(p, tp, rtol=1e-12, atol=1e-15), ORACLE_BUG
    return p


def vocab_result(result, what="build_vocab"):
    stoi, itos = pair(result, what, names=("stoi", "itos"))
    if not isinstance(stoi, dict):
        _fail("", f"expected stoi to be a dict (token -> id), got {type(stoi).__name__}")
    if not isinstance(itos, list):
        _fail("", f"expected itos to be a list (id -> token), got {type(itos).__name__}")
    return stoi, itos


HOLMES_LINE = "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name."


# ================================================================== make_windows (22.13)
def test_make_windows_docstring_example(seq):
    X, y = pair(call(seq.make_windows, np.arange(6.0), window=3, expected="(X, y)"), "make_windows", ("X", "y"))
    assert_array(X, [[[0.0], [1.0], [2.0]], [[1.0], [2.0], [3.0]], [[2.0], [3.0], [4.0]]],
                 "X of make_windows(arange(6.), window=3), shape (N, w, 1)")
    assert_array(y, [3.0, 4.0, 5.0], "y of make_windows(arange(6.), window=3)")
    _, y2 = pair(call(seq.make_windows, np.arange(6.0), window=3, horizon=2, expected="(X, y)"), "make_windows",
                 ("X", "y"))
    assert_array(y2, [4.0, 5.0], "y with horizon=2: the value 2 steps after the last of the window")


@pytest.mark.parametrize("t,w,h,s", [
    pytest.param(20, 4, 1, 1, id="T20-w4"), pytest.param(20, 4, 1, 3, id="T20-w4-s3"),
    pytest.param(23, 5, 6, 2, id="T23-w5-h6-s2"), pytest.param(10, 9, 1, 1, id="T10-w9-one-window"),
    pytest.param(31, 1, 1, 5, id="T31-w1-s5"), pytest.param(17, 6, 3, 4, id="T17-w6-h3-s4"),
])
def test_make_windows_matches_sliding_window_view(seq, t, w, h, s):
    series = np.random.default_rng(t + w + h + s).normal(size=t)
    want_X, want_y = windows_oracle(series, w, h, s)
    X, y = pair(call(seq.make_windows, series.copy(), w, horizon=h, stride=s, expected="(X, y)"), "make_windows",
                ("X", "y"))
    assert want_X.shape[0] == (t - w - h) // s + 1, ORACLE_BUG
    assert_array(X, want_X, f"X of make_windows(T={t}, window={w}, horizon={h}, stride={s}), N = (T - w - h) // s + 1")
    assert_array(y, want_y, f"y of make_windows(T={t}, window={w}, horizon={h}, stride={s})")


def test_make_windows_multivariate_series(seq):
    series = np.arange(30.0).reshape(10, 3)
    want_X, want_y = windows_oracle(series, 4, 2, 1)
    X, y = pair(call(seq.make_windows, series.copy(), 4, horizon=2, expected="(X, y)"), "make_windows", ("X", "y"))
    assert_array(X, want_X, "X of a (T, D) series: shape (N, w, D)")
    assert_array(y, want_y, "y of a (T, D) series: shape (N, D)")


def test_make_windows_keeps_the_dtype(seq):
    series = np.arange(12, dtype=np.int64)
    X, y = pair(call(seq.make_windows, series, 3, expected="(X, y)"), "make_windows", ("X", "y"))
    assert_array(X, windows_oracle(series, 3)[0], "X of an int64 series: same dtype", dtype=np.int64)
    assert_array(y, windows_oracle(series, 3)[1], "y of an int64 series: same dtype", dtype=np.int64)
    f32 = np.linspace(0, 1, 9).astype(np.float32)
    X32, _ = pair(call(seq.make_windows, f32, 2, expected="(X, y)"), "make_windows", ("X", "y"))
    assert_array(X32, windows_oracle(f32, 2)[0], "X of a float32 series: same dtype", dtype=np.float32)


def test_make_windows_accepts_a_list(seq):
    X, y = pair(call(seq.make_windows, [5, 6, 7, 8, 9], 2, stride=2, expected="(X, y)"), "make_windows", ("X", "y"))
    assert_array(X, [[[5], [6]], [[7], [8]]], "X of make_windows([5, 6, 7, 8, 9], 2, stride=2)")
    assert_array(y, [7, 9], "y of make_windows([5, 6, 7, 8, 9], 2, stride=2)")


def test_make_windows_returns_new_arrays(seq):
    series = np.arange(8.0)
    X, y = pair(call(seq.make_windows, series, 3, expected="(X, y)"), "make_windows", ("X", "y"))
    assert_new_array(X, [series], "X")
    assert_new_array(y, [series], "y")


def test_make_windows_does_not_modify_its_input(seq):
    series = np.random.default_rng(1).normal(size=15)
    saved = series.copy()
    call(seq.make_windows, series, 4, horizon=2, stride=3)
    if not np.array_equal(series, saved):
        _fail("", "expected the series unchanged by make_windows (build the windows in new arrays)")


@pytest.mark.parametrize("kwargs,why", [
    pytest.param(dict(window=0), "window = 0", id="window-0"),
    pytest.param(dict(horizon=0), "horizon = 0 (the target would be inside the window)", id="horizon-0"),
    pytest.param(dict(stride=0), "stride = 0", id="stride-0"),
    pytest.param(dict(window=2.5), "window = 2.5 is not an int", id="window-float"),
    pytest.param(dict(window=True), "window = True: a bool is not an int", id="window-bool"),
    pytest.param(dict(window=8), "a series of 8 steps is too short for a window of 8 and a horizon of 1",
                 id="too-short"),
    pytest.param(dict(window=3, horizon=6), "8 steps: no room for a window of 3 and a horizon of 6", id="horizon-too-far"),
])
def test_make_windows_refuses_invalid_arguments(seq, kwargs, why):
    args = dict(window=3, horizon=1, stride=1)
    args.update(kwargs)
    assert_raises(ValueError, seq.make_windows, np.arange(8.0), why=why, **args)


def test_make_windows_refuses_a_3d_series(seq):
    assert_raises(ValueError, seq.make_windows, np.zeros((10, 2, 2)), 3,
                  why="a series of shape (10, 2, 2): only (T,) or (T, D)")


# ================================================================== make_sequence_pairs (22.13)
def test_make_sequence_pairs_docstring_example(seq):
    X, Y = pair(call(seq.make_sequence_pairs, np.arange(7), seq_len=3, expected="(X, Y)"), "make_sequence_pairs",
                ("X", "Y"))
    assert_array(X, [[0, 1, 2], [3, 4, 5]], "X of make_sequence_pairs(arange(7), seq_len=3)")
    assert_array(Y, [[1, 2, 3], [4, 5, 6]], "Y of make_sequence_pairs(arange(7), seq_len=3): X shifted by one step")


@pytest.mark.parametrize("t,length,stride", [
    pytest.param(25, 4, None, id="T25-L4-chunks"), pytest.param(25, 4, 1, id="T25-L4-s1"),
    pytest.param(30, 7, 3, id="T30-L7-s3"), pytest.param(9, 8, None, id="T9-L8-one-chunk"),
    pytest.param(41, 10, 15, id="T41-L10-s15"),
])
def test_make_sequence_pairs_matches_a_slicing_loop(seq, t, length, stride):
    ids = np.random.default_rng(t + length).integers(0, 50, size=t)
    want_X, want_Y = pairs_oracle(ids, length, stride)
    X, Y = pair(call(seq.make_sequence_pairs, ids.copy(), length, stride=stride, expected="(X, Y)"),
                "make_sequence_pairs", ("X", "Y"))
    s = length if stride is None else stride
    assert want_X.shape[0] == (t - 1 - length) // s + 1, ORACLE_BUG
    assert_array(X, want_X, f"X of make_sequence_pairs(T={t}, seq_len={length}, stride={stride}), "
                            "N = (T - 1 - L) // stride + 1", dtype=ids.dtype)
    assert_array(Y, want_Y, f"Y of make_sequence_pairs(T={t}, seq_len={length}, stride={stride})", dtype=ids.dtype)
    assert_array(Y[:, :-1], X[:, 1:], "Y[:, :-1] must equal X[:, 1:] (the same chunk shifted by one step)")


def test_make_sequence_pairs_multivariate(seq):
    series = np.arange(24.0).reshape(12, 2)
    want_X, want_Y = pairs_oracle(series, 3, 2)
    X, Y = pair(call(seq.make_sequence_pairs, series.copy(), 3, stride=2, expected="(X, Y)"), "make_sequence_pairs",
                ("X", "Y"))
    assert_array(X, want_X, "X of a (T, D) sequence: shape (N, L, D)")
    assert_array(Y, want_Y, "Y of a (T, D) sequence: shape (N, L, D)")


def test_make_sequence_pairs_accepts_a_list(seq):
    X, Y = pair(call(seq.make_sequence_pairs, [3, 1, 4, 1, 5, 9], 2, expected="(X, Y)"), "make_sequence_pairs",
                ("X", "Y"))
    assert_array(X, [[3, 1], [4, 1]], "X of make_sequence_pairs([3, 1, 4, 1, 5, 9], 2)")
    assert_array(Y, [[1, 4], [1, 5]], "Y of make_sequence_pairs([3, 1, 4, 1, 5, 9], 2)")


def test_make_sequence_pairs_returns_new_arrays(seq):
    ids = np.arange(10)
    X, Y = pair(call(seq.make_sequence_pairs, ids, 3, expected="(X, Y)"), "make_sequence_pairs", ("X", "Y"))
    assert_new_array(X, [ids], "X")
    assert_new_array(Y, [ids], "Y")


def test_make_sequence_pairs_does_not_modify_its_input(seq):
    ids = np.random.default_rng(2).integers(0, 9, size=20)
    saved = ids.copy()
    call(seq.make_sequence_pairs, ids, 4, stride=3)
    if not np.array_equal(ids, saved):
        _fail("", "expected the sequence unchanged by make_sequence_pairs (build the pairs in new arrays)")


@pytest.mark.parametrize("kwargs,why", [
    pytest.param(dict(seq_len=0), "seq_len = 0", id="seq_len-0"),
    pytest.param(dict(stride=0), "stride = 0", id="stride-0"),
    pytest.param(dict(seq_len=2.0), "seq_len = 2.0 is not an int", id="seq_len-float"),
    pytest.param(dict(stride=False), "stride = False: a bool is not an int", id="stride-bool"),
    pytest.param(dict(seq_len=8), "8 steps: a chunk of 8 has no target for its last step", id="too-short"),
])
def test_make_sequence_pairs_refuses_invalid_arguments(seq, kwargs, why):
    args = dict(seq_len=3, stride=None)
    args.update(kwargs)
    assert_raises(ValueError, seq.make_sequence_pairs, np.arange(8), why=why, **args)


# ================================================================== build_vocab (22.12)
def test_build_vocab_docstring_example(seq):
    stoi, itos = vocab_result(call(seq.build_vocab, "hello", expected="(stoi, itos)"))
    if itos != ["l", "e", "h", "o"]:
        _fail("", f"expected itos ['l', 'e', 'h', 'o'] for 'hello' (most frequent first, then sorted order), "
                  f"got {itos!r}")
    if stoi != {"l": 0, "e": 1, "h": 2, "o": 3}:
        _fail("", f"expected stoi {{'l': 0, 'e': 1, 'h': 2, 'o': 3}}, got {stoi!r}")
    _, itos2 = vocab_result(call(seq.build_vocab, ["the", "cat", "the", "dog"], max_size=2, unk_token="<unk>",
                                 expected="(stoi, itos)"))
    if itos2 != ["<unk>", "the"]:
        _fail("", f"expected ['<unk>', 'the'] with max_size=2 and unk_token='<unk>' (unk first, counted in "
                  f"max_size), got {itos2!r}")


@pytest.mark.parametrize("text", [
    pytest.param(HOLMES_LINE, id="holmes-line"),
    pytest.param("abracadabra, ABRACADABRA!", id="case-and-punctuation"),
    pytest.param("Zz aA 1 9 ", id="ties-everywhere"),
    pytest.param("é è à ç Où ? « » ...", id="french-characters"),
])
def test_build_vocab_characters_match_counter(seq, text):
    want_stoi, want_itos = vocab_oracle(text)
    stoi, itos = vocab_result(call(seq.build_vocab, text, expected="(stoi, itos)"))
    if itos != want_itos:
        first = next((i for i, (a, b) in enumerate(zip(itos, want_itos)) if a != b), min(len(itos), len(want_itos)))
        _fail("", f"expected itos[{first}] = {want_itos[first] if first < len(want_itos) else None!r}, got "
                  f"{itos[first] if first < len(itos) else None!r} (rank by decreasing count, ties in sorted order: "
                  f"{len(want_itos)} tokens expected, {len(itos)} given)")
    if stoi != want_stoi:
        _fail("", "expected stoi[itos[i]] == i for every i (stoi and itos must agree)")


def test_build_vocab_words_with_max_size_min_freq_and_unk(seq):
    words = HOLMES_LINE.lower().replace(".", " .").split() * 2 + ["holmes", "watson", "watson", "the", "the"]
    for kwargs in [dict(max_size=5), dict(min_freq=3), dict(unk_token="<unk>"), dict(max_size=4, unk_token="<unk>"),
                   dict(min_freq=2, max_size=6, unk_token="<unk>")]:
        want_stoi, want_itos = vocab_oracle(words, kwargs.get("max_size"), kwargs.get("min_freq", 1),
                                            kwargs.get("unk_token"))
        stoi, itos = vocab_result(call(seq.build_vocab, list(words), expected="(stoi, itos)", **kwargs))
        if itos != want_itos:
            _fail("", f"expected itos {want_itos!r} with {kwargs} (unk first and counted in max_size, then decreasing "
                      f"count, ties in sorted order), got {itos!r}")
        if stoi != want_stoi:
            _fail("", f"expected stoi to agree with itos with {kwargs}")


def test_build_vocab_unk_token_present_in_the_text(seq):
    words = ["<unk>", "a", "a", "<unk>", "<unk>", "b"]
    stoi, itos = vocab_result(call(seq.build_vocab, words, unk_token="<unk>", expected="(stoi, itos)"))
    if itos != ["<unk>", "a", "b"]:
        _fail("", f"expected ['<unk>', 'a', 'b']: the unk token comes first (id 0) once, even when it appears in the "
                  f"text, got {itos!r}")


def test_build_vocab_accepts_a_generator(seq):
    words = ["sherlock", "holmes", "watson", "holmes"]
    stoi, itos = vocab_result(call(seq.build_vocab, (w for w in words), expected="(stoi, itos)"))
    if itos != vocab_oracle(words)[1]:
        _fail("", f"expected {vocab_oracle(words)[1]!r} from a generator of words (iterate over it once), got {itos!r}")


def test_build_vocab_ids_are_contiguous(seq):
    stoi, itos = vocab_result(call(seq.build_vocab, HOLMES_LINE, expected="(stoi, itos)"))
    if sorted(stoi.values()) != list(range(len(itos))):
        _fail("", f"expected the ids 0 to {len(itos) - 1}, each once, got {sorted(stoi.values())[:10]}…")
    if any(itos[i] != tok for tok, i in stoi.items()):
        _fail("", "expected itos[stoi[t]] == t for every token t")


@pytest.mark.parametrize("args,kwargs,why", [
    pytest.param(("",), {}, "an empty string: nothing to count", id="empty-string"),
    pytest.param(([],), {}, "an empty list of words", id="empty-list"),
    pytest.param(("abc",), dict(max_size=0), "max_size = 0", id="max_size-0"),
    pytest.param(("abc",), dict(min_freq=0), "min_freq = 0", id="min_freq-0"),
    pytest.param(("abc",), dict(max_size=2.5), "max_size = 2.5 is not an int", id="max_size-float"),
])
def test_build_vocab_refuses_invalid_arguments(seq, args, kwargs, why):
    assert_raises(ValueError, seq.build_vocab, *args, why=why, **kwargs)


# ================================================================== encode (22.12)
def test_encode_docstring_example(seq):
    stoi = {"l": 0, "e": 1, "h": 2, "o": 3}
    ids = call(seq.encode, "hole", dict(stoi))
    assert_array(ids, [2, 3, 0, 1], "encode('hole', stoi of 'hello')", dtype=np.int64)


def test_encode_matches_a_dictionary_lookup(seq):
    stoi, _ = vocab_oracle(HOLMES_LINE)
    want = np.array([stoi[c] for c in HOLMES_LINE], dtype=np.int64)
    assert_array(call(seq.encode, HOLMES_LINE, dict(stoi)), want, "encode of a line of Holmes", dtype=np.int64)


def test_encode_words(seq):
    stoi = {"<unk>": 0, "the": 1, "cat": 2}
    ids = call(seq.encode, ["the", "dog", "cat"], dict(stoi), unk_token="<unk>")
    assert_array(ids, [1, 0, 2], "encode of words with unk_token: 'dog' becomes id 0", dtype=np.int64)


def test_encode_empty_text(seq):
    ids = call(seq.encode, "", {"a": 0})
    assert_array(ids, np.zeros(0, dtype=np.int64), "encode('') is an empty int64 array of shape (0,)", dtype=np.int64)


def test_encode_unknown_token_without_unk_is_a_value_error(seq):
    assert_raises(ValueError, seq.encode, "abz", {"a": 0, "b": 1},
                  why="'z' is not in stoi and unk_token is None (a ValueError, not a KeyError)")


def test_encode_unk_token_missing_from_stoi(seq):
    assert_raises(ValueError, seq.encode, "ab", {"a": 0, "b": 1}, unk_token="<unk>",
                  why="unk_token '<unk>' is not a key of stoi")


def test_encode_does_not_modify_stoi(seq):
    stoi = {"a": 0, "b": 1, "<unk>": 2}
    saved = dict(stoi)
    call(seq.encode, "abcab", stoi, unk_token="<unk>")
    if stoi != saved:
        _fail("", f"expected stoi unchanged by encode, got {stoi!r} (do not add the unknown tokens to it)")


# ================================================================== decode (22.12)
def test_decode_docstring_example(seq):
    text = call(seq.decode, [2, 3, 0, 1], ["l", "e", "h", "o"])
    if text != "hole":
        _fail("", f"expected 'hole', got {text!r}")
    words = call(seq.decode, [1, 0], ["cat", "the"], sep=" ")
    if words != "the cat":
        _fail("", f"expected 'the cat' with sep=' ', got {words!r}")


def test_decode_round_trip_with_encode(seq):
    stoi, itos = vocab_oracle(HOLMES_LINE)
    ids = np.array([stoi[c] for c in HOLMES_LINE], dtype=np.int64)
    text = call(seq.decode, ids, list(itos))
    if text != HOLMES_LINE:
        _fail("", f"expected the round trip decode(encode(text)) == text, got {text[:40]!r}…")


def test_decode_empty(seq):
    text = call(seq.decode, np.zeros(0, dtype=np.int64), ["a"])
    if text != "":
        _fail("", f"expected '' for no ids, got {text!r}")


@pytest.mark.parametrize("ids,why", [
    pytest.param([0, -1], "id -1: Python would silently take the last token", id="negative"),
    pytest.param([3], "id 3 for a vocabulary of 3 tokens", id="too-large"),
    pytest.param([[0, 1]], "ids of shape (1, 2): they must be 1-D", id="2d"),
])
def test_decode_refuses_invalid_ids(seq, ids, why):
    assert_raises(ValueError, seq.decode, ids, ["a", "b", "c"], why=why)


# ================================================================== temperature_softmax (22.20)
def test_temperature_softmax_docstring_example(seq):
    assert_close(call(seq.temperature_softmax, [1.0, 2.0, 3.0]), softmax_oracle([1.0, 2.0, 3.0]),
                 msg="temperature_softmax([1, 2, 3])")
    assert_close(call(seq.temperature_softmax, [1.0, 2.0, 3.0], temperature=0.5),
                 softmax_oracle([1.0, 2.0, 3.0], 0.5), msg="temperature_softmax([1, 2, 3], temperature=0.5)")


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 10.0])
@pytest.mark.parametrize("shape", [(7,), (4, 9), (2, 3, 5)], ids=["V", "N-V", "batch-N-V"])
def test_temperature_softmax_matches_torch(seq, temperature, shape):
    logits = np.random.default_rng(len(shape) * 10 + int(temperature * 10)).normal(scale=3, size=shape)
    want = softmax_oracle(logits, temperature)
    got = call(seq.temperature_softmax, logits.copy(), temperature=temperature)
    assert_close(got, want, msg=f"softmax(logits / {temperature}) along the last axis, as torch.softmax(dim=-1)")


def test_temperature_softmax_rows_sum_to_one(seq):
    p = call(seq.temperature_softmax, np.random.default_rng(3).normal(size=(5, 11)), temperature=0.7)
    assert_close(np.asarray(p).sum(axis=-1), np.ones(5), msg="every row of probabilities sums to 1")


def test_temperature_softmax_no_overflow(seq):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p = call(seq.temperature_softmax, [1000.0, 999.0, -1000.0], temperature=0.5)
    assert_close(p, softmax_oracle([1000.0, 999.0, -1000.0], 0.5),
                 msg="logits of +-1000 (subtract the maximum before exp: no RuntimeWarning, no nan)")


def test_temperature_softmax_low_and_high_temperature(seq):
    logits = [1.0, 2.0, 3.0]
    cold = np.asarray(call(seq.temperature_softmax, logits, temperature=0.05))
    hot = np.asarray(call(seq.temperature_softmax, logits, temperature=100.0))
    assert_close(cold, [0.0, 0.0, 1.0], atol=1e-8, msg="T = 0.05: almost all the mass on the argmax")
    assert_close(hot, np.full(3, 1 / 3), atol=5e-3, msg="T = 100: almost uniform")


def test_temperature_softmax_does_not_modify_its_input(seq):
    logits = np.array([[0.5, -1.0, 2.0]])
    saved = logits.copy()
    call(seq.temperature_softmax, logits, temperature=0.3)
    if not np.array_equal(logits, saved):
        _fail("", "expected the logits unchanged by temperature_softmax (divide into a new array, not with /=)")


@pytest.mark.parametrize("temperature", [0.0, -1.0], ids=["zero", "negative"])
def test_temperature_softmax_refuses_a_non_positive_temperature(seq, temperature):
    assert_raises(ValueError, seq.temperature_softmax, [1.0, 2.0], temperature=temperature,
                  why=f"temperature = {temperature} (T must be > 0)")


# ================================================================== sample_from_logits (22.20)
def test_sample_from_logits_docstring_example(seq):
    first = call(seq.sample_from_logits, [1.0, 5.0, 2.0], temperature=0)
    if first != 1 or type(first) is not int:
        _fail("", f"expected the Python int 1 (the argmax) with temperature=0, got {first!r} ({type(first).__name__})")
    top1 = call(seq.sample_from_logits, [1.0, 5.0, 2.0], top_k=1, rng=np.random.default_rng(0))
    if top1 != 1:
        _fail("", f"expected 1 with top_k=1 (only the best logit is kept), got {top1!r}")
    many = call(seq.sample_from_logits, np.zeros((4, 3)), rng=np.random.default_rng(0))
    if not isinstance(many, np.ndarray) or many.shape != (4,):
        _fail("", f"expected an array of shape (4,) for (4, 3) logits, got {type(many).__name__} "
                  f"{getattr(many, 'shape', '')}")


def test_sample_from_logits_greedy_takes_the_first_maximum(seq):
    got = call(seq.sample_from_logits, [0.0, 3.0, 3.0, 1.0], temperature=0)
    if got != 1:
        _fail("", f"expected 1, the FIRST maximum, with temperature=0 (two logits equal 3), got {got!r}")
    rows = call(seq.sample_from_logits, [[0.0, 3.0, 3.0], [5.0, 1.0, 5.0], [0.0, 0.0, 9.0]], temperature=0)
    assert_array(rows, [1, 0, 2], "greedy decoding of (N, V) logits: the first argmax of each row", dtype=np.int64)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_sample_from_logits_frequencies_match_the_softmax(seq, temperature):
    logits = np.array([2.0, 0.5, -1.0, 1.0, 0.0])
    rng = np.random.default_rng(int(temperature * 100))
    draws = [call(seq.sample_from_logits, logits, temperature=temperature, rng=rng) for _ in range(20000)]
    if any(type(d) is not int for d in draws[:5]):
        _fail("", f"expected Python ints for (V,) logits, got {type(draws[0]).__name__}")
    freq = np.bincount(np.asarray(draws), minlength=5) / 20000
    want = softmax_oracle(logits, temperature)
    # 4.5 standard deviations of a frequency over 20 000 draws: a fair sampler fails about once in 100 000 runs
    tol = 4.5 * np.sqrt(want * (1 - want) / 20000) + 1e-4
    if (np.abs(freq - want) > tol).any():
        _fail("", f"expected frequencies close to softmax(logits / {temperature}) = {_short(want, 4)}, got "
                  f"{_short(freq, 4)} over 20 000 draws")


def test_sample_from_logits_batch_rows_are_independent_distributions(seq):
    logits = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    rng = np.random.default_rng(5)
    draws = np.stack([np.asarray(call(seq.sample_from_logits, logits, rng=rng)) for _ in range(10000)])
    if draws.shape != (10000, 2) or draws.dtype != np.int64:
        _fail("", f"expected an int64 array of shape (2,) per call for (2, 3) logits, got {draws.dtype} "
                  f"{draws.shape[1:]}")
    for row, want in enumerate(softmax_oracle(logits)):
        freq = np.bincount(draws[:, row], minlength=3) / 10000
        tol = 4.5 * np.sqrt(want * (1 - want) / 10000) + 1e-4
        if (np.abs(freq - want) > tol).any():
            _fail("", f"expected the frequencies of row {row} close to {_short(want, 4)}, got {_short(freq, 4)} "
                      "(each row is its own distribution)")


def test_sample_from_logits_top_k_restricts_the_support(seq):
    logits = np.array([0.0, 2.0, 1.0, 1.5, -1.0])
    rng = np.random.default_rng(6)
    draws = np.array([call(seq.sample_from_logits, logits, top_k=2, rng=rng) for _ in range(5000)])
    support = set(np.unique(draws).tolist())
    if not support <= {1, 3}:
        _fail("", f"expected only the ids 1 and 3 with top_k=2 (the two largest logits), got {sorted(support)}")
    freq = np.bincount(draws, minlength=5)[[1, 3]] / 5000
    want = softmax([2.0, 1.5])
    tol = 4.5 * np.sqrt(want * (1 - want) / 5000) + 1e-4
    if (np.abs(freq - want) > tol).any():
        _fail("", f"expected the frequencies of ids 1 and 3 close to {_short(want, 4)} (softmax of the kept logits), "
                  f"got {_short(freq, 4)}")


def test_sample_from_logits_top_k_keeps_ties(seq):
    logits = np.array([1.0, 3.0, 0.0, 3.0])
    rng = np.random.default_rng(7)
    draws = np.array([call(seq.sample_from_logits, logits, top_k=1, rng=rng) for _ in range(2000)])
    support = set(np.unique(draws).tolist())
    if support != {1, 3}:
        _fail("", f"expected the ids 1 and 3 with top_k=1: two logits tie for the largest and both are kept, "
                  f"got {sorted(support)}")


def test_sample_from_logits_top_k_with_greedy(seq):
    got = call(seq.sample_from_logits, [0.0, 2.0, 5.0], temperature=0, top_k=2)
    if got != 2:
        _fail("", f"expected 2 (the argmax) with temperature=0 and top_k=2, got {got!r}")


def test_sample_from_logits_is_reproducible_with_a_seed(seq):
    logits = np.random.default_rng(8).normal(size=(6, 10))
    a = call(seq.sample_from_logits, logits.copy(), temperature=0.8, rng=np.random.default_rng(42))
    b = call(seq.sample_from_logits, logits.copy(), temperature=0.8, rng=np.random.default_rng(42))
    assert_array(np.asarray(a), np.asarray(b), "two generators created with the same seed must give the same ids")


def test_sample_from_logits_accepts_no_generator(seq):
    got = call(seq.sample_from_logits, [0.0, 1.0, 2.0], rng=None)
    if type(got) is not int or not 0 <= got < 3:
        _fail("", f"expected a Python int in [0, 3) with rng=None (np.random.default_rng()), got {got!r}")


def test_sample_from_logits_does_not_modify_its_input(seq):
    logits = np.array([[0.5, -1.0, 2.0, 0.0]])
    saved = logits.copy()
    call(seq.sample_from_logits, logits, temperature=0.5, top_k=2, rng=np.random.default_rng(0))
    if not np.array_equal(logits, saved):
        _fail("", "expected the logits unchanged by sample_from_logits (top_k must work on a copy, not set -inf "
                  "in place)")


@pytest.mark.parametrize("kwargs,why", [
    pytest.param(dict(temperature=-0.5), "temperature = -0.5 (T must be >= 0)", id="negative-temperature"),
    pytest.param(dict(top_k=0), "top_k = 0", id="top_k-0"),
    pytest.param(dict(top_k=4), "top_k = 4 for V = 3 logits", id="top_k-above-V"),
    pytest.param(dict(top_k=1.5), "top_k = 1.5 is not an int", id="top_k-float"),
])
def test_sample_from_logits_refuses_invalid_arguments(seq, kwargs, why):
    assert_raises(ValueError, seq.sample_from_logits, [1.0, 2.0, 3.0], rng=np.random.default_rng(0), why=why, **kwargs)
