"""Tests of mylearn.embeddings (bonus chapter B2): oracle tests and properties.

    pytest tests/test_chb2_embeddings.py              # your code (mon_travail/mylearn/embeddings.py)
    pytest tests/test_chb2_embeddings.py --impl=ref   # the reference

Oracles: an explicit double loop over the positions and the distances (co-occurrences); the explicit float64
formula of the docstring with ``np.log``, cell by cell, and hand-computed values (PPMI, with ``alpha``);
``sklearn.metrics.pairwise.cosine_similarity`` and the formula of the docstring with its ``eps`` (cosine);
``torch.topk`` on ``torch.nn.functional.cosine_similarity`` (when PyTorch is installed),
``sklearn.neighbors.NearestNeighbors(metric='cosine')`` and a stable sort of the similarities (neighbours).
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_cooccurrence_matrix_ or test_ppmi_"`` (B2.22),
``-k "test_cosine_similarity_matrix_ or test_most_similar_"`` (B2.23).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified
and may be lists (integer arrays too); ``cooccurrence_matrix`` returns a float array of shape
``(vocab_size, vocab_size)``, counts a word next to itself (``C[w, w]``) and accepts a sequence shorter than the
window; ``ppmi`` gives 0 (never NaN) in a row or a column of zeros; ``most_similar`` returns NumPy arrays
(integer indices, float similarities), an empty pair when every row is excluded, and refuses a query of shape
``(1, d)``.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

ORACLE_BUG = ("bug in tests/test_chb2_embeddings.py (not in your code): the oracle of this test "
              "disagrees with another one")


@pytest.fixture
def emb(mylearn_module):
    return mylearn_module("embeddings")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3, max_line_width=10**6)
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


def assert_float_array(result, shape, msg):
    if result is None:
        _fail(msg, "expected an array, got None (did you forget the return?)")
    if not isinstance(result, np.ndarray):
        _fail(msg, f"expected a NumPy array, got {type(result).__name__}")
    if result.shape != shape:
        _fail(msg, f"expected shape {shape}, got shape {result.shape}")
    if not np.issubdtype(result.dtype, np.floating):
        _fail(msg, f"expected a float array, got dtype {result.dtype}")


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


def neighbours(result, msg):
    """The pair (indices, similarities) of most_similar, checked."""
    if result is None:
        _fail(msg, "expected the tuple (indices, similarities), got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 2:
        _fail(msg, f"expected the tuple (indices, similarities), got {type(result).__name__}"
                   + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else ""))
    idx, sims = result
    for value, what in ((idx, "indices"), (sims, "similarities")):
        if not isinstance(value, np.ndarray):
            _fail(msg, f"expected {what} as a NumPy array, got {type(value).__name__}")
        if value.ndim != 1:
            _fail(msg, f"expected {what} of shape (k,), got shape {value.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        _fail(msg, f"expected integer indices, got dtype {idx.dtype}")
    if not np.issubdtype(sims.dtype, np.floating):
        _fail(msg, f"expected float similarities, got dtype {sims.dtype}")
    return idx, sims


def assert_indices(idx, want, msg):
    want = list(map(int, want))
    got = [int(i) for i in idx]
    if got != want:
        i = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        _fail(msg, f"expected index {want[i] if i < len(want) else 'nothing'} at rank {i}, got "
                   f"{got[i] if i < len(got) else 'nothing'} ({len(want)} indices expected, {len(got)} given)",
              f"expected {want[:20]}\n     got {got[:20]}")


# ------------------------------------------------------------------ oracles
def cooc_oracle(ids, vocab_size, window=2, weighting="uniform", symmetric=True):
    """Double loop over the positions i and the positions j within the window."""
    ids = list(ids)
    C = np.zeros((vocab_size, vocab_size))
    for i, w in enumerate(ids):
        for j, c in enumerate(ids):
            d = j - i
            if d == 0 or abs(d) > window or (d < 0 and not symmetric):
                continue
            C[w, c] += 1.0 if weighting == "uniform" else 1.0 / abs(d)
    return C


def ppmi_oracle(C, alpha=1.0):
    """The formula of the docstring, cell by cell, in float64 with np.log."""
    C = np.asarray(C, dtype=np.float64)
    total = C.sum()
    rows, cols = C.sum(axis=1), C.sum(axis=0)
    smooth = cols ** alpha
    out = np.zeros_like(C)
    for w in range(C.shape[0]):
        for c in range(C.shape[1]):
            if C[w, c] > 0:
                p_wc = C[w, c] / total
                p_w = rows[w] / total
                p_c = smooth[c] / smooth.sum()
                out[w, c] = max(0.0, float(np.log(p_wc / (p_w * p_c))))
    return out


def cosine_oracle(A, B, eps=1e-12):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    S = np.empty((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            S[i, j] = A[i] @ B[j] / ((np.sqrt(A[i] @ A[i]) + eps) * (np.sqrt(B[j] @ B[j]) + eps))
    return S


def topk_oracle(E, query, k, exclude=()):
    """Similarities of the docstring formula, excluded rows removed, stable sort (ties: smaller index first)."""
    E = np.asarray(E, dtype=float)
    sims = cosine_oracle(np.asarray(query, dtype=float)[None, :], E)[0]
    rows = np.array([i for i in range(E.shape[0]) if i not in set(exclude)], dtype=int)
    order = sorted(rows, key=lambda i: (-sims[i], i))[:k]
    order = np.array(order, dtype=int)
    if torch is not None and order.size and len(set(np.round(sims[rows], 12))) == rows.size:
        t = F.cosine_similarity(torch.tensor(E[rows]), torch.tensor(np.asarray(query, float))[None, :], dim=1)
        values, pos = torch.topk(t, min(k, rows.size))
        assert np.array_equal(rows[pos.numpy()], order), ORACLE_BUG
        assert np.allclose(values.numpy(), sims[order], rtol=1e-9, atol=1e-12), ORACLE_BUG
    return order, sims[order] if order.size else np.zeros(0)


# ================================================================== cooccurrence_matrix (B2.22)
def test_cooccurrence_matrix_docstring_examples(emb):
    for kwargs, want, what in [
        (dict(window=1), [[0, 1, 0], [1, 0, 2], [0, 2, 0]], "window=1"),
        (dict(window=1, symmetric=False), [[0, 1, 0], [0, 0, 1], [0, 1, 0]], "window=1, symmetric=False"),
        (dict(window=2, weighting="harmonic"), [[0, 1, 0.5], [1, 1, 2], [0.5, 2, 0]], "window=2, harmonic"),
    ]:
        C = call(emb.cooccurrence_matrix, [0, 1, 2, 1], expected="the matrix", vocab_size=3, **kwargs)
        assert_float_array(C, (3, 3), f"cooccurrence_matrix([0, 1, 2, 1], 3, {what})")
        assert_close(C, want, msg=f"cooccurrence_matrix([0, 1, 2, 1], 3, {what})")


@pytest.mark.parametrize("n,vocab_size,window,weighting,symmetric", [
    pytest.param(30, 6, 2, "uniform", True, id="n30-V6-w2"),
    pytest.param(30, 6, 3, "uniform", False, id="n30-V6-w3-right-only"),
    pytest.param(40, 5, 4, "harmonic", True, id="n40-V5-w4-harmonic"),
    pytest.param(25, 8, 5, "harmonic", False, id="n25-V8-w5-harmonic-right-only"),
    pytest.param(50, 3, 1, "uniform", True, id="n50-V3-w1-many-repeats"),
    pytest.param(3, 4, 5, "uniform", True, id="shorter-than-the-window"),
    pytest.param(1, 4, 2, "harmonic", True, id="one-token"),
])
def test_cooccurrence_matrix_matches_a_double_loop(emb, n, vocab_size, window, weighting, symmetric):
    ids = np.random.default_rng(n + vocab_size + window).integers(0, vocab_size, size=n)
    want = cooc_oracle(ids, vocab_size, window, weighting, symmetric)
    C = call(emb.cooccurrence_matrix, ids.copy(), vocab_size, window=window, weighting=weighting,
             symmetric=symmetric, expected="the matrix")
    msg = f"cooccurrence_matrix(n={n}, vocab_size={vocab_size}, window={window}, {weighting!r}, symmetric={symmetric})"
    assert_float_array(C, (vocab_size, vocab_size), msg)
    assert_close(C, want, msg=msg + " (every pair (word, context) at distance d = 1 .. window adds its weight)")


def test_cooccurrence_matrix_counts_a_word_next_to_itself(emb):
    C = call(emb.cooccurrence_matrix, [0, 0, 0, 1], 2, window=1, expected="the matrix")
    assert_close(C, cooc_oracle([0, 0, 0, 1], 2, 1), msg="cooccurrence_matrix([0, 0, 0, 1], 2, window=1): "
                 "C[0, 0] = 4 (the repeated pairs add up: np.add.at, not C[rows, cols] += 1)")


@pytest.mark.parametrize("n,window", [pytest.param(20, 1, id="n20-w1"), pytest.param(57, 3, id="n57-w3"),
                                      pytest.param(9, 8, id="n9-w8")])
def test_cooccurrence_matrix_uniform_symmetric_properties(emb, n, window):
    ids = np.random.default_rng(n).integers(0, 7, size=n)
    C = call(emb.cooccurrence_matrix, list(ids), 7, window=window, expected="the matrix")
    assert_float_array(C, (7, 7), "cooccurrence_matrix")
    assert_close(C, C.T, msg="'uniform' + symmetric=True must give C == C.T")
    total = 2 * sum(n - d for d in range(1, window + 1))
    assert_close(C.sum(), total, msg=f"C.sum() = 2 * sum(n - d for d in 1..window) = {total} (n={n}, window={window})")


def test_cooccurrence_matrix_accepts_integer_arrays_and_lists(emb):
    for ids in ([2, 0, 1, 2], np.array([2, 0, 1, 2], dtype=np.int32), np.array([2, 0, 1, 2], dtype=np.uint8)):
        C = call(emb.cooccurrence_matrix, ids, 3, window=2, expected="the matrix")
        assert_float_array(C, (3, 3), f"ids of type {type(ids).__name__}"
                           + (f" {ids.dtype}" if isinstance(ids, np.ndarray) else ""))
        assert_close(C, cooc_oracle([2, 0, 1, 2], 3, 2), msg="cooccurrence_matrix([2, 0, 1, 2], 3, window=2)")


def test_cooccurrence_matrix_does_not_modify_its_input(emb):
    ids = np.random.default_rng(3).integers(0, 5, size=30)
    saved = ids.copy()
    call(emb.cooccurrence_matrix, ids, 5, window=3, weighting="harmonic")
    if not np.array_equal(ids, saved):
        _fail("", "expected the ids unchanged by cooccurrence_matrix")


@pytest.mark.parametrize("ids,kwargs,why", [
    pytest.param([[0, 1], [1, 0]], {}, "ids of shape (2, 2) is not 1-D", id="ids-2d"),
    pytest.param([0, 1, 2], dict(window=0), "window = 0", id="window-0"),
    pytest.param([0, 1, 2], dict(weighting="gaussian"), "weighting 'gaussian' is unknown", id="weighting"),
    pytest.param([0, 1, 3], {}, "id 3 is outside [0, 3)", id="id-equal-to-vocab-size"),
    pytest.param([0, -1, 2], {}, "id -1 is outside [0, 3) (no Python negative index)", id="negative-id"),
])
def test_cooccurrence_matrix_refuses_invalid_arguments(emb, ids, kwargs, why):
    assert_raises(ValueError, emb.cooccurrence_matrix, ids, 3, why=why, **kwargs)


# ================================================================== ppmi (B2.22)
DOC_C = [[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]]


def test_ppmi_docstring_examples(emb):
    P = call(emb.ppmi, np.array(DOC_C), expected="the PPMI")
    assert_float_array(P, (3, 3), "ppmi(C)")
    assert_close(P, [[0, 0.47, 0], [0.47, 0, 0.5878], [0, 0.5878, 0]], rtol=0, atol=5e-5, msg="ppmi(C) of the docstring")
    P75 = call(emb.ppmi, np.array(DOC_C), alpha=0.75, expected="the PPMI")
    assert_close(P75, [[0, 0.5218, 0], [0.3941, 0, 0.5838], [0, 0.6396, 0]], rtol=0, atol=5e-5,
                 msg="ppmi(C, alpha=0.75) of the docstring (smoothing of the contexts only: the result is not "
                     "symmetric)")


def test_ppmi_hand_computed_cell(emb):
    C = np.array([[4.0, 1.0], [1.0, 2.0]])
    # p(0, 0) = 4/8, p(w=0) = 5/8, p(c=0) = 5/8 -> log(0.5 / (25/64)) = log(1.28)
    want00 = np.log(1.28)
    # alpha = 0.5: p_alpha(c=0) = sqrt(5) / (sqrt(5) + sqrt(3))
    want00_half = np.log(0.5 / (5 / 8 * np.sqrt(5) / (np.sqrt(5) + np.sqrt(3))))
    P = call(emb.ppmi, C.copy(), expected="the PPMI")
    assert_close(P[0, 0], want00, msg="PPMI(0, 0) of [[4, 1], [1, 2]] = log(p(w, c) / (p(w) p(c))) = log(1.28)")
    Ph = call(emb.ppmi, C.copy(), alpha=0.5, expected="the PPMI")
    assert_close(Ph[0, 0], want00_half, msg="PPMI(0, 0) with alpha=0.5: p_alpha(c) = count(c)**0.5 / "
                 "sum(count(c')**0.5), only for the contexts (the columns)")
    assert_close(Ph[1, 0], 0.0, msg="PPMI(1, 0) with alpha=0.5 (negative PMI: max(0, .))")


@pytest.mark.parametrize("shape,alpha,density", [
    pytest.param((6, 6), 1.0, 0.7, id="6x6-alpha1"),
    pytest.param((5, 8), 1.0, 0.5, id="5x8-alpha1"),
    pytest.param((7, 4), 0.75, 0.6, id="7x4-alpha0.75"),
    pytest.param((9, 9), 0.5, 0.4, id="9x9-alpha0.5"),
    pytest.param((4, 6), 2.0, 0.8, id="4x6-alpha2"),
])
def test_ppmi_matches_the_explicit_formula(emb, shape, alpha, density):
    rng = np.random.default_rng(shape[0] * 10 + shape[1])
    C = rng.integers(1, 20, size=shape) * (rng.random(shape) < density)
    C = C.astype(float)
    want = ppmi_oracle(C, alpha)
    P = call(emb.ppmi, C.copy(), alpha=alpha, expected="the PPMI")
    msg = f"ppmi of a random {shape} matrix, alpha={alpha}"
    assert_float_array(P, shape, msg)
    assert_close(P, want, msg=msg + " (max(0, log(p(w, c) / (p(w) p_alpha(c)))), natural log)")


def test_ppmi_is_zero_where_c_is_zero_and_never_negative(emb):
    rng = np.random.default_rng(7)
    C = rng.integers(0, 6, size=(8, 8)).astype(float)
    P = call(emb.ppmi, C.copy(), alpha=0.75, expected="the PPMI")
    assert_float_array(P, (8, 8), "ppmi")
    if np.isnan(P).any():
        _fail("", "expected no NaN in the PPMI, got NaN")
    if (P[C == 0] != 0).any():
        _fail("", "expected PPMI = 0 wherever C == 0")
    if (P < 0).any():
        _fail("", f"expected PPMI >= 0 everywhere, got {P.min():.6g} (keep max(0, PMI))")


def test_ppmi_of_a_row_and_a_column_of_zeros(emb):
    C = np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 0.0], [2.0, 0.0, 5.0]])
    P = call(emb.ppmi, C.copy(), expected="the PPMI")
    assert_close(P, ppmi_oracle(C), msg="ppmi of a matrix with a row and a column of zeros: 0 there, not NaN "
                                        "(0 / 0 in p(w, c) / (p(w) p(c)))")


@pytest.mark.parametrize("seed", range(3))
def test_ppmi_of_an_independent_matrix_is_zero(emb, seed):
    rng = np.random.default_rng(seed)
    C = np.outer(rng.integers(1, 10, size=6), rng.integers(1, 10, size=5)).astype(float)
    P = call(emb.ppmi, C.copy(), expected="the PPMI")
    assert_close(P, np.zeros_like(C), rtol=0, atol=1e-12,
                 msg="ppmi(np.outer(u, v)): p(w, c) = p(w) p(c), so PMI = 0 everywhere (up to rounding)")


def test_ppmi_accepts_integer_counts_and_lists(emb):
    C = np.array(DOC_C, dtype=np.int64)
    P = call(emb.ppmi, C, expected="the PPMI of integer counts")
    assert_close(P, ppmi_oracle(DOC_C), msg="ppmi of an int64 matrix (compute in float)")
    P2 = call(emb.ppmi, [list(map(int, row)) for row in DOC_C], alpha=0.75, expected="the PPMI of a list")
    assert_close(P2, ppmi_oracle(DOC_C, 0.75), msg="ppmi of a list of lists, alpha=0.75")


def test_ppmi_does_not_modify_its_input(emb):
    C = np.random.default_rng(4).integers(0, 9, size=(5, 5)).astype(float)
    saved = C.copy()
    call(emb.ppmi, C, alpha=0.75)
    if not np.array_equal(C, saved):
        _fail("", "expected C unchanged by ppmi (compute the probabilities in a new array)")


@pytest.mark.parametrize("C,why", [
    pytest.param(np.array([1.0, 2.0]), "C of shape (2,) is not 2-D", id="1d"),
    pytest.param(np.array([[1.0, -1.0], [2.0, 3.0]]), "C has a negative entry", id="negative"),
    pytest.param(np.zeros((3, 3)), "C sums to 0", id="all-zero"),
])
def test_ppmi_refuses_invalid_counts(emb, C, why):
    assert_raises(ValueError, emb.ppmi, C, why=why)


# ================================================================== cosine_similarity_matrix (B2.23)
def test_cosine_similarity_matrix_docstring_examples(emb):
    A = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    S = call(emb.cosine_similarity_matrix, A.copy(), expected="the similarities")
    assert_float_array(S, (3, 3), "cosine_similarity_matrix(A)")
    r = 1 / np.sqrt(2)
    assert_close(S, [[1, r, 0], [r, 1, 0], [0, 0, 0]], rtol=0, atol=1e-9,
                 msg="cosine_similarity_matrix(A) of the docstring (a row of zeros has similarity 0)")
    S2 = call(emb.cosine_similarity_matrix, A.copy(), [[0.0, 2.0], [-3.0, 0.0]], expected="the similarities")
    assert_float_array(S2, (3, 2), "cosine_similarity_matrix(A, B) with A (3, 2) and B (2, 2)")
    assert_close(S2, [[0, -1], [r, -r], [0, 0]], rtol=0, atol=1e-9, msg="cosine_similarity_matrix(A, B) of the docstring")


@pytest.mark.parametrize("n,m,d", [pytest.param(5, 7, 3, id="5x3-vs-7x3"), pytest.param(10, 1, 6, id="10x6-vs-1x6"),
                                   pytest.param(1, 4, 50, id="1x50-vs-4x50")])
def test_cosine_similarity_matrix_matches_sklearn(emb, n, m, d):
    rng = np.random.default_rng(n * m + d)
    A, B = rng.normal(size=(n, d)), rng.normal(size=(m, d)) * 3.0
    want = cosine_similarity(A, B)
    assert np.allclose(want, cosine_oracle(A, B), rtol=1e-9, atol=1e-12), ORACLE_BUG
    S = call(emb.cosine_similarity_matrix, A.copy(), B.copy(), expected="the similarities")
    msg = f"cosine_similarity_matrix of A {A.shape} and B {B.shape}"
    assert_float_array(S, (n, m), msg + ": shape (n, m)")
    assert_close(S, want, rtol=1e-9, atol=1e-12, msg=msg + " (as sklearn cosine_similarity)")


def test_cosine_similarity_matrix_b_none_is_a(emb):
    A = np.random.default_rng(1).normal(size=(6, 4))
    A[2] = 0.0
    S = call(emb.cosine_similarity_matrix, A.copy(), expected="the similarities")
    assert_close(S, cosine_similarity(A), rtol=1e-9, atol=1e-12, msg="cosine_similarity_matrix(A) = (A, A), "
                 "zero row included")


@pytest.mark.parametrize("eps", [0.5, 2.0])
def test_cosine_similarity_matrix_adds_eps_to_each_norm(emb, eps):
    rng = np.random.default_rng(int(eps * 10))
    A, B = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    S = call(emb.cosine_similarity_matrix, A.copy(), B.copy(), eps=eps, expected="the similarities")
    assert_close(S, cosine_oracle(A, B, eps), msg=f"A[i] . B[j] / ((||A[i]|| + eps) (||B[j]|| + eps)) with eps={eps}")


def test_cosine_similarity_matrix_stays_in_minus_one_one(emb):
    rng = np.random.default_rng(5)
    A = np.vstack([rng.normal(size=(20, 8)), rng.normal(size=(1, 8)) * 1e-8, np.full((1, 8), 1e8)])
    S = call(emb.cosine_similarity_matrix, A.copy(), -A[:5].copy(), expected="the similarities")
    if not (np.abs(S) <= 1 + 1e-12).all():
        _fail("", f"expected similarities in [-1, 1], got {np.abs(S).max():.17g}")


def test_cosine_similarity_matrix_accepts_lists_and_integers(emb):
    A = [[1, 2], [3, 4], [0, 5]]
    S = call(emb.cosine_similarity_matrix, A, [[2, 1]], expected="the similarities")
    assert_close(S, cosine_similarity(np.array(A, float), [[2.0, 1.0]]), rtol=1e-9, atol=1e-12,
                 msg="cosine_similarity_matrix of lists of int")
    Ai = np.array(A, dtype=np.int64)
    S2 = call(emb.cosine_similarity_matrix, Ai, expected="the similarities of an int64 array")
    assert_close(S2, cosine_similarity(Ai.astype(float)), rtol=1e-9, atol=1e-12,
                 msg="cosine_similarity_matrix of an int64 array (normalise in float)")


def test_cosine_similarity_matrix_does_not_modify_its_inputs(emb):
    rng = np.random.default_rng(6)
    A, B = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    sa, sb = A.copy(), B.copy()
    call(emb.cosine_similarity_matrix, A, B)
    call(emb.cosine_similarity_matrix, A)
    if not (np.array_equal(A, sa) and np.array_equal(B, sb)):
        _fail("", "expected A and B unchanged by cosine_similarity_matrix (normalise into new arrays)")


@pytest.mark.parametrize("A,B,why", [
    pytest.param(np.ones(3), None, "A of shape (3,) is not 2-D", id="a-1d"),
    pytest.param(np.ones((2, 3)), np.ones(3), "B of shape (3,) is not 2-D", id="b-1d"),
    pytest.param(np.ones((2, 3)), np.ones((2, 4)), "A has 3 columns and B 4", id="columns"),
])
def test_cosine_similarity_matrix_refuses_invalid_shapes(emb, A, B, why):
    assert_raises(ValueError, emb.cosine_similarity_matrix, A, B, why=why)


# ================================================================== most_similar (B2.23)
DOC_E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [2.0, 0.1]]


def test_most_similar_docstring_examples(emb):
    idx, sims = neighbours(call(emb.most_similar, np.array(DOC_E), [1.0, 0.0], k=3, expected="(indices, similarities)"),
                           "most_similar(E, [1, 0], k=3)")
    assert_indices(idx, [0, 4, 2], "most_similar(E, [1, 0], k=3) of the docstring")
    assert_close(sims, [1.0, 0.9988, 0.7071], rtol=0, atol=5e-5, msg="the similarities of the docstring")
    idx2, _ = neighbours(call(emb.most_similar, np.array(DOC_E), [1.0, 0.0], k=3, exclude=(0,),
                              expected="(indices, similarities)"), "most_similar(..., exclude=(0,))")
    assert_indices(idx2, [4, 2, 1], "most_similar(E, [1, 0], k=3, exclude=(0,)): the row 0 is never returned")


@pytest.mark.parametrize("n,d,k,n_exclude", [
    pytest.param(50, 8, 5, 0, id="50x8-k5"), pytest.param(200, 16, 10, 3, id="200x16-k10-exclude3"),
    pytest.param(30, 4, 30, 5, id="30x4-k-all"), pytest.param(12, 3, 1, 2, id="12x3-k1"),
])
def test_most_similar_matches_torch_topk_and_nearest_neighbors(emb, n, d, k, n_exclude):
    rng = np.random.default_rng(n + d + k)
    E = rng.normal(size=(n, d))
    query = rng.normal(size=d)
    exclude = tuple(int(i) for i in rng.choice(n, size=n_exclude, replace=False))
    want_idx, want_sims = topk_oracle(E, query, k, exclude)
    kept = np.array([i for i in range(n) if i not in exclude])
    nn = NearestNeighbors(n_neighbors=min(k, kept.size), metric="cosine", algorithm="brute").fit(E[kept])
    dist, pos = nn.kneighbors(query[None, :])
    assert np.array_equal(kept[pos[0]], want_idx), ORACLE_BUG
    assert np.allclose(1 - dist[0], want_sims, rtol=1e-9, atol=1e-9), ORACLE_BUG
    idx, sims = neighbours(call(emb.most_similar, E.copy(), query.copy(), k=k, exclude=exclude,
                                expected="(indices, similarities)"), "most_similar")
    msg = f"most_similar(E {E.shape}, query, k={k}, exclude={exclude})"
    assert_indices(idx, want_idx, msg + " (by decreasing cosine similarity)")
    assert_close(sims, want_sims, msg=msg + ": the similarities of the returned rows")


def test_most_similar_analogy_excludes_the_three_words(emb):
    # a : b :: c : ?   with E[b] - E[a] + E[c] equal to E[c] shifted like b from a
    E = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.2, 1.0],
                  [1.0, 0.9, 0.1]])
    a, b, c = 0, 1, 2
    query = E[b] - E[a] + E[c]
    want_idx, want_sims = topk_oracle(E, query, 3, (a, b, c))
    assert want_idx[0] == 3, ORACLE_BUG
    idx, sims = neighbours(call(emb.most_similar, E.copy(), query.copy(), k=3, exclude=(a, b, c),
                                expected="(indices, similarities)"), "most_similar")
    assert_indices(idx, want_idx, "the analogy 0 : 1 :: 2 : ? with exclude=(0, 1, 2) (answer 3; the three words "
                                  "of the question never come back)")
    assert_close(sims, want_sims, msg="the similarities of the analogy")


def test_most_similar_ties_go_to_the_smaller_index(emb):
    E = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    idx, _ = neighbours(call(emb.most_similar, E.copy(), [1.0, 0.0], k=4, expected="(indices, similarities)"),
                        "most_similar")
    assert_indices(idx, [1, 3, 4, 0], "three identical rows 1, 3, 4 (ties: smaller index first), then row 0")
    big = np.repeat(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 40, axis=0)[np.random.default_rng(0).permutation(120)]
    want, _ = topk_oracle(big, [1.0, 0.2], 60)
    idx2, _ = neighbours(call(emb.most_similar, big.copy(), [1.0, 0.2], k=60, expected="(indices, similarities)"),
                         "most_similar")
    assert_indices(idx2, want, "120 rows, three distinct ones repeated 40 times: each tie in increasing index order "
                               "(a stable sort)")
    idx3, sims3 = neighbours(call(emb.most_similar, E.copy(), [0.0, 0.0], k=3, expected="(indices, similarities)"),
                             "most_similar")
    assert_indices(idx3, [0, 1, 2], "a zero query: every similarity is 0, so the 3 smallest indices")
    assert_close(sims3, [0.0, 0.0, 0.0], msg="a zero query has similarity 0 with every row")


def test_most_similar_returns_all_remaining_rows_when_fewer_than_k(emb):
    idx, sims = neighbours(call(emb.most_similar, np.array(DOC_E), [1.0, 0.0], k=10, exclude=(1, 3),
                                expected="(indices, similarities)"), "most_similar")
    assert_indices(idx, [0, 4, 2], "k=10 with 3 rows left after exclude=(1, 3): all of them")
    if sims.shape != (3,):
        _fail("", f"expected 3 similarities, got shape {sims.shape}")
    idx0, sims0 = neighbours(call(emb.most_similar, np.array(DOC_E), [1.0, 0.0], k=2, exclude=range(5),
                                  expected="(indices, similarities)"), "most_similar")
    if idx0.shape != (0,) or sims0.shape != (0,):
        _fail("", f"expected two arrays of shape (0,) when every row is excluded, got {idx0.shape} and {sims0.shape}")


def test_most_similar_accepts_lists(emb):
    idx, sims = neighbours(call(emb.most_similar, [list(r) for r in DOC_E], [1, 0], k=2, exclude=[4],
                                expected="(indices, similarities)"), "most_similar")
    assert_indices(idx, [0, 2], "most_similar of a list of lists, query [1, 0] (ints), exclude=[4]")


def test_most_similar_does_not_modify_its_inputs(emb):
    rng = np.random.default_rng(8)
    E, q = rng.normal(size=(10, 4)), rng.normal(size=4)
    se, sq = E.copy(), q.copy()
    call(emb.most_similar, E, q, k=3, exclude=(1,))
    if not (np.array_equal(E, se) and np.array_equal(q, sq)):
        _fail("", "expected E and query unchanged by most_similar")


@pytest.mark.parametrize("query,k,why", [
    pytest.param([1.0, 0.0], 0, "k = 0", id="k-0"),
    pytest.param([1.0, 0.0, 0.0], 2, "a query of length 3 for E with 2 columns", id="query-length"),
    pytest.param([[1.0, 0.0]], 2, "a query of shape (1, 2) is not a vector", id="query-2d"),
])
def test_most_similar_refuses_invalid_arguments(emb, query, k, why):
    assert_raises(ValueError, emb.most_similar, np.array(DOC_E), query, k=k, why=why)
