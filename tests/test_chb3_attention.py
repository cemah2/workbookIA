"""Tests of mylearn.attention (bonus chapter B3): oracle tests and properties.

    pytest tests/test_chb3_attention.py              # your code (mon_travail/mylearn/attention.py)
    pytest tests/test_chb3_attention.py --impl=ref   # the reference

Oracles: the formulas of the docstrings transcribed in this file with explicit loops (one batch, one head, one
query at a time; the RoPE rotation with complex numbers), and, when PyTorch is installed, checked at every test
against ``torch.tril``, ``torch.nn.functional.scaled_dot_product_attention`` (boolean mask, True = may attend,
``dropout_p=0``), an explicit torch softmax, ``torch.nn.MultiheadAttention`` (weights copied, mask passed as
``~mask``), ``view(N, T, H, D // H).transpose(1, 2)``, a pre-LN block written in PyTorch from the docstring
(``F.layer_norm``, ``F.gelu``) and ``torch.nn.TransformerEncoderLayer(norm_first=True)``; a disagreement between the
transcription and torch is a bug of this file, not yours. Without PyTorch, the transcriptions are the oracle, so no
test is skipped (except the comparison with Hugging Face ``transformers``, ``LlamaRotaryEmbedding`` and
``apply_rotary_pos_emb``, which needs that library). The sinusoidal encoding is compared with the closed-form
formula in float64 (implementation of the PyTorch tutorial).
Your functions are always called on copies of the test data, and the tests ``..._does_not_modify_...`` check that
the inputs are left as they were.
``scaled_dot_product_attention`` uses your ``softmax`` (chapter 17) and ``transformer_block`` your ``layer_norm``
(chapter 20) and ``gelu`` (chapter 17): if one of these tests fails, run the tests of those chapters first.

Every test name starts with the name of the function it tests, so that each exercise runs its own group:
``-k "test_causal_mask_ or test_padding_mask_ or test_scaled_dot_product_attention_"`` (B3.12),
``-k "test_split_heads_ or test_merge_heads_ or test_multi_head_attention_"`` (B3.14),
``-k "test_positional_encoding_ or test_apply_rope_"`` (B3.15), ``-k "test_transformer_block_"`` (B3.16).

Contract precisions written in the statements of these exercises and checked here: the inputs are never modified
and may be lists (``lengths``, ``q``, ``k``, ``v``, ``mask``, ``positions``, the ``x`` of ``split_heads``);
the masks are boolean arrays; ``scale=0.0`` is a scale (only ``None`` means ``1 / sqrt(d_k)``); a mask row with no
allowed key raises ``ValueError``, also with a 4-D mask (a sequence fully masked by a ``(N, 1, 1, S)`` padding
mask); a missing parameter raises ``ValueError`` (not ``KeyError``); ``multi_head_attention``
accepts any mapping (extra keys ignored) and a mask of shape ``(L, S)``, ``(N, 1, 1, S)``, ``(N, 1, L, S)`` or
``(N, H, L, S)``; ``eps`` is used by both LayerNorms.
The values are compared with rtol=1e-6, atol=1e-6 (a computation in float32 passes), except against Hugging Face,
which computes the RoPE angles in float32 (rtol=1e-5, atol=1e-5). The masks and the heads are compared exactly.
The first line of every failure message says what was expected (it is the line that ``pytest -rf`` shows).
"""

from __future__ import annotations

import cmath
import math
from types import MappingProxyType

import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # the PyTorch oracles run on Colab; the transcriptions take over here
    torch = None

ORACLE_BUG = ("bug in tests/test_chb3_attention.py (not in your code): the oracle of this test "
              "disagrees with torch")

MHA_KEYS = ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o")
BLOCK_KEYS = ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "W_1", "b_1", "W_2", "b_2")


@pytest.fixture
def att(mylearn_module):
    return mylearn_module("attention")


# ------------------------------------------------------------------ helpers
def _short(values, digits: int = 6) -> str:
    """A compact, one-line view of a number or an array, for a message."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        item = arr.item()
        return f"{item:.{digits}g}" if isinstance(item, float) else repr(item)
    text = np.array2string(arr, precision=digits, separator=", ", threshold=12, edgeitems=3,
                           max_line_width=10**6)
    return " ".join(text.split())


def _fail(msg: str, head: str, data: str = ""):
    """First line: the explanation and what was expected (shown by `pytest -rf`); then the data."""
    lines = [f"{msg}: {head}" if msg else head]
    if data:
        lines.append(data)
    raise AssertionError("\n".join(lines)) from None


# Tolerance of the comparisons: a computation in float32 (errors up to about 3e-7) passes, a wrong formula does not.
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


def assert_array(result, shape, msg, kind="float"):
    """A NumPy array of the given shape and kind ('float', 'bool' or None for any dtype)."""
    if result is None:
        _fail(msg, "expected an array, got None (did you forget the return?)")
    if not isinstance(result, np.ndarray):
        _fail(msg, f"expected a NumPy array, got {type(result).__name__}")
    if result.shape != tuple(shape):
        _fail(msg, f"expected shape {tuple(shape)}, got shape {result.shape}")
    if kind == "float" and not np.issubdtype(result.dtype, np.floating):
        _fail(msg, f"expected a float array, got dtype {result.dtype}")
    if kind == "bool" and result.dtype != bool:
        _fail(msg, f"expected a boolean array (dtype bool), got dtype {result.dtype}")


def assert_equal(result, expected, msg):
    """Exact equality (masks, heads); the message gives the first differing index."""
    got, want = np.asarray(result), np.asarray(expected)
    if got.shape != want.shape:
        _fail(msg, f"expected shape {want.shape}, got shape {got.shape}")
    diff = got != want
    if diff.any():
        index = tuple(int(i) for i in np.argwhere(diff)[0])
        head = (f"expected {_short(want[index])} at index {index}, got {_short(got[index])} "
                f"({int(diff.sum())} differing value(s) out of {want.size})")
        data = f"expected {_short(want)}\n     got {_short(got)}" if want.size <= 36 else ""
        _fail(msg, head, data)


def assert_raises(error, function, *args, why="", **kwargs):
    """The call must raise `error`; `why` names the case in the failure message."""
    name = getattr(function, "__name__", "the function")
    try:
        function(*args, **kwargs)
    except NotImplementedError:
        raise                  # before `except error`: NotImplementedError is a RuntimeError; shown as "⏳"
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
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = getattr(function, "__name__", "the function")
        raise AssertionError(f"expected {expected}, but {name} raised {type(exc).__name__}: {exc}") from None


def pair(result, what, names=("output", "weights")):
    """Unpack a 2-tuple result, or fail with a message that says what was expected."""
    label = f"the tuple ({', '.join(names)})"
    if result is None:
        _fail("", f"expected {label} from {what}, got None (did you forget the return?)")
    if not isinstance(result, tuple) or len(result) != 2:
        _fail("", f"expected {label} from {what}, got {type(result).__name__}"
                  + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else ""))
    return result


def copies(*arrays):
    return tuple(np.array(a, copy=True) for a in arrays)


def copy_params(params):
    return {key: np.array(value, copy=True) for key, value in params.items()}


# ------------------------------------------------------------------ oracles
def causal_oracle(L, S=None):
    """M[i, j] = (j <= i + S - L), double loop; checked against torch.tril(..., diagonal=S - L)."""
    S = L if S is None else S
    M = np.zeros((L, S), dtype=bool)
    for i in range(L):
        for j in range(S):
            M[i, j] = j <= i + S - L
    if torch is not None:
        want = torch.tril(torch.ones(L, S, dtype=torch.bool), diagonal=S - L).numpy()
        assert np.array_equal(M, want), ORACLE_BUG
    return M


def padding_oracle(lengths, S):
    """M[n, j] = j < lengths[n], double loop; checked against torch.arange(S) < lengths[:, None]."""
    lengths = [int(n) for n in lengths]
    M = np.array([[j < n for j in range(S)] for n in lengths], dtype=bool).reshape(len(lengths), S)
    if torch is not None:
        want = (torch.arange(S) < torch.tensor(lengths)[:, None]).numpy()
        assert np.array_equal(M, want), ORACLE_BUG
    return M


def _torch_softmax_weights(q, k, mask, scale):
    t = torch.tensor(q) @ torch.tensor(k).transpose(-1, -2) * scale
    if mask is not None:
        t = t.masked_fill(~torch.tensor(mask), float("-inf"))
    return torch.softmax(t, dim=-1).numpy()


def sdpa_oracle(q, k, v, mask=None, scale=None):
    """softmax(q k^T * scale, masked keys removed) v, one batch index and one query at a time."""
    q, k, v = (np.asarray(a, dtype=float) for a in (q, k, v))
    batch = np.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    L, d = q.shape[-2:]
    S, dv = v.shape[-2:]
    qb = np.broadcast_to(q, batch + (L, d))
    kb = np.broadcast_to(k, batch + (S, d))
    vb = np.broadcast_to(v, batch + (S, dv))
    mb = (np.ones(batch + (L, S), dtype=bool) if mask is None
          else np.broadcast_to(np.asarray(mask, dtype=bool), batch + (L, S)))
    s = 1.0 / math.sqrt(d) if scale is None else float(scale)
    out = np.zeros(batch + (L, dv))
    weights = np.zeros(batch + (L, S))
    for idx in np.ndindex(*batch):
        for i in range(L):
            allowed = [j for j in range(S) if mb[idx + (i, j)]]
            scores = [s * float(np.dot(qb[idx + (i,)], kb[idx + (j,)])) for j in allowed]
            top = max(scores)
            e = [math.exp(x - top) for x in scores]
            total = sum(e)
            for j, ej in zip(allowed, e):
                weights[idx + (i, j)] = ej / total
                out[idx + (i,)] += ej / total * vb[idx + (j,)]
    if torch is not None:
        tq, tk, tv = (torch.tensor(np.ascontiguousarray(a)) for a in (qb, kb, vb))
        tm = None if mask is None else torch.tensor(np.ascontiguousarray(mb))
        want = F.scaled_dot_product_attention(tq, tk, tv, attn_mask=tm, dropout_p=0.0, scale=scale).numpy()
        assert np.allclose(out, want, rtol=1e-9, atol=1e-10), ORACLE_BUG
        w = _torch_softmax_weights(qb, kb, None if mask is None else mb, s)
        assert np.allclose(weights, w, rtol=1e-9, atol=1e-12), ORACLE_BUG
    return out, weights


def _torch_mha(query, key, value, params, H, mask):
    """torch.nn.MultiheadAttention with the weights of `params` (mylearn layout: W.T) and attn_mask = ~mask."""
    N, L, D = query.shape
    S = key.shape[1]
    m = torch.nn.MultiheadAttention(D, H, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        m.in_proj_weight.copy_(torch.tensor(np.concatenate([params[f"W_{c}"].T for c in "qkv"])))
        m.in_proj_bias.copy_(torch.tensor(np.concatenate([params[f"b_{c}"] for c in "qkv"])))
        m.out_proj.weight.copy_(torch.tensor(params["W_o"].T))
        m.out_proj.bias.copy_(torch.tensor(params["b_o"]))
    attn_mask = None
    if mask is not None:
        full = np.broadcast_to(np.asarray(mask, dtype=bool), (N, H, L, S)).reshape(N * H, L, S)
        attn_mask = torch.tensor(~full)                     # True = masked in nn.MultiheadAttention
    with torch.no_grad():
        out, w = m(torch.tensor(query), torch.tensor(key), torch.tensor(value), attn_mask=attn_mask,
                   need_weights=True, average_attn_weights=False)
    return out.numpy(), w.numpy()


def mha_oracle(query, key, value, params, H, mask=None):
    """Project, attend in each head h on the features h*d_h .. (h+1)*d_h - 1, concatenate, project."""
    query, key, value = (np.asarray(a, dtype=float) for a in (query, key, value))
    N, L, D = query.shape
    S = key.shape[1]
    d_h = D // H
    Q = query @ params["W_q"] + params["b_q"]
    K = key @ params["W_k"] + params["b_k"]
    V = value @ params["W_v"] + params["b_v"]
    m = None if mask is None else np.broadcast_to(np.asarray(mask, dtype=bool), (N, H, L, S))
    heads = np.zeros((N, L, D))
    weights = np.zeros((N, H, L, S))
    for h in range(H):
        cols = slice(h * d_h, (h + 1) * d_h)
        o, w = sdpa_oracle(Q[:, :, cols], K[:, :, cols], V[:, :, cols], None if m is None else m[:, h])
        heads[:, :, cols] = o
        weights[:, h] = w
    out = heads @ params["W_o"] + params["b_o"]
    if torch is not None:
        t_out, t_w = _torch_mha(query, key, value, params, H, mask)
        assert np.allclose(out, t_out, rtol=1e-9, atol=1e-10), ORACLE_BUG
        assert np.allclose(weights, t_w, rtol=1e-9, atol=1e-12), ORACLE_BUG
    return out, weights


_ERF = np.vectorize(math.erf, otypes=[float])


def _layer_norm(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _torch_block_by_hand(x, params, H, mask, eps):
    """The docstring written in PyTorch: h = x + MHA(LN1(x)), out = h + gelu(LN2(h) W_1 + b_1) W_2 + b_2."""
    t = {key: torch.tensor(value) for key, value in params.items()}
    tx = torch.tensor(x)
    N, T, D = tx.shape

    def heads(z):
        return z.view(N, T, H, D // H).transpose(1, 2)

    a = F.layer_norm(tx, (D,), t["ln1_gamma"], t["ln1_beta"], eps)
    q, k, v = (heads(a @ t[f"W_{c}"] + t[f"b_{c}"]) for c in "qkv")
    tm = None if mask is None else torch.tensor(np.ascontiguousarray(np.broadcast_to(mask, (N, H, T, T))))
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=tm, dropout_p=0.0)
    h = tx + o.transpose(1, 2).reshape(N, T, D) @ t["W_o"] + t["b_o"]
    m = F.layer_norm(h, (D,), t["ln2_gamma"], t["ln2_beta"], eps)
    return (h + F.gelu(m @ t["W_1"] + t["b_1"]) @ t["W_2"] + t["b_2"]).numpy()


def _torch_encoder_layer(x, params, H, mask, eps):
    """torch.nn.TransformerEncoderLayer(norm_first=True, activation='gelu', dropout=0) with the same weights."""
    N, T, D = x.shape
    Fd = params["W_1"].shape[1]
    layer = torch.nn.TransformerEncoderLayer(D, H, Fd, dropout=0.0, activation="gelu", batch_first=True,
                                             norm_first=True, layer_norm_eps=eps, dtype=torch.float64).eval()
    tt = torch.tensor
    with torch.no_grad():
        sa = layer.self_attn
        sa.in_proj_weight.copy_(tt(np.concatenate([params[f"W_{c}"].T for c in "qkv"])))
        sa.in_proj_bias.copy_(tt(np.concatenate([params[f"b_{c}"] for c in "qkv"])))
        sa.out_proj.weight.copy_(tt(params["W_o"].T))
        sa.out_proj.bias.copy_(tt(params["b_o"]))
        layer.linear1.weight.copy_(tt(params["W_1"].T))
        layer.linear1.bias.copy_(tt(params["b_1"]))
        layer.linear2.weight.copy_(tt(params["W_2"].T))
        layer.linear2.bias.copy_(tt(params["b_2"]))
        layer.norm1.weight.copy_(tt(params["ln1_gamma"]))
        layer.norm1.bias.copy_(tt(params["ln1_beta"]))
        layer.norm2.weight.copy_(tt(params["ln2_gamma"]))
        layer.norm2.bias.copy_(tt(params["ln2_beta"]))
        src_mask = None
        if mask is not None:
            full = np.broadcast_to(np.asarray(mask, dtype=bool), (N, H, T, T)).reshape(N * H, T, T)
            src_mask = tt(~full)
        return layer(tt(x), src_mask=src_mask).numpy()


def block_oracle(x, params, H, mask=None, eps=1e-5):
    """Pre-LN block of the docstring with the exact GELU (math.erf), the attention of mha_oracle."""
    x = np.asarray(x, dtype=float)
    a = _layer_norm(x, params["ln1_gamma"], params["ln1_beta"], eps)
    attn, _ = mha_oracle(a, a, a, params, H, mask)
    h = x + attn
    z = _layer_norm(h, params["ln2_gamma"], params["ln2_beta"], eps) @ params["W_1"] + params["b_1"]
    out = h + 0.5 * z * (1.0 + _ERF(z / math.sqrt(2.0))) @ params["W_2"] + params["b_2"]
    if torch is not None:
        assert np.allclose(out, _torch_block_by_hand(x, params, H, mask, eps), rtol=1e-9, atol=1e-10), ORACLE_BUG
        assert np.allclose(out, _torch_encoder_layer(x, params, H, mask, eps), rtol=1e-9, atol=1e-10), ORACLE_BUG
    return out


def pe_oracle(T, D, base=10000.0):
    """Implementation of the PyTorch tutorial (exp / log), checked against a double loop of the docstring formula."""
    position = np.arange(T, dtype=np.float64)[:, None]
    div_term = np.exp(np.arange(0, D, 2, dtype=np.float64) * (-math.log(base) / D))
    pe = np.zeros((T, D))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    loop = np.zeros((T, D))
    for p in range(T):
        for i in range(D // 2):
            loop[p, 2 * i] = math.sin(p / base ** (2 * i / D))
            loop[p, 2 * i + 1] = math.cos(p / base ** (2 * i / D))
    assert np.allclose(pe, loop, rtol=0, atol=1e-9), "bug in tests/test_chb3_attention.py: two PE oracles disagree"
    return pe


def rope_oracle(x, positions=None, base=10000.0):
    """Each pair (x[i], x[i + d/2]) read as the complex number x1 + i x2, multiplied by exp(i theta)."""
    x = np.asarray(x, dtype=float)
    T, d = x.shape[-2:]
    half = d // 2
    pos = np.arange(T, dtype=float) if positions is None else np.asarray(positions, dtype=float)
    out = np.empty_like(x)
    for idx in np.ndindex(*x.shape[:-2]):
        for t in range(T):
            for i in range(half):
                theta = pos[t] * base ** (-2.0 * i / d)
                z = complex(x[idx + (t, i)], x[idx + (t, i + half)]) * cmath.exp(1j * theta)
                out[idx + (t, i)] = z.real
                out[idx + (t, i + half)] = z.imag
    return out


def random_params(D, F=None, seed=0, scale=0.5):
    """MHA parameters (and the block's, if F is given), mylearn layout W (n_in, n_out)."""
    rng = np.random.default_rng(seed)
    p = {f"W_{c}": rng.normal(scale=scale, size=(D, D)) for c in "qkvo"}
    p |= {f"b_{c}": rng.normal(scale=0.3, size=D) for c in "qkvo"}
    if F is not None:
        p |= {"ln1_gamma": 1 + rng.normal(scale=0.3, size=D), "ln1_beta": rng.normal(scale=0.3, size=D),
              "ln2_gamma": 1 + rng.normal(scale=0.3, size=D), "ln2_beta": rng.normal(scale=0.3, size=D),
              "W_1": rng.normal(scale=0.5, size=(D, F)), "b_1": rng.normal(scale=0.3, size=F),
              "W_2": rng.normal(scale=0.5, size=(F, D)), "b_2": rng.normal(scale=0.3, size=D)}
    return p


def random_mask(shape, seed):
    """A boolean mask with at least one True per row (the first key of a random row is forced)."""
    rng = np.random.default_rng(seed)
    m = rng.random(shape) < 0.6
    m[..., 0] |= ~m.any(axis=-1)
    return m


# ================================================================== causal_mask (B3.12)
def test_causal_mask_docstring_examples(att):
    M = call(att.causal_mask, 3, expected="the (3, 3) mask")
    assert_array(M, (3, 3), "causal_mask(3)", kind="bool")
    assert_equal(M, [[True, False, False], [True, True, False], [True, True, True]], "causal_mask(3)")
    M = call(att.causal_mask, 2, 4, expected="the (2, 4) mask")
    assert_array(M, (2, 4), "causal_mask(2, 4)", kind="bool")
    assert_equal(M, [[True, True, True, False], [True, True, True, True]],
                 "causal_mask(2, 4) (aligned bottom-right: the last query sees every key)")


@pytest.mark.parametrize("L,S", [
    pytest.param(1, None, id="L1"),
    pytest.param(5, None, id="L5"),
    pytest.param(1, 6, id="L1-S6-one-new-token"),
    pytest.param(3, 7, id="L3-S7"),
    pytest.param(6, 6, id="L6-S6-given"),
    pytest.param(4, 5, id="L4-S5"),
])
def test_causal_mask_matches_torch_tril(att, L, S):
    M = call(att.causal_mask, L, S, expected="the mask")
    shape = (L, L if S is None else S)
    msg = f"causal_mask({L}, {S})"
    assert_array(M, shape, msg, kind="bool")
    assert_equal(M, causal_oracle(L, S), msg + " (M[i, j] = j <= i + S - L, torch.tril(..., diagonal=S - L))")


@pytest.mark.parametrize("L", [1, 4, 7])
def test_causal_mask_is_the_mask_of_is_causal_when_l_equals_s(att, L):
    M = call(att.causal_mask, L, expected="the mask")
    assert_array(M, (L, L), f"causal_mask({L})", kind="bool")
    assert_equal(M, np.tril(np.ones((L, L), dtype=bool)), f"causal_mask({L}): the lower triangle (diagonal included)")
    if torch is not None:
        rng = np.random.default_rng(L)
        q, k, v = (torch.tensor(rng.normal(size=(2, L, 4))) for _ in range(3))
        want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        got = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tensor(M))
        assert_close(got.numpy(), want.numpy(), msg=f"F.scaled_dot_product_attention with attn_mask=causal_mask({L}) "
                     "and with is_causal=True")


def test_causal_mask_with_a_kv_cache(att):
    L, S = 3, 8
    M = call(att.causal_mask, L, S, expected="the mask")
    assert_array(M, (L, S), f"causal_mask({L}, {S})", kind="bool")
    assert_equal(M[-1], np.ones(S, dtype=bool), f"causal_mask({L}, {S}): the last query sees every key")
    assert_equal(M.sum(axis=1), [S - L + 1 + i for i in range(L)],
                 f"causal_mask({L}, {S}): number of keys seen by each query (query i is the token at "
                 "position i + S - L, it sees the keys 0 .. i + S - L)")


@pytest.mark.parametrize("args,why", [
    pytest.param((0,), "q_len = 0", id="q-len-0"),
    pytest.param((-2,), "q_len = -2", id="q-len-negative"),
    pytest.param((0, 3), "q_len = 0 (with kv_len = 3)", id="q-len-0-kv-3"),
    pytest.param((3, 2), "kv_len = 2 < q_len = 3", id="kv-shorter-than-q"),
    pytest.param((2, 0), "kv_len = 0 < q_len = 2", id="kv-len-0"),
])
def test_causal_mask_refuses_invalid_lengths(att, args, why):
    assert_raises(ValueError, att.causal_mask, *args, why=why)


# ================================================================== padding_mask (B3.12)
def test_padding_mask_docstring_example(att):
    M = call(att.padding_mask, [2, 3], 4, expected="the (2, 4) mask")
    assert_array(M, (2, 4), "padding_mask([2, 3], 4)", kind="bool")
    assert_equal(M, [[True, True, False, False], [True, True, True, False]], "padding_mask([2, 3], 4)")


@pytest.mark.parametrize("lengths,S", [
    pytest.param([1], 1, id="one-token"),
    pytest.param([5, 1, 3], 5, id="list"),
    pytest.param(np.array([4, 4, 4], dtype=np.int64), 4, id="no-padding"),
    pytest.param(np.array([2, 7, 1, 5], dtype=np.int32), 9, id="int32"),
    pytest.param(np.array([1, 1]), 6, id="all-length-1"),
])
def test_padding_mask_matches_torch_arange(att, lengths, S):
    want = padding_oracle(lengths, S)
    arg = list(lengths) if isinstance(lengths, list) else np.array(lengths, copy=True)
    M = call(att.padding_mask, arg, S, expected="the mask")
    msg = f"padding_mask({_short(lengths)}, {S})"
    assert_array(M, want.shape, msg, kind="bool")
    assert_equal(M, want, msg + " (M[n, j] = j < lengths[n])")


@pytest.mark.skipif(torch is None, reason="PyTorch absent : ce test d'oracle tourne sur Colab")
def test_padding_mask_negated_is_the_key_padding_mask_of_torch(att):
    lengths, S, D = [4, 2, 6], 6, 4
    M = call(att.padding_mask, list(lengths), S, expected="the mask")
    assert_array(M, (3, S), "padding_mask([4, 2, 6], 6)", kind="bool")
    m = torch.nn.MultiheadAttention(D, 2, batch_first=True, dtype=torch.float64)
    x = torch.tensor(np.random.default_rng(0).normal(size=(3, S, D)))
    with torch.no_grad():
        _, w = m(x, x, x, key_padding_mask=torch.tensor(~M), average_attn_weights=False)
    w = w.numpy()
    zero = np.broadcast_to(~padding_oracle(lengths, S)[:, None, None, :], w.shape)
    if (w[zero] != 0).any() or (w[~zero] == 0).any():
        _fail("", "expected nn.MultiheadAttention(key_padding_mask=~padding_mask(...)) to give a weight of 0 "
                  "exactly to the padding keys (and only to them)")


def test_padding_mask_does_not_modify_lengths(att):
    lengths = np.array([3, 1, 2])
    saved = lengths.copy()
    call(att.padding_mask, lengths, 3)
    if not np.array_equal(lengths, saved):
        _fail("", "expected lengths unchanged by padding_mask")


@pytest.mark.parametrize("lengths,S,why", [
    pytest.param([2, 0], 3, "a length of 0", id="length-0"),
    pytest.param([-1, 2], 3, "a negative length", id="negative-length"),
    pytest.param([3, 5], 4, "a length 5 > max_len = 4", id="longer-than-max-len"),
    pytest.param(np.array([1, 2, 7]), 6, "a length 7 > max_len = 6 (array)", id="array-too-long"),
])
def test_padding_mask_refuses_invalid_lengths(att, lengths, S, why):
    assert_raises(ValueError, att.padding_mask, lengths, S, why=why)


# ================================================================== scaled_dot_product_attention (B3.12)
def sdpa_result(att, *args, what="scaled_dot_product_attention", **kwargs):
    """Call the learner's function on copies; return (output, weights) checked as a tuple of arrays."""
    args = tuple(np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in args)
    kwargs = {key: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for key, v in kwargs.items()}
    out, w = pair(call(att.scaled_dot_product_attention, *args, expected="the tuple (output, weights)", **kwargs),
                  what)
    return out, w


def test_scaled_dot_product_attention_docstring_examples(att):
    q = np.array([[1.0, 0.0]])
    k = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = np.array([[1.0, 2.0], [3.0, 4.0]])
    out, w = sdpa_result(att, q, k, v)
    assert_array(w, (1, 2), "weights of the docstring example")
    assert_array(out, (1, 2), "output of the docstring example")
    assert_close(w, [[0.66976155, 0.33023845]], rtol=0, atol=5e-5,
                 msg="weights of the docstring example (softmax of q k^T / sqrt(d_k))")
    assert_close(out, [[1.6604769, 2.6604769]], rtol=0, atol=5e-5, msg="output of the docstring example")
    out, _ = sdpa_result(att, q, k, v, mask=np.array([[True, False]]))
    assert_close(out, [[1.0, 2.0]], msg="output of the docstring example with mask [[True, False]] "
                                         "(only the first key is allowed: its value is returned)")


SDPA_CASES = [
    # id, q shape, k shape, v shape, mask kind
    ("2d-no-mask", (4, 6), (5, 6), (5, 3), None),
    ("2d-L-equals-S-causal", (5, 4), (5, 4), (5, 2), "causal"),
    ("2d-kv-cache-causal-bottom-right", (2, 4), (6, 4), (6, 3), "causal"),
    ("2d-random-mask", (4, 8), (7, 8), (7, 5), "random"),
    ("batch-3d", (3, 4, 4), (3, 6, 4), (3, 6, 2), None),
    ("batch-heads-4d-causal", (2, 3, 5, 4), (2, 3, 5, 4), (2, 3, 5, 6), "causal"),
    ("batch-heads-padding-N11S", (2, 3, 4, 8), (2, 3, 6, 8), (2, 3, 6, 4), "padding"),
    ("batch-heads-full-mask", (2, 2, 3, 4), (2, 2, 5, 4), (2, 2, 5, 4), "full"),
    ("broadcast-leading-axes", (2, 3, 4, 4), (1, 3, 5, 4), (3, 5, 2), "causal"),
]


def _sdpa_mask(kind, q_shape, k_shape, seed):
    L, S = q_shape[-2], k_shape[-2]
    if kind is None:
        return None
    if kind == "causal":
        return causal_oracle(L, S)
    if kind == "random":
        return random_mask((L, S), seed)
    if kind == "padding":
        lengths = [S, 2][: q_shape[0]]
        return padding_oracle(lengths, S).reshape(q_shape[0], 1, 1, S)
    return random_mask(q_shape[:-2] + (L, S), seed)


@pytest.mark.parametrize("case", SDPA_CASES, ids=[c[0] for c in SDPA_CASES])
def test_scaled_dot_product_attention_matches_torch(att, case):
    name, qs, ks, vs, kind = case
    rng = np.random.default_rng(len(name))
    q, k, v = rng.normal(size=qs), rng.normal(size=ks), rng.normal(size=vs)
    mask = _sdpa_mask(kind, qs, ks, len(name))
    want_out, want_w = sdpa_oracle(q, k, v, mask)
    out, w = sdpa_result(att, q, k, v, mask=mask)
    msg = f"scaled_dot_product_attention, case {name} (q {qs}, k {ks}, v {vs}, mask {kind})"
    assert_array(w, want_w.shape, msg + ", weights")
    assert_array(out, want_out.shape, msg + ", output")
    assert_close(w, want_w, msg=msg + ", weights (softmax over the keys of q k^T / sqrt(d_k), masked keys at -inf)")
    assert_close(out, want_out, msg=msg + ", output (weights @ v)")


@pytest.mark.parametrize("scale", [1.0, 0.25, 2.5, -1.0, 0.0])
def test_scaled_dot_product_attention_uses_the_scale_given(att, scale):
    rng = np.random.default_rng(11)
    q, k, v = rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 5, 4)), rng.normal(size=(2, 5, 3))
    mask = causal_oracle(3, 5)
    want_out, want_w = sdpa_oracle(q, k, v, mask, scale=scale)
    out, w = sdpa_result(att, q, k, v, mask=mask, scale=scale)
    msg = f"scaled_dot_product_attention(..., scale={scale})"
    if scale == 0.0:
        msg += " (0.0 is a scale: only None means 1 / sqrt(d_k), test `scale is None`)"
    assert_close(w, want_w, msg=msg + ", weights")
    assert_close(out, want_out, msg=msg + ", output")


def test_scaled_dot_product_attention_masked_keys_get_exactly_zero(att):
    rng = np.random.default_rng(5)
    q, k, v = rng.normal(size=(2, 2, 6, 4)), rng.normal(size=(2, 2, 6, 4)), rng.normal(size=(2, 2, 6, 3))
    mask = random_mask((2, 2, 6, 6), 9)
    _, w = sdpa_result(att, q, k, v, mask=mask)
    assert_array(w, (2, 2, 6, 6), "weights")
    if (w[~mask] != 0).any():
        bad = tuple(int(i) for i in np.argwhere((w != 0) & ~mask)[0])
        _fail("", f"expected a weight of exactly 0 for every masked key, got {w[bad]:.6g} at index {bad} "
                  "(False = may not attend; a score of -inf gets exp(-inf) = 0)")
    assert_close(w.sum(axis=-1), np.ones((2, 2, 6)), msg="sum of each row of weights")


def test_scaled_dot_product_attention_masked_keys_do_not_change_the_output(att):
    rng = np.random.default_rng(6)
    q, k, v = rng.normal(size=(4, 4)), rng.normal(size=(6, 4)), rng.normal(size=(6, 2))
    mask = np.array([[True, True, False, True, False, False]] * 4)
    out1, _ = sdpa_result(att, q, k, v, mask=mask)
    k2, v2 = k.copy(), v.copy()
    k2[~mask[0]] *= 50.0
    v2[~mask[0]] = 1e3
    out2, _ = sdpa_result(att, q, k2, v2, mask=mask)
    assert_close(out2, out1, msg="output after changing the keys and values of the masked positions "
                                 "(it must not change: a masked key takes no part in the result)")


def test_scaled_dot_product_attention_large_inputs_give_no_nan(att):
    rng = np.random.default_rng(8)
    q, k, v = 40 * rng.normal(size=(3, 8)), 40 * rng.normal(size=(5, 8)), rng.normal(size=(5, 2))
    mask = random_mask((3, 5), 3)
    want_out, want_w = sdpa_oracle(q, k, v, mask)
    out, w = sdpa_result(att, q, k, v, mask=mask)
    if np.isnan(np.asarray(w, dtype=float)).any() or np.isnan(np.asarray(out, dtype=float)).any():
        _fail("", "expected no NaN with scores of order 1e3, got NaN (subtract the maximum of each row before exp: "
                  "your softmax of chapter 17 does it)")
    assert_close(w, want_w, msg="weights with scores of order 1e3")
    assert_close(out, want_out, msg="output with scores of order 1e3")


def test_scaled_dot_product_attention_accepts_lists(att):
    q = [[1.0, 0.0], [0.0, 2.0]]
    k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    v = [[1.0], [2.0], [4.0]]
    mask = [[True, False, True], [True, True, True]]
    want_out, want_w = sdpa_oracle(q, k, v, mask)
    out, w = sdpa_result(att, q, k, v, mask=mask)
    assert_close(w, want_w, msg="weights for q, k, v and mask given as lists")
    assert_close(out, want_out, msg="output for q, k, v and mask given as lists")


def test_scaled_dot_product_attention_does_not_modify_its_inputs(att):
    rng = np.random.default_rng(2)
    q, k, v = rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 3, 2))
    mask = causal_oracle(3)
    saved = copies(q, k, v, mask)
    call(att.scaled_dot_product_attention, q, k, v, mask=mask, expected="the tuple (output, weights)")
    for name, arr, before in zip(("q", "k", "v", "mask"), (q, k, v, mask), saved):
        if not np.array_equal(arr, before):
            _fail("", f"expected {name} unchanged by scaled_dot_product_attention (mask the scores in a new array)")


@pytest.mark.parametrize("mask,why", [
    pytest.param(np.array([[True, True, False], [False, False, False], [True, False, True]]),
                 "row 1 of the mask is all False", id="one-row-all-false"),
    pytest.param(np.zeros((3, 3), dtype=bool), "the mask is all False", id="all-false"),
    pytest.param(np.array([True, False, True, False, False, False]).reshape(2, 1, 1, 3),
                 "the (N, 1, 1, S) mask hides every key of sequence 1",
                 id="broadcast-sequence-fully-masked"),
])
def test_scaled_dot_product_attention_refuses_a_row_without_allowed_key(att, mask, why):
    rng = np.random.default_rng(4)
    shape = (2, 2, 3, 4) if mask.ndim == 4 else (3, 4)
    q, k, v = rng.normal(size=shape), rng.normal(size=shape), rng.normal(size=shape)
    assert_raises(ValueError, att.scaled_dot_product_attention, q, k, v, mask=mask, why=why)


@pytest.mark.parametrize("qs,ks,vs,mask_shape,why", [
    pytest.param((3, 4), (5, 3), (5, 2), None, "q has d_k = 4, k has d_k = 3", id="dk-mismatch"),
    pytest.param((3, 4), (5, 4), (6, 2), None, "k has S = 5 keys, v has 6", id="S-mismatch"),
    pytest.param((3, 4), (5, 4), (5, 2), (3, 6), "mask (3, 6) does not broadcast to (3, 5)", id="mask-shape"),
    pytest.param((2, 3, 4), (3, 5, 4), (3, 5, 2), None, "batch of 2 against batch of 3", id="batch-mismatch"),
])
def test_scaled_dot_product_attention_refuses_incompatible_shapes(att, qs, ks, vs, mask_shape, why):
    rng = np.random.default_rng(0)
    mask = None if mask_shape is None else np.ones(mask_shape, dtype=bool)
    assert_raises(ValueError, att.scaled_dot_product_attention, rng.normal(size=qs), rng.normal(size=ks),
                  rng.normal(size=vs), mask=mask, why=why)


# ================================================================== split_heads (B3.14)
def split_oracle(x, H):
    """out[n, h, t, :] = x[n, t, h*d_h:(h+1)*d_h]; checked against torch view(N, T, H, d_h).transpose(1, 2)."""
    x = np.asarray(x)
    N, T, D = x.shape
    d_h = D // H
    out = np.zeros((N, H, T, d_h), dtype=x.dtype)
    for n in range(N):
        for h in range(H):
            for t in range(T):
                out[n, h, t] = x[n, t, h * d_h:(h + 1) * d_h]
    if torch is not None:
        want = torch.tensor(x).view(N, T, H, d_h).transpose(1, 2).numpy()
        assert np.array_equal(out, want), ORACLE_BUG
    return out


def test_split_heads_docstring_examples(att):
    x = np.arange(12.0).reshape(1, 2, 6)
    s = call(att.split_heads, x.copy(), 3, expected="the heads")
    assert_array(s, (1, 3, 2, 2), "split_heads(x, 3) with x of shape (1, 2, 6)", kind=None)
    assert_equal(s[0, 1], [[2.0, 3.0], [8.0, 9.0]], "split_heads(x, 3)[0, 1]: head 1 gets the features 2 and 3")


@pytest.mark.parametrize("shape,H", [
    pytest.param((1, 2, 6), 3, id="1x2x6-H3"),
    pytest.param((2, 5, 8), 2, id="2x5x8-H2"),
    pytest.param((3, 4, 12), 4, id="3x4x12-H4"),
    pytest.param((2, 3, 6), 1, id="H1"),
    pytest.param((2, 3, 6), 6, id="H-equals-D"),
    pytest.param((2, 1, 8), 4, id="one-token"),
])
def test_split_heads_matches_torch_view_transpose(att, shape, H):
    x = np.random.default_rng(sum(shape)).normal(size=shape)
    s = call(att.split_heads, x.copy(), H, expected="the heads")
    N, T, D = shape
    msg = f"split_heads(x of shape {shape}, {H})"
    assert_array(s, (N, H, T, D // H), msg, kind=None)
    assert_equal(s, split_oracle(x, H), msg + " (head h gets the features h*d_h .. (h+1)*d_h - 1 of every token: "
                 "reshape to (N, T, H, d_h), then swap the axes T and H)")


def test_split_heads_accepts_a_list(att):
    x = np.arange(16.0).reshape(1, 2, 8)
    s = call(att.split_heads, x.tolist(), 2, expected="the heads")
    assert_array(s, (1, 2, 2, 4), "split_heads(list of shape (1, 2, 8), 2)", kind=None)
    assert_equal(s, split_oracle(x, 2), "split_heads(list of shape (1, 2, 8), 2)")


def test_split_heads_does_not_modify_its_input(att):
    x = np.random.default_rng(1).normal(size=(2, 3, 4))
    saved = x.copy()
    call(att.split_heads, x, 2)
    if not np.array_equal(x, saved):
        _fail("", "expected x unchanged by split_heads")


@pytest.mark.parametrize("shape,H,why", [
    pytest.param((3, 6), 2, "x of shape (3, 6) is 2-D", id="x-2d"),
    pytest.param((1, 2, 3, 6), 2, "x of shape (1, 2, 3, 6) is 4-D", id="x-4d"),
    pytest.param((2, 3, 6), 4, "D = 6 is not a multiple of n_heads = 4", id="D-not-divisible"),
    pytest.param((2, 3, 4), 8, "D = 4 is not a multiple of n_heads = 8", id="more-heads-than-features"),
])
def test_split_heads_refuses_invalid_shapes(att, shape, H, why):
    assert_raises(ValueError, att.split_heads, np.zeros(shape), H, why=why)


# ================================================================== merge_heads (B3.14)
def merge_oracle(x):
    """Heads concatenated along the last axis, head 0 first; checked against torch transpose(1, 2).reshape."""
    x = np.asarray(x)
    N, H, T, d_h = x.shape
    out = np.zeros((N, T, H * d_h), dtype=x.dtype)
    for n in range(N):
        for t in range(T):
            out[n, t] = np.concatenate([x[n, h, t] for h in range(H)])
    if torch is not None:
        want = torch.tensor(x).transpose(1, 2).reshape(N, T, H * d_h).numpy()
        assert np.array_equal(out, want), ORACLE_BUG
    return out


def test_merge_heads_docstring_example(att):
    x = np.arange(12.0).reshape(1, 2, 6)
    heads = split_oracle(x, 3)
    m = call(att.merge_heads, heads.copy(), expected="the merged array")
    assert_array(m, (1, 2, 6), "merge_heads(heads of shape (1, 3, 2, 2))", kind=None)
    assert_equal(m, x, "merge_heads(split_heads(x, 3)) == x for x = np.arange(12.0).reshape(1, 2, 6)")


@pytest.mark.parametrize("shape", [
    pytest.param((1, 3, 2, 2), id="1x3x2x2"),
    pytest.param((2, 2, 5, 4), id="2x2x5x4"),
    pytest.param((3, 4, 3, 3), id="3x4x3x3"),
    pytest.param((2, 1, 4, 6), id="one-head"),
    pytest.param((2, 5, 1, 2), id="one-token"),
])
def test_merge_heads_matches_torch_transpose_reshape(att, shape):
    x = np.random.default_rng(sum(shape)).normal(size=shape)
    m = call(att.merge_heads, x.copy(), expected="the merged array")
    N, H, T, d_h = shape
    msg = f"merge_heads(x of shape {shape})"
    assert_array(m, (N, T, H * d_h), msg, kind=None)
    assert_equal(m, merge_oracle(x), msg + " (swap the axes H and T back, then reshape: head 0 first)")


@pytest.mark.parametrize("shape,H", [pytest.param((2, 5, 8), 2, id="2x5x8-H2"),
                                     pytest.param((3, 4, 12), 3, id="3x4x12-H3"),
                                     pytest.param((1, 6, 6), 6, id="1x6x6-H6")])
def test_merge_heads_inverts_split_heads(att, shape, H):
    x = np.random.default_rng(7).normal(size=shape)
    m = call(att.merge_heads, split_oracle(x, H), expected="the merged array")
    assert_equal(m, x, f"merge_heads applied to the heads of x of shape {shape} (H = {H}) must give x back")


def test_merge_heads_does_not_modify_its_input(att):
    x = np.random.default_rng(3).normal(size=(2, 2, 3, 4))
    saved = x.copy()
    call(att.merge_heads, x)
    if not np.array_equal(x, saved):
        _fail("", "expected x unchanged by merge_heads")


@pytest.mark.parametrize("shape,why", [
    pytest.param((2, 3, 4), "x of shape (2, 3, 4) is 3-D", id="x-3d"),
    pytest.param((1, 2, 3, 4, 5), "x of shape (1, 2, 3, 4, 5) is 5-D", id="x-5d"),
])
def test_merge_heads_refuses_invalid_shapes(att, shape, why):
    assert_raises(ValueError, att.merge_heads, np.zeros(shape), why=why)


# ================================================================== multi_head_attention (B3.14)
def mha_result(att, query, key, value, params, H, mask=None, what="multi_head_attention"):
    args = copies(query, key, value)
    kwargs = {} if mask is None else {"mask": np.array(mask, copy=True)}
    return pair(call(att.multi_head_attention, *args, copy_params(params), H, expected="the tuple (output, weights)",
                     **kwargs), what)


def test_multi_head_attention_docstring_example(att):
    rng = np.random.default_rng(0)
    D, H = 8, 2
    params = {f"W_{c}": rng.normal(size=(D, D)) for c in "qkvo"}
    params |= {f"b_{c}": np.zeros(D) for c in "qkvo"}
    x = rng.normal(size=(2, 5, D))
    out, w = mha_result(att, x, x, x, params, H, causal_oracle(5))
    assert_array(out, (2, 5, 8), "output of the docstring example")
    assert_array(w, (2, 2, 5, 5), "weights of the docstring example (N, H, L, S)")
    assert_close(w.sum(axis=-1), np.ones((2, 2, 5)), msg="sum of each row of weights")
    assert_close(w[0, 0, 0], [1.0, 0.0, 0.0, 0.0, 0.0], msg="w[0, 0, 0] of the docstring example (with causal_mask(5), "
                                                           "the first token sees only itself)")


MHA_CASES = [
    # id, N, L, S, D, H, mask kind, self-attention
    ("self-no-mask-H2", 2, 5, 5, 8, 2, None, True),
    ("self-causal-H4", 2, 6, 6, 8, 4, "causal", True),
    ("self-causal-H1", 1, 4, 4, 6, 1, "causal", True),
    ("cross-L3-S5-no-mask", 2, 3, 5, 6, 3, None, False),
    ("cross-kv-cache-causal", 2, 2, 6, 8, 2, "causal", False),
    ("padding-N11S", 3, 5, 5, 8, 2, "padding", True),
    ("causal-and-padding-N1LS", 2, 6, 6, 8, 2, "causal+padding", True),
    ("one-mask-per-head-NHLS", 2, 4, 5, 8, 4, "per-head", False),
]


def _mha_mask(kind, N, L, S, H, seed):
    if kind is None:
        return None
    if kind == "causal":
        return causal_oracle(L, S)
    lengths = [S, 2, 3][:N]
    pad = padding_oracle(lengths, S).reshape(N, 1, 1, S)
    if kind == "padding":
        return pad
    if kind == "causal+padding":
        return pad & causal_oracle(L, S)[None, None]
    return random_mask((N, H, L, S), seed)


@pytest.mark.parametrize("case", MHA_CASES, ids=[c[0] for c in MHA_CASES])
def test_multi_head_attention_matches_torch_multihead_attention(att, case):
    name, N, L, S, D, H, kind, self_attention = case
    rng = np.random.default_rng(len(name) + D)
    params = random_params(D, seed=len(name))
    query = rng.normal(size=(N, L, D))
    key = query if self_attention else rng.normal(size=(N, S, D))
    value = query if self_attention else rng.normal(size=(N, S, D))
    mask = _mha_mask(kind, N, L, S, H, len(name))
    want_out, want_w = mha_oracle(query, key, value, params, H, mask)
    out, w = mha_result(att, query, key, value, params, H, mask)
    msg = f"multi_head_attention, case {name} (N={N}, L={L}, S={S}, D={D}, H={H}, mask {kind})"
    assert_array(w, (N, H, L, S), msg + ", weights")
    assert_array(out, (N, L, D), msg + ", output")
    assert_close(w, want_w, msg=msg + ", weights of each head (Q = query @ W_q + b_q, ... then split_heads)")
    assert_close(out, want_out, msg=msg + ", output (merge_heads, then @ W_o + b_o)")


def test_multi_head_attention_ignores_other_keys_of_params(att):
    rng = np.random.default_rng(12)
    params = random_params(8, F=16, seed=3)
    x = rng.normal(size=(2, 4, 8))
    want_out, want_w = mha_oracle(x, x, x, params, 2, causal_oracle(4))
    extra = copy_params(params) | {"junk": np.zeros(3), "W_x": np.ones((2, 2))}
    out, w = pair(call(att.multi_head_attention, x.copy(), x.copy(), x.copy(), MappingProxyType(extra), 2,
                       mask=causal_oracle(4), expected="the tuple (output, weights)"),
                  "multi_head_attention with the parameters of a whole block (read-only mapping)")
    assert_close(out, want_out, msg="output with the parameters of a whole block plus other keys (they are ignored)")
    assert_close(w, want_w, msg="weights with the parameters of a whole block plus other keys")


def test_multi_head_attention_does_not_modify_its_inputs(att):
    rng = np.random.default_rng(13)
    params = random_params(6, seed=4)
    x = rng.normal(size=(2, 3, 6))
    mask = causal_oracle(3)
    saved_x, saved_mask, saved_params = x.copy(), mask.copy(), copy_params(params)
    call(att.multi_head_attention, x, x, x, params, 3, mask=mask, expected="the tuple (output, weights)")
    if not np.array_equal(x, saved_x):
        _fail("", "expected query = key = value = x unchanged by multi_head_attention")
    if not np.array_equal(mask, saved_mask):
        _fail("", "expected mask unchanged by multi_head_attention")
    if set(params) != set(saved_params) or any(not np.array_equal(params[k], saved_params[k]) for k in params):
        _fail("", "expected params unchanged by multi_head_attention (same keys, same arrays)")


@pytest.mark.parametrize("missing", MHA_KEYS)
def test_multi_head_attention_refuses_a_missing_parameter(att, missing):
    params = random_params(4, seed=1)
    del params[missing]
    x = np.random.default_rng(0).normal(size=(1, 3, 4))
    assert_raises(ValueError, att.multi_head_attention, x, x, x, params, 2,
                  why=f"params has no {missing!r} (ValueError, not KeyError)")


@pytest.mark.parametrize("q_shape,kv_shape,v_shape,H,why", [
    pytest.param((2, 3, 6), (2, 3, 6), (2, 3, 6), 4, "D = 6 is not a multiple of n_heads = 4", id="D-not-divisible"),
    pytest.param((2, 3, 6), (2, 4, 4), (2, 4, 4), 2, "key has D = 4, query has D = 6", id="key-other-D"),
    pytest.param((2, 3, 6), (2, 4, 6), (2, 5, 6), 2, "key has S = 4, value has S = 5", id="value-other-S"),
    pytest.param((3, 6), (3, 6), (3, 6), 2, "query of shape (3, 6) is 2-D, not (N, L, D)", id="query-2d"),
])
def test_multi_head_attention_refuses_incompatible_shapes(att, q_shape, kv_shape, v_shape, H, why):
    rng = np.random.default_rng(0)
    params = random_params(6, seed=2)
    assert_raises(ValueError, att.multi_head_attention, rng.normal(size=q_shape), rng.normal(size=kv_shape),
                  rng.normal(size=v_shape), params, H, why=why)


# ================================================================== positional_encoding (B3.15)
def test_positional_encoding_docstring_example(att):
    pe = call(att.positional_encoding, 3, 4, expected="the (3, 4) encoding")
    assert_array(pe, (3, 4), "positional_encoding(3, 4)")
    assert_close(pe, [[0, 1, 0, 1], [0.84147098, 0.54030231, 0.00999983, 0.99995],
                      [0.90929743, -0.41614684, 0.01999867, 0.99980001]], rtol=0, atol=5e-5,
                 msg="positional_encoding(3, 4) of the docstring")


@pytest.mark.parametrize("T,D,base", [
    pytest.param(1, 2, 10000.0, id="T1-D2"),
    pytest.param(3, 4, 10000.0, id="T3-D4"),
    pytest.param(50, 16, 10000.0, id="T50-D16"),
    pytest.param(128, 64, 10000.0, id="T128-D64"),
    pytest.param(10, 6, 100.0, id="T10-D6-base100"),
    pytest.param(7, 8, 1.0, id="T7-D8-base1"),
    pytest.param(1000, 10, 10000.0, id="T1000-D10-time-steps"),
])
def test_positional_encoding_matches_the_closed_form(att, T, D, base):
    pe = call(att.positional_encoding, T, D, base=base, expected="the encoding")
    msg = f"positional_encoding({T}, {D}, base={base})"
    assert_array(pe, (T, D), msg)
    assert_close(pe, pe_oracle(T, D, base), msg=msg + " (PE[p, 2i] = sin(p / base**(2i / d)), "
                 "PE[p, 2i + 1] = cos(p / base**(2i / d)): the same angle for the columns 2i and 2i + 1)")


@pytest.mark.parametrize("k", [1, 5, 17])
def test_positional_encoding_shift_is_a_rotation_of_each_pair(att, k):
    T, D, base = 40, 12, 10000.0
    pe = call(att.positional_encoding, T, D, base=base, expected="the encoding")
    assert_array(pe, (T, D), f"positional_encoding({T}, {D})")
    omega = base ** (-np.arange(0, D, 2) / D)
    c, s = np.cos(k * omega), np.sin(k * omega)
    sin_p, cos_p = pe[: T - k, 0::2], pe[: T - k, 1::2]
    assert_close(pe[k:, 0::2], sin_p * c + cos_p * s,
                 msg=f"sin columns of PE[p + {k}] (sin(a + b) = sin a cos b + cos a sin b, b = {k} * omega_i)")
    assert_close(pe[k:, 1::2], cos_p * c - sin_p * s,
                 msg=f"cos columns of PE[p + {k}] (cos(a + b) = cos a cos b - sin a sin b, b = {k} * omega_i)")


@pytest.mark.parametrize("args,why", [
    pytest.param((4, 5), "d_model = 5 is odd", id="d-odd"),
    pytest.param((0, 4), "seq_len = 0", id="seq-len-0"),
    pytest.param((-3, 4), "seq_len = -3", id="seq-len-negative"),
    pytest.param((4, 0), "d_model = 0", id="d-model-0"),
    pytest.param((4, -2), "d_model = -2", id="d-model-negative"),
    pytest.param((4, 4, 0.5), "base = 0.5 < 1", id="base-below-1"),
    pytest.param((4, 4, 0.0), "base = 0", id="base-0"),
])
def test_positional_encoding_refuses_invalid_arguments(att, args, why):
    assert_raises(ValueError, att.positional_encoding, *args, why=why)


# ================================================================== apply_rope (B3.15)
def test_apply_rope_docstring_examples(att):
    r = call(att.apply_rope, np.ones((3, 4)), expected="the rotated array")
    assert_array(r, (3, 4), "apply_rope(np.ones((3, 4)))")
    assert_close(r, [[1, 1, 1, 1], [-0.30116868, 0.98995017, 1.38177329, 1.00994983],
                     [-1.32544426, 0.97980067, 0.49315059, 1.01979867]], rtol=0, atol=5e-5,
                 msg="apply_rope(np.ones((3, 4))) of the docstring")
    assert_close(np.linalg.norm(r, axis=-1), [2.0, 2.0, 2.0], msg="norms of the rows of apply_rope(np.ones((3, 4)))")


ROPE_CASES = [
    # id, shape, positions, base
    ("T3-d4", (3, 4), None, 10000.0),
    ("T5-d8-positions-list", (5, 8), [0, 1, 2, 3, 4], 10000.0),
    ("batch-N2-T4-d6", (2, 4, 6), None, 10000.0),
    ("heads-N2-H3-T5-d8", (2, 3, 5, 8), None, 10000.0),
    ("offset-positions-kv-cache", (2, 3, 8), [10, 11, 12], 10000.0),
    ("arbitrary-positions", (4, 6), [0, 3, 3, 100], 10000.0),
    ("base-500", (2, 2, 6, 10), None, 500.0),
    ("d2", (4, 2), [1, 2, 3, 4], 10000.0),
]


@pytest.mark.parametrize("case", ROPE_CASES, ids=[c[0] for c in ROPE_CASES])
def test_apply_rope_matches_the_complex_rotation(att, case):
    name, shape, positions, base = case
    x = np.random.default_rng(len(name)).normal(size=shape)
    kwargs = {"base": base} if base != 10000.0 else {}
    pos_arg = None if positions is None else list(positions)
    r = call(att.apply_rope, x.copy(), pos_arg, expected="the rotated array", **kwargs)
    msg = f"apply_rope, case {name} (x {shape}, positions {positions}, base {base})"
    assert_array(r, shape, msg)
    assert_close(r, rope_oracle(x, positions, base), msg=msg + " (pair (x[i], x[i + d/2]) rotated by "
                 "position * base**(-2i / d))")


@pytest.mark.parametrize("shape,positions", [
    pytest.param((2, 2, 7, 8), [0, 1, 2, 5, 9, 20, 40], id="N2-H2-T7-d8"),
    pytest.param((1, 4, 5, 16), None, id="N1-H4-T5-d16"),
])
def test_apply_rope_matches_hugging_face_llama(att, shape, positions):
    pytest.importorskip("transformers", reason="Hugging Face transformers absent : ce test d'oracle tourne sur Colab")
    if torch is None:
        pytest.skip("PyTorch absent : ce test d'oracle tourne sur Colab")
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

    x = np.random.default_rng(shape[-1]).normal(size=shape)
    T, d = shape[-2:]
    pos = np.arange(T) if positions is None else np.asarray(positions)
    config = LlamaConfig(hidden_size=d * shape[1], num_attention_heads=shape[1], head_dim=d,
                         max_position_embeddings=128, rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
    tx = torch.tensor(x)
    cos, sin = LlamaRotaryEmbedding(config)(tx, torch.tensor(pos)[None, :])
    want, _ = apply_rotary_pos_emb(tx, tx, cos, sin)
    r = call(att.apply_rope, x.copy(), None if positions is None else list(positions), expected="the rotated array")
    assert_array(r, shape, f"apply_rope(x of shape {shape})")
    assert_close(r, want.numpy(), rtol=1e-5, atol=1e-5,
                 msg=f"apply_rope(x of shape {shape}, positions {positions}) against LlamaRotaryEmbedding + "
                     "apply_rotary_pos_emb (angles in float32 there, hence the tolerance 1e-5)")


def test_apply_rope_preserves_norms(att):
    x = np.random.default_rng(21).normal(size=(2, 3, 9, 12))
    r = call(att.apply_rope, x.copy(), expected="the rotated array")
    assert_array(r, x.shape, "apply_rope(x of shape (2, 3, 9, 12))")
    assert_close(np.linalg.norm(r, axis=-1), np.linalg.norm(x, axis=-1),
                 msg="norm of each rotated vector (a rotation keeps the norm)")


def test_apply_rope_position_zero_is_not_rotated(att):
    x = np.random.default_rng(22).normal(size=(5, 8))
    r = call(att.apply_rope, x.copy(), [0, 0, 0, 0, 0], expected="the rotated array")
    assert_close(r, x, msg="apply_rope(x, positions=[0, 0, 0, 0, 0]) (angle 0: x unchanged)")
    r_none = call(att.apply_rope, x.copy(), expected="the rotated array")
    r_range = call(att.apply_rope, x.copy(), np.arange(5), expected="the rotated array")
    assert_close(r_none, r_range, msg="apply_rope(x) and apply_rope(x, np.arange(5)) (None means arange(T))")


@pytest.mark.parametrize("m,n", [(3, 1), (7, 5), (102, 100), (2, 0)])
def test_apply_rope_dot_product_depends_only_on_the_offset(att, m, n):
    rng = np.random.default_rng(23)
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
    rq = call(att.apply_rope, q.copy(), [m], expected="the rotated query")
    rk = call(att.apply_rope, k.copy(), [n], expected="the rotated key")
    assert_array(rq, (1, 8), "apply_rope(q of shape (1, 8), [m])")
    assert_array(rk, (1, 8), "apply_rope(k of shape (1, 8), [n])")
    want = float((rope_oracle(q, [m - n]) @ k[0])[0])     # rotate q by m - n, k stays at position 0
    assert_close(float(rq[0] @ rk[0]), want, msg=f"apply_rope(q, [{m}]) . apply_rope(k, [{n}]) "
                 f"(it depends only on m - n = {m - n}: the same as apply_rope(q, [{m - n}]) . k)")


def test_apply_rope_does_not_modify_its_inputs(att):
    x = np.random.default_rng(24).normal(size=(2, 4, 6))
    positions = np.array([3, 4, 5, 6])
    saved_x, saved_p = x.copy(), positions.copy()
    call(att.apply_rope, x, positions, expected="the rotated array")
    if not np.array_equal(x, saved_x):
        _fail("", "expected x unchanged by apply_rope (build the result in a new array)")
    if not np.array_equal(positions, saved_p):
        _fail("", "expected positions unchanged by apply_rope")


@pytest.mark.parametrize("shape,positions,why", [
    pytest.param((3, 5), None, "d = 5 is odd", id="d-odd"),
    pytest.param((4, 6), [0, 1, 2], "3 positions for T = 4 vectors", id="positions-too-short"),
    pytest.param((2, 3, 6), [0, 1, 2, 3], "4 positions for T = 3 vectors", id="positions-too-long"),
    pytest.param((3, 6), [7], "1 position for T = 3 vectors (NumPy would broadcast it: check the length)",
                 id="one-position-for-three-vectors"),
])
def test_apply_rope_refuses_invalid_arguments(att, shape, positions, why):
    assert_raises(ValueError, att.apply_rope, np.ones(shape), positions, why=why)


# ================================================================== transformer_block (B3.16)
USES = " (transformer_block uses your multi_head_attention, layer_norm and gelu: their tests come first)"


def block_result(att, x, params, H, mask=None, what="transformer_block", **kwargs):
    if mask is not None:
        kwargs["mask"] = np.array(mask, copy=True)
    out = call(att.transformer_block, np.array(x, copy=True), copy_params(params), H, expected="the output",
               **kwargs)
    assert_array(out, np.shape(x), what)
    return out


def test_transformer_block_docstring_example(att):
    rng = np.random.default_rng(0)
    D, Fd = 8, 32
    params = {f"W_{c}": np.zeros((D, D)) for c in "qkvo"}
    params |= {f"b_{c}": np.zeros(D) for c in "qkvo"}
    params |= {"ln1_gamma": np.ones(D), "ln1_beta": np.zeros(D), "ln2_gamma": np.ones(D), "ln2_beta": np.zeros(D),
               "W_1": np.zeros((D, Fd)), "b_1": np.zeros(Fd), "W_2": np.zeros((Fd, D)), "b_2": np.zeros(D)}
    x = rng.normal(size=(2, 5, D))
    out = block_result(att, x, params, 2, causal_oracle(5), what="transformer_block of the docstring example")
    assert_close(out, x, msg="transformer_block with all weights and biases at zero (both residual branches give 0, "
                             "so the output is x)")


BLOCK_CASES = [
    # id, N, T, D, F, H, mask kind, eps
    ("causal-H2", 2, 5, 8, 16, 2, "causal", 1e-5),
    ("no-mask-H4", 2, 4, 8, 12, 4, None, 1e-5),
    ("causal-H1-F-smaller-than-D", 1, 6, 6, 4, 1, "causal", 1e-5),
    ("padding-N11S", 3, 5, 8, 16, 2, "padding", 1e-5),
    ("causal-and-padding", 2, 6, 8, 16, 2, "causal+padding", 1e-5),
    ("causal-eps-0.5", 2, 5, 8, 16, 2, "causal", 0.5),
]


@pytest.mark.parametrize("case", BLOCK_CASES, ids=[c[0] for c in BLOCK_CASES])
def test_transformer_block_matches_torch_encoder_layer(att, case):
    name, N, T, D, Fd, H, kind, eps = case
    rng = np.random.default_rng(len(name))
    params = random_params(D, F=Fd, seed=len(name) + 1)
    x = rng.normal(size=(N, T, D))
    mask = _mha_mask(kind, N, T, T, H, len(name))
    want = block_oracle(x, params, H, mask, eps)
    kwargs = {} if eps == 1e-5 else {"eps": eps}
    out = block_result(att, x, params, H, mask, what=f"transformer_block, case {name}", **kwargs)
    assert_close(out, want, msg=f"transformer_block, case {name} (N={N}, T={T}, D={D}, F={Fd}, H={H}, mask {kind}, "
                                f"eps={eps}): h = x + MHA(LN1(x)), out = h + gelu(LN2(h) @ W_1 + b_1) @ W_2 + b_2"
                                + USES)


def test_transformer_block_attention_sublayer(att):
    """With W_1, b_1, W_2, b_2 at zero, the block is x + MHA(LN1(x))."""
    rng = np.random.default_rng(31)
    params = random_params(8, F=16, seed=31)
    for key in ("W_1", "b_1", "W_2", "b_2"):
        params[key] = np.zeros_like(params[key])
    x = rng.normal(size=(2, 5, 8))
    mask = causal_oracle(5)
    out = block_result(att, x, params, 2, mask, what="transformer_block with a zero feed-forward")
    assert_close(out, block_oracle(x, params, 2, mask),
                 msg="transformer_block with W_1, b_1, W_2, b_2 at zero: expected x + MHA(LN1(x)) (pre-LN: the "
                     "LayerNorm is applied to the input of the attention, the residual adds x itself)" + USES)


def test_transformer_block_feed_forward_sublayer(att):
    """With the value and output projections at zero, the attention gives 0 and the block is x + FFN(LN2(x))."""
    rng = np.random.default_rng(32)
    params = random_params(8, F=16, seed=32)
    for key in ("W_v", "b_v", "W_o", "b_o"):
        params[key] = np.zeros_like(params[key])
    x = rng.normal(size=(2, 4, 8))
    out = block_result(att, x, params, 2, causal_oracle(4), what="transformer_block with a zero attention")
    assert_close(out, block_oracle(x, params, 2, causal_oracle(4)),
                 msg="transformer_block with W_v, b_v, W_o, b_o at zero: expected x + gelu(LN2(x) @ W_1 + b_1) @ W_2 "
                     "+ b_2 (exact GELU with erf, LN2 with ln2_gamma and ln2_beta)" + USES)


def test_transformer_block_uses_eps(att):
    rng = np.random.default_rng(33)
    params = random_params(6, F=10, seed=33)
    x = 0.3 * rng.normal(size=(1, 4, 6))
    out = block_result(att, x, params, 3, causal_oracle(4), what="transformer_block(..., eps=0.8)", eps=0.8)
    assert_close(out, block_oracle(x, params, 3, causal_oracle(4), eps=0.8),
                 msg="transformer_block(..., eps=0.8): eps goes to both LayerNorms" + USES)


def test_transformer_block_does_not_modify_its_inputs(att):
    rng = np.random.default_rng(34)
    params = random_params(8, F=16, seed=34)
    x = rng.normal(size=(2, 3, 8))
    mask = causal_oracle(3)
    saved_x, saved_mask, saved_params = x.copy(), mask.copy(), copy_params(params)
    call(att.transformer_block, x, params, 2, mask=mask, expected="the output")
    if not np.array_equal(x, saved_x):
        _fail("", "expected x unchanged by transformer_block (h = x + ... is a new array; x += ... modifies x)")
    if not np.array_equal(mask, saved_mask):
        _fail("", "expected mask unchanged by transformer_block")
    if set(params) != set(saved_params) or any(not np.array_equal(params[k], saved_params[k]) for k in params):
        _fail("", "expected params unchanged by transformer_block (same keys, same arrays)")


@pytest.mark.parametrize("missing", BLOCK_KEYS + ("W_q",))
def test_transformer_block_refuses_a_missing_parameter(att, missing):
    params = random_params(4, F=8, seed=5)
    del params[missing]
    x = np.random.default_rng(0).normal(size=(1, 3, 4))
    assert_raises(ValueError, att.transformer_block, x, params, 2,
                  why=f"params has no {missing!r} (ValueError, not KeyError)")


@pytest.mark.parametrize("change,x_shape,why", [
    pytest.param({"W_2": (9, 4)}, (1, 3, 4), "W_1 is (4, 8) but W_2 is (9, 4)", id="W2-other-F"),
    pytest.param({"b_1": (9,)}, (1, 3, 4), "b_1 of shape (9,) with F = 8", id="b1-shape"),
    pytest.param({"ln1_gamma": (5,)}, (1, 3, 4), "ln1_gamma of shape (5,) with D = 4", id="ln1-gamma-shape"),
    pytest.param({}, (3, 4), "x of shape (3, 4) is 2-D, not (N, T, D)", id="x-2d"),
    pytest.param({}, (1, 3, 6), "x has D = 6, the parameters D = 4", id="x-other-D"),
])
def test_transformer_block_refuses_incompatible_shapes(att, change, x_shape, why):
    params = random_params(4, F=8, seed=6)
    for key, shape in change.items():
        params[key] = np.ones(shape)
    x = np.random.default_rng(0).normal(size=x_shape)
    assert_raises(ValueError, att.transformer_block, x, params, 2, why=why)
