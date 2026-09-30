"""Attention — mylearn, chapter B3 (Attention and Transformers: a mini-GPT from scratch).

Scaled dot-product attention, masks, multi-head attention, position encodings (sinusoidal
and rotary) and a full pre-LN Transformer block, in NumPy. Conventions: dense layers as
everywhere in mylearn (``W`` of shape ``(n_in, n_out)``, ``z = x @ W + b``), and boolean
masks as in ``torch.nn.functional.scaled_dot_product_attention`` (True = may attend).
``positional_encoding`` is reused in chapter B5 to encode the diffusion time step.

Complete every function or method that raises ``NotImplementedError``.
Check your work with ``python -m pytest tests/ -q`` (tests compare your code
with trusted libraries: NumPy, PyTorch and Hugging Face transformers).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

# Your own functions from chapter 17 (softmax, gelu) and chapter 20 (layer_norm).
from .nn.activations import gelu, softmax
from .nn.regularization import layer_norm


def causal_mask(q_len: int, kv_len: int | None = None) -> np.ndarray:
    """Boolean causal mask: True where query ``i`` may attend to key ``j``.

    The mask is aligned bottom-right, ``M[i, j] = (j <= i + S - L)``: the last query sees
    every key. When ``L < S`` (the ``S - L`` first keys come from a KV cache), query ``i``
    is the token at position ``i + S - L``.

    Parameters
    ----------
    q_len : int
        Number of queries ``L``.
    kv_len : int or None, default=None
        Number of keys ``S`` (``L`` if None).

    Returns
    -------
    np.ndarray of shape (L, S)
        Boolean mask.

    Raises
    ------
    ValueError
        If ``q_len < 1`` or ``kv_len < q_len``.

    Notes
    -----
    Tested against ``torch.tril(torch.ones(L, S, dtype=torch.bool), diagonal=S - L)``;
    for ``L == S``, the mask applied by
    ``F.scaled_dot_product_attention(..., is_causal=True)``.

    Examples
    --------
    >>> causal_mask(3)
    array([[ True, False, False],
           [ True,  True, False],
           [ True,  True,  True]])
    >>> causal_mask(2, 4)
    array([[ True,  True,  True, False],
           [ True,  True,  True,  True]])
    """
    # TODO: compare a column of query indices with a row of key indices (broadcasting).
    raise NotImplementedError("causal_mask() is not implemented yet")


def padding_mask(lengths: ArrayLike, max_len: int) -> np.ndarray:
    """Key mask that is True on real tokens and False on padding.

    For a batch of sequences padded to the same length ``S``: ``M[n, j] = j < lengths[n]``.
    Reshape it to ``(N, 1, 1, S)`` to broadcast it over the heads and the queries.

    Parameters
    ----------
    lengths : array-like of shape (N,)
        True length of each sequence.
    max_len : int
        Padded length ``S``.

    Returns
    -------
    np.ndarray of shape (N, S)
        Boolean mask.

    Raises
    ------
    ValueError
        If a length is < 1 or > ``max_len``.

    Notes
    -----
    Tested against ``torch.arange(max_len) < lengths[:, None]``, the logical negation of
    the ``key_padding_mask`` of ``torch.nn.MultiheadAttention`` (True = ignored there).

    Examples
    --------
    >>> padding_mask([2, 3], 4)
    array([[ True,  True, False, False],
           [ True,  True,  True, False]])
    """
    # TODO: one comparison with broadcasting.
    raise NotImplementedError("padding_mask() is not implemented yet")


def scaled_dot_product_attention(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    mask: ArrayLike | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Attention ``softmax(q k^T * scale + M) v`` over the last two axes.

    ``M`` is 0 where ``mask`` is True and -inf where it is False, so a masked key gets a
    weight of exactly 0. Any leading axes (batch, heads) are kept and broadcast.

    Parameters
    ----------
    q : array-like of shape (..., L, d_k)
        Queries.
    k : array-like of shape (..., S, d_k)
        Keys.
    v : array-like of shape (..., S, d_v)
        Values.
    mask : array-like of bool, broadcastable to (..., L, S), or None, default=None
        True = the query may attend to the key. None = no mask.
    scale : float or None, default=None
        Factor applied to the scores; ``1 / sqrt(d_k)`` if None.

    Returns
    -------
    output : np.ndarray of shape (..., L, d_v)
        Weighted averages of the values.
    weights : np.ndarray of shape (..., L, S)
        Attention weights; each row sums to 1.

    Raises
    ------
    ValueError
        If the shapes are incompatible, or if a query row has no allowed key.

    Notes
    -----
    Tested against ``torch.nn.functional.scaled_dot_product_attention`` (same boolean
    convention, ``dropout_p=0``); the weights against an explicit torch softmax.
    Reuses your ``softmax`` (chapter 17): subtracting the maximum first gives a weight of
    exactly 0 to a score of -inf.

    Examples
    --------
    >>> q = np.array([[1., 0.]])
    >>> k = np.array([[1., 0.], [0., 1.]])
    >>> v = np.array([[1., 2.], [3., 4.]])
    >>> out, w = scaled_dot_product_attention(q, k, v)
    >>> np.round(w, 4)
    array([[0.6698, 0.3302]])
    >>> np.round(out, 4)
    array([[1.6605, 2.6605]])
    >>> scaled_dot_product_attention(q, k, v, mask=np.array([[True, False]]))[0]
    array([[1., 2.]])
    """
    # TODO: scores, masking with -inf, softmax over the keys, weighted sum of the values.
    raise NotImplementedError("scaled_dot_product_attention() is not implemented yet")


def split_heads(x: ArrayLike, n_heads: int) -> np.ndarray:
    """Split the last axis into heads: ``(N, T, D) -> (N, H, T, D // H)``.

    Head ``h`` receives the features ``h * d_h`` to ``(h + 1) * d_h - 1`` of each token
    (``d_h = D // H``): reshape to ``(N, T, H, d_h)``, then swap the time and head axes.

    Parameters
    ----------
    x : array-like of shape (N, T, D)
        Batch of sequences of vectors.
    n_heads : int
        Number of heads ``H``; must divide ``D``.

    Returns
    -------
    np.ndarray of shape (N, H, T, D // H)
        The heads.

    Raises
    ------
    ValueError
        If ``x`` is not 3-D or ``D % n_heads != 0``.

    Notes
    -----
    Tested against ``x.view(N, T, H, D // H).transpose(1, 2)`` in PyTorch.

    Examples
    --------
    >>> x = np.arange(12.0).reshape(1, 2, 6)
    >>> split_heads(x, 3).shape
    (1, 3, 2, 2)
    >>> split_heads(x, 3)[0, 1]  # head 1: features 2 and 3 of both tokens
    array([[2., 3.],
           [8., 9.]])
    """
    # TODO: one reshape, one transpose.
    raise NotImplementedError("split_heads() is not implemented yet")


def merge_heads(x: ArrayLike) -> np.ndarray:
    """Inverse of ``split_heads``: ``(N, H, T, d_h) -> (N, T, H * d_h)``.

    Parameters
    ----------
    x : array-like of shape (N, H, T, d_h)
        The heads.

    Returns
    -------
    np.ndarray of shape (N, T, H * d_h)
        Heads concatenated along the last axis, head 0 first.

    Raises
    ------
    ValueError
        If ``x`` is not 4-D.

    Notes
    -----
    Tested with the property ``merge_heads(split_heads(x, H)) == x`` and against
    ``x.transpose(1, 2).reshape(N, T, H * d_h)`` in PyTorch.

    Examples
    --------
    >>> x = np.arange(12.0).reshape(1, 2, 6)
    >>> np.array_equal(merge_heads(split_heads(x, 3)), x)
    True
    """
    # TODO: undo split_heads in the reverse order.
    raise NotImplementedError("merge_heads() is not implemented yet")


def multi_head_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    params: Mapping[str, np.ndarray],
    n_heads: int,
    mask: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-head attention: project, split into heads, attend, merge, project.

    ``Q = query @ W_q + b_q`` (same for K and V), ``split_heads``, scaled dot-product
    attention in each head, ``merge_heads``, then ``output = merged @ W_o + b_o``.
    Self-attention is ``query = key = value = x``.

    Parameters
    ----------
    query : array-like of shape (N, L, D)
        Sequences that ask.
    key : array-like of shape (N, S, D)
        Sequences that are looked at.
    value : array-like of shape (N, S, D)
        Content returned by the attention.
    params : mapping of str to np.ndarray
        ``W_q``, ``W_k``, ``W_v``, ``W_o`` of shape (D, D) and ``b_q``, ``b_k``, ``b_v``,
        ``b_o`` of shape (D,). Other keys are ignored (so the parameters of a whole
        Transformer block can be passed).
    n_heads : int
        Number of heads ``H``; must divide ``D``.
    mask : array-like of bool, broadcastable to (N, H, L, S), or None, default=None
        True = may attend (e.g. ``causal_mask(L)``, or ``padding_mask(...)`` reshaped to
        ``(N, 1, 1, S)``).

    Returns
    -------
    output : np.ndarray of shape (N, L, D)
        Result of the output projection.
    weights : np.ndarray of shape (N, H, L, S)
        Attention weights of each head.

    Raises
    ------
    ValueError
        If a parameter is missing or the shapes are incompatible.

    Notes
    -----
    Tested against ``torch.nn.MultiheadAttention(D, H, batch_first=True)`` with
    ``in_proj_weight = cat(W_q.T, W_k.T, W_v.T)``, ``in_proj_bias = cat(b_q, b_k, b_v)``,
    ``out_proj = (W_o.T, b_o)``, ``average_attn_weights=False`` and ``attn_mask = ~mask``
    (careful: in ``nn.MultiheadAttention``, True means *masked*).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> D, H = 8, 2
    >>> params = {f"W_{c}": rng.normal(size=(D, D)) for c in "qkvo"}
    >>> params |= {f"b_{c}": np.zeros(D) for c in "qkvo"}
    >>> x = rng.normal(size=(2, 5, D))
    >>> out, w = multi_head_attention(x, x, x, params, n_heads=H, mask=causal_mask(5))
    >>> out.shape, w.shape
    ((2, 5, 8), (2, 2, 5, 5))
    >>> np.allclose(w.sum(axis=-1), 1.0), w[0, 0, 0]  # the first token sees only itself
    (True, array([1., 0., 0., 0., 0.]))
    """
    # TODO: three projections, split_heads, scaled_dot_product_attention, merge_heads,
    # output projection.
    raise NotImplementedError("multi_head_attention() is not implemented yet")


def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Sinusoidal position encoding of Vaswani et al. (2017).

    ``PE[p, 2i] = sin(p / base**(2i / d))`` and ``PE[p, 2i + 1] = cos(p / base**(2i / d))``
    for position ``p`` and ``i = 0 .. d/2 - 1``.

    Parameters
    ----------
    seq_len : int
        Number of positions ``T``.
    d_model : int
        Dimension ``D``; must be even.
    base : float, default=10000.0
        Wavelength scale.

    Returns
    -------
    np.ndarray of shape (T, D)
        Float array with values in [-1, 1].

    Raises
    ------
    ValueError
        If ``d_model`` is odd, or if ``seq_len``, ``d_model`` or ``base`` is < 1.

    Notes
    -----
    Tested against the closed-form formula in float64 (implementation of the PyTorch
    tutorial). Property: shifting every position by ``k`` applies the same rotation to
    each (sin, cos) pair of columns.

    Examples
    --------
    >>> np.round(positional_encoding(3, 4), 4)
    array([[ 0.    ,  1.    ,  0.    ,  1.    ],
           [ 0.8415,  0.5403,  0.01  ,  1.    ],
           [ 0.9093, -0.4161,  0.02  ,  0.9998]])
    """
    # TODO: an angle matrix (positions x frequencies), then sin in the even columns and
    # cos in the odd ones.
    raise NotImplementedError("positional_encoding() is not implemented yet")


def apply_rope(
    x: ArrayLike, positions: ArrayLike | None = None, base: float = 10000.0
) -> np.ndarray:
    """Rotary position embedding (RoPE), "rotate half" convention of Hugging Face Llama.

    Dimension ``i`` is paired with dimension ``i + d/2`` (``i < d/2``), and each pair
    ``(x1, x2)`` is rotated by the angle ``theta = position * base**(-2i / d)``:
    ``(x1 cos theta - x2 sin theta, x2 cos theta + x1 sin theta)``.

    Parameters
    ----------
    x : array-like of shape (..., T, d)
        Queries or keys (one vector per position), ``d`` even.
    positions : array-like of shape (T,) or None, default=None
        Position of each of the ``T`` vectors; ``np.arange(T)`` if None.
    base : float, default=10000.0
        Frequency base (10 000 in RoFormer and Llama 2).

    Returns
    -------
    np.ndarray
        Rotated array with the shape of ``x``.

    Raises
    ------
    ValueError
        If ``d`` is odd, or if ``positions`` does not have length ``T``.

    Notes
    -----
    Tested against ``LlamaRotaryEmbedding`` and ``apply_rotary_pos_emb`` of
    ``transformers.models.llama.modeling_llama`` (which compute the angles in float32,
    hence a tolerance of about 1e-6). Properties: norms are preserved, and
    ``apply_rope(q, [m]) . apply_rope(k, [n])`` depends only on ``m - n``.

    Examples
    --------
    >>> r = apply_rope(np.ones((3, 4)))
    >>> np.round(r, 4)  # position 0 is not rotated
    array([[ 1.    ,  1.    ,  1.    ,  1.    ],
           [-0.3012,  0.99  ,  1.3818,  1.0099],
           [-1.3254,  0.9798,  0.4932,  1.0198]])
    >>> np.round(np.linalg.norm(r, axis=-1), 4)
    array([2., 2., 2.])
    """
    # TODO: angles (T, d/2), then combine the two halves of x with cos and sin.
    raise NotImplementedError("apply_rope() is not implemented yet")


def transformer_block(
    x: ArrayLike,
    params: Mapping[str, np.ndarray],
    n_heads: int,
    mask: ArrayLike | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Pre-LN Transformer block (as in GPT-2), without dropout.

    ``h = x + MHA(LN1(x))`` (self-attention), then
    ``out = h + gelu(LN2(h) @ W_1 + b_1) @ W_2 + b_2`` with the exact GELU.

    Parameters
    ----------
    x : array-like of shape (N, T, D)
        Input sequences.
    params : mapping of str to np.ndarray
        The ``multi_head_attention`` parameters, plus ``ln1_gamma``, ``ln1_beta``,
        ``ln2_gamma``, ``ln2_beta`` of shape (D,), ``W_1`` (D, F), ``b_1`` (F,),
        ``W_2`` (F, D) and ``b_2`` (D,).
    n_heads : int
        Number of heads ``H``.
    mask : array-like of bool or None, default=None
        As in ``multi_head_attention`` (``causal_mask(T)`` for a decoder such as GPT).
    eps : float, default=1e-5
        Epsilon of both LayerNorms.

    Returns
    -------
    np.ndarray of shape (N, T, D)
        Output of the block.

    Raises
    ------
    ValueError
        If a parameter is missing or the shapes are incompatible.

    Notes
    -----
    Tested against ``torch.nn.TransformerEncoderLayer(D, H, F, dropout=0.0,
    activation='gelu', batch_first=True, norm_first=True)`` in eval mode with the same
    weights (mask passed as ``~mask``). Reuses your ``layer_norm`` (chapter 20) and
    ``gelu`` (chapter 17).

    Examples
    --------
    With all weights and biases at zero, both residual branches output 0:

    >>> rng = np.random.default_rng(0)
    >>> D, F = 8, 32
    >>> params = {f"W_{c}": np.zeros((D, D)) for c in "qkvo"}
    >>> params |= {f"b_{c}": np.zeros(D) for c in "qkvo"}
    >>> params |= {"ln1_gamma": np.ones(D), "ln1_beta": np.zeros(D),
    ...            "ln2_gamma": np.ones(D), "ln2_beta": np.zeros(D),
    ...            "W_1": np.zeros((D, F)), "b_1": np.zeros(F),
    ...            "W_2": np.zeros((F, D)), "b_2": np.zeros(D)}
    >>> x = rng.normal(size=(2, 5, D))
    >>> out = transformer_block(x, params, n_heads=2, mask=causal_mask(5))
    >>> out.shape, np.allclose(out, x)
    ((2, 5, 8), True)
    """
    # TODO: two residual sub-layers, each starting with its own layer_norm.
    raise NotImplementedError("transformer_block() is not implemented yet")
