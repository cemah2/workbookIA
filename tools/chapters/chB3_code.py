"""Code cells of the chapter B3 notebook, first half (parts A and B: B3.12 to B3.20), shared by build_chB3.py and
history_chB3.py (used by Claude).

Each constant is the source of one cell: TOOLS_* (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
"""

TOOLS_A = r'''# Tools for the notebook exercises of part A (your NumPy library attention.py) and of the bug hunt B3.17
import copy
import math
import time
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from wb import gpt as wbgpt
from wb.nbtools import StopCheck, checked, filled, fr, pair, refuses, returned, run_pytest_group, same, verdict, written

TEST_ATT = "tests/test_chb3_attention.py"      # the tests of your library attention.py
DATA = wbgpt.char_data("holmes")              # Holmes in character ids: the frozen split of wb.text.train_val


def causal_np(L, S=None):
    """The causal mask of the checks, aligned bottom-right (True = may attend), written once without your code."""
    S = L if S is None else S
    return np.tril(np.ones((L, S), dtype=bool), k=S - L)


def params_from_mha(mha):
    """The parameters of an nn.MultiheadAttention in the convention of multi_head_attention (z = x @ W + b,
    float64): PyTorch stores W_q, W_k and W_v transposed and stacked in in_proj_weight, (3D, D)."""
    D = mha.embed_dim
    W = mha.in_proj_weight.detach().double().numpy()
    b = mha.in_proj_bias.detach().double().numpy()
    return {"W_q": W[:D].T.copy(), "W_k": W[D:2 * D].T.copy(), "W_v": W[2 * D:].T.copy(),
            "W_o": mha.out_proj.weight.detach().double().numpy().T.copy(),
            "b_q": b[:D].copy(), "b_k": b[D:2 * D].copy(), "b_v": b[2 * D:].copy(),
            "b_o": mha.out_proj.bias.detach().double().numpy().copy()}


def block_params_from_torch(layer):
    """The parameters of an nn.TransformerEncoderLayer in the convention of transformer_block (float64)."""
    t = lambda w: w.detach().double().numpy().T.copy()        # noqa: E731  a dense layer: W = weight.T
    v = lambda b: b.detach().double().numpy().copy()          # noqa: E731
    return params_from_mha(layer.self_attn) | {
        "ln1_gamma": v(layer.norm1.weight), "ln1_beta": v(layer.norm1.bias),
        "ln2_gamma": v(layer.norm2.weight), "ln2_beta": v(layer.norm2.bias),
        "W_1": t(layer.linear1.weight), "b_1": v(layer.linear1.bias),
        "W_2": t(layer.linear2.weight), "b_2": v(layer.linear2.bias)}


print(f"Holmes : {len(DATA.train_text):_} caractères d'entraînement, {len(DATA.val_text):_} de validation, "
      f"{len(DATA.itos)} caractères distincts.".replace("_", " "))'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="B3")   # reload your saved file\n'

# --- 🔨 B3.12 -----------------------------------------------------------------------------------------------------------
CHECK_12 = r'''with checked("B3.12"):
    A = mylearn.attention
    m_12 = A.causal_mask(3)
    if returned("B3.12", "causal_mask", m_12):
        arr_12 = np.asarray(m_12)
        verdict("B3.12", arr_12.dtype == bool and np.array_equal(arr_12, causal_np(3)),
                "causal_mask(3) : le triangle inférieur, diagonale comprise, en booléens.",
                f"causal_mask(3) doit rendre le triangle inférieur, diagonale comprise, de dtype bool ; reçu "
                f"{arr_12.astype(int).tolist()} de dtype {arr_12.dtype}.")
    for L_12, S_12 in ((2, 4), (5, 5), (3, 8)):
        want_12 = torch.tril(torch.ones(L_12, S_12, dtype=torch.bool), diagonal=S_12 - L_12).numpy()
        got_12 = np.asarray(A.causal_mask(L_12, S_12))
        verdict("B3.12", got_12.shape == want_12.shape and np.array_equal(got_12, want_12),
                f"causal_mask({L_12}, {S_12}) = torch.tril(…, diagonal={S_12 - L_12}) : aligné en bas à droite.",
                f"causal_mask({L_12}, {S_12}) doit valoir torch.tril(torch.ones({L_12}, {S_12}), diagonal={S_12 - L_12}) "
                f"(la dernière requête voit toutes les clés) ; reçu {got_12.astype(int).tolist()}.")
    refuses("B3.12", "causal_mask(3, 2) (moins de clés que de requêtes)", lambda: A.causal_mask(3, 2))
    pad_12 = np.asarray(A.padding_mask([2, 3], 4))
    verdict("B3.12", pad_12.dtype == bool and np.array_equal(pad_12, [[True, True, False, False], [True, True, True, False]]),
            "padding_mask([2, 3], 4) : True sur les vrais tokens, False sur le padding.",
            f"padding_mask([2, 3], 4) doit rendre [[T, T, F, F], [T, T, T, F]] de dtype bool ; reçu {pad_12.astype(int).tolist()}.")
    refuses("B3.12", "padding_mask([0, 2], 4) (une longueur nulle)", lambda: A.padding_mask([0, 2], 4))
    refuses("B3.12", "padding_mask([5], 4) (plus long que max_len)", lambda: A.padding_mask([5], 4))

    # the attention by hand of the fiche (§203.2): one query, four keys, d_k = 4
    q_f = [[1.0, 0.0, 1.0, 0.0]]                                         # a list: accepted
    k_f = np.array([[1.0, 1, 0, 0], [0, 1, 1, 0], [2, 0, 2, 0], [0, 0, 0, 2]])
    v_f = np.array([[1.0, 0], [0, 1], [2, 2], [-1, 1]])
    k_copy_12 = k_f.copy()
    scores_f = np.array(q_f) @ k_f.T / 2.0
    w_want_12 = np.exp(scores_f) / np.exp(scores_f).sum()
    out_f, w_f = pair("B3.12", "scaled_dot_product_attention", A.scaled_dot_product_attention(q_f, k_f, v_f),
                      names=("output", "weights"))
    same("B3.12", "poids de l'exemple de la fiche (§203.2)", w_f, w_want_12, rtol=1e-6, atol=1e-6)
    same("B3.12", "sortie de l'exemple de la fiche", out_f, w_want_12 @ v_f, rtol=1e-6, atol=1e-6)
    verdict("B3.12", np.array_equal(k_f, k_copy_12), "les clés reçues n'ont pas changé.",
            "scaled_dot_product_attention a modifié les clés reçues.")
    _, w_c = A.scaled_dot_product_attention(q_f, k_f, v_f, mask=[[True, True, True, False]])
    verdict("B3.12", np.asarray(w_c)[0, 3] == 0 and np.isclose(np.asarray(w_c).sum(), 1),
            "une clé masquée reçoit un poids de 0 exactement, et la ligne somme encore à 1.",
            f"avec la clé 3 masquée, attendu un poids de 0 exactement et une ligne qui somme à 1 ; reçu {np.round(w_c, 4).tolist()}.")

    # a batch with heads and a padding mask (N, 1, 1, S), against F.scaled_dot_product_attention (float64)
    rng_12 = np.random.default_rng(12)
    q_12, k_12, v_12 = rng_12.normal(size=(2, 3, 5, 8)), rng_12.normal(size=(2, 3, 7, 8)), rng_12.normal(size=(2, 3, 7, 4))
    keys_12 = (np.arange(7) < np.array([7, 4])[:, None])[:, None, None, :]          # sequence 1 has 4 real keys
    want_12 = F.scaled_dot_product_attention(*(torch.from_numpy(a) for a in (q_12, k_12, v_12)),
                                             attn_mask=torch.from_numpy(keys_12)).numpy()
    out_12, w_12 = pair("B3.12", "scaled_dot_product_attention", A.scaled_dot_product_attention(q_12, k_12, v_12, mask=keys_12),
                        names=("output", "weights"))
    same("B3.12", "sortie (2, 3, 5, 4) avec un masque de padding (2, 1, 1, 7), contre F.scaled_dot_product_attention",
         out_12, want_12, rtol=1e-6, atol=1e-6)
    out_c12, _ = A.scaled_dot_product_attention(q_12[..., :5, :], k_12[..., :5, :], v_12[..., :5, :], mask=causal_np(5))
    want_c12 = F.scaled_dot_product_attention(*(torch.from_numpy(a[..., :5, :]) for a in (q_12, k_12, v_12)), is_causal=True).numpy()
    same("B3.12", "masque causal (5, 5) contre is_causal=True", out_c12, want_c12, rtol=1e-6, atol=1e-6)
    _, w0_12 = A.scaled_dot_product_attention(q_12, k_12, v_12, scale=0.0)
    verdict("B3.12", np.allclose(w0_12, 1 / 7),
            "scale=0.0 : des scores nuls, donc des poids uniformes (seul None donne 1/√d_k).",
            "scale=0.0 doit s'appliquer tel quel (des scores nuls, des poids uniformes de 1/7) : seul None donne 1/√d_k.")
    empty_12 = keys_12.copy()
    empty_12[1] = False                                                  # sequence 1 entirely masked
    refuses("B3.12", "un masque (2, 1, 1, 7) qui n'autorise aucune clé à la séquence 1",
            lambda: A.scaled_dot_product_attention(q_12, k_12, v_12, mask=empty_12))'''

# --- 🔮 B3.13 ----------------------------------------------------------------------------------------------------------
GIVEN_13 = r'''D_13, QUERIES_13 = 64, 200                                           # d_k = 64, 200 queries
KEYS_13 = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000)       # numbers of keys of the experiment


def copied_weight_13(n_keys, scaled=True, d_k=D_13, n_queries=QUERIES_13, seed=13, chunk=10_000):
    """For n_queries Gaussian queries q of dimension d_k (components N(0, 1)): the attention weight of the key equal
    to q among n_keys keys (q itself, then n_keys - 1 Gaussian keys), with or without the division by sqrt(d_k).
    The softmax over the keys is summed by blocks of `chunk` keys (log-sum-exp), so that 100 000 keys fit in memory."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_queries, d_k))
    scale = 1 / math.sqrt(d_k) if scaled else 1.0
    own = (q * q).sum(axis=1) * scale                                   # the score of the copied key
    others = np.full(n_queries, -np.inf)                                # log of the sum of exp(score) of the others
    left = n_keys - 1
    while left > 0:
        k = rng.normal(size=(min(chunk, left), d_k))
        s = q @ k.T * scale
        top = s.max(axis=1)
        others = np.logaddexp(others, top + np.log(np.exp(s - top[:, None]).sum(axis=1)))
        left -= len(k)
    return 1 / (1 + np.exp(others - own))                               # exp(own) / (exp(own) + the others)'''

TODO_13 = r'''exponent_13 = ...      # a) WITH the division by sqrt(d_k): the integer k such that the median weight of the copied
                       #    key falls under 1/2 between 10**k and 10**(k + 1) keys
keeps_half_13 = ...    # b) WITHOUT the division: with 100 000 keys, is the median weight of the copied key still
                       #    above 1/2? True or False'''

EXPERIMENT_13 = r'''# 1. the attention map: 6 queries; keys 0 to 5 copy them, keys 6 to 31 are random (your scaled_dot_product_attention)
rng_13 = np.random.default_rng(130)
q_13 = rng_13.normal(size=(6, D_13))
k_13 = np.vstack([q_13, rng_13.normal(size=(26, D_13))])
maps_13 = None
with checked("B3.13"):
    maps_13 = {"avec la division par √d_k": np.asarray(mylearn.attention.scaled_dot_product_attention(q_13, k_13, k_13)[1]),
               "sans la division (scale = 1)": np.asarray(mylearn.attention.scaled_dot_product_attention(q_13, k_13, k_13, scale=1.0)[1])}
if maps_13 is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 2.8))
    for ax, (title, w) in zip(axes, maps_13.items()):
        im = ax.imshow(w, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("clé (0 à 5 : copies des requêtes)")
        ax.set_ylabel("requête")
    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.show()
    print("poids de la clé copiée, requête par requête :",
          {title: np.round(np.diag(w[:, :6]), 3).tolist() for title, w in maps_13.items()})
else:
    print("⏳ Ex B3.13 : la carte attend ta scaled_dot_product_attention (B3.12) ; la courbe ci-dessous n'en a pas besoin.")

# 2. the median weight of the copied key over 200 queries, from 10 to 100 000 keys
medians_13 = {scaled: [float(np.median(copied_weight_13(n, scaled))) for n in KEYS_13] for scaled in (True, False)}
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.semilogx(KEYS_13, medians_13[True], "o-", label="avec la division par √d_k")
ax.semilogx(KEYS_13, medians_13[False], "s--", label="sans la division")
ax.axhline(0.5, color="gray", lw=0.8)
ax.set_xlabel("nombre de clés (la clé copiée et des clés aléatoires)")
ax.set_ylabel("poids médian de la clé copiée")
ax.set_ylim(-0.03, 1.03)
ax.legend()
plt.show()
for n_13, with_13, without_13 in zip(KEYS_13, medians_13[True], medians_13[False]):
    print(f"{n_13:>7_} clés : {fr(with_13, 3)} avec la division, {fr(without_13, 6)} sans".replace("_", " "))
wb.check("B3.13a", exponent_13)
wb.check("B3.13b", keeps_half_13)'''

# --- 🔨 B3.14 -----------------------------------------------------------------------------------------------------------
CHECK_14 = r'''with checked("B3.14"):
    A = mylearn.attention
    x_14 = np.arange(12.0).reshape(1, 2, 6)
    h_14 = A.split_heads(x_14, 3)
    if returned("B3.14", "split_heads", h_14):
        verdict("B3.14", np.shape(h_14) == (1, 3, 2, 2) and np.array_equal(np.asarray(h_14)[0, 1], [[2.0, 3.0], [8.0, 9.0]]),
                "split_heads(arange(12).reshape(1, 2, 6), 3) : forme (1, 3, 2, 2), la tête 1 a les features 2 et 3.",
                f"split_heads(x, 3) doit rendre la forme (1, 3, 2, 2) et la tête 1 [[2, 3], [8, 9]] ; reçu la forme "
                f"{np.shape(h_14)} et la tête 1 {np.asarray(h_14)[0, 1].tolist() if np.ndim(h_14) == 4 else '?'}.")
    rng_14 = np.random.default_rng(14)
    x_14 = rng_14.normal(size=(2, 5, 12))
    want_14 = torch.from_numpy(x_14).view(2, 5, 4, 3).transpose(1, 2).numpy()
    same("B3.14", "split_heads(x (2, 5, 12), 4) contre view(2, 5, 4, 3).transpose(1, 2)", A.split_heads(x_14, 4), want_14,
         rtol=0, atol=0)
    same("B3.14", "merge_heads(split_heads(x, 4)), qui doit redonner x", A.merge_heads(A.split_heads(x_14, 4)), x_14, rtol=0, atol=0)
    refuses("B3.14", "split_heads d'un x 2-D", lambda: A.split_heads(np.ones((5, 12)), 4))
    refuses("B3.14", "split_heads avec 5 têtes pour D = 12", lambda: A.split_heads(x_14, 5))

    # multi_head_attention against nn.MultiheadAttention with the same weights (float64)
    torch.manual_seed(14)
    D_14, H_14 = 12, 3
    mha_14 = nn.MultiheadAttention(D_14, H_14, batch_first=True).double()
    params_14 = params_from_mha(mha_14) | {"ln1_gamma": np.ones(D_14)}         # an extra key: ignored
    saved_14 = {name: p.copy() for name, p in params_14.items()}
    xt_14 = torch.from_numpy(x_14)
    with torch.no_grad():
        want_o, want_w = mha_14(xt_14, xt_14, xt_14, attn_mask=torch.from_numpy(~causal_np(5)), average_attn_weights=False)
    out_14, w_14 = pair("B3.14", "multi_head_attention", A.multi_head_attention(x_14, x_14, x_14, MappingProxyType(params_14), H_14,
                                                                             mask=causal_np(5)), names=("output", "weights"))
    same("B3.14", "auto-attention causale (2, 5, 12), 3 têtes, contre nn.MultiheadAttention (masque ~mask)", out_14, want_o.numpy(),
         rtol=1e-6, atol=1e-6)
    same("B3.14", "poids par tête (2, 3, 5, 5)", w_14, want_w.numpy(), rtol=1e-6, atol=1e-6)
    enc_14 = rng_14.normal(size=(2, 9, D_14))
    keys_14 = np.arange(9) < np.array([9, 6])[:, None]                        # sequence 1: 6 real keys out of 9
    with torch.no_grad():
        want_x, _ = mha_14(xt_14, torch.from_numpy(enc_14), torch.from_numpy(enc_14), key_padding_mask=torch.from_numpy(~keys_14))
    out_x, w_x = pair("B3.14", "multi_head_attention", A.multi_head_attention(x_14, enc_14, enc_14, params_14, H_14,
                                                                           mask=keys_14[:, None, None, :]), names=("output", "weights"))
    same("B3.14", "attention croisée (5 requêtes, 9 clés) avec un masque de padding (2, 1, 1, 9)", out_x, want_x.numpy(),
         rtol=1e-6, atol=1e-6)
    verdict("B3.14", all(np.array_equal(params_14[name], saved_14[name]) for name in saved_14),
            "les paramètres reçus n'ont pas changé.", "multi_head_attention a modifié les paramètres reçus.")
    no_wo_14 = {name: p for name, p in params_14.items() if name != "W_o"}
    refuses("B3.14", "multi_head_attention sans W_o (ValueError, pas KeyError)",
            lambda: A.multi_head_attention(x_14, x_14, x_14, no_wo_14, H_14))'''

# --- 🔨 B3.15 -----------------------------------------------------------------------------------------------------------
CHECK_15 = r'''with checked("B3.15"):
    A = mylearn.attention
    pos_15 = np.arange(3)[:, None] / 10000.0 ** (np.arange(0, 4, 2) / 4)       # the formula of the docstring
    want_15 = np.stack([np.sin(pos_15), np.cos(pos_15)], axis=-1).reshape(3, 4)
    same("B3.15", "positional_encoding(3, 4)", A.positional_encoding(3, 4), want_15, rtol=1e-6, atol=1e-6)
    pe_15 = np.asarray(A.positional_encoding(1000, 64))
    verdict("B3.15", pe_15.shape == (1000, 64) and pe_15.dtype == np.float64 and np.abs(pe_15).max() <= 1,
            "positional_encoding(1000, 64) : forme (1000, 64), float64, valeurs dans [−1, 1].",
            f"positional_encoding(1000, 64) : attendu un tableau float64 (1000, 64) dans [−1, 1] ; reçu {pe_15.dtype} {pe_15.shape}.")
    omega_15 = 10000.0 ** (-np.arange(32) / 32)                                  # one frequency per (sin, cos) pair
    c_15, s_15 = np.cos(7 * omega_15), np.sin(7 * omega_15)
    sin_15, cos_15 = pe_15[:-7, 0::2], pe_15[:-7, 1::2]
    shifted_15 = np.stack([sin_15 * c_15 + cos_15 * s_15, cos_15 * c_15 - sin_15 * s_15], axis=-1).reshape(993, 64)
    same("B3.15", "un décalage de 7 positions = une rotation fixe de chaque paire (sin, cos) (§203.5)", pe_15[7:], shifted_15,
         rtol=1e-6, atol=1e-6)
    refuses("B3.15", "positional_encoding(4, 5) (d_model impair)", lambda: A.positional_encoding(4, 5))
    refuses("B3.15", "positional_encoding(0, 4)", lambda: A.positional_encoding(0, 4))

    ones_15 = np.ones((3, 4))
    half_15 = 10000.0 ** (-2 * np.arange(2) / 4)                                  # theta = position * base**(-2i/d)
    z_15 = (1 + 1j) * np.exp(1j * np.arange(3)[:, None] * half_15)                # pair (x_i, x_{i+d/2}) as x_i + i x_{i+d/2}
    same("B3.15", "apply_rope(np.ones((3, 4))) (paires (i, i + d/2))", A.apply_rope(ones_15),
         np.concatenate([z_15.real, z_15.imag], axis=1), rtol=1e-6, atol=1e-6)
    rng_15 = np.random.default_rng(15)
    x_15 = rng_15.normal(size=(2, 3, 10, 8))
    same("B3.15", "normes de apply_rope(x (2, 3, 10, 8)), égales à celles de x", np.linalg.norm(A.apply_rope(x_15), axis=-1),
         np.linalg.norm(x_15, axis=-1), rtol=1e-6, atol=1e-6)
    q_15, k_15 = rng_15.normal(size=(1, 8)), rng_15.normal(size=(1, 8))
    dots_15 = [(np.asarray(A.apply_rope(q_15, [m])) @ np.asarray(A.apply_rope(k_15, [n])).T).item()
               for m, n in ((3, 1), (12, 10), (102, 100))]
    verdict("B3.15", np.allclose(dots_15, dots_15[0], rtol=1e-6, atol=1e-6),
            "⟨RoPE(q, m), RoPE(k, n)⟩ ne dépend que de m − n : " + ", ".join(fr(d, 4) for d in dots_15) + " pour (3, 1), (12, 10), (102, 100).",
            "⟨RoPE(q, m), RoPE(k, n)⟩ doit être le même pour (3, 1), (12, 10) et (102, 100) (m − n = 2) ; reçu "
            + ", ".join(fr(d, 4) for d in dots_15) + ".")
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
    except ImportError:
        print("⏳ Ex B3.15 : la comparaison à LlamaRotaryEmbedding attend la bibliothèque transformers (Colab).")
    else:
        config_15 = LlamaConfig(hidden_size=24, num_attention_heads=3, head_dim=8, max_position_embeddings=128,
                                rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
        cos_hf, sin_hf = LlamaRotaryEmbedding(config_15)(torch.from_numpy(x_15), torch.arange(10)[None, :])
        want_hf, _ = apply_rotary_pos_emb(torch.from_numpy(x_15), torch.from_numpy(x_15), cos_hf, sin_hf)
        same("B3.15", "apply_rope(x) contre LlamaRotaryEmbedding + apply_rotary_pos_emb de transformers (angles en float32 : 1e-5)",
             A.apply_rope(x_15), want_hf.numpy(), rtol=1e-5, atol=1e-5)
    refuses("B3.15", "apply_rope(np.ones((3, 4)), positions=[0]) (une position pour trois vecteurs)",
            lambda: A.apply_rope(ones_15, positions=[0]))
    refuses("B3.15", "apply_rope(np.ones((3, 5))) (d impair)", lambda: A.apply_rope(np.ones((3, 5))))'''

# --- 🔨 B3.16 -----------------------------------------------------------------------------------------------------------
CHECK_16 = r'''with checked("B3.16"):
    A = mylearn.attention
    for name_16, probe_16, where_16 in (
            ("gelu", lambda: mylearn.nn.activations.gelu(np.zeros(2)), "nn/activations.py, Ex 17.14"),
            ("layer_norm", lambda: mylearn.nn.regularization.layer_norm(np.zeros((1, 4)), np.ones(4), np.zeros(4)),
             "nn/regularization.py, Ex 20.17")):
        if not written(probe_16):
            print(f"ℹ️ Ex B3.16 : ta {name_16} ({where_16}) n'est pas encore écrite : écris-la, ou recode-la dans "
                  "transformer_block (l'énoncé le permet).")
    # the same block in PyTorch: nn.TransformerEncoderLayer, pre-LN, exact GELU, no dropout, in float64
    torch.manual_seed(16)
    D_16, H_16, F_16 = 16, 4, 64
    layer_16 = nn.TransformerEncoderLayer(D_16, H_16, F_16, dropout=0.0, activation="gelu", batch_first=True,
                                          norm_first=True).double().eval()
    params_16 = block_params_from_torch(layer_16)
    saved_16 = {name: p.copy() for name, p in params_16.items()}
    x_16 = np.random.default_rng(16).normal(size=(2, 7, D_16))
    x_copy_16 = x_16.copy()
    with torch.no_grad():
        want_c16 = layer_16(torch.from_numpy(x_16), src_mask=torch.from_numpy(~causal_np(7))).numpy()
        want_n16 = layer_16(torch.from_numpy(x_16)).numpy()
    same("B3.16", "transformer_block(x (2, 7, 16), 4 têtes, masque causal) contre nn.TransformerEncoderLayer(norm_first=True)",
         A.transformer_block(x_16, params_16, H_16, mask=causal_np(7)), want_c16, rtol=1e-6, atol=1e-6)
    same("B3.16", "le même bloc sans masque (un encodeur)", A.transformer_block(x_16, params_16, H_16), want_n16,
         rtol=1e-6, atol=1e-6)
    layer_e16 = nn.TransformerEncoderLayer(D_16, H_16, F_16, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True, layer_norm_eps=0.1).double().eval()
    layer_e16.load_state_dict(layer_16.state_dict())
    with torch.no_grad():
        want_e16 = layer_e16(torch.from_numpy(x_16)).numpy()
    same("B3.16", "eps=0.1 (les deux layer norms)", A.transformer_block(x_16, params_16, H_16, eps=0.1), want_e16,
         rtol=1e-6, atol=1e-6)
    verdict("B3.16", np.array_equal(x_16, x_copy_16) and all(np.array_equal(params_16[k], saved_16[k]) for k in saved_16),
            "x et les paramètres reçus n'ont pas changé.", "transformer_block a modifié x ou les paramètres reçus.")
    no_w1_16 = {name: p for name, p in params_16.items() if name != "W_1"}
    refuses("B3.16", "transformer_block sans W_1 (ValueError, pas KeyError)", lambda: A.transformer_block(x_16, no_w1_16, H_16))'''

# --- 🐛 B3.17 ----------------------------------------------------------------------------------------------------------
GIVEN_17 = r'''PAD_17 = len(DATA.itos)                    # one more id for the padding: 89 ids in all
VOCAB_17, LEN_17 = PAD_17 + 1, 64
LINES_17 = [line for line in DATA.train_text.split("\n") if 24 <= len(line) <= LEN_17]    # lines of Holmes
VAL_17 = DATA.val_ids[:LEN_17 * 64].view(64, LEN_17)                                       # 64 windows, no padding


def batch_17(lines, generator, batch_size=32):
    """A batch of lines of Holmes: inputs ids[:-1] and targets ids[1:], padded with PAD_17 to LEN_17 - 1 positions."""
    picks = torch.randint(len(lines), (batch_size,), generator=generator).tolist()
    width = max(len(lines[i]) for i in picks) - 1
    x = torch.full((batch_size, width), PAD_17)
    y = torch.full((batch_size, width), PAD_17)
    for row, i in enumerate(picks):
        ids = DATA.encode(lines[i])
        x[row, :len(ids) - 1], y[row, :len(ids) - 1] = ids[:-1], ids[1:]
    return x, y


def diagnose_17(model_class, loss_fn, iters=200):
    """Three diagnoses of the small language model, in a row: the first one that fails stops the chain (one error can
    hide another). Returns the trained model when the three pass, else None."""
    torch.manual_seed(17)
    model = model_class(VOCAB_17)
    gen = torch.Generator().manual_seed(17)
    x, y = batch_17(LINES_17, gen)

    def point1():
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y)
        if torch.isfinite(loss) and torch.isfinite(logits).all():
            return True, f"la loss d'un batch à l'initialisation est un nombre, {fr(loss.item(), 3)}", ""
        bad = sorted({int(t) for t in torch.nonzero(~torch.isfinite(logits).all(dim=-1))[:, 1]})
        return False, (f"la loss d'un batch vaut {loss.item()} ; les logits sont nan à la position {bad[0]} de chaque séquence"
                       if len(bad) == 1 else f"la loss d'un batch vaut {loss.item()} ; logits nan aux positions {bad[:6]}"), \
            "que voit cette position dans la couche d'attention ?"

    def point2():
        probe = VAL_17[:4].clone()
        changed = probe.clone()
        changed[:, 40] = (changed[:, 40] + 1) % PAD_17                      # change the character 40 of each window
        with torch.no_grad():
            gap = (model(probe)[:, :40] - model(changed)[:, :40]).abs().max().item()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(iters):
            xb, yb = batch_17(LINES_17, gen)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val = F.cross_entropy(model(VAL_17[:, :-1]).flatten(0, 1), VAL_17[:, 1:].flatten()).item()
        if gap < 1e-6:
            return True, (f"changer le caractère 40 d'une fenêtre ne change aucun logit des positions 0 à 39 ; après "
                          f"{iters} itérations, la loss de validation vaut {fr(val, 2)} nats par caractère"), ""
        return False, (f"changer le caractère 40 d'une fenêtre change les logits des positions 0 à 39 (écart maximal "
                       f"{gap:.2g}) ; après {iters} itérations, la loss de validation tombe à {fr(val, 2)} nat par "
                       "caractère"), "pour un modèle d'une seule couche, est-ce crédible ? D'où vient l'information ?"

    def point3():
        short = [line for line in LINES_17 if len(line) <= 40][:16]
        width = max(len(line) for line in short) - 1
        x_s = torch.full((16, width), PAD_17)
        y_s = torch.full((16, width), PAD_17)
        for row, line in enumerate(short):
            ids = DATA.encode(line)
            x_s[row, :len(ids) - 1], y_s[row, :len(ids) - 1] = ids[:-1], ids[1:]
        extra = 60 - width                                                   # the same lines, 60 positions wide
        x_l = torch.cat([x_s, torch.full((16, extra), PAD_17)], dim=1)
        y_l = torch.cat([y_s, torch.full((16, extra), PAD_17)], dim=1)
        with torch.no_grad():
            short_loss, long_loss = loss_fn(model(x_s), y_s).item(), loss_fn(model(x_l), y_l).item()
        if abs(short_loss - long_loss) < 1e-5:
            return True, (f"la loss d'un même batch ne dépend pas de son padding ({fr(short_loss, 4)} sur {width} "
                          f"positions comme sur 60)"), ""
        return False, (f"la loss d'un même batch de 16 lignes vaut {fr(short_loss, 4)} sur {width} positions et "
                       f"{fr(long_loss, 4)} une fois complété par {extra} positions de padding"), \
            "quelles positions entrent dans la moyenne ?"

    points = [("la loss est un nombre", point1), ("le modèle ne voit pas le futur", point2),
              ("le padding ne compte pas", point3)]
    for k, (name, run) in enumerate(points, start=1):
        ok, measured, why = run()
        if ok:
            print(f"✅ Ex B3.17 : point {k}, {name} ({measured}).")
        else:
            print(f"❌ Ex B3.17 : point {k} : {measured} ; {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex B3.17 : point {later} : corrige d'abord le point {k}.")
            return None
    print("✅ Ex B3.17 : les trois diagnostics passent.")
    return model'''

TODO_17 = r'''# The small language model to repair IN PLACE (same names, same signatures, same returns): the diagnostics of the
# next cell start from it. Its author first computed the attention with F.scaled_dot_product_attention, then switched
# to nn.MultiheadAttention to read the attention weights.
def causal_mask_17(T):
    """Boolean causal mask in the convention of F.scaled_dot_product_attention: True where query i may attend to key j."""
    return torch.ones(T, T, dtype=torch.bool).triu()


class TinyLM_17(nn.Module):
    """One causal attention layer over character and position embeddings, then a dense layer: ids (N, T) -> logits
    (N, T, vocab_size)."""

    def __init__(self, vocab_size, d_model=32, n_head=2, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(T))
        h, self.weights = self.attn(x, x, x, attn_mask=causal_mask_17(T))     # weights kept to draw the maps
        return self.head(self.ln(x + h))


def lm_loss_17(logits, targets):
    """Mean cross-entropy over the real characters of a batch: logits (N, T, V), targets (N, T) with PAD_17 on the
    padding."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))'''

SOLUTION_17 = r'''def causal_mask_17(T):
    """Boolean causal mask in the convention of F.scaled_dot_product_attention: True where query i may attend to key j."""
    return torch.ones(T, T, dtype=torch.bool).tril()                           # 2. the lower triangle: j <= i


class TinyLM_17(nn.Module):
    """One causal attention layer over character and position embeddings, then a dense layer: ids (N, T) -> logits
    (N, T, vocab_size)."""

    def __init__(self, vocab_size, d_model=32, n_head=2, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(T))
        h, self.weights = self.attn(x, x, x, attn_mask=~causal_mask_17(T))    # 1. nn.MultiheadAttention: True = masked
        return self.head(self.ln(x + h))


def lm_loss_17(logits, targets):
    """Mean cross-entropy over the real characters of a batch: logits (N, T, V), targets (N, T) with PAD_17 on the
    padding."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=PAD_17)                                # 3. the padding is left out of the mean'''

CHECK_17 = r'''with checked("B3.17"):
    model_17 = diagnose_17(TinyLM_17, lm_loss_17)'''

EXPERIMENT_17 = r'''if globals().get("model_17") is None:
    print("⏳ Ex B3.17 : la carte attend que les trois points de la cellule précédente passent.")
else:
    with checked("B3.17"):
        line_17 = "I had seen little of Holmes lately."                  # a line of the first story
        ids_17 = DATA.encode(line_17)[None, :-1]
        model_17.eval()
        with torch.no_grad():
            logits_17 = model_17(ids_17)
        w_17 = model_17.weights[0].numpy()                                 # (T, T), averaged over the 2 heads
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(w_17, cmap="viridis")
        ax.set_xticks(range(len(line_17) - 1), list(line_17[:-1]))
        ax.set_yticks(range(len(line_17) - 1), list(line_17[:-1]))
        ax.set_xlabel("clé")
        ax.set_ylabel("requête")
        ax.set_title("ta version réparée : poids d'attention (moyenne des deux têtes)")
        plt.show()
        print("au-dessus de la diagonale, le plus grand poids vaut", fr(float(np.triu(w_17, 1).max()), 6))'''

# --- Part B: tools ----------------------------------------------------------------------------------------------------
TOOLS_B = r'''# Tools of part B: the mini-GPT in PyTorch
MINI_CONFIG = dict(wbgpt.MINI_CONFIG)          # 4 blocks, 4 heads, d = 128, context of 128 characters, 88 ids
FORBIDDEN = (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder,
             nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)


def n_params(model):
    """The number of parameters of a module, each tied parameter counted once (model.parameters() lists it once)."""
    return sum(p.numel() for p in model.parameters())


def forbidden_in(model):
    """The names of the ready-made attention modules of PyTorch found in a model (the contract forbids them)."""
    return sorted({type(m).__name__ for m in model.modules() if isinstance(m, FORBIDDEN)})


def params_from_qkv(qkv, proj):
    """The parameters of an attention module (qkv: nn.Linear(d, 3d), queries then keys then values; proj:
    nn.Linear(d, d)) in the convention of multi_head_attention (z = x @ W + b, float64): W = weight.T."""
    D = proj.in_features
    W = qkv.weight.detach().double().numpy()
    b = qkv.bias.detach().double().numpy()
    return {"W_q": W[:D].T.copy(), "W_k": W[D:2 * D].T.copy(), "W_v": W[2 * D:].T.copy(),
            "W_o": proj.weight.detach().double().numpy().T.copy(), "b_q": b[:D].copy(), "b_k": b[D:2 * D].copy(),
            "b_v": b[2 * D:].copy(), "b_o": proj.bias.detach().double().numpy().copy()}


try:                                                   # the vocabulary: the rule of your build_vocab (ch. 22, 22.12)
    _, itos_vocab = mylearn.sequences.build_vocab(DATA.train_text + DATA.val_text)
    print("ℹ️ ton build_vocab (ch. 22) donne le vocabulaire de DATA :" if list(itos_vocab) == DATA.itos else
          "⚠️ ton build_vocab (ch. 22) ne donne pas le vocabulaire de DATA (décroissance de la fréquence, ex aequo dans "
          "l'ordre de sorted) : le notebook garde celui de DATA, relance les tests du ch. 22 ;",
          f"{len(DATA.itos)} caractères, les plus fréquents {DATA.itos[:6]}.")
except Exception:                                     # not written yet (NotImplementedError), or no module at all
    print(f"ℹ️ ton build_vocab (ch. 22, 22.12) n'est pas écrit : le notebook prend le vocabulaire de DATA, construit par "
          f"la même règle ({len(DATA.itos)} caractères).")'''

# --- 📦 B3.18 ----------------------------------------------------------------------------------------------------------
TODO_18 = r'''class CausalSelfAttention_18(nn.Module):
    """Causal multi-head self-attention (contract of Ex B3.18): x (N, T, d_model) -> (N, T, d_model)."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        raise NotImplementedError("CausalSelfAttention_18")

    def forward(self, x):
        raise NotImplementedError("CausalSelfAttention_18.forward")'''

SOLUTION_18 = r'''class CausalSelfAttention_18(nn.Module):
    """Causal multi-head self-attention (contract of Ex B3.18): x (N, T, d_model) -> (N, T, d_model)."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        if d_model % n_head:
            raise ValueError(f"n_head = {n_head} must divide d_model = {d_model}")
        self.n_head, self.dropout = n_head, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)       # the three projections in one product: q, then k, then v
        self.proj = nn.Linear(d_model, d_model)          # the output projection W_O
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        N, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)                                       # 3 x (N, T, D)
        q, k, v = (t.view(N, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))   # (N, H, T, d_h)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(N, T, D)                                      # merge the heads (reshape: copy)
        return self.resid_drop(self.proj(y))'''

CHECK_18 = r'''attn_ok_18 = False
with checked("B3.18"):
    torch.manual_seed(18)
    attn_18 = CausalSelfAttention_18(32, 4)
    shapes_18 = {name: tuple(p.shape) for name, p in attn_18.named_parameters()}
    want_18 = {"qkv.weight": (96, 32), "qkv.bias": (96,), "proj.weight": (32, 32), "proj.bias": (32,)}
    verdict("B3.18", shapes_18 == want_18, "deux couches denses, qkv (32 → 96) et proj (32 → 32), avec leurs biais.",
            f"paramètres attendus {want_18}, reçus {shapes_18} : relis le contrat (noms et formes).")
    if shapes_18 != want_18:
        raise StopCheck
    if forbidden_in(attn_18):
        print(f"❌ Ex B3.18 : ton module contient {forbidden_in(attn_18)} : le contrat demande F.scaled_dot_product_attention.")
        raise StopCheck
    attn_18.eval()
    x_18 = torch.randn(3, 10, 32)
    with torch.no_grad():
        y_18 = attn_18(x_18)
    if tuple(y_18.shape) != (3, 10, 32):
        print(f"❌ Ex B3.18 : x (3, 10, 32) donne une sortie {tuple(y_18.shape)} ; attendu (3, 10, 32).")
        raise StopCheck
    # the same attention in PyTorch's ready-made module, with the same weights: the oracle that is not yours
    params_18 = params_from_qkv(attn_18.qkv, attn_18.proj)
    mha_18 = nn.MultiheadAttention(32, 4, batch_first=True)
    with torch.no_grad():
        mha_18.in_proj_weight.copy_(attn_18.qkv.weight)
        mha_18.in_proj_bias.copy_(attn_18.qkv.bias)
        mha_18.out_proj.weight.copy_(attn_18.proj.weight)
        mha_18.out_proj.bias.copy_(attn_18.proj.bias)
        torch_18, _ = mha_18(x_18, x_18, x_18, attn_mask=torch.from_numpy(~causal_np(10)), need_weights=False)
    gap_torch_18 = (y_18 - torch_18).abs().max().item()
    verdict("B3.18", gap_torch_18 < 1e-5,
            f"ton module donne la sortie de nn.MultiheadAttention aux mêmes poids (écart maximal {gap_torch_18:.1e}).",
            f"ton module s'écarte de {gap_torch_18:.2g} de nn.MultiheadAttention aux mêmes poids (masque causal) : vérifie "
            "l'ordre q, k, v dans qkv, le découpage en têtes (view puis transpose), le masque causal et le recollage des têtes.")
    # your NumPy version (Ex B3.14) with the same weights, in float64
    x64_18 = x_18.double().numpy()
    try:
        out_np_18 = np.asarray(mylearn.attention.multi_head_attention(x64_18, x64_18, x64_18, params_18, 4,
                                                                      mask=causal_np(10))[0], dtype=float)
    except (NotImplementedError, ImportError, AttributeError):      # not written yet, or no attention.py at all
        out_np_18 = None
        print("⏳ Ex B3.18 : l'écart à ta version NumPy attend ta multi_head_attention (B3.14).")
    except Exception as exc:                                         # noqa: BLE001 - a bug of B3.14: say it, go on
        out_np_18 = None
        print(f"❌ Ex B3.18 : ta multi_head_attention (B3.14) lève {type(exc).__name__} : {str(exc)[:160]} ; relance ses tests.")
    if out_np_18 is not None and out_np_18.shape != (3, 10, 32):
        print(f"❌ Ex B3.18 : ta multi_head_attention (B3.14) rend une sortie {out_np_18.shape} au lieu de (3, 10, 32) ; "
              "relance ses tests.")
        out_np_18 = None
    if out_np_18 is not None:
        gap_18 = float(np.abs(y_18.double().numpy() - out_np_18).max())
        print(f"écart maximal entre ton module PyTorch (float32) et ta multi_head_attention NumPy (float64) : {gap_18:.1e}")
        if gap_18 >= 1e-5 and gap_torch_18 < 1e-5:
            print("❌ Ex B3.18 : ton module est d'accord avec nn.MultiheadAttention : c'est ta multi_head_attention (B3.14) "
                  "qui s'en écarte ; relance ses tests.")
        wb.check("B3.18", bool(gap_18 < 1e-5), computed=True)
    # causality, then dropout only in training
    x2_18 = x_18.clone()
    x2_18[:, 6] += 1.0
    with torch.no_grad():
        past_18 = (attn_18(x2_18)[:, :6] - y_18[:, :6]).abs().max().item()
    verdict("B3.18", past_18 == 0, "changer le token 6 ne change aucune sortie des tokens 0 à 5 : le module est causal.",
            f"changer le token 6 change les sorties des tokens 0 à 5 (écart {past_18:.2g}) : le module voit le futur.")
    drop_18 = CausalSelfAttention_18(32, 4, dropout=0.5)
    drop_18.load_state_dict(attn_18.state_dict())
    with torch.no_grad():
        drop_18.train()
        a_18, b_18 = drop_18(x_18), drop_18(x_18)
        drop_18.eval()
        c_18 = drop_18(x_18)
    verdict("B3.18", not torch.allclose(a_18, b_18) and torch.allclose(c_18, y_18),
            "avec dropout=0.5 : deux passes différentes en mode train, la sortie sans dropout en mode eval.",
            "avec dropout=0.5, attendu deux sorties différentes en mode train (dropout actif) et, en mode eval, la sortie "
            "du module sans dropout (dropout_p = 0 hors entraînement).")
    attn_ok_18 = gap_torch_18 < 1e-5 and past_18 == 0'''

RECORD_18 = r'''wb.record("B3.18", bool(gap_18 < 1e-5), mistakes={
    "l'écart à ta version NumPy dépasse 1e-5 : relis les lignes ❌ au-dessus (elles disent lequel des deux codes s'écarte de nn.MultiheadAttention)": False})'''

# --- 📦 B3.19 ----------------------------------------------------------------------------------------------------------
TODO_19 = r'''class Block_19(nn.Module):
    """Pre-LN Transformer block of the mini-GPT (contract of Ex B3.19): x (N, T, d_model) -> (N, T, d_model)."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Block_19")

    def forward(self, x):
        raise NotImplementedError("Block_19.forward")


class MiniGPT_19(nn.Module):
    """Character-level mini-GPT (contract of Ex B3.19): idx (N, T) int64 with T <= block_size -> logits
    (N, T, vocab_size)."""

    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, d_model=128, dropout=0.0):
        super().__init__()
        raise NotImplementedError("MiniGPT_19")

    def forward(self, idx):
        raise NotImplementedError("MiniGPT_19.forward")'''

SOLUTION_19 = r'''class Block_19(nn.Module):
    """Pre-LN Transformer block of the mini-GPT (contract of Ex B3.19): x (N, T, d_model) -> (N, T, d_model)."""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention_18(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                  # each sub-layer ADDS to the residual path
        return x + self.mlp(self.ln2(x))


class MiniGPT_19(nn.Module):
    """Character-level mini-GPT (contract of Ex B3.19): idx (N, T) int64 with T <= block_size -> logits
    (N, T, vocab_size)."""

    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, d_model=128, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)            # learned positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block_19(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                      # tied: the same tensor, counted once
        for module in self.modules():                               # nanoGPT: N(0, 0.02²), zero biases
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        for block in self.blocks:                                   # the projections that add to the residual path
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(block.mlp[2].weight, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, idx):
        N, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"{T} tokens, more than block_size = {self.block_size}")
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))'''

CHECK_19 = r'''model_ok_19 = False
with checked("B3.19"):
    torch.manual_seed(19)
    gpt_19 = MiniGPT_19(**MINI_CONFIG)
    if forbidden_in(gpt_19):
        print(f"❌ Ex B3.19 : ton modèle contient {forbidden_in(gpt_19)} : le contrat interdit les modules d'attention tout faits.")
        raise StopCheck
    got_19 = {name: tuple(t.shape) for name, t in gpt_19.state_dict().items()}
    want_19 = wbgpt.expected_state_shapes(MINI_CONFIG)
    missing_19 = [n for n in want_19 if n not in got_19]
    extra_19 = [n for n in got_19 if n not in want_19]
    wrong_19 = [f"{n} {got_19[n]} au lieu de {want_19[n]}" for n in want_19 if n in got_19 and got_19[n] != want_19[n]]
    verdict("B3.19", not (missing_19 or extra_19 or wrong_19),
            f"les {len(want_19)} tenseurs du state_dict ont les noms et les formes du contrat.",
            "le state_dict ne suit pas le contrat : " + "; ".join(filter(None, [
                f"manquants {missing_19[:4]}" if missing_19 else "", f"en trop {extra_19[:4]}" if extra_19 else "",
                f"formes {wrong_19[:3]}" if wrong_19 else ""])) + ".")
    if missing_19 or extra_19 or wrong_19:
        raise StopCheck
    tied_19 = gpt_19.head.weight is gpt_19.tok_emb.weight
    verdict("B3.19", tied_19, "la couche de sortie et la table des tokens sont le même tenseur (poids liés).",
            "head.weight doit être le tenseur tok_emb.weight lui-même (poids liés), pas une copie.")
    verdict("B3.19", n_params(gpt_19) == wbgpt.count_params(MINI_CONFIG),
            "le même nombre de paramètres que ton calcul de ✏️ B3.5 j) (la tête liée comptée une fois).",
            f"ton modèle a {n_params(gpt_19):_} paramètres : ce n'est pas le compte de ✏️ B3.5 j) ; relis le contrat (biais, "
            "positions apprises, layer norm finale, tête liée sans biais).".replace("_", " "))
    kinds_19 = {type(b).__name__ for b in gpt_19.blocks} | {type(b.attn).__name__ for b in gpt_19.blocks}
    verdict("B3.19", kinds_19 == {"Block_19", "CausalSelfAttention_18"},
            "4 blocs Block_19, chacun avec ton CausalSelfAttention_18.",
            f"blocks doit contenir des Block_19, dont attn est un CausalSelfAttention_18 (B3.18) ; reçu {sorted(kinds_19)}.")
    blk_19 = gpt_19.blocks[0].eval()
    xb_19 = torch.randn(2, 5, MINI_CONFIG["d_model"])
    with torch.no_grad():
        h_19 = xb_19 + blk_19.attn(blk_19.ln1(xb_19))
        pre_ln_19 = torch.allclose(blk_19(xb_19), h_19 + blk_19.mlp(blk_19.ln2(h_19)), atol=1e-6)
    verdict("B3.19", pre_ln_19, "Block_19 est pré-LN : h = x + attn(ln1(x)), puis h + mlp(ln2(h)).",
            "Block_19(x) ne vaut pas h + mlp(ln2(h)) avec h = x + attn(ln1(x)) : relis le bloc pré-LN du contrat (§203.6).")
    std_19 = {"tok_emb": gpt_19.tok_emb.weight.std().item(), "attn.qkv": gpt_19.blocks[0].attn.qkv.weight.std().item(),
              "attn.proj": gpt_19.blocks[0].attn.proj.weight.std().item(), "mlp.2": gpt_19.blocks[0].mlp[2].weight.std().item()}
    want_std_19 = {"tok_emb": 0.02, "attn.qkv": 0.02, "attn.proj": 0.02 / math.sqrt(8), "mlp.2": 0.02 / math.sqrt(8)}
    zero_bias_19 = all(float(m.bias.abs().max()) == 0 for m in gpt_19.modules() if isinstance(m, nn.Linear) and m.bias is not None)
    verdict("B3.19", all(abs(std_19[k] / want_std_19[k] - 1) < 0.1 for k in std_19) and zero_bias_19,
            "initialisation de nanoGPT : écarts-types 0,02 et 0,02/√8 pour les deux projections résiduelles, biais nuls.",
            "l'initialisation n'est pas celle du contrat : écarts-types mesurés "
            + ", ".join(f"{k} {fr(v, 4)}" for k, v in std_19.items()) + (", biais non nuls" if not zero_bias_19 else "") + ".")
    x_19, y_19 = wbgpt.get_batch(DATA.train_ids, 4, 128, torch.Generator().manual_seed(19))
    gpt_19.eval()
    with torch.no_grad():
        logits_19 = gpt_19(x_19)
        shape_ok_19 = tuple(logits_19.shape) == (4, 128, 88)
        verdict("B3.19", shape_ok_19, "des ids (4, 128) donnent des logits (4, 128, 88).",
                f"des ids (4, 128) donnent {tuple(logits_19.shape)} : attendu (4, 128, 88), un vecteur de logits par position.")
        if not shape_ok_19:
            raise StopCheck
        loss0_19 = F.cross_entropy(logits_19.flatten(0, 1), y_19.flatten()).item()
        verdict("B3.19", abs(loss0_19 - math.log(88)) < 0.1,
                "à l'initialisation, la loss est celle d'une prédiction presque uniforme (le contrôle de départ, §203.8).",
                f"à l'initialisation, la loss d'un batch vaut {fr(loss0_19, 2)} : loin d'une prédiction presque uniforme ; "
                "relis le contrôle de départ (§203.8) et l'initialisation du contrat.")
        repeated_19 = gpt_19(torch.zeros(1, 16, dtype=torch.long))[0]
        positions_19 = (repeated_19[1:] - repeated_19[:1]).abs().max().item() > 1e-6
        verdict("B3.19", positions_19,
                "les positions apprises servent : un même caractère répété 16 fois donne des logits qui changent d'une "
                "position à l'autre.",
                "un même caractère répété 16 fois donne les mêmes logits à chaque position : les positions apprises "
                "sont-elles ajoutées aux embeddings des tokens ?")
        changed_19 = x_19.clone()
        changed_19[:, 100] = (changed_19[:, 100] + 1) % 88
        future_19 = (gpt_19(changed_19)[:, :100] - logits_19[:, :100]).abs().max().item()
    verdict("B3.19", future_19 == 0, "changer le caractère 100 ne change aucun logit des positions 0 à 99 : le modèle est causal.",
            f"changer le caractère 100 change les logits des positions 0 à 99 (écart {future_19:.2g}) : le modèle voit le futur.")
    refuses("B3.19", "129 caractères pour un contexte de 128", lambda: gpt_19(torch.zeros(1, 129, dtype=torch.long)))
    model_ok_19 = all([tied_19, n_params(gpt_19) == wbgpt.count_params(MINI_CONFIG), pre_ln_19, positions_19,
                       kinds_19 == {"Block_19", "CausalSelfAttention_18"}, abs(loss0_19 - math.log(88)) < 0.1,
                       future_19 == 0])'''

# --- 📦 B3.20 ----------------------------------------------------------------------------------------------------------
GIVEN_20 = r'''CONFIG_20 = dict(MINI_CONFIG)                       # the model of B3.19
TRAIN_20 = wbgpt.train_settings(FAST_MODE)           # 600 iterations in fast mode, 5 000 in full mode
SEED_20 = wbgpt.SEED                                 # seed of the initialisation and of the training batches
get_batch = wbgpt.get_batch                          # the get_batch of the fiche (§203.8)
print(TRAIN_20)'''

TODO_20 = r'''def configure_optimizer_20(model, lr, weight_decay, betas=(0.9, 0.99)):
    """torch.optim.AdamW with two parameter groups: weight_decay on the parameters with 2 axes or more, 0.0 on the
    others; each parameter exactly once."""
    raise NotImplementedError("configure_optimizer_20")


def estimate_loss_20(model, data, n_batches, batch_size, block_size, generator):
    """Mean cross-entropy (nats per token, a Python float) of model on n_batches batches
    get_batch(data, batch_size, block_size, generator), in eval mode without gradient; model back in train mode."""
    raise NotImplementedError("estimate_loss_20")


def train_20(model, train_ids, val_ids, settings, generator, block_size=128):
    """Train model in place (contract of Ex B3.20) and return its history: {"train_loss": [...], "lr": [...],
    "val_iter": [...], "val_loss": [...]}."""
    raise NotImplementedError("train_20")'''

SOLUTION_20 = r'''def configure_optimizer_20(model, lr, weight_decay, betas=(0.9, 0.99)):
    """torch.optim.AdamW with two parameter groups: weight_decay on the parameters with 2 axes or more, 0.0 on the
    others; each parameter exactly once."""
    params = list(model.parameters())                      # lists the tied table once
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


@torch.no_grad()
def estimate_loss_20(model, data, n_batches, batch_size, block_size, generator):
    """Mean cross-entropy (nats per token, a Python float) of model on n_batches batches
    get_batch(data, batch_size, block_size, generator), in eval mode without gradient; model back in train mode."""
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, block_size, generator)
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return total / n_batches


def train_20(model, train_ids, val_ids, settings, generator, block_size=128):
    """Train model in place (contract of Ex B3.20) and return its history: {"train_loss": [...], "lr": [...],
    "val_iter": [...], "val_loss": [...]}."""
    s = settings
    optimizer = configure_optimizer_20(model, s["lr"], s["weight_decay"], (0.9, s["beta2"]))
    history = {"train_loss": [], "lr": [], "val_iter": [], "val_loss": []}

    def evaluate(done):                                    # the same validation batches at every evaluation
        history["val_iter"].append(done)
        history["val_loss"].append(estimate_loss_20(model, val_ids, s["eval_batches"], s["batch_size"], block_size,
                                                    torch.Generator().manual_seed(s["eval_seed"])))

    model.train()
    evaluate(0)
    for it in range(s["iters"]):
        if it < s["warmup"]:                               # linear warmup, then cosine down to min_lr (ch. 19)
            lr = s["lr"] * it / s["warmup"]
        else:
            progress = (it - s["warmup"]) / (s["iters"] - s["warmup"])
            lr = s["min_lr"] + (s["lr"] - s["min_lr"]) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_ids, s["batch_size"], block_size, generator)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), s["clip"])
        optimizer.step()
        history["train_loss"].append(loss.item())
        history["lr"].append(lr)
        if (it + 1) % s["eval_interval"] == 0 or it + 1 == s["iters"]:
            evaluate(it + 1)
    return history'''

CHECK_20 = r'''probes_ok_20 = False
SMALL_20 = {"vocab_size": 88, "block_size": 16, "n_layer": 2, "n_head": 2, "d_model": 16, "dropout": 0.0}
with checked("B3.20"):
    # 1. the two groups of AdamW, parameter by parameter (on the reference model of wb.gpt, so that B3.19 is not needed)
    torch.manual_seed(20)
    small_20 = wbgpt.ReferenceGPT(**SMALL_20)
    opt_20 = configure_optimizer_20(small_20, 1e-3, 0.1)
    if not isinstance(opt_20, torch.optim.AdamW):
        print(f"❌ Ex B3.20 : configure_optimizer_20 doit rendre un torch.optim.AdamW ; reçu {type(opt_20).__name__}.")
        raise StopCheck
    seen_20 = [id(p) for g in opt_20.param_groups for p in g["params"]]
    decay_20 = {id(p): g["weight_decay"] for g in opt_20.param_groups for p in g["params"]}
    wrong_20 = [n for n, p in small_20.named_parameters() if decay_20.get(id(p)) != (0.1 if p.dim() >= 2 else 0.0)]
    verdict("B3.20", sorted(seen_20) == sorted(id(p) for p in small_20.parameters()) and not wrong_20,
            "AdamW : chaque paramètre une fois, weight decay 0,1 sur les matrices et les tables, 0 sur les biais et les layer norms.",
            (f"weight decay mal réglé pour {wrong_20[:4]} : 0,1 sur les paramètres à deux axes ou plus, 0 sur les autres."
             if wrong_20 else "chaque paramètre du modèle doit être dans un groupe, et une seule fois (la tête liée est la "
             "table des tokens)."))
    verdict("B3.20", all(tuple(g["betas"]) == (0.9, 0.99) and g["lr"] == 1e-3 for g in opt_20.param_groups),
            "β = (0,9 ; 0,99) et le learning rate reçu.", "les groupes doivent avoir betas=(0.9, 0.99) et le learning rate reçu.")

    # 2. estimate_loss_20: eval mode and no gradient during the measure, train mode afterwards, the right value
    class Spy_20(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.calls = inner, []

        def forward(self, idx):
            self.calls.append((self.training, torch.is_grad_enabled()))
            return self.inner(idx)

    spy_20 = Spy_20(copy.deepcopy(small_20)).train()
    est_20 = estimate_loss_20(spy_20, DATA.val_ids, 3, 4, 16, torch.Generator().manual_seed(5))
    want_est_20 = wbgpt.estimate_loss(copy.deepcopy(small_20), DATA.val_ids, 3, 4, 16, torch.Generator().manual_seed(5))
    verdict("B3.20", type(est_20) is float and math.isclose(est_20, want_est_20, rel_tol=1e-6),
            f"estimate_loss_20 : la moyenne des loss de 3 batches, un float ({fr(est_20, 4)}).",
            f"estimate_loss_20 doit rendre un float Python, la moyenne des loss de 3 batches de get_batch(..., generator) : "
            f"attendu {fr(want_est_20, 4)}, reçu {est_20!r}.")
    verdict("B3.20", len(spy_20.calls) == 3 and all(not t and not g for t, g in spy_20.calls) and spy_20.training,
            "pendant la mesure : mode eval et sans gradient ; le modèle revient en mode train.",
            f"pendant la mesure, attendu 3 passes en mode eval sans gradient, puis le modèle en mode train ; vu "
            f"{[('train' if t else 'eval') + (', gradient' if g else '') for t, g in spy_20.calls]}, mode final "
            f"{'train' if spy_20.training else 'eval'}.")

    # 3. train_20 against the recipe of the contract: 6 iterations of a small model, a clipping that acts at every step
    probe_20 = TRAIN_20 | {"iters": 6, "warmup": 2, "eval_interval": 4, "eval_batches": 2, "batch_size": 4, "clip": 0.05}
    mine_20, ref_20 = copy.deepcopy(small_20), copy.deepcopy(small_20)
    hist_20 = train_20(mine_20, DATA.train_ids, DATA.val_ids, probe_20, torch.Generator().manual_seed(7), block_size=16)
    if not returned("B3.20", "train_20", hist_20):
        raise StopCheck
    want_hist_20 = wbgpt.train(ref_20, DATA.train_ids, DATA.val_ids, probe_20, torch.Generator().manual_seed(7), block_size=16)
    keys_ok_20 = isinstance(hist_20, dict) and all(k in hist_20 for k in want_hist_20)
    verdict("B3.20", keys_ok_20 and len(hist_20["train_loss"]) == len(hist_20["lr"]) == 6 and list(hist_20["val_iter"]) == [0, 4, 6],
            "historique : 6 loss et 6 learning rates, la validation avant la 1ʳᵉ itération, après la 4ᵉ et après la dernière.",
            "train_20 doit rendre un dict de listes train_loss et lr (une valeur par itération), val_iter et val_loss (avant "
            "la première itération, toutes les eval_interval itérations et après la dernière) ; avec 6 itérations et "
            f"eval_interval = 4, val_iter = [0, 4, 6] ; reçu {hist_20 if not keys_ok_20 else hist_20['val_iter']!r:.200}.")
    if not keys_ok_20:
        raise StopCheck
    verdict("B3.20", np.allclose(hist_20["lr"], want_hist_20["lr"], rtol=1e-9, atol=1e-12),
            "learning rate : montée linéaire de 0, puis cosinus jusqu'à min_lr (ton cosine_lr du ch. 19).",
            f"learning rates attendus {np.round(want_hist_20['lr'], 6).tolist()}, reçus {np.round(hist_20['lr'], 6).tolist()}.")
    same_20 = all(torch.allclose(a, b, rtol=1e-4, atol=1e-7) for a, b in zip(mine_20.parameters(), ref_20.parameters()))
    if same_20:
        print("✅ Ex B3.20 : après 6 itérations, les poids sont ceux de la recette du contrat (AdamW à deux groupes, "
              "learning rate de chaque itération, clipping, un get_batch par itération).")
    else:
        noclip_20 = copy.deepcopy(small_20)
        wbgpt.train(noclip_20, DATA.train_ids, DATA.val_ids, probe_20 | {"clip": 1e9}, torch.Generator().manual_seed(7), block_size=16)
        unclipped_20 = all(torch.allclose(a, b, rtol=1e-4, atol=1e-7) for a, b in zip(mine_20.parameters(), noclip_20.parameters()))
        print("❌ Ex B3.20 : après 6 itérations, tes poids ne sont pas ceux de la recette du contrat : "
              + ("ils sont ceux d'un entraînement sans clipping (la norme des gradients dépasse ici 0,05 à chaque itération)."
                 if unclipped_20 else "vérifie l'ordre de chaque itération (learning rate, get_batch, loss, gradients, "
                 "clipping, pas de l'optimiseur) et que l'évaluation tire ses batches avec un générateur à part."))
    probes_ok_20 = same_20'''

EXPERIMENT_20 = r'''model_20 = history_20 = val_char_20 = None
if not (globals().get("probes_ok_20") and globals().get("model_ok_19")):
    print("⏳ Ex B3.20 : l'entraînement attend ton modèle de B3.19 et les contrôles de la cellule précédente.")
else:
    with checked("B3.20"):
        torch.manual_seed(SEED_20)
        model_20 = MiniGPT_19(**CONFIG_20)
        started_20 = time.perf_counter()
        history_20 = train_20(model_20, DATA.train_ids, DATA.val_ids, TRAIN_20, torch.Generator().manual_seed(SEED_20))
        seconds_20 = time.perf_counter() - started_20
        val_char_20 = wbgpt.char_loss(model_20, DATA.val_ids)       # every character of the validation from index 128 on
        path_20 = wbgpt.save_checkpoint(wbgpt.learner_key(FAST_MODE), model_20, history_20, DATA.itos,
                                        extra={"char_loss": val_char_20, "seconds": seconds_20})
        print(f"{TRAIN_20['iters']} itérations en {seconds_20:.0f} s ; loss de validation estimée {fr(history_20['val_loss'][-1], 3)} "
              f"nats par token ; modèle sauvegardé dans {path_20}")
        verdict("B3.20", val_char_20 < TARGET_20,
                f"sur toute la validation, {fr(val_char_20, 3)} nats par caractère (perplexité {fr(math.exp(val_char_20), 2)}, "
                f"{fr(val_char_20 / math.log(2), 3)} bits par caractère) : sous {fr(TARGET_20, 2)}.",
                f"sur toute la validation, {fr(val_char_20, 3)} nats par caractère : attendu moins de {fr(TARGET_20, 2)} ; "
                "relance les contrôles de la cellule précédente, puis ceux de B3.19.")
        rows_20 = [(name, params, budget, loss) for name, params, budget, loss in LSTM_20]
        rows_20.append(("ton mini-GPT (" + ("mode rapide" if FAST_MODE else "mode complet") + ")", n_params(model_20),
                        f"{TRAIN_20['iters']} itérations, {seconds_20 / 60:.1f} min", val_char_20))
        print(f"\n{'modèle':<44} {'paramètres':>10}  {'budget':<30} {'nats/car.':>9} {'perplexité':>10}")
        for name, params, budget, loss in rows_20:
            print(f"{name:<44} {params:>10_}  {budget:<30} {fr(loss, 3):>9} {fr(math.exp(loss), 2):>10}".replace("_", " "))'''
