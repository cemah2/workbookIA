#!/usr/bin/env python
"""Build the two notebooks of bonus chapter B3 from a single source (used by Claude).

    python tools/chapters/build_chB3.py
    python tools/run_all_notebooks.py chapitres/b3_transformers/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/b3_transformers/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3 and of the ✏️ paper exercises
B3.1 to B3.5 (02_exercices.md). The derivations ∂ B3.6 and B3.7, the estimate 🧮 B3.8, the reflection (🗣️ B3.9,
⚖️ B3.10, 📄 B3.11) and the interview are corrected with 05_solutions.md. The answers are computed by hand formulas in
the context cell of the solutions notebook, then checked against the reference library (mylearn.attention,
mylearn.nn.regularization), PyTorch (F.scaled_dot_product_attention, nn.MultiheadAttention, nn.LayerNorm) and
transformers (GPT2Config on the meta device, no download).
Parts A and B (exercises B3.12 to B3.30) come at the second generation session (NEXT_SESSION): the NumPy bricks of
attention.py, then the mini-GPT in PyTorch.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (Paper, badge, check_unique_keys, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "B3"
FOLDER = "chapitres/b3_transformers"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np


def softmax_rows(s):
    """Stable softmax over the last axis (the keys); -inf gives a weight of exactly 0."""
    s = np.asarray(s, dtype=float)
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ✏️ B3.1: three tokens, d_k = 2 (token i is row i), no mask
X_B31 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
WQ_B31 = np.array([[0.0, 1.0], [1.0, 1.0]])
WK_B31 = np.array([[1.0, 0.0], [1.0, 2.0]])
WV_B31 = np.array([[1.0, -1.0], [0.0, 2.0]])
Q_B31, K_B31, V_B31 = X_B31 @ WQ_B31, X_B31 @ WK_B31, X_B31 @ WV_B31
RAW_B31 = Q_B31 @ K_B31.T                                           # (L, S) dot products
S_B31 = RAW_B31 / math.sqrt(2)                                      # scaled scores
A_B31 = softmax_rows(S_B31)                                         # attention map, softmax over the keys
O_B31 = A_B31 @ V_B31
A_NOSCALE_B31 = softmax_rows(RAW_B31)
A_COLUMNS_B31 = softmax_rows(S_B31.T).T                             # the classic mistake: softmax over the queries

# ✏️ B3.2: the same with a causal mask, then L = 2 new queries against S = 3 keys (KV cache)
CAUSAL_B32 = np.tril(np.ones((3, 3), dtype=bool))
TRIU_B32 = np.triu(np.ones((3, 3), dtype=bool))                     # the transposed mask of 🐛 B3.17
A_C_B32 = softmax_rows(np.where(CAUSAL_B32, S_B31, -np.inf))
O_C_B32 = A_C_B32 @ V_B31
A_ZERO_B32 = np.where(CAUSAL_B32, A_B31, 0.0)                       # the classic mistake: zeroed, not renormalized
A_NOSCALE_C_B32 = softmax_rows(np.where(CAUSAL_B32, RAW_B31, -np.inf))
A_TRIU_B32 = softmax_rows(np.where(TRIU_B32, S_B31, -np.inf))
BOTTOM_RIGHT_B32 = np.tril(np.ones((2, 3), dtype=bool), k=1)        # causal_mask(2, 3)
TOP_LEFT_B32 = np.tril(np.ones((2, 3), dtype=bool), k=0)            # is_causal=True of SDPA when L != S

# ✏️ B3.3: shapes and parameters of a multi-head attention
N_B33, T_B33, D_B33, H_B33, S_B33, HKV_B33 = 4, 10, 384, 6, 25, 2
DH_B33 = D_B33 // H_B33
MHA_B33 = 4 * D_B33 ** 2 + 4 * D_B33
GQA_B33 = 2 * (D_B33 * D_B33 + D_B33) + 2 * (D_B33 * HKV_B33 * DH_B33 + HKV_B33 * DH_B33)

# ✏️ B3.4: sinusoidal encoding, d = 8 (angular frequencies 1, 0.1, 0.01, 0.001)
D_B34 = 8
OMEGA_B34 = 10000.0 ** (-np.arange(0, D_B34, 2) / D_B34)


def pe_b34(p, omega=OMEGA_B34):
    """Row p of the sinusoidal encoding: sin in the even columns, cos in the odd ones."""
    row = np.zeros(2 * len(omega))
    row[0::2], row[1::2] = np.sin(p * omega), np.cos(p * omega)
    return row


PAIR_B34 = np.array([0.6, 0.8])                                     # (PE[p, 2], PE[p, 3]) for some real position p


def shift_b34(pair, angle):
    """The shift matrix of the fiche (203.5) applied to a (sin, cos) pair; a negative angle shifts backwards."""
    return np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]) @ pair


SHIFT_B34 = shift_b34(PAIR_B34, 10 * OMEGA_B34[1])                  # 10 positions on pair 1: 1 radian
PE_DOT_B34 = float(pe_b34(0) @ pe_b34(2))


# ✏️ B3.5: GPT-2 small (config.json of openai-community/gpt2) and the mini-GPT of the notebook
def gpt_params(n_layer, d, vocab, n_pos, tied=True):
    """Parameters of a GPT-2-style decoder: biases everywhere, F = 4d, learned positions, final layer norm."""
    attn, mlp, lns = 4 * d * d + 4 * d, 8 * d * d + 5 * d, 4 * d
    block = attn + mlp + lns
    return {"attn": attn, "mlp": mlp, "block": block, "blocks": n_layer * block, "wte": vocab * d,
            "wpe": n_pos * d, "ln_f": 2 * d,
            "total": n_layer * block + vocab * d * (1 if tied else 2) + n_pos * d + 2 * d,
            "non_embedding": n_layer * block + 2 * d}


GPT2_B35 = gpt_params(12, 768, 50257, 1024)
GPT2_UNTIED_B35 = gpt_params(12, 768, 50257, 1024, tied=False)["total"]
MINI_B35 = gpt_params(4, 128, 88, 128)
PAPER_B35 = 117e6                                                   # table 2 of the GPT-2 paper ("117M")

# quizzes and recalls with a computation
Q2C = (1 * 2 + 3 * 2) / math.sqrt(2)                                # q = (1, 3), k = (2, 2), d_k = 2
Q2D = 3 / (1 + 1 + 3)                                               # scores (0, 0, ln 3): exponentials 1, 1, 3
Q7E = 250 * 10000.0 ** (-2 * 16 / 64)                               # pair i = 16 of d_h = 64, position 250
Q8D = 2 * 512 * 2048 + 2048 + 512
Q11E = 2 * 2 * 100 * 64 * 4                                         # 2 (K and V) x layers x tokens x d x bytes
Q10E = math.log(65)
X_R3 = np.array([1.0, 2.0, 3.0, 6.0])
LN_R3 = (X_R3 - X_R3.mean()) / X_R3.std()                           # biased variance, as in your layer_norm
'''

LIBRARY_CHECKS = '''
# The same answers from the libraries (no download): mylearn (reference), PyTorch, transformers
import torch
import torch.nn.functional as F
from torch import nn

A = mylearn.attention
# ✏️ B3.1 and B3.2: scaled_dot_product_attention of the reference and of PyTorch, with and without the masks
out, w = A.scaled_dot_product_attention(Q_B31, K_B31, V_B31)
assert np.allclose(w, A_B31) and np.allclose(out, O_B31)
tq, tk, tv = (torch.tensor(a) for a in (Q_B31, K_B31, V_B31))
assert np.allclose(F.scaled_dot_product_attention(tq, tk, tv).numpy(), O_B31)
assert np.allclose(A.scaled_dot_product_attention(Q_B31, K_B31, V_B31, scale=1.0)[1], A_NOSCALE_B31)
assert np.array_equal(A.causal_mask(3), CAUSAL_B32) and int(A.causal_mask(3).sum()) == 6
out_c, w_c = A.scaled_dot_product_attention(Q_B31, K_B31, V_B31, mask=A.causal_mask(3))
assert np.allclose(w_c, A_C_B32) and np.allclose(out_c, O_C_B32)
assert np.allclose(F.scaled_dot_product_attention(tq, tk, tv, is_causal=True).numpy(), O_C_B32)
assert np.array_equal(A.causal_mask(2, 3), BOTTOM_RIGHT_B32)
cache_out, _ = A.scaled_dot_product_attention(Q_B31[1:], K_B31, V_B31, mask=A.causal_mask(2, 3))
assert np.allclose(cache_out, O_C_B32[1:])                          # the KV cache gives rows 1 and 2 of the full run
top_left = F.scaled_dot_product_attention(tq[1:], tk, tv, is_causal=True).numpy()
assert not np.allclose(top_left, O_C_B32[1:])
assert np.allclose(top_left[0], tv[0].numpy())                      # the first new query sees only key 0
from torch.nn.attention.bias import causal_lower_right
assert np.allclose(F.scaled_dot_product_attention(tq[1:], tk, tv, attn_mask=causal_lower_right(2, 3)).numpy(),
                   O_C_B32[1:])
w_triu = A.scaled_dot_product_attention(Q_B31, K_B31, V_B31, mask=TRIU_B32)[1]
assert w_triu[2, 0] == 0 and w_triu[2, 1] == 0 and np.all(w_triu[0] > 0)

# ✏️ B3.3: nn.MultiheadAttention, split_heads and multi_head_attention of the reference, GQA with nn.Linear
mha = nn.MultiheadAttention(D_B33, H_B33)
assert sum(p.numel() for p in mha.parameters()) == MHA_B33 and tuple(mha.in_proj_weight.shape) == (3 * D_B33, D_B33)
mha12 = nn.MultiheadAttention(D_B33, 12)
assert sum(p.numel() for p in mha12.parameters()) == MHA_B33
rng = np.random.default_rng(0)
params = {f"W_{c}": rng.normal(size=(D_B33, D_B33)) / 20 for c in "qkvo"} | {f"b_{c}": np.zeros(D_B33) for c in "qkvo"}
x = rng.normal(size=(N_B33, T_B33, D_B33))
assert A.split_heads(x @ params["W_q"], H_B33).shape == (N_B33, H_B33, T_B33, DH_B33)
o, w = A.multi_head_attention(x, x, x, params, n_heads=H_B33)
assert o.shape == (N_B33, T_B33, D_B33) and w.shape == (N_B33, H_B33, T_B33, T_B33) and w.size == 2400
enc = rng.normal(size=(N_B33, S_B33, D_B33))
assert A.multi_head_attention(x, enc, enc, params, n_heads=H_B33)[1].shape == (N_B33, H_B33, T_B33, S_B33)
gqa = nn.ModuleList([nn.Linear(D_B33, D_B33), nn.Linear(D_B33, HKV_B33 * DH_B33), nn.Linear(D_B33, HKV_B33 * DH_B33),
                     nn.Linear(D_B33, D_B33)])
assert sum(p.numel() for p in gqa.parameters()) == GQA_B33

# ✏️ B3.4: positional_encoding of the reference; the shift matrix on real positions and on integer ones
PE = A.positional_encoding(200, D_B34)
assert np.allclose(PE[25], pe_b34(25)) and np.isclose(PE[0] @ PE[2], PE_DOT_B34) and np.isclose(PE[100] @ PE[102], PE_DOT_B34)
p_real = math.asin(0.6) / OMEGA_B34[1]                               # the position of the statement, about 6.4
assert np.allclose(pe_b34(p_real)[2:4], PAIR_B34) and np.allclose(pe_b34(p_real + 10)[2:4], SHIFT_B34)
for p in (0, 7, 33):
    assert np.allclose(shift_b34(PE[p, 2:4], 10 * OMEGA_B34[1]), PE[p + 10, 2:4])
assert round(2 * math.pi / OMEGA_B34[2]) == 628

# ✏️ B3.5: GPT-2 small and the mini-GPT, built without their weights (meta device)
from transformers import GPT2Config, GPT2LMHeadModel
with torch.device("meta"):
    gpt2 = GPT2LMHeadModel(GPT2Config())
    gpt2_untied = GPT2LMHeadModel(GPT2Config(tie_word_embeddings=False))
    mini = GPT2LMHeadModel(GPT2Config(n_layer=4, n_head=4, n_embd=128, n_positions=128, vocab_size=88,
                                       bos_token_id=0, eos_token_id=0))
count = lambda model: sum(p.numel() for p in model.parameters())   # noqa: E731
assert count(gpt2) == GPT2_B35["total"] and count(gpt2_untied) == GPT2_UNTIED_B35 and count(mini) == MINI_B35["total"]
assert count(gpt2.transformer.h[0]) == GPT2_B35["block"] and count(gpt2.transformer.h[0].attn) == GPT2_B35["attn"]
assert count(gpt2.transformer.h[0].mlp) == GPT2_B35["mlp"]
assert count(gpt2) - count(gpt2.transformer.wte) - count(gpt2.transformer.wpe) == GPT2_B35["non_embedding"]
block = nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
assert count(block) == GPT2_B35["block"]

# quizzes and recalls with a computation
qa = torch.tensor([[1.0, 3.0]])
assert np.isclose((qa @ torch.tensor([[2.0], [2.0]])).item() / math.sqrt(2), Q2C)
assert np.isclose(torch.softmax(torch.tensor([0.0, 0.0, math.log(3)]), 0)[2].item(), Q2D)
assert int(A.causal_mask(5).sum()) == 15 and int(A.causal_mask(1, 9).sum()) == 9
angles = np.arange(251)[:, None] * 10000.0 ** (-2 * np.arange(32) / 64)
assert np.isclose(angles[250, 16], Q7E)
assert count(nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512))) == Q8D
assert count(nn.LayerNorm(1600)) == 3200 and count(nn.LayerNorm(768)) == 1536
assert np.allclose(mylearn.nn.regularization.layer_norm(X_R3[None], eps=0.0)[0], LN_R3)
assert np.allclose(F.layer_norm(torch.tensor(X_R3), (4,), eps=0.0).numpy(), LN_R3)
print("✅ the paper answers agree with mylearn (reference), PyTorch and transformers")
'''

mylearn = importlib.import_module("mylearn_ref")   # the builder computes the mistakes with the same helpers
for _module in ("attention", "nn.regularization"):
    importlib.import_module(f"mylearn_ref.{_module}")
exec(PAPER_CONTEXT)

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
WHOLE = "a whole number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
DEC4 = "a number, 4 decimals"
SHAPE = 'a shape as a string, e.g. "(2, 3)"'
VECTOR = "a list of numbers, e.g. [0.5, -1, 3]"
VECTOR3 = "a list of numbers, 3 decimals each"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    Paper("B3.Q1", "Le goulot de l'encodeur-décodeur récurrent : pourquoi un seul vecteur ne suffit pas", [
        ("a", LETTER, '"C"', letters("C", {"A": "relis « Traduire avec deux RNN » (§203.1) : le décodeur de 2014 relit-il la source ?",
                                           "B": "relis « Traduire avec deux RNN » : par où passent les mots source ?"})),
        ("b", TF, "False", r'''mistakes={"relis « Le goulot » (§203.1) : la taille de c dépend-elle de la phrase ?": True}'''),
        ("c", WHOLE, "12 * 9", r'''mistakes={"relis la question : on demande les poids de tous les pas de la traduction": 12,
          "relis la question : on demande les poids de tous les mots source, à chaque pas": 9,
          "tu as additionné : combien de poids à chaque pas, et combien de pas ?": 21}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "relis l'étape 2 de l'idée de Bahdanau : sur quoi porte la softmax ?",
                                           "C": "relis le paragraphe qui suit les quatre étapes : l'alignement est-il un choix d'un seul mot ?"})),
        ("e", TF, "True", r'''mistakes={"relis le paragraphe sur l'alignement souple (§203.1) : d'où viennent les poids ?": False}'''),
    ]),
    Paper("B3.Q2", "Requête, clé, valeur : un dictionnaire où l'on cherche « à peu près »", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis « Un dictionnaire où l'on cherche à peu près » : rend-on une seule valeur ?",
                                           "C": "relis le rôle de chaque vecteur : qu'est-ce que l'attention rend ?"})),
        ("b", TF, "False", r'''mistakes={"relis « D'où viennent les requêtes, les clés et les valeurs » : combien de projections ?": True}'''),
        ("c", DEC2, "Q2C", r'''decimals=2, mistakes={"relis la question : le score est mis à l'échelle": 8.0,
          "on divise par la racine de d_k, pas par d_k": 4.0}'''),
        ("d", DEC1, "Q2D", r'''decimals=1, mistakes={"les scores passent par l'exponentielle avant d'être normalisés": 1.0,
          "compare les exponentielles des trois scores : sont-elles égales ?": 1 / 3}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "c'est la forme des poids de la requête : on demande sa sortie",
                                           "B": "relis l'étape 5 de « La formule » : que devient la somme pondérée ?"})),
    ]),
    Paper("B3.Q3", "Pourquoi diviser les scores par √d_k ?", [
        ("a", WHOLE, "math.isqrt(256)", r'''mistakes={"c'est la variance : on demande l'écart-type": 256,
          "relis la question : d_k = 256": 8}'''),
        ("b", TF, "True", r'''mistakes={"relis « Pourquoi diviser par √d_k » : que vaut Var(aX) avec a = 1/√d_k ?": False}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "relis le paragraphe sur la softmax saturée : quel poids prend le plus grand score ?",
                                           "C": "une softmax peut-elle donner un poids négatif ?"})),
        ("d", LETTER, '"C"', letters("C", {"A": "l'échelle des scores compte avant l'exponentielle : relis l'effet de la variance des scores sur la softmax",
                                           "B": "diviser par plus que √d_k rend-il les scores plus grands ou plus petits ?"})),
        ("e", TF, "False", r'''mistakes={"relis l'argument de la variance : de quels vecteurs le score est-il fait ?": True}'''),
    ]),
    Paper("B3.Q4", "Masque causal : qui a le droit de regarder qui ?", [
        ("a", WHOLE, "int(np.tril(np.ones((5, 5))).sum())", r'''mistakes={"compte à nouveau : la diagonale est-elle autorisée ?": 10,
          "le masque causal interdit des cases : lesquelles ?": 25,
          "compte toutes les cases vraies, pas seulement la diagonale": 5}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "relis la phrase sous la formule du masque causal : combien de cases vraies a la ligne i ?",
                                           "C": "le masque causal cache-t-il le futur ou le passé ?"})),
        ("c", TF, "False", r'''mistakes={"relis la conséquence utile du masque causal (§203.3)": True}'''),
        ("d", WHOLE, "9", r'''mistakes={"c'est l'alignement en haut à gauche de is_causal=True : relis celui de causal_mask": 1,
          "la dernière requête voit-elle sa propre clé ?": 8}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "c'est le rôle d'un autre masque : relis « Le masque de padding »",
                                           "C": "relis le rôle du masque : change-t-il le nombre de scores calculés ?"})),
    ]),
    Paper("B3.Q5", "Auto-attention ou attention croisée : d'où viennent Q, K et V ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis la formule de l'attention croisée (§203.3) : d'où vient Q ?",
                                           "C": "relis « Auto-attention et attention croisée » : les clés et les requêtes viennent-elles de la même séquence ?"})),
        ("b", SHAPE, '"(7, 12)"', r'''mistakes={"relis l'ordre des axes de Q Kᵀ : les lignes sont-elles les requêtes ou les clés ?": "(12, 7)",
          "les clés viennent de la source : combien de colonnes ?": "(7, 7)",
          "les requêtes viennent de la traduction : combien de lignes ?": "(12, 12)"}'''),
        ("c", TF, "False", r'''mistakes={"relis le tableau des familles (§203.7) : quel masque dans l'attention croisée ?": True}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "les trois arguments ne sont pas les mêmes : relis l'ordre query, key, value",
                                           "B": "le premier argument fournit les requêtes : qui regarde qui ?"})),
        ("e", TF, "True", r'''mistakes={"relis la définition de l'auto-attention (§203.3)": False}'''),
    ]),
    Paper("B3.Q6", "Plusieurs têtes pour le prix d'une : que gagne-t-on ?", [
        ("a", WHOLE, "512 // 16", r'''mistakes={"tu as multiplié : relis la définition de d_h": 8192,
          "recompte avec H = 16": 64}'''),
        ("b", TF, "False", r'''mistakes={"relis « Compter les paramètres » (§203.4)": True}'''),
        ("c", WHOLE, "32 // 4", r'''mistakes={"c'est le nombre de têtes de clés : on demande un rapport": 4,
          "on demande un rapport, pas une différence": 28,
          "on demande un rapport, pas un produit": 128}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "relis le piège classique « un reshape direct » : y a-t-il une erreur ?",
                                           "C": "relis le piège classique « un reshape direct » : quelle tête reçoit quelles features ?"})),
        ("e", TF, "True", r'''mistakes={"relis le paragraphe sur repeat_interleave (§203.4), et écris les deux ordres de têtes": False}'''),
    ]),
    Paper("B3.Q7", "Sans encodage de position, l'attention voit-elle l'ordre des mots ?", [
        ("a", TF, "True", r'''mistakes={"relis « L'attention ne voit pas l'ordre » (§203.5)": False}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "sans position, l'attention sait-elle qui est voisin de qui ?",
                                           "C": "la sortie est une moyenne de valeurs : pourquoi serait-elle nulle ?"})),
        ("c", LETTER, '"B"', letters("B", {"A": "relis « Les positions apprises » : quelle est la limite de la table ?",
                                           "C": "relis « Les positions apprises » : la table a-t-elle une règle pour les positions trop grandes ?"})),
        ("d", LETTER, '"C"', letters("C", {"A": "relis le premier paragraphe de « RoPE » : touche-t-il aux embeddings ?",
                                           "B": "relis les trois points pratiques de RoPE : quels vecteurs sont tournés ?"})),
        ("e", DEC1, "Q7E", r'''decimals=1, mistakes={"vérifie l'exposant de la fréquence θ_i (fiche, « RoPE »)": 250 * 10000.0 ** (-16 / 64),
          "relis la question : la paire i = 16, pas la paire 0": 250.0,
          "on demande des radians, pas des degrés": math.degrees(Q7E)}'''),
    ]),
    Paper("B3.Q8", "Résiduel, pré-LN ou post-LN : ce qui rend un Transformer profond entraînable", [
        ("a", LETTER, '"C"', letters("C", {"A": "relis les deux formules du §203.6 : où se place la layer norm en pré-LN ?",
                                           "B": "relis les deux formules du §203.6 : que manque-t-il ?"})),
        ("b", TF, "True", r'''mistakes={"relis le paragraphe qui suit la figure des deux blocs (§203.6)": False}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "relis le paragraphe sur Xiong et coll. : où les gradients sont-ils grands ?",
                                           "C": "relis le paragraphe sur Xiong et coll. : est-ce le pré-LN ou le post-LN qui supporte un warmup court ?"})),
        ("d", WHOLE, "Q8D", r'''mistakes={"n'oublie pas les biais des deux couches denses": 2 * 512 * 2048,
          "vérifie la taille du biais de la seconde couche": 2 * 512 * 2048 + 2 * 2048,
          "vérifie la taille du biais de la première couche": 2 * 512 * 2048 + 2 * 512,
          "le MLP a deux couches denses": 512 * 2048 + 2048,
          "n'oublie pas le biais de la seconde couche dense": 2 * 512 * 2048 + 2048,
          "n'oublie pas le biais de la première couche dense": 2 * 512 * 2048 + 512}'''),
        ("e", TF, "False", r'''mistakes={"relis le rôle de la connexion résiduelle (§203.6 et le rappel R3)": True}'''),
    ]),
    Paper("B3.Q9", "BERT, GPT, T5 : quel masque, quel objectif, quel usage ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "c'est l'objectif d'un décodeur : relis « L'encodeur : BERT »",
                                           "C": "c'est l'objectif de T5 : relis « L'encodeur : BERT »"})),
        ("b", WHOLE, "200 * 15 // 100", r'''mistakes={"relis la question : 15 % de 200 tokens": 15,
          "relis la question : on demande toutes les positions choisies": 24,
          "on demande les positions choisies, pas les autres": 170}'''),
        ("c", WHOLE, "30 * 80 // 100", r'''mistakes={"relis la question : on demande celles qui deviennent [MASK]": 30,
          "relis la question : parmi les positions choisies seulement": 160,
          "relis la proportion des positions choisies qui deviennent [MASK] (§203.7)": 3,
          "relis les trois sorts d'une position choisie (§203.7) : un token tiré au hasard est-il un [MASK] ?": 27}'''),
        ("d", TF, "True", r'''mistakes={"relis le paragraphe sur le décodeur (§203.7) : que permet le prompt ?": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "relis la ligne « Sortie typique » du tableau des familles",
                                           "C": "relis la ligne « Sortie typique » du tableau des familles : que produit cette famille ?"})),
    ]),
    Paper("B3.Q10", "Décalage d'un token : ce que chaque position apprend à prédire", [
        ("a", LETTER, '"B"', letters("B", {"A": "les cibles sont-elles les entrées elles-mêmes ? Relis le tableau « chat » (§203.7)",
                                           "C": "relis le tableau « chat » : combien de cibles pour T entrées ?"})),
        ("b", WHOLE, "256", r'''mistakes={"relis le tableau « chat » : la dernière position a-t-elle une cible ?": 255,
          "relis la question : T = 256": 128}'''),
        ("c", TF, "False", r'''mistakes={"relis la définition du teacher forcing (§203.7) : quelle séquence est donnée en entrée ?": True}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "le modèle voit-il la cible dans son entrée ?",
                                           "B": "une loss qui diverge vient d'un learning rate ou d'une initialisation : ici, que voit le modèle ?"})),
        ("e", DEC2, "Q10E", r'''decimals=2, mistakes={"le logarithme est naturel ici, pas en base 2": math.log2(65),
          "le logarithme est naturel ici, pas décimal": math.log10(65),
          "on demande une loss, pas une probabilité": 1 / 65}'''),
    ]),
    Paper("B3.Q11", "KV cache : ce que l'on garde et pourquoi la génération s'accélère", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis « Le KV cache » : les anciennes requêtes resservent-elles ?",
                                           "C": "relis « Le KV cache » : que garde-t-on pour chaque couche ?"})),
        ("b", WHOLE, "1", r'''mistakes={"le nouveau token a-t-il besoin d'une requête ?": 0,
          "relis l'étape 1 du KV cache : combien de tokens passent dans le modèle ?": 2}'''),
        ("c", TF, "False", r'''mistakes={"relis la vérification de la fiche (une couche, cache contre calcul complet) : les sorties diffèrent-elles ?": True}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "relis « Ce que coûte le cache » : de quoi dépend la formule ?",
                                           "C": "relis la formule du cache : T y apparaît-il au carré ?"})),
        ("e", WHOLE, "Q11E", r'''mistakes={"recompte les facteurs de la formule du cache : il en manque un de 2": Q11E // 2,
          "tu as compté des valeurs, ou oublié deux facteurs : relis la formule du cache": Q11E // 4,
          "on demande des octets, pas des bits": Q11E * 8}'''),
    ]),
    Paper("B3.Q12", "Coût quadratique du contexte et ce que change FlashAttention", [
        ("a", WHOLE, "8 ** 2", r'''mistakes={"relis « La mémoire : la matrice T × T » : comment la mémoire croît-elle avec T ?": 8,
          "le carré de la longueur, pas son cube": 512,
          "relis la question : on multiplie la longueur par 8": 4}'''),
        ("b", TF, "False", r'''mistakes={"relis le résultat de FlashAttention (§203.10) : est-ce une approximation ?": True}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "FlashAttention change-t-il le modèle ?",
                                           "B": "relis le résultat de FlashAttention : que devient le nombre de FLOPs ?"})),
        ("d", WHOLE, "12 * 768", r'''mistakes={"le seuil n'est pas T = d : relis le rapport des deux coûts (§203.10)": 768,
          "relis la question : dans un décodeur causal, une requête voit-elle toutes les clés ?": 6 * 768,
          "compte les deux produits de l'attention : les scores, puis la somme pondérée des valeurs": 24 * 768}'''),
        ("e", TF, "True", r'''mistakes={"relis l'exemple de la fiche pour d = 4 096 (§203.10)": False}'''),
    ]),
    Paper("B3.R1", "De quoi dépend le nombre de tokens d'un texte ?", [
        ("a", TF, "False", r'''mistakes={"compare le nombre de tokens d'un même texte avec deux tokenizers de B2": True}'''),
        ("b", WHOLE, "4000 * 10 // 32", r'''mistakes={"relis la définition de la compression (B2) : des octets par token": 12800},
          fractional="on demande un nombre de tokens"'''),
        ("c", LETTER, '"C"', letters("C", {"A": "relis B2 (§202.3) : que fait une fusion au nombre de tokens du corpus ?",
                                           "B": "relis B2 : une fusion de plus change-t-elle le découpage ?"})),
        ("d", TF, "True", r'''mistakes={"relis la fertilité du français avec GPT-2 (B2, §202.6)": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "relis la compression de B2 : combien de caractères par token BPE ?",
                                           "C": "compare la longueur moyenne d'un token BPE à celle d'un caractère"})),
    ]),
    Paper("B3.R2", "Pourquoi le gradient s'évanouit dans un RNN déroulé", [
        ("a", DEC4, "0.9 ** 50", r'''decimals=4, mistakes={"relis le ch. 22 : le gradient traverse 50 jacobiennes l'une après l'autre": 0.9 * 50,
          "relis la question : 50 pas": 0.9}'''),
        ("b", TF, "False", r'''mistakes={"relis le ch. 22 : le LSTM atténue-t-il ou supprime-t-il le problème ?": True}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "c'est le chemin dans un RNN : relis le tableau de §203.1",
                                           "C": "relis le tableau de §203.1 : chemin entre deux tokens éloignés"})),
        ("d", LETTER, '"A"', letters("A", {"B": "le dropout régularise : relis ce que le ch. 22 fait contre l'explosion",
                                           "C": "la température sert à la génération"})),
        ("e", TF, "False", r'''mistakes={"relis « Et si l'on supprimait les RNN ? » (§203.1)": True}'''),
    ]),
    Paper("B3.R3", "Layer norm et connexion résiduelle, à quoi sert chacune ?", [
        ("a", DEC3, "LN_R3[0]", r'''decimals=3, mistakes={"tu as divisé par la variance : relis la formule de ta layer_norm": -2 / X_R3.var(),
          "relis la docstring de ta layer_norm : quelle variance utilise-t-elle ?": -2 / X_R3.std(ddof=1),
          "vérifie le signe": -LN_R3[0],
          "as-tu soustrait la moyenne ?": 1 / X_R3.std()}'''),
        ("b", TF, "False", r'''mistakes={"relis le ch. 20 : sur quel axe la layer norm calcule-t-elle ses statistiques ?": True}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "dérive y terme par terme : que donne le premier terme ?",
                                           "C": "dérive y terme par terme : que donne le second terme ?"})),
        ("d", TF, "True", r'''mistakes={"relis le 🔗 Lien avec le ch. 20 (§203.6)": False}'''),
        ("e", WHOLE, "2 * 1600", r'''mistakes={"relis le ch. 20 : combien de vecteurs de paramètres a une layer norm ?": 1600,
          "relis : γ et β sont-ils des matrices ?": 2 * 1600 ** 2}'''),
    ]),
    Paper("B3.1", "L'attention à la main : trois tokens, d_k = 2, scores, softmax et sortie", [
        ("a", VECTOR, "Q_B31[2].tolist()", r'''mistakes={"les tokens sont numérotés à partir de 0 : quelle ligne de X est x_2 ?": Q_B31[1].tolist(),
          "relis la question : on demande la requête, avec W_Q": K_B31[2].tolist()}'''),
        ("b", DEC3, "S_B31[2, 1]", r'''decimals=3, mistakes={"relis la définition de s_ij dans l'énoncé : le score est mis à l'échelle": RAW_B31[2, 1],
          "on divise par la racine de d_k, pas par d_k": RAW_B31[2, 1] / 2,
          "vérifie la clé : les tokens sont numérotés à partir de 0": S_B31[2, 2],
          "c'est le score de la requête 1 pour la clé 2 : relis l'ordre des indices": S_B31[1, 2]}'''),
        ("c", DEC3, "A_B31[2, 1]", r'''decimals=3, mistakes={"relis la définition de s_ij : la division par √d_k": A_NOSCALE_B31[2, 1],
          "la softmax porte sur les clés (chaque ligne), pas sur les requêtes": A_COLUMNS_B31[2, 1],
          "vérifie la clé : les tokens sont numérotés à partir de 0": A_B31[2, 2],
          "c'est le poids de la requête 1 pour la clé 2 : relis l'ordre des indices": A_B31[1, 2]}'''),
        ("d", VECTOR3, "O_B31[2].tolist()", r'''decimals=3, mistakes={"relis la définition de s_ij : la division par √d_k": (A_NOSCALE_B31[2] @ V_B31).tolist(),
          "la sortie moyenne les valeurs, pas les clés": (A_B31[2] @ K_B31).tolist(),
          "la softmax porte sur les clés (chaque ligne), pas sur les requêtes": (A_COLUMNS_B31[2] @ V_B31).tolist()}'''),
        ("e", DEC3, "A_NOSCALE_B31[2, 2]", r'''decimals=3, mistakes={"relis la question : ici, sans la division par √d_k": A_B31[2, 2],
          "la softmax porte sur les clés (chaque ligne), pas sur les requêtes": softmax_rows(RAW_B31.T).T[2, 2]}'''),
        ("f", TF, "False", r'''mistakes={"compare les scores des clés 1 et 2 pour la requête 0": True}'''),
        ("g", DEC3, "float(A_B31[:, 0].sum())", r'''decimals=3, mistakes={"relis la question : on additionne la colonne 0, sur les trois requêtes": 1.0,
          "relis la définition de s_ij : la division par √d_k": float(A_NOSCALE_B31[:, 0].sum())}'''),
    ]),
    Paper("B3.2", "Le même calcul avec un masque causal", [
        ("a", WHOLE, "int(CAUSAL_B32.sum())", r'''mistakes={"la diagonale est-elle autorisée ?": 3,
          "le masque causal interdit des cases : lesquelles ?": 9}'''),
        ("b", VECTOR3, "A_C_B32[1].tolist()", r'''decimals=3, mistakes={"un masque ne se contente pas de mettre des poids à 0 : relis l'exemple chiffré du §203.3": A_ZERO_B32[1].tolist(),
          "relis la définition de s_ij : la division par √d_k": A_NOSCALE_C_B32[1].tolist(),
          "relis la ligne 1 du masque causal : quelles clés sont autorisées ?": A_TRIU_B32[1].tolist(),
          "relis la question : avec le masque": A_B31[1].tolist()}'''),
        ("c", VECTOR3, "O_C_B32[1].tolist()", r'''decimals=3, mistakes={"les poids de la ligne masquée font-ils 1 ? Relis l'exemple chiffré du §203.3": (A_ZERO_B32[1] @ V_B31).tolist(),
          "relis la question : avec le masque": O_B31[1].tolist(),
          "relis la ligne 1 du masque causal : quelles clés sont autorisées ?": (A_TRIU_B32[1] @ V_B31).tolist()}'''),
        ("d", VECTOR3, "O_C_B32[0].tolist()", r'''decimals=3, mistakes={"relis la question : avec le masque": O_B31[0].tolist(),
          "un masque ne se contente pas de mettre des poids à 0 : relis l'exemple chiffré du §203.3": (A_ZERO_B32[0] @ V_B31).tolist()}'''),
        ("e", TF, "True", r'''mistakes={"quelles clés le masque causal retire-t-il à la dernière requête ?": False}'''),
        ("f", WHOLE, "int(BOTTOM_RIGHT_B32[0].sum())", r'''mistakes={"relis la question : on demande le masque de ta librairie, aligné en bas à droite": 1,
          "relis la docstring de causal_mask : la requête 0 est le token de quelle position ?": 3}'''),
        ("g", WHOLE, "int(TOP_LEFT_B32[0].sum())", r'''mistakes={"relis la question : on demande l'alignement de is_causal=True quand L ≠ S": 2,
          "relis l'encadré 🕰️ sur F.scaled_dot_product_attention : où s'aligne le triangle ?": 3}'''),
        ("h", TF, "False", r'''mistakes={"écris la ligne 2 du masque transposé : quelles clés autorise-t-elle ?": True}'''),
    ]),
    Paper("B3.3", "Suivre les formes dans l'attention multi-têtes et compter ses paramètres", [
        ("a", SHAPE, '"(4, 10, 384)"', r'''mistakes={"n'oublie pas l'axe des séquences du batch": "(10, 384)",
          "une seule matrice W_Q pour toutes les têtes : relis « Une seule matrice pour toutes les têtes »": "(4, 10, 64)",
          "relis la question : avant split_heads": "(4, 6, 10, 64)"}'''),
        ("b", SHAPE, '"(4, 6, 10, 64)"', r'''mistakes={"relis la docstring de split_heads : quel axe vient juste après N ?": "(4, 10, 6, 64)",
          "relis la docstring de split_heads : quel axe vient en premier ?": "(6, 4, 10, 64)",
          "relis : une tête a-t-elle toutes les features ?": "(4, 6, 10, 384)"}'''),
        ("c", SHAPE, '"(4, 6, 10, 10)"', r'''mistakes={"relis la docstring de multi_head_attention : une carte par tête ?": "(4, 10, 10)",
          "les poids ont une colonne par clé : relis l'ordre des axes de Q Kᵀ": "(4, 6, 10, 64)",
          "c'est la forme de la sortie : on demande les poids": "(4, 10, 384)"}'''),
        ("d", WHOLE, "N_B33 * H_B33 * T_B33 * T_B33", r'''mistakes={"compte toutes les têtes": 400,
          "compte toutes les séquences du batch": 600,
          "relis : combien de scores pour une requête ?": 240}'''),
        ("e", WHOLE, "MHA_B33", r'''mistakes={"n'oublie pas les quatre biais": 4 * D_B33 ** 2,
          "n'oublie pas la projection de sortie": 3 * D_B33 ** 2 + 3 * D_B33,
          "relis la forme de W_Q dans l'énoncé : (d, d)": 4 * D_B33 * DH_B33 + 4 * DH_B33}'''),
        ("f", TF, "True", r'''mistakes={"relis « Compter les paramètres » (§203.4)": False}'''),
        ("g", SHAPE, '"(1152, 384)"', r'''mistakes={"relis l'exemple de nn.MultiheadAttention(8, 2) (§203.4) : dans quel sens les matrices sont-elles rangées ?": "(384, 1152)",
          "relis : combien de matrices dans in_proj_weight ?": "(384, 384)",
          "relis : où est rangée W_O ?": "(1536, 384)"}'''),
        ("h", WHOLE, "GQA_B33", r'''mistakes={"relis la question : les clés et les valeurs n'ont que 2 têtes": MHA_B33,
          "n'oublie pas les biais": GQA_B33 - 2 * D_B33 - 2 * HKV_B33 * DH_B33,
          "relis la question : W_Q et W_O sont comprises": 2 * (D_B33 * HKV_B33 * DH_B33 + HKV_B33 * DH_B33),
          "relis la forme de W_K et de W_V : (384, 128)": 2 * (D_B33 * D_B33 + D_B33) + 2 * (D_B33 * DH_B33 + DH_B33)}'''),
        ("i", SHAPE, '"(4, 6, 10, 25)"', r'''mistakes={"relis l'ordre des axes de Q Kᵀ : les lignes sont-elles les requêtes ou les clés ?": "(4, 6, 25, 10)",
          "relis la question : les clés viennent des 25 tokens de l'encodeur": "(4, 6, 10, 10)",
          "relis la docstring de multi_head_attention : une carte par tête ?": "(4, 10, 25)"}'''),
    ]),
    Paper("B3.4", "Encodage sinusoïdal : quelques valeurs et l'effet d'un décalage de position", [
        ("a", WHOLE, "round(2 * math.pi / OMEGA_B34[2])", r'''mistakes={"relis la période d'une horloge de fréquence angulaire ω (§203.5)": 100,
          "relis la question : la paire i = 2 (colonnes 4 et 5)": 63,
          "vérifie ω_2 : l'exposant 2i/d avec i = 2": 6283},
          fractional="arrondis la période à l'entier le plus proche"'''),
        ("b", DEC3, "pe_b34(25)[0]", r'''decimals=3, mistakes={"les angles sont en radians": math.sin(math.radians(25)),
          "la colonne 0 est un sinus": math.cos(25),
          "vérifie le signe": -math.sin(25)}'''),
        ("c", DEC4, "pe_b34(25)[2]", r'''decimals=4, mistakes={"vérifie la paire : la colonne 2 est le sinus de la paire i = 1": math.sin(0.25),
          "relis la question : les colonnes paires sont des sinus": math.cos(2.5),
          "les angles sont en radians": math.sin(math.radians(2.5))}'''),
        ("d", DEC4, "pe_b34(25)[3]", r'''decimals=4, mistakes={"relis la question : les colonnes impaires sont des cosinus": math.sin(2.5),
          "vérifie la paire : la colonne 3 est le cosinus de la paire i = 1": math.cos(0.25),
          "vérifie le signe": -math.cos(2.5),
          "les angles sont en radians": math.cos(math.radians(2.5))}'''),
        ("e", DEC4, "SHIFT_B34[0]", r'''decimals=4, mistakes={"relis la matrice du décalage de la fiche : vérifie le signe de chaque terme": shift_b34(PAIR_B34, -1.0)[0],
          "le décalage de la paire 1 se fait à sa propre fréquence, ω_1": shift_b34(PAIR_B34, 10.0)[0],
          "relis la question : on demande la colonne 2, le sinus": SHIFT_B34[1]}'''),
        ("f", DEC4, "SHIFT_B34[1]", r'''decimals=4, mistakes={"relis la matrice du décalage de la fiche : vérifie le signe de chaque terme": shift_b34(PAIR_B34, -1.0)[1],
          "le décalage de la paire 1 se fait à sa propre fréquence, ω_1": shift_b34(PAIR_B34, 10.0)[1],
          "relis la question : on demande la colonne 3, le cosinus": SHIFT_B34[0]}'''),
        ("g", DEC4, "PE_DOT_B34", r'''decimals=4, mistakes={"relis la question : les deux positions sont différentes": 4.0,
          "les angles sont en radians": float(pe_b34(0) @ pe_b34(2, np.radians(OMEGA_B34))),
          "vérifie les fréquences : l'exposant est 2i/d": float(pe_b34(0) @ pe_b34(2, 10000.0 ** (-np.arange(4) / 8)))}'''),
        ("h", TF, "True", r'''mistakes={"relis la propriété du produit scalaire de deux encodages (§203.5)": False}'''),
    ]),
    Paper("B3.5", "Compter les paramètres d'un GPT et retrouver les 124 millions de GPT-2 small", [
        ("a", WHOLE, 'GPT2_B35["attn"]', r'''mistakes={"n'oublie pas les quatre biais": 4 * 768 ** 2,
          "n'oublie pas la projection de sortie": 3 * 768 ** 2 + 3 * 768}'''),
        ("b", WHOLE, 'GPT2_B35["mlp"]', r'''mistakes={"n'oublie pas les biais des deux couches denses": 8 * 768 ** 2,
          "vérifie la taille du biais de la première couche": 8 * 768 ** 2 + 2 * 768,
          "vérifie la taille du biais de la seconde couche": 8 * 768 ** 2 + 2 * 3072,
          "relis la largeur du MLP : F = 4d": 2 * 768 ** 2 + 2 * 768}'''),
        ("c", WHOLE, 'GPT2_B35["block"]', r'''mistakes={"n'oublie pas les deux layer norms": GPT2_B35["attn"] + GPT2_B35["mlp"],
          "relis le ch. 20 : combien de vecteurs de paramètres par layer norm ?": GPT2_B35["attn"] + GPT2_B35["mlp"] + 2 * 768,
          "on demande le compte exact, pas l'approximation 12d²": 12 * 768 ** 2}'''),
        ("d", WHOLE, 'GPT2_B35["blocks"]', r'''mistakes={"on demande le compte exact, pas l'approximation 12d²": 12 * 12 * 768 ** 2,
          "relis la question : les 12 blocs seuls": GPT2_B35["non_embedding"]}'''),
        ("e", WHOLE, 'GPT2_B35["wte"] + GPT2_B35["wpe"]', r'''mistakes={"relis la question : la table des positions aussi": GPT2_B35["wte"],
          "relis la question : la table des tokens aussi": GPT2_B35["wpe"],
          "la couche de sortie est liée : elle ne compte pas une seconde fois": 2 * GPT2_B35["wte"] + GPT2_B35["wpe"]}'''),
        ("f", WHOLE, 'GPT2_B35["total"]', r'''mistakes={"relis l'énoncé : la couche de sortie réutilise la table des tokens": GPT2_UNTIED_B35,
          "n'oublie pas la layer norm finale": GPT2_B35["total"] - 2 * 768,
          "n'oublie pas la table des positions": GPT2_B35["total"] - GPT2_B35["wpe"],
          "c'est le nombre de l'article, que le dépôt dit faux : compte les paramètres": 117000000}'''),
        ("g", WHOLE, 'GPT2_B35["non_embedding"]', r'''mistakes={"relis la question : la layer norm finale comprise": GPT2_B35["blocks"],
          "relis la question : sans la table des positions": GPT2_B35["non_embedding"] + GPT2_B35["wpe"],
          "relis la question : sans les tables d'embedding": GPT2_B35["total"]}'''),
        ("h", DEC1, '100 * (GPT2_B35["total"] - PAPER_B35) / PAPER_B35', r'''decimals=1, mistakes={"rapporte l'écart aux 117 millions de l'article": 100 * (GPT2_B35["total"] - PAPER_B35) / GPT2_B35["total"],
          "on demande un pourcentage, pas une proportion": (GPT2_B35["total"] - PAPER_B35) / PAPER_B35,
          "c'est l'écart en millions : on demande un pourcentage": (GPT2_B35["total"] - PAPER_B35) / 1e6}'''),
        ("i", TF, "GPT2_UNTIED_B35 > 160e6", r'''mistakes={"relis ce que les poids liés économisent (B2, §202.7)": False}'''),
        ("j", WHOLE, 'MINI_B35["total"]', r'''mistakes={"relis la question : on demande tous les paramètres, tables comprises": MINI_B35["non_embedding"],
          "relis l'énoncé : la couche de sortie est liée": gpt_params(4, 128, 88, 128, tied=False)["total"],
          "n'oublie pas la table des positions": MINI_B35["total"] - MINI_B35["wpe"],
          "relis la question : le compte exact, tables comprises": 4 * 12 * 128 ** 2,
          "relis la question : les tables d'embedding et la layer norm finale comptent aussi": MINI_B35["blocks"],
          "n'oublie pas la layer norm finale": MINI_B35["total"] - MINI_B35["ln_f"]}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ B3.1 à B3.5)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "une forme de tableau : entre guillemets (`\"(2, 3)\"`) ; un vecteur ou une ligne de matrice : une "
               "liste (`[0.5, -1, 3]`). Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les "
               "réponses pas encore remplies affichent ⏳. Les questions « dans ta copie », les dérivations ∂ B3.6 "
               "et B3.7, l'estimation 🧮 B3.8, la réflexion (🗣️ B3.9, ⚖️ B3.10, 📄 B3.11) et l'entretien se "
               "corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A and B (exercises B3.12 to B3.30): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "B3.12 à B3.17", "ta librairie NumPy : masques et attention, carte d'attention à prédire, têtes, positions "
                           "sinusoïdales et RoPE, bloc Transformer complet validé contre PyTorch, trois masques qui "
                           "mentent"),
    ("B", "B3.18 à B3.30", "le mini-GPT en PyTorch : module d'attention, modèle, entraînement sur Holmes, courbes, "
                           "Holmes amorcé en français, KV cache, cartes d'attention, ablations, BPE contre caractères, "
                           "contexte long, ViT, test de causalité, défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | B3.Q1 à B3.Q12, B3.R1 à B3.R3, B3.1 à B3.5 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(ex.table_row(part.key if i == 0 else ""))
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# B3 · Attention et Transformers : un mini-GPT from scratch — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/b3_transformers/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py B3`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# B3 · Attention et Transformers : un mini-GPT from scratch — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Calculer à la main une attention par produit scalaire, avec et sans masque causal, et justifier "
               "la division par $\\sqrt{d_k}$.\n"
               "- Implémenter en NumPy les masques, l'attention multi-têtes, les encodages de position (sinusoïdal, "
               "RoPE) et un bloc Transformer pré-LN, et les valider contre PyTorch.\n"
               "- Écrire et entraîner en PyTorch un mini-GPT sur Holmes, le faire générer avec un KV cache, et "
               "comparer sa perplexité à celle du LSTM du ch. 22.\n"
               "- Estimer les paramètres, les FLOPs et la mémoire d'un Transformer, et mesurer le coût d'un long "
               "contexte.\n\n"
               "**Rappel express.** $\\operatorname{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = "
               "\\operatorname{softmax}\\big(\\mathbf{Q}\\mathbf{K}^\\top / \\sqrt{d_k} + \\mathbf{M}\\big)\\mathbf{V}$, "
               "la softmax sur les clés, $M_{ij} = -\\infty$ là où le masque vaut `False`. Masque causal : la requête "
               "$i$ voit les clés $j \\le i + S - L$. Multi-têtes : $(N, T, d) \\to (N, H, T, d_h)$, $4d^2 + 4d$ "
               "paramètres. Encodage sinusoïdal : $PE_{p, 2i} = \\sin(p\\,\\omega_i)$, $PE_{p, 2i+1} = "
               "\\cos(p\\,\\omega_i)$, $\\omega_i = 10000^{-2i/d}$. RoPE : $\\langle R_{m\\theta} \\mathbf{q}, R_{n\\theta} \\mathbf{k} "
               "\\rangle = \\langle \\mathbf{q}, R_{(n-m)\\theta} \\mathbf{k} \\rangle$. Bloc pré-LN : $\\mathbf{h} = \\mathbf{x} + "
               "\\operatorname{MHA}(\\operatorname{LN}_1(\\mathbf{x}))$, $\\mathbf{y} = \\mathbf{h} + "
               "\\operatorname{MLP}(\\operatorname{LN}_2(\\mathbf{h}))$, environ $12d^2$ paramètres. Entraînement : "
               "environ $6ND$ FLOPs ; KV cache : $2 \\times L \\times T \\times d \\times$ octets par valeur.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code B3.12 à B3.30 (partie A : ta librairie NumPy ; partie B : le mini-GPT en "
             "PyTorch) seront ajoutés à la prochaine session de génération ; relance alors "
             "`python tools/start_chapter.py B3` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu calculer une attention à la main, masque causal compris, et dire pourquoi on divise par "
               "$\\sqrt{d_k}$ ?\n"
               "2. Sais-tu suivre les formes d'une attention multi-têtes et compter les paramètres d'un GPT ?\n"
               "3. Sais-tu dire ce que font un encodage sinusoïdal et RoPE, et démontrer la propriété de RoPE ?\n"
               "4. Sais-tu chiffrer le calcul d'un entraînement et la mémoire d'un KV cache ?\n\n"
               "**Pour aller plus loin** : la vidéo « Let's build GPT » d'A. Karpathy et *The Annotated "
               "Transformer*, cités dans la fiche. La suite : le bonus B4, les LLM en pratique."
               + later)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT + LIBRARY_CHECKS, PAPER_INTRO)
    for part in PARTS:
        cells += part_cells(part, kind)
    cells += footer_cells(kind)
    return cells


def main() -> int:
    write_notebook(f"{FOLDER}/03_notebook.ipynb", build("exercise"))
    write_notebook(f"{FOLDER}/05_solutions.ipynb", build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER)
    n_ex = sum(len(part.exercises) for part in PARTS)
    print(f"✅ {FOLDER}/03_notebook.ipynb and 05_solutions.ipynb written ({len(PAPER)} paper exercises, "
          f"{n_checks} paper checks, {n_ex} notebook exercises)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
