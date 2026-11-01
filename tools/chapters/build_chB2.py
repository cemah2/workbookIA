#!/usr/bin/env python
"""Build the two notebooks of bonus chapter B2 from a single source (used by Claude).

    python tools/chapters/build_chB2.py
    python tools/run_all_notebooks.py chapitres/b2_tokenisation/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/b2_tokenisation/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
B2.1 to B2.6 and of the estimate 🧮 B2.9 (02_exercices.md). The derivation ∂ B2.7, the estimate 🧮 B2.8, the reflection
(🗣️ B2.10, ⚖️ B2.11, 📄 B2.12) and the interview are corrected with 05_solutions.md. The answers are computed by hand
formulas in the context cell of the solutions notebook, then checked against the reference library (mylearn.tokenizer,
mylearn.embeddings, mylearn.sequences, mylearn.info), Hugging Face tokenizers and PyTorch (no download).
Parts A to E (exercises B2.13 to B2.31) come at the second generation session (NEXT_SESSION): real tokenizers and
their cost, your BPE, experiments on the vocabulary, word vectors (co-occurrences, word2vec), sentence embeddings,
delivery and the challenge.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (Paper, badge, check_unique_keys, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "B2"
FOLDER = "chapitres/b2_tokenisation"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math
import unicodedata

import numpy as np

# 🧠 Q2: « é » in NFC (one code point) and in NFD (« e » + U+0301)
E_ACUTE_NFC = "\\u00e9"
E_ACUTE_NFD = unicodedata.normalize("NFD", E_ACUTE_NFC)

# ✏️ B2.1: « Noël à Paris », typed in NFC
S_B21 = "No\\u00ebl \\u00e0 Paris"
B_B21 = S_B21.encode("utf-8")
NFD_B21 = unicodedata.normalize("NFD", S_B21)


def utf8_two_bytes(code_point):
    """110xxxxx 10xxxxxx: the 5 high bits, then the 6 low bits of an 11-bit code point (U+0080 to U+07FF)."""
    return [0b11000000 | (code_point >> 6), 0b10000000 | (code_point & 0b111111)]


# ✏️ B2.2: toy vectors (axes: royalty, masculine (+) or feminine (-), adult), « homme est à roi ce que femme est à ? »
WORDS_B22 = ["roi", "reine", "homme", "femme", "prince", "princesse", "trône"]
E_B22 = np.array([[0.6, 1.0, 0.9], [0.8, -0.5, 0.9], [0.1, 1.0, 0.9], [0.1, -1.0, 0.9],
                  [0.7, 0.8, 0.2], [0.7, -0.6, 0.2], [3.0, 0.0, 0.5]])
Q_B22 = E_B22[0] - E_B22[2] + E_B22[3]
COS_B22 = E_B22 @ Q_B22 / (np.linalg.norm(E_B22, axis=1) * np.linalg.norm(Q_B22))
E_UNIT_B22 = E_B22 / np.linalg.norm(E_B22, axis=1, keepdims=True)


def analogy_b22(E, a, b, c, exclude=True, cosine=True):
    """The word x of largest cos(x, b - a + c) (or dot product), without a, b and c when exclude is True."""
    k = {w: i for i, w in enumerate(WORDS_B22)}
    q = E[k[b]] - E[k[a]] + E[k[c]]
    scores = E @ q / (np.linalg.norm(E, axis=1) * np.linalg.norm(q)) if cosine else E @ q
    order = [i for i in np.argsort(-scores, kind="stable") if not (exclude and WORDS_B22[i] in (a, b, c))]
    return WORDS_B22[order[0]]


# ✏️ B2.3: three BPE merges by hand; ties go to the smallest pair of ids (BpeTrainer of tokenizers)
CORPUS_B23 = {"été": 4, "étés": 3, "pré": 2, "prés": 1}


def merge_ids(ids, pair, new_id):
    """Replace each occurrence of pair, left to right and without overlap, by new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def learn_bpe(corpus, n_merges):
    """The fiche's pseudo-code: alphabet sorted by code point, then n_merges merges (pair of ids) with their counts."""
    alphabet = sorted({ch for word in corpus for ch in word})
    words = {tuple(alphabet.index(ch) for ch in word): f for word, f in corpus.items()}
    merges, counts = [], []
    for _ in range(n_merges):
        pairs = {}
        for word, f in words.items():
            for pair in zip(word, word[1:]):
                pairs[pair] = pairs.get(pair, 0) + f
        best = min(pairs, key=lambda p: (-pairs[p], p))           # most frequent, then the smallest pair of ids
        merges.append(best)
        counts.append(pairs[best])
        words = {tuple(merge_ids(list(w), best, len(alphabet) + len(merges) - 1)): f for w, f in words.items()}
    return alphabet, merges, counts, words


ALPHABET_B23, MERGES_B23, COUNTS_B23, WORDS_AFTER_B23 = learn_bpe(CORPUS_B23, 3)
TOKENS_AFTER_B23 = sum(len(word) * f for word, f in WORDS_AFTER_B23.items())

# ✏️ B2.4: a BPE already learned (char level), applied by rank; an unseen character becomes <unk>
ALPHABET_B24 = list("acehnorst")
MERGES_B24 = [("n", "t"), ("c", "h"), ("a", "nt"), ("o", "n"), ("ch", "ant"), ("e", "r"), ("on", "s"),
              ("chant", "er")]
VOCAB_B24 = {ch: i for i, ch in enumerate(ALPHABET_B24)}
for _a, _b in MERGES_B24:
    VOCAB_B24[_a + _b] = len(VOCAB_B24)
UNK_B24 = len(VOCAB_B24)                                             # the special token comes after the merges
RANK_B24 = {(VOCAB_B24[a], VOCAB_B24[b]): (r, VOCAB_B24[a + b]) for r, (a, b) in enumerate(MERGES_B24)}


def encode_b24(word, by_rank=True):
    """Merge the adjacent pair of lowest rank (by_rank) or, the classic mistake, the leftmost known pair."""
    ids = [VOCAB_B24.get(ch, UNK_B24) for ch in word]
    while True:
        present = [(RANK_B24[p][0], i) for i, p in enumerate(zip(ids, ids[1:])) if p in RANK_B24]
        if not present:
            return ids
        _, i = min(present) if by_rank else min(present, key=lambda t: t[1])
        ids[i:i + 2] = [RANK_B24[(ids[i], ids[i + 1])][1]]


# ✏️ B2.5: one-hot times a table, and the gradient of L = sum over the positions t of w . e_t
E_B25 = np.array([[1, 0, 2], [0, 1, -1], [2, -1, 0], [-1, 3, 1], [0, 2, 2]], dtype=float)
IDS_B25 = [3, 0, 3, 1]
W_B25 = np.array([1.0, -2.0, 0.5])
X_B25 = np.eye(5)[IDS_B25]                                           # one one-hot row per position: (4, 5)
OUT_B25 = X_B25 @ E_B25                                              # (4, 3)
L_B25 = float((OUT_B25 @ W_B25).sum())
GRAD_B25 = X_B25.T @ np.tile(W_B25, (len(IDS_B25), 1))               # dL/dE = X^T dL/dOUT, shape (5, 3)
LR_B25 = 0.1

# ✏️ B2.6: co-occurrences in a window of 1, ids in alphabetical order (chat 0, court 1, le 2, rat 3)
WORDS_B26 = ["chat", "court", "le", "rat"]
IDS_B26 = [WORDS_B26.index(w) for w in "le chat court le chat le rat court".split()]
C_B26 = np.zeros((4, 4))
for _i in range(len(IDS_B26) - 1):
    C_B26[IDS_B26[_i], IDS_B26[_i + 1]] += 1                         # right context
    C_B26[IDS_B26[_i + 1], IDS_B26[_i]] += 1                         # left context
N_B26 = C_B26.sum()


def pmi_b26(w, c, alpha=1.0, alpha_rows=False):
    """log p(w, c) / (p(w) p_alpha(c)); alpha_rows smooths the rows too (a classic mistake)."""
    i, j = WORDS_B26.index(w), WORDS_B26.index(c)
    rows = C_B26.sum(axis=1) ** (alpha if alpha_rows else 1.0)
    cols = C_B26.sum(axis=0) ** alpha
    return math.log((C_B26[i, j] / N_B26) / ((rows[i] / rows.sum()) * (cols[j] / cols.sum())))


P_ALPHA_B26 = C_B26.sum(axis=0) ** 0.75 / (C_B26.sum(axis=0) ** 0.75).sum()

# 🧮 B2.9: embedding tables (config.json and safetensors headers of the Hugging Face Hub, read on 2026-11-01)
GPT2_TABLE_B29 = 50257 * 768
GPT2_TOTAL_B29 = 124_439_808                                         # GPT-2 small, its tied head counted once
L31_TABLE_B29 = 128256 * 4096                                        # Llama 3.1 8B, untied: two tables
L31_TOTAL_B29 = 8_030_261_248
L32_TABLE_B29 = 128256 * 2048                                        # Llama 3.2 1B, tied: one table
L32_TOTAL_B29 = 1_235_814_400

# quizzes and recalls with a computation
WP_AB_Q7 = 20 / (40 * 100)                                          # WordPiece score count(a, b) / (count(a) count(b))
WP_CD_Q7 = 30 / (300 * 200)
PMI_Q10 = math.log(6 * 1000 / (30 * 200))                           # p(w, c) / (p(w) p(c)), N simplified
H_Q11 = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 4.0], [0.0, 0.0]])   # the last position is a [PAD]
M_Q11 = np.array([1.0, 1.0, 1.0, 0.0])
S_Q11 = (M_Q11[:, None] * H_Q11).sum(axis=0) / M_Q11.sum()
P_R3 = np.array([0.5, 0.25, 0.125, 0.125])
H_R3 = float(-(P_R3 * np.log2(P_R3)).sum())
'''

LIBRARY_CHECKS = '''
# The same answers from the libraries (no download): mylearn (reference), tokenizers, PyTorch
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch import nn

assert len(E_ACUTE_NFC.encode()) == 2 and len(E_ACUTE_NFD) == 2 and len(E_ACUTE_NFD.encode()) == 3
assert len("\\u2019".encode()) == 3 and len("\\u4e2d".encode()) == 3
assert utf8_two_bytes(0xEB) == list("\\u00eb".encode()) and utf8_two_bytes(0x153) == list("\\u0153".encode())

T = mylearn.tokenizer
# ✏️ B2.3: the merges of BpeTrainer and of BPETokenizer, char level, Whitespace pre-tokenizer
text_b23 = " ".join(w for w, f in CORPUS_B23.items() for _ in range(f))
ref_b23 = T.BPETokenizer(vocab_size=len(ALPHABET_B23) + 3, level="char", min_frequency=1).fit(text_b23)
assert ref_b23.merges_ == MERGES_B23 and ref_b23.n_base_ == len(ALPHABET_B23)
hf_b23 = Tokenizer(models.BPE())
hf_b23.pre_tokenizer = pre_tokenizers.Whitespace()
hf_b23.train_from_iterator([text_b23], trainers.BpeTrainer(vocab_size=len(ALPHABET_B23) + 3, min_frequency=1,
                                                           show_progress=False))
assert [hf_b23.token_to_id(ch) for ch in ALPHABET_B23] == list(range(len(ALPHABET_B23)))
assert [hf_b23.token_to_id(ref_b23.vocab_[a].decode() + ref_b23.vocab_[b].decode()) for a, b in MERGES_B23] == [5, 6, 7]
for min_frequency, n in ((3, 3), (4, 2)):                             # 🧠 B2.3 h): the stopping rule
    assert len(T.BPETokenizer(vocab_size=8, level="char", min_frequency=min_frequency).fit(text_b23).merges_) == n

# ✏️ B2.4: the same merges in models.BPE and in BPETokenizer (attributes set by hand)
hf_b24 = Tokenizer(models.BPE(vocab=dict(VOCAB_B24, **{"<unk>": UNK_B24}), merges=MERGES_B24, unk_token="<unk>"))
ref_b24 = T.BPETokenizer(level="char", unk_token="<unk>")
ref_b24.n_base_, ref_b24.special_ids_ = len(ALPHABET_B24), {"<unk>": UNK_B24}
ref_b24.merges_ = [(VOCAB_B24[a], VOCAB_B24[b]) for a, b in MERGES_B24]
ref_b24.vocab_ = {i: s.encode() for s, i in VOCAB_B24.items()} | {UNK_B24: b"<unk>"}
for word in ("chantons", "ont", "chanteront", "chantez", "enchanter", "chantonner"):
    assert encode_b24(word) == hf_b24.encode(word).ids == ref_b24.encode(word), word
assert encode_b24("ont", by_rank=False) != encode_b24("ont")
byte_b24 = T.BPETokenizer(vocab_size=256, pre_tokenizer="gpt2").fit("x")       # 256 bytes, no merge
assert len(byte_b24.encode("été")) == 5 and ref_b24.decode(ref_b24.encode("chantez")) != "chantez"

# ✏️ B2.5: nn.Embedding, its gradient, padding_idx and the weight decay of SGD
emb = nn.Embedding(5, 3)
with torch.no_grad():
    emb.weight.copy_(torch.tensor(E_B25))
ids = torch.tensor(IDS_B25)
x = nn.functional.one_hot(ids, num_classes=5).double()
assert torch.allclose(x @ emb.weight.double(), emb(ids).double()) and np.allclose(emb(ids).detach(), OUT_B25)
loss = (emb(ids) @ torch.tensor(W_B25, dtype=torch.float32)).sum()
loss.backward()
assert np.isclose(loss.item(), L_B25) and np.allclose(emb.weight.grad, GRAD_B25)
torch.optim.SGD(emb.parameters(), lr=LR_B25).step()
assert np.allclose(emb.weight.detach()[3], E_B25[3] - LR_B25 * GRAD_B25[3])
pad = nn.Embedding(5, 3, padding_idx=0)
(pad(ids) @ torch.tensor(W_B25, dtype=torch.float32)).sum().backward()
assert torch.all(pad.weight.grad[0] == 0)
decay = nn.Embedding(5, 3)
before = decay.weight.detach().clone()
(decay(ids) @ torch.tensor(W_B25, dtype=torch.float32)).sum().backward()
torch.optim.SGD(decay.parameters(), lr=LR_B25, weight_decay=0.01).step()
assert not torch.equal(decay.weight.detach()[4], before[4])

# ✏️ B2.6: cooccurrence_matrix and ppmi of the reference
Em = mylearn.embeddings
assert np.array_equal(Em.cooccurrence_matrix(IDS_B26, 4, window=1), C_B26)
P1, P75 = Em.ppmi(C_B26), Em.ppmi(C_B26, alpha=0.75)
assert np.isclose(P1[0, 2], pmi_b26("chat", "le")) and P1[1, 2] == 0 and pmi_b26("court", "le") < 0
assert np.isclose(P75[0, 2], pmi_b26("chat", "le", 0.75)) and P75[1, 2] > 0 and not np.allclose(P75, P75.T)

# ✏️ B2.2: most_similar of the reference, with and without the exclusion
i22 = {w: i for i, w in enumerate(WORDS_B22)}
assert WORDS_B22[Em.most_similar(E_B22, Q_B22, k=1)[0][0]] == analogy_b22(E_B22, "homme", "roi", "femme", exclude=False)
assert WORDS_B22[Em.most_similar(E_B22, Q_B22, k=1, exclude=(i22["homme"], i22["roi"], i22["femme"]))[0][0]] == "reine"

# 🧮 B2.9: GPT-2 small built without its weights (meta device)
from transformers import GPT2Config, GPT2LMHeadModel
with torch.device("meta"):
    gpt2 = GPT2LMHeadModel(GPT2Config())
assert sum(p.numel() for p in gpt2.parameters()) == GPT2_TOTAL_B29
assert gpt2.transformer.wte.weight.numel() == GPT2_TABLE_B29 and gpt2.lm_head.weight is gpt2.transformer.wte.weight

# 🔁 R2 and R3: build_vocab, encode (ch. 22), entropy and Huffman (ch. 6)
S = mylearn.sequences
stoi, itos = S.build_vocab("banana")
assert stoi["b"] == 2
stoi_u, _ = S.build_vocab("banana", unk_token="<unk>")
assert S.encode("bandana", stoi_u, unk_token="<unk>").tolist() == [3, 1, 2, 0, 1, 2, 1]
stoi_w, itos_w = S.build_vocab(["le", "chat", "le", "rat"], max_size=2, unk_token="<unk>")
assert S.encode(["chat"], stoi_w, unk_token="<unk>").tolist() == [0]
assert np.isclose(mylearn.info.entropy(P_R3), H_R3)
assert min(len(c) for c in mylearn.info.huffman_code(list("abcd"), P_R3).values()) == 1
print("✅ the paper answers agree with mylearn (reference), tokenizers and PyTorch")
'''

mylearn = importlib.import_module("mylearn_ref")   # the builder computes the mistakes with the same helpers
for _module in ("tokenizer", "embeddings", "sequences", "info"):
    importlib.import_module(f"mylearn_ref.{_module}")
exec(PAPER_CONTEXT)
exec(LIBRARY_CHECKS)

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "B"'
WHOLE = "a whole number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
SHAPE = 'a shape as a string, e.g. "(2, 3)"'
BYTES = "a list of byte values, e.g. [0x4E, 0x6F]"
IDS = "a list of token ids, e.g. [4, 0, 7]"
PAIR = "a pair of ids as a list or a tuple, e.g. [2, 7]"
WORD = 'a word as a string, e.g. "mot"'
VECTOR = "a list of numbers, e.g. [0.5, -1, 3]"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    Paper("B2.Q1", "Caractères, mots ou sous-mots : taille du vocabulaire contre longueur des séquences", [
        ("a", LETTER, '"A"', letters("A", {"B": "un mot entier est un seul token : relis la colonne « longueur des séquences »",
                                           "C": "les sous-mots sont entre les deux : relis le tableau des trois granularités"})),
        ("b", TF, "True", r'''mistakes={"relis ce que devient un mot qui n'est pas parmi les V plus fréquents (§202.1)": False}'''),
        ("c", WHOLE, "3 ** 2", r'''mistakes={"le coût croît comme le carré de la longueur, pas comme la longueur": 3,
          "tu as doublé au lieu d'élever au carré": 6,
          "le carré de la longueur, pas son cube": 27}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "personne ne choisit les morceaux à la main : relis d'où vient le découpage",
                                           "C": "les morceaux n'ont pas de longueur fixe : relis l'exemple de « paisiblement »"})),
        ("e", TF, "False", r'''mistakes={"relis le tableau des trois granularités : combien de symboles distincts à chaque niveau ?": True}'''),
    ]),
    Paper("B2.Q2", "Octets, points de code, caractères affichés : combien « é » compte-t-il ?", [
        ("a", WHOLE, "len(E_ACUTE_NFC.encode('utf-8'))", r'''mistakes={"« é » n'est pas un caractère ASCII : relis le tableau d'UTF-8": 1,
          "relis les plages du tableau d'UTF-8 : dans laquelle tombe U+00E9 ?": 3}'''),
        ("b", WHOLE, "len(E_ACUTE_NFD)", r'''mistakes={"relis ce que fait NFD d'une lettre accentuée": 1,
          "on compte ici des points de code, pas des octets": 3}'''),
        ("c", WHOLE, "len(E_ACUTE_NFD.encode('utf-8'))", r'''mistakes={"compte les octets de chacun des deux points de code": 2,
          "recompte les octets de « e » : dans quelle plage du tableau tombe-t-il ?": 4}'''),
        ("d", WHOLE, "len(E_ACUTE_NFC)", r'''mistakes={"len d'un str compte les points de code, pas les octets": 2}'''),
        ("e", TF, "False", r'''mistakes={"U+2019 est au-delà de U+07FF : relis les plages du tableau d'UTF-8": True}'''),
    ]),
    Paper("B2.Q3", "NFC ou NFD : deux chaînes identiques à l'écran, différentes en mémoire", [
        ("a", TF, "False", r'''mistakes={"une chaîne en NFC et sa forme NFD n'ont pas les mêmes points de code : que compare == ?": True}'''),
        ("b", LETTER, '"C"', letters("C", {"A": "casefold() change la casse, pas les accents",
                                           "B": "en NFC, « é » est un seul point de code au-delà de 127 : tu supprimerais la lettre entière"})),
        ("c", TF, "False", r'''mistakes={"relis ce que NFKC fait de « ﬁ » et de « ² » : peut-on revenir en arrière ?": True}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "BPE compare des points de code, pas ce qui s'affiche : relis le piège classique du §202.2",
                                           "C": "un tokenizer ne vérifie pas la normalisation : il découpe sans rien dire"})),
        ("e", TF, "True", r'''mistakes={"NFC recompose tout ce que NFD a décomposé : une chaîne déjà composée revient-elle à l'identique ?": False}'''),
    ]),
    Paper("B2.Q4", "Fusions BPE : pourquoi la liste ordonnée des fusions est le modèle", [
        ("a", LETTER, '"B"', letters("B", {"A": "deux tokenizers aux mêmes tokens peuvent découper autrement : relis « Encoder un mot nouveau »",
                                           "C": "le corpus sert à apprendre les fusions, pas à encoder"})),
        ("b", TF, "False", r'''mistakes={"relis l'entraînement (§202.3) : une fusion franchit-elle la frontière d'un mot ?": True}'''),
        ("c", WHOLE, "3 * 2", r'''mistakes={"compte toutes les occurrences de la paire dans le mot": 3,
          "multiplie par la fréquence du mot": 2,
          "« n a » est une autre paire : on ne compte que « a » suivi de « n »": 9}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "une fusion crée un seul token nouveau",
                                           "C": "les deux tokens fusionnés restent dans le vocabulaire : d'autres mots s'en servent"})),
        ("e", TF, "True", r'''mistakes={"relis l'encadré 🕰️ sur le départage des fusions dans tokenizers": False}'''),
    ]),
    Paper("B2.Q5", "BPE au niveau octet : jamais de token inconnu, vrai ou faux ?", [
        ("a", TF, "True", r'''mistakes={"au niveau octet, quel alphabet couvre tout texte UTF-8 ?": False}'''),
        ("b", WHOLE, "len('\\u4e2d'.encode('utf-8'))", r'''mistakes={"sans fusion, chaque octet est un token : compte les octets du caractère": 1,
          "relis dans quelle plage du tableau d'UTF-8 tombent les caractères chinois": 4}'''),
        ("c", TF, "False", r'''mistakes={"relis l'encadré ⚠️ « un BPE qui jette des caractères » (§202.5)": True}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "relis la taille du vocabulaire de GPT-2 (§202.4)",
                                           "B": "décoder la liste entière des identifiants redonne le texte exact"})),
        ("e", TF, "True", r'''mistakes={"relis ce que donne le décodage d'un seul des deux octets de « Ç » (§202.4)": False}'''),
    ]),
    Paper("B2.Q6", "Tokens spéciaux et espace initial : « Ġthe » n'est pas « the »", [
        ("a", TF, "False", r'''mistakes={"relis la table octet → caractère : l'espace fait-elle partie du token ?": True}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "« Ġ » fait partie de tokens ordinaires comme « Ġthe » : relis la table octet → caractère",
                                           "C": "relis la table octet → caractère de GPT-2"})),
        ("c", LETTER, '"C"', letters("C", {"A": "un token spécial n'est jamais découpé",
                                           "B": "le modèle a besoin de ce signal : on ne le supprime pas"})),
        ("d", TF, "True", r'''mistakes={"relis le motif de GPT-2 : chaque caractère du texte appartient-il à un morceau ?": False}'''),
        ("e", WHOLE, "1", r'''mistakes={"relis la ligne EOS du tableau des tokens spéciaux": 0,
          "relis le paragraphe sous le tableau des tokens spéciaux": 2},
          fractional="on demande un nombre de tokens"'''),
    ]),
    Paper("B2.Q7", "WordPiece, Unigram, SentencePiece : quel critère pour quelle méthode ?", [
        ("a", LETTER, '"C"', letters("C", {"A": "BPE part des caractères et ajoute des fusions",
                                           "B": "WordPiece part aussi des caractères et fusionne"})),
        ("b", LETTER, '"A"', letters("A", {"B": "calcule le score de WordPiece des deux paires, pas leur compte"})),
        ("c", LETTER, '"B"', letters("B", {"A": "BPE ne regarde que le compte de la paire"})),
        ("d", TF, "False", r'''mistakes={"relis ce que SentencePiece fait des espaces (§202.5)": True}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "les tokens spéciaux de BERT s'écrivent entre crochets",
                                           "C": "relis l'exemple de « unbelievably »"})),
    ]),
    Paper("B2.Q8", "Pourquoi un LLM compte mal les lettres d'un mot et calcule mal avec les chiffres", [
        ("a", LETTER, '"C"', letters("C", {"A": "le modèle ne voit que des identifiants de tokens",
                                           "B": "même un BPE octet fusionne les octets des mots fréquents : relis « Les lettres »"})),
        ("b", TF, "False", r'''mistakes={"compare les découpages de « 1234567 » et de « 1234656 » par GPT-2 (§202.6)": True}'''),
        ("c", WHOLE, "len('2026')", r'''mistakes={"Qwen3 isole chaque chiffre": 1,
          "ce sont des groupes à la façon de Llama 3 : relis la règle de Qwen3": 2,
          "sans espace devant le nombre, pas de token d'espace": 5}'''),
        ("d", WHOLE, "math.ceil(len('12345678') / 3)", r'''mistakes={"le dernier groupe peut être plus court que les autres : il compte aussi": 2,
          "on compte des groupes, pas des chiffres": 8,
          "des groupes d'au plus 3 chiffres, pas de 2": 4}'''),
        ("e", TF, "True", r'''mistakes={"relis la fin du paragraphe « Les nombres » (§202.6)": False}'''),
    ]),
    Paper("B2.Q9", "nn.Embedding : une table de consultation entraînable, pas une couche magique", [
        ("a", SHAPE, '"(8, 20, 64)"', r'''mistakes={"n'oublie pas la dimension de l'embedding": "(8, 20)",
          "la table a 1 000 lignes, mais chaque id n'en lit qu'une": "(8, 20, 1000)",
          "les axes des ids restent dans le même ordre": "(20, 8, 64)"}'''),
        ("b", WHOLE, "5000 * 32", r'''mistakes={"une table de V lignes et de d colonnes : multiplie": 5032,
          "nn.Embedding n'a pas de biais": 160032}'''),
        ("c", TF, "True", r'''mistakes={"relis le calcul du one-hot fois la table (§202.7) : quelle ligne lit-il ?": False}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "règle de la chaîne : les contributions des occurrences s'additionnent",
                                           "C": "chaque occurrence contribue au gradient de la ligne"})),
        ("e", TF, "True", r'''mistakes={"relis le paragraphe sous l'exemple de nn.Embedding : une ligne jamais lue reçoit-elle un gradient ?": False}'''),
    ]),
    Paper("B2.Q10", "L'hypothèse distributionnelle : un mot se définit par ses voisins", [
        ("a", LETTER, '"B"', letters("B", {"A": "l'hypothèse distributionnelle ne regarde pas l'orthographe",
                                           "C": "se suivre n'est pas partager des contextes : « boire » et « café » se suivent, « thé » et « café » ont les mêmes voisins"})),
        ("b", TF, "True", r'''mistakes={"relis le sens du signe de la PMI dans l'encadré 🧮 du §202.8": False}'''),
        ("c", DEC2, "PMI_Q10", r'''decimals=2, mistakes={"c'est le rapport des probabilités : la PMI est son logarithme": 1.0,
          "divise chaque compte par N avant de faire le rapport": math.log(6 / (30 * 200))}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "prédire les contextes à partir du mot central, c'est le skip-gram",
                                           "B": "les mots tirés au hasard sont les négatifs de l'échantillonnage négatif"})),
        ("e", TF, "False", r'''mistakes={"relis la règle du sous-échantillonnage : quels mots sont jetés le plus souvent ?": True}'''),
    ]),
    Paper("B2.Q11", "Statique ou contextuel : l'« avocat » du tribunal et celui du marché", [
        ("a", LETTER, '"A"', letters("A", {"B": "un embedding statique ne regarde pas la phrase",
                                           "C": "le vecteur d'un mot du vocabulaire n'est pas nul"})),
        ("b", LETTER, '"B"', letters("B", {"A": "la ligne de la table est le point de départ ; la sortie du dernier bloc dépend de la phrase",
                                           "C": "un encodeur rend un vecteur par token"})),
        ("c", TF, "True", r'''mistakes={"relis le paragraphe « Un mot, plusieurs sens » (§202.10)": False}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "prendre un seul token, c'est un autre pooling : relis la formule du mean pooling",
                                           "B": "on moyenne les sorties du réseau, pas les lignes de la table"})),
        ("e", DEC1, "S_Q11[0]", r'''decimals=1, mistakes={"le masque retire le [PAD] de la moyenne : divise par le nombre de vrais tokens": 1.5,
          "on fait la moyenne des vecteurs, pas leur somme": 6.0}'''),
    ]),
    Paper("B2.Q12", "Cosinus ou distance euclidienne pour des embeddings normalisés ?", [
        ("a", DEC1, "2 - 2 * 0.8", r'''decimals=1, mistakes={"on demande la distance au carré, pas la distance": math.sqrt(0.4),
          "relis la relation de la fiche : un facteur 2 manque": 1 - 0.8,
          "le cosinus est multiplié par 2 dans la relation": 2 - 0.8}'''),
        ("b", TF, "True", r'''mistakes={"développe ‖a − b‖² pour deux vecteurs de norme 1": False}'''),
        ("c", DEC3, "math.sqrt(2 - 2 * 0)", r'''decimals=3, mistakes={"on demande la distance, pas son carré": 2.0,
          "deux vecteurs orthogonaux ne sont pas confondus : applique la relation avec leur cosinus": 0.0}'''),
        ("d", TF, "False", r'''mistakes={"un vecteur de grande norme gagne au produit scalaire sans changer de direction": True}'''),
        ("e", DEC1, "-1.0", r'''decimals=1, mistakes={"vérifie le signe du produit scalaire": 1.0,
          "le cosinus divise le produit scalaire par les deux normes": -50.0}'''),
    ]),
    Paper("B2.R1", "À quoi sert l'embedding d'image extrait d'un réseau pré-entraîné ?", [
        ("a", TF, "True", r'''mistakes={"relis le §201.3 de B1 : combien de fois calcule-t-on un embedding figé ?": False}'''),
        ("b", WHOLE, "512", r'''mistakes={"c'est le nombre de sorties de la tête ImageNet : on demande l'entrée de fc": 1000,
          "c'est la dimension de l'embedding de ResNet-50": 2048}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "un fine-tuning modifie le backbone : l'embedding n'est plus figé",
                                           "C": "un k plus proches voisins n'apprend aucun poids"})),
        ("d", TF, "False", r'''mistakes={"relis pourquoi B1 compare les embeddings par le cosinus (§201.3)": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "des pixels bruts ne sont pas un embedding",
                                           "C": "la tête produit des scores de classes, pas une représentation"})),
    ]),
    Paper("B2.R2", "build_vocab, encode, decode et le token inconnu", [
        ("a", WHOLE, "2", r'''mistakes={"les ids suivent la fréquence décroissante : quelle lettre est la plus fréquente ?": 0,
          "les ids commencent à 0": 3}'''),
        ("b", IDS, "[3, 1, 2, 0, 1, 2, 1]", r'''mistakes={"avec unk_token, le token inconnu prend l'id 0 et décale tous les autres": [2, 0, 1, 0, 0, 1, 0],
          "un caractère absent du vocabulaire ne reçoit pas de nouvel id : il devient le token inconnu": [3, 1, 2, 4, 1, 2, 1]}'''),
        ("c", TF, "False", r'''mistakes={"que devient « d » à l'aller, et que rend decode pour l'id du token inconnu ?": True}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "relis le rôle de unk_token dans encode",
                                           "B": "build_vocab ne découpe jamais un mot : c'est le travail d'un BPE"})),
        ("e", WHOLE, "0", r'''mistakes={"max_size compte aussi le token inconnu : combien de mots gardent leur propre identifiant ?": 2,
          "relis ce que compte max_size : « chat » a-t-il gardé un identifiant à lui ?": 1}'''),
    ]),
    Paper("B2.R3", "Entropie, Huffman et ce que l'on compresse vraiment", [
        ("a", DEC2, "H_R3", r'''decimals=2, mistakes={"l'entropie se mesure ici en bits : logarithme de base 2": -(P_R3 * np.log(P_R3)).sum(),
          "n'oublie pas le signe moins de la formule": -H_R3}'''),
        ("b", WHOLE, "1", r'''mistakes={"le symbole le plus probable reçoit le mot de code le plus court": 3,
          "un code de Huffman n'est pas de longueur fixe": 2}'''),
        ("c", TF, "True", r'''mistakes={"relis le lien avec le ch. 6 au début du §202.3": False}'''),
        ("d", TF, "False", r'''mistakes={"relis la borne du ch. 6 : l'entropie est-elle un plancher ou un plafond ?": True}'''),
        ("e", DEC1, "4000 / (1000 * 8)", r'''decimals=1, mistakes={"le taux du ch. 6 met les bits envoyés au numérateur": 2.0,
          "un octet vaut 8 bits : compare des bits à des bits": 4.0}'''),
    ]),
    Paper("B2.1", "« Noël à Paris » en UTF-8 : octets, points de code, longueur", [
        ("a", WHOLE, "len(S_B21)", r'''mistakes={"les espaces sont des points de code, eux aussi": 10,
          "relis la question : on compte des points de code, et en NFC une lettre accentuée n'en est qu'un": 14,
          "on compte des points de code, pas des mots": 3}'''),
        ("b", WHOLE, "len(B_B21)", r'''mistakes={"relis la question : on demande des octets, et une lettre accentuée en prend plus d'un": 12,
          "relis la question : on demande le texte tel qu'il est écrit, en NFC": 16,
          "relis les plages du tableau d'UTF-8 : dans laquelle tombent « ë » et « à » ?": 18}'''),
        ("c", BYTES, "list(B_B21[2:4])", r'''mistakes={"c'est le point de code U+00EB écrit sur deux octets, pas son écriture UTF-8": [0x00, 0xEB],
          "le second octet commence par les bits 10": [0xC3, 0x2B],
          "le premier octet commence par les bits 110": [0x03, 0xAB],
          "l'octet qui porte les bits de poids fort vient en premier": [0xAB, 0xC3]}'''),
        ("d", WHOLE, "B_B21[3]", r'''mistakes={"les indices d'un bytes comptent les octets, pas les caractères : combien de positions occupe « ë » ?": ord("l"),
          "les octets sont numérotés à partir de 0": B_B21[2],
          "c'est le point de code de « ë » : on demande un octet de son écriture UTF-8": 0xEB}'''),
        ("e", WHOLE, "len(NFD_B21.encode('utf-8'))", r'''mistakes={"relis la question : on demande les octets du texte décomposé (NFD)": 14,
          "relis les plages du tableau d'UTF-8 : dans laquelle tombent U+0300 et U+0308 ?": 18,
          "en NFD, chaque lettre accentuée devient deux points de code : compte les octets des deux": 12}'''),
        ("f", BYTES, "utf8_two_bytes(0x153)", r'''mistakes={"c'est le point de code U+0153 écrit sur deux octets, pas son écriture UTF-8": [0x01, 0x53],
          "le premier octet porte les 5 bits de poids fort du point de code : il ne vaut pas toujours C3": [0xC3, 0x93],
          "le second octet commence par les bits 10": [0xC5, 0x13],
          "le premier octet commence par les bits 110": [0x05, 0x93]}'''),
        ("g", TF, "False", r'''mistakes={"sans errors=, que fait decode d'octets qui ne forment pas de l'UTF-8 valide ?": True}'''),
    ]),
    Paper("B2.2", "Analogie à la main : roi − homme + femme sur des vecteurs de dimension 3", [
        ("a", DEC2, "float(Q_B22 @ E_B22[1])", r'''decimals=2, mistakes={"calcule d'abord la requête roi − homme + femme, puis son produit scalaire avec reine": float(E_B22[0] @ E_B22[1]),
          "la requête est roi − homme + femme : n'oublie pas de retrancher homme": float((E_B22[0] + E_B22[3]) @ E_B22[1])}'''),
        ("b", DEC3, "COS_B22[1]", r'''decimals=3, mistakes={"divise aussi par la norme de la requête": float(Q_B22 @ E_B22[1]) / np.linalg.norm(E_B22[1]),
          "divise aussi par la norme de reine": float(Q_B22 @ E_B22[1]) / np.linalg.norm(Q_B22),
          "le produit scalaire seul n'est pas le cosinus : divise par les deux normes": float(Q_B22 @ E_B22[1])}'''),
        ("c", DEC3, "COS_B22[3]", r'''decimals=3, mistakes={"divise aussi par la norme de la requête": float(Q_B22 @ E_B22[3]) / np.linalg.norm(E_B22[3]),
          "divise aussi par la norme de femme": float(Q_B22 @ E_B22[3]) / np.linalg.norm(Q_B22),
          "le produit scalaire seul n'est pas le cosinus : divise par les deux normes": float(Q_B22 @ E_B22[3])}'''),
        ("d", WORD, 'analogy_b22(E_B22, "homme", "roi", "femme", exclude=False)', r'''mistakes={"relis la question : ici, aucun mot n'est exclu": "reine",
          "compare les cosinus des sept mots, à 3 décimales": "princesse",
          "relis la question : on compare des cosinus, qui divisent par les normes": "trône"}'''),
        ("e", WORD, 'analogy_b22(E_B22, "homme", "roi", "femme")', r'''mistakes={"on exclut les trois mots de la question, femme comprise": "femme",
          "recalcule les cosinus des mots restants, à 3 décimales": "princesse",
          "relis la question : on compare des cosinus, qui divisent par les normes": "trône"}'''),
        ("f", WORD, 'analogy_b22(E_B22, "homme", "roi", "femme", cosine=False)', r'''mistakes={"relis la question : on compare ici des produits scalaires bruts, sans diviser par les normes": "reine",
          "n'oublie pas l'exclusion des trois mots de la question": "femme"}'''),
        ("g", TF, 'analogy_b22(E_UNIT_B22, "homme", "roi", "femme") == "reine"', r'''mistakes={"recalcule les cosinus avec les vecteurs ramenés à la norme 1 : le premier mot non exclu change-t-il ?": False}'''),
    ]),
    Paper("B2.3", "Trois fusions BPE à la main sur un mini-corpus", [
        ("a", WHOLE, "len(ALPHABET_B23)", r'''mistakes={"« é » est un caractère de l'alphabet comme les autres": 4}'''),
        ("b", WHOLE, "ALPHABET_B23.index('é')", r'''mistakes={"l'alphabet est trié par point de code, pas dans l'ordre alphabétique du français : où vient U+00E9 ?": 0,
          "les identifiants commencent à 0": 5}'''),
        ("c", PAIR, "list(MERGES_B23[0])", r'''mistakes={"à égalité, la règle prend la plus petite paire d'identifiants, pas la première lue dans le mot": [4, 3],
          "relis les identifiants de l'alphabet : il est trié par point de code": [0, 4],
          "les identifiants viennent de l'alphabet trié par point de code, et la paire se lit dans l'ordre du mot": [4, 0],
          "recompte cette paire : est-elle la plus fréquente ?": [4, 2]}'''),
        ("d", WHOLE, "COUNTS_B23[0]", r'''mistakes={"multiplie par la fréquence de chaque mot": 2,
          "on demande le compte de la seule paire fusionnée": 14}'''),
        ("e", PAIR, "list(MERGES_B23[1])", r'''mistakes={"relis le départage de la première fusion": [5, 4],
          "après la première fusion, « t » et « é » ne sont plus voisins dans « été »": [4, 3],
          "recompte cette paire après la première fusion : est-elle la plus fréquente ?": [5, 2]}'''),
        ("f", PAIR, "list(MERGES_B23[2])", r'''mistakes={"à égalité, la plus petite paire d'identifiants passe d'abord : quel est l'identifiant du token fusionné ?": [6, 2],
          "à égalité, compare d'abord les premiers identifiants des paires": [1, 4]}'''),
        ("g", WHOLE, "TOKENS_AFTER_B23", r'''mistakes={"c'est le nombre de mots du corpus : compte les tokens de chaque mot": 10,
          "multiplie le nombre de tokens de chaque mot par sa fréquence": 8,
          "c'est le nombre de caractères : applique les trois fusions": 34}'''),
        ("h", TF, "True", r'''mistakes={"relis la règle d'arrêt : on s'arrête quand la meilleure paire apparaît moins de min_frequency fois": False}'''),
    ]),
    Paper("B2.4", "Encoder un mot jamais vu avec des fusions déjà apprises", [
        ("a", IDS, 'encode_b24("chantons")', r'''mistakes={"applique les fusions jusqu'au bout : aucune paire voisine ne doit rester dans la table": [13, 12, 7]}'''),
        ("b", IDS, 'encode_b24("ont")', r'''mistakes={"relis la question : on applique les fusions par rang": encode_b24("ont", by_rank=False)}'''),
        ("c", IDS, 'encode_b24("ont", by_rank=False)', r'''mistakes={"relis la question : on suit ici l'erreur de lecture, la paire connue la plus à gauche": encode_b24("ont")}'''),
        ("d", IDS, 'encode_b24("chanteront")', r'''mistakes={"après la fusion « nt », « o » et « n » sont-ils encore voisins ?": encode_b24("chanteront", by_rank=False),
          "applique toutes les fusions de la table, jusqu'au rang 7": [13, 14, 5, 9]}'''),
        ("e", IDS, 'encode_b24("chantez")', r'''mistakes={"relis où les tokens spéciaux se placent dans la numérotation (§202.3)": [13, 2, 0],
          "un caractère inconnu devient un token, il ne disparaît pas": [13, 2]}'''),
        ("f", WHOLE, 'len("été".encode("utf-8"))', r'''mistakes={"au niveau octet, on compte des octets : combien en prend « é » ?": 3,
          "recompte les octets de « t » : dans quelle plage du tableau d'UTF-8 tombe-t-il ?": 6}'''),
        ("g", TF, "False", r'''mistakes={"que rend decode pour l'identifiant de <unk> ?": True}'''),
    ]),
    Paper("B2.5", "One-hot fois matrice : nn.Embedding à la main, et quelles lignes reçoivent un gradient", [
        ("a", SHAPE, '"(4, 5)"', r'''mistakes={"une ligne par position, une colonne par token du vocabulaire": "(5, 4)",
          "le one-hot a V colonnes, pas d": "(4, 3)"}'''),
        ("b", VECTOR, "OUT_B25[1].tolist()", r'''mistakes={"les positions sont numérotées à partir de 0 : t = 1 est la deuxième": OUT_B25[0].tolist(),
          "tu as pris la ligne 1 de la table : quel token est lu à la position t = 1 ?": E_B25[1].tolist()}'''),
        ("c", DEC1, "L_B25", r'''decimals=1, mistakes={"le token 3 apparaît deux fois : compte ses deux positions": -7.0}'''),
        ("d", VECTOR, "GRAD_B25[3].tolist()", r'''decimals=1, mistakes={"le token 3 apparaît deux fois : sa ligne reçoit la somme des deux gradients": W_B25.tolist(),
          "le gradient de w · e par rapport à e est w, pas e": (2 * E_B25[3]).tolist()}'''),
        ("e", WHOLE, "int((np.abs(GRAD_B25).sum(axis=1) > 0).sum())", r'''mistakes={"une ligne lue deux fois ne compte qu'une fois": 4,
          "les lignes jamais lues ne reçoivent rien": 5}'''),
        ("f", VECTOR, "(E_B25[3] - LR_B25 * GRAD_B25[3]).tolist()", r'''decimals=2, mistakes={"un pas de SGD retranche η fois le gradient": (E_B25[3] + LR_B25 * GRAD_B25[3]).tolist(),
          "la ligne 3 reçoit la somme des gradients de ses deux positions": (E_B25[3] - LR_B25 * W_B25).tolist()}'''),
        ("g", TF, "False", r'''mistakes={"relis ce que fait padding_idx (§202.7)": True}'''),
        ("h", TF, "True", r'''mistakes={"relis ce que la fiche dit des lignes qui ne sont pas lues, et de l'optimiseur (§202.7)": False}'''),
    ]),
    Paper("B2.6", "Co-occurrences et PPMI à la main (fenêtre de 1, quatre mots)", [
        ("a", WHOLE, "int(C_B26[0, 2])", r'''mistakes={"la fenêtre est symétrique : compte aussi « le » à gauche de « chat »": 1,
          "la fenêtre est symétrique : compte aussi « le » à droite de « chat »": 2}'''),
        ("b", WHOLE, "int(N_B26)", r'''mistakes={"chaque paire voisine compte dans les deux sens": 7,
          "un texte de n tokens a n − 1 paires voisines": 16}'''),
        ("c", DEC3, 'pmi_b26("chat", "le")', r'''decimals=3, mistakes={"la PMI est le logarithme du rapport": 2.1,
          "prends le logarithme naturel, pas en base 2": math.log2(2.1),
          "divise chaque somme par N, comme le compte de la paire": math.log(3 / 20)}'''),
        ("d", DEC3, 'pmi_b26("court", "le")', r'''decimals=3, mistakes={"relis la question : on demande la PMI, avant le max à 0": 0.0,
          "la PMI est le logarithme du rapport": 14 / 15,
          "prends le logarithme naturel, pas en base 2": math.log2(14 / 15)}'''),
        ("e", DEC3, "P_ALPHA_B26[WORDS_B26.index('le')]", r'''decimals=3, mistakes={"c'est p(le) sans lissage : élève les sommes des colonnes à la puissance α": 5 / 14}'''),
        ("f", DEC3, 'pmi_b26("chat", "le", alpha=0.75)', r'''decimals=3, mistakes={"relis la question : on demande la PPMI lissée, avec α = 0,75": pmi_b26("chat", "le"),
          "α ne porte que sur la loi des contextes, pas sur celle des mots": pmi_b26("chat", "le", 0.75, alpha_rows=True),
          "le mot est la ligne, le contexte la colonne : relis l'ordre": pmi_b26("le", "chat", 0.75)}'''),
        ("g", TF, "False", r'''mistakes={"compare PPMI(chat, le) et PPMI(le, chat) avec α = 0,75": True}'''),
        ("h", TF, "True", r'''mistakes={"recalcule PMI(court, le) avec p_α(le) au lieu de p(le)": False}'''),
    ]),
    Paper("B2.9", "Fermi : poids de la table d'embedding de GPT-2 et d'un LLM à 128 000 tokens", [
        ("a", WHOLE, "GPT2_TABLE_B29", r'''mistakes={"une table de V lignes et de d colonnes : multiplie": 50257 + 768,
          "GPT-2 lie ses poids : la tête de sortie est la même matrice que la table": 2 * GPT2_TABLE_B29}'''),
        ("b", DEC1, "100 * GPT2_TABLE_B29 / GPT2_TOTAL_B29", r'''decimals=1, mistakes={"on demande un pourcentage, pas une proportion": GPT2_TABLE_B29 / GPT2_TOTAL_B29,
          "GPT-2 lie ses poids : la table n'est comptée qu'une fois": 200 * GPT2_TABLE_B29 / GPT2_TOTAL_B29}'''),
        ("c", DEC1, "GPT2_TABLE_B29 * 4 / 1e6", r'''decimals=1, mistakes={"tu as compté des nombres : combien d'octets occupe chacun en float32 ?": GPT2_TABLE_B29 / 1e6,
          "1 Mo vaut ici 10⁶ octets, pas 2²⁰": GPT2_TABLE_B29 * 4 / 2 ** 20,
          "tu as compté des bits : on demande des octets": GPT2_TABLE_B29 * 32 / 1e6}'''),
        ("d", WHOLE, "L31_TABLE_B29", r'''mistakes={"une table de V lignes et de d colonnes : multiplie": 128256 + 4096,
          "relis la question : on demande la table d'entrée seule": 2 * L31_TABLE_B29}'''),
        ("e", DEC1, "200 * L31_TABLE_B29 / L31_TOTAL_B29", r'''decimals=1, mistakes={"compte les deux tables : Llama 3.1 8B ne lie pas ses poids": 100 * L31_TABLE_B29 / L31_TOTAL_B29,
          "on demande un pourcentage, pas une proportion": 2 * L31_TABLE_B29 / L31_TOTAL_B29}'''),
        ("f", WHOLE, "round(100 * L32_TABLE_B29 / L32_TOTAL_B29)", r'''mistakes={"Llama 3.2 1B lie ses poids : une seule table": round(200 * L32_TABLE_B29 / L32_TOTAL_B29),
          "on demande un pourcentage, pas une proportion": 0},
          fractional="arrondis le pourcentage à l'entier le plus proche"'''),
        ("g", TF, "L31_TABLE_B29 * 2 > 1e9", r'''mistakes={"relis l'énoncé : combien d'octets par nombre en bfloat16, et combien de paramètres dans la table ?": False}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ B2.1 à B2.6, 🧮 B2.9)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"B\"`) ; "
               "une forme de tenseur : entre guillemets (`\"(2, 3)\"`) ; un mot : entre guillemets (`\"mot\"`) ; "
               "des octets : une liste, en hexadécimal ou en décimal (`[0x4E, 0x6F]` ou `[78, 111]`) ; des "
               "identifiants, une paire d'identifiants ou un vecteur : une liste (`[4, 0, 7]`). Arrondis comme "
               "l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. "
               "Les questions « dans ta copie », la dérivation ∂ B2.7, l'estimation 🧮 B2.8, la réflexion (🗣️ B2.10, "
               "⚖️ B2.11, 📄 B2.12) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to E (exercises B2.13 to B2.31): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "B2.13 à B2.14", "des tokenizers réels sur Holmes et Verne, et ce que coûtent les langues, les nombres et "
                           "les emoji"),
    ("B", "B2.15 à B2.19", "ton BPE : compter et fusionner les paires, apprendre les fusions, encoder et décoder, "
                           "réparer un tokenizer qui abîme les accents, te comparer à la bibliothèque tokenizers"),
    ("C", "B2.20 à B2.21", "expériences : compression et entropie selon la taille du vocabulaire, tokenizer d'une "
                           "langue appliqué à l'autre"),
    ("D", "B2.22 à B2.27", "vecteurs de mots : co-occurrences et PPMI, cosinus et voisins, PPMI + SVD sur Holmes, "
                           "word2vec sur Verne, voisins de « Fogg » et « Passepartout », carte des embeddings"),
    ("E", "B2.28 à B2.31", "embeddings de phrases et recherche sémantique, statique contre contextuel, sauvegarder "
                           "un tokenizer, et le défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | B2.Q1 à B2.Q12, B2.R1 à B2.R3, B2.1 à B2.6, B2.9 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 "
            "| ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(ex.table_row(part.key if i == 0 else ""))
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# B2 · Tokenisation et embeddings — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/b2_tokenisation/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py B2`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# B2 · Tokenisation et embeddings — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Manipuler Unicode, UTF-8 et la normalisation d'un texte, et compter ses octets et ses points de "
               "code.\n"
               "- Écrire un tokenizer BPE (entraînement, encodage par rang, décodage) et le comparer à la "
               "bibliothèque `tokenizers`.\n"
               "- Comparer des tokenizers réels et mesurer leur coût (fertilité, compression) selon la langue.\n"
               "- Relier `nn.Embedding` à un produit one-hot × table, et entraîner des embeddings word2vec.\n"
               "- Construire des embeddings par co-occurrences (PPMI et SVD), les interroger par le cosinus, et "
               "chercher des passages par des embeddings de phrases.\n\n"
               "**Rappel express.** Fertilité $= \\frac{\\text{tokens}}{\\text{mots}}$, compression $= "
               "\\frac{\\text{octets}}{\\text{tokens}}$. Fusion BPE : la paire voisine la plus fréquente "
               "(à égalité, la plus petite paire d'identifiants) devient un token ; on encode en appliquant les "
               "fusions par rang. $\\mathbf{x}_{\\text{one-hot}}^\\top \\mathbf{E} = \\mathbf{E}[i, :]$. "
               "$\\operatorname{PPMI}(w, c) = \\max\\big(0, \\log \\frac{p(w, c)}{p(w)\\,p(c)}\\big)$. Skip-gram : "
               "$L = -\\log \\sigma(\\mathbf{u}_c^\\top \\mathbf{v}_w) - \\sum_k \\log \\sigma(-\\mathbf{u}_{n_k}^\\top "
               "\\mathbf{v}_w)$. Mean pooling : $\\mathbf{s} = \\frac{\\sum_t m_t \\mathbf{h}_t}{\\sum_t m_t}$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code B2.13 à B2.31 (parties A à E : tokenizers réels, ton BPE, expériences, "
             "vecteurs de mots, embeddings de phrases, livraison et défi) seront ajoutés à la prochaine session de "
             "génération ; relance alors `python tools/start_chapter.py B2` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu compter les octets et les points de code d'un texte, et écrire un caractère en UTF-8 ?\n"
               "2. Sais-tu dérouler un entraînement BPE à la main, départage compris, puis encoder un mot nouveau "
               "par rang ?\n"
               "3. Sais-tu dire quelles lignes d'une table d'embedding reçoivent un gradient, et calculer une "
               "PPMI ?\n"
               "4. Sais-tu chiffrer ce que coûtent une table d'embedding et un texte selon sa langue ?\n\n"
               "**Pour aller plus loin** : le chapitre 6 du cours LLM de Hugging Face et `minbpe` d'A. Karpathy, "
               "cités dans la fiche. La suite : le bonus B3, attention et Transformers."
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
