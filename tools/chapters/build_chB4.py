#!/usr/bin/env python
"""Build the two notebooks of bonus chapter B4 from a single source (used by Claude).

    python tools/chapters/build_chB4.py
    python tools/run_all_notebooks.py chapitres/b4_llm/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/b4_llm/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3 and of the ✏️ paper exercises
B4.1 to B4.5 (02_exercices.md). The derivation ∂ B4.6, the estimate 🧮 B4.7, the reflection (🗣️ B4.8, ⚖️ B4.9 and
B4.10, 📄 B4.11) and the interview are corrected with 05_solutions.md. The answers are computed by hand formulas in
the context cell of the solutions notebook, then checked against the reference library (mylearn.retrieval,
mylearn.embeddings), scikit-learn, transformers and peft (models built from local configurations on the meta device,
the chat template of SmolLM2 rendered by transformers' own Jinja renderer): part 0 downloads nothing. If the tokenizer
of SmolLM2-135M-Instruct happens to be in the local Hugging Face cache, the token counts of ✏️ B4.4 are also compared
with it (`local_files_only=True`).
Parts A and B (B4.12 to B4.21: usage and prompting, retrieval and the RAG) come from chB4_parts.py and chB4_code.py,
with the replies and the calibrations precomputed by history_chB4.py; B4.22 to B4.30 (the rest of the RAG, fine-tuning,
delivery) come at the next generation session (NEXT_SESSION).
"""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (Paper, badge, check_unique_keys, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "B4"
FOLDER = "chapitres/b4_llm"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = r'''# The short answers only use the numbers of their statements (02_exercices.md)
import math
from fractions import Fraction

import numpy as np

MB, GB = 10 ** 6, 10 ** 9                                           # the units of the chapter (decimal)

# ✏️ B4.1: SmolLM2-135M-Instruct (config.json, revision 12fd25f)
N_B41, LAYERS_B41, D_B41, V_B41, KV_HEADS_B41, DH_B41, CTX_B41 = 134_515_008, 30, 576, 49_152, 3, 64, 8_192
EMB_B41 = V_B41 * D_B41                                             # the token table (tied with the output layer)
EMB_SHARE_B41 = 100 * EMB_B41 / N_B41
FP32_B41, BF16_B41, INT4_B41 = 4 * N_B41 / MB, 2 * N_B41 / MB, 0.5 * N_B41 / MB
KV_TOKEN_B41 = 2 * LAYERS_B41 * KV_HEADS_B41 * DH_B41 * 2           # keys and values, every layer, bf16
KV_FULL_B41 = KV_TOKEN_B41 * CTX_B41 / MB
CONVERSATIONS_B41 = int((2 * GB - 2 * N_B41) // (KV_TOKEN_B41 * CTX_B41))
QWEN_FOOTPRINT, QWEN_SECONDS, QWEN_TOKENS = 1_976_131_328, 8.4, 64   # fiche §204.2, Qwen2.5-0.5B-Instruct in float32
SPEED_B41 = QWEN_TOKENS / QWEN_SECONDS * QWEN_FOOTPRINT / (4 * N_B41)

# ✏️ B4.2: five queries, five retrieved ids each
RETRIEVED_B42 = [[3, 1, 4, 0, 2], [5, 9, 2, 6, 8], [0, 7, 1, 2, 3], [6, 4, 8, 9, 1], [2, 0, 9, 3, 7]]
RELEVANT_B42 = [{1}, {2, 8}, {0, 5}, {7}, {3, 4, 9}]


def recall_b42(k, retrieved=RETRIEVED_B42, relevant=RELEVANT_B42, denominator="relevant"):
    """Mean of |top-k ∩ relevant| / |relevant| (denominator="k": the precision@k mistake)."""
    ratios = [len(set(r[:k]) & rel) / (len(rel) if denominator == "relevant" else k)
              for r, rel in zip(retrieved, relevant)]
    return sum(ratios) / len(ratios)


def first_rank_b42(r, rel):
    return next((i + 1 for i, doc in enumerate(r) if doc in rel), None)


MRR_B42 = sum(1 / rank if (rank := first_rank_b42(r, rel)) else 0
              for r, rel in zip(RETRIEVED_B42, RELEVANT_B42)) / 5
HIT3_B42 = sum(any(doc in rel for doc in r[:3]) for r, rel in zip(RETRIEVED_B42, RELEVANT_B42)) / 5
FULL5_B42 = sum(set(r[:5]) >= rel for r, rel in zip(RETRIEVED_B42, RELEVANT_B42))

# ✏️ B4.3: LoRA on Qwen2.5-0.5B-Instruct (24 layers, d_model = 896, k and v 896 -> 128, MLP 896 -> 4 864)
N_QWEN, LAYERS_QWEN, D_QWEN, KV_QWEN, MLP_QWEN, R_B43 = 494_032_768, 24, 896, 128, 4_864, 8


def lora_params(d_out, d_in, r=R_B43):
    return r * (d_out + d_in)


Q_B43, K_B43 = lora_params(D_QWEN, D_QWEN), lora_params(KV_QWEN, D_QWEN)
Q_SHARE_B43, K_SHARE_B43 = 100 * Q_B43 / D_QWEN ** 2, 100 * K_B43 / (KV_QWEN * D_QWEN)
LAYER_B43 = 2 * Q_B43 + 2 * K_B43                                   # q, k, v, o
MODEL_B43 = LAYERS_QWEN * LAYER_B43
TRAINABLE_PCT_B43 = 100 * MODEL_B43 / (N_QWEN + MODEL_B43)
ALL_LINEAR_B43 = LAYERS_QWEN * (LAYER_B43 + 3 * lora_params(MLP_QWEN, D_QWEN))
ADAPTER_MB_B43 = 4 * MODEL_B43 / MB
NO_SAVING_RANK_B43 = math.ceil(D_QWEN ** 2 / (2 * D_QWEN))           # r (d + k) >= d k

# ✏️ B4.4: the chat template of SmolLM2-135M-Instruct (revision 12fd25f), and the tokens of each piece
SMOL_TEMPLATE = ("{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
                 "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face"
                 "<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
                 "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
                 "{% endif %}")
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
S_B44 = {"role": "system", "content": "Answer in one short sentence."}
U_B44 = {"role": "user", "content": "Where does Holmes live?"}
A_B44 = {"role": "assistant", "content": "In London, at 221B Baker Street."}
U2_B44 = {"role": "user", "content": "Who is his friend?"}
CONTENT_TOKENS = {DEFAULT_SYSTEM: 16, S_B44["content"]: 6, U_B44["content"]: 5, A_B44["content"]: 12,
                  U2_B44["content"]: 5}
ROLE_TOKENS = {"system": 1, "user": 1, "assistant": 2}


def by_hand_b44(messages, add_generation_prompt):
    """The chat template applied by hand: the string, its special tokens and its token count."""
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}] + messages
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    specials = 2 * len(messages)
    tokens = sum(1 + ROLE_TOKENS[m["role"]] + 1 + CONTENT_TOKENS[m["content"]] + 1 + 1 for m in messages)
    if add_generation_prompt:
        text, specials, tokens = text + "<|im_start|>assistant\n", specials + 1, tokens + 1 + 2 + 1
    return text, specials, tokens


U_GEN_B44 = by_hand_b44([U_B44], True)
SU_GEN_B44, SU_NOGEN_B44 = by_hand_b44([S_B44, U_B44], True), by_hand_b44([S_B44, U_B44], False)
TURN2_B44 = by_hand_b44([S_B44, U_B44, A_B44, U2_B44], True)
DOUBLE_SYSTEM_B44 = by_hand_b44([{"role": "system", "content": DEFAULT_SYSTEM}, S_B44, U_B44], True)

# ✏️ B4.5: BM25 by hand (k1 = 1.5, b = 0.75), three documents, query "nemo nautilus"
DOCS_B45 = [["nemo", "commande", "le", "nautilus"], ["le", "nautilus", "plonge"],
            ["le", "nautilus", "croise", "un", "nautilus", "au", "grand", "large"]]
QUERY_B45 = ["nemo", "nautilus"]


def bm25_b45(query=QUERY_B45, docs=DOCS_B45, k1=1.5, b=0.75, idf_form="plus_one"):
    """BM25 scores of the docstring; idf_form="original" drops the "1 +" of the idf."""
    n, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    scores = np.zeros(n)
    for t in query:
        n_t = sum(t in doc for doc in docs)
        ratio = (n - n_t + 0.5) / (n_t + 0.5)
        idf = math.log(1 + ratio) if idf_form == "plus_one" else math.log(ratio)
        for i, doc in enumerate(docs):
            f = doc.count(t)
            if f:
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / avgdl))
    return scores


AVGDL_B45 = sum(map(len, DOCS_B45)) / 3
IDF_NAUTILUS_B45 = math.log(1 + 0.5 / 3.5)
SCORES_B45 = bm25_b45()
RANKING_B45 = [int(i) for i in np.argsort(-SCORES_B45, kind="stable")]
SCORES_B0_B45 = bm25_b45(b=0.0)
SCORES_K0_B45 = bm25_b45(k1=0.0)
SCORES_ORIGINAL_B45 = bm25_b45(idf_form="original")

# quizzes and recalls with a computation
Q3A, Q3B = 2 * 3e9 / GB, 0.5 * 3e9 / GB
Q8C = math.sqrt(0.8 * 0.2 / 100)
Q9E = 2 / (60 + 3)
Q11C = 4 * (512 + 512)
R1C = 2 * 24 * 2 * 64 * 2
A_R3, B_R3 = np.array([3.0, 4.0]), np.array([6.0, 8.0])
COS_R3 = float(A_R3 @ B_R3 / (np.linalg.norm(A_R3) * np.linalg.norm(B_R3)))
DIST_R3 = float(np.linalg.norm(A_R3 - B_R3))
SQ_R3 = 2 - 2 * 0.82
MULT_R3 = 50_000 * 768
'''

LIBRARY_CHECKS = r'''
# The same answers from the libraries (no download): mylearn (reference), scikit-learn, transformers and peft
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config
from transformers.utils.chat_template_utils import render_jinja_template

R = mylearn.retrieval
# ✏️ B4.2: recall_at_k and mean_reciprocal_rank of the reference
for k, expected in ((1, recall_b42(1)), (3, recall_b42(3)), (5, recall_b42(5))):
    assert math.isclose(R.recall_at_k(RETRIEVED_B42, RELEVANT_B42, k=k), expected)
assert math.isclose(R.mean_reciprocal_rank(RETRIEVED_B42, RELEVANT_B42), MRR_B42)
assert R.mean_reciprocal_rank(RETRIEVED_B42, RELEVANT_B42) >= R.recall_at_k(RETRIEVED_B42, RELEVANT_B42, k=1)

# ✏️ B4.5: BM25 of the reference, then the same corpus with b = 0, k1 = 0 and the original idf
bm25 = R.BM25(k1=1.5, b=0.75).fit(DOCS_B45)
assert bm25.avgdl_ == AVGDL_B45 and math.isclose(bm25.idf_["nautilus"], IDF_NAUTILUS_B45)
assert np.allclose(bm25.get_scores(QUERY_B45), SCORES_B45)
assert np.allclose(R.BM25(k1=1.5, b=0.0).fit(DOCS_B45).get_scores(QUERY_B45), SCORES_B0_B45)
assert np.allclose(R.BM25(k1=0.0, b=0.75).fit(DOCS_B45).get_scores(QUERY_B45), SCORES_K0_B45)
assert SCORES_ORIGINAL_B45[0] < 0                                   # below a document with no query word (score 0)
assert RANKING_B45 == [int(i) for i in np.argsort(-bm25.get_scores(QUERY_B45), kind="stable")]

# quiz Q9 e: a document ranked 3rd in two rankings (reciprocal_rank_fusion only returns the ids: the score by hand)
assert R.reciprocal_rank_fusion([[1, 2, 7], [3, 4, 7]])[0] == 7 and math.isclose(float(Fraction(2, 63)), Q9E)

# rappel R3: cosine and Euclidean distance, scikit-learn and the B2 reference
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
assert math.isclose(cosine_similarity([A_R3], [B_R3])[0, 0], COS_R3)
assert math.isclose(mylearn.embeddings.cosine_similarity_matrix([A_R3], [B_R3])[0, 0], COS_R3)
assert math.isclose(euclidean_distances([A_R3], [B_R3])[0, 0], DIST_R3)
u = np.array([1.0, 0.0]); v = np.array([0.82, math.sqrt(1 - 0.82 ** 2)])
assert math.isclose(float(np.sum((u - v) ** 2)), SQ_R3)

# ✏️ B4.1: SmolLM2-135M built from its configuration on the meta device (shapes only, no download)
smol_config = LlamaConfig(hidden_size=576, intermediate_size=1536, num_hidden_layers=30, num_attention_heads=9,
                          num_key_value_heads=3, vocab_size=49152, max_position_embeddings=8192,
                          tie_word_embeddings=True)
with torch.device("meta"):
    smol = AutoModelForCausalLM.from_config(smol_config)
assert smol.num_parameters() == N_B41 and smol.model.embed_tokens.weight.numel() == EMB_B41
assert smol.lm_head.weight is smol.model.embed_tokens.weight
head_dim = smol_config.hidden_size // smol_config.num_attention_heads
assert 2 * smol_config.num_hidden_layers * smol_config.num_key_value_heads * head_dim * 2 == KV_TOKEN_B41

# ✏️ B4.3 and rappel R1: Qwen2.5-0.5B built from its configuration, then wrapped by peft (r = 8)
qwen_config = Qwen2Config(hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
                          num_key_value_heads=2, vocab_size=151936, max_position_embeddings=32768,
                          tie_word_embeddings=True)


def peft_counts(target_modules, r=8):
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(qwen_config)
    return get_peft_model(base, LoraConfig(r=r, lora_alpha=16, target_modules=target_modules, lora_dropout=0.0,
                                           task_type="CAUSAL_LM")).get_nb_trainable_parameters()


with torch.device("meta"):
    qwen = AutoModelForCausalLM.from_config(qwen_config)
assert qwen.num_parameters() == N_QWEN
attention = qwen.model.layers[0].self_attn
assert attention.k_proj.weight.shape == (KV_QWEN, D_QWEN) and attention.q_proj.weight.shape == (D_QWEN, D_QWEN)
trainable, total = peft_counts(["q_proj", "k_proj", "v_proj", "o_proj"])
assert trainable == MODEL_B43 and total == N_QWEN + MODEL_B43
assert math.isclose(100 * trainable / total, TRAINABLE_PCT_B43)
assert peft_counts("all-linear")[0] == ALL_LINEAR_B43
assert peft_counts(["q_proj", "k_proj", "v_proj", "o_proj"], r=16)[0] == 2 * MODEL_B43
layer = nn.Sequential(nn.Linear(D_QWEN, D_QWEN))                     # one 896 x 896 layer, as in a)
assert get_peft_model(layer, LoraConfig(r=8, target_modules=["0"])).get_nb_trainable_parameters()[0] == Q_B43
assert 2 * qwen_config.num_hidden_layers * qwen_config.num_key_value_heads * 64 * 2 == R1C

# ✏️ B4.4: the template rendered by transformers' own Jinja renderer, as apply_chat_template does
def render_b44(messages, add_generation_prompt):
    return render_jinja_template(conversations=[messages], chat_template=SMOL_TEMPLATE,
                                 add_generation_prompt=add_generation_prompt)[0][0]


for messages, gen, by_hand in (([U_B44], True, U_GEN_B44), ([S_B44, U_B44], True, SU_GEN_B44),
                               ([S_B44, U_B44], False, SU_NOGEN_B44), ([S_B44, U_B44, A_B44, U2_B44], True, TURN2_B44)):
    rendered = render_b44(messages, gen)
    assert rendered == by_hand[0], (rendered, by_hand[0])
    assert rendered.count("<|im_start|>") + rendered.count("<|im_end|>") == by_hand[1]
try:                                                                 # optional: the real tokenizer, if already cached
    from transformers import AutoTokenizer
    smol_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct",
                                             revision="12fd25f77366fa6b3b4b768ec3050bf629380bac", local_files_only=True)
except OSError:
    print("ℹ️ SmolLM2's tokenizer is not in the local cache: token counts checked when the chapter was generated")
else:
    assert smol_tok.chat_template == SMOL_TEMPLATE
    for messages, gen, by_hand in (([U_B44], True, U_GEN_B44), ([S_B44, U_B44], True, SU_GEN_B44),
                                   ([S_B44, U_B44], False, SU_NOGEN_B44),
                                   ([S_B44, U_B44, A_B44, U2_B44], True, TURN2_B44)):
        ids = smol_tok.apply_chat_template(messages, add_generation_prompt=gen, return_dict=True)["input_ids"]
        assert len(ids) == by_hand[2] and len(smol_tok(by_hand[0])["input_ids"]) == by_hand[2]
    print("✅ token counts of ✏️ B4.4 checked with the tokenizer of SmolLM2-135M-Instruct (local cache)")
print("✅ the paper answers agree with mylearn (reference), scikit-learn, transformers and peft")
'''

mylearn = importlib.import_module("mylearn_ref")   # the builder computes the mistakes with the same helpers
for _module in ("retrieval", "embeddings"):
    importlib.import_module(f"mylearn_ref.{_module}")
exec(PAPER_CONTEXT)

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
WHOLE = "a whole number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
DEC4 = "a number, 4 decimals"
IDS = "a list of document ids, best first"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    Paper("B4.Q1", "Modèle de base, modèle instruct, modèle de chat : lequel pour quoi ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis « Trois étapes, trois sortes de modèles » : ce que l'on veut ici est-il un dialogue ?",
                                           "C": "relis l'étape 1 de « Trois étapes » : quel modèle continue n'importe quel texte ?"})),
        ("b", TF, "True", r'''mistakes={"compare les tokens du pré-entraînement et les paires du SFT (§204.1)": False}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "relis le paragraphe sur les noms du Hub : Qwen3 fait l'inverse des autres",
                                           "B": "relis le paragraphe sur les noms du Hub : que signifie le suffixe « -Instruct » ?"})),
        ("d", LETTER, '"C"', letters("C", {"A": "relis l'étape 1 de « Trois étapes » : que produit le pré-entraînement ?",
                                           "B": "un gabarit met en forme une conversation : apprend-il au modèle à suivre une consigne ?"})),
        ("e", TF, "False", r'''mistakes={"relis l'étape 1 de « Trois étapes » : qu'est-ce qui peut suivre une question dans un texte ?": True}'''),
    ]),
    Paper("B4.Q2", "« Open weights » n'est pas « open source » : lire la licence d'un modèle", [
        ("a", TF, "True", r'''mistakes={"relis la première puce de « Licences » (§204.1)": False}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "la licence d'origine impose-t-elle une autre licence aux dérivés ? Relis « Licences »",
                                           "C": "relis ce que la licence de Llama 3.2 demande pour un modèle dérivé redistribué"})),
        ("c", TF, "False", r'''mistakes={"relis le paragraphe sur l'Open Source AI Definition : que demande-t-elle en plus des poids ?": True}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "relis la troisième puce de « Licences » : que faut-il pour un accès contrôlé ?",
                                           "B": "un miroir non autorisé respecte-t-il les conditions de l'éditeur ?"})),
        ("e", LETTER, '"A"', letters("A", {"B": "transformers exécute ces modèles : le code d'inférence est-il caché ?",
                                           "C": "from_pretrained lit config.json : ce fichier manque-t-il ?"})),
    ]),
    Paper("B4.Q3", "dtype, device et mémoire : ce qui décide si un modèle tient sur ton ordinateur", [
        ("a", DEC1, "Q3A", r'''decimals=1, mistakes={"combien d'octets par paramètre en bfloat16 ? Relis « La mémoire des poids »": 12.0,
          "relis la question : bfloat16, pas int4": 1.5, "1 Go = 10⁹ octets : vérifie la conversion": 6000.0}'''),
        ("b", DEC1, "Q3B", r'''decimals=1, mistakes={"relis la question : int4, à 0,5 octet par paramètre": 6.0,
          "combien d'octets par paramètre en int4 ? Relis « La mémoire des poids »": 3.0,
          "c'est le nombre de bits : on demande des Go": 12.0}'''),
        ("c", TF, "True", r'''mistakes={"relis l'encadré 🧮 sur float32 et bfloat16 : combien de bits d'exposant ?": False}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "relis la formule du KV cache (§204.2) : V y apparaît-il ?",
                                           "B": "relis la formule du KV cache : quelle quantité grandit pendant la conversation ?"})),
        ("e", TF, "False", r'''mistakes={"relis « Vitesse sur CPU » : compare les deux mesures de la fiche": True}'''),
    ]),
    Paper("B4.Q4", "Chat template : pourquoi ne pas écrire soi-même « User: … Assistant: … »", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis « Rôles et chat template » : où le gabarit est-il rangé ?",
                                           "C": "generation_config.json règle la génération : relis « Rôles et chat template »"})),
        ("b", TF, "True", r'''mistakes={"relis la première des trois choses à retenir (§204.3)": False}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "le tokenizer réécrit-il le texte que tu lui donnes ? Relis le paragraphe qui suit les trois choses à retenir",
                                           "B": "relis le paragraphe qui suit les trois choses à retenir : le modèle a-t-il vu ce format ?"})),
        ("d", TF, "False", r'''mistakes={"relis la fin du paragraphe sur le format écrit à la main (§204.3)": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "skip_special_tokens sert au décodage : relis la troisième chose à retenir",
                                           "C": "add_generation_prompt agit sur la fin de la chaîne : relis la troisième chose à retenir"})),
    ]),
    Paper("B4.Q5", "Glouton, température, top-p : lequel pour extraire une date, lequel pour écrire un poème ?", [
        ("a", LETTER, '"C"', letters("C", {"A": "une température haute rend-elle la sortie plus ou moins variable ?",
                                           "B": "une graine différente à chaque appel donne-t-elle la même réponse ?"})),
        ("b", LETTER, '"A"', letters("A", {"B": "le décodage glouton donne-t-il dix réponses différentes ?",
                                           "C": "relis la troisième puce de « generate() et ses réglages » : une température de 0 se passe-t-elle ?"})),
        ("c", TF, "False", r'''mistakes={"relis le piège classique sur generation_config.json (§204.3)": True}'''),
        ("d", TF, "True", r'''mistakes={"relis la température au ch. 22 : diviser les logits par 0,5 écarte-t-il ou rapproche-t-il les probabilités ?": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "top-p compte des probabilités, pas une part du vocabulaire (B3)",
                                           "C": "top-p cumule des probabilités : relis B3, top-k et top-p"})),
    ]),
    Paper("B4.Q6", "Zéro-shot, few-shot, chain-of-thought : vrai ou faux", [
        ("a", TF, "False", r'''mistakes={"relis la définition de l'apprentissage en contexte (§204.4)": True}'''),
        ("b", TF, "True", r'''mistakes={"relis « Un prompt est fragile » : que montrent Lu et coll. ?": False}'''),
        ("c", TF, "True", r'''mistakes={"relis « Zéro-shot et few-shot » : comment écrit-on les exemples pour un modèle de chat ?": False}'''),
        ("d", TF, "False", r'''mistakes={"relis la première réserve sur le chain-of-thought (§204.4)": True}'''),
        ("e", TF, "False", r'''mistakes={"relis la seconde réserve sur le chain-of-thought (§204.4)": True}'''),
    ]),
    Paper("B4.Q7", "Pourquoi un LLM invente : hallucination et token le plus probable", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis « Ce que fait vraiment un LLM » : consulte-t-il une base de faits ?",
                                           "C": "relis « Ce que fait vraiment un LLM » : comment une réponse entière se construit-elle ?"})),
        ("b", TF, "False", r'''mistakes={"relis la dernière parade de « Les hallucinations » (§204.10)": True}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "une année plausible pour un fait qui n'existe pas : est-ce juste ?",
                                           "B": "Holmes a-t-il pu recevoir un prix Nobel ? Relis les causes d'hallucination"})),
        ("d", TF, "True", r'''mistakes={"relis les parades de « Les hallucinations » : un RAG garantit-il une réponse vraie ?": False}'''),
        ("e", TF, "False", r'''mistakes={"relis le premier paragraphe de « Les hallucinations » : qu'a récompensé l'entraînement ?": True}'''),
    ]),
    Paper("B4.Q8", "Évaluer un LLM : exact match, juge LLM, contamination des benchmarks", [
        ("a", WHOLE, "1", r'''mistakes={"relis la normalisation de l'exact match : que deviennent la casse et la ponctuation ?": 0}'''),
        ("b", WHOLE, "1", r'''mistakes={"relis la définition de « contient la réponse » : on cherche la référence dans la réponse": 0}'''),
        ("c", DEC3, "Q8C", r'''decimals=3, mistakes={"c'est la variance : on demande l'écart-type": 0.0016,
          "relis la formule de l'écart-type d'une proportion (§204.5) : n est-il sous la racine ?": 0.004,
          "relis la formule : n = 100 questions": 0.4}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "relis « Le juge LLM » : un petit modèle fait-il un bon juge ?",
                                           "B": "l'échelle de la note change-t-elle le biais de position ?"})),
        ("e", TF, "True", r'''mistakes={"relis « La contamination » : quel indice la fiche donne-t-elle ?": False}'''),
    ]),
    Paper("B4.Q9", "BM25 ou recherche dense : qui retrouve « Watson » et qui retrouve « le docteur » ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "l'idf pondère les mots communs : BM25 connaît-il des synonymes ?",
                                           "C": "le passage 0 contient-il le mot de la requête ?"})),
        ("b", TF, "True", r'''mistakes={"relis la première des quatre idées de BM25 (§204.6)": False}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "la recherche dense utilise-t-elle BM25 ? Relis « La recherche dense »",
                                           "C": "un embedding compte-t-il des lettres ? Relis « La recherche dense »"})),
        ("d", LETTER, '"A"', letters("A", {"B": "relis le paragraphe sur les forces des deux recherches : un embedding connaît-il bien un code ?",
                                           "C": "relis le paragraphe sur les forces des deux recherches : se valent-elles sur un code ?"})),
        ("e", DEC4, "Q9E", r'''decimals=4, mistakes={"relis la formule de la RRF : les rangs partent de 1, et k s'ajoute au rang": 2 / (60 + 2),
          "relis la question : le document est dans deux classements": 1 / 63,
          "relis la formule de la RRF : k = 60 s'ajoute au rang": 2 / 3}'''),
    ]),
    Paper("B4.Q10", "RAG ou fine-tuning pour ajouter des connaissances ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis « RAG ou fine-tuning ? » : un fine-tuning apprend-il bien des faits précis ?",
                                           "C": "relis « RAG ou fine-tuning ? » : que demande une documentation qui change chaque semaine ?"})),
        ("b", LETTER, '"B"', letters("B", {"A": "relis « RAG ou fine-tuning ? » : le RAG change-t-il un comportement ?",
                                           "C": "relis « RAG ou fine-tuning ? » : à quoi sert le fine-tuning ?"})),
        ("c", TF, "True", r'''mistakes={"relis les arguments du RAG dans « RAG ou fine-tuning ? »": False}'''),
        ("d", TF, "False", r'''mistakes={"relis « RAG ou fine-tuning ? » : un fine-tuning sur quelques exemples apprend-il bien des faits ?": True}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "relis l'étape 1 des quatre étapes du RAG : qu'est-ce qui se calcule une fois ?",
                                           "C": "l'index contient-il encore l'ancienne procédure ?"})),
    ]),
    Paper("B4.Q11", "LoRA : ce qui est gelé, ce qui est appris, ce que l'on fusionne", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis la première puce de « Ce que dit chaque morceau » (§204.8)",
                                           "C": "relis la formule de LoRA : W disparaît-elle ?"})),
        ("b", TF, "True", r'''mistakes={"relis la deuxième puce de « Ce que dit chaque morceau » : que vaut BA au départ ?": False}'''),
        ("c", WHOLE, "Q11C", r'''mistakes={"c'est le nombre de poids de la matrice : on demande ceux de LoRA": 512 * 512,
          "relis la formule r(d + k) : il y a deux matrices, A et B": 4 * 512,
          "relis la question : r = 4": 8 * 1024}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "relis « Fusionner l'adaptateur » : que deviennent les branches ?",
                                           "B": "relis « Fusionner l'adaptateur » : que devient W ?"})),
        ("e", TF, "False", r'''mistakes={"calcule le facteur α/r avec les valeurs de la question": True}'''),
    ]),
    Paper("B4.Q12", "Injection de prompt : quand un document récupéré devient une instruction", [
        ("a", LETTER, '"B"', letters("B", {"A": "c'est l'injection directe : relis « L'injection de prompt » (§204.10)",
                                           "C": "c'est une hallucination : relis la définition de l'injection"})),
        ("b", TF, "False", r'''mistakes={"relis « Les parades réduisent le risque sans l'annuler » (§204.10)": True}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "une température basse empêche-t-elle le modèle de suivre une instruction ?",
                                           "B": "relis les parades de l'injection : la taille du modèle en fait-elle partie ?"})),
        ("d", TF, "True", r'''mistakes={"relis la définition de l'injection indirecte : d'où viennent les instructions ?": False}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "relis l'encadré 🕰️ sur l'OWASP : à quel rang est l'agentivité excessive ?",
                                           "C": "relis l'encadré 🕰️ sur l'OWASP : à quel rang est la désinformation ?"})),
    ]),
    Paper("B4.R1", "Ce que le masque causal et le KV cache imposent à la génération", [
        ("a", WHOLE, "100", r'''mistakes={"avec un KV cache, le prompt repasse-t-il dans le modèle à chaque token ?": 500,
          "relis la question : la passe sur le prompt compte, et elle donne déjà le premier token": 99,
          "la passe sur le prompt produit-elle déjà un token ?": 101,
          "relis la question : on compte des passes avant, une par token produit": 400}'''),
        ("b", TF, "False", r'''mistakes={"relis l'encadré 🔗 « Lien avec B3 » (§204.3) : peut-on écrire le token 2 avant de connaître le token 1 ?": True}'''),
        ("c", WHOLE, "R1C", r'''mistakes={"n'oublie pas le facteur 2 des clés et des valeurs": R1C // 2,
          "combien d'octets par valeur en bfloat16 ?": R1C * 2,
          "relis la formule du KV cache (§204.2) : ce sont les têtes de clés et valeurs qui comptent (GQA)": 2 * 24 * 896 * 2}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "le nombre de têtes change-t-il le nombre de passes avant ?",
                                           "B": "relis l'encadré 🔗 « Lien avec B3 » de §204.3 : de quoi le temps d'une réponse est-il proportionnel ?"})),
        ("e", TF, "True", r'''mistakes={"relis le masque causal (B3) : le token 50 voit-il le token 49 ?": False}'''),
    ]),
    Paper("B4.R2", "Extraction de features ou fine-tuning : que change le nombre d'exemples ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis l'encadré 🔗 « Lien avec B1 » (§204.8) : avec peu d'exemples, qui gagnait ?",
                                           "C": "relis l'encadré 🔗 « Lien avec B1 » (§204.8) : les deux méthodes donnaient-elles la même chose ?"})),
        ("b", TF, "True", r'''mistakes={"relis B1 : que gèle l'extraction de features ?": False}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "relis l'encadré 🔗 « Lien avec B1 » (§204.8) : où LoRA place-t-il ses paramètres ?",
                                           "C": "relis la première puce de « Ce que dit chaque morceau » : W est-elle entraînée ?"})),
        ("d", TF, "False", r'''mistakes={"relis l'encadré 🔗 « Lien avec B1 » (§204.8) : que risque un fine-tuning complet avec peu d'exemples ?": True}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "c'est un problème de contexte, pas d'entraînement : relis B1",
                                           "C": "des poids sauvegardés se perdent-ils ? Relis la définition de B1"})),
    ]),
    Paper("B4.R3", "Les k plus proches voisins et le choix de la distance", [
        ("a", DEC2, "COS_R3", r'''decimals=2, mistakes={"c'est le produit scalaire : divise-le par les normes": 50.0,
          "c'est la distance euclidienne : on demande le cosinus": 5.0}'''),
        ("b", DEC2, "DIST_R3", r'''decimals=2, mistakes={"c'est le carré de la distance : prends sa racine": 25.0,
          "c'est le cosinus : on demande la distance euclidienne": 1.0,
          "la distance euclidienne est la racine de la somme des carrés des écarts": 7.0}'''),
        ("c", DEC2, "SQ_R3", r'''decimals=2, mistakes={"relis la formule de la fiche (§204.6) : il y a un facteur 2": 1 - 0.82,
          "c'est la distance : on demande son carré": math.sqrt(SQ_R3),
          "vérifie le signe dans la formule de la fiche": 2 + 2 * 0.82}'''),
        ("d", TF, "True", r'''mistakes={"relis la formule qui relie distance et cosinus pour des vecteurs de norme 1 (§204.6)": False}'''),
        ("e", WHOLE, "MULT_R3", r'''mistakes={"un produit scalaire de dimension 768 demande 768 multiplications : relis la question": 50_000,
          "relis la question : la dimension est 768": 50_000 * 384,
          "relis la question : 50 000 chunks": 10_000 * 768}'''),
    ]),
    Paper("B4.1", "Mémoire d'un modèle : paramètres, dtype et KV cache — tient-il sur ton ordinateur ?", [
        ("a", WHOLE, "EMB_B41", r'''mistakes={"les poids liés ne comptent qu'une fois : relis l'énoncé": 2 * EMB_B41,
          "la table a une ligne par token du vocabulaire, de taille d": V_B41 * 64,
          "relis l'énoncé : V = 49 152 tokens": 50_257 * D_B41}'''),
        ("b", DEC1, "EMB_SHARE_B41", r'''decimals=1, mistakes={"on demande un pourcentage, pas une proportion": EMB_SHARE_B41 / 100,
          "les poids liés ne comptent qu'une fois : relis l'énoncé": 2 * EMB_SHARE_B41}'''),
        ("c", DEC1, "FP32_B41", r'''decimals=1, mistakes={"combien d'octets par paramètre en float32 ?": BF16_B41,
          "1 Mo = 10⁶ octets : vérifie la conversion": 4 * N_B41 / 2 ** 20,
          "c'est le nombre de bits : on demande des octets": 32 * N_B41 / MB}'''),
        ("d", DEC1, "BF16_B41", r'''decimals=1, mistakes={"combien d'octets par paramètre en bfloat16 ?": FP32_B41,
          "1 Mo = 10⁶ octets : vérifie la conversion": 2 * N_B41 / 2 ** 20,
          "relis la question : bfloat16, pas int4": INT4_B41}'''),
        ("e", DEC1, "INT4_B41", r'''decimals=1, mistakes={"combien d'octets par paramètre en int4 ? Relis l'énoncé": 4 * N_B41 / MB,
          "relis l'énoncé : 0,5 octet par paramètre": N_B41 / MB,
          "1 Mo = 10⁶ octets : vérifie la conversion": 0.5 * N_B41 / 2 ** 20}'''),
        ("f", WHOLE, "KV_TOKEN_B41", r'''mistakes={"n'oublie pas le facteur 2 des clés et des valeurs": KV_TOKEN_B41 // 2,
          "combien d'octets par valeur en bfloat16 ?": KV_TOKEN_B41 // 2 * 4,
          "relis la formule du KV cache : ce sont les têtes de clés et valeurs (GQA) qui comptent": 2 * 30 * 9 * 64 * 2,
          "relis la formule du KV cache : une couche, ou toutes ?": KV_TOKEN_B41 // 30}'''),
        ("g", DEC1, "KV_FULL_B41", r'''decimals=1, mistakes={"relis la formule du KV cache : ce sont les têtes de clés et valeurs (GQA) qui comptent": 3 * KV_FULL_B41,
          "n'oublie pas le facteur 2 des clés et des valeurs": KV_FULL_B41 / 2,
          "1 Mo = 10⁶ octets : vérifie la conversion": KV_TOKEN_B41 * CTX_B41 / 2 ** 20,
          "relis l'énoncé : le contexte compte 8 192 tokens": KV_TOKEN_B41 * 8_000 / MB}'''),
        ("h", TF, "False", r'''mistakes={"compare tes réponses à g) et d)": True}'''),
        ("i", WHOLE, "CONVERSATIONS_B41", r'''mistakes={"n'oublie pas de retirer les poids des 2 Go": int(2 * GB // (KV_TOKEN_B41 * CTX_B41)),
          "relis la question : les poids sont en bfloat16": int((2 * GB - 4 * N_B41) // (KV_TOKEN_B41 * CTX_B41))},
          fractional="on demande un nombre de conversations entières"'''),
        ("j", WHOLE, "round(SPEED_B41)", r'''mistakes={"c'est la vitesse de Qwen2.5 : applique la proportion des mémoires": round(QWEN_TOKENS / QWEN_SECONDS),
          "un modèle plus petit va-t-il plus vite ou plus lentement ? Vérifie le sens du rapport": round(QWEN_TOKENS / QWEN_SECONDS * 4 * N_B41 / QWEN_FOOTPRINT),
          "relis la question : compare les mémoires en float32 des deux modèles": round(QWEN_TOKENS / QWEN_SECONDS * QWEN_FOOTPRINT / (2 * N_B41)),
          "relis l'énoncé : quelle mesure de Qwen2.5 prend-il ? Et arrondis à l'entier le plus proche, pas vers le bas": round(SPEED_B41) - 1},
          fractional="arrondis à l'entier le plus proche"'''),
        ("k", TF, "True", r'''mistakes={"compare ta réponse à c) à 1 Go = 1 000 Mo": False}'''),
    ]),
    Paper("B4.2", "Recall@k et MRR pour cinq requêtes", [
        ("a", DEC3, "recall_b42(1)", r'''decimals=3, mistakes={"relis la définition du recall@k : par quoi divise-t-on, pour chaque question ?": 0.2}'''),
        ("b", DEC3, "recall_b42(3)", r'''decimals=3, mistakes={"le recall divise par le nombre de pertinents, pas par k": recall_b42(3, denominator="k"),
          "relis la question : on demande le recall, pas la part des questions réussies": HIT3_B42,
          "relis la question : les trois premiers résultats": recall_b42(2)}'''),
        ("c", DEC3, "recall_b42(5)", r'''decimals=3, mistakes={"le recall divise par le nombre de pertinents, pas par k": recall_b42(5, denominator="k"),
          "relis la question : on demande le recall, pas la part des questions réussies": 0.8,
          "relis la question : les cinq premiers résultats": recall_b42(4)}'''),
        ("d", DEC3, "MRR_B42", r'''decimals=3, mistakes={"on prend le rang du premier pertinent, pas la somme sur tous les pertinents": (1 / 2 + 1 / 3 + 1 / 5 + 1 + 0 + 1 / 3 + 1 / 4) / 5,
          "une question sans pertinent retrouvé compte 0, et la moyenne porte sur les cinq questions": MRR_B42 * 5 / 4,
          "le MRR moyenne l'inverse du rang, pas le rang": (2 + 3 + 1 + 0 + 3) / 5}'''),
        ("e", DEC3, "HIT3_B42", r'''decimals=3, mistakes={"relis la question : on compte les questions qui ont au moins un pertinent, pas les pertinents": recall_b42(3),
          "relis la question : les trois premiers résultats": 0.4}'''),
        ("f", TF, "False", r'''mistakes={"compare tes réponses à b) et e)": True}'''),
        ("g", WHOLE, "FULL5_B42", r'''mistakes={"relis la question : tous les pertinents de la question, pas au moins un": 4,
          "recompte question par question : une question à deux pertinents peut aussi avoir un recall@5 de 1": 1,
          "recompte question par question : a-t-on retrouvé tous les pertinents de chacune ?": 3}'''),
        ("h", TF, "True", r'''mistakes={"pour une question, compare la part de ses pertinents au rang 1 et l'inverse du rang de son premier pertinent": False}'''),
    ]),
    Paper("B4.3", "LoRA : paramètres entraînables d'une couche 896×896 de rang 8, puis de tout un modèle", [
        ("a", WHOLE, "Q_B43", r'''mistakes={"relis la formule r(d + k) : il y a deux matrices, A et B": R_B43 * D_QWEN,
          "c'est le nombre de poids de la matrice : on demande ceux de LoRA": D_QWEN ** 2,
          "relis la formule r(d + k) : r s'ajoute-t-il ou se multiplie-t-il ?": 2 * D_QWEN + R_B43}'''),
        ("b", DEC2, "Q_SHARE_B43", r'''decimals=2, mistakes={"on demande un pourcentage, pas une proportion": Q_SHARE_B43 / 100,
          "relis la question : la matrice de q_proj, sans son biais": 100 * Q_B43 / (D_QWEN ** 2 + D_QWEN)}'''),
        ("c", DEC2, "K_SHARE_B43", r'''decimals=2, mistakes={"relis l'énoncé : k_proj va de 896 à 128": Q_SHARE_B43,
          "on demande un pourcentage, pas une proportion": K_SHARE_B43 / 100}'''),
        ("d", WHOLE, "LAYER_B43", r'''mistakes={"relis l'énoncé : k_proj et v_proj vont de 896 à 128": 4 * Q_B43,
          "relis la question : quatre matrices, q, k, v et o": 2 * Q_B43 + K_B43,
          "relis la question : une seule couche": MODEL_B43}'''),
        ("e", WHOLE, "MODEL_B43", r'''mistakes={"relis l'énoncé : k_proj et v_proj vont de 896 à 128": LAYERS_QWEN * 4 * Q_B43,
          "relis l'énoncé : 24 couches": 28 * LAYER_B43,
          "relis la question : q, k, v et o, pas seulement q et v": LAYERS_QWEN * (Q_B43 + K_B43)}'''),
        ("f", DEC4, "TRAINABLE_PCT_B43", r'''decimals=4, mistakes={"relis la question : « all params » compte aussi les adaptateurs": 100 * MODEL_B43 / N_QWEN,
          "on demande un pourcentage, pas une proportion": TRAINABLE_PCT_B43 / 100}'''),
        ("g", WHOLE, "ALL_LINEAR_B43", r'''mistakes={"relis l'énoncé : les trois couches du MLP comptent aussi": MODEL_B43,
          "relis l'énoncé : le MLP va de 896 à 4 864, puis revient à 896": LAYERS_QWEN * (LAYER_B43 + 3 * Q_B43),
          "relis la question : sans la tête de sortie": ALL_LINEAR_B43 + lora_params(151_936, D_QWEN)}'''),
        ("h", TF, "True", r'''mistakes={"relis la formule r(d + k) : comment dépend-elle de r ?": False}'''),
        ("i", DEC1, "ADAPTER_MB_B43", r'''decimals=1, mistakes={"combien d'octets par paramètre en float32 ?": 2 * MODEL_B43 / MB,
          "1 Mo = 10⁶ octets : vérifie la conversion": 4 * MODEL_B43 / 2 ** 20,
          "c'est la mémoire du modèle entier : on demande celle de l'adaptateur": 4 * (N_QWEN + MODEL_B43) / MB}'''),
        ("j", WHOLE, "NO_SAVING_RANK_B43", r'''mistakes={"relis la question : « au moins autant » ; vérifie le cas d'égalité": NO_SAVING_RANK_B43 - 1,
          "relis la formule r(d + k) : il y a deux matrices, A et B": D_QWEN,
          "écris l'inégalité r(d + k) ≥ dk et résous-la en r": NO_SAVING_RANK_B43 + 1},
          fractional="on demande un rang, un nombre entier"'''),
        ("k", TF, "False", r'''mistakes={"compare tes réponses à b) et c)": True}'''),
    ]),
    Paper("B4.4", "Un chat template à la main : rôles, tokens spéciaux et prompt de génération", [
        ("a", WHOLE, "U_GEN_B44[1]", r'''mistakes={"relis la condition `if loop.first and …` du gabarit : que se passe-t-il sans message système ?": U_GEN_B44[1] - 2,
          "relis la question : add_generation_prompt=True": U_GEN_B44[1] - 1,
          "les retours à la ligne ne sont pas des tokens spéciaux : relis le tableau": U_GEN_B44[1] + 5}'''),
        ("b", WHOLE, "U_GEN_B44[2]", r'''mistakes={"relis la condition `if loop.first and …` du gabarit : que se passe-t-il sans message système ?": U_GEN_B44[2] - 21,
          "relis le tableau : `assistant` compte 2 tokens": U_GEN_B44[2] - 1,
          "n'oublie pas les retours à la ligne, un token chacun": U_GEN_B44[2] - 5,
          "relis la question : add_generation_prompt=True": U_GEN_B44[2] - 4}'''),
        ("c", WHOLE, "SU_GEN_B44[2]", r'''mistakes={"relis la condition `if loop.first and …` : le message système par défaut est-il ajouté quand S existe ?": DOUBLE_SYSTEM_B44[2],
          "relis le tableau : `assistant` compte 2 tokens": SU_GEN_B44[2] - 1,
          "n'oublie pas les retours à la ligne, un token chacun": SU_GEN_B44[2] - 5,
          "relis la question : add_generation_prompt=True": SU_NOGEN_B44[2]}'''),
        ("d", WHOLE, "SU_NOGEN_B44[2]", r'''mistakes={"relis la question : add_generation_prompt=False": SU_GEN_B44[2],
          "n'oublie pas les retours à la ligne, un token chacun": SU_NOGEN_B44[2] - 4,
          "relis la condition `if loop.first and …` : le message système par défaut est-il ajouté quand S existe ?": DOUBLE_SYSTEM_B44[2] - 4}'''),
        ("e", WHOLE, "TURN2_B44[2]", r'''mistakes={"relis la question : le prompt du deuxième tour contient toute la conversation": SU_GEN_B44[2],
          "relis le tableau : `assistant` compte 2 tokens, aussi dans le tour de A": TURN2_B44[2] - 1,
          "le tour de A est-il fermé par `<|im_end|>\n` comme les autres ?": TURN2_B44[2] - 2,
          "relis la condition `if loop.first and …` : le message système par défaut s'ajoute-t-il ici ?": TURN2_B44[2] + 21}'''),
        ("f", LETTER, '"B"', letters("B", {"A": "relis la condition `if loop.first and …` : le message système par défaut est-il ajouté quand S existe ?",
                                           "C": "relis la question : add_generation_prompt=True"})),
        ("g", TF, "False", r'''mistakes={"relis le tableau : quels morceaux sont des tokens spéciaux ? En vois-tu un dans cette chaîne ?": True}'''),
        ("h", TF, "True", r'''mistakes={"relis le gabarit : la boucle `for` parcourt-elle tous les messages ?": False}'''),
    ]),
    Paper("B4.5", "BM25 à la main : trois documents, une requête de deux mots", [
        ("a", DEC4, "AVGDL_B45", r'''decimals=4, mistakes={"compte à nouveau les mots de chaque document": 16 / 3,
          "relis la question : la longueur moyenne, pas totale": 15.0}'''),
        ("b", DEC4, "IDF_NAUTILUS_B45", r'''decimals=4, mistakes={"relis la formule : l'idf de la docstring a un « 1 + »": math.log(0.5 / 3.5),
          "n_t compte les documents qui contiennent le terme, pas ses occurrences": math.log(1 + (3 - 4 + 0.5) / (4 + 0.5)),
          "relis la formule : n_t + 0,5 au dénominateur": math.log(1 + 0.5 / 3),
          "le logarithme est naturel ici, pas décimal": math.log10(1 + 0.5 / 3.5)}'''),
        ("c", DEC4, "SCORES_B45[1]", r'''decimals=4, mistakes={"relis la formule : le dénominateur dépend de la longueur relative du document": float(SCORES_B0_B45[1]),
          "relis la formule : le facteur k₁ + 1 au numérateur": float(SCORES_B45[1] / 2.5),
          "n_t compte les documents qui contiennent le terme, pas ses occurrences": float(SCORES_B45[1] / IDF_NAUTILUS_B45 * math.log(1 + (3 - 4 + 0.5) / 4.5))}'''),
        ("d", DEC4, "SCORES_B45[2]", r'''decimals=4, mistakes={"le doc 2 contient « nautilus » deux fois : f = 2": float(bm25_b45(docs=[DOCS_B45[0], DOCS_B45[1], ["le", "nautilus", "croise", "un", "x", "au", "grand", "large"]])[2]),
          "relis la formule : le dénominateur dépend de la longueur relative du document": float(SCORES_B0_B45[2]),
          "relis la formule : le facteur k₁ + 1 au numérateur": float(SCORES_B45[2] / 2.5)}'''),
        ("e", DEC4, "SCORES_B45[0]", r'''decimals=4, mistakes={"n'oublie pas la contribution de « nautilus »": float(bm25_b45(query=["nemo"])[0]),
          "n'oublie pas la contribution de « nemo »": float(bm25_b45(query=["nautilus"])[0]),
          "relis la formule : le dénominateur dépend de la longueur relative du document": float(SCORES_B0_B45[0]),
          "relis la formule : l'idf de la docstring a un « 1 + »": float(SCORES_ORIGINAL_B45[0])}'''),
        ("f", IDS, "RANKING_B45", r'''mistakes={"un document court peut-il battre un document qui contient le terme plus souvent ? Compare c) et d)": [0, 2, 1],
          "on classe du meilleur au moins bon": RANKING_B45[::-1]}'''),
        ("g", TF, "False", r'''mistakes={"recalcule les scores des docs 1 et 2 avec b = 0": True}'''),
        ("h", DEC4, "float(SCORES_K0_B45[0])", r'''decimals=4, mistakes={"relis la question : k₁ = 0, la fréquence ne compte plus": float(SCORES_B45[0]),
          "relis la question : on additionne les contributions des deux termes": float(SCORES_K0_B45[0] - IDF_NAUTILUS_B45)}'''),
        ("i", TF, "True", r'''mistakes={"calcule l'idf original de « nautilus », puis le score du doc 0": False}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ B4.1 à B4.5)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; un "
               "classement : une liste d'identifiants, du meilleur au moins bon (`[5, 6, 4]`). Arrondis comme "
               "l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. "
               "Les questions « dans ta copie », la dérivation ∂ B4.6, l'estimation 🧮 B4.7, la réflexion (🗣️ B4.8, "
               "⚖️ B4.9 et B4.10, 📄 B4.11) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A and B (exercises B4.12 to B4.21); B4.22 to B4.30 at the next generation session (NEXT_SESSION)
# ---------------------------------------------------------------------------
import chB4_code  # noqa: E402,F401  (the cells; history_chB4.py reuses them)
from chB4_parts import PARTS  # noqa: E402

NEXT_SESSION = [
    ("B", "B4.22 à B4.24", "suite de la recherche et du RAG : quatre défauts, évaluation et analyse des échecs, "
                           "injection de prompt"),
    ("C", "B4.25 à B4.28", "fine-tuning : LoRA à la main contre peft, style de Holmes, rang et modules ciblés, "
                           "classification de textes"),
    ("D", "B4.29 et B4.30", "livraison : pipeline reproductible et défi RAG"),
]


def part_cells_b4(part, kind: str) -> list:
    """part_cells, with the cache of the solutions notebook: its own file, read from the precomputed replies."""
    if kind == "solution" and chB4_code.LEARNER_CACHE in part.given:
        part = copy.copy(part)
        part.given = part.given.replace(chB4_code.LEARNER_CACHE, chB4_code.SOLUTION_CACHE_LINE)
    return part_cells(part, kind)


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | B4.Q1 à B4.Q12, B4.R1 à B4.R3, B4.1 à B4.5 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(ex.table_row(part.key if i == 0 else ""))
    if NEXT_SESSION:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# B4 · LLM en pratique : Hugging Face, prompting, RAG et LoRA — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/b4_llm/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py B4`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# B4 · LLM en pratique : Hugging Face, prompting, RAG et LoRA — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Estimer la mémoire d'un LLM (poids selon le dtype, KV cache) et appliquer à la main son chat "
               "template.\n"
               "- Charger et faire générer de petits modèles avec transformers, écrire et évaluer des prompts sur "
               "plusieurs formulations et graines.\n"
               "- Implémenter le découpage, BM25, la recherche dense, la fusion RRF, recall@k et MRR, puis "
               "construire, évaluer et diagnostiquer un RAG avec citations.\n"
               "- Compter, écrire à la main et appliquer avec peft une adaptation LoRA, et comparer fine-tuning, "
               "prompting et classifieur classique.\n\n"
               "**Rappel express.** Mémoire des poids $\\approx N \\times$ octets par paramètre (4 en float32, 2 en "
               "bfloat16, $\\approx 0{,}5$ en int4) ; KV cache $= 2 \\times L \\times T \\times n_{kv} d_h \\times$ "
               "octets par valeur. BM25 : $\\sum_{t \\in q} \\operatorname{idf}(t)\\,\\frac{f_{t,d}(k_1 + 1)}"
               "{f_{t,d} + k_1(1 - b + b\\,|d| / \\overline{|d|})}$, $\\operatorname{idf}(t) = \\ln\\big(1 + "
               "\\frac{N - n_t + 0{,}5}{n_t + 0{,}5}\\big)$. RRF : $\\sum_r 1/(k + \\operatorname{rang}_r(d))$, "
               "$k = 60$. recall@$k$ $= |\\text{top-}k \\cap \\text{pertinents}| / |\\text{pertinents}|$ ; MRR = "
               "moyenne de $1/\\operatorname{rang}$ du premier pertinent. LoRA : $\\mathbf{W}' = \\mathbf{W} + "
               "\\frac{\\alpha}{r}\\mathbf{B}\\mathbf{A}$, $\\mathbf{B}_0 = 0$, $r(d + k)$ paramètres au lieu de "
               "$dk$.")]


def footer_cells(kind: str) -> list:
    later = ("" if not NEXT_SESSION else
             "\n\n*Les exercices B4.22 à B4.30 (la suite du RAG, le fine-tuning, la livraison) seront ajoutés à la "
             "prochaine session de génération ; relance alors `python tools/start_chapter.py B4` pour obtenir le "
             "notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dire si un modèle tient en mémoire, poids et KV cache compris, selon son dtype ?\n"
               "2. Sais-tu appliquer à la main un chat template et compter les tokens d'un prompt ?\n"
               "3. Sais-tu calculer BM25, recall@k et MRR à la main ?\n"
               "4. Sais-tu compter les paramètres de LoRA et dériver les gradients de $\\mathbf{A}$ et $\\mathbf{B}$ ?\n\n"
               "**Pour aller plus loin** : le cours LLM de Hugging Face et l'article de LoRA, cités dans la fiche. "
               "La suite : le bonus B5, les modèles génératifs d'images."
               + later)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT + LIBRARY_CHECKS, PAPER_INTRO)
    for part in PARTS:
        cells += part_cells_b4(part, kind)
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
