"""Code cells of the chapter B2 notebook (tokenization phase, parts A to C), shared by build_chB2.py and history_chB2.py
(used by Claude).

Each constant is the source of one cell: TOOLS (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
"""

TOOLS = r'''# Tools for the notebook exercises of the tokenization phase (parts A to C)
import json
import math
import os
import re
import time
import unicodedata
import warnings
from collections import Counter
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wb import text as wbtext
from wb.nbtools import StopCheck, checked, filled, fr, refuses, returned, run_pytest_group, same, verdict, written

warnings.filterwarnings("ignore", message=".*IProgress not found.*")    # tqdm without ipywidgets (transformers)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()     # no « Token indices sequence length is longer… »: we count, no model reads
except ImportError:                       # transformers absent: the cells that need it say so
    pass

TEST_TOK = "tests/test_chb2_tokenizer.py"     # the tests of your library tokenizer.py
EXCERPT_H = wbtext.excerpt("holmes")           # 50 000 characters of Holmes, [100000:150000] (frozen)
EXCERPT_V = wbtext.excerpt("verne")            # 50 000 characters of Verne, [100000:150000]
HOLMES_TRAIN, HOLMES_VAL = wbtext.train_val("holmes")   # 90 % / 10 %, cut at a line, never shuffled
VERNE_TRAIN, VERNE_VAL = wbtext.train_val("verne")
TOKENIZER_REPOS = {name: (spec["repo"], spec["revision"]) for name, spec in wbtext.TOKENIZERS.items()}


def n_bytes(text):
    """The number of UTF-8 bytes of a text."""
    return len(text.encode("utf-8"))


def need(ex_id, *names):
    """Stop a check cell with ⏳ while a function of an earlier exercise is not defined yet."""
    for name in names:
        if globals().get(name) is None:
            print(f"⏳ Ex {ex_id} : définis d'abord {name}.")
            raise StopCheck


def bpe_written():
    """True once BPETokenizer.fit and encode of your library run (B2.16 and B2.17), else False."""
    return written(lambda: mylearn.tokenizer.BPETokenizer(vocab_size=258, level="byte").fit("aa aa").encode("aa"))


REBUILD_TEXT = "le chat et le rat, les chats et les rats ; l'été, été, étés."


def rebuilt(tok):
    """A new BPETokenizer with the hyperparameters and the four learned attributes of tok, set by hand (as
    fast_byte_bpe, truncated_bpe and load do), without fit."""
    copy = type(tok)(vocab_size=tok.vocab_size, level=tok.level, pre_tokenizer=tok.pre_tokenizer,
                     min_frequency=tok.min_frequency, special_tokens=tok.special_tokens, unk_token=tok.unk_token)
    copy.n_base_, copy.merges_ = tok.n_base_, list(tok.merges_)
    copy.vocab_, copy.special_ids_ = dict(tok.vocab_), dict(tok.special_ids_)
    return copy


def bpe_rebuild_ok(ex_id):
    """True when the encode of your library gives the same ids for a fitted tokenizer and for the same tokenizer
    rebuilt by hand without fit; otherwise a ❌ line that points to the contract of B2.16 and B2.17, and False."""
    try:
        tok = mylearn.tokenizer.BPETokenizer(vocab_size=290, pre_tokenizer="gpt2", min_frequency=1).fit(REBUILD_TEXT)
        same_ids = rebuilt(tok).encode(REBUILD_TEXT) == tok.encode(REBUILD_TEXT)
    except Exception as exc:  # noqa: BLE001 - any error of the learner's code: the same diagnosis
        same_ids = False
        print(f"   (ton code lève {type(exc).__name__} : {str(exc)[:160]})")
    if not same_ids:
        print(f"❌ Ex {ex_id} : un tokenizer reconstruit sans fit (ses quatre attributs n_base_, merges_, vocab_ et "
              "special_ids_ copiés à la main, comme le font fast_byte_bpe, truncated_bpe et load) n'encode pas comme "
              "l'original : ton encode lit un autre état, posé par fit. Relis le contrat de B2.16 et B2.17 : encode et "
              "decode ne lisent que les hyperparamètres et ces quatre attributs. Les mesures de cet exercice attendent "
              "cette correction.")
    return same_ids


def reference_char_bpe(text, vocab_size, min_frequency=2):
    """The BPE of the library tokenizers trained like BPETokenizer(level="char", pre_tokenizer="whitespace") (fiche
    §202.3): models.BPE, pre_tokenizers.Whitespace, BpeTrainer. For the checks."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    hf = Tokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.Whitespace()
    hf.train_from_iterator([text], trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                                       show_progress=False))
    return hf


def fast_byte_bpe(texts, vocab_size):
    """Your BPETokenizer(vocab_size, pre_tokenizer="gpt2"), whose merges are learned by the library tokenizers (Rust):
    the same merges as your fit (wbtext.byte_bpe_merges, B2.19), 50 to 200 times faster than Python. Your encode,
    decode and tokenize do the rest."""
    tok = mylearn.tokenizer.BPETokenizer(vocab_size=vocab_size, pre_tokenizer="gpt2")
    tok.n_base_, tok.merges_, tok.special_ids_ = 256, wbtext.byte_bpe_merges(texts, vocab_size), {}
    tok.vocab_ = {i: bytes([i]) for i in range(256)}
    for r, (a, b) in enumerate(tok.merges_):
        tok.vocab_[256 + r] = tok.vocab_[a] + tok.vocab_[b]
    return tok


def truncated_bpe(tok, vocab_size):
    """A copy of a fitted BPETokenizer without special token that keeps only its first vocab_size - n_base_ merges:
    the tokenizer that fit would have learned with this vocab_size (merges are learned one after the other, the same
    first ones whatever the final size, as long as min_frequency does not stop the training)."""
    if tok.special_ids_:
        raise ValueError("truncated_bpe: only for a tokenizer without special token")
    k = vocab_size - tok.n_base_
    if not 0 <= k <= len(tok.merges_):
        raise ValueError(f"truncated_bpe: vocab_size must be between {tok.n_base_} and "
                         f"{tok.n_base_ + len(tok.merges_)}, got {vocab_size}")
    small = type(tok)(vocab_size=vocab_size, level=tok.level, pre_tokenizer=tok.pre_tokenizer,
                      min_frequency=tok.min_frequency)
    small.n_base_, small.merges_, small.special_ids_ = tok.n_base_, list(tok.merges_[:k]), {}
    small.vocab_ = {i: tok.vocab_[i] for i in range(tok.n_base_ + k)}
    return small


print(f"Holmes : {len(HOLMES_TRAIN):_} caractères d'entraînement, {len(HOLMES_VAL):_} de validation ; Verne : "
      f"{len(VERNE_TRAIN):_} et {len(VERNE_VAL):_} ; extraits de {len(EXCERPT_H):_} caractères.".replace("_", " "))'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="B2")   # reload your saved file\n'

# --- Part A -----------------------------------------------------------------------------------------------------------
TODO_13 = r'''from transformers import AutoTokenizer


def load_tokenizers_13():
    """{"gpt2": ..., "bert": ..., "e5": ...}: the AutoTokenizer of each repository of TOKENIZER_REPOS, at its revision."""
    raise NotImplementedError("load_tokenizers_13")


def measure_13(tok, text):
    """{"tokens": int, "words": int, "fertility": float, "bytes_per_token": float} for one tokenizer and one text."""
    raise NotImplementedError("measure_13")'''

SOLUTION_13 = r'''from transformers import AutoTokenizer


def load_tokenizers_13():
    """{"gpt2": ..., "bert": ..., "e5": ...}: the AutoTokenizer of each repository of TOKENIZER_REPOS, at its revision."""
    out = {}
    for name in ("gpt2", "bert", "e5"):
        repo, revision = TOKENIZER_REPOS[name]
        out[name] = AutoTokenizer.from_pretrained(repo, revision=revision)
    return out


def measure_13(tok, text):
    """{"tokens": int, "words": int, "fertility": float, "bytes_per_token": float} for one tokenizer and one text."""
    n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])     # the text alone: no [CLS], [SEP], <s>, </s>
    n_words = len(text.split())
    return {"tokens": n_tokens, "words": n_words, "fertility": n_tokens / n_words,
            "bytes_per_token": len(text.encode("utf-8")) / n_tokens}'''

CHECK_13 = r'''REF_13 = {name: wbtext.try_tokenizer(name, "B2.13") for name in ("gpt2", "bert", "e5")}
KEYS_13 = ("tokens", "words", "fertility", "bytes_per_token")
rows_13 = {}
if any(tok is None for tok in REF_13.values()):
    print("⏳ Ex B2.13 : sans les tokenizers de référence, la vérification attend une connexion au Hub (ou Colab).")
else:
    with checked("B2.13"):
        toks_13 = load_tokenizers_13()
        if not isinstance(toks_13, dict) or set(toks_13) != set(REF_13):
            got = sorted(toks_13) if isinstance(toks_13, dict) else type(toks_13).__name__
            print(f"❌ Ex B2.13 : load_tokenizers_13 doit rendre un dictionnaire de clés 'gpt2', 'bert' et 'e5' ; reçu {got}.")
            raise StopCheck
        probe_13 = "Phileas Fogg paria 20 000 livres, déjà ! 🙂"
        for name, ref in REF_13.items():
            tok = toks_13[name]
            ok = (callable(tok) and type(tok).__name__ == type(ref).__name__ and len(tok) == len(ref)
                  and tok(probe_13)["input_ids"] == ref(probe_13)["input_ids"])
            verdict("B2.13", ok, f"{name} : {type(ref).__name__}, {len(ref):_} tokens, mêmes ids que la référence.".replace("_", " "),
                    f"{name} : ce n'est pas le tokenizer de {TOKENIZER_REPOS[name][0]} (type, taille du vocabulaire ou ids "
                    "d'une phrase d'essai différents de la référence).")
        for name in REF_13:
            for label, text in (("Holmes", EXCERPT_H), ("Verne", EXCERPT_V)):
                m = measure_13(toks_13[name], text)
                if not returned("B2.13", "measure_13", m):
                    raise StopCheck
                if not isinstance(m, dict) or any(key not in m for key in KEYS_13):
                    print(f"❌ Ex B2.13 : measure_13 doit rendre un dictionnaire de clés {', '.join(KEYS_13)} ; reçu "
                          f"{sorted(m) if isinstance(m, dict) else type(m).__name__}.")
                    raise StopCheck
                rows_13[(name, label)] = m
        table_13 = pd.DataFrame({f"{name} · {label}": m for (name, label), m in rows_13.items()}).T
        print(table_13.to_string(float_format=lambda x: f"{x:.3f}"))
        m = rows_13[("gpt2", "Holmes")]
        verdict("B2.13", m["words"] == len(EXCERPT_H.split()) and math.isclose(m["fertility"], m["tokens"] / m["words"])
                and math.isclose(m["bytes_per_token"], n_bytes(EXCERPT_H) / m["tokens"]),
                "les quatre mesures sont cohérentes (mots de split(), fertilité = tokens / mots, octets UTF-8 / tokens).",
                "relis le contrat : words = len(text.split()), fertility = tokens / words, bytes_per_token = octets UTF-8 "
                "du texte / tokens.")
        wb.check("B2.13a", rows_13[("gpt2", "Holmes")]["tokens"], computed=True)
        wb.check("B2.13b", rows_13[("bert", "Verne")]["tokens"], computed=True)
        wb.check("B2.13c", rows_13[("e5", "Holmes")]["fertility"], computed=True)
        wb.check("B2.13d", rows_13[("e5", "Verne")]["bytes_per_token"], computed=True)'''

RECORD_13 = r'''if not rows_13:
    print("⏳ Ex B2.13 : les réponses s'enregistrent avec les tokenizers de référence (connexion au Hub).")
else:
    specials_13 = {key: len(REF_13[name](text)["input_ids"])          # the classic mistake: [CLS] and [SEP], <s> and </s>
                   for key, (name, text) in {"bert_v": ("bert", EXCERPT_V), "e5_h": ("e5", EXCERPT_H), "e5_v": ("e5", EXCERPT_V)}.items()}
    wb.record("B2.13a", rows_13[("gpt2", "Holmes")]["tokens"], decimals=4, mistakes={
        "c'est le nombre de mots de l'extrait : on demande ses tokens": rows_13[("gpt2", "Holmes")]["words"],
        "c'est le compte de l'extrait de Verne : on demande celui de Holmes": rows_13[("gpt2", "Verne")]["tokens"]})
    wb.record("B2.13b", rows_13[("bert", "Verne")]["tokens"], decimals=4, mistakes={
        "les tokens spéciaux que BERT ajoute autour du texte sont comptés : on demande les tokens du texte seul": specials_13["bert_v"]})
    wb.record("B2.13c", rows_13[("e5", "Holmes")]["fertility"], decimals=4, mistakes={
        "les tokens spéciaux que e5 ajoute autour du texte sont comptés : on demande les tokens du texte seul": specials_13["e5_h"] / rows_13[("e5", "Holmes")]["words"],
        "la fertilité met les tokens au numérateur et les mots au dénominateur": rows_13[("e5", "Holmes")]["words"] / rows_13[("e5", "Holmes")]["tokens"]})
    wb.record("B2.13d", rows_13[("e5", "Verne")]["bytes_per_token"], decimals=4, mistakes={
        "tu as compté des caractères : la compression se mesure en octets UTF-8, et une lettre accentuée en prend deux": len(EXCERPT_V) / rows_13[("e5", "Verne")]["tokens"],
        "les tokens spéciaux que e5 ajoute autour du texte sont comptés : on demande les tokens du texte seul": n_bytes(EXCERPT_V) / specials_13["e5_v"]})'''

GIVEN_14 = r'''# Two more texts of about 6 000 to 12 000 bytes, next to the two excerpts of B2.13
NUMBERS_14 = " ".join(str(i * 7919 % 100000) for i in range(2000))    # 2 000 numbers of 1 to 5 digits, spaced
EMOJI_14 = "🙂🚀🎉👍🏽🐍❤️" * 200                                        # 1 200 emoji in a row, no space
TEXTS_14 = {"Holmes": EXCERPT_H, "Verne": EXCERPT_V, "nombres": NUMBERS_14, "emoji": EMOJI_14}
print({name: f"{len(text):_} points de code, {n_bytes(text):_} octets".replace("_", " ") for name, text in TEXTS_14.items()})'''

TODO_14 = r'''costliest_14 = ...    # a) with GPT-2, the text with the FEWEST bytes per token:
                      #    "A" Holmes, "B" Verne, "C" the numbers, "D" the emoji
ratio_14 = ...        # b) with CamemBERT, tokens of EXCERPT_V / tokens of EXCERPT_H (the same 50 000 characters):
                      #    "A" under 0.9, "B" from 0.9 to 1.1, "C" over 1.1
bert_emoji_14 = ...   # c) the number of tokens of BERT (bert-base-uncased) for EMOJI_14, special tokens left out:
                      #    "A" from 1 to 10, "B" from 11 to 2 000, "C" over 2 000'''

EXPERIMENT_14 = r'''names_14 = ("gpt2", "bert", "e5", "camembert", "qwen3", "smollm2")
toks_14 = {name: wbtext.try_tokenizer(name, "B2.14") for name in names_14}
counts_14 = {name: {label: wbtext.count_tokens(tok, text) for label, text in TEXTS_14.items()}
             for name, tok in toks_14.items() if tok is not None}
if any(name not in counts_14 for name in ("gpt2", "bert", "camembert")):
    print("⏳ Ex B2.14 : l'expérience attend les tokenizers de GPT-2, de BERT et de CamemBERT (connexion au Hub, ou Colab).")
else:
    tokens_14 = pd.DataFrame(counts_14).T
    per_token_14 = pd.DataFrame({name: {label: n_bytes(TEXTS_14[label]) / n for label, n in row.items()}
                                 for name, row in counts_14.items()}).T
    print("Nombre de tokens (tokens spéciaux exclus) :")
    print(tokens_14.to_string())
    print("\nOctets par token :")
    print(per_token_14.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nTokens de Verne / tokens de Holmes :", ", ".join(f"{name} {fr(row['Verne'] / row['Holmes'], 2)}"
                                                         for name, row in counts_14.items()))
    print("BERT sur les emoji :", toks_14["bert"].convert_ids_to_tokens(toks_14["bert"](EMOJI_14, add_special_tokens=False)["input_ids"])[:5])
    for id_14, value_14 in (("B2.14a", costliest_14), ("B2.14b", ratio_14), ("B2.14c", bert_emoji_14)):
        wb.check(id_14, value_14)'''

# --- Part B -----------------------------------------------------------------------------------------------------------
CHECK_15 = r'''with checked("B2.15"):
    T = mylearn.tokenizer
    corpus_15 = {(1, 2, 3): 2, (1, 2): 1, (4, 4, 4): 3, (5,): 7}
    arg_15 = dict(corpus_15)                                     # your function receives a copy
    counts_15 = T.get_pair_counts(MappingProxyType(arg_15))      # a read-only view, as in the tests
    if returned("B2.15", "get_pair_counts", counts_15):
        verdict("B2.15", isinstance(counts_15, dict) and dict(counts_15) == {(1, 2): 3, (2, 3): 2, (4, 4): 6},
                "get_pair_counts : (1, 2) 3 fois, (2, 3) 2 fois, (4, 4) 6 fois ; un mot d'un seul id n'a pas de paire.",
                f"get_pair_counts({corpus_15}) doit rendre {{(1, 2): 3, (2, 3): 2, (4, 4): 6}} (chaque paire voisine, "
                f"pondérée par la fréquence du mot, chevauchements compris) ; reçu {counts_15!r}.")
        verdict("B2.15", arg_15 == corpus_15, "le corpus reçu n'a pas changé.", "get_pair_counts a modifié le corpus reçu.")
    words_15 = Counter(tuple(map(ord, w)) for w in EXCERPT_V.split())      # the words of Verne, as code points
    expected_15 = Counter()
    for w, f in words_15.items():
        for p in zip(w, w[1:]):
            expected_15[p] += f
    got_15 = T.get_pair_counts(dict(words_15))
    verdict("B2.15", isinstance(got_15, dict) and dict(got_15) == dict(expected_15),
            f"sur les {len(words_15):_} mots distincts de l'extrait de Verne : {len(expected_15):_} paires, les mêmes comptes "
            "qu'un comptage direct.".replace("_", " "),
            "sur les mots de l'extrait de Verne, tes comptes diffèrent d'un comptage direct : as-tu pondéré par la "
            "fréquence et gardé les paires qui se chevauchent ?")
    refuses("B2.15", "get_pair_counts({(1, 2): -1}) (une fréquence négative)", lambda: T.get_pair_counts({(1, 2): -1}))

    ids_15 = (3, 1, 2, 1, 2, 1, 1, 1)                            # a tuple: merge_pair must accept it
    merged_15 = T.merge_pair(ids_15, (1, 2), 9)
    if returned("B2.15", "merge_pair", merged_15):
        verdict("B2.15", type(merged_15) is list and merged_15 == [3, 9, 9, 1, 1, 1],
                "merge_pair((3, 1, 2, 1, 2, 1, 1, 1), (1, 2), 9) = [3, 9, 9, 1, 1, 1], une list.",
                f"merge_pair({ids_15}, (1, 2), 9) doit rendre la list [3, 9, 9, 1, 1, 1] ; reçu {merged_15!r}.")
    overlap_15 = T.merge_pair([1, 1, 1], (1, 1), 5)
    verdict("B2.15", overlap_15 == [5, 1], "[1, 1, 1] avec (1, 1) donne [5, 1] : sans chevauchement, de gauche à droite.",
            f"merge_pair([1, 1, 1], (1, 1), 5) doit rendre [5, 1] (sans chevauchement, de gauche à droite) ; reçu {overlap_15!r}.")
    same_15 = [4, 5]
    out_15 = T.merge_pair(same_15, (1, 2), 9)
    verdict("B2.15", out_15 == [4, 5] and out_15 is not same_15 and same_15 == [4, 5],
            "sans occurrence de la paire : une liste neuve égale à l'entrée, l'entrée intacte.",
            "sans occurrence de la paire, merge_pair doit rendre une liste neuve (pas l'entrée elle-même), égale à "
            f"l'entrée, et laisser l'entrée intacte ; reçu {out_15!r}, entrée {same_15!r}.")
    refuses("B2.15", "merge_pair([1, 2], (1, 2, 3), 9) (une paire de trois ids)", lambda: T.merge_pair([1, 2], (1, 2, 3), 9))'''

CHECK_16 = r'''with checked("B2.16"):
    T = mylearn.tokenizer
    texts_16 = ["le chat dort, le rat court", "les chats et les rats", "un chaton"]
    copy_16 = list(texts_16)
    tok_16 = T.BPETokenizer(vocab_size=24, level="char", min_frequency=1)
    out_16 = tok_16.fit(copy_16)
    verdict("B2.16", out_16 is tok_16, "fit rend self.", "fit doit rendre self (return self à la fin).")
    verdict("B2.16", copy_16 == texts_16, "la liste des textes n'a pas changé.", "fit a modifié la liste des textes reçue.")
    alphabet_16 = sorted(set("".join(texts_16)) - {" "})
    got_alphabet_16 = [tok_16.vocab_[i] for i in range(getattr(tok_16, "n_base_", 0))]
    verdict("B2.16", got_alphabet_16 == [c.encode() for c in alphabet_16],
            f"alphabet : les {len(alphabet_16)} caractères des pré-tokens, triés par point de code (« , » en premier).",
            f"l'alphabet doit être les caractères des pré-tokens (sans les espaces), triés par point de code : attendu "
            f"{''.join(alphabet_16)!r}, reçu {b''.join(got_alphabet_16)!r}.")
    merges_16 = getattr(tok_16, "merges_", None)
    verdict("B2.16", isinstance(merges_16, list) and all(type(m) is tuple for m in merges_16)
            and len(tok_16.vocab_) == tok_16.n_base_ + len(merges_16) + len(tok_16.special_ids_) <= 24,
            f"merges_ : une liste de {len(merges_16 or [])} tuples ; len(vocab_) = n_base_ + fusions + spéciaux ≤ vocab_size.",
            "merges_ doit être une liste de tuples (a, b), et vocab_ contenir l'alphabet, les fusions et les tokens "
            "spéciaux, vocab_size au plus.")
    # the same merges as BpeTrainer of tokenizers, on the Verne excerpt
    hf_16 = reference_char_bpe(EXCERPT_V, 400)
    mine_16 = T.BPETokenizer(vocab_size=400, level="char").fit(EXCERPT_V)
    want_16 = [hf_16.id_to_token(i) for i in range(hf_16.get_vocab_size())]
    got_16 = [mine_16.vocab_[i].decode("utf-8", errors="replace") for i in range(len(mine_16.vocab_))]
    first_16 = next((i for i, (a, b) in enumerate(zip(got_16, want_16)) if a != b), None)
    verdict("B2.16", got_16 == want_16,
            f"extrait de Verne, niveau caractère, 400 tokens : les mêmes {len(want_16)} tokens, dans le même ordre, que "
            "BpeTrainer de tokenizers.",
            (f"extrait de Verne, 400 tokens : tes tokens diffèrent de ceux de BpeTrainer dès l'id {first_16} (toi "
             f"{got_16[first_16]!r}, tokenizers {want_16[first_16]!r}) : relis le départage et le tri de l'alphabet."
             if first_16 is not None else f"extrait de Verne : {len(got_16)} tokens chez toi, {len(want_16)} chez "
             "BpeTrainer : relis les conditions d'arrêt (vocab_size, min_frequency)."))
    # byte level, special tokens
    byte_16 = T.BPETokenizer(vocab_size=256 + 10 + 2, pre_tokenizer="gpt2", special_tokens=["<|endoftext|>", "<pad>"])
    byte_16.fit(EXCERPT_V)
    verdict("B2.16", byte_16.n_base_ == 256 and all(byte_16.vocab_[i] == bytes([i]) for i in range(256))
            and len(byte_16.merges_) == 10 and byte_16.special_ids_ == {"<|endoftext|>": 266, "<pad>": 267},
            "niveau octet : 256 octets, 10 fusions, puis <|endoftext|> en 266 et <pad> en 267 (vocab_size compte les spéciaux).",
            "niveau octet avec vocab_size = 268 et deux tokens spéciaux : attendu n_base_ = 256 (vocab_[i] = bytes([i])), "
            f"10 fusions, special_ids_ = {{'<|endoftext|>': 266, '<pad>': 267}} ; reçu n_base_ = "
            f"{getattr(byte_16, 'n_base_', None)}, {len(getattr(byte_16, 'merges_', []))} fusions, "
            f"special_ids_ = {getattr(byte_16, 'special_ids_', None)}.")
    refuses("B2.16", "level='word'", lambda: T.BPETokenizer(level="word").fit("abc"))
    refuses("B2.16", "unk_token au niveau octet", lambda: T.BPETokenizer(unk_token="<unk>").fit("abc"))
    refuses("B2.16", "vocab_size = 100 au niveau octet (moins que les 256 octets)", lambda: T.BPETokenizer(vocab_size=100).fit("abc"))'''

CHECK_17 = r'''with checked("B2.17"):
    T = mylearn.tokenizer
    char_17 = T.BPETokenizer(vocab_size=400, level="char").fit(EXCERPT_V)
    hf_17 = reference_char_bpe(EXCERPT_V, 400)
    sample_17 = EXCERPT_V[20000:24000]                    # characters seen during fit
    ids_17 = char_17.encode(sample_17)
    if returned("B2.17", "encode", ids_17):
        verdict("B2.17", type(ids_17) is list and all(isinstance(i, int) for i in ids_17) and ids_17 == hf_17.encode(sample_17).ids,
                f"niveau caractère : les {len(ids_17)} ids d'un passage de Verne sont ceux de tokenizers (fusions par rang).",
                "niveau caractère : encode doit rendre une list d'int identique aux ids de tokenizers (BpeTrainer) ; "
                "applique-t-il les fusions par rang, dans chaque pré-token ?")
    verdict("B2.17", char_17.encode("") == [], "encode('') = [].", f"encode('') doit rendre [] ; reçu {char_17.encode('')!r}.")
    unk_17 = T.BPETokenizer(vocab_size=300, level="char", unk_token="<unk>").fit(EXCERPT_V)
    u_17 = unk_17.special_ids_.get("<unk>")
    got_unk_17 = unk_17.encode("Fogg 🙂🙂 <unk>")
    verdict("B2.17", u_17 is not None and got_unk_17[-3:] == [u_17] * 3 and got_unk_17.count(u_17) == 3,
            "niveau caractère : chaque caractère inconnu donne l'id de <unk>, et « <unk> » écrit dans le texte aussi.",
            f"« Fogg 🙂🙂 <unk> » : attendu trois fois l'id {u_17} de <unk> (un par emoji inconnu, un pour « <unk> » "
            f"écrit tel quel) ; reçu {got_unk_17!r}.")
    byte_17 = T.BPETokenizer(vocab_size=600, pre_tokenizer="gpt2").fit(EXCERPT_V)
    text_17 = VERNE_VAL[:3000] + " 🙂 漢字, é !"
    try:
        back_17 = byte_17.decode(byte_17.encode(text_17))
    except UnicodeDecodeError as exc:
        back_17 = None
        print(f"❌ Ex B2.17 : decode(encode(texte)) lève UnicodeDecodeError ({exc.reason}) : decode doit recoller les "
              "octets de tous les tokens, puis décoder l'UTF-8 avec errors='replace'.")
    if back_17 is not None:
        at_17 = next((i for i, (a, b) in enumerate(zip(back_17, text_17)) if a != b), min(len(back_17), len(text_17)))
        verdict("B2.17", back_17 == text_17,
                "niveau octet, pré-tokenizer 'gpt2' : decode(encode(texte)) redonne le texte exact (emoji et idéogrammes compris).",
                "niveau octet avec 'gpt2', decode(encode(texte)) doit redonner le texte exact ; à partir du caractère "
                f"{at_17}, attendu {text_17[at_17:at_17 + 12]!r}, reçu {back_17[at_17:at_17 + 12]!r}.")
    try:
        lone_17 = byte_17.decode([0xC3])
        verdict("B2.17", lone_17 == "�", "decode([0xC3]) (un octet seul) = '�'.",
                f"decode([0xC3]), le premier octet de « é » seul, doit rendre '�' ; reçu {lone_17!r}.")
    except UnicodeDecodeError as exc:
        print(f"❌ Ex B2.17 : decode([0xC3]), le premier octet de « é » seul, lève UnicodeDecodeError ({exc.reason}) au "
              "lieu de rendre '�' (errors='replace').")
    pieces_17 = byte_17.tokenize("été")
    verdict("B2.17", isinstance(pieces_17, list) and all(isinstance(p, str) for p in pieces_17) and "".join(pieces_17) == "été",
            f"tokenize('été') = {pieces_17!r}.", f"tokenize doit rendre une liste de str ; reçu {pieces_17!r}.")
    refuses("B2.17", "encode avant fit", lambda: T.BPETokenizer().encode("abc"))
    refuses("B2.17", "decode([-1])", lambda: byte_17.decode([-1]))
    refuses("B2.17", "decode([len(vocab_)])", lambda: byte_17.decode([len(byte_17.vocab_)]))
    refuses("B2.17", "niveau caractère sans unk_token : encode('🙂')", lambda: char_17.encode("🙂"))
    import tempfile
    path_17 = os.path.join(tempfile.mkdtemp(prefix="wb_b2_17_"), "bpe.json")
    byte_17.save(path_17)                                  # save and load are provided: they keep the four attributes
    loaded_17 = type(byte_17).load(path_17)
    verdict("B2.17", loaded_17.encode(text_17) == byte_17.encode(text_17),
            "save puis load : le tokenizer rechargé encode exactement comme l'original.",
            "save puis load : le tokenizer rechargé n'encode pas comme l'original. load ne rend que les hyperparamètres "
            "et les quatre attributs n_base_, merges_, vocab_ et special_ids_ : ton encode lit-il un autre état, posé par "
            "fit (un cache privé) ? Relis le contrat.")'''

# 🐛 B2.18: the tokenizer to repair, its fixed version and the diagnostics
TODO_18 = r'''# The tokenizer to repair IN PLACE (same names, same signatures, same returns): the diagnostics of the next cell start
# from it
PIECES_18 = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")   # GPT-2 pre-tokens


def clean_18(text):
    """The text as the tokenizer learns it: accents written in a single standard form."""
    return unicodedata.normalize("NFD", text)


def merge_18(ids, pair, new_id):
    """A new list where each occurrence of pair, left to right and without overlap, becomes new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_18(text, vocab_size):
    """Byte-level BPE learned on text (GPT-2 pre-tokens; ties go to the smallest pair). Returns (merges, words): the
    merges in learning order, and {pre-token: its ids at the end of the training}."""
    text = clean_18(text)
    counts = Counter(PIECES_18.findall(text))
    words = {w: list(w.encode("utf-8")) for w in counts}
    merges = []
    while 256 + len(merges) < vocab_size:
        pairs = Counter()
        for w, ids in words.items():
            for pair in zip(ids, ids[1:]):
                pairs[pair] += counts[w]
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        words = {w: merge_18(ids, best, 255 + len(merges)) for w, ids in words.items()}
    return merges, words


def encode_18(text, merges):
    """The ids of text: inside each pre-token, its UTF-8 bytes, then the learned merges until none applies."""
    rank = {pair: r for r, pair in enumerate(merges)}
    out = []
    for piece in PIECES_18.findall(text):
        ids = list(piece.encode("utf-8"))
        i = 0
        while i < len(ids) - 1:
            pair = (ids[i], ids[i + 1])
            if pair in rank:
                ids[i:i + 2] = [256 + rank[pair]]
                i = 0
            else:
                i += 1
        out += ids
    return out


def decode_18(ids, merges):
    """The text of ids; a byte left alone (part of a character) becomes '�'."""
    vocab = {i: bytes([i]) for i in range(256)}
    for r, (a, b) in enumerate(merges):
        vocab[256 + r] = vocab[a] + vocab[b]
    return "".join(vocab[i].decode("utf-8") for i in ids)'''

SOLUTION_18 = r'''PIECES_18 = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")   # GPT-2 pre-tokens


def clean_18(text):
    """The text as the tokenizer learns it: accents written in a single standard form."""
    return unicodedata.normalize("NFC", text)                # 2. the form of the texts it will encode (NFC)


def merge_18(ids, pair, new_id):
    """A new list where each occurrence of pair, left to right and without overlap, becomes new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_18(text, vocab_size):
    """Byte-level BPE learned on text (GPT-2 pre-tokens; ties go to the smallest pair). Returns (merges, words): the
    merges in learning order, and {pre-token: its ids at the end of the training}."""
    text = clean_18(text)
    counts = Counter(PIECES_18.findall(text))
    words = {w: list(w.encode("utf-8")) for w in counts}
    merges = []
    while 256 + len(merges) < vocab_size:
        pairs = Counter()
        for w, ids in words.items():
            for pair in zip(ids, ids[1:]):
                pairs[pair] += counts[w]
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        words = {w: merge_18(ids, best, 255 + len(merges)) for w, ids in words.items()}
    return merges, words


def encode_18(text, merges):
    """The ids of text: inside each pre-token, its UTF-8 bytes, then the learned merges until none applies."""
    rank = {pair: r for r, pair in enumerate(merges)}
    out = []
    for piece in PIECES_18.findall(text):
        ids = list(piece.encode("utf-8"))
        while len(ids) > 1:                                  # 3. the pair of LOWEST RANK first, as in the training
            best = min(zip(ids, ids[1:]), key=lambda p: rank.get(p, math.inf))
            if best not in rank:
                break
            ids = merge_18(ids, best, 256 + rank[best])
        out += ids
    return out


def decode_18(ids, merges):
    """The text of ids; a byte left alone (part of a character) becomes '�'."""
    vocab = {i: bytes([i]) for i in range(256)}
    for r, (a, b) in enumerate(merges):
        vocab[256 + r] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")   # 1. join the bytes, then decode'''

GIVEN_18 = r'''TRAIN_18 = VERNE_TRAIN[:120_000]          # the training text of the diagnostics (≈ 1 s per training)
PROBE_18 = "« Phileas Fogg a gagné son pari ! » 🙂"


def diagnose_18(train, encode, decode, vocab_size=300):
    """Three diagnoses of a small tokenizer, in a row: the first one that fails stops the chain (one error can hide
    another). Returns True when the three pass."""
    result = train(TRAIN_18, vocab_size)
    if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict)):
        print(f"❌ Ex B2.18 : train_18 doit rendre le couple (merges, words) ; reçu {type(result).__name__}.")
        return False
    merges, words = result
    vocab = {i: bytes([i]) for i in range(256)}
    for r, (a, b) in enumerate(merges):
        vocab[256 + r] = vocab[a] + vocab[b]

    def shown(ids):
        return "[" + ", ".join(repr(vocab[i].decode("utf-8", errors="replace")) if i in vocab else str(i) for i in ids) + "]"

    def point1():
        try:
            back = decode(encode(PROBE_18, merges), merges)
        except UnicodeDecodeError as exc:
            return False, f"decode(encode(texte)) lève UnicodeDecodeError ({exc.reason}) sur « {PROBE_18} »", \
                "ce texte contient des caractères de plusieurs octets : à quel moment decode transforme-t-il des octets en texte ?"
        if back != PROBE_18:
            same_screen = unicodedata.normalize("NFC", back) == PROBE_18
            return False, f"decode(encode(texte)) rend {back!r} au lieu de {PROBE_18!r}", \
                ("les deux textes s'affichent pareil, mais leurs points de code diffèrent : compare-les"
                 if same_screen else "quels caractères changent, et combien d'octets ont-ils ?")
        try:
            lone = decode([0xC3], merges)
        except UnicodeDecodeError as exc:
            return False, f"decode([0xC3]) (le premier octet de « é », seul) lève UnicodeDecodeError ({exc.reason})", \
                "relis la docstring de decode_18 : que doit devenir un octet seul ?"
        if lone != "�":
            return False, f"decode([0xC3]) rend {lone!r} au lieu de '�'", "relis la docstring de decode_18."
        return True, "le texte d'essai revient à l'identique, et un octet seul donne '�'", ""

    def point2():
        ids = encode("é", merges)
        n = TRAIN_18.count("é")
        if len(ids) == 1:
            return True, f"« é » ({n} fois dans le texte d'entraînement) est un seul token", ""
        return False, f"« é » apparaît {n} fois dans le texte d'entraînement, mais encode le découpe en {len(ids)} tokens {[vocab.get(i, i) for i in ids]}", \
            "les fusions apprises pour « é » portent-elles sur les octets que reçoit encode ?"

    def point3():
        counts = Counter(PIECES_18.findall(TRAIN_18))
        diff = sorted((w for w in words if encode(w, merges) != words[w]), key=lambda w: (-counts.get(w, 0), w))
        if not diff:
            return True, f"les {len(words)} mots distincts du texte d'entraînement sont découpés par encode comme à la fin de l'entraînement", ""
        w = diff[0]
        return False, (f"{len(diff)} des {len(words)} mots distincts du texte d'entraînement ne sont pas découpés par encode "
                       f"comme à la fin de l'entraînement ; par exemple {w!r} : {shown(words[w])} à l'entraînement, "
                       f"{shown(encode(w, merges))} par encode"), \
            "compare les deux découpages : laquelle de ses fusions encode a-t-il faite en premier ?"

    points = [("decode redonne le texte", point1), ("« é » est un token", point2),
              ("encode découpe comme l'entraînement", point3)]
    for k, (name, run) in enumerate(points, start=1):
        ok, measured, why = run()
        if ok:
            print(f"✅ Ex B2.18 : point {k}, {name} ({measured}).")
        else:
            print(f"❌ Ex B2.18 : point {k} : {measured} ; {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex B2.18 : point {later} : corrige d'abord le point {k}.")
            return False
    print("✅ Ex B2.18 : les trois diagnostics passent.")
    return True'''

CHECK_18 = r'''with checked("B2.18"):
    ok_18 = diagnose_18(train_18, encode_18, decode_18)'''

EXPERIMENT_18 = r'''if not globals().get("ok_18"):
    print("⏳ Ex B2.18 : la comparaison attend que les trois points de la cellule précédente passent.")
else:
    with checked("B2.18"):
        merges_18, _ = train_18(TRAIN_18, 300)
        val_18 = VERNE_VAL[:20_000]
        ids_18 = encode_18(val_18, merges_18)
        print(f"Ta version réparée : {fr(n_bytes(val_18) / len(ids_18), 2)} octets par token sur 20 000 caractères de la "
              f"validation de Verne ; « {PROBE_18} » →", [decode_18([i], merges_18) for i in encode_18(PROBE_18, merges_18)])'''

# 📦 B2.19
TODO_19 = r'''from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def train_hf_19(text, vocab_size, min_frequency=2):
    """A Tokenizer of the library tokenizers: models.BPE, pre_tokenizers.Whitespace, trained by BpeTrainer on text
    (no special token, no progress bar)."""
    raise NotImplementedError("train_hf_19")


def hf_merges_19(hf_tok):
    """The merges of a trained BPE of tokenizers, as (left_id, right_id) tuples of ints, in learning order (rank)."""
    raise NotImplementedError("hf_merges_19")'''

SOLUTION_19 = r'''from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def train_hf_19(text, vocab_size, min_frequency=2):
    """A Tokenizer of the library tokenizers: models.BPE, pre_tokenizers.Whitespace, trained by BpeTrainer on text
    (no special token, no progress bar)."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, show_progress=False)
    tok.train_from_iterator([text], trainer)
    return tok


def hf_merges_19(hf_tok):
    """The merges of a trained BPE of tokenizers, as (left_id, right_id) tuples of ints, in learning order (rank)."""
    state = json.loads(hf_tok.to_str())                     # the tokenizer.json of the fiche (§202.5)
    vocab = hf_tok.get_vocab()                              # {token string: id}
    merges = state["model"]["merges"]                       # ["a", "b"] (or "a b" in older versions), rank order
    pairs = [m.split(" ", 1) if isinstance(m, str) else m for m in merges]
    return [(vocab[a], vocab[b]) for a, b in pairs]'''

CHECK_19 = r'''VOCAB_19 = 2000
ok_hf_19 = False
with checked("B2.19"):
    start_19 = time.perf_counter()
    hf_19 = train_hf_19(VERNE_TRAIN, VOCAB_19)
    seconds_hf_19 = time.perf_counter() - start_19
    merges_hf_19 = hf_merges_19(hf_19)
    if not returned("B2.19", "hf_merges_19", merges_hf_19):
        raise StopCheck
    if not (isinstance(merges_hf_19, list) and all(type(m) is tuple and len(m) == 2 and all(isinstance(i, int) for i in m)
                                                   for m in merges_hf_19)):
        print("❌ Ex B2.19 : hf_merges_19 doit rendre une liste de tuples de deux int (les ids des deux tokens fusionnés).")
        raise StopCheck
    n_base_19 = hf_19.get_vocab_size() - len(merges_hf_19)
    strings_19 = {i: s for s, i in hf_19.get_vocab().items()}
    rank_19 = next((r for r, (a, b) in enumerate(merges_hf_19) if strings_19.get(a, "") + strings_19.get(b, "") == "Passepartout"), None)
    val_ids_19 = hf_19.encode(VERNE_VAL).ids
    print(f"tokenizers : {hf_19.get_vocab_size()} tokens, dont {len(merges_hf_19)} fusions, appris en {fr(seconds_hf_19, 2)} s ; "
          f"la validation de Verne fait {len(val_ids_19)} tokens.")
    wb.check("B2.19a", n_base_19, computed=True)
    wb.check("B2.19b", rank_19, computed=True)
    wb.check("B2.19c", len(val_ids_19), computed=True)
    ok_hf_19 = True

if not bpe_written():
    print("⏳ Ex B2.19 : la comparaison à ton BPE attend BPETokenizer.fit et encode (B2.16 et B2.17).")
elif not ok_hf_19:
    print("⏳ Ex B2.19 : la comparaison à ton BPE attend train_hf_19 et hf_merges_19 (cellule précédente).")
else:
    with checked("B2.19"):
        start_19 = time.perf_counter()
        mine_19 = mylearn.tokenizer.BPETokenizer(vocab_size=VOCAB_19, level="char", pre_tokenizer="whitespace").fit(VERNE_TRAIN)
        seconds_mine_19 = time.perf_counter() - start_19
        first_19 = next((r for r, (a, b) in enumerate(zip(mine_19.merges_, merges_hf_19)) if tuple(a) != tuple(b)), None)
        verdict("B2.19", list(map(tuple, mine_19.merges_)) == merges_hf_19,
                f"ton BPE et tokenizers apprennent les mêmes {len(merges_hf_19)} fusions, dans le même ordre.",
                (f"ta fusion de rang {first_19} est {mine_19.merges_[first_19]}, celle de tokenizers {merges_hf_19[first_19]}"
                 if first_19 is not None else f"{len(mine_19.merges_)} fusions chez toi, {len(merges_hf_19)} chez tokenizers")
                + " : relis le départage et l'alphabet (B2.16), puis hf_merges_19.")
        verdict("B2.19", mine_19.encode(VERNE_VAL) == val_ids_19,
                "et les mêmes ids pour toute la validation de Verne.",
                "les ids de la validation diffèrent de ceux de tokenizers : relis encode (B2.17).")
        print(f"Temps de l'entraînement : ton BPETokenizer {fr(seconds_mine_19, 2)} s, tokenizers {fr(seconds_hf_19, 2)} s "
              f"(rapport {seconds_mine_19 / seconds_hf_19:.0f}, sur cette machine, pour cette exécution).")'''

RECORD_19 = r'''wb.record("B2.19a", n_base_19, decimals=4, mistakes={
    "c'est le nombre de fusions : on demande la taille de l'alphabet de base": len(merges_hf_19),
    "c'est la taille totale du vocabulaire : retire les fusions": hf_19.get_vocab_size()})
wb.record("B2.19b", rank_19, decimals=4, mistakes={
    "c'est l'identifiant du token : le rang d'une fusion part de 0, après l'alphabet": rank_19 + n_base_19,
    "les rangs commencent à 0": rank_19 + 1})
wb.record("B2.19c", len(val_ids_19), decimals=4, mistakes={
    "c'est le nombre de mots de la validation : on demande ses tokens": len(VERNE_VAL.split())})'''

# --- Part C -----------------------------------------------------------------------------------------------------------
TODO_20 = r'''def compression_20(tok, text):
    """The bytes per token of text: its number of UTF-8 bytes / the number of ids of tok.encode(text)."""
    raise NotImplementedError("compression_20")


def token_entropy_20(tok, text):
    """The entropy, in bits, of the frequencies of the ids of tok.encode(text)."""
    raise NotImplementedError("token_entropy_20")


def sweep_20(big_tok, text, sizes):
    """One dict per size of sizes, in their order: {"vocab_size", "bytes_per_token", "bits_per_token",
    "bits_per_byte", "used"} measured on text with truncated_bpe(big_tok, size); bits_per_byte = bits_per_token /
    bytes_per_token, "used" = the number of distinct ids of the encoded text."""
    raise NotImplementedError("sweep_20")'''

SOLUTION_20 = r'''def compression_20(tok, text):
    """The bytes per token of text: its number of UTF-8 bytes / the number of ids of tok.encode(text)."""
    return len(text.encode("utf-8")) / len(tok.encode(text))


def token_entropy_20(tok, text):
    """The entropy, in bits, of the frequencies of the ids of tok.encode(text)."""
    counts = np.array(list(Counter(tok.encode(text)).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def sweep_20(big_tok, text, sizes):
    """One dict per size of sizes, in their order: {"vocab_size", "bytes_per_token", "bits_per_token",
    "bits_per_byte", "used"} measured on text with truncated_bpe(big_tok, size); bits_per_byte = bits_per_token /
    bytes_per_token, "used" = the number of distinct ids of the encoded text."""
    rows = []
    for size in sizes:
        tok = truncated_bpe(big_tok, size)
        per_token = compression_20(tok, text)
        bits = token_entropy_20(tok, text)
        rows.append({"vocab_size": size, "bytes_per_token": per_token, "bits_per_token": bits,
                     "bits_per_byte": bits / per_token, "used": len(set(tok.encode(text)))})
    return rows'''

CHECK_20 = r'''class ListTokenizer_20:
    """A fake tokenizer for the check: encode gives the ids it was built with, whatever the text."""
    def __init__(self, ids):
        self.ids = list(ids)

    def encode(self, text):
        return list(self.ids)


ok_20 = False
with checked("B2.20"):
    c_20 = compression_20(ListTokenizer_20([7, 8]), "été")
    if returned("B2.20", "compression_20", c_20):
        verdict("B2.20", isinstance(c_20, (int, float)) and math.isclose(c_20, 2.5),
                "compression_20 : « été » en 2 tokens, 5 octets UTF-8, soit 2,5 octets par token.",
                f"« été » (5 octets UTF-8, 3 caractères) encodé en 2 tokens : attendu 2,5 octets par token ; reçu {c_20!r}.")
    h_20 = token_entropy_20(ListTokenizer_20([4, 4, 9, 2]), "x")
    if returned("B2.20", "token_entropy_20", h_20):
        verdict("B2.20", isinstance(h_20, (int, float)) and math.isclose(h_20, 1.5),
                "token_entropy_20 : les ids [4, 4, 9, 2] ont une entropie de 1,5 bit.",
                f"les ids [4, 4, 9, 2] (fréquences 1/2, 1/4, 1/4) : attendu 1,5 bit (logarithme de base 2) ; reçu {h_20!r}.")
    h0_20 = token_entropy_20(ListTokenizer_20([3, 3, 3]), "x")
    verdict("B2.20", isinstance(h0_20, (int, float)) and math.isclose(h0_20, 0.0, abs_tol=1e-12),
            "un seul id répété : entropie nulle.", f"un seul id répété doit donner une entropie nulle ; reçu {h0_20!r}.")
    ok_20 = (isinstance(c_20, (int, float)) and math.isclose(c_20, 2.5) and isinstance(h_20, (int, float))
             and math.isclose(h_20, 1.5) and math.isclose(h0_20, 0.0, abs_tol=1e-12))'''

EXPERIMENT_20 = r'''SIZES_20 = (300, 500, 1000, 2000, 4000, 8000)
if not bpe_written():
    print("⏳ Ex B2.20 : l'expérience attend BPETokenizer.fit et encode (B2.16 et B2.17).")
elif not bpe_rebuild_ok("B2.20"):
    pass
elif not globals().get("ok_20"):
    print("⏳ Ex B2.20 : l'expérience attend que compression_20 et token_entropy_20 passent la cellule précédente.")
else:
    with checked("B2.20"):
        start_20 = time.perf_counter()
        BIG_20 = fast_byte_bpe(VERNE_TRAIN, max(SIZES_20))
        print(f"Les {len(BIG_20.merges_)} fusions d'un BPE octet de {max(SIZES_20)} tokens sur VERNE_TRAIN, apprises par "
              f"tokenizers : {fr(time.perf_counter() - start_20, 1)} s.")
        part_20 = VERNE_TRAIN[:100_000]
        start_20 = time.perf_counter()
        mine_20 = mylearn.tokenizer.BPETokenizer(vocab_size=400, pre_tokenizer="gpt2").fit(part_20)
        seconds_20 = time.perf_counter() - start_20
        verdict("B2.20", mine_20.merges_ == fast_byte_bpe(part_20, 400).merges_,
                f"ton fit (400 tokens sur 100 000 caractères de Verne, {fr(seconds_20, 1)} s) apprend les mêmes fusions que "
                "fast_byte_bpe.",
                "ton fit et fast_byte_bpe n'apprennent pas les mêmes fusions sur 100 000 caractères de Verne : relis "
                "B2.16 (départage, min_frequency, niveau octet).")
        verdict("B2.20", truncated_bpe(BIG_20, 400).merges_ == fast_byte_bpe(VERNE_TRAIN, 400).merges_,
                f"les 144 premières fusions de l'entraînement à {max(SIZES_20)} tokens sont celles d'un entraînement à 400 : "
                "truncated_bpe redonne le tokenizer plus petit.",
                "les premières fusions d'un entraînement long devraient être celles d'un entraînement court.")
        rows_20 = sweep_20(BIG_20, VERNE_VAL, SIZES_20)
        if not returned("B2.20", "sweep_20", rows_20):
            raise StopCheck
        table_20 = pd.DataFrame(rows_20)
        need_20 = ["vocab_size", "bytes_per_token", "bits_per_token", "bits_per_byte", "used"]
        if list(table_20.get("vocab_size", [])) != list(SIZES_20) or any(c not in table_20 for c in need_20):
            print(f"❌ Ex B2.20 : sweep_20 doit rendre une ligne par taille de {SIZES_20}, dans cet ordre, avec les clés {need_20}.")
            raise StopCheck
        row_20 = rows_20[SIZES_20.index(1000)]
        tok1000_20 = truncated_bpe(BIG_20, 1000)
        ids1000_20 = tok1000_20.encode(VERNE_VAL)
        verdict("B2.20", math.isclose(row_20["bytes_per_token"], compression_20(tok1000_20, VERNE_VAL))
                and math.isclose(row_20["bits_per_token"], token_entropy_20(tok1000_20, VERNE_VAL))
                and math.isclose(row_20["bits_per_byte"], row_20["bits_per_token"] / row_20["bytes_per_token"])
                and row_20["used"] == len(set(ids1000_20)),
                "à 1 000 tokens, la ligne de sweep_20 reprend tes deux mesures, leur quotient et le nombre d'ids distincts.",
                "à 1 000 tokens, la ligne de sweep_20 doit reprendre compression_20, token_entropy_20 (sur "
                "truncated_bpe(big_tok, 1000)), leur quotient et le nombre d'identifiants distincts du texte encodé.")
        print(table_20[need_20].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        fig_20, axes_20 = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, col, label in zip(axes_20, ("bytes_per_token", "bits_per_token", "bits_per_byte"),
                                  ("octets par token", "entropie (bits par token)", "bits par octet")):
            ax.plot(table_20["vocab_size"], table_20[col], "o-")
            ax.set_xscale("log")
            ax.set_xticks(SIZES_20, [str(s) for s in SIZES_20])
            ax.set_xlabel("taille du vocabulaire")
            ax.set_title(label)
            ax.grid(alpha=0.3)
        fig_20.suptitle("BPE octet de Verne (VERNE_TRAIN), mesuré sur VERNE_VAL")
        fig_20.tight_layout()
        plt.show()'''

TODO_21 = r'''def train_tokenizers_21(vocab_size=2000):
    """Three byte-level BPETokenizer of your library, pre_tokenizer "gpt2", vocab_size tokens:
    {"holmes": learned on HOLMES_TRAIN, "verne": on VERNE_TRAIN, "mixte": on both texts}."""
    raise NotImplementedError("train_tokenizers_21")


def cross_table_21(tokenizers, texts):
    """A pandas DataFrame of the bytes per token (compression_20): one row per tokenizer, one column per text
    (tokenizers and texts are dicts name -> object)."""
    raise NotImplementedError("cross_table_21")'''

SOLUTION_21 = r'''def train_tokenizers_21(vocab_size=2000):
    """Three byte-level BPETokenizer of your library, pre_tokenizer "gpt2", vocab_size tokens:
    {"holmes": learned on HOLMES_TRAIN, "verne": on VERNE_TRAIN, "mixte": on both texts}."""
    return {"holmes": fast_byte_bpe(HOLMES_TRAIN, vocab_size),          # the merges of fit, learned in Rust
            "verne": fast_byte_bpe(VERNE_TRAIN, vocab_size),
            "mixte": fast_byte_bpe([HOLMES_TRAIN, VERNE_TRAIN], vocab_size)}


def cross_table_21(tokenizers, texts):
    """A pandas DataFrame of the bytes per token (compression_20): one row per tokenizer, one column per text
    (tokenizers and texts are dicts name -> object)."""
    return pd.DataFrame({text_name: {tok_name: compression_20(tok, text) for tok_name, tok in tokenizers.items()}
                         for text_name, text in texts.items()})'''

CHECK_21 = r'''if not bpe_written():
    print("⏳ Ex B2.21 : l'expérience attend BPETokenizer.fit et encode (B2.16 et B2.17).")
elif bpe_rebuild_ok("B2.21"):
    with checked("B2.21"):
        need("B2.21", "compression_20")
        start_21 = time.perf_counter()
        toks_21 = train_tokenizers_21(2000)
        print(f"Trois tokenizers de 2 000 tokens : {fr(time.perf_counter() - start_21, 1)} s.")
        if not returned("B2.21", "train_tokenizers_21", toks_21):
            raise StopCheck
        if not isinstance(toks_21, dict) or set(toks_21) != {"holmes", "verne", "mixte"}:
            print("❌ Ex B2.21 : train_tokenizers_21 doit rendre un dictionnaire de clés 'holmes', 'verne' et 'mixte'.")
            raise StopCheck
        for name, tok in toks_21.items():
            ok = (getattr(tok, "level", None) == "byte" and getattr(tok, "pre_tokenizer", None) == "gpt2"
                  and len(getattr(tok, "vocab_", {})) == 2000 and not getattr(tok, "special_ids_", {}))
            verdict("B2.21", ok, f"{name} : BPE octet 'gpt2' de 2 000 tokens.",
                    f"{name} : attendu un BPETokenizer de niveau octet, pré-tokenizer 'gpt2', 2 000 tokens sans token spécial.")
        if "BIG_20" in globals():
            verdict("B2.21", toks_21["verne"].merges_ == BIG_20.merges_[:2000 - 256],
                    "le tokenizer de Verne a les 1 744 premières fusions de celui de B2.20 : il est appris sur VERNE_TRAIN.",
                    "le tokenizer de Verne devrait avoir les premières fusions de BIG_20 (B2.20) : sur quel texte est-il appris ?")
        verdict("B2.21", toks_21["holmes"].merges_[:20] != toks_21["verne"].merges_[:20]
                and toks_21["mixte"].merges_ not in (toks_21["holmes"].merges_, toks_21["verne"].merges_),
                "les trois tokenizers ont des fusions différentes.",
                "deux de tes tokenizers ont les mêmes fusions : sont-ils appris sur des textes différents ?")
        texts_21 = {"Holmes (validation)": HOLMES_VAL, "Verne (validation)": VERNE_VAL}
        table_21 = cross_table_21(toks_21, texts_21)
        if not returned("B2.21", "cross_table_21", table_21):
            raise StopCheck
        if not isinstance(table_21, pd.DataFrame) or set(table_21.index) != set(toks_21) or set(table_21.columns) != set(texts_21):
            print("❌ Ex B2.21 : cross_table_21 doit rendre un DataFrame, une ligne par tokenizer, une colonne par texte.")
            raise StopCheck
        verdict("B2.21", math.isclose(float(table_21.loc["verne", "Verne (validation)"]),
                                      compression_20(toks_21["verne"], VERNE_VAL)),
                "le tableau contient les octets par token de compression_20.",
                "le tableau doit contenir compression_20(tokenizer, texte) : relis-le.")
        gpt2_21 = wbtext.try_tokenizer("gpt2", "B2.21")
        if gpt2_21 is not None:
            table_21.loc["GPT-2 (50 257 tokens)"] = pd.Series({name: n_bytes(t) / wbtext.count_tokens(gpt2_21, t)
                                                              for name, t in texts_21.items()})
        print(table_21.to_string(float_format=lambda x: f"{x:.2f}"))
        ax_21 = table_21.plot.bar(figsize=(7.5, 3.6), rot=0)
        ax_21.set_ylabel("octets par token")
        ax_21.set_title("Chaque tokenizer sur la validation des deux textes")
        ax_21.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()'''
