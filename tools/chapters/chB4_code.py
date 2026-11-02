"""Code cells of the chapter B4 notebook, first half (parts A and B: B4.12 to B4.21), shared by build_chB4.py and
history_chB4.py (used by Claude).

Each constant is the source of one cell: TOOLS_* (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell). The line that creates the
cache of replies differs between the notebooks: build_chB4.py replaces LEARNER_CACHE by SOLUTION_CACHE_LINE in the
solutions notebook (its own file, read from the replies precomputed by history_chB4.py; never the learner's).
"""

LEARNER_CACHE = 'CACHE = wbllm.GenerationCache("learner")          # your replies, cached outside the repository'
SOLUTION_CACHE_LINE = ('CACHE = wbllm.GenerationCache("solution", frozen=wb.repo_root() / wbllm.SOLUTION_CACHE)   '
                       '# precomputed replies; never the learner\'s')

TOOLS_A = r'''# Tools for the notebook exercises of part A (usage and prompting)
import json
import threading
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore", message=".*IProgress not found.*")    # tqdm without ipywidgets (transformers)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, TextStreamer, pipeline
from wb import llm as wbllm
from wb import rag as wbrag
from wb.nbtools import StopCheck, checked, filled, fr, pair, refuses, returned, run_pytest_group, same, verdict, written

SMOL = wbllm.LLMS["smollm2"]                      # repo, revision and licence of SmolLM2-135M-Instruct
''' + LEARNER_CACHE + r'''
LLM = wbllm.try_llm("smollm2", "B4.12", weights=True)   # the reference loader of the checks and the experiments


def spy_from_pretrained(calls):
    """A context manager: while it is open, AutoTokenizer.from_pretrained and AutoModelForCausalLM.from_pretrained
    write (kind, name, revision, dtype) into `calls`, then do their usual work; restored afterwards."""
    import contextlib

    @contextlib.contextmanager
    def spy():
        saved = {cls: cls.__dict__.get("from_pretrained") for cls in (AutoTokenizer, AutoModelForCausalLM)}
        for cls, kind in ((AutoTokenizer, "tokenizer"), (AutoModelForCausalLM, "model")):
            original = cls.from_pretrained

            def wrapper(*args, _original=original, _kind=kind, **kwargs):
                name = args[0] if args else kwargs.get("pretrained_model_name_or_path")
                calls.append((_kind, str(name), kwargs.get("revision"), kwargs.get("dtype", kwargs.get("torch_dtype"))))
                return _original(*args, **kwargs)

            cls.from_pretrained = wrapper
        try:
            yield calls
        finally:
            for cls, attr in saved.items():
                if attr is None:
                    del cls.from_pretrained
                else:
                    cls.from_pretrained = attr

    return spy()


print(f"Modèle des expériences : {SMOL['repo']} @ {SMOL['revision'][:7]} ({SMOL['license']}) ; "
      f"cache des réponses : {CACHE.path}")'''

# --- 📦 B4.12 ---------------------------------------------------------------------------------------------------------
GIVEN_12 = r'''PROMPT_12 = ("To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other "
             "name.")                                              # a raw text, continued without chat template
QUESTION_12 = "Who is Doctor Watson?"                              # a user message, for the pipeline'''

TODO_12 = r'''def load_12():
    """(tok, model): the tokenizer and the model SmolLM2-135M-Instruct (SMOL["repo"]) at the revision
    SMOL["revision"], the model in float32 and in eval mode."""
    raise NotImplementedError("load_12")


def complete_12(tok, model, prompt, max_new_tokens):
    """Greedy continuation of the raw text `prompt` (no chat template, at most max_new_tokens new tokens):
    (text, n_new), the generated text only (the prompt removed), decoded without special tokens, and the number of
    tokens generated."""
    raise NotImplementedError("complete_12")


def counts_12(tok, model):
    """(n_prompt, n_linear, n_block): the number of tokens of tok(PROMPT_12), the number of nn.Linear modules of the
    model (its output head included), the number of parameters of its first decoder block."""
    raise NotImplementedError("counts_12")


def make_pipe_12(tok, model):
    """A transformers pipeline "text-generation" built on these two objects."""
    raise NotImplementedError("make_pipe_12")


estimate_12 = ...   # your estimate of ✏️ B4.1 j): the generation speed of SmolLM2 in float32, in tokens per second'''

SOLUTION_12 = r'''def load_12():
    """(tok, model): SmolLM2-135M-Instruct at its frozen revision, float32, eval mode."""
    tok = AutoTokenizer.from_pretrained(SMOL["repo"], revision=SMOL["revision"])
    model = AutoModelForCausalLM.from_pretrained(SMOL["repo"], revision=SMOL["revision"], dtype=torch.float32)
    return tok, model.eval()                     # from_pretrained already gives eval(): said explicitly


def complete_12(tok, model, prompt, max_new_tokens):
    """Greedy continuation of a raw text: (generated text, number of generated tokens)."""
    enc = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0])


def counts_12(tok, model):
    """(tokens of PROMPT_12, nn.Linear modules, parameters of the first decoder block)."""
    n_prompt = len(tok(PROMPT_12)["input_ids"])
    n_linear = sum(isinstance(m, nn.Linear) for m in model.modules())
    n_block = sum(p.numel() for p in model.model.layers[0].parameters())
    return n_prompt, n_linear, n_block


def make_pipe_12(tok, model):
    return pipeline("text-generation", model=model, tokenizer=tok)


estimate_12 = 28.0   # ✏️ B4.1 j), by proportion of the memories (the solution of B4.1)'''

CHECK_12 = r'''tok_12 = model_12 = None
calls_12 = []
with checked("B4.12"):
    if LLM is None:
        print("⏳ Ex B4.12 : SmolLM2 n'a pas pu être chargé : l'exercice est « à valider sur Colab ».")
        raise StopCheck
    with spy_from_pretrained(calls_12), wb.transfer.quiet_hub():
        loaded_12 = load_12()
    tok_12, model_12 = pair("B4.12", "load_12", loaded_12, names=("tok", "model"))
    seen_12 = sorted({(kind, rev) for kind, name, rev, _ in calls_12 if name == SMOL["repo"]}, key=str)
    verdict("B4.12", {("tokenizer", SMOL["revision"]), ("model", SMOL["revision"])} <= set(seen_12),
            f"tokenizer et modèle lus par from_pretrained à la révision figée {SMOL['revision'][:7]}.",
            f"load_12 doit appeler AutoTokenizer.from_pretrained et AutoModelForCausalLM.from_pretrained avec "
            f"{SMOL['repo']!r} et revision=SMOL[\"revision\"] ; vu : {seen_12 or 'aucun appel'}.")
    if not isinstance(model_12, nn.Module):
        print(f"❌ Ex B4.12 : le second élément rendu par load_12 doit être le modèle (un nn.Module), pas "
              f"{type(model_12).__name__}.")
        model_12 = None
        raise StopCheck
    dtype_12 = next(model_12.parameters()).dtype
    verdict("B4.12", dtype_12 == torch.float32, "poids en float32.",
            f"les poids doivent être en float32 (dtype=torch.float32 dans from_pretrained) ; reçu {dtype_12}.")
    verdict("B4.12", not model_12.training, "modèle en mode eval.", "le modèle doit être en mode eval (model.eval()).")

if model_12 is not None:
    with checked("B4.12"):
        counted_12 = counts_12(tok_12, model_12)
        if returned("B4.12", "counts_12", counted_12):
            if not (isinstance(counted_12, (tuple, list)) and len(counted_12) == 3):
                print("❌ Ex B4.12 : counts_12 doit rendre un triplet (n_prompt, n_linear, n_block).")
                raise StopCheck
            for letter_12, value_12 in zip("abc", counted_12):
                wb.check(f"B4.12{letter_12}", value_12, computed=True)
    with checked("B4.12"):
        complete_12(tok_12, model_12, PROMPT_12, 4)                       # warm-up: the first call is slower
        start_12 = time.perf_counter()
        text_12, n_new_12 = pair("B4.12", "complete_12", complete_12(tok_12, model_12, PROMPT_12, 64),
                                 names=("text", "n_new"))
        seconds_12 = time.perf_counter() - start_12
        want_12 = LLM.generate(PROMPT_12, chat=False, max_new_tokens=64, cache=CACHE)
        verdict("B4.12", text_12 == want_12,
                "la continuation gloutonne est celle de la référence (texte brut, tokens générés seuls).",
                f"continuation inattendue : attendu {want_12[:70]!r}…, reçu {str(text_12)[:70]!r}… (texte brut sans "
                "chat template, do_sample=False, seulement les tokens générés, sans tokens spéciaux)")
        verdict("B4.12", isinstance(n_new_12, int) and 1 <= n_new_12 <= 64,
                f"{n_new_12} tokens générés.", f"n_new doit être le nombre de tokens générés (un entier de 1 à 64) ; "
                f"reçu {n_new_12!r}.")
        if isinstance(n_new_12, int) and n_new_12 > 0:
            speed_12 = n_new_12 / seconds_12
            print(f"Mesure : {n_new_12} tokens en {fr(seconds_12, 2)} s, soit {fr(speed_12, 1)} tokens par seconde "
                  "(une mesure, pas une réponse : elle dépend de la machine).")
            if filled(estimate_12):
                print(f"Ton estimation de ✏️ B4.1 j) : {fr(float(estimate_12), 1)} tokens par seconde ; "
                      f"mesure / estimation = {fr(speed_12 / float(estimate_12), 2)}.")
    with checked("B4.12"):
        pipe_12 = make_pipe_12(tok_12, model_12)
        with wb.text.quiet_transformers():
            out_12 = pipe_12([{"role": "user", "content": QUESTION_12}], max_new_tokens=24, do_sample=False)
        reply_12 = out_12[0]["generated_text"][-1]["content"]
        want_reply_12 = LLM.generate(QUESTION_12, max_new_tokens=24, cache=CACHE)
        verdict("B4.12", reply_12 == want_reply_12,
                f"le pipeline rend la réponse gloutonne attendue : {reply_12!r}",
                f"le pipeline devrait rendre {want_reply_12!r} (do_sample=False, 24 tokens) ; reçu {reply_12!r}.")'''

RECORD_12 = None   # built in chB4_parts.py (mistakes)

# --- 🔮 B4.13 ---------------------------------------------------------------------------------------------------------
GIVEN_13 = r'''PROMPT_13 = "Name the doctor who shares rooms with Sherlock Holmes. Answer with a name."
TEMPERATURES_13 = (0.0, 0.7, 1.5)
SEEDS_13 = range(10)


def draws_13(temperature):
    """The 10 replies of SmolLM2 to PROMPT_13 (24 new tokens at most): greedy for a temperature of 0, else a random
    draw at this temperature, without top-k or top-p filter, after torch.manual_seed(seed) for seed = 0 to 9."""
    if temperature == 0:
        return [LLM.generate(PROMPT_13, max_new_tokens=24, cache=CACHE) for _ in SEEDS_13]
    return [LLM.generate(PROMPT_13, max_new_tokens=24, do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                         seed=seed, cache=CACHE) for seed in SEEDS_13]'''

TODO_13 = r'''identical_13 = ...   # a) at 0.7: are at least 3 of the 10 replies identical, word for word, to the reply at
                     #    temperature 0? True or False
watson_13 = ...      # b) at 0.7: do at least 7 of the 10 replies name Watson? True or False'''

EXPERIMENT_13 = r'''if LLM is None:
    print("⏳ Ex B4.13 : SmolLM2 n'a pas pu être chargé : l'expérience est « à valider sur Colab ».")
else:
    replies_13 = {t: draws_13(t) for t in TEMPERATURES_13}
    greedy_13 = replies_13[0.0][0]
    for t, replies in replies_13.items():
        names = sum("watson" in r.lower() for r in replies)
        twins = sum(r == greedy_13 for r in replies)
        print(f"\n— température {fr(t, 1)} : {len(set(replies))} réponses distinctes sur 10, {names} nomment Watson, "
              f"{twins} identiques à la réponse gloutonne")
        for seed, r in zip(SEEDS_13, replies):
            print(f"  {seed} : {r!r}")
    wb.check("B4.13a", identical_13)
    wb.check("B4.13b", watson_13)'''

# --- 📦 B4.14 ---------------------------------------------------------------------------------------------------------
GIVEN_14 = r'''ONE_14 = [{"role": "user", "content": "Who is Sherlock Holmes?"}]
SYSTEM_14 = {
    "short": [{"role": "system", "content": "Answer in one short sentence."}] + ONE_14,
    "french": [{"role": "system", "content": "Tu réponds toujours en français."},
               {"role": "user", "content": "Qui est Phileas Fogg ?"}],
}
LIST_14 = [{"role": "user", "content": "Name three stories about Sherlock Holmes, one per line."}]'''

TODO_14 = r'''def chat_14(tok, model, messages, max_new_tokens=48, stop_strings=None, streamer=None):
    """Greedy reply of the chat model to `messages` (a list of {"role": ..., "content": ...}): the chat template with
    add_generation_prompt=True, generate() (do_sample=False, at most max_new_tokens new tokens, the stop strings
    and the streamer passed on), the new tokens only, decoded without special tokens, cut before the first stop
    string."""
    raise NotImplementedError("chat_14")


def stream_14(tok, model, messages, max_new_tokens=48):
    """The pieces of the same greedy reply, in order, as a TextIteratorStreamer gives them while generate() runs in
    another thread: a list of strings whose concatenation is the reply."""
    raise NotImplementedError("stream_14")'''

SOLUTION_14 = r'''def chat_14(tok, model, messages, max_new_tokens=48, stop_strings=None, streamer=None):
    """Greedy reply to a conversation (chat template, new tokens only, stop string removed)."""
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, stop_strings=stop_strings,
                             tokenizer=tok if stop_strings else None, streamer=streamer)
    reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for stop in stop_strings or ():
        reply = reply.split(stop)[0]             # cut before the stop string (its last token may hold more)
    return reply


def stream_14(tok, model, messages, max_new_tokens=48):
    """The pieces of the reply, as a TextIteratorStreamer yields them while generate() runs in a thread."""
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def run():
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)

    thread = threading.Thread(target=run)
    thread.start()
    pieces = list(streamer)                      # blocks until the generation ends
    thread.join()
    return pieces'''

CHECK_14 = r'''if LLM is not None and globals().get("model_12") is not None:
    TOK_14, MODEL_14 = tok_12, model_12
elif LLM is not None:
    TOK_14, MODEL_14 = LLM.tokenizer, LLM.model
    print("ℹ️ Ex B4.14 : load_12 (B4.12) n'est pas encore écrit : la suite prend le même SmolLM2, chargé par la référence.")
else:
    TOK_14 = MODEL_14 = None
    print("⏳ Ex B4.14 : SmolLM2 n'a pas pu être chargé : l'exercice est « à valider sur Colab ».")


def expected_14(messages, stop_strings=None):
    """The reference reply (greedy, 48 tokens), cut before its first stop string."""
    reply = LLM.generate(messages, max_new_tokens=48, stop_strings=stop_strings, cache=CACHE)
    for stop in stop_strings or ():
        reply = reply.split(stop)[0]
    return reply


if MODEL_14 is not None:
    with checked("B4.14"):
        print("Le prompt de ONE_14 après le chat template :", repr(LLM.render(ONE_14)))
        reply_14 = chat_14(TOK_14, MODEL_14, ONE_14)
        if returned("B4.14", "chat_14", reply_14):
            want_14 = expected_14(ONE_14)
            verdict("B4.14", reply_14 == want_14, f"une question : {reply_14!r}",
                    f"chat_14(ONE_14) devrait rendre {want_14!r} (chat template avec add_generation_prompt=True, "
                    f"glouton, tokens générés seuls) ; reçu {reply_14!r}.")
        for name_14, conv_14 in SYSTEM_14.items():
            got_14, want_14 = chat_14(TOK_14, MODEL_14, conv_14), expected_14(conv_14)
            verdict("B4.14", got_14 == want_14, f"message système « {conv_14[0]['content']} » : {got_14!r}",
                    f"avec le message système {conv_14[0]['content']!r}, attendu {want_14!r} ; reçu {got_14!r}.")
        two_14 = ONE_14 + [{"role": "assistant", "content": expected_14(ONE_14)},
                           {"role": "user", "content": "Who is his best friend?"}]
        got_14, want_14 = chat_14(TOK_14, MODEL_14, two_14), expected_14(two_14)
        verdict("B4.14", got_14 == want_14, f"second tour (l'historique repasse dans le prompt) : {got_14!r}",
                f"au second tour, attendu {want_14!r} ; reçu {got_14!r}.")
    with checked("B4.14"):
        full_14 = chat_14(TOK_14, MODEL_14, LIST_14)
        cut_14 = chat_14(TOK_14, MODEL_14, LIST_14, stop_strings=["\n"])
        want_cut_14 = expected_14(LIST_14, stop_strings=["\n"])
        verdict("B4.14", cut_14 == want_cut_14 and "\n" not in cut_14,
                f"arrêt au premier retour à la ligne, coupé avant lui : {cut_14!r} (sans arrêt : {full_14!r})",
                f"avec stop_strings=[\"\\n\"], attendu {want_cut_14!r} (le texte coupé avant la chaîne d'arrêt) ; "
                f"reçu {cut_14!r}.")
    with checked("B4.14"):
        print("Diffusé au fil de l'eau par TextStreamer : ", end="", flush=True)
        streamed_14 = chat_14(TOK_14, MODEL_14, ONE_14,
                              streamer=TextStreamer(TOK_14, skip_prompt=True, skip_special_tokens=True))
        verdict("B4.14", streamed_14 == expected_14(ONE_14), "avec un TextStreamer, la réponse rendue est la même.",
                "avec streamer=TextStreamer(...), chat_14 doit rendre la même réponse que sans.")
        pieces_14 = stream_14(TOK_14, MODEL_14, ONE_14)
        if returned("B4.14", "stream_14", pieces_14):
            ok_14 = isinstance(pieces_14, list) and all(isinstance(p, str) for p in pieces_14)
            joined_14 = "".join(pieces_14) if ok_14 else None
            verdict("B4.14", ok_14 and joined_14 == expected_14(ONE_14) and len(pieces_14) > 1,
                    f"TextIteratorStreamer : {len(pieces_14) if ok_14 else 0} morceaux, dont la concaténation est la "
                    f"réponse ; les premiers : {str(pieces_14)[:80]}",
                    "stream_14 doit rendre la liste des morceaux (plus d'un) dont la concaténation est la réponse "
                    f"gloutonne ; reçu {str(pieces_14)[:120]}.")'''

# --- 📦 B4.15 ---------------------------------------------------------------------------------------------------------
GIVEN_15 = r'''SYSTEM_15 = ("Extract information from the text. Reply with JSON only, with exactly three keys: name (the full name "
             "of the person named in the text, or null), place (the place named in the text, or null) and year (a "
             "year written in the text, as an integer, or null).")
EXTRACT_15 = [   # 8 short passages of Holmes and what they contain (written for the workbook)
    {"text": "One night—it was on the twentieth of March, 1888—I was returning from a journey to a patient (for I had "
             "now returned to civil practice), when my way led me through Baker Street.",
     "gold": {"name": None, "place": "Baker Street", "year": 1888}},
    {"text": "“Mr. Victor Hatherley, hydraulic engineer, 16A, Victoria Street (3rd floor).” That was the name, style, "
             "and abode of my morning visitor.",
     "gold": {"name": "Victor Hatherley", "place": "Victoria Street", "year": None}},
    {"text": "Mr. Henry Baker can have the same by applying at 6:30 this evening at 221B, Baker Street.",
     "gold": {"name": "Henry Baker", "place": "Baker Street", "year": None}},
    {"text": "Then he sealed it and addressed it to “Captain James Calhoun, Barque Lone Star, Savannah, Georgia.”",
     "gold": {"name": "James Calhoun", "place": "Savannah", "year": None}},
    {"text": "On account of the bequest of the late Ezekiah Hopkins, of Lebanon, Pennsylvania, U.S.A., there is now "
             "another vacancy open which entitles a member of the League to a salary of £ 4 a week.",
     "gold": {"name": "Ezekiah Hopkins", "place": "Lebanon", "year": None}},
    {"text": "“THE RED-HEADED LEAGUE IS DISSOLVED. October 9, 1890.”",
     "gold": {"name": None, "place": None, "year": 1890}},
    {"text": "“I am Alexander Holder, of the banking firm of Holder & Stevenson, of Threadneedle Street.”",
     "gold": {"name": "Alexander Holder", "place": "Threadneedle Street", "year": None}},
    {"text": "By degrees he made friends in the neighbourhood, and in 1887 he married the daughter of a local brewer, "
             "by whom he now has two children.",
     "gold": {"name": None, "place": None, "year": 1887}},
]
EXAMPLES_15 = [  # (text, answer): the examples of a few-shot prompt, other passages of Holmes
    ("“I have been a governess for five years,” said she, “in the family of Colonel Spence Munro, but two months ago "
     "the colonel received an appointment at Halifax, in Nova Scotia.”",
     {"name": "Spence Munro", "place": "Halifax", "year": None}),
    ("One day—it was in March, 1883—a letter with a foreign stamp lay upon the table in front of the colonel’s plate.",
     {"name": None, "place": None, "year": 1883}),
    ("“He travels for Westhouse & Marbank, the great claret importers of Fenchurch Street.”",
     {"name": None, "place": "Fenchurch Street", "year": None}),
]
KEYS_15 = ("name", "place", "year")


def field_ok_15(pred, gold):
    """A field is right when null is predicted for null, the same integer for a year, and a text that contains the
    expected words for a name or a place (« Lebanon, Pennsylvania » for « Lebanon »)."""
    if gold is None or isinstance(gold, int):
        return pred == gold and type(pred) is type(gold)
    return isinstance(pred, str) and wbrag.answer_in(pred, {"answer": gold, "aliases": []})


def score_15(predictions, golds):
    """{"valid": share of valid JSON replies, "name", "place", "year": accuracy of each field, "all": mean of the
    three}; an invalid reply (None) counts wrong for its three fields."""
    out = {"valid": float(np.mean([p is not None for p in predictions]))}
    for key in KEYS_15:
        out[key] = float(np.mean([p is not None and field_ok_15(p[key], g[key]) for p, g in zip(predictions, golds)]))
    out["all"] = float(np.mean([out[key] for key in KEYS_15]))
    return out'''

TODO_15 = r'''def parse_json_15(reply):
    """The dict {"name", "place", "year"} that the reply holds, or None: the reply stripped, without a Markdown fence
    (```json … ``` or ``` … ```), read by json.loads; None if it is not a JSON object with exactly the three keys of
    KEYS_15, or if a value has the wrong type (name and place: str or None; year: int, not bool, or None)."""
    raise NotImplementedError("parse_json_15")


def messages_15(text, examples=(), system=SYSTEM_15):
    """The chat messages of an extraction: the system message, then for each (text, answer) of `examples` a user
    message (the text) and an assistant message (json.dumps(answer)), then the user message `text`."""
    raise NotImplementedError("messages_15")


def vote_15(replies):
    """The majority vote: the most frequent dict among the replies that parse_json_15 accepts (two dicts are the same
    when their three values are equal), the one seen first in case of a tie; None if no reply is valid."""
    raise NotImplementedError("vote_15")'''

SOLUTION_15 = r'''def parse_json_15(reply):
    """The extracted dict, or None (fences removed; exactly the three keys; types checked)."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != set(KEYS_15):
        return None
    if not all(data[k] is None or isinstance(data[k], str) for k in ("name", "place")):
        return None
    year = data["year"]
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        return None
    return {k: data[k] for k in KEYS_15}


def messages_15(text, examples=(), system=SYSTEM_15):
    """System message, the examples as user/assistant turns, then the text."""
    messages = [{"role": "system", "content": system}]
    for example_text, answer in examples:
        messages += [{"role": "user", "content": example_text}, {"role": "assistant", "content": json.dumps(answer)}]
    return messages + [{"role": "user", "content": text}]


def vote_15(replies):
    """The most frequent valid answer (the first seen on a tie), or None."""
    parsed = [p for p in (parse_json_15(r) for r in replies) if p is not None]
    if not parsed:
        return None
    counts = Counter(tuple(p[k] for k in KEYS_15) for p in parsed)   # Counter keeps the order of first sight
    best = counts.most_common(1)[0][0]
    return dict(zip(KEYS_15, best))'''

CHECK_15 = r'''CASES_15 = [   # (reply, expected): replies that a small model gives
    ('{"name": "Irene Adler", "place": "London", "year": 1858}', {"name": "Irene Adler", "place": "London", "year": 1858}),
    ('```json\n{"name": null, "place": "Baker Street", "year": null}\n```', {"name": None, "place": "Baker Street", "year": None}),
    ('```\n{"name": "Holmes", "place": null, "year": 1890}\n```', {"name": "Holmes", "place": None, "year": 1890}),
    ('  {"year": 1888, "place": null, "name": null}  ', {"name": None, "place": None, "year": 1888}),
    ('Here is the JSON:\n{"name": "Watson", "place": null, "year": null}', None),
    ('{"name": "Watson", "place": null}', None),
    ('{"name": "Watson", "place": null, "year": null, "age": 35}', None),
    ('{"name": "Watson", "place": null, "year": "1888"}', None),
    ('{"name": "Watson", "place": null, "year": true}', None),
    ('{"name": ["Watson"], "place": null, "year": null}', None),
    ('[{"name": "Watson", "place": null, "year": null}]', None),
    ('{"name": "Watson", "place": null, "year": null', None),
    ('', None),
]
with checked("B4.15"):
    for reply_15, want_15 in CASES_15:
        got_15 = parse_json_15(reply_15)
        if got_15 != want_15 or (want_15 is not None and set(got_15) != set(KEYS_15)):
            print(f"❌ Ex B4.15 : parse_json_15({reply_15!r}) devrait rendre {want_15!r} ; reçu {got_15!r}.")
            break
    else:
        print(f"✅ Ex B4.15 : parse_json_15 juge comme attendu les {len(CASES_15)} réponses d'essai (balises "
              "Markdown, texte autour, clé en trop ou manquante, types).")
with checked("B4.15"):
    msgs_15 = messages_15("Some text.", EXAMPLES_15[:2])
    if returned("B4.15", "messages_15", msgs_15):
        roles_15 = [m.get("role") for m in msgs_15] if isinstance(msgs_15, list) else None
        verdict("B4.15", roles_15 == ["system", "user", "assistant", "user", "assistant", "user"]
                and msgs_15[0]["content"] == SYSTEM_15 and msgs_15[-1]["content"] == "Some text."
                and msgs_15[1]["content"] == EXAMPLES_15[0][0]
                and json.loads(msgs_15[2]["content"]) == EXAMPLES_15[0][1],
                "messages_15 : système, deux exemples (utilisateur puis assistant en JSON), puis le texte.",
                f"messages_15 avec deux exemples : rôles attendus system, user, assistant, user, assistant, user, le "
                f"texte en dernier et chaque réponse d'exemple en JSON ; reçu les rôles {roles_15}.")
        zero_15 = messages_15("Some text.")
        verdict("B4.15", [m["role"] for m in zero_15] == ["system", "user"],
                "messages_15 sans exemple : le zéro-shot, système puis texte.",
                f"sans exemple, attendu deux messages (system, user) ; reçu {[m.get('role') for m in zero_15]}.")
        other_15 = messages_15("Some text.", system="Be brief.")
        verdict("B4.15", other_15[0]["content"] == "Be brief.", "le message système se change par `system`.",
                "messages_15 doit prendre son message système dans l'argument `system` (B4.16 s'en sert).")
with checked("B4.15"):
    a_15 = '{"name": "Watson", "place": null, "year": null}'
    b_15 = '{"name": "Holmes", "place": null, "year": null}'
    c_15 = '```json\n{"name": "Holmes", "place": null, "year": null}\n```'
    replies_v15 = [a_15, "nonsense", b_15, c_15, a_15, "{}"]
    copy_v15 = list(replies_v15)
    for replies, want in (([a_15, "nonsense", b_15, c_15], {"name": "Holmes", "place": None, "year": None}),
                          ([a_15, b_15], {"name": "Watson", "place": None, "year": None}),
                          (["nonsense", "{}"], None), ([], None)):
        got = vote_15(replies)
        verdict("B4.15", got == want, f"vote_15 sur {len(replies)} réponses : {got}",
                f"vote_15 sur {replies!r} devrait rendre {want!r} (la plus fréquente parmi les réponses valides, la "
                f"première vue en cas d'égalité, None si aucune) ; reçu {got!r}.")
    vote_15(replies_v15)
    verdict("B4.15", replies_v15 == copy_v15, "la liste reçue n'a pas changé.", "vote_15 a modifié la liste reçue.")'''

EXPERIMENT_15 = r'''STRATEGIES_15 = ("zéro-shot", "few-shot (2 exemples)", "vote (5 tirages few-shot à T = 0,7)")
results_15 = None


def shown_15(reply):
    """A reply, or the beginning and the end of a long one (40 tokens at most: it may stop in the middle)."""
    return repr(reply) if len(reply) <= 100 else f"{reply[:60]!r} … {reply[-35:]!r}"


if LLM is None:
    print("⏳ Ex B4.15 : SmolLM2 n'a pas pu être chargé : l'expérience est « à valider sur Colab ».")
elif all(written(probe) for probe in (lambda: parse_json_15("{}"), lambda: messages_15("x"), lambda: vote_15([]))):
    with checked("B4.15"):
        golds_15 = [item["gold"] for item in EXTRACT_15]
        predictions_15 = {s: [] for s in STRATEGIES_15}
        for item in EXTRACT_15:
            zero = LLM.generate(messages_15(item["text"]), max_new_tokens=40, cache=CACHE)
            few_msgs = messages_15(item["text"], EXAMPLES_15[:2])
            few = LLM.generate(few_msgs, max_new_tokens=40, cache=CACHE)
            draws = [LLM.generate(few_msgs, max_new_tokens=40, do_sample=True, temperature=0.7, seed=s, cache=CACHE)
                     for s in range(5)]
            predictions_15[STRATEGIES_15[0]].append(parse_json_15(zero))
            predictions_15[STRATEGIES_15[1]].append(parse_json_15(few))
            predictions_15[STRATEGIES_15[2]].append(vote_15(draws))
            print(f"• {item['text'][:60]}…\n    zéro-shot : {shown_15(zero)}\n    few-shot  : {shown_15(few)}")
        results_15 = {s: score_15(predictions_15[s], golds_15) for s in STRATEGIES_15}
        print(f"\n{'stratégie':40} JSON valide   name   place   year   moyenne")
        for s, r in results_15.items():
            print(f"{s:40} {fr(r['valid'], 2):>11} {fr(r['name'], 2):>6} {fr(r['place'], 2):>7} "
                  f"{fr(r['year'], 2):>6} {fr(r['all'], 2):>9}")
else:
    print("⏳ Ex B4.15 : l'expérience attend tes trois fonctions.")'''

# --- 🔬 B4.16 ---------------------------------------------------------------------------------------------------------
GIVEN_16 = r'''FORMULATIONS_16 = [   # five system messages, the same instruction worded differently
    SYSTEM_15,
    "Read the text and return a JSON object with the keys name, place and year. Use null when the text does not "
    "give a value. The year is an integer. Output only the JSON.",
    "You are an information extraction system. Output format: {\"name\": string or null, \"place\": string or "
    "null, \"year\": integer or null}. No explanation.",
    "TASK: find the person, the place and the year mentioned in the text.\nFORMAT: one JSON object, keys name, "
    "place, year; null for a missing value.",
    "extract name, place and year from the text as json (null if absent, year as a number). only json.",
]
ORDERS_16 = [(0, 1, 2), (2, 0, 1), (1, 2, 0)]    # three orders of the three few-shot examples of EXAMPLES_15
SEEDS_16 = (0, 1, 2)
SNIPPETS_16 = EXTRACT_15[:6] if FAST_MODE else EXTRACT_15
MODEL_16 = LLM if FAST_MODE else wbllm.try_llm("qwen2.5", "B4.16")   # full mode: Qwen2.5-0.5B-Instruct (≈ 10 min)'''

TODO_16 = r'''def accuracy_16(system, examples, seed):
    """The mean field accuracy (score_15(...)["all"]) of the replies of MODEL_16 to the passages of SNIPPETS_16,
    with this system message and these few-shot examples (messages_15): one random draw per passage, at a
    temperature of 0.7, at most 40 new tokens, with this seed, through the cache (MODEL_16.generate(...,
    cache=CACHE)), each reply read by parse_json_15."""
    raise NotImplementedError("accuracy_16")


def spread_16(table):
    """table: (n_variants, n_seeds) accuracies, one row per prompt variant. (mean, between, within): the mean of the
    table, the standard deviation (ddof=0) of the row means, and the mean over the rows of their standard
    deviation (ddof=0) across the seeds."""
    raise NotImplementedError("spread_16")'''

SOLUTION_16 = r'''def accuracy_16(system, examples, seed):
    """Mean field accuracy over SNIPPETS_16, one draw at T = 0.7 per passage."""
    predictions = []
    for item in SNIPPETS_16:
        reply = MODEL_16.generate(messages_15(item["text"], examples, system), max_new_tokens=40, do_sample=True,
                                  temperature=0.7, seed=seed, cache=CACHE)
        predictions.append(parse_json_15(reply))
    return score_15(predictions, [item["gold"] for item in SNIPPETS_16])["all"]


def spread_16(table):
    """(mean, std of the row means, mean of the row stds), ddof = 0."""
    table = np.asarray(table, dtype=float)
    return float(table.mean()), float(table.mean(axis=1).std()), float(table.std(axis=1).mean())'''

CHECK_16 = r'''with checked("B4.16"):
    toy_16 = [[0.5, 0.7, 0.6], [0.2, 0.2, 0.2]]
    got_16 = spread_16(toy_16)
    if returned("B4.16", "spread_16", got_16):
        same("B4.16", "spread_16 sur un petit tableau (moyenne, entre les lignes, dans les lignes)", got_16,
             [0.4, 0.2, float(np.std([0.5, 0.7, 0.6])) / 2], rtol=1e-9, atol=1e-12)


class FakeModel16:
    """Stands for MODEL_16 during the check: it records the calls and answers a fixed JSON."""

    def __init__(self):
        self.calls = []

    def generate(self, messages, **settings):
        self.calls.append((messages, settings))
        return '{"name": "Henry Baker", "place": "Baker Street", "year": null}'


with checked("B4.16"):
    saved_16, fake_16 = MODEL_16, FakeModel16()
    try:
        MODEL_16 = fake_16
        value_16 = accuracy_16("Be brief.", EXAMPLES_15[:1], 5)
    finally:
        MODEL_16 = saved_16
    want_16 = score_15([{"name": "Henry Baker", "place": "Baker Street", "year": None}] * len(SNIPPETS_16),
                       [item["gold"] for item in SNIPPETS_16])["all"]
    calls_ok_16 = len(fake_16.calls) == len(SNIPPETS_16)
    verdict("B4.16", calls_ok_16, f"une réponse demandée par passage ({len(SNIPPETS_16)}).",
            f"accuracy_16 doit appeler MODEL_16.generate une fois par passage de SNIPPETS_16 ({len(SNIPPETS_16)}) ; "
            f"vu {len(fake_16.calls)} appels.")
    if calls_ok_16:
        msgs_16, settings_16 = fake_16.calls[0]
        verdict("B4.16", msgs_16 == messages_15(SNIPPETS_16[0]["text"], EXAMPLES_15[:1], "Be brief."),
                "les messages viennent de messages_15 avec le système et les exemples reçus.",
                "les messages de chaque appel doivent être messages_15(texte, examples, system).")
        verdict("B4.16", settings_16.get("do_sample") is True and settings_16.get("temperature") == 0.7
                and settings_16.get("seed") == 5 and settings_16.get("max_new_tokens") == 40
                and settings_16.get("cache") is CACHE,
                "un tirage à T = 0,7 avec la graine reçue, 40 tokens au plus, par le cache.",
                f"chaque appel doit passer do_sample=True, temperature=0.7, seed=seed, max_new_tokens=40 et "
                f"cache=CACHE ; vu {sorted((k, v) for k, v in settings_16.items() if k != 'cache')}.")
        same("B4.16", "la précision moyenne rendue (score_15(...)[\"all\"])", value_16, want_16, rtol=1e-12, atol=1e-12)'''

EXPERIMENT_16 = r'''grid_f16 = grid_o16 = None
if MODEL_16 is None:
    print("⏳ Ex B4.16 : le modèle n'a pas pu être chargé : l'expérience est « à valider sur Colab ».")
elif written(lambda: spread_16([[0.0]])) and written(lambda: accuracy_16(FORMULATIONS_16[0], EXAMPLES_15, SEEDS_16[0])):
    with checked("B4.16"):                        # the probe is the first case of the grid: read again from the cache
        start_16 = time.perf_counter()
        grid_f16 = np.array([[accuracy_16(f, EXAMPLES_15, s) for s in SEEDS_16] for f in FORMULATIONS_16])
        mean_16, between_16, within_16 = spread_16(grid_f16)
        print(f"formulations : {len(SNIPPETS_16)} passages, {len(SEEDS_16)} graines, "
              f"{fr(time.perf_counter() - start_16, 0)} s (une mesure)\n")
        print(f"formulation : moyenne {fr(mean_16, 3)}, écart-type entre les {len(grid_f16)} variantes "
              f"{fr(between_16, 3)}, écart-type moyen dû aux graines {fr(within_16, 3)}")
        for label, row in zip([f"F{i}" for i in range(len(FORMULATIONS_16))], grid_f16):
            print(f"  {label:10} " + "  ".join(fr(v, 3) for v in row) + f"   moyenne {fr(row.mean(), 3)}")
else:
    print("⏳ Ex B4.16 : l'expérience attend accuracy_16 et spread_16 (et les fonctions de B4.15).")'''

EXPERIMENT_16_ORDERS = r'''if globals().get("grid_f16") is None:
    print("⏳ Ex B4.16 : les ordres des exemples attendent le tableau des formulations (cellule précédente).")
else:
    with checked("B4.16"):
        start_16 = time.perf_counter()
        grid_o16 = np.array([[accuracy_16(SYSTEM_15, [EXAMPLES_15[i] for i in order], s) for s in SEEDS_16]
                             for order in ORDERS_16])
        mean_16, between_16, within_16 = spread_16(grid_o16)
        print(f"ordres des exemples : {fr(time.perf_counter() - start_16, 0)} s (une mesure)\n")
        print(f"ordre des exemples : moyenne {fr(mean_16, 3)}, écart-type entre les {len(grid_o16)} variantes "
              f"{fr(between_16, 3)}, écart-type moyen dû aux graines {fr(within_16, 3)}")
        for label, row in zip([str(o) for o in ORDERS_16], grid_o16):
            print(f"  {label:10} " + "  ".join(fr(v, 3) for v in row) + f"   moyenne {fr(row.mean(), 3)}")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
        for ax, grid, labels, title in ((axes[0], grid_f16, [f"F{i}" for i in range(len(FORMULATIONS_16))], "formulation"),
                                        (axes[1], grid_o16, [str(o) for o in ORDERS_16], "ordre des exemples")):
            ax.bar(labels, grid.mean(axis=1), yerr=grid.std(axis=1), capsize=4, color="tab:blue", alpha=0.7)
            for k, row in enumerate(grid):
                ax.plot([k] * len(row), row, "k.", ms=6)
            ax.set_title(f"{title} (points : les graines)")
            ax.set_ylim(0, 1.02)
        axes[0].set_ylabel("précision moyenne des champs")
        plt.show()
        if FAST_MODE and FULL_16:
            print(f"Protocole complet (précalculé par tools/chapters/history_chB4.py) : {FULL_16['model']}, "
                  f"{FULL_16['snippets']} passages, les mêmes formulations, ordres et graines :")
            for title, key in (("formulation", "grid_f"), ("ordre des exemples", "grid_o")):
                grid = np.array(FULL_16[key])
                print(f"  {title} : moyenne {fr(grid.mean(), 3)}, écart-type entre les variantes "
                      f"{fr(grid.mean(axis=1).std(), 3)}, écart-type moyen dû aux graines {fr(grid.std(axis=1).mean(), 3)}")'''

# --- 🔨 B4.17 and B4.18 -----------------------------------------------------------------------------------------------
TOOLS_B = r'''# Tools for the notebook exercises of part B (retrieval and RAG)
TEST_RET = "tests/test_chb4_retrieval.py"         # the tests of your library retrieval.py
QA = wbrag.load_qa("holmes")                      # the 30 annotated questions on Holmes
QA_FR = wbrag.load_qa("verne")                    # the 10 questions in French on Verne (chapters I to VII)
DOCS = wbrag.documents("holmes")                  # the 12 stories: [(title, text)]
DOCS_FR = wbrag.documents("verne")
REF_CHUNKS, REF_DOC = wbrag.reference_chunks("holmes")   # the reference chunking (120 words, overlap 20)
tokenize = wbrag.tokenize                         # the tokens of BM25: the same for the corpus and the queries
print(f"{len(QA)} questions sur Holmes ({sum(q['paraphrased'] for q in QA)} paraphrasées), {len(QA_FR)} sur Verne ; "
      f"Holmes : {len(DOCS)} nouvelles, {len(REF_CHUNKS)} chunks de référence de 120 mots.")
print("Exemple :", {k: QA[0][k] for k in ("id", "question", "answer", "evidence", "story_title")})'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="B4")   # reload your saved file\n'

CHECK_17 = r'''with checked("B4.17"):
    R = mylearn.retrieval
    sentence_17 = "Le docteur Watson prend son chapeau et suit Holmes dans la rue sombre"
    copy_17 = str(sentence_17)
    chunks_17 = R.chunk_text(sentence_17, chunk_size=5, overlap=2)
    want_17 = ["Le docteur Watson prend son", "prend son chapeau et suit", "et suit Holmes dans la", "dans la rue sombre"]
    verdict("B4.17", chunks_17 == want_17, "chunk_text : l'exemple de la fiche (5 mots, recouvrement de 2).",
            f"chunk_text(phrase, 5, 2) devrait rendre {want_17} ; reçu {chunks_17!r}.")
    verdict("B4.17", R.chunk_text("un  deux\ntrois", chunk_size=3, overlap=0) == ["un deux trois"],
            "un texte d'exactement chunk_size mots donne un seul chunk, les blancs réduits à une espace.",
            f"chunk_text('un  deux\\ntrois', 3, 0) devrait rendre ['un deux trois'] ; reçu "
            f"{R.chunk_text('un  deux' + chr(10) + 'trois', chunk_size=3, overlap=0)!r}.")
    chars_17 = R.chunk_text("ab\ncd ef", chunk_size=3, overlap=1, unit="char")
    verdict("B4.17", chars_17 == ["ab\n", "\ncd", "d e", "ef"],
            "unité 'char' : chaque caractère compte, espaces et retours à la ligne compris, sans rien retirer.",
            f"chunk_text('ab\\ncd ef', 3, 1, unit='char') devrait rendre ['ab\\n', '\\ncd', 'd e', 'ef'] (aucun "
            f"blanc retiré) ; reçu {chars_17!r}.")
    refuses("B4.17", "chunk_text(texte, chunk_size=0)", lambda: R.chunk_text("a b", chunk_size=0, overlap=0))

    docs_17 = ["holmes fume sa pipe", "watson lit le journal et holmes joue du violon", "le violon de holmes",
               "watson écrit"]
    corpus_17 = [d.split() for d in docs_17]
    bm_17 = R.BM25(k1=1.5, b=0.75).fit(corpus_17)
    if bm_17 is None:
        print("❌ Ex B4.17 : BM25.fit renvoie None : fit doit rendre self.")
        raise StopCheck
    want_bm_17 = wbrag.BM25Index(corpus_17).scores(["holmes", "violon"])
    same("B4.17", "BM25 sur les quatre documents de la fiche, requête « holmes violon »",
         bm_17.get_scores(["holmes", "violon"]), want_bm_17, rtol=1e-6, atol=1e-6)
    zero_17 = R.BM25(k1=0.0, b=1.0).fit([["a", "b"], [], ["b"]]).get_scores(["a", "c"])
    verdict("B4.17", np.all(np.isfinite(zero_17)) and np.asarray(zero_17)[1:].tolist() == [0.0, 0.0],
            "k1 = 0 et un document vide : des 0, jamais NaN.",
            f"avec k1 = 0, b = 1 et un document vide, les documents sans le terme doivent recevoir 0 (jamais NaN) ; "
            f"reçu {np.asarray(zero_17).tolist()}.")
    refuses("B4.17", "get_scores avant fit", lambda: R.BM25().get_scores(["a"]))

    # the real use: Holmes cut into chunks of 120 words, and the passages that BM25 ranks first
    holmes_17 = [c for _, body in DOCS for c in R.chunk_text(body, chunk_size=120, overlap=20)]
    verdict("B4.17", holmes_17 == REF_CHUNKS, f"Holmes : {len(holmes_17)} chunks, ceux de la référence.",
            f"sur les 12 nouvelles, chunk_text(texte, 120, 20) devrait donner {len(REF_CHUNKS)} chunks, ceux de "
            f"wbrag.reference_chunks ; reçu {len(holmes_17)} chunks.")
    index_17 = R.BM25().fit([tokenize(c) for c in holmes_17])
    scores_17 = np.asarray(index_17.get_scores(tokenize("violin music concert")))
    for rank_17, i_17 in enumerate(np.argsort(-scores_17, kind="stable")[:3], start=1):
        print(f"  {rank_17}. score {fr(scores_17[i_17], 2)} · {DOCS[REF_DOC[i_17]][0]} · {holmes_17[i_17][:110]}…")'''

CHECK_18 = r'''with checked("B4.18"):
    R = mylearn.retrieval
    D_18 = np.array([[1., 3.], [3., 1.], [1., 0.], [-1., -1.]])
    out_18 = R.cosine_top_k([1., 1.], D_18, k=3)
    idx_18, sims_18 = pair("B4.18", "cosine_top_k", out_18, names=("indices", "similarities"))
    verdict("B4.18", np.asarray(idx_18).tolist() == [0, 1, 2], "cosine_top_k : l'exemple de la fiche, ex aequo au plus "
            "petit indice.", f"cosine_top_k([1, 1], D, k=3) devrait rendre les indices [0, 1, 2] ; reçu {np.asarray(idx_18).tolist()}.")
    same("B4.18", "similarités de l'exemple de la fiche", sims_18, [2 / np.sqrt(5)] * 2 + [1 / np.sqrt(2)], rtol=1e-6, atol=1e-6)
    _, zero_18 = R.cosine_top_k([1., 1.], np.array([[0., 0.], [1., 0.]]), k=2)
    verdict("B4.18", np.all(np.isfinite(zero_18)) and 0.0 in np.round(np.asarray(zero_18), 12),
            "un vecteur nul a une similarité de 0 (pas de NaN).",
            f"un document nul doit recevoir une similarité de 0, jamais NaN ; reçu {np.asarray(zero_18).tolist()}.")
    refuses("B4.18", "cosine_top_k avec k > n", lambda: R.cosine_top_k([1., 1.], D_18, k=5))
    fused_18 = R.reciprocal_rank_fusion([[7, 2, 9, 4], [2, 5, 7, 1]], k=60)
    verdict("B4.18", np.asarray(fused_18).tolist() == [2, 7, 5, 9, 1, 4],
            "reciprocal_rank_fusion : l'exemple de la fiche, [2, 7, 5, 9, 1, 4].",
            f"la fusion de [7, 2, 9, 4] et [2, 5, 7, 1] (k = 60) devrait rendre [2, 7, 5, 9, 1, 4] ; reçu "
            f"{np.asarray(fused_18).tolist()}.")
    fused1_18 = R.reciprocal_rank_fusion([[3, 1, 2], [1, 2, 4]], k=1)        # with a small k, the ranks show
    verdict("B4.18", np.asarray(fused1_18).tolist() == [1, 2, 3, 4],
            "reciprocal_rank_fusion avec k = 1 : [1, 2, 3, 4] (rangs à partir de 1).",
            f"la fusion de [3, 1, 2] et [1, 2, 4] avec k = 1 devrait rendre [1, 2, 3, 4] ; reçu "
            f"{np.asarray(fused1_18).tolist()} : relis la formule (les rangs commencent à 1).")
    refuses("B4.18", "reciprocal_rank_fusion avec un identifiant répété dans un classement",
            lambda: R.reciprocal_rank_fusion([[1, 2, 1]]))
    empty_18 =np.asarray(R.reciprocal_rank_fusion([[]]))
    verdict("B4.18", empty_18.shape == (0,) and np.issubdtype(empty_18.dtype, np.integer),
            "une fusion vide rend un tableau d'entiers de forme (0,).",
            f"reciprocal_rank_fusion([[]]) doit rendre un tableau d'entiers vide ; reçu {empty_18!r}.")
    retrieved_18, relevant_18 = [[4, 8, 1, 3], [2, 6, 5, 0], [7, 3, 0, 5]], [{1}, {2, 9}, {6}]
    same("B4.18", "recall@1, recall@3 et MRR de l'exemple de la fiche",
         [R.recall_at_k(retrieved_18, relevant_18, 1), R.recall_at_k(retrieved_18, relevant_18, 3),
          R.mean_reciprocal_rank(retrieved_18, relevant_18)], [1 / 6, 0.5, 4 / 9], rtol=1e-9, atol=1e-12)
    rows_18, _ = R.cosine_top_k(np.eye(3)[:2], np.eye(3), k=2)              # a 2-D array of indices, as in B4.20
    same("B4.18", "recall@1 sur le tableau 2-D rendu par cosine_top_k", R.recall_at_k(rows_18, [{0}, {2}], 1), 0.5,
         rtol=1e-9, atol=1e-12)
    same("B4.18", "MRR avec un ensemble pertinent vide (la requête compte 0)",
         R.mean_reciprocal_rank([[1, 2], [3]], [{2}, set()]), 0.25, rtol=1e-9, atol=1e-12)'''

# --- 🔮 B4.19 ---------------------------------------------------------------------------------------------------------
GIVEN_19 = r'''SM_19 = wb.text.try_sentence_model("minilm", "B4.19")      # all-MiniLM-L6-v2 (B2), frozen; None if unavailable
PARA_19 = [q for q in QA if q["paraphrased"]]                # the 10 paraphrased questions
for q in PARA_19[:3]:
    print(f"{q['id']} · {q['question']}  →  {q['story_title']}")'''

TODO_19 = r'''bm25_half_19 = ...     # a) BM25 finds the relevant chunk of at least 5 of the 10 paraphrased questions in its first 5
                       #    results? True or False
dense_half_19 = ...    # b) the dense search (MiniLM) finds at least 5 of the 10 in its first 5 results? True or False
hybrid_best_19 = ...   # c) the hybrid search (RRF of the first 20 of both) finds at least as many as the better of
                       #    the two? True or False'''

EXPERIMENT_19 = r'''found_19 = None
if SM_19 is None:
    print("⏳ Ex B4.19 : MiniLM n'a pas pu être chargé : l'expérience est « à valider sur Colab ».")
else:
    with checked("B4.19"):
        R = mylearn.retrieval
        chunks_19 = [c for _, body in DOCS for c in R.chunk_text(body, chunk_size=120, overlap=20)]
        if chunks_19 != REF_CHUNKS:
            print("❌ Ex B4.19 : ton chunk_text ne rend pas le découpage de référence (B4.17) : l'expérience attend.")
            raise StopCheck
        bm25_19 = R.BM25().fit([tokenize(c) for c in chunks_19])
        start_19 = time.perf_counter()
        emb_19 = wb.text.passage_embeddings(SM_19, chunks_19, "b4-holmes-120-20")   # ≈ 40 s the first time, then cached
        print(f"embeddings des {len(chunks_19)} chunks : {fr(time.perf_counter() - start_19, 1)} s")
        found_19 = {"BM25": 0, "dense (MiniLM)": 0, "hybride (RRF)": 0}
        print(f"\n{'question':58} rang du chunk pertinent : BM25 · dense · hybride (« — » : au-delà de 20)")
        for item in PARA_19:
            relevant = wbrag.relevant_ids(chunks_19, item)
            bm = np.argsort(-np.asarray(bm25_19.get_scores(tokenize(item["question"]))), kind="stable")[:20]
            dense, _ = R.cosine_top_k(SM_19.embed([item["question"]], kind="query")[0], emb_19, k=20)
            hybrid = np.asarray(R.reciprocal_rank_fusion([bm, dense], k=60))[:20]
            ranks = []
            for name, ranking in zip(found_19, (bm, dense, hybrid)):
                rank = next((r for r, i in enumerate(np.asarray(ranking).tolist(), start=1) if i in relevant), None)
                found_19[name] += rank is not None and rank <= 5
                ranks.append("—" if rank is None else str(rank))
            print(f"{item['question'][:58]:58} {' · '.join(ranks)}")
        print("\nRetrouvées dans les 5 premiers résultats, sur 10 :", found_19)
if found_19 is not None:
    wb.check("B4.19a", bm25_half_19)
    wb.check("B4.19b", dense_half_19)
    wb.check("B4.19c", hybrid_best_19)
else:
    print("⏳ Ex B4.19 : tes prédictions seront vérifiées quand l'expérience aura tourné (B4.17 et B4.18, puis MiniLM).")'''

# --- 📦 B4.20 ---------------------------------------------------------------------------------------------------------
GIVEN_20 = r'''DEPTH_20 = 20     # the hybrid search fuses the first 20 results of BM25 and of the dense search (RRF, k = 60)
SM_E5 = wb.text.try_sentence_model("e5", "B4.20")      # multilingual-e5-small (B2): « query: » and « passage: »'''

TODO_20 = r'''def build_index_20(chunks, sentence_model, subset):
    """The index of a corpus: {"chunks": chunks, "bm25": your BM25 fitted on [tokenize(c) for c in chunks],
    "model": sentence_model, "embeddings": wb.text.passage_embeddings(sentence_model, chunks, subset)} (the
    passage embeddings, prefix « passage: » for e5, computed once then cached under the name `subset`)."""
    raise NotImplementedError("build_index_20")


def search_20(index, query, method="hybrid", k=5):
    """The ids of the k best chunks for `query`, best first, as a list of ints: "bm25" (your BM25 on
    tokenize(query)), "dense" (your cosine_top_k of index["model"].embed([query], kind="query")[0] against the
    embeddings), "hybrid" (your reciprocal_rank_fusion, k = 60, of the first DEPTH_20 results of the two, cut to k).
    ValueError for another method."""
    raise NotImplementedError("search_20")


def evaluate_20(index, items, method):
    """{"recall@1", "recall@5", "mrr"} of `method` on the questions `items`: for each question, the first DEPTH_20
    results of search_20 and its relevant chunks (wbrag.relevant_ids(index["chunks"], item)); your recall_at_k and
    mean_reciprocal_rank, as Python floats."""
    raise NotImplementedError("evaluate_20")'''

SOLUTION_20 = r'''def build_index_20(chunks, sentence_model, subset):
    """BM25 and the dense embeddings of the chunks."""
    R = mylearn.retrieval
    return {"chunks": list(chunks), "bm25": R.BM25().fit([tokenize(c) for c in chunks]), "model": sentence_model,
            "embeddings": wb.text.passage_embeddings(sentence_model, list(chunks), subset)}


def search_20(index, query, method="hybrid", k=5):
    """The ids of the k best chunks: BM25, dense, or their RRF fusion."""
    R = mylearn.retrieval
    if method not in ("bm25", "dense", "hybrid"):
        raise ValueError(f"unknown method {method!r}")
    depth = max(k, DEPTH_20)
    if method in ("bm25", "hybrid"):
        scores = np.asarray(index["bm25"].get_scores(tokenize(query)))
        bm = np.argsort(-scores, kind="stable")[:depth]
    if method in ("dense", "hybrid"):
        q = index["model"].embed([query], kind="query")[0]
        dense, _ = R.cosine_top_k(q, index["embeddings"], k=min(depth, len(index["chunks"])))
    if method == "bm25":
        return bm[:k].tolist()
    if method == "dense":
        return np.asarray(dense)[:k].tolist()
    return np.asarray(R.reciprocal_rank_fusion([bm[:DEPTH_20], dense[:DEPTH_20]], k=60))[:k].tolist()


def evaluate_20(index, items, method):
    """recall@1, recall@5 and MRR over the first DEPTH_20 results."""
    R = mylearn.retrieval
    retrieved = [search_20(index, item["question"], method, DEPTH_20) for item in items]
    relevant = [wbrag.relevant_ids(index["chunks"], item) for item in items]
    return {"recall@1": float(R.recall_at_k(retrieved, relevant, 1)),
            "recall@5": float(R.recall_at_k(retrieved, relevant, 5)),
            "mrr": float(R.mean_reciprocal_rank(retrieved, relevant))}'''

CHECK_20 = r'''INDEX_20 = None
if SM_19 is None:
    print("⏳ Ex B4.20 : MiniLM n'a pas pu être chargé : l'exercice est « à valider sur Colab ».")
else:
    with checked("B4.20"):
        INDEX_20 = build_index_20(REF_CHUNKS, SM_19, "b4-holmes-120-20")
        if returned("B4.20", "build_index_20", INDEX_20):
            if not (isinstance(INDEX_20, dict) and {"chunks", "bm25", "model", "embeddings"} <= set(INDEX_20)):
                print("❌ Ex B4.20 : build_index_20 doit rendre un dictionnaire aux clés chunks, bm25, model, embeddings.")
                INDEX_20 = None
                raise StopCheck
            emb_20 = np.asarray(INDEX_20["embeddings"])
            verdict("B4.20", emb_20.shape == (len(REF_CHUNKS), 384), f"embeddings {emb_20.shape}.",
                    f"les embeddings doivent avoir la forme ({len(REF_CHUNKS)}, 384) ; reçu {emb_20.shape}.")
        else:
            INDEX_20 = None
    if INDEX_20 is not None:
        ref_20 = wbrag.ReferenceIndex(REF_CHUNKS, SM_19, subset="b4-holmes-120-20")
        with checked("B4.20"):
            for q_20 in (QA[1], QA[22]):
                for method_20 in ("bm25", "dense", "hybrid"):
                    got_20, want_20 = search_20(INDEX_20, q_20["question"], method_20, 5), ref_20.search(q_20["question"], method_20, 5)
                    verdict("B4.20", list(np.asarray(got_20).tolist()) == want_20,
                            f"search_20 ({method_20}, {q_20['id']}) : {want_20}",
                            f"search_20(index, {q_20['question']!r}, {method_20!r}, 5) devrait rendre {want_20} ; "
                            f"reçu {np.asarray(got_20).tolist()}.")
            ties_20 = search_20(INDEX_20, "xyzzy", "bm25", 5)          # an unknown word: every score is 0
            verdict("B4.20", list(np.asarray(ties_20).tolist()) == [0, 1, 2, 3, 4],
                    "BM25 : à égalité de score, le plus petit identifiant d'abord.",
                    f"search_20(index, 'xyzzy', 'bm25', 5) : tous les scores sont nuls, attendu [0, 1, 2, 3, 4] (un tri "
                    f"stable par score décroissant) ; reçu {np.asarray(ties_20).tolist()}.")
            refuses("B4.20", "search_20 avec une méthode inconnue", lambda: search_20(INDEX_20, "x", "tfidf", 5))
        if SM_E5 is not None:                    # e5 needs « query: » before a query: Verne, chapters I to VII
            with checked("B4.20"):
                chunks_v20 = [c for _, body in DOCS_FR for c in wbrag.chunk_words(body)]
                index_v20 = build_index_20(chunks_v20, SM_E5, "b4-verne-120-20")
                ref_v20 = wbrag.ReferenceIndex(chunks_v20, SM_E5, subset="b4-verne-120-20")
                got_v20 = [list(np.asarray(search_20(index_v20, q["question"], "dense", 5)).tolist()) for q in QA_FR]
                want_v20 = [ref_v20.search(q["question"], "dense", 5) for q in QA_FR]
                verdict("B4.20", got_v20 == want_v20, "e5 sur Verne : la recherche dense de référence, préfixes compris.",
                        "avec e5, ta recherche dense ne rend pas celle de la référence sur les 10 questions de Verne : "
                        "relis le contrat de search_20 (l'embedding de la requête).")
        with checked("B4.20"):
            got_e20 = evaluate_20(INDEX_20, QA[:6], "hybrid")
            ret_e20 = [ref_20.search(q["question"], "hybrid", DEPTH_20) for q in QA[:6]]
            rel_e20 = [wbrag.relevant_ids(REF_CHUNKS, q) for q in QA[:6]]
            first_e20 = [next((r for r, i in enumerate(ret, 1) if i in rel), None) for ret, rel in zip(ret_e20, rel_e20)]
            want_e20 = {"recall@1": float(np.mean([f == 1 for f in first_e20])),
                        "recall@5": float(np.mean([len(set(ret[:5]) & rel) / len(rel) for ret, rel in zip(ret_e20, rel_e20)])),
                        "mrr": float(np.mean([1 / f if f else 0.0 for f in first_e20]))}
            if returned("B4.20", "evaluate_20", got_e20):
                same("B4.20", "evaluate_20 (hybride, 6 questions) : recall@1, recall@5, MRR",
                     [got_e20.get(k) for k in ("recall@1", "recall@5", "mrr")], list(want_e20.values()), rtol=1e-9)'''

EXPERIMENT_20 = r'''table_20 = None
if INDEX_20 is not None and written(lambda: evaluate_20(INDEX_20, QA[:1], "bm25")):
    with checked("B4.20"):
        start_20 = time.perf_counter()
        groups_20 = {"30 questions": QA, "20 directes": [q for q in QA if not q["paraphrased"]],
                     "10 paraphrasées": [q for q in QA if q["paraphrased"]]}
        table_20 = {}
        for method in ("bm25", "dense", "hybrid"):
            for group, items in groups_20.items():
                table_20["Holmes · MiniLM", method, group] = evaluate_20(INDEX_20, items, method)
        if SM_E5 is not None:
            chunks_fr = [c for _, body in DOCS_FR for c in mylearn.retrieval.chunk_text(body, chunk_size=120, overlap=20)]
            for name, model in (("Verne · e5", SM_E5), ("Verne · MiniLM", SM_19)):
                index_fr = build_index_20(chunks_fr, model, "b4-verne-120-20")
                for method in ("bm25", "dense", "hybrid"):
                    table_20[name, method, "10 questions"] = evaluate_20(index_fr, QA_FR, method)
        print(f"({fr(time.perf_counter() - start_20, 1)} s)\n{'corpus · modèle':18} {'méthode':8} {'questions':16} "
              "recall@1  recall@5   MRR")
        for (corpus, method, group), r in table_20.items():
            print(f"{corpus:18} {method:8} {group:16} {fr(r['recall@1'], 3):>8} {fr(r['recall@5'], 3):>9} "
                  f"{fr(r['mrr'], 3):>6}")
        try:                                     # e5 on Holmes, precomputed by tools/chapters/history_chB4.py
            E5_HOLMES_20 = json.loads((wb.repo_root() / "tools" / "chapters" / "history_chB4.json").read_text(
                encoding="utf-8"))["e5_holmes"]["rows"]
        except (OSError, KeyError, ValueError):
            E5_HOLMES_20 = []
        print("\ne5 sur Holmes (précalculé avec la solution ; ≈ 1 min d'embeddings sur CPU ; BM25 ne dépend pas du "
              "modèle d'embedding : ses lignes sont celles de MiniLM) :")
        for method, group, r in [row for row in E5_HOLMES_20 if row[0] != "bm25"]:
            print(f"{'Holmes · e5':18} {method:8} {group:16} {fr(r['recall@1'], 3):>8} {fr(r['recall@5'], 3):>9} "
                  f"{fr(r['mrr'], 3):>6}")
        labels_20 = ["bm25", "dense", "hybrid"]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        for k, group in enumerate(groups_20):
            ax.bar(np.arange(3) + 0.27 * (k - 1), [table_20["Holmes · MiniLM", m, group]["recall@5"] for m in labels_20],
                   width=0.27, label=group)
        ax.set_xticks(range(3), ["BM25", "dense (MiniLM)", "hybride (RRF)"])
        ax.set_ylabel("recall@5")
        ax.set_ylim(0, 1)
        ax.set_title("Holmes : recall@5 par méthode")
        ax.legend()
        plt.show()
else:
    print("⏳ Ex B4.20 : le tableau attend tes trois fonctions (et ta librairie de B4.17 et B4.18).")'''

# --- 📦 B4.21 ---------------------------------------------------------------------------------------------------------
GIVEN_21 = r'''QWEN = wbllm.try_llm("qwen2.5", "B4.21")      # Qwen2.5-0.5B-Instruct (494 M parameters), weights loaded at first use
SYSTEM_21 = wbrag.RAG_SYSTEM                     # the rules: only the passages, a short answer, its citation [n]
EXAMPLE_21 = wbrag.RAG_EXAMPLE                   # one example exchange (user, assistant): the format of the reply
QUESTIONS_21 = [q for q in QA if q["id"] in wbrag.FAST_IDS] if FAST_MODE else QA
print("Le message système :", SYSTEM_21)
print("L'exemple :", EXAMPLE_21)'''

TODO_21 = r'''def rag_messages_21(question, passages):
    """The messages of the RAG: the system message SYSTEM_21, the example exchange EXAMPLE_21, then one user
    message: "Passages:\n[1] <first passage>\n[2] <second>…\n\nQuestion: <question>" (the passages numbered from 1,
    in the order received, one per line; the question last)."""
    raise NotImplementedError("rag_messages_21")


def parse_reply_21(reply):
    """(answer, cited): the reply without its citations « [n] » (spaces collapsed, spaces and full stops removed at
    both ends), and the list of the numbers cited, in their order of first appearance, each once."""
    raise NotImplementedError("parse_reply_21")


def verify_21(answer, cited, passages):
    """True when the citations hold: `cited` is not empty, each number is a passage (1 to len(passages)), and at
    least one cited passage contains every content word of the answer (wbrag.content_words(answer), not empty,
    included in set(tokenize(passage)))."""
    raise NotImplementedError("verify_21")


def rag_21(question, index, llm, k=3):
    """The RAG: search_20(index, question, "hybrid", k), the passages, rag_messages_21, a greedy reply of llm (at
    most 24 new tokens, through the cache: llm.generate(..., max_new_tokens=24, cache=CACHE)), parse_reply_21 and
    verify_21. Returns {"ids", "passages", "reply", "answer", "cited", "verified"}."""
    raise NotImplementedError("rag_21")'''

SOLUTION_21 = r'''def rag_messages_21(question, passages):
    """System rules, one example exchange, then the numbered passages and the question (last)."""
    numbered = "\n".join(f"[{n}] {p}" for n, p in enumerate(passages, start=1))
    return ([{"role": "system", "content": SYSTEM_21}] + [dict(m) for m in EXAMPLE_21]
            + [{"role": "user", "content": f"Passages:\n{numbered}\n\nQuestion: {question}"}])


def parse_reply_21(reply):
    """The answer without its citations, and the cited numbers (first appearance, each once)."""
    import re
    cited = []
    for match in re.finditer(r"\[(\d+)\]", reply):
        if int(match.group(1)) not in cited:
            cited.append(int(match.group(1)))
    answer = " ".join(re.sub(r"\[\d+\]", " ", reply).split()).strip(" .")
    return answer, cited


def verify_21(answer, cited, passages):
    """Every citation is a passage, and one cited passage holds every content word of the answer."""
    words = wbrag.content_words(answer)
    if not cited or not words or any(not 1 <= n <= len(passages) for n in cited):
        return False
    return any(words <= set(tokenize(passages[n - 1])) for n in cited)


def rag_21(question, index, llm, k=3):
    """Search, prompt, greedy reply, parse and verify."""
    ids = search_20(index, question, "hybrid", k)
    passages = [index["chunks"][i] for i in ids]
    reply = llm.generate(rag_messages_21(question, passages), max_new_tokens=24, cache=CACHE)
    answer, cited = parse_reply_21(reply)
    return {"ids": ids, "passages": passages, "reply": reply, "answer": answer, "cited": cited,
            "verified": verify_21(answer, cited, passages)}'''

CHECK_21 = r'''with checked("B4.21"):
    msgs_21 = rag_messages_21("Who is it?", ["first passage", "second passage"])
    if returned("B4.21", "rag_messages_21", msgs_21):
        want_21 = wbrag.rag_messages("Who is it?", ["first passage", "second passage"])
        verdict("B4.21", msgs_21 == want_21, "rag_messages_21 : système, exemple, passages numérotés, question en dernier.",
                f"rag_messages_21 devrait rendre {len(want_21)} messages, le dernier : {want_21[-1]['content']!r} ; "
                f"reçu {str(msgs_21)[-160:]}.")
with checked("B4.21"):
    for reply_21, want_21 in (("Duke of Balmoral [2]", ("Duke of Balmoral", [2])),
                              ("Sir George Burnwell [1][3].", ("Sir George Burnwell", [1, 3])),
                              ("In London [2], at Baker Street [2].", ("In London , at Baker Street", [2])),
                              ("I don't know.", ("I don't know", [])),
                              ("  lawyer  ", ("lawyer", []))):
        got_21 = parse_reply_21(reply_21)
        verdict("B4.21", tuple(got_21) == want_21 if isinstance(got_21, (tuple, list)) else False,
                f"parse_reply_21({reply_21!r}) = {want_21}",
                f"parse_reply_21({reply_21!r}) devrait rendre {want_21} ; reçu {got_21!r}.")
with checked("B4.21"):
    passages_21 = ["Holmes plays the violin.", "The Duke of Balmoral has two sons."]
    for args_21, want_21 in ((("Duke of Balmoral", [2]), True), (("Duke of Balmoral", [1]), False),
                             (("Duke of Balmoral", [1, 2]), True), (("Duke of Balmoral", [3]), False),
                             (("Duke of Balmoral", []), False), (("the", [1]), False),
                             (("violin", [1, 3]), False), (("violin, of course", [1]), False)):
        got_21 = verify_21(*args_21, passages_21)
        verdict("B4.21", got_21 is want_21 or got_21 == want_21, f"verify_21{args_21} = {want_21}",
                f"verify_21{args_21} avec les passages {passages_21} devrait rendre {want_21} ; reçu {got_21!r}.")'''

EXPERIMENT_21 = r'''rows_21 = None
if QWEN is None or globals().get("INDEX_20") is None:
    print("⏳ Ex B4.21 : le RAG attend Qwen2.5 et ton index de B4.20.")
elif written(lambda: rag_21(QUESTIONS_21[0]["question"], INDEX_20, QWEN)):   # the first question, then the cache
    with checked("B4.21"):
        start_21, hits0_21, misses0_21 = time.perf_counter(), CACHE.hits, CACHE.misses
        rows_21, outs_21 = [], []
        for item in QUESTIONS_21:
            out = rag_21(item["question"], INDEX_20, QWEN)
            if not (isinstance(out, dict) and {"ids", "passages", "reply", "answer", "cited", "verified"} <= set(out)):
                print("❌ Ex B4.21 : rag_21 doit rendre un dictionnaire aux clés ids, passages, reply, answer, cited, "
                      f"verified ; reçu {str(out)[:120]}.")
                rows_21 = None
                raise StopCheck
            outs_21.append(out)
            relevant = wbrag.relevant_ids(INDEX_20["chunks"], item)
            model_answer = wbrag.parse_reply(str(out["reply"]))[0]          # what the model said, not out["answer"]
            rows_21.append({"id": item["id"], "found": bool(relevant & set(out["ids"])),
                            "correct": wbrag.answer_in(model_answer, item), "verified": bool(out["verified"]),
                            "idk": wbrag.is_idk(out["reply"]), "reply": out["reply"], "gold": item["answer"]})
        print(f"{len(rows_21)} questions en {fr(time.perf_counter() - start_21, 1)} s (cache : "
              f"{CACHE.hits - hits0_21} réponses relues, {CACHE.misses - misses0_21} générées)\n")
        print(f"{'id':4} {'trouvé':6} {'juste':6} {'citation':9} réponse  (attendu)")
        for r in rows_21:
            print(f"{r['id']:4} {'oui' if r['found'] else 'non':6} {'oui' if r['correct'] else 'non':6} "
                  f"{'vérifiée' if r['verified'] else '—':9} {r['reply']!r}  ({r['gold']})")
        n_21 = len(rows_21)
        print(f"\nPassage pertinent parmi ceux donnés au modèle : {sum(r['found'] for r in rows_21)}/{n_21} · réponses "
              f"justes : {sum(r['correct'] for r in rows_21)}/{n_21} · citations vérifiées : "
              f"{sum(r['verified'] for r in rows_21)}/{n_21} · justes et vérifiées : "
              f"{sum(r['correct'] and r['verified'] for r in rows_21)}/{n_21} · « I don't know » : "
              f"{sum(r['idk'] for r in rows_21)}/{n_21}")
        # every question against the reference chain: the passages, the reply of the model, what is read from it
        ref_rag_21 = wbrag.ReferenceRAG(QWEN, wbrag.ReferenceIndex(REF_CHUNKS, SM_19, subset="b4-holmes-120-20"),
                                        cache=CACHE)
        bad_ids_21, bad_reply_21, bad_read_21 = [], [], []
        for item, out in zip(QUESTIONS_21, outs_21):
            want = ref_rag_21.answer(item["question"])
            if [int(i) for i in out["ids"]] != want["ids"]:
                bad_ids_21.append(item["id"])
            elif out["reply"] != want["reply"]:
                bad_reply_21.append(item["id"])
            else:
                answer, cited = parse_reply_21(out["reply"])
                if (out["answer"], list(out["cited"])) != (answer, list(cited)) or list(out["passages"]) != want["passages"] \
                        or bool(out["verified"]) != bool(verify_21(answer, cited, want["passages"])):
                    bad_read_21.append(item["id"])
        verdict("B4.21", not bad_ids_21, f"pour les {n_21} questions, les passages de la recherche hybride (k = 3).",
                f"pour {', '.join(bad_ids_21)}, tes passages ne sont pas les 3 premiers de "
                "search_20(index, question, \"hybrid\", 3) : relis le contrat de rag_21 (hybride, k = 3).")
        verdict("B4.21", not bad_reply_21, "la réponse du modèle est celle qu'il donne au prompt du contrat.",
                f"pour {', '.join(bad_reply_21)}, out[\"reply\"] n'est pas la réponse que le modèle donne au prompt du "
                "contrat (rag_messages_21, glouton, 24 tokens au plus, par le cache).")
        verdict("B4.21", not bad_read_21, "passages, answer, cited et verified viennent de cette réponse.",
                f"pour {', '.join(bad_read_21)}, passages, answer, cited ou verified ne sont pas ceux que donnent les "
                "chunks de out[\"ids\"], parse_reply_21 et verify_21 appliqués à la réponse du modèle (out[\"reply\"]).")
else:
    print("⏳ Ex B4.21 : le RAG attend tes quatre fonctions.")'''
