"""Tests of wb.rag (chapter B4): the annotated questions, the corpora, the answer and citation helpers, the reference
retrieval (against the reference library mylearn_ref.retrieval) and the prompt of the reference RAG."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from wb import rag
from wb import text as wbtext

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "solutions") not in sys.path:      # the reference library is the oracle (absent from the learner mode path)
    sys.path.insert(0, str(ROOT / "solutions"))

_STOP_EXTRA = {"did", "does"}


def _stems(text: str) -> set[str]:
    """Content words of an English text, crudely stemmed (common suffixes cut, 5 letters kept): two forms of one word
    (« copying » and « copy », « Australian » and « Australia ») share a stem."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    out = set()
    for w in re.findall(r"[a-z]+", text.lower().replace("’", "'")):
        if w in ENGLISH_STOP_WORDS or w in _STOP_EXTRA or len(w) < 3:
            continue
        for suffix in ("ings", "ing", "edly", "ed", "es", "s", "ly", "ers", "er"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: -len(suffix)]
                break
        out.add(w[:5])
    return out


def _normalised(raw: str, start: int, end: int) -> str:
    return " ".join(raw[start:end].split())


# ---------------------------------------------------------------------------
# The question sets
# ---------------------------------------------------------------------------
def test_question_sets_have_the_sizes_of_the_contract():
    holmes, verne, extra = rag.load_qa("holmes"), rag.load_qa("verne"), rag.load_unanswerable()
    assert len(holmes) == 30 and sum(q["paraphrased"] for q in holmes) == 10
    assert len(verne) == 10 and not any(q["paraphrased"] for q in verne)
    assert len(extra) == 5 and {u["kind"] for u in extra} == {"absent", "false_premise"}
    ids = [q["id"] for q in holmes + verne] + [u["id"] for u in extra]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    fast = [q for q in holmes if q["id"] in rag.FAST_IDS]
    assert len(fast) == 10 and sum(q["paraphrased"] for q in fast) == 3


@pytest.mark.parametrize("name", ["holmes", "verne"])
def test_each_evidence_is_at_its_positions_once_and_holds_its_answer(name):
    raw = wbtext.load_text(name)
    whole = " ".join(raw.split())
    for q in rag.load_qa(name):
        assert _normalised(raw, q["start"], q["end"]) == q["evidence"], f"{q['id']}: evidence not at [start:end]"
        assert whole.count(q["evidence"]) == 1, f"{q['id']}: the evidence must occur once in the text"
        assert rag.answer_in(q["evidence"], q), f"{q['id']}: the answer {q['answer']!r} is not in its evidence"
        assert len(q["evidence"].split()) <= rag.OVERLAP + 1, f"{q['id']}: evidence longer than overlap + 1 words"
        assert q["question"].strip().endswith("?"), f"{q['id']}: a question ends with a question mark"


def test_holmes_questions_name_their_story():
    stories = wbtext.holmes_stories()
    raw = wbtext.load_text("holmes")
    for q in rag.load_qa("holmes"):
        title, body = stories[q["story"]]
        assert q["story_title"] == title
        assert q["evidence"] in " ".join(body.split()), f"{q['id']}: evidence outside its story"
        assert raw.find(body[:200]) <= q["start"]


def test_verne_questions_are_in_the_corpus_and_their_chapter():
    meta = rag._qa_file("verne")
    docs = rag.documents("verne")
    assert [t.split(".")[0] for t, _ in docs] == ["I", "II", "III", "IV", "V", "VI", "VII"]
    for q in rag.load_qa("verne"):
        assert meta["corpus_start"] <= q["start"] < q["end"] <= meta["corpus_stop"]
        body = dict((t.split(".")[0], b) for t, b in docs)[q["chapter"]]
        assert q["evidence"] in " ".join(body.split()), f"{q['id']}: evidence outside chapter {q['chapter']}"


def test_paraphrased_questions_share_no_keyword_with_the_text_around_their_evidence():
    raw = wbtext.load_text("holmes")
    words = raw.split()
    for q in rag.load_qa("holmes"):
        at = len(raw[: q["start"]].split())
        window = " ".join(words[max(0, at - 140): at + len(q["evidence"].split()) + 140])
        common = _stems(q["question"]) & _stems(window)
        if q["paraphrased"]:
            assert not common, f"{q['id']} shares keywords with its text: {sorted(common)}"
        else:
            assert common, f"{q['id']} is direct but shares no keyword with its text"


def test_unanswerable_questions_have_no_answer_in_the_stories():
    whole = " ".join(wbtext.load_text("holmes").split())
    for name in ("Mycroft", "Moriarty", "Napoleon of crime", "Paris to study"):
        assert name not in whole
    assert all(u["question"].endswith("?") and u["note"] for u in rag.load_unanswerable())


def test_every_question_has_a_relevant_reference_chunk():
    for name, size in (("holmes", 1045), ("verne", 93)):
        chunks, doc_ids = rag.reference_chunks(name)
        assert len(chunks) == size and len(doc_ids) == size
        for q in rag.load_qa(name):
            relevant = rag.relevant_ids(chunks, q)
            assert relevant, f"{q['id']}: no chunk holds the whole evidence"
            assert all(rag.answer_in(chunks[i], q) for i in relevant)


def test_load_qa_returns_fresh_copies():
    first = rag.load_qa("holmes")
    first[0]["answer"] = "changed"
    assert rag.load_qa("holmes")[0]["answer"] != "changed"
    with pytest.raises(ValueError):
        rag.load_qa("dickens")
    with pytest.raises(ValueError):
        rag.documents("dickens")


# ---------------------------------------------------------------------------
# Tokens, answers, citations
# ---------------------------------------------------------------------------
def test_tokenize_lowercases_and_splits_on_punctuation_and_underscores():
    assert rag.tokenize("Holmes’s pipe, 1884! _Gloria Scott_") == ["holmes", "s", "pipe", "1884", "gloria", "scott"]
    assert rag.tokenize("L'Éléphant") == ["l", "éléphant"]


def test_normalize_answer_and_answer_in():
    assert rag.normalize_answer("The Duke of Balmoral.") == "duke of balmoral"
    assert rag.normalize_answer("the gasfitters’ ball") == rag.normalize_answer("gasfitters' ball")
    item = {"answer": "7", "aliases": ["sept"]}
    assert rag.answer_in("au numéro 7 de Saville-row", item)
    assert not rag.answer_in("au numéro 17", item), "an answer is matched as whole words"
    assert rag.answer_in("Sept.", item)
    assert rag.answer_in("He was a LAWYER [1]", {"answer": "lawyer", "aliases": []})


def test_parse_reply_and_is_idk():
    assert rag.parse_reply("Duke of Balmoral [2]") == ("Duke of Balmoral", [2])
    assert rag.parse_reply("Sir George Burnwell [1][3] [1].") == ("Sir George Burnwell", [1, 3])
    assert rag.parse_reply("I don't know.") == ("I don't know", [])
    assert rag.is_idk("I don't know.") and rag.is_idk("Je ne sais pas.") and not rag.is_idk("London [1]")


def test_verify_citations():
    passages = ["Holmes plays the violin.", "The Duke of Balmoral has two sons."]
    assert rag.verify_citations("Duke of Balmoral", [2], passages)
    assert rag.verify_citations("Duke of Balmoral", [1, 2], passages)
    assert not rag.verify_citations("Duke of Balmoral", [1], passages)
    assert not rag.verify_citations("Duke of Balmoral", [3], passages), "a number outside the passages"
    assert not rag.verify_citations("Duke of Balmoral", [], passages), "no citation"
    assert not rag.verify_citations("the", [1], passages), "no content word"
    assert not rag.verify_citations("violin", [0], passages)


# ---------------------------------------------------------------------------
# Reference retrieval against the reference library
# ---------------------------------------------------------------------------
def test_chunk_words_is_the_word_chunking_of_the_reference_library():
    from mylearn_ref import retrieval as R

    rng = np.random.default_rng(0)
    for _ in range(30):
        words = [f"w{int(i)}" for i in rng.integers(0, 50, size=int(rng.integers(0, 60)))]
        text = "  ".join(words)
        size = int(rng.integers(1, 12))
        overlap = int(rng.integers(0, size))
        assert rag.chunk_words(text, size, overlap) == R.chunk_text(text, size, overlap, unit="word")


def test_bm25_index_matches_the_reference_library():
    from mylearn_ref import retrieval as R

    rng = np.random.default_rng(1)
    vocab = [f"t{i}" for i in range(12)]
    for _ in range(10):
        corpus = [list(rng.choice(vocab, size=int(rng.integers(1, 15)))) for _ in range(8)]
        query = list(rng.choice(vocab + ["absent"], size=3))
        np.testing.assert_allclose(rag.BM25Index(corpus).scores(query), R.BM25().fit(corpus).get_scores(query),
                                   rtol=1e-12, atol=1e-12)


def test_top_k_and_rrf_match_the_reference_library():
    from mylearn_ref import retrieval as R

    assert rag.top_k([0.5, 0.9, 0.9, 0.1], 3) == [1, 2, 0]
    for rankings in ([[7, 2, 9, 4], [2, 5, 7, 1]], [[3, 1], [1, 3]], [[5], [], [5, 6]]):
        assert rag.rrf(rankings) == R.reciprocal_rank_fusion(rankings).tolist()


def test_rag_messages_numbers_the_passages_and_ends_with_the_question():
    messages = rag.rag_messages("Who is it?", ["first", "second"])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == rag.RAG_SYSTEM
    assert messages[-1]["content"] == "Passages:\n[1] first\n[2] second\n\nQuestion: Who is it?"
    messages[1]["content"] = "changed"
    assert rag.RAG_EXAMPLE[0]["content"] != "changed", "the example is copied, never shared"


def test_reference_index_bm25_search_without_a_sentence_model():
    chunks = ["holmes fume sa pipe", "watson lit le journal et holmes joue du violon", "le violon de holmes",
              "watson écrit"]
    index = rag.ReferenceIndex(chunks)
    assert index.search("holmes violon", "bm25", 4) == [2, 1, 0, 3]
    with pytest.raises(ValueError):
        index.search("holmes", "dense", 2)


# --------------------------------------------------------------------------------------- control questions (B4.30)
def test_control_questions_hide_their_answers_and_cover_every_story():
    """No answer of the control is written in clear: not in its file (no run of its words has the hash of an
    answer), and not in this test either (the cases below use answers made up for the test)."""
    items = rag.load_control()
    assert [q["id"] for q in items] == [f"C{i:02d}" for i in range(1, 13)]
    assert sorted(q["story"] for q in items) == list(range(12))
    data = json.loads((ROOT / "data" / "qa" / rag.CONTROL_FILE).read_text(encoding="utf-8"))
    for item in items:                   # the file without the questions ("James Calhoun" may name another answer)
        rest = json.dumps({**data, "questions": [{k: v for k, v in q.items() if k != "question"}
                                                 for q in data["questions"]]}, ensure_ascii=False)
        assert not rag.control_correct(rest, item), f"an answer of the control is written in clear: {item['id']}"
    annotated = {q["question"] for q in rag.load_qa("holmes")}
    assert not annotated & {q["question"] for q in items}


def _hashed_item(*answers, max_words=None):
    """A control item for answers made up for the test (the real ones are never written in clear)."""
    import hashlib

    norms = [rag.normalize_answer(a) for a in answers]
    return {"answer_sha256": [hashlib.sha256(n.encode("utf-8")).hexdigest() for n in norms],
            "max_words": max_words or max(len(n.split()) for n in norms)}


@pytest.mark.parametrize("text, answers, expected", [
    ("Behind a velvet curtain [2]", ("velvet curtain",), True), ("A velvet cushion", ("velvet curtain",), False),
    ("the Red Lion.", ("Red Lion",), True), ("Red", ("Red Lion",), False),
    ("Marsh and Pembroke", ("Marsh and Pembroke", "Marsh & Pembroke"), True),
    ("Marsh & Pembroke", ("Marsh and Pembroke", "Marsh & Pembroke"), True),
    ("Pembroke", ("Marsh and Pembroke", "Marsh & Pembroke"), False),
    ("Agatha", ("Agatha",), True), ("Agathe", ("Agatha",), False), ("Agathas", ("Agatha",), False),
], ids=lambda v: str(v))
def test_control_correct_matches_whole_words_of_a_hashed_answer(text, answers, expected):
    assert rag.control_correct(text, _hashed_item(*answers)) is expected


def test_control_answers_are_in_the_stories():
    """Every hashed answer is found, as whole words, in the text of its story (the answers were taken from it)."""
    docs = rag.documents("holmes")
    for item in rag.load_control():
        assert rag.control_correct(docs[item["story"]][1], item), item["id"]
