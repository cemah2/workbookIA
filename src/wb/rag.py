"""Questions, corpora and a reference RAG for bonus chapter B4 (retrieval-augmented generation on Holmes and Verne).

Quick reference
---------------
>>> qa = load_qa("holmes")                   # 30 questions: id, question, answer, aliases, evidence, start, end, story,
>>> qa[0]["question"], qa[0]["answer"]       #   story_title, paraphrased (10 of them share no keyword with their text)
>>> extra = load_unanswerable()              # 5 questions without an answer in the stories (absent, false premise)
>>> fr = load_qa("verne")                    # 10 questions in French on chapters I to VII of Le Tour du monde
>>> docs = documents("holmes")               # [(title, text)]: the 12 stories (verne: chapters I to VII)
>>> chunks, doc_ids = reference_chunks("holmes")   # every document cut by chunk_text(text, 120, 20): 1 045 chunks
>>> relevant_ids(chunks, qa[0])              # the chunks that contain the evidence (whatever the chunking)
>>> tokenize("Holmes’s pipe, 1884!")         # ['holmes', 's', 'pipe', '1884']: the tokens of BM25 (corpus and queries)
>>> answer_in("He was a lawyer [1].", qa[1]) # the normalised answer (or an alias) appears in the text
>>> answer, cited = parse_reply("Duke of Balmoral [2]")   # ('Duke of Balmoral', [2])
>>> verify_citations(answer, cited, passages)             # every cited passage exists, one of them supports the answer
>>> rag = ReferenceRAG(llm, sentence_model, cache=cache)   # the reference RAG of Ex B4.20 and B4.21 (hybrid search,
>>> out = rag.answer(question)               #   numbered passages, one example, a short answer with its citation)

The questions are written for the workbook (data card: ``data/cards/qa.md``). An evidence is a short excerpt of the
text (at most 21 words, whitespace normalised) located by character positions in ``wb.datasets.load_holmes()`` or
``load_verne()``: a chunk is relevant when it contains the whole evidence, so the truth does not depend on the
chunking, and a chunking of 120 words with an overlap of 20 always keeps every evidence whole in some chunk.

``ReferenceRAG`` and the functions it uses are the reference solution of Ex B4.20 and B4.21 (and the fallback of the
later exercises when yours is missing); the notebook only reads them to check your functions.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from wb import text as wbtext

QA_DIR = Path(__file__).resolve().parents[2] / "data" / "qa"
QA_FILES = {"holmes": "holmes_qa.json", "verne": "verne_qa.json"}
CHUNK_SIZE, OVERLAP = 120, 20                  # words; an evidence of at most OVERLAP + 1 words stays whole
FAST_IDS = ("H02", "H03", "H05", "H08", "H12", "H16", "H19", "H21", "H25", "H28")   # the 10 questions of fast mode
RRF_K, DEPTH = 60, 20                          # the fusion of Ex B4.20: RRF (k = 60) of the top 20 of each method
IDK = "I don't know"

# ---------------------------------------------------------------------------
# Questions and corpora
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _qa_file(name: str) -> dict:
    if name not in QA_FILES:
        raise ValueError(f"unknown question set {name!r}; choose among {sorted(QA_FILES)}")
    return json.loads((QA_DIR / QA_FILES[name]).read_text(encoding="utf-8"))


def load_qa(name: str = "holmes") -> list[dict]:
    """The annotated questions of ``name`` (``"holmes"``: 30, in English; ``"verne"``: 10, in French), as fresh
    dictionaries (changing them changes nothing here)."""
    return json.loads(json.dumps(_qa_file(name)["questions"]))


def load_unanswerable() -> list[dict]:
    """5 questions on Holmes without an answer in the stories (``kind``: ``"absent"`` or ``"false_premise"``): the
    right reply is « I don't know ». They are not among the 30 of ``load_qa("holmes")``."""
    return json.loads(json.dumps(_qa_file("holmes")["unanswerable"]))


CONTROL_FILE = "holmes_control.json"


def load_control() -> list[dict]:
    """12 held-out questions on Holmes (``C01`` to ``C12``, one per story), the control of the challenge B4.30: ``id``,
    ``question``, ``story``, ``story_title``, ``max_words`` and ``answer_sha256``, the SHA-256 hashes of the
    normalised answers (``normalize_answer``) — the answers themselves are not stored. ``control_correct`` checks a
    reply against them."""
    data = json.loads((QA_DIR / CONTROL_FILE).read_text(encoding="utf-8"))
    return json.loads(json.dumps(data["questions"]))


def control_correct(text: str, item: dict) -> bool:
    """``answer_in`` for a control question: True when a run of consecutive words of the normalised ``text`` has the
    hash of a normalised answer of ``item``."""
    import hashlib

    words = normalize_answer(text).split()
    hashes = set(item["answer_sha256"])
    for n in range(1, int(item["max_words"]) + 1):
        for i in range(len(words) - n + 1):
            if hashlib.sha256(" ".join(words[i:i + n]).encode("utf-8")).hexdigest() in hashes:
                return True
    return False


def documents(name: str = "holmes") -> list[tuple[str, str]]:
    """``[(title, text)]``: the 12 stories of Holmes (``wb.text.holmes_stories()``), or chapters I to VII of Verne
    (the heading of each chapter is its title; the corpus of the 10 French questions)."""
    if name == "holmes":
        return list(wbtext.holmes_stories())
    if name != "verne":
        raise ValueError(f"unknown corpus {name!r}; choose 'holmes' or 'verne'")
    meta = _qa_file("verne")
    raw = wbtext.load_text("verne")[meta["corpus_start"]:meta["corpus_stop"]]
    parts = re.split(r"\n\n\n\n\n([IVXL]+)\n\n", raw)
    docs = []
    for k in range(1, len(parts), 2):
        body = parts[k + 1]
        heading, _, rest = body.partition("\n\n\n")
        docs.append((f"{parts[k]}. {' '.join(heading.split())}", rest))
    return docs


def chunk_words(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """The reference word chunking (the contract of ``chunk_text`` in ``unit="word"``): windows of ``chunk_size``
    words starting every ``chunk_size - overlap`` words, re-joined with single spaces."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("expected 0 <= overlap < chunk_size")
    words = text.split()
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            return chunks if words else []
        start += chunk_size - overlap


def reference_chunks(name: str = "holmes", chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """``(chunks, doc_ids)``: every document of ``documents(name)`` cut by ``chunk_words``, and the index of the
    document of each chunk (an int64 array). Holmes: 1 045 chunks; Verne: 103."""
    chunks, doc_ids = [], []
    for d, (_, body) in enumerate(documents(name)):
        for chunk in chunk_words(body, chunk_size, overlap):
            chunks.append(chunk)
            doc_ids.append(d)
    return chunks, np.array(doc_ids, dtype=np.int64)


def relevant_ids(chunks, item: dict) -> set[int]:
    """The indices of the chunks that contain the whole evidence of ``item`` (whitespace normalised)."""
    evidence = " ".join(item["evidence"].split())
    return {i for i, chunk in enumerate(chunks) if evidence in " ".join(str(chunk).split())}


# ---------------------------------------------------------------------------
# Tokens and answers
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """The tokens of BM25 in this chapter: the text in lower case, cut into runs of letters and digits (``\\w+``,
    accents kept, the underscore of the Gutenberg italics and the apostrophes are separators). The same function
    for the corpus and the queries."""
    return re.findall(r"[^\W_]+", text.lower())


_ARTICLES = {"a", "an", "the", "le", "la", "les", "l", "un", "une", "des", "du", "de", "d"}


def normalize_answer(text: str) -> str:
    """Lower case, Unicode NFC, curly quotes made straight, punctuation and underscores turned into spaces, the
    articles (English and French) removed, spaces collapsed: « The Duke of Balmoral. » → « duke of balmoral »."""
    text = unicodedata.normalize("NFC", str(text)).lower().replace("’", "'")
    words = re.findall(r"[^\W_]+", text)
    return " ".join(w for w in words if w not in _ARTICLES)


def answer_in(text: str, item: dict) -> bool:
    """True when the normalised answer of ``item``, or one of its aliases, appears in the normalised ``text`` as
    whole words (« 7 » is not found in « 17 »)."""
    hay = f" {normalize_answer(text)} "
    return any(f" {needle} " in hay for needle in (normalize_answer(a) for a in [item["answer"], *item["aliases"]])
               if needle)


def is_idk(reply: str) -> bool:
    """True when a reply says that the passages do not hold the answer (« I don't know », « je ne sais pas »)."""
    norm = normalize_answer(reply)
    return norm.startswith(("i don t know", "i do not know", "je ne sais pas")) or norm in ("unknown", "inconnu")


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
_CITATION = re.compile(r"\[(\d+)\]")


def parse_reply(reply: str) -> tuple[str, list[int]]:
    """``(answer, cited)``: the reply without its citations « [n] » (spaces and a final full stop removed), and the
    numbers cited, in their order, each once."""
    cited = []
    for match in _CITATION.finditer(reply):
        n = int(match.group(1))
        if n not in cited:
            cited.append(n)
    answer = " ".join(_CITATION.sub(" ", reply).split()).strip(" .")
    return answer, cited


_STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "was", "it", "he", "she", "his", "her",
         "by", "for", "with", "from", "as", "that", "this", "le", "la", "les", "l", "un", "une", "de", "des", "du",
         "d", "et", "à", "au", "en", "s"}


def content_words(text: str) -> set[str]:
    """The tokens of ``text`` (``tokenize``) without the most common little words: what an answer claims."""
    return {t for t in tokenize(text) if t not in _STOP}


def supports(passage: str, answer: str) -> bool:
    """True when every content word of ``answer`` is a token of ``passage`` (an answer without content word is not
    supported)."""
    words = content_words(answer)
    return bool(words) and words <= set(tokenize(passage))


def verify_citations(answer: str, cited, passages) -> bool:
    """True when ``cited`` is not empty, every number in it is a passage (from 1 to ``len(passages)``), and at least
    one cited passage supports the answer (``supports``)."""
    cited = list(cited)
    if not cited or any(not isinstance(n, (int, np.integer)) or not 1 <= n <= len(passages) for n in cited):
        return False
    return any(supports(passages[n - 1], answer) for n in cited)


# ---------------------------------------------------------------------------
# Reference retrieval (the reference solution of Ex B4.20)
# ---------------------------------------------------------------------------
class BM25Index:
    """BM25 over tokenised documents (the formula of the fiche, idf with + 1, k1 = 1.5, b = 0.75)."""

    def __init__(self, docs_tokens, k1: float = 1.5, b: float = 0.75):
        from collections import Counter

        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in docs_tokens]
        self.len = np.array([len(doc) for doc in docs_tokens], dtype=float)
        self.avgdl = float(self.len.mean()) if len(self.len) else 0.0
        df = Counter(term for counts in self.tf for term in counts)
        n = len(self.tf)
        self.idf = {t: float(np.log(1 + (n - c + 0.5) / (c + 0.5))) for t, c in df.items()}

    def scores(self, query_tokens) -> np.ndarray:
        out = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avgdl) if self.avgdl else np.full(len(self.tf), self.k1)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for d, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if f:
                    out[d] += idf * f * (self.k1 + 1) / (f + norm[d])
        return out


def cosine(query, docs, eps: float = 1e-12) -> np.ndarray:
    """The cosine of a query with every row of ``docs``, in float64 (the convention of ``cosine_similarity_matrix``
    of B2: the norms plus ``eps``, so that a row of zeros has a similarity of 0)."""
    q = np.asarray(query, dtype=float)
    D = np.asarray(docs, dtype=float)
    return (D @ q) / ((np.linalg.norm(D, axis=1) + eps) * (np.linalg.norm(q) + eps))


def top_k(scores, k: int) -> list[int]:
    """The indices of the ``k`` largest scores, best first, ties broken by the smaller index."""
    scores = np.asarray(scores, dtype=float)
    return np.argsort(-scores, kind="stable")[:k].tolist()


def rrf(rankings, k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: ids sorted by decreasing sum of 1 / (k + rank), ranks from 1, ties by smaller id."""
    from fractions import Fraction

    score: dict = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[int(doc)] = score.get(int(doc), Fraction(0)) + Fraction(1, k + rank)
    return sorted(score, key=lambda d: (-score[d], d))


class ReferenceIndex:
    """The chunks of a corpus, their BM25 index and (with a sentence model) their dense embeddings, cached by
    ``wb.text.passage_embeddings``; ``search(query, method, k)`` with ``method`` in ``"bm25"``, ``"dense"``,
    ``"hybrid"`` (RRF of the top ``DEPTH`` of both)."""

    def __init__(self, chunks, sentence_model=None, subset: str | None = None, verbose: bool = False):
        self.chunks = [str(c) for c in chunks]
        self.bm25 = BM25Index([tokenize(c) for c in self.chunks])
        self.sm = sentence_model
        self.embeddings = None
        if sentence_model is not None:
            self.embeddings = wbtext.passage_embeddings(sentence_model, self.chunks, subset or "b4-chunks",
                                                        kind="passage", verbose=verbose)

    def search(self, query: str, method: str = "hybrid", k: int = 5) -> list[int]:
        if method == "bm25":
            return top_k(self.bm25.scores(tokenize(query)), k)
        if self.embeddings is None:
            raise ValueError("dense and hybrid search need a sentence model")
        dense = top_k(cosine(self.sm.embed([query], kind="query")[0], self.embeddings), max(k, DEPTH))
        if method == "dense":
            return dense[:k]
        if method != "hybrid":
            raise ValueError(f"method must be 'bm25', 'dense' or 'hybrid', got {method!r}")
        return rrf([top_k(self.bm25.scores(tokenize(query)), DEPTH), dense[:DEPTH]])[:k]


# ---------------------------------------------------------------------------
# The reference RAG (Ex B4.21)
# ---------------------------------------------------------------------------
RAG_SYSTEM = ("You answer questions about the Sherlock Holmes stories. Use only the numbered passages. Give a short "
              "answer followed by the number of the passage that contains it, in square brackets. If no passage "
              f"contains the answer, reply exactly: {IDK}.")
RAG_EXAMPLE = [
    {"role": "user", "content": "Passages:\n[1] Watson took his hat and followed Holmes into the street.\n"
                                "[2] The letter was posted at Charing Cross on Monday.\n\n"
                                "Question: Where was the letter posted?"},
    {"role": "assistant", "content": "Charing Cross [2]"},
]


def rag_messages(question: str, passages, system: str = RAG_SYSTEM, example=RAG_EXAMPLE) -> list[dict]:
    """The prompt of the reference RAG: the system rules, one example exchange (the format of the reply), then the
    numbered passages and the question, last."""
    numbered = "\n".join(f"[{n}] {p}" for n, p in enumerate(passages, start=1))
    return ([{"role": "system", "content": system}] + [dict(m) for m in example]
            + [{"role": "user", "content": f"Passages:\n{numbered}\n\nQuestion: {question}"}])


class ReferenceRAG:
    """The reference RAG of Ex B4.21: ``ReferenceIndex`` (hybrid search, ``k`` passages), ``rag_messages``, a greedy
    reply of at most ``max_new_tokens`` tokens (through ``cache``), ``parse_reply`` and ``verify_citations``."""

    def __init__(self, llm, index: ReferenceIndex, *, k: int = 3, method: str = "hybrid", max_new_tokens: int = 24,
                 cache=None):
        self.llm, self.index, self.k, self.method = llm, index, k, method
        self.max_new_tokens, self.cache = max_new_tokens, cache

    def answer(self, question: str) -> dict:
        ids = self.index.search(question, self.method, self.k)
        passages = [self.index.chunks[i] for i in ids]
        reply = self.llm.generate(rag_messages(question, passages), max_new_tokens=self.max_new_tokens,
                                  cache=self.cache)
        answer, cited = parse_reply(reply)
        return {"question": question, "ids": ids, "passages": passages, "reply": reply, "answer": answer,
                "cited": cited, "idk": is_idk(reply), "verified": verify_citations(answer, cited, passages)}

