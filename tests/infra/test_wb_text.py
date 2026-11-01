"""Tests of wb.text: frozen texts, excerpts and splits, words, vocabulary, real tokenizers (bonus chapter B2)."""

import re

import numpy as np
import pytest

from wb import text as W


def test_texts_and_excerpts_are_frozen():
    holmes, verne = W.load_text("holmes"), W.load_text("verne")
    assert len(holmes) == 562_203 and len(verne) == 421_336, "expected the lengths of the B2 fiche (Gutenberg header removed)"
    assert W.excerpt("holmes") == holmes[100_000:150_000] and len(W.excerpt("verne")) == 50_000
    assert W.excerpt("verne", 0, 10) == verne[:10]
    assert W.load_text("holmes") is holmes, "expected the text to be read once per session"


def test_unknown_text_is_refused():
    with pytest.raises(ValueError, match="unknown text"):
        W.load_text("dickens")


@pytest.mark.parametrize("name", W.TEXTS)
def test_train_val_cut_at_a_line_boundary_about_90_percent(name):
    text = W.load_text(name)
    train, val = W.train_val(name)
    assert train + val == text, "expected train + val to give the whole text back (no shuffling, nothing lost)"
    assert train.endswith("\n"), "expected the cut at the start of a line (no word cut in two)"
    assert 0.89 < len(train) / len(text) <= 0.9, f"expected about 90 % of the characters in train, got {len(train) / len(text):.4f}"
    assert W.train_val(name) == (train, val), "expected the same split at every call (frozen)"


def test_split_point_on_a_small_text_and_bad_shares():
    text = "ab\ncd\nef\ngh\n"
    assert W.split_point(text, 0.5) == 6, "expected the start of the line that holds character 6"
    assert W.split_point("abcdefghij", 0.5) == 5, "expected the exact index when the text has no newline"
    for share in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            W.split_point(text, share)


def test_words_of_the_fiche_and_of_verne():
    holmes, verne = W.words("holmes"), W.words("verne")
    assert (len(holmes), len(set(holmes))) == (105_863, 7_817), "expected the counts of the map of the fiche"
    assert (len(verne), len(set(verne))) == (72_367, 8_793)
    assert all(re.fullmatch(r"[a-z]+", w) for w in holmes[:2000])
    assert "été" in set(verne) and "passepartout" in set(verne), "expected accents kept and lower case"
    assert W.words("L'été, Noël 1872 !", lang="verne") == ["l", "été", "noël"]
    assert W.words("Mr. Holmes's 221B", lang="holmes") == ["mr", "holmes", "s", "b"]


def test_words_of_a_text_need_a_language():
    with pytest.raises(ValueError, match="lang"):
        W.words("some text")
    with pytest.raises(ValueError, match="lang"):
        W.words("some text", lang="english")


def test_top_vocab_by_decreasing_frequency_ties_sorted():
    stoi, itos = W.top_vocab(["b", "a", "c", "a", "b", "d"], 3)
    assert itos == ["a", "b", "c"], "expected the most frequent first, ties in the order of sorted"
    assert stoi == {"a": 0, "b": 1, "c": 2}
    assert W.top_vocab(["x", "y"], 10)[1] == ["x", "y"], "expected every token when there are fewer than size"
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            W.top_vocab(["a"], bad)


def test_top_vocab_of_holmes_and_word_ids():
    words = W.words("holmes")
    stoi, itos = W.top_vocab(words, 4000)
    assert len(itos) == 4000 and itos[0] == "the"
    assert {"holmes", "watson", "letter"} <= set(stoi), "expected the words of 📦 B2.24 in the 4 000 most frequent"
    ids = W.word_ids(words, stoi)
    assert ids.dtype == np.int64 and ids.shape == (len(words),)
    assert ids.min() == -1 and ids.max() == 3999
    assert all(itos[i] == w for i, w in zip(ids[:500], words[:500]) if i >= 0)


def test_tokenizers_are_frozen_at_a_commit():
    assert set(W.TOKENIZERS) == {"gpt2", "bert", "e5", "camembert", "qwen3", "smollm2"}
    for name, spec in W.TOKENIZERS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), f"expected a full commit hash for {name}"
        assert spec["license"] in ("MIT", "Apache 2.0")


def test_unknown_tokenizer_is_refused():
    with pytest.raises(ValueError, match="unknown tokenizer"):
        W.load_tokenizer("llama3")


def test_try_tokenizer_falls_back_to_none_with_a_pending_line(monkeypatch, capsys):
    def no_network(name, *, reuse=True):
        raise W.TokenizerUnavailable(f"tokenizer {name} impossible à charger : OSError : no network")

    monkeypatch.setattr(W, "load_tokenizer", no_network)
    assert W.try_tokenizer("gpt2", "B2.13") is None
    out = capsys.readouterr().out
    assert out.startswith("⏳ Ex B2.13 : tokenizer gpt2") and "à valider sur Colab" in out


def test_load_tokenizer_wraps_errors(monkeypatch):
    transformers = pytest.importorskip("transformers")

    def broken(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", broken)
    monkeypatch.setattr(W, "_LOADED", {})
    with pytest.raises(W.TokenizerUnavailable, match="OSError"):
        W.load_tokenizer("bert")


@pytest.mark.network
def test_real_tokenizers_at_their_revisions():
    pytest.importorskip("transformers")
    sizes = {"gpt2": 50_257, "bert": 30_522, "e5": 250_002, "camembert": 32_006, "qwen3": 151_669, "smollm2": 49_152}
    for name, size in sizes.items():
        tok = W.load_tokenizer(name)
        assert len(tok) == size, f"expected len(tok) == {size} for {name} (fiche, table of 202.5), got {len(tok)}"
        assert W.load_tokenizer(name) is tok, "expected the tokenizer to be loaded once per session"
    gpt2 = W.load_tokenizer("gpt2")
    assert gpt2.encode("Ça coûte 15 €") == [127, 229, 64, 763, 42324, 660, 1315, 10432], "expected the ids of the fiche"
    bert = W.load_tokenizer("bert")
    assert W.count_tokens(bert, "a short one") == 3, "expected [CLS] and [SEP] left out"
    assert len(bert("a short one")["input_ids"]) == 5
    assert W.count_tokens(gpt2, W.excerpt("holmes")) == 13_419, "expected the count of EN_COURS.md §3"


def _naive_byte_merges(texts, vocab_size, min_frequency):
    """Every pair counted again after each merge; ties to the smallest pair (the rule of BPETokenizer.fit)."""
    from collections import Counter
    counts = Counter(p for t in texts for p in re.findall(W.GPT2_PATTERN, t))
    words = {w: list(w.encode("utf-8")) for w in counts}
    merges = []
    while 256 + len(merges) < vocab_size:
        pairs = Counter()
        for w, ids in words.items():
            for p in zip(ids, ids[1:]):
                pairs[p] += counts[w]
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[best] < min_frequency:
            break
        merges.append(best)
        new = 255 + len(merges)
        for w, ids in words.items():
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                    out.append(new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            words[w] = out
    return merges


def test_gpt2_pattern_is_the_one_of_the_stub():
    from wb.core import repo_root
    stub = (repo_root() / "templates" / "mylearn_stubs" / "tokenizer.py").read_text(encoding="utf-8")
    assert f'GPT2_PATTERN = r"{W.GPT2_PATTERN}"' in stub, "expected the 'gpt2' pattern of mylearn/tokenizer.py"


def test_byte_bpe_merges_docstring_example_and_a_small_corpus():
    pytest.importorskip("tokenizers")
    assert W.byte_bpe_merges("low lower lowest newer newest", 259) == [(119, 101), (108, 111), (32, 110)], \
        "expected the merges of the docstring of BPETokenizer (byte level, 'gpt2')"
    texts = [W.excerpt("verne", 0, 6000), "Ça coûte 15 € 🙂 l'été", W.excerpt("holmes", 0, 4000)]
    for vocab_size, min_frequency in ((300, 2), (400, 0), (320, 5)):
        assert W.byte_bpe_merges(texts, vocab_size, min_frequency) == _naive_byte_merges(texts, vocab_size, min_frequency), \
            f"expected the merges of a naive byte-level BPE (vocab_size={vocab_size}, min_frequency={min_frequency})"
    assert W.byte_bpe_merges(texts[0], 300) == W.byte_bpe_merges([texts[0]], 300), "expected a str to be one text"


# ---------------------------------------------------------------------------
# Embeddings of the phase « embeddings » (B2.24, B2.28, B2.29)
# ---------------------------------------------------------------------------
def test_reference_ppmi_svd_matches_the_dense_formula_on_a_small_vocabulary():
    import importlib

    from sklearn.decomposition import TruncatedSVD

    Em = importlib.import_module("mylearn_ref.embeddings")
    words = W.words("holmes")[:30_000]
    E, stoi, itos = W.reference_ppmi_svd(words, n_words=300, window=2, dim=20)
    assert E.shape == (300, 20) and np.allclose(np.linalg.norm(E, axis=1), 1), "expected 300 unit rows of 20 values"
    assert itos == W.top_vocab(words, 300)[1], "expected the vocabulary of top_vocab"
    ids = W.word_ids(words, stoi)
    ids = np.where(ids < 0, 300, ids)                       # an extra id for the unknown words, then dropped
    M = Em.ppmi(Em.cooccurrence_matrix(ids, 301, window=2)[:300, :300], alpha=0.75)
    svd = TruncatedSVD(20, algorithm="arpack", random_state=0)
    D = svd.fit_transform(M) / np.sqrt(svd.singular_values_)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    assert np.allclose(E @ E.T, D @ D.T, atol=1e-6), "expected the same cosines as the dense computation of mylearn"


def test_reference_ppmi_svd_of_holmes_has_the_neighbours_of_the_fiche():
    E, stoi, itos = W.reference_ppmi_svd(W.words("holmes"))
    sims = E @ E[stoi["father"]]
    order = [itos[i] for i in np.argsort(-sims, kind="stable") if itos[i] != "father"]
    assert order[0] == "sister" and {"son", "wife"} <= set(order[:5]), f"expected the neighbours of the fiche, got {order[:5]}"


def test_holmes_stories_and_passages_are_frozen_and_lose_no_word():
    stories = W.holmes_stories()
    assert [t for t, _ in stories][:2] == ["A Scandal in Bohemia", "The Red-Headed League"] and len(stories) == 12
    assert stories[6][0] == "The Blue Carbuncle" and stories[8][0] == "The Engineer’s Thumb"
    passages, story = W.holmes_passages()
    assert len(passages) == 1055 and story.dtype == np.int64 and story.shape == (1055,)
    assert max(len(p.split()) for p in passages) <= 120 and "\n" not in "".join(passages)
    assert np.all(np.diff(story) >= 0) and set(story.tolist()) == set(range(12)), "expected the stories in order"
    for s, (_, body) in enumerate(stories):
        got = " ".join(p for p, k in zip(passages, story) if k == s).split()
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if not re.fullmatch(r"[IVX]+\.", p.strip())]
        want = " ".join(paragraphs).split()                             # without the chapter numbers (« II. »)
        assert got == want, f"expected every word of story {s} once, in order"


class _FakeTokenizer:
    """Whitespace « tokens »: id = length of the word; padding with 0; truncation at max_length."""

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        import torch

        rows = [[len(w) for w in t.split()][:max_length] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        return {"input_ids": ids, "attention_mask": mask}


class _FakeModel:
    """last_hidden_state: (id, 1) for a real token, (100, 100) for padding (the mask must remove it)."""

    def __call__(self, input_ids, attention_mask):
        import torch
        from types import SimpleNamespace

        h = torch.stack([input_ids.float(), torch.ones_like(input_ids, dtype=torch.float)], dim=-1)
        h[attention_mask == 0] = 100.0
        return SimpleNamespace(last_hidden_state=h)


def test_sentence_model_embed_is_a_masked_mean_then_norm_1_in_the_order_of_the_texts():
    pytest.importorskip("torch")
    sm = W.SentenceModel("minilm", _FakeTokenizer(), _FakeModel())
    sm.dim = 2
    out = sm.embed(["aaa b", "cc", "dddd dddd dddd"], batch_size=2)
    want = np.array([[2.0, 1.0], [2.0, 1.0], [4.0, 1.0]])
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert out.dtype == np.float32 and np.allclose(out, want, atol=1e-6), "expected the masked mean, normalised"
    sm.prefixes["query"] = "zzzzzz "
    assert np.allclose(sm.embed(["b"], kind="query")[0], np.array([3.5, 1.0]) / np.linalg.norm([3.5, 1.0]), atol=1e-6)
    assert sm.embed([]).shape == (0, 2)
    with pytest.raises(ValueError, match="kind"):
        sm.embed(["x"], kind="document")


def test_passage_embeddings_are_cached_under_a_full_key(tmp_path):
    pytest.importorskip("torch")
    sm = W.SentenceModel("minilm", _FakeTokenizer(), _FakeModel())
    sm.dim = 2
    calls = []
    real = sm.embed
    sm.embed = lambda texts, kind=None, batch_size=32: calls.append(len(texts)) or real(texts, kind, batch_size)
    first = W.passage_embeddings(sm, ["a b", "ccc"], "toy", folder=tmp_path, verbose=False)
    again = W.passage_embeddings(sm, ["a b", "ccc"], "toy", folder=tmp_path, verbose=False)
    other = W.passage_embeddings(sm, ["a b", "cccc"], "toy", folder=tmp_path, verbose=False)
    assert calls == [2, 2], "expected one computation per set of passages, then the cache"
    assert np.array_equal(first, again) and not np.array_equal(first, other)


def test_passage_embeddings_are_computed_again_when_a_part_of_the_key_changes(tmp_path):
    pytest.importorskip("torch")
    sm = W.SentenceModel("e5", _FakeTokenizer(), _FakeModel())
    sm.dim = 2
    calls = []
    real = sm.embed
    sm.embed = lambda texts, kind=None, batch_size=32: calls.append(kind) or real(texts, kind, batch_size)
    passages = ["a b", "ccc"]
    W.passage_embeddings(sm, passages, "toy", folder=tmp_path, verbose=False)
    assert calls == ["passage"]
    W.passage_embeddings(sm, passages, "toy", kind="query", folder=tmp_path, verbose=False)      # another prefix
    W.passage_embeddings(sm, passages, "other", folder=tmp_path, verbose=False)                  # another subset
    sm.max_length = 128                                                                           # another truncation
    W.passage_embeddings(sm, passages, "toy", folder=tmp_path, verbose=False)
    sm.revision = "0" * 40                                                                        # another revision
    W.passage_embeddings(sm, passages, "toy", folder=tmp_path, verbose=False)
    assert len(calls) == 5, f"expected one computation per key, got {len(calls)} for 5 different keys"
    W.passage_embeddings(sm, passages, "toy", folder=tmp_path, verbose=False)
    assert len(calls) == 5, "expected the last key to be read from the cache"
    assert len(list(tmp_path.glob("*.npz"))) == 5, "expected one cache file per key"


def test_sentence_models_are_frozen_at_a_commit():
    assert set(W.SENTENCE_MODELS) == {"minilm", "e5"}
    for name, spec in W.SENTENCE_MODELS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), f"expected a full commit hash for {name}"
        assert spec["license"] in ("MIT", "Apache 2.0") and spec["dim"] == 384
    assert W.SENTENCE_MODELS["e5"]["revision"] == W.TOKENIZERS["e5"]["revision"], \
        "expected the tokenizer and the model of e5 read at the same commit"


def test_load_sentence_model_wraps_errors(monkeypatch):
    transformers = pytest.importorskip("transformers")

    def broken(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", broken)
    monkeypatch.setattr(W, "_SENTENCE_LOADED", {})
    with pytest.raises(W.SentenceModelUnavailable, match="OSError"):
        W.load_sentence_model("minilm")
    assert W._SENTENCE_LOADED == {}, "expected nothing kept after a failed load"


def test_try_sentence_model_falls_back_to_none_with_a_pending_line(monkeypatch, capsys):
    with pytest.raises(ValueError, match="unknown sentence model"):
        W.load_sentence_model("sbert")

    def no_network(name="minilm", *, reuse=True):
        raise W.SentenceModelUnavailable(f"modèle {name} impossible à charger : OSError : no network")

    monkeypatch.setattr(W, "load_sentence_model", no_network)
    assert W.try_sentence_model("minilm", "B2.28") is None
    assert capsys.readouterr().out.startswith("⏳ Ex B2.28 : modèle minilm impossible à charger")


@pytest.mark.network
def test_minilm_gives_the_cosines_of_the_fiche_and_reads_every_passage_whole():
    sm = W.load_sentence_model("minilm")
    assert sm.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" and not sm.model.training
    docs = ["The train to Lyon leaves at nine.", "My laptop will not start after the update.",
            "Our cat sleeps all day on the sofa.", "How do I reset a forgotten password?",
            "The bakery sells fresh bread every morning."]
    sims = sm.embed(["I can't log in to my account"]) @ sm.embed(docs).T
    assert np.round(np.sort(sims[0].astype(float))[::-1][:3], 3).tolist() == [0.502, 0.293, 0.042], "expected the cosines of the fiche"
    passages, _ = W.holmes_passages()
    assert max(len(sm.tokenizer(p)["input_ids"]) for p in passages) <= sm.max_length, "expected no passage truncated"


@pytest.mark.network
def test_minilm_word_vectors_contextual_and_static():
    sm = W.load_sentence_model("minilm")
    a = sm.word_vector("He sat on the bank of the river.", "bank")
    b = sm.word_vector("He works at the bank in the City.", "bank")
    assert a.shape == (384,) and a.dtype == np.float32 and not np.allclose(a, b), "expected one vector per use"
    batch = sm.tokenizer("He sat on the bank of the river.", return_tensors="pt")
    k = batch["input_ids"][0].tolist().index(sm.tokenizer.convert_tokens_to_ids("bank"))
    import torch
    with torch.inference_mode():
        want = sm.model(**batch).last_hidden_state[0, k].numpy()
    assert np.allclose(a, want, atol=1e-6), "expected the output of the last block at the token of the word"
    assert np.allclose(sm.static_vector("bank"), sm.model.embeddings.word_embeddings.weight[k and batch["input_ids"][0, k]].numpy())
    with pytest.raises(ValueError, match="not a word"):
        sm.word_vector("A banker.", "bank")
