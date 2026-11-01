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
