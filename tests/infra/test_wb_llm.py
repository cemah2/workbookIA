"""Tests of wb.llm (chapter B4): the frozen revisions, the generation settings, the cache of replies (keys, sources,
atomic writes, frozen file) and, with --run-network, a real greedy generation of SmolLM2."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from wb import llm as wbllm

ROOT = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    """Just enough of a tokenizer for the keys: a chat template and token ids."""

    chat_template = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"

    def apply_chat_template(self, messages, add_generation_prompt=True, return_dict=False, tokenize=True):
        text = "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + ("<|assistant|>" if add_generation_prompt else "")
        if not tokenize:
            return text
        return {"input_ids": [ord(c) for c in text]}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def fake_llm(defaults=None, name="smollm2"):
    """A ChatLM whose generation is a counter (no weights): ``llm.calls`` counts the real generations."""
    llm = wbllm.ChatLM(name, FakeTokenizer())
    llm._defaults = defaults or {}
    llm.calls = 0

    def generate(prompt, *, chat, seed, **settings):
        llm.calls += 1
        return f"reply {llm.calls} (seed {seed})"

    llm._generate = generate
    return llm


def test_revisions_are_frozen_commits_written_in_the_card():
    card = (ROOT / "data" / "cards" / "pretrained.md").read_text(encoding="utf-8")
    for name, spec in wbllm.LLMS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), name
        assert spec["revision"] in card, f"the revision of {name} is not in data/cards/pretrained.md"
        assert spec["license"] and spec["params"] > 0


def test_settings_complete_the_call_with_the_generation_config():
    llm = fake_llm({"do_sample": True, "temperature": 0.7, "top_k": 20, "top_p": 0.8, "repetition_penalty": 1.1})
    greedy = llm.settings(max_new_tokens=8)
    assert greedy == {"do_sample": False, "temperature": None, "top_k": None, "top_p": None,
                      "repetition_penalty": 1.1, "max_new_tokens": 8, "stop_strings": None}
    drawn = llm.settings(max_new_tokens=8, do_sample=True)
    assert (drawn["temperature"], drawn["top_k"], drawn["top_p"]) == (0.7, 20, 0.8)
    plain = fake_llm({})
    drawn = plain.settings(max_new_tokens=8, do_sample=True, temperature=1.5)
    assert (drawn["top_k"], drawn["top_p"], drawn["repetition_penalty"]) == (0, 1.0, 1.0), "no filter by default"
    with pytest.raises(ValueError):
        plain.settings(max_new_tokens=8, do_sample=True)          # a draw needs a temperature
    with pytest.raises(ValueError):
        plain.settings(max_new_tokens=0)


def test_key_holds_everything_that_decides_a_reply():
    llm = fake_llm()
    key = llm.generation_key("Who is Watson?", max_new_tokens=8, source="test")
    for field in ("model", "revision", "dtype", "template", "prompt", "n_prompt", "settings", "seed", "transformers",
                  "torch", "source"):
        assert field in key
    assert key["seed"] is None, "a greedy reply has no seed"
    assert llm.generation_key("Who is Watson ?", max_new_tokens=8, source="test")["prompt"] != key["prompt"]
    assert llm.generation_key("Who is Watson?", max_new_tokens=9, source="test")["settings"] != key["settings"]
    raw = llm.generation_key("Who is Watson?", chat=False, max_new_tokens=8, source="test")
    assert raw["template"] is None and raw["prompt"] != key["prompt"]
    drawn = llm.generation_key("Who is Watson?", max_new_tokens=8, do_sample=True, temperature=0.7, seed=3,
                               source="test")
    assert drawn["seed"] == 3


def test_cache_reads_a_reply_once_generated(tmp_path):
    llm, cache = fake_llm(), wbllm.GenerationCache("test", folder=tmp_path)
    first = llm.generate("Hello", max_new_tokens=8, cache=cache)
    assert llm.generate("Hello", max_new_tokens=8, cache=cache) == first
    assert llm.calls == 1 and (cache.hits, cache.misses) == (1, 1) and len(cache) == 1
    llm.generate("Hello", max_new_tokens=8, do_sample=True, temperature=0.7, seed=1, cache=cache)
    llm.generate("Hello", max_new_tokens=8, do_sample=True, temperature=0.7, seed=2, cache=cache)
    llm.generate("Hello", max_new_tokens=8, do_sample=True, temperature=0.7, seed=1, cache=cache)
    assert llm.calls == 3, "one generation per seed"
    llm.generate("Hello", max_new_tokens=8, do_sample=True, temperature=0.7, cache=cache)
    llm.generate("Hello", max_new_tokens=8, do_sample=True, temperature=0.7, cache=cache)
    assert llm.calls == 5, "a draw without a seed is never cached"
    assert not list(tmp_path.glob("*.tmp")), "no temporary file left"


def test_cache_disabled_neither_reads_nor_writes(tmp_path):
    llm, cache = fake_llm(), wbllm.GenerationCache("test", folder=tmp_path)
    llm.generate("Hello", max_new_tokens=8, cache=cache)
    with cache.disabled():
        llm.generate("Hello", max_new_tokens=8, cache=cache)
        llm.generate("Other", max_new_tokens=8, cache=cache)
    assert llm.calls == 3 and len(cache) == 1 and cache.enabled
    assert cache.clear() == 1 and len(cache) == 0


def test_sources_are_separate_and_the_frozen_file_is_read_only(tmp_path):
    llm = fake_llm()
    solution = wbllm.GenerationCache("solution", folder=tmp_path)
    llm.generate("Hello", max_new_tokens=8, cache=solution)
    frozen = tmp_path / "frozen.json"
    (tmp_path / "generations-solution.json").rename(frozen)
    before = frozen.read_bytes()

    reader = wbllm.GenerationCache("solution", folder=tmp_path / "local", frozen=frozen)
    assert llm.generate("Hello", max_new_tokens=8, cache=reader) == "reply 1 (seed None)" and llm.calls == 1
    llm.generate("New", max_new_tokens=8, cache=reader)
    assert frozen.read_bytes() == before, "the frozen file is never written"
    assert (tmp_path / "local" / "generations-solution.json").exists()

    learner = wbllm.GenerationCache("learner", folder=tmp_path / "local", frozen=None)
    llm.generate("Hello", max_new_tokens=8, cache=learner)
    assert llm.calls == 3, "the learner's cache never reads the replies of the solution"
    with pytest.raises(ValueError):
        learner.get(llm.generation_key("Hello", max_new_tokens=8, source="solution"))
    with pytest.raises(ValueError):
        wbllm.GenerationCache("Learner!")


def test_a_corrupt_cache_file_is_started_again(tmp_path):
    llm, cache = fake_llm(), wbllm.GenerationCache("test", folder=tmp_path)
    cache.path.write_text("{not json", encoding="utf-8")
    assert len(cache) == 0
    llm.generate("Hello", max_new_tokens=8, cache=cache)
    assert json.loads(cache.path.read_text(encoding="utf-8"))["source"] == "test"


def test_the_frozen_solution_cache_is_versioned_with_solution_keys():
    path = ROOT / wbllm.SOLUTION_CACHE
    if not path.exists():
        pytest.skip("tools/chapters/history_chB4.py has not written the frozen cache yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["source"] == "solution" and data["entries"]
    for digest, entry in data["entries"].items():
        assert entry["key"]["source"] == "solution"
        assert wbllm.GenerationCache.digest(entry["key"]) == digest
        assert entry["key"]["revision"] in {spec["revision"] for spec in wbllm.LLMS.values()}


def test_without_network_try_llm_says_pending_and_the_weights_raise_llm_unavailable(monkeypatch, capsys):
    """No network and no local copy: try_llm gives None with a ⏳ line (the notebook goes on), and weights that
    cannot be read raise LLMUnavailable, never the raw error of the Hub."""
    import transformers

    def offline(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files (simulated)")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", offline)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", offline)
    monkeypatch.setattr(wbllm, "_LOADED", {})
    assert wbllm.try_llm("qwen2.5", "B4.21") is None
    text = capsys.readouterr().out
    assert text.startswith("⏳ Ex B4.21 : modèle qwen2.5") and "à valider sur Colab" in text
    llm = wbllm.ChatLM("smollm2", FakeTokenizer())
    with pytest.raises(wbllm.LLMUnavailable, match="impossible à charger"):
        llm.model  # noqa: B018


def test_the_exercise_notebook_uses_the_learner_cache_and_never_the_frozen_file():
    """The cache line of chapter B4: the exercise notebook writes the source "learner" and reads no frozen file; only
    the solutions notebook reads the frozen replies, under the source "solution"."""
    def code(name):
        cells = json.loads((ROOT / "chapitres" / "b4_llm" / name).read_text(encoding="utf-8"))["cells"]
        return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")

    exercise, solution = code("03_notebook.ipynb"), code("05_solutions.ipynb")
    assert 'GenerationCache("learner")' in exercise
    assert "SOLUTION_CACHE" not in exercise and "frozen=" not in exercise and '"solution"' not in exercise
    assert 'GenerationCache("solution", frozen=wb.repo_root() / wbllm.SOLUTION_CACHE)' in solution
    assert 'GenerationCache("learner")' not in solution


def test_unknown_model_or_dtype_is_refused():
    with pytest.raises(ValueError):
        wbllm.load_llm("gpt-5")
    with pytest.raises(ValueError):
        wbllm.load_llm("smollm2", dtype="float16")


@pytest.mark.network
def test_smollm2_greedy_reply_is_deterministic_and_cached(tmp_path):
    llm = wbllm.load_llm("smollm2", reuse=False)
    cache = wbllm.GenerationCache("test", folder=tmp_path)
    first = llm.generate("Who is Sherlock Holmes?", max_new_tokens=8, cache=cache)
    with cache.disabled():
        again = llm.generate("Who is Sherlock Holmes?", max_new_tokens=8, cache=cache)
    assert first == again and first.strip()
    assert llm.generate("Who is Sherlock Holmes?", max_new_tokens=8, cache=cache) == first and cache.hits == 1
    assert llm.model.dtype.__str__() == "torch.float32" and not llm.model.training
