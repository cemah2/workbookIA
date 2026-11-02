"""Small language models of bonus chapter B4 at frozen revisions, and a cache of their generated replies.

Quick reference
---------------
>>> llm = load_llm("smollm2")                # SmolLM2-135M-Instruct at its frozen revision, float32 on CPU, eval()
>>> llm = try_llm("qwen2.5", "B4.21")        # None, with a ⏳ line, when its files cannot be read or downloaded
>>> llm.render([{"role": "user", "content": "Who is Watson?"}])     # the chat template, as a string
>>> llm.generate("Who is Watson?", max_new_tokens=20)              # greedy reply (the prompt is a user message)
>>> llm.generate(messages, max_new_tokens=20, do_sample=True, temperature=0.7, seed=3)   # one random draw
>>> cache = GenerationCache("learner")        # data/downloads/llm/generations-learner.json (never versioned)
>>> llm.generate(messages, max_new_tokens=20, cache=cache)          # read from the cache, or generated then written
>>> with cache.disabled(): ...                # measure cold: neither read nor written
>>> key = llm.generation_key(messages, max_new_tokens=20)           # the complete key of one generation

The models (``LLMS``) are read at the commits frozen on 2026-11-02 (``data/cards/pretrained.md``). On CPU they are
loaded in float32 (faster than bfloat16 on a CPU without bf16 instructions, fiche §204.2). The weights are loaded
lazily, at the first generation that the cache cannot answer: a notebook whose replies are all cached never loads
them. ``reuse=False`` gives a fresh model (a fine-tuning must not change the model of the other exercises).

The cache key holds everything that decides a reply: the model, its revision and dtype, the chat template (its
SHA-256) or ``chat=False``, the exact prompt (the SHA-256 of its token ids), the effective generation settings (those of
the call completed by the model's ``generation_config.json``), the seed of a random draw, ``max_new_tokens``, the stop
strings, the versions of transformers and torch, and the **source** of the cache (``"learner"``, ``"solution"``…): each
source has its own file, so that the solutions notebook never reads nor writes the replies of the learner. A random
draw without a seed is never cached. The solutions notebook also reads a frozen, versioned file of replies
precomputed by ``tools/chapters/history_chB4.py`` (``SOLUTION_CACHE``), and writes nothing there.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from wb import datasets

LLMS = {
    "smollm2": {"repo": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
                "license": "Apache 2.0", "params": 134_515_008,
                "kind": "instruct, anglais, 30 couches, 135 M de paramètres (≈ 540 Mo en float32)"},
    "smollm2-base": {"repo": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                     "license": "Apache 2.0", "params": 134_515_008,
                     "kind": "modèle de base de SmolLM2-135M-Instruct (il continue du texte)"},
    "qwen2.5": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "license": "Apache 2.0", "params": 494_032_768,
                "kind": "instruct, multilingue, 24 couches, 494 M de paramètres (≈ 2 Go en float32)"},
    "qwen3": {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
              "license": "Apache 2.0", "params": 596_049_920,
              "kind": "instruct à mode de raisonnement, multilingue, 28 couches, 596 M de paramètres une fois chargé"},
    "distilgpt2": {"repo": "distilbert/distilgpt2", "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                   "license": "Apache 2.0", "params": 81_912_576,
                   "kind": "modèle de base, anglais, 6 couches (GPT-2 distillé), 82 M de paramètres"},
}
CACHE_VERSION = 1
SOLUTION_CACHE = "tools/chapters/history_chB4_generations.json"   # relative to the repository (read only)
SETTING_NAMES = ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty")
_SOURCE = re.compile(r"^[a-z0-9_-]{1,40}$")


class LLMUnavailable(RuntimeError):
    """A model cannot be loaded here (no network and no cache, a missing library)."""


def _versions() -> dict:
    import torch
    import transformers

    return {"transformers": transformers.__version__, "torch": torch.__version__.split("+")[0]}


def _sha(text: str, n: int = 32) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


class ChatLM:
    """A causal language model of ``LLMS`` with its tokenizer: ``tokenizer`` (loaded at once), ``model`` (loaded at
    first use, in ``eval()``), ``name``, ``repo``, ``revision`` and ``dtype``."""

    def __init__(self, name: str, tokenizer, dtype: str = "float32"):
        spec = LLMS[name]
        self.name, self.repo, self.revision = name, spec["repo"], spec["revision"]
        self.tokenizer, self.dtype = tokenizer, dtype
        self._model = None

    def __repr__(self) -> str:
        state = "chargé" if self._model is not None else "poids pas encore chargés"
        return f"ChatLM({self.name!r}: {self.repo} @ {self.revision[:7]}, {self.dtype}, {state})"

    # -- the weights, loaded at first use ------------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        if self._model is None:
            import torch

            from wb.transfer import quiet_hub
            try:
                with quiet_hub():
                    from transformers import AutoModelForCausalLM

                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.repo, revision=self.revision, dtype=getattr(torch, self.dtype)).eval()
            except Exception as exc:  # noqa: BLE001 - no network, a proxy, a missing package
                raise LLMUnavailable(f"modèle {self.name} ({self.repo}) impossible à charger : "
                                     f"{type(exc).__name__} : {str(exc).strip()[:300]}") from exc
        return self._model

    def unload(self) -> bool:
        """Free the weights (they are loaded again at the next generation that the cache cannot answer); True when
        weights were loaded. A notebook that trains other models calls it to keep its memory low."""
        if self._model is None:
            return False
        import gc

        self._model = None
        gc.collect()
        return True

    # -- prompts -------------------------------------------------------------------------------------------------
    @staticmethod
    def messages(prompt) -> list:
        """A list of messages: ``prompt`` itself, or ``[{"role": "user", "content": prompt}]`` for a string."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        messages = [dict(m) for m in prompt]
        if not messages or any(set(m) - {"role", "content"} or "role" not in m or "content" not in m for m in messages):
            raise ValueError("a prompt is a string or a non-empty list of {'role': ..., 'content': ...} messages")
        return messages

    def render(self, prompt, add_generation_prompt: bool = True) -> str:
        """The text that the chat template makes of ``prompt`` (a string is one user message)."""
        return self.tokenizer.apply_chat_template(self.messages(prompt), tokenize=False,
                                                  add_generation_prompt=add_generation_prompt)

    def prompt_ids(self, prompt, chat: bool = True) -> list:
        """The token ids given to the model: the chat template with ``add_generation_prompt=True`` (``chat=True``), or
        the raw text without special tokens added (``chat=False``, ``prompt`` must be a string)."""
        if chat:
            enc = self.tokenizer.apply_chat_template(self.messages(prompt), add_generation_prompt=True,
                                                     return_dict=True)
            return list(enc["input_ids"])
        if not isinstance(prompt, str):
            raise ValueError("chat=False takes a raw string")
        return list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    # -- settings and keys ---------------------------------------------------------------------------------------
    def settings(self, *, max_new_tokens: int, do_sample: bool = False, temperature=None, top_k=None, top_p=None,
                 repetition_penalty=None, stop_strings=None) -> dict:
        """The effective generation settings: those given, completed by the model's ``generation_config.json``
        (``repetition_penalty`` 1.1 for Qwen2.5, even in greedy mode). In greedy mode, temperature, top-k and top-p
        play no part and are left out."""
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens!r}")
        config = self._generation_defaults()
        given = {"do_sample": bool(do_sample), "temperature": temperature, "top_k": top_k, "top_p": top_p,
                 "repetition_penalty": repetition_penalty}
        out = {name: (config.get(name) if given[name] is None else given[name]) for name in SETTING_NAMES}
        out["do_sample"] = bool(do_sample)
        if do_sample:
            if not out["temperature"] or out["temperature"] <= 0:
                raise ValueError("a random draw needs a temperature > 0 (greedy decoding: do_sample=False)")
            out["temperature"] = float(out["temperature"])
            out["top_k"] = 0 if out["top_k"] is None else int(out["top_k"])
            out["top_p"] = 1.0 if out["top_p"] is None else float(out["top_p"])
        else:
            out["temperature"] = out["top_k"] = out["top_p"] = None
        out["repetition_penalty"] = float(out["repetition_penalty"] or 1.0)
        out["max_new_tokens"] = max_new_tokens
        out["stop_strings"] = list(stop_strings) if stop_strings else None
        return out

    def _generation_defaults(self) -> dict:
        """The model's ``generation_config.json`` as written in its repository (not the completed object of
        transformers, whose own defaults change with its version); ``{}`` for a model without one."""
        if getattr(self, "_defaults", None) is None:
            path = None
            with contextlib.suppress(Exception):
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(self.repo, "generation_config.json", revision=self.revision)
            self._defaults = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        return self._defaults

    def generation_key(self, prompt, *, chat: bool = True, seed=None, source: str = "learner", **settings) -> dict:
        """The complete key of one generation (see the module docstring)."""
        ids = self.prompt_ids(prompt, chat=chat)
        template = self.tokenizer.chat_template if chat else None
        effective = self.settings(**settings)
        return {"version": CACHE_VERSION, "source": source, "model": self.repo, "revision": self.revision,
                "dtype": self.dtype, "chat": bool(chat), "template": _sha(template, 16) if template else None,
                "prompt": _sha(_canonical(ids)), "n_prompt": len(ids), "settings": effective,
                "seed": seed if effective["do_sample"] else None, **_versions()}

    # -- generation ----------------------------------------------------------------------------------------------
    def generate(self, prompt, *, chat: bool = True, max_new_tokens: int = 64, do_sample: bool = False,
                 temperature=None, top_k=None, top_p=None, repetition_penalty=None, seed=None, stop_strings=None,
                 cache: "GenerationCache | None" = None) -> str:
        """The reply to ``prompt`` (a string, one user message, or a list of messages; ``chat=False``: a raw text
        continued without chat template): the generated tokens only, decoded without the special tokens. A stop string
        stays at the end of the reply, as with ``generate()``. ``do_sample=True`` draws at random with ``temperature``
        after ``torch.manual_seed(seed)``; ``top_k`` and ``top_p`` come from the call, else from the model's
        ``generation_config.json`` (Qwen2.5: 20 and 0.8), else no filter (0 and 1.0: SmolLM2)."""
        settings = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k,
                        top_p=top_p, repetition_penalty=repetition_penalty, stop_strings=stop_strings)
        use_cache = cache is not None and cache.enabled and not (do_sample and seed is None)
        key = None
        if use_cache:
            key = self.generation_key(prompt, chat=chat, seed=seed, source=cache.source, **settings)
            found = cache.get(key)
            if found is not None:
                return found
        text = self._generate(prompt, chat=chat, seed=seed, **settings)
        if use_cache:
            cache.put(key, text)
        return text

    def _generate(self, prompt, *, chat, seed, **settings) -> str:
        import torch

        from wb.text import quiet_transformers

        effective = self.settings(**settings)
        model = self.model
        ids = torch.tensor([self.prompt_ids(prompt, chat=chat)])
        config = copy.deepcopy(model.generation_config)
        for name in SETTING_NAMES:
            setattr(config, name, effective[name])
        config.max_new_tokens = effective["max_new_tokens"]
        if effective["stop_strings"]:
            config.stop_strings = effective["stop_strings"]
        if effective["do_sample"]:
            torch.manual_seed(0 if seed is None else int(seed))
        with quiet_transformers(), torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config,
                                 tokenizer=self.tokenizer if effective["stop_strings"] else None)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


_LOADED: dict = {}


def load_llm(name: str = "smollm2", *, dtype: str = "float32", reuse: bool = True) -> ChatLM:
    """The model ``name`` (a key of ``LLMS``) at its frozen revision: its tokenizer now, its weights at first use
    (``ChatLM.model``). Raises :class:`LLMUnavailable` when the tokenizer cannot be read or downloaded."""
    if name not in LLMS:
        raise ValueError(f"unknown model {name!r}; choose among {sorted(LLMS)}")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError(f"dtype must be 'float32' or 'bfloat16', got {dtype!r}")
    if reuse and (name, dtype) in _LOADED:
        return _LOADED[name, dtype]
    spec = LLMS[name]
    from wb.text import quiet_transformers
    try:
        with quiet_transformers():
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    except Exception as exc:  # noqa: BLE001 - no network, a proxy, a missing package: the caller falls back
        raise LLMUnavailable(f"modèle {name} ({spec['repo']}) impossible à charger : "
                             f"{type(exc).__name__} : {str(exc).strip()[:300]}") from exc
    llm = ChatLM(name, tok, dtype)
    if reuse:
        _LOADED[name, dtype] = llm
    return llm


def try_llm(name: str = "smollm2", ex_id: str | None = None, *, weights: bool = False) -> ChatLM | None:
    """``load_llm(name)``, or None with a ⏳ line when it cannot be loaded (to be validated on Colab). With
    ``weights=True`` the weights are loaded at once (an exercise that will generate anyway)."""
    try:
        llm = load_llm(name)
        if weights:
            llm.model  # noqa: B018 - load now
        return llm
    except LLMUnavailable as exc:
        where = f"Ex {ex_id} : " if ex_id else ""
        print(f"⏳ {where}{exc}. Sans ce modèle, cette partie est « à valider sur Colab » ; le reste du notebook "
              "continue.")
        return None


# ---------------------------------------------------------------------------
# The cache of generated replies
# ---------------------------------------------------------------------------
def cache_dir() -> Path:
    """Where the caches of replies go: ``wb.datasets.cache_dir() / "llm"`` (never versioned)."""
    path = datasets.cache_dir() / "llm"
    path.mkdir(parents=True, exist_ok=True)
    return path


class GenerationCache:
    """A JSON file of generated replies, one per ``source`` (``generations-<source>.json`` in :func:`cache_dir`),
    written atomically after each new reply; ``frozen`` adds a read-only file (the solutions notebook: the versioned
    replies precomputed by ``tools/chapters/history_chB4.py``). ``hits`` and ``misses`` count the lookups."""

    def __init__(self, source: str = "learner", folder: str | Path | None = None, frozen: str | Path | None = None):
        if not isinstance(source, str) or not _SOURCE.match(source):
            raise ValueError(f"source must be a short lower-case name ('learner', 'solution'…), got {source!r}")
        self.source = source
        self.path = (Path(folder) if folder is not None else cache_dir()) / f"generations-{source}.json"
        self.frozen_path = Path(frozen) if frozen is not None else None
        self.enabled = True
        self.hits = self.misses = 0
        self._frozen = self._read(self.frozen_path) if self.frozen_path is not None else {}

    def __repr__(self) -> str:
        frozen = f", {len(self._frozen)} figées" if self._frozen else ""
        return f"GenerationCache({self.source!r}: {len(self._read(self.path))} réponses{frozen}, {self.path})"

    @staticmethod
    def digest(key: dict) -> str:
        return hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()

    @staticmethod
    def _read(path: Path | None) -> dict:
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):                      # truncated or corrupt: start again
            return {}
        entries = data.get("entries", {}) if isinstance(data, dict) else {}
        return entries if isinstance(entries, dict) else {}

    def get(self, key: dict) -> str | None:
        """The reply stored for exactly ``key`` (the frozen file first), else None; a disabled cache answers None."""
        if not self.enabled:
            return None
        if key.get("source") != self.source:
            raise ValueError(f"this cache holds the source {self.source!r}, not {key.get('source')!r}")
        digest = self.digest(key)
        for entries in (self._frozen, self._read(self.path)):
            entry = entries.get(digest)
            if isinstance(entry, dict) and entry.get("key") == json.loads(_canonical(key)) and isinstance(entry.get("text"), str):
                self.hits += 1
                return entry["text"]
        self.misses += 1
        return None

    def put(self, key: dict, text: str) -> None:
        """Store ``text`` for ``key`` (nothing when the cache is disabled): the file is read again, completed, then
        replaced atomically (a temporary file, then ``os.replace``)."""
        if not self.enabled:
            return
        if key.get("source") != self.source:
            raise ValueError(f"this cache holds the source {self.source!r}, not {key.get('source')!r}")
        if not isinstance(text, str):
            raise TypeError("a cached reply is a string")
        entries = self._read(self.path)
        entries[self.digest(key)] = {"key": json.loads(_canonical(key)), "text": text}
        write_json_atomic(self.path, {"version": CACHE_VERSION, "source": self.source, "entries": entries})

    def __len__(self) -> int:
        return len(self._read(self.path)) + len(self._frozen)

    def clear(self) -> int:
        """Remove this source's file (never the frozen one); the number of replies removed."""
        n = len(self._read(self.path))
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        return n

    @contextlib.contextmanager
    def disabled(self):
        """Neither read nor written inside the block (a cold measurement); the previous state comes back after."""
        previous, self.enabled = self.enabled, False
        try:
            yield self
        finally:
            self.enabled = previous


def write_json_atomic(path: str | Path, obj) -> None:
    """Write ``obj`` as JSON (UTF-8, sorted keys) into ``path``: a temporary file in the same folder, then
    ``os.replace``, so that a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=0, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
