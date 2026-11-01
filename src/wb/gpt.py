"""The mini-GPT of bonus chapter B3: Holmes as character ids, batches, a sliding validation loss, the single checkpoint
of the trained model, and a reference model for the exercises that need a trained model when yours is missing.

Quick reference
---------------
>>> data = char_data("holmes")              # the frozen split of wb.text.train_val, the vocabulary of build_vocab (ch. 22)
>>> data.train_ids, data.val_ids             # 1-D int64 tensors; data.stoi, data.itos, data.fingerprint
>>> x, y = get_batch(data.train_ids, 32, 128, generator)   # the get_batch of the fiche (§203.8)
>>> loss = char_loss(model, data.val_ids)    # nats per character, every target from index 128 on (sliding windows)
>>> key = learner_key(fast=True)             # the complete key of the model trained by Ex B3.20
>>> save_checkpoint(key, model, history, data.itos)   # written atomically, outside the repository
>>> ckpt = load_checkpoint(key)              # None when absent; else {"state_dict", "config", "itos", "history", ...}
>>> model, ckpt = reference_model(fast=True) # the reference mini-GPT, trained once (≈ 2 min on CPU), then reloaded
>>> config = mini_config(fast=True)          # MINI_CONFIG, with a dropout of 0.2 in full mode

The configuration of the contract (``MINI_CONFIG``: 4 blocks, 4 heads, d = 128, context 128, characters of Holmes) and
the training settings (``train_settings``: 600 iterations of 32 sequences in fast mode, 5 000 in full mode) are those
of Ex B3.19 and B3.20. ``ReferenceGPT``, ``configure_optimizer``, ``estimate_loss`` and ``train`` are the reference
solution of these two exercises (nanoGPT structure, A. Karpathy, MIT licence): the notebook only calls them when
your own model is missing, and you read them, if you wish, once your own version works.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from wb import datasets
from wb import text as wbtext

CHECKPOINT_VERSION = 1
MINI_CONFIG = {"vocab_size": 88, "block_size": 128, "n_layer": 4, "n_head": 4, "d_model": 128, "dropout": 0.0}
SEED = 2320            # the seed of the initialisation and of the training batches (Ex B3.20)
EVAL_SEED = 2321       # every estimate of the validation loss draws its batches again from this seed
SLIDING_STRIDE = 32    # char_loss: a window every 32 characters, each scoring its last 32 targets


def mini_config(fast: bool = True) -> dict:
    """The configuration of the mini-GPT of Ex B3.19 and B3.20: ``MINI_CONFIG``, with a dropout of 0.2 in full mode
    (5 000 iterations are about 40 passes over the training text: without dropout, the validation loss rises again
    after about 2 500 iterations), none in fast mode (600 iterations, far from overfitting)."""
    return MINI_CONFIG | {"dropout": 0.0 if fast else 0.2}


def train_settings(fast: bool = True) -> dict:
    """The training settings of Ex B3.20: AdamW (β2 = 0.99, weight decay 0.1 on the parameters with two axes or
    more), linear warmup then cosine decay from 5e-3 to 5e-4, clipping at a norm of 1, batches of 32 sequences of
    128 characters; 600 iterations in fast mode, 5 000 in full mode."""
    return {"iters": 600 if fast else 5000, "batch_size": 32, "lr": 5e-3, "min_lr": 5e-4,
            "warmup": 50 if fast else 200, "weight_decay": 0.1, "beta2": 0.99, "clip": 1.0,
            "eval_interval": 50 if fast else 250, "eval_batches": 8 if fast else 20,
            "seed": SEED, "eval_seed": EVAL_SEED}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def char_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    """``(stoi, itos)`` of the characters of ``text`` by decreasing frequency, ties in ``sorted`` order: the rule of
    ``build_vocab`` of ``mylearn.sequences`` (ch. 22)."""
    counts = Counter(text)
    itos = sorted(counts, key=lambda ch: (-counts[ch], ch))
    return {ch: i for i, ch in enumerate(itos)}, itos


@dataclass(frozen=True)
class CharData:
    """Holmes (or Verne) as character ids: the frozen split of ``wb.text.train_val`` and the vocabulary of the whole
    text (88 characters for Holmes, as in ch. 22)."""

    name: str
    train_text: str
    val_text: str
    stoi: dict
    itos: list
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    fingerprint: str

    def encode(self, text: str) -> torch.Tensor:
        """The int64 ids of ``text`` (ValueError on a character outside the vocabulary)."""
        missing = sorted(set(text) - set(self.stoi))
        if missing:
            raise ValueError(f"characters outside the vocabulary of {self.name}: {missing[:5]}")
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, ids) -> str:
        """The text of a sequence of ids (a tensor, an array or a list)."""
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(self.itos[int(i)] for i in values)


_DATA: dict = {}


def split_fingerprint(train_text: str, val_text: str) -> str:
    """16 hexadecimal digits of the SHA-256 of the split (train, then val)."""
    h = hashlib.sha256(train_text.encode("utf-8"))
    h.update(b"\0")
    h.update(val_text.encode("utf-8"))
    return h.hexdigest()[:16]


def char_data(name: str = "holmes") -> CharData:
    """The text ``name`` (``"holmes"`` or ``"verne"``) as character ids, built once per session."""
    if name not in _DATA:
        train, val = wbtext.train_val(name)
        stoi, itos = char_vocab(train + val)
        encode = lambda t: torch.tensor([stoi[ch] for ch in t], dtype=torch.long)   # noqa: E731
        _DATA[name] = CharData(name, train, val, stoi, itos, encode(train), encode(val),
                               split_fingerprint(train, val))
    return _DATA[name]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator | None = None):
    """Random windows of ``data`` (a 1-D tensor of ids): inputs (N, T) and targets shifted by one token (§203.8)."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# The validation loss of a whole text (comparison with the LSTM of ch. 22)
# ---------------------------------------------------------------------------
def sliding_windows(n: int, block_size: int = 128, stride: int = SLIDING_STRIDE) -> list[tuple[int, int, int]]:
    """``(start, first, last)`` for each window of :func:`char_loss` on a text of ``n`` ids: the window reads the
    inputs ``[start, start + block_size)`` and scores the targets ``first`` to ``last`` (included), so that every
    target from ``block_size`` to ``n - 1`` is scored once, with at least ``block_size - stride + 1`` characters of
    context."""
    if not 1 <= stride <= block_size:
        raise ValueError(f"stride must be in [1, block_size = {block_size}], got {stride}")
    if n <= block_size:
        raise ValueError(f"the text must have more than block_size = {block_size} ids, got {n}")
    windows, done, start = [], block_size - 1, 0
    while done < n - 1:
        start = min(start, n - 1 - block_size)
        last = start + block_size
        windows.append((start, done + 1, last))
        done = last
        start += stride
    return windows


@torch.no_grad()
def char_loss(model: nn.Module, ids: torch.Tensor, block_size: int = 128, stride: int = SLIDING_STRIDE,
              batch_size: int = 64) -> float:
    """Mean cross-entropy (nats per character) of ``model`` on the targets ``ids[block_size:]``, each predicted from
    the ``block_size - stride + 1`` to ``block_size`` characters that precede it (windows every ``stride`` characters).

    ``model(x)`` takes (N, T) int64 ids and returns logits (N, T, V); it is evaluated in ``eval()`` mode, then put
    back in its former mode. Deterministic: no random draw."""
    windows = sliding_windows(len(ids), block_size, stride)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    try:
        for b in range(0, len(windows), batch_size):
            chunk = windows[b:b + batch_size]
            x = torch.stack([ids[s:s + block_size] for s, _, _ in chunk]).to(device)
            y = torch.stack([ids[s + 1:s + 1 + block_size] for s, _, _ in chunk]).to(device)
            nll = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten(), reduction="none").view(len(chunk), -1)
            for row, (s, first, last) in enumerate(chunk):
                # the logits at local position j predict ids[s + j + 1]: targets first..last are j = first-s-1..last-s-1
                total += float(nll[row, first - s - 1:last - s].double().sum())
                count += last - first + 1
    finally:
        model.train(was_training)
    return total / count


# ---------------------------------------------------------------------------
# The reference model (solution of Ex B3.18 and B3.19, nanoGPT structure)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with ``F.scaled_dot_product_attention``: ``qkv`` (d → 3d: queries, keys,
    values), ``proj`` (d → d), dropout on the weights (training only) and on the output."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_head:
            raise ValueError(f"n_head = {n_head} must divide d_model = {d_model}")
        self.n_head, self.dropout = n_head, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(N, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(N, T, D)))


class Block(nn.Module):
    """Pre-LN block: ``x + attn(ln1(x))``, then ``+ mlp(ln2(x))`` with a GELU MLP of width 4d."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ReferenceGPT(nn.Module):
    """The mini-GPT of the contract: token and learned position embeddings, pre-LN blocks, final layer norm, output
    layer tied to the token table; nanoGPT initialisation (N(0, 0.02²), zero biases, residual projections scaled by
    1 / sqrt(2 n_layer)). ``forward(idx)`` returns the logits (N, T, V)."""

    def __init__(self, vocab_size: int, block_size: int = 128, n_layer: int = 4, n_head: int = 4,
                 d_model: int = 128, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                    # tied: one parameter
        self.apply(self._init_weights)
        for block in self.blocks:                                  # the two projections that write in the residual path
            for proj in (block.attn.proj, block.mlp[2]):
                nn.init.normal_(proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        if T > self.block_size:
            raise ValueError(f"sequence of {T} tokens, more than block_size = {self.block_size}")
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def count_params(config: dict | None = None) -> int:
    """The number of parameters of the mini-GPT of ``config`` (tied output layer counted once)."""
    c = {**MINI_CONFIG, **(config or {})}
    d, n_layer = c["d_model"], c["n_layer"]
    block = (4 * d * d + 4 * d) + (8 * d * d + 5 * d) + 4 * d
    return n_layer * block + c["vocab_size"] * d + c["block_size"] * d + 2 * d


def expected_state_shapes(config: dict | None = None) -> dict[str, tuple[int, ...]]:
    """The names and shapes of the ``state_dict`` of a mini-GPT that follows the contract of Ex B3.19."""
    c = {**MINI_CONFIG, **(config or {})}
    d, V = c["d_model"], c["vocab_size"]
    shapes = {"tok_emb.weight": (V, d), "pos_emb.weight": (c["block_size"], d)}
    for i in range(c["n_layer"]):
        p = f"blocks.{i}."
        shapes |= {p + "ln1.weight": (d,), p + "ln1.bias": (d,), p + "attn.qkv.weight": (3 * d, d),
                   p + "attn.qkv.bias": (3 * d,), p + "attn.proj.weight": (d, d), p + "attn.proj.bias": (d,),
                   p + "ln2.weight": (d,), p + "ln2.bias": (d,), p + "mlp.0.weight": (4 * d, d),
                   p + "mlp.0.bias": (4 * d,), p + "mlp.2.weight": (d, 4 * d), p + "mlp.2.bias": (d,)}
    shapes |= {"ln_f.weight": (d,), "ln_f.bias": (d,), "head.weight": (V, d)}
    return shapes


# ---------------------------------------------------------------------------
# The reference training (solution of Ex B3.20)
# ---------------------------------------------------------------------------
def lr_at(it: int, total: int, lr: float, warmup: int, min_lr: float) -> float:
    """Linear warmup from 0 to ``lr`` in ``warmup`` iterations, then cosine decay to ``min_lr`` at ``total``: the
    ``cosine_lr`` of ``mylearn.optim`` (ch. 19)."""
    if it < warmup:
        return lr * it / warmup
    if it >= total:
        return min_lr
    progress = (it - warmup) / (total - warmup)
    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def configure_optimizer(model: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.99)) -> torch.optim.AdamW:
    """AdamW with two groups: ``weight_decay`` on the parameters with two axes or more, 0 on the others."""
    params = list(model.parameters())                              # each tied parameter once
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=betas)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over every position: logits (N, T, V) and targets (N, T) flattened."""
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, n_batches: int, batch_size: int, block_size: int,
                  generator: torch.Generator) -> float:
    """Mean loss of ``model`` on ``n_batches`` batches of ``get_batch``, in eval mode, then back to train mode."""
    model.eval()
    losses = [lm_loss(model(x), y).item()
              for x, y in (get_batch(data, batch_size, block_size, generator) for _ in range(n_batches))]
    model.train()
    return sum(losses) / len(losses)


def train(model: nn.Module, train_ids: torch.Tensor, val_ids: torch.Tensor, settings: dict,
          generator: torch.Generator, block_size: int = 128, verbose: bool = False) -> dict:
    """Train ``model`` in place; return its history (train loss and learning rate of every iteration, validation
    loss before the first iteration, every ``eval_interval`` iterations and after the last)."""
    s = settings
    optimizer = configure_optimizer(model, s["lr"], s["weight_decay"], (0.9, s["beta2"]))
    history = {"train_loss": [], "lr": [], "val_iter": [], "val_loss": []}

    def evaluate(done: int) -> None:
        history["val_iter"].append(done)
        history["val_loss"].append(estimate_loss(model, val_ids, s["eval_batches"], s["batch_size"], block_size,
                                                 torch.Generator().manual_seed(s["eval_seed"])))
        if verbose:
            print(f"iteration {done:5d} · validation {history['val_loss'][-1]:.4f}", flush=True)

    model.train()
    evaluate(0)
    for it in range(s["iters"]):
        lr = lr_at(it, s["iters"], s["lr"], s["warmup"], s["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_ids, s["batch_size"], block_size, generator)
        loss = lm_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), s["clip"])
        optimizer.step()
        history["train_loss"].append(loss.item())
        history["lr"].append(lr)
        if (it + 1) % s["eval_interval"] == 0 or it + 1 == s["iters"]:
            evaluate(it + 1)
    return history


# ---------------------------------------------------------------------------
# The single checkpoint of the trained mini-GPT (outside the repository, atomic)
# ---------------------------------------------------------------------------
def checkpoints_dir() -> Path:
    """Where the trained mini-GPTs go: ``wb.datasets.cache_dir() / "gpt"`` (never versioned)."""
    path = datasets.cache_dir() / "gpt"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical(key: dict) -> str:
    return json.dumps(key, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def checkpoint_key(config: dict, settings: dict, *, fast: bool, source: str, data: CharData | None = None) -> dict:
    """The complete key of a trained model: its configuration, its training settings (seed included), the mode,
    who trained it (``"learner"`` or ``"reference"``) and the fingerprints of the split and of the vocabulary."""
    data = data or char_data("holmes")
    vocab = hashlib.sha256("".join(data.itos).encode("utf-8")).hexdigest()[:16]
    return {"version": CHECKPOINT_VERSION, "source": str(source), "fast": bool(fast), "config": dict(config),
            "settings": dict(settings), "split": data.fingerprint, "vocab": vocab, "text": data.name}


def learner_key(fast: bool = True) -> dict:
    """The key of the model trained and saved by Ex B3.20 (the configuration and the settings of the contract)."""
    return checkpoint_key(mini_config(fast), train_settings(fast), fast=fast, source="learner")


def reference_key(fast: bool = True) -> dict:
    """The key of the reference mini-GPT (``reference_model``)."""
    return checkpoint_key(mini_config(fast), train_settings(fast), fast=fast, source="reference")


def checkpoint_file(key: dict, folder: str | Path | None = None) -> Path:
    """The ``.pt`` file of a key: its source, its mode and 16 hexadecimal digits of the SHA-256 of the key."""
    digest = hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()[:16]
    mode = "fast" if key.get("fast") else "full"
    base = Path(folder) if folder is not None else checkpoints_dir()
    return base / f"minigpt-{key.get('source', 'model')}-{mode}-{digest}.pt"


def save_checkpoint(key: dict, model: nn.Module, history: dict, itos: list, folder: str | Path | None = None,
                    extra: dict | None = None) -> Path:
    """Save the ``state_dict`` of ``model``, the configuration of the key, the vocabulary and the history in the file
    of ``key``: written to a temporary file, then renamed (``os.replace``), so that an interrupted save never leaves
    a half-written checkpoint. Returns the path."""
    path = checkpoint_file(key, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}
    payload = {"key": _canonical(key), "config": dict(key["config"]), "itos": list(itos), "history": history,
               "state_dict": state, "extra": dict(extra or {}), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "__complete__": True}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_checkpoint(key: dict, folder: str | Path | None = None) -> dict | None:
    """The checkpoint of ``key`` (``state_dict``, ``config``, ``itos``, ``history``, ``extra``, ``path``), or None when
    there is none, when the file holds another key, or when it is incomplete or unreadable (any doubt: None)."""
    path = checkpoint_file(key, folder)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("__complete__") is not True or payload.get("key") != _canonical(key):
            return None
    except Exception:  # noqa: BLE001 - truncated, corrupt, another format: as if absent
        return None
    payload["path"] = path
    return payload


def build_from_checkpoint(model_class, ckpt: dict) -> nn.Module:
    """``model_class(**config)`` with the weights of ``ckpt`` loaded (strictly), in eval mode."""
    model = model_class(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def reference_model(fast: bool = True, folder: str | Path | None = None, verbose: bool = True):
    """``(model, checkpoint)`` of the reference mini-GPT: read from its checkpoint, or trained once with the settings
    of Ex B3.20 and the seed ``SEED`` (≈ 2 min on two CPU cores in fast mode, ≈ 35 min in full mode), then saved.
    The model is a :class:`ReferenceGPT` in eval mode."""
    key = reference_key(fast)
    ckpt = load_checkpoint(key, folder)
    if ckpt is None:
        data = char_data("holmes")
        if verbose:
            print(f"ℹ️ entraînement du mini-GPT de référence ({train_settings(fast)['iters']} itérations, une seule fois)",
                  flush=True)
        torch.manual_seed(SEED)
        model = ReferenceGPT(**mini_config(fast))
        started = time.perf_counter()
        history = train(model, data.train_ids, data.val_ids, train_settings(fast), torch.Generator().manual_seed(SEED))
        seconds = time.perf_counter() - started
        val = char_loss(model, data.val_ids)
        save_checkpoint(key, model, history, data.itos, folder, extra={"char_loss": val, "seconds": seconds})
        ckpt = load_checkpoint(key, folder)
    elif verbose:
        print(f"ℹ️ mini-GPT de référence relu ({ckpt['path'].name})")
    return build_from_checkpoint(ReferenceGPT, ckpt), ckpt
