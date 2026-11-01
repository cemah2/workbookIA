#!/usr/bin/env python
"""Calibrate the predictions, replay the bug hunt and precompute the long trainings of chapter B3, first half (used by
Claude).

    python tools/chapters/history_chB3.py                  # everything (about 65 min on a 2-core CPU)
    python tools/chapters/history_chB3.py --only bug17     # one entry: p13, bug17, gpt_fast, gpt_full, lstm

Writes tools/chapters/history_chB3.json (statistics only, no weights), which chB3_parts.py reads:
- "p13": 🔮 B3.13, the median weight of the copied key over 200 queries for 10 to 100 000 keys, with and without the
  division by sqrt(d_k), for the seed of the notebook and 10 other seeds; both answers must be the same for every seed;
- "bug17": 🐛 B3.17 replayed on its 2**3 partially fixed versions: the first point that fails must be the first error
  left (nothing must crash); the clipping of the probe of B3.20 must act (and change the weights);
- "gpt_fast": the mini-GPT of the solution (B3.19 and B3.20) trained in fast mode with three seeds: loss on the whole
  validation (wb.gpt.char_loss), time, history of the main seed; the threshold TARGET_20 of fast mode;
- "gpt_full": the same in full mode (5 000 iterations, dropout 0.2, one seed, ≈ 35 min): the 🚀 row of the comparison, the
  threshold of full mode, and the history that the second half of the chapter (📈 B3.21) can draw;
- "gpt_full_nodropout": the same full mode without dropout (the configuration of fast mode), kept from a run of
  session 68 and not recomputed by this script (run gpt_full with mini_config(True) to redo it): the validation
  loss is lowest around 2 500 iterations, then rises again (overfitting), 1 118 s;
- "lstm": the character LSTM of ch. 22 (solution of 22.25: two LSTM layers of 128, windows of 40 characters, stride 3,
  RMSprop, clipping), retrained on the split and the vocabulary of B3 in its fast mode (2 epochs on 240 000 characters)
  and in its full mode (the first 12 of its 20 epochs on the whole training text, the best epoch kept: in 22.26 the
  validation loss is lowest at epoch 9), and evaluated on the same validation
  targets as the mini-GPT (every character of the validation from index 128 on, read with its 40 previous characters).
The cells of the notebook are reused as is (tools/chapters/chB3_code.py), executed in one namespace.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(ROOT / "solutions")]
OUT = HERE / "history_chB3.json"

# The three fixes of 🐛 B3.17, one text edit each (point k of diagnose_17 checks fix k)
EDITS_17 = {
    1: ("h, self.weights = self.attn(x, x, x, attn_mask=causal_mask_17(T))",
        "h, self.weights = self.attn(x, x, x, attn_mask=~causal_mask_17(T))"),
    2: ("return torch.ones(T, T, dtype=torch.bool).triu()", "return torch.ones(T, T, dtype=torch.bool).tril()"),
    3: ("return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))",
        "return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD_17)"),
}
P13_SEEDS = (13, *range(10))
LSTM_FULL_EPOCHS = 12        # the full mode of 22.25 has 20 epochs; its validation loss is lowest at epoch 9 (22.26)


def namespace(blocks=()) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wb

    import chB3_code as C

    plt.show = lambda *a, **k: plt.close("all")
    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True), "FAST_MODE": True, "__name__": "history_chB3",
          "mylearn": wb.load_mylearn("ref")}
    for block in ("TOOLS_A", *blocks):
        exec(getattr(C, block), ns)
    return ns


def p13(ns) -> dict:
    keys = list(ns["KEYS_13"])
    runs = {}
    for seed in P13_SEEDS:
        runs[seed] = {name: [float(np.median(ns["copied_weight_13"](n, scaled, seed=seed))) for n in keys]
                      for name, scaled in (("scaled", True), ("unscaled", False))}
    answers = set()
    for seed, run in runs.items():
        # the first power of ten whose median is under 1/2, minus one: the crossing is between 10**k and 10**(k + 1)
        powers = [n for n in keys if n in (10, 100, 1000, 10_000, 100_000)]
        meds = dict(zip(keys, run["scaled"]))
        first_power = next(n for n in powers if meds[n] < 0.5)
        exponent = int(round(math.log10(first_power))) - 1
        answers.add((exponent, run["unscaled"][-1] > 0.5))
        print(f"  p13 seed {seed}: scaled {np.round(run['scaled'], 3).tolist()} → k = {exponent}; "
              f"unscaled at 100 000: {run['unscaled'][-1]:.6f}", flush=True)
    if len(answers) != 1:
        raise SystemExit(f"❌ p13: the answers depend on the seed: {answers}")
    exponent, keeps = answers.pop()
    at = {n: [runs[s]["scaled"][keys.index(n)] for s in P13_SEEDS] for n in (10 ** exponent, 10 ** (exponent + 1))}
    margin = min(min(at[10 ** exponent]) - 0.5, 0.5 - max(at[10 ** (exponent + 1)]))
    if margin < 0.05:
        raise SystemExit(f"❌ p13: the crossing is too close to a power of ten (margin {margin:.3f})")
    return {"keys": keys, "runs": {str(s): r for s, r in runs.items()}, "margin": margin,
            "answers": {"exponent": exponent, "keeps_half": bool(keeps)}}


def bug17(ns) -> dict:
    import chB3_code as C
    import torch
    rows = []
    for mask in itertools.product((False, True), repeat=3):
        fixed = [k for k, on in zip((1, 2, 3), mask) if on]
        src = C.TODO_17
        for k in fixed:
            if EDITS_17[k][0] not in src:
                raise SystemExit(f"❌ bug17: the edit {k} does not apply to TODO_17")
            src = src.replace(EDITS_17[k][0], EDITS_17[k][1])
        local = dict(ns)
        exec(src, local)
        out = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(out):
            exec(C.CHECK_17, local)
        text = out.getvalue()
        fails = [line for line in text.splitlines() if line.startswith("❌")]
        first = int(fails[0].split("point ")[1].split(" ")[0]) if fails else None
        want = next((k for k in (1, 2, 3) if k not in fixed), None)
        rows.append({"fixed": "".join(map(str, fixed)), "first_failure": first, "expected": want,
                     "seconds": round(time.perf_counter() - start, 2), "lines": text.strip().splitlines()})
        print(f"  bug17 fixed {rows[-1]['fixed'] or '-':<4} first failing point {first} (expected {want}) "
              f"in {rows[-1]['seconds']} s", flush=True)
        for line in text.strip().splitlines():
            print("     ", line[:200])
        if first != want or "Traceback" in text:
            raise SystemExit(f"❌ bug17: version {fixed} shows {fails}")
    # the probe of B3.20: the clipping at 0.05 acts at every iteration, and it changes the weights
    exec(C.TOOLS_B + "\n" + C.GIVEN_20 + "\n" + C.SOLUTION_20, ns)
    wbgpt, data = ns["wbgpt"], ns["DATA"]
    small = {"vocab_size": 88, "block_size": 16, "n_layer": 2, "n_head": 2, "d_model": 16, "dropout": 0.0}
    probe = ns["TRAIN_20"] | {"iters": 6, "warmup": 2, "eval_interval": 4, "eval_batches": 2, "batch_size": 4, "clip": 0.05}
    torch.manual_seed(20)
    base = wbgpt.ReferenceGPT(**small)
    import copy
    norms = []
    clip_ = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        total = clip_(params, max_norm, *a, **k)
        norms.append(float(total))
        return total
    torch.nn.utils.clip_grad_norm_ = spy
    try:
        clipped = copy.deepcopy(base)
        wbgpt.train(clipped, data.train_ids, data.val_ids, probe, torch.Generator().manual_seed(7), block_size=16)
    finally:
        torch.nn.utils.clip_grad_norm_ = clip_
    unclipped = copy.deepcopy(base)
    wbgpt.train(unclipped, data.train_ids, data.val_ids, probe | {"clip": 1e9}, torch.Generator().manual_seed(7),
                block_size=16)
    mine = copy.deepcopy(base)
    ns["train_20"](mine, data.train_ids, data.val_ids, probe, torch.Generator().manual_seed(7), block_size=16)
    gap_clip = max((a - b).abs().max().item() for a, b in zip(clipped.parameters(), unclipped.parameters()))
    gap_sol = max((a - b).abs().max().item() for a, b in zip(clipped.parameters(), mine.parameters()))
    print(f"  probe of B3.20: gradient norms {np.round(norms, 3).tolist()}, clipped - unclipped {gap_clip:.2e}, "
          f"solution - wb.gpt {gap_sol:.2e}", flush=True)
    if min(norms) <= 0.05 or gap_clip < 1e-4 or gap_sol > 1e-7:
        raise SystemExit("❌ bug17: the probe of B3.20 does not separate a clipped training from an unclipped one")
    return {"versions": rows, "probe20": {"grad_norms": norms, "gap_clip": gap_clip, "gap_solution": gap_sol}}


def _gpt_namespace(ns, fast: bool) -> dict:
    import chB3_code as C
    ns["FAST_MODE"] = fast
    for block in ("TOOLS_B", "SOLUTION_18", "SOLUTION_19", "GIVEN_20", "SOLUTION_20"):
        exec(getattr(C, block), ns)
    return ns


def _train_gpt(ns, seed: int, fast: bool) -> dict:
    import torch
    settings = ns["wbgpt"].train_settings(fast) | {"seed": seed}
    torch.manual_seed(seed)
    model = ns["MiniGPT_19"](**ns["CONFIG_20"])
    start = time.perf_counter()
    history = ns["train_20"](model, ns["DATA"].train_ids, ns["DATA"].val_ids, settings,
                             torch.Generator().manual_seed(seed))
    seconds = time.perf_counter() - start
    loss = ns["wbgpt"].char_loss(model, ns["DATA"].val_ids)
    print(f"  gpt {'fast' if fast else 'full'} seed {seed}: {seconds:.0f} s, char_loss {loss:.4f}, last estimate "
          f"{history['val_loss'][-1]:.4f}", flush=True)
    return {"seed": seed, "seconds": round(seconds, 1), "char_loss": loss, "history": history,
            "params": sum(p.numel() for p in model.parameters())}


def gpt_fast(ns) -> dict:
    ns = _gpt_namespace(ns, True)
    seed = ns["wbgpt"].SEED
    runs = [_train_gpt(ns, s, True) for s in (seed, seed + 1, seed + 2)]
    worst = max(r["char_loss"] for r in runs)
    target = math.ceil((worst + 0.08) * 20) / 20                 # a margin of 0.08 nat, rounded up to 0.05
    main = runs[0]
    return {"runs": [{k: v for k, v in r.items() if k != "history"} for r in runs], "history": main["history"],
            "target": target}


def gpt_full(ns) -> dict:
    ns = _gpt_namespace(ns, False)
    run = _train_gpt(ns, ns["wbgpt"].SEED, False)
    target = math.ceil((run["char_loss"] + 0.08) * 20) / 20
    s = ns["wbgpt"].train_settings(False)
    return {"name": "mini-GPT de référence, mode complet (dropout 0,2)", "params": run["params"],
            "budget": f"{s['iters']:_} itérations, {run['seconds'] / 60:.0f} min (CPU)".replace("_", " "), "char_loss": run["char_loss"],
            "seconds": run["seconds"], "history": run["history"], "target": target}


# --- the LSTM of ch. 22 (solution of 22.25), on the split and the vocabulary of B3 -------------------------------------
def _lstm_parts():
    import torch
    import torch.nn.functional as F
    from torch import nn

    class CharLSTM(nn.Module):
        """The generator of 22.25: one-hot -> nn.LSTM(V, 128, 2 layers) -> nn.Linear on the last step."""

        def __init__(self, vocab_size, hidden_size=128, num_layers=2):
            super().__init__()
            self.vocab_size = vocab_size
            self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, vocab_size)

        def forward(self, ids):
            out, _ = self.lstm(F.one_hot(ids, self.vocab_size).float())
            return self.head(out[:, -1])

    def train_epoch(model, X, y, optimizer, generator, batch_size=128):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(X), generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X[batch]), y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        return total / len(X)

    @torch.no_grad()
    def b3_loss(model, val_ids, block_size=128, window=40, batch_size=2000):
        """The targets of wb.gpt.char_loss (every validation character from index block_size on), each read with its
        `window` previous characters."""
        model.eval()
        targets = torch.arange(block_size, len(val_ids))
        total = 0.0
        for chunk in targets.split(batch_size):
            X = torch.stack([val_ids[p - window:p] for p in chunk.tolist()])
            total += F.cross_entropy(model(X), val_ids[chunk], reduction="sum").item()
        return total / len(targets)

    return CharLSTM, train_epoch, b3_loss


def lstm(ns) -> dict:
    import torch
    CharLSTM, train_epoch, b3_loss = _lstm_parts()
    data = ns["DATA"]

    def windows(ids, n):
        ids = ids[:n]
        starts = torch.arange(0, len(ids) - 40, 3)
        X = torch.stack([ids[s:s + 40] for s in starts.tolist()])
        return X, ids[starts + 40]

    rows, curves = [], {}
    for mode, n_chars, epochs in (("fast", 240_000, 2), ("full", len(data.train_ids), LSTM_FULL_EPOCHS)):
        torch.manual_seed(2225)
        model = CharLSTM(len(data.itos))
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
        generator = torch.Generator().manual_seed(0)
        X, y = windows(data.train_ids, n_chars)
        trained, curve = 0.0, []                          # training seconds, the evaluations left out
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            train_loss = train_epoch(model, X, y, optimizer, generator)
            trained += time.perf_counter() - start
            val = b3_loss(model, data.val_ids) if (mode == "full" or epoch == epochs) else None
            curve.append({"epoch": epoch, "train_loss": train_loss, "val_char_loss": val, "seconds": round(trained, 1)})
            print(f"  lstm {mode} epoch {epoch}: train {train_loss:.4f}, validation {val}, {trained:.0f} s of training",
                  flush=True)
        params = sum(p.numel() for p in model.parameters())
        if mode == "fast":
            rows.append({"name": "LSTM du ch. 22, mode rapide", "params": params,
                         "budget": f"2 epochs sur 240 000 car., {curve[-1]['seconds'] / 60:.1f} min".replace(".", ","),
                         "char_loss": curve[-1]["val_char_loss"]})
        else:
            best = min(curve, key=lambda e: e["val_char_loss"])
            rows.append({"name": "LSTM du ch. 22, mode complet (meilleure epoch)", "params": params,  # of 12
                         "budget": f"{best['epoch']} epochs sur tout l'entraînement, {best['seconds'] / 60:.0f} min",
                         "char_loss": best["val_char_loss"]})
        curves[mode] = curve
    return {"rows": rows, "curves": curves, "windows": {"fast": len(windows(data.train_ids, 240_000)[0]),
                                                        "full": len(windows(data.train_ids, len(data.train_ids))[0])}}


ENTRIES = {"p13": (("GIVEN_13",), p13), "bug17": (("GIVEN_17",), bug17), "gpt_fast": ((), gpt_fast),
           "gpt_full": ((), gpt_full), "lstm": ((), lstm)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(ENTRIES))
    args = parser.parse_args(argv)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name, (blocks, fn) in ENTRIES.items():
        if args.only and args.only != name:
            continue
        start = time.perf_counter()
        print(f"{name}…", flush=True)
        data[name] = fn(namespace(blocks))
        print(f"{name}: {time.perf_counter() - start:.0f} s", flush=True)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
