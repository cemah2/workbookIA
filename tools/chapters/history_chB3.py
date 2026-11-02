#!/usr/bin/env python
"""Calibrate the predictions, replay the bug hunt and precompute the long trainings of chapter B3, first half (used by
Claude).

    python tools/chapters/history_chB3.py                  # everything (about 3 h 30 on a 2-core CPU)
    python tools/chapters/history_chB3.py --only p22 p24   # some entries (names below)

Writes tools/chapters/history_chB3.json (statistics only, no weights), which chB3_parts.py reads:
- "p13": 🔮 B3.13, the median weight of the copied key over 200 queries for 10 to 100 000 keys, with and without the
  division by sqrt(d_k), for the seed of the notebook and 10 other seeds; both answers must be the same for every seed;
- "bug17": 🐛 B3.17 replayed on its 2**3 partially fixed versions: the first point that fails must be the first error
  left (nothing must crash), and on two versions where two errors cancel out (the mask of nn.MultiheadAttention passed
  as is: point 4); the clipping of the probe of B3.20 must act (and change the weights);
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
Second half (parts C to E, the cells of chB3_code2.py; the calibration models, the reference mini-GPT trained in fast
mode with seeds 2320 to 2322, are cached in wbgpt.checkpoints_dir() under the source "calibration", the full-mode one is
wbgpt.reference_model(False)):
- "p22": 🔮 B3.22 on the four models and four series of draws: both answers must agree everywhere;
- "p24": 📈 B3.24, the three scores of every head (previous character, spaces, entropy) of each model;
- "ablations": 🔬 B3.25, the five variants in the fast protocol of the notebook and in the full protocol (≈ 36 min);
- "bpe26": 🔬 B3.26, characters against BPE 512 at equal budget, both protocols (≈ 11 min);
- "vit28": 📦 B3.28, the ViT in fast mode (three seeds, the threshold) and in full mode (≈ 7 min);
- "challenge30": 🏆 B3.30, the reference method run by evaluate_30 (fast: five seeds, full: three; ≈ 50 min), the goal
  (worst seed + 5 %) and the gap of the control.
The cells of the notebook are reused as is (tools/chapters/chB3_code.py, chB3_code2.py), executed in one namespace.
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
        want = next((k for k in (1, 2, 3) if k not in fixed), None)          # point 4 passes once 1 to 3 are fixed
        rows.append({"fixed": "".join(map(str, fixed)), "first_failure": first, "expected": want,
                     "seconds": round(time.perf_counter() - start, 2), "lines": text.strip().splitlines()})
        print(f"  bug17 fixed {rows[-1]['fixed'] or '-':<4} first failing point {first} (expected {want}) "
              f"in {rows[-1]['seconds']} s", flush=True)
        for line in text.strip().splitlines():
            print("     ", line[:200])
        if first != want or "Traceback" in text:
            raise SystemExit(f"❌ bug17: version {fixed} shows {fails}")
    # two errors that cancel out: the mask of nn.MultiheadAttention (triu(1)) passed without "~", with and without
    # the fix of the loss; the model computes the right function, only point 4 (the docstring of the mask) sees it
    for fixed, want in (("2b", 3), ("2b3", 4)):
        src = C.TODO_17.replace(EDITS_17[2][0], "return torch.ones(T, T, dtype=torch.bool).triu(1)")
        if "3" in fixed:
            src = src.replace(EDITS_17[3][0], EDITS_17[3][1])
        local = dict(ns)
        exec(src, local)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(C.CHECK_17, local)
        text = out.getvalue()
        fails = [line for line in text.splitlines() if line.startswith("❌")]
        first = int(fails[0].split("point ")[1].split(" ")[0]) if fails else None
        rows.append({"fixed": fixed, "first_failure": first, "expected": want, "seconds": None,
                     "lines": text.strip().splitlines()})
        print(f"  bug17 compensated {fixed:<4} first failing point {first} (expected {want})", flush=True)
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


# --- the second half of the chapter (B3.21 to B3.30) --------------------------------------------------------------------
CALIBRATION_SEEDS = (2320, 2321, 2322)


def _second_namespace(ns, fast: bool = True) -> dict:
    """The namespace of parts C to E: the solutions of B3.18 to B3.20, then the tools of part C."""
    import chB3_code2 as C2
    ns = _gpt_namespace(ns, fast)
    ns["model_ok_19"], ns["probes_ok_20"], ns["CKPT_SOURCE_20"] = True, True, "solution"
    with contextlib.redirect_stdout(io.StringIO()):
        exec(C2.TOOLS_C, ns)
    return ns


def _calibration_models(ns) -> dict:
    """The reference mini-GPT trained in fast mode with three seeds (the first is the model of the solution, bit for
    bit), cached in wbgpt.checkpoints_dir() under the source "calibration", and the reference full-mode model."""
    import torch
    wbgpt, data, models = ns["wbgpt"], ns["DATA"], {}
    for seed in CALIBRATION_SEEDS:
        settings = wbgpt.train_settings(True) | {"seed": seed}
        key = wbgpt.checkpoint_key(wbgpt.mini_config(True), settings, fast=True, source="calibration")
        ckpt = wbgpt.load_checkpoint(key)
        if ckpt is None:
            torch.manual_seed(seed)
            model = wbgpt.ReferenceGPT(**wbgpt.mini_config(True))
            history = wbgpt.train(model, data.train_ids, data.val_ids, settings, torch.Generator().manual_seed(seed))
            wbgpt.save_checkpoint(key, model, history, data.itos, extra={"char_loss": wbgpt.char_loss(model, data.val_ids)})
            ckpt = wbgpt.load_checkpoint(key)
        models[f"fast {seed}"] = wbgpt.build_from_checkpoint(wbgpt.ReferenceGPT, ckpt)
    models["full"] = wbgpt.reference_model(False, verbose=False)[0]
    return models


def p22(ns) -> dict:
    """🔮 B3.22 on four models (three fast seeds, the full mode) and four generator seeds: both answers must agree."""
    import chB3_code2 as C2
    import torch
    ns = _second_namespace(ns)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(C2.GIVEN_22, ns)
    rows, answers = [], set()
    for name, model in _calibration_models(ns).items():
        french = ns["FRENCH_22"]
        loss = ns["wbgpt"].char_loss(model, french)
        majority = []
        for gen_seed in (22, 0, 1, 2):
            start = ns["DATA"].encode(ns["PROMPT_FR_22"])[None].repeat(10, 1)
            out = ns["wbgpt"].generate(model, start, 120, temperature=1.0, generator=torch.Generator().manual_seed(gen_seed))
            counts = [ns["languages_22"](ns["DATA"].decode(row[start.shape[1]:])) for row in out]
            majority.append(sum(f > e for e, f in counts))
        row = {"model": name, "french_loss": loss, "uniform": math.log(88), "majority_french": majority,
               "buckets": ns["loss_by_position_22"](model, french)}
        rows.append(row)
        answers.add((max(majority) > 5, loss < math.log(88), int(loss)))
        print(f"  p22 {name}: loss {loss:.3f}, majority french {majority}, buckets {np.round(row['buckets'], 3)}", flush=True)
    if len(answers) != 1:
        raise SystemExit(f"❌ p22: the answers depend on the model or the seed: {answers}")
    french, beats, k = answers.pop()
    return {"rows": rows, "answers": {"french": bool(french), "beats_uniform": bool(beats), "integer_part": k}}


def p24(ns) -> dict:
    """📈 B3.24: the three scores of every head (previous character, spaces, entropy) for each model."""
    import chB3_code2 as C2
    import torch
    ns = _second_namespace(ns)
    out = {}
    for name, model in _calibration_models(ns).items():
        ns["GPT"] = model
        with contextlib.redirect_stdout(io.StringIO()):
            exec(C2.GIVEN_24.split("\nif GPT is None:")[0], ns)
        maps = ns["attention_maps_24"](model, ns["WINDOWS_24"])
        rows = torch.arange(1, 128)
        scores = {"previous": maps[..., rows, rows - 1].mean(dim=(0, -1)),
                  "spaces": (maps * ns["SPACES_24"][:, None, None, None, :]).sum(-1)[..., 32:].mean(dim=(0, -1)),
                  "spread": -(maps * maps.clamp_min(1e-12).log()).sum(-1)[..., 32:].mean(dim=(0, -1))}
        out[name] = {k: v.numpy().round(3).tolist() for k, v in scores.items()}
        for k, v in scores.items():
            flat = sorted(v.flatten().tolist(), reverse=True)
            print(f"  p24 {name} {k}: best {flat[0]:.3f}, second {flat[1]:.3f}, within 10 % of the best: "
                  f"{sum(x >= 0.9 * flat[0] for x in flat)}", flush=True)
    return out


def _protocol_runs(ns, ex: str, mode: str, train_one) -> dict:
    import chB3_code2 as C2
    P = C2.PROTOCOLS[ex][mode]
    results, seconds = {}, {}
    for seed in P["seeds"]:
        for name, loss, sec in train_one(P, seed):
            results.setdefault(name, []).append(loss)
            seconds[name] = seconds.get(name, 0.0) + sec
            print(f"  {ex} {mode} seed {seed} {name}: {loss:.4f} ({sec:.0f} s)", flush=True)
    return {"losses": results, "seconds": {k: round(v / len(P["seeds"]), 1) for k, v in seconds.items()}}


def ablations(ns) -> dict:
    """🔬 B3.25: the five variants in both protocols (fast: the notebook; full: the 🚀 mode), with the solution."""
    import chB3_code2 as C2
    import torch
    ns = _second_namespace(ns)
    exec(C2.GIVEN_25.replace("«PROTOCOL_FAST_25»", "None").replace("«PROTOCOL_FULL_25»", "None")
         .replace("«FULL_25»", "None"), ns)
    exec(C2.SOLUTION_25, ns)
    wbgpt, data = ns["wbgpt"], ns["DATA"]

    def train_one(P, seed):
        for name in ns["VARIANTS_25"]:
            torch.manual_seed(seed)
            model = ns["make_variant_25"](name, P["config"])
            start = time.perf_counter()
            ns["train_20"](model, data.train_ids, data.val_ids, P["settings"], torch.Generator().manual_seed(seed),
                           block_size=P["config"]["block_size"])
            sec = time.perf_counter() - start
            yield name, wbgpt.char_loss(model, data.val_ids[:P["val_chars"]], block_size=P["config"]["block_size"]), sec

    return {mode: _protocol_runs(ns, "25", mode, train_one) for mode in ("fast", "full")}


def bpe26(ns) -> dict:
    """🔬 B3.26: characters against BPE 512 at equal budget, in both protocols, with the solution."""
    import chB3_code2 as C2
    import torch
    ns = _second_namespace(ns)
    ns["wb"] = __import__("wb")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(C2.GIVEN_26.replace("«PROTOCOL_FAST_26»", "None").replace("«PROTOCOL_FULL_26»", "None")
             .replace("«FULL_26»", "None"), ns)
    exec(C2.SOLUTION_26, ns)
    wbgpt, data, tok = ns["wbgpt"], ns["DATA"], ns["TOK_26"]
    cpt = ns["chars_per_token_26"](tok, data.train_text[:100_000])
    train_bpe = torch.tensor(tok.encode(data.train_text))

    def train_one(P, seed):
        block_bpe = ns["budget_block_26"](cpt, P["char_block"])
        val_bpe = torch.tensor(tok.encode(data.val_text[:P["val_chars"]]))
        for name, vocab, block, train_ids, val_ids in (("caractères", 88, P["char_block"], data.train_ids, data.val_ids),
                                                        ("BPE 512", 512, block_bpe, train_bpe, val_bpe)):
            torch.manual_seed(seed)
            model = ns["MiniGPT_19"](**(P["config"] | {"vocab_size": vocab, "block_size": block}))
            start = time.perf_counter()
            ns["train_20"](model, train_ids, val_ids, P["settings"], torch.Generator().manual_seed(seed), block_size=block)
            sec = time.perf_counter() - start
            if name == "caractères":
                loss = wbgpt.char_loss(model, data.val_ids[:P["val_chars"]], block_size=block)
            else:
                loss = ns["nats_per_char_26"](model, val_bpe, tok, block)
            yield name, loss, sec

    out = {mode: _protocol_runs(ns, "26", mode, train_one) for mode in ("fast", "full")}
    out["chars_per_token"] = {"train_100k": cpt, "val": ns["chars_per_token_26"](tok, data.val_text)}
    return out


VIT_SEEDS = (28, 29, 30)
VIT_FULL_EPOCHS = 12


def vit28(ns) -> dict:
    """📦 B3.28: the ViT of the solution, fast mode with three seeds (the threshold), full mode with one."""
    import chB3_code2 as C2
    import torch
    out = {}
    for fast in (True, False):
        local = _second_namespace(dict(ns), fast)
        src = C2.GIVEN_28.replace("«EPOCHS_FULL_28»", str(VIT_FULL_EPOCHS)).replace("«TARGET_28»", "0.0") \
            .replace("«TARGET_FULL_28»", "0.0")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(src, local)
        exec(C2.SOLUTION_28, local)
        runs = []
        for seed in (VIT_SEEDS if fast else VIT_SEEDS[:1]):
            torch.manual_seed(seed)
            model = local["ViT_28"]()
            start = time.perf_counter()
            losses = local["train_vit_28"](model, local["X_TRAIN_28"], local["Y_TRAIN_28"], local["EPOCHS_28"], seed=seed)
            sec = time.perf_counter() - start
            acc = local["accuracy_28"](model, local["X_TEST_28"], local["Y_TEST_28"])
            runs.append({"seed": seed, "accuracy": acc, "seconds": round(sec, 1), "losses": losses,
                         "params": sum(p.numel() for p in model.parameters())})
            print(f"  vit28 {'fast' if fast else 'full'} seed {seed}: {acc:.4f} in {sec:.0f} s", flush=True)
        out["fast" if fast else "full"] = runs
    worst = min(r["accuracy"] for r in out["fast"])
    out["target_fast"] = math.floor((worst - 0.02) * 100) / 100           # 2 points under the worst seed
    out["target_full"] = math.floor((out["full"][0]["accuracy"] - 0.02) * 100) / 100
    return out


CHALLENGE_SEEDS = {"fast": (3030, 3031, 3032, 3033, 3034), "full": (3030, 3031, 3032)}


def challenge30(ns, modes=("fast", "full")) -> dict:
    """🏆 B3.30: the reference method run by evaluate_30 itself (time limits, control) with several seeds, per mode."""
    import chB3_code2 as C2
    out = {}
    for mode in modes:
        local = _second_namespace(dict(ns), mode == "fast")
        src = C2.GIVEN_30
        for k, v in {"«GOAL_FAST_30»": "9.9", "«GOAL_FULL_30»": "9.9", "«REF_FAST_30»": "(0.0, 0.0)",
                     "«REF_FULL_30»": "(0.0, 0.0)", "«CONTROL_GAP_FAST_30»": "(0.0, 9.9)",
                     "«CONTROL_GAP_FULL_30»": "(0.0, 9.9)"}.items():
            src = src.replace(k, v)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(src, local)
        exec(C2.SOLUTION_30, local)
        runs = []
        for seed in CHALLENGE_SEEDS[mode]:
            local["SEED_30"] = seed
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                result = local["evaluate_30"](local["method_30"])
            fails = [line for line in text.getvalue().splitlines() if line.startswith("❌")]
            runs.append(result | {"seed": seed, "failures": fails})
            print(f"  challenge30 {mode} seed {seed}: {result['loss']:.4f} in {result['seconds']:.0f} s, control "
                  f"{result['control_val']:.3f} / {result['control_held']:.3f} in {result['control_seconds']:.0f} s, "
                  f"failures {fails}", flush=True)
        losses = [r["loss"] for r in runs]
        gaps = [r["control_val"] - r["control_held"] for r in runs]
        out[mode] = {"runs": runs, "reference": (min(losses), max(losses)),
                     "goal": math.ceil(max(losses) * 1.05 * 100) / 100,
                     "control_gap": (float(np.mean(gaps)), max(0.08, round(3 * (max(gaps) - min(gaps)), 2)))}
    return out


ENTRIES = {"p13": (("GIVEN_13",), p13), "bug17": (("GIVEN_17",), bug17), "gpt_fast": ((), gpt_fast),
           "gpt_full": ((), gpt_full), "lstm": ((), lstm), "p22": ((), p22), "p24": ((), p24),
           "ablations": ((), ablations), "bpe26": ((), bpe26), "vit28": ((), vit28), "challenge30": ((), challenge30)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(ENTRIES), nargs="+")
    args = parser.parse_args(argv)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name, (blocks, fn) in ENTRIES.items():
        if args.only and name not in args.only:
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
