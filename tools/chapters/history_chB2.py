#!/usr/bin/env python
"""Calibrate the predictions, replay the bug hunt and measure the times of chapter B2 (used by Claude).

    python tools/chapters/history_chB2.py                 # everything (about 6 min on a 2-core CPU, the Hub reachable)
    python tools/chapters/history_chB2.py --only bug18    # one entry: p14, bug18, timing, w2v, c31

Writes tools/chapters/history_chB2.json (statistics only), which chB2_parts.py reads:
- "p14": 🔮 B2.14, the counts of tokens of the six real tokenizers on the four texts (two frozen excerpts, the numbers,
  the emoji) and the answers of the three predictions; each measure must be far enough from the bounds of its
  categories (it is deterministic: frozen excerpts and frozen revisions);
- "bug18": 🐛 B2.18 replayed on its 2**3 partially fixed versions: the first point that fails must be the first error
  left (and nothing must crash); the fixed version must also encode by rank exactly like mylearn (reference);
- "timing": the training times of the reference BPETokenizer, of a naive version (every pair counted again and every
  word merged again after each merge, the way a learner first writes it), of tokenizers and of wb.text.byte_bpe_merges,
  for the statements, the hints and the solutions;
- "w2v": 🔮 B2.26 and 📈 B2.27, the word2vec of the solution (B2.25) trained with 4 seeds and 3 other settings (batch
  size, learning rate): the two nearest neighbours of « fogg », the characters among the neighbours of « passepartout »,
  the rank correlation between the frequency and the first axis of the PCA, the frequent words grouped on the UMAP;
  each answer must be the same in every run, with a margin;
- "c31": 🏆 B2.31, the reference method (no pre-tokenization inside a line, shortest segmentation) and the baseline of
  B2.20 on the frozen split and on 4 other splits of Verne, the goal checked against both with a margin, and the
  anti-leak rules tried on two methods that read the validation.
The cells of the notebook are reused as is (tools/chapters/chB2_code.py), executed in one namespace.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(ROOT / "solutions")]
OUT = HERE / "history_chB2.json"

# The three fixes of the tokenizer of 🐛 B2.18, one text edit each (point k of diagnose_18 checks fix k)
EDITS_18 = {
    1: ('    return "".join(vocab[i].decode("utf-8") for i in ids)',
        '    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")'),
    2: ('    return unicodedata.normalize("NFD", text)', '    return unicodedata.normalize("NFC", text)'),
    3: ("""        i = 0
        while i < len(ids) - 1:
            pair = (ids[i], ids[i + 1])
            if pair in rank:
                ids[i:i + 2] = [256 + rank[pair]]
                i = 0
            else:
                i += 1
""", """        while len(ids) > 1:
            best = min(zip(ids, ids[1:]), key=lambda p: rank.get(p, math.inf))
            if best not in rank:
                break
            ids = merge_18(ids, best, 256 + rank[best])
"""),
}

# 🔮 B2.14: the categories of the statement (chB2_code.TODO_14) and the margin each measure must keep from their bounds
P14_BOUNDS = {"ratio": (0.9, 1.1), "bert_emoji": (10, 2000)}


def namespace(blocks=(), blocks_de=()) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wb

    import chB2_code as C

    plt.show = lambda *a, **k: plt.close("all")
    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True), "FAST_MODE": True, "__name__": "history_chB2",
          "mylearn": wb.load_mylearn("ref")}
    for block in ("TOOLS", *blocks):
        exec(getattr(C, block), ns)
    if blocks_de:
        import chB2_code_de as D
        for block in ("TOOLS_D", *blocks_de):
            exec(getattr(D, block), ns)
    return ns


def _letter(value, bounds, letters="ABC"):
    low, high = bounds
    return letters[0] if value < low else (letters[1] if value <= high else letters[2])


def p14(ns) -> dict:
    from wb import text as wbtext
    counts = {}
    for name in wbtext.TOKENIZERS:
        tok = wbtext.load_tokenizer(name)
        counts[name] = {label: wbtext.count_tokens(tok, t) for label, t in ns["TEXTS_14"].items()}
        print(f"  p14 {name} {counts[name]}", flush=True)
    sizes = {label: ns["n_bytes"](t) for label, t in ns["TEXTS_14"].items()}
    per_token = {label: sizes[label] / counts["gpt2"][label] for label in sizes}
    order = sorted(per_token, key=per_token.get)
    if per_token[order[1]] - per_token[order[0]] < 0.5:
        raise SystemExit(f"❌ p14 a): the two cheapest texts are too close: {per_token}")
    ratio = counts["camembert"]["Verne"] / counts["camembert"]["Holmes"]
    bert = counts["bert"]["emoji"]
    for value, (low, high) in ((ratio, P14_BOUNDS["ratio"]), (bert, P14_BOUNDS["bert_emoji"])):
        if min(abs(value - low) / low, abs(value - high) / high) < 0.15:
            raise SystemExit(f"❌ p14: {value} is within 15 % of a bound of its categories {low, high}")
    answers = {"costliest": "ABCD"[["Holmes", "Verne", "nombres", "emoji"].index(order[0])],
               "ratio": _letter(ratio, P14_BOUNDS["ratio"]), "bert_emoji": _letter(bert, P14_BOUNDS["bert_emoji"])}
    print(f"  p14 bytes per GPT-2 token {per_token}, CamemBERT Verne/Holmes {ratio:.3f}, BERT emoji {bert}: {answers}", flush=True)
    return {"counts": counts, "bytes": sizes, "gpt2_bytes_per_token": {k: round(v, 3) for k, v in per_token.items()},
            "ratio": round(ratio, 4), "answers": answers}


def bug18(ns) -> dict:
    import chB2_code as C
    rows = []
    for mask in itertools.product((0, 1), repeat=3):
        src = C.TODO_18
        for k, on in zip((1, 2, 3), mask):
            if on:
                assert EDITS_18[k][0] in src, f"edit {k} does not apply"
                src = src.replace(EDITS_18[k][0], EDITS_18[k][1])
        exec(src, ns)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = ns["diagnose_18"](ns["train_18"], ns["encode_18"], ns["decode_18"])
        text = out.getvalue()
        fails = [line for line in text.splitlines() if line.startswith("❌")]
        first = int(fails[0].split("point ")[1].split(" ")[0]) if fails else None
        want = next((k for k, on in zip((1, 2, 3), mask) if not on), None)
        rows.append({"fixed": "".join(str(k) for k, on in zip((1, 2, 3), mask) if on), "first_failure": first,
                     "expected": want, "all_pass": bool(ok), "message": fails[0] if fails else ""})
        print(f"  bug18 fixed {rows[-1]['fixed'] or '-':<4} first failing point {first} (expected {want})", flush=True)
        if first != want or len(fails) > 1:
            raise SystemExit(f"❌ bug18: version {mask} shows {fails}")
    # the fixed version encodes like the reference library (byte level, 'gpt2' pre-tokenizer, same merges)
    exec(C.SOLUTION_18, ns)
    merges, _ = ns["train_18"](ns["TRAIN_18"], 300)
    ref = ns["mylearn"].tokenizer.BPETokenizer(vocab_size=300, pre_tokenizer="gpt2", min_frequency=0).fit(ns["TRAIN_18"])
    if ref.merges_ != merges:
        raise SystemExit("❌ bug18: the fixed train_18 does not learn the merges of mylearn (reference)")
    sample = ns["VERNE_VAL"][:20_000]
    if ns["encode_18"](sample, merges) != ref.encode(sample) or ns["decode_18"](ref.encode(sample), merges) != sample:
        raise SystemExit("❌ bug18: the fixed encode_18 or decode_18 differs from mylearn (reference)")
    return {"versions": rows, "e_in_train": ns["TRAIN_18"].count("é")}


def _naive_fit(text, vocab_size, pre_tokenizer, level="byte"):
    """A naive byte or char BPE: every pair counted again after each merge (min_frequency = 2)."""
    from collections import Counter

    from mylearn_ref.tokenizer import merge_pair, pre_tokenize
    counts = Counter(pre_tokenize(text, pre_tokenizer))
    if level == "byte":
        n_base, words = 256, {w: list(w.encode("utf-8")) for w in counts}
    else:
        alphabet = sorted({c for w in counts for c in w})
        n_base, words = len(alphabet), {w: [alphabet.index(c) for c in w] for w in counts}
    merges = []
    while n_base + len(merges) < vocab_size:
        pairs = Counter()
        for w, ids in words.items():
            for p in zip(ids, ids[1:]):
                pairs[p] += counts[w]
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[best] < 2:
            break
        merges.append(best)
        words = {w: merge_pair(ids, best, n_base + len(merges) - 1) for w, ids in words.items()}
    return merges


def timing(ns) -> dict:
    T = ns["mylearn"].tokenizer
    out = {}

    def best(fn, repeat=3):
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - start)
        return min(times), result

    for name, text in (("verne", ns["VERNE_TRAIN"]), ("holmes", ns["HOLMES_TRAIN"])):
        for size in (2000, 8000):
            seconds, _ = best(lambda: T.BPETokenizer(vocab_size=size, pre_tokenizer="gpt2").fit(text))
            out[f"ref_{name}_byte_{size}"] = round(seconds, 2)
    seconds, _ = best(lambda: T.BPETokenizer(vocab_size=2000, level="char").fit(ns["VERNE_TRAIN"]))
    out["ref_verne_char_2000"] = round(seconds, 2)
    seconds, _ = best(lambda: ns["reference_char_bpe"](ns["VERNE_TRAIN"], 2000))
    out["tokenizers_verne_char_2000"] = round(seconds, 3)
    seconds, _ = best(lambda: T.BPETokenizer(vocab_size=2000, pre_tokenizer="gpt2").fit([ns["HOLMES_TRAIN"], ns["VERNE_TRAIN"]]), 1)
    out["ref_mixed_byte_2000"] = round(seconds, 2)
    import wb.text as wbtext
    seconds, _ = best(lambda: wbtext.byte_bpe_merges(ns["VERNE_TRAIN"], 8000))
    out["tokenizers_verne_byte_8000"] = round(seconds, 2)
    for key, (text, size, level) in {"naive_verne_byte_2000": (ns["VERNE_TRAIN"], 2000, "byte"),
                                     "naive_verne_char_2000": (ns["VERNE_TRAIN"], 2000, "char"),
                                     "naive_part_byte_400": (ns["VERNE_TRAIN"][:100_000], 400, "byte")}.items():
        start = time.perf_counter()
        merges = _naive_fit(text, size, "gpt2" if level == "byte" else "whitespace", level)
        out[key] = round(time.perf_counter() - start, 1)
        ref = T.BPETokenizer(vocab_size=size, level=level, pre_tokenizer="gpt2" if level == "byte" else "whitespace").fit(text)
        if merges != ref.merges_:
            raise SystemExit(f"❌ timing: the naive version of {key} does not learn the merges of the reference")
        print(f"  timing {key}: {out[key]} s", flush=True)
    seconds, _ = best(lambda: ns["train_18"](ns["TRAIN_18"], 300), 2)
    out["train_18"] = round(seconds, 2)
    print(f"  timing {out}", flush=True)
    return out


# 🔮 B2.26 and 📈 B2.27: the settings of train_word2vec_25 tried, besides the contract (seeds 0 to 3)
W2V_VARIANTS = {"contract": ({}, (0, 1, 2, 3)), "batch1024_lr0.02": ({"batch_size": 1024, "lr": 0.02}, (0, 1)),
                "lr0.005": ({"lr": 0.005}, (0, 1)), "batch256": ({"batch_size": 256}, (0, 1))}


def w2v(ns) -> dict:
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors
    rows = []
    for name, (kwargs, seeds) in W2V_VARIANTS.items():
        for seed in seeds:
            start = time.perf_counter()
            model, losses = ns["train_word2vec_25"](ns["IDS_25"], ns["COUNTS_25"], seed=seed, **kwargs)
            seconds = time.perf_counter() - start
            E = model.center.weight.detach().double().numpy()
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            fogg = [w for w, _ in ns["nearest_words"](E, ns["ITOS_25"], "fogg", 10, ns["STOI_25"])]
            pp = [w for w, _ in ns["nearest_words"](E, ns["ITOS_25"], "passepartout", 10, ns["STOI_25"])]
            pca_all, umap_all, pca_groups, _ = ns["maps_27"](E, ns["COUNTS_25"], ns["GROUPS_27"])
            logf = np.log10(ns["COUNTS_25"])
            _, near = NearestNeighbors(n_neighbors=11).fit(umap_all).kneighbors(umap_all)
            top = np.arange(len(ns["ITOS_25"])) < 100
            row = {"variant": name, "seed": seed, "seconds": round(seconds, 1), "loss": [round(losses[0], 3), round(losses[-1], 3)],
                   "fogg": fogg, "passepartout": pp, "names": sum(w in ns["CHARACTERS_26"] for w in pp),
                   "rho_pc1": round(float(spearmanr(pca_all[:, 0], logf)[0]), 3),
                   "rho_pc2": round(float(spearmanr(pca_all[:, 1], logf)[0]), 3),
                   "umap_share": round(float(top[near[top, 1:]].mean()), 3)}
            rows.append(row)
            print(f"  w2v {name} {seed}: {row}", flush=True)
    chance = 100 / len(ns["ITOS_25"])
    checks = {"fogg": all(set(r["fogg"][:2]) == {"phileas", "mr"} for r in rows),
              "names": max(r["names"] for r in rows) <= 4,                      # b): at least 5 of the 10, never
              "pca_axis": max(abs(r["rho_pc1"]) for r in rows) < 0.35,
              "umap": min(r["umap_share"] for r in rows) > 3 * chance}
    if not all(checks.values()):
        raise SystemExit(f"❌ w2v: an answer is not the same in every run: {checks}")
    return {"runs": rows, "chance_umap": round(chance, 3),
            "answers": {"fogg": "B", "characters": False, "pca_axis": False, "umap_frequent": True}}


def c31(ns) -> dict:
    from wb import text as wbtext
    full = wbtext.load_text("verne")
    splits = {"frozen": (ns["VERNE_TRAIN"], ns["VERNE_VAL"])}
    for k in range(4):                                   # four other tenths of the text as the validation
        a, b = int(len(full) * (0.5 + 0.1 * k)), int(len(full) * (0.6 + 0.1 * k))
        a, b = full.rfind("\n", 0, a) + 1, full.rfind("\n", 0, b) + 1
        splits[f"tenth_{5 + k}"] = (full[:a] + full[b:], full[a:b])
    out = {}
    for name, (train, val) in splits.items():
        start = time.perf_counter()
        tok = ns["method_31"](train)
        ids = tok.encode(val)
        seconds = time.perf_counter() - start
        assert b"".join(tok.vocab_[i] for i in ids) == val.encode() and len(tok.vocab_) <= 2000
        base = ns["fast_byte_bpe"](train, 2000)
        base_dp = ns["LinesTokenizer31"](base.vocab_)
        n = len(val.encode())
        out[name] = {"reference": round(n / len(ids), 3), "seconds": round(seconds, 1),
                     "baseline": round(n / len(base.encode(val)), 3), "baseline_dp": round(n / len(base_dp.encode(val)), 3)}
        print(f"  c31 {name}: {out[name]}", flush=True)
    goal, frozen = ns["GOAL_31"], out["frozen"]
    if abs(frozen["baseline"] - ns["BASELINE_31"]) > 5e-4:
        raise SystemExit(f"❌ c31: the baseline of B2.20 is {frozen['baseline']}, not {ns['BASELINE_31']}")
    cut = wbtext.split_point(ns["VERNE_TRAIN"], 8 / 9)
    held, learned = ns["VERNE_TRAIN"][cut:], ns["VERNE_TRAIN"][:cut]
    control = round(len(held.encode()) / len(ns["fast_byte_bpe"](learned, 2000).encode(held)), 3)
    if abs(control - ns["BASELINE_CONTROL_31"]) > 5e-4:
        raise SystemExit(f"❌ c31: the baseline of the control is {control}, not {ns['BASELINE_CONTROL_31']}")
    if frozen["reference"] < 1.03 * goal or frozen["baseline_dp"] > 0.98 * goal:
        raise SystemExit(f"❌ c31: the goal {goal} is too close to the reference or to the baseline: {frozen}")
    for name, row in out.items():                         # the same goal, scaled by the baseline of each split
        if row["reference"] < 1.03 * goal * row["baseline"] / frozen["baseline"]:
            raise SystemExit(f"❌ c31: on {name}, the reference does not clear the scaled goal: {row}")
    # the anti-leak rules: a method that reads the validation through a global, and one that captured it before
    honest = ns["method_31"]

    def reads_global(text):
        return honest(text + "\n" + ns["VERNE_VAL"])

    def captured(text, _val=ns["VERNE_VAL"]):
        return honest(text + "\n" + _val)

    import contextlib
    import io
    leaks = {}
    for label, method in (("global", reads_global), ("captured", captured)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = ns["evaluate_31"](method)
        leaks[label] = {"ok": bool(result and result["ok"]), "score": round(result["bytes_per_token"], 3) if result else None,
                        "flagged": "❌ Ex B2.31 : contrôle" in buf.getvalue()}
        print(f"  c31 leak {label}: {leaks[label]}", flush=True)
    honest_run = io.StringIO()
    with contextlib.redirect_stdout(honest_run):
        honest_result = ns["evaluate_31"](honest)
    leaks["honest"] = {"ok": honest_result["ok"], "score": round(honest_result["bytes_per_token"], 3),
                       "control": [line for line in honest_run.getvalue().splitlines() if "contrôle" in line]}
    print(f"  c31 honest: {leaks['honest']}", flush=True)
    # the global variable is a decoy during the run (no gain); the captured validation is caught by the control
    if leaks["global"]["score"] > leaks["honest"]["score"] + 0.005 or leaks["captured"]["ok"] or not leaks["honest"]["ok"]:
        raise SystemExit(f"❌ c31: a method that reads the validation is not stopped: {leaks}")
    return {"splits": out, "goal": goal, "leaks": leaks}


ENTRIES = {"p14": (("GIVEN_14",), p14), "bug18": (("GIVEN_18",), bug18), "timing": (("GIVEN_18", "SOLUTION_18"), timing),
           "w2v": ((), w2v, ("GIVEN_25", "SOLUTION_25", "GIVEN_26", "GIVEN_27")),
           "c31": ((), c31, ("GIVEN_31", "SOLUTION_31"))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(ENTRIES))
    args = parser.parse_args(argv)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name, (blocks, fn, *blocks_de) in ENTRIES.items():
        if args.only and args.only != name:
            continue
        start = time.perf_counter()
        print(f"{name}…", flush=True)
        data[name] = fn(namespace(blocks, *blocks_de))
        print(f"{name}: {time.perf_counter() - start:.0f} s", flush=True)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
