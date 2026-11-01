#!/usr/bin/env python
"""Calibrate the predictions, replay the bug hunt and measure the times of chapter B2 (used by Claude).

    python tools/chapters/history_chB2.py                 # everything (about 6 min on a 2-core CPU, the Hub reachable)
    python tools/chapters/history_chB2.py --only bug18    # one entry: p14, bug18, timing

Writes tools/chapters/history_chB2.json (statistics only), which chB2_parts.py reads:
- "p14": 🔮 B2.14, the counts of tokens of the six real tokenizers on the four texts (two frozen excerpts, the numbers,
  the emoji) and the answers of the three predictions; each measure must be far enough from the bounds of its
  categories (it is deterministic: frozen excerpts and frozen revisions);
- "bug18": 🐛 B2.18 replayed on its 2**3 partially fixed versions: the first point that fails must be the first error
  left (and nothing must crash); the fixed version must also encode by rank exactly like mylearn (reference);
- "timing": the training times of the reference BPETokenizer, of a naive version (every pair counted again and every
  word merged again after each merge, the way a learner first writes it), of tokenizers and of wb.text.byte_bpe_merges,
  for the statements, the hints and the solutions.
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


def namespace(blocks=()) -> dict:
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


ENTRIES = {"p14": (("GIVEN_14",), p14), "bug18": (("GIVEN_18",), bug18), "timing": (("GIVEN_18", "SOLUTION_18"), timing)}


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
