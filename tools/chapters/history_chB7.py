#!/usr/bin/env python
"""Calibrate the prediction and replay the bug hunt of chapter B7, first half (used by Claude).

    python tools/chapters/history_chB7.py                 # everything (about 1 min on a 2-core CPU)
    python tools/chapters/history_chB7.py --only p12      # some entries (names below)

Writes tools/chapters/history_chB7.json (statistics only, no weights), which chB7_parts.py reads:
- "p12": 🔮 B7.12, the four comparisons of wbp.workers_run (0 workers replayed, 4 workers replayed, 0 against 4,
  0 against 4 without the flip) for the seeds 0, 1 and 2 and two sizes of the dataset; the four answers must be the
  same everywhere;
- "b16": 🐛 B7.16, the 2**5 partly repaired versions of the colleague's service (chB7_code.repaired_16): for each,
  the first diagnosis that fails must be the one of the first gap left, in the order of FIX_ORDER_16; and, for the
  versions with ONE gap left, the accuracy of the service on the 200 images (for 05_solutions.md);
- "t13": 🔬 B7.13, the reference command line (configs/fast.yaml) with the seeds 0 to 4, deterministic or not: the
  accuracies and the largest gaps between the weights.
The cells of the notebook are reused as is (tools/chapters/chB7_code.py), executed in one namespace.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(ROOT / "solutions")]
OUT = HERE / "history_chB7.json"


def namespace(blocks=()) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import wb

    import chB7_code as C

    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True), "FAST_MODE": True, "__name__": "history_chB7"}
    exec(C.TOOLS_A.replace("«KIND»", "solution"), ns)
    for block in blocks:
        exec(getattr(C, block), ns)
    return ns


def p12() -> dict:
    from wb import prod as wbp

    rows, answers = [], set()
    for seed in (0, 1, 2):
        for n in (512, 256):
            runs = {key: wbp.workers_run(w, seed=seed, n=n, augment=aug)
                    for key, (w, aug) in {"0a": (0, True), "0b": (0, True), "4a": (4, True), "4b": (4, True),
                                          "0n": (0, False), "4n": (4, False)}.items()}
            got = {"a": wbp.same_weights(runs["0a"], runs["0b"]), "b": wbp.same_weights(runs["4a"], runs["4b"]),
                   "c": wbp.same_weights(runs["0a"], runs["4a"]), "d": wbp.same_weights(runs["0n"], runs["4n"])}
            rows.append({"seed": seed, "n": n, **got, "gap_c": wbp.max_gap(runs["0a"], runs["4a"])})
            answers.add(tuple(sorted(got.items())))
            print("p12", rows[-1])
    if len(answers) != 1:
        raise SystemExit(f"p12: the answers change with the seed or the size: {answers}")
    return {"answers": dict(answers.pop()), "rows": rows}


def b16() -> dict:
    import chB7_code as C

    ns = namespace(["GIVEN_16"])
    order = C.FIX_ORDER_16
    table, mismatches, accuracy = [], [], {}
    for size in range(len(order) + 1):
        for fixed in itertools.combinations(order, size):
            exec(C.repaired_16(fixed), ns)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                ns["diagnose_16"]()
            lines = buffer.getvalue().splitlines()
            failing = next((int(line.split("point ")[1].split(",")[0].split(" ")[0]) for line in lines
                            if line.startswith("❌")), None)
            left = [k for k, name in enumerate(order, start=1) if name not in fixed]
            want = left[0] if left else None
            row = {"fixed": list(fixed), "first_failing": failing, "expected": want,
                   "message": next((line for line in lines if line.startswith("❌")), "")}
            table.append(row)
            if failing != want:
                mismatches.append(row)
            if len(left) == 1:
                try:
                    model, meta = ns["fresh_service_16"]()
                    names = ns["predict_16"](model, meta, ns["PATHS_16"])
                    truth = [ns["FRENCH_16"][ns["CLASSES"][k]] for k in ns["Y_16"]]
                    accuracy[order[left[0] - 1]] = float(np.mean([a == b for a, b in zip(names, truth)]))
                except Exception as error:  # noqa: BLE001 - a crash is a result too
                    accuracy[order[left[0] - 1]] = f"{type(error).__name__}: {str(error)[:100]}"
    print("b16", len(table), "versions,", len(mismatches), "mismatches;", accuracy)
    for row in mismatches:
        print("   ", row)
    if mismatches:
        raise SystemExit("b16: some partly repaired versions do not fail at the diagnosis of their first gap")
    return {"versions": len(table), "accuracy_one_gap_left": accuracy,
            "offline_accuracy": float((ns["OFFLINE_16"] == ns["Y_16"]).mean()),
            "messages": {",".join(r["fixed"]) or "none": r["message"] for r in table if len(r["fixed"]) >= 3 or not r["fixed"]}}


def t13() -> dict:
    from safetensors.torch import load_file

    from wb import prod as wbp

    project = wbp.project_dir("solution")
    for part in ("src", "configs"):
        shutil.copytree(wbp.reference_package() / part, project / part, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    wbp.write_subset(project / "data")
    runs = {}
    for seed in range(5):
        for det in (True, False):
            out = project / "out" / f"{seed}-{det}"
            r = wbp.cli(project, "train", "--config", "configs/fast.yaml", "--set", f"seed={seed}", "--set",
                        f"deterministic={str(det).lower()}", "--out", out, "--runs", project / "runs.jsonl")
            if not r.ok:
                raise SystemExit(f"t13: {r.tail(3)}")
            runs[(seed, det)] = {"weights": load_file(r.last_json()["weights"]), "val_acc": r.last_json()["val_acc"],
                                 "seconds": r.seconds}
    accs = [runs[(s, True)]["val_acc"] for s in range(5)]
    out = {"val_acc": accs, "seconds": [round(runs[(s, True)]["seconds"], 2) for s in range(5)],
           "det_equals_nondet": [wbp.same_weights(runs[(s, True)]["weights"], runs[(s, False)]["weights"]) for s in range(5)],
           "gap_seed": [wbp.max_gap(runs[(s, True)]["weights"], runs[(s + 1, True)]["weights"]) for s in range(4)],
           "mean5": float(np.mean(accs)), "s5": float(np.std(accs, ddof=1)),
           "half5": float(2.776 * np.std(accs, ddof=1) / np.sqrt(5))}
    print("t13", out)
    return out


ENTRIES = {"p12": p12, "b16": b16, "t13": t13}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=sorted(ENTRIES), default=None)
    args = parser.parse_args(argv)
    history = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in args.only or ENTRIES:
        start = time.perf_counter()
        history[name] = ENTRIES[name]()
        history[name]["seconds_to_compute"] = round(time.perf_counter() - start, 1)
    OUT.write_text(json.dumps(history, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
