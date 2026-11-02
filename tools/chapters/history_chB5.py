#!/usr/bin/env python
"""Calibrate the predictions and precompute the multi-seed protocols of chapter B5, first half (used by Claude).

    python tools/chapters/history_chB5.py                  # everything (about 5 min on a 2-core CPU)
    python tools/chapters/history_chB5.py --only p13 toy20  # some entries (names below)

Writes tools/chapters/history_chB5.json (statistics only, no weights), which chB5_parts.py reads:
- "p13": 🔮 B5.13, the accuracy of the classifier of wb.diffusion on the 1 000 test digits noised at the steps 100, 300
  and 600 of the linear and cosine schedules (T = 1 000), for the classifier of the notebook and four others (other
  training seeds), each with three draws of the noise; both answers must be the same everywhere, with a margin;
- "p15": 📈 B5.15, the readings of the figure (computed from the formulas);
- "b18": 🔬 B5.18, the errors of the five chains with the reference library, for T = 400 and T = 1 000;
- "toy20": 📦 B5.19 and 🔬 B5.20, the protocol of the notebook (the reference functions of wb.diffusion, the settings
  and the seeds of the cells) for five seeds: the quality of DDPM (B5.19), DDIM and Euler from 1 to 100 steps (B5.20),
  for T = 400 (fast mode) and T = 1 000 (full mode); both answers of B5.20 must hold for every seed and both modes.
The cells of the notebook are reused as is (tools/chapters/chB5_code.py), executed in one namespace.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(ROOT / "solutions")]
OUT = HERE / "history_chB5.json"

STEPS_20 = (1, 2, 5, 10, 20, 50, 100)
SEEDS = (0, 1, 2, 3, 4)               # seed k: the seeds of the notebook cells plus 100 k (k = 0: the notebook itself)
P13_CLASSIFIER_SEEDS = (2502, 0, 1, 2, 3)
P13_NOISE_SEEDS = (13, 1, 2)


def namespace(fast: bool = True, blocks=()) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wb

    import chB5_code as C

    plt.show = lambda *a, **k: plt.close("all")
    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=fast), "FAST_MODE": fast, "__name__": "history_chB5",
          "mylearn": wb.load_mylearn("ref")}
    for block in ("TOOLS_A", *blocks):
        exec(getattr(C, block), ns)
    return ns


def p13(ns) -> dict:
    from wb import diffusion as wbd

    import chB5_code as C

    exec(C.GIVEN_13, ns)                                     # the classifier of the notebook, cached as in the notebook
    labels = ns["DIGITS"].labels.numpy()
    rows, answers = [], set()
    folder = Path(tempfile.mkdtemp())
    for clf_seed in P13_CLASSIFIER_SEEDS:
        wbd.CLASSIFIER_SETTINGS["seed"] = clf_seed
        wbd._CLASSIFIERS.clear()
        clf = wbd.classifier("mnist", folder=None if clf_seed == 2502 else folder, verbose=False)
        for noise_seed in P13_NOISE_SEEDS:
            ns["EPS_13"] = np.random.default_rng(noise_seed).standard_normal(tuple(ns["DIGITS"].images.shape))
            acc = {f"{kind} {step}": float(np.mean(wbd.predict(clf, ns["noisy_13"](kind, step)) == labels))
                   for kind in ns["AB_13"] for step in ns["STEPS_13"]}
            strong = sum(v >= 0.7 for v in acc.values())
            cos600 = acc["cosinus 600"] > 0.5
            answers.add((strong, cos600))
            margin = min(min(abs(v - 0.7) for v in acc.values()), abs(acc["cosinus 600"] - 0.5))
            rows.append({"classifier_seed": clf_seed, "noise_seed": noise_seed, "accuracy": acc, "margin": margin})
            print(f"  p13 clf {clf_seed} noise {noise_seed}: {({k: round(v, 3) for k, v in acc.items()})}", flush=True)
    wbd.CLASSIFIER_SETTINGS["seed"] = 2502
    wbd._CLASSIFIERS.clear()
    if len(answers) != 1:
        raise SystemExit(f"❌ p13: the answers depend on the classifier or the noise: {answers}")
    margin = min(r["margin"] for r in rows)
    if margin < 0.04:
        raise SystemExit(f"❌ p13: an accuracy is too close to its threshold (margin {margin:.3f})")
    strong, cos600 = answers.pop()
    ranges = {case: [min(r["accuracy"][case] for r in rows), max(r["accuracy"][case] for r in rows)]
              for case in rows[0]["accuracy"]}
    return {"rows": rows, "ranges": ranges, "margin": margin, "answers": {"strong": strong, "cos600": bool(cos600)}}


def p15(ns) -> dict:
    from wb import diffusion as wbd

    ln_snr = {}
    for key, betas in (("L", wbd.schedule_betas("linear", 1000)), ("C", wbd.schedule_betas("cosine", 1000)),
                       ("D", np.linspace(1e-4, 0.02, 400)), ("S", wbd.schedule_betas("linear", 400))):
        ab = wbd.alpha_bars(betas)
        ln_snr[key] = np.log(ab / (1 - ab))
    ab_l = wbd.alpha_bars(wbd.schedule_betas("linear", 1000))
    at_400 = np.interp(np.arange(1, 401) / 400, np.arange(1, 1001) / 1000, ab_l)     # L read at the fractions of T = 400
    gap_s = float(np.abs(wbd.alpha_bars(wbd.schedule_betas("linear", 400)) - at_400).max())
    gap_d = float(np.abs(wbd.alpha_bars(np.linspace(1e-4, 0.02, 400)) - at_400).max())
    ends = {k: float(v[-1]) for k, v in ln_snr.items()}
    above = float(np.mean(ln_snr["C"] > ln_snr["L"]))
    if not (gap_s < 0.02 < 0.2 < gap_d):
        raise SystemExit(f"❌ p15: the curves S and D are not told apart ({gap_s:.3f}, {gap_d:.3f})")
    largest = max(ends, key=ends.get)
    share = "A" if above < 0.5 else "B" if above <= 0.9 else "C" if above > 0.95 else "?"
    if share == "?":
        raise SystemExit(f"❌ p15: the share {above:.3f} falls between two choices")
    print(f"  p15: gaps S {gap_s:.4f} D {gap_d:.4f}; final ln SNR {ends}; cosine above on {above:.3f}")
    return {"gap_S": gap_s, "gap_D": gap_d, "final_ln_snr": ends, "cos_above_share": above,
            "first_index_cos_below": int(np.argmax(ln_snr["C"] < ln_snr["L"])),
            "answers": {"same_curve": "S" if gap_s < gap_d else "D", "largest_end": largest, "cos_above": share}}


def b18(ns) -> dict:
    import chB5_code as C

    out = {}
    exec(C.SOLUTION_18, ns)
    ns.update(ddpm_fixed_18=..., ddpm_oracle_18=..., ddim_ten_18=..., ddim_eta1_18=...)   # no answer checked here
    for T in (400, 1000):
        from wb import diffusion as wbd

        ns["BETAS"], ns["T_STEPS"] = wbd.schedule_betas("linear", T), T
        exec(C.CHECK_18, ns)
        out[str(T)] = ns["ERRORS_18"]
        print(f"  b18 T = {T}: {ns['ERRORS_18']}", flush=True)
    keys = list(out["400"])
    answers = {"a": out["400"][keys[0]] < 1e-6, "b": out["400"][keys[1]] < 1e-6, "c": out["400"][keys[3]] < 1e-6,
               "d": out["400"][keys[4]] < 1e-6}
    for T in ("400", "1000"):
        k = list(out[T])
        got = {"a": out[T][k[0]] < 1e-6, "b": out[T][k[1]] < 1e-6, "c": out[T][k[3]] < 1e-6, "d": out[T][k[4]] < 1e-6}
        if got != answers or out[T][k[0]] < 1e-3 or out[T][k[4]] < 1e-3 or max(out[T][x] for x in k[1:4]) > 1e-9:
            raise SystemExit(f"❌ b18: the answers are not clear-cut for T = {T}: {out[T]}")
    return {"errors": out, "answers": answers}


def toy20(ns) -> dict:
    import torch
    from wb import diffusion as wbd

    import chB5_code as C

    exec(C.GIVEN_B, ns)
    S = wbd.TOY_SETTINGS
    data = ns["DATA_B"]
    result = {"seeds": list(SEEDS), "steps": list(STEPS_20), "floor": ns["FLOOR_B"], "blob": ns["BLOB_B"]}
    euler = []
    for k in SEEDS:
        torch.manual_seed(20 + 100 * k)
        fm = wbd.Denoiser2D(wbd.FM_POSITIONS)
        start = time.perf_counter()
        wbd.train_toy(fm, data, S["n_iters"], S["batch_size"], S["lr"], torch.Generator().manual_seed(200 + 100 * k),
                      flow=True)
        fm.eval()
        seconds = time.perf_counter() - start
        euler.append([ns["quality"](wbd.sample_toy_euler(fm, 2000, n, torch.Generator().manual_seed(201 + 100 * k)))[0]
                      for n in STEPS_20])
        print(f"  toy20 fm seed {k}: {np.round(euler[-1], 4).tolist()} ({seconds:.0f} s)", flush=True)
    result["euler"] = euler
    for T in (400, 1000):
        sched = ns["toy_schedule"](wbd.schedule_betas("linear", T))
        ddim, ddpm = [], []
        for k in SEEDS:
            torch.manual_seed(19 + 100 * k)
            eps = wbd.Denoiser2D(T)
            wbd.train_toy(eps, data, S["n_iters"], S["batch_size"], S["lr"],
                          torch.Generator().manual_seed(190 + 100 * k), sched["alpha_bars"])
            eps.eval()
            pts = wbd.sample_toy_ddpm(eps, 2000, sched["alpha_bars"], torch.Generator().manual_seed(191 + 100 * k),
                                      np.random.default_rng(192 + 100 * k))
            ddpm.append(list(ns["quality"](pts)))
            ddim.append([ns["quality"](wbd.sample_toy_ddim(eps, 2000, n, sched["alpha_bars"],
                                                           torch.Generator().manual_seed(201 + 100 * k)))[0]
                         for n in STEPS_20])
            print(f"  toy20 T = {T} eps seed {k}: ddpm {np.round(ddpm[-1], 4).tolist()} ddim "
                  f"{np.round(ddim[-1], 4).tolist()}", flush=True)
        result[f"ddim_{T}"], result[f"ddpm_{T}"] = ddim, ddpm
        for swd, on, modes in ddpm:
            if not (swd < 0.2 * 0.8 and on > 0.8 + 0.05 and modes == 8):
                raise SystemExit(f"❌ toy20: the reference DDPM of B5.19 is too close to the threshold of the check: "
                                 f"{(swd, on, modes)}")
    # a) the smallest number of steps from which Euler is within 3 times its distance at 100 steps, for every seed
    fewest = []
    for row in euler:
        ok = [n for n, v in zip(STEPS_20, row) if v <= 3 * row[-1]]
        fewest.append(min(ok))
    worst_ratio_ok = max(row[STEPS_20.index(min(fewest))] / row[-1] for row in euler)
    best_ratio_before = min(row[STEPS_20.index(min(fewest)) - 1] / row[-1] for row in euler)
    if len(set(fewest)) != 1 or worst_ratio_ok > 2.6 or best_ratio_before < 3.4:
        raise SystemExit(f"❌ toy20: answer a) is not clear-cut: {fewest}, {worst_ratio_ok:.2f}, {best_ratio_before:.2f}")
    # b) at 5 steps, Euler against DDIM, for every seed and both T (at 10 steps, one seed of T = 1 000 is a tie)
    i5 = STEPS_20.index(5)
    margins = [d[i5] / e[i5] for T in (400, 1000) for d, e in zip(result[f"ddim_{T}"], euler)]
    better = "E" if min(margins) > 1.5 else "D" if max(margins) < 1 / 1.5 else "?"
    if better == "?":
        raise SystemExit(f"❌ toy20: answer b) is not clear-cut: ratios DDIM / Euler at 5 steps {margins}")
    result["answers"] = {"fewest": fewest[0], "better5": better}
    result["ratios_5"] = margins
    result["fewest_ratios"] = {"at_answer_max": worst_ratio_ok, "before_answer_min": best_ratio_before}
    return result


ENTRIES = {"p13": p13, "p15": p15, "b18": b18, "toy20": toy20}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(ENTRIES), nargs="+")
    args = parser.parse_args(argv)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name, fn in ENTRIES.items():
        if args.only and name not in args.only:
            continue
        start = time.perf_counter()
        print(f"{name}…", flush=True)
        data[name] = fn(namespace())
        print(f"{name}: {time.perf_counter() - start:.0f} s", flush=True)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
