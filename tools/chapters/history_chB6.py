#!/usr/bin/env python
"""Calibrate the predictions and precompute the multi-seed protocols of chapter B6, first half (used by Claude).

    python tools/chapters/history_chB6.py                 # everything (about 2 min on a 2-core CPU)
    python tools/chapters/history_chB6.py --only p13      # some entries (names below)

Writes tools/chapters/history_chB6.json (statistics only, no weights), which chB6_parts.py reads:
- "p13": 🔮 B6.13, the four readings of the ICE curves of MedInc and Latitude (rise of the partial dependence of
  MedInc and of Latitude (end minus start), largest rise again of that of Latitude going north, share of the ICE curves of Latitude that end higher,
  standard deviations of the change of the ICE curves) for the gradient boosting of the notebook on five splits of
  California (wbx.california(seed)) and two seeds of the model each; the four answers must be the same everywhere,
  with a margin;
- "b16": 🔬 B6.16, the base value and the Shapley value of MedInc of the district of the notebook for the three
  backgrounds, the random ones drawn with five seeds;
- "c16": 🔬 B6.16, the duplicated feature (copy_experiment_16 of the solution) for the three noise levels and ten
  seeds.
The cells of the notebook are reused as is (tools/chapters/chB6_code.py), executed in one namespace.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(ROOT / "solutions")]
OUT = HERE / "history_chB6.json"

P13_SPLITS = (0, 1, 2, 3, 4)
P13_MODEL_SEEDS = (0, 1)
C16_SEEDS = tuple(range(10))
B16_SEEDS = (0, 1, 2, 3, 4)


def namespace(fast: bool = True, blocks=()) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wb

    import chB6_code as C

    plt.show = lambda *a, **k: plt.close("all")
    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=fast), "FAST_MODE": fast, "__name__": "history_chB6",
          "mylearn": wb.load_mylearn("ref")}
    for block in ("TOOLS_A", *blocks):
        exec(getattr(C, block), ns)
    return ns


def letter(value: float, edges, letters) -> str:
    return letters[int(np.searchsorted(edges, value, side="right"))]


def p13(ns) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import partial_dependence

    from wb import explain_data as wbx

    rows = []
    for split in P13_SPLITS:
        cal = wbx.california(split)
        for seed in P13_MODEL_SEEDS:
            model = HistGradientBoostingRegressor(random_state=seed).fit(cal.X_train, cal.y_train)
            stats = {"split": split, "seed": seed}
            for feature in ("MedInc", "Latitude"):
                r = partial_dependence(model, cal.X_val.iloc[:500], [feature], kind="both", grid_resolution=50)
                avg, ind = r["average"][0], r["individual"][0]
                change = ind[:, -1] - ind[:, 0]
                stats[feature] = {"rise": float(avg[-1] - avg[0]),
                                  "bump": float(np.max(avg - np.minimum.accumulate(avg))),
                                  "up": float(np.mean(change > 0)), "spread": float(np.std(change))}
            rows.append(stats)
    rise = [r["MedInc"]["rise"] for r in rows]
    bump = [r["Latitude"]["bump"] for r in rows]
    up = [r["Latitude"]["up"] for r in rows]
    ratio = [r["Latitude"]["spread"] / r["MedInc"]["spread"] for r in rows]
    lat_rise = [r["Latitude"]["rise"] for r in rows]
    answers = {"medinc_rise": sorted({letter(v, (0.5, 1.5, 3.0), "ABCD") for v in rise}),
               "lat_trend": sorted({letter(v, (-0.3, 0.3), "CBA") for v in lat_rise}),
               "lat_up": sorted({letter(v, (0.05, 0.30, 0.70), "ABCD") for v in up}),
               "spread": sorted({bool(v > 1) for v in ratio})}
    for name, values in answers.items():
        if len(values) != 1:
            raise SystemExit(f"p13: the answer {name} changes from one split to another: {values}")
    return {"rows": rows, "answers": {k: v[0] for k, v in answers.items()},
            "ranges": {"medinc_rise": [min(rise), max(rise)], "lat_bump": [min(bump), max(bump)],
                       "lat_up": [min(up), max(up)], "spread_ratio": [min(ratio), max(ratio)],
                       "lat_rise": [min(r["Latitude"]["rise"] for r in rows), max(r["Latitude"]["rise"] for r in rows)],
                       "medinc_spread": [min(r["MedInc"]["spread"] for r in rows), max(r["MedInc"]["spread"] for r in rows)],
                       "lat_spread": [min(r["Latitude"]["spread"] for r in rows),
                                      max(r["Latitude"]["spread"] for r in rows)]}}


def b16(ns) -> dict:
    import shap

    import chB6_code as C

    exec(C.GIVEN_16, ns)
    X_TRAIN, GBR, X_16 = ns["X_TRAIN"], ns["GBR"], ns["X_16"]
    rich = ns["RICH_16"]
    out = {}
    for seed in B16_SEEDS:
        for name, bg in (("population", X_TRAIN.sample(50, random_state=seed)), ("riches", rich.sample(50, random_state=seed))):
            sv = shap.TreeExplainer(GBR, data=bg, feature_perturbation="interventional")(X_16.to_frame().T)
            out.setdefault(name, []).append({"seed": seed, "base": float(np.ravel(sv.base_values)[0]),
                                             "phi": [float(v) for v in sv.values[0]]})
    return {"features": list(ns["FEATURES"]), "rows": out}


def c16(ns) -> dict:
    import chB6_code as C

    exec(C.GIVEN_16, ns)
    exec(C.SOLUTION_16, ns)
    out = {}
    for noise in (0.0, 0.01, 0.1):
        out[str(noise)] = [dict(seed=seed, **ns["copy_experiment_16"](noise, seed)) for seed in C16_SEEDS]
    return out


ENTRIES = {"p13": p13, "b16": b16, "c16": c16}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=sorted(ENTRIES))
    args = parser.parse_args()
    history = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    ns = namespace()
    for name in args.only or ENTRIES:
        start = time.perf_counter()
        history[name] = ENTRIES[name](ns)
        print(f"{name}: {time.perf_counter() - start:.0f} s", flush=True)
    OUT.write_text(json.dumps(history, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
