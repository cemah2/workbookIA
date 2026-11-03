#!/usr/bin/env python
"""Calibrate the predictions and precompute the multi-seed protocols of chapter B6, first half (used by Claude).

    python tools/chapters/history_chB6.py                 # everything (about 6 min on a 2-core CPU)
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
  seeds;
- "t25": 📈 B6.25, the audit table to read (the solution of B6.24, 2 000 bootstrap draws, on the validation part);
- "p26": 🔮 B6.26, the four predictions on removing sex (the AUC of a model that predicts the sex from the other
  features, with and without relationship and marital-status; ΔEO without sex over ΔEO with it; the four-fifths
  ratio without sex) on eight splits of Adult and two seeds of the model each; the four answers must be the same
  everywhere, with a margin;
- "c29": 🏆 B6.29, the reference method on eleven splits of Adult (the test of the MP3 with five validation splits,
  then six other tests): ΔEO by sex and loss of accuracy on the test against the gradient boosting without correction,
  and the balanced accuracy of the method trained on shuffled labels; the goal is checked against the worst split.
The cells of the notebook are reused as is (tools/chapters/chB6_code.py and chB6_code2.py), executed in one namespace.
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


P26_SPLITS = ((2026, 0), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (1, 0), (2, 0), (3, 0))
P26_MODEL_SEEDS = (0, 1)
C29_SPLITS = ((2026, 0), (2026, 1), (2026, 2), (2026, 3), (2026, 4), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0))
C29_SHUFFLED = ((2026, 0), (1, 0), (2, 0))


def part_c(ns) -> dict:
    """The tools of part C (TOOLS_C), executed once in the namespace."""
    import chB6_code2 as C2

    if "adult_pipeline" not in ns:
        exec(C2.TOOLS_C, ns)
    return ns


def t25(ns) -> dict:
    import chB6_code2 as C2

    part_c(ns)
    exec(C2.GIVEN_24, ns)
    exec(C2.SOLUTION_24, ns)
    y, p, n_boot = ns["YA_VAL"], ns["PRED_VAL"], 2000
    tables, gaps = {}, {}
    for name, groups in (("sex", ns["SEX_VAL"]), ("age", ns["AGE_VAL"])):
        table = ns["audit_24"](y, p, groups, n_boot=n_boot, seed=0)
        tables[name] = {str(k): {c: (int(v) if c in ("n", "positives") else float(v)) for c, v in row.items()}
                        for k, row in table.iterrows()}
        for kind in ("dp", "eo", "tpr", "fpr"):
            gaps[f"{name}_{kind}"] = list(ns["gap_ci_24"](y, p, groups, kind, n_boot=n_boot, seed=0))
    return {"n_boot": n_boot, "tables": tables, "gaps": gaps}


def p26(ns) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    from wb import explain_data as wbx

    part_c(ns)
    pipe, audit = ns["adult_pipeline"], ns["sk_audit"]
    rows = []
    for test_seed, val_seed in P26_SPLITS:
        A = wbx.adult(seed=test_seed, val_seed=val_seed)
        ytr, yva, sex = A.y_train.to_numpy(), A.y_val.to_numpy(), A.X_val["sex"].to_numpy()
        female_tr, female_va = (A.X_train["sex"] == "Female").to_numpy(int), (sex == "Female").astype(int)
        auc = {}
        for name, drop in (("all", ("sex",)), ("few", ("sex", "relationship", "marital-status"))):
            spy = pipe(drop=drop).fit(A.X_train, female_tr)
            auc[name] = float(roc_auc_score(female_va, spy.predict_proba(A.X_val)[:, 1]))
        for seed in P26_MODEL_SEEDS:
            out = {"test_seed": test_seed, "val_seed": val_seed, "seed": seed, "auc_all": auc["all"], "auc_few": auc["few"]}
            for name, drop in (("with", ()), ("without", ("sex",))):
                model = pipe(HistGradientBoostingClassifier(random_state=seed), drop=drop).fit(A.X_train, ytr)
                pred = (model.predict_proba(A.X_val)[:, 1] >= 0.5).astype(int)
                rates, gaps = audit(yva, pred, sex)
                out[name] = {"acc": float(np.mean(pred == yva)), "eo": gaps["eo"], "dp": gaps["dp"],
                             "ratio": float(rates["Female"]["selection_rate"] / rates["Male"]["selection_rate"])}
            out["eo_ratio"] = out["without"]["eo"] / out["with"]["eo"]
            rows.append(out)
            print(f"  p26 {test_seed}/{val_seed}/{seed}: AUC {auc['all']:.3f} {auc['few']:.3f}, ratio EO "
                  f"{out['eo_ratio']:.2f}, 4/5 {out['without']['ratio']:.3f}", flush=True)
    answers = {"proxy_auc": sorted({letter(r["auc_all"], (0.6, 0.75, 0.9), "ABCD") for r in rows}),
               "proxy_auc_few": sorted({letter(r["auc_few"], (0.6, 0.75, 0.9), "ABCD") for r in rows}),
               "eo_ratio": sorted({letter(r["eo_ratio"], (0.5, 1.5), "ABC") for r in rows}),
               "four_fifths": sorted({letter(r["without"]["ratio"], (0.4, 0.6, 0.8), "ABCD") for r in rows})}
    for name, values in answers.items():
        if len(values) != 1:
            raise SystemExit(f"p26: the answer {name} changes from one split to another: {values}")
    span = lambda key: [min(key(r) for r in rows), max(key(r) for r in rows)]
    return {"rows": rows, "answers": {k: v[0] for k, v in answers.items()},
            "ranges": {"auc_all": span(lambda r: r["auc_all"]), "auc_few": span(lambda r: r["auc_few"]),
                       "eo_ratio": span(lambda r: r["eo_ratio"]), "four_fifths": span(lambda r: r["without"]["ratio"]),
                       "acc_change": span(lambda r: r["without"]["acc"] - r["with"]["acc"])}}


def c29(ns) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    import chB6_code2 as C2
    from wb import explain_data as wbx

    part_c(ns)
    exec(C2.SOLUTION_29, ns)
    pipe, audit, method = ns["adult_pipeline"], ns["sk_audit"], ns["method_29"]
    rows = []
    for test_seed, val_seed in C29_SPLITS:
        A = wbx.adult(seed=test_seed, val_seed=val_seed)
        yte, sex = A.y_test.to_numpy(), A.X_test["sex"].to_numpy()
        base = pipe().fit(A.X_train, A.y_train.to_numpy())
        base_acc = float(np.mean((base.predict_proba(A.X_test)[:, 1] >= 0.5) == yte))
        decide = method(A.X_train, A.y_train.to_numpy(), A.X_val, A.y_val.to_numpy())
        d = decide(A.X_test)
        rates, gaps = audit(yte, d, sex)
        out = {"test_seed": test_seed, "val_seed": val_seed, "base_acc": base_acc, "acc": float(np.mean(d == yte)),
               "eo": gaps["eo"], "tpr_gap": gaps["tpr"], "fpr_gap": gaps["fpr"],
               "base_eo": audit(yte, (base.predict_proba(A.X_test)[:, 1] >= 0.5).astype(int), sex)[1]["eo"]}
        out["loss"] = base_acc - out["acc"]
        if (test_seed, val_seed) in C29_SHUFFLED:
            rng = np.random.default_rng(29)
            shuffled = method(A.X_train, rng.permutation(A.y_train.to_numpy()), A.X_val, rng.permutation(A.y_val.to_numpy()))
            out["shuffled_balanced"] = float(balanced_accuracy_score(yte, shuffled(A.X_test)))
        rows.append(out)
        print(f"  c29 {test_seed}/{val_seed}: EO {out['eo']:.3f} (base {out['base_eo']:.3f}), loss {100 * out['loss']:.2f} pt"
              + (f", shuffled {out['shuffled_balanced']:.3f}" if "shuffled_balanced" in out else ""), flush=True)
    eo, loss = [r["eo"] for r in rows], [r["loss"] for r in rows]
    return {"rows": rows, "eo": [min(eo), max(eo)], "loss": [min(loss), max(loss)],
            "shuffled_balanced": max(r["shuffled_balanced"] for r in rows if "shuffled_balanced" in r)}


ENTRIES = {"p13": p13, "b16": b16, "c16": c16, "t25": t25, "p26": p26, "c29": c29}


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
