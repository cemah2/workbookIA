#!/usr/bin/env python
"""Precompute the experiments of chapter 28 shown before a prediction or read as curves (used by Claude).

    python tools/chapters/history_ch28.py                  # everything (about 3 min on a CPU)
    python tools/chapters/history_ch28.py --only recon19   # one entry: p12, recon19, curves20 or calib29

Writes tools/chapters/history_ch28.json, which build_ch28.py reads:
- "p12": the images that excite 8 filters of conv1 and 8 filters of fire8 (the reference visualize_filter_13, 128
  pixels, 100 steps), summed up by their dominant period in pixels and their share of high frequencies (🔮 28.12);
- "recon19": the reconstructions from noise of the reference reconstruct_22 for three seeds: the pixel error to the
  photo of the content reconstructions from conv1 and from fire7, the correlation of the style reconstruction with
  the photo (🔮 28.19);
- "curves20": three style transfers of the reference style_transfer_23 (the settings of part C, beta x 100, a noise
  start), every 5 steps, under shuffled letters (📈 28.20);
- "calib29": the reference of the challenge 🏆 28.29 (make_29: content distance, style ratio, time) over the three paintings; the
  goal of the challenge is the content distance of the reference on the Hokusai, plus 15 %.
The code of the notebook is reused as is (tools/chapters/ch28_code.py), executed in one namespace.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
OUT = HERE / "history_ch28.json"
LETTERS_20 = {"standard": "B", "style_heavy": "C", "noise": "A"}


def namespace() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wb

    import ch28_code as C

    plt.show = lambda *a, **k: plt.close("all")
    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True), "FAST_MODE": True, "mylearn": wb.load_mylearn("ref"),
          "__name__": "history_ch28"}
    for block in ("TOOLS", "SOLUTION_11", "SOLUTION_13", "GIVEN_C", "SOLUTION_18", "SOLUTION_22", "SOLUTION_23",
                  "GIVEN_24", "SOLUTION_29"):
        exec(getattr(C, block), ns)
    return ns


def p12(ns) -> dict:
    out = {}
    for name, channels in (("conv1", [3, 11, 19, 27, 35, 43, 51, 59]), ("fire8", [10, 70, 130, 190, 250, 310, 370, 430])):
        rows = []
        for ch in channels:
            img = ns["visualize_filter_13"](ns["NET"], ns["LAYERS"][name], ch, size=128, steps=100)
            rows.append({"channel": ch, "period": round(ns["dominant_period"](img), 2),
                         "high": round(ns["high_freq_share"](img), 4)})
        out[name] = rows
        print(f"  p12 {name}: periods {[r['period'] for r in rows]}", flush=True)
    return out


def recon19(ns) -> dict:
    import numpy as np
    photo = ns["CONTENT_C"]
    grey = photo.mean(axis=0).ravel()
    runs = []
    for seed in range(3):
        rec = {name: ns["reconstruct_22"](ns["NET"], photo, [name], "content", steps=150, seed=seed) for name in ("conv1", "fire7")}
        style = ns["reconstruct_22"](ns["NET"], ns["STYLE_C"], ns["STYLE_LAYERS"], "style", steps=150, seed=seed)
        style = style[:, : photo.shape[1], : photo.shape[2]]
        row = {"seed": seed, **{f"mse_{n}": round(float(((r - photo) ** 2).mean()), 5) for n, r in rec.items()},
               "corr_style": round(float(np.corrcoef(style.mean(axis=0).ravel(), grey)[0, 1]), 4)}
        runs.append(row)
        print(f"  recon19 {row}", flush=True)
    return {"runs": runs}


def curves20(ns) -> dict:
    settings = {"standard": dict(beta=ns["BETA"]), "style_heavy": dict(beta=100 * ns["BETA"]),
                "noise": dict(beta=ns["BETA"], init="noise")}
    runs = {}
    for name, kw in settings.items():
        _, hist = ns["style_transfer_23"](ns["NET"], ns["CONTENT_C"], ns["STYLE_C"], ns["CONTENT_LAYERS"], ns["STYLE_LAYERS"],
                                          alpha=ns["ALPHA"], tv=ns["TV"], steps=150, every=5, **kw)
        runs[LETTERS_20[name]] = {k: [round(v, 3) for v in vals] for k, vals in hist.items()}
        h = runs[LETTERS_20[name]]
        print(f"  curves20 {name} ({LETTERS_20[name]}): content {h['content'][0]} -> {h['content'][-1]}, style "
              f"{h['style'][0]} -> {h['style'][-1]}, tv {h['tv'][0]} -> {h['tv'][-1]}", flush=True)
    return {"runs": dict(sorted(runs.items()))}


def calib29(ns) -> dict:
    import wb
    out = []
    content = wb.datasets.load_sample_image("flower", ns["SIZE"])
    for style_name in ("great_wave", "starry_night", "composition_vii"):
        style = wb.datasets.load_style_image(style_name, ns["SIZE"])
        start = time.perf_counter()
        img = ns["make_29"](content, style)
        seconds = time.perf_counter() - start
        d_c, d_s = ns["measure_24"](img, content, style)
        d_s0 = ns["measure_24"](content, content, style)[1]
        out.append({"style": style_name, "seconds": round(seconds, 1), "content": round(d_c, 3),
                    "style_ratio": round(d_s0 / d_s, 2)})
        print(f"  calib29 {out[-1]}", flush=True)
    return {"runs": out}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["p12", "recon19", "curves20", "calib29"])
    args = parser.parse_args(argv)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    ns = namespace()
    for name, fn in (("p12", p12), ("recon19", recon19), ("curves20", curves20), ("calib29", calib29)):
        if args.only and args.only != name:
            continue
        start = time.perf_counter()
        print(f"{name}…", flush=True)
        data[name] = fn(ns)
        print(f"{name}: {time.perf_counter() - start:.0f} s", flush=True)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
