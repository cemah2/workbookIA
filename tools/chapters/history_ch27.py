#!/usr/bin/env python
"""Precompute the multi-seed experiments of chapter 27 (used by Claude).

    python tools/chapters/history_ch27.py                   # everything (about 15 min on a CPU)
    python tools/chapters/history_ch27.py --only curves14   # one entry: runs13, curves14 or calib

Writes tools/chapters/history_ch27.json, which build_ch27.py reads:
- "runs13": the trainings of 27.12 (the reference gan_step_12 and train_gan_12, 10 epochs on DATA_1D) for the seeds 0
  to 4, shown by the prediction 🔮 27.13: after each epoch, the mean and the standard deviation of the generated numbers,
  and the epoch means of loss_d, loss_g, d_real, d_fake;
- "curves14": four trainings of the same toy GAN with other learning rates (seed 14), under shuffled letters, read by
  📈 27.14: the means of loss_d, loss_g, d_real and d_fake over each quarter of epoch, the generated mean and standard
  deviation after each epoch, and the final W1 distance to the data;
- "calib": the outcomes of the prediction 🔮 27.17 over eight seeds (the notebook uses seed 0), and the reference of
  the challenge 🏆 27.28 (Fréchet distances on the public and hidden test images, training time) over three seeds,
  with the baseline of 27.20 trained 3 epochs.
The code of the notebook is reused as is: the cells of build_ch27.py are executed in one namespace.
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
OUT = HERE / "history_ch27.json"

SETTINGS_14 = {                       # name -> (lr_g, lr_d); 10 epochs, mini-batches of 64, seed 14
    "standard": (1e-3, 1e-3),
    "slow": (1e-4, 1e-4),
    "strong_d": (1e-4, 1e-2),
    "g_fast": (1e-2, 1e-4),
}
LETTERS_14 = {"slow": "A", "g_fast": "B", "standard": "C", "strong_d": "D"}


def namespace() -> dict:
    import numpy as np
    import wb

    import build_ch27 as B

    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True), "np": np, "mylearn": wb.load_mylearn("ref"),
          "FAST_MODE": True, "__name__": "history_ch27"}
    exec(B.TOOLS, ns)
    exec(B.GIVEN_A, ns)
    exec(B.SOLUTION_12, ns)
    return ns


def _epoch_runs(ns, seed, lr_g=1e-3, lr_d=1e-3, n_epochs=10, chunks=1):
    np, torch = ns["np"], ns["torch"]
    stats = {0: ns["stats_1d"](ns["make_toy_gan"](dim=1, seed=seed)[0])}
    G, D = ns["make_toy_gan"](dim=1, seed=seed)
    history = ns["train_gan_12"](G, D, ns["DATA_1D"], n_epochs, lr_g=lr_g, lr_d=lr_d, seed=seed,
                                 on_epoch=lambda e, G: stats.__setitem__(e, ns["stats_1d"](G)))
    steps = ns["STEPS_1D"]
    per = {key: [round(float(v), 4) for v in ns["mean_epochs"](history, key, steps // chunks)] for key in history}
    del torch
    return G, stats, per, np


def runs13(ns) -> dict:
    rows = []
    for seed in range(5):
        _, stats, per, _ = _epoch_runs(ns, seed)
        rows.append({"seed": seed, "mean": [round(stats[e][0], 4) for e in range(11)],
                     "std": [round(stats[e][1], 4) for e in range(11)], **per})
        print(f"  runs13 seed {seed}: mean {rows[-1]['mean'][1]:.2f} -> {rows[-1]['mean'][-1]:.2f}, std "
              f"{min(rows[-1]['std'][2:]):.2f}..{max(rows[-1]['std'][2:]):.2f}, loss_d {rows[-1]['loss_d'][-1]:.3f}",
              flush=True)
    return {"runs": rows}


def curves14(ns) -> dict:
    from scipy.stats import wasserstein_distance

    runs = {}
    for name, (lr_g, lr_d) in SETTINGS_14.items():
        G, stats, per, _ = _epoch_runs(ns, 14, lr_g=lr_g, lr_d=lr_d, chunks=4)
        x = ns["generate"](G, ns["Z_FIXED_1D"])[:, 0]
        w1 = float(wasserstein_distance(x, ns["DATA_1D"][:2000, 0].numpy()))
        runs[LETTERS_14[name]] = {**per, "mean": [round(stats[e][0], 4) for e in range(11)],
                                  "std": [round(stats[e][1], 4) for e in range(11)], "w1": round(w1, 4)}
        print(f"  curves14 {name} ({LETTERS_14[name]}): loss_d {per['loss_d'][-1]:.3f} loss_g {per['loss_g'][-1]:.3f} "
              f"d_real {per['d_real'][-1]:.3f} d_fake {per['d_fake'][-1]:.3f} mean {stats[10][0]:.2f} "
              f"std {stats[10][1]:.2f} w1 {w1:.3f}", flush=True)
    return {"quarters": [round(0.25 * k, 2) for k in range(1, 41)], "runs": dict(sorted(runs.items())),
            "settings": {LETTERS_14[k]: k for k in SETTINGS_14}}


def calib(ns) -> dict:
    import build_ch27 as B
    out = {"p17": []}
    for seed in range(8):
        _, base, _, _ = _epoch_runs(ns, seed)
        _, strong, per, _ = _epoch_runs(ns, seed, lr_d=1e-2)
        spread = [max(s[e][1] for e in range(6, 11)) - min(s[e][1] for e in range(6, 11)) for s in (base, strong)]
        row = {"seed": seed, "d_fake_epoch1": per["d_fake"][0], "final_mean": round(strong[10][0], 4),
               "spread_base": round(spread[0], 4), "spread_strong": round(spread[1], 4)}
        out["p17"].append(row)
        print(f"  calib p17 {row}", flush=True)
    torch = ns["torch"]
    exec(B.GIVEN_B, ns)
    exec(B.SOLUTION_20, ns)
    exec(B.SOLUTION_23, ns)
    exec(B.GIVEN_28.replace("{GOAL}", "0.0").replace("{SECONDS}", "150"), ns)
    out["challenge28"] = []
    for seed in (0, 1, 2):
        for name in ("baseline", "reference"):
            if name == "reference":
                exec(B._solution_28().replace("seed=2828", f"seed={2828 + seed}"), ns)
            else:
                def make_gan_28():
                    G, D = ns["make_generator_20"](ns["NZ"], 32), ns["make_discriminator_20"](32)
                    G.apply(ns["init_weights_20"])
                    D.apply(ns["init_weights_20"])
                    return G, D

                def train_28(G, D, images, seed=seed):
                    ns["train_gan_12"](G, D, images[:10_000], 3, batch_size=64, lr_g=2e-4, lr_d=2e-4, seed=2828 + seed)
                ns["make_gan_28"], ns["train_28"] = make_gan_28, train_28
            torch.manual_seed(2728 + seed)
            G, D = ns["make_gan_28"]()
            start = time.perf_counter()
            ns["train_28"](G, D, ns["X_FASHION"])
            seconds = time.perf_counter() - start
            public = ns["fid_23"](G, ns["CLF_FASHION"], ns["PUBLIC_FEATS_28"], n=2000, seed=28)
            hidden = ns["fid_23"](G, ns["CLF_FASHION"], ns["HIDDEN_FEATS_28"], n=2000, seed=2828)
            out["challenge28"].append({"seed": seed, "name": name, "seconds": round(seconds, 1),
                                       "public": round(public, 3), "hidden": round(hidden, 3)})
            print(f"  calib challenge28 {out['challenge28'][-1]}", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["runs13", "curves14", "calib"])
    args = parser.parse_args(argv)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    ns = namespace()
    for name, fn in (("runs13", runs13), ("curves14", curves14), ("calib", calib)):
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
