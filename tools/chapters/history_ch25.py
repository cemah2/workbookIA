#!/usr/bin/env python
"""Precompute the long trainings and the multi-seed calibrations of chapter 25 (used by Claude).

    python tools/chapters/history_ch25.py                 # everything (about 15 min on a CPU)
    python tools/chapters/history_ch25.py --only calib    # one entry: collapse34 or calib

Writes tools/chapters/history_ch25.json, which build_ch25.py reads:
- "collapse34": the four VAE trainings of 📈 25.34 (20 latents, 20 epochs on 20 000 MNIST images, validation on the
  2 000 test images of the notebook): beta = 1, beta = 20, beta = 100, and beta rising from 0 to 1 over the first
  10 epochs, under shuffled letters; per epoch, the validation reconstruction and KL (nats per image) and the number
  of active latents (mean KL > 0.05 nat);
- "calib": the answers of the predictions 🔮 25.14, 25.17 and 25.31 over five seeds (the notebook uses one), and
  the reference of the challenge 🏆 25.39 over three seeds, whose mean validation PSNR minus 0.5 dB is the goal.
The code of the notebook is reused as is: the cells of build_ch25.py are executed in one namespace.
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
OUT = HERE / "history_ch25.json"


def notebook_namespace(fast: bool) -> dict:
    """The tools and given cells of the notebook, executed as in the solutions notebook."""
    import numpy as np
    import torch
    import wb
    if not OUT.exists():                                  # build_ch25 reads the history: give it a placeholder
        OUT.write_text(json.dumps({"collapse34": None, "calib": {"goal_39": 0.0}}), encoding="utf-8")
    import build_ch25 as B
    torch.set_num_threads(2)
    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=fast), "FAST_MODE": fast, "np": np, "__name__": "history_ch25"}
    ns["mylearn"] = wb.load_mylearn("ref")
    import matplotlib
    matplotlib.use("Agg")                                 # no window: the figures of the given cells are not drawn
    exec(B.TOOLS, ns)
    ns["show_rows"] = lambda *args, **kwargs: None
    for cell in (B.GIVEN_TRAIN, B.GIVEN_14, B.GIVEN_DEEP.split("torch.manual_seed(2516)")[0], B.GIVEN_B,
                 B.GIVEN_DENOISER, B.SOLUTION_26, B.GIVEN_VAE.split("torch.manual_seed(2530)")[0], B.GIVEN_30_HELPER):
        exec(cell, ns)
    return ns


def collapse34(ns) -> dict:
    import numpy as np
    import torch
    settings = {"A": "anneal", "B": "beta1", "C": "beta100", "D": "beta20"}
    betas = {"beta1": lambda e: 1.0, "beta20": lambda e: 20.0, "beta100": lambda e: 100.0,
             "anneal": lambda e: min(1.0, (e - 1) / 10)}          # 0 at epoch 1, 1 from epoch 11 on
    x = ns["to_tensor"](ns["X_MNIST"][:20_000])
    runs = {}
    for letter, setting in settings.items():
        torch.manual_seed(2534)
        model = ns["VAE"](20).to(ns["DEVICE"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(2534)
        run = {"rec": [], "kl": [], "active": []}
        start = time.perf_counter()
        for epoch in range(1, 21):
            model.train()
            beta = betas[setting](epoch)
            order = torch.randperm(len(x), generator=g)
            for i in range(0, len(x), 128):
                xb = x[order[i:i + 128]].to(ns["DEVICE"])
                loss, _, _ = ns["vae_loss"](xb, *model(xb), beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            rec, kl, per_latent = ns["vae_eval"](model, ns["XM_TEST"])
            run["rec"].append(round(rec, 3))
            run["kl"].append(round(kl, 3))
            run["active"].append(int(np.sum(per_latent > 0.05)))
        runs[letter] = run
        print(f"collapse34 {letter} ({setting}): rec {run['rec'][-1]:.1f} kl {run['kl'][-1]:.2f} "
              f"active {run['active'][-1]} ({time.perf_counter() - start:.0f} s)", flush=True)
    return {"settings": settings, "runs": runs}


def calib(ns) -> dict:
    import numpy as np
    import torch
    out = {}
    # 🔮 25.14: one image, five seeds
    rows = []
    for seed in range(2514, 2519):
        with_bias = ns["train_one_image_14"](ns["BOOT_14"], seed=seed)
        no_bias = ns["train_one_image_14"](ns["BOOT_14"], bias=False, seed=seed)
        out_b = ns["reconstruct"](with_bias, np.stack([ns["BLACK_14"], ns["SHIRT_14"]]))
        m = ns["mse_per_image"]
        rows.append({"seed": seed,
                     "black_vs_boot": float(m(out_b[:1], ns["BOOT_14"][None])[0]),
                     "shirt_vs_boot": float(m(out_b[1:], ns["BOOT_14"][None])[0]),
                     "shirt_vs_shirt": float(m(out_b[1:], ns["SHIRT_14"][None])[0]),
                     "no_bias_mse": float(m(ns["reconstruct"](no_bias, ns["BOOT_14"][None]), ns["BOOT_14"][None])[0])})
    out["p14"] = {"rows": rows, "answers": [all(r["black_vs_boot"] >= 0.01 for r in rows) and False or
                                            any(r["black_vs_boot"] < 0.01 for r in rows),
                                            all(r["shirt_vs_boot"] < r["shirt_vs_shirt"] for r in rows),
                                            all(r["no_bias_mse"] < 0.001 for r in rows)]}
    print("p14", out["p14"]["answers"], flush=True)

    # 🔮 25.17 and 25.31: AE_DEEP, VAE_C and CLF_31 retrained with five seeds
    from build_ch25 import GIVEN_31
    rows17, rows31 = [], []
    for k, seed in enumerate((2516, 2517, 2518, 2519, 2520)):
        torch.manual_seed(seed)
        ae = ns["make_deep_ae"](20)
        ns["train_ae"](ae, ns["X_MNIST"][:ns["N_FIXED"]], 8, seed=seed)
        d = ns["mse_per_image"](ns["reconstruct"](ae, ns["XM_TEST"]), ns["XM_TEST"])
        c = ns["mse_per_image"](ns["reconstruct"](ae, ns["XF_TEST"]), ns["XF_TEST"])
        per = [float(c[ns["YF_TEST"] == j].mean()) for j in range(10)]
        rows17.append({"seed": seed, "ratio": float(c.mean() / d.mean()), "best_two": sorted(np.argsort(per)[:2].tolist()),
                       "below_black": bool(c.mean() < ns["mse_per_image"](np.zeros_like(ns["XF_TEST"]), ns["XF_TEST"]).mean())})
        torch.manual_seed(seed + 14)
        vae = ns["VAE"](20)
        ns["train_vae"](vae, ns["X_MNIST"][:ns["N_FIXED"]], 10, seed=seed + 14)
        sub = dict(ns)
        exec(GIVEN_31.replace("torch.manual_seed(2531)", f"torch.manual_seed({2531 + k})")
             .replace("manual_seed(2531)", f"manual_seed({2531 + k})"), sub)
        res = {}
        for name, images in (("ae", ns["decode"](ae, sub["Z_31"])), ("vae", ns["vae_images"](vae, sub["Z_31"]))):
            pred, conf = sub["classify_31"](images)
            res[name + "_top"] = float(np.bincount(pred, minlength=10).max() / len(pred))
            res[name + "_conf"] = float(np.mean(conf > 0.9))
        rows31.append({"seed": seed, **res})
        print("p17", rows17[-1], "p31", rows31[-1], flush=True)
    out["p17"] = {"rows": rows17}
    out["p31"] = {"rows": rows31}

    # 🏆 25.39: the reference solution, three seeds
    from build_ch25 import SOLUTION_39
    rows39 = []
    for seed in (2539, 2540, 2541):
        sub = dict(ns)
        sub.update({"MAX_PARAMS_39": 100_000})
        exec(SOLUTION_39.replace("seed=2539", f"seed={seed}"), sub)
        torch.manual_seed(seed)
        model = sub["make_denoiser_39"]()
        start = time.perf_counter()
        sub["train_39"](model.to(ns["DEVICE"]), ns["X_FASHION"])
        seconds = time.perf_counter() - start
        test_clean = ns["XF_TEST"][1000:2000]
        test_noisy = np.clip(test_clean + 0.3 * np.random.default_rng(2539).standard_normal(test_clean.shape), 0, 1).astype(np.float32)
        rows39.append({"seed": seed, "seconds": round(seconds, 1), "params": ns["n_params"](model),
                       "val_psnr": round(ns["psnr"](ns["VAL_CLEAN_B"], ns["reconstruct"](model, ns["VAL_NOISY_B"])), 3),
                       "test_psnr": round(ns["psnr"](test_clean, ns["reconstruct"](model, test_noisy)), 3)})
        print("p39", rows39[-1], flush=True)
    out["p39"] = {"rows": rows39}
    out["goal_39"] = round(float(np.mean([r["val_psnr"] for r in rows39])) - 0.5, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["collapse34", "calib"])
    args = parser.parse_args()
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if args.only in (None, "calib"):
        data["calib"] = calib(notebook_namespace(fast=True))
        OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    if args.only in (None, "collapse34"):
        data["collapse34"] = collapse34(notebook_namespace(fast=False))
        OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
