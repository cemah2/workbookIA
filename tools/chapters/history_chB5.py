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


# =====================================================================================================================
# Second half (B5.22 to B5.31)
# =====================================================================================================================
PROTOCOLS = ROOT / "src" / "wb" / "diffusion_protocols.json"
CURVE_STEPS = (5, 10, 20, 50, 100)
UNET_SEEDS = (0, 1, 2)                 # seed k: the notebook's seeds plus 100 k (k = 0: the notebook itself)
RHO_SOLUTION = 0.99                    # RHO_23 of the solution, fast mode


def _fd(real, clf, images):
    from mylearn_ref.gen_metrics import frechet_distance_from_features
    from wb import diffusion as wbd
    return float(frechet_distance_from_features(real, wbd.features(clf, images.clamp(-1, 1))))


def b22(ns) -> dict:
    """🐛 B5.22: the four diagnoses on the 2**4 versions of the pipeline (each must stop at its first bug, without
    crashing), and the samples of each version with the perfect denoiser."""
    import contextlib
    import io
    import itertools

    import chB5_code2 as C2

    exec(C2.GIVEN_22, ns)
    fixes = {"A": ("[:, None] / 255", "[:, None] / 127.5 - 1"),
             "C": ("[t - 1].view(-1, 1, 1, 1)   # step t of the paper", "[t].view(-1, 1, 1, 1)"),
             "B": ("return ab * x0", "return ab.sqrt() * x0"),
             "D": ("        x = x + math.sqrt(betas[t]) * torch.randn(x.shape, generator=generator)",
                   "        if t > 0:\n            x = x + math.sqrt(betas[t]) * torch.randn(x.shape, generator=generator)")}
    order = ["A", "C", "B", "D"]                      # the order of the diagnoses: scale, index, coefficients, last step
    rows = {}
    for present in itertools.product((True, False), repeat=4):
        code = C2.TODO_22
        for letter, bug in zip(order, present):
            if not bug:
                assert fixes[letter][0] in code, letter
                code = code.replace(fixes[letter][0], fixes[letter][1])
        exec(code, ns)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = ns["diagnose_22"](ns["prepare_22"], ns["noisy_22"], ns["sample_22"])
            out = ns["show_pipeline_22"](ns["prepare_22"], ns["noisy_22"], ns["sample_22"], "x")
        text = buf.getvalue()
        first = next((k for k, bug in enumerate(present, start=1) if bug), None)
        passed = text.count("✅ Ex B5.22 : point")
        failed_at = None if ok else passed + 1
        name = "".join(letter for letter, bug in zip(order, present) if bug) or "-"
        if failed_at != first or "Traceback" in text or out is None:
            raise SystemExit(f"❌ b22: version {name}: stops at {failed_at}, expected {first}\n{text}")
        rows[name] = {"stops_at": failed_at, "mean": float(out.mean()), "min": float(out.min()), "max": float(out.max())}
        print(f"  b22 bugs {name:>4}: stops at point {failed_at}; samples mean {out.mean():.3f}", flush=True)
    return {"versions": rows}


def _train_mnist(seed_k, kind="linear", n_images=None, steps=None):
    """The recipe of B5.23 (fast mode, the solution's rho) with the seeds of the notebook plus 100 k; with n_images,
    on the first n_images of the frozen subset for the same number of optimizer steps as the recipe."""
    import torch
    from wb import diffusion as wbd

    s = wbd.unet_settings(True) | {"ema": RHO_SOLUTION}
    n_steps_recipe = wbd.n_updates(s["n_train"], s)
    data = wbd.image_subset("mnist", n_images or s["n_train"])
    if n_images is not None:
        s = s | {"epochs": math.ceil(n_steps_recipe / math.ceil(len(data) / s["batch_size"]))}
    torch.manual_seed(s["seed"] + 100 * seed_k)
    model = wbd.TinyUNet(s["T"], channels=s["channels"], t_dim=s["t_dim"])
    ema = wbd.TinyUNet(s["T"], channels=s["channels"], t_dim=s["t_dim"])
    ema.load_state_dict(model.state_dict())
    ab = wbd.alpha_bars(wbd.schedule_betas(kind, s["T"]))
    hist = wbd.train_unet(model, ema, data.images, s, torch.Generator().manual_seed(s["seed"] + 1 + 100 * seed_k), ab,
                          max_steps=n_steps_recipe)
    return ema, ab, data, hist


def unet25(ns) -> dict:
    """📦 B5.23 and 🔬 B5.25: the recipe of B5.23 with the linear (L) and the cosine (C) schedule, 3 seeds each; the
    distance of 256 DDIM samples (the noise of the judge of B5.23, seed 2324) for 5 to 100 steps; for L, the copy
    ratios of the samples at 50 steps (🔬 B5.28, 10 000 images)."""
    import torch
    from wb import diffusion as wbd

    clf = wbd.classifier("mnist")
    real = wbd.features(clf, wbd.image_subset("mnist", 2000, split="test").images)
    x = torch.randn((256, 1, 28, 28), generator=torch.Generator().manual_seed(2324))
    out = {"steps": list(CURVE_STEPS), "L": [], "C": [], "seconds": [], "loss_end": [], "copies": [], "median_r": []}
    out["floor"] = _fd(real, clf, wbd.image_subset("mnist", 10000).images[:256])
    for kind, key in (("linear", "L"), ("cosine", "C")):
        for k in UNET_SEEDS:
            ema, ab, data, hist = _train_mnist(k, kind)
            row = []
            for n in CURVE_STEPS:
                images = wbd.sample_ddim(ema, x, ab, n)
                row.append(_fd(real, clf, images))
                if key == "L" and n == 50:
                    r = wbd.copy_ratio(images, data.images)
                    out["copies"].append(float(np.mean(r < 1 / 3)))
                    out["median_r"].append(float(np.median(r)))
            out[key].append(row)
            if key == "L":
                out["seconds"].append(hist["seconds"])
                out["loss_end"].append(float(np.mean(hist["loss"][-100:])))
            print(f"  unet25 {key} seed {k}: {np.round(row, 1).tolist()} ({hist['seconds']:.0f} s)", flush=True)
    return out


def mem28(ns) -> dict:
    """🔬 B5.28: the recipe of B5.23 with the same number of optimizer steps on 500 and on 60 000 images, 2 seeds;
    the copy ratios of 256 DDIM samples (50 steps) against each training set."""
    import torch
    from wb import diffusion as wbd

    x = torch.randn((256, 1, 28, 28), generator=torch.Generator().manual_seed(2324))
    out = {"copies": {}, "median_r": {}}
    for n_images in (500, 60000):
        out["copies"][str(n_images)], out["median_r"][str(n_images)] = [], []
        for k in (0, 1):
            ema, ab, data, hist = _train_mnist(k, "linear", n_images=n_images)
            r = wbd.copy_ratio(wbd.sample_ddim(ema, x, ab, 50), data.images)
            out["copies"][str(n_images)].append(float(np.mean(r < 1 / 3)))
            out["median_r"][str(n_images)].append(float(np.median(r)))
            print(f"  mem28 {n_images} seed {k}: {hist['steps']} steps, copies {np.mean(r < 1 / 3):.3f}, median r "
                  f"{np.median(r):.3f}", flush=True)
    return out


GUIDANCE_SEEDS = (0, 1, 2, 3)


def guidance26(ns) -> dict:
    """🔮 B5.26 and 📦 B5.27: the conditional recipe of B5.27 (fast mode) on Fashion-MNIST, 4 seeds; for w = 0, 1, 3, 7,
    200 images (20 per class, DDIM 20 steps): distance, accuracy on the requested class, diversity, saturation."""
    import torch
    from mylearn_ref.gen_metrics import frechet_distance_from_features
    from wb import diffusion as wbd

    T = wbd.n_steps(True)
    ab = wbd.alpha_bars(wbd.schedule_betas("linear", T))
    s = wbd.unet_settings(True) | {"epochs": 2, "ema": 0.99, "p_uncond": 0.1, "seed": 2700}
    data = wbd.image_subset("fashion", s["n_train"])
    test = wbd.image_subset("fashion", 2000, split="test")
    clf = wbd.classifier("fashion")
    real = wbd.features(clf, test.images)
    y = torch.arange(10).repeat_interleave(20)
    by_w = {str(w): {"fd": [], "acc": [], "div": [], "sat": []} for w in (0, 1, 3, 7)}
    for k in GUIDANCE_SEEDS:
        torch.manual_seed(s["seed"] + 100 * k)
        model = wbd.TinyUNet(T, channels=s["channels"], t_dim=s["t_dim"], n_classes=10)
        ema = wbd.TinyUNet(T, channels=s["channels"], t_dim=s["t_dim"], n_classes=10)
        ema.load_state_dict(model.state_dict())
        wbd.train_unet(model, ema, data.images, s, torch.Generator().manual_seed(s["seed"] + 1 + 100 * k), ab,
                       labels=data.labels)
        x = torch.randn((200, 1, 28, 28), generator=torch.Generator().manual_seed(271 + 100 * k))
        for w in (0, 1, 3, 7):
            images = wbd.sample_ddim(ema, x, ab, 20, y=y, w=w).clamp(-1, 1)
            f = wbd.features(clf, images)
            row = by_w[str(w)]
            row["fd"].append(float(frechet_distance_from_features(real, f)))
            row["acc"].append(float(np.mean(wbd.predict(clf, images) == y.numpy())))
            row["div"].append(float(np.mean([f[y.numpy() == c].std(axis=0).mean() for c in range(10)])))
            row["sat"].append(float((images.abs() > 0.98).float().mean()))
        print(f"  guidance26 seed {k}: " + "; ".join(f"w={w} fd {by_w[str(w)]['fd'][-1]:.1f} acc {by_w[str(w)]['acc'][-1]:.2f} "
                                                   f"div {by_w[str(w)]['div'][-1]:.3f} sat {by_w[str(w)]['sat'][-1]:.3f}"
                                                   for w in (0, 1, 3, 7)), flush=True)
    real_div = float(np.mean([real[test.labels.numpy() == c].std(axis=0).mean() for c in range(10)]))
    table = {"n_seeds": len(GUIDANCE_SEEDS), "n_images": 200, "steps": 20, "by_w": by_w,
             "real": {"div": real_div, "sat": float((test.images.abs() > 0.98).float().mean()),
                      "fd": float(frechet_distance_from_features(real, wbd.features(clf, data.images[:200])))}}
    fd0, fd7 = np.array(by_w["0"]["fd"]), np.array(by_w["7"]["fd"])
    acc0 = np.array(by_w["0"]["acc"])
    sat0, sat7 = np.array(by_w["0"]["sat"]), np.array(by_w["7"]["sat"])
    div0 = np.array(by_w["0"]["div"])
    answers = {}
    for name, below, above in (("fd7", fd7 < 0.9 * fd0, fd7 > 1.1 * fd0), ("acc0", acc0 > 0.77, acc0 < 0.63),
                               ("div0", div0 > 1.1 * real_div, div0 < 0.9 * real_div)):
        if below.all():
            answers[name] = True
        elif above.all():
            answers[name] = False
        else:
            raise SystemExit(f"❌ guidance26: answer {name} depends on the seed")
    margins = {"fd7_over_fd0_max": float((fd7 / fd0).max()), "fd7_over_fd0_min": float((fd7 / fd0).min()),
               "acc0_range": [float(acc0.min()), float(acc0.max())],
               "sat7_over_sat0": [float((sat7 / sat0).min()), float((sat7 / sat0).max())]}
    print(f"  guidance26 answers {answers}, margins {margins}", flush=True)
    data_p = json.loads(PROTOCOLS.read_text(encoding="utf-8")) if PROTOCOLS.exists() else {}
    data_p["guidance_26"] = table
    PROTOCOLS.write_text(json.dumps(data_p, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"table": table, "answers": answers, "margins": margins}


DEFI_SEEDS = {"fast": (0, 1, 2, 3), "full": (0, 1)}
DEFI_LIMIT = {"fast": 60.0, "full": 600.0}
DEFI_W = (1.0, 2.0, 3.0)


def _defi_reference(images, labels, seconds, seed, kind="linear"):
    """The reference method of 🏆 B5.31 (SOLUTION_31 with the reference functions): a conditional small U-Net trained
    for `seconds`, an EMA that warms up; returns the EMA model and its alpha_bars."""
    import copy

    import torch
    from wb import diffusion as wbd

    F = torch.nn.functional
    start = time.perf_counter()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    T = 400
    ab_np = wbd.alpha_bars(wbd.schedule_betas(kind, T))
    ab = torch.tensor(ab_np, dtype=torch.float32)
    model = wbd.TinyUNet(T, channels=(16, 32, 64), n_classes=10)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    step = 0
    model.train()
    while time.perf_counter() - start < 0.95 * seconds:
        idx = torch.randint(len(images), (32,), generator=generator)
        x0, y = images[idx], labels[idx]
        t = torch.randint(0, T, (32,), generator=generator)
        eps = torch.randn(x0.shape, generator=generator)
        y = torch.where(torch.rand(32, generator=generator) < 0.1, torch.full_like(y, 10), y)
        a = ab[t].view(-1, 1, 1, 1)
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, y), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        wbd.ema_update(ema, model, min(0.99, (1 + step) / (10 + step)))
        step += 1
    ema.eval()
    return ema, ab_np, step, time.perf_counter() - start


def defi31(ns, modes=("fast", "full")) -> dict:
    """🏆 B5.31: the reference method on Fashion-MNIST, for each mode and seed (training of LIMIT seconds), the
    distance of N images (DDIM 25 steps with guidance, 50 passes) for w = 1, 2, 3, and their copy ratios."""
    import torch
    from mylearn_ref.gen_metrics import frechet_distance_from_features
    from wb import diffusion as wbd

    test = wbd.image_subset("fashion", 2000, split="test")
    clf = wbd.classifier("fashion")
    real = wbd.features(clf, test.images)
    out = {}
    for mode in modes:
        fast = mode == "fast"
        train = wbd.image_subset("fashion", wbd.n_train(fast))
        n = 256 if fast else 1000
        rows = []
        for k in DEFI_SEEDS[mode]:
            ema, ab, steps, seconds = _defi_reference(train.images, train.labels, DEFI_LIMIT[mode], 3100 + 100 * k)
            gen = torch.Generator().manual_seed(3132)
            x_T = torch.randn((n, 1, 28, 28), generator=gen)
            y = torch.arange(n) % 10
            row = {"steps": steps, "seconds": seconds, "fd": {}, "copies": {}}
            for w in DEFI_W:
                images = wbd.sample_ddim(ema, x_T, ab, 25, y=y, w=w).clamp(-1, 1)
                row["fd"][str(w)] = float(frechet_distance_from_features(real, wbd.features(clf, images)))
                row["copies"][str(w)] = float(np.mean(wbd.copy_ratio(images, train.images) < 1 / 3))
            rows.append(row)
            print(f"  defi31 {mode} seed {k}: {steps} steps; fd {row['fd']}; copies {row['copies']}", flush=True)
        out[mode] = rows
    return out


ENTRIES.update({"b22": b22, "guidance26": guidance26, "unet25": unet25, "mem28": mem28, "defi31": defi31})


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
