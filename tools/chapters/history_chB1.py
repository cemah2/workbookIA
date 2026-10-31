#!/usr/bin/env python
"""Calibrate the predictions and replay the bug hunt of chapter B1, parts A and B (used by Claude).

    python tools/chapters/history_chB1.py                 # everything (about 6 min on a 2-core CPU)
    python tools/chapters/history_chB1.py --only bug18    # one entry: p12, u15, bug18, variants or timing

Writes tools/chapters/history_chB1.json (statistics only), which chB1_parts.py reads:
- "p12": 🔮 B1.12 for three draws of the nested subsets (seeds): the linear probe accuracy on the embeddings of the
  32-pixel images and the share of the test images covered by the 5 most frequent ImageNet classes; the categories of
  the statement must be the same for every draw;
- "u15": 📈 B1.15 for three draws of the 600 test images: the 10-NN accuracy in the plane of each PCA and UMAP, and
  the share of the right family (vehicles or animals); the answers must be the same for every draw;
- "bug18": 🐛 B1.18 replayed on its 2**4 partially fixed versions: the first point that fails must be the first error
  left (and nothing must crash);
- "variants": the numbers of the variants of 05_solutions.md (B1.17 with gamma = 0.3, B1.19 with gamma = 0.5);
- "timing": one step of full fine-tuning at 96 and 224 pixels, for the full-mode estimates.
The cells of the notebook are reused as is (tools/chapters/chB1_code.py), executed in one namespace.
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
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
OUT = HERE / "history_chB1.json"
SEEDS = (2011, 1, 2)                        # 2011 = wb.datasets.CIFAR10_SUBSET_SEED, the draw of the notebook

# The four fixes of finetune_18, one text edit each (point k of diagnose_18 checks fix k)
EDITS_18 = {
    1: ("    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)\n    model.fc = nn.Linear(model.fc.in_features, 10)\n",
        "    model.fc = nn.Linear(model.fc.in_features, 10)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)\n"),
    2: ('        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)\n',
        '        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)\n'
        '        x = (x - torch.tensor(wb.datasets.IMAGENET_MEAN).view(1, 3, 1, 1)) / '
        'torch.tensor(wb.datasets.IMAGENET_STD).view(1, 3, 1, 1)\n'),
    3: ("    model.train()\n", "    model.train()\n    freeze_batchnorm_17(model)\n"),
    4: ("torch.optim.AdamW(model.parameters(), lr=1e-1)",
        'torch.optim.AdamW([{"params": model.fc.parameters(), "lr": 1e-3}, '
        '{"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 1e-4}])'),
}


def namespace(blocks=()) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wb

    import chB1_code as C

    plt.show = lambda *a, **k: plt.close("all")
    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True), "FAST_MODE": True, "__name__": "history_chB1"}
    for block in ("TOOLS", *blocks):
        exec(getattr(C, block), ns)
    return ns


def p12(ns) -> dict:
    import numpy as np
    import torch
    import wb
    runs = []
    for seed in SEEDS:
        X_tr, y_tr = wb.datasets.load_cifar10_subset(200, seed=seed)
        X_te, y_te = wb.datasets.load_cifar10_subset(200, "test", seed=seed)
        row = {"seed": seed}
        for size in (32, 96):
            E_tr, E_te = ns["RESNET"].embed(X_tr, size), ns["RESNET"].embed(X_te, size)
            with torch.inference_mode():
                pred = ns["RESNET"].full.fc(torch.from_numpy(E_te)).argmax(1).numpy()
            _, counts = np.unique(pred, return_counts=True)
            row[f"probe{size}"] = round(ns["reference_probe"](E_tr, y_tr, E_te, y_te), 4)
            row[f"top5_{size}"] = round(float(np.sort(counts)[::-1][:5].sum() / len(pred)), 4)
            row[f"classes{size}"] = int(len(counts))
        runs.append(row)
        print(f"  p12 {row}", flush=True)
    return {"runs": runs}


def u15(ns) -> dict:
    import warnings

    import numpy as np
    import wb
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import umap
    dino = ns["transfer"].load_backbone("dinov2_small")
    runs = []
    for seed in SEEDS:
        X, y = wb.datasets.load_cifar10_subset(60, "test", seed=seed)
        reps = {"A": X.reshape(len(X), -1) / 255.0, "B": ns["RESNET"].embed(X, ns["SIZE"]), "C": dino.embed(X, ns["DINO_SIZE"])}
        row = {"seed": seed}
        for key, E in reps.items():
            Z0 = StandardScaler().fit_transform(E)
            for method, Z in (("pca", PCA(2, svd_solver="full").fit_transform(Z0)),
                              ("umap", umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0, n_jobs=1).fit_transform(Z0))):
                pred = cross_val_predict(KNeighborsClassifier(10), Z, y, cv=5)
                row[f"{method}_knn2d_{key}"] = round(float(np.mean(pred == y)), 3)
                row[f"{method}_family_{key}"] = round(float(np.mean(np.isin(pred, [0, 1, 8, 9]) == np.isin(y, [0, 1, 8, 9]))), 3)
        runs.append(row)
        print(f"  u15 {row}", flush=True)
    separated = {max("ABC", key=lambda k: r[f"umap_knn2d_{k}"]) for r in runs}
    pixel_classes = {r["umap_knn2d_A"] >= 0.35 for r in runs}        # the groups of the pixels' UMAP are not classes
    vehicles = {r["umap_family_B"] >= 0.85 for r in runs}
    if len(separated) != 1 or len(pixel_classes) != 1 or len(vehicles) != 1:
        raise SystemExit(f"❌ u15: the answers change from one draw to another: {separated}, {pixel_classes}, {vehicles}")
    return {"runs": runs, "answers": {"separated": separated.pop(), "pixel_classes": pixel_classes.pop(), "vehicles": vehicles.pop()}}


def bug18(ns) -> dict:
    import chB1_code as C
    rows = []
    for mask in itertools.product((0, 1), repeat=4):
        src = C.TODO_18
        for k, on in zip((1, 2, 3, 4), mask):
            if on:
                assert EDITS_18[k][0] in src, f"edit {k} does not apply"
                src = src.replace(EDITS_18[k][0], EDITS_18[k][1])
        exec(src, ns)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = ns["diagnose_18"](ns["finetune_18"])
        text = out.getvalue()
        fails = [line for line in text.splitlines() if line.startswith("❌")]
        first = int(fails[0].split("point ")[1].split(" ")[0]) if fails else None
        want = next((k for k, on in zip((1, 2, 3, 4), mask) if not on), None)
        rows.append({"fixed": "".join(str(k) for k, on in zip((1, 2, 3, 4), mask) if on), "first_failure": first,
                     "expected": want, "all_pass": bool(ok)})
        print(f"  bug18 fixed {rows[-1]['fixed'] or '-':<5} first failing point {first} (expected {want})", flush=True)
        if first != want or len(fails) > 1:
            raise SystemExit(f"❌ bug18: version {mask} shows {fails}")
    return {"versions": rows}


def variants(ns) -> dict:
    import torch
    out = {}
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = ns["make_partial_17"]()
        ns["train_partial_17"](model, ns["XT_B"], ns["YT_B"], ns["FT_EPOCHS"], head_lr=1e-3, gamma=0.3)
        out["b17_gamma03_test"] = round(ns["evaluate"](model, ns["X_TEST"], ns["Y_TEST"]), 4)
    print(f"  variants {out}", flush=True)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = ns["make_full_19"]()
        hist = ns["fine_tune_19"](model, (ns["XT_B"], ns["YT_B"]), (ns["X_VAL"], ns["Y_VAL"]), epochs=ns["FT_EPOCHS"],
                                  head_lr=1e-3, gamma=0.5, t_warmup=16, patience=1)
        out["b19_gamma05_val"] = [round(v, 4) for v in hist["val_acc"]]
        out["b19_gamma05_test"] = round(ns["evaluate"](model, ns["X_TEST"], ns["Y_TEST"]), 4)
    print(f"  variants {out}", flush=True)
    return out


def timing(ns) -> dict:
    import torch
    import torch.nn.functional as F
    out = {}
    model = ns["make_full_19"]().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for size in (96, 224):
        x, y = torch.randn(32, 3, size, size), torch.randint(0, 10, (32,))
        times = []
        for _ in range(4):
            start = time.perf_counter()
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            times.append(time.perf_counter() - start)
        out[f"step32_{size}"] = [round(min(times[1:]), 2), round(max(times[1:]), 2)]
    print(f"  timing {out}", flush=True)
    return out


ENTRIES = {"p12": ((), p12), "u15": ((), u15), "bug18": (("GIVEN_B", "SOLUTION_17", "GIVEN_18"), bug18),
           "variants": (("GIVEN_B", "SOLUTION_17", "SOLUTION_19"), variants),
           "timing": (("GIVEN_B", "SOLUTION_17", "SOLUTION_19"), timing)}


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
