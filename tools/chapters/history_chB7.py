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
- "mp5data": the fingerprint of the 45 000 training images of the MP5 reference model (cifarkit.stratified_split, seed
  0), checked against wbp.MP5_DATA and against the channel means of its model.json;
- "l19": 🔮 B7.19, the latency protocol of EXPERIMENT_19 (reference API in another process) repeated for 2 and 1
  threads, FastAPI and http.server, three times: the two ratios and their categories must be the same everywhere;
- "d22": 📈 B7.22, the table of EXPERIMENT_22 (SOLUTION_22), and its answers again with other noise seeds and the
  reference and production halves swapped: the answers must not change;
- "c25": 🏆 B7.25, the chain of the self-check on the reference package (git clone, editable install, pytest, demo),
  three times, with wbp.machine_speed(): the speed constant of the challenge and the worst time.
The cells of the notebook are reused as is (tools/chapters/chB7_code*.py), executed in one namespace.
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


def mp5data() -> dict:
    import wb
    from wb import prod as wbp

    sys.path.insert(0, str(ROOT / "projets" / "partie_5_classifieur_cifar10" / "solution"))
    wb.load_mylearn("ref")
    import data as mp5
    from cifarkit.pipeline import stratified_split

    X, y = mp5.load_cifar10()
    train_idx, _ = stratified_split(y, 5000, 0)
    sha = wbp.array_fingerprint(X[train_idx], y[train_idx])
    mean = (X[train_idx] / 255.0).mean(axis=(0, 1, 2))
    meta = json.loads((wbp.reference_folder() / "model.json").read_text(encoding="utf-8"))
    out = {"sha256": sha, "n_train": int(len(train_idx)), "same_as_wbp": sha == wbp.MP5_DATA["sha256"],
           "mean_matches_model_json": bool(np.allclose(mean, meta["mean"], rtol=0, atol=1e-12))}
    print("mp5data", out)
    if not (out["same_as_wbp"] and out["mean_matches_model_json"]):
        raise SystemExit("mp5data: wbp.MP5_DATA does not match the MP5 split")
    return out


def l19() -> dict:
    import requests
    import torch

    from wb import prod as wbp

    ns = namespace()
    import chB7_code2 as C2
    exec(C2.GIVEN_17, ns)
    folder = wbp.reference_artifact(ns["scratch"]("artifact19") / "vgg")
    model, info = wbp.reference_model()
    x = ns["model_input"](ns["X_B"], info["mean"], info["std"])
    body1, body32 = {"pixels": ns["X_B"][0].tolist()}, {"images": ns["X_B"][:32].tolist()}
    rows, keep = [], torch.get_num_threads()
    for repeat in range(3):
        for threads in (2, 1):
            torch.set_num_threads(threads)
            with torch.inference_mode():
                direct = wbp.latencies(lambda: model(x[:1]), 200, 20)
            for backend in ("fastapi", "stdlib"):
                with wbp.serve_process(wbp.reference_package() / "src", folder, backend=backend,
                                       env={"OMP_NUM_THREADS": str(threads)}) as server:
                    s = requests.Session()
                    one = wbp.latencies(lambda: s.post(f"{server.url}/predict", json=body1, timeout=30), 200, 20)
                    many = wbp.latencies(lambda: s.post(f"{server.url}/predict_batch", json=body32, timeout=30), 40, 3)
                row = {"repeat": repeat, "threads": threads, "backend": backend,
                       "direct_p50": float(np.percentile(direct, 50)), "direct_p95": float(np.percentile(direct, 95)),
                       "one_p50": float(np.percentile(one, 50)), "one_p95": float(np.percentile(one, 95)),
                       "batch_p50": float(np.percentile(many, 50)), "batch_p95": float(np.percentile(many, 95))}
                row["a"] = row["one_p95"] / row["direct_p95"]
                row["b"] = row["batch_p95"] / row["one_p95"]
                row["gain"] = 32 * row["one_p50"] / row["batch_p50"]
                rows.append(row)
                print("l19", {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    torch.set_num_threads(keep)
    cat = lambda r, lo, hi: "A" if r < lo else "B" if r <= hi else "C"
    one = [r for r in rows if r["threads"] == 1]               # the protocol of the notebook: one thread
    answers = {(cat(r["a"], 2, 20), cat(r["b"], 3, 32)) for r in one}
    if len(answers) != 1:
        raise SystemExit(f"l19: the categories change from one run to another: {answers}")
    a, b = answers.pop()
    pick = lambda key, rs: [round(min(r[key] for r in rs), 2), round(max(r[key] for r in rs), 2)]
    keys = ("a", "b", "gain", "direct_p50", "direct_p95", "one_p50", "one_p95", "batch_p50", "batch_p95")
    return {"answers": {"a": a, "b": b}, "rows": rows, "ranges": {k: pick(k, one) for k in keys},
            "ranges_fastapi": {k: pick(k, [r for r in one if r["backend"] == "fastapi"]) for k in keys},
            "ranges_two_threads": {k: pick(k, [r for r in rows if r["threads"] == 2]) for k in keys},
            "direct_p50_default": float(np.median([r["direct_p50"] for r in one]))}


def d22() -> dict:
    import chB7_code2 as C2
    import chB7_code3 as C3

    ns = namespace()
    for block in (C2.GIVEN_17, C3.GIVEN_22, C3.SOLUTION_22, C3.CHECK_22):
        exec(block, ns)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ns["plt"].show = lambda *a, **k: None
        exec(C3.EXPERIMENT_22, ns)
    table = ns["TABLE_22"]

    def answers(t):
        letters = lambda f: "".join(k for k in sorted(t) if f(t[k]))
        return {"a": letters(lambda r: r["psi"] > 0.25), "b": letters(lambda r: r["p"] < 0.05),
                "c": letters(lambda r: r["psi_classes"] > 0.25),
                "d": letters(lambda r: r["accuracy"] < ns["acc_ref_22"] - 0.10), "e": round(t["D"]["psi"], 2)}

    main = answers(table)
    others = []
    for seed, swap in [(0, False), (1, False), (2, True), (22, True)]:
        X, Y = ns["X_22"], ns["Y_22"]
        ref, prod, ry, py = (X[1::2], X[::2], Y[1::2], Y[::2]) if swap else (X[::2], X[1::2], Y[::2], Y[1::2])
        wins = {"A": prod, "B": np.round(prod * 0.6).astype(np.uint8),
                "C": np.clip(prod + np.random.default_rng(seed).normal(0, 25, prod.shape), 0, 255).round().astype(np.uint8),
                "D": np.round(prod * 0.9).astype(np.uint8)}
        proba = {k: ns["offline_b"](ns["MODEL_22"], v, ns["REF_INFO"]["mean"], ns["REF_INFO"]["std"])[0]
                 for k, v in {"ref": ref, **wins}.items()}
        light = {k: ns["brightness_22"](v) for k, v in {"ref": ref, **wins}.items()}
        pred = {k: v.argmax(axis=1) for k, v in proba.items()}
        acc_ref = float((pred["ref"] == ry).mean())
        t = {}
        for k in wins:
            from scipy import stats
            t[k] = {"psi": ns["psi_22"](light["ref"], light[k]), "p": float(stats.ks_2samp(light["ref"], light[k]).pvalue),
                    "psi_classes": ns["psi_classes_22"](pred["ref"], pred[k]), "accuracy": float((pred[k] == py).mean())}
        ns["acc_ref_22"], keep = acc_ref, ns["acc_ref_22"]
        got = answers(t)
        ns["acc_ref_22"] = keep
        others.append({"seed": seed, "swap": swap, **{k: v for k, v in got.items() if k != "e"},
                       "psi_D": round(t["D"]["psi"], 3), "table": t})
        print("d22", seed, swap, got)
    stable = all(all(o[k] == main[k] for k in "abcd") for o in others)
    print("d22 main", main, "stable", stable)
    if not stable:
        raise SystemExit("d22: the letters change with the noise seed or the halves")
    return {"answers": main, "table": table, "acc_ref": ns["acc_ref_22"], "others": others,
            "psi_D_exact": table["D"]["psi"]}


def c25() -> dict:
    import tempfile

    from wb import prod as wbp

    runs = []
    for repeat in range(3):
        speed = wbp.machine_speed()
        project = Path(tempfile.mkdtemp(prefix="wb_b7_c25_")) / "cifar-prod"
        shutil.copytree(wbp.reference_package(), project, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        for args in (["git", "init", "-q"], ["git", "add", "-A"],
                     ["git", "-c", "user.name=w", "-c", "user.email=w@example.org", "commit", "-q", "-m", "c25"]):
            wbp.run(args, project, timeout=60)
        start = time.perf_counter()
        clone, how, _ = wbp.clone_project(project, project.parent / "clone")
        install = wbp.install_editable(clone)
        try:
            tests = install.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--strict-markers", "-c",
                                 str(clone / "pyproject.toml"), "--rootdir", str(clone)], clone, timeout=600)
            demo = install.run(["python", "scripts/demo.py"], clone, timeout=600)
        finally:
            install.cleanup()
        total = time.perf_counter() - start
        row = {"speed": speed, "how": how, "install": install.seconds, "tests": tests.seconds, "tests_ok": tests.ok,
               "demo": demo.seconds, "demo_ok": demo.ok, "total": total, "summary": demo.last_json(),
               "tests_summary": (tests.stdout.strip().splitlines() or [""])[-1]}
        runs.append(row)
        print("c25", {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        if not (tests.ok and demo.ok):
            raise SystemExit(f"c25: the reference chain fails: {tests.stdout[-500:]} {demo.stderr[-500:]}")
    return {"runs": runs, "calib_speed": float(np.median([r["speed"] for r in runs])),
            "worst_total": float(max(r["total"] for r in runs)),
            "worst_equivalent": float(max(r["total"] * np.median([q["speed"] for q in runs]) / r["speed"] for r in runs))}


ENTRIES = {"p12": p12, "b16": b16, "t13": t13, "mp5data": mp5data, "l19": l19, "d22": d22, "c25": c25}


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
