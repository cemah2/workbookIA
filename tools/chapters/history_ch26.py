#!/usr/bin/env python
"""Precompute the multi-seed experiments of chapter 26 (used by Claude).

    python tools/chapters/history_ch26.py                  # everything (about 2 min on a CPU)
    python tools/chapters/history_ch26.py --only curves23  # one entry: l16, curves23 or calib

Writes tools/chapters/history_ch26.json, which build_ch26.py reads:
- "l16": the L-learning results shown by the prediction 🔮 26.16 (and reproduced in 🔬 26.18): for five seeds, a table
  trained 6 000 games without the truck and one trained 6 000 games with a truck of probability 0.1, each then played
  once, greedily and without the truck, on the 506 non-winning boards;
- "curves23": the four FrozenLake trainings of 📈 26.23 (slippery lake, 4 000 episodes, seed 0) under shuffled
  letters: the moving average of the training returns and the success of the greedy policy (1 000 episodes);
- "calib": the outcomes of the predictions 🔮 26.28 over eight seeds (the notebook uses one), and the reference of the
  challenge 🏆 26.38 over three training seeds (mean length over 100 episodes of 500 steps, public and hidden seeds).
The code of the notebook is reused as is: the cells of build_ch26.py are executed in one namespace.
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
OUT = HERE / "history_ch26.json"

SETTINGS_23 = {                      # name -> (alpha, epsilon); gamma = 0.99, 4 000 episodes, 100 steps at most
    "decay": (0.1, "decay"),         # epsilon from 1 to 0.05 over the first 80 % of the episodes
    "greedy": (0.1, 0.0),            # no exploration (ties at random only)
    "alpha1": (1.0, "decay"),        # learning rate 1: each update replaces the value
    "eps05": (0.1, 0.5),             # a constant, large exploration
}
LETTERS_23 = {"decay": "C", "greedy": "A", "alpha1": "D", "eps05": "B"}


def namespace() -> dict:
    import numpy as np
    import wb

    import build_ch26 as B

    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True), "np": np, "mylearn": wb.load_mylearn("ref"),
          "FAST_MODE": True, "__name__": "history_ch26"}
    exec(B.TOOLS, ns)
    exec(B.SOLUTION_18, ns)
    return ns


def l16(ns) -> dict:
    np = ns["np"]
    rows = []
    for seed in range(5):
        for truck in (0.0, 0.1):
            L, _ = ns["l_learning_18"](ns["Flippers"](truck_prob=truck), 6000, np.random.default_rng(seed))
            result = ns["evaluate_flippers"](L, truck_prob=0.0, seed=16)
            extra = result["length"] - result["optimal"]
            rows.append({"seed": seed, "truck": truck, "won": float(result["won"].mean()),
                         "optimal": float(np.mean(extra == 0)), "mean_length": float(result["length"].mean()),
                         "zero_cells": float(np.mean(L == 0))})
            print(f"  l16 seed {seed} truck {truck}: {rows[-1]}", flush=True)
    return {"rows": rows}


def curves23(ns) -> dict:
    np, rl = ns["np"], ns["mylearn"].rl
    env = ns["make_env"]("FrozenLake-v1", seed=0, is_slippery=True)
    n = 4000
    runs = {}
    for name, (alpha, epsilon) in SETTINGS_23.items():
        if epsilon == "decay":
            epsilon = (lambda i: max(0.05, 1.0 - 0.95 * i / (0.8 * n)))
        Q, history = rl.train_tabular(env, n, alpha=alpha, gamma=0.99, epsilon=epsilon, max_steps=100,
                                      rng=np.random.default_rng(0))
        score = rl.evaluate_policy(env, rl.greedy_policy(Q), 1000, max_steps=100, rng=np.random.default_rng(23))
        curve = ns["moving_average"](history["returns"], 100)
        runs[LETTERS_23[name]] = {"curve": [round(float(v), 4) for v in curve[49::50]],
                                  "greedy_success": round(score["mean_return"], 4)}
        print(f"  curves23 {name} ({LETTERS_23[name]}): greedy {score['mean_return']:.3f}, last {curve[-1]:.3f}", flush=True)
    return {"episodes": list(range(50, n + 1, 50)), "runs": dict(sorted(runs.items())),
            "settings": {LETTERS_23[k]: k for k in SETTINGS_23}}


def calib(ns) -> dict:
    np, rl = ns["np"], ns["mylearn"].rl
    out = {"cliff28": []}
    for seed in range(8):
        row = {"seed": seed}
        for algorithm in ("q_learning", "sarsa"):
            env = ns["make_env"]("CliffWalking-v1", seed=0)
            Q, history = rl.train_tabular(env, 500, algorithm=algorithm, alpha=0.5, gamma=1.0, epsilon=0.1,
                                          max_steps=200, rng=np.random.default_rng(seed))
            falls = -(history["returns"] + history["lengths"]) / 99
            state, _ = env.reset(seed=0)
            path = [state]
            for _ in range(100):
                state, _, terminated, truncated, _ = env.step(int(np.argmax(Q[state])))
                path.append(int(state))
                if terminated or truncated:
                    break
            row[algorithm] = {"falls": int(round(falls.sum())), "path_length": len(path) - 1,
                              "top_row": bool(min(path) < 12)}
        out["cliff28"].append(row)
        print(f"  calib cliff28 {row}", flush=True)
    import build_ch26 as B
    exec(B.SOLUTION_22, ns)
    exec(B.GIVEN_36, ns)
    exec(B.SOLUTION_36, ns)
    exec(B.GIVEN_38, ns)
    exec(B.SOLUTION_38, ns)
    out["challenge38"] = []
    for seed in (2638, 1, 2):
        exec(B.SOLUTION_38.replace("default_rng(2638)", f"default_rng({seed})"), ns)
        start = time.perf_counter()
        Q = ns["train_38"](ns["make_cartpole_38"]())
        seconds = time.perf_counter() - start
        policy = rl.greedy_policy(Q)
        scores = [rl.evaluate_policy(ns["make_cartpole_38"](), policy, 100, max_steps=500,
                                     rng=np.random.default_rng(s))["mean_length"] for s in (38, 3838)]
        out["challenge38"].append({"seed": seed, "seconds": round(seconds, 1), "public": scores[0], "hidden": scores[1]})
        print(f"  calib challenge38 {out['challenge38'][-1]}", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["l16", "curves23", "calib"])
    args = parser.parse_args(argv)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    ns = namespace()
    for name, fn in (("l16", l16), ("curves23", curves23), ("calib", calib)):
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
