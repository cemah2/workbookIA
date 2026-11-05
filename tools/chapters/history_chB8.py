#!/usr/bin/env python
"""Calibrate the checks and precompute the multi-seed protocols of chapter B8 (used by Claude).

    python tools/chapters/history_chB8.py                    # everything of the first half (about 30 min, 2 cores)
    python tools/chapters/history_chB8.py --only a15 p16     # some entries (names below)

Writes two files, merging the entries computed into those already there (another entry is never touched):
- ``tools/chapters/history_chB8.json``: the calibrations read by chB8_parts.py when the notebooks are built (thresholds,
  chosen seeds, the statistics quoted by 05_solutions.md);
- ``src/wb/deep_rl_protocols.json``: what the notebooks read at run time (``wbr.precomputed(name)``): the tabular
  table of chapter 26 and the curves of several seeds that a fast-mode notebook cannot afford to train, each with its
  protocol (seeds, budget, settings), so that a 🔮 never shows its outcome in the source of a cell.

Entries of the first half (session 78, B8.12 to B8.20):
- "tab26": the tabular reference of chapter 26 (the recipe of the solution of Ex 26.38 with mylearn_ref.rl), its
  table Q and its greedy evaluation with max_steps=500 and with the default (200);
- "c12": 📦 B8.12, the mean return of the random policy (one generator for the actions, another for the seeds) on 100
  episodes for the seeds 0 to 15, the one-decimal value of the fiche (22.3) excluded, and the values of the classic
  mistakes for the chosen seed;
- "c13": 📦 B8.13, the reference DQN (wbr.train_dqn, DQN_SETTINGS) on 8 seeds: the greedy evaluation of the final
  network (100 episodes, EVAL_SEED), the first and last training returns; the threshold of the check;
- "a15": 🔬 B8.15, the four variants (dqn, no_target, no_replay, double) on three seeds, 20 000 steps, the greedy
  evaluation every 1 000 steps (20 episodes): the curves of the figure, and the prefix that the notebook replays;
- "p16": 🔮 B8.16, REINFORCE without baseline, with the mean baseline and with normalised returns, 300 episodes, on
  six seeds: the training returns and the greedy evaluation at the end; the answers of the predictions must be the
  same on every seed (or are counts over the seeds);
- "c17": 📦 B8.17, the spread of the per-episode gradients with and without baseline (wbr.episode_gradients) for
  untrained and half-trained policies on several seeds: the ratio must be clearly above 1 everywhere;
- "c20": 📦 B8.20, the reference PPO (wbr.train_ppo, PPO_SETTINGS, 50 000 steps) on six seeds: the deterministic
  evaluation (100 episodes, EVAL_SEED); the threshold of the check;
- "c28": 🏆 B8.28 (first measure, for the second half): the reference PPO with 100 000 steps on three seeds, the mean
  return on 100 episodes with max_steps=500;
- "b14": 🐛 B8.14, the 2**4 partly repaired versions of the colleague's DQN (chB8_code.repaired_14): for each, the first
  diagnosis that fails must be that of the first bug left (FIX_ORDER_14).
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

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE), str(ROOT / "solutions")]
OUT = HERE / "history_chB8.json"
RUNTIME = ROOT / "src" / "wb" / "deep_rl_protocols.json"


def r4(x):
    return [round(float(v), 4) for v in x] if isinstance(x, (list, tuple, np.ndarray)) else round(float(x), 4)


# ------------------------------------------------------------------------------------------------------------------- tab26
def tab26() -> tuple[dict, dict]:
    import gymnasium as gym
    from mylearn_ref import rl

    from wb import deep_rl_env as R

    env = R.Discretize(gym.make("CartPole-v1"), R.TABULAR_EDGES)
    n = 800
    schedule = lambda i: max(0.01, 1.0 - (1.0 - 0.01) * i / (0.5 * n))   # epsilon_schedule_22(n, 1.0, 0.01, 0.5)
    start = time.perf_counter()
    Q, _ = rl.train_tabular(env, n, alpha=0.2, gamma=0.99, epsilon=schedule, max_steps=200,
                            rng=np.random.default_rng(2638))
    seconds = time.perf_counter() - start
    policy = rl.greedy_policy(Q)
    full = rl.evaluate_policy(R.Discretize(gym.make("CartPole-v1"), R.TABULAR_EDGES), policy, 100, max_steps=500,
                              rng=np.random.default_rng(26))
    short = rl.evaluate_policy(R.Discretize(gym.make("CartPole-v1"), R.TABULAR_EDGES), policy, 100,
                               rng=np.random.default_rng(26))
    print("tab26", full, short, f"{seconds:.1f} s")
    calib = {"eval_500": full, "eval_default": short, "train_seconds": round(seconds, 1)}
    return calib, {"Q": Q.tolist(), "edges": R.TABULAR_EDGES, "protocol": "solution of Ex 26.38: 800 episodes, "
                   "alpha 0.2, gamma 0.99, epsilon 1 -> 0.01 over half of the training, max_steps 200, seed 2638"}


# --------------------------------------------------------------------------------------------------------------------- c12
def c12() -> dict:
    from mylearn_ref import rl

    from wb import deep_rl_env as R

    rows = {}
    for seed in range(16):
        act = np.random.default_rng(seed)
        res = rl.evaluate_policy(R.make_env(), lambda s, i: int(act.integers(2)), 100, max_steps=500,
                                 rng=np.random.default_rng(seed))
        rows[seed] = res["mean_return"]
    print("c12", rows)
    return {"means": rows}


# --------------------------------------------------------------------------------------------------------------------- c13
def c13() -> dict:
    from wb import deep_rl_env as R

    rows = []
    for seed in range(8):
        q, h = R.train_dqn(seed, eval_every=0)
        ev = R.evaluate(q, 100, R.EVAL_SEED)
        r = np.array(h["episode_returns"])
        rows.append({"seed": seed, "eval": ev["mean_return"], "first10": float(r[:10].mean()),
                     "last10": float(r[-10:].mean()), "episodes": int(r.size), "seconds": round(h["seconds"], 1)})
        print("c13", rows[-1], flush=True)
    worst = min(row["eval"] for row in rows)
    return {"rows": rows, "worst": worst, "threshold": float(5 * np.floor(0.6 * worst / 5)),
            "settings": R.DQN_SETTINGS, "steps": R.dqn_steps(True)}


# --------------------------------------------------------------------------------------------------------------------- a15
A15_SEEDS = (0, 1, 2)
A15_EVERY = 1_000


def a15() -> tuple[dict, dict]:
    from wb import deep_rl_env as R

    runs, table = {}, {}
    for variant in R.DQN_VARIANTS:
        for seed in A15_SEEDS:
            q, h = R.train_dqn(seed, R.dqn_steps(True), variant=variant, eval_every=A15_EVERY)
            final = R.evaluate(q, 100, R.EVAL_SEED)["mean_return"]
            runs[f"{variant}/{seed}"] = {"eval_steps": h["eval_steps"], "eval_returns": r4(h["eval_returns"]),
                                         "episode_returns": h["episode_returns"], "episode_ends": h["episode_ends"],
                                         "final_eval_100": final, "seconds": round(h["seconds"], 1)}
            print("a15", variant, seed, final, r4(h["eval_returns"])[:6], f"{h['seconds']:.0f} s", flush=True)
        curves = np.array([runs[f"{variant}/{s}"]["eval_returns"] for s in A15_SEEDS])
        table[variant] = {"final_median": float(np.median([runs[f"{variant}/{s}"]["final_eval_100"] for s in A15_SEEDS])),
                          "finals": [runs[f"{variant}/{s}"]["final_eval_100"] for s in A15_SEEDS],
                          "auc_median": float(np.median(curves.mean(axis=1))),
                          "aucs": r4(curves.mean(axis=1))}
    protocol = {"seeds": list(A15_SEEDS), "steps": R.dqn_steps(True), "eval_every": A15_EVERY, "eval_episodes": 20,
                "eval_seed": R.EVAL_SEED, "settings": R.DQN_SETTINGS, "variants": list(R.DQN_VARIANTS)}
    print("a15", json.dumps(table))
    return {"table": table, "protocol": protocol}, {"protocol": protocol, "runs": runs}


# --------------------------------------------------------------------------------------------------------------------- p16
P16_SEEDS = (0, 1, 2, 3, 4, 5)


def p16() -> tuple[dict, dict]:
    from wb import deep_rl_env as R

    runs, rows = {}, []
    for seed in P16_SEEDS:
        for baseline in R.BASELINES:
            net, h = R.train_reinforce(seed, 300, baseline)
            r = np.array(h["episode_returns"])
            final = R.evaluate(net, 100, R.EVAL_SEED)["mean_return"]
            runs[f"{baseline}/{seed}"] = {"episode_returns": h["episode_returns"], "final_eval_100": final,
                                          "seconds": round(h["seconds"], 1)}
            rows.append({"seed": seed, "baseline": baseline, "last50": float(r[-50:].mean()),
                         "first50": float(r[:50].mean()), "first_200": int(np.argmax(R.moving_average(r, 20) >= 200))
                         if (R.moving_average(r, 20) >= 200).any() else None, "final": final,
                         "seconds": round(h["seconds"], 1)})
            print("p16", rows[-1], flush=True)
    protocol = {"seeds": list(P16_SEEDS), "episodes": 300, "settings": R.REINFORCE_SETTINGS,
                "baselines": list(R.BASELINES), "eval_seed": R.EVAL_SEED}
    return {"rows": rows, "protocol": protocol}, {"protocol": protocol, "runs": runs}


# --------------------------------------------------------------------------------------------------------------------- c17
def c17() -> dict:
    import torch

    from wb import deep_rl_env as R

    rows = []
    for seed in range(5):
        for trained in (0, 60):
            if trained:
                net, _ = R.train_reinforce(seed, trained, "mean")
            else:
                net = R.policy_network(64, seed=seed)
            episodes = R.collect_episodes(net, 32, 1700 + seed)
            spreads = {b: R.gradient_spread(R.episode_gradients(net, episodes, b)) for b in ("none", "mean")}
            means = {b: R.episode_gradients(net, episodes, b).mean(axis=0) for b in ("none", "mean")}
            cos = float(means["none"] @ means["mean"] / (np.linalg.norm(means["none"]) * np.linalg.norm(means["mean"])))
            rows.append({"seed": seed, "trained": trained, "ratio": spreads["none"] / spreads["mean"],
                         "mean_length": float(np.mean([len(e["rewards"]) for e in episodes])), "cos": cos})
            print("c17", rows[-1], flush=True)
    del torch
    return {"rows": rows, "min_ratio": min(r["ratio"] for r in rows)}


# --------------------------------------------------------------------------------------------------------------- c20, c28
def c20() -> dict:
    from wb import deep_rl_env as R

    rows = []
    for seed in range(6):
        agent, logs = R.train_ppo(seed, R.ppo_steps(True))
        ev = R.evaluate(agent, 100, R.EVAL_SEED)
        rows.append({"seed": seed, "eval": ev["mean_return"], "seconds": round(logs["seconds"], 1),
                     "first5": float(np.nanmean(logs["mean_return"][:5])),
                     "last5": float(np.nanmean(logs["mean_return"][-5:]))})
        print("c20", rows[-1], flush=True)
    worst = min(row["eval"] for row in rows)
    return {"rows": rows, "worst": worst, "settings": R.PPO_SETTINGS, "steps": R.ppo_steps(True)}


def c28() -> dict:
    from wb import deep_rl_env as R

    rows = []
    for seed in (0, 1, 2):
        agent, logs = R.train_ppo(seed, 100_000)
        ev = R.evaluate(agent, 100, 2828)
        rows.append({"seed": seed, "mean_return_100": ev["mean_return"], "min_return": min(ev["returns"]),
                     "seconds": round(logs["seconds"], 1)})
        print("c28", rows[-1], flush=True)
    return {"rows": rows, "settings": R.PPO_SETTINGS, "steps": 100_000, "eval_seed": 2828,
            "all_solved": all(r["mean_return_100"] >= R.SOLVED for r in rows)}


# --------------------------------------------------------------------------------------------------------------------- b14
def namespace(blocks=()) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import wb

    import chB8_code as C

    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True, verbose=False), "FAST_MODE": True, "__name__": "history_chB8"}
    exec(C.TOOLS_A.replace("«KIND»", "ref"), ns)
    for block in blocks:
        exec(getattr(C, block), ns)
    return ns


def b14() -> dict:
    import chB8_code as C

    ns = namespace(["GIVEN_14"])
    order = C.FIX_ORDER_14
    table, mismatches = [], []
    for size in range(len(order) + 1):
        for fixed in itertools.combinations(order, size):
            exec(C.repaired_14(fixed), ns)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                ns["diagnose_14"]()
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
    print("b14", len(table), "versions,", len(mismatches), "mismatches")
    for row in mismatches:
        print("   ", row)
    if mismatches:
        raise SystemExit("b14: some partly repaired versions do not fail at the diagnosis of their first bug")
    return {"versions": len(table), "messages": {",".join(r["fixed"]) or "none": r["message"] for r in table}}


ENTRIES = {"tab26": tab26, "c12": c12, "c13": c13, "a15": a15, "p16": p16, "c17": c17, "c20": c20, "c28": c28,
           "b14": b14}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="*", choices=sorted(ENTRIES), help="entries to compute")
    args = parser.parse_args()
    names = args.only or list(ENTRIES)
    calib = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8")) if RUNTIME.exists() else {}
    for name in names:
        start = time.perf_counter()
        result = ENTRIES[name]()
        if isinstance(result, tuple):
            calib[name], runtime[name] = result
            RUNTIME.write_text(json.dumps(runtime, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        else:
            calib[name] = result
        calib[name] = {**calib[name], "computed_in_s": round(time.perf_counter() - start, 1)} \
            if isinstance(calib[name], dict) else calib[name]
        OUT.write_text(json.dumps(calib, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"✅ {name} ({time.perf_counter() - start:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
