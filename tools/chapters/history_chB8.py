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
- "c14": 🐛 B8.14, the repaired colleague's DQN (8 000 steps) on five seeds: the threshold of the check; and the
  versions with all the bugs, then with one bug left (seed 0), for 05_solutions.md;
- "c17": 📦 B8.17, the spread of the per-episode gradients with and without baseline (wbr.episode_gradients) for
  untrained policies and policies trained 60 episodes, five seeds each; the weights of the trained policy of the
  notebook (seed 1) and the two ratios of the notebook's episodes;
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
C17_SEED = 1


def c17() -> tuple[dict, dict]:
    import torch

    from wb import deep_rl_env as R

    rows, notebook, kept = [], {}, None
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
            if seed == C17_SEED and trained:
                kept = {k: v.detach().double().numpy().tolist() for k, v in net.state_dict().items()}
    for name, net in (("untrained", R.policy_network(64, seed=17)), ("trained", None)):
        if net is None:
            net = R.policy_network(64)
            net.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in kept.items()})
        episodes = R.collect_episodes(net, 32, 1700)
        spreads = {b: R.gradient_spread(R.episode_gradients(net, episodes, b)) for b in ("none", "mean")}
        notebook[name] = {"ratio": spreads["none"] / spreads["mean"], "spreads": spreads,
                          "mean_length": float(np.mean([len(e["rewards"]) for e in episodes]))}
        print("c17 notebook", name, notebook[name], flush=True)
    calib = {"rows": rows, "min_ratio_trained": min(r["ratio"] for r in rows if r["trained"]),
             "kept_seed": C17_SEED, "notebook": notebook}
    return calib, {"state_dict": kept, "protocol": f"wbr.train_reinforce(seed={C17_SEED}, n_episodes=60, "
                                                     "baseline='mean'), its policy network"}


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

    ns = {"wb": wb, "cfg": wb.setup(seed=42, fast=True, verbose=False), "FAST_MODE": True, "__name__": "history_chB8",
          "mylearn": wb.load_mylearn("ref")}
    exec(C.TOOLS_A.replace("«KIND»", "solution"), ns)
    for block in blocks:
        exec(fill(getattr(C, block) if block.isupper() else block), ns)
    return ns


def fill(text: str) -> str:
    """A cell with its «NAME» placeholders replaced by neutral values (the calibrations do not depend on them)."""
    import re
    return re.sub(r"«[A-Z0-9_]+»", "0", text)


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


def c14() -> dict:
    import chB8_code as C

    ns = namespace(["GIVEN_14"])
    exec(C.SOLUTION_14, ns)
    rows = []
    for seed in range(5):
        start = time.perf_counter()
        online, log = ns["run_colleague_14"](seed, 8_000)
        ev = ns["wbr"].evaluate(ns["wbr"].greedy_q(online), 100, ns["EVAL_SEED"])
        rows.append({"seed": seed, "eval": ev["mean_return"], "episodes": len(log["returns"]),
                     "last20": float(np.mean(log["returns"][-20:])), "q_end": log["q_mean"][-1],
                     "seconds": round(time.perf_counter() - start, 1)})
        print("c14", rows[-1], flush=True)
    buggy = {}
    exec(C.TODO_14, ns)
    online, log = ns["run_colleague_14"](0, 8_000)
    buggy["all_bugs"] = {"eval": ns["wbr"].evaluate(ns["wbr"].greedy_q(online), 100, ns["EVAL_SEED"])["mean_return"],
                         "q_mean": log["q_mean"], "last20": float(np.mean(log["returns"][-20:]))}
    for name in C.FIX_ORDER_14:                      # one bug left
        exec(C.repaired_14([f for f in C.FIX_ORDER_14 if f != name]), ns)
        online, log = ns["run_colleague_14"](0, 8_000)
        buggy[name] = {"eval": ns["wbr"].evaluate(ns["wbr"].greedy_q(online), 100, ns["EVAL_SEED"])["mean_return"],
                       "q_mean": log["q_mean"][-1], "last20": float(np.mean(log["returns"][-20:]))}
        print("c14 one bug left", name, buggy[name], flush=True)
    worst = min(r["eval"] for r in rows)
    return {"rows": rows, "worst": worst, "threshold": float(5 * np.floor(0.6 * worst / 5)), "buggy": buggy}



# ======================================================================================== second half (B8.21 to B8.28)
# --------------------------------------------------------------------------------------------------------------------- l21
L21_SEEDS = (21, 22, 23)            # the notebook shows the first; the other two check that each signature holds
L21_RUNS = {"healthy": {}, "lr_high": {"lr": 1e-2, "anneal_lr": False}, "no_clip": {"clip_eps": 1e6},
            "entropy_sign": {"ent_coef": -0.2}, "no_critic": {"vf_coef": 0.0}}
L21_LETTERS = "ABCDE"


def _log_signature(logs: dict) -> dict:
    """The statistics that tell the runs of B8.21 apart (computed on the logs of the reference PPO)."""
    n = len(logs["step"])
    first = slice(0, max(1, n // 2))
    ev = np.array(logs["explained_variance"], dtype=float)
    return {"kl_median": float(np.median(logs["approx_kl"][first])),
            "clipfrac_median": float(np.median(logs["clipfrac"][first])),
            "clipfrac_max": float(np.max(logs["clipfrac"])),
            "entropy_quarter": float(logs["entropy"][max(0, n // 4 - 1)]),
            "entropy_min": float(np.min(logs["entropy"])),
            "ev_median": float(np.nanmedian(ev[first])),
            "return_last10": float(np.nanmean(np.array(logs["mean_return"][-10:], dtype=float)))}


def l21_checks(rows: list) -> list:
    """Each signature of B8.21 on each seed: the largest clip fraction for the large learning rate, a clip fraction
    always 0 without clipping (and a KL above the healthy run), the lowest entropy reached for the wrong-sign
    bonus, the lowest explained variance for the critic that does not learn."""
    checks = []
    for seed in sorted({r["seed"] for r in rows}):
        by = {r["run"]: r for r in rows if r["seed"] == seed}
        others = lambda key, run: [by[n][key] for n in by if n != run]       # noqa: E731
        checks.append({"seed": seed,
                       "lr_high_largest_clipfrac": by["lr_high"]["clipfrac_median"]
                       > max(others("clipfrac_median", "lr_high")),
                       "no_clip_never_clips": by["no_clip"]["clipfrac_max"] == 0.0,
                       "no_clip_kl_above_healthy": by["no_clip"]["kl_median"] > by["healthy"]["kl_median"],
                       "entropy_sign_lowest_entropy": by["entropy_sign"]["entropy_min"]
                       < min(others("entropy_min", "entropy_sign")),
                       "no_critic_lowest_ev": by["no_critic"]["ev_median"] < min(others("ev_median", "no_critic"))})
    return checks


def l21() -> tuple[dict, dict]:
    from wb import deep_rl_env as R

    rows, kept = [], {}
    order = list(np.random.default_rng(24).permutation(len(L21_RUNS)))       # run i -> letter order[i]
    letters = {name: L21_LETTERS[order[i]] for i, name in enumerate(L21_RUNS)}
    for seed in L21_SEEDS:
        for name, change in L21_RUNS.items():
            agent, logs = R.train_ppo(seed, R.ppo_steps(True), {**R.PPO_SETTINGS, **change})
            ev = R.evaluate(agent, 100, R.EVAL_SEED)["mean_return"]
            sig = {"seed": seed, "run": name, "eval": ev, "seconds": round(logs["seconds"], 1), **_log_signature(logs)}
            rows.append(sig)
            print("l21", json.dumps(sig), flush=True)
            if seed == L21_SEEDS[0]:
                kept[letters[name]] = {k: r4(logs[k]) if k not in ("step", "iteration", "episodes") else logs[k]
                                       for k in R.LOG_KEYS}
    checks = l21_checks(rows)
    print("l21 checks", json.dumps(checks))
    protocol = {"seed": L21_SEEDS[0], "steps": R.ppo_steps(True), "settings": R.PPO_SETTINGS,
                "changes": "five runs of the reference PPO; one has the reference settings, each other one changes "
                           "one of them (see the statement)"}
    calib = {"rows": rows, "letters": letters, "checks": checks,
             "all_hold": all(all(v for k, v in c.items() if k != "seed") for c in checks)}
    return calib, {"protocol": protocol, "runs": kept}


# --------------------------------------------------------------------------------------------------------------------- p22
P22_SEEDS = (220, 221, 222)
P22_CONFIGS = {"reference": {}, "clip_0.1": {"clip_eps": 0.1}, "clip_0.4": {"clip_eps": 0.4},
               "lambda_0.8": {"gae_lambda": 0.8}, "lambda_1": {"gae_lambda": 1.0},
               "epochs_4": {"n_epochs": 4}, "epochs_20": {"n_epochs": 20}}


def p22() -> tuple[dict, dict]:
    from wb import deep_rl_env as R

    runs, rows = {}, []
    for name, change in P22_CONFIGS.items():
        for seed in P22_SEEDS:
            agent, logs = R.train_ppo(seed, R.ppo_steps(True), {**R.PPO_SETTINGS, **change})
            ev = R.evaluate(agent, 100, R.EVAL_SEED)["mean_return"]
            runs[f"{name}/{seed}"] = {"step": logs["step"], "mean_return": r4(logs["mean_return"]),
                                      "approx_kl": r4(logs["approx_kl"]), "clipfrac": r4(logs["clipfrac"]),
                                      "eval": ev, "seconds": round(logs["seconds"], 1)}
            rows.append({"config": name, "seed": seed, "eval": ev, **R.log_summary(logs),
                         "seconds": round(logs["seconds"], 1)})
            print("p22", json.dumps(rows[-1]), flush=True)
    protocol = {"seeds": list(P22_SEEDS), "steps": R.ppo_steps(True), "settings": R.PPO_SETTINGS,
                "configs": P22_CONFIGS, "eval_seed": R.EVAL_SEED, "eval_episodes": 100}
    return {"rows": rows, "protocol": protocol}, {"protocol": protocol, "runs": runs}


# --------------------------------------------------------------------------------------------------------------- c23, g24
C23_SEEDS = (0, 1, 2, 3, 4, 5)
C23_PHYSICS = {"normal": {}, "gravity_x2": {"gravity": 19.6}, "length_x2": {"length": 1.0},
               "length_x4": {"length": 2.0}}
C23_EVAL_SEEDS = (8000, 2323)
G24_GRAVITY = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
G24_LENGTH = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
G24_EPISODES = 50


def c23() -> tuple[dict, dict]:
    """🔮 B8.23 and 🔬 B8.24: six reference PPO agents (50 000 steps, the protocol of B8.20), evaluated on the normal
    physics, a doubled gravity and a pole twice as long (two sets of evaluation seeds), then on the grids of B8.24."""
    from wb import deep_rl_env as R

    rows, grid = [], {}
    for seed in C23_SEEDS:
        agent, logs = R.train_ppo(seed, R.ppo_steps(True))
        row = {"seed": seed}
        for name, phys in C23_PHYSICS.items():
            for es in C23_EVAL_SEEDS:
                ev = R.evaluate(agent, 100, es, **phys)
                row[f"{name}/{es}"] = {"mean": ev["mean_return"], "min": min(ev["returns"]),
                                       "max": max(ev["returns"])}
        for key, factors, base in (("gravity", G24_GRAVITY, 9.8), ("length", G24_LENGTH, 0.5)):
            grid[f"{key}/{seed}"] = [R.evaluate(agent, G24_EPISODES, R.EVAL_SEED, **{key: base * f})["mean_return"]
                                     for f in factors]
        rows.append(row)
        print("c23", json.dumps(row), json.dumps({k: v for k, v in grid.items() if k.endswith(f"/{seed}")}),
              flush=True)
    protocol = {"seeds": list(C23_SEEDS), "steps": R.ppo_steps(True), "settings": R.PPO_SETTINGS,
                "eval_seeds": list(C23_EVAL_SEEDS), "physics": C23_PHYSICS, "gravity_factors": list(G24_GRAVITY),
                "length_factors": list(G24_LENGTH), "grid_episodes": G24_EPISODES, "grid_eval_seed": R.EVAL_SEED}
    return {"rows": rows, "grid": grid, "protocol": protocol}, {"protocol": protocol, "rows": rows, "grid": grid}


# --------------------------------------------------------------------------------------------------------------------- c28b
C28_SEEDS = tuple(range(280, 290))
C28_SETTINGS = {"reference": ({}, C28_SEEDS),
                "cleanrl": ({"lr": 2.5e-4, "n_epochs": 4}, C28_SEEDS[:5]),
                "epochs_4": ({"n_epochs": 4}, C28_SEEDS[:5])}


def c28b() -> dict:
    """🏆 B8.28: the reference PPO through the harness of the challenge (wbr.run_challenge: the steps counted at the
    class level, the held-out evaluation), with its settings on ten seeds, and two other reasonable settings (the
    defaults of CleanRL, lr 2.5e-4 and 4 epochs; 4 epochs alone) on five."""
    from wb import deep_rl_env as R

    train = R.train_ppo                     # captured before the run: the harness guards the name, not this object
    rows = []
    for name, (change, seeds) in C28_SETTINGS.items():
        settings = {**R.PPO_SETTINGS, **change}
        for seed in seeds:
            res = R.run_challenge(lambda s, budget: train(s, budget, settings)[0], seed)
            rows.append({"settings": name, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}})
            print("c28b", json.dumps(rows[-1]), flush=True)
    summary = {name: {"solved": sum(r["solved"] for r in rows if r["settings"] == name),
                      "runs": sum(1 for r in rows if r["settings"] == name),
                      "worst": min(r["mean_return"] for r in rows if r["settings"] == name)} for name in C28_SETTINGS}
    print("c28b", json.dumps(summary))
    return {"rows": rows, "summary": summary, "budget": R.CHALLENGE_BUDGET, "eval_seed": R.CHALLENGE_EVAL_SEED,
            "settings": {k: v[0] for k, v in C28_SETTINGS.items()}}


# --------------------------------------------------------------------------------------------------------------------- r26
R26_RUNS = 5                         # data seed 26 + k, held-out pairs 126 + k, reward model and PPO seed 26 + k
R26_PAIRS, R26_HELD = 200, 100
R26_STEPS = 30_000
R26_DATA = {"broad": None, "narrow": (1.0,)}


def r26() -> dict:
    """📦 B8.26: the miniature RLHF on five seeds (k = 0 is the notebook's): for the broad preferences (the noisy rule
    and the random policy) and the narrow ones (the random policy only), the reference reward model, its preference
    accuracy (its own held-out pairs, broad held-out pairs), then the reference PPO (30 000 steps) on it; and two
    PPO agents for comparison, on the true reward and on the +1 of CartPole (survival). Each agent is measured by
    wbr.rlhf_measure (20 deterministic episodes)."""
    import torch

    from wb import deep_rl_env as R

    rows = []
    for k in range(R26_RUNS):
        seed = 26 + k
        broad_held = R.preference_pairs(R26_HELD, 126 + k)
        models, row = {}, {"k": k, "seed": seed}
        for name, behaviours in R26_DATA.items():
            extra = {} if behaviours is None else {"behaviours": behaviours}
            start = time.perf_counter()
            pairs = R.preference_pairs(R26_PAIRS, seed, **extra)
            held = R.preference_pairs(R26_HELD, 126 + k, **extra)
            model, losses = R.train_reward_model(pairs, seed)
            models[name] = model
            row[f"rm_{name}"] = {"acc_train": R.preference_accuracy(model, pairs), "acc_held": R.preference_accuracy(model, held),
                                 "acc_broad": R.preference_accuracy(model, broad_held), "loss_first": losses[0],
                                 "loss_last": losses[-1], "seconds": round(time.perf_counter() - start, 1),
                                 "mean_length": float(np.mean([len(t) for t in pairs["first"] + pairs["second"]])),
                                 "max_abs_x": float(max(np.abs(t[:, 0]).max() for t in pairs["first"] + pairs["second"]))}

        def learned(model):
            def reward(obs, actions, next_obs):
                with torch.no_grad():
                    return model(torch.as_tensor(next_obs, dtype=torch.float32)).numpy()
            return reward

        fns = {"rlhf_broad": learned(models["broad"]), "rlhf_narrow": learned(models["narrow"]),
               "true": lambda obs, actions, next_obs: R.centered_reward(next_obs), "survival": None}
        for name, fn in fns.items():
            agent, logs = R.train_ppo(seed, R26_STEPS, reward_fn=fn)
            row[name] = {"broad": R.rlhf_measure(agent, models["broad"], 20, R.EVAL_SEED),
                         "narrow_learned_per_step": R.rlhf_measure(agent, models["narrow"], 20, R.EVAL_SEED)["learned_per_step"],
                         "seconds": round(logs["seconds"], 1)}
        rows.append(row)
        print("r26", json.dumps(row), flush=True)
    return {"rows": rows, "pairs": R26_PAIRS, "held": R26_HELD, "steps": R26_STEPS,
            "settings": {"epochs": 200, "lr": 3e-3, "hidden": 32, "width": R.RLHF_WIDTH, "max_steps": R.RLHF_STEPS}}

ENTRIES = {"tab26": tab26, "c14": c14, "c12": c12, "c13": c13, "a15": a15, "p16": p16, "c17": c17, "c20": c20, "c28": c28,
           "b14": b14, "l21": l21, "p22": p22, "c23": c23, "c28b": c28b, "r26": r26}


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
