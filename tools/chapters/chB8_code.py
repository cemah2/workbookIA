"""Code cells of the chapter B8 notebook, first half (part A, the value: B8.12 to B8.15; start of part B, the policy:
B8.16 to B8.20), shared by build_chB8.py (through chB8_parts.py) and history_chB8.py (used by Claude).

Each constant is the source of one cell: TOOLS_* and GIVEN_* (both notebooks), TODO_* (exercise notebook), SOLUTION_*
(solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions). The cells go
through the reference of wb.deep_rl_env (``wbr``: environments, evaluation, reference agents) for every expected value,
never through another exercise's code. «KIND» is replaced by "learner" (exercise notebook) or "solution"; other
placeholders «NAME» are filled by chB8_parts.py from tools/chapters/history_chB8.json.
The second half (B8.21 to B8.28) reuses TOOLS_A, TOOLS_B, SETTINGS_20, train_ppo_20 and the agent saved by B8.20
(AGENT_KEY_20, ``load_agent_20``).
"""

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="B8")   # reload your saved file\n'

# =====================================================================================================================
# Part A · the value
# =====================================================================================================================
TOOLS_A = r'''# Tools of part A: CartPole, the evaluation and the reference agents of the chapter (wb.deep_rl_env, `wbr`)
import copy
import inspect
import math
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from wb import deep_rl_env as wbr
from wb.nbtools import (StopCheck, checked, filled, fr, own_code, pair, refuses, returned, run_pytest_group, same,
                        verdict, written)

torch.set_num_threads(1)           # tiny networks: one thread is faster, and a seed gives the same numbers every time
SOURCE = "«KIND»"                  # who trained the agents this notebook saves ("solution" in the solutions notebook)
TEST_DEEP_RL = "tests/test_chb8_deep_rl.py"   # the tests of your library deep_rl.py
EVAL_SEED = wbr.EVAL_SEED          # the seeds of every evaluation of the notebook (never used to train)
EPISODE_STEPS = wbr.EPISODE_STEPS  # 500: the time limit of CartPole-v1
RANDOM_LEVEL = 22.3                # the mean return of a random policy (fiche, 208.1)
REFERENCE_CODE = wbr.reference_answers()   # what your code must not call: the checks would measure the reference

_RL_PROBES = {
    "evaluate_policy": (lambda: mylearn.rl.evaluate_policy(wbr.make_env(), lambda s, i: 0, 1, max_steps=3,
                                                           rng=np.random.default_rng(0)), "26.21"),
    "ReplayBuffer": (lambda: (lambda b: (b.push(np.zeros(4, np.float32), 0, 1.0, np.zeros(4, np.float32), False),
                                         len(b), b.sample(1)))(mylearn.rl.ReplayBuffer(2, obs_shape=(4,),
                                                                                       obs_dtype="float32")), "26.31"),
    "returns_to_go": (lambda: mylearn.rl.returns_to_go([1.0, 1.0], 0.5), "26.14"),
}


def rl_ready(ex_id, *names):
    """True when the functions `names` of your mylearn/rl.py (chapter 26) are written; otherwise one ⏳ line that
    names them and their exercise of chapter 26."""
    missing = [name for name in names if not written(_RL_PROBES[name][0])]
    if missing:
        where = ", ".join(f"{name} (Ex {_RL_PROBES[name][1]})" for name in missing)
        print(f"⏳ Ex {ex_id} : cet exercice se sert de ta librairie rl.py du ch. 26 : {where}, pas encore écrit.")
    return not missing


def plot_returns(ax, returns, label, ends=None, window=20, **kwargs):
    """The returns of the training episodes as a moving average of `window` episodes (against the episode number, or
    against the environment step of their end when `ends` is given)."""
    returns = np.asarray(returns, dtype=float)
    x = np.arange(1, len(returns) + 1) if ends is None else np.asarray(ends)
    ax.plot(x, wbr.moving_average(returns, window), label=label, **kwargs)


print(f"CartPole-v1 (gymnasium {gym.__version__}) : {EPISODE_STEPS} pas au plus, résolu à {wbr.SOLVED:g} de moyenne sur "
      f"100 épisodes · évaluations sur les graines tirées de {EVAL_SEED} · PyTorch {torch.__version__} sur un thread")'''

# --- 📦 B8.12 ---------------------------------------------------------------------------------------------------------
GIVEN_12 = r'''SEED_12 = «SEED_12»                                    # the seed of the random policy and of the episodes
MAKE_TABULAR_12, Q_TABULAR_12 = wbr.tabular_reference()   # the table of Ex 26.38 (12 states), read back, not trained
TABULAR_POLICY_12 = Q_TABULAR_12.argmax(axis=1)            # its greedy policy: one action per state (a table)'''

TODO_12 = r'''def describe_12(env):
    """What the agent sees and may do in env, READ from env (never typed): a dict {"obs_shape": the shape of one
    observation (a tuple of ints), "n_actions": the number of actions (an int), "max_steps": the time limit of an
    episode (an int, from env.spec), "threshold": the mean return that counts as solved (a float, from env.spec)}."""
    raise NotImplementedError("describe_12")


def random_policy_12(seed):
    """A policy (state, info) -> int that plays 0 or 1 at random: each call draws int(rng.integers(2)) from ONE
    generator rng = np.random.default_rng(seed), created once by random_policy_12 (not at each call)."""
    raise NotImplementedError("random_policy_12")


def evaluate_12(env, policy, seed, n_episodes=100):
    """The dict of mylearn.rl.evaluate_policy for n_episodes WHOLE episodes of env (max_steps = the time limit of env,
    read in env.spec, not the default of evaluate_policy), the seeds of the episodes drawn from
    rng=np.random.default_rng(seed)."""
    raise NotImplementedError("evaluate_12")'''

SOLUTION_12 = r'''def describe_12(env):
    """What the agent sees and may do in env, READ from env (never typed): a dict {"obs_shape": the shape of one
    observation (a tuple of ints), "n_actions": the number of actions (an int), "max_steps": the time limit of an
    episode (an int, from env.spec), "threshold": the mean return that counts as solved (a float, from env.spec)}."""
    return {"obs_shape": tuple(env.observation_space.shape), "n_actions": int(env.action_space.n),
            "max_steps": int(env.spec.max_episode_steps), "threshold": float(env.spec.reward_threshold)}


def random_policy_12(seed):
    """A policy (state, info) -> int that plays 0 or 1 at random: each call draws int(rng.integers(2)) from ONE
    generator rng = np.random.default_rng(seed), created once by random_policy_12 (not at each call)."""
    rng = np.random.default_rng(seed)        # once: each call goes on with the same stream

    def policy(state, info):
        return int(rng.integers(2))
    return policy


def evaluate_12(env, policy, seed, n_episodes=100):
    """The dict of mylearn.rl.evaluate_policy for n_episodes WHOLE episodes of env (max_steps = the time limit of env,
    read in env.spec, not the default of evaluate_policy), the seeds of the episodes drawn from
    rng=np.random.default_rng(seed)."""
    return mylearn.rl.evaluate_policy(env, policy, n_episodes, max_steps=env.spec.max_episode_steps,
                                      rng=np.random.default_rng(seed))'''

CHECK_12 = r'''with checked("B8.12"):
    info_12 = describe_12(gym.make("CartPole-v1"))
    if returned("B8.12", "describe_12", info_12):
        want_12 = {"obs_shape": (4,), "n_actions": 2, "max_steps": 500, "threshold": 475.0}
        if not isinstance(info_12, dict) or set(info_12) != set(want_12):
            print(f"❌ Ex B8.12 : describe_12 doit rendre un dict aux clés {sorted(want_12)}, reçu "
                  f"{sorted(info_12) if isinstance(info_12, dict) else type(info_12).__name__}.")
        else:
            got_12 = {k: (tuple(v) if k == "obs_shape" and not isinstance(v, str) else v) for k, v in info_12.items()}
            verdict("B8.12", got_12 == want_12 and isinstance(got_12["n_actions"], int),
                    "describe_12 lit les espaces et env.spec de CartPole-v1.",
                    f"describe_12 sur CartPole-v1 : reçu {got_12} ; relis 208.1 (« CartPole, vu de près »).")
            import dataclasses                      # a CartPole whose spec says 200 steps and a threshold of 195
            short_spec_12 = dataclasses.replace(gym.spec("CartPole-v1"), max_episode_steps=200, reward_threshold=195.0)
            short_12 = describe_12(gym.make(short_spec_12))
            limits_12 = {k: short_12.get(k) for k in ("max_steps", "threshold")}
            verdict("B8.12", limits_12 == {"max_steps": 200, "threshold": 195.0},
                    "la limite de temps et le seuil viennent bien de env.spec (200 et 195 sur un CartPole dont la spec "
                    "le dit).",
                    f"sur un CartPole dont env.spec dit 200 pas et un seuil de 195, describe_12 rend "
                    f"max_steps = {limits_12['max_steps']!r} et threshold = {limits_12['threshold']!r} : les lit-elle "
                    "vraiment dans env.spec ?")

with checked("B8.12"):
    actions_12 = [random_policy_12(SEED_12)(None, {}) for _ in range(3)]
    policy_12 = random_policy_12(SEED_12)
    stream_12 = [policy_12(None, {}) for _ in range(40)]
    want_12 = [int(a) for a in np.random.default_rng(SEED_12).integers(2, size=40)]
    if not all(type(a) is int for a in stream_12):
        print(f"❌ Ex B8.12 : la politique doit rendre un int Python (gymnasium refuse un tenseur), reçu "
              f"{type(stream_12[0]).__name__}.")
    elif stream_12 != want_12:
        constant_12 = len(set(stream_12)) == 1
        print("❌ Ex B8.12 : " + ("40 appels de la politique rendent toujours la même action : le générateur "
                                  "repart-il de sa graine à chaque appel ?" if constant_12 else
                                  "les 40 premières actions ne sont pas les tirages int(rng.integers(2)) de "
                                  "np.random.default_rng(seed) : relis le contrat."))
    else:
        verdict("B8.12", True, "random_policy_12 tire ses actions d'un seul générateur à graine.", "")

with checked("B8.12"):
    if rl_ready("B8.12", "evaluate_policy"):
        try:
            rule_12 = evaluate_12(gym.make("CartPole-v1"), wbr.rule_policy, 1, n_episodes=5)
        except AttributeError as error:          # an int given as rng to an evaluate_policy that wants a generator
            if "object has no attribute 'integers'" not in str(error):
                raise
            print("❌ Ex B8.12 : evaluate_12 lève AttributeError (" + str(error)[:80] + ") : evaluate_policy attend un "
                  "générateur, rng=np.random.default_rng(seed), pas un entier.")
            raise StopCheck
        if returned("B8.12", "evaluate_12", rule_12):
            verdict("B8.12", rule_12["mean_length"] == EPISODE_STEPS,
                    "la règle d'une ligne de 208.1 tient 500 pas : tes épisodes vont jusqu'à la limite de CartPole-v1.",
                    f"la règle d'une ligne de 208.1 tient {rule_12['mean_length']:g} pas en moyenne au lieu de 500 : "
                    "tes épisodes s'arrêtent-ils avant la limite de temps de l'environnement ? (piège de 208.6)")
            random_12 = evaluate_12(gym.make("CartPole-v1"), random_policy_12(SEED_12), SEED_12)
            tab_12 = evaluate_12(MAKE_TABULAR_12(), TABULAR_POLICY_12, 26)
            tab_short_12 = mylearn.rl.evaluate_policy(MAKE_TABULAR_12(), TABULAR_POLICY_12, 100,
                                                      rng=np.random.default_rng(26))
            print(f"politique aléatoire (graine {SEED_12}) : {fr(random_12['mean_return'], 2)} de retour moyen sur 100 "
                  f"épisodes (écart-type {fr(random_12['std_return'], 1)})")
            print(f"table du ch. 26 (12 états, 26.38) : {fr(tab_12['mean_return'], 1)} sur 100 épisodes entiers ; "
                  f"{fr(tab_short_12['mean_return'], 1)} avec le max_steps par défaut d'evaluate_policy")
            wb.check("B8.12", random_12["mean_return"], computed=True)'''

RECORD_12 = r'''shared_12 = np.random.default_rng(SEED_12)             # one generator for the actions AND the seeds of the episodes
again_12 = lambda s, i: int(np.random.default_rng(SEED_12).integers(2))   # a generator created at each call
coin_12 = (lambda g: (lambda s, i: int(g.random() < 0.5)))(np.random.default_rng(SEED_12))
wb.record("B8.12", random_12["mean_return"], decimals=4, mistakes={
    "les actions et les graines des épisodes viennent-elles du même générateur ? Relis les contrats de "
    "random_policy_12 et d'evaluate_12": mylearn.rl.evaluate_policy(
        gym.make("CartPole-v1"), lambda s, i: int(shared_12.integers(2)), 100, max_steps=500, rng=shared_12)["mean_return"],
    "ta politique joue-t-elle toujours la même action ? Où le générateur est-il créé ?": evaluate_12(
        gym.make("CartPole-v1"), again_12, SEED_12)["mean_return"],
    "relis le contrat : chaque appel tire int(rng.integers(2))": evaluate_12(
        gym.make("CartPole-v1"), coin_12, SEED_12)["mean_return"]})'''

# --- 📦 B8.13 ---------------------------------------------------------------------------------------------------------
GIVEN_13 = r'''SETTINGS_13 = dict(wbr.DQN_SETTINGS)   # the settings of the reference DQN of the chapter (printed below)
STEPS_13 = wbr.dqn_steps(FAST_MODE)       # 20 000 environment steps in fast mode, 50 000 in full mode
SEED_13 = 13
THRESHOLD_13 = «THRESHOLD_13»             # the greedy evaluation that the check asks (calibrated on 8 seeds of the reference)
print(SETTINGS_13)'''

TODO_13 = r'''def make_q_net_13(hidden=64):
    """A new Q network for CartPole: 4 inputs (the observation), two hidden layers of `hidden` neurons with ReLU, 2
    outputs (one value per action), float32."""
    raise NotImplementedError("make_q_net_13")


def dqn_targets_13(q_target, next_states, rewards, terminated, gamma):
    """The DQN targets of a mini-batch, a tensor (B,) that carries NO gradient:
    r + gamma (1 - terminated) max_a' Q_target(s', a'). next_states: float32 (B, 4); rewards: float (B,); terminated:
    bool (B,)."""
    raise NotImplementedError("dqn_targets_13")


def epsilon_13(step, total_steps, settings=SETTINGS_13):
    """ε of the step `step` (0, 1, ...): linear from settings["eps_start"] to settings["eps_end"] over the first
    settings["eps_fraction"] * total_steps steps, then settings["eps_end"]."""
    raise NotImplementedError("epsilon_13")


def train_dqn_13(env, seed, total_steps, settings=SETTINGS_13):
    """Train a DQN for total_steps steps of env (the contract of the statement); return (q_net, buffer, history), with
    history = {"returns": the return of each finished episode, "ends": the number of steps done (1, 2, ...) when each
    of them ended}."""
    raise NotImplementedError("train_dqn_13")'''

SOLUTION_13 = r'''def make_q_net_13(hidden=64):
    """A new Q network for CartPole: 4 inputs (the observation), two hidden layers of `hidden` neurons with ReLU, 2
    outputs (one value per action), float32."""
    return nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def dqn_targets_13(q_target, next_states, rewards, terminated, gamma):
    """The DQN targets of a mini-batch, a tensor (B,) that carries NO gradient:
    r + gamma (1 - terminated) max_a' Q_target(s', a'). next_states: float32 (B, 4); rewards: float (B,); terminated:
    bool (B,)."""
    with torch.no_grad():                                        # the target is a constant for the gradient
        following = q_target(next_states).max(dim=1).values      # one value per row: the best action of s'
        return rewards + gamma * (1.0 - terminated.float()) * following


def epsilon_13(step, total_steps, settings=SETTINGS_13):
    """ε of the step `step` (0, 1, ...): linear from settings["eps_start"] to settings["eps_end"] over the first
    settings["eps_fraction"] * total_steps steps, then settings["eps_end"]."""
    start, end = settings["eps_start"], settings["eps_end"]
    progress = min(1.0, step / (settings["eps_fraction"] * total_steps))
    return start + progress * (end - start)


def train_dqn_13(env, seed, total_steps, settings=SETTINGS_13):
    """Train a DQN for total_steps steps of env (the contract of the statement); return (q_net, buffer, history), with
    history = {"returns": the return of each finished episode, "ends": the number of steps done (1, 2, ...) when each
    of them ended}."""
    s = settings
    torch.manual_seed(seed)
    q_net = make_q_net_13(s["hidden"])
    q_target = copy.deepcopy(q_net)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=s["lr"])
    rng = np.random.default_rng(seed)
    buffer = mylearn.rl.ReplayBuffer(s["capacity"], obs_shape=(4,), obs_dtype="float32", random_state=seed)
    history = {"returns": [], "ends": []}
    state, _ = env.reset(seed=seed)
    episode_return = 0.0
    for step in range(total_steps):
        if rng.random() < epsilon_13(step, total_steps, s):
            action = int(rng.integers(2))
        else:
            with torch.no_grad():
                action = int(q_net(torch.as_tensor(state).unsqueeze(0)).argmax(dim=1))
        next_state, reward, terminated, truncated, _ = env.step(action)
        buffer.push(state, action, reward, next_state, terminated)     # terminated only: a time limit bootstraps
        episode_return += reward
        state = next_state
        if terminated or truncated:
            history["returns"].append(episode_return)
            history["ends"].append(step + 1)
            episode_return = 0.0
            state, _ = env.reset()
        if step + 1 >= s["learning_starts"] and len(buffer) >= s["batch_size"] and step % s["train_freq"] == 0:
            batch = buffer.sample(s["batch_size"])
            y = dqn_targets_13(q_target, torch.as_tensor(batch["next_states"]),
                               torch.as_tensor(batch["rewards"], dtype=torch.float32),
                               torch.as_tensor(batch["terminated"]), s["gamma"])
            q_sa = q_net(torch.as_tensor(batch["states"])).gather(1, torch.as_tensor(batch["actions"])[:, None])
            loss = F.huber_loss(q_sa.squeeze(1), y, delta=s["huber_delta"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), s["max_grad_norm"])
            optimizer.step()
        if (step + 1) % s["target_every"] == 0:
            q_target.load_state_dict(q_net.state_dict())
    return q_net, buffer, history'''

CHECK_13 = r'''ok_13 = False
with checked("B8.13"):
    net_13 = make_q_net_13()
    if returned("B8.13", "make_q_net_13", net_13):
        out_13 = net_13(torch.zeros(5, 4)) if isinstance(net_13, nn.Module) else None
        verdict("B8.13", out_13 is not None and tuple(out_13.shape) == (5, 2) and out_13.dtype == torch.float32,
                f"make_q_net_13 : {sum(p.numel() for p in net_13.parameters())} paramètres, une sortie (5, 2) pour 5 "
                "observations.",
                f"make_q_net_13 : attendu un nn.Module qui rend (5, 2) en float32 pour 5 observations, reçu "
                f"{type(net_13).__name__}" + (f" de sortie {tuple(out_13.shape)} {out_13.dtype}" if out_13 is not None
                                              else "") + ".")

targets_ok_13 = False
with checked("B8.13"):
    probe_13 = nn.Linear(4, 2)                 # Q_target(s', .) = the first two numbers of s'
    with torch.no_grad():
        probe_13.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
        probe_13.bias.zero_()
    next_13 = torch.tensor([[2.0, 3.0, 0.3, 0.0], [5.0, -1.0, 0.0, 0.2], [0.5, 0.25, 0.1, 0.1], [-1.0, -2.0, 0.0, 0.0]])
    rewards_13 = torch.tensor([1.0, 0.0, 1.0, -1.0])
    term_13 = torch.tensor([False, True, False, False])
    y_13 = dqn_targets_13(probe_13, next_13.clone(), rewards_13.clone(), term_13.clone(), 0.9)
    if returned("B8.13", "dqn_targets_13", y_13):
        if getattr(y_13, "requires_grad", False):
            print("❌ Ex B8.13 : les cibles portent un gradient (requires_grad=True) : la loss serait dérivée aussi à "
                  "travers elles ; relis « Apprendre une valeur par régression » (208.1).")
        else:
            targets_ok_13 = same("B8.13", "dqn_targets_13 sur 4 transitions (la 2ᵉ terminée, γ = 0,9)", y_13,
                                 rewards_13 + 0.9 * (~term_13).float() * torch.tensor([3.0, 5.0, 0.5, -1.0]),
                                 rtol=1e-6, atol=1e-6)

eps_ok_13 = False
with checked("B8.13"):
    eps_steps_13 = [0, 1500, 3000, 6000, 9999]
    eps_ok_13 = same("B8.13", "epsilon_13 aux pas 0, 1 500, 3 000, 6 000 et 9 999 d'un entraînement de 10 000 pas",
         [epsilon_13(k, 10_000) for k in eps_steps_13],
         [wbr.linear_epsilon(k, 10_000, 1.0, 0.05, SETTINGS_13["eps_fraction"]) for k in eps_steps_13])

with checked("B8.13"):
    if not targets_ok_13:
        print("⏳ Ex B8.13 : l'essai de train_dqn_13 attend des cibles justes (dqn_targets_13, ci-dessus).")
    elif rl_ready("B8.13", "ReplayBuffer"):
        small_13 = {**SETTINGS_13, "learning_starts": 200, "capacity": 1_000, "batch_size": 32, "target_every": 100}
        counter_13 = wbr.StepCounter(wbr.make_env(max_episode_steps=15))   # many episodes cut by a 15-step time limit
        targets_seen_13 = []                      # the target network given to each call of dqn_targets_13 (by name)
        real_targets_13 = dqn_targets_13

        def spy_targets_13(q_target, *args, **kwargs):
            with torch.no_grad():
                targets_seen_13.append(torch.cat([p.detach().reshape(-1) for p in q_target.parameters()]).clone())
            return real_targets_13(q_target, *args, **kwargs)

        dqn_targets_13 = spy_targets_13
        try:
            with own_code("B8.13", "train_dqn_13", REFERENCE_CODE):
                out_a_13 = train_dqn_13(counter_13, 0, 600, small_13)
        finally:
            dqn_targets_13 = real_targets_13
        q_a_13, buf_a_13, hist_a_13 = pair("B8.13", "train_dqn_13", out_a_13, names=("q_net", "buffer", "history"))
        good_13 = []
        good_13.append(counter_13.steps == 600)
        verdict("B8.13", good_13[-1], "600 pas d'environnement pour total_steps = 600.",
                f"train_dqn_13(env, 0, 600, ...) a joué {counter_13.steps} pas d'environnement au lieu de 600.")
        is_buffer_13 = isinstance(buf_a_13, mylearn.rl.ReplayBuffer)
        good_13.append(is_buffer_13 and buf_a_13.capacity == 1_000 and len(buf_a_13) == 600)
        verdict("B8.13", good_13[-1], "la mémoire est une ReplayBuffer de ta librairie : capacité 1 000, 600 transitions.",
                "attendu en deuxième sortie la ReplayBuffer de mylearn.rl, de capacité settings[\"capacity\"] (1 000) "
                "et qui contient les 600 transitions, reçu " + (f"une ReplayBuffer de capacité {buf_a_13.capacity} et "
                f"{len(buf_a_13)} transitions" if is_buffer_13 else type(buf_a_13).__name__) + ".")
        if is_buffer_13:
            stored_13 = int(np.sum(buf_a_13._terminated[:min(len(buf_a_13), buf_a_13.capacity)]))
            good_13.append(stored_13 == counter_13.terminations)
            verdict("B8.13", good_13[-1],
                    f"la mémoire range {stored_13} fins d'épisode : les {counter_13.terminations} vraies fins, pas les "
                    f"{counter_13.truncations} limites de temps.",
                    f"la mémoire range {stored_13} transitions marquées comme fin, pour {counter_13.terminations} vraies "
                    f"fins (terminated) et {counter_13.truncations} limites de temps (truncated) ; relis l'encadré ⚠️ "
                    "« la limite de temps n'est pas une fin » (208.2).")
        returns_a_13 = list(hist_a_13.get("returns", [])) if isinstance(hist_a_13, dict) else None
        ends_a_13 = list(hist_a_13.get("ends", [])) if isinstance(hist_a_13, dict) else None
        n_ends_13 = counter_13.terminations + counter_13.truncations
        good_13.append(returns_a_13 is not None and len(returns_a_13) == len(ends_a_13) == n_ends_13
                       and ends_a_13 == sorted(ends_a_13) and (not ends_a_13 or (ends_a_13[-1] <= 600
                       and abs(sum(returns_a_13) - ends_a_13[-1]) < 1e-9)))
        verdict("B8.13", good_13[-1], f"history : {n_ends_13} épisodes finis, leurs retours et leurs fins.",
                f"history doit avoir \"returns\" et \"ends\", une valeur par épisode fini ({n_ends_13} ici), les fins "
                "comptées en pas d'environnement (1, 2, …) et la somme des retours égale au pas de la dernière fin ; "
                f"reçu {len(returns_a_13) if returns_a_13 is not None else '?'} retours.")
        seeds_13 = counter_13.reset_seeds
        later_13 = sum(s is not None for s in seeds_13[1:])
        good_13.append(bool(seeds_13) and seeds_13[0] == 0 and later_13 == 0)
        verdict("B8.13", good_13[-1], f"la graine va au premier reset seulement ({len(seeds_13)} resets en 600 pas).",
                f"env.reset reçoit {seeds_13[0] if seeds_13 else None!r} au premier épisode et une graine à {later_13} "
                f"des {max(len(seeds_13) - 1, 0)} suivants : le contrat la donne au premier épisode seulement "
                "(reset(seed=seed), puis reset()) ; sinon chaque épisode repart du même état.")
        changes_13 = sum(not torch.equal(a, b) for a, b in zip(targets_seen_13, targets_seen_13[1:]))
        good_13.append(bool(targets_seen_13) and 3 <= changes_13 <= 5)
        verdict("B8.13", good_13[-1],
                f"le réseau cible que reçoit dqn_targets_13 change {changes_13} fois en 600 pas : une copie tous les "
                "target_every = 100 pas, une fois les mises à jour commencées.",
                "train_dqn_13 n'appelle pas dqn_targets_13 : le contrat calcule les cibles avec elle (appelle-la par ce "
                "nom : la vérification l'observe)." if not targets_seen_13 else
                f"le réseau cible que reçoit dqn_targets_13 change {changes_13} fois en 600 pas (mises à jour du pas "
                "200 au pas 600, target_every = 100), au lieu de 4 : " + ("est-ce le réseau en ligne lui-même ?"
                if changes_13 > len(targets_seen_13) // 2 else "est-il recopié (load_state_dict) tous les "
                "target_every pas ?") + " Relis « Les deux stabilisateurs » (208.2).")
        counter_b_13 = wbr.StepCounter(wbr.make_env(max_episode_steps=15))
        q_b_13, _, hist_b_13 = train_dqn_13(counter_b_13, 0, 600, small_13)
        probe_states_13 = torch.as_tensor(np.random.default_rng(13).normal(0, 0.1, (64, 4)), dtype=torch.float32)
        with torch.no_grad():
            same_q_13 = torch.equal(q_a_13(probe_states_13), q_b_13(probe_states_13))
        good_13.append(same_q_13 and hist_b_13 == hist_a_13)
        verdict("B8.13", good_13[-1], "deux entraînements de même graine donnent le même réseau et le même historique.",
                "deux entraînements de même graine ne donnent pas le même " + ("historique" if same_q_13 else "réseau")
                + " : relis le contrat des graines (torch.manual_seed, le premier reset, le générateur de "
                "l'exploration, random_state de la mémoire).")
        ok_13 = all(good_13) and eps_ok_13'''

EXPERIMENT_13 = r'''if not ok_13:
    print("⏳ Ex B8.13 : l'entraînement complet attend que la vérification ci-dessus passe.")
else:
    with checked("B8.13"):
        start_13 = time.perf_counter()
        with own_code("B8.13", "train_dqn_13", REFERENCE_CODE):
            Q_NET_13, BUFFER_13, HISTORY_13 = train_dqn_13(wbr.make_env(), SEED_13, STEPS_13)
        seconds_13 = time.perf_counter() - start_13
        returns_13 = np.asarray(HISTORY_13["returns"], dtype=float)
        eval_13 = wbr.evaluate(wbr.greedy_q(Q_NET_13), 100, EVAL_SEED)
        fig_13, ax_13 = plt.subplots(figsize=(7, 3))
        ax_13.plot(HISTORY_13["ends"], returns_13, ".", alpha=0.25, label="un épisode d'entraînement")
        plot_returns(ax_13, returns_13, "moyenne glissante sur 20 épisodes", ends=HISTORY_13["ends"])
        ax_13.axhline(eval_13["mean_return"], color="k", ls="--", lw=1, label="évaluation gloutonne finale")
        ax_13.set(xlabel="pas d'environnement", ylabel="retour", title=f"Ton DQN, graine {SEED_13}")
        ax_13.legend(fontsize=8)
        plt.show()
        print(f"{STEPS_13} pas en {seconds_13:.0f} s, {len(returns_13)} épisodes · 20 premiers épisodes : "
              f"{fr(returns_13[:20].mean(), 1)} · 20 derniers : {fr(returns_13[-20:].mean(), 1)} · politique gloutonne "
              f"sur 100 épisodes : {fr(eval_13['mean_return'], 1)} (écart-type {fr(eval_13['std_return'], 1)})")
        verdict("B8.13", returns_13[-20:].mean() >= 2 * returns_13[:20].mean(),
                "le retour d'entraînement monte : les 20 derniers épisodes valent au moins le double des 20 premiers.",
                "le retour d'entraînement ne monte pas : les 20 derniers épisodes ne valent pas le double des 20 premiers ; "
                "l'essai de 600 pas passe, alors relis la loss : Q(s, a) de l'action jouée (gather), pas max_a Q(s, a) "
                "(208.2).")
        verdict("B8.13", eval_13["mean_return"] >= THRESHOLD_13,
                f"la politique gloutonne dépasse {THRESHOLD_13:g} (la référence, sur 8 graines, fait au moins "
                f"«WORST_13»).",
                f"la politique gloutonne tient {fr(eval_13['mean_return'], 1)} pas en moyenne : moins que les "
                f"{THRESHOLD_13:g} attendus (la référence, sur 8 graines, fait au moins «WORST_13») ; relis la loss : "
                "Q(s, a) de l'action jouée (gather), pas max_a Q(s, a) (208.2).")'''

# --- 🐛 B8.14 ---------------------------------------------------------------------------------------------------------
GIVEN_14 = r'''STEPS_14 = 8_000 if FAST_MODE else 20_000
THRESHOLD_14 = «THRESHOLD_14»       # the greedy evaluation of the repaired DQN (calibrated on 5 seeds of the repair)


def run_colleague_14(seed, total_steps):
    """The training loop of the colleague (right as it is): it calls new_memory, remember, exploration_rate and
    bellman_targets and the constants of the cell below. Returns (online, log): log["returns"] and log["ends"] (the
    finished episodes), log["eps"] (ε every 100 steps) and log["q_mean"] (the mean of max Q over 256 fixed states,
    every 500 steps)."""
    torch.manual_seed(seed)
    online = wbr.q_network(64)
    frozen = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    memory = new_memory(seed)
    env = wbr.make_env()
    obs, _ = env.reset(seed=seed)
    probe = torch.as_tensor(np.random.default_rng(14).uniform(-0.05, 0.05, (256, 4)), dtype=torch.float32)
    log = {"returns": [], "ends": [], "eps": [], "q_mean": []}
    episode_return = 0.0
    for step in range(total_steps):
        eps = exploration_rate(step)
        if step % 100 == 0:
            log["eps"].append(float(eps))
        if rng.random() < eps:
            action = int(rng.integers(2))
        else:
            with torch.no_grad():
                action = int(online(torch.as_tensor(obs).unsqueeze(0)).argmax(dim=1))
        next_obs, reward, term, trunc, _ = env.step(action)
        over = remember(memory, obs, action, reward, next_obs, term, trunc)
        episode_return += reward
        obs = next_obs
        if over:
            log["returns"].append(episode_return)
            log["ends"].append(step + 1)
            episode_return = 0.0
            obs, _ = env.reset()
        if step >= 500 and len(memory) >= MINI_BATCH:
            batch = memory.sample(MINI_BATCH)
            y = bellman_targets(online, frozen, batch)
            q_sa = online(torch.as_tensor(batch["states"])).gather(1, torch.as_tensor(batch["actions"])[:, None])
            loss = F.huber_loss(q_sa.squeeze(1), y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
        if (step + 1) % 500 == 0:
            frozen.load_state_dict(online.state_dict())
            with torch.no_grad():
                log["q_mean"].append(float(online(probe).max(dim=1).values.mean()))
    return online, log


class _SpyMemory_14:
    """Records what remember stores (point 2), without a real memory."""

    def __init__(self):
        self.pushed = []

    def push(self, state, action, reward, next_state, terminated):
        self.pushed.append(terminated)


def diagnose_14():
    """Four diagnoses of the colleague's pieces, in a row: the first that fails stops the chain. True when they pass."""
    rng = np.random.default_rng(140)
    states = rng.uniform(-0.1, 0.1, (5, 4)).astype(np.float32)
    next_states = rng.uniform(-0.1, 0.1, (5, 4)).astype(np.float32)
    batch = {"states": states, "actions": np.array([0, 1, 1, 0, 1]), "rewards": np.ones(5),
             "next_states": next_states, "terminated": np.array([False, False, True, False, False])}

    def point1():
        online, frozen = wbr.q_network(16, seed=1), wbr.q_network(16, seed=2)
        y = bellman_targets(online, frozen, {k: v.copy() for k, v in batch.items()})
        if getattr(y, "requires_grad", False):
            return False, ("les cibles de 5 transitions portent un gradient (requires_grad=True) : loss.backward() "
                           "dériverait aussi à travers elles"), \
                "relis « Apprendre une valeur par régression » (208.1) : qu'est la cible pendant le calcul du gradient ?"
        want = wbr.dqn_targets(online, frozen, torch.as_tensor(next_states), torch.ones(5),
                               torch.as_tensor(batch["terminated"]), DISCOUNT)
        y = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
        if y.shape == want.shape and torch.allclose(y, want, atol=1e-6):
            return True, "les cibles sont constantes pour le gradient et viennent du réseau figé", ""
        moved = wbr.q_network(16, seed=3)
        y_moved = torch.as_tensor(bellman_targets(moved, frozen, {k: v.copy() for k, v in batch.items()})).reshape(-1)
        if y_moved.shape == y.shape and not torch.allclose(y_moved, y):
            return False, (f"les cibles bougent (jusqu'à {fr(float((y_moved - y).abs().max()), 3)}) quand seuls les poids "
                           "du réseau en ligne changent"), \
                "relis « Les deux stabilisateurs » (208.2) : quel réseau calcule la cible, et pourquoi ?"
        return False, f"les cibles s'écartent de r + γ(1 − terminated) max Q(s′, ·)", "relis bellman_targets ligne à ligne"

    def point2():
        cases = [(False, True, "une limite de temps (terminated=False, truncated=True)"),
                 (True, False, "une vraie fin (terminated=True, truncated=False)"),
                 (False, False, "un pas ordinaire")]
        for term, trunc, what in cases:
            spy = _SpyMemory_14()
            over = remember(spy, states[0], 1, 1.0, next_states[0], term, trunc)
            if len(spy.pushed) != 1:
                return False, f"pour {what}, remember range {len(spy.pushed)} transitions au lieu d'une", \
                    "relis remember"
            if bool(over) != (term or trunc):
                return False, (f"pour {what}, remember rend {bool(over)} : la boucle " +
                               ("ne remettrait pas l'environnement à zéro" if term or trunc else
                                "remettrait l'environnement à zéro en plein épisode")), "relis sa docstring"
            if bool(spy.pushed[0]) != term:
                return False, (f"{what} entre dans la mémoire avec le drapeau de fin {bool(spy.pushed[0])}" +
                               (" : sa cible n'amorcera pas sur l'état suivant" if trunc else "")), \
                    "relis l'encadré ⚠️ « la limite de temps n'est pas une fin » (208.2)"
        return True, "seule une vraie fin entre dans la mémoire comme une fin ; les deux fins remettent à zéro", ""

    def point3():
        steps = [0, EPS_STEPS // 2, EPS_STEPS, 2 * EPS_STEPS]
        got = [float(exploration_rate(k)) for k in steps]
        want = [wbr.linear_epsilon(k, EPS_STEPS, EPS_FIRST, EPS_LAST, 1.0) for k in steps]
        if np.allclose(got, want, atol=1e-9):
            return True, f"ε passe de {fr(got[0], 2)} à {fr(got[2], 2)} en {EPS_STEPS} pas", ""
        shown = ", ".join(f"{fr(v, 3)} au pas {k}" for v, k in zip(got, steps))
        if max(got) - min(got) < 1e-9:
            return False, f"ε vaut {shown} : la politique ne quitte jamais le même niveau d'exploration", \
                "que doit valoir ε à la fin annoncée par la docstring ?"
        return False, f"ε vaut {shown}, ce qui ne suit pas la docstring", "relis exploration_rate ligne à ligne"

    def point4():
        memory = new_memory(0)
        for k in range(2000):
            memory.push(np.array([k, 0, 0, 0], np.float32), 0, 1.0, np.array([k + 1, 0, 0, 0], np.float32), False)
        drawn = np.concatenate([memory.sample(MINI_BATCH)["states"][:, 0] for _ in range(20)])
        recent = float(np.mean(drawn >= 1900))
        if recent <= 0.15:
            return True, (f"après 2 000 pas, {fr(100 * recent, 0)} % des transitions tirées datent des 100 derniers pas "
                          "(les mini-batches mêlent des moments éloignés)"), ""
        return False, (f"après 2 000 pas, {fr(100 * recent, 0)} % des transitions de 20 mini-batches de {MINI_BATCH} "
                       "datent des 100 derniers pas, soit deux épisodes d'un agent qui tient 50 pas : chaque mini-batch "
                       "reste fait de pas qui se suivent"), \
            "relis « Ces nombres montrent surtout… » (208.2) et 🧠 Q3 : que doit garder la mémoire ?"

    points = [("la cible", point1), ("les fins d'épisode", point2), ("l'exploration", point3), ("la mémoire", point4)]
    for k, (name, run) in enumerate(points, start=1):
        try:
            ok, measured, why = run()
        except NotImplementedError as error:
            print(f"⏳ Ex B8.14 : point {k}, {name} : pas encore fait ({error}) : la mémoire est ta ReplayBuffer "
                  "(rl.py, Ex 26.31).")
            return False
        except Exception as error:  # noqa: BLE001 - the colleague's code, as repaired: one ❌ line, the chain stops
            ok, measured, why = False, f"une erreur {type(error).__name__} : {str(error)[:200]}", "relis la fonction en cause"
        if ok:
            print(f"✅ Ex B8.14 : point {k}, {name} ({measured}).")
        else:
            print(f"❌ Ex B8.14 : point {k}, {name} : {measured} ; {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex B8.14 : point {later} : corrige d'abord le point {k}.")
            return False
    print("✅ Ex B8.14 : les quatre diagnostics passent.")
    return True'''

TODO_14 = r'''# The DQN of a colleague, to repair IN PLACE (same names, same signatures). It does not learn as it should on
# CartPole; four mistakes slipped into its pieces (run_colleague_14, the loop of the cell above that calls them, is right).
LEARNING_RATE = 5e-4
MINI_BATCH = 64
MEMORY_SLOTS = 100
DISCOUNT = 0.99
EPS_FIRST, EPS_LAST, EPS_STEPS = 1.0, 0.05, 3_000


def new_memory(seed):
    """The replay memory of the agent: your mylearn.rl.ReplayBuffer, for CartPole observations."""
    return mylearn.rl.ReplayBuffer(MEMORY_SLOTS, obs_shape=(4,), obs_dtype="float32", random_state=seed)


def remember(memory, obs, action, reward, next_obs, term, trunc):
    """Store one transition in the memory; return True when the episode is over (the loop then resets the env)."""
    finished = term | trunc
    memory.push(obs, action, reward, next_obs, finished)
    return finished


def exploration_rate(step):
    """ε of a step: from EPS_FIRST down to EPS_LAST, linearly over the first EPS_STEPS steps, then EPS_LAST."""
    weight = max(0.0, 1.0 - step / EPS_STEPS)          # 1 at the start, 0 after EPS_STEPS steps
    return weight * EPS_FIRST + (1.0 - weight) * EPS_FIRST


def bellman_targets(online, frozen, batch):
    """The targets of a mini-batch (the dict of memory.sample), a tensor (B,):
    y = r + DISCOUNT (1 - terminated) max_a' Q_frozen(s', a'), a constant for the gradient."""
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
    alive = 1.0 - torch.as_tensor(batch["terminated"], dtype=torch.float32)
    follow = online(torch.as_tensor(batch["next_states"])).max(dim=1).values
    return rewards + DISCOUNT * alive * follow'''

FIX_ORDER_14 = ("target", "done", "epsilon", "memory")
FIXES_14 = {
    "target": [("    follow = online(torch.as_tensor(batch[\"next_states\"])).max(dim=1).values\n",
                "    with torch.no_grad():                                   # 1. the frozen network, no gradient\n"
                "        follow = frozen(torch.as_tensor(batch[\"next_states\"])).max(dim=1).values\n")],
    "done": [("    memory.push(obs, action, reward, next_obs, finished)\n",
              "    memory.push(obs, action, reward, next_obs, term)           # 2. a time limit is not an end\n")],
    "epsilon": [("    return weight * EPS_FIRST + (1.0 - weight) * EPS_FIRST\n",
                 "    return weight * EPS_FIRST + (1.0 - weight) * EPS_LAST    # 3. the end of the schedule\n")],
    "memory": [("MEMORY_SLOTS = 100\n", "MEMORY_SLOTS = 10_000                  # 4. a memory of many episodes\n")],
}


def repaired_14(fixed=FIX_ORDER_14) -> str:
    """The colleague's code with the repairs ``fixed`` applied (all of them: the solution)."""
    text = TODO_14.replace("# The DQN of a colleague, to repair IN PLACE (same names, same signatures). It does not learn "
                           "as it should on\n# CartPole; four mistakes slipped into its pieces (run_colleague_14, the loop "
                           "of the cell above that calls them, is right).",
                           "# The DQN of the colleague, repaired: four changes, numbered in the comments")
    for name in fixed:
        for old, new in FIXES_14[name]:
            if text.count(old) != 1:
                raise ValueError(f"repair {name}: {old!r} is not found once")
            text = text.replace(old, new)
    return text


SOLUTION_14 = repaired_14()

CHECK_14 = r'''repaired_14_ok = False
with checked("B8.14"):
    repaired_14_ok = diagnose_14()'''

EXPERIMENT_14 = r'''# The colleague's DQN as it is now (before the repair: its symptoms; after: the repaired agent), STEPS_14 steps
with checked("B8.14"):
    start_14 = time.perf_counter()
    online_14, log_14 = run_colleague_14(0, STEPS_14)
    eval_14 = wbr.evaluate(wbr.greedy_q(online_14), 100, EVAL_SEED)
    fig_14, axes_14 = plt.subplots(1, 3, figsize=(11, 2.8))
    if log_14["returns"]:
        axes_14[0].plot(log_14["ends"], log_14["returns"], ".", alpha=0.3)
        plot_returns(axes_14[0], log_14["returns"], "moyenne sur 20", ends=log_14["ends"])
    axes_14[0].set(xlabel="pas", ylabel="retour", title="retours d'entraînement")
    axes_14[1].plot(np.arange(len(log_14["eps"])) * 100, log_14["eps"])
    axes_14[1].set(xlabel="pas", ylabel="ε", title="exploration", ylim=(0, 1.05))
    axes_14[2].plot(np.arange(1, len(log_14["q_mean"]) + 1) * 500, log_14["q_mean"], marker="o")
    axes_14[2].set(xlabel="pas", ylabel="moyenne de max Q", title="valeurs de 256 états de départ")
    plt.tight_layout()
    plt.show()
    print(f"{STEPS_14} pas en {time.perf_counter() - start_14:.0f} s, {len(log_14['returns'])} épisodes · 20 derniers "
          f"épisodes d'entraînement : {fr(np.mean(log_14['returns'][-20:]), 1) if log_14['returns'] else '?'} · politique "
          f"gloutonne sur 100 épisodes : {fr(eval_14['mean_return'], 1)} · max Q moyen à la fin : "
          f"{fr(log_14['q_mean'][-1], 1) if log_14['q_mean'] else '?'}")
    if repaired_14_ok:
        verdict("B8.14", eval_14["mean_return"] >= THRESHOLD_14,
                f"réparé, le DQN apprend : sa politique gloutonne dépasse {THRESHOLD_14:g}.",
                f"les quatre diagnostics passent, mais la politique gloutonne ne tient que {fr(eval_14['mean_return'], 1)} "
                f"pas (au moins {THRESHOLD_14:g} attendus) : as-tu changé autre chose que les quatre erreurs ?")
    else:
        print("⏳ Ex B8.14 : ces courbes montrent le DQN tel qu'il est ; la vérification ci-dessus dit ce qui reste.")'''

# --- 🔬 B8.15 ---------------------------------------------------------------------------------------------------------
GIVEN_15 = r'''VARIANTS_15 = wbr.DQN_VARIANTS                 # ("dqn", "no_target", "no_replay", "double")
NAMES_15 = {"dqn": "DQN", "no_target": "sans réseau cible", "no_replay": "sans rejeu", "double": "Double DQN"}
A15 = wbr.precomputed("a15")                   # the curves of the reference: 4 variants x 3 seeds (protocol inside)
REPLAY_STEPS_15 = 3_000                        # the prefix of one recorded run that this notebook trains again'''

TODO_15 = r'''def targets_15(online, frozen, next_states, rewards, terminated, gamma, variant):
    """The targets of a mini-batch for the four variants, a tensor (B,) without gradient: "dqn" and "no_replay", those of
    dqn_targets_13 with the target network `frozen`; "no_target", the same formula with the online network in place of
    the target network; "double", the action chosen by the online network (argmax), its value read in `frozen`."""
    raise NotImplementedError("targets_15")


def memory_size_15(variant, settings=SETTINGS_13):
    """The capacity of the replay memory of a variant: settings["capacity"], except for "no_replay":
    settings["batch_size"] (each mini-batch is then the last batch_size transitions, in a random order)."""
    raise NotImplementedError("memory_size_15")'''

SOLUTION_15 = r'''def targets_15(online, frozen, next_states, rewards, terminated, gamma, variant):
    """The targets of a mini-batch for the four variants, a tensor (B,) without gradient: "dqn" and "no_replay", those of
    dqn_targets_13 with the target network `frozen`; "no_target", the same formula with the online network in place of
    the target network; "double", the action chosen by the online network (argmax), its value read in `frozen`."""
    with torch.no_grad():
        if variant == "double":
            best = online(next_states).argmax(dim=1, keepdim=True)        # the online network chooses...
            following = frozen(next_states).gather(1, best).squeeze(1)    # ...the target network values it
        else:
            following = (online if variant == "no_target" else frozen)(next_states).max(dim=1).values
        return rewards + gamma * (1.0 - terminated.float()) * following


def memory_size_15(variant, settings=SETTINGS_13):
    """The capacity of the replay memory of a variant: settings["capacity"], except for "no_replay":
    settings["batch_size"] (each mini-batch is then the last batch_size transitions, in a random order)."""
    return settings["batch_size"] if variant == "no_replay" else settings["capacity"]'''

CHECK_15 = r'''ok_15 = False
with checked("B8.15"):
    online_15, frozen_15 = wbr.q_network(32, seed=150), wbr.q_network(32, seed=151)
    next_15 = torch.as_tensor(np.random.default_rng(15).normal(0, 1, (6, 4)), dtype=torch.float32)
    rewards_15 = torch.tensor([1.0, 1.0, 0.5, 1.0, -1.0, 1.0])
    term_15 = torch.tensor([False, True, False, False, False, False])
    with torch.no_grad():
        disagree_15 = int((online_15(next_15).argmax(1) != frozen_15(next_15).argmax(1)).sum())
    good_15 = []
    swapped_15 = wbr.dqn_targets(frozen_15, online_15, next_15, rewards_15, term_15, 0.99, "double")   # roles swapped
    for variant_15 in VARIANTS_15:
        with own_code("B8.15", "targets_15", REFERENCE_CODE):
            y_15 = targets_15(online_15, frozen_15, next_15.clone(), rewards_15.clone(), term_15.clone(), 0.99, variant_15)
        if not returned("B8.15", f"targets_15(..., {variant_15!r})", y_15):
            good_15.append(False)
            continue
        if getattr(y_15, "requires_grad", False):
            print(f"❌ Ex B8.15 : targets_15(..., {variant_15!r}) porte un gradient (requires_grad=True).")
            good_15.append(False)
            continue
        want_15 = wbr.dqn_targets(online_15, frozen_15, next_15, rewards_15, term_15, 0.99, variant_15)
        flat_15 = np.asarray(torch.as_tensor(y_15, dtype=torch.float32).detach()).reshape(-1)
        if (variant_15 == "double" and flat_15.shape == (6,) and not np.allclose(flat_15, want_15.numpy(), atol=1e-6)
                and np.allclose(flat_15, swapped_15.numpy(), atol=1e-6)):
            print("❌ Ex B8.15 : targets_15(..., 'double') : les rôles sont inversés ; le réseau en ligne choisit l'action "
                  "(argmax), le réseau cible donne sa valeur (208.2, « La surestimation, et Double DQN »).")
            good_15.append(False)
            continue
        good_15.append(same("B8.15", f"targets_15, variante {variant_15!r} (6 transitions, la 2ᵉ terminée)", y_15,
                            want_15, rtol=1e-6, atol=1e-6))
    sizes_15 = {v: memory_size_15(v) for v in VARIANTS_15}
    want_sizes_15 = {v: (SETTINGS_13["batch_size"] if v == "no_replay" else SETTINGS_13["capacity"]) for v in VARIANTS_15}
    good_15.append(sizes_15 == want_sizes_15)
    verdict("B8.15", good_15[-1], f"memory_size_15 : {sizes_15}.",
            f"memory_size_15 : attendu {want_sizes_15}, reçu {sizes_15}.")
    print(f"(le réseau en ligne et le réseau cible de la vérification choisissent une autre action dans "
          f"{disagree_15} des 6 états s′ : Double DQN y diffère de DQN)")
    ok_15 = all(good_15)'''

EXPERIMENT_15 = r'''# The four variants on three seeds (fast mode: the curves of the reference, precomputed; full mode: your pieces in the
# reference loop, 3 seeds x 4 variants, 🚀), then the median and the range of the three seeds
protocol_15 = A15["protocol"]
if FAST_MODE:
    runs_15 = A15["runs"]
    replay_15, replay_hist_15 = wbr.train_dqn(protocol_15["seeds"][0], protocol_15["steps"], variant="double",
                                              eval_every=protocol_15["eval_every"],
                                              eval_episodes=protocol_15["eval_episodes"],
                                              eval_seed=protocol_15["eval_seed"], stop_after=REPLAY_STEPS_15)
    recorded_15 = runs_15[f"double/{protocol_15['seeds'][0]}"]["eval_returns"][:len(replay_hist_15["eval_returns"])]
    if np.allclose(replay_hist_15["eval_returns"], recorded_15, atol=1e-6):
        print(f"✅ Ex B8.15 : Double DQN, graine {protocol_15['seeds'][0]}, rejoué sur {REPLAY_STEPS_15} pas : "
              f"{replay_hist_15['eval_returns']}, comme dans le fichier : les courbes ci-dessous sont celles de ce code.")
    else:
        print(f"ℹ️ Double DQN, graine {protocol_15['seeds'][0]}, rejoué sur {REPLAY_STEPS_15} pas : "
              f"{replay_hist_15['eval_returns']}, le fichier dit {recorded_15} : une autre machine ou une autre version "
              "de PyTorch arrondit autrement, et un écart minuscule change la suite d'un run de RL (208.6). Les courbes "
              "ci-dessous restent celles du fichier.")
elif not ok_15:
    runs_15 = None
    print("⏳ Ex B8.15 : le mode complet entraîne tes variantes ; il attend que la vérification ci-dessus passe.")
else:
    runs_15 = {}
    for variant_15 in VARIANTS_15:
        for seed_15 in protocol_15["seeds"]:
            _, h_15 = wbr.train_dqn(seed_15, wbr.dqn_steps(False), variant=variant_15, eval_every=2_500,
                                    eval_episodes=protocol_15["eval_episodes"], eval_seed=protocol_15["eval_seed"],
                                    targets_fn=targets_15, capacity=memory_size_15(variant_15))
            runs_15[f"{variant_15}/{seed_15}"] = h_15
            print(f"{NAMES_15[variant_15]}, graine {seed_15} : {h_15['seconds']:.0f} s")
if runs_15:
    fig_15, ax_15 = plt.subplots(figsize=(8, 3.6))
    finals_15 = {}
    for variant_15 in VARIANTS_15:
        hs_15 = [runs_15[f"{variant_15}/{s}"] for s in protocol_15["seeds"]]
        steps_15 = hs_15[0]["eval_steps"]
        b_15 = wbr.band([h["eval_returns"] for h in hs_15])
        line_15, = ax_15.plot(steps_15, b_15["median"], label=NAMES_15[variant_15])
        ax_15.fill_between(steps_15, b_15["low"], b_15["high"], color=line_15.get_color(), alpha=0.15)
        finals_15[variant_15] = [h["eval_returns"][-1] for h in hs_15]
    ax_15.axhline(RANDOM_LEVEL, color="gray", ls=":", lw=1)
    ax_15.set(xlabel="pas d'environnement", ylabel="retour glouton (20 épisodes)",
              title="Médiane et intervalle (min-max) sur trois graines")
    ax_15.legend(fontsize=8)
    plt.show()
    for variant_15, values_15 in finals_15.items():
        print(f"{NAMES_15[variant_15]:18s} dernière évaluation par graine : {[round(v, 1) for v in values_15]} · "
              f"médiane {fr(float(np.median(values_15)), 1)}")'''

# =====================================================================================================================
# Part B · the policy (first half: B8.16 to B8.20)
# =====================================================================================================================
# --- 🔮 B8.16 ---------------------------------------------------------------------------------------------------------
TODO_16 = r'''# Your predictions, BEFORE running the next cell (an int, or True/False)
best_16 = ...        # a) 0 (no baseline), 1 (mean baseline) or 2 (normalised returns)
first_200_16 = ...   # b) an int from 0 to 6
beats_16 = ...       # c) True or False'''

EXPERIMENT_16 = r'''P16 = wbr.precomputed("p16")                  # REINFORCE, 3 variants x 6 seeds x 300 episodes (protocol inside)
names_16 = {"none": "sans baseline", "mean": "baseline moyenne", "normalize": "retours normalisés"}
seeds_16 = P16["protocol"]["seeds"]
fig_16, ax_16 = plt.subplots(figsize=(8, 3.4))
last50_16 = {}
for baseline_16, name_16 in names_16.items():
    curves_16 = [wbr.moving_average(P16["runs"][f"{baseline_16}/{s}"]["episode_returns"], 20) for s in seeds_16]
    b_16 = wbr.band(curves_16)
    line_16, = ax_16.plot(np.arange(1, 301), b_16["median"], label=name_16)
    ax_16.fill_between(np.arange(1, 301), b_16["low"], b_16["high"], color=line_16.get_color(), alpha=0.12)
    last50_16[baseline_16] = [float(np.mean(P16["runs"][f"{baseline_16}/{s}"]["episode_returns"][-50:])) for s in seeds_16]
ax_16.axhline(200, color="gray", ls=":", lw=1)
ax_16.set(xlabel="épisode", ylabel="retour (moyenne glissante sur 20)",
          title=f"REINFORCE sur CartPole : médiane et intervalle sur {len(seeds_16)} graines")
ax_16.legend(fontsize=8)
plt.show()
for baseline_16, name_16 in names_16.items():
    reached_16 = sum(bool((wbr.moving_average(P16["runs"][f"{baseline_16}/{s}"]["episode_returns"], 20) >= 200).any())
                     for s in seeds_16)
    print(f"{name_16:20s} moyenne des 50 derniers épisodes, par graine : {[round(v) for v in last50_16[baseline_16]]} · "
          f"médiane {fr(float(np.median(last50_16[baseline_16])), 1)} · moyenne glissante ≥ 200 atteinte sur "
          f"{reached_16} graines sur {len(seeds_16)}")
wb.check("B8.16a", best_16)
wb.check("B8.16b", first_200_16)
wb.check("B8.16c", beats_16)'''

# --- 📦 B8.17 ---------------------------------------------------------------------------------------------------------
GIVEN_17 = r'''C17 = wbr.precomputed("c17")                  # a policy trained 60 episodes by the reference (its weights)
HALF_TRAINED_17 = wbr.policy_network(64)
HALF_TRAINED_17.load_state_dict({k: torch.tensor(v) for k, v in C17["state_dict"].items()})
POLICIES_17 = {"non entraînée": wbr.policy_network(64, seed=17), "entraînée 60 épisodes": HALF_TRAINED_17}
EPISODES_17 = {name: wbr.collect_episodes(net, 32, 1700) for name, net in POLICIES_17.items()}   # 32 episodes each
N_17 = 300                                       # training episodes of each variant (B8.16 shows the same protocol)
SEED_17 = 17
THRESHOLD_17 = «THRESHOLD_17»                    # the last 50 episodes with the mean baseline (6 seeds of the reference)
for name_17, eps_17 in EPISODES_17.items():
    print(f"politique {name_17} : 32 épisodes, {fr(np.mean([len(e['rewards']) for e in eps_17]), 1)} pas en moyenne")'''

TODO_17 = r'''def make_policy_17(hidden=64):
    """A new policy network for CartPole: 4 inputs, one hidden layer of `hidden` neurons with tanh, 2 outputs (the
    logits of the two actions)."""
    raise NotImplementedError("make_policy_17")


def episode_gradients_17(policy_net, episodes, baseline, gamma=0.99):
    """One REINFORCE gradient per episode (the contract of the statement): a float64 array (M, P), row i the gradient
    of -mean_t(log pi(a_t | s_t) (G_t - b)) of episode i with respect to all the parameters of policy_net, flattened
    in the order of policy_net.parameters(); b = 0 ("none") or the mean of the G_t of ALL the episodes ("mean")."""
    raise NotImplementedError("episode_gradients_17")


def train_reinforce_17(seed, n_episodes, baseline, lr=1e-2, gamma=0.99):
    """REINFORCE on CartPole-v1 (the contract of the statement): one Adam step per episode; baseline "none" or "mean"
    (the mean of the G_t of the 20 previous episodes, 0 for the first). Returns (policy_net, returns), the return of
    each training episode."""
    raise NotImplementedError("train_reinforce_17")'''

SOLUTION_17 = r'''def make_policy_17(hidden=64):
    """A new policy network for CartPole: 4 inputs, one hidden layer of `hidden` neurons with tanh, 2 outputs (the
    logits of the two actions)."""
    return nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 2))


def episode_gradients_17(policy_net, episodes, baseline, gamma=0.99):
    """One REINFORCE gradient per episode (the contract of the statement): a float64 array (M, P), row i the gradient
    of -mean_t(log pi(a_t | s_t) (G_t - b)) of episode i with respect to all the parameters of policy_net, flattened
    in the order of policy_net.parameters(); b = 0 ("none") or the mean of the G_t of ALL the episodes ("mean")."""
    returns = [mylearn.rl.returns_to_go(e["rewards"], gamma) for e in episodes]
    b = float(np.mean(np.concatenate(returns))) if baseline == "mean" else 0.0
    params = list(policy_net.parameters())
    rows = []
    for e, G in zip(episodes, returns):
        log_pi = torch.log_softmax(policy_net(torch.as_tensor(e["states"])), dim=1)
        log_pi_a = log_pi.gather(1, torch.as_tensor(e["actions"])[:, None]).squeeze(1)
        loss = -(log_pi_a * torch.as_tensor(G - b, dtype=torch.float32)).mean()
        grads = torch.autograd.grad(loss, params)                 # leaves the .grad of the network alone
        rows.append(torch.cat([g.reshape(-1) for g in grads]).double().numpy())
    return np.stack(rows)


def train_reinforce_17(seed, n_episodes, baseline, lr=1e-2, gamma=0.99):
    """REINFORCE on CartPole-v1 (the contract of the statement): one Adam step per episode; baseline "none" or "mean"
    (the mean of the G_t of the 20 previous episodes, 0 for the first). Returns (policy_net, returns), the return of
    each training episode."""
    torch.manual_seed(seed)
    policy_net = make_policy_17()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    env = wbr.make_env()
    env.reset(seed=seed)
    past, returns = [], []
    for _ in range(n_episodes):
        state, _ = env.reset()
        log_probs, rewards, done = [], [], False
        while not done:
            dist = torch.distributions.Categorical(logits=policy_net(torch.as_tensor(state)))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(int(action))
            rewards.append(reward)
            done = terminated or truncated
        G = mylearn.rl.returns_to_go(rewards, gamma)
        b = float(np.mean(np.concatenate(past))) if baseline == "mean" and past else 0.0
        loss = -(torch.stack(log_probs) * torch.as_tensor(G - b, dtype=torch.float32)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        past = (past + [G])[-20:]
        returns.append(float(sum(rewards)))
    return policy_net, returns'''

CHECK_17 = r'''grads_ok_17 = False
with checked("B8.17"):
    net_17 = make_policy_17()
    if returned("B8.17", "make_policy_17", net_17):
        out_17 = net_17(torch.zeros(3, 4)) if isinstance(net_17, nn.Module) else None
        verdict("B8.17", out_17 is not None and tuple(out_17.shape) == (3, 2),
                f"make_policy_17 : {sum(p.numel() for p in net_17.parameters())} paramètres, deux logits par observation.",
                "make_policy_17 : attendu un nn.Module qui rend (3, 2) pour 3 observations.")

with checked("B8.17"):
    if rl_ready("B8.17", "returns_to_go"):
        for name_17, net_17 in POLICIES_17.items():
            for baseline_17 in ("none", "mean"):
                copy_17 = copy.deepcopy(net_17)
                with own_code("B8.17", "episode_gradients_17", REFERENCE_CODE):
                    got_17 = episode_gradients_17(copy_17, EPISODES_17[name_17], baseline_17)
                want_17 = wbr.episode_gradients(net_17, EPISODES_17[name_17], baseline_17)
                if not returned("B8.17", "episode_gradients_17", got_17):
                    raise StopCheck
                arr_17 = np.asarray(got_17.detach() if isinstance(got_17, torch.Tensor) else got_17, dtype=float)
                label_17 = f"episode_gradients_17, politique {name_17}, baseline {baseline_17!r}"
                lengths_17 = np.array([len(e["rewards"]) for e in EPISODES_17[name_17]], dtype=float)
                near_17 = lambda a, b: a.shape == b.shape and np.linalg.norm(a - b) <= 1e-3 * np.linalg.norm(b)
                if near_17(arr_17, -want_17):
                    print(f"❌ Ex B8.17 : {label_17} : tous les signes sont inversés : on minimise la loss −mean(log π · (G − b)).")
                    raise StopCheck
                if near_17(arr_17, want_17 * lengths_17[:, None]):
                    print(f"❌ Ex B8.17 : {label_17} : chaque ligne vaut T fois l'attendu (T, la longueur de son épisode) : "
                          "une somme sur les pas au lieu d'une moyenne ?")
                    raise StopCheck
                if baseline_17 == "mean" and near_17(arr_17, np.concatenate(
                        [wbr.episode_gradients(net_17, [e], "mean") for e in EPISODES_17[name_17]])):
                    print(f"❌ Ex B8.17 : {label_17} : ta baseline est la moyenne des G_t de chaque épisode ; le contrat "
                          "demande une seule constante, la moyenne des G_t de tous les épisodes donnés.")
                    raise StopCheck
                if near_17(arr_17, wbr.episode_gradients(net_17, EPISODES_17[name_17], baseline_17, gamma=1.0)):
                    print(f"❌ Ex B8.17 : {label_17} : ce sont les gradients avec γ = 1 : tes retours sont-ils actualisés "
                          "avec gamma ?")
                    raise StopCheck
                if not same("B8.17", label_17, arr_17, want_17, rtol=1e-4, atol=1e-6):
                    raise StopCheck
        grads_ok_17 = True
        if grads_ok_17:
            for name_17, net_17 in POLICIES_17.items():
                spread_17 = {b: wbr.gradient_spread(episode_gradients_17(copy.deepcopy(net_17), EPISODES_17[name_17], b))
                             for b in ("none", "mean")}
                print(f"politique {name_17} : dispersion des gradients (moyenne de ‖g_i − ḡ‖²) sans baseline "
                      f"{spread_17['none']:.3g}, avec {spread_17['mean']:.3g}, rapport {fr(spread_17['none'] / spread_17['mean'], 2)}")'''

EXPERIMENT_17 = r'''# Training: REINFORCE without and with the mean baseline, N_17 episodes each, against the reference curves of B8.16
if not grads_ok_17:
    print("⏳ Ex B8.17 : l'entraînement attend que tes gradients par épisode passent la vérification ci-dessus.")
else:
    with checked("B8.17"):
        P16_17 = wbr.precomputed("p16")
        runs_17 = {}
        for baseline_17 in ("none", "mean"):
            start_17 = time.perf_counter()
            with own_code("B8.17", "train_reinforce_17", REFERENCE_CODE):
                out_17 = train_reinforce_17(SEED_17, N_17, baseline_17)
            net_17, returns_17 = pair("B8.17", "train_reinforce_17", out_17, names=("policy_net", "returns"))
            runs_17[baseline_17] = (net_17, np.asarray(returns_17, dtype=float), time.perf_counter() - start_17)
        fig_17, axes_17 = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        for ax_17, baseline_17 in zip(axes_17, ("none", "mean")):
            ref_17 = wbr.band([wbr.moving_average(P16_17["runs"][f"{baseline_17}/{s}"]["episode_returns"], 20)
                               for s in P16_17["protocol"]["seeds"]])
            ax_17.fill_between(np.arange(1, 301), ref_17["low"], ref_17["high"], color="gray", alpha=0.2,
                               label="référence, 6 graines (min-max)")
            plot_returns(ax_17, runs_17[baseline_17][1], f"toi, graine {SEED_17}")
            ax_17.set(xlabel="épisode", title={"none": "sans baseline", "mean": "baseline moyenne"}[baseline_17])
        axes_17[0].set_ylabel("retour (moyenne glissante sur 20)")
        axes_17[0].legend(fontsize=8)
        plt.show()
        for baseline_17, (net_17, returns_17, seconds_17) in runs_17.items():
            ev_17 = wbr.evaluate(wbr.greedy_logits(net_17), 100, EVAL_SEED)
            print(f"{baseline_17:5s} : {len(returns_17)} épisodes en {seconds_17:.0f} s · 50 derniers : "
                  f"{fr(returns_17[-50:].mean(), 1)} · politique déterministe sur 100 épisodes : {fr(ev_17['mean_return'], 1)}")
        mean_17 = runs_17["mean"][1]
        rose_17 = len(mean_17) >= 20 and float(wbr.moving_average(mean_17, 20).max()) >= 50
        verdict("B8.17", len(mean_17) == N_17 and mean_17[-50:].mean() >= THRESHOLD_17,
                f"avec la baseline, tes 50 derniers épisodes dépassent {THRESHOLD_17:g} (la pire des 6 graines de la "
                f"référence fait «WORST_17»).",
                f"avec la baseline, attendu {N_17} épisodes dont les 50 derniers dépassent {THRESHOLD_17:g} en moyenne, "
                f"reçu {len(mean_17)} épisodes et {fr(float(mean_17[-50:].mean()), 1) if len(mean_17) else '?'} ; "
                + ("ta courbe monte puis retombe : si elle finit vers 9 (une politique devenue déterministe, qui pousse "
                   "toujours du même côté), relance avec une autre SEED_17 avant de chercher une erreur : REINFORCE "
                   "s'effondre parfois." if rose_17 else
                   "ta courbe ne décolle jamais (sa moyenne glissante reste sous 50) : tes actions sont-elles tirées "
                   "dans la politique (Categorical(...).sample()), et la loss a-t-elle le signe de 208.3 ?"))'''

# --- 🔨 B8.18 ---------------------------------------------------------------------------------------------------------
CHECK_18 = r'''with checked("B8.18"):
    rng_18 = np.random.default_rng(18)                  # the expected values first, then your library, watched
    cols_18 = [rng_18.normal(size=(9, 3)) for _ in range(3)] + [rng_18.random((9, 3)) < 0.2,
                                                                 rng_18.random((9, 3)) < 0.2]
    one_18 = np.stack([wbr.gae_numpy(*[c[:, j] for c in cols_18], 0.95, 0.9)[0] for j in range(3)], axis=1)
    with own_code("B8.18", "ta librairie deep_rl.py", REFERENCE_CODE):
        r_18, v_18 = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        out_18 = mylearn.deep_rl.compute_gae(r_18, v_18, v_18, [False, False, True], [False, False, False], gamma=0.5,
                                             lam=1.0)
        adv_18, ret_18 = pair("B8.18", "compute_gae", out_18, names=("advantages", "returns"))
        same("B8.18", "exemple 1 de la docstring (avantages)", adv_18, [1.75, 1.5, 1.0])
        adv_18, _ = pair("B8.18", "compute_gae", mylearn.deep_rl.compute_gae(
            r_18, [0.5, 0.5, 0.5], [0.5, 0.5, 2.0], [False, False, False], [False, True, False], gamma=0.9, lam=0.5),
            names=("advantages", "returns"))
        same("B8.18", "exemple 2 de la docstring (troncature au pas 1)", adv_18, [1.3775, 0.95, 2.3])
        v_fiche_18 = [2.0, 1.5, 0.8]
        adv_18, ret_18 = pair("B8.18", "compute_gae", mylearn.deep_rl.compute_gae(
            r_18, v_fiche_18, [1.5, 0.8, 0.0], [False, False, True], [False, False, False], gamma=0.9, lam=0.5),
            names=("advantages", "returns"))
        same("B8.18", "le tableau de 208.4 (avantages)", adv_18, [0.4895, 0.31, 0.2])
        same("B8.18", "le tableau de 208.4 (cibles du critique)", ret_18, [2.4895, 1.81, 1.0])
        passed_18 = [c.copy() for c in cols_18]
        adv2_18, _ = pair("B8.18", "compute_gae", mylearn.deep_rl.compute_gae(*passed_18, 0.95, 0.9),
                          names=("advantages", "returns"))
        verdict("B8.18", all(np.array_equal(a, b) for a, b in zip(passed_18, cols_18)), "compute_gae laisse ses entrées intactes.",
                "compute_gae a modifié un de ses tableaux d'entrée : travaille sur des copies.")
        same("B8.18", "un rollout (9, 3) : chaque colonne comme un rollout à elle seule", adv2_18, one_18, rtol=1e-6, atol=1e-6)
        ev_18 = mylearn.deep_rl.explained_variance([1.5, 1.5, 3.5, 3.5], [1.0, 2.0, 3.0, 4.0])
        if isinstance(ev_18, (int, float, np.floating)) and abs(float(ev_18) - 0.75) < 1e-9:     # 1 - Var(y - p) / Var(p)
            print("❌ Ex B8.18 : variance expliquée de 208.4 : attendu 0.8, reçu 0.75, la variance des prédictions au "
                  "dénominateur : l'ordre est (y_pred, y_true), l'inverse de scikit-learn.")
        else:
            same("B8.18", "variance expliquée de 208.4 (prédictions d'abord)", ev_18, 0.8)
        same("B8.18", "une prédiction décalée de 5 partout", mylearn.deep_rl.explained_variance([6.0, 7.0, 8.0, 9.0],
                                                                                             [1.0, 2.0, 3.0, 4.0]), 1.0)
        same("B8.18", "des retours tous égaux", mylearn.deep_rl.explained_variance([1.0, 2.0], [3.0, 3.0]), np.nan)
        b84_18 = mylearn.deep_rl.compute_gae([1.0] * 4, [4.0, 3.8, 2.0, 2.4], [3.8, 3.6, 2.4, 2.6], [False] * 4,
                                             [False, True, False, False], gamma=0.9, lam=0.95)[0]
        print("le rollout de ✏️ B8.4 (λ = 0,95) avec ta fonction :", np.round(np.asarray(b84_18, dtype=float), 4).tolist(),
              "← compare à tes réponses de B8.4 d) et e)")'''

# --- 🔨 B8.19 ---------------------------------------------------------------------------------------------------------
CHECK_19 = r'''with checked("B8.19"):
    loss_19, grad_19 = pair("B8.19", "clipped_surrogate", mylearn.deep_rl.clipped_surrogate([0.7, 1.1, 1.35],
                                                                                           [-1.0, 2.0, -1.0], 0.2),
                            names=("loss", "gradient"), floats=("loss",))
    same("B8.19", "l'exemple de 208.5 (loss)", loss_19, -(-0.8 + 2.2 - 1.35) / 3)
    same("B8.19", "l'exemple de 208.5 (gradient)", grad_19, [0.0, -2 / 3, 1 / 3])
    rng_19 = np.random.default_rng(19)
    for eps_19 in (0.2, 0.5, 1.5):
        ratio_19 = np.concatenate([rng_19.uniform(0.0, 2.5, 6), [1 - eps_19 if eps_19 < 1 else 0.0, 1 + eps_19, 1.0]])
        adv_19 = np.concatenate([rng_19.normal(size=6), [1.5, -1.5, 0.0]])
        r_t = torch.tensor(ratio_19, requires_grad=True)
        a_t = torch.tensor(adv_19)
        want_19 = -torch.min(r_t * a_t, r_t.clamp(1 - eps_19, 1 + eps_19) * a_t).mean()
        want_19.backward()
        loss_19, grad_19 = pair("B8.19", "clipped_surrogate",
                                mylearn.deep_rl.clipped_surrogate(ratio_19.copy(), adv_19.copy(), eps_19),
                                names=("loss", "gradient"), floats=("loss",))
        same("B8.19", f"9 ratios (deux sur un bord exact), ε = {fr(eps_19, 1)} : la loss contre l'autograd", loss_19,
             want_19.item(), rtol=1e-9, atol=1e-12)
        same("B8.19", f"9 ratios, ε = {fr(eps_19, 1)} : le gradient contre l'autograd", grad_19, r_t.grad.numpy(),
             rtol=1e-9, atol=1e-12)'''

# --- 📦 B8.20 ---------------------------------------------------------------------------------------------------------
GIVEN_20 = r'''SETTINGS_20 = dict(wbr.PPO_SETTINGS)      # the settings of the reference PPO (printed below)
STEPS_20 = wbr.ppo_steps(FAST_MODE)         # 50 000 environment steps (all copies) in fast mode, 100 000 in full mode
SEED_20 = 20
THRESHOLD_20 = «THRESHOLD_20»               # the deterministic evaluation that the check asks (6 seeds of the reference)
AGENT_KEY_20 = wbr.agent_key(SETTINGS_20, source=SOURCE, fast=FAST_MODE, seed=SEED_20, total_steps=STEPS_20)


def load_agent_20(verbose=True):
    """(agent, source) for the rest of the chapter: the agent trained by Ex B8.20 in this session (AGENT_20), else the
    one an earlier session saved under AGENT_KEY_20, else the reference PPO agent of wbr (an explicit fallback, said).
    The solutions notebook never reads the learner's agent: its key has the source "solution"."""
    def weak(score):
        if verbose and score is not None and score < THRESHOLD_20:
            print(f"⚠️ ton agent de B8.20 ne tient que {score:.1f} pas en évaluation (seuil {THRESHOLD_20:g}) : la suite "
                  "s'en sert quand même ; refais B8.20, ou supprime-le (wbr.delete_agent(AGENT_KEY_20)) pour prendre "
                  "l'agent de référence.")

    if globals().get("AGENT_20") is not None:
        weak(globals().get("eval_20", {}).get("mean_return"))
        return AGENT_20, SOURCE
    try:
        loaded = wbr.load_agent(AGENT_KEY_20, build=ActorCritic20)
    except NotImplementedError:
        loaded = None
    if loaded is not None:
        if verbose:
            print(f"ℹ️ agent de B8.20 relu ({loaded[1]['path'].name}, enregistré le {loaded[1]['saved_at']})")
        weak(loaded[1]["extra"].get("eval"))
        return loaded[0], SOURCE
    agent, _ = wbr.reference_agent(FAST_MODE, verbose=verbose)
    if verbose:
        print("ℹ️ pas d'agent de B8.20 enregistré : repli sur l'agent PPO de référence (wbr.reference_agent)")
    return agent, "reference"


print(SETTINGS_20)'''

TODO_20 = r'''class ActorCritic20(nn.Module):
    """The agent: an actor (4 -> hidden -> hidden -> 2 logits, tanh) and a critic (4 -> hidden -> hidden -> 1, tanh),
    two separate networks in the attributes `actor` and `critic`. forward(obs) returns the logits (N, 2);
    value(obs) returns V(obs), of shape (N,)."""

    def __init__(self, hidden=64):
        super().__init__()
        raise NotImplementedError("ActorCritic20.__init__")

    def forward(self, obs):
        raise NotImplementedError("ActorCritic20.forward")

    def value(self, obs):
        raise NotImplementedError("ActorCritic20.value")


def rollout_20(agent, envs, obs, just_reset, n_steps, generator):
    """One rollout of n_steps steps in each copy of envs (the contract of the statement). Returns (batch, obs,
    just_reset): batch a dict of NumPy arrays of shape (n_steps, N, ...) "obs", "actions", "log_probs", "values",
    "rewards", "terminated", "truncated", "next_obs", "next_values", "valid"; obs and just_reset for the next rollout."""
    raise NotImplementedError("rollout_20")


def ppo_loss_20(agent, obs, actions, old_log_probs, advantages, returns, settings=SETTINGS_20):
    """The PPO loss of a mini-batch of tensors (the contract of the statement): (loss, parts), loss a tensor with its
    graph, parts a dict of floats "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"."""
    raise NotImplementedError("ppo_loss_20")


def train_ppo_20(seed, total_steps, settings=SETTINGS_20, reward_fn=None):
    """PPO on settings["n_envs"] copies of CartPole-v1 (the contract of the statement). Returns (agent, logs), logs a
    dict of lists (one value per iteration) with the keys of wbr.LOG_KEYS and "surrogate_gap"."""
    raise NotImplementedError("train_ppo_20")'''

SOLUTION_20 = r'''def _layer_20(n_in, n_out, gain):
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)       # the orthogonal initialisation of CleanRL (208.5, "37 détails")
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic20(nn.Module):
    """The agent: an actor (4 -> hidden -> hidden -> 2 logits, tanh) and a critic (4 -> hidden -> hidden -> 1, tanh),
    two separate networks in the attributes `actor` and `critic`. forward(obs) returns the logits (N, 2);
    value(obs) returns V(obs), of shape (N,)."""

    def __init__(self, hidden=64):
        super().__init__()
        g = math.sqrt(2)
        self.actor = nn.Sequential(_layer_20(4, hidden, g), nn.Tanh(), _layer_20(hidden, hidden, g), nn.Tanh(),
                                   _layer_20(hidden, 2, 0.01))
        self.critic = nn.Sequential(_layer_20(4, hidden, g), nn.Tanh(), _layer_20(hidden, hidden, g), nn.Tanh(),
                                    _layer_20(hidden, 1, 1.0))

    def forward(self, obs):
        return self.actor(obs)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def rollout_20(agent, envs, obs, just_reset, n_steps, generator):
    """One rollout of n_steps steps in each copy of envs (the contract of the statement). Returns (batch, obs,
    just_reset): batch a dict of NumPy arrays of shape (n_steps, N, ...) "obs", "actions", "log_probs", "values",
    "rewards", "terminated", "truncated", "next_obs", "next_values", "valid"; obs and just_reset for the next rollout."""
    keys = ("obs", "actions", "log_probs", "values", "rewards", "terminated", "truncated", "next_obs", "valid")
    rows = {k: [] for k in keys}
    for _ in range(n_steps):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            log_pi = torch.log_softmax(agent(x), dim=1)
            actions = torch.multinomial(log_pi.exp(), 1, generator=generator).squeeze(1)
            values = agent.value(x)
        next_obs, rewards, terminated, truncated, _ = envs.step(actions.numpy())
        step = (obs, actions.numpy(), log_pi.gather(1, actions[:, None]).squeeze(1).numpy(), values.numpy(), rewards,
                terminated, truncated, next_obs, ~just_reset)          # valid: False when the step only resets a copy
        for k, v in zip(keys, step):
            rows[k].append(np.array(v, copy=True))
        just_reset = terminated | truncated
        obs = next_obs
    batch = {k: np.stack(v) for k, v in rows.items()}
    with torch.no_grad():                         # V of the observation that FOLLOWS each step: at a time limit, the
        flat = torch.as_tensor(batch["next_obs"].reshape(-1, 4), dtype=torch.float32)   # real last observation
        batch["next_values"] = agent.value(flat).numpy().reshape(n_steps, -1)
    return batch, obs, just_reset


def ppo_loss_20(agent, obs, actions, old_log_probs, advantages, returns, settings=SETTINGS_20):
    """The PPO loss of a mini-batch of tensors (the contract of the statement): (loss, parts), loss a tensor with its
    graph, parts a dict of floats "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"."""
    eps = settings["clip_eps"]
    log_pi = torch.log_softmax(agent(obs), dim=1)
    log_ratio = log_pi.gather(1, actions.long()[:, None]).squeeze(1) - old_log_probs
    ratio = log_ratio.exp()
    policy_loss = -torch.min(ratio * advantages, ratio.clamp(1 - eps, 1 + eps) * advantages).mean()
    value_loss = 0.5 * ((agent.value(obs) - returns) ** 2).mean()
    entropy = -(log_pi.exp() * log_pi).sum(dim=1).mean()
    loss = policy_loss + settings["vf_coef"] * value_loss - settings["ent_coef"] * entropy
    with torch.no_grad():
        parts = {"policy_loss": float(policy_loss), "value_loss": float(value_loss), "entropy": float(entropy),
                 "approx_kl": float(((ratio - 1) - log_ratio).mean()),
                 "clipfrac": float(((ratio - 1).abs() > eps).float().mean())}
    return loss, parts


def train_ppo_20(seed, total_steps, settings=SETTINGS_20, reward_fn=None):
    """PPO on settings["n_envs"] copies of CartPole-v1 (the contract of the statement). Returns (agent, logs), logs a
    dict of lists (one value per iteration) with the keys of wbr.LOG_KEYS and "surrogate_gap"."""
    s = settings
    torch.manual_seed(seed)
    agent = ActorCritic20(s["hidden"])
    optimizer = torch.optim.Adam(agent.parameters(), lr=s["lr"], eps=s["adam_eps"])
    generator = torch.Generator().manual_seed(seed)
    envs = gym.make_vec("CartPole-v1", num_envs=s["n_envs"], vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP})
    obs, _ = envs.reset(seed=seed)
    just_reset = np.zeros(s["n_envs"], dtype=bool)
    running = np.zeros(s["n_envs"])                      # the true return of the running episode of each copy
    n_iter = total_steps // (s["n_envs"] * s["n_steps"])
    logs = {k: [] for k in (*wbr.LOG_KEYS, "surrogate_gap")}
    for it in range(n_iter):
        lr = s["lr"] * (1 - it / n_iter) if s["anneal_lr"] else s["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch, obs, just_reset = rollout_20(agent, envs, obs, just_reset, s["n_steps"], generator)
        valid = batch["valid"]
        finished = []
        for t in range(s["n_steps"]):                    # the episodes that end in this rollout, with their true returns
            running += np.where(valid[t], batch["rewards"][t], 0.0)
            for i in np.flatnonzero(valid[t] & (batch["terminated"][t] | batch["truncated"][t])):
                finished.append(running[i])
                running[i] = 0.0
        if reward_fn is not None:                        # RLHF: the agent learns from another reward
            batch["rewards"] = np.stack([reward_fn(batch["obs"][t], batch["actions"][t], batch["next_obs"][t])
                                         for t in range(s["n_steps"])]).astype(float)
        adv, ret = mylearn.deep_rl.compute_gae(batch["rewards"], batch["values"], batch["next_values"],
                                               batch["terminated"], batch["truncated"], s["gamma"], s["gae_lambda"])
        ev = mylearn.deep_rl.explained_variance(batch["values"][valid], ret[valid])
        flat = {k: torch.as_tensor(batch[k][valid]) for k in ("obs", "actions", "log_probs")}
        adv_all = torch.as_tensor(adv[valid], dtype=torch.float32)
        ret_all = torch.as_tensor(ret[valid], dtype=torch.float32)
        n = len(adv_all)
        size = n // s["n_minibatches"]
        parts_all, gap = [], 0.0
        for _ in range(s["n_epochs"]):
            order = torch.randperm(n, generator=generator)
            for k in range(s["n_minibatches"]):
                idx = order[k * size:(k + 1) * size]
                a = adv_all[idx]
                if s["norm_adv"]:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                loss, parts = ppo_loss_20(agent, flat["obs"][idx], flat["actions"][idx], flat["log_probs"][idx], a,
                                          ret_all[idx], s)
                with torch.no_grad():                    # your clipped_surrogate (NumPy) against the torch loss
                    ratio = (torch.log_softmax(agent(flat["obs"][idx]), dim=1)
                             .gather(1, flat["actions"][idx][:, None]).squeeze(1) - flat["log_probs"][idx]).exp()
                mine, _ = mylearn.deep_rl.clipped_surrogate(ratio.double().numpy(), a.double().numpy(), s["clip_eps"])
                gap = max(gap, abs(float(mine) - parts["policy_loss"]))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), s["max_grad_norm"])
                optimizer.step()
                parts_all.append(parts)
        row = {"step": (it + 1) * s["n_envs"] * s["n_steps"], "iteration": it + 1, "lr": lr,
               **{k: float(np.mean([p[k] for p in parts_all])) for k in parts_all[0]},
               "explained_variance": float(ev), "episodes": len(finished),
               "mean_return": float(np.mean(finished)) if finished else float("nan"), "surrogate_gap": gap}
        for k in logs:
            logs[k].append(row[k])
    envs.close()
    return agent, logs'''

CHECK_20 = r'''ok_20 = False
with checked("B8.20"):
    torch.manual_seed(0)
    agent_20 = ActorCritic20()
    obs_20 = torch.as_tensor(np.random.default_rng(20).normal(0, 0.1, (5, 4)), dtype=torch.float32)
    logits_20, value_20 = agent_20(obs_20), agent_20.value(obs_20)
    has_parts_20 = isinstance(getattr(agent_20, "actor", None), nn.Module) and isinstance(getattr(agent_20, "critic", None),
                                                                                         nn.Module)
    verdict("B8.20", tuple(logits_20.shape) == (5, 2) and tuple(value_20.shape) == (5,) and has_parts_20,
            f"ActorCritic20 : acteur et critique séparés, {sum(p.numel() for p in agent_20.parameters())} paramètres ; "
            "forward rend (5, 2), value rend (5,).",
            f"ActorCritic20 : attendu des attributs actor et critic, forward de forme (5, 2) et value de forme (5,), reçu "
            f"{tuple(logits_20.shape)} et {tuple(value_20.shape)}" + ("" if has_parts_20 else ", sans actor ou critic") + ".")

rollout_ok_20 = False
with checked("B8.20"):
    torch.manual_seed(20)
    ref_agent_20 = wbr.ActorCritic(64)                    # a fixed agent of the reference, so that only rollout_20 is tested
    envs_20 = wbr.make_vec(2, seed=5, max_episode_steps=12)
    gen_20 = torch.Generator().manual_seed(20)
    with own_code("B8.20", "rollout_20", REFERENCE_CODE):
        out_a_20 = rollout_20(ref_agent_20, envs_20, envs_20.first_obs.copy(), np.zeros(2, bool), 10, gen_20)
    batch_a_20, obs_a_20, reset_a_20 = pair("B8.20", "rollout_20", out_a_20, names=("batch", "obs", "just_reset"))
    with own_code("B8.20", "rollout_20", REFERENCE_CODE):
        out_b_20 = rollout_20(ref_agent_20, envs_20, obs_a_20, reset_a_20, 10, gen_20)
    batch_b_20, _, _ = pair("B8.20", "rollout_20", out_b_20, names=("batch", "obs", "just_reset"))
    keys_20 = ("obs", "actions", "log_probs", "values", "rewards", "terminated", "truncated", "next_obs", "next_values",
               "valid")
    missing_20 = [k for k in keys_20 if not isinstance(batch_a_20, dict) or k not in batch_a_20]
    if missing_20:
        print(f"❌ Ex B8.20 : le batch de rollout_20 doit avoir les clés {list(keys_20)} ; il manque {missing_20}.")
        raise StopCheck
    shapes_20 = {k: np.shape(batch_a_20[k]) for k in keys_20}
    want_shapes_20 = {k: ((10, 2, 4) if k in ("obs", "next_obs") else (10, 2)) for k in keys_20}
    if shapes_20 != want_shapes_20:
        print(f"❌ Ex B8.20 : formes du batch : attendu {want_shapes_20}, reçu {shapes_20}.")
        raise StopCheck
    replay_20 = wbr.make_vec(2, seed=5, max_episode_steps=12)          # the same copies, played with YOUR actions
    actions_20 = np.concatenate([batch_a_20["actions"], batch_b_20["actions"]]).astype(np.int64)
    expect_20 = {k: [] for k in ("obs", "rewards", "terminated", "truncated", "next_obs", "valid")}
    o_20, jr_20 = replay_20.first_obs, np.zeros(2, bool)
    for t_20 in range(20):
        n_20, r_20, te_20, tr_20, _ = replay_20.step(actions_20[t_20])
        for k_20, v_20 in zip(expect_20, (o_20, r_20, te_20, tr_20, n_20, ~jr_20)):
            expect_20[k_20].append(np.array(v_20, copy=True))
        jr_20, o_20 = te_20 | tr_20, n_20
    both_20 = {k: np.concatenate([np.asarray(batch_a_20[k]), np.asarray(batch_b_20[k])]) for k in keys_20}
    problems_20 = []
    for k_20, v_20 in expect_20.items():
        if not np.allclose(both_20[k_20].astype(float), np.stack(v_20).astype(float)):
            problems_20.append(k_20)
    with torch.no_grad():
        x_20 = torch.as_tensor(both_20["obs"].reshape(-1, 4), dtype=torch.float32)
        lp_want_20 = torch.log_softmax(ref_agent_20(x_20), 1).gather(
            1, torch.as_tensor(actions_20.reshape(-1, 1))).squeeze(1).numpy().reshape(20, 2)
        v_want_20 = ref_agent_20.value(x_20).numpy().reshape(20, 2)
        nv_want_20 = ref_agent_20.value(torch.as_tensor(both_20["next_obs"].reshape(-1, 4), dtype=torch.float32)
                                        ).numpy().reshape(20, 2)
        greedy_20 = ref_agent_20(x_20).argmax(1).numpy().reshape(20, 2)      # p close to 0.5: 40 argmax have p ≈ 2^-40
    if np.array_equal(actions_20, greedy_20):
        problems_20.append("actions")
    for k_20, want_20 in (("log_probs", lp_want_20), ("values", v_want_20), ("next_values", nv_want_20)):
        if not np.allclose(both_20[k_20].astype(float), want_20, atol=1e-5):
            problems_20.append(k_20)
    ends_20 = int((np.stack(expect_20["terminated"]) | np.stack(expect_20["truncated"])).sum())
    if not problems_20:
        rollout_ok_20 = True
        print(f"✅ Ex B8.20 : deux rollouts de 10 pas sur 2 copies coupées à 12 pas ({ends_20} fins d'épisode) : tout "
              "correspond à l'environnement rejoué avec tes actions ; les pas de reset sont écartés (valid), "
              "next_values est V de l'observation qui suit chaque pas, et le second rollout reprend où le premier s'arrête.")
    else:
        hints_20 = {"valid": "valid doit valoir False exactement au pas qui suit une fin d'épisode dans la même copie "
                             "(ce pas ne fait que le reset, 208.5), y compris d'un rollout au suivant (just_reset)",
                    "next_values": "next_values[t] est V de l'observation que rend le pas t (la vraie dernière quand "
                                   "l'épisode finit), pas la valeur du pas suivant du rollout",
                    "obs": "obs[t] est l'observation AVANT le pas t ; le second rollout repart de l'observation rendue "
                           "par le premier",
                    "log_probs": "log_probs est le log de la probabilité de l'action jouée, sous la politique de l'agent",
                    "values": "values est V(obs[t]) du critique de l'agent",
                    "actions": "les 40 actions sont toutes celles de plus grande probabilité : elles doivent être "
                               "tirées dans la politique (torch.multinomial ou Categorical, avec generator)"}
        for k_20 in problems_20:
            print(f"❌ Ex B8.20 : rollout_20, « {k_20} » ne correspond pas à l'environnement rejoué avec tes actions"
                  + (f" : {hints_20[k_20]}." if k_20 in hints_20 else "."))
        if problems_20 == ["actions"]:
            print("❌ Ex B8.20 : rollout_20 joue l'action la plus probable : l'évaluation est déterministe, pas le "
                  "rollout, dont les données doivent suivre la politique (sinon les log-probabilités et les ratios de "
                  "PPO ne décrivent pas ce qui a été joué).")

loss_ok_20 = False
with checked("B8.20"):
    rng_20 = np.random.default_rng(200)
    mb_20 = {"obs": torch.as_tensor(rng_20.normal(0, 0.5, (16, 4)), dtype=torch.float32),
             "actions": torch.as_tensor(rng_20.integers(0, 2, 16)),
             "advantages": torch.as_tensor(rng_20.normal(size=16), dtype=torch.float32),
             "returns": torch.as_tensor(rng_20.normal(5, 2, 16), dtype=torch.float32)}
    with torch.no_grad():                                # old log-probabilities a little away from the current ones
        mb_20["old_log_probs"] = torch.log_softmax(ref_agent_20(mb_20["obs"]), 1).gather(
            1, mb_20["actions"][:, None]).squeeze(1) + torch.as_tensor(rng_20.normal(0, 0.3, 16), dtype=torch.float32)
    args_20 = (mb_20["obs"], mb_20["actions"], mb_20["old_log_probs"], mb_20["advantages"], mb_20["returns"])
    with own_code("B8.20", "ppo_loss_20", REFERENCE_CODE):
        out_loss_20 = ppo_loss_20(ref_agent_20, *args_20, SETTINGS_20)
    loss_20, parts_20 = pair("B8.20", "ppo_loss_20", out_loss_20, names=("loss", "parts"))
    want_loss_20, want_parts_20 = wbr.ppo_loss(ref_agent_20, *args_20, SETTINGS_20)
    with torch.no_grad():
        log_ratio_20 = torch.log_softmax(ref_agent_20(mb_20["obs"]), 1).gather(
            1, mb_20["actions"][:, None]).squeeze(1) - mb_20["old_log_probs"]
    good_20 = [same("B8.20", "ppo_loss_20 sur un mini-batch de 16 (la loss)", loss_20.detach(), want_loss_20.detach(),
                    rtol=1e-5, atol=1e-6)]
    wrong_20 = {}
    for k_20, v_20 in want_parts_20.items():
        got_20 = parts_20.get(k_20) if isinstance(parts_20, dict) else None
        good_20.append(got_20 is not None and abs(float(got_20) - v_20) <= 1e-5 * max(1.0, abs(v_20)))
        if not good_20[-1]:
            wrong_20[k_20] = got_20
    for k_20, got_20 in wrong_20.items():
        hint_20 = ""
        if (k_20 == "approx_kl" and isinstance(got_20, (int, float, np.floating))
                and abs(float(got_20) + float(log_ratio_20.mean())) < 1e-5):
            hint_20 = (" : c'est l'estimateur naïf −mean(ln ρ), qui peut être négatif ; le contrat demande la moyenne de "
                       "(ρ − 1) − ln ρ, jamais négative (208.5, « Les journaux de PPO »)")
        elif k_20 in ("approx_kl", "clipfrac") and got_20 == 0 and {"approx_kl", "clipfrac"} <= set(wrong_20):
            hint_20 = (" : ton ratio vaut 1 partout ; ρ compare la politique courante aux log-probabilités du rollout "
                       "(old_log_probs), pas à elle-même détachée")
        elif k_20 == "policy_loss" and not ({"value_loss", "entropy", "approx_kl", "clipfrac"} & set(wrong_20)):
            hint_20 = " : relis L^π, le min de ρ Â et de clip(ρ, 1 − ε, 1 + ε) Â : ε borne le ratio, pas l'avantage"
        print(f"❌ Ex B8.20 : ppo_loss_20, parts[{k_20!r}] : attendu {want_parts_20[k_20]:.6g}, reçu {got_20!r}{hint_20}.")
    if not good_20[0] and all(good_20[1:]):
        print("❌ Ex B8.20 : les cinq parties sont justes, mais la loss ne les combine pas comme le contrat : "
              "policy_loss + vf_coef × value_loss − ent_coef × entropy (signes et coefficients de SETTINGS_20).")
    ref_agent_20.zero_grad()
    loss_20.backward()
    reached_20 = {name: any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.parameters())
                  for name, m in (("acteur", ref_agent_20.actor), ("critique", ref_agent_20.critic))}
    good_20.append(all(reached_20.values()))
    verdict("B8.20", good_20[-1], "le gradient de la loss atteint l'acteur et le critique.",
            f"le gradient de la loss n'atteint pas {', '.join(n for n, ok in reached_20.items() if not ok)} : un .detach() "
            "ou un .item() de trop ?")
    loss_ok_20 = all(good_20)

with checked("B8.20"):
    if not (rollout_ok_20 and loss_ok_20):
        print("⏳ Ex B8.20 : la boucle se vérifie une fois rollout_20 et ppo_loss_20 justes.")
    elif not (written(lambda: mylearn.deep_rl.compute_gae([1.0], [0.0], [0.0], [True], [False]))
              and written(lambda: mylearn.deep_rl.clipped_surrogate([1.0], [1.0]))):
        print("⏳ Ex B8.20 : ta boucle se sert de compute_gae et de clipped_surrogate de ta librairie (B8.18, B8.19), pas "
              "encore écrites.")
    else:
        seen_20 = {"gae": [], "surrogate": 0, "rollouts": [], "minibatches": []}
        real_gae_20, real_clip_20 = mylearn.deep_rl.compute_gae, mylearn.deep_rl.clipped_surrogate
        real_rollout_20, real_loss_20, real_make_vec_20 = rollout_20, ppo_loss_20, gym.make_vec

        def spy_rollout_20(*args, **kwargs):                # each rollout opens an iteration
            out = real_rollout_20(*args, **kwargs)
            seen_20["rollouts"].append(out[0] if isinstance(out, (tuple, list)) and out else {})
            seen_20["minibatches"].append([])
            return out

        def spy_loss_20(*args, **kwargs):                   # the size and the advantages of each mini-batch
            adv = torch.as_tensor(inspect.signature(real_loss_20).bind(*args, **kwargs).arguments["advantages"])
            adv = adv.detach().double().reshape(-1)
            if seen_20["minibatches"]:
                seen_20["minibatches"][-1].append((len(adv), float(adv.mean()) if len(adv) else 0.0,
                                                   float(adv.std(unbiased=False)) if len(adv) > 1 else 1.0))
            return real_loss_20(*args, **kwargs)

        def spy_gae_20(*args, **kwargs):
            out = real_gae_20(*args, **kwargs)
            bound = inspect.signature(real_gae_20).bind(*args, **kwargs).arguments
            seen_20["gae"].append((tuple(np.shape(bound.get(k)) for k in ("rewards", "values", "next_values",
                                                                           "terminated", "truncated")), args, kwargs))
            return out

        def spy_clip_20(*args, **kwargs):
            seen_20["surrogate"] += 1
            return real_clip_20(*args, **kwargs)

        mylearn.deep_rl.compute_gae, mylearn.deep_rl.clipped_surrogate = spy_gae_20, spy_clip_20
        rollout_20, ppo_loss_20 = spy_rollout_20, spy_loss_20
        gym.make_vec = lambda *a, **k: real_make_vec_20(*a, **{**k, "max_episode_steps": 20})   # time limits, here only
        try:
            short_20 = {**SETTINGS_20, "n_epochs": 2}
            per_iter_20 = short_20["n_envs"] * short_20["n_steps"]
            with own_code("B8.20", "train_ppo_20", REFERENCE_CODE):
                out_s_20 = train_ppo_20(3, 2 * per_iter_20, short_20)
        finally:
            mylearn.deep_rl.compute_gae, mylearn.deep_rl.clipped_surrogate = real_gae_20, real_clip_20
            rollout_20, ppo_loss_20, gym.make_vec = real_rollout_20, real_loss_20, real_make_vec_20
        agent_s_20, logs_s_20 = pair("B8.20", "train_ppo_20", out_s_20, names=("agent", "logs"))
        want_keys_20 = set(wbr.LOG_KEYS) | {"surrogate_gap"}
        good_20 = []
        good_20.append(isinstance(logs_s_20, dict) and want_keys_20 <= set(logs_s_20)
                       and all(len(logs_s_20[k]) == 2 for k in want_keys_20))
        verdict("B8.20", good_20[-1], "les journaux : une valeur par itération pour chaque clé de wbr.LOG_KEYS et surrogate_gap.",
                f"logs doit avoir les clés {sorted(want_keys_20)}, une valeur par itération (2 ici), reçu "
                + (str({k: len(v) for k, v in logs_s_20.items()}) if isinstance(logs_s_20, dict) else type(logs_s_20).__name__) + ".")
        shape_20 = (short_20["n_steps"], short_20["n_envs"])
        good_20.append(len(seen_20["gae"]) == 2 and all(s == (shape_20,) * 5 for s, _, _ in seen_20["gae"]))
        verdict("B8.20", good_20[-1], f"compute_gae de ta librairie : appelée une fois par itération sur des tableaux {shape_20}.",
                f"attendu un appel de mylearn.deep_rl.compute_gae par itération (2) sur des tableaux {shape_20} (le temps "
                f"d'abord), reçu {len(seen_20['gae'])} appels" + (f" de formes {seen_20['gae'][0][0]}" if seen_20["gae"] else "")
                + " ; l'appelles-tu par ce nom (la vérification l'observe) ?")
        if good_20[-1] and len(seen_20["rollouts"]) == 2:
            gae_args_20 = [inspect.signature(real_gae_20).bind(*a, **k).arguments for _, a, k in seen_20["gae"]]
            arrays_20 = ("rewards", "values", "next_values", "terminated", "truncated")
            differ_20 = sorted({k for b, r in zip(gae_args_20, seen_20["rollouts"]) for k in arrays_20
                                if not np.allclose(np.asarray(b[k], dtype=float), np.asarray(r[k], dtype=float),
                                                   atol=1e-6)})
            flag_20 = lambda r, k: np.asarray(r[k]).astype(bool)
            cut_20 = sum(int(np.sum(flag_20(r, "truncated") & ~flag_20(r, "terminated"))) for r in seen_20["rollouts"])
            good_20.append(not differ_20)
            if not differ_20:
                print(f"✅ Ex B8.20 : compute_gae reçoit les tableaux du rollout de son itération ({cut_20} limites de "
                      "temps, ici à 20 pas, gardées distinctes des fins).")
            elif "terminated" in differ_20 and all(np.array_equal(flag_20(b, "terminated"), flag_20(r, "terminated")
                                                                  | flag_20(r, "truncated"))
                                                   for b, r in zip(gae_args_20, seen_20["rollouts"])):
                print(f"❌ Ex B8.20 : compute_gae reçoit comme terminated les {cut_20} limites de temps du rollout (ici à "
                      "20 pas) : GAE n'amorce plus sur next_values à une limite de temps ; terminated et truncated vont "
                      "chacun à sa place (208.4, « Fin d'épisode ou limite de temps »).")
            else:
                print(f"❌ Ex B8.20 : compute_gae ne reçoit pas les tableaux {differ_20} du rollout de son itération : "
                      "GAE prend rewards, values, next_values, terminated et truncated tels que rollout_20 les rend.")
            rows_20 = [(sum(n for n, _, _ in mbs), int(np.sum(r["valid"])))
                       for mbs, r in zip(seen_20["minibatches"], seen_20["rollouts"])]
            epochs_20, n_split_20 = short_20["n_epochs"], short_20["n_minibatches"]
            good_20.append(all(epochs_20 * (v - v % n_split_20) <= n <= epochs_20 * v for n, v in rows_20))
            verdict("B8.20", good_20[-1],
                    f"la mise à jour passe {epochs_20} fois sur les transitions valides ({', '.join(str(v) for _, v in rows_20)} "
                    "par itération) : les pas de reset sont écartés.",
                    f"ppo_loss_20 voit {', '.join(str(n) for n, _ in rows_20)} transitions par itération, pour "
                    f"{', '.join(str(v) for _, v in rows_20)} transitions valides du rollout et {epochs_20} epochs : "
                    + ("les pas qui ne font que le reset (valid False) sont-ils écartés de la mise à jour ?"
                       if any(n > epochs_20 * v for n, v in rows_20) else
                       "chaque epoch passe-t-elle sur toutes les transitions valides ?"))
            normed_20 = [(m, s) for mbs in seen_20["minibatches"] for n, m, s in mbs if n > 1]
            good_20.append(bool(normed_20) and all(abs(m) < 1e-4 and abs(s - 1) < 0.02 for m, s in normed_20))
            verdict("B8.20", good_20[-1], f"les avantages de chacun des {len(normed_20)} mini-batches sont normalisés "
                    "(moyenne 0, écart-type 1).",
                    "les avantages que reçoit ppo_loss_20 ne sont pas centrés réduits dans chaque mini-batch (norm_adv) : "
                    f"moyennes de {fr(min(m for m, _ in normed_20), 3)} à {fr(max(m for m, _ in normed_20), 3)}, "
                    f"écarts-types de {fr(min(s for _, s in normed_20), 3)} à {fr(max(s for _, s in normed_20), 3)} ; "
                    "moins la moyenne du mini-batch, divisés par son écart-type (208.5)." if normed_20 else
                    "ppo_loss_20 n'est pas appelée par ce nom dans train_ppo_20 (la vérification l'observe).")
        n_mb_20 = 2 * short_20["n_epochs"] * short_20["n_minibatches"]
        good_20.append(seen_20["surrogate"] >= n_mb_20)
        verdict("B8.20", good_20[-1], f"clipped_surrogate de ta librairie : appelée à chaque mini-batch ({seen_20['surrogate']} fois).",
                f"attendu au moins {n_mb_20} appels de mylearn.deep_rl.clipped_surrogate (un par mini-batch), reçu "
                f"{seen_20['surrogate']}.")
        if good_20[0]:
            gap_20 = max(logs_s_20["surrogate_gap"])
            good_20.append(gap_20 <= 1e-5)
            verdict("B8.20", good_20[-1], f"ta clipped_surrogate et la loss torch coïncident (écart {gap_20:.1e}).",
                    f"ta clipped_surrogate et la loss torch du même mini-batch diffèrent de {gap_20:.2g} : leur passes-tu "
                    "les mêmes ratios et les mêmes avantages (normalisés) ?")
        ok_20 = all(good_20)'''

EXPERIMENT_20 = r'''# Training: your PPO, STEPS_20 steps on 4 copies, then the deterministic policy on 100 episodes; the agent is saved
if not ok_20:
    print("⏳ Ex B8.20 : l'entraînement attend que la vérification ci-dessus passe.")
else:
    with checked("B8.20"):
        AGENT_20 = None                                      # a new training replaces the agent of a former one
        wbr.delete_agent(AGENT_KEY_20)
        start_20 = time.perf_counter()
        with own_code("B8.20", "train_ppo_20", REFERENCE_CODE):
            trained_20, LOGS_20 = train_ppo_20(SEED_20, STEPS_20)
        seconds_20 = time.perf_counter() - start_20
        reference_20 = wbr.load_agent(wbr.reference_key(FAST_MODE))
        if not isinstance(trained_20, ActorCritic20) or (reference_20 is not None
                                                       and wbr.same_weights(trained_20, reference_20[0])):
            print("❌ Ex B8.20 : train_ppo_20 doit rendre l'agent qu'il vient d'entraîner, un ActorCritic20 neuf, reçu "
                  + ("l'agent PPO de référence" if isinstance(trained_20, nn.Module) and reference_20 is not None
                     and wbr.same_weights(trained_20, reference_20[0]) else type(trained_20).__name__) + " : rien n'est "
                  "enregistré.")
            raise StopCheck
        AGENT_20 = trained_20
        eval_20 = wbr.evaluate(wbr.greedy_logits(AGENT_20), 100, EVAL_SEED)
        steps_20, returns_20 = np.asarray(LOGS_20["step"]), np.asarray(LOGS_20["mean_return"], dtype=float)
        fig_20, ax_20 = plt.subplots(figsize=(7, 3))
        ax_20.plot(steps_20, returns_20, marker=".", label="retour moyen des épisodes finis de chaque rollout")
        ax_20.axhline(eval_20["mean_return"], color="k", ls="--", lw=1, label="politique déterministe, 100 épisodes")
        ax_20.set(xlabel="pas d'environnement (4 copies)", ylabel="retour", title=f"Ton PPO, graine {SEED_20}")
        ax_20.legend(fontsize=8)
        plt.show()
        print(f"{STEPS_20} pas en {seconds_20:.0f} s ({len(steps_20)} itérations, {int(np.sum(LOGS_20['episodes']))} "
              "épisodes finis) · politique déterministe : "
              f"{fr(eval_20['mean_return'], 1)} sur 100 épisodes (écart-type {fr(eval_20['std_return'], 1)}) · écart "
              f"maximal entre ta clipped_surrogate et la loss torch : {max(LOGS_20['surrogate_gap']):.1e}")
        finite_20 = returns_20[np.isfinite(returns_20)]          # an iteration where no episode ends has nan
        early_20, late_20 = float(finite_20[:5].mean()), float(finite_20[-5:].mean())
        verdict("B8.20", late_20 >= 2 * early_20,
                f"le retour monte : {fr(early_20, 1)} sur les 5 premières itérations, {fr(late_20, 1)} sur les 5 dernières.",
                f"le retour ne monte pas assez : {fr(early_20, 1)} sur les 5 premières itérations, {fr(late_20, 1)} sur "
                "les 5 dernières.")
        verdict("B8.20", eval_20["mean_return"] >= THRESHOLD_20,
                f"la politique déterministe dépasse {THRESHOLD_20:g} (la référence, sur 6 graines : «C20_EVALS»).",
                f"la politique déterministe tient {fr(eval_20['mean_return'], 1)} pas : moins que les {THRESHOLD_20:g} "
                f"attendus (la référence, sur 6 graines : «C20_EVALS»).")
        print(f"agent enregistré pour la suite du chapitre (source {SOURCE!r}) : "
              f"{wbr.save_agent(AGENT_KEY_20, AGENT_20, LOGS_20, extra={'eval': eval_20['mean_return']}).name}")'''
