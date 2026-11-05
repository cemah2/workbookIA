"""Modern RL on CartPole for bonus chapter B8: environments and seeds, the evaluation of a policy, the reference
agents (DQN and its ablations, REINFORCE, PPO with its logs), the single checkpoint (with its provenance) of a trained
PPO agent, and the precomputed multi-seed protocols of the notebook.

Quick reference
---------------
>>> env = make_env()                                   # gym.make("CartPole-v1"): 500 steps at most, threshold 475
>>> env = make_env(gravity=19.6, length=1.0)           # another physics (polemass_length and total_mass recomputed)
>>> envs = make_vec(4, seed=0)                         # gym.make_vec, 4 synchronous copies, next-step autoreset
>>> policy = greedy_q(q_net)                           # (state, info) -> int, the action of largest value
>>> policy = greedy_logits(actor)                      # the action of largest probability (deterministic evaluation)
>>> evaluate(policy, 100, seed=EVAL_SEED)              # mean_return, std_return, mean_length, returns (500 steps max)
>>> q_net, history = train_dqn(seed=0, variant="double")         # the reference DQN (Ex B8.13 to B8.15)
>>> policy_net, history = train_reinforce(seed=0, baseline="mean")   # the reference REINFORCE (Ex B8.16, B8.17)
>>> agent, logs = train_ppo(seed=0)                    # the reference PPO (Ex B8.20 on), with its logs (LOG_KEYS)
>>> agent, info = reference_agent()                    # the reference PPO agent of the mode, trained once, then read
>>> key = agent_key(settings, source="learner", fast=True); save_agent(key, agent, logs); load_agent(key)
>>> precomputed("a15")                                 # a protocol of tools/chapters/history_chB8.py

Conventions: observations are float32 arrays of shape (4,) (one copy) or (N, 4) (N copies); actions are Python ints
for one copy and int64 arrays for N copies; every trainer fixes its own seeds (``torch.manual_seed``, NumPy
generators, the first ``reset``) and runs PyTorch on one thread (``single_thread``), so that the same seed gives the
same run on the same machine. The evaluation draws the seed of each episode as ``mylearn.rl.evaluate_policy`` does
(``int(rng.integers(2**31))`` with ``rng = np.random.default_rng(seed)``): both give the same episodes.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

ENV_ID = "CartPole-v1"
EPISODE_STEPS = 500             # the time limit of CartPole-v1 (gym.make adds a TimeLimit)
SOLVED = 475.0                  # env.spec.reward_threshold: the mean of 100 consecutive episodes
EVAL_SEED = 8000                # the seeds of the evaluations of the notebook (never used for training)
SOURCES = ("learner", "solution", "reference")
CHECKPOINT_VERSION = 1
PHYSICS = ("gravity", "length", "masspole", "masscart", "force_mag")

# The tabular reference of chapter 26 (the solution of Ex 26.38): the angle in 4 intervals and its speed in 3, the
# position and the speed of the cart ignored, 12 states; Q learned by mylearn_ref.rl.train_tabular (800 episodes,
# alpha 0.2, gamma 0.99, epsilon from 1 to 0.01 over half of the training, 200 steps at most, seed 2638). The table
# is read here, not trained again (tools/chapters/history_chB8.py, entry "tab26", recomputes it and checks it).
TABULAR_EDGES = [[], [], [-0.05, 0.0, 0.05], [-0.5, 0.5]]
TABULAR_Q = None                # filled from the precomputed protocols (see tabular_reference)

LOG_KEYS = ("step", "iteration", "lr", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
            "explained_variance", "episodes", "mean_return")


def _torch():
    import torch

    return torch


# ---------------------------------------------------------------------------------------------------------------------
# Environments, physics and seeds
# ---------------------------------------------------------------------------------------------------------------------
def set_physics(env, **physics):
    """Change the physics of a CartPole (``env.unwrapped`` of a ``gym.make`` environment, or every copy of a
    synchronous vector environment) and recompute the derived constants ``total_mass = masspole + masscart`` and
    ``polemass_length = masspole * length`` (``CartPoleEnv.__init__`` computes them once: changing ``length`` alone
    would leave the old ``polemass_length`` in the equations). ``length`` is HALF the pole, as in gymnasium. Returns
    ``env``. Unknown names raise ``ValueError``."""
    unknown = sorted(set(physics) - set(PHYSICS))
    if unknown:
        raise ValueError(f"unknown physics {unknown}; allowed: {PHYSICS}")
    copies = [e.unwrapped for e in env.envs] if hasattr(env, "envs") else [env.unwrapped]
    for core in copies:
        for name, value in physics.items():
            if value is not None:
                if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                    raise ValueError(f"{name} must be a positive number, got {value!r}")
                setattr(core, name, float(value))
        core.total_mass = core.masspole + core.masscart
        core.polemass_length = core.masspole * core.length
    return env


def physics_of(env) -> dict:
    """The physics of a CartPole (one copy): the five settable constants and the two derived ones."""
    core = (env.envs[0] if hasattr(env, "envs") else env).unwrapped
    return {name: float(getattr(core, name)) for name in PHYSICS + ("total_mass", "polemass_length")}


def make_env(*, max_episode_steps: int | None = None, **physics):
    """``gym.make("CartPole-v1")`` (500 steps at most), optionally with another physics (``set_physics``) or another
    time limit."""
    kwargs = {} if max_episode_steps is None else {"max_episode_steps": int(max_episode_steps)}
    env = gym.make(ENV_ID, **kwargs)
    if physics:
        set_physics(env, **physics)
    return env


def make_vec(n_envs: int = 4, seed: int | None = None, *, autoreset: str = "next_step", physics: dict | None = None,
             max_episode_steps: int | None = None):
    """``gym.make_vec("CartPole-v1", num_envs=n_envs, vectorization_mode="sync")`` with an EXPLICIT autoreset mode
    (``"next_step"``, the default of gymnasium 1.x: the call after the end of an episode only resets that copy, or
    ``"same_step"``: the last observation goes through ``infos["final_obs"]``). With ``seed``, the copies are reset
    once with the seeds ``seed, seed + 1, ...`` (the observations of that reset are in ``envs.first_obs``)."""
    modes = {"next_step": gym.vector.AutoresetMode.NEXT_STEP, "same_step": gym.vector.AutoresetMode.SAME_STEP}
    if autoreset not in modes:
        raise ValueError(f"autoreset must be 'next_step' or 'same_step', got {autoreset!r}")
    kwargs = {} if max_episode_steps is None else {"max_episode_steps": int(max_episode_steps)}
    envs = gym.make_vec(ENV_ID, num_envs=int(n_envs), vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": modes[autoreset]}, **kwargs)
    if physics:
        set_physics(envs, **physics)
    if seed is not None:
        envs.first_obs, _ = envs.reset(seed=int(seed))
    return envs


def seed_everything(seed: int) -> None:
    """``random``, NumPy's global generator and PyTorch (``torch.manual_seed``) from one seed. The environment is seeded
    apart, at its first ``reset(seed=...)``, and so is its action space (``env.action_space.seed(...)``)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    _torch().manual_seed(seed)


@contextlib.contextmanager
def single_thread():
    """Run PyTorch on one thread inside the block (faster for these small networks, and the same arithmetic on every
    run), then restore the number of threads."""
    torch = _torch()
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


class StepCounter(gym.Wrapper):
    """Counts the steps, the true ends (``terminated``) and the time limits (``truncated``) of an environment, keeps
    the seed given to each ``reset`` (``reset_seeds``: None for a reset without seed) and the last ``keep`` transitions
    ``(obs, action, reward, next_obs, terminated, truncated)`` (copies)."""

    def __init__(self, env, keep: int = 0):
        super().__init__(env)
        self.steps = self.terminations = self.truncations = self.resets = 0
        self.reset_seeds: list = []
        self.keep = int(keep)
        self.transitions: list = []
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.resets += 1
        self.reset_seeds.append(kwargs.get("seed"))
        self._obs = np.array(obs, copy=True)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.terminations += bool(terminated)
        self.truncations += bool(truncated) and not bool(terminated)
        if self.keep:
            self.transitions.append((self._obs, int(action), float(reward), np.array(obs, copy=True),
                                     bool(terminated), bool(truncated)))
            del self.transitions[:-self.keep]
        self._obs = np.array(obs, copy=True)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------------------------------------------------
# Policies and evaluation
# ---------------------------------------------------------------------------------------------------------------------
def random_policy(seed: int):
    """A policy ``(state, info) -> int`` that plays 0 or 1 at random from ONE generator ``np.random.default_rng(seed)``
    (``env.action_space.sample()`` is not reset by ``reset(seed=...)``)."""
    rng = np.random.default_rng(seed)
    return lambda state, info: int(rng.integers(2))


def _net_policy(net, pick):
    torch = _torch()

    def policy(state, info):
        with torch.no_grad():
            out = net(torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0))
        return int(pick(out))
    return policy


def greedy_q(q_net):
    """``(state, info) -> int``: the action of largest value of a Q network (the greedy policy of a DQN)."""
    return _net_policy(q_net, lambda out: out.argmax(dim=1))


def greedy_logits(net):
    """``(state, info) -> int``: the action of largest probability of a policy network that returns logits (or of an
    agent with an ``actor`` attribute): the deterministic policy of a REINFORCE or PPO agent."""
    actor = getattr(net, "actor", net)
    return _net_policy(actor, lambda out: out.argmax(dim=1))


def rule_policy(state, info) -> int:
    """The one-line rule of the fiche (208.1): push towards the side the pole leans to, anticipating with its angular
    speed. It holds CartPole-v1 for 500 steps."""
    return int(state[2] + 0.5 * state[3] > 0)


def evaluate(policy, n_episodes: int = 100, seed: int = EVAL_SEED, *, max_steps: int = EPISODE_STEPS, env=None,
             **physics) -> dict:
    """Play ``policy`` (a function ``(state, info) -> int``, a Q network or an agent with an ``actor``) for
    ``n_episodes`` episodes on ``env`` (default: ``make_env(**physics)``), with the episode seeds drawn as
    ``mylearn.rl.evaluate_policy`` draws them, each episode cut at ``max_steps``. No exploration, no update. Returns
    ``mean_return``, ``std_return`` (ddof=0), ``mean_length`` and ``returns`` (a list)."""
    if not callable(policy):
        raise TypeError("policy must be a function (state, info) -> int or a torch module")
    if hasattr(policy, "parameters"):                  # a network: greedy on its output (or on its actor)
        policy = greedy_logits(policy) if hasattr(policy, "actor") else greedy_q(policy)
    env = env if env is not None else make_env(**physics)
    rng = np.random.default_rng(seed)
    returns, lengths = [], []
    for _ in range(int(n_episodes)):
        state, info = env.reset(seed=int(rng.integers(2**31)))
        total, steps = 0.0, 0
        while steps < max_steps:
            state, reward, terminated, truncated, info = env.step(policy(state, info))
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)), "returns": [float(r) for r in returns]}


class Discretize(gym.ObservationWrapper):
    """The discretisation of Ex 26.36: the index of the cell of the grid ``edges`` (np.digitize per variable, then
    np.ravel_multi_index), a Python int; ``observation_space`` becomes ``Discrete``."""

    def __init__(self, env, edges):
        super().__init__(env)
        self.edges = [np.asarray(e, dtype=float) for e in edges]
        self.dims = tuple(len(e) + 1 for e in self.edges)
        self.observation_space = gym.spaces.Discrete(int(np.prod(self.dims)))

    def observation(self, observation):
        indices = [int(np.digitize(value, e)) for value, e in zip(observation, self.edges)]
        return int(np.ravel_multi_index(indices, self.dims))


def tabular_reference():
    """``(make_tabular_env, Q)``: the tabular reference of chapter 26 (the solution of Ex 26.38), read back: a function
    that makes the discretised CartPole-v1 (12 states: the angle and its speed) and its table Q (12, 2), whose greedy
    policy ``mylearn.rl.greedy_policy(Q)`` is evaluated on that environment."""
    table = np.array(precomputed("tab26")["Q"], dtype=float)
    return (lambda: Discretize(make_env(), TABULAR_EDGES)), table


def moving_average(values, window: int = 20) -> np.ndarray:
    """The mean of each window of ``window`` consecutive values (shorter at the start), same length as ``values``."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    sums = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(1, values.size + 1)
    lo = np.maximum(0, idx - window)
    return (sums[idx] - sums[lo]) / (idx - lo)


# ---------------------------------------------------------------------------------------------------------------------
# The reference DQN (Ex B8.13 to B8.15): replay memory, target network, Huber loss, decreasing epsilon; ablations
# ---------------------------------------------------------------------------------------------------------------------
DQN_SETTINGS = {"hidden": 64, "lr": 5e-4, "batch_size": 128, "capacity": 10_000, "learning_starts": 1_000,
                "train_freq": 1, "target_every": 500, "gamma": 0.99, "eps_start": 1.0, "eps_end": 0.05,
                "eps_fraction": 0.3, "huber_delta": 1.0, "max_grad_norm": 10.0}
DQN_VARIANTS = ("dqn", "no_target", "no_replay", "double")


def dqn_steps(fast: bool = True) -> int:
    """The number of environment steps of a DQN run: 20 000 in fast mode, 50 000 in full mode."""
    return 20_000 if fast else 50_000


def q_network(hidden: int = 64, seed: int | None = None):
    """The Q network of the reference: 4 -> hidden -> hidden -> 2, ReLU (PyTorch's default initialisation, drawn after
    ``torch.manual_seed(seed)`` when ``seed`` is given)."""
    torch = _torch()
    nn = torch.nn
    if seed is not None:
        torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def linear_epsilon(step: int, total_steps: int, start: float = 1.0, end: float = 0.05, fraction: float = 0.3) -> float:
    """ε of a step: linear from ``start`` to ``end`` over the first ``fraction * total_steps`` steps, then ``end``."""
    span = max(1.0, fraction * total_steps)
    return float(end + (start - end) * max(0.0, 1.0 - step / span))


def dqn_targets(online, target, next_states, rewards, terminated, gamma: float, variant: str = "dqn"):
    """The targets of a mini-batch, without gradient: ``r + gamma (1 - terminated) max_a' Q_target(s', a')`` for
    ``"dqn"`` and ``"no_replay"``; the online network in place of the target network for ``"no_target"``; the action
    chosen by the online network and valued by the target network for ``"double"``."""
    torch = _torch()
    if variant not in DQN_VARIANTS:
        raise ValueError(f"variant must be one of {DQN_VARIANTS}, got {variant!r}")
    with torch.no_grad():
        if variant == "double":
            best = online(next_states).argmax(dim=1, keepdim=True)
            following = target(next_states).gather(1, best).squeeze(1)
        else:
            following = (online if variant == "no_target" else target)(next_states).max(dim=1).values
        return rewards + gamma * (1.0 - terminated.float()) * following


class _Memory:
    """A FIFO replay memory (the reference of ``rl.ReplayBuffer``: sampling without replacement)."""

    def __init__(self, capacity: int, rng):
        self.capacity, self.rng = int(capacity), rng
        self.s = np.zeros((capacity, 4), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float64)
        self.s2 = np.zeros((capacity, 4), np.float32)
        self.d = np.zeros(capacity, bool)
        self.pos = self.size = 0

    def push(self, s, a, r, s2, d):
        self.s[self.pos], self.a[self.pos], self.r[self.pos], self.s2[self.pos], self.d[self.pos] = s, a, r, s2, d
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idx = self.rng.choice(self.size, size=n, replace=False)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


def train_dqn(seed: int = 0, total_steps: int | None = None, settings: dict | None = None, variant: str = "dqn", *,
              eval_every: int = 2_000, eval_episodes: int = 20, eval_seed: int = EVAL_SEED, fast: bool = True,
              env=None, targets_fn=None, capacity: int | None = None, stop_after: int | None = None):
    """Train the reference DQN on CartPole-v1 and return ``(q_net, history)``.

    One environment step: ε-greedy action (``linear_epsilon``), the transition stored with ``terminated`` only (a time
    limit keeps its bootstrap), an episode that ends (terminated or truncated) reset; from ``learning_starts`` steps on,
    every ``train_freq`` steps, one Adam step on the Huber loss (``F.huber_loss`` with ``huber_delta``) between
    ``Q(s, a)`` and the targets of ``dqn_targets``, gradient norm clipped; the target network is a copy of the online
    one every ``target_every`` steps. Variants: ``"no_target"`` (targets from the online network), ``"no_replay"``
    (a memory of ``batch_size`` transitions: each mini-batch is the last ``batch_size`` steps, in a random order),
    ``"double"`` (Double DQN). Every ``eval_every`` steps, the greedy policy is evaluated on ``eval_episodes`` episodes
    (``evaluate``, its own environment and seeds: the training run does not depend on it). ``targets_fn`` (same
    arguments as ``dqn_targets``) and ``capacity`` replace the targets and the size of the memory of the variant: the
    full mode of Ex B8.15 runs this loop with the learner's ``targets_15`` and ``memory_size_15``. ``stop_after`` stops
    the run after that many steps (the same schedule as the whole run: a prefix of the recorded curve).

    ``history``: ``eval_steps``, ``eval_returns``, ``episode_returns``, ``episode_ends`` (the step of each end),
    ``seconds``, and the protocol (``seed``, ``variant``, ``total_steps``, ``settings``)."""
    torch = _torch()
    F = torch.nn.functional
    s = {**DQN_SETTINGS, **(settings or {})}
    total_steps = int(total_steps if total_steps is not None else dqn_steps(fast))
    if variant not in DQN_VARIANTS:
        raise ValueError(f"variant must be one of {DQN_VARIANTS}, got {variant!r}")
    if capacity is None:
        capacity = s["batch_size"] if variant == "no_replay" else s["capacity"]
    targets_fn = targets_fn if targets_fn is not None else dqn_targets
    start = time.perf_counter()
    with single_thread():
        q_net = q_network(s["hidden"], seed=seed)
        frozen = copy.deepcopy(q_net)
        optimizer = torch.optim.Adam(q_net.parameters(), lr=s["lr"])
        rng = np.random.default_rng(seed)
        memory = _Memory(capacity, np.random.default_rng(seed + 1))
        env = env if env is not None else make_env()
        state, _ = env.reset(seed=int(seed))
        history = {"eval_steps": [], "eval_returns": [], "episode_returns": [], "episode_ends": []}
        ret = 0.0
        for step in range(total_steps if stop_after is None else min(total_steps, int(stop_after))):
            eps = linear_epsilon(step, total_steps, s["eps_start"], s["eps_end"], s["eps_fraction"])
            if rng.random() < eps:
                action = int(rng.integers(2))
            else:
                with torch.no_grad():
                    action = int(q_net(torch.as_tensor(state).unsqueeze(0)).argmax(dim=1))
            next_state, reward, terminated, truncated, _ = env.step(action)
            memory.push(state, action, reward, next_state, terminated)
            ret += float(reward)
            state = next_state
            if terminated or truncated:
                history["episode_returns"].append(ret)
                history["episode_ends"].append(step + 1)
                ret = 0.0
                state, _ = env.reset()
            if step + 1 >= s["learning_starts"] and memory.size >= s["batch_size"] and step % s["train_freq"] == 0:
                bs, ba, br, bs2, bd = memory.sample(s["batch_size"])
                bs, bs2 = torch.as_tensor(bs), torch.as_tensor(bs2)
                br = torch.as_tensor(br, dtype=torch.float32)
                y = targets_fn(q_net, frozen, bs2, br, torch.as_tensor(bd), s["gamma"], variant)
                pred = q_net(bs).gather(1, torch.as_tensor(ba).unsqueeze(1)).squeeze(1)
                loss = F.huber_loss(pred, y, delta=s["huber_delta"])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), s["max_grad_norm"])
                optimizer.step()
            if (step + 1) % s["target_every"] == 0:
                frozen.load_state_dict(q_net.state_dict())
            if eval_every and (step + 1) % eval_every == 0:
                history["eval_steps"].append(step + 1)
                history["eval_returns"].append(evaluate(greedy_q(q_net), eval_episodes, eval_seed)["mean_return"])
    history.update(seconds=time.perf_counter() - start, seed=int(seed), variant=variant, total_steps=total_steps,
                   settings=s)
    return q_net.eval(), history


# ---------------------------------------------------------------------------------------------------------------------
# The reference REINFORCE (Ex B8.16, B8.17)
# ---------------------------------------------------------------------------------------------------------------------
REINFORCE_SETTINGS = {"hidden": 64, "lr": 1e-2, "gamma": 0.99, "baseline_window": 20}
BASELINES = ("none", "mean", "normalize")


def policy_network(hidden: int = 64, seed: int | None = None):
    """The policy network of the reference: 4 -> hidden -> 2 logits, tanh."""
    torch = _torch()
    nn = torch.nn
    if seed is not None:
        torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 2))


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    """``G_t = r_t + gamma G_(t+1)`` of one episode (the reference of ``rl.returns_to_go``), float64."""
    out = np.zeros(len(rewards))
    future = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        future = float(rewards[t]) + gamma * future
        out[t] = future
    return out


def reinforce_weights(returns, baseline: str, past: list) -> np.ndarray:
    """The weights of ``log pi(a_t | s_t)`` in the loss of one episode: ``G_t`` (``"none"``), ``G_t - b`` with ``b``
    the mean of the returns ``G_t`` of the previous episodes kept in ``past`` (``"mean"``; 0 for the first episode),
    or ``(G_t - mean) / (std + 1e-8)`` over the episode (``"normalize"``)."""
    returns = np.asarray(returns, dtype=float)
    if baseline == "none":
        return returns
    if baseline == "mean":
        return returns - (float(np.mean(np.concatenate(past))) if past else 0.0)
    if baseline == "normalize":
        return (returns - returns.mean()) / (returns.std() + 1e-8)
    raise ValueError(f"baseline must be one of {BASELINES}, got {baseline!r}")


def play_episode(policy_net, env, generator=None, max_steps: int = EPISODE_STEPS):
    """One episode with actions drawn from the policy (``Categorical(logits=...)``, ``generator`` for the draws):
    ``(log_probs, rewards)``, a list of 0-D tensors that require grad and a list of floats."""
    torch = _torch()
    state, _ = env.reset()
    log_probs, rewards = [], []
    for _ in range(max_steps):
        logits = policy_net(torch.as_tensor(state).unsqueeze(0))[0]
        probs = torch.softmax(logits, dim=0)
        action = int(torch.multinomial(probs.detach(), 1, generator=generator))
        log_probs.append(torch.log_softmax(logits, dim=0)[action])
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return log_probs, rewards


def train_reinforce(seed: int = 0, n_episodes: int = 300, baseline: str = "none", settings: dict | None = None, *,
                    eval_every: int = 0, eval_episodes: int = 20, eval_seed: int = EVAL_SEED):
    """Train the reference REINFORCE on CartPole-v1: one Adam step per episode on
    ``-mean_t(log pi(a_t | s_t) * w_t)``, the weights ``w_t`` of ``reinforce_weights`` (the mean baseline over the
    returns of the last ``baseline_window`` episodes). Returns ``(policy_net, history)`` with ``episode_returns``,
    ``eval_episodes``/``eval_returns`` (when ``eval_every``), ``seconds`` and the protocol."""
    torch = _torch()
    s = {**REINFORCE_SETTINGS, **(settings or {})}
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, got {baseline!r}")
    start = time.perf_counter()
    with single_thread():
        net = policy_network(s["hidden"], seed=seed)
        optimizer = torch.optim.Adam(net.parameters(), lr=s["lr"])
        generator = torch.Generator().manual_seed(seed)
        env = make_env()
        env.reset(seed=int(seed))
        past: list = []
        history = {"episode_returns": [], "eval_episodes": [], "eval_returns": []}
        for episode in range(int(n_episodes)):
            log_probs, rewards = play_episode(net, env, generator)
            returns = discounted_returns(rewards, s["gamma"])
            weights = torch.as_tensor(reinforce_weights(returns, baseline, past), dtype=torch.float32)
            loss = -(torch.stack(log_probs) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            past = (past + [returns])[-s["baseline_window"]:]
            history["episode_returns"].append(float(sum(rewards)))
            if eval_every and (episode + 1) % eval_every == 0:
                history["eval_episodes"].append(episode + 1)
                history["eval_returns"].append(evaluate(greedy_logits(net), eval_episodes, eval_seed)["mean_return"])
    history.update(seconds=time.perf_counter() - start, seed=int(seed), baseline=baseline, n_episodes=int(n_episodes),
                   settings=s)
    return net.eval(), history


def collect_episodes(policy_net, n_episodes: int, seed: int, max_steps: int = EPISODE_STEPS) -> list:
    """``n_episodes`` episodes of CartPole-v1 played by ``policy_net`` without learning (actions drawn by
    ``torch.multinomial`` from a generator seeded with ``seed``, the environment reset once with ``seed``): a list of
    dicts ``states`` (T, 4) float32, ``actions`` (T,) int64 and ``rewards`` (T,) float64. The same arguments give the
    same episodes."""
    torch = _torch()
    generator = torch.Generator().manual_seed(seed)
    env = make_env()
    state, _ = env.reset(seed=int(seed))
    episodes = []
    with single_thread(), torch.no_grad():
        for _ in range(int(n_episodes)):
            states, actions, rewards = [], [], []
            for _ in range(max_steps):
                probs = torch.softmax(policy_net(torch.as_tensor(state).unsqueeze(0))[0], dim=0)
                action = int(torch.multinomial(probs, 1, generator=generator))
                states.append(np.array(state, dtype=np.float32))
                actions.append(action)
                state, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            episodes.append({"states": np.stack(states), "actions": np.array(actions, dtype=np.int64),
                             "rewards": np.array(rewards, dtype=float)})
            state, _ = env.reset()
    return episodes


def episode_gradients(policy_net, episodes, baseline: str = "none", gamma: float = 0.99) -> np.ndarray:
    """One REINFORCE gradient per episode, flattened over all the parameters of ``policy_net``: the gradient of
    ``-mean_t(log pi(a_t | s_t) * w_t)`` with ``w_t = G_t`` (``"none"``) or ``w_t = G_t - b``, ``b`` the mean of the
    returns ``G_t`` of ALL the episodes given (``"mean"``, one constant for the batch). Returns float64 (M, P); the
    network is left unchanged (its ``.grad`` too)."""
    torch = _torch()
    if baseline not in ("none", "mean"):
        raise ValueError(f"baseline must be 'none' or 'mean', got {baseline!r}")
    returns = [discounted_returns(e["rewards"], gamma) for e in episodes]
    b = float(np.mean(np.concatenate(returns))) if baseline == "mean" else 0.0
    params = [p for p in policy_net.parameters() if p.requires_grad]
    rows = []
    with single_thread():
        for e, G in zip(episodes, returns):
            logits = policy_net(torch.as_tensor(e["states"]))
            logp = torch.log_softmax(logits, dim=1).gather(1, torch.as_tensor(e["actions"])[:, None]).squeeze(1)
            loss = -(logp * torch.as_tensor(G - b, dtype=torch.float32)).mean()
            grads = torch.autograd.grad(loss, params)
            rows.append(torch.cat([g.reshape(-1) for g in grads]).double().numpy())
    return np.stack(rows)


def gradient_spread(grads) -> float:
    """The spread of per-episode gradients (M, P): the mean squared distance to their mean,
    ``mean_i ||g_i - mean(g)||^2`` (the trace of their covariance, ddof=0)."""
    g = np.asarray(grads, dtype=float)
    return float(np.mean(np.sum((g - g.mean(axis=0)) ** 2, axis=1)))


# ---------------------------------------------------------------------------------------------------------------------
# The reference PPO (Ex B8.20 on): vector environments, actor-critic, GAE, clipped objective, entropy, logs
# ---------------------------------------------------------------------------------------------------------------------
PPO_SETTINGS = {"n_envs": 4, "n_steps": 128, "n_minibatches": 4, "n_epochs": 10, "lr": 1e-3, "anneal_lr": True,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
                "max_grad_norm": 0.5, "norm_adv": True, "hidden": 64, "adam_eps": 1e-5}


def ppo_steps(fast: bool = True) -> int:
    """The number of environment steps of a PPO run (all copies): 50 000 in fast mode, 100 000 in full mode."""
    return 50_000 if fast else 100_000


def _actor_critic_class():
    torch = _torch()
    nn = torch.nn

    def layer(n_in, n_out, std):
        lin = nn.Linear(n_in, n_out)
        nn.init.orthogonal_(lin.weight, std)
        nn.init.zeros_(lin.bias)
        return lin

    class ActorCritic(nn.Module):
        """Separate actor (4 -> hidden -> hidden -> 2 logits) and critic (4 -> hidden -> hidden -> 1), tanh,
        orthogonal initialisation (gain sqrt(2), 0.01 for the logits, 1 for the value), as in CleanRL's ppo.py."""

        def __init__(self, hidden: int = 64):
            super().__init__()
            self.hidden = hidden
            g = math.sqrt(2)
            self.actor = nn.Sequential(layer(4, hidden, g), nn.Tanh(), layer(hidden, hidden, g), nn.Tanh(),
                                       layer(hidden, 2, 0.01))
            self.critic = nn.Sequential(layer(4, hidden, g), nn.Tanh(), layer(hidden, hidden, g), nn.Tanh(),
                                        layer(hidden, 1, 1.0))

        def value(self, obs):
            return self.critic(obs).squeeze(-1)

        def forward(self, obs):
            return self.actor(obs)

    return ActorCritic


def __getattr__(name: str):
    if name == "ActorCritic":
        return _actor_critic_class()
    raise AttributeError(f"module 'wb.deep_rl_env' has no attribute {name!r}")


def gae_numpy(rewards, values, next_values, terminated, truncated, gamma: float, lam: float):
    """GAE over a rollout (T, N), the reference of ``mylearn.deep_rl.compute_gae``: ``(advantages, returns)``."""
    rewards, values, next_values = (np.asarray(a, dtype=float) for a in (rewards, values, next_values))
    term, done = np.asarray(terminated, bool), np.asarray(terminated, bool) | np.asarray(truncated, bool)
    delta = rewards + gamma * np.where(term, 0.0, next_values) - values
    adv = np.zeros_like(delta)
    following = np.zeros_like(delta[0])
    for t in range(len(delta) - 1, -1, -1):
        following = delta[t] + gamma * lam * np.where(done[t], 0.0, following)
        adv[t] = following
    return adv, adv + values


def explained_variance_numpy(y_pred, y_true) -> float:
    """``1 - Var(y_true - y_pred) / Var(y_true)``, nan when ``Var(y_true) == 0`` (the reference of
    ``mylearn.deep_rl.explained_variance``)."""
    y_pred, y_true = np.asarray(y_pred, float), np.asarray(y_true, float)
    var = np.var(y_true)
    return float("nan") if var == 0 else float(1.0 - np.var(y_true - y_pred) / var)


def collect_rollout(agent, envs, obs, just_reset, n_steps: int, generator, ep_returns, finished: list, step0: int = 0,
                    reward_fn=None):
    """One rollout of ``n_steps`` steps in each copy of ``envs`` (next-step autoreset), with actions drawn from the
    agent. Returns ``(batch, obs, just_reset)``: ``batch`` holds (T, N, ...) arrays ``obs``, ``actions``,
    ``log_probs``, ``values``, ``rewards`` (those of ``reward_fn(obs, actions, next_obs)`` when given), ``terminated``,
    ``truncated``, ``next_obs``, ``next_values`` and ``valid`` (False on the steps that only reset a copy: they are
    not transitions). ``ep_returns`` (N,) accumulates the TRUE rewards of the running episodes; ``finished`` receives
    ``(step, return)`` for each episode that ends."""
    torch = _torch()
    N = envs.num_envs
    keys = ("obs", "actions", "log_probs", "values", "rewards", "terminated", "truncated", "next_obs", "valid")
    buf = {k: [] for k in keys}
    for t in range(n_steps):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            logits = agent.actor(x)
            probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            log_probs = torch.log_softmax(logits, dim=1).gather(1, actions[:, None]).squeeze(1)
            values = agent.value(x)
        acts = actions.numpy()
        next_obs, rewards, terminated, truncated, _ = envs.step(acts)
        valid = ~just_reset
        true_rewards = np.asarray(rewards, dtype=float)
        learn_rewards = true_rewards if reward_fn is None else np.asarray(reward_fn(obs, acts, next_obs), dtype=float)
        for k, v in zip(keys, (obs, acts, log_probs.numpy(), values.numpy(), learn_rewards, terminated, truncated,
                               next_obs, valid)):
            buf[k].append(np.array(v, copy=True))
        ep_returns += np.where(valid, true_rewards, 0.0)
        ended = (terminated | truncated) & valid
        for i in np.flatnonzero(ended):
            finished.append((step0 + (t + 1) * N, float(ep_returns[i])))
            ep_returns[i] = 0.0
        just_reset = terminated | truncated
        obs = next_obs
    batch = {k: np.stack(v) for k, v in buf.items()}
    batch["rewards"] = np.where(batch["valid"], batch["rewards"], 0.0)
    with torch.no_grad():
        nv = agent.value(torch.as_tensor(batch["next_obs"].reshape(-1, 4), dtype=torch.float32))
    batch["next_values"] = nv.numpy().reshape(n_steps, N)
    return batch, obs, just_reset


def ppo_loss(agent, obs, actions, old_log_probs, advantages, returns, settings: dict | None = None):
    """The PPO loss of a mini-batch (tensors) and its parts, the reference of the learner's ``ppo_loss_20``:
    ``(loss, parts)`` with ``loss = policy_loss + vf_coef * value_loss - ent_coef * entropy``, ``policy_loss =
    -mean(min(rho A, clip(rho, 1 - eps, 1 + eps) A))``, ``rho = exp(log pi(a|s) - old_log_prob)``, ``value_loss =
    0.5 mean((V(s) - returns)^2)``, ``entropy`` the mean entropy of ``pi(.|s)``; the advantages are used as given.
    ``parts``: floats ``policy_loss``, ``value_loss``, ``entropy``, ``approx_kl`` (mean of ``(rho - 1) - log rho``),
    ``clipfrac`` (share of ``|rho - 1| > eps``)."""
    torch = _torch()
    s = {**PPO_SETTINGS, **(settings or {})}
    log_all = torch.log_softmax(agent.actor(obs), dim=1)
    new_logp = log_all.gather(1, actions.long()[:, None]).squeeze(1)
    entropy = -(log_all.exp() * log_all).sum(dim=1).mean()
    log_ratio = new_logp - old_log_probs
    ratio = log_ratio.exp()
    eps = s["clip_eps"]
    policy_loss = -torch.min(ratio * advantages, ratio.clamp(1 - eps, 1 + eps) * advantages).mean()
    value_loss = 0.5 * ((agent.value(obs) - returns) ** 2).mean()
    loss = policy_loss + s["vf_coef"] * value_loss - s["ent_coef"] * entropy
    with torch.no_grad():
        parts = {"policy_loss": float(policy_loss), "value_loss": float(value_loss), "entropy": float(entropy),
                 "approx_kl": float(((ratio - 1) - log_ratio).mean()),
                 "clipfrac": float(((ratio - 1).abs() > eps).float().mean())}
    return loss, parts


def ppo_update(agent, optimizer, batch, advantages, returns, s: dict, generator) -> dict:
    """``n_epochs`` passes over the valid transitions of the rollout, in ``n_minibatches`` shuffled mini-batches: one
    Adam step each on ``-L^CLIP + vf_coef * 0.5 * MSE(V, returns) - ent_coef * entropy`` (advantages normalised per
    mini-batch when ``norm_adv``, gradient norm clipped at ``max_grad_norm``). Returns the logs of the update: the
    means over the mini-batches of ``policy_loss``, ``value_loss``, ``entropy``, ``clipfrac`` and ``approx_kl`` (the
    estimator ``(rho - 1) - log rho``)."""
    torch = _torch()
    valid = batch["valid"].reshape(-1)
    flat = {k: torch.as_tensor(batch[k].reshape(-1, *batch[k].shape[2:])[valid]) for k in ("obs", "actions",
                                                                                          "log_probs")}
    adv_all = torch.as_tensor(advantages.reshape(-1)[valid], dtype=torch.float32)
    ret_all = torch.as_tensor(returns.reshape(-1)[valid], dtype=torch.float32)
    obs_all = flat["obs"].float()
    n = len(adv_all)
    size = max(1, n // s["n_minibatches"])
    logs = {"policy_loss": [], "value_loss": [], "entropy": [], "clipfrac": [], "approx_kl": []}
    for _ in range(s["n_epochs"]):
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n - size + 1, size):
            idx = order[lo:lo + size]
            adv = adv_all[idx]
            if s["norm_adv"] and len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss, parts = ppo_loss(agent, obs_all[idx], flat["actions"][idx], flat["log_probs"][idx], adv,
                                   ret_all[idx], s)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), s["max_grad_norm"])
            optimizer.step()
            for k, v in parts.items():
                logs[k].append(v)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in logs.items()}


def train_ppo(seed: int = 0, total_steps: int | None = None, settings: dict | None = None, *, fast: bool = True,
              reward_fn=None, physics: dict | None = None, callback=None):
    """Train the reference PPO on ``settings["n_envs"]`` copies of CartPole-v1 (``make_vec``, next-step autoreset, the
    reset steps left out, a time limit bootstrapped on the real last observation) and return ``(agent, logs)``.

    ``agent`` is an ``ActorCritic`` in eval mode. ``logs`` is a dict of lists, one value per iteration, with the keys
    ``LOG_KEYS``: ``step`` (environment steps so far, all copies), ``iteration``, ``lr``, ``policy_loss``,
    ``value_loss``, ``entropy``, ``approx_kl``, ``clipfrac`` (means over the mini-batches of the update),
    ``explained_variance`` (of the values of the rollout against the returns of GAE, before the update), ``episodes``
    (episodes finished during the rollout) and ``mean_return`` (their mean TRUE return, nan if none); plus
    ``episode_steps``/``episode_returns`` (every finished episode), ``seconds`` and the protocol (``seed``,
    ``total_steps``, ``settings``). ``reward_fn(obs, actions, next_obs) -> (N,)`` replaces the reward the agent learns
    from (RLHF); ``physics`` changes the CartPole of the copies; ``callback(iteration, agent, logs)`` is called after
    each update (return True to stop)."""
    torch = _torch()
    s = {**PPO_SETTINGS, **(settings or {})}
    total_steps = int(total_steps if total_steps is not None else ppo_steps(fast))
    per_iter = s["n_envs"] * s["n_steps"]
    n_iter = max(1, total_steps // per_iter)
    start = time.perf_counter()
    logs = {k: [] for k in LOG_KEYS}
    logs.update(episode_steps=[], episode_returns=[])
    with single_thread():
        torch.manual_seed(seed)
        agent = _actor_critic_class()(s["hidden"])
        optimizer = torch.optim.Adam(agent.parameters(), lr=s["lr"], eps=s["adam_eps"])
        generator = torch.Generator().manual_seed(seed)
        envs = make_vec(s["n_envs"], seed=seed, physics=physics)
        obs, just_reset = envs.first_obs, np.zeros(s["n_envs"], bool)
        ep_returns = np.zeros(s["n_envs"])
        for it in range(n_iter):
            lr = s["lr"] * (1.0 - it / n_iter) if s["anneal_lr"] else s["lr"]
            for group in optimizer.param_groups:
                group["lr"] = lr
            finished: list = []
            agent.eval()
            batch, obs, just_reset = collect_rollout(agent, envs, obs, just_reset, s["n_steps"], generator,
                                                     ep_returns, finished, it * per_iter, reward_fn)
            adv, ret = gae_numpy(batch["rewards"], batch["values"], batch["next_values"], batch["terminated"],
                                 batch["truncated"], s["gamma"], s["gae_lambda"])
            valid = batch["valid"]
            ev = explained_variance_numpy(batch["values"][valid], ret[valid])
            agent.train()
            update = ppo_update(agent, optimizer, batch, adv, ret, s, generator)
            row = {"step": (it + 1) * per_iter, "iteration": it + 1, "lr": lr, **update, "explained_variance": ev,
                   "episodes": len(finished),
                   "mean_return": float(np.mean([r for _, r in finished])) if finished else float("nan")}
            for k in LOG_KEYS:
                logs[k].append(row[k])
            logs["episode_steps"] += [st for st, _ in finished]
            logs["episode_returns"] += [r for _, r in finished]
            if callback is not None and callback(it + 1, agent, logs):
                break
        envs.close()
    logs.update(seconds=time.perf_counter() - start, seed=int(seed), total_steps=total_steps, settings=s)
    return agent.eval(), logs


def log_summary(logs: dict) -> dict:
    """Two numbers of a PPO log (the format of ``LOG_KEYS``), the reference of the learner's ``summary_22``: ``auc``,
    the mean of the finite values of ``mean_return`` (the area under the curve divided by its length; nan if there is
    none) and ``first_475``, the first ``step`` whose ``mean_return`` is >= 475 (an int; None if it never is)."""
    returns = np.asarray(logs["mean_return"], dtype=float)
    finite = returns[np.isfinite(returns)]
    reached = np.flatnonzero(np.nan_to_num(returns, nan=-np.inf) >= SOLVED)
    return {"auc": float(finite.mean()) if finite.size else float("nan"),
            "first_475": int(logs["step"][reached[0]]) if reached.size else None}

# ---------------------------------------------------------------------------------------------------------------------
# The single checkpoint of a trained PPO agent (outside the repository, atomic, with its provenance)
# ---------------------------------------------------------------------------------------------------------------------
REFERENCE_SEED = 2020           # the seed of the reference PPO agent (reference_agent)


def checkpoints_dir() -> Path:
    """Where the trained agents go: ``wb.datasets.cache_dir() / "deep_rl"`` (never versioned; on Colab,
    ``/content/wb_cache/deep_rl``, erased with the session)."""
    from wb import datasets

    path = datasets.cache_dir() / "deep_rl"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical(key: dict) -> str:
    return json.dumps(key, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def agent_key(settings: dict, *, source: str, fast: bool, seed: int, total_steps: int, kind: str = "ppo") -> dict:
    """The complete key of a trained agent: its kind, who trained it (``"learner"``, ``"solution"`` or
    ``"reference"``), the mode, the seed, the budget and every setting. Two agents whose keys differ never share a
    file: the solutions notebook never reads nor overwrites the learner's agent."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return {"version": CHECKPOINT_VERSION, "kind": kind, "source": source, "fast": bool(fast), "seed": int(seed),
            "total_steps": int(total_steps), "settings": {k: settings[k] for k in sorted(settings)}}


def agent_file(key: dict, folder: str | Path | None = None) -> Path:
    """The ``.pt`` file of a key: its kind, its source, its mode and 16 hexadecimal digits of the SHA-256 of the key."""
    digest = hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()[:16]
    base = Path(folder) if folder is not None else checkpoints_dir()
    return base / f"{key.get('kind', 'agent')}-{key.get('source')}-{'fast' if key.get('fast') else 'full'}-{digest}.pt"


def save_agent(key: dict, agent, logs: dict | None = None, folder: str | Path | None = None,
               extra: dict | None = None) -> Path:
    """Save the ``state_dict`` of ``agent``, its logs and ``extra`` in the file of ``key``: written to a temporary file,
    then renamed (``os.replace``), so that an interrupted save never leaves a half-written checkpoint."""
    torch = _torch()
    path = agent_file(key, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": _canonical(key), "state_dict": {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
               "hidden": int(getattr(agent, "hidden", 64)), "logs": json.dumps(logs or {}, default=float),
               "extra": json.dumps(extra or {}, default=str), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "__complete__": True}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_agent(key: dict, folder: str | Path | None = None, build=None):
    """``(agent, info)`` for the checkpoint of ``key`` (``build()``, an untrained module of the same class, with the
    weights loaded strictly, in eval mode; by default an ``ActorCritic`` of the saved width; ``info``: ``logs``,
    ``extra``, ``saved_at``, ``path``), or None when there is none, when the file holds another key, or when it is
    incomplete or unreadable."""
    torch = _torch()
    path = agent_file(key, folder)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("__complete__") is not True or payload.get("key") != _canonical(key):
            return None
        agent = build() if build is not None else _actor_critic_class()(payload["hidden"])
        agent.load_state_dict(payload["state_dict"])
    except Exception:  # noqa: BLE001 - truncated, corrupt, another format: as if absent
        return None
    info = {"logs": json.loads(payload["logs"]), "extra": json.loads(payload["extra"]), "saved_at": payload["saved_at"],
            "path": path}
    return agent.eval(), info


def delete_agent(key: dict, folder: str | Path | None = None) -> bool:
    """Remove the checkpoint of ``key`` if there is one (a new training removes the old agent first, so that a failed
    run never leaves the previous one in place); True when a file was removed."""
    path = agent_file(key, folder)
    if path.exists():
        path.unlink()
        return True
    return False


def reference_key(fast: bool = True) -> dict:
    """The checkpoint key of the reference PPO agent of the mode (``reference_agent``)."""
    return agent_key(dict(PPO_SETTINGS), source="reference", fast=fast, seed=REFERENCE_SEED,
                     total_steps=ppo_steps(fast))


def reference_agent(fast: bool = True, folder: str | Path | None = None, verbose: bool = True):
    """``(agent, info)``: the reference PPO agent of the mode, trained with ``train_ppo`` (``PPO_SETTINGS``, seed
    ``REFERENCE_SEED``, ``ppo_steps(fast)`` steps) the first time (about 20 s on a CPU core in fast mode), then read
    from its checkpoint (source ``"reference"``). The explicit fallback when the learner's agent is missing."""
    key = reference_key(fast)
    loaded = load_agent(key, folder)
    if loaded is None:
        if verbose:
            print("ℹ️ entraînement de l'agent PPO de référence (une seule fois)", flush=True)
        agent, logs = train_ppo(REFERENCE_SEED, ppo_steps(fast))
        save_agent(key, agent, logs, folder)
        loaded = load_agent(key, folder)
    elif verbose:
        print(f"ℹ️ agent PPO de référence relu ({loaded[1]['path'].name})")
    return loaded


def same_weights(a, b, near: float = 0.0) -> bool:
    """True when two torch modules hold the same parameters and buffers, in the same order (an agent handed back in
    place of one trained from scratch): exactly, or within a relative L2 distance ``near`` (a copy with noise)."""
    return _same_parameters(_parameters_of(a), _parameters_of(b), near=near)


ANSWER_FUNCTIONS = ("linear_epsilon", "dqn_targets", "train_dqn", "discounted_returns", "reinforce_weights",
                    "play_episode", "train_reinforce", "episode_gradients", "gae_numpy", "explained_variance_numpy",
                    "collect_rollout", "ppo_loss", "ppo_update", "train_ppo", "reference_agent", "load_agent",
                    "precomputed")


def reference_answers() -> dict:
    """``{"wbr.<name>": function}`` for the functions of this module that compute what the learner writes in the
    notebook (``ANSWER_FUNCTIONS``: ε, the DQN targets and trainer, REINFORCE and its gradients, GAE, the explained
    variance, the PPO rollout, loss, update and trainer, the reference agent, the checkpoints and the precomputed
    runs), plus ``torch.load``: what the learner's code must not call during a check (``wb.nbtools.own_code``)."""
    torch = _torch()
    module = globals()
    return {**{f"wbr.{name}": module[name] for name in ANSWER_FUNCTIONS}, "torch.load": torch.load}


# ---------------------------------------------------------------------------------------------------------------------
# The precomputed protocols of the notebook (tools/chapters/history_chB8.py)
# ---------------------------------------------------------------------------------------------------------------------
PROTOCOLS_FILE = Path(__file__).with_name("deep_rl_protocols.json")


def precomputed(name: str) -> dict:
    """A precomputed protocol of chapter B8 (``tools/chapters/history_chB8.py`` writes them into
    ``deep_rl_protocols.json``), read at run time: the curves of several seeds that a fast-mode notebook cannot
    afford to train, with their protocol (seeds, budget, settings)."""
    data = json.loads(PROTOCOLS_FILE.read_text(encoding="utf-8")) if PROTOCOLS_FILE.exists() else {}
    if name not in data:
        raise KeyError(f"no precomputed protocol {name!r} in {PROTOCOLS_FILE.name}")
    return data[name]


def band(curves) -> dict:
    """The median and the range (min, max) of several curves of the same length, point by point (the figure of the
    contract: « médiane et intervalle sur trois graines »)."""
    arr = np.asarray(curves, dtype=float)
    return {"median": np.median(arr, axis=0), "low": arr.min(axis=0), "high": arr.max(axis=0)}


# ---------------------------------------------------------------------------------------------------------------------
# RLHF in miniature (Ex B8.26): a hidden "true" reward, simulated preferences, a reward model of Bradley-Terry
# ---------------------------------------------------------------------------------------------------------------------
RLHF_WIDTH = 0.5                # the true reward of a step: exp(-(x / RLHF_WIDTH)^2), x the position of the cart
RLHF_STEPS = 100                # a trajectory of the preference data: at most 100 steps
OBS_SCALE = (2.4, 3.0, 0.2095, 3.0)   # the order of magnitude of each variable (the reward model divides by it)
# The behaviours that play the trajectories of the preference data: the one-line rule of 208.1 with a share of random
# actions (1.0: the random policy). The rule ignores the position of the cart: some of its trajectories stay near the
# centre, others drift away.
RLHF_BEHAVIOURS = (1.0, 0.0, 0.1, 0.3, 0.5)


def centered_reward(obs) -> np.ndarray:
    """The hidden "true" reward of B8.26: ``exp(-(x / RLHF_WIDTH)^2)`` for each observation (the last axis holds the
    four variables; x the position of the cart): 1 at the centre, 0.37 at 0.5 m, almost 0 beyond 1 m."""
    x = np.asarray(obs, dtype=float)[..., 0]
    return np.exp(-(x / RLHF_WIDTH) ** 2)


def _behaviour_trajectory(env, share_random: float, rng, max_steps: int) -> np.ndarray:
    state, _ = env.reset(seed=int(rng.integers(2**31)))
    states = []
    for _ in range(max_steps):
        action = int(rng.integers(2)) if rng.random() < share_random else rule_policy(state, None)
        state, _, terminated, truncated, _ = env.step(action)
        states.append(np.array(state, dtype=np.float32))
        if terminated or truncated:
            break
    return np.stack(states)


def preference_pairs(n_pairs: int, seed: int, max_steps: int = RLHF_STEPS, behaviours=RLHF_BEHAVIOURS) -> dict:
    """``n_pairs`` pairs of trajectories of CartPole-v1 (at most ``max_steps`` steps each), played by ``behaviours``
    drawn at random (shares of random actions; by default ``RLHF_BEHAVIOURS``; ``(1.0,)``: the random policy only),
    and labelled by the hidden true reward (simulated preferences, without a
    human). Returns a dict: ``first`` and ``second`` (lists of float32 arrays (T_i, 4): the observation that FOLLOWS
    each step, the one a reward of the step is computed on), ``true_first`` and ``true_second`` (float arrays (n,):
    the sums of ``centered_reward`` over each trajectory), ``preferred`` (int array (n,): 0 when the first trajectory
    has the larger true return, 1 otherwise; a tie goes to the first) and ``behaviours`` (the share of random actions
    of each trajectory, (n, 2)). The same arguments give the same pairs."""
    rng = np.random.default_rng(seed)
    env = make_env()
    trajectories, shares = [], []
    for _ in range(2 * int(n_pairs)):
        share = float(behaviours[int(rng.integers(len(behaviours)))])
        trajectories.append(_behaviour_trajectory(env, share, rng, max_steps))
        shares.append(share)
    first, second = trajectories[0::2], trajectories[1::2]
    true_first = np.array([centered_reward(t).sum() for t in first])
    true_second = np.array([centered_reward(t).sum() for t in second])
    return {"first": first, "second": second, "true_first": true_first, "true_second": true_second,
            "preferred": np.where(true_first >= true_second, 0, 1).astype(int),
            "behaviours": np.array(shares).reshape(-1, 2)}


def _reward_model_class():
    torch = _torch()
    nn = torch.nn

    class RewardModel(nn.Module):
        """The reward model of the reference: obs (N, 4) -> r (N,), an MLP 4 -> hidden -> hidden -> 1 (tanh) on the
        observations divided by OBS_SCALE."""

        def __init__(self, hidden: int = 32):
            super().__init__()
            self.register_buffer("scale", torch.tensor(OBS_SCALE, dtype=torch.float32))
            self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 1))

        def forward(self, obs):
            return self.net(obs / self.scale).squeeze(-1)

    return RewardModel


def reward_model(hidden: int = 32, seed: int | None = None):
    """A new reward model of the reference (``torch.manual_seed(seed)`` first when ``seed`` is given)."""
    if seed is not None:
        _torch().manual_seed(seed)
    return _reward_model_class()(hidden)


def trajectory_returns(model, trajectories):
    """The learned return of each trajectory: the sum of ``model(states)`` over its steps, a tensor (M,) with its
    graph (one forward pass on all the states at once)."""
    torch = _torch()
    lengths = [len(t) for t in trajectories]
    rewards = model(torch.as_tensor(np.concatenate(trajectories), dtype=torch.float32))
    index = torch.as_tensor(np.repeat(np.arange(len(trajectories)), lengths))
    return torch.zeros(len(trajectories), dtype=rewards.dtype).index_add(0, index, rewards)


def train_reward_model(pairs: dict, seed: int = 0, epochs: int = 200, lr: float = 3e-3, hidden: int = 32):
    """The reference reward model: ``epochs`` full-batch Adam steps on the Bradley-Terry loss
    ``mean(-logsigmoid(R(chosen) - R(rejected)))`` of the pairs, R the learned return of a trajectory. Returns
    ``(model, losses)``."""
    torch = _torch()
    F = torch.nn.functional
    with single_thread():
        model = reward_model(hidden, seed=seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        first_wins = torch.as_tensor(np.asarray(pairs["preferred"]) == 0)
        losses = []
        for _ in range(int(epochs)):
            r1, r2 = trajectory_returns(model, pairs["first"]), trajectory_returns(model, pairs["second"])
            loss = -F.logsigmoid(torch.where(first_wins, r1 - r2, r2 - r1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return model.eval(), losses


def preference_accuracy(model, pairs: dict) -> float:
    """The share of the pairs (those whose true returns differ) that the learned returns put in the true order."""
    torch = _torch()
    with torch.no_grad():
        r1 = trajectory_returns(model, pairs["first"]).numpy()
        r2 = trajectory_returns(model, pairs["second"]).numpy()
    t1, t2 = np.asarray(pairs["true_first"]), np.asarray(pairs["true_second"])
    keep = t1 != t2
    return float(np.mean((r1 > r2)[keep] == (t1 > t2)[keep])) if keep.any() else float("nan")


def policy_trajectories(policy, n_episodes: int, seed: int, max_steps: int = EPISODE_STEPS) -> list:
    """The observations (T_i, 4) that follow each step of ``n_episodes`` episodes played by ``policy`` (a function,
    or a network made deterministic as in ``evaluate``), the episode seeds drawn as ``evaluate`` draws them."""
    if hasattr(policy, "parameters"):
        policy = greedy_logits(policy) if hasattr(policy, "actor") else greedy_q(policy)
    env = make_env()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_episodes)):
        state, info = env.reset(seed=int(rng.integers(2**31)))
        states = []
        for _ in range(max_steps):
            state, _, terminated, truncated, info = env.step(policy(state, info))
            states.append(np.array(state, dtype=np.float32))
            if terminated or truncated:
                break
        out.append(np.stack(states))
    return out


def rlhf_measure(policy, model, n_episodes: int = 20, seed: int = EVAL_SEED) -> dict:
    """What an agent earns under the learned reward and under the true one, on ``n_episodes`` deterministic episodes
    of 500 steps at most: ``length`` (mean), ``true_return`` and ``learned_return`` (means of the sums over each
    episode), ``true_per_step`` and ``learned_per_step``, ``abs_x`` (mean |x| of the visited states) and
    ``onpolicy_accuracy``: the preference accuracy of the model on pairs made of the agent's own trajectories cut at
    ``RLHF_STEPS`` steps (what a reward model has never seen)."""
    torch = _torch()
    episodes = policy_trajectories(policy, n_episodes, seed)
    with torch.no_grad():
        learned = [float(model(torch.as_tensor(e)).sum()) for e in episodes]
    true = [float(centered_reward(e).sum()) for e in episodes]
    lengths = [len(e) for e in episodes]
    cut = [e[:RLHF_STEPS] for e in episodes]
    half = len(cut) // 2
    pairs = {"first": cut[:half], "second": cut[half:2 * half],
             "true_first": np.array([centered_reward(e).sum() for e in cut[:half]]),
             "true_second": np.array([centered_reward(e).sum() for e in cut[half:2 * half]])}
    return {"length": float(np.mean(lengths)), "true_return": float(np.mean(true)),
            "learned_return": float(np.mean(learned)), "true_per_step": float(np.sum(true) / np.sum(lengths)),
            "learned_per_step": float(np.sum(learned) / np.sum(lengths)),
            "abs_x": float(np.mean(np.abs(np.concatenate(episodes)[:, 0]))),
            "onpolicy_accuracy": preference_accuracy(model, pairs)}


# ---------------------------------------------------------------------------------------------------------------------
# The challenge B8.28: a budget of environment steps counted by the infrastructure, a held-out evaluation
# ---------------------------------------------------------------------------------------------------------------------
CHALLENGE_BUDGET = 100_000      # environment steps, all copies, every CartPole created during the run
CHALLENGE_SEEDS = (280, 281, 282)
CHALLENGE_EVAL_SEED = 28_028    # held out: no training and no other evaluation of the notebook use it
CHALLENGE_GUARDED = ("train_dqn", "train_reinforce", "train_ppo", "reference_agent", "load_agent", "precomputed")
CHALLENGE_MIN_STEPS = 10_000    # a policy that comes out of fewer counted steps is not a trained agent
CHALLENGE_NEAR = 1e-3           # an agent within this relative L2 distance of an agent that existed before is a copy


class BudgetExceeded(RuntimeError):
    """Raised by a CartPole step beyond the budget of the challenge."""


class ChallengeRuleBroken(RuntimeError):
    """Raised during the challenge by a forbidden call (a reference trainer, a checkpoint, torch.load, async copies)."""


def evaluation_seeds(seed: int, n_episodes: int) -> set:
    """The episode seeds that ``evaluate(policy, n_episodes, seed)`` draws (the held-out episodes of the challenge)."""
    rng = np.random.default_rng(seed)
    return {int(rng.integers(2**31)) for _ in range(int(n_episodes))}


@contextlib.contextmanager
def counted_steps(budget: int, held: set | None = None):
    """Count every step of every CartPole of the process inside the block, however it was created (``gym.make``,
    ``gym.make_vec`` in "sync" mode, ``make_env``, ``make_vec``, a copy of an environment, ``env.unwrapped``):
    ``CartPoleEnv.step`` and ``CartPoleVectorEnv.step`` are patched at the class level (the vector version counts
    one step per copy). The step that would go beyond ``budget`` raises ``BudgetExceeded``. ``"async"`` copies, which
    step in other processes, raise ``ChallengeRuleBroken``, and so does a ``reset`` with one of the seeds of ``held``
    (the held-out evaluation episodes). Yields a dict whose ``"steps"`` is the count."""
    from gymnasium.envs.classic_control import cartpole

    counter = {"steps": 0, "budget": int(budget)}
    held = set(held or ())
    single, vector = cartpole.CartPoleEnv.step, cartpole.CartPoleVectorEnv.step
    single_reset, vector_reset = cartpole.CartPoleEnv.reset, cartpole.CartPoleVectorEnv.reset
    async_init = gym.vector.AsyncVectorEnv.__init__

    def refuse_held(seeds):
        if held & set(seeds):
            raise ChallengeRuleBroken("cette graine est celle d'un épisode de l'évaluation tenue à l'écart : aucun "
                                      "entraînement ne s'en sert")

    def reset(self, *args, seed=None, **kwargs):
        if seed is not None:
            refuse_held([int(seed)])
        return single_reset(self, *args, seed=seed, **kwargs)

    def vector_reset_(self, *args, seed=None, **kwargs):
        if seed is not None:
            refuse_held([int(s) for s in seed] if isinstance(seed, (list, tuple)) else
                        [int(seed) + i for i in range(self.num_envs)])
        return vector_reset(self, *args, seed=seed, **kwargs)

    def step(self, action):
        if counter["steps"] >= counter["budget"]:
            raise BudgetExceeded(f"les {counter['budget']} pas d'environnement du budget sont épuisés")
        counter["steps"] += 1
        return single(self, action)

    def vector_step(self, action):
        if counter["steps"] + self.num_envs > counter["budget"]:
            raise BudgetExceeded(f"les {counter['budget']} pas d'environnement du budget sont épuisés")
        counter["steps"] += self.num_envs
        return vector(self, action)

    def refuse_async(self, *args, **kwargs):
        raise ChallengeRuleBroken("vectorization_mode='async' joue dans d'autres processus, hors du compteur : "
                                  "utilise 'sync'")

    cartpole.CartPoleEnv.step, cartpole.CartPoleVectorEnv.step = step, vector_step
    cartpole.CartPoleEnv.reset, cartpole.CartPoleVectorEnv.reset = reset, vector_reset_
    gym.vector.AsyncVectorEnv.__init__ = refuse_async
    try:
        yield counter
    finally:
        cartpole.CartPoleEnv.step, cartpole.CartPoleVectorEnv.step = single, vector
        cartpole.CartPoleEnv.reset, cartpole.CartPoleVectorEnv.reset = single_reset, vector_reset
        gym.vector.AsyncVectorEnv.__init__ = async_init


def _parameters_of(module) -> list:
    return [p.detach().cpu().clone() for p in module.state_dict().values()]


def _same_parameters(a: list, b: list, near: float = 0.0) -> bool:
    """True when the two lists of tensors are equal, or, with ``near`` > 0, when their relative L2 distance (over all
    the tensors) is at most ``near``: a copy with a little noise added."""
    torch = _torch()
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        return False
    if near <= 0:
        return all(torch.equal(x, y) for x, y in zip(a, b))
    gap = sum(float((x.double() - y.double()).pow(2).sum()) for x, y in zip(a, b))
    size = sum(float(y.double().pow(2).sum()) for y in b)
    return gap <= near ** 2 * max(size, 1e-12)


def _holders_of_modules(namespace: dict) -> list:
    """The names of the notebook's globals that hold a torch module: the module itself, or a dict, list or tuple with
    at least one module among its values (one level deep)."""
    torch = _torch()
    names = []
    for name, obj in list(namespace.items()):
        if name.startswith("_"):
            continue
        values = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else [obj]
        try:
            if any(isinstance(v, torch.nn.Module) for v in values):
                names.append(name)
        except Exception:  # noqa: BLE001 - an exotic container: not an agent
            continue
    return names


def _known_agents(namespace: dict) -> dict:
    """The trained modules that already exist before a run: the torch modules of the notebook's globals and the
    checkpoints of this chapter (the agent of B8.20, the reference agents), as lists of parameters."""
    torch = _torch()
    known = {}
    for name in _holders_of_modules(namespace):
        obj = namespace[name]
        items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, (list, tuple)) else [("", obj)]
        for key, value in items:
            if isinstance(value, torch.nn.Module):
                known[f"{name}[{key!r}]" if key != "" else name] = _parameters_of(value)
    for path in sorted(checkpoints_dir().glob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            known[path.name] = [v.detach().cpu().clone() for v in payload["state_dict"].values()]
        except Exception:  # noqa: BLE001 - an unreadable file holds no agent to compare with
            continue
    return known


def _shown(names, at_most: int = 3) -> str:
    """A few names of masked globals for a message: the constants (AGENT_20...) first, then « … »."""
    ordered = sorted(names, key=lambda n: (not n.isupper(), n))
    return ", ".join(ordered[:at_most]) + (", …" if len(ordered) > at_most else "")


def _deterministic(policy, eval_seed: int, n_states: int = 50) -> bool:
    """True when the function ``policy`` gives the same action twice on each of ``n_states`` states of an episode
    played by the rule of 208.1 (a seed that is not an evaluation seed of the challenge, after the run: not counted)."""
    probe = make_env()
    state, info = probe.reset(seed=int(eval_seed) - 1)
    for _ in range(n_states):
        if policy(state.copy(), dict(info)) != policy(state.copy(), dict(info)):
            return False
        state, _, terminated, truncated, info = probe.step(rule_policy(state, info))
        if terminated or truncated:
            state, info = probe.reset()
    return True


def run_challenge(train_fn, seed: int, *, budget: int = CHALLENGE_BUDGET, namespace: dict | None = None,
                  eval_seed: int = CHALLENGE_EVAL_SEED, n_episodes: int = 100,
                  min_steps: int = CHALLENGE_MIN_STEPS) -> dict:
    """One run of the challenge B8.28: ``train_fn(seed, budget)`` trains from scratch and returns a policy (a torch
    module, evaluated by its most probable action, or a deterministic function ``(state, info) -> int``).

    During the run: every CartPole step is counted (``counted_steps``; beyond ``budget``, ``BudgetExceeded`` stops
    the run); a reset with the seed of an evaluation episode, the trainers and checkpoints of this module
    (``CHALLENGE_GUARDED``, under any name) and ``torch.load`` raise ``ChallengeRuleBroken``; the torch modules of
    ``namespace`` (the notebook's globals, such as the agent of B8.20) and the dicts, lists or tuples that hold one,
    are set to None, then restored. After the run: a policy that comes out of fewer than ``min_steps`` counted steps,
    a module whose parameters are those of an agent that existed before (within ``CHALLENGE_NEAR``), and a function
    that is not deterministic are refused; the policy is evaluated on ``n_episodes`` episodes drawn from
    ``eval_seed`` (held out), 500 steps at most. Returns ``seed``, ``steps``, ``seconds``, ``mean_return``, ``min_return``, ``solved`` (mean >=
    475) and ``error`` (None, or the reason the run does not count, in French: the notebook shows it)."""
    import torch

    from wb.nbtools import ReferenceCalled, watched

    namespace = namespace if namespace is not None else {}
    known = _known_agents(namespace)
    masked = {name: namespace[name] for name in _holders_of_modules(namespace)}
    module = globals()
    saved = {name: module[name] for name in CHALLENGE_GUARDED}
    loads = (torch.load, torch.serialization.load)

    def guard(name):
        def refuse(*args, **kwargs):
            raise ChallengeRuleBroken(f"{name} est interdit pendant le défi : on entraîne à partir de rien, avec les "
                                      "pas du budget")
        return refuse

    result = {"seed": int(seed), "steps": 0, "seconds": 0.0, "mean_return": float("nan"), "min_return": float("nan"),
              "solved": False, "error": None}
    policy = None
    start = time.perf_counter()
    try:
        for name in masked:
            namespace[name] = None
        for name in CHALLENGE_GUARDED:
            module[name] = guard(f"wbr.{name}")
        torch.load = torch.serialization.load = guard("torch.load")
        with counted_steps(budget, held=evaluation_seeds(eval_seed, n_episodes)) as counter:
            try:
                with watched({**{f"wbr.{name}": saved[name] for name in CHALLENGE_GUARDED}, "torch.load": loads[0]}):
                    policy = train_fn(seed, budget)        # the same functions under another name (an alias) too
            except BudgetExceeded as exc:
                result["error"] = f"budget dépassé : {exc}"
            except ChallengeRuleBroken as exc:
                result["error"] = f"règle enfreinte : {exc}"
            except ReferenceCalled as exc:
                result["error"] = (f"règle enfreinte : {exc.label} est interdit pendant le défi (appelé sous un autre "
                                   "nom) : on entraîne à partir de rien, avec les pas du budget")
            result["steps"] = counter["steps"]
    finally:
        torch.load, torch.serialization.load = loads
        module.update(saved)
        namespace.update(masked)
        result["seconds"] = time.perf_counter() - start
    if result["error"] is None:
        if policy is None or not callable(policy):
            result["error"] = ("train_fn doit rendre une politique : un module torch, ou une fonction (state, info) -> "
                               "int" + (f" ; pendant le défi, les agents du notebook ({_shown(masked)}) valent None : "
                                        "entraîne le tien" if masked else ""))
        elif isinstance(policy, torch.nn.Module) and any(
                _same_parameters(_parameters_of(policy), params, near=CHALLENGE_NEAR) for params in known.values()):
            same = next(name for name, params in known.items()
                        if _same_parameters(_parameters_of(policy), params, near=CHALLENGE_NEAR))
            result["error"] = (f"l'agent rendu a les paramètres d'un agent entraîné avant le run ({same}), au bruit "
                               "près : on entraîne à partir de rien")
        elif result["steps"] < min_steps:
            result["error"] = (f"{result['steps']} pas d'environnement joués, moins de {min_steps} : la "
                               "politique rendue doit sortir d'un entraînement sur CartPole (une règle écrite à la "
                               "main n'est pas un agent entraîné)")
        elif not isinstance(policy, torch.nn.Module) and not _deterministic(policy, eval_seed):
            result["error"] = ("la politique rendue n'est pas déterministe : deux appels sur le même état rendent des "
                               "actions différentes (rends l'action la plus probable)")
    if result["error"] is None:
        with torch.no_grad():
            ev = evaluate(policy, n_episodes, eval_seed)
        result.update(mean_return=ev["mean_return"], min_return=float(min(ev["returns"])),
                      solved=ev["mean_return"] >= SOLVED)
    return result


# ---------------------------------------------------------------------------------------------------------------------
# The code review B8.27: CleanRL's ppo.py at a pinned commit (MIT licence), read, never installed
# ---------------------------------------------------------------------------------------------------------------------
CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
CLEANRL_PPO_URL = f"https://raw.githubusercontent.com/vwxyzjn/cleanrl/{CLEANRL_COMMIT}/cleanrl/ppo.py"
CLEANRL_PPO_PAGE = f"https://github.com/vwxyzjn/cleanrl/blob/{CLEANRL_COMMIT}/cleanrl/ppo.py"
CLEANRL_PPO_SHA256 = "d8dfbb7ac0b21e747b77d6e6673e566db4d6cdacc83bcebcd19cf87ad0e459ae"
CLEANRL_PPO_LINES = 312


def cleanrl_ppo_lines(download: bool = True) -> list | None:
    """The lines of CleanRL's ``ppo.py`` at the commit ``CLEANRL_COMMIT`` (312 lines, MIT licence), downloaded once into
    the cache and checked by its SHA-256; None when it cannot be read (no network, another file). Line ``k`` of the
    file is ``lines[k - 1]``."""
    import hashlib

    from wb import datasets

    path = datasets.cache_dir() / "cleanrl" / f"ppo_{CLEANRL_COMMIT[:7]}.py"
    if not path.exists() and download:
        try:
            datasets.download(CLEANRL_PPO_URL, path, timeout=20, retries=1)
        except ConnectionError:
            return None
    if not path.exists():
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CLEANRL_PPO_SHA256:
        return None
    return data.decode("utf-8").splitlines()
