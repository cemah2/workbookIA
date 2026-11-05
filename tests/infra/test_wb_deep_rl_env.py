"""Tests of wb.deep_rl_env: environments and physics, seeds, the evaluation of a policy, the reference agents (DQN,
REINFORCE, PPO) and their pieces, the checkpoint of an agent and the precomputed protocols (bonus chapter B8)."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from wb import deep_rl_env as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _reference(module: str):
    """A module of the reference library, imported by its own path (learner mode has no solutions/ on sys.path)."""
    if str(ROOT / "solutions") not in sys.path:
        sys.path.insert(0, str(ROOT / "solutions"))
    return importlib.import_module(f"mylearn_ref.{module}")


# ----------------------------------------------------------------------------------------------- environments, physics
def test_make_env_is_cartpole_v1_with_its_time_limit_and_threshold():
    env = R.make_env()
    assert env.spec.max_episode_steps == R.EPISODE_STEPS == 500
    assert env.spec.reward_threshold == R.SOLVED == 475.0
    assert R.make_env(max_episode_steps=15).spec.max_episode_steps == 15


def test_set_physics_recomputes_the_derived_constants():
    env = R.make_env(gravity=19.6, length=1.0)
    p = R.physics_of(env)
    assert p["gravity"] == 19.6 and p["length"] == 1.0
    assert p["polemass_length"] == pytest.approx(0.1 * 1.0) and p["total_mass"] == pytest.approx(1.1)
    heavy = R.physics_of(R.make_env(masspole=0.5))
    assert heavy["total_mass"] == pytest.approx(1.5) and heavy["polemass_length"] == pytest.approx(0.25)
    assert R.physics_of(R.make_env())["polemass_length"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="unknown physics"):
        R.make_env(friction=1.0)
    with pytest.raises(ValueError, match="positive"):
        R.make_env(gravity=-1.0)
    envs = R.make_vec(2, physics={"length": 1.0})
    assert all(e.unwrapped.polemass_length == pytest.approx(0.1) for e in envs.envs)


def test_make_vec_seeds_the_copies_and_keeps_an_explicit_next_step_autoreset():
    envs = R.make_vec(3, seed=42)
    for i in range(3):
        single, _ = gym.make("CartPole-v1").reset(seed=42 + i)
        assert np.allclose(envs.first_obs[i], single)
    assert envs.metadata["autoreset_mode"] == gym.vector.AutoresetMode.NEXT_STEP
    assert R.make_vec(2, autoreset="same_step").metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    with pytest.raises(ValueError, match="autoreset"):
        R.make_vec(2, autoreset="never")
    short = R.make_vec(1, seed=0, max_episode_steps=3)
    flags = []
    for _ in range(4):
        _, reward, term, trunc, _ = short.step(np.array([0]))
        flags.append((float(reward[0]), bool(term[0]), bool(trunc[0])))
    assert flags[2][2] or flags[2][1], "the third step ends the episode (time limit of 3)"
    assert flags[3] == (0.0, False, False), "next-step autoreset: the call after an end only resets the copy"


def test_step_counter_counts_terminations_and_truncations_apart():
    env = R.StepCounter(R.make_env(max_episode_steps=10), keep=5)
    env.reset(seed=0)
    for _ in range(200):
        _, _, term, trunc, _ = env.step(int(env.steps % 2))
        if term or trunc:
            env.reset()
    assert env.steps == 200 and env.truncations > 0
    assert env.terminations + env.truncations in (env.resets - 1, env.resets)
    assert len(env.transitions) == 5 and len(env.transitions[0]) == 6


# ------------------------------------------------------------------------------------------------ policies, evaluation
def test_evaluate_plays_the_same_episodes_as_rl_evaluate_policy():
    rl = _reference("rl")
    for seed in (0, 13):
        act = np.random.default_rng(seed)
        want = rl.evaluate_policy(R.make_env(), lambda s, i: int(act.integers(2)), 30, max_steps=500,
                                  rng=np.random.default_rng(seed))
        got = R.evaluate(R.random_policy(seed), 30, seed)
        assert got["mean_return"] == want["mean_return"] and got["std_return"] == pytest.approx(want["std_return"])
        assert got["mean_length"] == want["mean_length"] and len(got["returns"]) == 30


def test_evaluate_reaches_the_time_limit_with_the_rule_and_cuts_at_max_steps():
    assert R.evaluate(R.rule_policy, 5, 0)["mean_length"] == 500
    assert R.evaluate(R.rule_policy, 5, 0, max_steps=200)["mean_return"] == 200
    assert R.evaluate(R.rule_policy, 3, 0, length=1.0)["mean_length"] > 0     # another physics
    with pytest.raises(TypeError):
        R.evaluate(42)


def test_greedy_policies_return_python_ints_and_accept_a_network():
    q = R.q_network(16, seed=0)
    action = R.greedy_q(q)(np.zeros(4, np.float32), {})
    assert type(action) is int and action in (0, 1)
    agent = R.ActorCritic(16)
    assert type(R.greedy_logits(agent)(np.zeros(4, np.float32), {})) is int
    assert R.evaluate(q, 2, 0)["mean_length"] >= 1 and R.evaluate(agent, 2, 0)["mean_length"] >= 1


def test_random_policy_draws_from_one_seeded_generator():
    p = R.random_policy(7)
    assert [p(None, {}) for _ in range(20)] == [int(a) for a in np.random.default_rng(7).integers(2, size=20)]


def test_tabular_reference_is_the_table_of_chapter_26_and_holds_500_steps():
    make, Q = R.tabular_reference()
    assert Q.shape == (12, 2)
    env = make()
    assert env.observation_space.n == 12
    rl = _reference("rl")
    assert rl.evaluate_policy(env, Q.argmax(axis=1), 5, max_steps=500, rng=np.random.default_rng(0))["mean_length"] == 500


def test_moving_average_and_band():
    assert np.allclose(R.moving_average([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
    assert R.moving_average([], 3).size == 0
    b = R.band([[1, 5, 3], [3, 1, 3], [2, 2, 9]])
    assert np.allclose(b["median"], [2, 2, 3]) and np.allclose(b["low"], [1, 1, 3]) and np.allclose(b["high"], [3, 5, 9])


# --------------------------------------------------------------------------------------------------------------- DQN
def test_linear_epsilon():
    assert R.linear_epsilon(0, 1000, 1.0, 0.05, 0.5) == 1.0
    assert R.linear_epsilon(250, 1000, 1.0, 0.05, 0.5) == pytest.approx(0.525)
    assert R.linear_epsilon(500, 1000, 1.0, 0.05, 0.5) == pytest.approx(0.05)
    assert R.linear_epsilon(900, 1000, 1.0, 0.05, 0.5) == pytest.approx(0.05)


def test_dqn_targets_of_the_variants_carry_no_gradient():
    online, target = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)
    with torch.no_grad():
        online.weight.zero_(); online.bias.copy_(torch.tensor([4.1, 3.0, 0.5]))
        target.weight.zero_(); target.bias.copy_(torch.tensor([2.0, 3.5, 1.2]))
    s2, r, d = torch.zeros(2, 4), torch.tensor([1.0, 1.0]), torch.tensor([False, True])
    assert torch.allclose(R.dqn_targets(online, target, s2, r, d, 0.9), torch.tensor([4.15, 1.0]))
    assert torch.allclose(R.dqn_targets(online, target, s2, r, d, 0.9, "double"), torch.tensor([2.8, 1.0]))
    assert torch.allclose(R.dqn_targets(online, target, s2, r, d, 0.9, "no_target"), torch.tensor([1 + 0.9 * 4.1, 1.0]))
    assert not R.dqn_targets(online, target, s2, r, d, 0.9).requires_grad
    with pytest.raises(ValueError, match="variant"):
        R.dqn_targets(online, target, s2, r, d, 0.9, "rainbow")


def test_train_dqn_is_reproducible_and_a_prefix_replays_the_recorded_run():
    a, ha = R.train_dqn(3, 1500, eval_every=500, eval_episodes=3)
    b, hb = R.train_dqn(3, 1500, eval_every=500, eval_episodes=3, stop_after=1000)
    assert hb["eval_returns"] == ha["eval_returns"][:2]
    _, hc = R.train_dqn(3, 1500, eval_every=500, eval_episodes=3)
    assert hc["episode_returns"] == ha["episode_returns"] and hc["eval_returns"] == ha["eval_returns"]
    assert sum(ha["episode_returns"]) == ha["episode_ends"][-1]
    assert ha["settings"]["capacity"] == R.DQN_SETTINGS["capacity"] and ha["variant"] == "dqn"


def test_train_dqn_takes_the_pieces_of_the_learner():
    calls = []

    def targets(online, frozen, s2, r, d, gamma, variant):
        calls.append(variant)
        return R.dqn_targets(online, frozen, s2, r, d, gamma, variant)

    R.train_dqn(0, 1200, eval_every=0, targets_fn=targets, capacity=500, variant="double")
    assert calls and set(calls) == {"double"}


# --------------------------------------------------------------------------------------------------------- REINFORCE
def test_discounted_returns_and_reinforce_weights():
    rl = _reference("rl")
    r = [1.0, 0.0, 2.0, 1.0]
    assert np.allclose(R.discounted_returns(r, 0.5), rl.returns_to_go(r, 0.5))
    G = np.array([3.0, 2.0, 1.0])
    assert np.allclose(R.reinforce_weights(G, "none", []), G)
    assert np.allclose(R.reinforce_weights(G, "mean", []), G)                       # no past episode: b = 0
    assert np.allclose(R.reinforce_weights(G, "mean", [np.array([1.0, 3.0])]), G - 2.0)
    assert np.allclose(R.reinforce_weights(G, "normalize", []), (G - 2) / (G.std() + 1e-8))
    with pytest.raises(ValueError):
        R.reinforce_weights(G, "median", [])


def test_episode_gradients_match_a_direct_autograd_and_leave_the_network_alone():
    net = R.policy_network(8, seed=0)
    episodes = R.collect_episodes(net, 4, 11)
    assert all(np.array_equal(a["actions"], b["actions"]) for a, b in zip(episodes, R.collect_episodes(net, 4, 11)))
    before = [p.detach().clone() for p in net.parameters()]
    grads = R.episode_gradients(net, episodes, "mean")
    assert grads.shape == (4, sum(p.numel() for p in net.parameters())) and grads.dtype == np.float64
    assert all(torch.equal(p, q) for p, q in zip(before, net.parameters())) and all(p.grad is None for p in net.parameters())
    returns = [R.discounted_returns(e["rewards"], 0.99) for e in episodes]
    b = float(np.mean(np.concatenate(returns)))
    e = episodes[1]
    logp = torch.log_softmax(net(torch.as_tensor(e["states"])), 1)[torch.arange(len(e["actions"])), e["actions"]]
    (-(logp * torch.as_tensor(returns[1] - b, dtype=torch.float32)).mean()).backward()
    want = torch.cat([p.grad.reshape(-1) for p in net.parameters()]).double().numpy()
    assert np.allclose(grads[1], want, rtol=1e-5, atol=1e-7)
    g = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    assert R.gradient_spread(g) == pytest.approx(np.mean(np.sum((g - g.mean(0)) ** 2, axis=1)))


def test_train_reinforce_is_reproducible():
    _, a = R.train_reinforce(1, 5, "mean")
    _, b = R.train_reinforce(1, 5, "mean")
    assert a["episode_returns"] == b["episode_returns"] and len(a["episode_returns"]) == 5


# --------------------------------------------------------------------------------------------------------------- PPO
def test_gae_numpy_and_explained_variance_numpy_match_the_reference_library():
    deep = _reference("deep_rl")
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=(7, 3)) for _ in range(3)] + [rng.random((7, 3)) < 0.2, rng.random((7, 3)) < 0.2]
    for got, want in zip(R.gae_numpy(*arrays, 0.97, 0.9), deep.compute_gae(*arrays, 0.97, 0.9)):
        assert np.allclose(got, want)
    y, p = rng.normal(size=20), rng.normal(size=20)
    assert R.explained_variance_numpy(p, y) == pytest.approx(deep.explained_variance(p, y))
    assert np.isnan(R.explained_variance_numpy([1.0, 2.0], [3.0, 3.0]))


def test_collect_rollout_leaves_out_the_reset_steps_and_bootstraps_on_the_real_last_observation():
    torch.manual_seed(0)
    agent = R.ActorCritic(16)
    envs = R.make_vec(2, seed=1, max_episode_steps=4)
    gen = torch.Generator().manual_seed(0)
    finished = []
    batch, obs, just_reset = R.collect_rollout(agent, envs, envs.first_obs, np.zeros(2, bool), 9, gen, np.zeros(2),
                                               finished)
    ends = batch["terminated"] | batch["truncated"]
    assert ends.any()
    for t in range(1, 9):
        assert np.array_equal(batch["valid"][t], ~ends[t - 1])
    with torch.no_grad():
        nv = agent.value(torch.as_tensor(batch["next_obs"].reshape(-1, 4))).numpy().reshape(9, 2)
    assert np.allclose(batch["next_values"], nv)
    assert np.allclose(batch["rewards"][~batch["valid"]], 0.0)
    assert len(finished) == int((ends & batch["valid"]).sum())
    assert np.array_equal(just_reset, ends[-1])


def test_ppo_loss_parts():
    torch.manual_seed(0)
    agent = R.ActorCritic(8)
    obs, actions = torch.randn(6, 4), torch.tensor([0, 1, 1, 0, 1, 0])
    with torch.no_grad():
        old = torch.log_softmax(agent(obs), 1).gather(1, actions[:, None]).squeeze(1)
    adv, ret = torch.randn(6), torch.randn(6)
    loss, parts = R.ppo_loss(agent, obs, actions, old, adv, ret)
    assert parts["approx_kl"] == pytest.approx(0.0, abs=1e-7) and parts["clipfrac"] == 0.0
    assert parts["policy_loss"] == pytest.approx(float(-adv.mean()), abs=1e-6)
    s = R.PPO_SETTINGS
    assert float(loss) == pytest.approx(parts["policy_loss"] + s["vf_coef"] * parts["value_loss"]
                                        - s["ent_coef"] * parts["entropy"], abs=1e-6)


def test_train_ppo_logs_and_reward_fn():
    settings = {"n_steps": 32, "n_epochs": 1}
    agent, logs = R.train_ppo(0, 2 * 4 * 32, settings)
    assert all(len(logs[k]) == 2 for k in R.LOG_KEYS) and logs["step"] == [128, 256]
    seen = []

    def reward_fn(obs, actions, next_obs):
        seen.append(obs.shape)
        return np.zeros(len(obs))

    _, logs_r = R.train_ppo(0, 4 * 32, settings, reward_fn=reward_fn)
    assert seen and seen[0] == (4, 4)
    assert logs_r["mean_return"][0] == logs["mean_return"][0], "mean_return keeps the true rewards"


# ------------------------------------------------------------------------------------------------- agent checkpoints
def test_agent_checkpoint_roundtrip_key_and_corruption(tmp_path):
    torch.manual_seed(0)
    agent = R.ActorCritic(8)
    key = R.agent_key({"a": 1}, source="learner", fast=True, seed=0, total_steps=10)
    path = R.save_agent(key, agent, {"step": [1]}, folder=tmp_path, extra={"eval": 3.0})
    assert path.name.startswith("ppo-learner-fast-")
    loaded, info = R.load_agent(key, folder=tmp_path)
    assert all(torch.equal(p, q) for p, q in zip(agent.parameters(), loaded.parameters()))
    assert info["logs"] == {"step": [1]} and info["extra"] == {"eval": 3.0} and not loaded.training
    other = R.agent_key({"a": 1}, source="solution", fast=True, seed=0, total_steps=10)
    assert R.agent_file(other, tmp_path) != path and R.load_agent(other, folder=tmp_path) is None
    loaded_b, _ = R.load_agent(key, folder=tmp_path, build=lambda: R.ActorCritic(8))
    assert torch.equal(next(loaded_b.parameters()), next(agent.parameters()))
    path.write_bytes(b"not a checkpoint")
    assert R.load_agent(key, folder=tmp_path) is None
    assert R.delete_agent(key, folder=tmp_path) and not R.delete_agent(key, folder=tmp_path)
    with pytest.raises(ValueError, match="source"):
        R.agent_key({}, source="teacher", fast=True, seed=0, total_steps=1)


# --------------------------------------------------------------------------------------------- precomputed protocols
def test_precomputed_protocols_have_their_protocol():
    with pytest.raises(KeyError):
        R.precomputed("nothing")
    a15 = R.precomputed("a15")
    assert set(a15["protocol"]["variants"]) == set(R.DQN_VARIANTS)
    assert all(f"{v}/{s}" in a15["runs"] for v in R.DQN_VARIANTS for s in a15["protocol"]["seeds"])
    p16 = R.precomputed("p16")
    assert all(len(p16["runs"][f"{b}/{s}"]["episode_returns"]) == p16["protocol"]["episodes"]
               for b in R.BASELINES for s in p16["protocol"]["seeds"])
    assert set(R.precomputed("c17")["state_dict"]) == set(R.policy_network(64).state_dict())
