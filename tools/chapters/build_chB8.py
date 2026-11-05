#!/usr/bin/env python
"""Build the two notebooks of bonus chapter B8 from a single source (used by Claude).

    python tools/chapters/build_chB8.py
    python tools/run_all_notebooks.py chapitres/b8_rl_moderne/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/b8_rl_moderne/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3 and of the ✏️ paper exercises
B8.1 to B8.6 (02_exercices.md). The derivation ∂ B8.7, the estimate 🧮 B8.8, the reflection (🗣️ B8.9, ⚖️ B8.10,
📄 B8.11) and the interview are corrected with 05_solutions.md. The answers are computed by hand formulas in the
context cell of the solutions notebook, then checked against the libraries: the reference ``mylearn.deep_rl``
(compute_gae, clipped_surrogate, bradley_terry_loss, dpo_loss) and ``mylearn.rl`` (ReplayBuffer), PyTorch
(F.huber_loss, F.smooth_l1_loss, log_softmax and autograd, torch.optim.Adam, clip_grad_norm_, torch.min/clamp,
F.logsigmoid, torch.distributions.Categorical), and gymnasium (CartPole-v1 spaces, the seeds of reset and of the
action space), with exact enumerations for the overestimation quiz: part 0 downloads nothing and trains no agent.
The notebook exercises come in three phases: A, the value (B8.12 to B8.15); B, the policy (B8.16 to B8.24); C, the
preferences (B8.25 to B8.28). Session 78 writes them in two halves: B8.12 to B8.20 (chB8_parts.py, chB8_code.py,
infrastructure wb.deep_rl_env), then B8.21 to B8.28 (NEXT_SESSION); history_chB8.py precomputes the calibrations and the
multi-seed protocols (tools/chapters/history_chB8.json, src/wb/deep_rl_protocols.json).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (Paper, badge, check_unique_keys, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "B8"
FOLDER = "chapitres/b8_rl_moderne"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = r'''# The short answers only use the numbers of their statements (02_exercices.md)
import math
from itertools import product

import numpy as np


def sigmoid(u):
    return 1.0 / (1.0 + math.exp(-u))


def softplus(u):                                       # -log sigmoid(-u) = log(1 + e^u)
    return math.log1p(math.exp(u))


# quiz Q1: CartPole discretised into 6 intervals per variable, 2 actions ; a Q network on a batch of 32 states
Q1A = 6 ** 4 * 2
# quiz Q3: a full memory of 10 000 transitions, one replaced per step, 64 distinct transitions drawn per update
CAP_Q3, BATCH_Q3 = 10_000, 64
Q3A = CAP_Q3 * BATCH_Q3 / CAP_Q3                       # steps of stay x probability of being drawn at each step
Q3D = (1 - 0.005) ** 200                                # weight left on the target weights of 200 steps ago
# quiz Q4: three actions of value 0, estimates +1 or -1 with probability 1/2, independently
OUTCOMES_Q4 = list(product([-1, 1], repeat=3))
Q4B = float(np.mean([max(o) for o in OUTCOMES_Q4]))
Q4C = float(np.mean([second[int(np.argmax(first))] for first in OUTCOMES_Q4 for second in OUTCOMES_Q4]))
# quiz Q5: 6 motors with 10 values each ; a softmax with logits (ln 2, ln 6, 0)
Q5B = 10 ** 6
LOGITS_Q5 = np.array([math.log(2), math.log(6), 0.0])
Q5C = float(np.exp(LOGITS_Q5)[1] / np.exp(LOGITS_Q5).sum())
# quiz Q6: a biased coin, p(heads) = theta = 0.3 ; the score function d/dtheta log p(x)
THETA_Q6 = 0.3
SCORE_TAILS_Q6 = -1 / (1 - THETA_Q6)
Q6B = THETA_Q6 * (1 / THETA_Q6) + (1 - THETA_Q6) * SCORE_TAILS_Q6
# quiz Q7: one TD error ; the weight of delta_(t+10) with gamma = 0.99 and lambda = 0.95
Q7A = 1 + 0.9 * 2.5 - 3.0
Q7D = (0.99 * 0.95) ** 10
# quiz Q8: N = 8 environments, T = 256 steps, 4 epochs, mini-batches of 256 ; a ratio from two log-probabilities
Q8C = 4 * (8 * 256) // 256
Q8D = math.exp(-0.9 - (-1.2))
# quiz Q10: an evaluation cut at max_steps = 300 ; the standard error of a mean of 100 returns
Q10A, Q10B = 300, 40 / math.sqrt(100)
# quiz Q11: GRPO advantages in a group (1, 0, 0, 0), population standard deviation
REWARDS_Q11 = np.array([1.0, 0.0, 0.0, 0.0])
Q11C = float((REWARDS_Q11[0] - REWARDS_Q11.mean()) / REWARDS_Q11.std())
Q11C_DDOF1 = float((REWARDS_Q11[0] - REWARDS_Q11.mean()) / REWARDS_Q11.std(ddof=1))
# recall R1: one Q-learning update
Q_R1, R_R1, GAMMA_R1, MAX_R1, ALPHA_R1 = 0.5, 0.3, 0.8, 0.6, 0.2
R1A = R_R1 + GAMMA_R1 * MAX_R1
R1B = Q_R1 + ALPHA_R1 * (R1A - Q_R1)
R1C = Q_R1 + ALPHA_R1 * (R_R1 - Q_R1)
# recall R2: the first step of Adam with bias correction ; clipping by the global norm (PyTorch adds 1e-6)
W_R2, G_R2, LR_R2, B1_R2, B2_R2, EPS_R2 = 1.0, -40.0, 1e-3, 0.9, 0.999, 1e-8
M_R2, V_R2 = (1 - B1_R2) * G_R2, (1 - B2_R2) * G_R2 ** 2
R2A = W_R2 - LR_R2 * (M_R2 / (1 - B1_R2)) / (math.sqrt(V_R2 / (1 - B2_R2)) + EPS_R2)
R2A_NO_CORRECTION = W_R2 - LR_R2 * M_R2 / (math.sqrt(V_R2) + EPS_R2)
GRADS_R2 = [np.array([3.0, 4.0]), np.array([12.0])]
NORM_R2 = math.sqrt(sum(float((g ** 2).sum()) for g in GRADS_R2))
R2B = 3.0 * min(1.0, 0.5 / (NORM_R2 + 1e-6))
# recall R3: epsilon-greedy over 2 actions ; a softmax with temperature 0.5 ; an incremental mean
R3A = 1 - 0.2 + 0.2 / 2
R3B = float(np.exp(1.5 / 0.5) / (np.exp(1.0 / 0.5) + np.exp(1.5 / 0.5)))
R3C = 4 + (8 - 4) / 4

# ✏️ B8.1: three transitions, gamma = 0.99, r = 1 ; transition 2 is terminated, transition 3 truncated
GAMMA_B81 = 0.99
QT_B81 = np.array([[12.0, 13.7], [4.0, 3.1], [40.2, 41.6]])        # Q_target(s', .)
QO_B81 = np.array([[12.6, 12.2], [2.5, 3.8], [42.5, 41.7]])        # Q_online(s', .)
TERM_B81 = np.array([False, True, False])
PRED_B81 = np.array([15.0, 2.9, 45.3])                              # Q_online(s, a) of the played action
DQN_B81 = 1 + GAMMA_B81 * (1 - TERM_B81) * QT_B81.max(axis=1)
DDQN_B81 = 1 + GAMMA_B81 * (1 - TERM_B81) * QT_B81[np.arange(3), QO_B81.argmax(axis=1)]
GAP_B81 = float(np.mean(DQN_B81 - DDQN_B81))
AGREE_B81 = bool(QT_B81[0].argmax() == QO_B81[0].argmax())
X3_B81 = float(PRED_B81[2] - DQN_B81[2])

# ✏️ B8.2: three gaps x = Q - y, Huber threshold 1.5
X_B82, DELTA_B82 = np.array([0.6, -2.4, 3.2]), 1.5


def huber_b82(x, delta=DELTA_B82):
    x = np.asarray(x, float)
    return np.where(np.abs(x) <= delta, 0.5 * x ** 2, delta * (np.abs(x) - delta / 2))


def huber_grad_b82(x, delta=DELTA_B82):
    return np.clip(np.asarray(x, float), -delta, delta)


TERMS_B82 = huber_b82(X_B82)
MEAN_B82 = float(TERMS_B82.mean())
GRAD3_B82 = float(huber_grad_b82(X_B82)[2] / 3)
GRAD3_MSE_B82 = float(X_B82[2] / 3)
SMOOTH_B82 = MEAN_B82 / DELTA_B82                                   # smooth_l1_loss(beta) = huber(delta=beta) / beta
SAME_GRAD_B82 = bool(huber_grad_b82(0.6) == 0.6)

# ✏️ B8.3: softmax policy with logits (0.5, 1.5, -0.5), action 2 played, advantage -2, step 0.1
Z_B83, A_B83, ADV_B83, STEP_B83 = np.array([0.5, 1.5, -0.5]), 2, -2.0, 0.1


def softmax_b83(z):
    e = np.exp(np.asarray(z, float) - np.max(z))
    return e / e.sum()


PI_B83 = softmax_b83(Z_B83)
GRAD_B83 = np.eye(3)[A_B83] - PI_B83                                # grad_z log softmax(z)_a = e_a - softmax(z)
Z_NEW_B83 = Z_B83 + STEP_B83 * ADV_B83 * GRAD_B83
PI_NEW_B83 = softmax_b83(Z_NEW_B83)
ENTROPY_B83 = float(-(PI_B83 * np.log(PI_B83)).sum())

# ✏️ B8.4: four steps, gamma = 0.9, truncated at step 1 (no terminated step), the rollout ends after step 3
R_B84 = np.ones(4)
V_B84 = np.array([4.0, 3.8, 2.0, 2.4])
NEXT_B84 = np.array([3.8, 3.6, 2.4, 2.6])
TERM_B84 = np.zeros(4, dtype=bool)
TRUNC_B84 = np.array([False, True, False, False])
GAMMA_B84 = 0.9


def gae_b84(lam, terminated=TERM_B84, truncated=TRUNC_B84, values=V_B84, next_values=NEXT_B84):
    """Backward loop of 208.4: delta_t = r + gamma (1 - term) V(s') - V(s), A_t = delta_t + gamma lam (1 - done) A_(t+1)."""
    deltas = R_B84 + GAMMA_B84 * (1 - terminated) * next_values - values
    adv, future = np.zeros(4), 0.0
    for t in range(3, -1, -1):
        future = deltas[t] + GAMMA_B84 * lam * (1 - (terminated[t] or truncated[t])) * future
        adv[t] = future
    return deltas, adv


DELTA_B84, ADV0_B84 = gae_b84(0.0)
_, ADV95_B84 = gae_b84(0.95)
_, ADV1_B84 = gae_b84(1.0)
RET95_B84 = ADV95_B84 + V_B84
_, ADV95_AS_END_B84 = gae_b84(0.95, terminated=TRUNC_B84, truncated=TERM_B84)    # truncation taken as a true end
_, ADV95_NO_STOP_B84 = gae_b84(0.95, truncated=TERM_B84)                          # the recursion does not stop
MC_B84 = bool(math.isclose(ADV1_B84[0] + V_B84[0], 1 + GAMMA_B84 * 1 + GAMMA_B84 ** 2 * 3.6))

# ✏️ B8.5: four transitions, eps = 0.2 (then 0.3)
RATIO_B85 = np.array([1.4, 0.6, 0.75, 1.35])
ADV_B85 = np.array([2.0, 1.5, -1.2, -0.8])


def clip_b85(eps):
    unclipped = RATIO_B85 * ADV_B85
    clipped = np.clip(RATIO_B85, 1 - eps, 1 + eps) * ADV_B85
    terms = np.minimum(unclipped, clipped)
    grad = np.where(unclipped <= clipped, -ADV_B85 / 4, 0.0)
    return terms, -float(terms.mean()), grad


TERMS_B85, LOSS_B85, GRAD_B85 = clip_b85(0.2)
ZERO_B85 = int((GRAD_B85 == 0).sum())
CLIPFRAC_B85 = float(np.mean(np.abs(RATIO_B85 - 1) > 0.2))
ZERO_EPS3_B85 = int((clip_b85(0.3)[2] == 0).sum())
# a classic mistake for d): the max instead of the min
LOSS_MAX_B85 = -float(np.maximum(RATIO_B85 * ADV_B85, np.clip(RATIO_B85, 0.8, 1.2) * ADV_B85).mean())

# ✏️ B8.6: Bradley-Terry on one pair, then DPO
RW_B86, RL_B86 = 2.3, 1.1
P_B86 = sigmoid(RW_B86 - RL_B86)
LOSS_B86 = softplus(-(RW_B86 - RL_B86))
GRAD_W_B86 = -(1 - P_B86)
LOSS_SWAP_B86 = softplus(RW_B86 - RL_B86)
LOGP_B86 = {"pc": -35.2, "pr": -38.9, "rc": -36.0, "rr": -37.5}


def dpo_b86(beta):
    chosen = LOGP_B86["pc"] - LOGP_B86["rc"]
    rejected = LOGP_B86["pr"] - LOGP_B86["rr"]
    margin = beta * (chosen - rejected)
    return beta * chosen, margin, softplus(-margin), -beta * sigmoid(-margin)


IMPLICIT_B86, MARGIN_B86, DPO_B86, DPO_GRAD_B86 = dpo_b86(0.1)
DPO5_B86 = dpo_b86(0.5)[2]
'''

LIBRARY_CHECKS = r'''
# The same answers from the libraries (no download, no training): the reference mylearn.deep_rl and mylearn.rl,
# PyTorch (losses, autograd, Adam, clip_grad_norm_, Categorical) and gymnasium (CartPole-v1, seeds)
import warnings

import gymnasium as gym
import torch
import torch.nn.functional as F

DRL, RL = mylearn.deep_rl, mylearn.rl

# quiz Q1: CartPole-v1 has 4 state variables and 2 actions ; a Q network rends one value per action
env = gym.make("CartPole-v1")
assert env.observation_space.shape == (4,) and env.action_space.n == 2
assert env.spec.max_episode_steps == 500 and env.spec.reward_threshold == 475.0
assert Q1A == 6 ** env.observation_space.shape[0] * env.action_space.n
torch.manual_seed(0)
q_net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
assert tuple(q_net(torch.zeros(32, 4)).shape) == (32, 2)

# quiz Q1 d): the semi-gradient (target detached) and the full gradient differ
s, s_next = torch.randn(1, 4), torch.randn(1, 4)
semi = (q_net(s)[0, 0] - (1 + 0.9 * q_net(s_next).max().detach())) ** 2
full = (q_net(s)[0, 0] - (1 + 0.9 * q_net(s_next).max())) ** 2
g_semi = torch.autograd.grad(semi, list(q_net.parameters()))
g_full = torch.autograd.grad(full, list(q_net.parameters()))
assert any((a - b).abs().max() > 1e-6 for a, b in zip(g_semi, g_full))

# quiz Q3 a): simulate the full memory of rl.ReplayBuffer, 64 distinct transitions per update
buffer = RL.ReplayBuffer(capacity=1_000, obs_shape=(4,), obs_dtype="float32", random_state=0)
for i in range(1_000):
    buffer.push(np.zeros(4, np.float32), 0, float(i), np.zeros(4, np.float32), False)
draws = np.zeros(1_000)
for _ in range(2_000):
    batch = buffer.sample(64)
    assert len(np.unique(batch["rewards"])) == 64                     # distinct transitions within a mini-batch
    draws[batch["rewards"].astype(int)] += 1
assert abs(draws.mean() * 1_000 / 2_000 - BATCH_Q3) < 1               # per 1 000 steps of stay: 64 draws
buffer.push(np.ones(4, np.float32), 1, -1.0, np.ones(4, np.float32), True)
assert len(buffer) == 1_000 and -1.0 in buffer.sample(1_000)["rewards"] and 0.0 not in buffer.sample(1_000)["rewards"]

# quiz Q4: Monte Carlo check of the exact enumerations
rng = np.random.default_rng(0)
est = rng.choice([-1, 1], size=(200_000, 2, 3))
assert abs(est[:, 0].max(axis=1).mean() - Q4B) < 0.01
assert abs(est[np.arange(200_000), 1, est[:, 0].argmax(axis=1)].mean() - Q4C) < 0.01

# quiz Q5 c), quiz R3 b): softmax by torch
assert math.isclose(torch.softmax(torch.tensor(LOGITS_Q5), 0)[1].item(), Q5C, rel_tol=1e-9)
assert math.isclose(torch.softmax(torch.tensor([1.0, 1.5], dtype=torch.float64) / 0.5, 0)[1].item(), R3B)

# quiz Q6: the score of a Bernoulli by autograd
theta = torch.tensor(THETA_Q6, dtype=torch.float64, requires_grad=True)
torch.log(1 - theta).backward()
assert math.isclose(theta.grad.item(), SCORE_TAILS_Q6) and abs(Q6B) < 1e-12

# quiz Q7: one TD error and the GAE weight with the reference compute_gae (lambda = 0 gives the TD error)
adv, _ = DRL.compute_gae([1.0], [3.0], [2.5], [False], [False], gamma=0.9, lam=0.0)
assert math.isclose(adv[0], Q7A)
impulse = np.zeros(11)
impulse[10] = 1.0                                  # a TD error of 1 at step 10 only: r = delta, V = 0
adv, _ = DRL.compute_gae(impulse, np.zeros(11), np.zeros(11), np.zeros(11, bool), np.zeros(11, bool), gamma=0.99, lam=0.95)
assert math.isclose(adv[0], Q7D)

# quiz Q8 f), ✏️ B8.5: the clipped objective by torch.min/torch.clamp and autograd, and the reference
for eps, (terms, loss, grad) in ((0.2, clip_b85(0.2)), (0.3, clip_b85(0.3))):
    ratio = torch.tensor(RATIO_B85, requires_grad=True)
    adv_t = torch.tensor(ADV_B85)
    loss_t = -torch.min(ratio * adv_t, ratio.clamp(1 - eps, 1 + eps) * adv_t).mean()
    loss_t.backward()
    assert math.isclose(loss_t.item(), loss) and np.allclose(ratio.grad.numpy(), grad)
    ref_loss, ref_grad = DRL.clipped_surrogate(RATIO_B85, ADV_B85, clip_eps=eps)
    assert math.isclose(ref_loss, loss) and np.allclose(ref_grad, grad)
_, grad_zero_adv = DRL.clipped_surrogate([0.3, 1.0, 2.5], [0.0, 0.0, 0.0])
assert np.all(grad_zero_adv == 0)

# quiz Q10: gymnasium seeds ; evaluate_policy cut at max_steps (a perfect rule of 208.1)
env.reset(seed=3)
first = env.reset(seed=3)[0]
assert not np.allclose(env.reset()[0], first)                     # reset() without a seed continues the sequence
env.reset(seed=3)
a1 = [env.action_space.sample() for _ in range(20)]
env.reset(seed=3)
a2 = [env.action_space.sample() for _ in range(20)]
assert a1 != a2                                                   # reset(seed=...) does not seed the action space
rule = lambda state, info: int(state[2] + 0.5 * state[3] > 0)    # noqa: E731
assert RL.evaluate_policy(env, rule, n_episodes=5, max_steps=Q10A, rng=np.random.default_rng(1))["mean_return"] == Q10A

# quiz Q11: the GRPO advantages of a group
assert math.isclose(Q11C, 3 / math.sqrt(3)) and math.isclose(Q11C_DDOF1, 1.5)

# recall R2: Adam and clip_grad_norm_ in PyTorch
w = torch.nn.Parameter(torch.tensor([W_R2], dtype=torch.float64))
opt = torch.optim.Adam([w], lr=LR_R2, betas=(B1_R2, B2_R2), eps=EPS_R2)
w.grad = torch.tensor([G_R2], dtype=torch.float64)
opt.step()
assert math.isclose(w.item(), R2A, rel_tol=1e-12)
p1, p2 = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))
p1.grad, p2.grad = torch.tensor([3.0, 4.0]), torch.tensor([12.0])
assert math.isclose(torch.nn.utils.clip_grad_norm_([p1, p2], max_norm=0.5).item(), NORM_R2)
assert math.isclose(p1.grad[0].item(), R2B, rel_tol=1e-6)
assert math.isclose(float(torch.cat([p1.grad, p2.grad]).norm()), 0.5, rel_tol=1e-5)

# ✏️ B8.1: the targets in PyTorch (max, then argmax + gather), under no_grad
with torch.no_grad():
    qt, qo = torch.tensor(QT_B81), torch.tensor(QO_B81)
    term = torch.tensor(TERM_B81)
    y = 1 + GAMMA_B81 * (1 - term.double()) * qt.max(dim=1).values
    y_double = 1 + GAMMA_B81 * (1 - term.double()) * qt.gather(1, qo.argmax(dim=1, keepdim=True)).squeeze(1)
assert np.allclose(y.numpy(), DQN_B81) and np.allclose(y_double.numpy(), DDQN_B81)

# ✏️ B8.2: F.huber_loss and F.smooth_l1_loss, values and gradients
pred = torch.tensor(X_B82, requires_grad=True)
target = torch.zeros(3, dtype=torch.float64)
huber = F.huber_loss(pred, target, delta=DELTA_B82)
huber.backward()
assert math.isclose(huber.item(), MEAN_B82) and math.isclose(pred.grad[2].item(), GRAD3_B82)
assert np.allclose(F.huber_loss(pred.detach(), target, delta=DELTA_B82, reduction="none").numpy(), TERMS_B82)
assert math.isclose(F.smooth_l1_loss(pred.detach(), target, beta=DELTA_B82).item(), SMOOTH_B82)
mse = pred.detach().clone().requires_grad_()
(0.5 * mse ** 2).mean().backward()
assert math.isclose(mse.grad[2].item(), GRAD3_MSE_B82)

# ✏️ B8.3: log_softmax and autograd, one step, Categorical's entropy
z = torch.tensor(Z_B83, requires_grad=True)
torch.log_softmax(z, dim=0)[A_B83].backward()
assert np.allclose(z.grad.numpy(), GRAD_B83)
assert np.allclose(torch.softmax(torch.tensor(Z_NEW_B83), 0).numpy(), PI_NEW_B83)
assert math.isclose(torch.distributions.Categorical(logits=torch.tensor(Z_B83)).entropy().item(), ENTROPY_B83)

# ✏️ B8.4: the reference compute_gae for the three lambdas, and the two classic mistakes
for lam, expected in ((0.0, ADV0_B84), (0.95, ADV95_B84), (1.0, ADV1_B84)):
    adv, ret = DRL.compute_gae(R_B84, V_B84, NEXT_B84, TERM_B84, TRUNC_B84, gamma=GAMMA_B84, lam=lam)
    assert np.allclose(adv, expected) and np.allclose(ret, expected + V_B84)
adv, _ = DRL.compute_gae(R_B84, V_B84, NEXT_B84, TRUNC_B84, TERM_B84, gamma=GAMMA_B84, lam=0.95)
assert np.allclose(adv, ADV95_AS_END_B84)

# ✏️ B8.6: the reference losses and F.logsigmoid with autograd
loss, g_w, g_l = DRL.bradley_terry_loss([RW_B86], [RL_B86])
assert math.isclose(loss, LOSS_B86) and math.isclose(g_w[0], GRAD_W_B86) and math.isclose(g_l[0], -GRAD_W_B86)
assert math.isclose(DRL.bradley_terry_loss([RW_B86 + 10], [RL_B86 + 10])[0], LOSS_B86, rel_tol=1e-9)
assert math.isclose(DRL.bradley_terry_loss([RL_B86], [RW_B86])[0], LOSS_SWAP_B86)
pc = torch.tensor([LOGP_B86["pc"]], dtype=torch.float64, requires_grad=True)
for beta, expected in ((0.1, DPO_B86), (0.5, DPO5_B86)):
    loss_d, margins = DRL.dpo_loss([LOGP_B86["pc"]], [LOGP_B86["pr"]], [LOGP_B86["rc"]], [LOGP_B86["rr"]], beta=beta)
    assert math.isclose(loss_d, expected)
margin_t = 0.1 * ((pc - LOGP_B86["rc"]) - (LOGP_B86["pr"] - LOGP_B86["rr"]))
(-F.logsigmoid(margin_t)).mean().backward()
assert math.isclose(pc.grad.item(), DPO_GRAD_B86) and math.isclose(margin_t.item(), MARGIN_B86)
env.close()
print("✅ the paper answers agree with mylearn.deep_rl, mylearn.rl, PyTorch and gymnasium")
'''

exec(PAPER_CONTEXT)  # the builder computes a few mistakes with the same constants

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
LETTERS = 'the letters, e.g. "EF"'
WHOLE = "a whole number"
SHAPE = 'a shape as a string, e.g. "(3, 5)"'
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
DEC4 = "a number, 4 decimals"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


def several(answer: str, options: str, where: str) -> str:
    """Mistakes for a question answered by several letters: every other non-empty set of the options, grouped into
    three messages that name neither the letter in excess nor the missing one (« il manque », « en trop », both)."""
    groups = {"missing": [], "extra": [], "both": []}
    for size in range(1, len(options) + 1):
        for combo in combinations(options, size):
            chosen = "".join(combo)
            if chosen == answer:
                continue
            missing, extra = set(answer) - set(chosen), set(chosen) - set(answer)
            groups["both" if missing and extra else "missing" if missing else "extra"].append(chosen)
    messages = {"missing": f"il manque au moins une lettre : relis {where}",
                "extra": f"au moins une des lettres choisies n'a pas sa place : relis {where}",
                "both": f"il manque au moins une lettre, et une autre n'a pas sa place : relis {where}"}
    return "mistakes={" + ", ".join(f'"{messages[k]}": {set(v)!r}' for k, v in groups.items() if v) + "}"


PAPER = [
    Paper("B8.Q1", "Pourquoi une table Q ne suffit plus pour CartPole", [
        ("a", WHOLE, "Q1A", r'''mistakes={"une case par état discrétisé et par action : n'oublie pas les actions": 6 ** 4,
          "les intervalles des quatre variables se combinent : ils se multiplient, ils ne s'additionnent pas": 48,
          "6 intervalles pour chacune des 4 variables : vérifie la base et l'exposant": 4 ** 6 * 2,
          "relis la question : 6 intervalles par variable, et quatre variables": 24}'''),
        ("b", WHOLE, "1", r'''mistakes={"relis ce que rend le réseau de 208.1 : une valeur pour un couple (état, action), ou un vecteur ?": 2,
          "il faut bien calculer les valeurs pour choisir : relis « Pourquoi la table ne suffit plus » (§208.1)": 0}'''),
        ("c", TF, "True", r'''mistakes={"relis « Pourquoi la table ne suffit plus » (§208.1) : que partagent deux états voisins dans un réseau ?": False}'''),
        ("d", TF, "False", r'''mistakes={"relis « Apprendre une valeur par régression » (§208.1) : que devient la méthode si la cible est dérivée aussi ?": True}'''),
        ("e", SHAPE, '"(32, 2)"', r'''mistakes={"le réseau rend une valeur par action : combien d'actions a CartPole ?": "(32, 1)",
          "4 est la dimension de l'entrée : que rend le réseau ?": "(32, 4)",
          "relis la question : un batch de 32 états passe d'un coup, pas un seul état": "(2,)",
          "la forme commence par l'axe des exemples": "(2, 32)"}'''),
    ]),
    Paper("B8.Q2", "La triade mortelle : approximation, amorçage et apprentissage off-policy", [
        ("a", LETTERS, '"ACD"', several("ACD", "ABCDE", "« Instabilité : la triade mortelle » (§208.1)")),
        ("b", WHOLE, "2", r'''mistakes={"une table est-elle une approximation de fonction ? Relis la triade (§208.1)": 3,
          "relis la cible du Q-learning (ch. 26) : amorce-t-elle ? Et apprend-elle la valeur de la politique qui explore ?": 1,
          "compte chaque ingrédient séparément : approximation, amorçage, off-policy": 0}'''),
        ("c", WHOLE, "1", r'''mistakes={"des retours observés jusqu'à la fin d'un épisode : y a-t-il une estimation dans la cible ?": 2,
          "un réseau qui estime V est-il une approximation de fonction ?": 0,
          "relis les trois ingrédients : les transitions viennent-elles d'une autre politique ?": 3}'''),
        ("d", WHOLE, "2", r'''mistakes={"SARSA apprend la valeur de la politique qu'il suit (ch. 26) : est-ce off-policy ?": 3,
          "la cible de SARSA contient Q(s', a') : est-ce un amorçage ? Et un réseau, une approximation ?": 1,
          "compte chaque ingrédient séparément : approximation, amorçage, off-policy": 0}'''),
        ("e", TF, "False", r'''mistakes={"relis la fin de « Instabilité : la triade mortelle » (§208.1) : DQN renonce-t-il à l'un des trois ?": True}'''),
        ("f", TF, "True", r'''mistakes={"relis le troisième ingrédient (§208.1) : de quelle politique viennent les transitions de la mémoire ?": False}'''),
    ]),
    Paper("B8.Q3", "Réseau cible et mémoire de rejeu : quel problème chacun résout-il ?", [
        ("a", WHOLE, "round(Q3A)", r'''mistakes={"une mise à jour tire 64 transitions, pas une seule : ta probabilité d'être tirée à chaque pas en tient-elle compte ?": 1,
          "relis la question : combien de fois une même transition est-elle tirée, pas combien de mini-batches tient la mémoire": 156,
          "une transition reste dans la mémoire plus d'un pas : combien de pas, et quelle chance d'être tirée à chacun ?": 0},
          fractional="compte les tirages d'une transition pendant tout son séjour : un nombre entier en moyenne ici"'''),
        ("b", TF, "False", r'''mistakes={"refais le calcul de a) avec 1 000 : la durée du séjour et la probabilité d'être tirée changent toutes les deux": True}'''),
        ("c", TF, "True", r'''mistakes={"relis « Les deux stabilisateurs » (§208.2) : quels poids calculent la cible entre deux copies ?": False}'''),
        ("d", DEC2, "Q3D", r'''decimals=2, mistakes={"les parts se multiplient d'un pas à l'autre, elles ne se retranchent pas": 0.0,
          "relis la question : 200 pas, pas un seul": 0.995,
          "relis la question : la part des poids d'il y a 200 pas qui reste, pas celle de ce qui a été ajouté depuis": 1 - Q3D}'''),
        ("e", TF, "True", r'''mistakes={"relis « Les deux stabilisateurs » (§208.2) : combien de fois sert chaque transition de la mémoire ?": False}'''),
    ]),
    Paper("B8.Q4", "Double DQN : d'où vient la surestimation des valeurs Q ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "chaque estimation est juste en moyenne, mais leur maximum ? Relis « La surestimation, et Double DQN » (§208.2)",
                                           "C": "le max choisit l'estimation la plus haute : dans quel sens penche-t-il ?"})),
        ("b", DEC2, "Q4B", r'''decimals=2, mistakes={"chaque estimation est juste en moyenne, mais le maximum des trois ?": 0.0,
          "compte les huit cas de trois estimations, pas les quatre cas de deux": 0.5,
          "le maximum vaut −1 dans un cas : lequel, et avec quelle probabilité ?": 1.0,
          "relis la question : l'espérance du maximum, pas la probabilité qu'il vaille +1": 0.875}'''),
        ("c", WHOLE, "round(Q4C)", r'''mistakes={"le bruit qui a fait gagner l'action dans le premier jeu est-il celui qu'on lit dans le second ?": 1,
          "les deux jeux sont indépendants : relis « La surestimation, et Double DQN » (§208.2)": -1},
          fractional="la valeur lue ne profite plus du bruit qui a fait gagner l'action : son espérance est un entier ici"'''),
        ("d", LETTER, '"A"', letters("A", {"B": "relis la formule de Double DQN (§208.2) : quel réseau prend l'argmax ?",
                                           "C": "Double DQN n'ajoute aucun réseau : relis la fin de « La surestimation, et Double DQN »"})),
        ("e", TF, "False", r'''mistakes={"relis la fin de « La surestimation, et Double DQN » (§208.2) : combien de paramètres le changement ajoute-t-il ?": True}'''),
        ("f", TF, "True", r'''mistakes={"la politique gloutonne prend l'argmax : que lui fait une même constante ajoutée à toutes les valeurs ?": False}'''),
    ]),
    Paper("B8.Q5", "Méthodes par valeur ou par politique : actions continues et politiques aléatoires", [
        ("a", LETTER, '"C"', letters("C", {"A": "calcule b) : combien de colonnes aurait cette table, et que perd-on entre deux valeurs d'un moteur ?",
                                           "B": "un DQN prend un argmax sur une liste d'actions : relis la première raison de « Apprendre la politique elle-même » (§208.3)"})),
        ("b", WHOLE, "Q5B", r'''mistakes={"les choix des six moteurs se combinent : ils se multiplient": 60,
          "relis : 10 valeurs pour chacun des 6 moteurs (vérifie la base et l'exposant)": 6 ** 10,
          "relis la question : 10 valeurs possibles par moteur": 2 ** 6}'''),
        ("c", DEC2, "Q5C", r'''decimals=2, mistakes={"c'est la probabilité d'une autre action : relis la numérotation à partir de 0": 2 / 9,
          "la softmax passe par l'exponentielle des logits, pas par les logits eux-mêmes": math.log(6) / (math.log(2) + math.log(6)),
          "relis la question : trois actions, et l'exponentielle de 0 vaut 1": 0.75,
          "trois logits différents ne donnent pas trois probabilités égales": 1 / 3}'''),
        ("d", LETTERS, '"ABD"', several("ABD", "ABCD", "« Apprendre la politique elle-même » (§208.3) et la mémoire de rejeu (§208.2)")),
        ("e", TF, "False", r'''mistakes={"si ta politique est déterministe, que fait un adversaire qui l'a devinée ? Relis la deuxième raison de « Apprendre la politique elle-même » (§208.3)": True}'''),
    ]),
    Paper("B8.Q6", "REINFORCE : pourquoi soustraire une baseline ne biaise pas le gradient", [
        ("a", DEC2, "SCORE_TAILS_Q6", r'''decimals=2, mistakes={"c'est le score de pile : relis la question": 1 / THETA_Q6,
          "vérifie le signe : ln(1 − θ) baisse quand θ monte": -SCORE_TAILS_Q6,
          "dérive ln(1 − θ), pas ln θ": -1 / THETA_Q6}'''),
        ("b", DEC2, "Q6B", r'''decimals=2, mistakes={"pondère chaque issue par sa probabilité": 1 / THETA_Q6 + SCORE_TAILS_Q6,
          "relis la question : l'espérance sur les deux issues, pas le score d'une seule": SCORE_TAILS_Q6}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "une moyenne d'épisodes passés dépend-elle de l'action jouée maintenant ? Relis « La baseline et la variance » (§208.3)",
                                           "B": "un critique dépend de l'état, pas de l'action : relis « La baseline et la variance » (§208.3)"})),
        ("d", TF, "True", r'''mistakes={"une constante ne dépend pas de l'action : que devient l'espérance ? Et (G − b)², quand b est loin des retours ?": False}'''),
        ("e", TF, "False", r'''mistakes={"relis la fin de « REINFORCE » (§208.3) : l'estimation de REINFORCE a-t-elle un biais à corriger ?": True}'''),
    ]),
    Paper("B8.Q7", "Avantage, erreur TD et GAE : ce que règle λ", [
        ("a", DEC2, "Q7A", r'''decimals=2, mistakes={"γ multiplie la valeur de l'état suivant": 1 + 2.5 - 3.0,
          "l'épisode continue : la cible amorce sur V(s_{t+1})": 1 - 3.0,
          "l'erreur TD retranche la valeur de l'état actuel": 1 + 0.9 * 2.5,
          "vérifie le signe : la cible moins la valeur prévue": -Q7A}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "relis les cas particuliers de GAE (§208.4) : quel λ redonne G_t − V(s_t) ?",
                                           "C": "relis la formule de GAE (§208.4) : que reste-t-il de la somme quand γλ = 0 ?"})),
        ("c", LETTER, '"C"', letters("C", {"A": "relis les cas particuliers de GAE (§208.4) : quel réglage porte le hasard de tout le reste de l'épisode ?",
                                           "B": "λ = 0 ne garde qu'un pas de hasard : relis « L'erreur TD, première estimation de l'avantage »"})),
        ("d", DEC2, "Q7D", r'''decimals=2, mistakes={"le poids combine γ et λ, à chaque pas": 0.95 ** 10,
          "le poids combine γ et λ, à chaque pas : λ manque": 0.99 ** 10,
          "relis la question : dix pas plus loin, pas un seul": 0.99 * 0.95,
          "vérifie l'exposant : δ_{t+10} est dix pas après δ_t": (0.99 * 0.95) ** 11}'''),
        ("e", TF, "False", r'''mistakes={"relis le cas λ = 1 (§208.4) : que reste-t-il du critique dans G_t − V(s_t) ?": True}'''),
        ("f", TF, "True", r'''mistakes={"relis « Fin d'épisode ou limite de temps » (§208.4) : la récursion continue-t-elle après un pas tronqué ?": False}'''),
    ]),
    Paper("B8.Q8", "PPO : à quoi sert le clip, et pourquoi réutiliser plusieurs fois le même lot ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis « Réutiliser un rollout sans tout casser » (§208.5) : le clip rend-il l'estimation exacte ?",
                                           "C": "relis le début de « Réutiliser un rollout sans tout casser » (§208.5) : un gradient de politique est-il off-policy ?"})),
        ("b", TF, "False", r'''mistakes={"relis le cas de l'avantage négatif (§208.5) : quel terme est le minimum quand le ratio est monté ?": True}'''),
        ("c", WHOLE, "Q8C", r'''mistakes={"relis la question : 4 epochs, pas une seule": Q8C // 4,
          "chaque epoch découpe tout le rollout en mini-batches : combien en tient-il ?": 4,
          "relis la question : on demande des pas d'Adam, pas des transitions": 8 * 256}'''),
        ("d", DEC2, "Q8D", r'''decimals=2, mistakes={"vérifie le sens du ratio : la nouvelle politique sur l'ancienne": 1 / Q8D,
          "le ratio est l'exponentielle de la différence des log-probabilités": 0.3,
          "on divise des probabilités, pas des log-probabilités": 0.75}'''),
        ("e", TF, "False", r'''mistakes={"relis « Une région de confiance » (§208.5) : comment TRPO résout-il son problème ?": True}'''),
        ("f", TF, "True", r'''mistakes={"relis le gradient par rapport au ratio (§208.5) : que vaut −A/n quand A = 0 ?": False}'''),
    ]),
    Paper("B8.Q9", "On-policy ou off-policy : DQN, REINFORCE et PPO", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis la fin de « REINFORCE » (§208.3) : un épisode sert-il encore après le pas ?",
                                           "C": "PPO corrige l'écart par un ratio : relis « Réutiliser un rollout sans tout casser » (§208.5)"})),
        ("b", TF, "False", r'''mistakes={"relis l'encadré 💼 de 208.5 : quelle famille consomme beaucoup d'interactions, et pourquoi ?": True}'''),
        ("c", TF, "True", r'''mistakes={"relis la fin de « REINFORCE » (§208.3) : pour quelle politique le gradient vaut-il ?": False}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "le ratio de probabilités appartient à PPO : relis la cible de DQN (§208.2)",
                                           "C": "même avec ε = 0,1, la cible reste celle de la politique gloutonne : relis la cible de DQN (§208.2)"})),
        ("e", LETTER, '"C"', letters("C", {"A": "relis l'encadré 💼 de 208.5 : que coûte PPO en interactions, ici où chacune prend du temps réel ?",
                                           "B": "une estimation sans biais mais très dispersée demande beaucoup d'épisodes : relis « La baseline et la variance » (§208.3)"})),
    ]),
    Paper("B8.Q10", "Évaluer un agent : graines, épisodes, politique déterministe", [
        ("a", WHOLE, "Q10A", r'''mistakes={"relis l'encadré ⚠️ « un plafond caché dans l'évaluation » (§208.6) : que fait max_steps ?": 500,
          "relis l'appel : max_steps n'a pas sa valeur par défaut ici": 200,
          "relis la question : une politique parfaite, qui ne tombe jamais": 475}'''),
        ("b", DEC1, "Q10B", r'''decimals=1, mistakes={"relis la question : l'erreur type de la moyenne, pas l'écart-type des retours": 40.0,
          "l'erreur type ne divise pas l'écart-type par le nombre d'épisodes : relis le ch. 2": 0.4}'''),
        ("c", TF, "False", r'''mistakes={"relis « Les graines » (§208.6) : que font les reset() suivants, sans graine ?": True}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "plus d'épisodes d'évaluation réduisent le bruit de l'évaluation, pas celui de l'entraînement : relis « Les graines » (§208.6)",
                                           "C": "garder la meilleure graine, c'est choisir un tirage favorable : relis « Rapporter plusieurs graines » (§208.6)"})),
        ("e", TF, "True", r'''mistakes={"relis « Les graines » (§208.6) : l'espace d'actions a-t-il son propre générateur ?": False}'''),
        ("f", LETTER, '"A"', letters("A", {"B": "relis « Évaluer un agent » (§208.6) : quelle politique évalue-t-on, et sans quel hasard ?",
                                           "C": "V dépend de l'état seul : peut-il choisir une action ?"})),
    ]),
    Paper("B8.Q11", "RLHF, DPO, GRPO : qui a besoin d'un modèle de récompense ou d'un critique ?", [
        ("a", LETTERS, '"A"', several("A", "ABC", "208.7 : DPO apprend-il une récompense à part, et d'où vient celle de GRPO ?")),
        ("b", LETTERS, '"AC"', several("AC", "ABC", "208.7 : lesquelles sont des méthodes de RL, et laquelle est une loss supervisée ?")),
        ("c", DEC2, "Q11C", r'''decimals=2, mistakes={"relis la question : l'écart-type de population, divisé par le nombre de réponses": Q11C_DDOF1,
          "l'avantage divise l'écart à la moyenne par l'écart-type du groupe": 0.75,
          "relis la question : l'avantage de la première réponse, celle qui a la récompense 1": -1 / math.sqrt(3)}'''),
        ("d", TF, "True", r'''mistakes={"relis le dernier piège du tableau de la fiche : π_ref bouge-t-elle ?": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "une récompense 0 ou 1 varie d'une réponse à l'autre : relis « GRPO et les récompenses vérifiables » (§208.7)",
                                           "C": "relis « GRPO et les récompenses vérifiables » (§208.7) : que coûte le critique de PPO ?"})),
        ("f", TF, "True", r'''mistakes={"calcule les avantages d'un groupe de récompenses toutes égales : que pousse le gradient ?": False}'''),
    ]),
    Paper("B8.Q12", "Piratage de la récompense (reward hacking) : quand l'agent optimise la lettre et pas l'esprit", [
        ("a", LETTERS, '"BD"', several("BD", "ABCD", "« Le piratage de la récompense » (§208.7) : l'agent atteint-il la récompense sans faire ce qu'on voulait ?")),
        ("b", LETTER, '"B"', letters("B", {"A": "relis « Le piratage de la récompense » (§208.7) : que fait la vraie récompense après un moment ?",
                                           "C": "relis « Le piratage de la récompense » (§208.7) : la vraie récompense monte-t-elle d'abord ?"})),
        ("c", TF, "False", r'''mistakes={"relis « Le RLHF : PPO sous une laisse KL » (§208.7) : que fait une laisse plus forte ?": True}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "plus d'optimisation contre le même modèle aggrave-t-elle ou corrige-t-elle le piratage ? Relis la courbe de Gao et coll.",
                                           "B": "relis le rôle de la pénalité KL (§208.7) : que se passe-t-il loin des réponses de la référence ?"})),
        ("e", TF, "True", r'''mistakes={"relis le paragraphe sur la flagornerie (§208.7) : d'où viennent les préférences pour une réponse flatteuse ?": False}'''),
    ]),
    Paper("B8.R1", "La mise à jour du Q-learning et le rôle de la mémoire de rejeu", [
        ("a", DEC2, "R1A", r'''decimals=2, mistakes={"la cible prend le maximum de la ligne de s', pas la valeur de la case actuelle": R_R1 + GAMMA_R1 * Q_R1,
          "s' n'est pas terminal : la cible amorce sur la ligne de s'": R_R1,
          "γ multiplie la valeur de la suite": R_R1 + MAX_R1}'''),
        ("b", DEC3, "R1B", r'''decimals=3, mistakes={"α mélange la cible et l'ancienne valeur : relis la mise à jour du ch. 26 (§26.6.1)": R1A,
          "relis la question : la nouvelle valeur de la case, pas l'erreur TD": R1A - Q_R1,
          "relis la question : la nouvelle valeur de la case, pas seulement sa variation": ALPHA_R1 * (R1A - Q_R1)}'''),
        ("c", DEC2, "R1C", r'''decimals=2, mistakes={"un état terminal n'a pas de suite : relis le rappel maths du ch. 26 (§26.6.1)": R1B,
          "α mélange la cible et l'ancienne valeur": R_R1}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "c'est le rejeu du livre : relis l'encadré 🕰️ de 26.9 et la docstring de ReplayBuffer",
                                           "C": "relis la docstring de ReplayBuffer (🔨 26.31) : que range push ?"})),
        ("e", TF, "True", r'''mistakes={"relis la docstring de ReplayBuffer (🔨 26.31) : que devient la transition la plus ancienne ?": False}'''),
        ("f", TF, "False", r'''mistakes={"relis le ⚠️ du ch. 26 sur SARSA (§26.7) : de quelle politique SARSA apprend-il la valeur ?": True}'''),
    ]),
    Paper("B8.R2", "Adam et le gradient clipping", [
        ("a", DEC4, "R2A", r'''decimals=4, mistakes={"c'est le pas de la descente de gradient simple : relis Adam au ch. 19 (§19.5.5)": W_R2 - LR_R2 * G_R2,
          "vérifie le signe : le gradient est négatif, le poids monte": 2 * W_R2 - R2A,
          "relis la correction de biais d'Adam (ch. 19, §19.5.5) : m et v partent de 0": R2A_NO_CORRECTION}'''),
        ("b", DEC3, "R2B", r'''decimals=3, mistakes={"la norme est globale : tous les gradients mis bout à bout, et un seul facteur": 3.0 * 0.5 / 5.0,
          "le clipping par la norme ne borne pas chaque composante à ±0,5": 0.5,
          "la norme globale dépasse 0,5 : le gradient est réduit": 3.0}'''),
        ("c", TF, "False", r'''mistakes={"relis « le gradient clipping » (ch. 19) : à quel moment les gradients existent-ils, et quand sont-ils utilisés ?": True}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "relis la formule d'Adam (ch. 19, §19.5.5) : où ε apparaît-il ?",
                                           "C": "borner la norme est le rôle de max_grad_norm : relis la formule d'Adam"})),
        ("e", TF, "True", r'''mistakes={"relis « le gradient clipping » (ch. 19) : tous les gradients sont-ils multipliés par le même facteur ?": False}'''),
    ]),
    Paper("B8.R3", "Exploration, softmax et bandits", [
        ("a", DEC2, "R3A", r'''decimals=2, mistakes={"en exploration, ε-greedy tire parmi toutes les actions, l'action gloutonne comprise": 0.8,
          "relis la question : la probabilité de jouer l'action gloutonne, pas d'explorer": 0.2,
          "relis la question : deux actions pour CartPole": 1 - 0.2 + 0.2 / 4}'''),
        ("b", DEC3, "R3B", r'''decimals=3, mistakes={"la température divise les valeurs, elle ne les multiplie pas": 1 / (1 + math.exp(-0.25)),
          "relis la question : la température vaut 0,5, pas 1": 1 / (1 + math.exp(-0.5)),
          "la softmax passe par l'exponentielle, pas par Q / ΣQ": 0.6}'''),
        ("c", DEC1, "R3C", r'''decimals=1, mistakes={"compte les récompenses reçues par ce bras, la nouvelle comprise : quel n ?": 4 + 4 / 3,
          "la moyenne incrémentale ajoute (R − Q)/n, elle ne remplace pas Q par R": 8.0,
          "relis la question : la nouvelle moyenne, après la récompense 8": 4.0}'''),
        ("d", TF, "False", r'''mistakes={"relis la softmax du ch. 26 (§26.6.2) : grande température, presque uniforme ou presque gloutonne ?": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "ε-greedy ne monte aucune préférence : relis l'encadré 🔗 de 208.3",
                                           "C": "UCB ajoute un bonus d'incertitude : relis l'encadré 🔗 de 208.3"})),
    ]),
    Paper("B8.1", "Cible DQN à la main : état terminal, puis cible Double DQN", [
        ("a", DEC2, "DQN_B81[0]", r'''decimals=2, mistakes={"DQN prend le max du réseau cible, pas du réseau en ligne": 1 + GAMMA_B81 * QO_B81[0].max(),
          "γ multiplie la valeur de la suite : n'oublie pas la récompense r = 1": GAMMA_B81 * QT_B81[0].max(),
          "relis la question : on demande la cible DQN ; quelle action de s' retient-elle ?": 1 + GAMMA_B81 * QT_B81[0, 0]}'''),
        ("b", DEC2, "DQN_B81[1]", r'''decimals=2, mistakes={"relis « la cible se réduit à r » (§208.2) : s' est-il terminal ?": 1 + GAMMA_B81 * QT_B81[1].max(),
          "même terminale, la transition a reçu sa récompense": 0.0}'''),
        ("c", DEC2, "DQN_B81[2]", r'''decimals=2, mistakes={"relis l'encadré ⚠️ « la limite de temps n'est pas une fin » (§208.2)": 1.0,
          "DQN prend le max du réseau cible, pas du réseau en ligne": 1 + GAMMA_B81 * QO_B81[2].max(),
          "γ multiplie la valeur de la suite : n'oublie pas la récompense r = 1": GAMMA_B81 * QT_B81[2].max()}'''),
        ("d", DEC2, "DDQN_B81[0]", r'''decimals=2, mistakes={"Double DQN lit la valeur dans le réseau cible, pas dans le réseau en ligne": 1 + GAMMA_B81 * QO_B81[0, 0],
          "relis la formule de Double DQN (§208.2) : quel réseau choisit l'action ?": 1 + GAMMA_B81 * QT_B81[0, 1]}'''),
        ("e", DEC2, "DDQN_B81[2]", r'''decimals=2, mistakes={"Double DQN lit la valeur dans le réseau cible, pas dans le réseau en ligne": 1 + GAMMA_B81 * QO_B81[2, 0],
          "relis la formule de Double DQN (§208.2) : quel réseau choisit l'action ?": 1 + GAMMA_B81 * QT_B81[2, 1],
          "relis l'encadré ⚠️ « la limite de temps n'est pas une fin » (§208.2)": 1.0}'''),
        ("f", DEC2, "GAP_B81", r'''decimals=2, mistakes={"relis la question : la moyenne sur les trois transitions, la transition 2 comprise": GAP_B81 * 3 / 2,
          "vérifie le sens : la cible de DQN moins celle de Double DQN": -GAP_B81,
          "relis la question : la moyenne, pas la somme": GAP_B81 * 3}'''),
        ("g", TF, "AGREE_B81", r'''mistakes={"prends l'argmax de chacun des deux réseaux, puis compare": not AGREE_B81}'''),
        ("h", DEC2, "X3_B81", r'''decimals=2, mistakes={"vérifie le sens : la prédiction moins la cible": -X3_B81,
          "relis la question : l'écart à la cible DQN, pas à celle de Double DQN": float(PRED_B81[2] - DDQN_B81[2]),
          "relis l'encadré ⚠️ « la limite de temps n'est pas une fin » (§208.2) : que vaut la cible de la transition 3 ?": float(PRED_B81[2] - 1.0)}'''),
    ]),
    Paper("B8.2", "Loss de Huber et son gradient pour trois erreurs TD", [
        ("a", DEC2, "float(TERMS_B82[0])", r'''decimals=2, mistakes={"|0,6| est sous le seuil : relis la branche quadratique, avec son ½": 0.6 ** 2,
          "|0,6| est sous le seuil : la branche linéaire ne s'applique pas": DELTA_B82 * (0.6 - DELTA_B82 / 2)}'''),
        ("b", DEC3, "float(TERMS_B82[1])", r'''decimals=3, mistakes={"|−2,4| dépasse le seuil : relis la branche linéaire": 0.5 * 2.4 ** 2,
          "la branche linéaire retranche δ/2 à |x|, pas δ": DELTA_B82 * (2.4 - DELTA_B82),
          "vérifie le signe : la loss de Huber prend la valeur absolue": DELTA_B82 * (-2.4 - DELTA_B82 / 2),
          "la branche linéaire multiplie par δ": 2.4 - DELTA_B82 / 2}'''),
        ("c", DEC2, "MEAN_B82", r'''decimals=2, mistakes={"relis l'énoncé : la moyenne des trois termes, pas leur somme": float(TERMS_B82.sum()),
          "c'est la moyenne des ½x² : relis le seuil": float((0.5 * X_B82 ** 2).mean()),
          "tu as divisé par δ : relis l'énoncé, F.huber_loss": MEAN_B82 / DELTA_B82}'''),
        ("d", DEC1, "float(huber_grad_b82(-2.4))", r'''decimals=1, mistakes={"au-delà du seuil, la dérivée ne dépend plus de x : relis le rappel maths de 208.2": -2.4,
          "vérifie le signe : x est négatif": DELTA_B82,
          "c'est la dérivée de smooth_l1_loss : relis la question, ℓ_δ": -1.0}'''),
        ("e", DEC2, "GRAD3_B82", r'''decimals=2, mistakes={"la loss du mini-batch est une moyenne : divise par 3": float(huber_grad_b82(3.2)),
          "au-delà du seuil, la dérivée ne dépend plus de x": 3.2 / 3,
          "vérifie le signe : x = Q − y, et Q monte avec x": -GRAD3_B82}'''),
        ("f", DEC3, "GRAD3_MSE_B82", r'''decimals=3, mistakes={"la loss du mini-batch est une moyenne : divise par 3": 3.2,
          "la dérivée de ½x² vaut x, pas 2x": 2 * 3.2 / 3,
          "relis la question : la moyenne des ½x², sans seuil": GRAD3_B82}'''),
        ("g", DEC3, "SMOOTH_B82", r'''decimals=3, mistakes={"relis le rappel maths de 208.2 : smooth_l1_loss(beta) et huber_loss(delta) ne coïncident qu'en 1": MEAN_B82,
          "smooth_l1_loss divise la loss de Huber par β, elle ne la multiplie pas": MEAN_B82 * DELTA_B82,
          "relis l'énoncé : la moyenne, pas la somme": float(TERMS_B82.sum()) / DELTA_B82}'''),
        ("h", TF, "SAME_GRAD_B82", r'''mistakes={"|0,6| est-il sous le seuil 1,5 ? Compare les deux dérivées dans ce cas": not SAME_GRAD_B82}'''),
    ]),
    Paper("B8.3", "Gradient du log d'une politique softmax à trois actions", [
        ("a", DEC4, "float(PI_B83[2])", r'''decimals=4, mistakes={"la softmax normalise les exponentielles par leur somme": float(np.exp(-0.5)),
          "c'est la probabilité d'une autre action : relis la numérotation à partir de 0": float(PI_B83[0]),
          "les logits ne sont pas des probabilités : passe par l'exponentielle": -0.5 / 1.5}'''),
        ("b", DEC4, "float(PI_B83[1])", r'''decimals=4, mistakes={"c'est la probabilité d'une autre action : relis la numérotation à partir de 0": float(PI_B83[2]),
          "la softmax normalise les exponentielles par leur somme": float(np.exp(1.5))}'''),
        ("c", DEC4, "float(GRAD_B83[0])", r'''decimals=4, mistakes={"vérifie le signe : e_a − softmax(z), et z_0 n'est pas l'action jouée": float(PI_B83[0]),
          "relis la formule de 208.3 : e_a vaut 0 en position 0 ici, l'action jouée est la 2": float(1 - PI_B83[0]),
          "relis la question : la composante de z_0": float(GRAD_B83[2])}'''),
        ("d", DEC4, "float(GRAD_B83[2])", r'''decimals=4, mistakes={"relis la formule de 208.3 : e_a vaut 1 en position a": float(-PI_B83[2]),
          "relis la question : la dérivée de log π(2), pas celle de π(2)": float(PI_B83[2] * (1 - PI_B83[2]))}'''),
        ("e", TF, "True", r'''mistakes={"additionne e_a − softmax(z) composante par composante : que vaut la somme de la softmax ?": False}'''),
        ("f", DEC3, "float(Z_NEW_B83[2])", r'''decimals=3, mistakes={"vérifie le signe : l'avantage est négatif, le logit de l'action jouée doit baisser": float(Z_B83[2] - STEP_B83 * ADV_B83 * GRAD_B83[2]),
          "relis l'énoncé : le pas multiplie aussi l'avantage": float(Z_B83[2] + STEP_B83 * GRAD_B83[2]),
          "relis l'énoncé : le pas vaut 0,1": float(Z_B83[2] + ADV_B83 * GRAD_B83[2])}'''),
        ("g", DEC4, "float(PI_NEW_B83[2])", r'''decimals=4, mistakes={"relis la question : après le pas, avec les trois nouveaux logits": float(PI_B83[2]),
          "les trois logits bougent : refais la softmax avec les trois nouveaux": float(softmax_b83([0.5, 1.5, Z_NEW_B83[2]])[2]),
          "vérifie le signe du pas : l'avantage est négatif": float(softmax_b83(Z_B83 - STEP_B83 * ADV_B83 * GRAD_B83)[2])}'''),
        ("h", DEC3, "ENTROPY_B83", r'''decimals=3, mistakes={"l'entropie de la fiche est en nats : le logarithme naturel": float(-(PI_B83 * np.log2(PI_B83)).sum()),
          "vérifie le signe : H = −Σ π ln π est positive": -ENTROPY_B83,
          "relis la question : l'entropie avant le pas": float(-(PI_NEW_B83 * np.log(PI_NEW_B83)).sum())}'''),
    ]),
    Paper("B8.4", "GAE à la main sur quatre pas, pour λ = 0, 0,95 et 1", [
        ("a", DEC2, "float(DELTA_B84[1])", r'''decimals=2, mistakes={"relis « Fin d'épisode ou limite de temps » (§208.4) : un pas tronqué amorce-t-il ?": float(1 - V_B84[1]),
          "next_values[1] est la valeur de la vraie dernière observation : relis l'énoncé": float(1 + GAMMA_B84 * V_B84[2] - V_B84[1]),
          "γ multiplie la valeur de l'état suivant": float(1 + NEXT_B84[1] - V_B84[1])}'''),
        ("b", DEC2, "float(DELTA_B84[3])", r'''decimals=2, mistakes={"le rollout s'arrête, pas l'épisode : relis la docstring de compute_gae (next_values)": float(1 - V_B84[3]),
          "γ multiplie la valeur de l'état suivant": float(1 + NEXT_B84[3] - V_B84[3])}'''),
        ("c", DEC2, "float(ADV0_B84[2])", r'''decimals=2, mistakes={"relis le cas λ = 0 (§208.4)": float(ADV1_B84[2]),
          "relis le cas λ = 0 (§208.4) : avantage et cible du critique sont deux sorties différentes": float(ADV0_B84[2] + V_B84[2])}'''),
        ("d", DEC4, "float(ADV95_B84[2])", r'''decimals=4, mistakes={"la récursion multiplie par γλ, pas par λ seul": float(DELTA_B84[2] + 0.95 * DELTA_B84[3]),
          "la récursion multiplie par γλ, pas par γ seul": float(DELTA_B84[2] + GAMMA_B84 * DELTA_B84[3]),
          "relis la question : λ = 0,95, l'erreur TD suivante compte aussi": float(DELTA_B84[2]),
          "relis la question : l'avantage, pas la cible du critique": float(RET95_B84[2])}'''),
        ("e", DEC4, "float(ADV95_B84[0])", r'''decimals=4, mistakes={"relis « Fin d'épisode ou limite de temps » (§208.4) : la récursion s'arrête-t-elle au pas 1 ?": float(ADV95_NO_STOP_B84[0]),
          "relis « Fin d'épisode ou limite de temps » (§208.4) : le pas tronqué amorce encore": float(ADV95_AS_END_B84[0]),
          "la récursion multiplie par γλ, pas par λ seul": float(DELTA_B84[0] + 0.95 * DELTA_B84[1])}'''),
        ("f", DEC4, "float(RET95_B84[0])", r'''decimals=4, mistakes={"relis la docstring de compute_gae : les cibles valent avantages + values": float(ADV95_B84[0] + NEXT_B84[0]),
          "relis la question : la cible du critique (avantage + valeur)": float(ADV95_B84[0])}'''),
        ("g", DEC3, "float(ADV1_B84[0])", r'''decimals=3, mistakes={"relis « Fin d'épisode ou limite de temps » (§208.4) : la récursion s'arrête-t-elle au pas 1 ?": float(gae_b84(1.0, truncated=TERM_B84)[1][0]),
          "relis la question : λ = 1": float(ADV95_B84[0]),
          "relis la question : l'avantage, pas la cible du critique": float(ADV1_B84[0] + V_B84[0])}'''),
        ("h", DEC3, "float(ADV1_B84[2])", r'''decimals=3, mistakes={"relis la question : λ = 1": float(ADV95_B84[2]),
          "le pas 3 n'est pas la fin de l'épisode : son erreur TD compte aussi": float(DELTA_B84[2])}'''),
        ("i", DEC3, "float(ADV95_AS_END_B84[0])", r'''decimals=3, mistakes={"relis la question : sans amorçage au pas 1, l'erreur TD de ce pas change": float(ADV95_B84[0]),
          "relis la question : λ = 0,95": float(gae_b84(1.0, terminated=TRUNC_B84, truncated=TERM_B84)[1][0])}'''),
        ("j", TF, "MC_B84", r'''mistakes={"écris Â_0 = δ_0 + γ δ_1 avec λ = 1, puis ajoute V(s_0) : les valeurs intermédiaires se simplifient-elles ?": not MC_B84}'''),
    ]),
    Paper("B8.5", "Objectif clippé de PPO : quatre cas, valeur et gradient nul ou non", [
        ("a", DEC2, "float(TERMS_B85[0])", r'''decimals=2, mistakes={"le min prend le plus petit des deux termes : relis le cas de l'avantage positif (§208.5)": float(RATIO_B85[0] * ADV_B85[0]),
          "le clip porte sur le ratio, pas sur le produit": 1.2}'''),
        ("b", DEC2, "float(TERMS_B85[2])", r'''decimals=2, mistakes={"le min prend le plus petit des deux termes : avec un avantage négatif, lequel est le plus petit ?": float(RATIO_B85[2] * ADV_B85[2]),
          "relis la borne basse de l'intervalle : 1 − ε": float(1.2 * ADV_B85[2])}'''),
        ("c", DEC2, "float(TERMS_B85[3])", r'''decimals=2, mistakes={"le min prend le plus petit des deux termes : avec un avantage négatif, lequel est le plus petit ?": float(1.2 * ADV_B85[3]),
          "vérifie le signe : l'avantage est négatif": float(-RATIO_B85[3] * ADV_B85[3])}'''),
        ("d", DEC3, "LOSS_B85", r'''decimals=3, mistakes={"vérifie le signe : la loss est l'opposé de l'objectif": -LOSS_B85,
          "relis l'énoncé : la moyenne des quatre termes, pas leur somme": LOSS_B85 * 4,
          "relis le rôle du min (§208.5) : il prend le plus pessimiste des deux termes": LOSS_MAX_B85}'''),
        ("e", DEC3, "float(GRAD_B85[1])", r'''decimals=3, mistakes={"vérifie le signe : la loss baisse si ce ratio monte": -float(GRAD_B85[1]),
          "le gradient de la loss moyenne divise par le nombre de transitions": -float(ADV_B85[1]),
          "compare les deux termes de la transition 1 : lequel est le minimum ?": 0.0}'''),
        ("f", DEC2, "float(GRAD_B85[3])", r'''decimals=2, mistakes={"compare les deux termes de la transition 3 : le non clippé est-il le minimum ?": 0.0,
          "vérifie le signe : la loss baisse si ce ratio redescend": -float(GRAD_B85[3]),
          "le gradient de la loss moyenne divise par le nombre de transitions": -float(ADV_B85[3])}'''),
        ("g", WHOLE, "ZERO_B85", r'''mistakes={"un ratio hors de l'intervalle n'a pas toujours un gradient nul : compare les deux termes, transition par transition": 4,
          "compare les deux termes de chaque transition : où le terme clippé est-il strictement le plus petit ?": 1,
          "relis la figure 1 de l'article : dans quelles zones le gradient s'annule-t-il ?": 3}'''),
        ("h", DEC2, "CLIPFRAC_B85", r'''decimals=2, mistakes={"la fraction clippée compte les ratios hors de l'intervalle, pas les gradients nuls": ZERO_B85 / 4,
          "relis la définition : |ρ − 1| > ε, pour chaque transition": 0.75}'''),
        ("i", WHOLE, "ZERO_EPS3_B85", r'''mistakes={"avec ε = 0,3, l'intervalle devient [0,7 ; 1,3] : refais la comparaison des deux termes": ZERO_B85,
          "avec ε = 0,3, compare encore les deux termes de chaque transition : le terme clippé est-il quelque part strictement le plus petit ?": 0,
          "relis les quatre ratios contre [0,7 ; 1,3]": 3}'''),
    ]),
    Paper("B8.6", "Préférences : loss de Bradley-Terry puis loss DPO sur une paire de réponses", [
        ("a", DEC3, "P_B86", r'''decimals=3, mistakes={"vérifie le sens de la différence : la préférée moins l'autre": sigmoid(RL_B86 - RW_B86),
          "Bradley-Terry passe la différence par une sigmoïde": RW_B86 / (RW_B86 + RL_B86)}'''),
        ("b", DEC3, "LOSS_B86", r'''decimals=3, mistakes={"la loss est −log σ(d), le logarithme naturel": -math.log10(P_B86),
          "vérifie le sens de la différence : la préférée moins l'autre": LOSS_SWAP_B86,
          "relis la question : la loss, pas la probabilité": 1 - P_B86}'''),
        ("c", DEC3, "GRAD_W_B86", r'''decimals=3, mistakes={"vérifie le signe : monter r(y_w) fait baisser la loss": -GRAD_W_B86,
          "la dérivée de −log σ(d) vaut −(1 − σ(d)), pas −σ(d)": -P_B86}'''),
        ("d", DEC3, "LOSS_SWAP_B86", r'''decimals=3, mistakes={"relis la question : le choix inverse, avec les mêmes récompenses": LOSS_B86,
          "la loss est −log σ(d), le logarithme naturel": -math.log10(1 - P_B86)}'''),
        ("e", TF, "True", r'''mistakes={"relis « Le modèle de récompense de Bradley-Terry » (§208.7) : de quoi dépend la loss ?": False}'''),
        ("f", DEC2, "IMPLICIT_B86", r'''decimals=2, mistakes={"vérifie le sens du logarithme : la politique sur la référence": -IMPLICIT_B86,
          "relis la question : multiplie par β": IMPLICIT_B86 / 0.1,
          "relis la question : la récompense implicite de y_w seule": MARGIN_B86}'''),
        ("g", DEC2, "MARGIN_B86", r'''decimals=2, mistakes={"relis la formule de DPO (§208.7) : β multiplie la différence": MARGIN_B86 / 0.1,
          "vérifie le signe de la partie de y_l : on retranche son log-ratio": 0.1 * ((LOGP_B86["pc"] - LOGP_B86["rc"]) + (LOGP_B86["pr"] - LOGP_B86["rr"])),
          "vérifie le sens des log-ratios : la politique moins la référence": -MARGIN_B86}'''),
        ("h", DEC3, "DPO_B86", r'''decimals=3, mistakes={"vérifie le signe de la marge dans −log σ(·)": softplus(MARGIN_B86),
          "relis la formule de DPO : β multiplie la différence": softplus(-MARGIN_B86 / 0.1),
          "la loss est −log σ(marge), le logarithme naturel": -math.log10(sigmoid(MARGIN_B86))}'''),
        ("i", DEC3, "DPO5_B86", r'''decimals=3, mistakes={"relis la question : β = 0,5": DPO_B86,
          "vérifie le signe de la marge dans −log σ(·)": softplus(5 * MARGIN_B86)}'''),
        ("j", DEC4, "DPO_GRAD_B86", r'''decimals=4, mistakes={"vérifie le signe : monter log π(y_w) fait baisser la loss": -DPO_GRAD_B86,
          "la dérivée de la marge par rapport à log π(y_w) vaut β : n'oublie pas ce facteur": DPO_GRAD_B86 / 0.1,
          "la dérivée de −log σ(m) vaut −σ(−m), pas −σ(m)": -0.1 * sigmoid(MARGIN_B86)}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ B8.1 à B8.6)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "plusieurs choix : les lettres ensemble, entre guillemets, dans n'importe quel ordre (`\"EF\"`) ; une "
               "forme : entre guillemets (`\"(3, 5)\"`). Arrondis comme l'énoncé le demande, et seulement à la fin "
               "du calcul. Les réponses pas encore remplies affichent ⏳. Les questions « dans ta copie », la "
               "dérivation ∂ B8.7, l'estimation 🧮 B8.8, la réflexion (🗣️ B8.9, ⚖️ B8.10, 📄 B8.11) et "
               "l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to C (exercises B8.12 to B8.28). Session 78 writes them in two halves: B8.12 to B8.20 (chB8_parts.py,
# chB8_code.py: part A and the start of part B), then B8.21 to B8.28 (NEXT_SESSION: the end of part B and part C).
# history_chB8.py precomputes the calibrations and the multi-seed protocols.
# ---------------------------------------------------------------------------
import dataclasses  # noqa: E402

import chB8_code  # noqa: E402,F401  (the cells; history_chB8.py reuses them)
from chB8_parts import PART_B, PARTS  # noqa: E402,F401

for _part in PARTS:
    for _ex in _part.exercises:
        check_unique_keys(_ex.record or "x = 1", _ex.id)

NEXT_SESSION = [
    ("B", "B8.21 à B8.24", "la fin de la politique : les journaux de ton PPO, ses hyperparamètres sur trois graines, "
                           "sa robustesse face à une autre physique"),
    ("C", "B8.25 à B8.28", "les préférences : Bradley-Terry et DPO dans ta librairie, un RLHF miniature sur "
                           "CartPole, la revue de code face à CleanRL, et le défi"),
]


def kind_part(part, kind: str):
    """The part with its tools for this notebook: «KIND» is "learner" (exercise notebook) or "solution"."""
    return dataclasses.replace(part, given=part.given.replace("«KIND»", "learner" if kind == "exercise" else "solution"))


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | B8.Q1 à B8.Q12, B8.R1 à B8.R3, B8.1 à B8.6 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(ex.table_row(part.key if i == 0 else ""))
    rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
             for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# B8 · RL moderne : DQN, gradient de politique, PPO et RLHF — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/b8_rl_moderne/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py B8`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# B8 · RL moderne : DQN, gradient de politique, PPO et RLHF — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Expliquer pourquoi et comment un réseau remplace la table Q, et ce qui rend DQN stable.\n"
               "- Dériver le gradient de politique et justifier la baseline, l'avantage et GAE.\n"
               "- Implémenter en PyTorch un DQN, REINFORCE et PPO sur CartPole, avec les calculs clés testés dans "
               "`mylearn`.\n"
               "- Diagnostiquer un entraînement RL à partir de ses journaux et comparer des variantes sur "
               "plusieurs graines.\n"
               "- Expliquer RLHF, DPO et GRPO et entraîner un modèle de récompense sur des préférences.\n\n"
               "**Rappel express.** Cible DQN $y = r + \\gamma(1 - d)\\max_{a'} Q_{\\bar\\theta}(s', a')$ ; Double "
               "DQN : $Q_{\\bar\\theta}(s', \\arg\\max_{a'} Q_\\theta(s', a'))$. Huber : $\\frac12 x^2$ si "
               "$|x| \\le \\delta$, $\\delta(|x| - \\frac{\\delta}{2})$ sinon. $\\nabla_{\\mathbf z} \\log "
               "\\operatorname{softmax}(\\mathbf z)_a = \\mathbf e_a - \\operatorname{softmax}(\\mathbf z)$. "
               "$\\delta_t = r_t + \\gamma(1 - \\text{terminated}_t)V(s_{t+1}) - V(s_t)$, $\\hat A_t = \\delta_t + "
               "\\gamma\\lambda(1 - \\text{done}_t)\\hat A_{t+1}$. PPO : $-\\operatorname{mean}\\min(\\rho \\hat A, "
               "\\operatorname{clip}(\\rho, 1 \\pm \\varepsilon)\\hat A)$. Bradley-Terry : $-\\log\\sigma(r_w - r_l)$ ; "
               "DPO : $-\\log\\sigma\\big(\\beta[(\\log\\pi_w - \\log\\pi^{\\text{ref}}_w) - (\\log\\pi_l - "
               "\\log\\pi^{\\text{ref}}_l)]\\big)$.")]


def footer_cells(kind: str) -> list:
    later = ("" if not NEXT_SESSION else
             "\n\n*Les exercices B8.21 à B8.28 (la fin de la politique, les préférences et le défi) seront ajoutés à "
             "la prochaine session de génération ; relance alors `python tools/start_chapter.py B8` pour obtenir le "
             "notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu calculer une cible DQN et Double DQN, en distinguant une vraie fin d'une limite de temps ?\n"
               "2. Sais-tu dériver le gradient du log d'une politique softmax, et dire pourquoi une baseline ne "
               "biaise pas le gradient ?\n"
               "3. Sais-tu calculer GAE à rebours et l'objectif clippé de PPO avec son gradient, cas par cas ?\n"
               "4. Sais-tu calculer une loss de Bradley-Terry et une loss DPO, et dire ce que règle $\\beta$ ?\n\n"
               "**Pour aller plus loin** : l'article de Schulman et coll. (📄 B8.11) et les « 37 détails » de "
               "Huang et coll., cités dans la fiche. B8 est le dernier chapitre bonus."
               + later)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT + LIBRARY_CHECKS, PAPER_INTRO)
    for part in PARTS:
        cells += part_cells(kind_part(part, kind), kind)
    cells += footer_cells(kind)
    return cells


def main() -> int:
    write_notebook(f"{FOLDER}/03_notebook.ipynb", build("exercise"))
    write_notebook(f"{FOLDER}/05_solutions.ipynb", build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER)
    n_ex = sum(len(part.exercises) for part in PARTS)
    print(f"✅ {FOLDER}/03_notebook.ipynb and 05_solutions.ipynb written ({len(PAPER)} paper exercises, "
          f"{n_checks} paper checks, {n_ex} notebook exercises)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
