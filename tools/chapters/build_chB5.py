#!/usr/bin/env python
"""Build the two notebooks of bonus chapter B5 from a single source (used by Claude).

    python tools/chapters/build_chB5.py
    python tools/run_all_notebooks.py chapitres/b5_diffusion/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/b5_diffusion/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q11, of the recalls R1 to R3 and of the ✏️ paper exercises
B5.1 to B5.5 (02_exercices.md). The derivations ∂ B5.6 and B5.7, the estimate 🧮 B5.8, the reflection (🗣️ B5.9,
⚖️ B5.10, 📄 B5.11) and the interview are corrected with 05_solutions.md. The answers are computed by hand formulas in
the context cell of the solutions notebook, then checked against the reference library (mylearn.diffusion,
mylearn.attention, mylearn.gen_metrics, mylearn.vae) and, if it is installed, against diffusers (DDPMScheduler and
DDIMScheduler built from the toy schedule of ✏️ B5.2, their noise set to zero): part 0 downloads nothing.
The notebook itself (exercises B5.12 to B5.31) comes at the second generation session (NEXT_SESSION), in three
phases: the mathematics of diffusion.py and their checks, 2-D toys, the U-Net on MNIST and Fashion-MNIST.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (Paper, badge, check_unique_keys, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "B5"
FOLDER = "chapitres/b5_diffusion"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = r'''# The short answers only use the numbers of their statements (02_exercices.md). Steps: t = 1..T in the
# formulas (Ho et al.), indices 0..T-1 in the code (step t is index t - 1).
import math

import numpy as np

# ✏️ B5.1: classifier-free guidance on two pixels, (1 + w) eps_c - w eps_uncond
EC_B51, EU_B51 = np.array([0.8, -0.5]), np.array([0.5, 0.1])


def cfg_b51(w, cond=EC_B51, uncond=EU_B51):
    return [float(v) for v in (1 + w) * cond - w * uncond]


W0_B51, W1_B51, W3_B51 = cfg_b51(0), cfg_b51(1), cfg_b51(3)
GAP_B51 = 3 * float(np.linalg.norm(EC_B51 - EU_B51))                # || eps_tilde - eps_c || for w = 3
EVALS_B51 = 2 * 50                                                  # two predictions per step, 50 DDIM steps

# ✏️ B5.2: toy linear schedule T = 4, beta from 0.05 to 0.35 (it serves B5.4 to B5.7 too)
BETAS_B52 = np.linspace(0.05, 0.35, 4)
ALPHAS_B52 = 1 - BETAS_B52
AB_B52 = np.cumprod(ALPHAS_B52)                                     # alpha_bar, indices 0..3 (steps 1..4)
BETA2_B52, AB2_B52, AB4_B52 = float(BETAS_B52[1]), float(AB_B52[1]), float(AB_B52[3])
KEEP4_B52 = math.sqrt(AB4_B52)
APPROX4_B52 = math.exp(-float(BETAS_B52.sum()))
SUM1000_B52 = 1000 * (1e-4 + 0.02) / 2                              # arithmetic series: T (beta_1 + beta_T) / 2
AB1000_E5_B52 = math.exp(-SUM1000_B52) * 1e5
KEEP1000_PCT_B52 = 100 * math.sqrt(AB1000_E5_B52 * 1e-5)
SUM400_B52 = 400 * (1e-4 + 0.02) / 2
SUM400_SCALED_B52 = 400 * (1e-4 + 0.02) * (1000 / 400) / 2

# ✏️ B5.3: signal-to-noise ratio of the linear schedule (approximation exp(-S(t))) and of the cosine schedule (exact)
T_B53, B1_B53, BT_B53, S_B53 = 1000, 1e-4, 0.02, 0.008


def lin_sum_b53(t, square=False):
    """S(t), the sum of the first t betas; square=True: the t**2 slip instead of t (t - 1)."""
    return t * B1_B53 + (BT_B53 - B1_B53) * (t * t if square else t * (t - 1)) / (2 * (T_B53 - 1))


def snr_b53(ab):
    return ab / (1 - ab)


def cos_ab_b53(u, s=S_B53):
    f = lambda v: math.cos((v + s) / (1 + s) * math.pi / 2) ** 2
    return f(u) / f(0)


def lin_cross_b53(target_sum):
    """The t (real) where S(t) = target_sum: a t**2 + b t - target_sum = 0, then the fraction t / T."""
    a = (BT_B53 - B1_B53) / (2 * (T_B53 - 1))
    b = B1_B53 - a
    return (-b + math.sqrt(b * b + 4 * a * target_sum)) / (2 * a) / T_B53


def cos_cross_b53(alpha_bar, s=S_B53):
    """The fraction u where the cosine alpha_bar(u) equals alpha_bar."""
    f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    theta = math.acos(math.sqrt(alpha_bar * f0))
    return theta * 2 / math.pi * (1 + s) - s


S500_B53 = lin_sum_b53(500)
LIN500_B53 = snr_b53(math.exp(-S500_B53))
COS500_B53 = snr_b53(cos_ab_b53(0.5))
LIN1_B53, COS1_B53 = lin_cross_b53(math.log(2)), cos_cross_b53(0.5)           # SNR = 1: alpha_bar = 1/2
LIN01_B53, COS01_B53 = lin_cross_b53(math.log(11)), cos_cross_b53(1 / 11)     # SNR = 0.1: alpha_bar = 1/11

# ✏️ B5.4: one DDPM step at step t = 3 (index 2) of the toy schedule
I_B54, X3_B54, EPS_B54, Z_B54 = 2, 0.8, 0.5, -1.0


def ddpm_mean_b54(x, eps, i=I_B54, sign=-1.0, root=True):
    """Mean of Algorithm 2; sign=+1: the sign slip; root=False: beta / (1 - alpha_bar) without the square root."""
    ab = float(AB_B52[i])
    coef = BETAS_B52[i] / (math.sqrt(1 - ab) if root else (1 - ab))
    return float((x + sign * coef * eps) / math.sqrt(ALPHAS_B52[i]))


def posterior_b54(i=I_B54):
    """coef1, coef2 and beta tilde of the posterior at index i (alpha_bar_prev = 1 at index 0)."""
    ab, ab_prev = float(AB_B52[i]), (float(AB_B52[i - 1]) if i > 0 else 1.0)
    beta = float(BETAS_B52[i])
    return (beta * math.sqrt(ab_prev) / (1 - ab), (1 - ab_prev) * math.sqrt(ALPHAS_B52[i]) / (1 - ab),
            beta * (1 - ab_prev) / (1 - ab))


COEF_B54 = float(BETAS_B52[I_B54] / math.sqrt(1 - AB_B52[I_B54]))
MEAN_B54 = ddpm_mean_b54(X3_B54, EPS_B54)
C1_B54, C2_B54, BTILDE_B54 = posterior_b54()
X2_B54 = MEAN_B54 + math.sqrt(BTILDE_B54) * Z_B54
X2_BETA_B54 = MEAN_B54 + math.sqrt(BETAS_B52[I_B54]) * Z_B54
X3C_B54, EPSC_B54 = 1.5, -0.6
X0HAT_C_B54 = (X3C_B54 - math.sqrt(1 - AB_B52[I_B54]) * EPSC_B54) / math.sqrt(AB_B52[I_B54])
CLIP_MEAN_B54 = C1_B54 * float(np.clip(X0HAT_C_B54, -1, 1)) + C2_B54 * X3C_B54
NOCLIP_MEAN_B54 = C1_B54 * X0HAT_C_B54 + C2_B54 * X3C_B54
LAST_B54 = ddpm_mean_b54(0.5, 0.2, i=0)                                         # step 1 (index 0): no noise

# ✏️ B5.5: DDIM from step 4 (index 3), x_4 = -0.5, eps_theta = -1.2
X4_B55, EPS_B55 = -0.5, -1.2


def x0hat_b55(x=X4_B55, eps=EPS_B55, i=3):
    return float((x - math.sqrt(1 - AB_B52[i]) * eps) / math.sqrt(AB_B52[i]))


def ab_b55(i):
    return 1.0 if i == -1 else float(AB_B52[i])


def ddim_b55(i, i_prev, eta=0.0, x=X4_B55, eps=EPS_B55, noise_term=True):
    """The deterministic part of a DDIM step (z = 0), and sigma."""
    a, ap = ab_b55(i), ab_b55(i_prev)
    sigma = eta * math.sqrt((1 - ap) / (1 - a)) * math.sqrt(1 - a / ap)
    x0 = (x - math.sqrt(1 - a) * eps) / math.sqrt(a)
    return math.sqrt(ap) * x0 + (math.sqrt(1 - ap - sigma ** 2) * eps if noise_term else 0.0), sigma


X0HAT_B55 = x0hat_b55()
X1_B55 = ddim_b55(3, 0)[0]
X2_B55 = ddim_b55(3, 1)[0]
SIGMA2_43_B55 = ddim_b55(3, 2, eta=1.0)[1] ** 2
SIGMA_41_B55 = ddim_b55(3, 0, eta=1.0)[1]
DIR_41_B55 = math.sqrt(1 - AB_B52[0] - SIGMA_41_B55 ** 2)

# quizzes and recalls with a computation
Q1E = 1000 * 0.02
Q2B = float(np.mean([-1, 0, 2]))
Q2C = float(np.mean((np.array([-1, 0, 2]) - Q2B) ** 2))
Q3A, Q3B, Q3D = math.sqrt(0.64), math.sqrt(1 - 0.64), 0.64 / 0.36
Q4B = (0.5 - math.sqrt(1 - 0.64) * 0.2) / math.sqrt(0.64)
Q5C = 128 // 2
Q6B, Q6D = 128 + 64, 32 // 2 ** 3
Q7A = 1000 * 0.05
Q9A = 3 * 0.3 - 2 * 0.5
Q11A = (256 // 4) ** 2 * 3
R1B = 2 * math.pi
R2A = (0 - 1) ** 2 + (1 + 4 - 2 * math.sqrt(1 * 4))
R2E = 0 ** 2 + (1 + 0 - 2 * math.sqrt(1 * 0))
R3A = 0.5 + math.exp(0.5 * math.log(0.25)) * (-2)
R3B = 0.5 * (0.5 ** 2 + 0.25 - 1 - math.log(0.25))
'''

LIBRARY_CHECKS = r'''
# The same answers from the libraries (no download): mylearn (reference) and, if installed, diffusers
D = mylearn.diffusion
toy = D.noise_schedule(D.linear_beta_schedule(4, 0.05, 0.35))
assert np.allclose(toy["betas"], BETAS_B52) and np.allclose(toy["alpha_bars"], AB_B52)

# ✏️ B5.2: the sums of the betas, and the exact alpha_bar_T that the approximation overestimates
lin1000 = D.noise_schedule(D.linear_beta_schedule(1000))
assert math.isclose(lin1000["betas"].sum(), SUM1000_B52)
assert math.isclose(D.linear_beta_schedule(400).sum(), SUM400_B52)
assert math.isclose(D.linear_beta_schedule(400, 1e-4 * 2.5, 0.02 * 2.5).sum(), SUM400_SCALED_B52)
assert lin1000["alpha_bars"][-1] < AB1000_E5_B52 * 1e-5 and toy["alpha_bars"][-1] < APPROX4_B52

# ✏️ B5.3: the exact schedules of the library (index i = step i + 1) cross the thresholds at the same 2 decimals
cos1000 = D.noise_schedule(D.cosine_beta_schedule(1000))
for sched, answers in ((lin1000, (LIN1_B53, LIN01_B53)), (cos1000, (COS1_B53, COS01_B53))):
    snr = sched["alpha_bars"] / (1 - sched["alpha_bars"])
    for threshold, answer in zip((1.0, 0.1), answers):
        first_step = int(np.argmax(snr < threshold)) + 1
        assert round(first_step / 1000, 2) == round(answer, 2), (threshold, first_step, answer)
assert math.isclose(cos1000["alpha_bars"][499], cos_ab_b53(0.5), rel_tol=1e-9)
assert round(float(lin1000["alpha_bars"][499] / (1 - lin1000["alpha_bars"][499])), 2) == round(LIN500_B53, 2)

# ✏️ B5.4: ddpm_step; its mean is step(x) - step(0) with the same seed, its variance posterior_variance
def mean_of_ddpm_step(x, i, eps, **kw):
    one = D.ddpm_step(np.array([[x]]), i, np.array([[eps]]), toy, rng=np.random.default_rng(0), **kw)
    zero = D.ddpm_step(np.array([[0.0]]), i, np.array([[0.0]]), toy, rng=np.random.default_rng(0),
                       variance=kw.get("variance", "posterior"))
    return float((one - zero)[0, 0])


assert math.isclose(mean_of_ddpm_step(X3_B54, I_B54, EPS_B54), MEAN_B54)
assert math.isclose(mean_of_ddpm_step(X3_B54, I_B54, EPS_B54, variance="beta"), MEAN_B54)
assert math.isclose(mean_of_ddpm_step(X3C_B54, I_B54, EPSC_B54, clip_x0=1.0), CLIP_MEAN_B54)
assert math.isclose(toy["posterior_variance"][I_B54], BTILDE_B54)
assert math.isclose(toy["posterior_mean_coef1"][I_B54], C1_B54) and math.isclose(toy["posterior_mean_coef2"][I_B54], C2_B54)
for variance in ("posterior", "beta"):                                 # step 1 (index 0): no noise, whatever the seed
    assert math.isclose(float(D.ddpm_step(np.array([[0.5]]), 0, np.array([[0.2]]), toy, rng=7, variance=variance)[0, 0]),
                        LAST_B54)

# ✏️ B5.5: predict_x0_from_eps and ddim_step (eta = 0: deterministic)
assert math.isclose(float(D.predict_x0_from_eps(np.array([X4_B55]), [3], np.array([EPS_B55]), AB_B52)[0]), X0HAT_B55)
x4 = np.array([[X4_B55]]); e4 = np.array([[EPS_B55]])
assert math.isclose(float(D.ddim_step(x4, 3, 0, e4, AB_B52)[0, 0]), X1_B55)
assert math.isclose(float(D.ddim_step(x4, 3, 1, e4, AB_B52)[0, 0]), X2_B55)
assert math.isclose(float(D.ddim_step(D.ddim_step(x4, 3, 1, e4, AB_B52), 1, 0, e4, AB_B52)[0, 0]), X1_B55)
assert math.isclose(float(D.ddim_step(x4, 3, -1, e4, AB_B52)[0, 0]), X0HAT_B55)
assert math.isclose(SIGMA2_43_B55, toy["posterior_variance"][3])

# recalls: positional_encoding (B3), frechet_distance (ch. 27), reparameterize and gaussian_kl (ch. 25)
pe = mylearn.attention.positional_encoding(1000, 128)
assert pe[0, 1] == 1.0 and pe.shape[1] // 2 == Q5C
assert np.allclose(pe[:, 0], np.sin(np.arange(1000)))                 # column 0: sin(p), period 2 pi
assert 2 * math.pi * 10000 ** (126 / 128) > 1000                      # the slowest pair: less than one turn
fd = mylearn.gen_metrics.frechet_distance
assert math.isclose(fd([0.0], [[1.0]], [1.0], [[4.0]]), R2A) and math.isclose(fd([0.0], [[1.0]], [0.0], [[0.0]]), R2E)
z, _ = mylearn.vae.reparameterize([[0.5]], [[math.log(0.25)]], rng=0)
assert z.shape == (1, 1)
assert math.isclose(float(mylearn.vae.gaussian_kl([[0.5]], [[math.log(0.25)]])[0]), R3B)
print("✅ the paper answers agree with mylearn (reference): diffusion, attention, gen_metrics, vae")

# Optional: diffusers' schedulers on the toy schedule (float64 tensors, noise set to zero)
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")   # tqdm, imported by diffusers
try:
    import torch
    import diffusers.schedulers.scheduling_ddpm as ddpm_module
    from diffusers import DDIMScheduler, DDPMScheduler
except ImportError:
    print("ℹ️ diffusers is not installed: the scheduler comparison is skipped (the closed forms above are checked)")
else:
    def as_double(scheduler, names):
        for name in names:
            setattr(scheduler, name, getattr(scheduler, name).double())
        return scheduler

    def diffusers_ddpm(x, i, eps, clip=None):
        sch = as_double(DDPMScheduler(num_train_timesteps=4, trained_betas=BETAS_B52.tolist(),
                                      clip_sample=clip is not None, clip_sample_range=clip or 1.0),
                        ("betas", "alphas", "alphas_cumprod", "one"))
        saved = ddpm_module.randn_tensor
        ddpm_module.randn_tensor = lambda shape, **kw: torch.zeros(shape, dtype=torch.float64)
        try:
            out = sch.step(torch.tensor([[eps]], dtype=torch.float64), i, torch.tensor([[x]], dtype=torch.float64))
        finally:
            ddpm_module.randn_tensor = saved
        return float(out.prev_sample), float(sch._get_variance(i))

    def diffusers_ddim(x, i, eps, n_steps):
        sch = as_double(DDIMScheduler(num_train_timesteps=4, trained_betas=BETAS_B52.tolist(), clip_sample=False,
                                      set_alpha_to_one=True), ("alphas_cumprod", "final_alpha_cumprod"))
        sch.set_timesteps(n_steps)
        out = sch.step(torch.tensor([[eps]], dtype=torch.float64), i, torch.tensor([[x]], dtype=torch.float64),
                       eta=0.0)
        return float(out.prev_sample), float(out.pred_original_sample), sch

    mean, var = diffusers_ddpm(X3_B54, I_B54, EPS_B54)
    assert math.isclose(mean, MEAN_B54, rel_tol=1e-6) and math.isclose(var, BTILDE_B54, rel_tol=1e-6)
    assert math.isclose(diffusers_ddpm(X3C_B54, I_B54, EPSC_B54, clip=1.0)[0], CLIP_MEAN_B54, rel_tol=1e-6)
    assert math.isclose(diffusers_ddpm(0.5, 0, 0.2)[0], LAST_B54, rel_tol=1e-6)
    prev, x0_pred, ddim = diffusers_ddim(X4_B55, 3, EPS_B55, 2)          # 4 // 2 = 2: index 3 -> index 1
    assert math.isclose(prev, X2_B55, rel_tol=1e-6) and math.isclose(x0_pred, X0HAT_B55, rel_tol=1e-6)
    assert math.isclose(diffusers_ddim(X4_B55, 3, EPS_B55, 1)[0], X0HAT_B55, rel_tol=1e-6)   # index 3 -> clean
    assert math.isclose(float(ddim._get_variance(3, 2)), SIGMA2_43_B55, rel_tol=1e-6)
    assert math.isclose(float(ddim._get_variance(3, 0)), SIGMA_41_B55 ** 2, rel_tol=1e-6)
    print("✅ ✏️ B5.4 and B5.5 also agree with diffusers' DDPMScheduler and DDIMScheduler")
'''

mylearn = importlib.import_module("mylearn_ref")   # the builder computes the mistakes with the same helpers
for _module in ("diffusion", "attention", "gen_metrics", "vae"):
    importlib.import_module(f"mylearn_ref.{_module}")
exec(PAPER_CONTEXT)

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
WHOLE = "a whole number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
DEC4 = "a number, 4 decimals"
VEC2 = "a list of two numbers, 2 decimals"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    Paper("B5.Q1", "Diffusion, VAE, GAN : comment chacun fabrique-t-il une image ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis l'encadré 🔗 « Lien avec les ch. 25 et 27 » : combien de passes du décodeur pour générer avec un VAE ?",
                                           "C": "pour générer, part-on d'une image du dataset ? Relis l'encadré 🔗 de 205.1"})),
        ("b", TF, "False", r'''mistakes={"relis « Détruire est facile » (§205.1) : le processus direct s'apprend-il ?": True}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "relis l'encadré 🔗 de 205.1 : la diffusion s'entraîne-t-elle contre un adversaire ?",
                                           "B": "le VAE a-t-il un discriminateur ? Relis l'encadré 🔗 de 205.1"})),
        ("d", TF, "True", r'''mistakes={"relis l'encadré 🔗 « Lien avec les ch. 25 et 27 » : quelle taille ont les codes de la diffusion ?": False}'''),
        ("e", WHOLE, "round(Q1E)", r'''mistakes={"c'est le nombre de passes : on demande des secondes": 1000,
          "vérifie la multiplication par 1 000": 2}'''),
    ]),
    Paper("B5.Q2", "Pourquoi détruire l'image en mille petits pas de bruit plutôt qu'en un seul ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis l'encadré 🧮 « la MSE apprend une moyenne » (§205.1) : que rend un réseau entraîné par MSE quand plusieurs cibles sont possibles ?",
                                           "C": "un réseau entraîné par MSE recopie-t-il un exemple ? Relis l'encadré 🧮 de 205.1"})),
        ("b", DEC2, "Q2B", r'''decimals=2, mistakes={"c'est la médiane : la MSE est minimisée par une autre grandeur (encadré 🧮 de 205.1)": 0.0,
          "c'est le milieu entre la plus petite et la plus grande cible : relis l'encadré 🧮 de 205.1": 0.5}'''),
        ("c", DEC2, "Q2C", r'''decimals=2, mistakes={"c'est la MSE de la prédiction 0 : prends la prédiction de b)": 5 / 3,
          "la MSE est une moyenne des carrés des écarts : divise par le nombre de cibles": 3 * Q2C,
          "élève les écarts au carré avant de faire la moyenne": float(np.mean(np.abs(np.array([-1, 0, 2]) - Q2B)))}'''),
        ("d", TF, "True", r'''mistakes={"relis « Ce qu'on voudrait » (§205.3) : que dit-on de ce pas quand β_t est petit ?": False}'''),
        ("e", TF, "False", r'''mistakes={"relis l'approximation ln ᾱ_T ≈ −Σβ (§205.2) : à somme égale, que deviennent les β quand il y a deux fois plus de pas ?": True}'''),
    ]),
    Paper("B5.Q3", "ᾱ_t : quelle part de l'image reste-t-il au pas t ?", [
        ("a", DEC2, "Q3A", r'''decimals=2, mistakes={"c'est ᾱ_t : relis la forme fermée, le coefficient de x₀ est-il ᾱ_t ?": 0.64,
          "relis la question : on demande le coefficient de x₀, pas celui du bruit": Q3B}'''),
        ("b", DEC2, "Q3B", r'''decimals=2, mistakes={"c'est 1 − ᾱ_t : relis la forme fermée, le coefficient du bruit est-il 1 − ᾱ_t ?": 0.36,
          "relis la question : on demande le coefficient du bruit, pas celui de x₀": Q3A}'''),
        ("c", TF, "False", r'''mistakes={"calcule la somme de tes réponses a) et b) ; relis la dernière phrase de « La forme fermée »": True}'''),
        ("d", DEC2, "Q3D", r'''decimals=2, mistakes={"le SNR met la part de l'image au numérateur": 0.36 / 0.64,
          "le SNR est un rapport de variances, pas de coefficients": Q3A / Q3B}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "relis la définition de ᾱ_t : un produit de nombres plus petits que 1 peut-il croître ?",
                                           "B": "regarde la figure `plannings.png` (§205.2) : ᾱ_t reste-t-il constant ?"})),
    ]),
    Paper("B5.Q4", "Prédire le bruit, l'image ou la vitesse : la même information ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis « Prédire le bruit » (§205.3) : la formule de x̂₀ est-elle une approximation quand le bruit est le vrai ?",
                                           "C": "relis « Prédire le bruit » (§205.3) : la forme fermée s'inverse-t-elle ?"})),
        ("b", DEC3, "Q4B", r'''decimals=3, mistakes={"relis la formule de x̂₀ : on divise par la racine de ᾱ_t, et le bruit est multiplié par la racine de 1 − ᾱ_t": (0.5 - 0.36 * 0.2) / 0.64,
          "vérifie le signe : on retire le bruit de x_t": (0.5 + 0.6 * 0.2) / 0.8,
          "n'oublie pas de diviser par la racine de ᾱ_t": 0.5 - 0.6 * 0.2}'''),
        ("c", TF, "True", r'''mistakes={"relis le paragraphe qui suit la moyenne μ_θ (§205.3) : qu'ont trouvé Ho et coll. pour la prédiction de x₀ ?": False}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "le réseau reçoit-il encore t quand il prédit le bruit ? Relis « Dire au réseau quel est le pas »",
                                           "B": "le bruit est-il toujours plus petit que l'image ? Relis la raison donnée en 205.3"})),
        ("e", TF, "False", r'''mistakes={"relis le paragraphe qui suit l'encadré 🧮 sur Euler (§205.9) : les deux « v » sont-ils la même grandeur ?": True}'''),
    ]),
    Paper("B5.Q5", "Le pas de temps est une entrée du réseau : pourquoi, et comment l'encoder ?", [
        ("a", LETTER, '"C"', letters("C", {"A": "relis « Reconstruire s'apprend » (§205.1) : combien de réseaux sert-on pour tous les pas ?",
                                           "B": "ᾱ_t se calcule-t-il dans le réseau ? Relis « Dire au réseau quel est le pas »"})),
        ("b", TF, "False", r'''mistakes={"relis « Dire au réseau quel est le pas » (§205.4) : que dit la fiche d'un nombre de 0 à 999 donné tel quel ?": True}'''),
        ("c", WHOLE, "Q5C", r'''mistakes={"c'est le nombre de colonnes : chaque couple en occupe deux": 128,
          "relis l'appel : d_model = 128": 32}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "relis « Un bloc résiduel conditionné par le pas » : l'encodage est-il concaténé ?",
                                           "C": "relis la ligne `h = h + self.time(temb)[:, :, None, None]` du bloc résiduel"})),
        ("e", TF, "True", r'''mistakes={"relis la fin de « Dire au réseau quel est le pas » : que dit la fiche des permutations de colonnes ?": False}'''),
    ]),
    Paper("B5.Q6", "U-Net : à quoi servent les connexions de saut ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis « Le U-Net et ses connexions de saut » : à quelle résolution relient-elles l'encodeur et le décodeur ?",
                                           "C": "c'est une connexion résiduelle : relis le piège classique « deux skip connections »"})),
        ("b", WHOLE, "Q6B", r'''mistakes={"la connexion de saut concatène : les canaux de l'encodeur s'ajoutent à ceux du décodeur": 128,
          "relis la question : 128 canaux du décodeur et 64 de l'encodeur": 256,
          "relis la question : la convolution suivante reçoit aussi les canaux du décodeur": 64}'''),
        ("c", TF, "True", r'''mistakes={"relis le paragraphe qui suit la figure du U-Net (§205.4) : que deviendrait la sortie sans connexions de saut ?": False}'''),
        ("d", WHOLE, "Q6D", r'''mistakes={"recompte les réductions de l'énoncé : en as-tu appliqué une de moins ?": 8,
          "tu n'as appliqué qu'une réduction : relis l'énoncé": 16}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "la group norm a-t-elle des paramètres γ et β ? Relis le compte du bloc résiduel",
                                           "B": "relis la définition de la group norm (§205.4) : que normalise-t-elle ?"})),
    ]),
    Paper("B5.Q7", "Échantillonnage DDPM : pourquoi est-il lent et aléatoire ?", [
        ("a", WHOLE, "round(Q7A)", r'''mistakes={"c'est le nombre de passes : on demande des secondes": 1000,
          "vérifie la multiplication : as-tu décalé la virgule ?": 500}'''),
        ("b", TF, "False", r'''mistakes={"relis « Ce que coûte un échantillon » (§205.5) : que tire-t-on à chaque pas ?": True}'''),
        ("c", TF, "True", r'''mistakes={"relis la phrase qui suit l'algorithme 2 (§205.5)": False}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "le planning interdit-il les batches ? Relis « Ce que coûte un échantillon »",
                                           "C": "un réseau peut-il servir plusieurs calculs ? Relis « Ce que coûte un échantillon » : de quoi chaque passe a-t-elle besoin ?"})),
        ("e", LETTER, '"B"', letters("B", {"A": "relis « Quelle variance ? » (§205.5) : le réseau prédit-il la variance ?",
                                           "C": "relis la fin de 205.3 : β̃_t ≤ β_t, et l'égalité ne vaut qu'au premier pas ?"})),
    ]),
    Paper("B5.Q8", "DDIM : moins de pas avec le même réseau, vrai ou faux", [
        ("a", TF, "False", r'''mistakes={"relis « L'idée de DDIM » (§205.6) : l'entraînement change-t-il ?": True}'''),
        ("b", TF, "True", r'''mistakes={"relis la formule DDIM avec η = 0 : y a-t-il un tirage ?": False}'''),
        ("c", TF, "True", r'''mistakes={"relis « Le paramètre η » (§205.6) : que vaut σ² quand η = 1 et t' = t − 1 ?": False}'''),
        ("d", TF, "True", r'''mistakes={"relis l'étape 2 de DDIM (§205.6) : t' doit-il être le pas juste avant ?": False}'''),
        ("e", TF, "False", r'''mistakes={"relis « Pourquoi c'est plus rapide » (§205.6) : une trajectoire se suit-elle aussi bien avec très peu de pas ?": True}'''),
    ]),
    Paper("B5.Q9", "Guidance sans classifieur : ce que règle l'échelle w", [
        ("a", DEC1, "Q9A", r'''decimals=1, mistakes={"vérifie le signe de l'extrapolation": -Q9A,
          "(1 + w) multiplie la prédiction avec condition, w celle sans condition": 3 * 0.5 - 2 * 0.3,
          "relis la question : w = 2": 0.3}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "relis la formule avec w = 0 : quelle prédiction reste-t-il ?",
                                           "C": "relis la formule avec w = 0 : y a-t-il une moyenne ?"})),
        ("c", TF, "False", r'''mistakes={"relis le paragraphe qui suit la figure de la guidance (§205.7)": True}'''),
        ("d", WHOLE, "2", r'''mistakes={"relis le piège classique « deux conventions pour l'échelle de guidance » (§205.7)": 3,
          "guidance_scale = 1 + w : résous pour w": 4}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "relis « La guidance sans classifieur » : combien de réseaux entraîne-t-on ?",
                                           "B": "le réseau peut-il prédire sans condition s'il n'a jamais vu ∅ ? Relis « La guidance sans classifieur »"})),
    ]),
    Paper("B5.Q10", "Un bon FID peut-il cacher de la copie des données d'entraînement ?", [
        ("a", TF, "True", r'''mistakes={"relis « La mémorisation » (§205.8) : que verrait une distance de Fréchet d'un modèle qui recopie ?": False}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "plus d'images rend-il la distance de Fréchet sensible à la copie ? Relis « La mémorisation »",
                                           "C": "la loss dit-elle quelles images sont recopiées ? Relis « La mémorisation »"})),
        ("c", TF, "False", r'''mistakes={"relis la dernière puce des résultats de Carlini et coll. (§205.8)": True}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "relis les trois règles de « La distance de Fréchet » (§205.8)",
                                           "C": "la distance dépend-elle du nombre d'images de façon proportionnelle ? Relis les trois règles de 205.8"})),
        ("e", TF, "False", r'''mistakes={"relis « Fidélité et diversité » (§205.8) : un seul nombre sépare-t-il les deux qualités ?": True}'''),
    ]),
    Paper("B5.Q11", "Diffusion latente, transformers de diffusion, flow matching : ce qui a changé depuis 2020", [
        ("a", WHOLE, "Q11A", r'''mistakes={"n'oublie pas les 3 canaux du latent": (256 // 4) ** 2,
          "relis l'énoncé : le latent a 3 canaux": (256 // 4) ** 2 * 4,
          "le facteur f divise la largeur et la hauteur, chacune": 256 * 256 * 3 // 4}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "relis « Les transformers de diffusion » (§205.9) : d'où viennent les tokens d'un DiT ?",
                                           "C": "relis « Les transformers de diffusion » : combien de tokens pour un latent de 32 × 32 × 4 ?"})),
        ("c", LETTER, '"C"', letters("C", {"A": "dérive x_s par rapport à s : vérifie le signe",
                                           "B": "dérive x_s par rapport à s : le coefficient de x₀ est 1 − s"})),
        ("d", TF, "True", r'''mistakes={"relis l'encadré 🧮 « la méthode d'Euler » (§205.9) : d'où vient l'erreur d'Euler ?": False}'''),
        ("e", TF, "False", r'''mistakes={"relis « Les transformers de diffusion » et « Flow matching et rectified flow » (§205.9)": True}'''),
    ]),
    Paper("B5.R1", "L'encodage sinusoïdal des positions", [
        ("a", WHOLE, "1", r'''mistakes={"la colonne 1 est impaire : sinus ou cosinus ?": 0}'''),
        ("b", DEC2, "R1B", r'''decimals=2, mistakes={"une période de sinus est un tour complet : relis la formule de la colonne 0": math.pi,
          "relis la formule : à la colonne 0, la fréquence vaut 1": 1.0}'''),
        ("c", TF, "False", r'''mistakes={"relis l'encodage sinusoïdal en B3 : quelles colonnes changent vite d'une position à la suivante ?": True}'''),
        ("d", TF, "True", r'''mistakes={"calcule la période 2π · 10 000^(126/128) de la dernière paire et compare-la à 1 000": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "relis la formule de l'énoncé : PE[p, 2i] est-il un cosinus ?",
                                           "C": "relis la formule de l'énoncé"})),
    ]),
    Paper("B5.R2", "Ce que mesure la distance de Fréchet et l'effondrement de mode", [
        ("a", WHOLE, "round(R2A)", r'''mistakes={"il manque un des deux termes de la formule : la moyenne ou la covariance": 1,
          "n'oublie pas le terme −2(Σ₁Σ₂)^(1/2)": 6},
          fractional="relis le calcul : avec ces nombres, le résultat est entier"'''),
        ("b", LETTER, '"A"', letters("A", {"B": "la distance de Fréchet juge-t-elle chaque image à part ? Relis la formule",
                                           "C": "une distance nulle demanderait la même moyenne et la même covariance : est-ce le cas ?"})),
        ("c", TF, "True", r'''mistakes={"chaque terme de la formule est-il nul quand les moyennes et les covariances sont égales, et seulement alors ?": False}'''),
        ("d", TF, "False", r'''mistakes={"relis le ch. 27 : sur quoi le FID calcule-t-il ses moyennes et covariances ?": True}'''),
        ("e", WHOLE, "round(R2E)", r'''mistakes={"un générateur qui rend toujours la même feature a-t-il la covariance des vraies images ?": 0},
          fractional="relis le calcul : avec ces nombres, le résultat est entier"'''),
    ]),
    Paper("B5.R3", "L'astuce de reparamétrisation et le terme KL du VAE", [
        ("a", DEC2, "R3A", r'''decimals=2, mistakes={"ln σ² = ln 0,25 : σ est la racine de 0,25, pas 0,25": 0.5 + 0.25 * (-2),
          "z = μ + σ ε, avec σ = exp(0,5 ln σ²), pas ln σ²": 0.5 + math.log(0.25) * (-2),
          "vérifie le signe : ε = −2": 0.5 + 0.5 * 2}'''),
        ("b", DEC4, "R3B", r'''decimals=4, mistakes={"n'oublie pas le facteur 1/2 de la formule": 2 * R3B,
          "il manque le terme − ln σ²": 0.5 * (0.25 + 0.25 - 1),
          "σ² = 0,25 : relis le terme de variance de la formule": 0.5 * (0.25 + 0.5 - 1 - math.log(0.25))}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "la reparamétrisation change-t-elle la taille du latent ? Relis le ch. 25",
                                           "C": "z reste-t-il aléatoire avec la reparamétrisation ? Relis le ch. 25"})),
        ("d", TF, "True", r'''mistakes={"relis l'encadré 🔗 « Lien avec la reparamétrisation (ch. 25) » (§205.3)": False}'''),
        ("e", TF, "False", r'''mistakes={"relis « Le lien avec l'ELBO » (§205.3) : ce terme dépend-il des paramètres du réseau ?": True}'''),
    ]),
    Paper("B5.1", "Guidance sans classifieur : combiner les deux prédictions pour w = 0, 1 et 3", [
        ("a", VEC2, "W0_B51", r'''decimals=2, mistakes={"avec w = 0, quelle prédiction garde-t-on ? Relis la formule": [0.5, 0.1]}'''),
        ("b", VEC2, "W1_B51", r'''decimals=2, mistakes={"la guidance n'est pas une moyenne des deux prédictions : relis la formule": [0.65, -0.2],
          "(1 + w) multiplie la prédiction avec condition, w celle sans condition": cfg_b51(1, EU_B51, EC_B51),
          "relis la question : w = 1": W0_B51}'''),
        ("c", VEC2, "W3_B51", r'''decimals=2, mistakes={"(1 + w) multiplie la prédiction avec condition, w celle sans condition": cfg_b51(3, EU_B51, EC_B51),
          "c'est guidance_scale = 3, la convention de diffusers : relis le piège classique de 205.7": cfg_b51(2),
          "relis la question : w = 3": W1_B51}'''),
        ("d", WHOLE, "4", r'''mistakes={"relis le piège classique « deux conventions pour l'échelle de guidance » (§205.7)": 3,
          "guidance_scale = 1 + w : vérifie le sens": 2}'''),
        ("e", DEC2, "GAP_B51", r'''decimals=2, mistakes={"multiplie l'écart des deux prédictions par w": GAP_B51 / 3,
          "la norme euclidienne est la racine de la somme des carrés": 3 * (0.3 + 0.6),
          "relis la question : on demande l'écart à ε_c, pas la norme de ε̃": float(np.linalg.norm(W3_B51))}'''),
        ("f", TF, "True", r'''mistakes={"remplace w par −1 dans la formule": False}'''),
        ("g", TF, "False", r'''mistakes={"compare ta réponse c) aux deux prédictions, composante par composante": True}'''),
        ("h", WHOLE, "EVALS_B51", r'''mistakes={"la guidance demande deux prédictions à chaque pas": 50,
          "relis la question : 50 pas DDIM, pas 1 000": 2 * 1000}'''),
    ]),
    Paper("B5.2", "Planning linéaire : β_t, α_t et ᾱ_t pour T = 4, puis ᾱ_T pour T = 1 000", [
        ("a", DEC2, "BETA2_B52", r'''decimals=2, mistakes={"4 valeurs de 0,05 à 0,35 font 3 intervalles, pas 4": 0.05 + 0.3 / 4,
          "relis la question : β_2 est la deuxième valeur, à l'indice 1": 0.25}'''),
        ("b", DEC4, "AB2_B52", r'''decimals=4, mistakes={"c'est α_2 : ᾱ_2 est le produit des α jusqu'au pas 2": float(ALPHAS_B52[1]),
          "ᾱ est un produit des 1 − β, pas 1 moins la somme des β": 1 - 0.05 - 0.15,
          "relis la question : on demande ᾱ_2, au pas 2": float(AB_B52[0])}'''),
        ("c", DEC4, "AB4_B52", r'''decimals=4, mistakes={"c'est α_4 : ᾱ_4 est le produit des α jusqu'au pas 4": float(ALPHAS_B52[3]),
          "ᾱ est un produit des 1 − β, pas 1 moins la somme des β": 1 - float(BETAS_B52.sum()),
          "on demande ᾱ_4, pas sa racine": KEEP4_B52,
          "relis la question : le produit va jusqu'au pas 4": float(AB_B52[2])}'''),
        ("d", DEC4, "KEEP4_B52", r'''decimals=4, mistakes={"c'est ᾱ_4 : relis la forme fermée, le coefficient de x₀ est sa racine": AB4_B52,
          "relis la question : on demande le coefficient de x₀, pas celui du bruit": math.sqrt(1 - AB4_B52)}'''),
        ("e", DEC4, "APPROX4_B52", r'''decimals=4, mistakes={"recompte la somme des quatre β": math.exp(-1.0),
          "relis la question : on demande l'approximation e^(−Σβ), pas la valeur exacte": AB4_B52}'''),
        ("f", TF, "False", r'''mistakes={"compare tes réponses c) et e) : calcule l'écart relatif": True}'''),
        ("g", DEC2, "SUM1000_B52", r'''decimals=2, mistakes={"la somme d'une suite arithmétique est T × (premier + dernier) / 2 : n'oublie pas le / 2": 2 * SUM1000_B52,
          "relis la question : on demande la somme des 1 000 β, pas leur moyenne": SUM1000_B52 / 1000,
          "relis l'énoncé : le dernier β vaut 0,02": 1000 * (1e-4 + 0.2) / 2}'''),
        ("h", DEC2, "AB1000_E5_B52", r'''decimals=2, mistakes={"garde la somme des β de g) sans l'arrondir à 10": math.exp(-10) * 1e5,
          "relis la question : multiplie l'approximation par 10⁵": math.exp(-SUM1000_B52) * 1e6,
          "tu as arrondi e^(−Σβ) à cinq décimales : pour un si petit nombre, garde cinq chiffres significatifs": 4.0,
          "c'est le produit exact des 1 − β : on demande l'approximation e^(−Σβ)": float(np.prod(1 - np.linspace(1e-4, 0.02, 1000))) * 1e5}'''),
        ("i", DEC2, "KEEP1000_PCT_B52", r'''decimals=2, mistakes={"h) est multiplié par 10⁵ : reviens à ᾱ_T avant de prendre la racine": math.sqrt(AB1000_E5_B52),
          "relis la question : en pourcentage, donc multiplié par 100": KEEP1000_PCT_B52 / 100,
          "c'est la racine du produit exact : l'énoncé demande celle de ta valeur de h)": 100 * math.sqrt(float(np.prod(1 - np.linspace(1e-4, 0.02, 1000))))}'''),
        ("j", DEC2, "SUM400_B52", r'''decimals=2, mistakes={"relis la question : les mêmes bornes, mais 400 β seulement": SUM1000_B52,
          "n'oublie pas le / 2 de la somme d'une suite arithmétique": 2 * SUM400_B52}'''),
        ("k", DEC2, "SUM400_SCALED_B52", r'''decimals=2, mistakes={"relis la question : les deux bornes sont multipliées par 1 000/400": SUM400_B52,
          "n'oublie pas le / 2 de la somme d'une suite arithmétique": 2 * SUM400_SCALED_B52}'''),
        ("l", TF, "True", r'''mistakes={"compare tes réponses g) et k)": False}'''),
    ]),
    Paper("B5.3", "Rapport signal sur bruit : à quel pas un chiffre devient-il illisible, planning linéaire ou cosinus ?", [
        ("a", DEC3, "S500_B53", r'''decimals=3, mistakes={"relis la formule de S(t) : t(t − 1), pas t²": lin_sum_b53(500, square=True),
          "S(500) somme les 500 premiers β, pas 500 fois la moyenne de tous les β": 500 * (1e-4 + 0.02) / 2}'''),
        ("b", DEC2, "LIN500_B53", r'''decimals=2, mistakes={"le SNR met la part de l'image au numérateur, celle du bruit au dénominateur": 1 / LIN500_B53,
          "c'est ᾱ_t : on demande le SNR": math.exp(-S500_B53)}'''),
        ("c", DEC2, "COS500_B53", r'''decimals=2, mistakes={"n'oublie pas le décalage s = 0,008": 1.0,
          "c'est ᾱ_t : on demande le SNR": cos_ab_b53(0.5)}'''),
        ("d", DEC2, "LIN1_B53", r'''decimals=2, mistakes={"SNR = 1 veut dire ᾱ = 1/2 : relis le lien entre ᾱ_t et S(t)": lin_cross_b53(1.0),
          "résous l'équation du second degré en t, puis divise t par T": lin_cross_b53(math.log(2)) / 2}'''),
        ("e", DEC2, "COS1_B53", r'''decimals=2, mistakes={"SNR = 1 veut dire ᾱ = 1/2 : c'est le carré du cosinus qui vaut 1/2, pas le cosinus": cos_cross_b53(0.25)}'''),
        ("f", DEC2, "LIN01_B53", r'''decimals=2, mistakes={"SNR = 0,1 ne veut pas dire ᾱ = 0,1 : écris ᾱ en fonction du SNR": lin_cross_b53(math.log(10)),
          "relis la question : le seuil de lisibilité est 0,1, pas 1": LIN1_B53}'''),
        ("g", DEC2, "COS01_B53", r'''decimals=2, mistakes={"n'oublie pas le décalage s = 0,008": cos_cross_b53(1 / 11, s=0.0),
          "SNR = 0,1 ne veut pas dire ᾱ = 0,1 : écris ᾱ en fonction du SNR": cos_cross_b53(0.1)}'''),
        ("h", TF, "True", r'''mistakes={"compare ta réponse f) à 0,5": False}'''),
        ("i", TF, "False", r'''mistakes={"relis ta réponse g) : quelle part des pas reste lisible ?": True}'''),
        ("j", LETTER, '"B"', letters("B", {"A": "compare f) − d) et g) − e)",
                                           "C": "calcule f) − d) et g) − e) : sont-elles égales ?"})),
    ]),
    Paper("B5.4", "Un pas DDPM à la main, en dimension 1, avec un bruit prédit donné", [
        ("a", DEC4, "COEF_B54", r'''decimals=4, mistakes={"relis l'algorithme 2 : β_t est divisé par la racine de 1 − ᾱ_t": float(BETAS_B52[2] / (1 - AB_B52[2])),
          "décalage d'un cran : le pas 3 est l'indice 2, avec β_3 = 0,25": float(BETAS_B52[3] / math.sqrt(1 - AB_B52[3])),
          "relis l'algorithme 2 : c'est 1 − ᾱ_t, pas 1 − α_t": float(BETAS_B52[2] / math.sqrt(BETAS_B52[2]))}'''),
        ("b", DEC4, "MEAN_B54", r'''decimals=4, mistakes={"vérifie le signe : on retire le bruit prédit": ddpm_mean_b54(X3_B54, EPS_B54, sign=1.0),
          "relis l'algorithme 2 : β_t est divisé par la racine de 1 − ᾱ_t": ddpm_mean_b54(X3_B54, EPS_B54, root=False),
          "on divise par la racine de α_t, pas de ᾱ_t": float((X3_B54 - COEF_B54 * EPS_B54) / math.sqrt(AB_B52[2])),
          "décalage d'un cran : le pas 3 est l'indice 2": ddpm_mean_b54(X3_B54, EPS_B54, i=3)}'''),
        ("c", DEC4, "BTILDE_B54", r'''decimals=4, mistakes={"c'est β_3 : la variance du posterior a un facteur en plus": float(BETAS_B52[2]),
          "décalage d'un cran : le pas 3 est l'indice 2": posterior_b54(3)[2],
          "relis la formule de β̃ : (1 − ᾱ_{t−1}) au numérateur, (1 − ᾱ_t) au dénominateur": float(BETAS_B52[2] * (1 - AB_B52[2]) / (1 - AB_B52[1]))}'''),
        ("d", DEC4, "X2_B54", r'''decimals=4, mistakes={"z multiplie l'écart-type σ_t, pas la variance": MEAN_B54 + BTILDE_B54 * Z_B54,
          "vérifie le signe de z : z = −1": MEAN_B54 - math.sqrt(BTILDE_B54) * Z_B54,
          "relis la question : on demande x₂, avec le tirage z": MEAN_B54}'''),
        ("e", DEC4, "X2_BETA_B54", r'''decimals=4, mistakes={"z multiplie l'écart-type, la racine de β_3": MEAN_B54 + float(BETAS_B52[2]) * Z_B54,
          "relis la question : la variance est β_3, pas β̃_3": X2_B54}'''),
        ("f", DEC4, "CLIP_MEAN_B54", r'''decimals=4, mistakes={"tu n'as pas écrêté x̂₀ dans [−1, 1]": NOCLIP_MEAN_B54,
          "relis la question : on demande la moyenne, pas x̂₀ écrêté": 1.0,
          "as-tu arrondi les coefficients du posterior à quatre décimales ? Garde-en cinq : ce chemin tombe à la limite d'un arrondi": 1.2036}'''),
        ("g", DEC4, "LAST_B54", r'''decimals=4, mistakes={"vérifie le signe : on retire le bruit prédit": ddpm_mean_b54(0.5, 0.2, i=0, sign=1.0),
          "relis la question : au pas 1, on prend les constantes de l'indice 0": ddpm_mean_b54(0.5, 0.2, i=1)}'''),
        ("h", TF, "False", r'''mistakes={"relis la docstring de ddpm_step (« no noise at all at t = 0 ») et la phrase qui suit l'algorithme 2": True}'''),
        ("i", TF, "True", r'''mistakes={"compare tes calculs de d) et e) : quelle partie change ?": False}'''),
    ]),
    Paper("B5.5", "Un pas DDIM à la main et l'image estimée x̂_0", [
        ("a", DEC4, "X0HAT_B55", r'''decimals=4, mistakes={"vérifie le signe : on retire le bruit de x_t": x0hat_b55(eps=-EPS_B55),
          "relis la formule de x̂₀ : des racines de ᾱ_t et de 1 − ᾱ_t": float((X4_B55 - (1 - AB_B52[3]) * EPS_B55) / AB_B52[3]),
          "décalage d'un cran : le pas 4 est l'indice 3": x0hat_b55(i=2)}'''),
        ("b", DEC4, "X1_B55", r'''decimals=4, mistakes={"le pas DDIM réutilise aussi le bruit prédit, multiplié par la racine de 1 − ᾱ_{t'}": ddim_b55(3, 0, noise_term=False)[0],
          "relis la question : on s'arrête au pas 1 (indice 0), pas à l'image propre": X0HAT_B55,
          "relis la question : du pas 4 au pas 1": X2_B55}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "ta librairie compte les pas en indices : le pas t est l'indice t − 1",
                                           "C": "l'indice −1 désigne l'image propre : relis la question, on va au pas 1"})),
        ("d", DEC4, "X2_B55", r'''decimals=4, mistakes={"le pas DDIM réutilise aussi le bruit prédit": ddim_b55(3, 1, noise_term=False)[0],
          "décalage d'un cran : le pas 2 est l'indice 1": ddim_b55(3, 2)[0]}'''),
        ("e", TF, "True", r'''mistakes={"recalcule x̂₀ à partir de ton x₂ et du même ε_θ : que trouves-tu ?": False}'''),
        ("f", DEC4, "SIGMA2_43_B55", r'''decimals=4, mistakes={"c'est β_4 : relis la formule de σ dans « Le paramètre η »": float(BETAS_B52[3]),
          "on demande σ², pas σ": math.sqrt(SIGMA2_43_B55),
          "relis la formule de σ : le premier rapport est (1 − ᾱ_{t'}) / (1 − ᾱ_t)": float((1 - AB_B52[3]) / (1 - AB_B52[2]) * (1 - AB_B52[3] / AB_B52[2]))}'''),
        ("g", DEC4, "SIGMA_41_B55", r'''decimals=4, mistakes={"on demande σ, pas σ²": SIGMA_41_B55 ** 2,
          "σ tombe près d'une limite d'arrondi : as-tu arrondi σ² à cinq décimales avant la racine ? Garde-en six": math.sqrt(round(SIGMA_41_B55 ** 2, 5)),
          "relis la question : du pas 4 au pas 1, soit de l'indice 3 à l'indice 0": ddim_b55(3, 1, eta=1.0)[1]}'''),
        ("h", DEC4, "DIR_41_B55", r'''decimals=4, mistakes={"n'oublie pas de retirer σ² sous la racine": math.sqrt(1 - AB_B52[0]),
          "prends la racine de 1 − ᾱ₁ − σ²": float(1 - AB_B52[0] - SIGMA_41_B55 ** 2),
          "1 − ᾱ₁ et σ² sont presque égaux : leur différence a perdu des chiffres ; garde six décimales de σ²": math.sqrt(1 - AB_B52[0] - round(SIGMA_41_B55 ** 2, 5)),
          "tu as repris σ arrondi de g) : 1 − ᾱ₁ − σ² est une petite différence, calcule σ² sans l'arrondir": math.sqrt(1 - AB_B52[0] - round(SIGMA_41_B55, 4) ** 2)}'''),
        ("i", TF, "False", r'''mistakes={"compare tes réponses g) et h)": True}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ B5.1 à B5.5)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; un "
               "vecteur : une liste de nombres (`[0.5, -1.25]`). Arrondis comme l'énoncé le demande, et seulement à "
               "la fin du calcul. Les réponses pas encore remplies affichent ⏳. Les questions « dans ta copie », les "
               "dérivations ∂ B5.6 et B5.7, l'estimation 🧮 B5.8, la réflexion (🗣️ B5.9, ⚖️ B5.10, 📄 B5.11) et "
               "l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to C (exercises B5.12 to B5.31): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "B5.12 à B5.18", "les mathématiques de ta librairie et leurs contrôles : plannings de bruit, bruitage en "
                           "forme fermée, posterior, pas DDPM et DDIM, aller-retour avec le vrai bruit"),
    ("B", "B5.19 et B5.20", "jouets 2D : un MLP débruiteur conditionné par le pas, puis le flow matching"),
    ("C", "B5.21 à B5.31", "le U-Net sur MNIST et Fashion-MNIST : quatre bugs, entraînement avec EMA, DDPM contre "
                           "DDIM, guidance, mémorisation, distance de Fréchet, script autonome et défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | B5.Q1 à B5.Q11, B5.R1 à B5.R3, B5.1 à B5.5 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(ex.table_row(part.key if i == 0 else ""))
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# B5 · Modèles de diffusion : un DDPM minimal — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/b5_diffusion/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py B5`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# B5 · Modèles de diffusion : un DDPM minimal — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Dériver la forme fermée du bruitage et le posterior d'un pas, et calculer à la main un pas DDPM "
               "et un pas DDIM.\n"
               "- Implémenter en NumPy les plannings de bruit, le bruitage et les pas d'échantillonnage, et les "
               "contrôler par l'aller-retour avec le vrai bruit.\n"
               "- Écrire et entraîner un petit U-Net conditionné par le pas, l'échantillonner par DDPM et DDIM, et "
               "le conditionner par la classe avec la guidance sans classifieur.\n"
               "- Évaluer un générateur (distance de Fréchet, plus proches voisins) et le situer face au VAE et au "
               "GAN.\n\n"
               "**Rappel express.** Pas de l'article $t = 1..T$, indice du code $t - 1$. $\\alpha_t = 1 - \\beta_t$, "
               "$\\bar\\alpha_t = \\prod_{s \\le t} \\alpha_s$, $\\mathbf{x}_t = \\sqrt{\\bar\\alpha_t}\\,\\mathbf{x}_0 + "
               "\\sqrt{1 - \\bar\\alpha_t}\\,\\boldsymbol{\\varepsilon}$, $\\operatorname{SNR}(t) = \\bar\\alpha_t / (1 - "
               "\\bar\\alpha_t)$. Posterior : $\\tilde\\beta_t = \\frac{1 - \\bar\\alpha_{t-1}}{1 - \\bar\\alpha_t}\\beta_t$. "
               "DDPM : $\\mathbf{x}_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}\\big(\\mathbf{x}_t - \\frac{\\beta_t}{\\sqrt{1 - "
               "\\bar\\alpha_t}}\\boldsymbol{\\varepsilon}_\\theta\\big) + \\sigma_t\\mathbf{z}$, sans bruit au dernier pas. "
               "DDIM ($\\eta = 0$) : $\\hat{\\mathbf{x}}_0 = (\\mathbf{x}_t - \\sqrt{1 - \\bar\\alpha_t}\\,"
               "\\boldsymbol{\\varepsilon}_\\theta) / \\sqrt{\\bar\\alpha_t}$, puis $\\mathbf{x}_{t'} = "
               "\\sqrt{\\bar\\alpha_{t'}}\\,\\hat{\\mathbf{x}}_0 + \\sqrt{1 - \\bar\\alpha_{t'}}\\,\\boldsymbol{\\varepsilon}_\\theta$. "
               "Guidance : $\\tilde{\\boldsymbol{\\varepsilon}} = (1 + w)\\,\\boldsymbol{\\varepsilon}_\\theta(\\mathbf{x}_t, c) - "
               "w\\,\\boldsymbol{\\varepsilon}_\\theta(\\mathbf{x}_t, \\varnothing)$, `guidance_scale` $= 1 + w$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code B5.12 à B5.31 (mathématiques de ta librairie, jouets 2D, U-Net sur "
             "MNIST et Fashion-MNIST) seront ajoutés à la prochaine session de génération ; relance alors "
             "`python tools/start_chapter.py B5` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu calculer à la main un planning de bruit, ses $\\bar\\alpha_t$ et son rapport signal sur "
               "bruit ?\n"
               "2. Sais-tu dériver la forme fermée et le posterior d'un pas, puis calculer un pas DDPM et un pas "
               "DDIM ?\n"
               "3. Sais-tu appliquer la guidance sans classifieur et traduire `guidance_scale` en $w$ ?\n"
               "4. Sais-tu estimer le coût d'un échantillonnage, et ce que gagnent DDIM, la distillation et l'espace "
               "latent ?\n\n"
               "**Pour aller plus loin** : l'article de Ho, Jain et Abbeel et le billet de L. Weng, cités dans la "
               "fiche. La suite : le bonus B6, l'explicabilité et l'équité."
               + later)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT + LIBRARY_CHECKS, PAPER_INTRO)
    for part in PARTS:
        cells += part_cells(part, kind)
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
