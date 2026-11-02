"""Code cells of the chapter B5 notebook, first half (parts A and B and the U-Net of part C: B5.12 to B5.21), shared by
build_chB5.py and history_chB5.py (used by Claude).

Each constant is the source of one cell: TOOLS_* (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
"""

TOOLS_A = r'''# Tools of parts A and B: your NumPy library diffusion.py, then the 2-D toys
import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from wb import diffusion as wbd
from wb.nbtools import (StopCheck, checked, filled, fr, pair, refuses, returned, run_pytest_group, same, shape_text,
                        verdict, written)

TEST_DIFF = "tests/test_chb5_diffusion.py"          # the tests of your library diffusion.py
T_STEPS = wbd.n_steps(FAST_MODE)                     # 400 steps in fast mode, 1 000 in full mode
BETAS = wbd.schedule_betas("linear", T_STEPS)        # linear, bounds x 1000 / T (computed here, without your code)
DIGITS = wbd.image_subset("mnist", 1000, split="test")   # 1 000 MNIST test digits in [-1, 1], always the same


def fr_e(x):
    """A small number in French scientific notation: 3,6e-05."""
    return f"{x:.1e}".replace(".", ",")


def toy_schedule(betas):
    """The nine constants of noise_schedule (B5.12), written once without your code, for the checks."""
    b = np.asarray(betas, dtype=float)
    a = 1 - b
    ab = np.cumprod(a)
    ab_prev = np.concatenate([[1.0], ab[:-1]])
    return {"betas": b, "alphas": a, "alpha_bars": ab, "alpha_bars_prev": ab_prev, "sqrt_alpha_bars": np.sqrt(ab),
            "sqrt_one_minus_alpha_bars": np.sqrt(1 - ab), "posterior_variance": b * (1 - ab_prev) / (1 - ab),
            "posterior_mean_coef1": b * np.sqrt(ab_prev) / (1 - ab),
            "posterior_mean_coef2": (1 - ab_prev) * np.sqrt(a) / (1 - ab)}


def show_images(images, titles=None, ncol=None, size=1.3, suptitle=None):
    """Images in [-1, 1] (a tensor or an array (N, 1, 28, 28)) in a grid of ncol columns."""
    arr = images.detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images, dtype=float)
    arr = wbd.to_unit(arr.reshape(-1, 28, 28))
    ncol = ncol or len(arr)
    nrow = math.ceil(len(arr) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow + (0.4 if suptitle else 0.1)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, img in enumerate(arr):
        axes.ravel()[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        if titles is not None:
            axes.ravel()[i].set_title(titles[i], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


print(f"T = {T_STEPS} pas en {'mode rapide' if FAST_MODE else 'mode complet'} ; planning linéaire de bornes "
      f"{fr_e(BETAS[0])} et {fr_e(BETAS[-1])} : ᾱ au dernier indice = {fr_e(wbd.alpha_bars(BETAS)[-1])}. "
      f"{len(DIGITS)} chiffres de test de MNIST dans [−1, 1].")'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="B5")   # reload your saved file\n'

# --- 🔨 B5.12 ----------------------------------------------------------------------------------------------------------
CHECK_12 = r'''with checked("B5.12"):
    M = mylearn.diffusion
    lin_12 = M.linear_beta_schedule(5)
    if returned("B5.12", "linear_beta_schedule", lin_12):
        verdict("B5.12", np.asarray(lin_12).dtype == np.float64, "linear_beta_schedule rend des float64.",
                f"linear_beta_schedule doit rendre des float64, reçu {np.asarray(lin_12).dtype}.")
        same("B5.12", "linear_beta_schedule(5) (exemple de la docstring)", lin_12, np.linspace(1e-4, 0.02, 5))
    same("B5.12", f"linear_beta_schedule({T_STEPS}, …) aux bornes × 1000/T, contre np.linspace",
         M.linear_beta_schedule(T_STEPS, 1e-4 * 1000 / T_STEPS, 0.02 * 1000 / T_STEPS), BETAS)
    same("B5.12", "linear_beta_schedule(1, 0.3, 0.5) : un seul pas, beta_start comme np.linspace",
         M.linear_beta_schedule(1, 0.3, 0.5), [0.3])
    refuses("B5.12", "linear_beta_schedule(10, beta_start=nan)", lambda: M.linear_beta_schedule(10, float("nan")))
    refuses("B5.12", "linear_beta_schedule(10, 0.02, 0.01) (beta_start > beta_end)",
            lambda: M.linear_beta_schedule(10, 0.02, 0.01))
    refuses("B5.12", "linear_beta_schedule(0)", lambda: M.linear_beta_schedule(0))

    f_12 = np.cos((np.arange(6) / 5 + 0.008) / 1.008 * np.pi / 2) ** 2          # f at the T + 1 = 6 points t / T
    same("B5.12", "cosine_beta_schedule(5) contre la forme fermée", M.cosine_beta_schedule(5),
         np.minimum(1 - f_12[1:] / f_12[:-1], 0.999))
    same("B5.12", "cosine_beta_schedule(1000) contre la forme fermée", M.cosine_beta_schedule(1000),
         wbd.schedule_betas("cosine", 1000))
    same("B5.12", "cosine_beta_schedule(8, max_beta=0.3) : l'écrêtage atteint au dernier indice",
         np.asarray(M.cosine_beta_schedule(8, max_beta=0.3))[-1], 0.3)
    refuses("B5.12", "cosine_beta_schedule(10, max_beta=1.0)", lambda: M.cosine_beta_schedule(10, max_beta=1.0))
    refuses("B5.12", "cosine_beta_schedule(10, s=-0.1)", lambda: M.cosine_beta_schedule(10, s=-0.1))

    betas_12 = np.array([0.1, 0.2, 0.5])
    sched_12 = M.noise_schedule(betas_12)
    if returned("B5.12", "noise_schedule", sched_12):
        want_12 = toy_schedule(betas_12)
        keys_12 = set(sched_12) if isinstance(sched_12, dict) else set()
        verdict("B5.12", keys_12 == set(want_12), "noise_schedule rend les neuf clés de la docstring.",
                f"noise_schedule doit rendre un dictionnaire aux clés {sorted(want_12)} ; manquent "
                f"{sorted(set(want_12) - keys_12)}, en trop {sorted(keys_12 - set(want_12))}.")
        for name_12 in sorted(set(want_12) & keys_12):
            same("B5.12", f"noise_schedule([0.1, 0.2, 0.5])[{name_12!r}]", sched_12[name_12], want_12[name_12])
        if "posterior_variance" in keys_12:
            verdict("B5.12", np.asarray(sched_12["posterior_variance"])[0] == 0,
                    "posterior_variance[0] vaut 0 exactement.",
                    "posterior_variance[0] doit valoir 0 exactement (alpha_bars_prev[0] = 1).")
    sched32_12 = M.noise_schedule(np.array([0.1, 0.2], dtype=np.float32))
    verdict("B5.12", all(np.asarray(v).dtype == np.float64 for v in sched32_12.values()),
            "des betas en float32 donnent des constantes en float64.",
            "noise_schedule doit rendre des float64, même pour des betas en float32.")
    verdict("B5.12", np.array_equal(betas_12, [0.1, 0.2, 0.5]), "les betas reçus n'ont pas changé.",
            "noise_schedule a modifié les betas reçus.")
    refuses("B5.12", "noise_schedule([0.1, nan])", lambda: M.noise_schedule([0.1, float("nan")]))
    refuses("B5.12", "noise_schedule([0.1, 1.0])", lambda: M.noise_schedule([0.1, 1.0]))
    refuses("B5.12", "noise_schedule([[0.1, 0.2]]) (2-D)", lambda: M.noise_schedule([[0.1, 0.2]]))

    end_default_12 = float(M.noise_schedule(M.linear_beta_schedule(T_STEPS))["alpha_bars"][-1])
    end_scaled_12 = float(M.noise_schedule(BETAS)["alpha_bars"][-1])
    print(f"ᾱ au dernier indice, T = {T_STEPS} : bornes par défaut {fr_e(end_default_12)}, bornes × 1000/T "
          f"{fr_e(end_scaled_12)} (avec ta librairie).")'''

# --- 🔮 B5.13 ----------------------------------------------------------------------------------------------------------
GIVEN_13 = r'''CLF = wbd.classifier("mnist")                      # the CNN of ch. 27, trained once on clean digits, then reloaded
STEPS_13 = (100, 300, 600)                          # steps of the paper: indices 99, 299, 599
AB_13 = {"linéaire": wbd.alpha_bars(wbd.schedule_betas("linear", 1000)),
         "cosinus": wbd.alpha_bars(wbd.schedule_betas("cosine", 1000))}
EPS_13 = np.random.default_rng(13).standard_normal(tuple(DIGITS.images.shape))   # one noise per digit, for every case
X0_13 = DIGITS.images.double().numpy()


def noisy_13(kind, step):
    """The 1 000 test digits at the step `step` of the paper (index step - 1) of the schedule `kind` (T = 1 000),
    all with the same noise EPS_13 (the closed form, written here without your q_sample)."""
    a = AB_13[kind][step - 1]
    return math.sqrt(a) * X0_13 + math.sqrt(1 - a) * EPS_13


print(f"Le classifieur reconnaît {fr(100 * np.mean(wbd.predict(CLF, X0_13) == DIGITS.labels.numpy()), 1)} % des "
      "1 000 chiffres propres.")'''

TODO_13 = r'''strong_13 = ...    # a) in how many of the 6 cases (steps 100, 300, 600 x linear, cosine) does the classifier
                   #    recognise at least 70 % of the noisy digits? an integer from 0 to 6
cos600_13 = ...    # b) cosine schedule, step 600: does the classifier recognise MORE than half of the digits?
                   #    True or False'''

EXPERIMENT_13 = r'''# 1. four digits, clean then at the steps 100, 300 and 600 of each schedule (the same noise everywhere)
for kind_13 in AB_13:
    row_13 = [X0_13[:4]] + [noisy_13(kind_13, s)[:4] for s in STEPS_13]
    show_images(np.stack(row_13, axis=1).reshape(-1, 1, 28, 28), ncol=16,
                titles=[f"{s}" for _ in range(4) for s in ("propre",) + STEPS_13], size=0.9,
                suptitle=f"planning {kind_13}, T = 1 000 : chaque chiffre propre, puis aux pas 100, 300, 600")

# 2. the accuracy of the classifier, and the SNR, in each of the 6 cases
acc_13 = {}
for kind_13 in AB_13:
    for step_13 in STEPS_13:
        a_13 = AB_13[kind_13][step_13 - 1]
        pred_13 = wbd.predict(CLF, noisy_13(kind_13, step_13))
        acc_13[kind_13, step_13] = float(np.mean(pred_13 == DIGITS.labels.numpy()))
        top_13 = np.bincount(pred_13, minlength=10)
        print(f"{kind_13:>9} pas {step_13} : SNR {fr(a_13 / (1 - a_13), 3):>7} · accuracy {fr(100 * acc_13[kind_13, step_13], 1):>5} % "
              f"· classe la plus prédite : {top_13.argmax()} ({fr(100 * top_13.max() / len(pred_13), 0)} % des chiffres)")
wb.check("B5.13a", strong_13)
wb.check("B5.13b", cos600_13)'''

# --- 🔨 B5.14 ----------------------------------------------------------------------------------------------------------
CHECK_14 = r'''xt_14 = None
with checked("B5.14"):
    M = mylearn.diffusion
    ab_14 = np.array([0.9, 0.72, 0.36])
    x0_14 = np.array([[1.0, -1.0], [0.5, 0.0]])
    out_14, noise_14 = pair("B5.14", "q_sample", M.q_sample(x0_14.copy(), [0, 2], ab_14, noise=np.ones((2, 2))),
                            names=("x_t", "noise"))
    same("B5.14", "q_sample de l'exemple de la docstring", out_14,
         np.sqrt(ab_14[[0, 2]])[:, None] * x0_14 + np.sqrt(1 - ab_14[[0, 2]])[:, None], rtol=1e-6, atol=1e-6)

    ab_nb = wbd.alpha_bars(BETAS)
    imgs_14 = DIGITS.images[:6].double().numpy()
    t_14 = np.array([0, 1, 50, 150, 300, T_STEPS - 1])
    eps_14 = np.random.default_rng(14).standard_normal(imgs_14.shape)
    copies_14 = (imgs_14.copy(), t_14.copy(), eps_14.copy(), ab_nb.copy())
    xt_14, n_14 = pair("B5.14", "q_sample", M.q_sample(imgs_14, t_14, ab_nb, noise=eps_14), names=("x_t", "noise"))
    want_14 = np.sqrt(ab_nb[t_14])[:, None, None, None] * imgs_14 + np.sqrt(1 - ab_nb[t_14])[:, None, None, None] * eps_14
    if not same("B5.14", "6 chiffres (6, 1, 28, 28), un pas par image", xt_14, want_14, rtol=1e-6, atol=1e-6):
        xt_14 = None
    same("B5.14", "le bruit rendu est celui reçu", n_14, eps_14, rtol=1e-6, atol=1e-6)
    verdict("B5.14", all(np.array_equal(a, b) for a, b in zip((imgs_14, t_14, eps_14, ab_nb), copies_14)),
            "les entrées n'ont pas changé.", "q_sample a modifié une de ses entrées (x0, t, noise ou alpha_bars).")
    same("B5.14", "x0 en 1-D (3 exemples d'une valeur)", M.q_sample(np.array([0.5, -0.5, 1.0]), [0, 1, 2], ab_14,
                                                                    noise=np.array([1.0, 2.0, 3.0]))[0],
         np.sqrt(ab_14) * [0.5, -0.5, 1.0] + np.sqrt(1 - ab_14) * [1.0, 2.0, 3.0], rtol=1e-6, atol=1e-6)
    x_r, n_r = pair("B5.14", "q_sample", M.q_sample(x0_14, [1, 1], ab_14, rng=np.random.default_rng(5)),
                    names=("x_t", "noise"))
    same("B5.14", "sans noise : le bruit de rng.standard_normal(x0.shape), rendu", n_r,
         np.random.default_rng(5).standard_normal((2, 2)), rtol=1e-6, atol=1e-6)
    refuses("B5.14", "t de longueur 1 pour 2 exemples", lambda: M.q_sample(x0_14, [1], ab_14, noise=np.ones((2, 2))))
    refuses("B5.14", "un bruit de forme (2,) pour x0 de forme (2, 2)",
            lambda: M.q_sample(x0_14, [0, 1], ab_14, noise=np.ones(2)))
    refuses("B5.14", "t = -1", lambda: M.q_sample(x0_14, [-1, 0], ab_14, noise=np.ones((2, 2))))
    refuses("B5.14", "t = T", lambda: M.q_sample(x0_14, [0, 3], ab_14, noise=np.ones((2, 2))))
if xt_14 is not None:
    show_images(xt_14, titles=[f"indice {t}" for t in t_14], size=1.5,
                suptitle=f"ta q_sample : 6 chiffres, un pas chacun (T = {T_STEPS})")'''

# --- 📈 B5.15 ----------------------------------------------------------------------------------------------------------
GIVEN_15 = r'''# The four schedules of the figure, against the fraction t / T of the chain (computed without your code)
SCHEDULES_15 = {"L": ("linéaire, T = 1 000", wbd.schedule_betas("linear", 1000)),
                "C": ("cosinus, T = 1 000", wbd.schedule_betas("cosine", 1000)),
                "D": ("linéaire, T = 400, bornes par défaut", np.linspace(1e-4, 0.02, 400)),
                "S": ("linéaire, T = 400, bornes × 1000/T", wbd.schedule_betas("linear", 400))}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for (letter, (label, betas)), style in zip(SCHEDULES_15.items(), ("-", "-", "--", ":")):
    ab = wbd.alpha_bars(betas)
    frac = np.arange(1, len(ab) + 1) / len(ab)
    ax1.plot(frac, ab, style, lw=2, label=f"{letter} : {label}")
    ax2.plot(frac, np.log(ab / (1 - ab)), style, lw=2, label=letter)
ax1.set_xlabel("fraction de la chaîne, (indice + 1) / T")
ax1.set_ylabel("ᾱ")
ax1.legend(fontsize=8)
ax2.set_xlabel("fraction de la chaîne, (indice + 1) / T")
ax2.set_ylabel("ln SNR")
ax2.legend()
plt.show()

# One digit at the indices 0, 99, ..., 999 of the two schedules of T = 1 000, with the same noise
idx_15 = [0] + list(range(99, 1000, 100))
x0_15 = DIGITS.images[0].double().numpy()
eps_15 = np.random.default_rng(15).standard_normal(x0_15.shape)
grid_15 = [math.sqrt(a) * x0_15 + math.sqrt(1 - a) * eps_15
           for key in ("L", "C") for a in wbd.alpha_bars(SCHEDULES_15[key][1])[idx_15]]
show_images(np.stack(grid_15), titles=[f"{k} {i}" for k in ("L", "C") for i in idx_15], ncol=len(idx_15), size=1.35,
            suptitle="le même chiffre et le même bruit, aux indices 0 à 999 : linéaire (L), puis cosinus (C)")'''

TODO_15 = r'''same_curve_15 = ...   # a) "D" or "S": which T = 400 schedule has the alpha_bar curve of L (against t / T)?
largest_end_15 = ...  # b) "L", "C", "D" or "S": which schedule keeps the most of the image at its last index?
cos_above_15 = ...    # c) "A", "B" or "C": over which share of the chain is ln SNR of C above that of L?
                      #    A: less than half; B: between half and 90 %; C: more than 95 %'''

CHECK_15 = r'''with wb.attempt("B5.15"):
    wb.check("B5.15a", same_curve_15)
with wb.attempt("B5.15"):
    wb.check("B5.15b", largest_end_15)
with wb.attempt("B5.15"):
    wb.check("B5.15c", cos_above_15)'''

# --- 🔨 B5.16 ----------------------------------------------------------------------------------------------------------
CHECK_16 = r'''with checked("B5.16"):
    M = mylearn.diffusion
    s_16 = toy_schedule([0.1, 0.2, 0.5])
    x0_16 = np.array([[1.0, -1.0], [0.5, 0.0]])
    xt_16 = np.array([[1.2649, -0.6325], [1.1, 0.8]])
    copies_16 = (x0_16.copy(), xt_16.copy())
    mean_16, var_16 = pair("B5.16", "q_posterior_mean_variance",
                           M.q_posterior_mean_variance(x0_16, xt_16, [0, 2], s_16), names=("mean", "variance"))
    t_16 = np.array([0, 2])
    same("B5.16", "moyenne du posterior (exemple de la docstring)", mean_16,
         s_16["posterior_mean_coef1"][t_16][:, None] * x0_16 + s_16["posterior_mean_coef2"][t_16][:, None] * xt_16,
         rtol=1e-6, atol=1e-6)
    same("B5.16", "variance du posterior, forme (2, 1)", var_16, s_16["posterior_variance"][t_16][:, None],
         rtol=1e-6, atol=1e-6)
    verdict("B5.16", np.asarray(var_16).ravel()[0] == 0, "à l'indice 0, la variance vaut 0 exactement.",
            "à l'indice 0, la variance du posterior doit valoir 0 exactement.")
    verdict("B5.16", all(np.array_equal(a, b) for a, b in zip((x0_16, xt_16), copies_16)),
            "les entrées n'ont pas changé.", "q_posterior_mean_variance a modifié x0 ou x_t.")
    _, var4_16 = M.q_posterior_mean_variance(np.zeros((3, 1, 2, 2)), np.ones((3, 1, 2, 2)), [0, 1, 2], s_16)
    verdict("B5.16", np.shape(var4_16) == (3, 1, 1, 1), "variance de forme (3, 1, 1, 1) pour x0 de forme (3, 1, 2, 2).",
            f"pour x0 de forme (3, 1, 2, 2), la variance doit avoir la forme (3, 1, 1, 1), reçu {np.shape(var4_16)}.")
    _, var1_16 = M.q_posterior_mean_variance(np.zeros(3), np.ones(3), [0, 1, 2], s_16)
    verdict("B5.16", np.shape(var1_16) == (3,), "variance de forme (3,) pour x0 en 1-D.",
            f"pour x0 de forme (3,), la variance doit avoir la forme (3,), reçu {np.shape(var1_16)}.")

    # Bayes on a fine grid, in 1-D: prior q(x_{t-1} | x0) times the likelihood q(x_t | x_{t-1}), at index 2
    x0_b, xt_b, t_b = -0.4, 0.7, 2
    grid = np.linspace(-15, 15, 600_001)
    ab, a = s_16["alpha_bars"], s_16["alphas"]
    log_post = (-(grid - math.sqrt(ab[t_b - 1]) * x0_b) ** 2 / (2 * (1 - ab[t_b - 1]))
                - (xt_b - math.sqrt(a[t_b]) * grid) ** 2 / (2 * (1 - a[t_b])))
    w_b = np.exp(log_post - log_post.max())
    m_b = float((grid * w_b).sum() / w_b.sum())
    mean_b, var_b = M.q_posterior_mean_variance(np.array([x0_b]), np.array([xt_b]), [t_b], s_16)
    same("B5.16", "moyenne contre Bayes sur une grille (x0 = −0,4, x_t = 0,7, indice 2)", mean_b, [m_b], rtol=1e-6, atol=1e-6)
    same("B5.16", "variance contre Bayes sur une grille", var_b, [float(((grid - m_b) ** 2 * w_b).sum() / w_b.sum())],
         rtol=1e-6, atol=1e-6)
    refuses("B5.16", "x_t de forme (2, 3) pour x0 de forme (2, 2)",
            lambda: M.q_posterior_mean_variance(x0_16, np.ones((2, 3)), [0, 1], s_16))
    refuses("B5.16", "t = 3 avec T = 3", lambda: M.q_posterior_mean_variance(x0_16, xt_16, [0, 3], s_16))

    eps_16 = np.array([[0.3, -2.0], [1.0, 0.5]])
    ab_16 = s_16["alpha_bars"]
    noisy_16 = np.sqrt(ab_16[[1, 2]])[:, None] * x0_16 + np.sqrt(1 - ab_16[[1, 2]])[:, None] * eps_16
    same("B5.16", "predict_x0_from_eps : l'aller-retour avec le vrai bruit redonne x0",
         M.predict_x0_from_eps(noisy_16, [1, 2], eps_16, ab_16), x0_16, rtol=1e-6, atol=1e-6)
    refuses("B5.16", "predict_x0_from_eps avec eps de forme (2,)",
            lambda: M.predict_x0_from_eps(noisy_16, [1, 2], np.ones(2), ab_16))
    refuses("B5.16", "predict_x0_from_eps avec t = -1", lambda: M.predict_x0_from_eps(noisy_16, [-1, 2], eps_16, ab_16))'''

# --- 🔨 B5.17 ----------------------------------------------------------------------------------------------------------
CHECK_17 = r'''with checked("B5.17"):
    M = mylearn.diffusion
    s_17 = toy_schedule([0.1, 0.2, 0.5])
    b, a, ab = s_17["betas"], s_17["alphas"], s_17["alpha_bars"]

    def algo2_17(x, t, eps):        # the mean of Algorithm 2 (Ho et al.)
        return (np.asarray(x) - b[t] / math.sqrt(1 - ab[t]) * np.asarray(eps)) / math.sqrt(a[t])

    x_17, e_17 = np.array([[0.5, -0.5]]), np.array([[1.0, 1.0]])
    same("B5.17", "ddpm_step à t = 0 (exemple de la docstring : pas de bruit)", M.ddpm_step(x_17, 0, e_17, s_17),
         algo2_17(x_17, 0, e_17), rtol=1e-6, atol=1e-6)
    xb_17 = np.random.default_rng(1).standard_normal((4, 3))
    eb_17 = np.random.default_rng(2).standard_normal((4, 3))
    copies_17 = (xb_17.copy(), eb_17.copy())
    full_17 = M.ddpm_step(xb_17, 2, eb_17, s_17, rng=np.random.default_rng(7))
    again_17 = M.ddpm_step(xb_17, 2, eb_17, s_17, rng=np.random.default_rng(7))
    verdict("B5.17", np.array_equal(np.asarray(full_17), np.asarray(again_17)), "même graine, même résultat.",
            "deux appels avec la même graine doivent rendre le même x_{t-1} : z est-il tiré de rng ?")
    noise_17 = np.asarray(M.ddpm_step(np.zeros((4, 3)), 2, np.zeros((4, 3)), s_17, rng=np.random.default_rng(7)))
    same("B5.17", "moyenne à l'indice 2 (même graine : la différence de deux appels)", np.asarray(full_17) - noise_17,
         algo2_17(xb_17, 2, eb_17), rtol=1e-6, atol=1e-6)
    beta_17 = np.asarray(M.ddpm_step(np.zeros((4, 3)), 2, np.zeros((4, 3)), s_17, rng=np.random.default_rng(7),
                                     variance="beta"))
    same("B5.17", "bruit avec variance='beta' contre 'posterior' : le rapport des écarts-types",
         beta_17, noise_17 * math.sqrt(b[2] / s_17["posterior_variance"][2]), rtol=1e-6, atol=1e-6)
    verdict("B5.17", all(np.array_equal(u, v) for u, v in zip((xb_17, eb_17), copies_17)), "les entrées n'ont pas changé.",
            "ddpm_step a modifié x_t ou eps_pred.")
    big_17 = np.array([[-3.0, 2.0, 0.1]])                                     # x0_hat far outside [-0.5, 0.5]
    xc_17 = np.array([[0.2, -0.1, 0.3]])
    x0hat_17 = (xc_17 - math.sqrt(1 - ab[1]) * big_17) / math.sqrt(ab[1])
    clipped_17 = (np.asarray(M.ddpm_step(xc_17, 1, big_17, s_17, rng=np.random.default_rng(3), clip_x0=0.5))
                  - np.asarray(M.ddpm_step(np.zeros((1, 3)), 1, np.zeros((1, 3)), s_17, rng=np.random.default_rng(3),
                                           clip_x0=0.5)))
    same("B5.17", "clip_x0 = 0.5 : moyenne du posterior de x̂₀ écrêté à [−0,5 ; 0,5]", clipped_17,
         s_17["posterior_mean_coef1"][1] * np.clip(x0hat_17, -0.5, 0.5) + s_17["posterior_mean_coef2"][1] * xc_17,
         rtol=1e-6, atol=1e-6)
    refuses("B5.17", "variance='betas' (inconnue), même à t = 0", lambda: M.ddpm_step(x_17, 0, e_17, s_17, variance="betas"))
    refuses("B5.17", "ddpm_step à t = 3 avec T = 3", lambda: M.ddpm_step(x_17, 3, e_17, s_17))
    refuses("B5.17", "ddpm_step à t = -1", lambda: M.ddpm_step(x_17, -1, e_17, s_17))

    x0_17 = np.array([[0.5, -1.0, 0.25]])
    eps_17 = np.array([[1.0, 0.0, -2.0]])
    xt_17 = math.sqrt(ab[2]) * x0_17 + math.sqrt(1 - ab[2]) * eps_17
    same("B5.17", "ddim_step de l'indice 2 à -1 avec le vrai bruit : x0", M.ddim_step(xt_17, 2, -1, eps_17, ab), x0_17,
         rtol=1e-6, atol=1e-6)
    same("B5.17", "ddim_step de l'indice 2 à 0 avec le vrai bruit : le x au pas 0 du même bruit",
         M.ddim_step(xt_17, 2, 0, eps_17, ab), math.sqrt(ab[0]) * x0_17 + math.sqrt(1 - ab[0]) * eps_17,
         rtol=1e-6, atol=1e-6)
    rng_17 = np.random.default_rng(11)
    state_17 = copy.deepcopy(rng_17.bit_generator.state)
    M.ddim_step(xt_17, 2, 1, eps_17, ab, eta=0.0, rng=rng_17)
    verdict("B5.17", rng_17.bit_generator.state == state_17, "avec eta = 0, rng n'a pas servi.",
            "avec eta = 0, ddim_step ne doit tirer aucun nombre de rng (son état a changé).")
    refuses("B5.17", "ddim_step avec t_prev = t", lambda: M.ddim_step(xt_17, 2, 2, eps_17, ab))
    refuses("B5.17", "ddim_step avec eta = 1.5", lambda: M.ddim_step(xt_17, 2, 1, eps_17, ab, eta=1.5))
    refuses("B5.17", "ddim_step avec eta = nan", lambda: M.ddim_step(xt_17, 2, 1, eps_17, ab, eta=float("nan")))'''

# --- 🔬 B5.18 ----------------------------------------------------------------------------------------------------------
TODO_18 = r'''def true_noise_18(x_t, t, x0, alpha_bars):
    """The noise eps such that x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps, for one step index t
    (an int, the same for the whole batch); a new array with the shape of x_t."""
    raise NotImplementedError("true_noise_18")


def ddpm_chain_18(x0, eps, schedule, rng, oracle=False):
    """The whole DDPM chain: x at index T - 1 made from x0 and eps by the closed form, then ddpm_step (your library)
    from index T - 1 down to 0, with as eps_pred the initial eps (oracle=False) or the true noise of the current x
    (true_noise_18, oracle=True). Returns the final x."""
    raise NotImplementedError("ddpm_chain_18")


def ddim_chain_18(x0, eps, alpha_bars, timesteps, eta=0.0, rng=None):
    """The DDIM chain on the decreasing sub-sequence timesteps: x at index timesteps[0] made from x0 and eps by the
    closed form, then ddim_step (your library) from each index to the next one, and from the last one to -1, always
    with eps_pred = eps, the initial noise. Returns the final x."""
    raise NotImplementedError("ddim_chain_18")


ddpm_fixed_18 = ...    # a) DDPM, initial eps at every step: does it return x0 within 1e-6? True or False
ddpm_oracle_18 = ...   # b) DDPM, true noise of the current x at every step: within 1e-6 of x0? True or False
ddim_ten_18 = ...      # c) DDIM, eta = 0, 10 steps, initial eps: within 1e-6 of x0? True or False
ddim_eta1_18 = ...     # d) DDIM, eta = 1, all T steps, initial eps: within 1e-6 of x0? True or False'''

SOLUTION_18 = r'''def true_noise_18(x_t, t, x0, alpha_bars):
    """The noise eps such that x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps, for one step index t
    (an int, the same for the whole batch); a new array with the shape of x_t."""
    ab = float(np.asarray(alpha_bars)[t])
    return (np.asarray(x_t, dtype=float) - math.sqrt(ab) * np.asarray(x0, dtype=float)) / math.sqrt(1 - ab)


def ddpm_chain_18(x0, eps, schedule, rng, oracle=False):
    """The whole DDPM chain: x at index T - 1 made from x0 and eps by the closed form, then ddpm_step (your library)
    from index T - 1 down to 0, with as eps_pred the initial eps (oracle=False) or the true noise of the current x
    (true_noise_18, oracle=True). Returns the final x."""
    ab = schedule["alpha_bars"]
    T = len(ab)
    x = math.sqrt(ab[T - 1]) * np.asarray(x0, dtype=float) + math.sqrt(1 - ab[T - 1]) * np.asarray(eps, dtype=float)
    for t in range(T - 1, -1, -1):
        eps_pred = true_noise_18(x, t, x0, ab) if oracle else eps
        x = mylearn.diffusion.ddpm_step(x, t, eps_pred, schedule, rng=rng)
    return x


def ddim_chain_18(x0, eps, alpha_bars, timesteps, eta=0.0, rng=None):
    """The DDIM chain on the decreasing sub-sequence timesteps: x at index timesteps[0] made from x0 and eps by the
    closed form, then ddim_step (your library) from each index to the next one, and from the last one to -1, always
    with eps_pred = eps, the initial noise. Returns the final x."""
    steps = [int(t) for t in timesteps] + [-1]
    a0 = float(alpha_bars[steps[0]])
    x = math.sqrt(a0) * np.asarray(x0, dtype=float) + math.sqrt(1 - a0) * np.asarray(eps, dtype=float)
    for t, t_prev in zip(steps[:-1], steps[1:]):
        x = mylearn.diffusion.ddim_step(x, t, t_prev, eps, alpha_bars, eta=eta, rng=rng)
    return x'''

ANSWERS_18 = r'''ddpm_fixed_18, ddpm_oracle_18, ddim_ten_18, ddim_eta1_18 = False, True, True, False'''

CHECK_18 = r'''chains_ok_18 = False
with checked("B5.18"):
    s_18 = toy_schedule([0.1, 0.2, 0.5])
    ab_18 = s_18["alpha_bars"]
    x0_c = np.array([[0.3, -0.7], [1.2, 0.1]])
    e_c = np.array([[0.5, 1.0], [-1.5, 0.2]])
    xt_c = math.sqrt(ab_18[2]) * x0_c + math.sqrt(1 - ab_18[2]) * e_c
    copy_c = xt_c.copy()
    ok_18 = same("B5.18", "true_noise_18 redonne le bruit de x_t", true_noise_18(xt_c, 2, x0_c, ab_18), e_c,
                 rtol=1e-9, atol=1e-12)
    verdict("B5.18", np.array_equal(xt_c, copy_c), "true_noise_18 n'a pas modifié x_t.", "true_noise_18 a modifié x_t.")
    ok_18 &= np.array_equal(xt_c, copy_c)
    # the DDPM chain with T = 3 and the initial noise, written step by step without your functions
    rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
    x_ref = xt_c.copy()
    for t in (2, 1, 0):
        mean = (x_ref - s_18["betas"][t] / math.sqrt(1 - ab_18[t]) * e_c) / math.sqrt(s_18["alphas"][t])
        x_ref = mean if t == 0 else mean + np.asarray(mylearn.diffusion.ddpm_step(
            np.zeros_like(x_ref), t, np.zeros_like(x_ref), s_18, rng=rng_b))
    ok_18 &= same("B5.18", "ddpm_chain_18 (T = 3, bruit initial) contre la chaîne pas à pas",
                  ddpm_chain_18(x0_c, e_c, s_18, rng_a), x_ref, rtol=1e-6, atol=1e-6)
    x_dd = math.sqrt(ab_18[2]) * x0_c + math.sqrt(1 - ab_18[2]) * e_c
    x0h = (x_dd - math.sqrt(1 - ab_18[2]) * e_c) / math.sqrt(ab_18[2])
    ok_18 &= same("B5.18", "ddim_chain_18 sur [2, 0] (η = 0) contre la formule", ddim_chain_18(x0_c, e_c, ab_18, [2, 0]),
                  x0h, rtol=1e-6, atol=1e-6)
    chains_ok_18 = bool(ok_18)
if not chains_ok_18:
    print("⏳ Ex B5.18 : l'expérience attend tes trois fonctions (et ddpm_step, ddim_step de B5.17).")
else:
    X0_18 = wbd.ring(500, seed=18)                                   # 500 points of the ring of 8 Gaussians
    EPS_18 = np.random.default_rng(180).standard_normal(X0_18.shape)
    SCHED_18 = mylearn.diffusion.noise_schedule(BETAS)
    AB_18 = SCHED_18["alpha_bars"]
    runs_18 = {"DDPM, bruit initial à chaque pas": ddpm_chain_18(X0_18, EPS_18, SCHED_18, np.random.default_rng(1)),
               "DDPM, vrai bruit du x courant": ddpm_chain_18(X0_18, EPS_18, SCHED_18, np.random.default_rng(1), oracle=True),
               f"DDIM η = 0, les {T_STEPS} pas": ddim_chain_18(X0_18, EPS_18, AB_18, np.arange(T_STEPS - 1, -1, -1)),
               "DDIM η = 0, 10 pas": ddim_chain_18(X0_18, EPS_18, AB_18, wbd.ddim_timesteps(T_STEPS, 10)),
               f"DDIM η = 1, les {T_STEPS} pas": ddim_chain_18(X0_18, EPS_18, AB_18, np.arange(T_STEPS - 1, -1, -1),
                                                                eta=1.0, rng=np.random.default_rng(2))}
    ERRORS_18 = {name: float(np.abs(np.asarray(x) - X0_18).max()) for name, x in runs_18.items()}
    for name, err in ERRORS_18.items():
        print(f"{name:>32} : écart maximal à x0 = {fr_e(err)}")
    verdict("B5.18", max(ERRORS_18[k] for k in list(ERRORS_18)[1:4]) < 1e-9,
            "les trois chaînes exactes retrouvent x0 à moins de 1e-9.",
            "une chaîne qui devrait être exacte s'écarte de x0 de plus de 1e-9 : relis tes trois fonctions.")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))         # each with its own scale
    for ax, name in zip(axes, [list(runs_18)[0], list(runs_18)[3], list(runs_18)[4]]):
        ax.scatter(*X0_18.T, s=4, alpha=0.4, label="x0")
        ax.scatter(*np.asarray(runs_18[name]).T, s=4, alpha=0.4, label="fin de la chaîne")
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
    axes[0].legend(fontsize=8)
    plt.show()
    for letter, answer in zip("abcd", [ddpm_fixed_18, ddpm_oracle_18, ddim_ten_18, ddim_eta1_18]):
        with wb.attempt("B5.18"):
            wb.check(f"B5.18{letter}", answer)'''

# --- 📦 B5.19 ----------------------------------------------------------------------------------------------------------
GIVEN_B = r'''# Part B: the ring of 8 Gaussians, the settings of the two trainings, the measures of quality
DATA_B = torch.tensor(wbd.ring(50_000, seed=1900), dtype=torch.float32)   # 50 000 training points
REAL_B = wbd.ring(2000, seed=2020)                                         # 2 000 other points, for the measures
SETTINGS_B = dict(wbd.TOY_SETTINGS)                    # 3 000 iterations, batches of 512, AdamW at 2e-3 (≈ 10 s)
SCHED_B = toy_schedule(BETAS)                          # the schedule of the notebook, T = T_STEPS (without your code)
FLOOR_B = wbd.sliced_wasserstein(wbd.ring(2000, seed=2021), REAL_B)        # two samples of the ring itself
BLOB_B = wbd.sliced_wasserstein(np.random.default_rng(3).standard_normal((2000, 2)), REAL_B)   # pure noise


def quality(points):
    """(sliced Wasserstein distance to REAL_B, share of points on a mode, modes reached) of 2 000 points."""
    report = wbd.mode_report(points)
    return wbd.sliced_wasserstein(points, REAL_B), report["on_mode"], report["modes"]


print(f"distance de Wasserstein en tranches : {fr(FLOOR_B, 3)} entre deux échantillons de l'anneau, "
      f"{fr(BLOB_B, 3)} pour du bruit gaussien.")'''

TODO_19 = r'''class Denoiser2D_19(nn.Module):
    """MLP that predicts the noise of a 2-D point x_t at the integer position t (contract of the statement)."""

    def __init__(self, n_positions, t_dim=32, hidden=128):
        super().__init__()
        raise NotImplementedError("Denoiser2D_19")

    def forward(self, x, t):
        raise NotImplementedError("Denoiser2D_19.forward")


def train_19(model, data, alpha_bars, n_iters, batch_size, lr, generator):
    """Train model to predict the noise (recipe of the contract); returns the list of the losses (floats)."""
    raise NotImplementedError("train_19")


def sample_19(model, n, schedule, generator, rng):
    """n points by DDPM ancestral sampling with your ddpm_step; a float64 array (n, 2)."""
    raise NotImplementedError("sample_19")'''

SOLUTION_19 = r'''class Denoiser2D_19(nn.Module):
    """MLP that predicts the noise of a 2-D point x_t at the integer position t (contract of the statement)."""

    def __init__(self, n_positions, t_dim=32, hidden=128):
        super().__init__()
        table = mylearn.attention.positional_encoding(n_positions, t_dim)        # your encoding of B3
        self.register_buffer("pe", torch.tensor(table, dtype=torch.float32), persistent=False)
        self.net = nn.Sequential(nn.Linear(2 + t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.pe[t]], dim=1))


def train_19(model, data, alpha_bars, n_iters, batch_size, lr, generator):
    """Train model to predict the noise (recipe of the contract); returns the list of the losses (floats)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ab = torch.tensor(np.asarray(alpha_bars), dtype=torch.float32)
    losses = []
    for _ in range(n_iters):
        x0 = data[torch.randint(len(data), (batch_size,), generator=generator)]
        t = torch.randint(0, len(ab), (batch_size,), generator=generator)
        eps = torch.randn((batch_size, 2), generator=generator)
        a = ab[t][:, None]
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def sample_19(model, n, schedule, generator, rng):
    """n points by DDPM ancestral sampling with your ddpm_step; a float64 array (n, 2)."""
    x = torch.randn((n, 2), generator=generator).double().numpy()
    with torch.no_grad():
        for t in range(len(schedule["alpha_bars"]) - 1, -1, -1):
            eps = model(torch.tensor(x, dtype=torch.float32), torch.full((n,), t, dtype=torch.long)).double().numpy()
            x = mylearn.diffusion.ddpm_step(x, t, eps, schedule, rng=rng)
    return x'''

CHECK_19 = r'''model_ok_19 = False
with checked("B5.19"):
    torch.manual_seed(0)
    mine_19 = Denoiser2D_19(50)
    ref_19 = wbd.Denoiser2D(50)
    names_19 = {k: tuple(v.shape) for k, v in mine_19.state_dict().items()}
    want_19 = {k: tuple(v.shape) for k, v in ref_19.state_dict().items()}
    ok_19 = names_19 == want_19
    verdict("B5.19", ok_19, "noms et formes des paramètres : ceux du contrat (net.0 à net.6).",
            f"le state_dict doit avoir les noms et formes {want_19} ; reçu {names_19} (le buffer pe est-il non persistant ?).")
    if not ok_19:
        raise StopCheck
    pe_19 = getattr(mine_19, "pe", None)
    pe_ok_19 = (pe_19 is not None and tuple(pe_19.shape) == (50, 32)
                and np.allclose(pe_19.double().numpy(), wbd.sinusoidal_table(50, 32), atol=1e-6))
    verdict("B5.19", pe_ok_19, "le buffer pe est la table de positional_encoding(50, 32) (B3).",
            "self.pe doit être la table positional_encoding(n_positions, t_dim) de B3, en float32.")
    ok_19 &= pe_ok_19
    with torch.no_grad():
        for p_ in ref_19.parameters():
            p_.add_(0.3 * torch.randn(p_.shape))                    # weights far from the initialisation
    mine_19.load_state_dict(ref_19.state_dict())
    xq_19, tq_19 = torch.randn(7, 2), torch.tensor([0, 3, 9, 17, 25, 33, 49])
    with torch.no_grad():
        ok_19 &= same("B5.19", "sortie de ton MLP contre celle du contrat, aux mêmes poids", mine_19(xq_19, tq_19),
                      ref_19(xq_19, tq_19), rtol=1e-5, atol=1e-6)
    # 5 iterations of train_19 against the recipe of the contract (the same draws, the same model at the start)
    ab_19 = toy_schedule(np.linspace(1e-3, 0.2, 50))["alpha_bars"]
    a_19, b_19 = copy.deepcopy(ref_19), copy.deepcopy(ref_19)
    got_19 = train_19(a_19, DATA_B, ab_19, 5, 64, 1e-3, torch.Generator().manual_seed(5))
    want_l19 = wbd.train_toy(b_19, DATA_B, 5, 64, 1e-3, torch.Generator().manual_seed(5), ab_19)
    if returned("B5.19", "train_19", got_19):
        ok_19 &= same("B5.19", "les 5 premières loss de train_19 contre la recette du contrat (mêmes tirages, dans le même ordre)", got_19, want_l19,
                      rtol=1e-5, atol=1e-6)
        ok_19 &= same("B5.19", "les poids après 5 itérations", a_19.net[0].weight, b_19.net[0].weight, rtol=1e-4,
                      atol=1e-6)
    else:
        ok_19 = False
    sched_19 = toy_schedule(np.linspace(1e-3, 0.2, 50))
    b_19.eval()
    pts_19 = sample_19(b_19, 6, sched_19, torch.Generator().manual_seed(8), np.random.default_rng(9))
    if returned("B5.19", "sample_19", pts_19):
        ok_19 &= same("B5.19", "sample_19 (6 points, T = 50) contre l'échantillonnage DDPM du contrat", pts_19,
                      wbd.sample_toy_ddpm(b_19, 6, sched_19["alpha_bars"], torch.Generator().manual_seed(8),
                                          np.random.default_rng(9)), rtol=1e-5, atol=1e-5)
        model_ok_19 = bool(ok_19)'''

EXPERIMENT_19 = r'''# Train on the ring, then 2 000 points by DDPM (≈ 10 s of training)
MODEL_19 = None
if not model_ok_19:
    print("⏳ Ex B5.19 : l'entraînement attend ton MLP, train_19 et sample_19 (cellule précédente sans ❌).")
else:
    torch.manual_seed(19)
    MODEL_19 = Denoiser2D_19(T_STEPS)
    start_19 = time.perf_counter()
    LOSSES_19 = train_19(MODEL_19, DATA_B, SCHED_B["alpha_bars"], SETTINGS_B["n_iters"], SETTINGS_B["batch_size"],
                         SETTINGS_B["lr"], torch.Generator().manual_seed(190))
    MODEL_19.eval()
    POINTS_19 = sample_19(MODEL_19, 2000, SCHED_B, torch.Generator().manual_seed(191), np.random.default_rng(192))
    swd_19, on_19, modes_19 = quality(POINTS_19)
    print(f"entraînement et échantillonnage : {time.perf_counter() - start_19:.0f} s (une mesure, pas une réponse)")
    print(f"loss moyenne des 200 dernières itérations : {fr(np.mean(LOSSES_19[-200:]), 3)} ; distance en tranches "
          f"{fr(swd_19, 3)} ; {fr(100 * on_19, 1)} % des points sur un mode ; {modes_19} modes sur 8")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(np.convolve(LOSSES_19, np.ones(50) / 50, mode="valid"))
    ax1.set_xlabel("itération")
    ax1.set_ylabel("loss (moyenne glissante sur 50)")
    ax2.scatter(*REAL_B.T, s=3, alpha=0.3, label="anneau")
    ax2.scatter(*POINTS_19.T, s=3, alpha=0.3, label="DDPM")
    ax2.set_aspect("equal")
    ax2.legend()
    plt.show()
    verdict("B5.19", swd_19 < 0.2 and on_19 > 0.8 and modes_19 == 8,
            "les points générés couvrent les 8 modes, à distance de l'anneau proche de celle de deux échantillons.",
            "les points générés sont loin de l'anneau (distance ≥ 0,2, moins de 80 % sur un mode, ou un mode manqué) : "
            "relis l'entraînement (le pas t donné au modèle, la cible) et l'échantillonnage.")'''

# --- 🔬 B5.20 ----------------------------------------------------------------------------------------------------------
TODO_20 = r'''def train_fm_20(model, data, n_iters, batch_size, lr, generator):
    """Train model to predict the velocity eps - x0 at s = u / 1000 (recipe of the contract); returns the losses."""
    raise NotImplementedError("train_fm_20")


def euler_20(model, n, n_steps, generator):
    """n points by Euler integration of the flow from s = 1 to s = 0 in n_steps steps; a float64 array (n, 2)."""
    raise NotImplementedError("euler_20")


def ddim_20(model, n, n_steps, alpha_bars, generator):
    """n points by DDIM (eta = 0, your ddim_step) on wbd.ddim_timesteps(T, n_steps), then to -1; float64 (n, 2)."""
    raise NotImplementedError("ddim_20")'''

SOLUTION_20 = r'''def train_fm_20(model, data, n_iters, batch_size, lr, generator):
    """Train model to predict the velocity eps - x0 at s = u / 1000 (recipe of the contract); returns the losses."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_iters):
        x0 = data[torch.randint(len(data), (batch_size,), generator=generator)]
        u = torch.randint(1, 1001, (batch_size,), generator=generator)
        eps = torch.randn((batch_size, 2), generator=generator)
        s = (u.float() / 1000)[:, None]
        loss = F.mse_loss(model((1 - s) * x0 + s * eps, u), eps - x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def euler_20(model, n, n_steps, generator):
    """n points by Euler integration of the flow from s = 1 to s = 0 in n_steps steps; a float64 array (n, 2)."""
    x = torch.randn((n, 2), generator=generator)
    with torch.no_grad():
        for k in range(n_steps):
            u = round(1000 * (1 - k / n_steps))                     # s = 1, 1 - 1/n, ..., 1/n
            x = x - model(x, torch.full((n,), u, dtype=torch.long)) / n_steps
    return x.double().numpy()


def ddim_20(model, n, n_steps, alpha_bars, generator):
    """n points by DDIM (eta = 0, your ddim_step) on wbd.ddim_timesteps(T, n_steps), then to -1; float64 (n, 2)."""
    x = torch.randn((n, 2), generator=generator).double().numpy()
    steps = [int(t) for t in wbd.ddim_timesteps(len(alpha_bars), n_steps)] + [-1]
    with torch.no_grad():
        for t, t_prev in zip(steps[:-1], steps[1:]):
            eps = model(torch.tensor(x, dtype=torch.float32), torch.full((n,), t, dtype=torch.long)).double().numpy()
            x = mylearn.diffusion.ddim_step(x, t, t_prev, eps, alpha_bars)
    return x'''

CHECK_20 = r'''fm_ok_20 = False
with checked("B5.20"):
    torch.manual_seed(1)
    ref_20 = wbd.Denoiser2D(wbd.FM_POSITIONS)
    a_20, b_20 = copy.deepcopy(ref_20), copy.deepcopy(ref_20)
    got_20 = train_fm_20(a_20, DATA_B, 5, 64, 1e-3, torch.Generator().manual_seed(6))
    ok_20 = returned("B5.20", "train_fm_20", got_20)
    if ok_20:
        ok_20 = same("B5.20", "les 5 premières loss de train_fm_20 contre la recette du contrat (mêmes tirages, dans le même ordre)", got_20,
                     wbd.train_toy(b_20, DATA_B, 5, 64, 1e-3, torch.Generator().manual_seed(6), flow=True),
                     rtol=1e-5, atol=1e-6)
    b_20.eval()
    for steps_c in (1, 3, 8):
        ok_20 &= same("B5.20", f"euler_20 en {steps_c} pas contre l'intégration du contrat",
                      euler_20(b_20, 5, steps_c, torch.Generator().manual_seed(7)),
                      wbd.sample_toy_euler(b_20, 5, steps_c, torch.Generator().manual_seed(7)), rtol=1e-5, atol=1e-5)
    torch.manual_seed(2)
    eps_net_20 = wbd.Denoiser2D(50).eval()
    ab_20 = toy_schedule(np.linspace(1e-3, 0.2, 50))["alpha_bars"]
    for steps_c in (1, 4, 50):
        ok_20 &= same("B5.20", f"ddim_20 en {steps_c} pas contre le DDIM du contrat",
                      ddim_20(eps_net_20, 5, steps_c, ab_20, torch.Generator().manual_seed(8)),
                      wbd.sample_toy_ddim(eps_net_20, 5, steps_c, ab_20, torch.Generator().manual_seed(8)),
                      rtol=1e-5, atol=1e-5)
    fm_ok_20 = bool(ok_20)'''

EXPERIMENT_20 = r'''# The flow matching model, the same MLP and the same budget as in B5.19, then both samplers from 1 to 100 steps
MEASURED_20 = None
if not (fm_ok_20 and MODEL_19 is not None):
    print("⏳ Ex B5.20 : l'expérience attend tes trois fonctions (cellule précédente sans ❌) et le modèle de B5.19.")
else:
    torch.manual_seed(20)
    MODEL_FM_20 = Denoiser2D_19(wbd.FM_POSITIONS)
    start_20 = time.perf_counter()
    LOSSES_FM_20 = train_fm_20(MODEL_FM_20, DATA_B, SETTINGS_B["n_iters"], SETTINGS_B["batch_size"], SETTINGS_B["lr"],
                               torch.Generator().manual_seed(200))
    MODEL_FM_20.eval()
    print(f"entraînement du flow matching : {time.perf_counter() - start_20:.0f} s")
    MEASURED_20 = {"DDIM (bruit prédit, B5.19)": [], "Euler (vitesse prédite)": []}
    SAMPLES_20 = {}
    for n_20 in STEPS_20:
        pts_d = ddim_20(MODEL_19, 2000, n_20, SCHED_B["alpha_bars"], torch.Generator().manual_seed(201))
        pts_e = euler_20(MODEL_FM_20, 2000, n_20, torch.Generator().manual_seed(201))
        SAMPLES_20[n_20] = (pts_d, pts_e)
        MEASURED_20["DDIM (bruit prédit, B5.19)"].append(quality(pts_d)[0])
        MEASURED_20["Euler (vitesse prédite)"].append(quality(pts_e)[0])
    print("pas :", "  ".join(f"{n:>6}" for n in STEPS_20))
    for name, values in MEASURED_20.items():
        print(f"{name} :", "  ".join(f"{fr(v, 3):>6}" for v in values))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    for (name, values), color in zip(MEASURED_20.items(), ("C0", "C1")):
        ax1.plot(STEPS_20, values, "o-", color=color, label=f"{name}, ton protocole")
        rows = np.array(FULL_20[name])                         # (seeds, steps) of the full protocol
        ax1.fill_between(STEPS_20, rows.min(axis=0), rows.max(axis=0), color=color, alpha=0.15)
    ax1.axhline(FLOOR_B, color="gray", lw=0.8, ls=":", label="deux échantillons de l'anneau")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("nombre de pas d'échantillonnage")
    ax1.set_ylabel("distance de Wasserstein en tranches")
    ax1.legend(fontsize=7)
    for ax, (pts, title) in zip((ax2, ax3), ((SAMPLES_20[5][0], "DDIM, 5 pas"), (SAMPLES_20[5][1], "Euler, 5 pas"))):
        ax.scatter(*REAL_B.T, s=3, alpha=0.2, color="gray")
        ax.scatter(*np.clip(pts, -4, 4).T, s=3, alpha=0.4)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(title)
    plt.show()
    print(f"protocole complet ({len(FULL_20['seeds'])} graines, bandes du graphique) : distance médiane par nombre de pas")
    for name in MEASURED_20:
        print(f"{name} :", "  ".join(f"{fr(v, 3):>6}" for v in np.median(np.array(FULL_20[name]), axis=0)))'''

TODO_20B = r'''fewest_20 = ...     # a) the smallest number of steps of STEPS_20 from which Euler is within 3 times its distance at
                    #    100 steps, for every seed of the full protocol: an integer of STEPS_20
better5_20 = ...    # b) at 5 steps, which sampler has the smaller distance for every seed: "D" (DDIM) or "E" (Euler)?'''

CHECK_20B = r'''with wb.attempt("B5.20"):
    wb.check("B5.20a", fewest_20)
with wb.attempt("B5.20"):
    wb.check("B5.20b", better5_20)'''

# --- 📦 B5.21 ----------------------------------------------------------------------------------------------------------
TODO_21 = r'''class ResBlock_21(nn.Module):
    """Residual block conditioned by the step (contract of the statement)."""

    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        raise NotImplementedError("ResBlock_21")

    def forward(self, x, temb):
        raise NotImplementedError("ResBlock_21.forward")


class TinyUNet_21(nn.Module):
    """A tiny U-Net that predicts the noise of x_t (contract of the statement)."""

    def __init__(self, n_steps, channels=(32, 64, 128), t_dim=128, n_classes=None):
        super().__init__()
        raise NotImplementedError("TinyUNet_21")

    def forward(self, x, t, y=None):
        raise NotImplementedError("TinyUNet_21.forward")


n_small_21 = ...    # a) the number of parameters of TinyUNet_21(400, channels=(16, 32, 64), t_dim=64), with your model
u1_in_21 = ...      # b) the shape of the input of u1 (after the concatenation) for that model and a batch of 8 images,
                    #    a tuple of 4 integers'''

SOLUTION_21 = r'''class ResBlock_21(nn.Module):
    """Residual block conditioned by the step (contract of the statement)."""

    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]              # one bias per channel, from the step
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)                                 # residual connection


class TinyUNet_21(nn.Module):
    """A tiny U-Net that predicts the noise of x_t (contract of the statement)."""

    def __init__(self, n_steps, channels=(32, 64, 128), t_dim=128, n_classes=None):
        super().__init__()
        c1, c2, c3 = channels
        self.config = {"n_steps": n_steps, "channels": [c1, c2, c3], "t_dim": t_dim, "n_classes": n_classes}
        self.n_classes = n_classes
        table = mylearn.attention.positional_encoding(n_steps, t_dim)            # your encoding of B3
        self.register_buffer("pe", torch.tensor(table, dtype=torch.float32), persistent=False)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        if n_classes is not None:
            self.class_emb = nn.Embedding(n_classes + 1, t_dim)                 # the last row is the empty class
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1 = ResBlock_21(c1, c1, t_dim)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)                  # 28 -> 14
        self.d2 = ResBlock_21(c1, c2, t_dim)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)                  # 14 -> 7
        self.mid = ResBlock_21(c2, c3, t_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)           # 7 -> 14
        self.u2 = ResBlock_21(2 * c2, c2, t_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)           # 14 -> 28
        self.u1 = ResBlock_21(2 * c1, c1, t_dim)
        self.out = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x, t, y=None):
        temb = self.t_mlp(self.pe[t])
        if self.n_classes is not None:
            temb = temb + self.class_emb(torch.full_like(t, self.n_classes) if y is None else y)
        h1 = self.d1(self.inp(x), temb)
        h2 = self.d2(self.down1(h1), temb)
        h = self.mid(self.down2(h2), temb)
        h = self.u2(torch.cat([self.up2(h), h2], dim=1), temb)
        h = self.u1(torch.cat([self.up1(h), h1], dim=1), temb)
        return self.out(h)


n_small_21 = sum(p.numel() for p in TinyUNet_21(400, channels=(16, 32, 64), t_dim=64).parameters())
u1_in_21 = (8, 32, 28, 28)'''

CHECK_21 = r'''UNET_OK_21 = False
with checked("B5.21"):
    torch.manual_seed(0)
    ok_21 = True
    blk_21, ref_blk = ResBlock_21(16, 32, 24), wbd.ResBlock(16, 32, 24)
    with torch.no_grad():
        for p_ in ref_blk.parameters():
            p_.add_(0.2 * torch.randn(p_.shape))
    if {k: tuple(v.shape) for k, v in blk_21.state_dict().items()} == \
            {k: tuple(v.shape) for k, v in ref_blk.state_dict().items()}:
        blk_21.load_state_dict(ref_blk.state_dict())
        xb_21, tb_21 = torch.randn(2, 16, 6, 6), torch.randn(2, 24)
        with torch.no_grad():
            ok_21 &= same("B5.21", "ResBlock_21(16, 32, 24) contre le bloc du contrat, aux mêmes poids", blk_21(xb_21, tb_21),
                          ref_blk(xb_21, tb_21), rtol=1e-5, atol=1e-5)
    else:
        ok_21 = False
        print(f"❌ Ex B5.21 : ResBlock_21(16, 32, 24) doit avoir les paramètres "
              f"{sorted(ref_blk.state_dict())} (noms et formes du contrat) ; reçu {sorted(blk_21.state_dict())}.")
    for kw_21 in ({}, {"channels": (16, 32, 64), "t_dim": 64}, {"n_classes": 10}):
        mine_21, ref_21 = TinyUNet_21(T_STEPS, **kw_21), wbd.TinyUNet(T_STEPS, **kw_21)
        shapes_m = {k: tuple(v.shape) for k, v in mine_21.state_dict().items()}
        shapes_r = {k: tuple(v.shape) for k, v in ref_21.state_dict().items()}
        if shapes_m != shapes_r:
            ok_21 = False
            diff_21 = sorted(set(shapes_m.items()) ^ set(shapes_r.items()))[:4]
            print(f"❌ Ex B5.21 : TinyUNet_21({T_STEPS}, {kw_21}) : le state_dict n'a pas les noms et les formes du "
                  f"contrat (premières différences, attendu ou reçu : {diff_21}).")
            continue
        pe_21 = getattr(mine_21, "pe", None)
        ok_21 &= same("B5.21", f"le buffer pe ({kw_21 or 'réglage par défaut'}) : la table de positional_encoding",
                      pe_21 if pe_21 is not None else torch.zeros(0), torch.tensor(wbd.sinusoidal_table(
                          T_STEPS, kw_21.get("t_dim", 128)), dtype=torch.float32), rtol=1e-5, atol=1e-6)
        with torch.no_grad():
            for p_ in ref_21.parameters():
                p_.add_(0.05 * torch.randn(p_.shape))                # weights far from the initialisation
        mine_21.load_state_dict(ref_21.state_dict())
        x_21, t_21 = torch.randn(3, 1, 28, 28), torch.tensor([0, T_STEPS // 2, T_STEPS - 1])
        with torch.no_grad():
            if "n_classes" in kw_21:
                y_21 = torch.tensor([3, 10, 7])
                ok_21 &= same("B5.21", "U-Net conditionné (y donné, dont la classe vide 10) contre le contrat",
                              mine_21(x_21, t_21, y_21), ref_21(x_21, t_21, y_21), rtol=1e-4, atol=1e-5)
                ok_21 &= same("B5.21", "U-Net conditionné, y = None : la classe vide", mine_21(x_21, t_21),
                              ref_21(x_21, t_21, torch.full((3,), 10)), rtol=1e-4, atol=1e-5)
            else:
                ok_21 &= same("B5.21", f"U-Net {kw_21 or '(32, 64, 128)'} contre le contrat, aux mêmes poids",
                              mine_21(x_21, t_21), ref_21(x_21, t_21), rtol=1e-4, atol=1e-5)
        if not kw_21:
            got_s, want_s = wbd.stage_shapes(mine_21, x_21, t_21), wbd.stage_shapes(ref_21, x_21, t_21)
            ok_21 &= got_s == want_s
            verdict("B5.21", got_s == want_s, "les formes de chaque étage sont celles du contrat.",
                    f"formes des étages : attendu {want_s}, reçu {got_s}.")
    UNET_OK_21 = bool(ok_21)
with wb.attempt("B5.21"):
    wb.check("B5.21a", n_small_21)
with wb.attempt("B5.21"):
    wb.check("B5.21b", shape_text(u1_in_21))'''

EXPERIMENT_21 = r'''# The model of part C: your U-Net if it passed its checks, otherwise the U-Net of the contract (wb.diffusion)
UNET_CLASS = TinyUNet_21 if UNET_OK_21 else wbd.TinyUNet
if not UNET_OK_21:
    print("ℹ️ Ex B5.21 : ton U-Net n'est pas encore écrit, ou n'a pas passé ses contrôles ; la suite du chapitre prend "
          "celui du contrat (wbd.TinyUNet). Relance cette cellule une fois le tien prêt.")
net_21 = UNET_CLASS(T_STEPS).eval()
x_21, t_21 = torch.randn(64, 1, 28, 28), torch.randint(0, T_STEPS, (64,))
with torch.no_grad():
    net_21(x_21, t_21)
    times_21 = []
    for _ in range(3):
        start_21 = time.perf_counter()
        net_21(x_21, t_21)
        times_21.append(time.perf_counter() - start_21)
SECONDS_21 = min(times_21)
print(f"{'ton' if UNET_OK_21 else 'le'} U-Net : une passe sur un batch de 64 images en {fr(SECONDS_21, 3)} s (le "
      f"meilleur de trois) ; {T_STEPS} pas de DDPM pour 64 images coûteraient ≈ {T_STEPS * SECONDS_21:.0f} s.")'''
